"""격자 지도 + Frontier + A* + 자동 경로 추종 (초기 수동 모드).
월드의 TurtleBot3Burger.extensionSlot에 추가:
Display { name "map_display" width 300 height 300 }
필요 장치: camera, LDS-01, left/right wheel motor, map_display.
E: 자동 탐색 시작, Space: 정지/수동, WASD: 수동 전환, R: 지도 초기화.
기록은 실행 중 메모리에 유지되며 재시작하면 초기화됩니다.
"""
from controller import Robot, Keyboard
import math
import heapq
from collections import deque

# 1. 로봇과 입력/센서 초기화
robot = Robot()
timestep = int(robot.getBasicTimeStep())
keyboard = Keyboard()
keyboard.enable(timestep)
camera = robot.getDevice("camerae")
camera.enable(timestep)
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
for motor in (left_motor, right_motor):
    motor.setPosition(float("inf"))  # 위치 목표 없이 회전 속도로 제어
    motor.setVelocity(0.0)
left_encoder = left_motor.getPositionSensor()
right_encoder = right_motor.getPositionSensor()
left_encoder.enable(timestep)
right_encoder.enable(timestep)
lidar = robot.getDevice("LDS-01")
lidar.enable(timestep)
lidar.enablePointCloud()  # 배열 인덱스 대신 센서 좌표의 실제 관측점 사용
map_display = robot.getDevice("map_display")
if map_display is None:
    raise RuntimeError('extensionSlot에 name "map_display"인 Display를 추가하세요.')

# 기존 로봇에 자이로가 있으면 회전 측정을 사용. 없으면 엔코더로 동작.
# 센서 Z축이 로봇 수직축이고 양수가 반시계인지 먼저 확인하세요.
gyro = None
for i in range(robot.getNumberOfDevices()):
    device = robot.getDeviceByIndex(i)
    if 'gyro' in device.getName().lower() and hasattr(device, 'getValues'):
        gyro = device
        gyro.enable(timestep)
        break
GYRO_SIGN = 1.0
print('방향 추정:', '자이로 Z축' if gyro is not None else '엔코더 (자이로 없음)')

# 2. 지도/주행 설정. 실제 PROTO의 치수와 센서 장착 위치를 확인해 보정하세요.
WHEEL_RADIUS = 0.033
WHEEL_DISTANCE = 0.160
LIDAR_OFFSET_X = 0.0
LIDAR_OFFSET_Y = 0.0
LIDAR_YAW = 0.0
CELL_SIZE = 0.05              # 5cm 격자로 좁은 통로 표현 개선
ROBOT_RADIUS = 0.12           # 몸체를 보수적인 원으로 근사, m
SAFETY_MARGIN = 0.02         # 몸체 크기는 유지하고 추가 여유만 2cm로 조정
INFLATION_RADIUS = ROBOT_RADIUS + SAFETY_MARGIN
MAP_RANGE = 3.5               # 유효 관측도 이 거리까지만 지도에 사용
MAP_PERIOD = 0.20
DRAW_PERIOD = 0.40
PLAN_PERIOD = 2.0
MAX_GRID_CELLS = 100000       # 한도에 도달하면 정지하고 기록 유지
AUTO_SPEED = 0.09             # 저속 시작, m/s
MANUAL_SPEED = 0.099
MAX_OMEGA = 0.8
WAYPOINT_TOLERANCE = 0.03
GOAL_TOLERANCE = 0.10
BASE_SCALE = 0.02
VIEW_MARGIN = 14

# 시작 자세 기준 상대 좌표. +x 전방, +y 좌측, 회전은 반시계 양수.
x = y = theta = 0.0
previous_left = previous_right = None
previous_gyro_z = None
path_history = [(0.0, 0.0)]
# 값은 점유 증거 점수: <= -1 빈 공간 / >= 2 장애물 / 나머지 미확인.
# 사전에 없는 칸도 미확인. 빈 공간과 장애물을 별도로 구분해야 Frontier를 찾을 수 있음.
grid = {}
local_hits = []               # 현재 관측: 로봇 기준 (전방, 좌측) 좌표 리스트
frontier_cells = set()
planned_path = []             # 앞으로 따라갈 지도 좌표 리스트
waypoint_index = 0
goal = None
failed_goals = {}             # 실패 목표 칸: 다시 허용할 시뮬레이션 시각
mode = 'MANUAL'
visit_counts = {}            # 실제로 지난 30cm 구역의 방문 횟수
last_visit_region = None
selected_reason = ''
visited_goals = {}             # 성공 목표는 실패 목록과 분리
plan_reason = ''
scan_started = 0.0
scan_rotation = 0.0
scan_previous_theta = 0.0
next_scan_time = 0.0
last_diagnostic = -100.0
last_map = last_draw = last_plan = -100.0
blocked_since = None
progress_distance = float('inf')
progress_time = 0.0
scan_ready = False
map_full = False
view_scale = BASE_SCALE
width, height = map_display.getWidth(), map_display.getHeight()
motor_limit = min(left_motor.getMaxVelocity(), right_motor.getMaxVelocity())
CARDINAL = ((1, 0), (-1, 0), (0, 1), (0, -1))


def cell_of(a, b):
    return (math.floor(a / CELL_SIZE), math.floor(b / CELL_SIZE))


def center_of(cell):
    return ((cell[0] + 0.5) * CELL_SIZE, (cell[1] + 0.5) * CELL_SIZE)


def adjacent(cell):
    for dx, dy in CARDINAL:
        yield (cell[0] + dx, cell[1] + dy)


def ray_cells(start, end):
    """Bresenham: LiDAR 시작점부터 끝점까지 통과하는 격자 칸을 나열."""
    ax, ay = start
    bx, by = end
    dx, dy = abs(bx - ax), -abs(by - ay)
    sx, sy = (1 if ax < bx else -1), (1 if ay < by else -1)
    error = dx + dy
    result = []
    while True:
        result.append((ax, ay))
        if (ax, ay) == (bx, by):
            return result
        twice = 2 * error
        if twice >= dy:
            error += dy
            ax += sx
        if twice <= dx:
            error += dx
            ay += sy


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def update_pose():
    """모터 명령이 아닌 엔코더 측정 차이로 위치를 추정."""
    global x, y, theta, previous_left, previous_right, previous_gyro_z
    left, right = left_encoder.getValue(), right_encoder.getValue()
    if not all(math.isfinite(a) for a in (left, right)):
        return False
    gyro_z = None
    if gyro is not None:
        value = gyro.getValues()[2]
        if math.isfinite(value):
            gyro_z = value * GYRO_SIGN
    if previous_left is not None:
        dl = (left - previous_left) * WHEEL_RADIUS
        dr = (right - previous_right) * WHEEL_RADIUS
        ds, da = (dl + dr) / 2, (dr - dl) / WHEEL_DISTANCE
        # 바퀴가 헛돌아도 자이로는 몸체의 실제 회전율을 측정.
        # 시간 적분 오차는 남으므로 장시간 지도 정합을 보장하지 않음.
        if gyro_z is not None and previous_gyro_z is not None:
            da = (gyro_z + previous_gyro_z) * 0.5 * timestep / 1000.0
        x += ds * math.cos(theta + da / 2)
        y += ds * math.sin(theta + da / 2)
        theta = wrap(theta + da)
        if math.hypot(x-path_history[-1][0], y-path_history[-1][1]) >= 0.03:
            path_history.append((x, y))
            if len(path_history) > 20000:
                del path_history[0]
    previous_gyro_z = gyro_z
    previous_left, previous_right = left, right
    return True


def read_scan():
    """유효 점군을 로봇 좌표로 변환. inf/NaN은 장애물로 기록하지 않음."""
    global scan_ready
    local_hits.clear()
    ranges = lidar.getRangeImage()
    # 관측이 아직 준비되지 않았으면 정지. inf는 범위 내 반사 없음으로 취급.
    scan_ready = bool(ranges) and any(not math.isnan(r) for r in ranges)
    ca, sa = math.cos(LIDAR_YAW), math.sin(LIDAR_YAW)
    for p in lidar.getPointCloud():
        if not all(math.isfinite(a) for a in (p.x, p.y, p.z)):
            continue
        distance = math.sqrt(p.x*p.x + p.y*p.y + p.z*p.z)
        if lidar.getMinRange() < distance < lidar.getMaxRange():
            local_hits.append((LIDAR_OFFSET_X + ca*p.x - sa*p.y,
                               LIDAR_OFFSET_Y + sa*p.x + ca*p.y))


def update_grid():
    """관측 광선 내부는 빈 공간, 끝점은 장애물. 한 스캔당 한 칸 한 번 갱신.
    inf 광선은 좌표가 없으므로 여기서는 빈 공간으로 채우지 않음(보수적).
    MAP_RANGE를 넘어선 유효 광선은 잘린 끝까지 빈 공간으로만 기록.
    """
    global map_full
    cs, sn = math.cos(theta), math.sin(theta)
    ox = x + cs*LIDAR_OFFSET_X - sn*LIDAR_OFFSET_Y
    oy = y + sn*LIDAR_OFFSET_X + cs*LIDAR_OFFSET_Y
    start = cell_of(ox, oy)
    free, occupied = set(), set()
    for rx, ry in local_hits:
        dx, dy = rx-LIDAR_OFFSET_X, ry-LIDAR_OFFSET_Y
        distance = math.hypot(dx, dy)
        factor = min(1.0, MAP_RANGE / max(distance, 1e-9))
        gx = ox + cs*dx*factor - sn*dy*factor
        gy = oy + sn*dx*factor + cs*dy*factor
        cells = ray_cells(start, cell_of(gx, gy))
        if distance <= MAP_RANGE:
            free.update(cells[:-1])
            occupied.add(cells[-1])
        else:
            free.update(cells)
    free.add(cell_of(x, y))
    # 같은 스캔에서 빈 공간과 장애물이 겹치면 장애물을 우선.
    for cell in free - occupied:
        if cell in grid or len(grid) < MAX_GRID_CELLS:
            grid[cell] = max(-6, grid.get(cell, 0) - 1)
        else:
            map_full = True
    for cell in occupied:
        if cell in grid or len(grid) < MAX_GRID_CELLS:
            grid[cell] = min(6, grid.get(cell, 0) + 3)
        else:
            map_full = True


def navigable_cells():
    """장애물 칸을 몸체 반지름+여유만큼 확장. 모르는 칸은 A*에서 제외."""
    free = {c for c, score in grid.items() if score <= -1}
    blocked = set()
    radius = math.ceil(INFLATION_RADIUS / CELL_SIZE)
    # 칸의 면적까지 고려해 보수적으로 확장
    offsets = [(a, b) for a in range(-radius, radius+1)
               for b in range(-radius, radius+1)
               if math.hypot(max(abs(a)-0.5, 0), max(abs(b)-0.5, 0))*CELL_SIZE <= INFLATION_RADIUS]
    for c, score in grid.items():
        if score >= 2:
            blocked.update((c[0]+a, c[1]+b) for a, b in offsets)
    return free - blocked


def astar(start, target, allowed):
    """4방향 A*: 벽 모서리를 대각선으로 통과하지 않고 미확인 칸은 제외."""
    if start not in allowed or target not in allowed:
        return []
    def heuristic(c):
        return abs(c[0]-target[0]) + abs(c[1]-target[1])
    queue = [(heuristic(start), 0, start)]
    cost, parent = {start: 0}, {}
    while queue:
        _, g, current = heapq.heappop(queue)
        if g != cost.get(current):
            continue
        if current == target:
            result = [current]
            while current in parent:
                current = parent[current]
                result.append(current)
            return result[::-1]
        for nxt in adjacent(current):
            ng = g + 1
            if nxt in allowed and ng < cost.get(nxt, float('inf')):
                cost[nxt], parent[nxt] = ng, current
                heapq.heappush(queue, (ng+heuristic(nxt), ng, nxt))
    return []


def observation_signature(cell):
    """목표 주변 지도의 상태가 바뀌면 같은 목표도 다시 평가할 수 있음."""
    return frozenset((a, b, -1 if grid.get((a, b), 0) <= -1 else
                      2 if grid.get((a, b), 0) >= 2 else 0)
                     for a in range(cell[0]-4, cell[0]+5)
                     for b in range(cell[1]-4, cell[1]+5))


def choose_frontier(allowed, now):
    """경계 검출과 안전한 관측 목표 선택을 분리. 기존 길 재방문 허용."""
    global frontier_cells, plan_reason, selected_reason
    free = {c for c, score in grid.items() if score <= -1}
    # 경계는 확장 장애물 적용 전 지도에서 검출해야 문 근처 경계도 남음.
    frontier_cells = {c for c in free
                      if any(-1 < grid.get(n, 0) < 2 for n in adjacent(c))}
    start = cell_of(x, y)
    if start not in allowed:
        plan_reason = '현재 로봇 칸이 안전 영역 밖: 위치 오차/장애물 확장 확인'
        return None
    reachable, distance = {start}, {start: 0}
    queue = deque([start])
    while queue:
        c = queue.popleft()
        for n in adjacent(c):
            if n in allowed and n not in reachable:
                reachable.add(n)
                distance[n] = distance[c]+1
                queue.append(n)
    # 실패 목표는 임시 제외 후 다시 허용. 성공은 주변 지도 갱신 시 즉시 재평가.
    for c in list(failed_goals):
        if failed_goals[c] <= now:
            del failed_goals[c]
    pending = set(frontier_cells)
    best, best_score = None, -float('inf')
    groups = candidates_count = 0
    while pending:
        first = pending.pop()
        group, queue = [first], deque([first])
        while queue:
            for n in adjacent(queue.popleft()):
                if n in pending:
                    pending.remove(n)
                    group.append(n)
                    queue.append(n)
        groups += 1  # 작은 경계도 버리지 않고 평가
        candidates = set()
        # Frontier 위에 직접 올라가지 않고 근처의 안전한 관측 위치를 선택.
        for edge in group:
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    c = (edge[0]+dx, edge[1]+dy)
                    if c not in reachable or distance[c]*CELL_SIZE < 0.20:
                        continue
                    if math.hypot(dx, dy)*CELL_SIZE > 0.40:
                        continue
                    # 벽 너머의 경계를 관측할 수 있다고 가정하지 않음.
                    if any(grid.get(q, 0) >= 2 for q in ray_cells(c, edge)):
                        continue
                    if any(math.hypot(c[0]-f[0], c[1]-f[1])*CELL_SIZE < 0.25
                           for f in failed_goals):
                        continue
                    visited = visited_goals.get(c)
                    if visited and now-visited[0] < 20.0 and observation_signature(c) == visited[1]:
                        continue
                    candidates.add(c)
        candidates_count += len(candidates)
        for c in candidates:
            # 지나온 길 여부는 비용에 넣지 않음. 먼 경계로 복귀 가능.
            edge_distance = min(math.hypot(c[0]-e[0], c[1]-e[1]) for e in group)*CELL_SIZE
            gx, gy = center_of(c)
            visits = visit_counts.get((math.floor(gx/0.3), math.floor(gy/0.3)), 0)
            travel = distance[c]*CELL_SIZE
            # 그룹 길이를 예상 새 관측량의 대용 지표로 사용.
            # 실제 정보 이득 전체를 예측하는 정밀 센서 모델은 아님.
            information = len(group)*CELL_SIZE
            score = information / ((0.3+travel)*(1+0.7*visits)) - edge_distance

            if score > best_score:
                best, best_score = c, score
                selected_reason = (f'경로 {travel:.2f}m, 경계 길이 {information:.2f}m, '
                                   f'방문 {visits}회, 점수 {score:.2f}: '
                                   '짧은 이동/새 관측/적은 방문 우선')
    plan_reason = (f'경계 {len(frontier_cells)}칸/{groups}그룹, '
                   f'연결된 안전 칸 {len(reachable)}, 관측 후보 {candidates_count}')
    return best


def begin_rescan(now):
    """전진 없이 회전 관측 시작. 회전량은 시간 대신 엔코더 방향으로 측정."""
    global mode, scan_started, scan_rotation, scan_previous_theta
    mode = 'SCAN'
    scan_started = now
    scan_rotation = 0.0
    scan_previous_theta = theta
    print('재탐지: 주변 회전 관측 시작')


def discard_goal(now, failed=False):
    global goal, waypoint_index, blocked_since
    if failed and goal is not None:
        failed_goals[cell_of(*goal)] = now + 12.0
    goal = None
    planned_path.clear()
    waypoint_index = 0
    blocked_since = None


def plan(now):
    global goal, waypoint_index, progress_distance, progress_time, last_plan, plan_reason
    allowed = navigable_cells()
    start = cell_of(x, y)
    target = cell_of(*goal) if goal is not None else choose_frontier(allowed, now)
    route = astar(start, target, allowed) if target is not None else []
    if not route:
        if target is not None:
            plan_reason = '선택한 목표까지 A* 경로 없음'
        discard_goal(now, failed=goal is not None)
        last_plan = now
        return False
    goal = center_of(target)
    planned_path[:] = [center_of(c) for c in route]
    # 현재 칸 중심으로 돌아가지 않고 다음 칸부터 추종.
    waypoint_index = min(1, len(planned_path)-1)
    progress_distance = (len(planned_path)-waypoint_index)*CELL_SIZE + math.hypot(
        planned_path[waypoint_index][0]-x, planned_path[waypoint_index][1]-y)
    progress_time = now
    last_plan = now
    return True


def follow_path():
    """각 경유점에 방향을 맞춘 뒤 전진. 회전 중에는 전진 속도를 줄임."""
    global waypoint_index
    while waypoint_index < len(planned_path)-1:
        a, b = planned_path[waypoint_index]
        if math.hypot(a-x, b-y) > WAYPOINT_TOLERANCE:
            break
        waypoint_index += 1
    if not planned_path:
        return 0.0, 0.0
    gx, gy = planned_path[waypoint_index]
    error = wrap(math.atan2(gy-y, gx-x) - theta)
    omega = max(-MAX_OMEGA, min(MAX_OMEGA, 2.0*error))
    v = 0.0 if abs(error) > 0.65 else min(AUTO_SPEED, math.hypot(gx-x, gy-y))*max(0.2, math.cos(error))
    return v, omega


def safe_motion(v, omega):
    """현재 관측으로 이동 통로를 검사하고 가까운 물체에서는 감속/정지.
    몸체를 원으로 근사하고 여유가 좁으면 보수적으로 회전도 정지.
    LiDAR보다 낮은 물체 등 관측할 수 없는 장애물은 대응하지 못함.
    """
    if not scan_ready:
        return 0.0, 0.0
    clearance = ROBOT_RADIUS + 0.015
    if any(math.hypot(a, b) < clearance for a, b in local_hits):
        return 0.0, 0.0
    if v != 0:
        sign = 1 if v > 0 else -1
        ahead = [sign*a for a, b in local_hits if sign*a > 0 and abs(b) < INFLATION_RADIUS]
        distance = min(ahead, default=float('inf'))
        stop_distance = ROBOT_RADIUS + 0.04
        factor = max(0.0, min(1.0, (distance-stop_distance)/0.15))
        v *= factor
        # 좁은 통로에서는 측면 거리를 보고 전진 속도만 낮춤.
        side = min((abs(b) for a, b in local_hits if abs(a) < 0.20), default=float('inf'))
        if side < ROBOT_RADIUS+0.08:
            v = max(-0.045, min(0.045, v))
        if factor == 0:
            omega = 0.0
    return v, omega


def set_motion(v, omega):
    """선속도/각속도를 좌우 모터 회전속도로 바꾸고 공통 배율로 제한."""
    left = (v - omega*WHEEL_DISTANCE/2)/WHEEL_RADIUS
    right = (v + omega*WHEEL_DISTANCE/2)/WHEEL_RADIUS
    factor = min(1.0, motor_limit/max(abs(left), abs(right), 1e-9))
    left_motor.setVelocity(left*factor)
    right_motor.setVelocity(right*factor)


def to_pixel(a, b):
    return (round((width-1)/2 + (a-x)/view_scale),
            round((height-1)/2 - (b-y)/view_scale))


def draw_map():
    """300x300, 로봇 중앙, 알려진 전체 지도에 맞춰 자동 축소. 글자 없음."""
    global view_scale
    view_scale = BASE_SCALE
    hw, hh = max(1, width/2-VIEW_MARGIN), max(1, height/2-VIEW_MARGIN)
    for c in grid:
        a, b = center_of(c)
        view_scale = max(view_scale, (abs(a-x)+CELL_SIZE)/hw, (abs(b-y)+CELL_SIZE)/hh)
    map_display.setColor(0x111827)  # 미확인
    map_display.fillRectangle(0, 0, width, height)
    # 빈 공간 먼저, 장애물 나중: 축소 시 한 픽셀에서 장애물이 사라지지 않도록
    size = max(1, math.ceil(CELL_SIZE/view_scale))
    for occupied in (False, True):
        map_display.setColor(0x475569 if not occupied else 0xF1F5F9)
        for c, score in grid.items():
            if not ((score >= 2) if occupied else (score <= -1)):
                continue
            a, b = center_of(c)
            px, py = to_pixel(a-CELL_SIZE/2, b+CELL_SIZE/2)
            if 0 <= px < width and 0 <= py < height:
                map_display.fillRectangle(px, py, min(size, width-px), min(size, height-py))
    map_display.setColor(0xA78BFA)  # Frontier
    for c in frontier_cells:
        px, py = to_pixel(*center_of(c))
        if 0 <= px < width and 0 <= py < height:
            map_display.drawPixel(px, py)
    for points, color in ((path_history, 0x38BDF8), (planned_path, 0xFACC15)):
        map_display.setColor(color)
        for a, b in zip(points, points[1:]):
            ax, ay = to_pixel(*a)
            bx, by = to_pixel(*b)
            if all((0 <= ax < width, 0 <= bx < width, 0 <= ay < height, 0 <= by < height)):
                map_display.drawLine(ax, ay, bx, by)
    if goal is not None:
        map_display.setColor(0xFACC15)
        px, py = to_pixel(*goal)
        if 4 <= px < width-4 and 4 <= py < height-4:
            map_display.drawOval(px, py, 4, 4)
    # 화면 중앙 표시. 녹색 자동 / 청록 수동 / 빨강 정지 상태
    map_display.setColor(0xF59E0B if mode == 'SCAN' else (0x22C55E if mode == 'AUTO' else (0xEF4444 if mode == 'WAIT' else 0x2DD4BF)))
    px, py = to_pixel(x, y)
    map_display.fillOval(px, py, 4, 4)
    map_display.drawLine(px, py, px+round(12*math.cos(theta)), py-round(12*math.sin(theta)))


print('E: 자동 Frontier 탐색 / Space: 정지 / WASD: 수동 / R: 지도 초기화')
print('초기 수동 모드. 저속으로 벽의 지도 정렬을 확인한 뒤 E를 누르세요.')
previous_keys = set()
while robot.step(timestep) != -1:
    now = robot.getTime()
    ready = update_pose()
    # 같은 구역에 머무르는 시간 대신 구역 재진입 횟수를 기록.
    region = (math.floor(x/0.3), math.floor(y/0.3))
    if ready and region != last_visit_region:
        visit_counts[region] = visit_counts.get(region, 0)+1
        last_visit_region = region
    read_scan()  # 충돌 검사는 매 제어 주기마다 최신 센서로
    keys = set()
    key = keyboard.getKey()
    while key != -1:
        keys.add(key)
        key = keyboard.getKey()
    keys = {chr(k).upper() for k in keys if 0 <= k < 128}
    pressed = keys - previous_keys  # 키를 계속 누를 때 모드를 반복 변경하지 않음
    previous_keys = keys
    if ' ' in keys:
        mode = 'MANUAL'
        discard_goal(now)
    elif 'R' in pressed:
        mode = 'MANUAL'
        x = y = theta = 0.0
        grid.clear()
        frontier_cells.clear()
        failed_goals.clear()
        visited_goals.clear()
        visit_counts.clear()
        last_visit_region = None
        next_scan_time = 0.0
        path_history[:] = [(0.0, 0.0)]
        map_full = False
        discard_goal(now)
        last_map = -100.0
    elif 'E' in pressed:
        mode = 'AUTO'
        discard_goal(now)
        last_plan = -100.0
        next_scan_time = 0.0
        print('자동 탐색 시작')
    elif keys & set('WASD'):
        mode = 'MANUAL'
        discard_goal(now)
    if ready and scan_ready and now-last_map >= MAP_PERIOD:
        update_grid()
        last_map = now
    v = omega = 0.0
    if mode == 'MANUAL':
        if 'W' in keys:
            v = MANUAL_SPEED
        elif 'S' in keys:
            v = -MANUAL_SPEED
        elif 'A' in keys:
            omega = MAX_OMEGA
        elif 'D' in keys:
            omega = -MAX_OMEGA
    elif mode in ('AUTO', 'WAIT', 'SCAN') and ready and scan_ready:
        # 재탐지 도중에도 지도는 계속 갱신되며 2초마다 경로 선택을 재시도.
        if mode == 'SCAN':
            scan_rotation += abs(wrap(theta-scan_previous_theta))
            scan_previous_theta = theta
        if goal is not None and math.hypot(goal[0]-x, goal[1]-y) < GOAL_TOLERANCE:
            c = cell_of(*goal)
            visited_goals[c] = (now, observation_signature(c))
            discard_goal(now, failed=False)  # 도착을 실패로 기록하지 않음
            begin_rescan(now)
            last_plan = now  # 새 관측을 먼저 수집한 뒤 목표 선택
        if goal is not None and now-last_plan >= PLAN_PERIOD:
            allowed = navigable_cells()
            if any(cell_of(*p) not in allowed for p in planned_path[waypoint_index:]):
                plan(now)
            else:
                last_plan = now
        if goal is None and now-last_plan >= PLAN_PERIOD:
            if plan(now):
                mode = 'AUTO'
                print('새 Frontier 관측 목표:', tuple(round(a, 2) for a in goal))
                print('선택 이유:', selected_reason)
            else:
                if now-last_diagnostic >= 8.0:
                    print('재계획 대기:', plan_reason)
                    last_diagnostic = now
                if mode != 'SCAN' and now >= next_scan_time:
                    begin_rescan(now)
                elif mode != 'SCAN':
                    mode = 'WAIT'
        if mode == 'SCAN':
            # 한 바퀴 관측 또는 18초 제한 후 잠시 대기, 이후 다시 관측.
            # 회전이 안전 검사로 막혀도 무한히 같은 회전을 명령하지 않음.
            if scan_rotation >= 2*math.pi-0.12 or now-scan_started >= 18.0:
                mode = 'WAIT'
                next_scan_time = now+6.0
                print('재탐지 관측 종료: 지도 갱신 후 경로 재평가')
            else:
                omega = 0.45
        if goal is not None:
            mode = 'AUTO'
            v, omega = follow_path()
            # 목표 직선거리 대신 현재 경유점 진행으로 감시: 우회 이동도 정상 진행.
            remaining = (len(planned_path)-waypoint_index)*CELL_SIZE + math.hypot(
                planned_path[waypoint_index][0]-x, planned_path[waypoint_index][1]-y)
            if remaining < progress_distance-0.03:
                progress_distance, progress_time = remaining, now
            if now-progress_time > 20.0:
                print('경유점 진행 정체: 목표 임시 제외 후 재탐지')
                discard_goal(now, failed=True)
                begin_rescan(now)
                v = omega = 0.0
    desired_v, desired_omega = v, omega
    v, omega = safe_motion(v, omega)
    if mode == 'AUTO' and (abs(desired_v) + abs(desired_omega) > 0.01) and abs(v)+abs(omega) < 0.001:
        if blocked_since is None:
            blocked_since = now
        elif now-blocked_since > 3.0:
            print("충돌 방지 정지 지속: 재탐지로 전환")
            discard_goal(now, failed=True)
            begin_rescan(now)
            v = omega = 0.0
    else:
        blocked_since = None
    if not ready or map_full or ' ' in keys or 'R' in keys:
        v = omega = 0.0
        if map_full:
            mode = 'WAIT'
    set_motion(v, omega)
    if ready and now-last_draw >= DRAW_PERIOD:
        draw_map()
        last_draw = now
