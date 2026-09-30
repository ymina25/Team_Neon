"""WASD 수동 주행 + 엔코더 이동 경로 + LiDAR 관측 지도.
월드의 TurtleBot3Burger.extensionSlot에 추가:
Display { name "map_display" width 300 height 300 }
필요 장치: camera, LDS-01, left/right wheel motor, map_display.
R: 지도와 상대 위치 초기화(로봇의 실제 위치는 바꾸지 않음).
기록은 실행 중 메모리에 유지되며 재시작하면 초기화됩니다.
"""
from controller import Robot, Keyboard
import math

# 1. 로봇과 입력/센서 초기화
robot = Robot()
timestep = int(robot.getBasicTimeStep())
keyboard = Keyboard()
keyboard.enable(timestep)
camera = robot.getDevice("camera")
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

# 2. 설정값: 실제 PROTO의 바퀴 치수/센서 장착 위치와 비교해 조정
SPEED = 3.0                       # 모터 회전 속도, rad/s
WHEEL_RADIUS = 0.033               # Burger 기준 초기값, m
WHEEL_DISTANCE = 0.160             # 좌우 바퀴 중심 사이 초기값, m
LIDAR_OFFSET_X = 0.0               # 로봇 기준 LiDAR 위치, m
LIDAR_OFFSET_Y = 0.0               # 장착 오프셋은 PROTO 확인 후 보정
LIDAR_YAW = 0.0                    # 로봇 대비 센서 회전, rad
METERS_PER_PIXEL = 0.02            # 최소 배율: 300px에 6m 표시
AUTO_FIT = True                    # 누적 지도 전체가 보이도록 자동 축소
VIEW_MARGIN = 16                   # 화면 가장자리 여백, px
view_scale = METERS_PER_PIXEL
CELL_SIZE = 0.05                   # 같은 5cm 칸의 관측은 한 번만 기록
MAX_OBSTACLES = 30000              # 리스트가 무한히 커지지 않게 제한
MAX_PATH = 20000
DRAW_PERIOD = 0.20                 # 지도 갱신 주기, 시뮬레이션 초
PATH_SPACING = 0.02                # 2cm 이상 이동했을 때 경로 추가

# 3. 리스트 기록: 위치/방향은 시작 자세를 원점으로 하는 상대 좌표
# x: 시작 시 로봇 앞쪽, y: 시작 시 로봇 왼쪽, theta: 반시계 방향
x, y, theta = 0.0, 0.0, 0.0
path_history = [(x, y)]            # [(x, y), ...] 이동 경로
obstacle_points = []              # [(x, y), ...] 누적 장애물 관측
current_scan_points = []          # 현재 스캔만 기록
obstacle_cells = set()            # 중복 검사용 보조 자료구조
previous_left = previous_right = None
last_draw = -DRAW_PERIOD
width, height = map_display.getWidth(), map_display.getHeight()
limit = min(left_motor.getMaxVelocity(), right_motor.getMaxVelocity())
SPEED = min(SPEED, limit)


def to_pixel(px, py):
    """현재 로봇 위치를 빼서 중앙 고정. 방향축은 고정하고 지도만 이동."""
    return (int(round((width - 1) / 2 + (px - x) / view_scale)),
            int(round((height - 1) / 2 - (py - y) / view_scale)))


def visible(px, py):
    return 0 <= px < width and 0 <= py < height


def update_pose():
    """엔코더의 누적 각도 차이를 바퀴 이동 거리로 변환."""
    global x, y, theta, previous_left, previous_right
    left = left_encoder.getValue()
    right = right_encoder.getValue()
    if not (math.isfinite(left) and math.isfinite(right)):
        return False  # 첫 센서값이 준비되기 전에는 계산하지 않음
    if previous_left is not None:
        dl = (left - previous_left) * WHEEL_RADIUS
        dr = (right - previous_right) * WHEEL_RADIUS
        ds = (dl + dr) / 2
        da = (dr - dl) / WHEEL_DISTANCE
        # 중간 방향으로 적분하면 회전하며 이동할 때 오차가 줄어듦
        x += ds * math.cos(theta + da / 2)
        y += ds * math.sin(theta + da / 2)
        theta = (theta + da + math.pi) % (2 * math.pi) - math.pi
        if math.hypot(x - path_history[-1][0], y - path_history[-1][1]) >= PATH_SPACING:
            path_history.append((x, y))
            if len(path_history) > MAX_PATH:
                del path_history[0]
    previous_left, previous_right = left, right
    return True


def update_scan():
    """LiDAR 로컬 관측점을 로봇 좌표, 다시 지도 좌표로 변환.
    센서 로컬 +x 전방/+y 좌측 및 수평 장착을 가정함.
    흰 점은 관측 끝점만 의미하며 빈 공간/미확인 칸을 구분하지 않음.
    """
    current_scan_points.clear()
    cs, sn = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(LIDAR_YAW), math.sin(LIDAR_YAW)
    for point in lidar.getPointCloud():
        sx, sy, sz = point.x, point.y, point.z
        if not all(math.isfinite(v) for v in (sx, sy, sz)):
            continue  # 범위 밖(inf)이나 유효하지 않은 측정은 버림
        distance = math.sqrt(sx*sx + sy*sy + sz*sz)
        if not lidar.getMinRange() < distance < lidar.getMaxRange():
            continue
        rx = LIDAR_OFFSET_X + ca * sx - sa * sy
        ry = LIDAR_OFFSET_Y + sa * sx + ca * sy
        gx = x + cs * rx - sn * ry
        gy = y + sn * rx + cs * ry
        current_scan_points.append((gx, gy))
        cell = (math.floor(gx / CELL_SIZE), math.floor(gy / CELL_SIZE))
        if cell not in obstacle_cells:
            obstacle_cells.add(cell)
            obstacle_points.append((gx, gy))
    # 오래된 점을 지우고 중복 검사 집합도 함께 갱신
    if len(obstacle_points) > MAX_OBSTACLES:
        del obstacle_points[:-MAX_OBSTACLES]
        obstacle_cells.clear()
        obstacle_cells.update((math.floor(a / CELL_SIZE), math.floor(b / CELL_SIZE))
                              for a, b in obstacle_points)


def draw_points(points, color):
    map_display.setColor(color)
    for a, b in points:
        px, py = to_pixel(a, b)
        if visible(px, py):
            map_display.drawPixel(px, py)


def draw_map():
    """글자/격자 없이 누적 관측, 현재 스캔, 경로, 중앙 로봇을 그림."""
    map_display.setColor(0x111827)
    map_display.fillRectangle(0, 0, width, height)
    # 알려진 지도 크기만 사용해 자동 배율 계산. 미탐색 공간은 알 수 없음.
    # 로봇 중앙 기준으로 가장 먼 점도 여백 안에 들어오도록 맞춤.
    global view_scale
    view_scale = METERS_PER_PIXEL
    if AUTO_FIT:
        half_w = max(1, (width - 1) / 2 - VIEW_MARGIN)
        half_h = max(1, (height - 1) / 2 - VIEW_MARGIN)
        for points in (obstacle_points, current_scan_points, path_history):
            for gx, gy in points:
                view_scale = max(view_scale, abs(gx-x) / half_w,
                                 abs(gy-y) / half_h)
    draw_points(obstacle_points, 0xE5E7EB)
    draw_points(current_scan_points, 0xF59E0B)
    map_display.setColor(0x38BDF8)
    for a, b in zip(path_history, path_history[1:]):
        ax, ay = to_pixel(*a)
        bx, by = to_pixel(*b)
        if visible(ax, ay) and visible(bx, by):
            map_display.drawLine(ax, ay, bx, by)
    px, py = to_pixel(x, y)
    if visible(px, py):
        map_display.setColor(0x22C55E)
        map_display.fillOval(px, py, 5, 5)
        # 초록 선은 로봇이 현재 바라보는 앞쪽 방향
        hx, hy = to_pixel(x + 0.4 * math.cos(theta), y + 0.4 * math.sin(theta))
        if visible(hx, hy):
            map_display.drawLine(px, py, hx, hy)


print('WASD: 수동 이동 / 키를 놓으면 정지 / R: 상대 지도 초기화')
print('자동 회피 기능은 없습니다. 벽 근처에서는 짧게 조작하세요.')

# 4. 메인 루프: 위치 추정 → 키 입력 → 모터 출력 → 주기적 지도 갱신
while robot.step(timestep) != -1:
    pose_ready = update_pose()
    keys = set()
    key = keyboard.getKey()
    while key != -1:  # 이번 단계에 들어온 키 입력을 모두 소비
        keys.add(key)
        key = keyboard.getKey()
    left_speed = right_speed = 0.0
    if ord('W') in keys or ord('w') in keys:
        left_speed = right_speed = SPEED
    elif ord('S') in keys or ord('s') in keys:
        left_speed = right_speed = -SPEED
    elif ord('A') in keys or ord('a') in keys:
        left_speed, right_speed = -SPEED, SPEED
    elif ord('D') in keys or ord('d') in keys:
        left_speed, right_speed = SPEED, -SPEED
    if ord('R') in keys or ord('r') in keys:
        x = y = theta = 0.0
        path_history[:] = [(0.0, 0.0)]
        obstacle_points.clear()
        obstacle_cells.clear()
        current_scan_points.clear()
        last_draw = -DRAW_PERIOD
        left_speed = right_speed = 0.0
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
    if pose_ready and robot.getTime() - last_draw >= DRAW_PERIOD:
        update_scan()
        draw_map()
        last_draw = robot.getTime()