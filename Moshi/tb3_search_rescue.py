"""
TurtleBot3 Burger - Autonomous Search & Rescue controller  (2026 PNU TECH WEEK, Webots R2025a, Python 3.9+, numpy, ultralytics)

Mission : START -> explore the UNKNOWN map -> avoid obstacles and walking people -> find EVERY target
          (apples; their number and positions are not known) -> reach each one ->
          come back to the start position and orientation -> FINISH

Sensors used (the rules): wheel encoders, gyro/IMU, 2D LiDAR (LDS-01), camera.   NOT used: GPS/GNSS, compass, ground truth.

Pipeline (Perception -> Localization -> Mapping -> Planning -> Control)
  Perception    camera + YOLO11n (models/YOLO/yolo11n.pt) -> apple / orange / sports-ball boxes -> colour check of the
                target colour -> distance from the floor-contact row and the box width.  No YOLO available: colour blobs.
  Localization  wheel odometry + gyro heading, corrected by LiDAR scan matching against the map
  Mapping       log-odds occupancy grid + a "camera coverage" grid (floor the camera has really looked at);
                walking people are tracked and erased from the map when they leave
  Global plan   frontier exploration, grid search on the inflated costmap (wide-clearance route first) + line-of-sight smoothing
  Local plan    Dynamic Window Approach with the real footprint and predicted people
  Supervisor    EXPLORE (+ 360-degree SWEEPs) -> APPROACH -> DWELL -> ... -> LOOK -> RETURN -> ALIGN -> FINISH
                + RECOVER and a fall-back chain so that the way home can never dead-lock

ON THE DAY change only the three lines under "ON THE DAY" below (time limit, target colour, number of targets).
"""
import math
from collections import deque

import numpy as np
from controller import Robot as _Robot
try:
    from controller import Node as _Node
except ImportError:                                 # pragma: no cover
    _Node = None
try:
    from controller import Supervisor as _Base      # only used to READ the true pose for the debug log
except ImportError:                                 # pragma: no cover
    _Base = _Robot

# ============================== CONFIG ======================================
DEBUG = True                  # write a compact once-per-half-second trace to debug_log.txt (next to this file)
DEBUG_TRUTH = False           # log the true pose if the robot node has supervisor TRUE (never used for control)
EXPECTED_TARGETS = None       # None = number of targets unknown -> explore everything reachable.
                              # int  = go home as soon as that many targets have been reached.
TIME_LIMIT = 600.0            # [s] competition time budget; the trip home starts early enough to fit
START_RADIUS = 0.22           # [m] "back at start" tolerance
ALIGN_AT_START = True         # finish by turning to the original start orientation
TARGET_REACH = 0.9            # [m] target counts as reached inside this distance (robot centre)
DWELL = 0.5                   # [s] stand still at each target ("identify")
WP_TOL, WP_TOL_LAST, WP_TOL_HOME = 0.30, 0.25, 0.15   # [m] a waypoint counts as passed inside this distance (last one / last one on the way home)
LOOKAHEAD, DENSE_STEP = 1.0, 0.4   # [m] look-ahead distance of the local planner (+1.3 s of speed), spacing of the waypoints
REACH_LENIENT = None          # [m] 'got as close as the map allows' also counts as reached inside this distance (None = 2 x TARGET_REACH)
APPROACH_MIN = None           # [m] closest distance to a target the robot centre may be sent to (None = footprint + 0.2, at most 0.62)
APPROACH_OUT = 0.1            # [m] the goal ring around a target reaches this far beyond TARGET_REACH

USE_RECOGNITION = True        # Webots camera recognition; False = rule-based colour detection
USE_SCAN_MATCH = True         # LiDAR scan matching to reduce odometry drift

# --- robot / motion   (EDIT THESE FOR ANOTHER ROBOT - everything else adapts to them)
WHEEL_R, TRACK_CMD, MAX_MOTOR = 0.10, 0.80, 20.0    # wheel radius [m], wheel separation used for commands [m], motor limit [rad/s]
HL, HW = 0.45, 0.36           # half length / half width of the robot footprint incl. wheels [m]

# --- hardware: names are looked up first; if a name is missing the controller searches by device type instead
LEFT_MOTORS = ["wheel_fl", "wheel_bl"]     # all motors on the left side (1 for a two-wheel robot)
RIGHT_MOTORS = ["wheel_fr", "wheel_br"]
HEADING_SOURCE = "auto"       # "auto" | "imu" | "gyro" | "compass" | "odometry"
                              # auto: InertialUnit (IMU) > Gyro > wheel odometry only.  Compass and GPS/GNSS are NOT used
USE_COMPASS = False           # the rules of the competition do not allow a compass: it is ignored unless this is set True
COMPASS_SIGN = 1              # +1 / -1: which way the compass counts.  A Gyro + Compass pair is compared while the robot turns
                              # at start-up: the sign is corrected automatically, an unusable compass is ignored
LIDAR_MOUNT = {"front_lidar": (0.42, 0.0), "back_lidar": (-0.42, math.pi)}   # name -> (x [m], yaw [rad]) in the robot frame
LIDAR_DEFAULT_MOUNT = (0.0, 0.0)           # for a LiDAR whose name is not listed above
CAMERA_NAME = None            # None = first Camera found
TARGET_RGB = (1.0, 0.0, 0.0)  # colour of the objects to rescue (used by the colour fallback and to filter recognition results)
COLOR_TOL = 0.30              # allowed hue difference to TARGET_RGB in radians (0.30 = 17 degrees); lower it if other things get picked up
COLOR_SAT = 0.45              # minimum saturation of a pixel to count as coloured (0 = grey, 1 = pure colour)
V_MAX, W_MAX = 1.2, 1.8       # [m/s], [rad/s]
A_V, A_BRAKE, A_W = 1.5, 3.0, 4.0     # accel / brake / angular accel limits
DW, SIM_DT, SIM_STEPS = 0.30, 0.25, 7 # DWA window [s], rollout step [s], rollout steps (1.75 s)
PEOPLE_STEPS = 12             # people are predicted this many rollout steps ahead (3 s)
MARGIN = 0.12                 # [m] minimum free space between footprint and obstacles (grows with speed)

# --- mapping / planning
RES = 0.10                    # [m] grid resolution
GX0, GY0 = -25.0, -25.0       # grid origin in the START frame (start = 0,0 facing +x)
GW, GH = 500, 500             # 50 m x 50 m around the start
LIDAR_USE = 12.0              # [m] max range used for mapping
COV_RANGE = 7.0               # [m] how far the camera is trusted to spot a target (camera coverage map)
LOOK_MIN = 500                # [cells] unseen floor (0.01 m2 each) that makes a far spot worth a 360-degree look
SWEEP_MIN = 600               # [cells] same, for a quick look on the spot while exploring
SWEEP_GAP = 5.0               # [m] minimum distance between two 360-degree looks
SWEEP_W = 1.8                 # [rad/s] turn rate during a 360-degree look
EXPLORE_MAX = None            # [s] hard cap on searching: after this the robot always heads home (None = EXPLORE_FRAC * TIME_LIMIT)
EXPLORE_FRAC = 0.55           # share of TIME_LIMIT that may be spent searching (the trip home is budgeted separately)
TRAVEL_V = None               # [m/s] typical driving speed used to estimate the way home (None = the lower of 0.6 and 85% of V_MAX)
NOVELTY_WIN = 40.0            # [s] go home when the map, camera coverage and target list stop growing this long
LOOK_BUDGET = 60.0            # [s] longest 'go and look at unseen areas' phase (restarts when a new target is found)
FINE_MIN = 50                 # [cells] short-sighted camera only: even a small unseen corner (0.5 m2) is worth a look before going home
FINE_BUDGET = 90.0            # [s] longest 'look into the small corners' phase
FINE_MIN_OPEN = 150           # [cells] cameras that see far (recognition / depth): only corners of at least 1.5 m2 are re-visited
FINE_BUDGET_OPEN = 60.0       # [s] ... and for a shorter time
SELF_MASK_M = 0.75            # [m] rays already blocked this close at start-up = robot itself (0 = off)
CAM_X, CAM_Z = 0.44, 0.42     # [m] camera position in the robot frame
CAM_PITCH_PX = 0.0            # [px] image row of the horizon below the image centre (a robot that leans forward a little sees the world tilted); refined online from the object sizes
CAM_SYNC = False              # update the camera in the step in which a new picture arrives (False = every 4th step)
CAM_LAG = 0.0                 # [s] the camera image is this much older than the robot pose: the heading is corrected by yaw_rate * CAM_LAG

# --- YOLO object detector (Ultralytics YOLO11n, runs on the CPU).  Used for the targets when the package and the model file are found;
#     otherwise the rule-based colour detection above is used.
USE_YOLO = False               # True = detect the targets with YOLO (apples / balls).  Off here: the boxes of the MiR / practice worlds are not COCO objects
YOLO_MODEL = None             # path to yolo11n.pt.  None = look for ../../models/YOLO/yolo11n.pt (the competition layout), then next to this file
YOLO_CLASSES = (32, 47, 49)   # COCO classes accepted as a target: 32 sports ball, 47 apple, 49 orange  (a small apple is often called 'sports ball').  None = all
YOLO_CONF = 0.20              # minimum confidence
YOLO_IMGSZ = 640              # network input size (a 10 cm apple is only ~15 px wide at 3.5 m: do not lower it)
YOLO_RANGE = 4.0              # [m] detections farther than this are ignored (their distance is too uncertain)
YOLO_COV_RANGE = 3.0          # [m] floor closer than this counts as 'looked at' for the coverage map (YOLO finds a 10 cm apple reliably up to about here)
YOLO_PERIOD = 0.5             # [s] minimum time between two inferences (0.25 s while the robot turns)
YOLO_COLOR_TOL = 0.45         # [rad] allowed hue difference between the detected object and the target colour (26 degrees)
YOLO_COLOR_MIN = 0.20         # this share of the object's pixels must have the target colour
OBJ_SIZE = 0.10               # [m] real diameter of a target: used to judge the distance from the box width and to reject wrong-sized objects (None = off)
OBJ_HALF = 0.05               # [m] target radius: the target centre is this far behind the surface the camera sees
TARGET_COLOR = None           # colour(s) to rescue by name: "red" "green" "orange" "purple" or a tuple of names, "any" = every colour.
                              # None = use TARGET_RGB above.  Enter the colour that is announced on the day here.
GATE_MIN, GATE_K = 1.2, 0.3   # two sightings closer than max(GATE_MIN, GATE_K * range) are the same target
GATE_REACHED = 0.6            # [m] extra gate around a target that is already done (stops duplicates after a push)
TARGET_LIDAR_VISIBLE = True   # False: targets are lower than the LiDAR plane (apples), so the LiDAR is not used to correct their distance
COLOR_TABLE = {"red": (1.0, 0.0, 0.0), "orange": (1.0, 0.73, 0.0), "purple": (0.56, 0.0, 1.0), "green": (0.59, 0.75, 0.28),
               "yellow": (1.0, 0.9, 0.0), "blue": (0.0, 0.2, 1.0)}
# ============================================================================

# ======================= TURTLEBOT3 BURGER PROFILE =========================
# ---- ON THE DAY: edit these three -------------------------------------------------------------------------------
TIME_LIMIT = 600.0            # [s] time the organizers give for the mission (the robot plans its trip home with it)
TARGET_COLOR = "red"          # colour to rescue: "red" "green" "orange" "purple", several as ("red", "orange"), or "any"
EXPECTED_TARGETS = 2          # two red apples: go home as soon as both are reached (None = unknown number -> explore everything)
# ---- robot: TurtleBot3 Burger (Webots) --------------------------------------------------------------------------
LEFT_MOTORS, RIGHT_MOTORS = ["left wheel motor"], ["right wheel motor"]
WHEEL_R, TRACK_CMD, MAX_MOTOR = 0.033, 0.160, 6.67
HL, HW = 0.105, 0.105
LIDAR_MOUNT = {"LDS-01": (-0.032, 0.0)}
LIDAR_DEFAULT_MOUNT = (-0.032, 0.0)
CAM_X, CAM_Z = 0.02, 0.073
CAM_SYNC, CAM_LAG = True, 0.064
CAM_PITCH_PX = 4.5            # the Burger leans forward a little: its horizon is 4-5 rows below the middle of the picture
V_MAX, W_MAX = 0.22, 2.0
A_V, A_BRAKE, A_W = 0.5, 1.0, 3.0
MARGIN = 0.10
TARGET_REACH = 0.32           # [m] robot centre to apple centre (the robot body touches an apple at 0.16 m)
REACH_LENIENT = 0.42
APPROACH_MIN, APPROACH_OUT = 0.20, 0.03
START_RADIUS = 0.15
SELF_MASK_M = 0.0
LIDAR_USE = 3.5
TARGET_LIDAR_VISIBLE = False  # apples are lower than the LiDAR plane
GATE_MIN, GATE_K, GATE_REACHED = 0.30, 0.12, 0.30
EXPLORE_FRAC = 0.80
WP_TOL, WP_TOL_LAST, WP_TOL_HOME = 0.15, 0.08, 0.06
LOOKAHEAD, DENSE_STEP = 0.5, 0.2
NOVELTY_WIN = 100.0           # [s] go home when nothing new was seen for this long (slow robot: long)
LOOK_BUDGET = 150.0
FINE_BUDGET_OPEN = 120.0
USE_YOLO = True               # needs the ultralytics package and models/YOLO/yolo11n.pt; otherwise the colour detection is used
# ===========================================================================
import os as _os
import re as _re
import sys as _sys
# Overrides without editing this file:  Robot node field  controllerArgs [ "WHEEL_R=0.08" "HW=0.26" ]   (used by the practice robots)
# or, for experiments, the environment variable  MIR_CFG="USE_SCAN_MATCH=False,V_MAX=0.8"
_over = list(_sys.argv[1:]) + _re.split(r",(?=\s*[A-Za-z_][A-Za-z_0-9]*\s*=)", _os.environ.get("MIR_CFG", ""))
for _kv in _over:
    if "=" in _kv:
        _k, _v = _kv.split("=", 1)
        if _k.strip() in globals():
            globals()[_k.strip()] = eval(_v)
if isinstance(TARGET_COLOR, str) and TARGET_COLOR.lower() != "any":
    TARGET_COLOR = (TARGET_COLOR,)
if TARGET_COLOR is None:
    TARGET_LIST = [tuple(TARGET_RGB)]
elif isinstance(TARGET_COLOR, str):                     # "any"
    TARGET_LIST = []
else:
    TARGET_LIST = [tuple(COLOR_TABLE[str(n).lower()]) if str(n).lower() in COLOR_TABLE else tuple(n) if not isinstance(n, str) else tuple(TARGET_RGB) for n in TARGET_COLOR]
    TARGET_RGB = TARGET_LIST[0]
LEVELS = (2.0 * HW, 1.6 * HW, 1.3 * HW)          # obstacle inflation levels [m], widest first (scale with the robot half-width)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def approach(cur, target, rate):
    return cur + clip(target - cur, -rate, rate)


def dilate(mask, rc):
    """Binary dilation with a disc of radius rc cells."""
    if rc <= 0:
        return mask.copy()
    W, H = mask.shape
    out = np.zeros_like(mask)
    for dx in range(-rc, rc + 1):
        for dy in range(-rc, rc + 1):
            if dx * dx + dy * dy > rc * rc:
                continue
            out[max(dx, 0):W + min(dx, 0), max(dy, 0):H + min(dy, 0)] |= \
                mask[max(-dx, 0):W + min(-dx, 0), max(-dy, 0):H + min(-dy, 0)]
    return out


def count_nb(mask, r):
    """Number of True cells in the (2r+1)^2 neighbourhood."""
    W, H = mask.shape
    out = np.zeros((W, H), np.int16)
    m = mask.astype(np.int16)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            out[max(dx, 0):W + min(dx, 0), max(dy, 0):H + min(dy, 0)] += \
                m[max(-dx, 0):W + min(-dx, 0), max(-dy, 0):H + min(-dy, 0)]
    return out


def box_sum(mask, r):
    """Sum of mask over the (2r+1)^2 box around every cell (integral image)."""
    W, H = mask.shape
    c = np.zeros((W + 1, H + 1), np.int32)
    c[1:, 1:] = mask.astype(np.int32).cumsum(0).cumsum(1)
    i, j = np.arange(W), np.arange(H)
    i0, i1 = np.clip(i - r, 0, W), np.clip(i + r + 1, 0, W)
    j0, j1 = np.clip(j - r, 0, H), np.clip(j + r + 1, 0, H)
    return c[np.ix_(i1, j1)] - c[np.ix_(i0, j1)] - c[np.ix_(i1, j0)] + c[np.ix_(i0, j0)]


def line_free(trav, a, b):
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    xs = np.rint(np.linspace(a[0], b[0], n)).astype(int)
    ys = np.rint(np.linspace(a[1], b[1], n)).astype(int)
    return bool(trav[xs, ys].all())


def smooth_cells(path, trav):
    """Greedy line-of-sight shortcutting of a grid path."""
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = min(len(path) - 1, i + 90)
        while j > i + 1 and not line_free(trav, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def densify(origin, pts, step=None):
    step = DENSE_STEP if step is None else step
    out, prev = [], origin
    for p in pts:
        d = math.hypot(p[0] - prev[0], p[1] - prev[1])
        n = int(d / step)
        for k in range(1, n + 1):
            f = k / (n + 1.0) if n else 1.0
            out.append((prev[0] + (p[0] - prev[0]) * f, prev[1] + (p[1] - prev[1]) * f))
        out.append(p)
        prev = p
    return out


def polyline_len(pts):
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts[:-1], pts[1:])))


def maxfilt(A, r):
    """Maximum over a (2r+1)x(2r+1) neighbourhood (separable)."""
    B = A.copy()
    for k in range(1, r + 1):
        np.maximum(B[k:, :], A[:-k, :], out=B[k:, :])
        np.maximum(B[:-k, :], A[k:, :], out=B[:-k, :])
    C = B.copy()
    for k in range(1, r + 1):
        np.maximum(C[:, k:], B[:, :-k], out=C[:, k:])
        np.maximum(C[:, :-k], B[:, k:], out=C[:, :-k])
    return C


class Lidar:
    def __init__(self, dev, mx, myaw, dt):
        dev.enable(dt)
        self.dev, self.mx, self.myaw = dev, mx, myaw
        self.n, self.fov, self.max = dev.getHorizontalResolution(), dev.getFov(), dev.getMaxRange()
        self.base = np.full(self.n, np.inf, np.float32)     # self-hit baseline
        self.set_order(True)
        self.r = np.full(self.n, self.max, np.float32)
        self.valid = np.ones(self.n, bool)
        self.dyn = np.zeros(self.n, bool)                   # rays that hit something moving

    def set_order(self, left_first):
        self.left_first = left_first
        i = np.arange(self.n)
        h = self.fov / 2
        self.ang_local = (h - i * self.fov / (self.n - 1)) if left_first else (-h + i * self.fov / (self.n - 1))
        self.ang = self.ang_local + self.myaw               # robot frame
        self.cos, self.sin = np.cos(self.ang), np.sin(self.ang)

    def read(self):
        r = np.array(self.dev.getRangeImage(), dtype=np.float32)
        r[~np.isfinite(r)] = self.max
        px, py = self.mx + r * self.cos, r * self.sin               # hit point in the robot frame
        inside = (np.abs(px) < HL) & (np.abs(py) < HW)              # inside our own footprint = we are seeing ourselves
        self.valid = (r >= 0.06) & (np.abs(r - self.base) > 0.03) & ~inside   # (e.g. the other LiDAR at 0.8 m)
        r[~self.valid] = self.max
        self.r = r


class Mapper:
    def __init__(self):
        self.L = np.zeros((GW, GH), np.float32)             # log-odds, exactly 0 = unknown
        self.ts = np.arange(0.15, LIDAR_USE, 0.08)
        self.blk = np.zeros((GW, GH), bool)                 # blacklisted frontier area
        self.obs_pts = []                                   # centres of small objects below the LiDAR plane
        self.cov = np.zeros((GW, GH), bool)                 # floor the CAMERA has looked at
        self.vpvis = np.zeros((GW, GH), bool)               # floor that was in open view of a place where we did a 360-degree look
        self.cx = GX0 + (np.arange(GW) + 0.5) * RES
        self.cy = GY0 + (np.arange(GH) + 0.5) * RES

    def cell(self, x, y):
        return int((x - GX0) / RES), int((y - GY0) / RES)

    def center(self, ix, iy):
        return GX0 + (ix + 0.5) * RES, GY0 + (iy + 0.5) * RES

    def inside(self, ix, iy):
        return 4 <= ix < GW - 4 and 4 <= iy < GH - 4

    def update(self, sx, sy, yaw, lid):
        a = yaw + lid.ang
        r = np.minimum(lid.r, LIDAR_USE)
        hit = lid.valid & (lid.r < min(lid.max, LIDAR_USE) - 0.05)
        ca, sa = np.cos(a), np.sin(a)
        T = self.ts[None, :]
        m = (T < (r[:, None] - 0.5 * RES)) & lid.valid[:, None]
        ix = ((sx + T * ca[:, None] - GX0) / RES).astype(np.int32)
        iy = ((sy + T * sa[:, None] - GY0) / RES).astype(np.int32)
        m &= (ix >= 0) & (ix < GW) & (iy >= 0) & (iy < GH)
        self.L[ix[m], iy[m]] -= 0.35
        hx = ((sx + r * ca - GX0) / RES).astype(np.int32)
        hy = ((sy + r * sa - GY0) / RES).astype(np.int32)
        ok = hit & (hx >= 0) & (hx < GW) & (hy >= 0) & (hy < GH)
        self.L[hx[ok], hy[ok]] += np.where(lid.dyn, 0.25, 1.0)[ok]     # moving things leave little trace
        np.clip(self.L, -4.0, 4.0, out=self.L)

    def mark_cov(self, ox, oy, yaw, offs, rng):
        """Camera rays (bearings offs relative to yaw, lengths rng) mark the floor they looked at."""
        a = yaw + offs
        T = np.arange(0.2, COV_RANGE, 0.1)[None, :]
        m = T < rng[:, None]
        ix = ((ox + T * np.cos(a)[:, None] - GX0) / RES).astype(np.int32)
        iy = ((oy + T * np.sin(a)[:, None] - GY0) / RES).astype(np.int32)
        m &= (ix >= 0) & (ix < GW) & (iy >= 0) & (iy < GH)
        self.cov[ix[m], iy[m]] = True

    def mark_vp(self, ox, oy, reach):
        """Cells in open line of sight (walls and shelves block) within `reach` of a place where we looked around."""
        a = np.arange(0.0, 2 * math.pi, 0.02)
        T = np.arange(0.1, reach, 0.1)[None, :]
        ix = ((ox + T * np.cos(a)[:, None] - GX0) / RES).astype(np.int32)
        iy = ((oy + T * np.sin(a)[:, None] - GY0) / RES).astype(np.int32)
        ok = (ix >= 0) & (ix < GW) & (iy >= 0) & (iy < GH)
        ix, iy = np.clip(ix, 0, GW - 1), np.clip(iy, 0, GH - 1)
        hit = (self.L[ix, iy] > 0.5) & ok
        seen = np.cumsum(hit, axis=1) - hit == 0                  # up to and including the first obstacle
        self.vpvis[ix[seen & ok], iy[seen & ok]] = True

    def window(self, pts, pad=10):
        """Bounding box (i0, i1, j0, j1) of everything known plus the given world points."""
        known = self.L != 0
        rows, cols = np.flatnonzero(known.any(axis=1)), np.flatnonzero(known.any(axis=0))
        il, jl = [], []
        if len(rows):
            il += [int(rows[0]), int(rows[-1])]
            jl += [int(cols[0]), int(cols[-1])]
        for x, y in pts:
            i, j = self.cell(x, y)
            il.append(i)
            jl.append(j)
        return (max(min(il) - pad, 2), min(max(il) + pad + 1, GW - 2),
                max(min(jl) - pad, 2), min(max(jl) + pad + 1, GH - 2))

    def derive(self, win, rx, ry, radius_m, unknown_ok=False):
        """Traversable + frontier masks inside the window."""
        i0, i1, j0, j1 = win
        L = self.L[i0:i1, j0:j1]
        occ = L > 1.0
        for ox, oy in self.obs_pts:                         # small objects the LiDAR cannot see (apples on the floor)
            a, b = self.cell(ox, oy)
            if i0 <= a < i1 and j0 <= b < j1:
                occ[a - i0, b - j0] = True
        ix, iy = self.cell(rx, ry)
        ix, iy = ix - i0, iy - j0
        a0, b0 = max(ix - 8, 0), max(iy - 8, 0)
        sub = occ[a0:ix + 9, b0:iy + 9]
        if sub.any():                                       # robot already close to something: shrink
            ii, jj = np.nonzero(sub)
            d_occ = float(np.hypot(ii + a0 - ix, jj + b0 - iy).min()) * RES
            radius_m = clip(min(radius_m, d_occ - 0.03), 0.35, radius_m)
        infl = dilate(occ, int(round(radius_m / RES)))
        free = (L <= 0.5) if unknown_ok else (L < -0.25)
        trav = free & ~infl
        trav[max(ix - 3, 0):ix + 4, max(iy - 3, 0):iy + 4] = True     # always let the robot leave its spot
        trav[[0, -1], :] = False
        trav[:, [0, -1]] = False
        unk = L == 0
        un = np.zeros_like(unk)
        un[1:, :] |= unk[:-1, :]
        un[:-1, :] |= unk[1:, :]
        un[:, 1:] |= unk[:, :-1]
        un[:, :-1] |= unk[:, 1:]
        fr = trav & un & ~self.blk[i0:i1, j0:j1]
        if fr.any():
            fr &= count_nb(fr, 2) >= 4                      # ignore isolated 1-2 cell "frontiers"
        return trav, fr

    def unseen_around(self, x, y, rng=COV_RANGE):
        """Known-free floor within `rng` that the camera has never looked at (number of cells)."""
        R = int(rng / RES)
        ix, iy = self.cell(x, y)
        a0, b0 = max(ix - R, 0), max(iy - R, 0)
        a1, b1 = min(ix + R + 1, GW), min(iy + R + 1, GH)
        un = (self.L[a0:a1, b0:b1] < -0.25) & ~self.cov[a0:a1, b0:b1]
        ii = (np.arange(a0, a1) - ix)[:, None]
        jj = (np.arange(b0, b1) - iy)[None, :]
        return int((un & (ii * ii + jj * jj <= R * R)).sum())

    def blacklist(self, x, y, r=7):
        ix, iy = self.cell(x, y)
        self.blk[max(ix - r, 0):ix + r + 1, max(iy - r, 0):iy + r + 1] = True

    @staticmethod
    def bfs(trav, start, goal):
        """8-connected BFS from start (i, j) to the nearest goal cell (arrays are window-local)."""
        W, H = trav.shape
        tv = bytearray(trav.astype(np.uint8).ravel().tobytes())
        gv = bytearray(goal.astype(np.uint8).ravel().tobytes())
        s = start[0] * H + start[1]
        prev = [-2] * (W * H)
        prev[s] = -1
        nb = (1, -1, H, -H, H + 1, H - 1, -H + 1, -H - 1)
        dq = deque([s])
        while dq:
            u = dq.popleft()
            if gv[u] and u != s:
                path = []
                while u != -1:
                    path.append((u // H, u % H))
                    u = prev[u]
                return path[::-1]
            for d in nb:
                v = u + d
                if 0 <= v < W * H and prev[v] == -2 and tv[v]:
                    prev[v] = u
                    dq.append(v)
        return None

    def dump_pgm(self, path, robot, path_pts, extra=()):
        """Debug picture around the known area: free=254, unknown=128, occupied=0, camera-unseen free=230,
        path=160, robot=60 (5x5), extra points=100."""
        win = self.window([robot] + list(path_pts) + list(extra), pad=5)
        i0, i1, j0, j1 = win
        L = self.L[i0:i1, j0:j1]
        img = np.full(L.shape, 128, np.uint8)
        img[L < -0.25] = 254
        img[(L < -0.25) & ~self.cov[i0:i1, j0:j1]] = 225
        img[L > 1.0] = 0
        for pts, val, r in ((path_pts, 160, 0), (extra, 100, 2), ([robot], 60, 2)):
            for x, y in pts:
                ix, iy = self.cell(x, y)
                if i0 + r <= ix < i1 - r and j0 + r <= iy < j1 - r:
                    img[ix - i0 - r:ix - i0 + r + 1, iy - j0 - r:iy - j0 + r + 1] = val
        with open(path, "wb") as f:
            f.write(b"P5\n%d %d\n255\n" % (img.shape[0], img.shape[1]))
            f.write(np.flipud(img.T).tobytes())

    def save_pgm(self, path, trail, targets=()):
        win = self.window([(0.0, 0.0)] + list(trail), pad=5)
        i0, i1, j0, j1 = win
        L = self.L[i0:i1, j0:j1]
        img = np.full(L.shape, 128, np.uint8)
        img[L < -0.25] = 254
        img[(L < -0.25) & ~self.cov[i0:i1, j0:j1]] = 225                # floor the camera never looked at: light grey
        img[L > 1.0] = 0
        for x, y in trail:
            ix, iy = self.cell(x, y)
            if i0 <= ix < i1 and j0 <= iy < j1:
                img[ix - i0, iy - j0] = 200
        for x, y in targets:
            ix, iy = self.cell(x, y)
            if i0 + 2 <= ix < i1 - 2 and j0 + 2 <= iy < j1 - 2:
                img[ix - i0 - 2:ix - i0 + 3, iy - j0 - 2:iy - j0 + 3] = 90
        with open(path, "wb") as f:
            f.write(b"P5\n%d %d\n255\n" % (img.shape[0], img.shape[1]))
            f.write(np.flipud(img.T).tobytes())


class Agent:
    def __init__(self):
        self.robot = _Base() if DEBUG_TRUTH else _Robot()
        r = self.robot
        self.dt = int(r.getBasicTimeStep())
        self.t = 0.0
        self.setup_hardware(r)
        self.cw, self.ch = self.cam.getWidth(), self.cam.getHeight()
        self.cam_f = (self.cw / 2) / math.tan(self.cam.getFov() / 2)
        # how far a target can be judged without a depth camera: the distance from the image row has an error of about
        # range^2 / (camera height * focal length) per pixel, so a low-resolution camera is only trusted at short range
        self.los_looks = self.depth is None and not self.use_recog
        self.explore_max = EXPLORE_MAX if EXPLORE_MAX is not None else EXPLORE_FRAC * TIME_LIMIT
        self.travel_v = TRAVEL_V if TRAVEL_V is not None else min(0.6, 0.85 * V_MAX)
        self.fine_min = FINE_MIN if self.los_looks else FINE_MIN_OPEN
        self.fine_budget = FINE_BUDGET if self.los_looks else FINE_BUDGET_OPEN
        self.det_range = COV_RANGE if (self.depth is not None or self.use_recog) else min(COV_RANGE, math.sqrt(0.7 * CAM_Z * self.cam_f))
        if self.yolo is not None:
            self.det_range = min(self.det_range, YOLO_COV_RANGE)
        self.need_hits = 3 if (self.use_recog or self.yolo is not None) else 6      # sightings before a target is believed

        r.step(self.dt)
        self.calibrate_lidars()
        self.prev_enc = [e.getValue() for e in self.enc]
        self.init_heading()
        self.yaw_bias = 0.0
        self.x = self.y = self.yaw = 0.0
        self.ybmax = 0.05 if self.hsrc == "imu" else 0.6      # how far scan matching may bend the heading
        self.map = Mapper()
        self.trail = []
        self.crumbs = [(0.0, 0.0)]                          # breadcrumb trail for the way back
        for _ in range(4):                                  # the start area, seen from the exact start pose
            for l in self.lidars:
                l.read()
                self.map.update(l.mx, 0.0, 0.0, l)
            r.step(self.dt)
        self.L0_init = self.map.L.copy()

        self.state, self.prev_state = "EXPLORE", "EXPLORE"
        self.path, self.goal_xy = [], None
        self.goal_pick_t = -99.0
        self.arrived = False
        self.last_plan, self.goal_t0 = -99.0, 0.0
        self.targets = []                                   # dicts: p, hits, seen, range, status
        self.cand = None
        self.approach_t0 = 0.0
        self.appr_anchor, self.appr_fb = None, False
        self.dwell_until = self.align_until = 0.0
        self.recover_until, self.recover_dir = 0.0, 1.0
        self.stuck_events = deque()
        self.hist = deque()
        self.blocked_since = self.idle_since = None
        self.retry_done = False
        self.vp_done, self.look_goal, self.look_t0 = [], None, 0.0
        self.vp_swept = []                                  # every place where a real 360-degree look happened
        self.vp_real = set()                                # viewpoints where a 360-degree look really happened
        self.sweep_turned = self.sweep_prev = self.sweep_until = 0.0
        self.last_sweep_t = -99.0
        self.blk_f = self.blk_b = False
        self.look_first = None
        self.depth_np = None
        self.ret_mode, self.ret_best, self.ret_prog_t = 0, 1e9, 0.0
        self.v = self.w = 0.0
        self.tracks = []                                    # moving obstacles (people)
        self.Lref, self.lref_t = self.map.L.copy(), 0.0     # map snapshot from ~1.5 s ago (what was free then)
        self.clear_now = 3.0
        self.step_i = 0
        self.log_t = 0.0
        self.Lref_nb2 = self.Lref_nb4 = None
        self.goal_is_look = False                           # current EXPLORE goal is a viewpoint, not a frontier
        self.contact_t = None                               # last time a person was pressed against the robot
        self.nov = deque()                                  # (t, camera-seen cells, free cells, targets known)
        self.L0 = self.L0_init                              # map of the start area, made before the robot moved
        self.anchor_t, self.anchor_checks = -9.0, 0
        self.w_wheel = 0.0                                  # turn rate the wheel commands imply
        self.yaw_rate = 0.0                                 # measured [rad/s] (IMU)
        self.v_enc = 0.0                                    # measured [m/s] (wheel encoders)
        self.wgain = 1.0                                    # measured turn rate / commanded turn rate (skid steering)
        self.flip_votes, self.swap_t = 0, -99.0
        self.near_home_t = None
        self.fine_look, self.fine_t0, self.contact_last = False, 0.0, -99.0
        self.lost_t, self.reloc_t = None, -99.0
        self.flung_t, self.flung_pending = -99.0, False
        self.prev_scan = None
        self._gw = deque()                                  # (dt, commanded w, measured dyaw) for the gain estimate
        self.scan_shift = 0.0                               # total distance moved by scan matching (debug)
        self.dbg_t = -1.0
        self.dbg_file = None
        self.me = self.truth0 = None
        self.persons, self.pers_hit, self.min_pdist = [], False, 9.0
        self.tnodes = []
        self.dump_t = -99.0
        if DEBUG:
            try:
                self.dbg_file = open("debug_log.txt", "w")
            except OSError:
                self.dbg_file = None
            if DEBUG_TRUTH:
                try:
                    self.me = self.robot.getSelf()
                    if self.me is not None:
                        p, o = self.me.getPosition(), self.me.getOrientation()
                        self.truth0 = (p[0], p[1], math.atan2(o[3], o[0]))
                        for k in range(1, 9):
                            nd = self.robot.getFromDef("PERSON_%d" % k)
                            if nd is not None:
                                self.persons.append(nd)
                            nd = self.robot.getFromDef("TARGET_%d" % k)
                            if nd is not None:
                                self.tnodes.append(nd)
                except Exception:
                    self.me = None

    # ------------------------------------------------------------ hardware discovery
    def devices_of(self, node_types):
        out = []
        for i in range(self.robot.getNumberOfDevices()):
            d = self.robot.getDeviceByIndex(i)
            if d.getNodeType() in node_types:
                out.append(d.getName())
        return out

    def setup_hardware(self, r):
        N = _Node
        names = set(self.devices_of({N.ROTATIONAL_MOTOR, N.LINEAR_MOTOR}) if N else [])
        left = [n for n in LEFT_MOTORS if n in names]
        right = [n for n in RIGHT_MOTORS if n in names]
        if not left or not right:                                   # other names: sort by 'left' / 'right' in the name, else by order
            allm = sorted(names)
            left = [n for n in allm if "left" in n.lower() or n.lower().startswith("l")]
            right = [n for n in allm if n not in left and ("right" in n.lower() or n.lower().startswith("r"))]
            if not left or not right:
                left, right = allm[:len(allm) // 2], allm[len(allm) // 2:]
        print("Motors: left %s, right %s" % (left, right))
        self.motors_l = [r.getDevice(n) for n in left]
        self.motors_r = [r.getDevice(n) for n in right]
        self.motors = self.motors_l + self.motors_r
        self.enc = []
        for m in self.motors:
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
            e = m.getPositionSensor()
            if e is None:
                e = r.getDevice(m.getName() + "_sensor")
            e.enable(self.dt)
            self.enc.append(e)
        self.max_motor = 0.995 * min(MAX_MOTOR, min(m.getMaxVelocity() for m in self.motors) if all(m.getMaxVelocity() > 0 for m in self.motors) else MAX_MOTOR)   # a hair below the limit: no "exceeds maxVelocity" warnings
        # heading sensors
        imus = self.devices_of({N.INERTIAL_UNIT}) if N else []
        gyros = self.devices_of({N.GYRO}) if N else []
        comps = self.devices_of({N.COMPASS}) if (N and USE_COMPASS) else []      # (off by default: not allowed)
        src = HEADING_SOURCE
        if src == "auto":
            src = "imu" if imus else ("gyro" if gyros else ("compass" if comps else "odometry"))
        if src == "imu" and not imus or src == "gyro" and not gyros or src == "compass" and not comps:
            src = "odometry"
        self.hsrc = src
        self.imu = r.getDevice(imus[0]) if src == "imu" else None
        self.gyro = r.getDevice(gyros[0]) if src == "gyro" else None
        self.compass = r.getDevice(comps[0]) if src in ("gyro", "compass") and comps else None
        for d in (self.imu, self.gyro, self.compass):
            if d is not None:
                d.enable(self.dt)
        print("Heading source: %s%s" % (src, " + compass" if src == "gyro" and self.compass else ""))
        # LiDARs
        lnames = self.devices_of({N.LIDAR}) if N else []
        self.lidars = []
        for n in lnames:
            mx, my = LIDAR_MOUNT.get(n, LIDAR_DEFAULT_MOUNT)
            self.lidars.append(Lidar(r.getDevice(n), mx, my, self.dt))
        if not self.lidars:
            raise RuntimeError("no Lidar found on this robot")
        angs = np.concatenate([np.arctan2(np.sin(l.ang), np.cos(l.ang)) for l in self.lidars])
        self.rear_cover = bool((np.abs(angs) > 2.6).any())          # can we see behind us? (else never reverse)
        # camera (+ optional range finder)
        period = self.dt * 4
        cams = self.devices_of({N.CAMERA}) if N else []
        self.cam = r.getDevice(CAMERA_NAME or cams[0])
        self.cam.enable(period)
        self.use_recog = USE_RECOGNITION and bool(self.cam.hasRecognition())
        if self.use_recog:
            self.cam.recognitionEnable(period)
        rfs = self.devices_of({N.RANGE_FINDER}) if N else []
        self.depth = r.getDevice(rfs[0]) if rfs else None
        if self.depth is not None:
            self.depth.enable(period)
        self.setup_yolo()
        print("Camera %s (%s), depth sensor: %s" % (self.cam.getName(), "recognition" if self.use_recog else ("YOLO11n" if self.yolo is not None else "colour detection"),
                                                    self.depth.getName() if self.depth else "none -> using LiDAR / floor geometry"))

    def setup_yolo(self):
        self.yolo, self.yolo_t, self.yolo_n = None, -99.0, 0
        self._cam_sig, self._cam_step = None, 0
        self.horizon_dy, self.hz_hist = CAM_PITCH_PX, deque(maxlen=25)
        self.objs = []
        self.objs_off_until = 0.0
        if not USE_YOLO or self.use_recog:
            return
        try:
            from ultralytics import YOLO
        except Exception as e:
            print("YOLO: the ultralytics package is not available (%s) -> using colour detection" % e)
            return
        here = _os.path.dirname(_os.path.abspath(__file__))
        cands = ([YOLO_MODEL] if YOLO_MODEL else []) + [
            _os.path.join(here, "..", "..", "models", "YOLO", "yolo11n.pt"), _os.path.join(here, "yolo11n.pt"),
            _os.path.join(here, "..", "..", "yolo11n.pt"), _os.path.join(_os.getcwd(), "..", "..", "models", "YOLO", "yolo11n.pt")]
        path = next((c for c in cands if c and _os.path.isfile(c)), None)
        if path is None:
            print("YOLO: yolo11n.pt not found (looked in %s) -> using colour detection" % ", ".join(_os.path.normpath(c) for c in cands if c))
            return
        try:
            m = YOLO(path)
            m.to("cpu")
            m.predict(source=np.zeros((self.cam.getHeight(), self.cam.getWidth(), 3), np.uint8), imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False, device="cpu")   # warm-up
            self.yolo = m
            print("YOLO: %s loaded (classes %s, conf >= %.2f, range <= %.1f m)" % (_os.path.normpath(path), YOLO_CLASSES, YOLO_CONF, YOLO_RANGE))
        except Exception as e:
            print("YOLO: could not run the model (%s) -> using colour detection" % e)

    def init_heading(self):
        self._gyro_yaw = 0.0
        self.odo_dyaw = 0.0
        self.compass_sign, self.compass_ok = 1, None                # ok: None = not yet compared with the gyro
        self._comp0 = self.compass_raw() if self.compass is not None else 0.0
        self._comp_prev, self._chk_g, self._chk_c = self._comp0, 0.0, 0.0
        self.yaw0 = self.imu.getRollPitchYaw()[2] if self.imu is not None else 0.0
        self._prev_yaw_raw = self.yaw0

    def compass_raw(self):
        v = self.compass.getValues()
        return COMPASS_SIGN * self.compass_sign * -math.atan2(v[1], v[0])   # heading of the robot against the compass' north

    def heading_step(self, dts):
        """Returns (raw heading, change since the last call) from whatever heading sensor this robot has."""
        if self.hsrc == "imu":
            raw = self.imu.getRollPitchYaw()[2]
            d = wrap(raw - self._prev_yaw_raw)
        elif self.hsrc == "gyro":
            d = self.gyro.getValues()[2] * dts
            self._gyro_yaw += d
            if self.compass is not None and self.compass_ok is not False:
                if self.compass_ok is None:                         # does the compass turn like the gyro?  (needs a real turn)
                    c = self.compass_raw()
                    self._chk_g += d
                    self._chk_c += wrap(c - self._comp_prev)
                    self._comp_prev = c
                    if abs(self._chk_g) > 2.0:
                        r = self._chk_c / self._chk_g
                        if 0.75 < abs(r) < 1.3:
                            self.compass_ok = True
                            if r < 0:                               # counts the other way: flip it
                                self.compass_sign, self._comp0 = -1, -self._comp0
                            print("Compass agrees with the gyro (ratio %.2f)%s" % (r, ", sign flipped" if r < 0 else ""))
                        else:
                            self.compass_ok = False
                            print("Compass does not follow the gyro (ratio %.2f) -> ignored, gyro only" % r)
                if self.compass_ok:                                 # slow pull towards the compass: cancels gyro drift
                    err = wrap(wrap(self.compass_raw() - self._comp0) - self._gyro_yaw)
                    self._gyro_yaw += 0.01 * err
            raw = self._gyro_yaw
        elif self.hsrc == "compass":
            raw = wrap(self.compass_raw() - self._comp0)
            d = wrap(raw - self._prev_yaw_raw)
        else:                                                       # wheels only
            d = self.odo_dyaw
            raw = self._prev_yaw_raw + d
        self._prev_yaw_raw = raw
        return raw, d

    # ------------------------------------------------------------ start-up calibration
    def calibrate_lidars(self):
        """1) find the true ray order with the point cloud, 2) remember rays that hit the robot itself."""
        r = self.robot
        for l in self.lidars:
            l.dev.enablePointCloud()
        r.step(self.dt)
        r.step(self.dt)
        for l in self.lidars:
            try:
                pts = l.dev.getPointCloud()
                x = np.array([p.x for p in pts], float)
                y = np.array([p.y for p in pts], float)
                z = np.array([p.z for p in pts], float)
            except Exception:
                x = y = z = np.array([])
            l.dev.disablePointCloud()
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (np.hypot(x, y) > 0.06)
            if len(x) != l.n or ok.sum() < 5:
                print("LiDAR %s: point cloud unusable, assuming index 0 = left" % l.dev.getName())
                continue
            idx = np.nonzero(ok)[0][:80]
            best = None
            for lf in (True, False):
                l.set_order(lf)
                for frame, meas in (("FLU", np.arctan2(y[idx], x[idx])), ("NUE", np.arctan2(-x[idx], -z[idx]))):
                    d = meas - l.ang_local[idx]
                    err = float(np.mean(np.abs(np.arctan2(np.sin(d), np.cos(d)))))
                    if best is None or err < best[0]:
                        best = (err, lf, frame)
            l.set_order(best[1] if best[0] < 0.2 else True)
            print("LiDAR %s: ray order %s (frame %s, err %.3f rad)"
                  % (l.dev.getName(), "left->right" if l.left_first else "right->left", best[2], best[0]))
        if SELF_MASK_M > 0:
            mins = [np.full(l.n, np.inf, np.float32) for l in self.lidars]
            for _ in range(6):
                r.step(self.dt)
                for l, mn in zip(self.lidars, mins):
                    raw = np.array(l.dev.getRangeImage(), np.float32)
                    raw[~np.isfinite(raw)] = np.inf
                    np.minimum(mn, raw, out=mn)
            for l, mn in zip(self.lidars, mins):
                l.base = np.where(mn < SELF_MASK_M, mn, np.inf).astype(np.float32)
                if np.isfinite(l.base).any():
                    print("LiDAR %s: %d rays hit something within %.2f m at start-up -> ignored as self-hits"
                          % (l.dev.getName(), int(np.isfinite(l.base).sum()), SELF_MASK_M))

    # ------------------------------------------------------------ localization
    def update_contact(self):
        """Is something touching the front / back of the robot? (wheels cannot push it through)"""
        P = self.scan_points(1.5)
        self.blk_f = self.blk_b = False
        if not len(P):
            return
        d = np.hypot(np.maximum(np.abs(P[:, 0]) - HL, 0.0), np.maximum(np.abs(P[:, 1]) - HW, 0.0))
        m = d < 0.035
        if m.any():
            self.blk_f = bool((P[m, 0] > 0.2).any())
            self.blk_b = bool((P[m, 0] < -0.2).any())

    def odometry(self):
        cur = [e.getValue() for e in self.enc]
        d = [c - p for c, p in zip(cur, self.prev_enc)]
        self.prev_enc = cur
        nl = len(self.motors_l)
        dl, dr = sum(d[:nl]) / nl, sum(d[nl:]) / max(len(d) - nl, 1)
        ds = WHEEL_R * (dl + dr) / 2
        self.odo_dyaw = WHEEL_R * (dr - dl) / TRACK_CMD
        dts = self.dt / 1000.0
        a = 0.15                                                    # low-pass the measured speeds
        self.v_enc += a * (ds / dts - self.v_enc)
        if (ds > 0 and self.blk_f) or (ds < 0 and self.blk_b):
            ds = 0.0                                                # pushing against an obstacle: the wheels slip, no motion
        raw, dyaw = self.heading_step(dts)
        self.yaw_rate += a * (dyaw / dts - self.yaw_rate)
        self._gw.append((dts, self.w_wheel, dyaw))                  # what we asked the wheels for vs what the body did
        if len(self._gw) > 64:                                      # ~1 s
            self._gw.popleft()
        if self.step_i % 32 == 0 and len(self._gw) >= 60:
            cmd = sum(g[0] * g[1] for g in self._gw)
            act = sum(g[2] for g in self._gw)
            if self.hsrc != "odometry" and abs(cmd) > 0.9 and self.clear_now > 0.15:
                if cmd * act > 0:
                    gain = clip(act / cmd, 0.35, 1.5)
                    self.wgain += 0.35 * (gain - self.wgain)
                    self.flip_votes = 0
                elif cmd * act < -0.5 * abs(cmd) and self.t - self.swap_t > 8.0:
                    self.flip_votes += 1                            # turning the wrong way, again and again: left / right are swapped
                    if self.flip_votes >= 3:
                        self.swap_sides()
        self.yaw = wrap(raw - self.yaw0 + self.yaw_bias)
        self.x += ds * math.cos(self.yaw)
        self.y += ds * math.sin(self.yaw)

    def match_points(self, maxn=120):
        P = []
        for l in self.lidars:
            m = l.valid & (l.r < 11.0)
            if m.any():
                P.append(np.stack([l.mx + l.r[m] * l.cos[m], l.r[m] * l.sin[m]], 1))
        if not P:
            return np.zeros((0, 2))
        P = np.concatenate(P)
        if len(P) > maxn:
            P = P[::int(math.ceil(len(P) / maxn))]
        return P

    def _match(self, P, d, dth, pen_xy, pen_th, Lmap=None):
        """Score every pose offset (dx, dy, dtheta) of the scan P against the grid. Returns best, zero-offset score."""
        Lm = self.map.L if Lmap is None else Lmap
        DX, DY, DT = (a.ravel() for a in np.meshgrid(d, d, dth, indexing="ij"))
        th = self.yaw + DT
        c, s = np.cos(th)[:, None], np.sin(th)[:, None]
        X = self.x + DX[:, None] + c * P[None, :, 0] - s * P[None, :, 1]
        Y = self.y + DY[:, None] + s * P[None, :, 0] + c * P[None, :, 1]
        ix = np.clip(((X - GX0) / RES).astype(np.int32), 0, GW - 1)
        iy = np.clip(((Y - GY0) / RES).astype(np.int32), 0, GH - 1)
        score = np.clip(Lm[ix, iy], -1.0, 3.0).sum(axis=1)
        score -= pen_xy * np.hypot(DX, DY) + pen_th * np.abs(DT)     # prefer staying with the odometry
        k = int(np.argmax(score))
        zero = int(np.argmin(np.hypot(DX, DY) + np.abs(DT)))
        return (DX[k], DY[k], DT[k]), float(score[k] - score[zero])

    def scan_match(self):
        """Correlative scan matching: small search around the odometry pose against the grid."""
        if self.blk_f or self.blk_b or self.clear_now < 0.12 or abs(self.yaw_rate) > 0.6:
            return                                                  # touching something / spinning: the scan is not trustworthy
        P = self.match_points()
        if len(P) < 40 or int((self.map.L > 1.0).sum()) < 150:
            return
        (dx, dy, dt), gain = self._match(P, np.arange(-0.25, 0.2501, 0.05), np.arange(-0.06, 0.0601, 0.03), 4.0, 10.0)
        if gain > 3.0:
            g = 0.8
            self.scan_shift += g * math.hypot(dx, dy)
            self.x += g * dx
            self.y += g * dy
            self.yaw_bias = clip(self.yaw_bias + g * dt, -self.ybmax, self.ybmax)     # the IMU heading is trusted more than the scan

    def pose_fit(self, P):
        """(scan points on cells the map knows, of those: points on occupied cells) at the current pose estimate."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        ix = np.clip(((self.x + c * P[:, 0] - s * P[:, 1] - GX0) / RES).astype(np.int32), 0, GW - 1)
        iy = np.clip(((self.y + s * P[:, 0] + c * P[:, 1] - GY0) / RES).astype(np.int32), 0, GH - 1)
        Lv = self.map.L[ix, iy]
        return int((np.abs(Lv) > 0.25).sum()), int((Lv > 0.5).sum())

    def remember_scan(self):
        """World-frame copy of the latest calm scan: what the robot saw just before it might get thrown."""
        P = self.match_points(200)
        if len(P) >= 60:
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            self.prev_scan = np.stack([self.x + c * P[:, 0] - s * P[:, 1], self.y + s * P[:, 0] + c * P[:, 1]], 1)

    def global_relocalize(self, why, ref=0):
        """The robot was thrown or pushed a long way (a person hit it, the wheels did not notice).  Two searches over +-6 m:
        A) the new scan against the picture of the scan taken just before the throw (works in unmapped areas too),
        B) the new scan against the whole map.  The nearest convincing fit wins."""
        self.reloc_t = self.t
        self.reloc_ok = False
        P = self.match_points(160)
        if len(P) < 60:
            return 0.0
        x0, y0, yb0 = self.x, self.y, self.yaw_bias
        nk0, no0 = self.pose_fit(P)                                     # how well the pose we still believe in fits
        no0 = max(no0, ref)                                             # (or the best small correction found so far)
        best = None
        cands = []
        if self.prev_scan is not None:
            Lt = np.zeros((GW, GH), np.float32)                        # the old scan as a tiny map
            ix = ((self.prev_scan[:, 0] - GX0) / RES).astype(np.int32)
            iy = ((self.prev_scan[:, 1] - GY0) / RES).astype(np.int32)
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    okk = (ix + ox >= 0) & (ix + ox < GW) & (iy + oy >= 0) & (iy + oy < GH)
                    Lt[ix[okk] + ox, iy[okk] + oy] = 3.0
            cands.append(("previous scan", Lt))
        cands.append(("map", self.map.L))
        for name, Lm in cands:
            self.x, self.y, self.yaw_bias = x0, y0, yb0
            (dx, dy, dt), gain = self._match(P, np.arange(-6.0, 6.001, 0.3), np.arange(-0.10, 0.1001, 0.05), 0.6, 6.0, Lm)
            self.x += dx
            self.y += dy
            self.yaw_bias = clip(self.yaw_bias + dt, -self.ybmax, self.ybmax)
            (dx2, dy2, dt2), _ = self._match(P, np.arange(-0.3, 0.3001, 0.05), np.arange(-0.06, 0.0601, 0.03), 1.0, 4.0, Lm)
            self.x += dx2
            self.y += dy2
            self.yaw_bias = clip(self.yaw_bias + dt2, -self.ybmax, self.ybmax)
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            ix = np.clip(((self.x + c * P[:, 0] - s * P[:, 1] - GX0) / RES).astype(np.int32), 0, GW - 1)
            iy = np.clip(((self.y + s * P[:, 0] + c * P[:, 1] - GY0) / RES).astype(np.int32), 0, GH - 1)
            Lv = Lm[ix, iy]
            on = int((Lv > 1.0).sum())                                  # points that land on a wall of the reference
            on5 = int((Lv > 0.5).sum())                                 # (same threshold as pose_fit, to compare with the old pose)
            # convincing: most points on walls; or, when the old pose clearly does not fit, at least half of them and
            # clearly more than the old pose (a busy room has people in the scan, so 60% is not always reachable)
            ok = gain >= max(30.0, 0.25 * len(P)) and on5 >= no0 + 0.05 * len(P) and \
                (on >= 0.6 * len(P) or (on >= 0.5 * len(P) and on5 >= no0 + 0.1 * len(P)))
            self.dlog("global relocalization vs %s: moved %.2f m, gain %.0f, %d/%d points on walls (old pose %d) -> %s"
                      % (name, math.hypot(self.x - x0, self.y - y0), gain, on, len(P), no0, "ok" if ok else "rejected"))
            if ok and (best is None or on > best[0] + 0.15 * len(P) and False):
                best = (on, self.x, self.y, self.yaw_bias, name)
            if ok:
                break
        if best is None:
            self.x, self.y, self.yaw_bias = x0, y0, yb0                 # no convincing place: keep the old estimate
            return 0.0
        _, self.x, self.y, self.yaw_bias, name = best
        self.reloc_ok = True
        moved = math.hypot(self.x - x0, self.y - y0)
        if moved > 0.3:
            self.log("pose jumped %.1f m (%s) -> relocalized using the %s" % (moved, why, name))
        if moved > 0.8:                                                 # sightings made with the wrong pose are worthless
            self.targets = [g for g in self.targets if not (g["status"] == "pending" and self.t - g["seen"] < 8.0)]
            self.path, self.last_plan = [], -99.0
        return moved

    def relocalize(self, rng=1.6, why="", use0=False):
        """Wide, coarse-to-fine scan match: repairs the pose after wheel slip or a long jam.
        use0=True matches against the map of the start area (built while the pose was still exact)."""
        Lm = self.L0 if (use0 and self.L0 is not None) else self.map.L
        P = self.match_points(160)
        if len(P) < 40 or int((Lm > 1.0).sum()) < 150:
            return 0.0
        step = 0.1 if use0 else 0.2
        (dx, dy, dt), gain = self._match(P, np.arange(-rng, rng + 1e-6, step), np.arange(-0.06, 0.0601, 0.03),
                                         1.0 if use0 else 1.5, 6.0, Lm)
        if gain < (6.0 if use0 else 8.0):
            return 0.0
        self.x += dx
        self.y += dy
        self.yaw_bias = clip(self.yaw_bias + dt, -self.ybmax, self.ybmax)
        (dx2, dy2, dt2), _ = self._match(P, np.arange(-0.2, 0.2001, 0.025 if use0 else 0.05),
                                         np.arange(-0.04, 0.0401, 0.02), 1.0, 4.0, Lm)
        self.x += dx2
        self.y += dy2
        self.yaw_bias = clip(self.yaw_bias + dt2, -self.ybmax, self.ybmax)
        moved = math.hypot(dx + dx2, dy + dy2)
        if moved > 0.3:
            self.log("pose corrected by %.2f m (%s)" % (moved, why))
        return moved

    def swap_sides(self):
        """The heading sensor says the robot turns the opposite way to what the wheels were told: LEFT_MOTORS and
        RIGHT_MOTORS are the wrong way round for this robot.  Swap them (commands and odometry together)."""
        info = {id(m): (e, pe) for m, e, pe in zip(self.motors, self.enc, self.prev_enc)}
        self.motors_l, self.motors_r = self.motors_r, self.motors_l
        self.motors = self.motors_l + self.motors_r
        self.enc = [info[id(m)][0] for m in self.motors]
        self.prev_enc = [info[id(m)][1] for m in self.motors]
        self._gw.clear()
        self.flip_votes, self.swap_t = 0, self.t
        self.log("turning the wrong way -> left and right motors swapped: left = %s, right = %s"
                 % ([m.getName() for m in self.motors_l], [m.getName() for m in self.motors_r]))

    # ------------------------------------------------------------ helpers
    def drive(self, v, w):
        ww = w / self.wgain                                         # skid steering turns slower than the wheels suggest
        vl = (v - ww * TRACK_CMD / 2) / WHEEL_R
        vr = (v + ww * TRACK_CMD / 2) / WHEEL_R
        big = max(abs(vl), abs(vr))
        if big > self.max_motor:                                    # keep the v / w ratio when the motors saturate
            vl, vr, ww = vl * self.max_motor / big, vr * self.max_motor / big, ww * self.max_motor / big
        self.w_wheel = ww
        for m in self.motors_l:
            m.setVelocity(vl)
        for m in self.motors_r:
            m.setVelocity(vr)
        self.v, self.w = v, w

    def brake(self):
        dt = 2 * self.dt / 1000.0
        self.drive(approach(self.v, 0.0, A_BRAKE * dt), approach(self.w, 0.0, A_W * dt))

    def dist(self, p):
        return math.hypot(p[0] - self.x, p[1] - self.y)

    def scan_points(self, rmax=4.0):
        """Valid LiDAR hits in the robot frame (x forward, y left), own footprint removed."""
        P = []
        for l in self.lidars:
            m = l.valid & (l.r < rmax)
            if m.any():
                P.append(np.stack([l.mx + l.r[m] * l.cos[m], l.r[m] * l.sin[m]], 1))
        if not P:
            return np.zeros((0, 2))
        P = np.concatenate(P)
        return P[~((np.abs(P[:, 0]) < HL - 0.04) & (np.abs(P[:, 1]) < HW - 0.04))]     # deep inside the footprint = self

    # ------------------------------------------------------------ moving obstacles
    def detect_dynamic(self):
        """LiDAR hits that land in space the map held as clearly free 1.5 s ago = something moving. Track them."""
        if self.t - self.lref_t > 1.5 or self.Lref_nb2 is None:
            self.Lref, self.lref_t = self.map.L.copy(), self.t
            self.Lref_nb2, self.Lref_nb4 = maxfilt(self.Lref, 2), maxfilt(self.Lref, 4)
        t = self.t
        if t < 2.5:                                                 # the map is still empty: everything looks 'new'
            for l in self.lidars:
                l.dyn = np.zeros(l.n, bool)
            return
        L, N2, N4 = self.Lref, self.Lref_nb2, self.Lref_nb4
        turning = abs(self.yaw_rate) > 0.4                          # bearing error grows with range while turning
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        pts = []
        for l in self.lidars:
            l.dyn = np.zeros(l.n, bool)
            m = np.flatnonzero(l.valid & (l.r < 6.0))
            if len(m) == 0:
                continue
            px, py = l.mx + l.r[m] * l.cos[m], l.r[m] * l.sin[m]
            wx, wy = self.x + c * px - s * py, self.y + s * px + c * py
            ix, iy = ((wx - GX0) / RES).astype(np.int32), ((wy - GY0) / RES).astype(np.int32)
            ok = (ix > 5) & (ix < GW - 6) & (iy > 5) & (iy < GH - 6)
            m, ix, iy, wx, wy = m[ok], ix[ok], iy[ok], wx[ok], wy[ok]
            if len(m) == 0:
                continue
            if turning:
                nb = N4[ix, iy]
            else:
                nb = np.where(l.r[m] > 4.5, N4[ix, iy], N2[ix, iy])
            d = (L[ix, iy] < -1.5) & (nb < -0.5)            # free around, free before: it just arrived
            l.dyn[m[d]] = True
            if d.any():
                pts.append(np.stack([wx[d], wy[d]], 1))
        if pts:
            P = np.concatenate(pts)[:200]
            cl = []
            for x, y in P:
                for c_ in cl:
                    if math.hypot(x - c_[0] / c_[2], y - c_[1] / c_[2]) < 0.5:
                        c_[0] += x
                        c_[1] += y
                        c_[2] += 1
                        break
                else:
                    cl.append([x, y, 1])
            for c_ in cl:
                if c_[2] < 3:
                    continue
                cx, cy = c_[0] / c_[2], c_[1] / c_[2]
                best, bd = None, 0.8
                for tr in self.tracks:
                    dd = math.hypot(cx - tr["x"], cy - tr["y"])
                    if dd < bd:
                        best, bd = tr, dd
                if best is None:
                    self.tracks.append(dict(x=cx, y=cy, hist=deque([(t, cx, cy)]), v=(0.0, 0.0), seen=t, born=t, disp=0.0))
                    self.dlog("new moving-object track at (%.1f, %.1f) n=%d dist=%.1f" % (cx, cy, c_[2], self.dist((cx, cy))))
                    continue
                best["x"], best["y"], best["seen"] = cx, cy, t
                h = best["hist"]
                h.append((t, cx, cy))
                while h and t - h[0][0] > 1.0:
                    h.popleft()
                if t - h[0][0] >= 0.4:
                    vx, vy = (cx - h[0][1]) / (t - h[0][0]), (cy - h[0][2]) / (t - h[0][0])
                    sp = math.hypot(vx, vy)
                    if sp > 1.6:
                        vx, vy = vx * 1.6 / sp, vy * 1.6 / sp
                    best["v"] = (vx, vy)
                    best["disp"] = max(best["disp"], math.hypot(cx - h[0][1], cy - h[0][2]))
        self.tracks = [tr for tr in self.tracks if t - tr["seen"] < 2.5]

    def moving_tracks(self):
        """People that are really walking: seen just now, tracked for a while, and they have actually travelled."""
        return [tr for tr in self.tracks if self.t - tr["seen"] < 0.4 and self.t - tr["born"] >= 0.2
                and math.hypot(*tr["v"]) > 0.2]

    def movers(self):
        """Walking people we have seen in the last 2.5 s: (x, y, vx, vy) extrapolated to now."""
        out = []
        for tr in self.tracks:
            age = self.t - tr["seen"]
            if age < 2.5 and math.hypot(*tr["v"]) > 0.2 and self.t - tr["born"] >= 0.2:
                out.append((tr["x"] + tr["v"][0] * age, tr["y"] + tr["v"][1] * age, tr["v"][0], tr["v"][1]))
        return out

    def people_mask(self, X, Y):
        """Cells a walking person is expected to occupy (or come close to) during the next ~3.5 s."""
        mv = self.movers()
        if not mv:
            return None
        m = np.zeros(np.broadcast(X, Y).shape, bool)
        for x, y, vx, vy in mv:
            if math.hypot(x - self.x, y - self.y) > 14.0:
                continue
            for tau in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
                m |= np.hypot(X - (x + vx * tau), Y - (y + vy * tau)) < 0.72 + 0.05 * tau
        r = np.hypot(X - self.x, Y - self.y) < 0.55                 # never wall the robot itself in
        m &= ~r
        return m

    # ------------------------------------------------------------ target perception
    def pixel_to_world(self, px, dep):
        dx = px - (self.cw - 1) / 2
        lat = -dx / self.cam_f * dep                        # +left
        yaw = self.yaw - CAM_LAG * self.yaw_rate
        c, s = math.cos(yaw), math.sin(yaw)
        cx, cy = self.x + CAM_X * c, self.y + CAM_X * s
        return cx + dep * c - lat * s, cy + dep * s + lat * c

    def camera_fresh(self):
        """True in the time step in which the camera delivers a NEW picture (its own period is not in phase with our loop:
        reading the picture up to three steps late would put every sighting made while turning 20 degrees off)."""
        try:
            sig = hash(bytes(self.cam.getImage()[::4099]))
        except Exception:
            return self.step_i % 4 == 0
        if sig != self._cam_sig or self.step_i - self._cam_step >= 8:
            self._cam_sig, self._cam_step = sig, self.step_i
            return True
        return False

    def update_camera(self):
        offs = np.arctan2(-(np.arange(0, self.cw, 2) - (self.cw - 1) / 2), self.cam_f)
        if self.depth is not None:
            self.depth_np = np.array(self.depth.getRangeImage(), np.float32).reshape(self.ch, self.cw)
            band = self.depth_np[self.ch // 4:self.ch // 2 + 1, ::2]          # rows at/above the horizon: no floor hits
            z = np.where(np.isfinite(band), band, 99.0).min(axis=0)
            rng = np.minimum(z / np.cos(offs), COV_RANGE)
        else:                                                               # no depth camera: what the LiDAR sees in that direction
            A, R = self.lidar_polar()
            rng = np.minimum(np.interp(offs, A, R, period=2 * math.pi), self.det_range) if len(A) else np.full(len(offs), self.det_range)
        yaw = self.yaw - CAM_LAG * self.yaw_rate
        c, s = math.cos(yaw), math.sin(yaw)
        self.map.mark_cov(self.x + CAM_X * c, self.y + CAM_X * s, yaw, offs, rng)
        self.update_target_belief()

    def lidar_polar(self):
        """All valid LiDAR hits as (bearing, range) seen from the robot centre, sorted by bearing."""
        A, R = [], []
        for l in self.lidars:
            px, py = l.mx + l.r * l.cos, l.r * l.sin
            A.append(np.arctan2(py, px))
            R.append(np.hypot(px, py))
        if not A:
            return np.zeros(0), np.zeros(0)
        A, R = np.concatenate(A), np.concatenate(R)
        o = np.argsort(A)
        return A[o], R[o]

    def color_close(self, rgb, tol=None, target=None):
        """True where a colour (0..1 floats or 0..255 arrays, last axis = R,G,B) has the hue of TARGET_RGB.

        Hue + saturation, so a lit / shaded / slightly tinted red box still matches while grey walls,
        brown wood, skin and blue clothes do not.  tol = allowed hue difference in radians."""
        rgb = np.asarray(rgb, np.float32)
        mx = rgb.max(axis=-1)
        a = rgb[..., 0] - 0.5 * (rgb[..., 1] + rgb[..., 2])
        b = 0.8660254 * (rgb[..., 1] - rgb[..., 2])
        chroma = np.hypot(a, b)
        t = TARGET_RGB if target is None else target
        th = math.atan2(0.8660254 * (t[1] - t[2]), t[0] - 0.5 * (t[1] + t[2]))
        dh = np.abs(np.arctan2(np.sin(np.arctan2(b, a) - th), np.cos(np.arctan2(b, a) - th)))
        return (dh < (COLOR_TOL if tol is None else tol)) & (chroma > COLOR_SAT * (mx + 1e-6))

    def depth_at(self, px, py, win=3):
        depth = self.depth_np
        x0, x1 = max(int(px) - win, 0), min(int(px) + win + 1, self.cw)
        y0, y1 = max(int(py) - win, 0), min(int(py) + win + 1, self.ch)
        d = depth[y0:y1, x0:x1].ravel()
        d = d[np.isfinite(d) & (d > 0.1)]
        return float(np.median(d)) if len(d) else None

    def detect_targets(self):
        """All currently visible targets as (x, y, surface_range) in the START frame."""
        out = []
        if self.use_recog:
            for o in self.cam.getRecognitionObjects():
                px, py = o.getPositionOnImage()
                q = o.getPosition()                                 # object centre in the camera frame (x fwd, y left)
                dep = float(q[0])
                if not (0.1 < dep < 12.0):
                    continue
                try:                                                # only objects of the target colour (if the object tells its colours)
                    nc = int(o.getNumberOfColors())                 # getColors() is a raw C pointer: never iterate it without the count
                    cp = o.getColors()
                    cols = [float(cp[k]) for k in range(3 * nc)] if (nc > 0 and cp) else []
                    if len(cols) >= 3 and not any(self.color_close(cols[k:k + 3], 0.6) for k in range(0, len(cols) - 2, 3)):
                        continue
                except Exception:
                    pass
                c_, s_ = math.cos(self.yaw), math.sin(self.yaw)
                cx_, cy_ = self.x + CAM_X * c_, self.y + CAM_X * s_
                x, y = cx_ + q[0] * c_ - q[1] * s_, cy_ + q[0] * s_ + q[1] * c_
                near = self.lidar_range_toward(x, y)
                if DEBUG and self.step_i % 16 == 0:
                    ex = ""
                    if self.tnodes and self.truth0 is not None:
                        x0, y0, a0 = self.truth0
                        cc, ss = math.cos(-a0), math.sin(-a0)
                        best = None
                        for nd in self.tnodes:
                            p = nd.getPosition()
                            tx, ty = cc * (p[0] - x0) - ss * (p[1] - y0), ss * (p[0] - x0) + cc * (p[1] - y0)
                            tr = self.truth()
                            dd = math.hypot(tx - x, ty - y)
                            if best is None or dd < best[0]:
                                best = (dd, tx, ty, math.hypot(tx - tr[0], ty - tr[1]) if tr else -1)
                        ex = " | nearest true target (%.2f, %.2f) off by %.2f m, true range %.2f" % (best[1], best[2], best[0], best[3])
                    self.dlog("recognised object px=%.0f py=%.0f depth=%.2f -> (%.2f, %.2f) lidar range there %.2f (robot-target %.2f)%s"
                              % (px, py, dep, x, y, near, math.hypot(x - self.x, y - self.y), ex))
                out.append((x, y, dep))
            return out
        if self.yolo is not None:
            return self.detect_yolo()
        img = np.frombuffer(self.cam.getImage(), np.uint8).reshape(self.ch, self.cw, 4)   # rule-based colour (BGRA)
        rgbf = img[..., 2::-1].astype(np.float32)
        hit = self.color_close(rgbf) & (rgbf.sum(axis=-1) > 90)
        ys, xs = np.nonzero(hit)
        if DEBUG and self.step_i % 32 == 0:
            self.dlog("colour camera: %d target-coloured pixels | brightest pixel %s | mean %s" % (len(xs), rgbf.reshape(-1, 3)[rgbf.sum(axis=-1).argmax()].astype(int).tolist(), rgbf.reshape(-1, 3).mean(0).astype(int).tolist()))
            if _os.environ.get("MIR_DUMPIMG") and self.step_i % 320 == 0:
                open("cam_%05d.ppm" % self.step_i, "wb").write(b"P6 %d %d 255\n" % (self.cw, self.ch) + rgbf.astype(np.uint8).tobytes())
        if len(xs) < 8:
            return out
        cols = np.unique(xs)                                        # one blob per group of neighbouring columns
        cuts = np.nonzero(np.diff(cols) > 3)[0]
        groups = np.split(cols, cuts + 1)
        for gcols in groups:
            sel = (xs >= gcols[0]) & (xs <= gcols[-1])
            if sel.sum() < 8:
                continue
            gx, gy = xs[sel], ys[sel]
            px = float(gx.mean())
            bottom = int(gy.max())
            edge = gcols[0] <= 1 or gcols[-1] >= self.cw - 2
            if self.depth is not None and self.depth_np is not None:
                d = self.depth_np[gy, gx]
                z = CAM_Z + d * ((self.ch - 1) / 2 - gy) / self.cam_f
                keep = np.isfinite(d) & (d > 0.1) & (z > 0.08) & (z < 1.0)     # ignore the floor
                if keep.sum() < 8:
                    continue
                dep = float(np.median(d[keep]))
            else:                                                   # no depth camera: distance from where the box meets the floor
                below = bottom + 0.5 - ((self.ch - 1) / 2 + self.horizon_dy)
                if below < 0.5 or bottom >= self.ch - 1:
                    continue                                        # base not visible (too close / cut off)
                dep = CAM_Z * self.cam_f / below
                if dep > self.det_range:
                    continue                                        # too far to place it reliably: wait until we are closer
                if OBJ_SIZE and not edge:                           # an apple is small and round: fire extinguishers, doors, red furniture and
                    wpx = float(gx.max() - gx.min() + 1)            # things that are not standing on the floor fail one of these two tests
                    hpx = float(gy.max() - gy.min() + 1)
                    if not (0.55 < dep / (self.cam_f * OBJ_SIZE / wpx) < 1.8) or not (0.5 < hpx / wpx < 1.8):
                        continue
                x0, y0 = self.pixel_to_world(px, dep)
                near = self.lidar_range_toward(x0, y0)              # the LiDAR is far more exact: use it when it agrees
                dl = near * math.cos(math.atan2(-(px - (self.cw - 1) / 2), self.cam_f)) - CAM_X
                if TARGET_LIDAR_VISIBLE and abs(dl - dep) < max(0.9, 1.5 * dep * dep / (CAM_Z * self.cam_f)):
                    dep = dl
            if edge and self.cw - 1 - gcols[-1] < 2 and gcols[0] <= 1:
                continue
            if dep < 0.1 or dep > COV_RANGE + 3:
                continue
            if self.depth is None:                                  # colour + floor geometry only: be strict
                if dep > self.det_range + 0.4:
                    continue
                if self.contact_t is not None or self.t - self.contact_last < 1.5 or self.t - self.flung_t < 2.0:
                    continue                                        # being shoved: pose and camera tilt cannot be trusted
            x, y = self.pixel_to_world(px, dep + 0.2)
            if self.depth is None and not self.plausible_target(x, y):
                continue
            out.append((x, y, dep))
        return out

    def colour_match(self, rgb, tol):
        """True where a pixel has the hue of ANY colour we are asked to rescue (every colour when TARGET_COLOR = "any")."""
        if not TARGET_LIST:
            rgb = np.asarray(rgb, np.float32)
            mx = rgb.max(axis=-1)
            return (mx - rgb.min(axis=-1)) > COLOR_SAT * (mx + 1e-6)
        m = None
        for t in TARGET_LIST:
            c = self.color_close(rgb, tol, t)
            m = c if m is None else (m | c)
        return m

    def detect_yolo(self):
        """Targets found by YOLO11n: class filter -> target-colour check -> distance from the floor contact row and the box width."""
        out = []
        period = 0.25 if abs(self.yaw_rate) > 0.6 else YOLO_PERIOD
        if self.t - self.yolo_t < period - 1e-6:
            return out
        img = np.frombuffer(self.cam.getImage(), np.uint8).reshape(self.ch, self.cw, 4)
        rgbf = img[..., 2::-1].astype(np.float32)                  # RGB
        if TARGET_LIST:                                             # cheap pre-check: no pixel of the wanted colour anywhere -> nothing to find
            small = rgbf[::2, ::2]
            if int((self.colour_match(small, 0.6) & (small.sum(axis=-1) > 60)).sum()) < 3:
                return out
        self.yolo_t = self.t
        bgr = np.ascontiguousarray(img[..., :3])
        try:
            res = self.yolo.predict(source=bgr, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=0.5, verbose=False, device="cpu",
                                    classes=list(YOLO_CLASSES) if YOLO_CLASSES else None)[0]
        except Exception as e:
            self.dlog("YOLO failed: %s" % e)
            return out
        self.yolo_n += 1
        moving = self.contact_t is not None or self.t - self.contact_last < 1.5 or self.t - self.flung_t < 2.0
        for b in res.boxes:
            conf = float(b.conf[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            w, h = x2 - x1, y2 - y1
            why, dep, x, y = None, None, 0.0, 0.0
            if w < 4 or h < 4 or h / w > 2.2 or w / h > 2.2:
                why = "shape"
            elif x1 <= 1 or x2 >= self.cw - 2 or y2 >= self.ch - 2:
                why = "cut off"
            elif moving:
                why = "being shoved"
            if why is None:
                below = y2 - ((self.ch - 1) / 2 + self.horizon_dy)  # rows between the horizon and the point where the object touches the floor
                dg = CAM_Z * self.cam_f / below if below > 1.5 else None
                ds = self.cam_f * OBJ_SIZE / w if OBJ_SIZE else None
                if dg is None:
                    why = "not on the floor"
                elif ds is not None and not (0.55 < dg / ds < 1.8):
                    why = "wrong size (floor %.1f m, width %.1f m)" % (dg, ds)
                else:
                    if ds is not None:                              # both estimates, each weighted by how many pixels it rests on
                        wg, ws = (below / 2.0) ** 2, (w / 1.5) ** 2
                        dep = math.exp((wg * math.log(dg) + ws * math.log(ds)) / (wg + ws))
                        if 0.35 < ds < 2.2:                         # a close object of known size shows where the horizon really is
                            self.hz_hist.append(y2 - CAM_Z * self.cam_f / ds)
                            if len(self.hz_hist) >= 5:
                                self.horizon_dy = clip(float(np.median(self.hz_hist)) - (self.ch - 1) / 2, -25.0, 25.0)
                    else:
                        dep = dg
                    if dep > YOLO_RANGE:
                        why = "too far"
            if why is None:
                x, y = self.pixel_to_world(0.5 * (x1 + x2), dep)
                if not self.plausible_target(x, y):
                    why = "not in known free space"
            if why is None:
                self.note_object(x, y, dep)                         # any apple-like thing on the floor is an obstacle, whatever its colour
                if TARGET_LIST and not self.yolo_colour_ok(rgbf, x1, y1, x2, y2):
                    why = "colour"
                else:
                    out.append((x, y, max(0.1, dep - OBJ_HALF)))
            if DEBUG and (why is None or self.step_i % 8 == 0):
                self.dlog("YOLO %s %.2f box (%d,%d,%d,%d) -> %s%s" % (self.yolo.names.get(int(b.cls[0]), "?"), conf, x1, y1, x2, y2,
                                                                    ("target at %.2f m -> (%.2f, %.2f)" % (dep, x, y)) if why is None else "ignored: " + why,
                                                                    self.truth_note(x, y) if why is None else ""))
        return out

    def active_obs(self):
        """Known small objects to drive around.  While approaching a target, that target and its neighbours are left out
        (the robot has to stop ~0.3 m from the target, which may stand in a row of apples)."""
        if self.t < self.objs_off_until:                            # boxed in by them: for a moment, push through
            return []
        pts = [o["p"] for o in self.objs if o["n"] >= 3]
        if self.cand is not None and self.state == "APPROACH":
            cx, cy = self.cand["p"]
            pts = [p for p in pts if math.hypot(p[0] - cx, p[1] - cy) > 0.45]
        return pts

    def note_object(self, x, y, dep):
        """Objects below the LiDAR plane (apples) are not in the map: remember them, so that the planner drives around them."""
        wgt = 1.0 / (dep * dep)
        for o in self.objs:
            if math.hypot(o["p"][0] - x, o["p"][1] - y) < 0.30:
                sw = o["w"] + wgt
                o["p"] = ((o["p"][0] * o["w"] + x * wgt) / sw, (o["p"][1] * o["w"] + y * wgt) / sw)
                o["w"], o["n"] = sw, o["n"] + 1
                break
        else:
            self.objs.append(dict(p=(x, y), w=wgt, n=1))

    def truth_note(self, x, y):
        """Debug only: how far the nearest real target (DEF TARGET_n) is from a detection (needs supervisor TRUE + DEBUG_TRUTH)."""
        try:
            if not self.tnodes or self.truth0 is None:
                return ""
            x0, y0, a0 = self.truth0
            cc, ss = math.cos(-a0), math.sin(-a0)
            best = None
            for nd in self.tnodes:
                p = nd.getPosition()
                tx, ty = cc * (p[0] - x0) - ss * (p[1] - y0), ss * (p[0] - x0) + cc * (p[1] - y0)
                dd = math.hypot(tx - x, ty - y)
                if best is None or dd < best[0]:
                    best = (dd, tx, ty)
            return " | nearest real target (%.2f, %.2f) off by %.2f m" % (best[1], best[2], best[0])
        except Exception:
            return ""

    def yolo_colour_ok(self, rgbf, x1, y1, x2, y2):
        cx, cy, hw, hh = 0.5 * (x1 + x2), 0.5 * (y1 + y2), 0.35 * (x2 - x1), 0.35 * (y2 - y1)
        patch = rgbf[int(max(cy - hh, 0)):int(min(cy + hh + 1, self.ch)), int(max(cx - hw, 0)):int(min(cx + hw + 1, self.cw))]
        if patch.size == 0:
            return False
        return float(self.colour_match(patch, YOLO_COLOR_TOL).mean()) >= YOLO_COLOR_MIN

    def plausible_target(self, x, y):
        """A box we can see must be in open space that we know about: no wall between us and it, and free floor next to it."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        ox, oy = self.x + CAM_X * c, self.y + CAM_X * s
        d = math.hypot(x - ox, y - oy)
        L = self.map.L
        for k in range(int((d - 0.5) / 0.05)):
            ix, iy = self.map.cell(ox + (x - ox) * (k * 0.05) / d, oy + (y - oy) * (k * 0.05) / d)
            if self.map.inside(ix, iy) and L[ix, iy] > 1.0:
                return False                                        # seen through a wall: mis-ranged or pose error
        ix, iy = self.map.cell(x, y)
        if not self.map.inside(ix - 8, iy - 8) or not self.map.inside(ix + 8, iy + 8):
            return False
        return bool((L[ix - 8:ix + 9, iy - 8:iy + 9] < -0.25).any())     # free floor within 0.8 m

    def lidar_range_toward(self, x, y):
        """Shortest LiDAR range within +-4 degrees of the direction towards world point (x, y) (9 m if none)."""
        a = wrap(math.atan2(y - self.y, x - self.x) - self.yaw)
        best = 9.0
        for l in self.lidars:
            d = np.abs(np.arctan2(np.sin(l.ang - a), np.cos(l.ang - a)))
            m = (d < 0.07) & l.valid
            if m.any():
                best = min(best, float(np.hypot(l.mx + l.r[m] * l.cos[m], l.r[m] * l.sin[m]).min()))
        return best

    def update_target_belief(self):
        for x, y, rng in self.detect_targets():
            best = None
            for tg in self.targets:
                gate = max(GATE_MIN, GATE_K * (rng if self.use_recog else max(rng, tg.get("best", 0.0))))   # far sightings are less accurate
                d = math.hypot(x - tg["p"][0], y - tg["p"][1])
                if d < gate + (GATE_REACHED if tg["status"] == "reached" else 0.0) and (best is None or d < best[0]):
                    best = (d, tg)
            if best is None:
                self.targets.append(dict(p=(x, y), hits=1, seen=self.t, range=rng, best=rng, status="pending"))
                continue
            tg = best[1]
            tg["seen"], tg["range"] = self.t, rng
            tg["hits"] += 1
            if tg["status"] != "reached":                           # a close sighting is worth more than a far one
                w = max(0.1, tg.get("best", rng) ** 2 / (tg.get("best", rng) ** 2 + rng ** 2)) if not self.use_recog else 0.3
                tg["p"] = ((1 - w) * tg["p"][0] + w * x, (1 - w) * tg["p"][1] + w * y)
            tg["best"] = min(tg.get("best", rng), rng)

    def n_reached(self):
        return sum(tg["status"] == "reached" for tg in self.targets)

    # ------------------------------------------------------------ global planning
    def goal_mask(self, kind, trav, fr, X, Y, win):
        rx, ry = self.x, self.y
        if kind == "LOOK":                                      # spots from which a lot of never-seen floor is in reach
            i0, i1, j0, j1 = win
            unseen = (self.map.L[i0:i1, j0:j1] < -0.25) & ~self.map.cov[i0:i1, j0:j1]
            lo, rad = (self.fine_min, 25) if self.fine_look else (LOOK_MIN, 40)
            sc = box_sum(unseen, rad)
            top = int(sc.max())
            if top < lo:
                return np.zeros_like(trav)
            g = trav & (sc >= max(lo, 0.4 * top))
            for vx, vy in self.vp_done:
                if (vx, vy) in self.vp_real:                    # a real look: only what it could see through open space is done
                    g &= (np.hypot(X - vx, Y - vy) > 1.5) & ~self.map.vpvis[i0:i1, j0:j1]
                else:
                    g &= np.hypot(X - vx, Y - vy) > SWEEP_GAP
            return g
        if kind == "EXPLORE":
            return fr & (np.hypot(X - rx, Y - ry) > 0.6)
        if kind == "APPROACH":
            dd = np.hypot(X - self.cand["p"][0], Y - self.cand["p"][1])
            g = trav & (dd > (APPROACH_MIN if APPROACH_MIN is not None else min(0.62, HL + 0.2))) & (dd < TARGET_REACH + APPROACH_OUT)      # (a small robot may stand closer to the box)
            if g.any() or not self.appr_fb:
                return g
            # nothing usable next to the target (unmapped / behind furniture): the nearest cell that is clearly CLOSER than we are now.
            # (never "any cell within 2 m": that includes our own spot and the path collapses to nothing -> the robot freezes)
            dr = math.hypot(rx - self.cand["p"][0], ry - self.cand["p"][1])
            return trav & (dd > min(0.45, HL + 0.1)) & (dd < min(2.0, dr - 0.3))
        dd = np.hypot(X, Y)                                     # RETURN
        g = trav & (dd < 0.8)
        return g if g.any() else trav & (dd < 1.6)

    def plan(self, kind, t, unknown_ok=False):
        if kind == "APPROACH" and not unknown_ok:
            # 1) a spot next to the target in mapped floor, 2) the same through not-yet-mapped floor (the camera sees further than the LiDAR
            # map reaches: drive there and let the scans fill it in), 3) at least a mapped spot clearly closer to the target than we are
            for uk, fb in ((False, False), (True, False), (False, True)):
                self.appr_fb = fb
                if self._plan(kind, t, uk):
                    return True
            return False
        self.appr_fb = False
        return self._plan(kind, t, unknown_ok)

    def _plan(self, kind, t, unknown_ok=False):
        self.last_plan = t
        self.map.obs_pts = self.active_obs()
        rx, ry = self.x, self.y
        if not self.map.inside(*self.map.cell(rx, ry)):
            return False
        extra = [(rx, ry), (0.0, 0.0)] + ([self.cand["p"]] if kind == "APPROACH" else [])
        win = self.map.window(extra)
        i0, i1, j0, j1 = win
        X = self.map.cx[i0:i1][:, None]
        Y = self.map.cy[j0:j1][None, :]
        six, siy = self.map.cell(rx, ry)
        start = (six - i0, siy - j0)

        found = []
        pmask = self.people_mask(X, Y)
        for use_people in ((True, False) if pmask is not None else (False,)):   # avoid people's paths if at all possible
            for lvl in LEVELS:                                  # widest clearance first
                trav, fr = self.map.derive(win, rx, ry, lvl, unknown_ok)
                if use_people:
                    trav = trav & ~pmask
                    fr = fr & ~pmask
                goal = self.goal_mask(kind, trav, fr, X, Y, win)
                lg = None
                if kind == "EXPLORE":                           # finish this room first: nearby unseen spots count too
                    lg = self.goal_mask("LOOK", trav, fr, X, Y, win) & (np.hypot(X - rx, Y - ry) < 6.0)
                    goal = goal | lg
                if not goal.any():
                    continue
                path = Mapper.bfs(trav, start, goal)
                if path:
                    self.goal_is_look = bool(lg is not None and lg[path[-1][0], path[-1][1]] and not fr[path[-1][0], path[-1][1]])
                    found.append((path, trav, fr, goal))
                    if len(found) == 2:
                        break
            if found:
                break
        if not found:
            self.dlog("plan(%s) found no route (window %s)" % (kind, win))
            return False
        path, trav, fr, gmask = found[0]
        if len(found) == 2 and len(path) * 1.0 > 1.35 * len(found[1][0]) + 20:   # wide route is a big detour
            path, trav, fr, gmask = found[1]

        dgoal = math.hypot(self.goal_xy[0] - rx, self.goal_xy[1] - ry) if self.goal_xy is not None else 0.0
        if kind == "EXPLORE" and self.goal_xy is not None and (dgoal > 1.0 or (t - self.goal_pick_t < 8.0 and dgoal > 0.35)):
            gx, gy = self.goal_xy                               # hysteresis: keep the old goal if it is still ahead of us and reasonable
            dg = np.hypot(X - gx, Y - gy)
            g2 = (fr | gmask) & (dg < (1.0 if dgoal > 1.0 else 0.5))
            if g2.any():                                        # commit to the cell of the old goal, not to whatever frontier is nearest to us
                ki = int(np.argmin(np.where(g2, dg, 1e9)))
                g2 = np.zeros_like(g2)
                g2.flat[ki] = True
                p2 = Mapper.bfs(trav, start, g2)
                if p2 and len(p2) <= 1.5 * len(path) + 30:
                    path = p2

        cells = smooth_cells(path, trav)
        pts = [self.map.center(i + i0, j + j0) for i, j in cells[1:]]
        if kind == "RETURN" and math.hypot(pts[-1][0], pts[-1][1]) > 0.05:
            pts.append((0.0, 0.0))
        self.path = densify((rx, ry), pts)
        newg = pts[-1]
        if kind == "EXPLORE" and (self.goal_xy is None or math.hypot(newg[0] - self.goal_xy[0], newg[1] - self.goal_xy[1]) > 0.5):
            self.goal_pick_t = t                                # a new goal was chosen: stick with it for a few seconds
        if kind == "EXPLORE" and (self.goal_xy is None or math.hypot(newg[0] - self.goal_xy[0], newg[1] - self.goal_xy[1]) > 1.0):
            self.goal_t0 = t                                    # a new goal: restart its chase timer
        self.goal_xy = newg
        self.dlog("plan(%s) ok: %d waypoints, goal (%.1f, %.1f), look-goal %s" % (kind, len(self.path), newg[0], newg[1], self.goal_is_look))
        if kind == "LOOK":
            self.look_goal = newg
        self.arrived = False
        return True

    def use_crumbs(self):
        c = self.crumbs
        i = min(range(len(c)), key=lambda k: math.hypot(c[k][0] - self.x, c[k][1] - self.y))
        self.path = list(c[i::-2]) + [(0.0, 0.0)]
        self.arrived = False
        self.last_plan = self.t

    def path_blocked(self):
        L = self.map.L
        for x, y in self.path[:25]:
            ix, iy = self.map.cell(x, y)
            if not self.map.inside(ix, iy):
                return True
            if (L[ix - 3:ix + 4, iy - 3:iy + 4] > 1.0).any():
                return True
        return False

    def plan_return(self, t):
        """Way home with a fallback chain: map path -> path through unknown -> breadcrumbs -> direct."""
        while self.ret_mode < 2:
            if self.plan("RETURN", t, unknown_ok=(self.ret_mode == 1)):
                return
            self.ret_mode += 1
            self.log("no safe map route home -> fallback level %d" % self.ret_mode)
        if self.ret_mode == 2:
            self.use_crumbs()
            return
        self.path, self.arrived, self.last_plan = [(0.0, 0.0)], False, t     # direct homing

    # ------------------------------------------------------------ local planning (DWA)
    def points_for_dwa(self):
        P = self.scan_points(3.2)
        obs = self.active_obs()
        if obs:
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            ex = []
            for ox, oy in obs:
                dx, dy = ox - self.x, oy - self.y
                if dx * dx + dy * dy < 3.2 * 3.2:
                    xr, yr = c * dx + s * dy, -s * dx + c * dy
                    ex += [(xr + 0.05 * math.cos(a), yr + 0.05 * math.sin(a)) for a in (0.0, 1.57, 3.14, 4.71)]
            if ex:
                P = np.vstack([P.reshape(-1, 2), np.array(ex, np.float32)])
        if len(P) > 120:                                            # one point per 5 cm cell keeps thin obstacles
            key = np.rint(P / 0.05).astype(np.int64)
            _, idx = np.unique(key[:, 0] * 100003 + key[:, 1], return_index=True)
            P = P[idx]
        if len(P) > 220:
            P = P[::int(math.ceil(len(P) / 220))]
        return P.astype(np.float32)

    def rollout(self, VV, WW, P, steps=SIM_STEPS):
        """Simulate constant (v, w) arcs. Returns xs, ys, th, dmin (footprint-to-obstacle distance)."""
        K = len(VV)
        wdt = (WW[:, None] * SIM_DT).astype(np.float32)
        th = np.cumsum(np.ones((K, steps), np.float32) * wdt, axis=1)
        mid = th - wdt / 2
        xs = np.cumsum(VV[:, None] * np.cos(mid) * SIM_DT, axis=1)
        ys = np.cumsum(VV[:, None] * np.sin(mid) * SIM_DT, axis=1)
        if len(P):
            dx = P[None, None, :, 0] - xs[:, :, None]
            dy = P[None, None, :, 1] - ys[:, :, None]
            cc, ss = np.cos(th)[:, :, None], np.sin(th)[:, :, None]
            xr, yr = dx * cc + dy * ss, -dx * ss + dy * cc
            d = np.hypot(np.maximum(np.abs(xr) - HL, 0.0), np.maximum(np.abs(yr) - HW, 0.0))
            dmin = d.min(axis=(1, 2))
        else:
            dmin = np.full(K, 5.0)
        return xs, ys, th, dmin

    def dwa(self, tgt, vmax):
        """Dynamic Window Approach. tgt = lookahead point in the START frame. Returns (v, w) or None."""
        c0, s0 = math.cos(self.yaw), math.sin(self.yaw)
        gx, gy = tgt[0] - self.x, tgt[1] - self.y
        tx, ty = c0 * gx + s0 * gy, -s0 * gx + c0 * gy          # target in robot frame
        P = self.points_for_dwa()
        if len(P):
            d_now = float(np.hypot(np.maximum(np.abs(P[:, 0]) - HL, 0.0), np.maximum(np.abs(P[:, 1]) - HW, 0.0)).min())
            fwd = P[(P[:, 0] > 0) & (np.abs(P[:, 1]) < HW + 0.05)]
            gap = float(fwd[:, 0].min() - HL) if len(fwd) else 9.0
        else:
            d_now, gap = 5.0, 9.0
        self.clear_now = d_now
        mt = self.moving_tracks()
        near_dyn = any(self.dist((tr["x"], tr["y"])) < 2.5 for tr in mt)
        if near_dyn:
            vmax = min(vmax, 0.8)
        elif any(self.dist((tr["x"], tr["y"])) < 5.0 for tr in mt):
            vmax = min(vmax, 0.95)                                  # somebody is walking around: no full speed
        bearing = abs(math.atan2(ty, tx))
        vmax = min(vmax, max(0.25, V_MAX * (1.6 - bearing)))        # turn first, then speed up
        vmax = min(vmax, math.sqrt(2 * 1.5 * max(gap - 0.15, 0.0)))     # always able to stop before what we see

        if d_now < MARGIN:
            vmax = min(vmax, 0.5)                                   # tight spot: slow and careful
        back_ok = self.rear_cover and (d_now < 0.30 or near_dyn)                          # squeezed, or a person is close: may reverse
        v_lo = max(-0.3 if back_ok else 0.0, self.v - A_BRAKE * DW)
        v_hi = max(v_lo, min(vmax, self.v + A_V * DW))
        vs = np.linspace(v_lo, v_hi, 7)
        ws = np.linspace(max(-W_MAX, self.w - A_W * DW), min(W_MAX, self.w + A_W * DW), 13)
        VV, WW = (a.ravel() for a in np.meshgrid(vs, ws, indexing="ij"))
        xs, ys, th, dmin = self.rollout(VV, WW, P)
        if mt:                                                      # where will the people be during each rollout?
            xp, yp, thp, _ = self.rollout(VV, WW, np.zeros((0, 2), np.float32), PEOPLE_STEPS)   # longer look-ahead for people
            taus = SIM_DT * np.arange(1, PEOPLE_STEPS + 1)
            cc, ss = np.cos(thp), np.sin(thp)
            for tr in mt:
                pxw, pyw = tr["x"] + tr["v"][0] * taus - self.x, tr["y"] + tr["v"][1] * taus - self.y
                pr, qr = c0 * pxw + s0 * pyw, -s0 * pxw + c0 * pyw
                dx, dy = pr[None, :] - xp, qr[None, :] - yp
                xr, yr = dx * cc + dy * ss, -dx * ss + dy * cc
                dd = np.hypot(np.maximum(np.abs(xr) - HL, 0.0), np.maximum(np.abs(yr) - HW, 0.0)) - 0.30 - 0.06 * taus[None, :]
                dmin = np.minimum(dmin, dd.min(axis=1))
        req = MARGIN + 0.08 * np.maximum(VV, 0.0)                   # more room needed at higher speed
        ok = dmin >= np.minimum(req, 0.8 * d_now)
        if d_now < MARGIN:
            ok = dmin >= 0.6 * d_now                                # squeezed: creep on, but do not push into the wall
        if not ok.any():
            if near_dyn:                                            # someone is coming: at least maximise the distance
                k = int(np.argmax(dmin))
                return float(VV[k]), float(WW[k])
            return None

        xf, yf, tf = xs[:, -1], ys[:, -1], th[:, -1]
        d0 = math.hypot(tx, ty) + 0.5
        progress = (d0 - np.hypot(xf - tx, yf - ty)) / d0
        ang = np.arctan2(ty - yf, tx - xf) - tf
        head = np.abs(np.arctan2(np.sin(ang), np.cos(ang))) / math.pi
        pp = np.array(self.path[:14]) if self.path else np.zeros((0, 2))
        if len(pp):
            px, py = pp[:, 0] - self.x, pp[:, 1] - self.y
            rxp, ryp = c0 * px + s0 * py, -s0 * px + c0 * py
            dpath = np.hypot(xf[:, None] - rxp[None, :], yf[:, None] - ryp[None, :]).min(axis=1)
        else:
            dpath = np.zeros(len(VV))
        score = (2.0 * progress + 1.2 * (1.0 - head) + 2.0 * np.minimum(dmin, 0.45) / 0.45
                 + 0.8 * VV / V_MAX - 0.2 * np.abs(WW - self.w) / W_MAX - 0.9 * np.minimum(dpath, 1.5))
        score[~ok] = -1e9
        k = int(np.argmax(score))
        return float(VV[k]), float(WW[k])

    def escape(self):
        """Recovery: the safest small move (back away / turn) judged with the same footprint check."""
        P = self.points_for_dwa()
        turn = 0.0
        if self.path:                                               # prefer turning towards where we want to go
            tgt = self.path[min(3, len(self.path) - 1)]
            turn = wrap(math.atan2(tgt[1] - self.y, tgt[0] - self.x) - self.yaw)
        VV = np.array([-0.3, -0.3, -0.3, 0.0, 0.0, 0.3, 0.3, 0.3])
        WW = np.array([0.0, 0.7, -0.7, 0.9, -0.9, 0.0, 0.7, -0.7])
        _, _, _, dmin = self.rollout(VV, WW, P)
        score = np.minimum(dmin, 0.25) + 0.05 * (VV < 0)            # prefer backing away when equally safe
        if not self.rear_cover:
            score = np.where(VV < 0, -1.0, score)                   # blind behind us: never reverse
        if abs(turn) > 0.8:
            score = score + 0.10 * (WW * turn > 0)
        k = int(np.argmax(score))
        return float(VV[k]), float(WW[k])

    def follow(self, vcap=V_MAX):
        if len(self.path) > 1:                                      # waypoints we have already passed are dropped
            n = min(len(self.path), 20)
            k = min(range(n), key=lambda i: self.dist(self.path[i]))
            if k > 0:
                del self.path[:k]
        last_r = WP_TOL_HOME if self.state == "RETURN" else WP_TOL_LAST
        while self.path and self.dist(self.path[0]) < (WP_TOL if len(self.path) > 1 else last_r):
            self.path.pop(0)
            if not self.path:
                self.arrived = True
        if not self.path:
            self.brake()
            return
        look = LOOKAHEAD + 1.3 * max(self.v, 0.0)
        tgt = self.path[-1]
        for p in self.path:
            if self.dist(p) >= look:
                tgt = p
                break
        vmax = min(vcap, 0.25 + 0.7 * self.dist(self.path[-1]))     # slow down when arriving
        res = self.dwa(tgt, vmax)
        dt = 2 * self.dt / 1000.0                                   # control period (every 2nd step)
        if res is None:                                             # nothing safe: brake and wait
            self.blocked_since = self.blocked_since or self.t
            self.brake()
            return
        self.blocked_since = None
        v = approach(self.v, res[0], (A_V if res[0] > self.v else A_BRAKE) * dt)
        w = approach(self.w, res[1], A_W * dt)
        self.drive(v, w)

    # ------------------------------------------------------------ supervisor
    def stuck(self):
        if self.hist and self.t - self.hist[-1][0] > 0.4:            # was in SWEEP / DWELL / RECOVER: start fresh
            self.hist.clear()
            self.idle_since = self.blocked_since = None
        self.hist.append((self.t, self.x, self.y, self.v > 0.05, self.yaw, self.v))
        while self.hist and self.t - self.hist[0][0] > 8.0:
            self.hist.popleft()
        t = self.t
        # somebody walking close by: waiting for them to pass is not "stuck"
        wait = 10.0 if any(self.dist((tr["x"], tr["y"])) < 3.0 for tr in self.moving_tracks()) else 3.0
        if self.blocked_since and t - self.blocked_since > wait:
            return True
        if self.path and abs(self.v) < 0.05 and abs(self.w) < 0.15:  # wants to go somewhere but is not moving
            self.idle_since = self.idle_since or t
            if t - self.idle_since > wait:
                self.idle_since = None
                return True
        else:
            self.idle_since = None
        win = 9.0 if wait > 5.0 else 6.0                            # wants to go somewhere, yet neither moves nor turns
        if self.path and t - self.hist[0][0] >= win - 0.3 and len(self.hist) > 8:
            times = np.array([h[0] for h in self.hist])
            H2 = np.array([(h[1], h[2], h[4]) for h in self.hist])[t - times <= win]
            ext2 = float(np.hypot(*(H2[:, :2].max(axis=0) - H2[:, :2].min(axis=0))))
            turned2 = float(np.abs(np.diff(np.unwrap(H2[:, 2]))).sum())
            if ext2 < 0.2 and turned2 < 0.7:
                self.log("no progress for %.0f s" % win)
                return True
        span = t - self.hist[0][0]
        if len(self.hist) > 8 and span > 3.4:
            times = np.array([h[0] for h in self.hist])
            H = np.array([(h[1], h[2], h[4], h[5]) for h in self.hist])
            rec = H[t - times < 3.4]
            ext_rec = float(np.hypot(*(rec[:, :2].max(axis=0) - rec[:, :2].min(axis=0))))
            if t - times[t - times < 3.4].min() > 3.0 and float(np.abs(rec[:, 3]).mean()) > 0.4 and ext_rec < 0.45:
                self.log("wheels spinning without progress")           # odometry says fast, the map says not moving
                return True
            if span > 7.5:
                trying = sum(h[3] for h in self.hist) > 0.7 * len(self.hist)
                extent = float(np.hypot(*(H[:, :2].max(axis=0) - H[:, :2].min(axis=0))))
                if trying and extent < 0.12:
                    return True
                turned = float(np.abs(np.diff(np.unwrap(H[:, 2]))).sum())
                if turned > 6.5 and extent < 2.5 and float(H[:, 3].mean()) > 0.2:   # driving in circles
                    self.log("driving in circles")
                    return True
        return False

    def start_recover(self):
        t = self.t
        self.stuck_events.append(t)
        while self.stuck_events and t - self.stuck_events[0] > 45.0:
            self.stuck_events.popleft()
        repeated = len(self.stuck_events) >= 3
        n_here = sum(te >= self.approach_t0 for te in self.stuck_events) if self.state == "APPROACH" else 0
        self.log("stuck (%d in 45 s) -> recovery manoeuvre" % len(self.stuck_events))
        if any(o["n"] >= 3 and math.hypot(o["p"][0] - self.x, o["p"][1] - self.y) < 0.6 for o in self.objs):
            self.objs_off_until = t + 12.0                          # stuck next to small objects (apples): do not treat them as walls for a while
        self.debug_dump("stuck")
        self.appr_anchor = None
        self.recover_until = t + 2.5
        self.recover_dir = 1.0 if self.step_i % 4 < 2 else -1.0
        self.hist.clear()
        self.relocalize(1.6, "after a jam")
        self.blocked_since = self.idle_since = None
        if self.state == "EXPLORE" and self.goal_xy:
            self.map.blacklist(*self.goal_xy)
        if self.state == "LOOK" and self.look_goal:
            self.vp_done.append(self.look_goal)
        if repeated:                                                # same problem again and again: change strategy
            self.stuck_events.clear()
            if self.state == "RETURN":
                self.ret_mode = min(self.ret_mode + 1, 3)
                self.log("repeatedly stuck on the way home -> fallback level %d" % self.ret_mode)
        if self.state == "APPROACH" and self.cand and n_here >= 2:
            self.skip_target(self.cand)                              # this target keeps getting us stuck: try later
            self.log("target unreachable right now -> will retry later")
        self.prev_state, self.state = self.state, "RECOVER"

    def replan_soon(self, state=None):
        if state:
            self.state = state
        self.path, self.last_plan, self.arrived = [], -99.0, False

    def go_home(self, why):
        self.log("returning to start (%s)" % why)
        if _os.environ.get("MIR_DUMPMAP"):
            np.savez("map_dump.npz", L=self.map.L, cov=self.map.cov, vp=np.array(self.vp_done).reshape(-1, 2), trail=np.array(self.trail).reshape(-1, 2) if len(self.trail) else np.zeros((0, 2)))
        self.ret_mode, self.ret_best, self.ret_prog_t = 0, 1e9, self.t
        self.cand = None
        self.replan_soon("RETURN")

    def cand_reached(self, tg):
        d = self.dist(tg["p"])
        if d < TARGET_REACH:
            return True
        if self.t - tg["seen"] < 1.0 and tg["range"] < 0.78 * TARGET_REACH:
            return True                                             # camera says: it is right here
        lenient = REACH_LENIENT if REACH_LENIENT is not None else 2.0 * TARGET_REACH
        if self.arrived and self.cand is tg and self.t - self.approach_t0 > 30.0:
            lenient = max(lenient, 0.8)                             # a long approach that ends here: this is as close as it gets
        return self.arrived and d < lenient                         # got as close as the map allows

    def begin_fine_look(self, t):
        """A short-sighted camera (colour only, ~5 m) can miss a target in a corner that looks 'small' to the normal test:
        before going home, look into every unseen corner of at least self.fine_min cells."""
        self.fine_look, self.fine_t0 = True, t
        if not self.los_looks:                                      # far-seeing camera: until now a look counted as 'done' within 5 m,
            for v in self.vp_swept:                                 # even when a shelf or wall hid the corner. From now on only
                self.vp_real.add(v)                                 # what was really in open view of a look is done.
                self.map.mark_vp(v[0], v[1], SWEEP_GAP)
        self.log("normal search finished -> checking the small unseen corners (%.1f m2 of unseen floor)"
                 % (float(((self.map.L < -0.25) & ~self.map.cov).sum()) * RES * RES))
        return True

    def skip_target(self, tg):
        tg["status"], tg["skip_t"] = "skipped", self.t
        tg["skips"] = tg.get("skips", 0) + 1

    def pick_pending(self):
        for g in self.targets:                                      # a skipped target gets another chance later
            if g["status"] == "skipped" and g.get("skips", 0) < 3 and self.t - g.get("skip_t", 0.0) > 45.0:
                g["status"] = "pending"
        need_hits = self.need_hits
        pend = [g for g in self.targets if g["status"] == "pending" and g["hits"] >= need_hits]
        return min(pend, key=lambda g: self.dist(g["p"])) if pend else None

    def reach_target(self, tg, why=""):
        tg["status"] = "reached"
        self.look_first = None
        if self.fine_look:
            self.fine_t0 = self.t
        self.log("TARGET %d reached at (%.1f, %.1f)%s" % (self.n_reached(), tg["p"][0], tg["p"][1], (" (%s)" % why) if why else ""))
        self.cand = None
        self.dwell_until = self.t + DWELL
        self.state = "DWELL"

    def start_approach(self, tg):
        self.cand, self.approach_t0 = tg, self.t
        self.appr_anchor = None
        self.log("target seen at (%.1f, %.1f) -> approaching" % tg["p"])
        self.replan_soon("APPROACH")

    def budget_left(self):
        return TIME_LIMIT - (1.6 * math.hypot(self.x, self.y) / self.travel_v + 30.0) - self.t

    def sweep_wanted(self):
        """Standing here, would a 360-degree look reveal a lot of floor the camera has never seen?"""
        if self.t - self.last_sweep_t < 5.0:
            return False
        for vx, vy in self.vp_done:
            d = math.hypot(self.x - vx, self.y - vy)
            if d < (1.5 if (vx, vy) in self.vp_real else SWEEP_GAP):
                return False
        ci, cj = self.map.cell(self.x, self.y)
        if 0 <= ci < GW and 0 <= cj < GH and self.map.vpvis[ci, cj]:
            return False
        return self.map.unseen_around(self.x, self.y) >= SWEEP_MIN

    def start_sweep(self):
        """Stand still and turn 360 degrees so the camera sees everything around this spot."""
        self.vp_done.append((self.x, self.y))
        self.last_sweep_t = self.t
        P = self.scan_points(2.0)
        if (len(P) and float(np.hypot(P[:, 0], P[:, 1]).min()) < math.hypot(HL, HW) + 0.04) or \
                any(self.dist((tr["x"], tr["y"])) < 2.0 for tr in self.moving_tracks()):
            self.log("no room / people close -> skipping the 360-degree look here")
            self.replan_soon("EXPLORE")
            return
        self.vp_swept.append((self.x, self.y))
        if self.los_looks:                                          # short-sighted camera: what a look could not see through walls stays open
            self.vp_real.add(self.vp_done[-1])
            self.map.mark_vp(self.x, self.y, SWEEP_GAP)
        elif self.fine_look:                                        # far-seeing camera, corner pass: same rule from now on
            self.vp_real.add(self.vp_done[-1])
            self.map.mark_vp(self.x, self.y, SWEEP_GAP)
        self.state, self.sweep_turned, self.sweep_prev, self.sweep_until = "SWEEP", 0.0, self.yaw, self.t + 10.0
        self.log("360-degree look at (%.1f, %.1f)" % (self.x, self.y))

    def tick(self):
        t = self.t
        st = self.state
        dt = 2 * self.dt / 1000.0
        if st == "FINISH":
            self.brake()
            return
        if st == "DWELL":
            self.brake()
            if t > self.dwell_until:
                self.replan_soon("EXPLORE")
            return
        if st == "ALIGN":
            err = wrap(0.0 - self.yaw)
            if abs(err) < 0.05 or t > self.align_until:
                self.finish("back at start")
            else:
                self.drive(0.0, clip(2.5 * err, -0.9, 0.9))
            return
        if st == "RECOVER":
            if t > self.recover_until:
                self.replan_soon(self.prev_state)
            else:
                v, w = self.escape()
                self.drive(approach(self.v, v, 3.0 * dt), approach(self.w, w, 6.0 * dt))
            return
        if st == "SWEEP":
            if self.budget_left() < 0:
                self.go_home("time budget")
                return
            tg = self.pick_pending()
            if tg:
                self.start_approach(tg)
                return
            self.sweep_turned += wrap(self.yaw - self.sweep_prev)
            self.sweep_prev = self.yaw
            danger = any(self.dist((tr["x"], tr["y"])) < 1.8 for tr in self.moving_tracks())
            if self.sweep_turned >= 6.1 or t > self.sweep_until or danger:
                self.dlog("sweep ends: turned %.2f rad, %s" % (self.sweep_turned,
                          "person close" if danger else ("timeout" if t > self.sweep_until else "done")))
                self.replan_soon("EXPLORE")
            else:
                if abs(self.v) > 0.08:                              # first come to a halt (no skidding while spinning up)
                    self.drive(approach(self.v, 0.0, A_BRAKE * dt), approach(self.w, 0.0, A_W * dt))
                else:
                    self.drive(0.0, approach(self.w, SWEEP_W, 2.5 * dt))
            return

        if self.stuck():
            self.start_recover()
            return

        # ---- shoved by a person? the wheels did not move, the robot did: re-anchor on the map when the contact is over
        if any(self.dist((m[0], m[1])) < 1.3 for m in self.movers()) and self.clear_now < 0.08:
            if self.contact_t is None:
                self.dlog("person pressed against the robot: map updates paused")
            self.contact_t = self.contact_last = t
        elif self.contact_t is not None and t - self.contact_t > 0.6 and self.clear_now > 0.2 and abs(self.yaw_rate) < 0.6:
            self.contact_t = None
            pre = (self.x, self.y, self.yaw_bias)
            moved = self.relocalize(2.0, "after contact with a person")
            P = self.match_points(160)
            if len(P) >= 60:
                nk, no = self.pose_fit(P)
                self.dlog("after contact: small correction %.2f m, fit %d/%d" % (moved, no, nk))
                if nk >= 60 and (no < 0.5 * nk or moved > 1.0):         # does not fit, or a suspiciously big correction:
                    loc = (self.x, self.y, self.yaw_bias)               # it may have been thrown further than 2 m
                    self.x, self.y, self.yaw_bias = pre                 # search wide around where the robot was before the push
                    m2 = self.global_relocalize("after contact with a person", ref=no)
                    if self.reloc_ok:
                        moved = m2
                    else:
                        self.x, self.y, self.yaw_bias = loc             # nothing better: keep the small correction
            self.dlog("contact with a person over: re-anchored on the map (moved %.2f m)" % moved)
            if moved > 0.15:
                self.replan_soon()
                return

        need_hits = self.need_hits
        self.targets = [g for g in self.targets
                        if not (g["status"] == "pending" and g["hits"] < need_hits and t - g["seen"] > 3.0)]

        # ---- mission decisions
        if st in ("EXPLORE", "LOOK") and self.step_i % 40 == 0:     # remember how much is known; nothing new -> go home
            self.nov.append((t, int(self.map.cov.sum()), int((self.map.L < -1.0).sum()), len(self.targets)))
            while self.nov and t - self.nov[0][0] > NOVELTY_WIN + 5:
                self.nov.popleft()
            o = self.nov[0]
            pend = any(g["status"] == "pending" for g in self.targets)
            if (t - o[0] >= NOVELTY_WIN and not pend and len(self.targets) == o[3]
                    and self.nov[-1][1] - o[1] < 250 and self.nov[-1][2] - o[2] < 400):
                self.go_home("nothing new for %.0f s, %d target(s) reached" % (NOVELTY_WIN, self.n_reached()))
                return
            if t > self.explore_max:
                self.go_home("search time limit, %d target(s) reached" % self.n_reached())
                return
        if st in ("EXPLORE", "APPROACH", "LOOK"):
            if EXPECTED_TARGETS is not None and self.n_reached() >= EXPECTED_TARGETS:
                self.go_home("all %d expected targets reached" % EXPECTED_TARGETS)
                return
            if self.budget_left() < 0:
                self.go_home("time budget")
                return
        if st in ("EXPLORE", "LOOK"):
            tg = self.pick_pending()
            if tg:
                self.start_approach(tg)
                st = "APPROACH"
        if st == "LOOK":
            if self.arrived or (self.look_goal and self.dist(self.look_goal) < 0.6):
                self.start_sweep()
                return
            if self.fine_look:
                if t - self.fine_t0 > self.fine_budget:
                    self.replan_soon("EXPLORE")
                    return
            elif self.look_first is not None and t - self.look_first > LOOK_BUDGET:
                self.replan_soon("EXPLORE")
                return
            if t - self.look_t0 > 35.0:
                self.log("viewpoint not reachable -> next one")
                self.vp_done.append(self.look_goal or (self.x, self.y))
                self.look_t0 = t
                self.replan_soon("LOOK")
                return
        if st == "APPROACH":
            tg = self.cand
            if tg is None or tg["status"] != "pending":
                self.replan_soon("EXPLORE")
                return
            here = (t, self.x, self.y)                                  # stall watchdog: an approach that stays on the spot must END, not freeze
            if self.appr_anchor is None or math.hypot(self.x - self.appr_anchor[1], self.y - self.appr_anchor[2]) > 0.2 \
                    or any(self.dist((m[0], m[1])) < 3.0 for m in self.movers()):
                self.appr_anchor = here
            elif t - self.appr_anchor[0] > 6.0:
                self.appr_anchor = None
                if self.dist(tg["p"]) < 0.8:
                    self.reach_target(tg, "as close as it gets")
                else:
                    self.log("approach stalled %.1f m from the target -> will retry later" % self.dist(tg["p"]))
                    self.skip_target(tg)
                    self.map.blacklist(*tg["p"])
                    self.replan_soon("EXPLORE")
                return
            if self.cand_reached(tg):
                self.reach_target(tg)
                return
            if t - self.approach_t0 > 70.0:
                self.log("target not reachable in time -> will retry later")
                self.skip_target(tg)
                self.map.blacklist(*tg["p"])
                self.replan_soon("EXPLORE")
                return
        elif st == "RETURN":
            hd = math.hypot(self.x, self.y)
            if hd < 5.0 and t - self.anchor_t > 0.7 and abs(self.yaw_rate) < 0.5 and self.clear_now > 0.10:
                self.anchor_t = t                                   # the start area was mapped from the exact start pose: snap onto it
                self.relocalize(1.4 if hd > 1.5 else 0.7, "start map", use0=True)
                hd = math.hypot(self.x, self.y)
            if hd < START_RADIUS and self.anchor_checks < 3 and self.L0 is not None:
                self.anchor_checks += 1                             # verify before declaring victory
                self.relocalize(0.7, "final check", use0=True)
                hd = math.hypot(self.x, self.y)
            if hd < 0.5:                                            # hovering next to the start without ever getting inside the radius: accept
                self.near_home_t = self.near_home_t or t
                if t - self.near_home_t > 12.0:
                    hd = 0.0
            else:
                self.near_home_t = None
            if hd < START_RADIUS:
                if ALIGN_AT_START and abs(wrap(self.yaw)) > 0.05 and self.clear_now > 0.2:
                    self.state, self.align_until = "ALIGN", t + 8.0
                else:
                    self.finish("back at start")
                return
            if t > TIME_LIMIT + 120.0:
                self.finish("time out")
                return
            if self.path:                                           # progress watchdog for the way home
                rem = self.dist(self.path[0]) + polyline_len(self.path)
                if rem < self.ret_best - 0.5:
                    self.ret_best, self.ret_prog_t = rem, t
                elif t - self.ret_prog_t > 40.0:
                    self.ret_mode = self.ret_mode + 1 if self.ret_mode < 3 else 0
                    self.ret_best, self.ret_prog_t = 1e9, t
                    self.log("no progress toward start -> fallback level %d" % self.ret_mode)
                    self.replan_soon()
        elif st == "EXPLORE" and self.goal_xy and t - self.goal_t0 > 45.0:
            self.map.blacklist(*self.goal_xy)                       # chasing one frontier too long
            self.replan_soon()

        if st == "EXPLORE" and self.goal_is_look and self.goal_xy and (self.arrived or self.dist(self.goal_xy) < 0.6):
            self.goal_is_look = False
            self.start_sweep()                                  # reached a viewpoint: look around (marks it done)
            return
        if st == "EXPLORE" and self.step_i % 8 == 0 and self.sweep_wanted():
            self.start_sweep()
            return

        # ---- (re)plan
        period = 1.3 if self.movers() else 2.5
        need = (not self.path) or (t - self.last_plan > period and st != "LOOK")
        if not need and self.step_i % 16 == 0:
            need = self.path_blocked()
        if need and not (st == "RETURN" and self.ret_mode >= 2 and self.path and t - self.last_plan < 6.0):
            if st == "RETURN":
                self.plan_return(t)
            elif not self.plan(st, t):
                if st == "EXPLORE":
                    if t < 3.0:
                        self.drive(0, 0.5)                          # look around while the map fills
                    elif (self.look_first is None or t - self.look_first < LOOK_BUDGET) and self.plan("LOOK", t):
                        # LiDAR frontiers are gone: go and look at what the camera never saw
                        self.state, self.look_t0 = "LOOK", t
                        self.look_first = self.look_first if self.look_first is not None else t
                        self.log("no frontier left -> looking around for unseen areas")
                    elif t - (self.fine_t0 if self.fine_look else t) < self.fine_budget and self.budget_left() > 60.0 \
                            and (self.fine_look or self.begin_fine_look(t)) and self.plan("LOOK", t):
                        self.state, self.look_t0 = "LOOK", t
                        self.log("looking into the small unseen corners")
                    elif not self.retry_done and (self.map.blk.any() or any(g["status"] == "skipped" for g in self.targets)):
                        self.log("nothing left to look at -> second pass")
                        self.retry_done = True
                        self.map.blk[:] = False
                        for g in self.targets:
                            if g["status"] == "skipped":
                                g["status"] = "pending"
                        self.last_plan = t - 1.5
                    else:
                        left = [g for g in self.targets if g["status"] != "reached"]
                        self.go_home("exploration complete, %d target(s) found%s"
                                     % (self.n_reached(), ", %d unreachable" % len(left) if left else ""))
                    return
                if st == "LOOK":
                    self.replan_soon("EXPLORE")
                    return
                if st == "APPROACH":
                    if self.cand is not None and self.dist(self.cand["p"]) < 0.8:
                        self.reach_target(self.cand, "no closer spot on the map")
                        return
                    self.log("no path to target -> will retry later")
                    self.skip_target(self.cand)
                    self.replan_soon("EXPLORE")
                    return
        self.follow()

    # ------------------------------------------------------------ misc
    def log(self, msg):
        line = "[%6.1fs] %-8s %s" % (self.t, self.state, msg)
        print(line)
        if self.dbg_file:
            self.dbg_file.write(line + "\n")
            self.dbg_file.flush()

    def dlog(self, msg):
        """Debug-file only."""
        if self.dbg_file:
            self.dbg_file.write("[%6.1fs] %-8s %s\n" % (self.t, self.state, msg))
            self.dbg_file.flush()

    def truth(self):
        """True pose in the start frame (debug only)."""
        if self.me is None or self.truth0 is None:
            return None
        try:
            p, o = self.me.getPosition(), self.me.getOrientation()
        except Exception:
            return None
        x0, y0, a0 = self.truth0
        dx, dy = p[0] - x0, p[1] - y0
        c, s = math.cos(-a0), math.sin(-a0)
        return c * dx - s * dy, s * dx + c * dy, wrap(math.atan2(o[3], o[0]) - a0)

    def check_person_contact(self):
        """Debug only: true footprint-to-person distance (people are ~0.25 m radius)."""
        try:
            me = self.me.getPosition()
            o = self.me.getOrientation()
            yaw = math.atan2(o[3], o[0])
            c, s = math.cos(yaw), math.sin(yaw)
            worst = 9.0
            for k, nd in enumerate(self.persons):
                p = nd.getPosition()
                dx, dy = p[0] - me[0], p[1] - me[1]
                xr, yr = c * dx + s * dy, -s * dx + c * dy
                d = math.hypot(max(abs(xr) - 0.445, 0.0), max(abs(yr) - 0.29, 0.0)) - 0.25
                worst = min(worst, d)
                if d < 0.0 and not self.pers_hit:
                    self.pers_hit = True
                    self.dlog("*** COLLISION with person %d (gap %.2f m) robot v=%.2f" % (k + 1, d, self.v_enc))
                    print("[%6.1fs] *** COLLISION with person %d" % (self.t, k + 1))
            if worst > 0.15:
                self.pers_hit = False
            self.min_pdist = min(self.min_pdist, worst)
        except Exception:
            self.persons = []

    def debug_dump(self, tag):
        if not DEBUG or self.dbg_file is None:
            return
        try:
            ex = [(tr["x"], tr["y"]) for tr in self.tracks] + [g["p"] for g in self.targets]
            self.map.dump_pgm("dbg_%s_%03d.pgm" % (tag, int(self.t)), (self.x, self.y), self.path[:40], ex)
        except Exception as e:                                      # never let a debug picture stop the robot
            self.dlog("dump failed: %r" % (e,))

    def debug_line(self):
        tr = self.truth()
        tt = ""
        if tr is not None:
            tt = " | TRUE (%.2f, %.2f, %4.0f) err %.2f m %.0f deg" % (
                tr[0], tr[1], math.degrees(tr[2]), math.hypot(tr[0] - self.x, tr[1] - self.y),
                math.degrees(wrap(self.yaw - tr[2])))
        if self.persons and self.truth0 is not None:
            try:
                x0, y0, a0 = self.truth0
                cc, ss = math.cos(-a0), math.sin(-a0)
                for k, nd in enumerate(self.persons):
                    p = nd.getPosition()
                    dx, dy = p[0] - x0, p[1] - y0
                    tt += " P%d(%.2f,%.2f)" % (k + 1, cc * dx - ss * dy, ss * dx + cc * dy)
            except Exception:
                pass
        self._dl = getattr(self, "_dl", 0) + 1
        if self._dl % 10 == 1:
            self.dlog("lidars: " + ", ".join("%s valid %d/%d min %.2f" % (l.dev.getName(), int(l.valid.sum()), l.n, float(l.r[l.valid].min()) if l.valid.any() else -1) for l in self.lidars))
        try:
            P = self.match_points(160)
            nk, no = self.pose_fit(P) if len(P) else (0, 0)
        except Exception:
            nk = no = 0
        self.dlog("est (%.2f, %.2f, %4.0f) cmd v=%.2f w=%.2f | meas v=%.2f w=%.2f gain=%.2f | clr=%.2f tracks=%d/%d "
                  "path=%d scan_shift=%.2f fit=%d/%d%s"
                  % (self.x, self.y, math.degrees(self.yaw), self.v, self.w, self.v_enc, self.yaw_rate, self.wgain,
                     self.clear_now, len(self.moving_tracks()), len(self.tracks), len(self.path), self.scan_shift, no, nk, tt))

    def finish(self, why):
        self.drive(0, 0)
        reached = [g["p"] for g in self.targets if g["status"] == "reached"]
        self.log("FINISH (%s): %d target(s) reached %s, distance to start %.2f m, heading error %.0f deg, time %.1f s"
                 % (why, len(reached), ["(%.1f, %.1f)" % p for p in reached],
                    math.hypot(self.x, self.y), math.degrees(abs(wrap(self.yaw))), self.t))
        try:
            self.map.save_pgm("search_rescue_map.pgm", self.trail, reached)
            self.log("map saved to search_rescue_map.pgm")
        except OSError:
            pass
        self.state = "FINISH"
        if _os.environ.get("MIR_QUIT"):
            try:
                self.dbg_file and self.dbg_file.flush()
                self.robot.simulationQuit(0)
            except Exception:
                pass

    def detect_fling(self):
        """The robot was thrown or shoved hard (a person hit it): the scan changes a lot between two readings although the
        wheels barely moved.  Returns True while that lasts; the pose is repaired afterwards (global_relocalize)."""
        if getattr(self, "nofling", False):
            return False
        hit = tot = 0
        for l in self.lidars:
            pr = getattr(l, "_pr", None)
            if pr is not None:
                m = l.valid & pr[1]
                n = int(m.sum())
                if n > 40:
                    hit += int((np.abs(l.r[m] - pr[0][m]) > 0.3).sum())
                    tot += n
            l._pr = (l.r.copy(), l.valid.copy())
        if tot > 60 and hit > 0.25 * tot and abs(self.yaw_rate) < 0.5 and abs(self.w) < 0.8:
            if self.t - self.flung_t > 1.0:
                self.dlog("scan jumped (%d of %d rays changed by >0.3 m): robot thrown or shoved" % (hit, tot))
            self.flung_t, self.flung_pending = self.t, True
        return self.t - self.flung_t < 0.6

    def test_teleport(self):
        """Test only (needs a Supervisor robot): MIR_TELEPORT="60,0,-4" throws the robot 4 m in world -y at t = 60 s."""
        spec = _os.environ.get("MIR_TELEPORT")
        if not spec or self.me is None or getattr(self, "_tp_done", False):
            return
        t0, dx, dy = (float(v) for v in spec.split(","))
        if self.t >= t0:
            self._tp_done = True
            f = self.me.getField("translation")
            p = f.getSFVec3f()
            f.setSFVec3f([p[0] + dx, p[1] + dy, p[2]])
            self.me.resetPhysics()
            if _os.environ.get("MIR_TP_CONTACT"):                       # test the 'person pushed me' path instead of the fling detector
                self.nofling, self.contact_t = True, self.t
            self.log("TEST: robot teleported by (%.1f, %.1f) m" % (dx, dy))

    def run(self):
        while self.robot.step(self.dt) != -1:
            self.t = self.robot.getTime()
            self.step_i += 1
            self.test_teleport()
            self.odometry()
            if math.hypot(self.x - self.crumbs[-1][0], self.y - self.crumbs[-1][1]) > 0.4:
                self.crumbs.append((self.x, self.y))
            if self.step_i % 2 == 0:
                for l in self.lidars:
                    l.read()
                self.update_contact()
                flung = self.detect_fling()
                if self.flung_pending and not flung:                # it has come to rest: find out where it landed
                    self.flung_pending = False
                    self.global_relocalize("thrown or shoved")
                shoved = self.contact_t is not None or flung        # being pushed: the pose is unreliable, keep the map clean
                if not shoved and self.step_i % 8 == 0 and abs(self.yaw_rate) < 3.0:
                    self.remember_scan()
                if USE_SCAN_MATCH and self.step_i % 8 == 0 and not shoved:
                    self.scan_match()
                self.detect_dynamic()
                for l in (self.lidars if not shoved else []):
                    self.map.update(self.x + l.mx * math.cos(self.yaw),
                                    self.y + l.mx * math.sin(self.yaw), self.yaw, l)
                cam_new = self.camera_fresh()
                if self.state in ("EXPLORE", "APPROACH", "LOOK", "SWEEP") and (cam_new if CAM_SYNC else self.step_i % 4 == 0):
                    self.update_camera()
                self.tick()
            if self.step_i % 25 == 0:
                self.trail.append((self.x, self.y))
            if self.persons and self.step_i % 2 == 0:
                self.check_person_contact()
            if DEBUG and self.t - self.dbg_t >= 0.5:
                self.dbg_t = self.t
                self.debug_line()
                if int(self.t) % 60 == 59 and self.t - self.dump_t > 5.0:
                    self.dump_t = self.t
                    self.debug_dump("t")
            if self.t - self.log_t > 5.0:
                self.log_t = self.t
                self.log("pose (%.1f, %.1f, %.0f deg) targets %d/%d seen | v=%.2f m/s clearance=%.2fm path=%d"
                         % (self.x, self.y, math.degrees(self.yaw), self.n_reached(), len(self.targets),
                            self.v, self.clear_now, len(self.path)))


if __name__ == "__main__":
    Agent().run()
