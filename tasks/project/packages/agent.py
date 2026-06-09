"""
Convoying follower — final project agent.

One Duckiebot follows another. The leader carries the Duckietown circle
grid (3 rows x 7 dots) on its back; we detect it with cv2.findCirclesGrid,
steer to keep it centred, and modulate speed to hold a safe distance.
Sign detection is leader-only; this bot mimics the leader via the dot grid.
"""

import os

import cv2
import numpy as np
import yaml

# Published for the web UI / debugging (the sim server reads these).
# Harmless on real hardware — nothing reads them there.
DEBUG_FRAME = None
STATUS = {}

# Runtime config (mutable, read by the web UI for live tuning).
CFG = None
_leader = None
_ctrl = None


# =====================================================================
# SECTION 1: CONFIG LOADING
#   Loads config/project_config.yaml. Every value has a baked-in default
#   so the agent still runs if the file or a key is missing.
# =====================================================================

_DEFAULTS = {
    'leader': {
        'grid_cols': 7, 'grid_rows': 3, 'dot_spacing_m': 0.0125,
        'target_span': 0.45, 'stop_span': 0.60, 'span_deadband': 0.03,
        'blob_min_area': 6.0,
    },
    'control': {
        'max_speed': 0.45, 'chase_speed': 0.30,
        'steer_kp': 0.55, 'steer_kd': 0.30,
        'dist_kp': 2.0, 'accel_rate': 0.05, 'decel_rate': 0.08,
        'search_turn': 0.10, 'search_after_frames': 24, 'loop_hz': 20,
    },
    'camera': {'matrix': None, 'dist_coeffs': None},
}


# Set by the server before calling main() to select which config file to load.
# virtual_server sets this to 'project_config_sim.yaml'; real_server leaves it as-is.
CONFIG_FILE = 'project_config.yaml'


def _config_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, '..', '..', '..', 'config', CONFIG_FILE)


def load_config():
    cfg = {k: dict(v) for k, v in _DEFAULTS.items()}
    path = _config_path()
    try:
        with open(path, 'r') as f:
            loaded = yaml.safe_load(f) or {}
        for section, values in loaded.items():
            if isinstance(values, dict):
                cfg.setdefault(section, {}).update(values)
            else:
                cfg[section] = values
        print(f'[agent] Loaded config from {CONFIG_FILE}')
    except FileNotFoundError:
        print(f'[agent] {CONFIG_FILE} not found, using defaults')
    return cfg


# =====================================================================
# SECTION 2: LEADER DETECTION  (Duckietown circle grid on the leader's back)
#   findCirclesGrid returns the 21 dot centres. From them we derive:
#     lateral_error : horizontal offset of the grid in [-1, 1]
#                     (-1 = leader far left, +1 = leader far right)
#     span          : grid width as a fraction of frame width
#                     (bigger => leader closer)  -> our distance proxy
# =====================================================================

class LeaderDetector:
    def __init__(self, cfg):
        self.cols = int(cfg['leader']['grid_cols'])
        self.rows = int(cfg['leader']['grid_rows'])
        self.pattern = (self.cols, self.rows)
        self.flags = cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
        # A blob detector tuned for SMALL dark dots so the grid is still
        # found when the leader is far away (lower min_area => longer range).
        self._blob = self._make_blob_detector(float(cfg['leader']['blob_min_area']))

    @staticmethod
    def _make_blob_detector(min_area):
        p = cv2.SimpleBlobDetector_Params()
        p.filterByColor = True
        p.blobColor = 0                  # the dots are dark on white
        p.filterByArea = True
        p.minArea = min_area             # small => detect distant (tiny) dots
        p.maxArea = 8000.0
        p.filterByCircularity = True
        p.minCircularity = 0.6
        p.filterByInertia = False
        p.filterByConvexity = False
        p.minDistBetweenBlobs = 3.0
        return cv2.SimpleBlobDetector_create(p)

    def detect(self, bgr):
        """Return (found, lateral_error, span, centers_or_None)."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        found, centers = cv2.findCirclesGrid(
            gray, self.pattern, flags=self.flags, blobDetector=self._blob)
        if not found or centers is None:
            return False, 0.0, 0.0, None

        pts = centers.reshape(-1, 2)
        h, w = gray.shape[:2]
        mean_x = float(np.mean(pts[:, 0]))
        span = float(pts[:, 0].max() - pts[:, 0].min()) / float(w)
        lateral_error = (mean_x - w / 2.0) / (w / 2.0)
        return True, float(np.clip(lateral_error, -1.0, 1.0)), span, pts

    # --- Optional, for real hardware: true metric distance via solvePnP ---
    # Build the real 3-D dot positions (z=0 plane) once, then on each frame:
    #   ok, rvec, tvec = cv2.solvePnP(obj_pts, centers, K, dist)
    #   distance_m = float(np.linalg.norm(tvec))
    # Requires calibrated intrinsics in config (camera.matrix). The span
    # method above needs no calibration, so it is the default for sim.
    def build_grid_object_points(self, spacing_m):
        obj = np.zeros((self.cols * self.rows, 3), np.float32)
        grid = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
        obj[:, :2] = grid * spacing_m
        return obj


# =====================================================================
# SECTION 3: CONTROL  (PID steering + distance speed control + smooth ramp)
#   - steering : PID on lateral_error -> differential turn
#   - speed    : PURE proportional on span error around target_span. At the
#                target distance the speed is exactly 0 (it holds position);
#                when too close (span >= stop_span) speed is clamped to 0.
#   - ramp     : current speed eases toward target so starts/stops are soft
# =====================================================================

class Controller:
    def __init__(self, cfg):
        c = cfg['control']
        self.max_speed = float(c['max_speed'])
        self.chase_speed = float(c['chase_speed'])    # forward creep while reacquiring far leader
        self.steer_kp = float(c['steer_kp'])
        self.steer_kd = float(c['steer_kd'])
        self.dist_kp = float(c['dist_kp'])
        self.accel = float(c['accel_rate'])
        self.decel = float(c['decel_rate'])
        self.search_turn = float(c['search_turn'])
        self.search_after = int(c['search_after_frames'])
        self.target_span = float(cfg['leader']['target_span'])
        self.stop_span = float(cfg['leader']['stop_span'])
        self.deadband = float(cfg['leader']['span_deadband'])

        self._cur_v = 0.0       # ramped forward speed (may go negative when backing)
        self._prev_e = 0.0      # previous lateral error (for D term)

    def steering(self, lateral_error):
        d = lateral_error - self._prev_e
        self._prev_e = lateral_error
        return self.steer_kp * lateral_error + self.steer_kd * d

    def distance_speed(self, span, speed_cap):
        """Forward speed to hold the target distance (before ramping)."""
        if span >= self.stop_span:
            return 0.0                          # too close -> hold still, don't reverse
        err = self.target_span - span           # >0 = leader too far -> go forward
        if abs(err) < self.deadband:
            return 0.0                          # at the right distance -> hold still
        return float(np.clip(self.dist_kp * err, 0.0, speed_cap))

    def ramp(self, target_v):
        """Ease current speed toward target_v (smooth accel/decel)."""
        if target_v > self._cur_v:
            self._cur_v = min(target_v, self._cur_v + self.accel)
        else:
            self._cur_v = max(target_v, self._cur_v - self.decel)
        return self._cur_v

    def reset_steer(self):
        self._prev_e = 0.0

    @property
    def current_speed(self):
        return self._cur_v


def _sync_cfg():
    """Re-read leader/control config into the live objects (live tuning)."""
    global _ctrl, CFG
    if _ctrl is None or CFG is None:
        return
    c = CFG['control']
    _ctrl.max_speed = float(c['max_speed'])
    _ctrl.chase_speed = float(c['chase_speed'])
    _ctrl.steer_kp = float(c['steer_kp'])
    _ctrl.steer_kd = float(c['steer_kd'])
    _ctrl.dist_kp = float(c['dist_kp'])
    _ctrl.accel = float(c['accel_rate'])
    _ctrl.decel = float(c['decel_rate'])
    _ctrl.search_turn = float(c['search_turn'])
    _ctrl.search_after = int(c['search_after_frames'])
    _ctrl.target_span = float(CFG['leader']['target_span'])
    _ctrl.stop_span = float(CFG['leader']['stop_span'])
    _ctrl.deadband = float(CFG['leader']['span_deadband'])


# =====================================================================
# SECTION 4: LED SIGNALLING
#   Corner LEDs communicate the follower's state. Guarded with `if leds:`
#   because the LED hardware may fail to initialise on a real bot.
#   Indices: 0=front-left, 2=front-right, 3=back-left, 4=back-right.
# =====================================================================

_COLORS = {
    'FOLLOW': [0.0, 1.0, 0.0],   # green
    'HOLD':   [1.0, 0.7, 0.0],   # amber — holding distance / leader stopped
    'SEARCH': [0.0, 0.3, 1.0],   # blue
}


def set_leds(leds, state, turn):
    if not leds:
        return
    base = _COLORS.get(state, [0.0, 0.0, 0.0])
    leds.set_rgb(3, base)
    leds.set_rgb(4, base)
    # Front LEDs double as turn indicators while following.
    if state in ('FOLLOW', 'HOLD') and abs(turn) > 0.12:
        if turn > 0:                         # turning right
            leds.set_rgb(2, [1.0, 0.6, 0.0]); leds.set_rgb(0, base)
        else:                                # turning left
            leds.set_rgb(0, [1.0, 0.6, 0.0]); leds.set_rgb(2, base)
    else:
        leds.set_rgb(0, base); leds.set_rgb(2, base)


# =====================================================================
# SECTION 5: STATE MACHINE + main()
#   States: SEARCH / FOLLOW / HOLD (leader too close or stopped).
# =====================================================================

def _annotate(bgr, state, span, lateral_error, centers):
    """Draw a small debug overlay (shown in the sim web UI)."""
    img = bgr.copy()
    if centers is not None:
        for (x, y) in centers.astype(int):
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
    txt = f'{state}  span={span:.2f}  e={lateral_error:+.2f}'
    cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 4)
    cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1)
    return img


def main(camera, wheels, leds, stop_event):
    global DEBUG_FRAME, STATUS, CFG, _leader, _ctrl

    CFG = load_config()
    _leader = LeaderDetector(CFG)
    _ctrl = Controller(CFG)

    dt = 1.0 / float(CFG['control']['loop_hz'])

    lost_count = 0
    last_e = 0.0               # last lateral error while the leader was seen
    last_span = 0.0            # last span while seen (tells us WHY we lost it)
    last_turn = 0.0            # last steering command while following

    try:
        while not stop_event.is_set():
            _sync_cfg()
            ok, frame = camera.read()
            if not ok or frame is None:
                stop_event.wait(0.02)
                continue

            found, lateral_error, span, centers = _leader.detect(frame)

            speed_cap = _ctrl.max_speed

            # --- pick state + commands ---
            turn = 0.0
            if found:
                lost_count = 0
                last_e, last_span = lateral_error, span
                turn = _ctrl.steering(lateral_error)
                last_turn = turn
                target_v = _ctrl.distance_speed(span, speed_cap)
                if span >= _ctrl.stop_span:
                    state = 'HOLD'           # too close -> hold position
                else:
                    state = 'FOLLOW'
            else:
                lost_count += 1
                if lost_count < _ctrl.search_after:
                    # brief loss: HOLD heading (never spin). React to why we lost:
                    if last_span >= _ctrl.target_span:
                        # was close -> camera likely clipped; hold still
                        target_v = 0.0
                        turn = 0.0
                        state = 'HOLD'
                    else:
                        # was far -> keep chasing straight to reacquire
                        target_v = _ctrl.chase_speed
                        turn = last_turn * 0.5
                        state = 'FOLLOW'
                else:
                    # long loss: gentle scan toward the last-seen side
                    state = 'SEARCH'
                    target_v = 0.0
                    turn = _ctrl.search_turn * (1.0 if last_e >= 0 else -1.0)

            # --- ramp speed, mix into wheels ---
            v = _ctrl.ramp(target_v)
            left = float(np.clip(v + turn, -1.0, 1.0))
            right = float(np.clip(v - turn, -1.0, 1.0))
            wheels.set_wheels_speed(left, right)

            # --- signal + publish debug ---
            set_leds(leds, state, turn)
            DEBUG_FRAME = _annotate(frame, state, span, lateral_error, centers)
            STATUS = {
                'state': state, 'found': found, 'span': round(span, 3),
                'lateral_error': round(lateral_error, 3), 'speed': round(v, 3),
                'turn': round(turn, 3),
            }

            stop_event.wait(dt)
    finally:
        # Contract: always stop the motors and lights on exit.
        wheels.set_wheels_speed(0.0, 0.0)
        if leds:
            leds.all_off()
