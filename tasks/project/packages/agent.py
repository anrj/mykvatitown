"""
Convoying follower — final project agent.

One Duckiebot follows another. The leader carries the Duckietown circle
grid (3 rows x 7 dots) on its back; we detect it with cv2.findCirclesGrid,
steer to keep it centred, and modulate speed to hold a safe distance.

Turn detection uses two complementary, calibration-free channels:
  * lateral excursion  — grid swinging left/right vs a learned baseline
                         (absorbs each wonky camera's resting offset).
  * row tilt           — mean angle of the grid's rows vs a learned tilt
                         baseline (bias-immune to lateral mounting offset;
                         this is the perspective/orientation channel).

A turn is declared when a channel exceeds its threshold (raised while the
follower is itself correcting — the self-stable gate) for a sustained run
of same-sign frames, with hysteresis on clear. The follower signals the
direction via LEDs / STATUS for a later preemptive turn.

No AprilTag / sign detection here — the LEADER handles stop signs; the
follower mimics the leader via the dot grid (stops when the leader stops,
because span grows past stop_span). No solvePnP / calibration required.
"""

import os
import threading

import cv2
import numpy as np
import yaml

from tasks.project.packages import preprocessing

# Published for the web UI / debugging (the servers read these).
# DETECTION is a lock-protected dict of the latest detection + control results.
# The video thread reads it to draw the overlay on the LIVE frame (so the feed
# never freezes — it always shows fresh camera frames with the overlay drawn on
# top at detection rate). STATUS is the /status panel payload.
DETECTION = {}
_det_lock = threading.Lock()
STATUS = {}

# Runtime config (mutable, read by the web UI for live tuning).
CFG = None
_leader = None
_turn = None
_ctrl = None

# Pause flag — servers set this; the loop holds wheels at zero while paused
# but still runs detection so you can verify it stationary.
PAUSED = True


# =====================================================================
# SECTION 1: CONFIG LOADING
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
        'error_alpha': 0.3, 'd_deadband': 0.01,
    },
    'detection': {
        'hold_frames': 3,     # keep last pose this many frames on a miss
        'roi_pad': 30,        # px padding around last bbox for ROI retry
        'clahe': True,        # CLAHE-gray retry on a miss
    },
    'turn': {
        'excursion_thr': 0.35,        # |lateral_error - baseline| to start a turn
        'excursion_thr_strong': 0.55, # required while the follower is self-correcting
        'tilt_thr': 0.10,             # rad, |tilt - tilt_baseline| to start a turn
        'tilt_thr_strong': 0.18,      # rad, required while self-correcting
        'sustain_frames': 4,          # consecutive same-sign frames to declare
        'clear_thr': 0.12,            # excursion below this to clear a turn
        'clear_tilt': 0.04,           # tilt below this to clear a turn
        'baseline_alpha': 0.02,       # EMA rate for lateral/tilt baselines
        'self_stable_thr': 0.10,      # |steer| EMA above this => follower unstable
        'self_stable_alpha': 0.15,    # EMA rate for |steer|
        'sign_invert': False,         # flip L/R label (confirm empirically once)
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
#   Robust, calibration-free pipeline (each step only runs on a miss):
#     1. raw grayscale              (primary, known-working)
#     2. CLAHE grayscale retry      (catches poor/uneven lighting)
#     3. ROI-crop retry             (less clutter around last bbox, bridges
#                                    motion blur / partial occlusion)
#     4. temporal hold of last pose (bridges 1-2 frame flicker)
#   Returns (found, lateral_error, span, centers, method, quality)
#     lateral_error : horizontal offset in [-1,1] (-1 leader left, +1 right)
#     span          : grid width / frame width  (bigger => closer)
#     method        : 'raw' | 'clahe' | 'roi' | 'held' | ''
#     quality       : 1.0 fresh, decaying while held
# =====================================================================

class LeaderDetector:
    def __init__(self, cfg):
        self.cols = int(cfg['leader']['grid_cols'])
        self.rows = int(cfg['leader']['grid_rows'])
        self.pattern = (self.cols, self.rows)
        self.flags = cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING
        self._blob = self._make_blob_detector(float(cfg['leader']['blob_min_area']))
        d = cfg.get('detection', {})
        self.hold_frames = int(d.get('hold_frames', 3))
        self.roi_pad = int(d.get('roi_pad', 30))
        self.use_clahe = bool(d.get('clahe', True))
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self._last_centers = None      # for temporal hold + ROI seeding
        self._last_bbox = None
        self._hold_left = 0

    @staticmethod
    def _make_blob_detector(min_area):
        p = cv2.SimpleBlobDetector_Params()
        p.filterByColor = True
        p.blobColor = 0                  # dark dots on white
        p.filterByArea = True
        p.minArea = min_area             # small => detect distant (tiny) dots
        p.maxArea = 8000.0
        p.filterByCircularity = True
        p.minCircularity = 0.6
        p.filterByInertia = False
        p.filterByConvexity = False
        p.minDistBetweenBlobs = 3.0
        return cv2.SimpleBlobDetector_create(p)

    def _try(self, gray):
        found, centers = cv2.findCirclesGrid(
            gray, self.pattern, flags=self.flags, blobDetector=self._blob)
        if found and centers is not None:
            return centers.reshape(-1, 2)
        return None

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        pts = self._try(gray)
        method = 'raw' if pts is not None else ''

        if pts is None and self.use_clahe:
            pts = self._try(self._clahe.apply(gray))
            if pts is not None:
                method = 'clahe'

        if pts is None and self._last_bbox is not None:
            x0, y0, x1, y1 = self._last_bbox
            x0 = max(0, x0 - self.roi_pad); y0 = max(0, y0 - self.roi_pad)
            x1 = min(w, x1 + self.roi_pad); y1 = min(h, y1 + self.roi_pad)
            if x1 - x0 > 20 and y1 - y0 > 20:
                crop = bgr[y0:y1, x0:x1]
                cpts = self._try(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
                if cpts is not None:
                    pts = cpts + np.array([x0, y0], dtype=np.float32)
                    method = 'roi'

        if pts is not None:
            self._last_centers = pts
            xs = pts[:, 0]
            self._last_bbox = (int(xs.min()), int(pts[:, 1].min()),
                               int(xs.max()), int(pts[:, 1].max()))
            self._hold_left = self.hold_frames
            quality = 1.0
        elif self._last_centers is not None and self._hold_left > 0:
            # brief miss -> reuse last pose a few frames (decaying confidence)
            pts = self._last_centers
            self._hold_left -= 1
            quality = self._hold_left / float(max(1, self.hold_frames))
            method = 'held'
        else:
            return False, 0.0, 0.0, None, '', 0.0

        xs = pts[:, 0]
        mean_x = float(np.mean(xs))
        span = float(xs.max() - xs.min()) / float(w)
        lateral_error = (mean_x - w / 2.0) / (w / 2.0)
        return (True, float(np.clip(lateral_error, -1.0, 1.0)), span,
                pts, method, quality)


# =====================================================================
# SECTION 3: TURN DETECTION
#   Two calibration-free channels:
#     lateral excursion : lateral_error - lateral_baseline
#     row tilt          : mean row angle - tilt_baseline
#   Baselines are EMAs updated only while tracking straight & not turning,
#   so each wonky camera's resting lateral offset / roll is absorbed. The
#   self-stable gate raises thresholds while the follower is actively
#   correcting its own heading, so the follower's wobble is not misread as
#   a leader turn.
# =====================================================================

class TurnDetector:
    def __init__(self, cfg):
        t = cfg.get('turn', {})
        self.excursion_thr = float(t.get('excursion_thr', 0.35))
        self.excursion_thr_strong = float(t.get('excursion_thr_strong', 0.55))
        self.tilt_thr = float(t.get('tilt_thr', 0.10))
        self.tilt_thr_strong = float(t.get('tilt_thr_strong', 0.18))
        self.sustain_frames = int(t.get('sustain_frames', 4))
        self.clear_thr = float(t.get('clear_thr', 0.12))
        self.clear_tilt = float(t.get('clear_tilt', 0.04))
        self.baseline_alpha = float(t.get('baseline_alpha', 0.02))
        self.self_stable_thr = float(t.get('self_stable_thr', 0.10))
        self.self_stable_alpha = float(t.get('self_stable_alpha', 0.15))
        self.sign_invert = bool(t.get('sign_invert', False))

        self.lateral_baseline = 0.0
        self.tilt_baseline = 0.0
        self._self_stable_ema = 0.0
        self._sustain = 0
        self._sustain_sign = 0
        self.turn_dir = 'none'
        self.turn_active = False
        self.excursion = 0.0     # lateral_error - lateral_baseline
        self.tilt = 0.0          # row_tilt - tilt_baseline

    @staticmethod
    def _row_tilt(pts, rows, cols):
        """Orientation (rad) of the grid's long axis via PCA. Ordering- and
        overlap-agnostic: per-row y-grouping breaks when tilted rows overlap
        in y (this grid is wide and short, so they do at small rotations).
        The long axis (7-dot direction) is PC1; we force it to point rightward
        so the sign is a stable tilt convention (not the arbitrary eigenvector
        sign). Angle == the grid's in-image yaw."""
        p = np.asarray(pts, dtype=np.float64)
        p = p - p.mean(axis=0)
        cov = np.cov(p[:, 0], p[:, 1])
        _, v = np.linalg.eigh(cov)            # v[:, -1] = largest-variance axis
        axis = v[:, -1]
        if axis[0] < 0:
            axis = -axis                      # force pointing right
        return float(np.arctan2(axis[1], axis[0]))

    def update(self, found, lateral_error, pts, steer, rows, cols):
        # how much the follower is currently correcting its own heading
        self._self_stable_ema = ((1 - self.self_stable_alpha) * self._self_stable_ema
                                 + self.self_stable_alpha * abs(steer))
        unstable = self._self_stable_ema > self.self_stable_thr

        if found:
            raw_tilt = self._row_tilt(pts, rows, cols) if pts is not None else 0.0
            self.tilt = raw_tilt - self.tilt_baseline
            self.excursion = lateral_error - self.lateral_baseline
            # learn baselines only while steadily tracking straight, no turn
            if not unstable and not self.turn_active:
                a = self.baseline_alpha
                self.lateral_baseline = (1 - a) * self.lateral_baseline + a * lateral_error
                self.tilt_baseline = (1 - a) * self.tilt_baseline + a * raw_tilt
        else:
            self.excursion = 0.0
            self.tilt = 0.0

        e_thr = self.excursion_thr_strong if unstable else self.excursion_thr
        t_thr = self.tilt_thr_strong if unstable else self.tilt_thr

        # candidate direction: either channel over threshold; if both over,
        # they must agree in sign, else it's noise -> no candidate.
        es = 1 if self.excursion > e_thr else (-1 if self.excursion < -e_thr else 0)
        ts = 1 if self.tilt > t_thr else (-1 if self.tilt < -t_thr else 0)
        if es != 0 and ts != 0:
            sign = es if es == ts else 0
        else:
            sign = es or ts

        if sign != 0:
            if sign == self._sustain_sign:
                self._sustain += 1
            else:
                self._sustain_sign = sign
                self._sustain = 1
            if self._sustain >= self.sustain_frames:
                self.turn_active = True
                raw = 'L' if sign < 0 else 'R'
                self.turn_dir = ('R' if raw == 'L' else 'L') if self.sign_invert else raw
        else:
            if self.turn_active and abs(self.excursion) < self.clear_thr \
                    and abs(self.tilt) < self.clear_tilt:
                self.turn_active = False
                self.turn_dir = 'none'
                self._sustain = 0
                self._sustain_sign = 0
            elif not self.turn_active:
                self._sustain = max(0, self._sustain - 1)
                if self._sustain == 0:
                    self._sustain_sign = 0

        return self.turn_dir, self.turn_active


# =====================================================================
# SECTION 4: CONTROL  (PD steering + distance speed control + smooth ramp)
#   - steering : PD on lateral_error -> differential turn
#   - speed    : pure proportional on span error around target_span. At the
#                target distance speed is 0 (it holds position); when too
#                close (span >= stop_span) speed is clamped to 0 (this is
#                how the follower stops when the leader stops).
#   - ramp     : current speed eases toward target so starts/stops are soft
# =====================================================================

class Controller:
    def __init__(self, cfg):
        c = cfg['control']
        self.max_speed = float(c['max_speed'])
        self.chase_speed = float(c['chase_speed'])    # creep while reacquiring a far leader
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

        # Low-pass smoothing on the raw detection signals. Without this, a few-px
        # blob-centre jitter (or a raw->clahe method switch shifting mean_x a bit)
        # gets differentiated by the D-term into an alternating steer kick that
        # rocks the bot and then waterfalls (wobble -> noisier detection -> bigger
        # kick). error_alpha is the weight of the NEW sample (0.3 = fairly smooth,
        # like the leader's lane agent). A D-term deadband stops sub-threshold
        # noise from being differentiated at all.
        self.error_alpha = float(c.get('error_alpha', 0.3))
        self.d_deadband = float(c.get('d_deadband', 0.01))

        self._cur_v = 0.0       # ramped forward speed
        self._prev_e = 0.0      # previous (filtered) lateral error (for D term)
        self._filt_e = 0.0      # low-passed lateral_error
        self._filt_span = 0.0   # low-passed span

    def steering(self, lateral_error):
        # low-pass the error so detection jitter doesn't kick the wheels
        self._filt_e = (1.0 - self.error_alpha) * self._filt_e \
            + self.error_alpha * lateral_error
        e = self._filt_e
        d = e - self._prev_e
        self._prev_e = e
        # D-term deadband: ignore tiny deltas that are just measurement noise
        if abs(d) < self.d_deadband:
            d = 0.0
        return self.steer_kp * e + self.steer_kd * d

    def filtered_span(self, span):
        """Low-pass the span so distance speed doesn't twitch on detection jitter."""
        self._filt_span = (1.0 - self.error_alpha) * self._filt_span \
            + self.error_alpha * span
        return self._filt_span

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
        self._filt_e = 0.0
        self._filt_span = 0.0

    @property
    def current_speed(self):
        return self._cur_v


def _sync_cfg():
    """Re-read config into the live objects (live tuning from the dashboard)."""
    global _ctrl, _turn, _leader, CFG
    if CFG is None:
        return
    c = CFG['control']
    if _ctrl is not None:
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
        _ctrl.error_alpha = float(c.get('error_alpha', _ctrl.error_alpha))
        _ctrl.d_deadband = float(c.get('d_deadband', _ctrl.d_deadband))
    t = CFG.get('turn', {})
    if _turn is not None:
        _turn.excursion_thr = float(t.get('excursion_thr', _turn.excursion_thr))
        _turn.excursion_thr_strong = float(t.get('excursion_thr_strong', _turn.excursion_thr_strong))
        _turn.tilt_thr = float(t.get('tilt_thr', _turn.tilt_thr))
        _turn.tilt_thr_strong = float(t.get('tilt_thr_strong', _turn.tilt_thr_strong))
        _turn.sustain_frames = int(t.get('sustain_frames', _turn.sustain_frames))
        _turn.clear_thr = float(t.get('clear_thr', _turn.clear_thr))
        _turn.clear_tilt = float(t.get('clear_tilt', _turn.clear_tilt))
        _turn.baseline_alpha = float(t.get('baseline_alpha', _turn.baseline_alpha))
        _turn.self_stable_thr = float(t.get('self_stable_thr', _turn.self_stable_thr))
        _turn.self_stable_alpha = float(t.get('self_stable_alpha', _turn.self_stable_alpha))
        _turn.sign_invert = bool(t.get('sign_invert', _turn.sign_invert))
    d = CFG.get('detection', {})
    if _leader is not None:
        _leader.hold_frames = int(d.get('hold_frames', _leader.hold_frames))
        _leader.roi_pad = int(d.get('roi_pad', _leader.roi_pad))
        _leader.use_clahe = bool(d.get('clahe', _leader.use_clahe))


# =====================================================================
# SECTION 5: LED SIGNALLING
#   Corner LEDs communicate state + the detected leader turn direction.
#   Guarded with `if leds:` because the LED hardware may fail to init.
#   Indices: 0=front-left, 2=front-right, 3=back-left, 4=back-right.
# =====================================================================

_COLORS = {
    'FOLLOW': [0.0, 1.0, 0.0],   # green
    'HOLD':   [1.0, 0.7, 0.0],   # amber — holding distance / leader stopped
    'SEARCH': [0.0, 0.3, 1.0],   # blue
}


def set_leds(leds, state, turn_dir='none', turn_active=False):
    if not leds:
        return
    base = _COLORS.get(state, [0.0, 0.0, 0.0])
    leds.set_rgb(3, base)
    leds.set_rgb(4, base)
    if turn_active and turn_dir in ('L', 'R'):
        amber = [1.0, 0.6, 0.0]
        if turn_dir == 'R':
            leds.set_rgb(2, amber); leds.set_rgb(0, base)
        else:
            leds.set_rgb(0, amber); leds.set_rgb(2, base)
    else:
        leds.set_rgb(0, base); leds.set_rgb(2, base)


# =====================================================================
# SECTION 6: DEBUG OVERLAY + STATE MACHINE + main()
#   States: SEARCH (no leader) / FOLLOW / HOLD (leader too close / stopped).
#   SEARCH currently just stops (future: follow lanes to reacquire).
# =====================================================================

def _annotate(bgr, det):
    """Rich debug overlay drawn by the VIDEO THREAD on the live frame.
    `det` is the DETECTION dict published by the agent (lock-protected).
    Draws: grid bbox + dots, baseline line, turn arrow, excursion bar,
    state + span + excursion + tilt + detection method."""
    if not det:
        return bgr
    img = bgr.copy()
    h, w = img.shape[:2]

    centers = det.get('centers')
    method = det.get('method', '')
    if centers is not None:
        c = centers.astype(int)
        x0, y0 = int(c[:, 0].min()), int(c[:, 1].min())
        x1, y1 = int(c[:, 0].max()), int(c[:, 1].max())
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 1)
        col = (0, 255, 0) if method != 'held' else (0, 180, 255)
        for (x, y) in c:
            cv2.circle(img, (x, y), 3, col, -1)
        # lateral baseline as a magenta vertical line (the bot's "rest" centre)
        bx = int(w / 2.0 + det.get('lateral_baseline', 0.0) * (w / 2.0))
        cv2.line(img, (bx, 0), (bx, h), (255, 0, 255), 1)
        cv2.putText(img, 'base', (bx + 2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    # big turn arrow, top-center
    if det.get('turn_active') and det.get('turn_dir') in ('L', 'R'):
        arrow = '<<< LEFT' if det['turn_dir'] == 'L' else 'RIGHT >>>'
        cv2.putText(img, arrow, (w // 2 - 70, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(img, arrow, (w // 2 - 70, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

    # excursion bar (bottom-center): yellow tick = how far off baseline
    bar_w = int(w * 0.4)
    bar_x = w // 2 - bar_w // 2
    bar_y = h - 22
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (40, 40, 40), -1)
    cv2.line(img, (w // 2, bar_y - 2), (w // 2, bar_y + 10), (90, 90, 90), 1)
    ex = int(np.clip(det.get('excursion', 0.0), -1, 1) * bar_w / 2)
    cv2.line(img, (w // 2, bar_y + 4), (w // 2 + ex, bar_y + 4), (0, 255, 255), 2)

    state = det.get('state', '')
    span = det.get('span', 0.0)
    lat = det.get('lateral_error', 0.0)
    exc = det.get('excursion', 0.0)
    tilt = det.get('tilt', 0.0)
    quality = det.get('quality', 0.0)
    txt = (f'{state}  span={span:.2f}  e={lat:+.2f}  '
           f'exc={exc:+.2f}  tilt={tilt:+.2f}  '
           f'{method or "-"}  q={quality:.1f}')
    cv2.putText(img, txt, (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
    cv2.putText(img, txt, (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main(camera, wheels, leds, stop_event):
    global STATUS, CFG, _leader, _turn, _ctrl

    CFG = load_config()
    _leader = LeaderDetector(CFG)
    _turn = TurnDetector(CFG)
    _ctrl = Controller(CFG)

    dt = 1.0 / float(CFG['control']['loop_hz'])

    lost_count = 0
    last_e = 0.0               # last lateral error while the leader was seen
    last_span = 0.0            # last span while seen (tells us WHY we lost it)
    last_turn = 0.0            # last steering command while following
    red_stop_active = False    # track if we've already signaled a red stop
    red_consecutive_count = 0  # require N consecutive frames of red before stopping
    red_stop_sustain = 1       # require 1 frame of red (immediate response, no delay)

    try:
        while not stop_event.is_set():
            _sync_cfg()
            ok, frame = camera.read()
            if not ok or frame is None:
                stop_event.wait(0.02)
                continue

            # Check for red stop lines first
            red_pixels = preprocessing.detect_red_stop(frame, CFG.get('red_stop', {}))
            red_threshold = CFG.get('red_stop', {}).get('detection_threshold', 800)
            red_frame_detected = red_pixels >= red_threshold
            
            # Count consecutive frames of red (require sustain frames to avoid noise)
            if red_frame_detected:
                red_consecutive_count += 1
            else:
                red_consecutive_count = 0
            
            # Trigger stop only if sustained red detection
            red_detected = red_consecutive_count >= red_stop_sustain
            
            # Signal leader to stop if red detected
            if red_detected and not red_stop_active:
                if wheels:
                    wheels.stop_leader()
                red_stop_active = True
            elif not red_detected and red_stop_active:
                red_stop_active = False

            found, lateral_error, span, centers, method, quality = _leader.detect(frame)

            speed_cap = _ctrl.max_speed

            # --- pick state + commands (runs identically paused or active) ---
            steer = 0.0
            if found:
                lost_count = 0
                last_e, last_span = lateral_error, span
                steer = _ctrl.steering(lateral_error)
                last_turn = steer
                # distance control on the FILTERED span (jitter-resistant)
                fspan = _ctrl.filtered_span(span)
                target_v = _ctrl.distance_speed(fspan, speed_cap)
                state = 'HOLD' if fspan >= _ctrl.stop_span else 'FOLLOW'
            else:
                lost_count += 1
                if lost_count < _ctrl.search_after:
                    # brief loss: react to why we lost it (never spin)
                    if last_span >= _ctrl.target_span:
                        target_v, steer, state = 0.0, 0.0, 'HOLD'
                    else:
                        target_v, steer, state = _ctrl.chase_speed, last_turn * 0.5, 'FOLLOW'
                else:
                    # long loss: stop for now (future: follow lanes to reacquire)
                    target_v, steer, state = 0.0, 0.0, 'SEARCH'
                    _ctrl.reset_steer()

            # turn detection runs every frame (held frames keep the last pose)
            tdir, tactive = _turn.update(
                found, lateral_error, centers, steer,
                _leader.rows, _leader.cols)

            # --- ramp speed, mix into wheels ---
            # PAUSE only stops the wheels — detection, control decision, LEDs,
            # overlay and status all keep running so you can verify behaviour
            # stationary. The dashboard red dot is the sole pause indicator.
            v = _ctrl.ramp(target_v)
            if PAUSED:
                wheels.set_wheels_speed(0.0, 0.0)
            else:
                left = float(np.clip(v + steer, -1.0, 1.0))
                right = float(np.clip(v - steer, -1.0, 1.0))
                wheels.set_wheels_speed(left, right)

            # --- signal + publish results (identical paused or active) ---
            set_leds(leds, state, tdir, tactive)
            det = {
                'found': found, 'centers': centers, 'method': method,
                'quality': quality, 'span': span,
                'lateral_error': lateral_error, 'state': state,
                'steer': steer, 'turn_dir': tdir, 'turn_active': tactive,
                'excursion': _turn.excursion, 'tilt': _turn.tilt,
                'lateral_baseline': _turn.lateral_baseline,
                'red_detected': red_detected, 'red_pixels': red_pixels,
            }
            with _det_lock:
                DETECTION.update(det)
            STATUS = {
                'state': state, 'found': found, 'span': round(span, 3),
                'lateral_error': round(lateral_error, 3), 'steer': round(steer, 3),
                'speed': round(v, 3),
                'turn_dir': tdir, 'turn_active': tactive,
                'baseline': round(_turn.lateral_baseline, 3),
                'excursion': round(_turn.excursion, 3),
                'tilt': round(_turn.tilt, 3),
                'detection_method': method, 'quality': round(quality, 2),
                'lane_fallback': False,
                'red_detected': red_detected, 'red_pixels': red_pixels,
            }

            stop_event.wait(dt)
    finally:
        # Contract: always stop the motors and lights on exit.
        wheels.set_wheels_speed(0.0, 0.0)
        if leds:
            leds.all_off()
