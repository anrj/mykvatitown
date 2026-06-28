"""
Convoying follower — lane-primary architecture with LED-blink turn trigger.

STEERING is always from the shared LaneServoingAgent (stays in lane).
The dot grid on the leader's back controls SPEED (span → distance).
TURN DETECTION is driven by the leader's back-LED amber blink (5 Hz): the
follower detects amber pixels in the left/right halves of the frame, watches
the ~5 Hz cadence for a few frames, then runs the same open-loop arc as the
leader. This replaces the fragile PCA-on-the-dot-grid turn detector that
spuriously fired when the grid foreshortened mid-turn.

FSM:
  LANE_FOLLOW  — lane steering + grid speed modulation + LED-blink detection
  TURNING      — open-loop arc (same direction/params as leader), blind
  STOP         — wheels at 0; resume LANE_FOLLOW if grid reacquired

No AprilTag / sign detection — the LEADER handles stop signs. PAUSED stops
only the wheels.
"""

import os
import time
import threading

import cv2
import numpy as np
import yaml

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent

# Published for the web UI / debugging (the servers read these).
DETECTION = {}
_det_lock = threading.Lock()
STATUS = {}

CFG = None
_leader = None
_blink = None
_ctrl = None
_lane_agent = None

PAUSED = True
CONFIG_FILE = 'project_config.yaml'

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
        'search_turn': 0.10, 'search_after_frames': 24, 'loop_hz': 24,
        'error_alpha': 0.3, 'd_deadband': 0.01,
    },
    'detection': {
        'hold_frames': 3,
        'roi_pad': 30,
        'clahe': True,
    },
    'turn': {
        # LED blink detector parameters
        'blink_hz': 5.0,                 # leader's expected blink cadence
        'blink_sustain_frames': 4,      # consecutive amber frames to confirm
        'blink_blink_frames': 3,        # how many of those must be "on"
        'min_amber_frac': 0.002,        # min amber pixel fraction to be "lit"
        'side_ratio': 1.8,              # left/right amber-count ratio to pick a side
        'blink_window_s': 0.6,                # sliding window for blink edge counting
        'blink_required_edges': 3,            # edges in window to confirm a turn
        # Open-loop arc params (same finetuned values as the leader)
        'preturn_right_s': 0.83,
        'preturn_left_s': 0.33,
        'preturn_speed': 0.20,
        'turn_right_s': 1.04,
        'turn_left_s': 2.33,
        'turn_right_pwm': [0.60, 0.05],
        'turn_left_pwm': [0.20, 0.50],
        'exit_s': 0.125,
        'exit_speed': 0.40,
        'turn_cooldown_s': 3.0,         # suppress turn detection this long after a turn
    },
    'lane_fallback': {
        'enabled': True,
        'lane_follow_timeout_s': 5.0,
    },
    'camera': {'matrix': None, 'dist_coeffs': None},
}


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
# SECTION 2: LEADER DETECTION  (dot grid — used for speed/distance only)
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
        self._last_centers = None
        self._last_bbox = None
        self._hold_left = 0

    @staticmethod
    def _make_blob_detector(min_area):
        p = cv2.SimpleBlobDetector_Params()
        p.filterByColor = True
        p.blobColor = 0
        p.filterByArea = True
        p.minArea = min_area
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
# SECTION 3: LED-BLINK TURN DETECTION
# Replaces the dot-grid PCA detector. The leader blinks its back LEDs amber
# at ~5 Hz during its STOP+turn phase; the follower watches for that amber
# cadence, with L/R disambiguated by which side of the frame the amber
# dominates.
# =====================================================================

class LEDBlinkDetector:
    """Detects the leader's amber LED blink and returns (turn_dir, turn_active).

    Tracks lit↔unlit edges (False→True transitions) in a sliding time window.
    A real 5 Hz amber blink with ~50% duty produces one edge every ~200 ms, so
    `required_edges` edges within `window_s` is a robust cadence signal that
    survives single-frame detection misses (unlike a naive sustained-lit count).
    L/R is decided from which half of the frame the amber pixels dominate.
    """

    # Amber HSV — matches DuckieTown's LED amber RGB (1.0, 0.6, 0.0), which is
    # roughly H~25, S~170, V~255 in OpenCV scale.
    _AMBER_LOW = np.array([15, 100, 100], dtype=np.uint8)
    _AMBER_HIGH = np.array([35, 255, 255], dtype=np.uint8)

    def __init__(self, cfg):
        t = cfg.get('turn', {})
        self.blink_hz = float(t.get('blink_hz', 5.0))
        period = 1.0 / max(0.01, self.blink_hz)
        # Default window = ~3 blink periods; required_edges = ~3 edges.
        self.window_s = float(t.get('blink_window_s', period * 3.0))
        self.required_edges = int(t.get('blink_required_edges', 3))
        self.min_amber_frac = float(t.get('min_amber_frac', 0.002))
        self.side_ratio = float(t.get('side_ratio', 1.8))

        self._edge_times = []        # monotonic timestamps of lit-edges in window
        self._last_was_lit = False
        self.turn_dir = 'none'
        self.turn_active = False
        self.amber_left = 0
        self.amber_right = 0
        self.last_lit = False
        self.edges_in_window = 0
        self.last_edge_delta = 0.0
        self._prev_edge_time = 0.0

    def _detect_amber(self, frame_bgr):
        """Count amber pixels in left and right halves of the frame (full frame)."""
        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._AMBER_LOW, self._AMBER_HIGH)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        left = mask[:, :w // 2]
        right = mask[:, w // 2:]
        left_count = int(np.count_nonzero(left))
        right_count = int(np.count_nonzero(right))
        total = left_count + right_count
        frac = total / float(max(1, h * w))
        return left_count, right_count, frac

    def update(self, frame_bgr):
        now = time.monotonic()
        left, right, frac = self._detect_amber(frame_bgr)
        self.amber_left = left
        self.amber_right = right
        self.last_lit = frac >= self.min_amber_frac

        just_lit_edge = self.last_lit and not self._last_was_lit
        if just_lit_edge:
            # Cadence check (optional): if previous edge was at wildly wrong
            # interval vs the expected period, drop this edge.
            if self._prev_edge_time > 0.0:
                self.last_edge_delta = now - self._prev_edge_time
            self._edge_times.append(now)
            self._prev_edge_time = now

        # Drop edges older than the sliding window.
        cutoff = now - self.window_s
        while self._edge_times and self._edge_times[0] < cutoff:
            self._edge_times.pop(0)
        self.edges_in_window = len(self._edge_times)

        # Decide whether we have a confirmed blink.
        if self.edges_in_window >= self.required_edges:
            # Snap the dominant side from the current frame's amber-split.
            if left > self.side_ratio * max(1, right):
                self.turn_dir = 'L'
            elif right > self.side_ratio * max(1, left):
                self.turn_dir = 'R'
            else:
                self.turn_dir = 'L' if left >= right else 'R'
            self.turn_active = True
        else:
            if not self.turn_active:
                self.turn_dir = 'none'

        self._last_was_lit = self.last_lit
        return self.turn_dir, self.turn_active

    def reset(self):
        """Full reset after the open-loop arc has been completed."""
        self._edge_times = []
        self._last_was_lit = False
        self._prev_edge_time = 0.0
        self.last_edge_delta = 0.0
        self.turn_dir = 'none'
        self.turn_active = False
        self.amber_left = 0
        self.amber_right = 0
        self.edges_in_window = 0


# =====================================================================
# SECTION 4: SPEED CONTROL  (distance from grid span — steering is lane-based)
# =====================================================================

class Controller:
    def __init__(self, cfg):
        c = cfg['control']
        self.max_speed = float(c['max_speed'])
        self.chase_speed = float(c['chase_speed'])
        self.dist_kp = float(c['dist_kp'])
        self.accel = float(c['accel_rate'])
        self.decel = float(c['decel_rate'])
        self.target_span = float(cfg['leader']['target_span'])
        self.stop_span = float(cfg['leader']['stop_span'])
        self.deadband = float(cfg['leader']['span_deadband'])
        self.error_alpha = float(c.get('error_alpha', 0.3))
        self._filt_span = 0.0
        self._cur_v = 0.0

    def filtered_span(self, span):
        self._filt_span = (1.0 - self.error_alpha) * self._filt_span \
            + self.error_alpha * span
        return self._filt_span

    def distance_speed(self, span, speed_cap):
        if span >= self.stop_span:
            return 0.0
        err = self.target_span - span
        if abs(err) < self.deadband:
            return 0.0
        return float(np.clip(self.dist_kp * err, 0.0, speed_cap))

    def ramp(self, target_v):
        if target_v > self._cur_v:
            self._cur_v = min(target_v, self._cur_v + self.accel)
        else:
            self._cur_v = max(target_v, self._cur_v - self.decel)
        return self._cur_v

    def reset(self):
        self._filt_span = 0.0
        self._cur_v = 0.0


def _sync_cfg():
    global _ctrl, _blink, _leader, CFG
    if CFG is None:
        return
    c = CFG['control']
    if _ctrl is not None:
        _ctrl.max_speed = float(c['max_speed'])
        _ctrl.chase_speed = float(c['chase_speed'])
        _ctrl.dist_kp = float(c['dist_kp'])
        _ctrl.accel = float(c['accel_rate'])
        _ctrl.decel = float(c['decel_rate'])
        _ctrl.target_span = float(CFG['leader']['target_span'])
        _ctrl.stop_span = float(CFG['leader']['stop_span'])
        _ctrl.deadband = float(CFG['leader']['span_deadband'])
        _ctrl.error_alpha = float(c.get('error_alpha', _ctrl.error_alpha))
    t = CFG.get('turn', {})
    if _blink is not None:
        _blink.blink_hz = float(t.get('blink_hz', _blink.blink_hz))
        _blink.window_s = float(t.get('blink_window_s', _blink.window_s))
        _blink.required_edges = int(t.get('blink_required_edges', _blink.required_edges))
        _blink.min_amber_frac = float(t.get('min_amber_frac', _blink.min_amber_frac))
        _blink.side_ratio = float(t.get('side_ratio', _blink.side_ratio))
    d = CFG.get('detection', {})
    if _leader is not None:
        _leader.hold_frames = int(d.get('hold_frames', _leader.hold_frames))
        _leader.roi_pad = int(d.get('roi_pad', _leader.roi_pad))
        _leader.use_clahe = bool(d.get('clahe', _leader.use_clahe))


# =====================================================================
# SECTION 5: LED SIGNALLING  (the follower's own dashboard LEDs)
# =====================================================================

_COLORS = {
    'LANE_FOLLOW': [0.0, 1.0, 0.0],   # green
    'TURNING':     [0.3, 0.3, 1.0],   # blue
    'STOP':        [1.0, 0.0, 0.0],   # red
}


def set_leds(leds, state, turn_dir='none'):
    if not leds:
        return
    base = _COLORS.get(state, [0.0, 0.0, 0.0])
    for i in (3, 4):
        leds.set_rgb(i, base)
    if state == 'TURNING' and turn_dir in ('L', 'R'):
        amber = [1.0, 0.6, 0.0]
        if turn_dir == 'R':
            leds.set_rgb(2, amber); leds.set_rgb(0, base)
        else:
            leds.set_rgb(0, amber); leds.set_rgb(2, base)
    else:
        leds.set_rgb(0, base); leds.set_rgb(2, base)


# =====================================================================
# SECTION 6: DEBUG OVERLAY + STATE MACHINE + main()
# =====================================================================

def _annotate(bgr, det):
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

    if det.get('turn_dir') in ('L', 'R'):
        arrow = '<<< LEFT' if det['turn_dir'] == 'L' else 'RIGHT >>>'
        cv2.putText(img, arrow, (w // 2 - 70, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(img, arrow, (w // 2 - 70, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

    # LED blink side bar (left/right amber counts).
    bar_w = int(w * 0.4)
    bar_x = w // 2 - bar_w // 2
    bar_y = h - 22
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (40, 40, 40), -1)
    cv2.line(img, (w // 2, bar_y - 2), (w // 2, bar_y + 10), (90, 90, 90), 1)
    left = int(det.get('amber_left', 0))
    right = int(det.get('amber_right', 0))
    max_side = max(left, right, 1)
    half = bar_w // 2
    lw = int(half * (left / float(max_side)))
    rw = int(half * (right / float(max_side)))
    cv2.rectangle(img, (w // 2 - lw, bar_y + 1), (w // 2, bar_y + 7), (0, 165, 255), -1)
    cv2.rectangle(img, (w // 2, bar_y + 1), (w // 2 + rw, bar_y + 7), (0, 165, 255), -1)

    state = det.get('state', '')
    span = det.get('span', 0.0)
    lat = det.get('lateral_error', 0.0)
    quality = det.get('quality', 0.0)
    speed_scale = det.get('speed_scale', 1.0)
    lane_det = det.get('lane_detected', False)
    lit = det.get('lit', False)
    edges = det.get('edges_in_window', 0)
    txt = (f'{state}  span={span:.2f}  scale={speed_scale:.2f}  '
           f'lane={lane_det}  e={lat:+.2f}  '
           f'amber L={left} R={right} lit={int(lit)} edges={edges}  '
           f'{method or "-"}  q={quality:.1f}')
    cv2.putText(img, txt, (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 4)
    cv2.putText(img, txt, (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return img


def _reset_lane_agent_filters():
    """Reset LaneServoingAgent's smoothing filters so the first frame after a
    blind TURNING arc doesn't apply a stale steering command (fixes the
    post-turn hard-steer jerk)."""
    if _lane_agent is None:
        return
    _lane_agent._filtered_error = 0.0
    _lane_agent._prev_error = 0.0
    _lane_agent._filtered_steering = 0.0
    _lane_agent._left_history.clear()
    _lane_agent._right_history.clear()


def main(camera, wheels, leds, stop_event):
    global STATUS, CFG, _leader, _blink, _ctrl, _lane_agent

    CFG = load_config()
    _leader = LeaderDetector(CFG)
    _blink = LEDBlinkDetector(CFG)
    _ctrl = Controller(CFG)
    _lane_agent = LaneServoingAgent()

    dt = 1.0 / float(CFG['control']['loop_hz'])
    t = CFG.get('turn', {})
    lf = CFG.get('lane_fallback', {})
    lane_follow_timeout_s = float(lf.get('lane_follow_timeout_s', 5.0))

    # FSM state
    state = 'LANE_FOLLOW'
    turn_state_start = 0.0
    turn_arc_dir = 'none'
    turn_phase = ''        # 'preturn' | 'turn' | 'exit'
    lost_since = None
    turn_cooldown_until = 0.0
    turn_cooldown_s = float(t.get('turn_cooldown_s', 3.0))

    try:
        while not stop_event.is_set():
            _sync_cfg()
            ok, frame = camera.read()
            if not ok or frame is None:
                stop_event.wait(0.02)
                continue

            now = time.monotonic()

            # --- lane following (skipped during TURNING to avoid filter contamination) ---
            lane_l = lane_r = 0.0
            lane_info = {}
            lane_detected = False
            if state != 'TURNING':
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                lane_l, lane_r = _lane_agent.compute_commands(rgb)
                lane_info = _lane_agent.last_debug_info
                lane_detected = bool(lane_info.get('lane_detected', False))

            # --- grid detection (for speed) ---
            found, lateral_error, span, centers, method, quality = _leader.detect(frame)

            # --- LED blink turn detection (only in LANE_FOLLOW, suppressed by cooldown) ---
            tdir, tactive = 'none', False
            if state == 'LANE_FOLLOW' and now >= turn_cooldown_until:
                tdir, tactive = _blink.update(frame)

            # --- FSM ---
            pwm_l, pwm_r = 0.0, 0.0
            speed_scale = 1.0

            if state == 'LANE_FOLLOW':
                if found:
                    lost_since = None
                    fspan = _ctrl.filtered_span(span)

                    # speed modulation from grid distance
                    if fspan >= _ctrl.stop_span:
                        speed_scale = 0.0        # leader stopped / too close
                    else:
                        target_speed = _ctrl.distance_speed(fspan, _ctrl.max_speed)
                        lane_base = max(_lane_agent.base_speed, 0.01)
                        speed_scale = float(np.clip(target_speed / lane_base, 0.0, 2.0))

                    pwm_l = lane_l * speed_scale
                    pwm_r = lane_r * speed_scale

                    if tactive and tdir in ('L', 'R'):
                        state = 'TURNING'
                        turn_arc_dir = tdir
                        turn_phase = 'preturn'
                        turn_state_start = now
                else:
                    # grid lost — keep lane-following at full speed
                    speed_scale = 1.0
                    pwm_l = lane_l
                    pwm_r = lane_r
                    if lost_since is None:
                        lost_since = now
                    if now - lost_since > lane_follow_timeout_s:
                        state = 'STOP'
                        _ctrl.reset()

            elif state == 'TURNING':
                # blind open-loop arc — lane steering intentionally discarded
                if turn_phase == 'preturn':
                    ps = float(t.get('preturn_speed', 0.20))
                    pwm_l = pwm_r = ps
                    preturn_s = (float(t.get('preturn_right_s', 0.83)) if turn_arc_dir == 'R'
                                 else float(t.get('preturn_left_s', 0.33)))
                    if now - turn_state_start >= preturn_s:
                        turn_phase = 'turn'
                        turn_state_start = now

                elif turn_phase == 'turn':
                    if turn_arc_dir == 'R':
                        pwm_l, pwm_r = t.get('turn_right_pwm', [0.60, 0.05])
                        turn_s = float(t.get('turn_right_s', 1.04))
                    else:
                        pwm_l, pwm_r = t.get('turn_left_pwm', [0.20, 0.50])
                        turn_s = float(t.get('turn_left_s', 2.33))
                    if now - turn_state_start >= turn_s:
                        turn_phase = 'exit'
                        turn_state_start = now

                elif turn_phase == 'exit':
                    es = float(t.get('exit_speed', 0.40))
                    pwm_l = pwm_r = es
                    if now - turn_state_start >= float(t.get('exit_s', 0.125)):
                        state = 'LANE_FOLLOW'
                        _blink.reset()
                        _ctrl.reset()
                        lost_since = None
                        turn_cooldown_until = now + turn_cooldown_s
                        # Drop lane agent's stale filter state so the first
                        # post-turn frame isn't steered by old commands.
                        _reset_lane_agent_filters()

                tdir = turn_arc_dir
                tactive = True

            elif state == 'STOP':
                pwm_l = pwm_r = 0.0
                if found:
                    state = 'LANE_FOLLOW'
                    lost_since = None
                    _ctrl.reset()

            # --- wheels (paused = movement only) ---
            if PAUSED:
                wheels.set_wheels_speed(0.0, 0.0)
            else:
                wheels.set_wheels_speed(float(pwm_l), float(pwm_r))

            # --- LEDs ---
            set_leds(leds, state, turn_arc_dir if state == 'TURNING' else tdir)

            # --- publish results ---
            det = {
                'found': found, 'centers': centers, 'method': method,
                'quality': quality, 'span': span,
                'lateral_error': lateral_error, 'state': state,
                'speed_scale': speed_scale,
                'turn_dir': tdir, 'turn_active': tactive,
                'amber_left': _blink.amber_left, 'amber_right': _blink.amber_right,
                'lit': _blink.last_lit,
                'edges_in_window': _blink.edges_in_window,
                'lane_detected': lane_detected,
            }
            with _det_lock:
                DETECTION.update(det)
            STATUS = {
                'state': state, 'found': found, 'span': round(span, 3),
                'lateral_error': round(lateral_error, 3),
                'speed_scale': round(speed_scale, 3),
                'turn_dir': tdir, 'turn_active': tactive,
                'amber_left': _blink.amber_left, 'amber_right': _blink.amber_right,
                'lit': _blink.last_lit,
                'edges': _blink.edges_in_window,
                'detection_method': method, 'quality': round(quality, 2),
                'lane_detected': lane_detected,
            }

            stop_event.wait(dt)
    finally:
        wheels.set_wheels_speed(0.0, 0.0)
        if leds:
            leds.all_off()