"""
Convoying follower — lane-primary architecture.

STEERING is always from the shared LaneServoingAgent (stays in lane).
The dot grid on the leader's back controls SPEED (span → distance) and
TURN DETECTION (lateral excursion + row-tilt via PCA). When a turn is
detected, the follower executes the same open-loop arc as the leader.

FSM:
  LANE_FOLLOW        — lane steering + grid speed modulation + turn detection
  TURNING            — open-loop arc (same direction/params as leader), blind
  LANE_FOLLOW_TIMEOUT — grid lost, no recent turn: keep lane-following, then stop
  STOP               — wheels at 0; resume LANE_FOLLOW if grid reacquired

No AprilTag / sign detection — the LEADER handles stop signs. No solvePnP /
calibration required. PAUSED stops only the wheels.
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
_turn = None
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
        'excursion_thr': 0.35,
        'excursion_thr_strong': 0.55,
        'tilt_thr': 0.10,
        'tilt_thr_strong': 0.18,
        'sustain_frames': 4,
        'clear_thr': 0.12,
        'clear_tilt': 0.04,
        'baseline_alpha': 0.02,
        'self_stable_thr': 0.10,
        'self_stable_alpha': 0.15,
        'sign_invert': False,
        # Turn arc params — same finetuned values as the leader
        'preturn_right_s': 0.83,
        'preturn_left_s': 0.33,
        'preturn_speed': 0.20,
        'turn_right_s': 1.04,
        'turn_left_s': 2.33,
        'turn_right_pwm': [0.60, 0.05],
        'turn_left_pwm': [0.20, 0.50],
        'exit_s': 0.125,
        'exit_speed': 0.40,
        'turn_cooldown_s': 3.0,       # suppress turn detection this long after a turn
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
# SECTION 2: LEADER DETECTION  (dot grid — unchanged from follower-v2)
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
# SECTION 3: TURN DETECTION  (unchanged from follower-v2)
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
        self.excursion = 0.0
        self.tilt = 0.0

    @staticmethod
    def _row_tilt(pts, rows, cols):
        p = np.asarray(pts, dtype=np.float64)
        p = p - p.mean(axis=0)
        cov = np.cov(p[:, 0], p[:, 1])
        _, v = np.linalg.eigh(cov)
        axis = v[:, -1]
        if axis[0] < 0:
            axis = -axis
        return float(np.arctan2(axis[1], axis[0]))

    def update(self, found, lateral_error, pts, steer, rows, cols):
        self._self_stable_ema = ((1 - self.self_stable_alpha) * self._self_stable_ema
                                 + self.self_stable_alpha * abs(steer))
        unstable = self._self_stable_ema > self.self_stable_thr

        if found:
            raw_tilt = self._row_tilt(pts, rows, cols) if pts is not None else 0.0
            self.tilt = raw_tilt - self.tilt_baseline
            self.excursion = lateral_error - self.lateral_baseline
            if not unstable and not self.turn_active:
                a = self.baseline_alpha
                self.lateral_baseline = (1 - a) * self.lateral_baseline + a * lateral_error
                self.tilt_baseline = (1 - a) * self.tilt_baseline + a * raw_tilt
        else:
            self.excursion = 0.0
            self.tilt = 0.0

        e_thr = self.excursion_thr_strong if unstable else self.excursion_thr
        t_thr = self.tilt_thr_strong if unstable else self.tilt_thr

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

    def reset(self):
        """Full reset after a completed turn — clears state AND baselines so
        the detector doesn't immediately re-fire on the stale excursion/tilt
        values (which would cause an infinite turn loop)."""
        self.turn_active = False
        self.turn_dir = 'none'
        self._sustain = 0
        self._sustain_sign = 0
        self.excursion = 0.0
        self.tilt = 0.0
        self.lateral_baseline = 0.0
        self.tilt_baseline = 0.0
        self._self_stable_ema = 0.0


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
    global _ctrl, _turn, _leader, CFG
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
# =====================================================================

_COLORS = {
    'LANE_FOLLOW':         [0.0, 1.0, 0.0],   # green
    'TURNING':             [0.3, 0.3, 1.0],   # blue
    'LANE_FOLLOW_TIMEOUT': [1.0, 0.7, 0.0],   # amber
    'STOP':                [1.0, 0.0, 0.0],   # red
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
#   LANE_FOLLOW  — lane steering + grid speed + turn detection
#   TURNING      — open-loop arc (blind, same params as leader)
#   LANE_FOLLOW_TIMEOUT — grid lost, keep lane-following, then stop
#   STOP         — wheels at 0
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
        bx = int(w / 2.0 + det.get('lateral_baseline', 0.0) * (w / 2.0))
        cv2.line(img, (bx, 0), (bx, h), (255, 0, 255), 1)
        cv2.putText(img, 'base', (bx + 2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    if det.get('turn_dir') in ('L', 'R'):
        arrow = '<<< LEFT' if det['turn_dir'] == 'L' else 'RIGHT >>>'
        cv2.putText(img, arrow, (w // 2 - 70, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(img, arrow, (w // 2 - 70, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

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
    speed_scale = det.get('speed_scale', 1.0)
    lane_det = det.get('lane_detected', False)
    txt = (f'{state}  span={span:.2f}  scale={speed_scale:.2f}  '
           f'lane={lane_det}  e={lat:+.2f}  exc={exc:+.2f}  tilt={tilt:+.2f}  '
           f'{method or "-"}  q={quality:.1f}')
    cv2.putText(img, txt, (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 4)
    cv2.putText(img, txt, (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def main(camera, wheels, leds, stop_event):
    global STATUS, CFG, _leader, _turn, _ctrl, _lane_agent

    CFG = load_config()
    _leader = LeaderDetector(CFG)
    _turn = TurnDetector(CFG)
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
    lost_since = None      # monotonic time when grid was first lost
    turn_cooldown_until = 0.0  # suppress turn detection until this time
    turn_cooldown_s = float(t.get('turn_cooldown_s', 3.0))

    try:
        while not stop_event.is_set():
            _sync_cfg()
            ok, frame = camera.read()
            if not ok or frame is None:
                stop_event.wait(0.02)
                continue

            now = time.monotonic()

            # --- lane following (always runs for steering) ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lane_l, lane_r = _lane_agent.compute_commands(rgb)
            lane_info = _lane_agent.last_debug_info
            lane_detected = bool(lane_info.get('lane_detected', False))

            # --- grid detection (for speed + turn detection) ---
            found, lateral_error, span, centers, method, quality = _leader.detect(frame)

            # --- FSM ---
            pwm_l, pwm_r = 0.0, 0.0
            speed_scale = 1.0
            tdir, tactive = 'none', False

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

                    # turn detection (suppressed during cooldown after a turn)
                    if now >= turn_cooldown_until:
                        tdir, tactive = _turn.update(
                            found, lateral_error, centers,
                            abs(lane_l - lane_r),  # steer magnitude from lane agent
                            _leader.rows, _leader.cols)
                    else:
                        tdir, tactive = 'none', False

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
                # blind open-loop arc (same params as leader)
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
                        _turn.reset()
                        _ctrl.reset()
                        lost_since = None
                        turn_cooldown_until = now + turn_cooldown_s

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
                'excursion': _turn.excursion, 'tilt': _turn.tilt,
                'lateral_baseline': _turn.lateral_baseline,
                'lane_detected': lane_detected,
            }
            with _det_lock:
                DETECTION.update(det)
            STATUS = {
                'state': state, 'found': found, 'span': round(span, 3),
                'lateral_error': round(lateral_error, 3),
                'speed_scale': round(speed_scale, 3),
                'turn_dir': tdir, 'turn_active': tactive,
                'baseline': round(_turn.lateral_baseline, 3),
                'excursion': round(_turn.excursion, 3),
                'tilt': round(_turn.tilt, 3),
                'detection_method': method, 'quality': round(quality, 2),
                'lane_detected': lane_detected,
                'lane_fallback': state in ('LANE_FOLLOW_TIMEOUT',),
            }

            stop_event.wait(dt)
    finally:
        wheels.set_wheels_speed(0.0, 0.0)
        if leds:
            leds.all_off()
