import os
import time
import threading

import cv2
import numpy as np
import yaml

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent

DETECTION = {}
_det_lock = threading.Lock()
STATUS = {}

CFG = None
_leader = None
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
        'dist_kp': 2.0, 'accel_rate': 0.05, 'decel_rate': 0.08,
        'search_after_frames': 24, 'loop_hz': 24,
        'error_alpha': 0.3,
    },
    'detection': {
        'hold_frames': 3,
        'roi_pad': 30,
        'clahe': True,
    },
    'turn_bias': {
        'bias_amount': 0.15,
        'bias_duration_s': 1.0,
        'bias_ramp_s': 0.3,
        'bias_excursion_thr': 0.3,
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
    global _ctrl, _leader, CFG
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
    d = CFG.get('detection', {})
    if _leader is not None:
        _leader.hold_frames = int(d.get('hold_frames', _leader.hold_frames))
        _leader.roi_pad = int(d.get('roi_pad', _leader.roi_pad))
        _leader.use_clahe = bool(d.get('clahe', _leader.use_clahe))


_COLORS = {
    'LANE_FOLLOW': [0.0, 1.0, 0.0],
    'STOP':        [1.0, 0.0, 0.0],
}


def set_leds(leds, state):
    if not leds:
        return
    base = _COLORS.get(state, [0.0, 0.0, 0.0])
    for i in (0, 2, 3, 4):
        leds.set_rgb(i, base)


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

    state = det.get('state', '')
    span = det.get('span', 0.0)
    lat = det.get('lateral_error', 0.0)
    quality = det.get('quality', 0.0)
    speed_scale = det.get('speed_scale', 1.0)
    lane_det = det.get('lane_detected', False)
    txt = (f'{state}  span={span:.2f}  scale={speed_scale:.2f}  '
           f'lane={lane_det}  e={lat:+.2f}  '
           f'{method or "-"}  q={quality:.1f}')
    cv2.putText(img, txt, (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 4)
    cv2.putText(img, txt, (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return img


def main(camera, wheels, leds, stop_event):
    global STATUS, CFG, _leader, _ctrl, _lane_agent

    CFG = load_config()
    _leader = LeaderDetector(CFG)
    _ctrl = Controller(CFG)
    _lane_agent = LaneServoingAgent()

    dt = 1.0 / float(CFG['control']['loop_hz'])
    lf = CFG.get('lane_fallback', {})
    lane_follow_timeout_s = float(lf.get('lane_follow_timeout_s', 5.0))
    tb = CFG.get('turn_bias', {})
    bias_amount = float(tb.get('bias_amount', 0.15))
    bias_duration_s = float(tb.get('bias_duration_s', 1.0))
    bias_ramp_s = float(tb.get('bias_ramp_s', 0.3))
    bias_excursion_thr = float(tb.get('bias_excursion_thr', 0.3))

    state = 'LANE_FOLLOW'
    lost_since = None
    _last_lat = 0.0
    _bias = 0.0
    _bias_target = 0.0
    _bias_start = 0.0

    try:
        while not stop_event.is_set():
            _sync_cfg()
            ok, frame = camera.read()
            if not ok or frame is None:
                stop_event.wait(0.02)
                continue

            now = time.monotonic()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lane_l, lane_r = _lane_agent.compute_commands(rgb)
            lane_info = _lane_agent.last_debug_info
            lane_detected = bool(lane_info.get('lane_detected', False))

            found, lateral_error, span, centers, method, quality = _leader.detect(frame)

            pwm_l, pwm_r = 0.0, 0.0
            speed_scale = 1.0

            if state == 'LANE_FOLLOW':
                if found:
                    lost_since = None
                    _last_lat = lateral_error
                    fspan = _ctrl.filtered_span(span)
                    if fspan >= _ctrl.stop_span:
                        speed_scale = 0.0
                    else:
                        target_speed = _ctrl.distance_speed(fspan, _ctrl.max_speed)
                        lane_base = max(_lane_agent.base_speed, 0.01)
                        speed_scale = float(np.clip(target_speed / lane_base, 0.0, 2.0))

                    pwm_l = lane_l * speed_scale
                    pwm_r = lane_r * speed_scale
                else:
                    # Trigger bias — leader lost with strong lateral excursion
                    if _bias_target == 0.0 and abs(_last_lat) > bias_excursion_thr:
                        _bias_target = bias_amount * (1.0 if _last_lat > 0 else -1.0)
                        _bias_start = now
                        print(f'[agent] turn bias target {_bias_target:+.2f} (last_lat={_last_lat:+.2f})')

                    # Compute ramped bias value
                    _bias = 0.0
                    if _bias_target != 0.0:
                        elapsed = now - _bias_start
                        if elapsed >= bias_duration_s:
                            _bias_target = 0.0
                            _last_lat = 0.0
                        else:
                            ramp = min(1.0, elapsed / bias_ramp_s)
                            _bias = _bias_target * ramp

                    speed_scale = 1.0
                    pwm_l = lane_l
                    pwm_r = lane_r
                    if _bias != 0.0:
                        pwm_l = max(0.0, pwm_l + _bias)
                        pwm_r = max(0.0, pwm_r - _bias)
                    if lost_since is None:
                        lost_since = now
                    if now - lost_since > lane_follow_timeout_s:
                        state = 'STOP'
                        _ctrl.reset()

            elif state == 'STOP':
                pwm_l = pwm_r = 0.0
                if found:
                    state = 'LANE_FOLLOW'
                    lost_since = None
                    _ctrl.reset()

            if PAUSED:
                wheels.set_wheels_speed(0.0, 0.0)
            else:
                wheels.set_wheels_speed(float(pwm_l), float(pwm_r))

            set_leds(leds, state)

            det = {
                'found': found, 'centers': centers, 'method': method,
                'quality': quality, 'span': span,
                'lateral_error': lateral_error, 'state': state,
                'speed_scale': speed_scale,
                'lane_detected': lane_detected,
                'turn_bias': round(_bias, 3),
            }
            with _det_lock:
                DETECTION.update(det)
            STATUS = {
                'state': state, 'found': found, 'span': round(span, 3),
                'lateral_error': round(lateral_error, 3),
                'speed_scale': round(speed_scale, 3),
                'detection_method': method, 'quality': round(quality, 2),
                'lane_detected': lane_detected,
                'turn_bias': round(_bias, 3),
            }

            stop_event.wait(dt)
    finally:
        wheels.set_wheels_speed(0.0, 0.0)
        if leds:
            leds.all_off()
