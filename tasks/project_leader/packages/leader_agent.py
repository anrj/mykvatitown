"""
Convoying leader — lane follow + stop when a configured lane edge ends.

Uses left/right lane edges (same as visual_lane_servoing), not hardcoded colours.
Typical Duckietown setup: stop when the left (dashed) edge ends, then follow the
right (solid) edge through the stop line.
"""

import os
import time

import cv2
import yaml

from tasks.project_leader.packages.leader_lane import LeaderLaneAgent, _norm_side

DEBUG_FRAME = None
STATUS = {}
CFG = None
_lane = None

MODE_CRUISE = 'CRUISE'
MODE_AT_LINE = 'AT_LINE'
MODE_STOP = 'STOP'

CONFIG_FILE = 'leader_config.yaml'
_LANE_CONFIG = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'leader_lane_config.yaml',
))

_DEFAULTS = {
    'lane_stop': {
        'stop_trigger_line': 'left',
        'at_line_follow_line': 'right',
        'min_line_frames': 10,
        'line_lost_frames': 4,
        'bottom_line_px': 25,
        'stop_hold_s': 1.0,
        'line_speed_mult': 0.35,
    },
    'control': {
        'loop_hz': 20,
        'accel_rate': 0.05,
    },
}

_COLORS = {
    MODE_CRUISE:   [0.0, 1.0, 0.0],
    MODE_AT_LINE:  [1.0, 0.7, 0.0],
    MODE_STOP:     [1.0, 0.0, 0.0],
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
        _migrate_lane_stop_keys(cfg.get('lane_stop', {}))
        print(f'[leader] Loaded config from {CONFIG_FILE}')
    except FileNotFoundError:
        print(f'[leader] {CONFIG_FILE} not found, using defaults')
    return cfg


def _migrate_lane_stop_keys(ls: dict) -> None:
    """Accept legacy yellow_* config keys."""
    if 'min_yellow_frames' in ls and 'min_line_frames' not in ls:
        ls['min_line_frames'] = ls.pop('min_yellow_frames')
    if 'yellow_lost_frames' in ls and 'line_lost_frames' not in ls:
        ls['line_lost_frames'] = ls.pop('yellow_lost_frames')
    if 'bottom_yellow_px' in ls and 'bottom_line_px' not in ls:
        ls['bottom_line_px'] = ls.pop('bottom_yellow_px')


def _lane_stop_cfg(ls: dict) -> dict:
    _migrate_lane_stop_keys(ls)
    return {
        'trigger': _norm_side(ls.get('stop_trigger_line', 'left')),
        'follow': _norm_side(ls.get('at_line_follow_line', 'right')),
        'min_frames': int(ls.get('min_line_frames', 10)),
        'lost_frames': int(ls.get('line_lost_frames', 4)),
        'bottom_px': int(ls.get('bottom_line_px', 25)),
        'stop_hold_s': float(ls.get('stop_hold_s', 1.0)),
        'line_speed': float(ls.get('line_speed_mult', 0.35)),
    }


def _visualize(bgr, lane_info, mode, trigger_visible, line_count, line_lost, trigger_side):
    h, w = bgr.shape[:2]
    img = bgr.copy()

    ym = lane_info.get('yellow_mask')
    wm = lane_info.get('white_mask')
    if ym is not None and wm is not None and ym.shape[:2] == (h, w):
        overlay = img.copy()
        overlay[ym > 0] = (0, 200, 255)
        overlay[wm > 0] = (255, 255, 255)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)

    slice_ys = lane_info.get('slice_ys') or []
    if slice_ys:
        y = int(slice_ys[-1])
        color = (0, 220, 255) if trigger_visible else (0, 0, 255)
        cv2.line(img, (0, y), (w, y), color, 2)

    follow = lane_info.get('follow_line', '')
    txt = (f'{mode}  trigger={trigger_side} vis={trigger_visible}'
           f'  follow={follow or "-"}  seen={line_count}  lost={line_lost}'
           f'  lane={lane_info.get("lane_detected", False)}')
    cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 4)
    cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    return img


def main(camera, wheels, leds, stop_event):
    global DEBUG_FRAME, STATUS, CFG, _lane

    CFG = load_config()
    _lane = LeaderLaneAgent(config_path=_LANE_CONFIG)

    dt = 1.0 / float(CFG['control']['loop_hz'])

    mode = MODE_CRUISE
    speed_mult = 1.0
    line_count = 0
    line_lost = 0
    stop_until = 0.0
    stop_armed = True

    try:
        while not stop_event.is_set():
            ok, frame = camera.read()
            if not ok or frame is None:
                stop_event.wait(0.02)
                continue

            now = time.time()
            lc = _lane_stop_cfg(CFG['lane_stop'])
            accel = float(CFG['control']['accel_rate'])

            if mode == MODE_STOP and now >= stop_until:
                mode = MODE_AT_LINE

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prev_info = _lane.last_debug_info or {}
            prev_trigger = _lane.bottom_line_visible(
                prev_info, lc['trigger'], lc['bottom_px'])

            use_single = (
                mode == MODE_AT_LINE
                or (mode == MODE_CRUISE
                    and line_count >= lc['min_frames']
                    and not prev_trigger)
            )

            if mode == MODE_STOP:
                lane_info = prev_info
                trigger_visible = _lane.bottom_line_visible(
                    lane_info, lc['trigger'], lc['bottom_px'])
                pwm_l = pwm_r = 0.0
            elif use_single:
                pwm_l, pwm_r = _lane.compute_single_line_commands(rgb, lc['follow'])
                lane_info = _lane.last_debug_info
                trigger_visible = _lane.bottom_line_visible(
                    lane_info, lc['trigger'], lc['bottom_px'])
            else:
                pwm_l, pwm_r = _lane.compute_commands(rgb)
                lane_info = _lane.last_debug_info
                trigger_visible = _lane.bottom_line_visible(
                    lane_info, lc['trigger'], lc['bottom_px'])

            if trigger_visible:
                line_count = min(line_count + 1, lc['min_frames'] * 3)
                line_lost = 0
                if mode == MODE_AT_LINE:
                    mode = MODE_CRUISE
                    stop_armed = True
                    _lane.reset_steering_state()
            else:
                line_lost += 1

            if (mode in (MODE_CRUISE, MODE_AT_LINE) and stop_armed
                    and line_count >= lc['min_frames']
                    and line_lost >= lc['lost_frames']):
                mode = MODE_STOP
                stop_until = now + lc['stop_hold_s']
                stop_armed = False
                _lane.reset_steering_state()
                pwm_l = pwm_r = 0.0
                speed_mult = 0.0
            elif mode == MODE_CRUISE and line_count >= lc['min_frames'] and line_lost >= 1:
                mode = MODE_AT_LINE

            if mode == MODE_STOP:
                speed_mult = 0.0
            elif mode == MODE_AT_LINE:
                speed_mult = lc['line_speed']
            else:
                speed_mult = min(1.0, speed_mult + accel)

            left = pwm_l * speed_mult
            right = pwm_r * speed_mult
            wheels.set_wheels_speed(left, right)

            if leds:
                color = _COLORS.get(mode, [0.0, 1.0, 0.0])
                leds.set_rgb(0, color)
                leds.set_rgb(2, color)

            DEBUG_FRAME = _visualize(
                frame, lane_info, mode, trigger_visible, line_count, line_lost,
                lc['trigger'],
            )
            STATUS = {
                'mode': mode,
                'stop_trigger_line': lc['trigger'],
                'at_line_follow_line': lc['follow'],
                'trigger_visible': trigger_visible,
                'line_count': line_count,
                'line_lost': line_lost,
                'stop_armed': stop_armed,
                'speed_mult': round(speed_mult, 3),
                'lane_detected': bool(lane_info.get('lane_detected')),
                'lateral_error': round(float(lane_info.get('lateral_error', 0)), 3),
                'speed_l': round(left, 3),
                'speed_r': round(right, 3),
            }

            stop_event.wait(dt)
    finally:
        wheels.set_wheels_speed(0.0, 0.0)
        if leds:
            leds.all_off()
