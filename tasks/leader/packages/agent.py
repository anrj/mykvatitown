import os
import time
import threading

import cv2
import numpy as np
import yaml

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.leader.packages.red_line import detect_red_line

DETECTION = {}
_det_lock = threading.Lock()
STATUS = {}

CFG = None
_lane = None

PAUSED = True
CONFIG_FILE = 'leader_config.yaml'

_DEFAULTS = {
    'control': {
        'loop_hz': 24,
    },
    'red_line': {
        'red_strip_frac': 0.32,
        'red_approach_frac': 0.50,
        'red_roi_left': 0.12,
        'red_roi_right': 0.90,
        'red_pixel_frac': 0.012,
        'red_approach_pixel_frac': 0.003,
        'red_min_area': 60,
        'red_min_width_frac': 0.12,
        'red_line_close_y2_ratio': 0.3,
    },
    'turn': {
        'stop_hold_s': 2.08,
        'preturn_right_s': 0.83,
        'preturn_left_s': 0.33,
        'preturn_speed': 0.20,
        'turn_right_s': 1.04,
        'turn_left_s': 2.33,
        'turn_right_pwm': [0.60, 0.05],
        'turn_left_pwm': [0.20, 0.50],
        'exit_s': 0.125,
        'exit_speed': 0.40,
        'red_ignore_s': 4.17,
        'approach_slow_factor': 0.80,
        'turn_sequence': ['R', 'L', 'S', 'S'],
        'turn_cooldown_s': 3.0,
    },
    'led': {
        'blink_hz': 5.0,
    },
}

_COLORS = {
    'CRUISE':   [0.0, 1.0, 0.0],
    'APPROACH': [1.0, 0.7, 0.0],
    'STOP':     [1.0, 0.0, 0.0],
    'PRE_TURN': [0.3, 0.3, 1.0],
    'TURN':     [0.3, 0.3, 1.0],
    'EXIT':     [0.0, 1.0, 0.0],
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
        print(f'[leader] Loaded config from {CONFIG_FILE}')
    except FileNotFoundError:
        print(f'[leader] {CONFIG_FILE} not found, using defaults')
    return cfg


def _set_leds(leds, state, turn_dir='none', blink_on=False):
    if not leds:
        return
    if state in ('STOP', 'PRE_TURN', 'TURN') and turn_dir in ('L', 'R') and blink_on:
        amber = [1.0, 0.6, 0.0]
        off = [0.0, 0.0, 0.0]
        if turn_dir == 'R':
            leds.set_rgb(4, amber); leds.set_rgb(3, off)
        else:
            leds.set_rgb(3, amber); leds.set_rgb(4, off)
        leds.set_rgb(0, off); leds.set_rgb(2, off)
    else:
        base = _COLORS.get(state, [0.0, 0.0, 0.0])
        for i in (0, 2, 3, 4):
            leds.set_rgb(i, base)


def _annotate(bgr, det):
    if not det:
        return bgr
    img = bgr.copy()
    h, w = img.shape[:2]

    state = det.get('state', '')
    red_ratio = det.get('red_ratio', 0.0)
    approach = det.get('red_approach', False)
    turn_dir = det.get('turn_dir', 'none')
    turn_count = det.get('turn_count', 0)
    lane_detected = det.get('lane_detected', False)
    lateral_error = det.get('lateral_error', 0.0)
    speed_l = det.get('speed_l', 0.0)
    speed_r = det.get('speed_r', 0.0)

    label = state
    if turn_dir in ('L', 'R', 'S'):
        label += f'  TURN={turn_dir}  #{turn_count}'
    if approach:
        label += '  (approach!)'
    cv2.putText(img, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    txt = (f'red={red_ratio:.3f}  lane={lane_detected}  '
           f'e={lateral_error:+.2f}  L={speed_l:.2f} R={speed_r:.2f}')
    cv2.putText(img, txt, (10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
    cv2.putText(img, txt, (10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def main(camera, wheels, leds, stop_event):
    global STATUS, CFG, _lane, DETECTION

    CFG = load_config()
    _lane = LaneServoingAgent()

    dt = 1.0 / float(CFG['control']['loop_hz'])
    red_cfg = CFG['red_line']
    turn_cfg = CFG['turn']
    blink_hz = float(CFG['led']['blink_hz'])
    blink_period = 1.0 / blink_hz if blink_hz > 0 else 0.2

    turn_sequence = list(turn_cfg.get('turn_sequence', ['R', 'L', 'S', 'S']))
    turn_cooldown_s = float(turn_cfg.get('turn_cooldown_s', 3.0))

    state = 'CRUISE'
    state_start = time.monotonic()
    turn_count = 0
    turn_dir = 'none'
    slow_factor = 1.0
    red_ignore_until = 0.0

    try:
        while not stop_event.is_set():
            ok, frame = camera.read()
            if not ok or frame is None:
                stop_event.wait(0.02)
                continue

            now = time.monotonic()
            elapsed = now - state_start

            # --- lane following (always runs for steering + debug) ---
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lane_l, lane_r = _lane.compute_commands(rgb)
            lane_info = _lane.last_debug_info

            red_detected, red_ratio, red_approach = False, 0.0, False
            if now > red_ignore_until:
                red_detected, red_ratio, red_approach = detect_red_line(frame, red_cfg)

            pwm_l, pwm_r = lane_l, lane_r

            if state == 'CRUISE':
                if red_approach:
                    state = 'APPROACH'
                    state_start = now
                    slow_factor = 1.0

            elif state == 'APPROACH':
                slow_factor *= turn_cfg['approach_slow_factor']
                pwm_l = lane_l * slow_factor
                pwm_r = lane_r * slow_factor
                if red_detected or slow_factor < 0.06:
                    state = 'STOP'
                    state_start = now
                    turn_count += 1
                    if turn_count <= len(turn_sequence):
                        turn_dir = turn_sequence[turn_count - 1]
                    else:
                        turn_dir = 'S'

            elif state == 'STOP':
                pwm_l = pwm_r = 0.0
                if elapsed >= turn_cfg['stop_hold_s']:
                    if turn_dir == 'S':
                        state = 'CRUISE'
                        state_start = now
                        red_ignore_until = now + max(
                            turn_cfg['red_ignore_s'], turn_cooldown_s)
                        turn_dir = 'none'
                    else:
                        state = 'PRE_TURN'
                        state_start = now

            elif state == 'PRE_TURN':
                ps = turn_cfg['preturn_speed']
                pwm_l = pwm_r = ps
                preturn_s = (turn_cfg['preturn_right_s'] if turn_dir == 'R'
                             else turn_cfg['preturn_left_s'])
                if elapsed >= preturn_s:
                    state = 'TURN'
                    state_start = now

            elif state == 'TURN':
                if turn_dir == 'R':
                    pwm_l, pwm_r = turn_cfg['turn_right_pwm']
                    turn_s = turn_cfg['turn_right_s']
                else:
                    pwm_l, pwm_r = turn_cfg['turn_left_pwm']
                    turn_s = turn_cfg['turn_left_s']
                if elapsed >= turn_s:
                    state = 'EXIT'
                    state_start = now

            elif state == 'EXIT':
                es = turn_cfg['exit_speed']
                pwm_l = pwm_r = es
                if elapsed >= turn_cfg['exit_s']:
                    state = 'CRUISE'
                    state_start = now
                    red_ignore_until = now + max(
                        turn_cfg['red_ignore_s'], turn_cooldown_s)
                    turn_dir = 'none'
                    
            blink_on = (state in ('STOP', 'PRE_TURN', 'TURN')
                        and (now % blink_period) < (blink_period * 0.5))

            if PAUSED:
                wheels.set_wheels_speed(0.0, 0.0)
            else:
                wheels.set_wheels_speed(float(pwm_l), float(pwm_r))

            _set_leds(leds, state, turn_dir, blink_on)

            det = {
                'state': state,
                'red_ratio': red_ratio,
                'red_approach': red_approach,
                'turn_dir': turn_dir,
                'turn_count': turn_count,
                'lane_detected': lane_info.get('lane_detected', False),
                'lateral_error': lane_info.get('lateral_error', 0.0),
                'speed_l': pwm_l,
                'speed_r': pwm_r,
            }
            with _det_lock:
                DETECTION.update(det)
            STATUS = {
                'state': state,
                'turn_dir': turn_dir,
                'turn_count': turn_count,
                'red_ratio': round(red_ratio, 4),
                'red_approach': red_approach,
                'lane_detected': bool(lane_info.get('lane_detected')),
                'lateral_error': round(float(lane_info.get('lateral_error', 0)), 3),
                'speed_l': round(float(pwm_l), 3),
                'speed_r': round(float(pwm_r), 3),
            }

            stop_event.wait(dt)
    finally:
        wheels.set_wheels_speed(0.0, 0.0)
        if leds:
            leds.all_off()
