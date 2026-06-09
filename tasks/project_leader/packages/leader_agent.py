"""
Convoying leader — lane follow + slow / stop / go at traffic signs.

Only the leader reads signs. The follower mimics via the dot matrix.
"""

import os
import time

import cv2
import yaml

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.project_leader.packages.sign_behavior import SignBehavior

DEBUG_FRAME = None
STATUS = {}
CFG = None
_lane = None
_signs = None

CONFIG_FILE = 'leader_config.yaml'
_LANE_CONFIG = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'leader_lane_config.yaml',
))

_DEFAULTS = {
    'signs': {
        'enabled': True,
        'min_tag_px': 42,
        'slow_tag_px': 22,
        'stop_hold_s': 1.0,
        'red_slow_area': 0.0025,
        'red_stop_area': 0.009,
        'red_slow_bottom': 0.58,
        'red_stop_bottom': 0.72,
        'use_detection': True,
        'tag_meanings': {0: 'stop', 1: 'slow'},
    },
    'control': {
        'loop_hz': 20,
        'cruise_speed_mult': 1.0,
        'slow_speed_mult': 0.35,
        'depart_speed_mult': 0.45,
        'depart_slow_s': 2.5,
        'single_lane_cap': 0.55,
        'recovery_cap': 0.20,
        'accel_rate': 0.05,
        'decel_rate': 0.07,
        'detection_slow_threshold': 0.06,
        'detection_stop_threshold': 0.15,
    },
}

_COLORS = {
    'CRUISE':   [0.0, 1.0, 0.0],
    'DEPART':   [0.0, 0.8, 0.5],
    'SLOW':     [1.0, 0.7, 0.0],
    'STOP':     [1.0, 0.0, 0.0],
    'RECOVERY': [0.0, 0.4, 1.0],
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


def set_leds(leds, state):
    if not leds:
        return
    color = _COLORS.get(state, [0.0, 0.0, 0.0])
    for idx in (0, 2, 3, 4):
        leds.set_rgb(idx, color)


def _scale_preserving_steer(left: float, right: float, mult: float):
    """Scale forward speed but keep full steering — stays in lane at low speed."""
    if mult <= 0.0:
        return 0.0, 0.0
    forward = (left + right) * 0.5
    turn = (right - left) * 0.5
    forward *= mult
    left = forward - turn
    right = forward + turn
    return float(max(-1.0, min(1.0, left))), float(max(-1.0, min(1.0, right)))


def _both_lanes_visible(lane_info: dict) -> bool:
    return bool(lane_info.get('yellow_xs')) and bool(lane_info.get('white_xs'))


def _annotate(bgr, state, phase, source, strength, lane_detected, speed_mult):
    img = bgr.copy()
    txt = (f'{state}  phase={phase}  src={source}  str={strength:.3f}'
           f'  spd={speed_mult:.2f}  lane={lane_detected}')
    cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
    cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def main(camera, wheels, leds, stop_event):
    global DEBUG_FRAME, STATUS, CFG, _lane, _signs

    CFG = load_config()
    _lane = LaneServoingAgent(config_path=_LANE_CONFIG)
    _signs = SignBehavior(CFG)

    if _signs._det_error:
        print(f'[leader] Object detection: {_signs._det_error}')
    elif _signs.detector_ready:
        print('[leader] Object detection model loaded')

    stop_hold_s = float(CFG['signs']['stop_hold_s'])
    dt = 1.0 / float(CFG['control']['loop_hz'])
    cruise_mult = float(CFG['control']['cruise_speed_mult'])
    slow_mult = float(CFG['control']['slow_speed_mult'])
    accel = float(CFG['control']['accel_rate'])
    decel = float(CFG['control']['decel_rate'])
    depart_mult = float(CFG['control']['depart_speed_mult'])
    depart_slow_s = float(CFG['control']['depart_slow_s'])
    single_lane_cap = float(CFG['control']['single_lane_cap'])
    recovery_cap = float(CFG['control']['recovery_cap'])

    speed_mult = cruise_mult
    stop_until = 0.0
    depart_until = 0.0
    stop_armed = True          # one stop per sign; re-arm after sign clears
    last_stop_strength = 0.0   # debounce: only stop when signal grows (approaching)

    try:
        while not stop_event.is_set():
            ok, frame = camera.read()
            if not ok or frame is None:
                stop_event.wait(0.02)
                continue

            now = time.time()
            lane_info = getattr(_lane, 'last_debug_info', {})
            phase, source, strength = _signs.assess(frame, lane_info)

            # One-shot stop: latch once per approach, never extend while sitting at sign.
            approaching = strength >= last_stop_strength
            if (phase == 'stop' and stop_armed and not (now < stop_until)
                    and approaching):
                stop_until = now + stop_hold_s
                depart_until = stop_until + depart_slow_s
                stop_armed = False
            last_stop_strength = strength if phase in ('slow', 'stop') else 0.0

            # Re-arm after we've driven past (sign no longer triggers slow/stop).
            if _signs.sign_cleared(phase, strength) and now >= stop_until:
                stop_armed = True

            holding = now < stop_until
            departing = (not holding) and (now < depart_until)
            lane_ok = bool(lane_info.get('lane_detected'))
            both_lanes = _both_lanes_visible(lane_info)

            if holding:
                target_mult = 0.0
                drive_state = 'STOP'
            elif departing:
                # Creep away from sign — don't jump to full speed while re-acquiring lane.
                target_mult = depart_mult
                drive_state = 'DEPART'
            elif phase == 'slow' and stop_armed:
                target_mult = slow_mult
                drive_state = 'SLOW'
            else:
                target_mult = cruise_mult
                drive_state = 'CRUISE'

            # Lane safety caps — never blast full speed without both lines visible.
            if not lane_ok:
                target_mult = min(target_mult, recovery_cap)
                drive_state = 'RECOVERY'
            elif not both_lanes:
                target_mult = min(target_mult, single_lane_cap)

            # Smooth ramp — decelerate into slow/stop, accelerate away after hold.
            if target_mult > speed_mult:
                speed_mult = min(target_mult, speed_mult + accel)
            else:
                speed_mult = max(target_mult, speed_mult - decel)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            left, right = _lane.compute_commands(rgb)

            if holding:
                left = right = 0.0
            else:
                left, right = _scale_preserving_steer(left, right, speed_mult)

            wheels.set_wheels_speed(left, right)
            state = drive_state

            set_leds(leds, state)
            lane_detected = bool(_lane.last_debug_info.get('lane_detected'))
            DEBUG_FRAME = _annotate(frame, state, phase, source, strength,
                                    lane_detected, speed_mult)
            STATUS = {
                'state': state,
                'phase': phase,
                'source': source,
                'strength': round(strength, 4),
                'speed_mult': round(speed_mult, 3),
                'lane_detected': lane_detected,
                'speed_l': round(left, 3),
                'speed_r': round(right, 3),
                'holding_stop': holding,
                'stop_armed': stop_armed,
                'detector_ready': _signs.detector_ready,
            }

            stop_event.wait(dt)
    finally:
        wheels.set_wheels_speed(0.0, 0.0)
        if leds:
            leds.all_off()
