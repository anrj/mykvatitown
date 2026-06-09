"""
Convoying leader — lane follow + stop at traffic signs.

Only the leader reads signs. The follower mimics the leader via the dot
matrix on the leader's back (see tasks/project/packages/agent.py).
"""

import os
import time

import cv2
import yaml

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.project_leader.packages.sign_detector import SignDetector

DEBUG_FRAME = None
STATUS = {}
CFG = None
_lane = None
_signs = None

CONFIG_FILE = 'leader_config.yaml'

_DEFAULTS = {
    'signs': {
        'enabled': True,
        'min_tag_px': 38,
        'stop_hold_s': 1.0,
        'tag_meanings': {0: 'stop', 1: 'slow'},
    },
    'control': {'loop_hz': 20},
}

_COLORS = {
    'LANE':     [0.0, 1.0, 0.0],
    'STOP':     [1.0, 0.0, 0.0],
    'RECOVERY': [1.0, 0.7, 0.0],
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


def _annotate(bgr, state, sign, tag_px, lane_detected):
    img = bgr.copy()
    txt = f'{state}  lane={lane_detected}'
    if sign:
        txt += f'  sign={sign}  px={tag_px:.0f}'
    cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def main(camera, wheels, leds, stop_event):
    global DEBUG_FRAME, STATUS, CFG, _lane, _signs

    CFG = load_config()
    _lane = LaneServoingAgent()
    _signs = SignDetector(CFG)

    stop_hold_s = float(CFG['signs']['stop_hold_s'])
    dt = 1.0 / float(CFG['control']['loop_hz'])

    stop_until = 0.0
    last_stop_px = 0.0

    try:
        while not stop_event.is_set():
            ok, frame = camera.read()
            if not ok or frame is None:
                stop_event.wait(0.02)
                continue

            now = time.time()
            sign, tag_px = _signs.detect(frame)

            # Debounce: trigger only when the tag grows (arriving, not leaving).
            if sign == 'stop' and tag_px >= last_stop_px and now > stop_until + 0.5:
                stop_until = now + stop_hold_s
            last_stop_px = tag_px if sign == 'stop' else 0.0

            holding = now < stop_until

            if holding:
                wheels.set_wheels_speed(0.0, 0.0)
                state = 'STOP'
                left = right = 0.0
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                left, right = _lane.compute_commands(rgb)
                wheels.set_wheels_speed(left, right)
                lane_ok = bool(_lane.last_debug_info.get('lane_detected'))
                state = 'LANE' if lane_ok else 'RECOVERY'

            set_leds(leds, state)
            lane_detected = bool(_lane.last_debug_info.get('lane_detected'))
            DEBUG_FRAME = _annotate(frame, state, sign, tag_px, lane_detected)
            STATUS = {
                'state': state,
                'sign': sign or 'none',
                'tag_px': round(tag_px, 1),
                'lane_detected': lane_detected,
                'speed_l': round(left, 3),
                'speed_r': round(right, 3),
                'holding_stop': holding,
            }

            stop_event.wait(dt)
    finally:
        wheels.set_wheels_speed(0.0, 0.0)
        if leds:
            leds.all_off()
