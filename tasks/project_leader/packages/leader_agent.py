"""
Convoying leader — visual lane follow + red-line stop + intersection turns.

Route: CRUISE → red line STOP → CROSS → PRE_TURN → turn (left/right) or straight → EXIT_IX.
"""

import os
import time

import cv2
import numpy as np
import yaml

from tasks.project_leader.packages.intersection_turn import IntersectionPlanner
from tasks.project_leader.packages.leader_lane import LeaderLaneAgent
from tasks.project_leader.packages.sign_behavior import SignBehavior

DEBUG_FRAME = None
STATUS = {}
CFG = None
LANE = None
RUNNING = False

MODE_CRUISE = 'CRUISE'
MODE_STOP = 'STOP'
MODE_CROSS = 'CROSS'      # POST_STOP — creep over the red line
MODE_PRE_TURN = 'PRE_TURN'
MODE_TURN = 'TURN'
MODE_EXIT = 'EXIT_IX'

CONFIG_FILE = 'leader_config.yaml'

_LANE_CONFIG = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'leader_lane_config.yaml'
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
    'signs': {
        'red_strip_frac': 0.08,
        'red_pixel_frac': 0.035,
        'red_line_stop': 0.035,
        'red_line_slow': 0.012,
        'red_hsv_low1': [0, 100, 100],
        'red_hsv_high1': [10, 255, 255],
        'red_hsv_low2': [160, 100, 100],
        'red_hsv_high2': [179, 255, 255],
        'lane_margin_px': 12,
        'stop_tag_px': 55,
        'slow_tag_px': 38,
        'min_tag_px': 38,
        'tag_meanings': {0: 'stop', 1: 'slow'},
        'red_confirm_frames': 3,
        'red_clear_frames': 12,
        'red_center_frac': 0.22,
    },
    'intersection': {
        'enabled': True,
        'turns': ['left', 'right', 'straight'],
        'stop_hold_s': 1.0,
        'turn_angle_deg': 90.0,
        'turn_timeout_s': 4.0,
        'deadband_deg': 3.0,
        'tolerance_deg': 5.0,
    },
    'control': {
        'loop_hz': 20,
        'accel_rate': 0.05,
        'startup_grace_s': 2.0,
        'intersection_exit_s': 2.5,
        'intersection_exit_speed': 0.22,
    },
    'maneuver': {
        'post_stop_frames': 25,
        'post_stop_speed': 0.20,
        'preturn_left_frames': 18,
        'preturn_right_frames': 18,
        'preturn_straight_frames': 18,
        'preturn_speed': 0.22,
    },
}

_COLORS = {
    MODE_CRUISE: (0.0, 1.0, 0.0),
    MODE_CROSS: (1.0, 0.85, 0.0),
    MODE_PRE_TURN: (1.0, 0.55, 0.0),
    MODE_EXIT: (1.0, 0.7, 0.0),
    MODE_STOP: (1.0, 0.0, 0.0),
    MODE_TURN: (1.0, 1.0, 1.0),
}


def _config_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, '..', '..', '..', 'config', CONFIG_FILE)


def load_config():
    cfg = {k: dict(v) for k, v in _DEFAULTS.items()}
    path = _config_path()
    try:
        with open(path) as f:
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


def _lane_stop_cfg(cfg):
    ls = dict(cfg.get('lane_stop', {}))
    if 'min_yellow_frames' in ls and 'min_line_frames' not in ls:
        ls['min_line_frames'] = ls.pop('min_yellow_frames')
    if 'yellow_lost_frames' in ls and 'line_lost_frames' not in ls:
        ls['line_lost_frames'] = ls.pop('yellow_lost_frames')
    if 'bottom_yellow_px' in ls and 'bottom_line_px' not in ls:
        ls['bottom_line_px'] = ls.pop('bottom_yellow_px')
    return ls


def _visualize(bgr, mode, lane_info, trigger, follow, line_count, line_lost,
               phase, source, turn_dir):
    img = bgr.copy()
    yellow = lane_info.get('yellow_mask')
    white = lane_info.get('white_mask')
    if yellow is not None:
        overlay = img.copy()
        yidx = yellow > 0
        yval = (yellow[yidx] // 2).astype(np.uint8)
        overlay[yidx, 0] = 0
        overlay[yidx, 1] = yval
        overlay[yidx, 2] = yval
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    if white is not None:
        overlay = img.copy()
        widx = white > 0
        wval = (white[widx] // 2).astype(np.uint8)
        overlay[widx, 0] = wval
        overlay[widx, 1] = wval
        overlay[widx, 2] = wval
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

    txt = (f'{mode}  sign={phase}/{source}  trigger={trigger}  follow={follow}  '
           f'lost={line_lost}/{line_count}  turn={turn_dir}')
    cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def _set_leds(leds, mode):
    if not leds:
        return
    color = _COLORS.get(mode, (0.0, 0.0, 0.0))
    leds.set_rgb(0, color)
    leds.set_rgb(2, color)


def _sync_lane_params(lane, cfg):
    """Apply live-tuned lane PID/speed from CFG (updated by the web UI)."""
    if lane is None or cfg is None:
        return
    lc = cfg.get('lane', {})
    if 'p_gain' in lc:
        lane.p_gain = float(lc['p_gain'])
    if 'd_gain' in lc:
        lane.d_gain = float(lc['d_gain'])
    if 'base_speed' in lc:
        lane.base_speed = float(lc['base_speed'])
    if 'curve_speed' in lc:
        lane.curve_speed = float(lc['curve_speed'])


def main(camera, wheels, leds, stop_event):
    global DEBUG_FRAME, STATUS, CFG, LANE, RUNNING

    CFG = load_config()
    if 'lane' not in CFG:
        CFG['lane'] = {}
    control = CFG['control']
    maneuver = CFG.get('maneuver', {})
    dt = 1.0 / float(control.get('loop_hz', 20))
    startup_grace = float(control.get('startup_grace_s', 2.0))
    accel_rate = float(control.get('accel_rate', 0.05))

    try:
        lane = LeaderLaneAgent(config_path=_LANE_CONFIG)
        LANE = lane
        CFG['lane'].setdefault('p_gain', lane.p_gain)
        CFG['lane'].setdefault('d_gain', lane.d_gain)
        CFG['lane'].setdefault('base_speed', lane.base_speed)
        CFG['lane'].setdefault('curve_speed', lane.curve_speed)
        signs = SignBehavior(CFG)
        intersection = IntersectionPlanner(CFG)
    except Exception as e:
        print(f'[leader] Failed to init: {e}')
        return

    mode = MODE_CRUISE
    speed_mult = 0.0
    red_consumed = False
    red_seen_frames = 0
    red_clear_count = 0
    red_near_frac = 0.0
    red_far_frac = 0.0
    lane_info = {}
    started_at = time.monotonic()
    exit_ix_until = 0.0
    red_confirm = int(CFG.get('signs', {}).get('red_confirm_frames', 4))
    red_clear_need = int(CFG.get('signs', {}).get('red_clear_frames', 20))
    exit_ix_s = float(control.get('intersection_exit_s', 2.5))
    exit_ix_speed = float(control.get('intersection_exit_speed', 0.22))
    post_stop_frames = int(maneuver.get('post_stop_frames', 25))
    post_stop_speed = float(maneuver.get('post_stop_speed', 0.20))
    preturn_left_frames = int(maneuver.get('preturn_left_frames', 18))
    preturn_right_frames = int(maneuver.get('preturn_right_frames', 18))
    preturn_straight_frames = int(maneuver.get('preturn_straight_frames', 18))
    preturn_speed = float(maneuver.get('preturn_speed', 0.22))
    post_stop_count = 0
    preturn_count = 0

    try:
        while not stop_event.is_set():
            _sync_lane_params(lane, CFG)
            red_confirm = int(CFG.get('signs', {}).get('red_confirm_frames', red_confirm))
            red_clear_need = int(CFG.get('signs', {}).get('red_clear_frames', red_clear_need))

            ok, frame = camera.read()
            if not ok or frame is None:
                stop_event.wait(0.02)
                continue

            now = time.monotonic()
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[-1] == 3 else frame
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            turn_dir = (
                intersection.pending_direction()
                if intersection.turn_active() else 'none'
            )
            left, right = 0.0, 0.0
            in_exit = mode == MODE_EXIT or now < exit_ix_until

            # --- driving commands ------------------------------------------------
            if mode == MODE_TURN:
                done, left, right = intersection.pid_step(wheels, dt, now)
                if done:
                    lane.reset_steering_state()
                    mode = MODE_EXIT
                    exit_ix_until = now + exit_ix_s
                    red_consumed = True
                    red_clear_count = 0
            elif mode == MODE_STOP:
                left, right = 0.0, 0.0
                if intersection.stop_complete(now):
                    post_stop_count = 0
                    mode = MODE_CROSS
                    lane.reset_steering_state()
            elif mode == MODE_CROSS:
                left, right = lane.compute_commands(rgb)
                lane_info = dict(lane.last_debug_info)
                post_stop_count += 1
                if post_stop_count >= post_stop_frames:
                    preturn_count = 0
                    mode = MODE_PRE_TURN
                    lane.reset_steering_state()
            elif mode == MODE_PRE_TURN:
                left, right = lane.compute_commands(rgb)
                lane_info = dict(lane.last_debug_info)
                preturn_count += 1
                next_turn = intersection.pending_direction()
                if next_turn == 'left':
                    preturn_need = preturn_left_frames
                elif next_turn == 'right':
                    preturn_need = preturn_right_frames
                else:
                    preturn_need = preturn_straight_frames
                if preturn_count >= preturn_need:
                    if not intersection.has_pending_turn():
                        mode = MODE_CRUISE
                    elif intersection.is_spin_turn():
                        intersection.begin_turn(now)
                        mode = MODE_TURN
                    else:
                        intersection.advance_maneuver()
                        lane.reset_steering_state()
                        mode = MODE_EXIT
                        exit_ix_until = now + exit_ix_s
                        red_consumed = True
                        red_clear_count = 0
            else:
                left, right = lane.compute_commands(rgb)
                lane_info = dict(lane.last_debug_info)

            if mode not in (MODE_CRUISE, MODE_EXIT, MODE_CROSS, MODE_PRE_TURN) and not lane_info:
                lane.compute_commands(rgb)
                lane_info = dict(lane.last_debug_info)
            elif mode in (MODE_CRUISE, MODE_EXIT, MODE_CROSS, MODE_PRE_TURN):
                lane_info = dict(lane.last_debug_info)

            # --- red line (only in CRUISE, not inside intersection) ------------
            phase, source, strength = signs.assess(bgr, lane_info)
            red_hit, red_near_frac, red_far_frac = signs.red_line_detected(
                rgb, lane_info, at_stop_line=True,
            )
            past_grace = (now - started_at) >= startup_grace
            allow_red = (
                past_grace and RUNNING and mode == MODE_CRUISE
                and not in_exit and not red_consumed
            )

            if allow_red and red_hit:
                red_seen_frames += 1
                red_clear_count = 0
            else:
                red_seen_frames = 0
                if red_consumed and not red_hit:
                    red_clear_count += 1

            if red_consumed and red_clear_count >= red_clear_need:
                red_consumed = False
                red_clear_count = 0

            if allow_red and red_seen_frames >= red_confirm:
                intersection.arm_stop(now)
                mode = MODE_STOP
                red_consumed = True
                red_seen_frames = 0
                lane.reset_steering_state()
            elif (allow_red and phase == 'stop' and source == 'apriltag'):
                intersection.arm_stop(now)
                mode = MODE_STOP
                lane.reset_steering_state()

            if mode == MODE_EXIT and now >= exit_ix_until:
                mode = MODE_CRUISE

            # --- speed scaling -------------------------------------------------
            if mode == MODE_STOP:
                target_mult = 0.0
            elif mode == MODE_CROSS:
                target_mult = post_stop_speed
            elif mode == MODE_PRE_TURN:
                target_mult = preturn_speed
            elif mode == MODE_EXIT:
                target_mult = exit_ix_speed
            elif mode == MODE_TURN:
                target_mult = 1.0
            else:
                target_mult = 0.35 if phase == 'slow' else 1.0

            # Slow down when drifting off-centre so we don't stop nosed into the corner.
            lat_err = abs(float(lane_info.get('lateral_error', 0.0)))
            if mode == MODE_CRUISE and lat_err > 0.22:
                target_mult = min(target_mult, 0.55)
            elif mode in (MODE_CROSS, MODE_PRE_TURN) and lat_err > 0.18:
                creep_cap = post_stop_speed if mode == MODE_CROSS else preturn_speed
                target_mult = min(target_mult, creep_cap * 0.85)

            if target_mult > speed_mult:
                speed_mult = min(target_mult, speed_mult + accel_rate)
            else:
                speed_mult = max(target_mult, speed_mult - accel_rate)

            if not RUNNING:
                speed_mult = 0.0

            if RUNNING:
                wheels.set_wheels_speed(left * speed_mult, right * speed_mult)
            else:
                wheels.set_wheels_speed(0.0, 0.0)
            _set_leds(leds, mode)

            DEBUG_FRAME = _visualize(
                bgr, mode, lane_info,
                'red_line', 'centreline', 0, 0,
                phase, source, turn_dir,
            )
            next_turn = intersection.pending_direction()
            if next_turn == 'left':
                preturn_need = preturn_left_frames
            elif next_turn == 'right':
                preturn_need = preturn_right_frames
            else:
                preturn_need = preturn_straight_frames
            STATUS = {
                'running': RUNNING,
                'mode': mode,
                'phase': phase,
                'source': source,
                'red_near': round(red_near_frac, 3),
                'red_far': round(red_far_frac, 3),
                'red_consumed': red_consumed,
                'exit_ix': in_exit,
                'post_stop_count': post_stop_count,
                'post_stop_need': post_stop_frames,
                'preturn_count': preturn_count,
                'preturn_need': preturn_need,
                'lane_detected': bool(lane_info.get('lane_detected')),
                'lateral_error': round(float(lane_info.get('lateral_error', 0.0)), 3),
                'raw_lateral_error': round(float(lane_info.get('raw_lateral_error', 0.0)), 3),
                'lane_half_width_px': lane_info.get('lane_half_width_px'),
                'single_line_mode': bool(lane_info.get('single_line_mode')),
                'left_speed': round(left * speed_mult, 3),
                'right_speed': round(right * speed_mult, 3),
                'turn': turn_dir,
                'turn_angle_deg': intersection.turn_angle_deg,
                'intersection_idx': intersection.turn_idx,
            }

            stop_event.wait(dt)
    finally:
        wheels.set_wheels_speed(0.0, 0.0)
        if leds:
            leds.all_off()
