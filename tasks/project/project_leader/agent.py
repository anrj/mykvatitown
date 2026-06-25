import os
import time

import yaml

from tasks.project.packages.control import apply_leds, motors_from_decision
from tasks.project.project_leader.fsm import (
    STATE_CROSS, STATE_TURN_L, STATE_TURN_R, LeadFSM,
)
from tasks.project.project_leader.perception import LeadPerception


def _smooth_wheel_speed(target_left, target_right, prev_left, prev_right,
                        max_delta=0.08):
    def step(target, prev):
        delta = target - prev
        if delta > max_delta:
            return prev + max_delta
        if delta < -max_delta:
            return prev - max_delta
        return target
    return step(target_left, prev_left), step(target_right, prev_right)

_CONFIG_FILE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "project_lead_config.yaml")
)

_MANEUVER_STATES = {STATE_TURN_L, STATE_TURN_R, STATE_CROSS}


def _load_cfg() -> dict:
    try:
        with open(_CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def main(camera, wheels, leds, stop_event,
         frame_queue=None, debug_lock=None, debug_dict=None, encoders=None):
    cfg = _load_cfg()
    perception = LeadPerception(cfg)
    fsm = LeadFSM(cfg)
    baseline = float(cfg.get("wheel_baseline_m", 0.1))

    last_state = None
    last_hw_warn = 0.0
    last_dbg = 0.0
    last_fps_update = 0.0
    frame_count = 0
    fps = 0.0
    in_maneuver = False
    prev_left = 0.0
    prev_right = 0.0
    cam_fail = 0

    try:
        while not stop_event.is_set():
            ok, frame = camera.read()
            if not ok:
                cam_fail += 1
                if cam_fail == 5:
                    print("[lead] camera returned no frames — check nvargus-daemon and "
                          "that no other process is holding the camera")
                # Auto-recover from a transient nvargus/caps hiccup that would
                # otherwise leave the video stuck on "Waiting for frames" forever.
                if cam_fail % 150 == 0:
                    print(f"[lead] re-initializing camera after {cam_fail} empty reads...")
                    try:
                        camera.stop()
                        camera.start()
                    except Exception as e:
                        print(f"[lead] camera re-init failed: {e}")
                time.sleep(0.02)
                continue
            if cam_fail:
                print(f"[lead] camera recovered after {cam_fail} empty reads")
                cam_fail = 0

            if frame_queue is not None:
                try:
                    frame_queue.put_nowait(frame.copy())
                except Exception:
                    pass

            now = time.monotonic()
            wm = perception.update(frame, now)

            # Encoder odometry closes the loop on maneuvers when available: yaw
            # for turns, forward distance for straight crosses. None on either
            # falls back to lane-reacquisition + timeout inside the FSM.
            turn_yaw = None
            fwd_dist = None
            if encoders is not None and in_maneuver:
                try:
                    dl = encoders.left.distance_m()
                    dr = encoders.right.distance_m()
                    turn_yaw = (dr - dl) / max(baseline, 1e-3)
                    fwd_dist = 0.5 * (dl + dr)
                except Exception:
                    turn_yaw = None
                    fwd_dist = None

            decision = fsm.step(wm, turn_yaw_rad=turn_yaw, fwd_dist_m=fwd_dist)
            if fsm.request_lane_reset:
                perception.reset_lane()

            left, right = motors_from_decision(decision)
            left, right = _smooth_wheel_speed(left, right, prev_left, prev_right)
            prev_left, prev_right = left, right

            # Reset encoders at maneuver entry so yaw integrates from zero.
            is_man = decision.state_name in _MANEUVER_STATES
            if is_man and not in_maneuver and encoders is not None:
                try:
                    encoders.reset()
                    encoders.set_directions(True, True)
                except Exception:
                    pass
            in_maneuver = is_man

            frame_count += 1
            if now - last_fps_update > 1.0:
                fps = frame_count / (now - last_fps_update)
                frame_count = 0
                last_fps_update = now

            if debug_lock is not None and debug_dict is not None:
                dbg = perception.last_debug_info
                with debug_lock:
                    debug_dict.update({
                        'state': decision.state_name,
                        'base_speed': decision.base_speed,
                        'steering': decision.steering,
                        'left_speed': left,
                        'right_speed': right,
                        'apriltag_ids': dbg.get('apriltag_ids', []),
                        'red_line': dbg.get('red_line'),
                        'route_idx': fsm.route_idx,
                        'fps': fps,
                    })

            if now - last_dbg > 0.5:
                dbg = perception.last_debug_info
                print(f"[lead] {decision.state_name} "
                      f"route={fsm.route_idx}/{len(fsm.route)} "
                      f"tags={dbg.get('apriltag_ids')} redline={dbg.get('red_line')} "
                      f"base={decision.base_speed:.2f} steer={decision.steering:+.2f} "
                      f"L={left:.2f} R={right:.2f}")
                last_dbg = now

            try:
                wheels.set_wheels_speed(left, right)
            except OSError as e:
                if now - last_hw_warn > 2.0:
                    print(f"[lead] wheels I/O error: {e} (check battery / HAT)")
                    last_hw_warn = now
            try:
                apply_leds(leds, decision)
            except OSError as e:
                if now - last_hw_warn > 2.0:
                    print(f"[lead] leds I/O error: {e}")
                    last_hw_warn = now

            if decision.state_name != last_state:
                print(f"[lead] state={decision.state_name} "
                      f"speed={decision.base_speed:.2f} steer={decision.steering:+.2f}")
                last_state = decision.state_name
    finally:
        try:
            wheels.set_wheels_speed(0.0, 0.0)
        except Exception:
            pass
        if leds is not None:
            try:
                leds.all_off()
            except Exception:
                pass
