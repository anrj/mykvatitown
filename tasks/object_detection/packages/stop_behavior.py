import time
from typing import List, Tuple

Detection = Tuple[Tuple[int, int, int, int], float, int]
class_names = {0: 'duckie', 1: 'truck', 2: 'sign'}

# --- STATE MACHINE CONSTANTS ---
LANE_FOLLOWING = "lane_following"
OBSTACLE_PRESENT = "obstacle_present"
INITIATE_PASS = "initiate_pass"
ALIGN_LEFT = "align_left"
PASSING_CLEARANCE = "passing_clearance"
RETURN_RIGHT = "return_right"
ALIGN_RIGHT = "align_right"

# --- GLOBAL TRACKING VARIABLES ---
_state = LANE_FOLLOWING
_seen_frames = 0
_clear_frames = 0
_cooldown_until = 0.0
_state_start_time = 0.0


def should_stop(detections: List[Detection], img_size: int, current_lane_omega: float = 0.0) -> Tuple[
    bool, str, float, float]:
    global _state, _seen_frames, _clear_frames, _state_start_time, _cooldown_until

    current_time = time.time()
    obstacle_found = False
    is_close_enough_to_pass = False
    reason = ""

    # Check if the lane agent is heavily cornering
    is_turning_heavily = abs(current_lane_omega) > 0.15

    if current_time < _cooldown_until:
        return False, "Passing Cooldown Active", -1.0, -1.0

    # -----------------------------------------------------------
    # RUNTIME MANEUVER OVERRIDES (GLOBAL SPEEDS SCALED DOWN ~30%)
    # -----------------------------------------------------------
    if _state == INITIATE_PASS:
        print("STATE = INITIATE_PASS")
        elapsed = current_time - _state_start_time
        if elapsed < 1.20:  # Slightly extended time to compensate for lower speed
            return True, "Overtake Step 1: Swerving Left", 0.10, 0.40
        else:
            _state = ALIGN_LEFT
            _state_start_time = current_time

    if _state == ALIGN_LEFT:
        print("STATE = ALIGN_LEFT")
        elapsed = current_time - _state_start_time
        if elapsed < 0.55:
            return True, "Overtake Step 2: Straightening Left Lane", 0.10, -0.18
        else:
            _state = PASSING_CLEARANCE
            _state_start_time = current_time

    if _state == PASSING_CLEARANCE:
        print("STATE = PASSING_CLEARANCE")
        obstacle_visible = False

        for bbox, score, cls_id in detections:
            if cls_id in [0, 1]:
                obstacle_visible = True
                break

        target_v = 0.06 if is_turning_heavily else 0.09

        # Follow road curvature while overtaking
        adaptive_omega = current_lane_omega

        if obstacle_visible:
            _clear_frames = 0

            return True, \
                "Passing obstacle", \
                target_v, \
                adaptive_omega

        _clear_frames += 1

        if _clear_frames < 5:
            return True, \
                "Verifying clearance", \
                target_v, \
                adaptive_omega

        _state = RETURN_RIGHT
        _state_start_time = current_time

    if _state == RETURN_RIGHT:
        print("STATE = RETURN_RIGHT")
        elapsed = current_time - _state_start_time
        if elapsed < 0.70:
            return True, "Overtake Step 4: Swerving Right", 0.09, -0.40
        else:
            _state = ALIGN_RIGHT
            _state_start_time = current_time

    if _state == ALIGN_RIGHT:
        elapsed = current_time - _state_start_time
        if elapsed < 0.55:
            return True, "Overtake Step 5: Straightening Home Lane", 0.09, 0.32
        else:
            print("[FSM] Clean parallel handoff to Lane Agent.")
            _state = LANE_FOLLOWING
            _cooldown_until = current_time + 4.0
            return False, "Maneuver Completed", -1.0, -1.0

    # -----------------------------------------------------------
    # TWO-ZONE SPATIAL FILTERING
    # -----------------------------------------------------------
    for bbox, score, cls_id in detections:
        if cls_id not in [0, 1]:
            continue

        xmin, ymin, xmax, ymax = bbox
        area = (xmax - xmin) * (ymax - ymin)
        cx = (xmin + xmax) / 2

        if cx < img_size * 0.40:
            continue

        if is_turning_heavily:
            if ymax < img_size * 0.65 and area < (img_size * img_size * 0.025):
                continue

        if cx > img_size * 0.78:
            continue

        # ZONE 1: Detection Confirmation (Duckie is seen ahead)
        if area > img_size * img_size * 0.012 or ymax > img_size * 0.55:
            obstacle_found = True

            # ZONE 2: Proximity Gate (Only true when duckie is close to our bumper)
            # Increase this multiplier (e.g., 0.68 -> 0.72) to make the bot get even closer before turning
            if ymax > img_size * 0.62 or area > (img_size * img_size * 0.045):
                is_close_enough_to_pass = True
            break

    # -----------------------------------------------------------
    # PROXIMITY-CONTROLLED STATE TRANSITIONS
    # -----------------------------------------------------------
    if obstacle_found:
        _seen_frames += 1
        if _seen_frames >= 2:
            if is_turning_heavily:
                _state = OBSTACLE_PRESENT
                return True, "Curve Lockout: Holding Lane", 0.0, current_lane_omega

            if is_close_enough_to_pass:
                # Duckie is close enough; execute safe wide-berth pass
                _state = INITIATE_PASS
                _state_start_time = current_time
                return True, "Close Proximity Verified: Initiating Overtake", 0.10, 0.40
            else:
                # Duckie is seen but far away; drop to creep speed to stabilize tracking
                _state = OBSTACLE_PRESENT
                return True, "Approach State: Slowing down and closing distance", 0.05, current_lane_omega
        return False, "Analyzing track corridor...", -1.0, -1.0

    if not obstacle_found:
        _seen_frames = 0

    if _state == OBSTACLE_PRESENT:
        _clear_frames += 1
        if _clear_frames < 15:
            return True, "Creeping forward / Waiting for trace stability", 0.04, current_lane_omega

    _state = LANE_FOLLOWING
    _seen_frames = 0
    _clear_frames = 0
    return False, "", -1.0, -1.0


def reset_fsm():
    global _state, _seen_frames, _clear_frames, _cooldown_until
    _state = LANE_FOLLOWING
    _seen_frames = 0
    _clear_frames = 0
    _cooldown_until = 0.0