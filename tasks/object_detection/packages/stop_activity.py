from typing import List, Tuple, Optional

Detection = Tuple[Tuple[int, int, int, int], float, int]

class_names = {0: 'duckie', 1: 'truck', 2: 'sign'}

_stop_active = False
STOP_THRESHOLD = 0.15
CLEAR_THRESHOLD = 0.05
LANE_MARGIN = 30
ASSUMED_LANE_WIDTH = 250


def _in_lane(cx: float, lane_info: dict, orig_w: int) -> bool:
    yellow_xs = lane_info.get('yellow_xs', [])
    white_xs = lane_info.get('white_xs', [])
    left_boundary = min(yellow_xs) if yellow_xs else None
    right_boundary = max(white_xs) if white_xs else None
    if left_boundary is not None and right_boundary is not None:
        return left_boundary - LANE_MARGIN < cx < right_boundary + LANE_MARGIN
    if left_boundary is not None:
        return left_boundary - LANE_MARGIN < cx < left_boundary + ASSUMED_LANE_WIDTH
    if right_boundary is not None:
        return right_boundary - ASSUMED_LANE_WIDTH < cx < right_boundary + LANE_MARGIN
    return True


def reset_stop_state():
    """Call this when the simulation resets."""
    global _stop_active
    _stop_active = False


def should_stop(
    detections: List[Detection],
    img_size: int = None,
    orig_w: int = None,
    orig_h: int = None,
    lane_info: Optional[dict] = None,
) -> Tuple[bool, str]:
    global _stop_active

    if not detections:
        _stop_active = False
        return False, ""

    if orig_w is None or orig_h is None:
        if img_size is not None:
            orig_w = orig_w or int(img_size * 640 / 416)
            orig_h = orig_h or int(img_size * 480 / 416)
        else:
            raise ValueError("Provide either orig_w+orig_h or img_size")

    max_danger = 0.0
    worst = None

    for bbox, score, cls_id in detections:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2

        if lane_info and not _in_lane(cx, lane_info, orig_w):
            continue

        cx_norm = cx / orig_w
        y2_ratio = y2 / orig_h

        margin = 0.25 - 0.15 * y2_ratio
        if cx_norm < margin or cx_norm > (1 - margin):
            continue

        area = (x2 - x1) * (y2 - y1)
        area_ratio = area / (orig_w * orig_h)
        class_mult = {0: 2.5, 1: 1.5, 2: 1.0}.get(cls_id, 1.0)
        danger = y2_ratio * area_ratio * 50 * class_mult

        if danger > max_danger:
            max_danger = danger
            worst = (bbox, cls_id)

    if worst is None:
        _stop_active = False
        return False, ""

    threshold = CLEAR_THRESHOLD if _stop_active else STOP_THRESHOLD
    should = max_danger > threshold
    _stop_active = should

    if should:
        _, cls_id = worst
        return True, f"{class_names[cls_id]} ahead"
    return False, ""
