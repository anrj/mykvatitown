from typing import List, Tuple
import numpy as np


def detect_curve(
    yellow_xs: List[int],
    white_xs:  List[int],
    curve_threshold: int = 350,
) -> Tuple[bool, int]:
    shifts = []

    if len(yellow_xs) >= 2:
        shifts.append(yellow_xs[-1] - yellow_xs[0])

    if len(white_xs) >= 2:
        shifts.append(white_xs[-1] - white_xs[0])

    if not shifts:
        return False, 0

    mean_shift = float(np.mean(shifts))

    if abs(mean_shift) > curve_threshold:
        curve_direction = -1 if mean_shift > 0 else 1
        return True, curve_direction

    return False, 0
