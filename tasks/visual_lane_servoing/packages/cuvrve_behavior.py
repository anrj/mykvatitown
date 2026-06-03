from typing import List, Tuple
import numpy as np


def detect_curve(yellow_xs: List[int],white_xs:  List[int],curve_threshold: int = 350,
    ) -> Tuple[bool, int]:
    def get_shift(xs: List[int]) -> int:
        valid_xs = [x for x in xs if x >= 0]
        if len(valid_xs) >= 2:
            return valid_xs[-1] - valid_xs[0]
        return 0

    shift_yellow = get_shift(yellow_xs)
    shift_white = get_shift(white_xs)
    
    max_shift = shift_yellow if abs(shift_yellow) > abs(shift_white) else shift_white
    
    if abs(max_shift) > curve_threshold:
        return True, int(np.sign(max_shift))
        
    return False, 0
