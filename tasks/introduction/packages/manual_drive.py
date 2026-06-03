import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

SPEED = 1
TURN = 0.5


def get_motor_speeds(keys_pressed: Dict[str, bool]) -> Tuple[float, float]:
    left, right = 0, 0
    if keys_pressed.get("up", False):
        left += SPEED
        right += SPEED
    if keys_pressed.get("down", False):
        left -= SPEED
        right -= SPEED
    if keys_pressed.get("left", False):
        left -= TURN
        right += TURN
    if keys_pressed.get("right", False):
        left += TURN
        right -= TURN
    return left, right
