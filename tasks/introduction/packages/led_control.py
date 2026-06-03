import colorsys
from typing import List


def set_turning_leds(direction: str) -> dict:
    if direction == "left":
        return {
            0: [1.0, 1.0, 0.0],
            2: [0.0, 0.0, 0.0],
            3: [0.0, 0.0, 0.0],
            4: [1.0, 1.0, 0.0],
        }
    elif direction == "right":
        return {
            0: [0.0, 0.0, 0.0],
            2: [1.0, 1.0, 0.0],
            3: [1.0, 1.0, 0.0],
            4: [0.0, 0.0, 0.0],
        }
    elif direction == "stop":
        return {
            0: [0.0, 0.0, 0.0],
            2: [0.0, 0.0, 0.0],
            3: [1.0, 0.0, 0.0],
            4: [1.0, 0.0, 0.0],
        }
    else:
        return {
            0: [0.0, 0.0, 0.0],
            2: [0.0, 0.0, 0.0],
            3: [0.0, 0.0, 0.0],
            4: [0.0, 0.0, 0.0],
        }
