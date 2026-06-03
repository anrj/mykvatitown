from typing import Tuple

import numpy as np


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """Left motor weight matrix: highest at bottom-left, decreasing toward top-right."""
    h, w = shape
    y = np.linspace(0.0, 1.0, h)[:, np.newaxis]
    x = np.linspace(1.0, 0.0, w)[np.newaxis, :]
    return y * x


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    """Right motor weight matrix: highest at bottom-right, decreasing toward top-left."""
    h, w = shape
    y = np.linspace(0.0, 1.0, h)[:, np.newaxis]
    x = np.linspace(0.0, 1.0, w)[np.newaxis, :]
    return y * x
