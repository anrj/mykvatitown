from typing import Tuple

import numpy as np


def delta_phi(ticks: int, prev_ticks: int, resolution: int) -> Tuple[float, float]:
    delta_ticks = ticks - prev_ticks
    alpha = 2 * np.pi / resolution
    dphi = alpha * delta_ticks

    return float(dphi), float(ticks)


def pose_estimation(
    R: float,
    baseline: float,
    x_prev: float,
    y_prev: float,
    theta_prev: float,
    delta_phi_left: float,
    delta_phi_right: float,
) -> Tuple[float, float, float]:
    # distance travelled by each wheel
    d_left = R * delta_phi_left
    d_right = R * delta_phi_right

    # distance travelled by the robot center and the rotation of the robot
    d_A = (d_left + d_right) / 2.0
    delta_theta = (d_right - d_left) / baseline

    x = x_prev + d_A * np.cos(theta_prev)
    y = y_prev + d_A * np.sin(theta_prev)
    theta = theta_prev + delta_theta

    return float(x), float(y), float(theta)
