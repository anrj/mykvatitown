import os
from typing import Tuple

import numpy as np
import yaml

_GAINS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "config", "modcon_config.yaml"
)
try:
    with open(_GAINS_FILE) as _f:
        _g = yaml.safe_load(_f) or {}
except FileNotFoundError:
    _g = {}

K_P = _g.get("k_P", 0.0)
K_I = _g.get("k_I", 0.0)
K_D = _g.get("k_D", 0.0)
MAX_OMEGA = _g.get("max_omega", 8.0)
MIN_OMEGA = -MAX_OMEGA


def PIDController(
    v_0: float,
    theta_ref: float,
    theta_hat: float,
    prev_e: float,
    prev_int: float,
    delta_t: float,
) -> Tuple[float, float, float, float]:
    # error
    e = theta_ref - theta_hat

    # integral
    e_int = prev_int + (e * delta_t)

    # derivative
    if delta_t > 0:
        e_der = (e - prev_e) / delta_t
    else:
        e_der = 0.0

    # control signal
    omega = (K_P * e) + (K_I * e_int) + (K_D * e_der)
    omega = max(MIN_OMEGA, min(MAX_OMEGA, omega))

    return v_0, omega, e, e_int
