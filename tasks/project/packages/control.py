from typing import Tuple

from tasks.project.project_leader.fsm import Decision


def motors_from_decision(d: Decision) -> Tuple[float, float]:
    # Clamp steering so both wheels keep moving forward — a single stopped
    # wheel pivots in place and easily overshoots (~180°) on right turns.
    base = d.base_speed
    min_wheel = 0.06
    max_steer = max(0.0, base - min_wheel)
    steer = max(-max_steer, min(max_steer, d.steering))
    left = base - steer
    right = base + steer
    return _clip01(left), _clip01(right)


def apply_leds(leds, d: Decision) -> None:
    if leds is None:
        return
    for idx, color in d.leds.items():
        try:
            leds.set_rgb(idx, list(color))
        except Exception:
            pass


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)
