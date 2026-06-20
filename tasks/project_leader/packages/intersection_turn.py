"""
Intersection turns — modcon PID heading control.

After a stop at the red line: turn in place with PIDController until the
heading error converges.
"""

import os
from typing import Optional, Tuple

import yaml

from tasks.modcon.packages.pid_controller import PIDController

_MODCON_CONFIG = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'modcon_config.yaml'
))

_TURN_ORDER = ('left', 'right', 'straight')


def _norm_turn(name: str) -> str:
    s = str(name).lower()
    if s in ('straight', 'forward', 'fwd'):
        return 'straight'
    return 'left' if s.startswith('l') else 'right'


def _angle_diff(target: float, current: float) -> float:
    diff = target - current
    while diff > 180.0:
        diff -= 360.0
    while diff < -180.0:
        diff += 360.0
    return diff


class IntersectionPlanner:
    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        ic = cfg.get('intersection', cfg)
        self.enabled = bool(ic.get('enabled', True))
        turns = ic.get('turns', list(_TURN_ORDER))
        self.turns = tuple(_norm_turn(t) for t in turns)
        self.turn_idx = 0
        self.turn_angle_deg = float(ic.get('turn_angle_deg', 90.0))
        self.stop_hold_s = float(ic.get('stop_hold_s', cfg.get('stop_hold_s', 1.0)))
        self.turn_v = float(ic.get('turn_v', 0.0))
        self.turn_timeout_s = float(ic.get('turn_timeout_s', 4.0))
        self.deadband_deg = float(ic.get('deadband_deg', 3.0))
        self.tolerance_deg = float(ic.get('tolerance_deg', 5.0))
        self.converge_samples = int(ic.get('converge_samples', 3))

        robot = self._load_robot(_MODCON_CONFIG)
        self.baseline = float(robot.get('baseline', 0.1))
        self.encoder_resolution = int(robot.get('encoder_resolution', 135))
        self.radius = float(robot.get('radius', 0.0318))

        self._stop_until = -1.0
        self._turn_target_deg: Optional[float] = None
        self._turn_start = -1.0
        self._heading_deg = 0.0
        self._prev_e = 0.0
        self._prev_int = 0.0
        self._converged = 0

    @staticmethod
    def _load_robot(path: str) -> dict:
        try:
            with open(path) as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

    def has_pending_turn(self) -> bool:
        return self.enabled and self.turn_idx < len(self.turns)

    def pending_direction(self) -> str:
        if not self.has_pending_turn():
            return 'none'
        return self.turns[self.turn_idx]

    def is_spin_turn(self) -> bool:
        """True when the next maneuver is a left/right PID turn."""
        return self.pending_direction() in ('left', 'right')

    def advance_maneuver(self) -> None:
        """Mark current intersection maneuver done (straight-through or after PID)."""
        if self.has_pending_turn():
            self.turn_idx += 1

    def arm_stop(self, now: float) -> None:
        self._stop_until = now + self.stop_hold_s

    def stop_complete(self, now: float) -> bool:
        return now >= self._stop_until

    def begin_turn(self, now: float) -> None:
        if not self.has_pending_turn() or not self.is_spin_turn():
            return
        direction = self.turns[self.turn_idx]
        sign = 1.0 if direction == 'left' else -1.0
        self._turn_target_deg = self._heading_deg + sign * self.turn_angle_deg
        self._turn_start = now
        self._prev_e = 0.0
        self._prev_int = 0.0
        self._converged = 0

    def complete_turn(self) -> None:
        if self._turn_target_deg is not None:
            self._heading_deg = self._turn_target_deg
        self._turn_target_deg = None
        self._turn_start = -1.0
        self.advance_maneuver()

    def turn_active(self) -> bool:
        return self._turn_target_deg is not None

    def pid_step(self, wheels, dt: float, now: float) -> Tuple[bool, float, float]:
        """One PID control step — returns (done, left_pwm, right_pwm)."""
        if self._turn_target_deg is None:
            return True, 0.0, 0.0

        err = _angle_diff(self._turn_target_deg, self._heading_deg)
        if abs(err) <= self.tolerance_deg:
            self._converged += 1
        else:
            self._converged = 0

        timed_out = (now - self._turn_start) >= self.turn_timeout_s
        done = self._converged >= self.converge_samples or timed_out

        if done:
            self.complete_turn()
            return True, 0.0, 0.0

        _, omega, e, e_int = PIDController(
            self.turn_v,
            self._turn_target_deg * 3.14159265 / 180.0,
            self._heading_deg * 3.14159265 / 180.0,
            self._prev_e,
            self._prev_int,
            max(dt, 1e-3),
        )
        self._prev_e = e
        self._prev_int = e_int

        # Integrate heading from commanded omega (sim-friendly).
        self._heading_deg += (omega * dt) * 180.0 / 3.14159265

        half = self.baseline / 2.0
        if abs(err) <= self.deadband_deg:
            left = right = 0.0
        elif err > 0:
            left, right = -0.25, 0.25
        else:
            left, right = 0.25, -0.25

        return False, float(left), float(right)
