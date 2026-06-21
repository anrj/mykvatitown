"""
Intersection maneuvers — ModCon odometry + PID heading + distance goals.

Uses wheel odometry (real encoders or PWM estimate in sim) so cross/turn
do not depend on frame counts or fake integrated heading.
"""

import math
import os
import time
from typing import Any, Optional, Tuple

import numpy as np
import yaml

from tasks.modcon.packages.odometry_activity import delta_phi, pose_estimation
from tasks.modcon.packages.pid_controller import PIDController

_MODCON_CONFIG = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'modcon_config.yaml'
))

_TURN_ORDER = ('left', 'right', 'straight')
_TURN_DEADBAND_RAD = 0.035  # ~2°, matches modcon virtual_server
_SIM_PWM_TO_TICKS_PER_SEC = 676.0  # modcon sim calibration


def _norm_turn(name: str) -> str:
    s = str(name).lower()
    if s in ('straight', 'forward', 'fwd'):
        return 'straight'
    return 'left' if s.startswith('l') else 'right'


def _wrap_angle_rad(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def pwm_from_velocity(v: float, omega: float, radius: float, baseline: float) -> Tuple[float, float]:
    """Differential-drive PWM from linear and angular velocity (modcon)."""
    v_left = (v - omega * baseline / 2.0) / radius
    v_right = (v + omega * baseline / 2.0) / radius
    return (
        float(np.clip(v_left, -1.0, 1.0)),
        float(np.clip(v_right, -1.0, 1.0)),
    )


class LeaderOdometry:
    """Pose tracker — encoders on hardware, PWM integration in sim."""

    def __init__(self, radius: float = 0.0318, baseline: float = 0.1, resolution: int = 135):
        self.R = radius
        self.baseline = baseline
        self.resolution = resolution
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.ticks_left = 0
        self.ticks_right = 0
        self._prev_ticks_left = 0
        self._prev_ticks_right = 0
        self._last_update = time.monotonic()

    @property
    def theta_deg(self) -> float:
        return float(math.degrees(self.theta))

    def reset_pose(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.theta = theta
        self.ticks_left = 0
        self.ticks_right = 0
        self._prev_ticks_left = 0
        self._prev_ticks_right = 0
        self._last_update = time.monotonic()

    def _estimate_ticks_from_pwm(self, left_pwm: float, right_pwm: float, dt: float) -> None:
        k = _SIM_PWM_TO_TICKS_PER_SEC
        dl = int(k * abs(left_pwm) * dt)
        dr = int(k * abs(right_pwm) * dt)
        if left_pwm < 0:
            dl = -dl
        if right_pwm < 0:
            dr = -dr
        self.ticks_left += dl
        self.ticks_right += dr

    def update(self, wheels: Any, dt: float) -> None:
        dt = max(dt, 1e-4)
        encoders = getattr(wheels, 'encoders', None)
        if encoders is not None:
            self.ticks_left = int(encoders.left.ticks)
            self.ticks_right = int(encoders.right.ticks)
        else:
            self._estimate_ticks_from_pwm(
                float(getattr(wheels, 'left_pwm', 0.0)),
                float(getattr(wheels, 'right_pwm', 0.0)),
                dt,
            )

        dphi_left, self._prev_ticks_left = delta_phi(
            self.ticks_left, self._prev_ticks_left, self.resolution,
        )
        dphi_right, self._prev_ticks_right = delta_phi(
            self.ticks_right, self._prev_ticks_right, self.resolution,
        )
        self.x, self.y, self.theta = pose_estimation(
            self.R, self.baseline,
            self.x, self.y, self.theta,
            dphi_left, dphi_right,
        )
        self._last_update = time.monotonic()


class IntersectionPlanner:
    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        ic = cfg.get('intersection', cfg)
        mc = cfg.get('maneuver', {})

        self.enabled = bool(ic.get('enabled', True))
        turns = ic.get('turns', ic.get('maneuvers', list(_TURN_ORDER)))
        self.turns = tuple(_norm_turn(t) for t in turns)
        self.turn_idx = 0
        self.turn_angle_deg = float(ic.get('turn_angle_deg', 90.0))
        self.stop_hold_s = float(ic.get('stop_hold_s', cfg.get('stop_hold_s', 1.0)))
        self.turn_v = float(ic.get('turn_v', 0.0))
        self.turn_timeout_s = float(ic.get('turn_timeout_s', 12.0))
        self.deadband_deg = float(ic.get('deadband_deg', 2.0))
        self.tolerance_deg = float(ic.get('tolerance_deg', 5.0))
        self.converge_samples = int(ic.get('converge_samples', 8))

        self.cross_distance_m = float(mc.get('cross_distance_m', 0.12))
        self.cross_timeout_s = float(mc.get('cross_timeout_s', 8.0))

        robot = self._load_robot(_MODCON_CONFIG)
        self.baseline = float(robot.get('baseline', 0.1))
        self.encoder_resolution = int(robot.get('encoder_resolution', 135))
        self.radius = float(robot.get('radius', 0.0318))

        self.odometry = LeaderOdometry(
            radius=self.radius,
            baseline=self.baseline,
            resolution=self.encoder_resolution,
        )

        self._stop_until = -1.0
        self._turn_target_rad: Optional[float] = None
        self._turn_start = -1.0
        self._prev_e = 0.0
        self._prev_int = 0.0
        self._recent_errors: list = []
        self._pid_error_deg = 0.0

        self._cross_start_x = 0.0
        self._cross_start_y = 0.0
        self._cross_started_at = -1.0
        self._cross_active = False

    @property
    def pid_error_deg(self) -> float:
        return self._pid_error_deg

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
        return self.pending_direction() in ('left', 'right')

    def advance_maneuver(self) -> None:
        if self.has_pending_turn():
            self.turn_idx += 1

    def arm_stop(self, now: float) -> None:
        self._stop_until = now + self.stop_hold_s

    def stop_complete(self, now: float) -> bool:
        return now >= self._stop_until

    def begin_cross_segment(self) -> None:
        self._cross_start_x = self.odometry.x
        self._cross_start_y = self.odometry.y
        self._cross_started_at = time.monotonic()
        self._cross_active = True

    def segment_distance(self) -> float:
        dx = self.odometry.x - self._cross_start_x
        dy = self.odometry.y - self._cross_start_y
        return float(math.sqrt(dx * dx + dy * dy))

    def cross_segment_complete(self, now: float) -> bool:
        if not self._cross_active:
            return False
        if self.segment_distance() >= self.cross_distance_m:
            return True
        if self._cross_started_at >= 0 and (now - self._cross_started_at) >= self.cross_timeout_s:
            return True
        return False

    def end_cross_segment(self) -> None:
        self._cross_active = False
        self._cross_started_at = -1.0

    def begin_turn(self, now: float) -> None:
        if not self.has_pending_turn() or not self.is_spin_turn():
            return
        direction = self.turns[self.turn_idx]
        sign = 1.0 if direction == 'left' else -1.0
        self._turn_target_rad = self.odometry.theta + sign * math.radians(self.turn_angle_deg)
        self._turn_start = now
        self._prev_e = 0.0
        self._prev_int = 0.0
        self._recent_errors = []
        self._pid_error_deg = 0.0

    def complete_turn(self) -> None:
        self._turn_target_rad = None
        self._turn_start = -1.0
        self._recent_errors = []
        self._pid_error_deg = 0.0
        self.advance_maneuver()

    def turn_active(self) -> bool:
        return self._turn_target_rad is not None

    def record_motion(self, wheels: Any, dt: float) -> None:
        self.odometry.update(wheels, dt)

    def pid_step(self, wheels: Any, dt: float, now: float) -> Tuple[bool, float, float]:
        """ModCon-style PID turn — returns (done, left_pwm, right_pwm)."""
        if self._turn_target_rad is None:
            return True, 0.0, 0.0

        dt = max(dt, 1e-3)
        theta = self.odometry.theta
        error = _wrap_angle_rad(self._turn_target_rad - theta)
        self._pid_error_deg = math.degrees(error)
        self._recent_errors.append(abs(error))
        if len(self._recent_errors) > 20:
            self._recent_errors.pop(0)

        timed_out = (now - self._turn_start) >= self.turn_timeout_s
        tol_rad = math.radians(self.tolerance_deg)
        converged = (
            len(self._recent_errors) >= self.converge_samples
            and max(self._recent_errors[-self.converge_samples:]) < tol_rad
        )
        if timed_out or converged:
            self.complete_turn()
            return True, 0.0, 0.0

        _, omega, e, e_int = PIDController(
            self.turn_v,
            self._turn_target_rad,
            theta,
            self._prev_e,
            self._prev_int,
            dt,
        )
        self._prev_e = e
        self._prev_int = e_int

        if abs(error) < _TURN_DEADBAND_RAD:
            omega = 0.0

        left, right = pwm_from_velocity(self.turn_v, omega, self.radius, self.baseline)
        return False, left, right
