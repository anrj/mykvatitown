"""
Leader bot agent for convoying project.

The leader detects yellow/white lane markings and steers to stay centered.
It also detects red stop lines and stops when encountering them.
"""

import os
import yaml
import numpy as np
import cv2
from collections import deque
from typing import Tuple

from tasks.visual_lane_servoing.packages import visual_servoing_activity as student

_CONFIG_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_config.yaml'
))

_LINE_OFFSET = 160
_ROI_START   = 0.47
_NUM_SLICES  = 3
_SLICE_TOL   = 5


def detect_lines_in_slices(
    mask_yellow: np.ndarray,
    mask_white:  np.ndarray,
    h: int,
) -> Tuple[list, list]:
    """Detect yellow and white line positions in image slices."""
    slice_height = int(h * 0.35 / _NUM_SLICES)
    start_y      = int(h * _ROI_START)
    yellow_xs, white_xs = [], []

    for i in range(_NUM_SLICES):
        y = start_y + i * slice_height + slice_height // 2

        strip_y = mask_yellow[y - _SLICE_TOL: y + _SLICE_TOL, :]
        idx = np.where(strip_y > 0)[1]
        if len(idx) > 0:
            yellow_xs.append(int(np.mean(idx)))

        strip_w = mask_white[y - _SLICE_TOL: y + _SLICE_TOL, :]
        idx = np.where(strip_w > 0)[1]
        if len(idx) > 0:
            white_xs.append(int(np.mean(idx)))

    return yellow_xs, white_xs


class LeaderAgent:
    """Leader bot that follows lanes and detects red stop lines."""

    def __init__(self, config_path: str = None):
        path = config_path or _CONFIG_FILE
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        self.p_gain              = cfg.get('p_gain',              0.1)
        self.d_gain              = cfg.get('d_gain',              0.35)
        self.max_steer           = cfg.get('max_steer',           0.4)
        self.base_speed          = cfg.get('base_speed',          0.2)
        self.curve_speed         = cfg.get('curve_speed',         0.2)
        self.curve_threshold     = cfg.get('curve_threshold',     350)
        self.steering_threshold  = cfg.get('steering_threshold',  0.2)
        self.curve_boost         = cfg.get('curve_boost',         1.3)
        self.detection_threshold = cfg.get('detection_threshold', 500)

        self.frame_count        = 0
        self._prev_error        = 0.0
        self._filtered_error    = 0.0
        self._lane_half_width   = float(_LINE_OFFSET)
        self._left_history      = deque(maxlen=3)
        self._right_history     = deque(maxlen=3)

        # Red stop line detection state
        self.red_consecutive_count = 0
        self.red_stop_sustain = 1  # immediate response
        self.red_stop_active = False

    def _calculate_error(self, yellow_xs, white_xs, left_det, right_det, w):
        """Calculate steering error based on lane detection."""
        if left_det and right_det and yellow_xs and white_xs:
            y_mean = float(np.mean(yellow_xs))
            w_mean = float(np.mean(white_xs))
            measured = (w_mean - y_mean) / 2.0
            if measured > 20:
                self._lane_half_width = 0.9 * self._lane_half_width + 0.1 * measured
            error = w / 2.0 - (y_mean + w_mean) / 2.0
        elif left_det and yellow_xs:
            error = w / 2.0 - (float(np.mean(yellow_xs)) + self._lane_half_width)
        elif right_det and white_xs:
            error = w / 2.0 - (float(np.mean(white_xs)) - self._lane_half_width)
        else:
            error = self._prev_error

        return float(np.clip(error / (w / 2.0), -1.0, 1.0))

    def _calculate_steering(self, error: float) -> float:
        """Calculate steering command from error."""
        error_diff       = error - self._prev_error
        self._prev_error = error
        steering = self.p_gain * error + self.d_gain * error_diff
        return float(np.clip(steering, -self.max_steer, self.max_steer))

    def _motor_commands(self, steering: float, recovery: bool, is_curve: bool, both_visible: bool):
        """Convert steering error to motor PWM commands."""
        if recovery:
            return 0.0, 0.0

        speed = self.curve_speed if is_curve else self.base_speed
        
        if not both_visible:
            speed *= 0.8

        left  = speed - steering
        right = speed + steering

        if is_curve and abs(steering) > self.steering_threshold:
            if steering > 0:
                right *= 5
            else:
                left  *= self.curve_boost

        return float(np.clip(left, 0.0, 1.0)), float(np.clip(right, 0.0, 1.0))

    def _smooth(self, left, right, both_visible):
        """Smooth motor commands over frames."""
        buf = 2 if both_visible else 1
        if self._left_history.maxlen != buf:
            self._left_history  = deque(maxlen=buf)
            self._right_history = deque(maxlen=buf)
        self._left_history.append(left)
        self._right_history.append(right)
        return (sum(self._left_history)  / len(self._left_history),
                sum(self._right_history) / len(self._right_history))

    def compute_commands(self, image: np.ndarray) -> Tuple[float, float]:
        """
        Detect lanes and red lines, compute motor commands.
        
        Args:
            image: RGB image from camera
            
        Returns:
            (left_pwm, right_pwm) motor commands
        """
        self.frame_count += 1
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            mask_left, mask_right = student.detect_lane_markings(bgr)
            mask_red = student.detect_red_line(bgr)
        except Exception as e:
            print(f"[LeaderAgent] detection error: {e}")
            return 0.0, 0.0

        # Check for red stop line
        red_pixels = int(np.count_nonzero(mask_red))
        red_frame_detected = red_pixels > 100  # threshold for red line presence
        
        # Accumulate red frames with hysteresis
        if red_frame_detected:
            self.red_consecutive_count += 1
        else:
            self.red_consecutive_count = 0
        
        red_detected = self.red_consecutive_count >= self.red_stop_sustain
        
        # Signal red stop if detected
        if red_detected and not self.red_stop_active:
            self.red_stop_active = True
            return 0.0, 0.0  # Stop immediately
        elif not red_detected and self.red_stop_active:
            self.red_stop_active = False

        if red_detected:
            return 0.0, 0.0  # Stay stopped

        # Detect lanes
        h, w = bgr.shape[:2]
        yellow_xs, white_xs = detect_lines_in_slices(mask_left, mask_right, h)

        left_det  = len(yellow_xs) > 0
        right_det = len(white_xs) > 0
        both_visible = left_det and right_det

        if not (left_det or right_det):
            # Lost both lines - stop to avoid collision
            return 0.0, 0.0

        # Calculate steering error
        error = self._calculate_error(yellow_xs, white_xs, left_det, right_det, w)
        steering = self._calculate_steering(error)

        # Check for curve (high pixel count = sharp turn)
        total_lane_pixels = int(np.count_nonzero(mask_left)) + int(np.count_nonzero(mask_right))
        is_curve = total_lane_pixels > self.curve_threshold

        # Compute motor commands
        left, right = self._motor_commands(steering, False, is_curve, both_visible)
        left, right = self._smooth(left, right, both_visible)

        return left, right
