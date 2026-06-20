"""
Autonomous lane-following leader agent for convoying project.

The bot detects yellow/white lane markings and steers to stay centered on the road.
It also detects red stop lines and stops when encountering them.
"""

import os
import threading
import cv2
import numpy as np
import yaml
from collections import deque
from typing import Tuple

from tasks.visual_lane_servoing.packages import visual_servoing_activity as student

# Published for the web UI / debugging
DETECTION = {}
_det_lock = threading.Lock()
STATUS = {}

# Runtime config
CFG = None
PAUSED = True

_LINE_OFFSET = 160
_ROI_START = 0.47
_NUM_SLICES = 3
_SLICE_TOL = 5

_DEFAULTS = {
    'p_gain': 0.1,
    'd_gain': 0.35,
    'max_steer': 0.4,
    'base_speed': 0.2,
    'curve_speed': 0.2,
    'curve_threshold': 350,
    'steering_threshold': 0.2,
    'curve_boost': 1.3,
    'detection_threshold': 500,
}

CONFIG_FILE = 'project_config_sim.yaml'


def _config_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, '..', '..', '..', 'config', CONFIG_FILE)


def _load_config():
    global CFG
    try:
        with open(_config_path()) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    CFG = {**_DEFAULTS, **cfg}


def detect_lines_in_slices(mask_yellow, mask_white, h):
    """Detect lane line positions in image slices."""
    slice_height = int(h * 0.35 / _NUM_SLICES)
    start_y = int(h * _ROI_START)
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


class AutonomousAgent:
    """Lane-following autonomous agent."""

    def __init__(self):
        _load_config()
        
        self.p_gain = CFG.get('p_gain', 0.1)
        self.d_gain = CFG.get('d_gain', 0.35)
        self.max_steer = CFG.get('max_steer', 0.4)
        self.base_speed = CFG.get('base_speed', 0.2)
        self.curve_speed = CFG.get('curve_speed', 0.2)
        self.curve_threshold = CFG.get('curve_threshold', 350)
        self.steering_threshold = CFG.get('steering_threshold', 0.2)
        self.curve_boost = CFG.get('curve_boost', 1.3)
        self.detection_threshold = CFG.get('detection_threshold', 500)

        self.frame_count = 0
        self._prev_error = 0.0
        self._lane_half_width = float(_LINE_OFFSET)
        self._left_history = deque(maxlen=3)
        self._right_history = deque(maxlen=3)

        # Red stop line detection
        self.red_consecutive_count = 0
        self.red_stop_sustain = 1
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

    def _calculate_steering(self, error):
        """Calculate steering command."""
        error_diff = error - self._prev_error
        self._prev_error = error
        steering = self.p_gain * error + self.d_gain * error_diff
        return float(np.clip(steering, -self.max_steer, self.max_steer))

    def _motor_commands(self, steering, is_curve, both_visible):
        """Convert steering to motor PWM commands."""
        speed = self.curve_speed if is_curve else self.base_speed
        
        if not both_visible:
            speed *= 0.8

        left = speed - steering
        right = speed + steering

        if is_curve and abs(steering) > self.steering_threshold:
            if steering > 0:
                right *= self.curve_boost
            else:
                left *= self.curve_boost

        return float(np.clip(left, 0.0, 1.0)), float(np.clip(right, 0.0, 1.0))

    def _smooth(self, left, right, both_visible):
        """Smooth motor commands over frames."""
        buf = 2 if both_visible else 1
        if self._left_history.maxlen != buf:
            self._left_history = deque(maxlen=buf)
            self._right_history = deque(maxlen=buf)
        self._left_history.append(left)
        self._right_history.append(right)
        return (sum(self._left_history) / len(self._left_history),
                sum(self._right_history) / len(self._right_history))

    def compute_commands(self, image_rgb):
        """Compute motor commands from camera image."""
        self.frame_count += 1
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        try:
            mask_yellow, mask_white = student.detect_lane_markings(bgr)
            mask_red = student.detect_red_line(bgr)
        except Exception as e:
            print(f"[AutonomousAgent] detection error: {e}")
            return 0.0, 0.0

        # Check for red stop line
        red_pixels = int(np.count_nonzero(mask_red))
        red_frame_detected = red_pixels > 100
        
        if red_frame_detected:
            self.red_consecutive_count += 1
        else:
            self.red_consecutive_count = 0
        
        red_detected = self.red_consecutive_count >= self.red_stop_sustain
        
        if red_detected and not self.red_stop_active:
            self.red_stop_active = True
            return 0.0, 0.0
        elif not red_detected and self.red_stop_active:
            self.red_stop_active = False

        if red_detected:
            return 0.0, 0.0

        # Detect lanes
        h, w = bgr.shape[:2]
        yellow_xs, white_xs = detect_lines_in_slices(mask_yellow, mask_white, h)

        left_det = len(yellow_xs) > 0
        right_det = len(white_xs) > 0
        both_visible = left_det and right_det

        # If no lanes detected, stop
        if not (left_det or right_det):
            return 0.0, 0.0

        # Calculate steering
        error = self._calculate_error(yellow_xs, white_xs, left_det, right_det, w)
        steering = self._calculate_steering(error)

        # Check for curve
        total_pixels = int(np.count_nonzero(mask_yellow)) + int(np.count_nonzero(mask_white))
        is_curve = total_pixels > self.curve_threshold

        # Motor commands
        left, right = self._motor_commands(steering, is_curve, both_visible)
        left, right = self._smooth(left, right, both_visible)

        return left, right


def main(camera, wheels, leds, stop_event):
    """Main loop: read frames, compute commands, drive motors."""
    global CFG
    
    _load_config()
    agent = AutonomousAgent()

    while not stop_event.is_set():
        ok, frame = camera.read()
        if not ok or frame is None:
            stop_event.wait(0.02)
            continue

        if not PAUSED:
            try:
                left, right = agent.compute_commands(frame)
                wheels.set_wheels_speed(left, right)
            except Exception as e:
                print(f"[Agent] error: {e}")
                wheels.set_wheels_speed(0, 0)
        else:
            wheels.set_wheels_speed(0, 0)

        stop_event.wait(0.02)
