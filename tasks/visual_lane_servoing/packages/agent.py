import os
import yaml
import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple

from tasks.visual_lane_servoing.packages import visual_servoing_activity as student
from tasks.visual_lane_servoing.packages.cuvrve_behavior import detect_curve

_CONFIG_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_config.yaml'
))

_LINE_OFFSET = 160
_ROI_START   = 0.47
_NUM_SLICES  = 3
_SLICE_TOL   = 5


def _edge_x(idx: np.ndarray, side: str) -> int:
    """Left edge for yellow (inner), right edge for white (outer).

    At a 90° corner a horizontal strip can hit two legs of the L; the mean
    lands between them. Extrema track the lane edge the bot should follow.
    """
    if side == "left":
        return int(np.percentile(idx, 15))
    return int(np.percentile(idx, 85))


def detect_lines_in_slices(
    mask_yellow: np.ndarray,
    mask_white:  np.ndarray,
    h: int,
) -> Tuple[list, list]:
    slice_height = int(h * 0.35 / _NUM_SLICES)
    start_y      = int(h * _ROI_START)
    yellow_xs, white_xs = [], []

    for i in range(_NUM_SLICES):
        y = start_y + i * slice_height + slice_height // 2

        strip_y = mask_yellow[y - _SLICE_TOL: y + _SLICE_TOL, :]
        idx = np.where(strip_y > 0)[1]
        if len(idx) > 0:
            yellow_xs.append(_edge_x(idx, "left"))

        strip_w = mask_white[y - _SLICE_TOL: y + _SLICE_TOL, :]
        idx = np.where(strip_w > 0)[1]
        if len(idx) > 0:
            white_xs.append(_edge_x(idx, "right"))

    return yellow_xs, white_xs


class LaneServoingAgent:

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
        self.curve_boost         = cfg.get('curve_boost',         1.4)
        self.detection_threshold = cfg.get('detection_threshold', 500)
        self.sharp_corner_threshold = cfg.get('sharp_corner_threshold', 220)
        # Slice weights: index 0 = far, -1 = near bumper. Near slices matter more
        # on bends; at a sharp 90° outer white line the top slice still sees the
        # old leg and would pull the bot straight if weighted equally.
        self.slice_weights       = cfg.get('slice_weights', [0.15, 0.30, 0.55])
        self.sharp_slice_weights = cfg.get('sharp_slice_weights', [0.05, 0.20, 0.75])

        self.frame_count        = 0
        self._prev_error        = 0.0
        self._filtered_error    = 0.0
        self._lane_half_width   = float(_LINE_OFFSET)
        self._left_history      = deque(maxlen=3)
        self._right_history     = deque(maxlen=3)
        self.last_debug_info    = self._empty_debug_info(480, 640)

    def _weighted_slice_error(self, yellow_xs, white_xs, weights, w) -> Optional[float]:
        errs, wts = [], []
        for i, wt in enumerate(weights):
            if wt <= 0:
                continue
            yx = yellow_xs[i] if i < len(yellow_xs) else None
            wx = white_xs[i] if i < len(white_xs) else None
            if yx is not None and wx is not None:
                err = w / 2.0 - (yx + wx) / 2.0
            elif yx is not None:
                err = w / 2.0 - (yx + self._lane_half_width)
            elif wx is not None:
                err = w / 2.0 - (wx - self._lane_half_width)
            else:
                continue
            errs.append(err)
            wts.append(wt)
        if not errs:
            return None
        return float(sum(e * wt for e, wt in zip(errs, wts)) / sum(wts))

    def _calculate_error(self, yellow_xs, white_xs, left_det, right_det, w,
                         weights=None):
        weights = weights or self.slice_weights
        sliced = self._weighted_slice_error(yellow_xs, white_xs, weights, w)

        if sliced is not None:
            if left_det and right_det and yellow_xs and white_xs:
                y_near = yellow_xs[-1]
                w_near = white_xs[-1]
                measured = (w_near - y_near) / 2.0
                if measured > 20:
                    self._lane_half_width = 0.9 * self._lane_half_width + 0.1 * measured
            error = sliced
        elif left_det and yellow_xs:
            error = w / 2.0 - (float(yellow_xs[-1]) + self._lane_half_width)
        elif right_det and white_xs:
            error = w / 2.0 - (float(white_xs[-1]) - self._lane_half_width)
        else:
            error = self._prev_error

        return float(np.clip(error / (w / 2.0), -1.0, 1.0))

    def _calculate_steering(self, error: float) -> float:
        error_diff       = error - self._prev_error
        self._prev_error = error
        steering = self.p_gain * error + self.d_gain * error_diff
        return float(np.clip(steering, -self.max_steer, self.max_steer))

    def _motor_commands(self, steering: float, recovery: bool, is_curve: bool, both_visible: bool):
        if recovery:
            return 0.0, 0.0

        speed = self.curve_speed if is_curve else self.base_speed
        
        if not both_visible:
            speed *= 0.8

        left  = speed - steering
        right = speed + steering

        if is_curve and abs(steering) > self.steering_threshold:
            # Boost the outside wheel symmetrically so 90° bends get enough yaw.
            if steering > 0:
                right *= self.curve_boost
            else:
                left  *= self.curve_boost

        return float(np.clip(left, 0.0, 1.0)), float(np.clip(right, 0.0, 1.0))

    def _smooth(self, left, right, both_visible):
        buf = 2 if both_visible else 1
        if self._left_history.maxlen != buf:
            self._left_history  = deque(maxlen=buf)
            self._right_history = deque(maxlen=buf)
        self._left_history.append(left)
        self._right_history.append(right)
        return (sum(self._left_history)  / len(self._left_history),
                sum(self._right_history) / len(self._right_history))

    def compute_commands(self, image: np.ndarray) -> Tuple[float, float]:
        self.frame_count += 1
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            mask_left, mask_right = student.detect_lane_markings(bgr)
        except Exception as e:
            print(f"[Agent] detect_lane_markings error: {e}")
            return 0.0, 0.0

        mask_y = (mask_left  * 255).astype(np.uint8)
        mask_w = (mask_right * 255).astype(np.uint8)

        yellow_pixels = int(np.count_nonzero(mask_y))
        white_pixels  = int(np.count_nonzero(mask_w))
        total_pixels  = yellow_pixels + white_pixels

        combined = np.clip(mask_left + mask_right, 0, 1)
        self.last_debug_info = {
            'roi':               image,
            'lane_mask':         (combined * 255).astype(np.uint8),
            'white_mask':        mask_w,
            'yellow_mask':       mask_y,
            'total_lane_pixels': total_pixels,
            'lateral_error':     float(np.clip(self._prev_error, -1.0, 1.0)),
            'lane_detected':     total_pixels >= self.detection_threshold,
            'frame_count':       self.frame_count,
        }

        h, w      = mask_y.shape
        left_det  = yellow_pixels > 0
        right_det = white_pixels  > 0
        recovery  = total_pixels  < self.detection_threshold

        yellow_xs, white_xs = detect_lines_in_slices(mask_y, mask_w, h)
        both_visible        = left_det and right_det and not recovery
        is_curve, curve_dir = detect_curve(yellow_xs, white_xs, self.curve_threshold)
        shift = max(abs((yellow_xs[-1] - yellow_xs[0]) if len(yellow_xs) >= 2 else 0),
                    abs((white_xs[-1] - white_xs[0]) if len(white_xs) >= 2 else 0))
        sharp = shift >= self.sharp_corner_threshold
        weights = self.sharp_slice_weights if sharp else self.slice_weights
        if sharp:
            is_curve = True

        raw_error            = self._calculate_error(
            yellow_xs, white_xs, left_det, right_det, w, weights=weights)
        self._filtered_error = 0.7 * self._filtered_error + 0.3 * raw_error
        steering             = self._calculate_steering(self._filtered_error)
        left, right          = self._motor_commands(steering, recovery, is_curve, both_visible)
        left, right          = self._smooth(left, right, both_visible)

        slice_height = int(h * 0.35 / _NUM_SLICES)
        start_y      = int(h * _ROI_START)
        self.last_debug_info.update({
            'yellow_xs': yellow_xs,
            'white_xs':  white_xs,
            'slice_ys':  [start_y + i * slice_height + slice_height // 2 for i in range(_NUM_SLICES)],
            'is_curve':  is_curve,
            'curve_dir': curve_dir,
            'sharp_corner': sharp,
        })

        return left, right

    def reset(self) -> None:
        self._prev_error = 0.0
        self._filtered_error = 0.0
        self._left_history.clear()
        self._right_history.clear()

    def step(self, image: np.ndarray, wheels_driver) -> Tuple[float, float]:
        left, right = self.compute_commands(image)
        wheels_driver.set_wheels_speed(left, right)
        return left, right

    def get_debug_info(self, image: np.ndarray) -> dict:
        return self.last_debug_info

    def _empty_debug_info(self, h, w):
        return {
            'roi':               np.zeros((h, w, 3), dtype=np.uint8),
            'lane_mask':         np.zeros((h, w),    dtype=np.uint8),
            'white_mask':        np.zeros((h, w),    dtype=np.uint8),
            'yellow_mask':       np.zeros((h, w),    dtype=np.uint8),
            'total_lane_pixels': 0,
            'lateral_error':     0.0,
            'lane_detected':     False,
            'frame_count':       0,
        }
