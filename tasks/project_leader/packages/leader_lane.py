"""
LeaderLaneAgent — centreline visual lane following for the convoy leader.

Extends LaneServoingAgent with weighted centreline steering, curve
anticipation, and helpers for stop-line / intersection behaviour.
"""

import os
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

from tasks.visual_lane_servoing.packages import visual_servoing_activity as student
from tasks.visual_lane_servoing.packages.agent import (
    LaneServoingAgent,
    detect_lines_in_slices,
    _ROI_START,
    _NUM_SLICES,
)
from tasks.visual_lane_servoing.packages.cuvrve_behavior import detect_curve

_CONFIG_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'leader_lane_config.yaml'
))

_BOTTOM_MIN_PX = 12
_DEFAULT_SLICE_WEIGHTS = (0.2, 0.35, 0.45)


def _norm_side(name: str) -> str:
    s = str(name).lower()
    if s in ('left', 'yellow', 'dashed'):
        return 'left'
    return 'right'


def centerline_xs(
    yellow_xs: List[int],
    white_xs: List[int],
    half_width: float,
) -> List[int]:
    """Midpoint per slice; offset by half_width when only one edge is visible."""
    n = max(len(yellow_xs), len(white_xs))
    hw = int(half_width)
    out = []
    for i in range(n):
        y = yellow_xs[i] if i < len(yellow_xs) and yellow_xs[i] >= 0 else None
        wl = white_xs[i] if i < len(white_xs) and white_xs[i] >= 0 else None
        if y is not None and wl is not None:
            out.append(int((y + wl) / 2))
        elif y is not None:
            out.append(int(y + hw))
        elif wl is not None:
            out.append(int(wl - hw))
    return out


class LeaderLaneAgent(LaneServoingAgent):

    def __init__(self, config_path: str = None):
        path = config_path or _CONFIG_FILE
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        super().__init__(config_path=path)

        self.lane_half_width_px = float(cfg.get('lane_half_width_px', 200))
        self.min_lane_half_width_px = float(cfg.get('min_lane_half_width_px', 180))
        self.error_filter_alpha = float(cfg.get('error_filter_alpha', 0.3))
        self.curve_inner_scale = float(cfg.get('curve_inner_scale', 1.12))
        self.curve_outer_scale = float(cfg.get('curve_outer_scale', 1.18))
        self.curve_wheel_boost = float(cfg.get('curve_wheel_boost', 1.45))
        self.curve_steer_scale = float(cfg.get('curve_steer_scale', 0.78))
        self.curve_center_bias_px = float(cfg.get('curve_center_bias_px', 28))
        self.slice_weights = tuple(cfg.get('slice_weights', _DEFAULT_SLICE_WEIGHTS))
        self.intersection_half_width_px = float(cfg.get('intersection_half_width_px', 230))
        self.single_line_width_factor = float(cfg.get('single_line_width_factor', 0.88))
        self.yellow_single_bias_px = float(cfg.get('yellow_single_bias_px', 40))
        self.white_single_bias_px = float(cfg.get('white_single_bias_px', 0))
        self.error_recovery_threshold = float(cfg.get('error_recovery_threshold', 0.28))
        self.error_recovery_steer_boost = float(cfg.get('error_recovery_steer_boost', 1.3))
        self._lane_half_width = self.lane_half_width_px

    def _effective_half_width(self, both_visible: bool) -> float:
        """Learned lane width; use a wider estimate only when both edges are seen."""
        if both_visible:
            return self._lane_half_width
        return min(self._lane_half_width, self.intersection_half_width_px)

    def _weighted_mean(self, values: List[int]) -> Optional[float]:
        if not values:
            return None
        n = min(len(values), len(self.slice_weights))
        weights = self.slice_weights[-n:]
        wsum = sum(weights)
        return sum(v * w for v, w in zip(values[-n:], weights)) / wsum

    def _curve_factor(self, yellow_xs: List[int], white_xs: List[int]) -> float:
        shift = 0
        for xs in (yellow_xs, white_xs):
            valid = [x for x in xs if x >= 0]
            if len(valid) >= 2:
                shift = valid[-1] - valid[0]
                break
        if abs(shift) < 40:
            return 1.0
        return self.curve_outer_scale if shift > 0 else self.curve_inner_scale

    def _error_from_centerline(self, yellow_xs: List[int], white_xs: List[int], w: int,
                               curve_dir: int = 0, half_width: Optional[float] = None) -> float:
        hw = half_width if half_width is not None else self._lane_half_width
        mids = centerline_xs(yellow_xs, white_xs, hw)
        center_x = self._weighted_mean(mids)
        if center_x is None:
            return self._prev_error * (w / 2.0)
        if len(yellow_xs) >= 2 and len(white_xs) >= 2:
            valid_y = [x for x in yellow_xs if x >= 0]
            valid_w = [x for x in white_xs if x >= 0]
            if valid_y and valid_w:
                measured = (float(np.mean(valid_w)) - float(np.mean(valid_y))) / 2.0
                if measured > 20:
                    self._lane_half_width = max(
                        self.min_lane_half_width_px,
                        0.9 * self._lane_half_width + 0.1 * measured,
                    )
        if curve_dir > 0:
            center_x -= self.curve_center_bias_px
        elif curve_dir < 0:
            center_x += self.curve_center_bias_px
        return w / 2.0 - center_x

    def _single_line_error(self, yellow_xs, white_xs, left_det, right_det, w):
        """One edge visible — use learned half-width, not a fixed inset."""
        hw = self._effective_half_width(both_visible=False)
        hw *= self.single_line_width_factor
        if left_det and yellow_xs:
            y_mean = float(np.mean([x for x in yellow_xs if x >= 0]))
            # Bias target left when hugging the right edge (yellow-only approach).
            center_x = y_mean + hw - self.yellow_single_bias_px
        elif right_det and white_xs:
            w_mean = float(np.mean([x for x in white_xs if x >= 0]))
            center_x = w_mean - hw + self.white_single_bias_px
        else:
            return self._prev_error
        return float(np.clip((w / 2.0 - center_x) / (w / 2.0), -1.0, 1.0))

    def _calculate_error(self, yellow_xs, white_xs, left_det, right_det, w, curve_dir=0,
                         both_visible=False):
        has_y = bool(yellow_xs and any(x >= 0 for x in yellow_xs))
        has_w = bool(white_xs and any(x >= 0 for x in white_xs))
        if has_y and has_w:
            error = self._error_from_centerline(
                yellow_xs, white_xs, w, curve_dir=curve_dir,
            )
        elif has_y or has_w:
            error = self._single_line_error(yellow_xs, white_xs, has_y, has_w, w)
        else:
            error = self._prev_error
        return float(np.clip(error, -1.0, 1.0))

    def _filter_error(self, raw_error: float) -> float:
        a = self.error_filter_alpha
        self._filtered_error = (1.0 - a) * self._filtered_error + a * raw_error
        return self._filtered_error

    def _motor_commands(self, steering, recovery, is_curve, both_visible, curve_factor=1.0):
        """Softer curve mixing than LaneServoingAgent (no right-wheel x5)."""
        if recovery:
            return 0.0, 0.0

        speed = self.curve_speed if is_curve else self.base_speed
        if not both_visible:
            speed *= 0.8

        if is_curve:
            steering = steering * self.curve_steer_scale

        left = speed - steering
        right = speed + steering

        if is_curve and abs(steering) > self.steering_threshold:
            boost = min(self.curve_wheel_boost, curve_factor)
            if steering > 0:
                right = min(1.0, right * boost)
            elif steering < 0:
                left = min(1.0, left * boost)

        return float(np.clip(left, 0.0, 1.0)), float(np.clip(right, 0.0, 1.0))

    def _process_lane(self, image: np.ndarray, follow_side: Optional[str] = None):
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            mask_left, mask_right = student.detect_lane_markings(bgr)
        except Exception as e:
            print(f'[LeaderLane] detect_lane_markings error: {e}')
            return 0.0, 0.0, self.last_debug_info

        mask_y = (mask_left * 255).astype(np.uint8)
        mask_w = (mask_right * 255).astype(np.uint8)
        h, w = mask_y.shape

        yellow_pixels = int(np.count_nonzero(mask_y))
        white_pixels = int(np.count_nonzero(mask_w))
        total_pixels = yellow_pixels + white_pixels
        left_det = yellow_pixels > 0
        right_det = white_pixels > 0
        recovery = total_pixels < self.detection_threshold

        yellow_xs, white_xs = detect_lines_in_slices(mask_y, mask_w, h)
        both_visible = left_det and right_det and not recovery
        is_curve, curve_dir = detect_curve(yellow_xs, white_xs, self.curve_threshold)
        curve_factor = self._curve_factor(yellow_xs, white_xs)

        if follow_side == 'left' and yellow_xs:
            raw_error = self._single_line_error(yellow_xs, white_xs, True, False, w)
        elif follow_side == 'right' and white_xs:
            raw_error = self._single_line_error(yellow_xs, white_xs, False, True, w)
        else:
            raw_error = self._calculate_error(
                yellow_xs, white_xs, left_det, right_det, w,
                curve_dir=curve_dir, both_visible=both_visible,
            )

        error = self._filter_error(raw_error)
        steering = self._calculate_steering(error)
        if abs(error) >= self.error_recovery_threshold:
            steering = float(np.clip(
                steering * self.error_recovery_steer_boost,
                -self.max_steer, self.max_steer,
            ))
        left, right = self._motor_commands(
            steering, recovery, is_curve, both_visible, curve_factor,
        )
        left, right = self._smooth(left, right, both_visible)

        has_y = bool(yellow_xs and any(x >= 0 for x in yellow_xs))
        has_w = bool(white_xs and any(x >= 0 for x in white_xs))
        single_line = (has_y or has_w) and not (has_y and has_w)

        combined = np.clip(mask_left + mask_right, 0, 1)
        slice_height = int(h * 0.35 / _NUM_SLICES)
        start_y = int(h * _ROI_START)
        self.last_debug_info = {
            'roi': image,
            'lane_mask': (combined * 255).astype(np.uint8),
            'white_mask': mask_w,
            'yellow_mask': mask_y,
            'total_lane_pixels': total_pixels,
            'lateral_error': float(np.clip(error, -1.0, 1.0)),
            'raw_lateral_error': float(np.clip(raw_error, -1.0, 1.0)),
            'lane_half_width_px': round(self._lane_half_width, 1),
            'single_line_mode': single_line,
            'lane_detected': total_pixels >= self.detection_threshold,
            'frame_count': self.frame_count,
            'yellow_xs': yellow_xs,
            'white_xs': white_xs,
            'centerline_xs': centerline_xs(yellow_xs, white_xs, self._lane_half_width),
            'slice_ys': [start_y + i * slice_height + slice_height // 2 for i in range(_NUM_SLICES)],
            'is_curve': is_curve,
            'curve_dir': curve_dir,
            'follow_line': follow_side,
        }
        return left, right, self.last_debug_info

    def compute_commands(self, image: np.ndarray):
        self.frame_count += 1
        left, right, _ = self._process_lane(image)
        return left, right

    def reset_steering_state(self):
        self._prev_error = 0.0
        self._filtered_error = 0.0
        self._left_history = deque(maxlen=3)
        self._right_history = deque(maxlen=3)

    def bottom_line_visible(self, lane_info: dict, side: str, bottom_px: int) -> bool:
        side = _norm_side(side)
        mask = lane_info.get('yellow_mask') if side == 'left' else lane_info.get('white_mask')
        if mask is None:
            return False
        h = mask.shape[0]
        band = mask[max(0, h - bottom_px):, :]
        return int(np.count_nonzero(band)) >= _BOTTOM_MIN_PX

    def compute_single_line_commands(self, image: np.ndarray, follow_side: str):
        self.frame_count += 1
        follow = _norm_side(follow_side)
        left, right, _ = self._process_lane(image, follow_side=follow)
        return left, right

    def compute_intersection_commands(self, image: np.ndarray, turn_dir: str):
        """Stay on centreline through a curve — do not hug the outside white line."""
        self.frame_count += 1
        left, right, _ = self._process_lane(image)
        return left, right
