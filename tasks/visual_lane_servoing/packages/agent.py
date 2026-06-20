import os
import yaml
import numpy as np
import cv2
from collections import deque
from typing import Tuple

from tasks.visual_lane_servoing.packages import visual_servoing_activity as student
from tasks.visual_lane_servoing.packages.cuvrve_behavior import detect_curve

_CONFIG_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "config", "lane_servoing_config.yaml"
))

_LINE_OFFSET = 160
_ROI_START = 0.50
_SLICE_TOL = 6

# More slices = more stable on curves.
_SLICE_Y_RATIOS = [0.56, 0.64, 0.72, 0.80, 0.88]


def _strip_center_x(mask: np.ndarray, y: int, white_x=None, prefer_right: bool = False):
    h, w = mask.shape[:2]

    y1 = max(0, y - _SLICE_TOL)
    y2 = min(h, y + _SLICE_TOL)

    strip = mask[y1:y2, :]
    idx = np.where(strip > 0)[1]

    if len(idx) == 0:
        return None

    idx = np.sort(idx)

    # Split pixels into separate clusters.
    gaps = np.where(np.diff(idx) > 20)[0] + 1
    clusters = np.split(idx, gaps)

    valid_clusters = []

    for cluster in clusters:
        if len(cluster) < 3:
            continue

        cluster_left = int(cluster[0])
        cluster_right = int(cluster[-1])
        cluster_width = cluster_right - cluster_left + 1
        cluster_center = float(np.median(cluster))

        # Reject tiny noise.
        if cluster_width < 3:
            continue

        if prefer_right:
            # WHITE LINE:
            # We only want the right-side white border.
            if cluster_center < w * 0.35:
                continue
        else:
            # YELLOW LINE:
            # Yellow road line should usually be left/center, not far right.
            if cluster_center > w * 0.75:
                continue

        valid_clusters.append(cluster)

    if not valid_clusters:
        return None

    if prefer_right:
        # WHITE:
        # Choose nearest right-side white cluster.
        # Then use its LEFT edge, because the lane is left of the white line.
        best = min(valid_clusters, key=lambda c: np.median(c))

        left_edge = int(best[0])
        return left_edge

    # YELLOW:
    # Choose strongest yellow cluster.
    # Then use its RIGHT edge, because the lane is right of the yellow line.
    candidates = valid_clusters

    # don't take yellow near white
    if white_x is not None:
        DIST_THRESH = 200

        filtered = [
            c for c in candidates
            if abs(np.median(c) - white_x) > DIST_THRESH
        ]

        if len(filtered) > 0:
            candidates = filtered

    best = max(candidates, key=lambda c: np.median(c))

    right_edge = int(best[-1])
    return right_edge


def detect_lines_in_slices(
    mask_yellow: np.ndarray,
    mask_white: np.ndarray,
    h: int,
) -> Tuple[list, list]:
    yellow_xs = []
    white_xs = []

    for ratio in _SLICE_Y_RATIOS:
        y = int(h * ratio)

        white_x = _strip_center_x(mask_white, y, prefer_right=True)
        yellow_x = _strip_center_x(mask_yellow, y, white_x, prefer_right=False)

        if yellow_x is not None:
            yellow_xs.append(yellow_x)

        if white_x is not None:
            white_xs.append(white_x)

    return yellow_xs, white_xs


class LaneServoingAgent:

    def __init__(self, config_path: str = None):
        path = config_path or _CONFIG_FILE

        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        self.p_gain = cfg.get("p_gain", 0.20)
        self.d_gain = cfg.get("d_gain", 0.11)
        self.max_steer = cfg.get("max_steer", 0.22)
        self.base_speed = cfg.get("base_speed", 0.30)
        self.curve_speed = cfg.get("curve_speed", 0.1)
        self.curve_threshold = cfg.get("curve_threshold", 350)
        self.steering_threshold = cfg.get("steering_threshold", 0.2)
        self.curve_boost = cfg.get("curve_boost", 1.0)
        self.detection_threshold = cfg.get("detection_threshold", 80)

        self.frame_count = 0

        self._prev_error = 0.0
        self._prev_diff = 0.0
        self._filtered_error = 0.0
        self._filtered_steering = 0.0

        self._lane_half_width = float(_LINE_OFFSET)

        self._left_history = deque(maxlen=3)
        self._right_history = deque(maxlen=3)

        self.last_debug_info = self._empty_debug_info(480, 640)

    def _calculate_error(self, yellow_xs, white_xs, left_det, right_det, w):
        if left_det and right_det and yellow_xs and white_xs:
            y_mean = float(np.median(yellow_xs))
            w_mean = float(np.median(white_xs))

            # If white is not actually to the right of yellow, ignore the bad pair.
            if w_mean <= y_mean + 50:
                error = self._prev_error * (w / 2.0)
            else:
                measured = (w_mean - y_mean) / 2.0

                # Update lane width only with believable values.
                if 60 < measured < 360:
                    self._lane_half_width = 0.90 * self._lane_half_width + 0.10 * measured

                lane_center = (y_mean + w_mean) / 2.0
                error = w / 2.0 - lane_center

        elif left_det and yellow_xs:
            y_mean = float(np.median(yellow_xs))
            lane_center = y_mean + self._lane_half_width
            error = w / 2.0 - lane_center

        elif right_det and white_xs:
            w_mean = float(np.median(white_xs))
            lane_center = w_mean - self._lane_half_width
            error = w / 2.0 - lane_center

        else:
            error = self._prev_error * (w / 2.0)

        return float(np.clip(error / (w / 2.0), -1.0, 1.0))

    def _calculate_steering(self, error: float) -> float:
        raw_diff = error - self._prev_error
        error_diff = 0.70 * self._prev_diff + 0.30 * raw_diff

        self._prev_diff = error_diff
        self._prev_error = error

        raw_steering = self.p_gain * error + self.d_gain * error_diff
        raw_steering = float(np.clip(raw_steering, -self.max_steer, self.max_steer))

        # Smooth steering to prevent sudden hard turns when white/yellow line flickers.
        max_delta = 0.035
        delta = np.clip(raw_steering - self._filtered_steering, -max_delta, max_delta)

        self._filtered_steering += delta
        self._filtered_steering = float(
            np.clip(self._filtered_steering, -self.max_steer, self.max_steer)
        )

        return self._filtered_steering

    def _motor_commands(self, steering: float, recovery: bool, is_curve: bool, both_visible: bool):
        if recovery:
            return 0.0, 0.0

        speed = self.curve_speed if is_curve else self.base_speed

        if not both_visible:
            speed *= 0.75

        # Slow down when steering is large.
        speed *= max(0.65, 1.0 - abs(steering) * 1.8)

        left = speed - steering
        right = speed + steering

        return float(np.clip(left, 0.0, 0.35)), float(np.clip(right, 0.0, 0.35))

    def _smooth(self, left, right, both_visible):
        buf = 3 if both_visible else 2

        if self._left_history.maxlen != buf:
            self._left_history = deque(maxlen=buf)
            self._right_history = deque(maxlen=buf)

        self._left_history.append(left)
        self._right_history.append(right)

        return (
            sum(self._left_history) / len(self._left_history),
            sum(self._right_history) / len(self._right_history),
        )

    def compute_commands(self, image: np.ndarray) -> Tuple[float, float]:
        self.frame_count += 1

        # Incoming image is RGB. Detection function expects BGR.
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            mask_left, mask_right = student.detect_lane_markings(bgr)
        except Exception as e:
            print(f"[Agent] detect_lane_markings error: {e}")
            return 0.0, 0.0

        mask_y = (mask_left * 255).astype(np.uint8)
        mask_w = (mask_right * 255).astype(np.uint8)

        yellow_pixels = int(np.count_nonzero(mask_y))
        white_pixels = int(np.count_nonzero(mask_w))
        total_pixels = yellow_pixels + white_pixels

        combined = np.clip(mask_left + mask_right, 0, 1)

        self.last_debug_info = {
            "roi": image,
            "lane_mask": (combined * 255).astype(np.uint8),
            "white_mask": mask_w,
            "yellow_mask": mask_y,
            "total_lane_pixels": total_pixels,
            "lateral_error": float(np.clip(self._prev_error, -1.0, 1.0)),
            "lane_detected": total_pixels >= self.detection_threshold,
            "frame_count": self.frame_count,
        }

        h, w = mask_y.shape

        yellow_xs, white_xs = detect_lines_in_slices(mask_y, mask_w, h)

        left_det = len(yellow_xs) > 0
        right_det = len(white_xs) > 0

        recovery = total_pixels < self.detection_threshold
        both_visible = left_det and right_det and not recovery

        is_curve, curve_dir = detect_curve(yellow_xs, white_xs, self.curve_threshold)

        raw_error = self._calculate_error(yellow_xs, white_xs, left_det, right_det, w)

        # Stronger filtering. Prevents jumping when mask flickers.
        self._filtered_error = 0.82 * self._filtered_error + 0.18 * raw_error

        steering = self._calculate_steering(self._filtered_error)

        left, right = self._motor_commands(steering, recovery, is_curve, both_visible)
        left, right = self._smooth(left, right, both_visible)

        self.last_debug_info.update({
            "yellow_xs": yellow_xs,
            "white_xs": white_xs,
            "slice_ys": [int(h * r) for r in _SLICE_Y_RATIOS],
            "is_curve": is_curve,
            "curve_dir": curve_dir,
        })

        return left, right

    def step(self, image: np.ndarray, wheels_driver) -> Tuple[float, float]:
        left, right = self.compute_commands(image)
        wheels_driver.set_wheels_speed(left, right)
        return left, right

    def get_debug_info(self, image: np.ndarray) -> dict:
        return self.last_debug_info

    def _empty_debug_info(self, h, w):
        return {
            "roi": np.zeros((h, w, 3), dtype=np.uint8),
            "lane_mask": np.zeros((h, w), dtype=np.uint8),
            "white_mask": np.zeros((h, w), dtype=np.uint8),
            "yellow_mask": np.zeros((h, w), dtype=np.uint8),
            "total_lane_pixels": 0,
            "lateral_error": 0.0,
            "lane_detected": False,
            "frame_count": 0,
        }
