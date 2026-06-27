import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RedLineState:
    red_line_close: bool
    lane_disabled: bool
    disabled_remaining: float
    red_ratio: float
    red_row_ratio: float


class RedLineGate:
    def __init__(
        self,
        disable_seconds: float = 5.0,
        roi_y_start: float = 0.62,
        roi_y_end: float = 0.95,
        min_red_ratio: float = 0.015,
        min_red_row_ratio: float = 0.18,
    ):
        self.disable_seconds = disable_seconds
        self.roi_y_start = roi_y_start
        self.roi_y_end = roi_y_end
        self.min_red_ratio = min_red_ratio
        self.min_red_row_ratio = min_red_row_ratio

        self._disabled_until = 0.0
        self.last_state = RedLineState(
            red_line_close=False,
            lane_disabled=False,
            disabled_remaining=0.0,
            red_ratio=0.0,
            red_row_ratio=0.0,
        )

    def update(self, frame_rgb: np.ndarray) -> RedLineState:
        now = time.monotonic()

        red_line_close, red_ratio, red_row_ratio = self._detect_close_red_line(frame_rgb)

        if red_line_close:
            self._disabled_until = now + self.disable_seconds

        remaining = max(0.0, self._disabled_until - now)
        lane_disabled = remaining > 0.0

        self.last_state = RedLineState(
            red_line_close=red_line_close,
            lane_disabled=lane_disabled,
            disabled_remaining=remaining,
            red_ratio=red_ratio,
            red_row_ratio=red_row_ratio,
        )

        return self.last_state

    def reset(self) -> None:
        self._disabled_until = 0.0
        self.last_state = RedLineState(
            red_line_close=False,
            lane_disabled=False,
            disabled_remaining=0.0,
            red_ratio=0.0,
            red_row_ratio=0.0,
        )

    def _detect_close_red_line(self, frame_rgb: np.ndarray):
        if frame_rgb is None:
            return False, 0.0, 0.0

        h, w = frame_rgb.shape[:2]

        y1 = int(h * self.roi_y_start)
        y2 = int(h * self.roi_y_end)

        # Ignore extreme left/right borders.
        x1 = int(w * 0.08)
        x2 = int(w * 0.92)

        roi = frame_rgb[y1:y2, x1:x2]

        if roi.size == 0:
            return False, 0.0, 0.0

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        lower_red_1 = np.array([0, 80, 60])
        upper_red_1 = np.array([12, 255, 255])

        lower_red_2 = np.array([165, 80, 60])
        upper_red_2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

        red_mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        red_pixels = int(np.count_nonzero(red_mask))
        total_pixels = red_mask.shape[0] * red_mask.shape[1]

        red_ratio = red_pixels / float(total_pixels) if total_pixels > 0 else 0.0

        row_counts = np.count_nonzero(red_mask, axis=1)
        max_row_count = int(np.max(row_counts)) if row_counts.size > 0 else 0
        red_row_ratio = max_row_count / float(red_mask.shape[1]) if red_mask.shape[1] > 0 else 0.0

        red_line_close = (
            red_ratio >= self.min_red_ratio
            and red_row_ratio >= self.min_red_row_ratio
        )

        return red_line_close, red_ratio, red_row_ratio