"""
LeaderLaneAgent — lane following for the convoy leader.

Thin extension of the shared ``LaneServoingAgent`` (from visual_lane_servoing)
that adds the few helpers the leader's stop-line logic needs:

  * ``reset_steering_state()``        — clear the PID / smoothing history
  * ``bottom_line_visible(...)``      — is a given lane edge still painted at
                                        the bottom of the frame? (stop trigger)
  * ``compute_single_line_commands``  — follow only one edge (used while the
                                        trigger edge is missing near a stop line)

The base agent already follows whichever edge is visible, so single-line follow
is just a biased variant of the normal controller.
"""

import cv2
import numpy as np

from tasks.visual_lane_servoing.packages import visual_servoing_activity as student
from tasks.visual_lane_servoing.packages.agent import (
    LaneServoingAgent,
    detect_lines_in_slices,
    _NUM_SLICES,
    _ROI_START,
)

# Minimum lit pixels in the bottom band before we call a line "visible".
_BOTTOM_MIN_PX = 12


def _norm_side(side) -> str:
    """Normalise a side spec to 'left' or 'right' (default 'left')."""
    s = str(side or '').strip().lower()
    if s in ('right', 'r', 'white', 'solid'):
        return 'right'
    return 'left'


class LeaderLaneAgent(LaneServoingAgent):
    """LaneServoingAgent + convoy-leader stop-line helpers."""

    def reset_steering_state(self) -> None:
        self._prev_error = 0.0
        self._filtered_error = 0.0
        self._left_history.clear()
        self._right_history.clear()

    def bottom_line_visible(self, lane_info, side, bottom_px) -> bool:
        """True when the requested edge has paint in the bottom band of the frame.

        side 'left'  -> yellow mask, side 'right' -> white mask.
        """
        info = lane_info or {}
        side = _norm_side(side)
        mask = info.get('yellow_mask') if side == 'left' else info.get('white_mask')
        if mask is None or getattr(mask, 'size', 0) == 0:
            return False
        h = mask.shape[0]
        band = mask[max(0, h - int(bottom_px)):, :]
        return int(np.count_nonzero(band)) >= _BOTTOM_MIN_PX

    def compute_single_line_commands(self, image, follow_side):
        """Servo on a single lane edge using the learned half-lane width."""
        side = _norm_side(follow_side)
        self.frame_count += 1
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            mask_left, mask_right = student.detect_lane_markings(bgr)
        except Exception as e:
            print(f"[LeaderLane] detect_lane_markings error: {e}")
            return 0.0, 0.0

        mask_y = (mask_left * 255).astype(np.uint8)
        mask_w = (mask_right * 255).astype(np.uint8)
        h, w = mask_y.shape

        yellow_xs, white_xs = detect_lines_in_slices(mask_y, mask_w, h)

        if side == 'right' and white_xs:
            error = w / 2.0 - (float(np.mean(white_xs)) - self._lane_half_width)
        elif side == 'left' and yellow_xs:
            error = w / 2.0 - (float(np.mean(yellow_xs)) + self._lane_half_width)
        else:
            error = self._prev_error

        raw_error = float(np.clip(error / (w / 2.0), -1.0, 1.0))
        self._filtered_error = 0.7 * self._filtered_error + 0.3 * raw_error
        steering = self._calculate_steering(self._filtered_error)

        speed = self.base_speed
        left = float(np.clip(speed - steering, 0.0, 1.0))
        right = float(np.clip(speed + steering, 0.0, 1.0))

        slice_h = int(h * 0.35 / _NUM_SLICES)
        start_y = int(h * _ROI_START)
        total = int(np.count_nonzero(mask_y)) + int(np.count_nonzero(mask_w))
        self.last_debug_info = {
            'roi':               image,
            'white_mask':        mask_w,
            'yellow_mask':       mask_y,
            'total_lane_pixels': total,
            'lateral_error':     raw_error,
            'lane_detected':     total >= self.detection_threshold,
            'frame_count':       self.frame_count,
            'yellow_xs':         yellow_xs,
            'white_xs':          white_xs,
            'slice_ys':          [start_y + i * slice_h + slice_h // 2
                                  for i in range(_NUM_SLICES)],
            'follow_line':       side,
        }
        return left, right
