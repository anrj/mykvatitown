"""
Sign / stop-line detection for the convoy leader.

Red stop lines are checked in a bottom strip of the frame, optionally
restricted to the lane strip between detected lane markings.

Includes ``detect_red_line`` (adapted from another team's implementation).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from tasks.project_leader.packages.sign_detector import SignDetector

Phase = str  # 'cruise' | 'slow' | 'stop'


@dataclass
class SignConfig:
    red_strip_frac: float = 0.08
    red_close_strip_frac: float = 0.045
    red_pixel_frac: float = 0.035
    red_line_close: float = 0.065
    red_hsv_low1: Tuple[int, int, int] = (0, 100, 100)
    red_hsv_high1: Tuple[int, int, int] = (10, 255, 255)
    red_hsv_low2: Tuple[int, int, int] = (160, 100, 100)
    red_hsv_high2: Tuple[int, int, int] = (179, 255, 255)
    lane_margin_px: int = 12
    red_line_stop: float = 0.035
    red_line_slow: float = 0.012
    stop_tag_px: float = 55.0
    slow_tag_px: float = 38.0
    red_confirm_frames: int = 3
    red_clear_frames: int = 12
    red_center_frac: float = 0.22
    red_min_band_rows: float = 0.35
    red_near_over_far: float = 1.15
    require_lane_for_red: bool = True


@dataclass
class LaneMode:
    left_px: Optional[int]
    right_px: Optional[int]
    reliable: bool


def detect_red_line(
    sign_behavior: 'SignBehavior',
    frame_rgb: np.ndarray,
    left_px: Optional[int] = None,
    right_px: Optional[int] = None,
    lane_reliable: bool = True,
    at_stop_line: bool = False,
) -> Tuple[bool, float, float]:
    """
    Detect OUR stop line at an intersection.

    ``at_stop_line=True`` uses a tighter bottom ROI and higher threshold so we
    only stop when the line is directly under the bot — not when it is visible
  far ahead down the road.
    """
    h, w = frame_rgb.shape[:2]
    cfg = sign_behavior.cfg

    if cfg.require_lane_for_red and not lane_reliable:
        return False, 0.0, 0.0

    frac = cfg.red_close_strip_frac if at_stop_line else cfg.red_strip_frac
    strip_h = max(2, int(h * frac))

    if left_px is not None and right_px is not None:
        l, r = int(left_px), int(right_px)
    else:
        half = int(w * cfg.red_center_frac)
        l, r = w // 2 - half, w // 2 + half

    l = max(0, min(l, w - 2))
    r = max(l + 2, min(r, w))

    strip = frame_rgb[h - strip_h:, l:r]
    strip_w = r - l

    hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
    lo1 = np.array(cfg.red_hsv_low1, dtype=np.uint8)
    hi1 = np.array(cfg.red_hsv_high1, dtype=np.uint8)
    lo2 = np.array(cfg.red_hsv_low2, dtype=np.uint8)
    hi2 = np.array(cfg.red_hsv_high2, dtype=np.uint8)
    mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)

    mid = max(1, strip_h // 2)
    far_mask = mask[:mid, :]
    near_mask = mask[mid:, :]
    far_fraction = float(far_mask.sum()) / (255.0 * max(1, mid * strip_w))
    near_fraction = float(near_mask.sum()) / (255.0 * max(1, (strip_h - mid) * strip_w))
    near_threshold = (
        cfg.red_line_close if at_stop_line else cfg.red_pixel_frac
    )

    row_cov = np.mean(near_mask > 0, axis=1) if near_mask.size else np.array([])
    band_rows = float(np.sum(row_cov >= 0.35)) / max(1, len(row_cov))
    min_band = cfg.red_min_band_rows if at_stop_line else cfg.red_min_band_rows * 0.8

    detected = (
        near_fraction >= near_threshold
        and near_fraction >= far_fraction * cfg.red_near_over_far
        and band_rows >= min_band
    )
    if detected and at_stop_line:
        print(
            f'[SignBehavior] AT stop line — '
            f'near={near_fraction:.3f} far={far_fraction:.3f} '
            f'band={band_rows:.2f}'
        )
    return detected, near_fraction, far_fraction


def _resolve_lane(lane_info: dict, w: int) -> LaneMode:
    """Resolve left/right lane edges from lane debug info."""
    yellow_xs = [int(x) for x in (lane_info.get('yellow_xs') or []) if x >= 0]
    white_xs = [int(x) for x in (lane_info.get('white_xs') or []) if x >= 0]
    left_white = [int(x) for x in (lane_info.get('left_white_xs') or []) if x >= 0]

    left = yellow_xs[-1] if yellow_xs else (left_white[-1] if left_white else None)
    right = white_xs[-1] if white_xs else None

    if left is None and right is not None:
        left = max(0, right - int(w * 0.35))
    if right is None and left is not None:
        right = min(w - 1, left + int(w * 0.35))

    reliable = left is not None and right is not None and right > left + 20
    return LaneMode(left_px=left, right_px=right, reliable=reliable)


class SignBehavior:
    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        s = cfg.get('signs', cfg)
        self.cfg = SignConfig(
            red_strip_frac=float(s.get('red_strip_frac', 0.08)),
            red_pixel_frac=float(s.get('red_pixel_frac', s.get('red_line_stop', 0.035))),
            red_hsv_low1=tuple(s.get('red_hsv_low1', [0, 100, 100])),
            red_hsv_high1=tuple(s.get('red_hsv_high1', [10, 255, 255])),
            red_hsv_low2=tuple(s.get('red_hsv_low2', [160, 100, 100])),
            red_hsv_high2=tuple(s.get('red_hsv_high2', [179, 255, 255])),
            lane_margin_px=int(s.get('lane_margin_px', 12)),
            red_line_close=float(s.get('red_line_close', 0.065)),
            red_close_strip_frac=float(s.get('red_close_strip_frac', 0.045)),
            red_line_slow=float(s.get('red_line_slow', 0.012)),
            stop_tag_px=float(s.get('stop_tag_px', 55.0)),
            slow_tag_px=float(s.get('slow_tag_px', 38.0)),
            red_confirm_frames=int(s.get('red_confirm_frames', 3)),
            red_clear_frames=int(s.get('red_clear_frames', 12)),
            red_center_frac=float(s.get('red_center_frac', 0.22)),
            red_min_band_rows=float(s.get('red_min_band_rows', 0.35)),
            red_near_over_far=float(s.get('red_near_over_far', 1.15)),
            require_lane_for_red=bool(s.get('require_lane_for_red', True)),
        )
        self._tags = SignDetector(cfg)

    def lane_mode(self, lane_info: dict, w: int) -> LaneMode:
        return _resolve_lane(lane_info, w)

    def lane_bounds(self, lane_info: dict, w: int) -> Optional[Tuple[int, int]]:
        mode = self.lane_mode(lane_info, w)
        if mode.left_px is None or mode.right_px is None:
            return None
        margin = self.cfg.lane_margin_px
        return mode.left_px + margin, mode.right_px - margin

    def red_line_detected(
        self,
        frame_rgb: np.ndarray,
        lane_info: Optional[dict] = None,
        at_stop_line: bool = False,
    ) -> Tuple[bool, float, float]:
        h, w = frame_rgb.shape[:2]
        mode = self.lane_mode(lane_info or {}, w) if lane_info else LaneMode(None, None, False)
        bounds = self.lane_bounds(lane_info, w) if lane_info else None
        left_px, right_px = bounds if bounds else (None, None)
        return detect_red_line(
            self, frame_rgb, left_px, right_px,
            lane_reliable=mode.reliable,
            at_stop_line=at_stop_line,
        )

    def at_stop_line(self, frame_rgb: np.ndarray, lane_info: Optional[dict] = None) -> bool:
        hit, _, _ = self.red_line_detected(frame_rgb, lane_info, at_stop_line=True)
        return hit

    def assess(self, bgr: np.ndarray, lane_info: Optional[dict] = None) -> Tuple[Phase, str, float]:
        """
        Return (phase, source, strength).

        Priority: AprilTag stop → red line stop → AprilTag slow → cruise.
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        meaning, tag_px = self._tags.detect(gray)

        if meaning == 'stop' and tag_px >= self.cfg.stop_tag_px:
            return 'stop', 'apriltag', tag_px / 200.0
        red_hit, near_frac, _ = self.red_line_detected(rgb, lane_info, at_stop_line=True)
        if red_hit:
            return 'stop', 'red_line', near_frac
        if meaning == 'slow' and tag_px >= self.cfg.slow_tag_px:
            return 'slow', 'apriltag', tag_px / 200.0
        return 'cruise', 'none', 0.0

    def sign_cleared(self, phase: Phase, strength: float) -> bool:
        return phase == 'cruise' or strength < self.cfg.red_line_slow * 0.5
