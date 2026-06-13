"""
Sign / stop-line detection for the convoy leader.

Red stop lines are only checked BETWEEN the detected lane markings.

Lane marking support
--------------------
The original code only worked when a yellow (left) line was visible.
This version handles all four real-world cases:

  Case A  yellow-left + white-right   (standard lane, most common)
  Case B  white-left  + white-right   (highway, parking lot, inner lanes)
  Case C  yellow-left only            (right line occluded / off-frame)
  Case D  white-right only            (left line occluded / off-frame)

Detection priority for each side:
  LEFT  — yellow_xs first, then left_white_xs (if provided), then fallback
  RIGHT — white_xs first, then right_white_xs (if provided), then fallback

lane_info keys recognised
--------------------------
  yellow_xs       list[float]   x-coords of yellow (left) line, bottom-to-top
  white_xs        list[float]   x-coords of white right line, bottom-to-top
  left_white_xs   list[float]   x-coords of white LEFT line (Case B)
  right_white_xs  list[float]   alias for white_xs (accepted too)

The last element of each list is used (nearest to the vehicle).
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tasks.project_leader.packages.sign_detector import SignDetector

Detection = Tuple[Tuple[int, int, int, int], float, int]
SIGN_CLASS_ID = 2

# Fallback half-width when only one lane edge is visible (~lane width in px at 640).
_ASSUMED_LANE_HALF_PX = 130


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _red_mask(hsv: np.ndarray) -> np.ndarray:
    low1  = np.array([0,   90,  60], dtype=np.uint8)
    high1 = np.array([12, 255, 255], dtype=np.uint8)
    low2  = np.array([168, 90,  60], dtype=np.uint8)
    high2 = np.array([180, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv, low1, high1) | cv2.inRange(hsv, low2, high2)


# ---------------------------------------------------------------------------
# Lane-strip helpers
# ---------------------------------------------------------------------------

class LaneMode:
    """Resolved lane configuration for one frame."""
    __slots__ = ('left_x', 'right_x', 'left_src', 'right_src', 'reliable')

    def __init__(self, left_x, right_x, left_src, right_src, reliable):
        self.left_x   = left_x    # float px, or None
        self.right_x  = right_x   # float px, or None
        self.left_src = left_src  # 'yellow' | 'white' | 'fallback' | None
        self.right_src= right_src
        self.reliable = reliable  # True when at least one real line was found

    def __repr__(self):
        return (f"LaneMode(left={self.left_x}({self.left_src}), "
                f"right={self.right_x}({self.right_src}), "
                f"reliable={self.reliable})")


def _resolve_lane(lane_info: Optional[dict], w: int) -> LaneMode:
    """
    Determine left and right lane-edge x-coordinates from whatever
    line data is available.  Never raises; always returns a LaneMode.

    Priority
    --------
    Left edge:  yellow_xs  >  left_white_xs  >  fallback from right
    Right edge: white_xs   >  right_white_xs >  fallback from left
    """
    info = lane_info or {}

    def _last(key) -> Optional[float]:
        xs = info.get(key) or []
        return float(xs[-1]) if xs else None

    y_left   = _last('yellow_xs')
    w_right  = _last('white_xs') or _last('right_white_xs')
    w_left   = _last('left_white_xs')

    left_x   = y_left  if y_left  is not None else w_left
    right_x  = w_right

    left_src  = ('yellow' if y_left  is not None else
                 'white'  if w_left  is not None else None)
    right_src = ('white'  if w_right is not None else None)

    reliable  = (left_x is not None or right_x is not None)

    # Fallback: project the missing side using assumed lane width
    if left_x is None and right_x is not None:
        left_x   = right_x - 2 * _ASSUMED_LANE_HALF_PX
        left_src = 'fallback'
    if right_x is None and left_x is not None:
        right_x   = left_x + 2 * _ASSUMED_LANE_HALF_PX
        right_src = 'fallback'

    # Clamp to image bounds
    if left_x  is not None: left_x  = max(0.0, min(float(w - 1), left_x))
    if right_x is not None: right_x = max(0.0, min(float(w - 1), right_x))

    return LaneMode(left_x, right_x, left_src, right_src, reliable)


def _lane_strip_bounds(lane_info: Optional[dict], w: int,
                       margin_px: int = 10,
                       require_yellow: bool = False) -> Optional[Tuple[int, int]]:
    """
    Return (left_px, right_px) of the current lane with inward margin applied.

    Parameters
    ----------
    require_yellow : kept for backward-compatibility; when True the function
                     still returns None if no yellow line is found AND no white
                     left line is available — i.e. we have no left boundary at all.
                     Set to False (default) to allow white-left fallback.
    """
    mode = _resolve_lane(lane_info, w)

    if not mode.reliable:
        return None

    # Honour legacy require_yellow: only block if truly no left line
    if require_yellow and mode.left_src not in ('yellow', 'white'):
        return None  # left side is pure fallback — not trustworthy enough

    left  = int(mode.left_x)  + margin_px
    right = int(mode.right_x) - margin_px

    left  = max(0, left)
    right = min(w - 1, right)

    if right <= left + 8:
        return None
    return left, right


# ---------------------------------------------------------------------------
# Red-line / proximity helpers  (unchanged logic, updated to use new bounds)
# ---------------------------------------------------------------------------

def _red_stop_line_ratio(bgr: np.ndarray, lane_info: Optional[dict],
                         y0: float, y1: float, margin_px: int = 10) -> float:
    """Red painted line in the bottom band, only inside the current lane strip."""
    h, w = bgr.shape[:2]
    bounds = _lane_strip_bounds(lane_info, w, margin_px=margin_px)
    if bounds is None:
        return 0.0
    x_left, x_right = bounds
    roi = bgr[int(h * y0):int(h * y1), x_left:x_right]
    if roi.size == 0:
        return 0.0
    mask = _red_mask(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    return float(np.count_nonzero(mask)) / float(mask.size)


def _red_sign_proximity(bgr: np.ndarray, lane_info: Optional[dict],
                        roi_start: float = 0.30) -> Tuple[float, float, float]:
    h, w = bgr.shape[:2]
    bounds = _lane_strip_bounds(lane_info, w, margin_px=5)
    if bounds is None:
        return 0.0, 0.0, 0.0
    x_left, x_right = bounds
    y0  = int(h * roi_start)
    roi = bgr[y0:, x_left:x_right]
    if roi.size == 0:
        return 0.0, 0.0, 0.0

    mask   = _red_mask(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, 0.0

    x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if bw * bh < 30:
        return 0.0, 0.0, 0.0

    area_ratio   = float(bw * bh) / float(h * w)
    bottom_ratio = float(y0 + y + bh) / float(h)
    proximity    = bottom_ratio * (area_ratio ** 0.5) * 100.0
    return area_ratio, bottom_ratio, proximity


def _sign_danger(detections: List[Detection], orig_w: int, orig_h: int,
                 lane_info: Optional[dict]) -> float:
    bounds = _lane_strip_bounds(lane_info, orig_w, margin_px=0)
    if bounds is None:
        return 0.0
    x_left, x_right = bounds
    max_danger = 0.0
    for bbox, _score, cls_id in detections:
        if cls_id != SIGN_CLASS_ID:
            continue
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        if cx < x_left or cx > x_right:
            continue
        y2_ratio   = y2 / orig_h
        area_ratio = (x2 - x1) * (y2 - y1) / float(orig_w * orig_h)
        max_danger = max(max_danger, y2_ratio * area_ratio * 50.0)
    return max_danger


# ---------------------------------------------------------------------------
# SignBehavior  (public API unchanged; lane handling now multi-mode)
# ---------------------------------------------------------------------------

class SignBehavior:
    def __init__(self, cfg):
        s = cfg['signs']
        c = cfg['control']
        self._april         = SignDetector(cfg)
        self.use_detection  = bool(s.get('use_detection', True))
        self.slow_tag_px    = float(s.get('slow_tag_px', 18))
        self.stop_tag_px    = float(s.get('min_tag_px', 32))
        self.line_slow      = float(s.get('red_line_slow', 0.012))
        self.line_stop      = float(s.get('red_line_stop', 0.035))
        self.red_slow_area  = float(s.get('red_slow_area', 0.0018))
        self.red_stop_area  = float(s.get('red_stop_area', 0.005))
        self.red_slow_bottom= float(s.get('red_slow_bottom', 0.52))
        self.red_stop_bottom= float(s.get('red_stop_bottom', 0.62))
        self.lane_margin_px = int(s.get('lane_margin_px', 10))
        self.det_slow       = float(c.get('detection_slow_threshold', 0.06))
        self.det_stop       = float(c.get('detection_stop_threshold', 0.12))
        self._det_agent     = None
        self._det_error     = None
        if self.use_detection:
            self._try_load_detector()

    def _try_load_detector(self):
        model_path = os.path.normpath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..',
            'tasks', 'object_detection', 'models', 'best.onnx',
        ))
        if not os.path.isfile(model_path):
            self._det_error = 'best.onnx not found — AprilTag + red line only'
            return
        try:
            from tasks.object_detection.packages.agent import ObjectDetectionAgent
            self._det_agent = ObjectDetectionAgent()
            if not self._det_agent.model_loaded:
                self._det_error = self._det_agent.load_error
                self._det_agent = None
        except Exception as e:
            self._det_error = str(e)
            self._det_agent = None

    @property
    def detector_ready(self) -> bool:
        return self._det_agent is not None and self._det_agent.model_loaded

    def lane_mode(self, lane_info: Optional[dict], w: int) -> LaneMode:
        """Expose resolved lane configuration for HUD / debugging."""
        return _resolve_lane(lane_info, w)

    def lane_bounds(self, lane_info: Optional[dict], w: int) -> Optional[Tuple[int, int]]:
        """Pixel (left, right) of the current lane strip, or None."""
        return _lane_strip_bounds(lane_info, w, self.lane_margin_px)

    def red_line_ratio(self, bgr: np.ndarray,
                       lane_info: Optional[dict] = None) -> float:
        return _red_stop_line_ratio(bgr, lane_info, 0.68, 0.92, self.lane_margin_px)

    def assess(self, bgr, lane_info=None) -> Tuple[str, str, float]:
        h, w = bgr.shape[:2]
        lane_info = lane_info or {}

        # ── 1. AprilTag ─────────────────────────────────────────────────
        sign, tag_px = self._april.detect(bgr)
        if sign == 'stop':
            if tag_px >= self.stop_tag_px:
                return 'stop', 'apriltag', tag_px
            if tag_px >= self.slow_tag_px:
                return 'slow', 'apriltag', tag_px

        # ── 2. Red painted stop line ─────────────────────────────────────
        line_ratio = _red_stop_line_ratio(
            bgr, lane_info, 0.68, 0.92, self.lane_margin_px)
        if line_ratio >= self.line_stop:
            return 'stop', 'red_line', line_ratio
        if line_ratio >= self.line_slow:
            return 'slow', 'red_line', line_ratio

        # ── 3. Red sign approaching ──────────────────────────────────────
        area_ratio, bottom_ratio, proximity = _red_sign_proximity(bgr, lane_info)
        if bottom_ratio >= self.red_stop_bottom and area_ratio >= self.red_stop_area:
            return 'stop', 'red_sign', proximity
        if bottom_ratio >= self.red_slow_bottom and area_ratio >= self.red_slow_area:
            return 'slow', 'red_sign', proximity

        # ── 4. YOLO object detection ─────────────────────────────────────
        if self.detector_ready:
            rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dets   = self._det_agent.detect(rgb) or []
            danger = _sign_danger(dets, w, h, lane_info)
            if danger >= self.det_stop:
                return 'stop', 'yolo', danger
            if danger >= self.det_slow:
                return 'slow', 'yolo', danger

        return 'cruise', 'none', 0.0

    def sign_cleared(self, phase: str, strength: float) -> bool:
        return phase == 'cruise' or (phase == 'slow' and strength < self.line_slow * 0.5)