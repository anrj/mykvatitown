"""
Sign approach behavior — slow down, stop, then go.

Combines:
  * AprilTag (SignDetector)
  * Red stop-sign colour (HSV, no model required)
  * YOLO sign detections (object_detection) when best.onnx is available
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from tasks.project_leader.packages.sign_detector import SignDetector

Detection = Tuple[Tuple[int, int, int, int], float, int]

SIGN_CLASS_ID = 2
SLOW_DANGER = 0.06
STOP_DANGER = 0.15
CLEAR_DANGER = 0.04


def _lane_info_defaults():
    return {'yellow_xs': [], 'white_xs': []}


def _in_lane(cx: float, lane_info: dict, orig_w: int) -> bool:
    yellow_xs = lane_info.get('yellow_xs', [])
    white_xs = lane_info.get('white_xs', [])
    margin = 30
    lane_w = 250
    left = min(yellow_xs) if yellow_xs else None
    right = max(white_xs) if white_xs else None
    if left is not None and right is not None:
        return left - margin < cx < right + margin
    if left is not None:
        return left - margin < cx < left + lane_w
    if right is not None:
        return right - lane_w < cx < right + margin
    return True


def _sign_danger(detections: List[Detection], orig_w: int, orig_h: int,
                 lane_info: Optional[dict]) -> float:
    """Danger score for sign class only (object_detection stop_activity logic)."""
    lane_info = lane_info or _lane_info_defaults()
    max_danger = 0.0
    for bbox, _score, cls_id in detections:
        if cls_id != SIGN_CLASS_ID:
            continue
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        if not _in_lane(cx, lane_info, orig_w):
            continue
        cx_norm = cx / orig_w
        y2_ratio = y2 / orig_h
        margin = 0.25 - 0.15 * y2_ratio
        if cx_norm < margin or cx_norm > (1 - margin):
            continue
        area_ratio = (x2 - x1) * (y2 - y1) / float(orig_w * orig_h)
        danger = y2_ratio * area_ratio * 50.0
        max_danger = max(max_danger, danger)
    return max_danger


def _red_sign_proximity(bgr: np.ndarray, roi_start: float = 0.35
                        ) -> Tuple[float, float, float]:
    """
    Proximity from the largest red blob (close sign = big + low in frame).

    Returns (area_ratio, bottom_ratio, proximity):
      area_ratio   — bbox area / full frame (bigger when nearer)
      bottom_ratio — blob bottom y / frame height (higher when nearer)
      proximity    — combined score used for debounce / debug
    """
    h, w = bgr.shape[:2]
    y0 = int(h * roi_start)
    roi = bgr[y0:, :]
    if roi.size == 0:
        return 0.0, 0.0, 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    low1 = np.array([0, 100, 70], dtype=np.uint8)
    high1 = np.array([10, 255, 255], dtype=np.uint8)
    low2 = np.array([170, 100, 70], dtype=np.uint8)
    high2 = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, low1, high1) | cv2.inRange(hsv, low2, high2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, 0.0

    x, y, bw, bh = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if bw * bh < 40:
        return 0.0, 0.0, 0.0

    cx = x + bw / 2.0
    if cx < 0.28 * w or cx > 0.72 * w:
        return 0.0, 0.0, 0.0

    area_ratio = float(bw * bh) / float(h * w)
    bottom_ratio = float(y0 + y + bh) / float(h)
    proximity = bottom_ratio * (area_ratio ** 0.5) * 100.0
    return area_ratio, bottom_ratio, proximity


class SignBehavior:
    def __init__(self, cfg):
        s = cfg['signs']
        c = cfg['control']
        self._april = SignDetector(cfg)
        self.use_detection = bool(s.get('use_detection', True))
        self.slow_tag_px = float(s.get('slow_tag_px', 22))
        self.stop_tag_px = float(s.get('min_tag_px', 42))
        # Red sign must be large AND low in frame (near), not a tiny speck far away.
        self.red_slow_area = float(s.get('red_slow_area', 0.0025))
        self.red_stop_area = float(s.get('red_stop_area', 0.009))
        self.red_slow_bottom = float(s.get('red_slow_bottom', 0.58))
        self.red_stop_bottom = float(s.get('red_stop_bottom', 0.72))
        self.det_slow = float(c.get('detection_slow_threshold', SLOW_DANGER))
        self.det_stop = float(c.get('detection_stop_threshold', STOP_DANGER))
        self._det_agent = None
        self._det_error = None
        if self.use_detection:
            self._try_load_detector()

    def _try_load_detector(self):
        model_path = os.path.normpath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..',
            'tasks', 'object_detection', 'models', 'best.onnx',
        ))
        if not os.path.isfile(model_path):
            self._det_error = 'best.onnx not found — using AprilTag + red colour only'
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

    def detect_objects(self, rgb: np.ndarray) -> List[Detection]:
        if not self.detector_ready:
            return []
        result = self._det_agent.detect(rgb)
        return result or []

    def assess(self, bgr, lane_info=None) -> Tuple[str, str, float]:
        """
        Return (phase, source, strength).
        phase: 'cruise' | 'slow' | 'stop'
        """
        h, w = bgr.shape[:2]
        lane_info = lane_info or _lane_info_defaults()

        sign, tag_px = self._april.detect(bgr)
        if sign == 'stop':
            if tag_px >= self.stop_tag_px:
                return 'stop', 'apriltag', tag_px
            if tag_px >= self.slow_tag_px:
                return 'slow', 'apriltag', tag_px

        area_ratio, bottom_ratio, proximity = _red_sign_proximity(bgr)
        if (bottom_ratio >= self.red_stop_bottom
                and area_ratio >= self.red_stop_area):
            return 'stop', 'red', proximity
        if (bottom_ratio >= self.red_slow_bottom
                and area_ratio >= self.red_slow_area):
            return 'slow', 'red', proximity

        if self.detector_ready:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dets = self.detect_objects(rgb)
            danger = _sign_danger(dets, w, h, lane_info)
            if danger >= self.det_stop:
                return 'stop', 'yolo', danger
            if danger >= self.det_slow:
                return 'slow', 'yolo', danger

        return 'cruise', 'none', 0.0

    def sign_cleared(self, phase: str, strength: float) -> bool:
        """True when the sign is no longer in slow/stop range (safe to re-arm)."""
        if phase == 'cruise':
            return True
        if phase == 'slow':
            return False
        # phase == 'stop' — treat as cleared once strength drops (driving past)
        if strength < self.red_slow_area * 50.0:
            return True
        return False
