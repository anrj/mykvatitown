"""AprilTag traffic-sign detection for the convoy leader."""

from typing import Optional, Tuple

import cv2


class SignDetector:
    def __init__(self, cfg: dict):
        s = cfg.get('signs', cfg)
        self.min_px = float(s.get('min_tag_px', 38))
        meanings = s.get('tag_meanings', {0: 'stop', 1: 'slow'})
        self.meanings = {int(k): str(v).lower() for k, v in meanings.items()}
        self.last_tag_ids = []
        self._detector = None
        try:
            d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            self._detector = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        except Exception as e:
            print(f'[SignDetector] AprilTag unavailable ({e})')

    def detect(self, gray) -> Tuple[Optional[str], float]:
        """Return (meaning_or_None, tag_px). Closest relevant tag wins."""
        self.last_tag_ids = []
        if self._detector is None:
            return None, 0.0
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return None, 0.0

        self.last_tag_ids = [int(i) for i in ids.flatten()]
        best_meaning, best_px = None, 0.0
        for c, i in zip(corners, ids.flatten()):
            meaning = self.meanings.get(int(i))
            if meaning is None:
                continue
            quad = c.reshape(-1, 2)
            size_px = float(cv2.norm(quad[0], quad[2]))
            if size_px < self.min_px:
                continue
            if size_px > best_px:
                best_meaning, best_px = meaning, size_px
        return best_meaning, best_px
