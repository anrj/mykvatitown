"""AprilTag traffic-sign detection (tag36h11) — leader bot only."""

import cv2


class SignDetector:
    def __init__(self, cfg):
        self.enabled = bool(cfg['signs']['enabled'])
        self.min_px = float(cfg['signs']['min_tag_px'])
        self.meanings = {int(k): str(v).lower()
                         for k, v in cfg['signs']['tag_meanings'].items()}
        self._detector = None
        if self.enabled:
            try:
                d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
                self._detector = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
            except Exception as e:
                print(f'[sign_detector] AprilTag unavailable ({e}); signs disabled')
                self.enabled = False

    def detect(self, bgr):
        """Return (meaning_or_None, tag_px). Closest relevant tag wins."""
        if not self.enabled or self._detector is None:
            return None, 0.0
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return None, 0.0

        best_meaning, best_px = None, 0.0
        for c, i in zip(corners, ids.flatten()):
            meaning = self.meanings.get(int(i))
            if meaning is None:
                continue
            quad = c.reshape(-1, 2)
            size_px = float(cv2.norm(quad[0], quad[2]))  # diagonal
            if size_px < self.min_px:
                continue
            if size_px > best_px:
                best_meaning, best_px = meaning, size_px
        return best_meaning, best_px
