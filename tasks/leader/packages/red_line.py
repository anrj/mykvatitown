"""Red line detection — based on the finetuned traffic-lights team's code.

Standalone detection (no FSM). Returns (detected, ratio, approach_detected)
so the leader's FSM can trigger APPROACH early (taller ROI) and STOP close
(bottom ROI).
"""

import numpy as np
import cv2


_RED_HSV_LOW1  = np.array([0,   100, 70],  dtype=np.uint8)
_RED_HSV_HIGH1 = np.array([12,  255, 255], dtype=np.uint8)
_RED_HSV_LOW2  = np.array([168, 100, 70],  dtype=np.uint8)
_RED_HSV_HIGH2 = np.array([179, 255, 255], dtype=np.uint8)


def _detect_in_zone(frame_bgr, cfg, strip_frac, pixel_frac):
    """Check for a red stop line in the bottom `strip_frac` of the frame.

    Returns (detected: bool, pixel_ratio: float).
    """
    h, w = frame_bgr.shape[:2]

    strip_h = max(2, int(h * strip_frac))
    y0 = h - strip_h
    y1 = h

    l = int(w * cfg.get('red_roi_left', 0.12))
    r = int(w * cfg.get('red_roi_right', 0.90))

    strip = frame_bgr[y0:y1, l:r]

    if strip.size == 0:
        return False, 0.0

    strip_h_actual, strip_w_actual = strip.shape[:2]

    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, _RED_HSV_LOW1, _RED_HSV_HIGH1)
    mask2 = cv2.inRange(hsv, _RED_HSV_LOW2, _RED_HSV_HIGH2)
    mask = mask1 | mask2

    kernel_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    red_pixels = int(np.count_nonzero(mask))
    strip_area = strip_h_actual * strip_w_actual

    if strip_area <= 0:
        return False, 0.0

    pixel_ratio = red_pixels / float(strip_area)

    if pixel_ratio < pixel_frac:
        return False, pixel_ratio

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    close_ratio = cfg.get('red_line_close_y2_ratio', 0.3)

    for cnt in contours:
        area = float(cv2.contourArea(cnt))

        if area < cfg.get('red_min_area', 60):
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw <= 1 or bh <= 1:
            continue

        y2 = y + bh
        aspect = bw / float(bh + 1e-6)

        if aspect < 1.6:
            continue

        if bw < strip_w_actual * cfg.get('red_min_width_frac', 0.12):
            continue

        if y2 < strip_h_actual * close_ratio:
            continue

        score = area * aspect

        if best is None or score > best:
            best = score

    return best is not None, pixel_ratio


def detect_red_line(frame_bgr, cfg):
    """Check for a red stop line.

    Returns (detected, pixel_ratio, approach_detected):
      detected         — red line in the CLOSE zone (bottom red_strip_frac)
      approach_detected — red line in the APPROACH zone (bottom red_approach_frac)
                          — taller, more sensitive, fires earlier
    """
    close_frac = cfg.get('red_strip_frac', 0.32)
    approach_frac = cfg.get('red_approach_frac', 0.50)
    close_pixel_frac = cfg.get('red_pixel_frac', 0.012)
    approach_pixel_frac = cfg.get('red_approach_pixel_frac', 0.003)

    detected, ratio = _detect_in_zone(frame_bgr, cfg, close_frac, close_pixel_frac)

    if detected:
        return True, ratio, True

    approach_detected, approach_ratio = _detect_in_zone(
        frame_bgr, cfg, approach_frac, approach_pixel_frac)

    return False, approach_ratio, approach_detected
