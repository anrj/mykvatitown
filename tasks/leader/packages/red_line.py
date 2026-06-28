"""Red line detection — ported from Duckietown's finetuned RedLineGate values.

Detection uses two gates (both must pass):
  1. min_red_ratio        — overall red-pixel fraction in the ROI >= 0.015
  2. min_red_row_ratio    — max horizontal projection of red pixels in any
                            row >= 0.18 of the ROI width (a real stop line is
                            horizontally wide, a red blob is not)

ROI is the bottom strip of the frame (y from 1-roi_y_end up to 1-roi_y_start),
ignoring the left/right borders (x from roi_x_left to roi_x_right).

After detection, callers should disable lane-following for `disable_seconds`
(=5.0 s) so the convoy controller follows only the leader through the
intersection.
"""

import numpy as np
import cv2


# Duckietown finetuned red HSV thresholds (proven on the real bot).
_RED_HSV_LOW1  = np.array([0,   80, 60],  dtype=np.uint8)
_RED_HSV_HIGH1 = np.array([12,  255, 255], dtype=np.uint8)
_RED_HSV_LOW2  = np.array([165, 80, 60],  dtype=np.uint8)
_RED_HSV_HIGH2 = np.array([180, 255, 255], dtype=np.uint8)

# Duckietown defaults — overridable via cfg.
_DEFAULT_ROI_Y_START      = 0.62     # strip top (fraction of frame height from bottom)
_DEFAULT_ROI_Y_END        = 0.95     # strip bottom
_DEFAULT_ROI_X_LEFT       = 0.08     # left border cutoff
_DEFAULT_ROI_X_RIGHT      = 0.92     # right border cutoff
_DEFAULT_MIN_RED_RATIO    = 0.015
_DEFAULT_MIN_RED_ROW      = 0.18
_DEFAULT_DISABLE_SECONDS  = 5.0


def disable_seconds(cfg):
    """How long lane-following should be disabled after a red line is seen."""
    return float(cfg.get('disable_seconds', _DEFAULT_DISABLE_SECONDS))


def _detect_in_zone(frame_bgr, cfg, strip_frac, pixel_frac):
    """Check for a red stop line in the bottom `strip_frac` of the frame.

    Returns (detected: bool, pixel_ratio: float).
    """
    h, w = frame_bgr.shape[:2]

    strip_h = max(2, int(h * strip_frac))
    y0 = h - strip_h
    y1 = h

    # Border cutoff (Duckietown: ignore outer 8% on each side).
    l = int(w * cfg.get('red_roi_left', _DEFAULT_ROI_X_LEFT))
    r = int(w * cfg.get('red_roi_right', _DEFAULT_ROI_X_RIGHT))

    strip = frame_bgr[y0:y1, l:r]

    if strip.size == 0:
        return False, 0.0

    strip_h_actual, strip_w_actual = strip.shape[:2]

    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, _RED_HSV_LOW1, _RED_HSV_HIGH1)
    mask2 = cv2.inRange(hsv, _RED_HSV_LOW2, _RED_HSV_HIGH2)
    mask = mask1 | mask2

    # 5x5 open+close (Duckietown) — removes noise and bridges the line.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    red_pixels = int(np.count_nonzero(mask))
    strip_area = strip_h_actual * strip_w_actual

    if strip_area <= 0:
        return False, 0.0

    pixel_ratio = red_pixels / float(strip_area)

    if pixel_ratio < pixel_frac:
        return False, pixel_ratio

    # Row-ratio gate (Duckietown min_red_row_ratio): a real stop line must
    # span a wide fraction of at least one ROI row. This rejects red blobs.
    min_row_ratio = float(cfg.get('red_min_row_ratio', _DEFAULT_MIN_RED_ROW))
    row_counts = np.count_nonzero(mask, axis=1)
    max_row_count = int(np.max(row_counts)) if row_counts.size > 0 else 0
    red_row_ratio = max_row_count / float(strip_w_actual) if strip_w_actual > 0 else 0.0
    if red_row_ratio < min_row_ratio:
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
    close_pixel_frac = cfg.get('red_pixel_frac', _DEFAULT_MIN_RED_RATIO)
    approach_pixel_frac = cfg.get('red_approach_pixel_frac', _DEFAULT_MIN_RED_RATIO * 0.5)

    detected, ratio = _detect_in_zone(frame_bgr, cfg, close_frac, close_pixel_frac)

    if detected:
        return True, ratio, True

    approach_detected, approach_ratio = _detect_in_zone(
        frame_bgr, cfg, approach_frac, approach_pixel_frac)

    return False, approach_ratio, approach_detected
