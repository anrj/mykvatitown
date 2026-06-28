from typing import Tuple
import os

import cv2
import numpy as np
import yaml


HSV_FILE = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "config",
        "lane_servoing_hsv_config.yaml",
    )
)

try:
    with open(HSV_FILE, "r") as f:
        _h = yaml.safe_load(f) or {}
except FileNotFoundError:
    _h = {}


_yellow_lower = np.array([
    _h.get("yellow_lower_h", 18),
    _h.get("yellow_lower_s", 70),
    _h.get("yellow_lower_v", 70),
], dtype=np.uint8)

_yellow_upper = np.array([
    _h.get("yellow_upper_h", 42),
    _h.get("yellow_upper_s", 255),
    _h.get("yellow_upper_v", 255),
], dtype=np.uint8)

_white_lower = np.array([
    _h.get("white_lower_h", 0),
    _h.get("white_lower_s", 0),
    _h.get("white_lower_v", 205),
], dtype=np.uint8)

_white_upper = np.array([
    _h.get("white_upper_h", 179),
    _h.get("white_upper_s", 42),
    _h.get("white_upper_v", 255),
], dtype=np.uint8)


_ROI_START = 0.42

# Hide left white line. Use 0.42 because on curves the right white line can move toward center.
_IGNORE_WHITE_LEFT_UNTIL = 0.42

# White filtering.
# Gray floor should fail because it is not bright enough / not neutral enough.
_WHITE_MIN_BGR = 145
_WHITE_MAX_CHANNEL_DIFF = 70


def _clean_mask(mask: np.ndarray, open_size: int = 3, close_size: int = 5) -> np.ndarray:
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))

    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k_close)

    return clean


def _filter_white_components(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    output = np.zeros_like(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 70:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw <= 2 or bh <= 2:
            continue

        x2 = x + bw
        y2 = y + bh
        cx = x + bw / 2.0

        bbox_area = bw * bh
        fill_ratio = area / float(bbox_area + 1e-6)

        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]

        if rw <= 1 or rh <= 1:
            long_aspect = max(bw, bh) / float(min(bw, bh) + 1e-6)
        else:
            long_aspect = max(rw, rh) / float(min(rw, rh) + 1e-6)

        # Ignore far/top noise.
        if y2 < h * 0.44:
            continue

        # Ignore left white line.
        # Component must be mostly on the right/front side.
        if cx < w * _IGNORE_WHITE_LEFT_UNTIL and x2 < w * 0.58:
            continue

        # Reject huge filled bright blobs from floor/reflection.
        if bw > w * 0.38 and bh > h * 0.16 and fill_ratio > 0.55:
            continue

        # White road line should be elongated or tall.
        if long_aspect < 1.35 and bh < h * 0.16:
            continue

        # Reject tiny short pieces.
        if bh < h * 0.035 and bw < w * 0.08:
            continue

        cv2.drawContours(output, [cnt], -1, 255, thickness=cv2.FILLED)

    return output


def _filter_yellow_components(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    output = np.zeros_like(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 25:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw <= 2 or bh <= 2:
            continue

        y2 = y + bh

        if y2 < h * 0.35:
            continue

        # Reject very large yellow blobs that are probably not lane dashes.
        if bw > w * 0.35 and bh > h * 0.20:
            continue

        cv2.drawContours(output, [cnt], -1, 255, thickness=cv2.FILLED)

    return output


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]

    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[int(h * _ROI_START):, :] = 255

    raw_yellow = cv2.inRange(imghsv, _yellow_lower, _yellow_upper)
    raw_white = cv2.inRange(imghsv, _white_lower, _white_upper)

    raw_yellow = cv2.bitwise_and(raw_yellow, roi_mask)
    raw_white = cv2.bitwise_and(raw_white, roi_mask)

    # Remove left-side white line before morphology.
    raw_white[:, :int(w * _IGNORE_WHITE_LEFT_UNTIL)] = 0

    # Extra white check:
    # True white tape is bright and has similar B/G/R values.
    min_channel = np.min(image, axis=2)
    max_channel = np.max(image, axis=2)
    channel_diff = max_channel - min_channel

    bright_neutral = (
        (min_channel >= _WHITE_MIN_BGR)
        & (channel_diff <= _WHITE_MAX_CHANNEL_DIFF)
    ).astype(np.uint8) * 255

    raw_white = cv2.bitwise_and(raw_white, bright_neutral)

    clean_yellow = _clean_mask(raw_yellow, open_size=3, close_size=5)
    clean_white = _clean_mask(raw_white, open_size=3, close_size=7)

    clean_yellow = _filter_yellow_components(clean_yellow, h, w)
    clean_white = _filter_white_components(clean_white, h, w)

    return (
        (clean_yellow > 0).astype(np.float32),
        (clean_white > 0).astype(np.float32),
    )


def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper):
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper

    _yellow_lower = np.array(yellow_lower, dtype=np.uint8)
    _yellow_upper = np.array(yellow_upper, dtype=np.uint8)
    _white_lower = np.array(white_lower, dtype=np.uint8)
    _white_upper = np.array(white_upper, dtype=np.uint8)


def get_hsv_bounds():
    return {
        "yellow_lower_h": int(_yellow_lower[0]),
        "yellow_upper_h": int(_yellow_upper[0]),
        "yellow_lower_s": int(_yellow_lower[1]),
        "yellow_upper_s": int(_yellow_upper[1]),
        "yellow_lower_v": int(_yellow_lower[2]),
        "yellow_upper_v": int(_yellow_upper[2]),

        "white_lower_h": int(_white_lower[0]),
        "white_upper_h": int(_white_upper[0]),
        "white_lower_s": int(_white_lower[1]),
        "white_upper_s": int(_white_upper[1]),
        "white_lower_v": int(_white_lower[2]),
        "white_upper_v": int(_white_upper[2]),
    }
