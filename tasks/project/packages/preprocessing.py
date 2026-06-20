"""Red stop line detection for convoying project."""

import numpy as np
import cv2


def detect_red_stop(image_bgr: np.ndarray, config: dict) -> int:
    """
    Detect red stop lines in the forward direction (center ROI only).
    
    Args:
        image_bgr: BGR image from camera
        config: red_stop config dict with HSV bounds and threshold
    
    Returns:
        Number of red pixels detected in forward ROI (0 if none)
    """
    if not config.get('enabled', True):
        return 0
    
    h, w = image_bgr.shape[:2]
    
    # Only check center lane (narrow ROI to ignore adjacent lanes)
    # Focus on just the robot's lane where it's actually traveling
    roi_left = int(w * 0.35)      # 35% from left (ignore left lane)
    roi_right = int(w * 0.65)     # 35% from right (ignore right lane)
    roi_top = int(h * 0.2)        # start from 20% down (catch lines very early)
    roi_bottom = h                 # to bottom of image
    
    roi = image_bgr[roi_top:roi_bottom, roi_left:roi_right]
    
    hsv_image = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Red hue wraps around at 180, so check both ranges
    lower1 = np.array([
        config.get('lower_h1', 0),
        config.get('lower_s', 120),
        config.get('lower_v', 60)
    ])
    upper1 = np.array([
        config.get('upper_h1', 10),
        config.get('upper_s', 255),
        config.get('upper_v', 255)
    ])
    
    lower2 = np.array([
        config.get('lower_h2', 170),
        config.get('lower_s', 120),
        config.get('lower_v', 60)
    ])
    upper2 = np.array([
        config.get('upper_h2', 179),
        config.get('upper_s', 255),
        config.get('upper_v', 255)
    ])
    
    red_mask1 = cv2.inRange(hsv_image, lower1, upper1)
    red_mask2 = cv2.inRange(hsv_image, lower2, upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    red_pixels = int(np.count_nonzero(red_mask))
    return red_pixels
