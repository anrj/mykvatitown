from typing import Tuple
import os
import numpy as np
import cv2
import yaml

_HSV_FILE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_hsv_config.yaml')
try:
    with open(_HSV_FILE) as _f:
        _h = yaml.safe_load(_f) or {}
except FileNotFoundError:
    _h = {}

_yellow_lower = np.array([_h.get('yellow_lower_h', 0),  _h.get('yellow_lower_s', 0),  _h.get('yellow_lower_v', 0)])
_yellow_upper = np.array([_h.get('yellow_upper_h', 0),  _h.get('yellow_upper_s', 0), _h.get('yellow_upper_v', 0)])

_white_lower = np.array([_h.get('white_lower_h', 0),   _h.get('white_lower_s', 0), _h.get('white_lower_v', 0)])
_white_upper = np.array([_h.get('white_upper_h', 0), _h.get('white_upper_s', 0), _h.get('white_upper_v', 0)])

# Red stop line detection (wraps around H=180)
_red_lower_h1 = int(_h.get('red_lower_h1', 0))
_red_upper_h1 = int(_h.get('red_upper_h1', 10))
_red_lower_h2 = int(_h.get('red_lower_h2', 170))
_red_upper_h2 = int(_h.get('red_upper_h2', 179))
_red_lower_s = int(_h.get('red_lower_s', 80))
_red_upper_s = int(_h.get('red_upper_s', 255))
_red_lower_v = int(_h.get('red_lower_v', 50))
_red_upper_v = int(_h.get('red_upper_v', 255))

def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    yellow_mask = cv2.inRange(hsv_image, _yellow_lower, _yellow_upper)
    white_mask = cv2.inRange(hsv_image, _white_lower, _white_upper)
    
    return yellow_mask, white_mask

def detect_red_line(image: np.ndarray) -> np.ndarray:
    """Detect red stop line. Red wraps around HSV hue, so check both ranges."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Red occupies two ranges due to hue wrapping at 180
    lower1 = np.array([_red_lower_h1, _red_lower_s, _red_lower_v])
    upper1 = np.array([_red_upper_h1, _red_upper_s, _red_upper_v])
    lower2 = np.array([_red_lower_h2, _red_lower_s, _red_lower_v])
    upper2 = np.array([_red_upper_h2, _red_upper_s, _red_upper_v])
    
    red_mask1 = cv2.inRange(hsv_image, lower1, upper1)
    red_mask2 = cv2.inRange(hsv_image, lower2, upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    return red_mask

def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper, red_lower_h1=None, red_upper_h1=None, 
                    red_lower_h2=None, red_upper_h2=None, red_lower_s=None, red_upper_s=None, 
                    red_lower_v=None, red_upper_v=None):
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper
    global _red_lower_h1, _red_upper_h1, _red_lower_h2, _red_upper_h2
    global _red_lower_s, _red_upper_s, _red_lower_v, _red_upper_v
    
    _yellow_lower    = np.array(yellow_lower)
    _yellow_upper    = np.array(yellow_upper)
    _white_lower = np.array(white_lower)
    _white_upper = np.array(white_upper)
    
    if red_lower_h1 is not None:
        _red_lower_h1 = int(red_lower_h1)
    if red_upper_h1 is not None:
        _red_upper_h1 = int(red_upper_h1)
    if red_lower_h2 is not None:
        _red_lower_h2 = int(red_lower_h2)
    if red_upper_h2 is not None:
        _red_upper_h2 = int(red_upper_h2)
    if red_lower_s is not None:
        _red_lower_s = int(red_lower_s)
    if red_upper_s is not None:
        _red_upper_s = int(red_upper_s)
    if red_lower_v is not None:
        _red_lower_v = int(red_lower_v)
    if red_upper_v is not None:
        _red_upper_v = int(red_upper_v)

def get_hsv_bounds():
    return {
        'yellow_lower_h': int(_yellow_lower[0]),    'yellow_upper_h': int(_yellow_upper[0]),
        'yellow_lower_s': int(_yellow_lower[1]),    'yellow_upper_s': int(_yellow_upper[1]),
        'yellow_lower_v': int(_yellow_lower[2]),    'yellow_upper_v': int(_yellow_upper[2]),
        'white_lower_h':  int(_white_lower[0]), 'white_upper_h':  int(_white_upper[0]),
        'white_lower_s':  int(_white_lower[1]), 'white_upper_s':  int(_white_upper[1]),
        'white_lower_v':  int(_white_lower[2]), 'white_upper_v':  int(_white_upper[2]),
        'red_lower_h1':   _red_lower_h1,         'red_upper_h1':   _red_upper_h1,
        'red_lower_h2':   _red_lower_h2,         'red_upper_h2':   _red_upper_h2,
        'red_lower_s':    _red_lower_s,          'red_upper_s':    _red_upper_s,
        'red_lower_v':    _red_lower_v,          'red_upper_v':    _red_upper_v,
    }