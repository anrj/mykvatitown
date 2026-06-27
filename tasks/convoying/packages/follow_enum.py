from dataclasses import dataclass
from typing import Optional, Tuple

BBox = Tuple[int, int, int, int]

FAR = "FAR"
GOOD = "GOOD"
CLOSE = "CLOSE"
TOO_CLOSE = "TOO_CLOSE"
LOST = "LOST"


@dataclass
class TargetInfo:
    found: bool
    bbox: Optional[BBox]
    center_x: Optional[float]
    center_y: Optional[float]
    bottom_y: Optional[int]
    area: int
    score: float
    class_id: Optional[int]
    distance_state: str
    reason: str


@dataclass
class ConvoyCommand:
    should_move: bool
    left_speed: float
    right_speed: float
    speed_multiplier: float
    reason: str