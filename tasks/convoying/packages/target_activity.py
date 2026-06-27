from typing import List, Optional, Tuple
import math

from tasks.convoying.packages.follow_enum import (
    TargetInfo,
    FAR,
    GOOD,
    CLOSE,
    TOO_CLOSE,
    LOST,
)

Detection = Tuple[Tuple[int, int, int, int], float, int]


class TargetTracker:
    """
    Tracks the object that our robot should follow.

    Detection format:
        ((x1, y1, x2, y2), score, class_id)

    Current known classes:
        0 = duckie
        1 = truck
        2 = sign

    Default target classes:
        1 = truck
        0 = duckie fallback, because the model may classify vehicle-like objects differently

    It rejects signs by default.
    """

    def __init__(
            self,
            target_class_ids: Tuple[int, ...] = (1,),
            rejected_class_ids: Tuple[int, ...] = (2,),
            min_score: float = 0.20,
            min_area: int = 800,
            max_center_shift_ratio: float = 0.70,
    ):
        self.target_class_ids = target_class_ids
        self.rejected_class_ids = rejected_class_ids
        self.min_score = min_score
        self.min_area = min_area
        self.max_center_shift_ratio = max_center_shift_ratio

        self._has_target = False
        self._last_center_x: Optional[float] = None
        self._last_center_y: Optional[float] = None

    def reset(self) -> None:
        self._has_target = False
        self._last_center_x = None
        self._last_center_y = None

    def update(
        self,
        detections: Optional[List[Detection]],
        image_height: int,
        image_width: Optional[int] = None,
    ) -> TargetInfo:
        if image_width is None:
            image_width = image_height

        if detections is None or len(detections) == 0:
            return self._lost("target_lost_no_detections")

        valid_targets = [
            detection for detection in detections
            if self._is_valid_target(detection)
        ]

        if not valid_targets:
            return self._lost(self._build_no_target_reason(detections))

        if not self._has_target:
            selected = self._choose_initial_target(valid_targets)
            return self._build_target_info(
                detection=selected,
                image_height=image_height,
                reason="initial_target_selected",
            )

        selected = self._match_previous_target(
            detections=valid_targets,
            image_width=image_width,
            image_height=image_height,
        )

        if selected is None:
            return self._lost("previous_target_not_matched")

        return self._build_target_info(
            detection=selected,
            image_height=image_height,
            reason="target_tracked",
        )

    def _is_valid_target(self, detection: Detection) -> bool:
        bbox, score, class_id = detection

        if class_id in self.rejected_class_ids:
            return False

        if class_id not in self.target_class_ids:
            return False

        if score < self.min_score:
            return False

        if self._area(bbox) < self.min_area:
            return False

        return True

    def _choose_initial_target(self, detections: List[Detection]) -> Detection:
        # Closest object usually has biggest area and lowest bottom_y.
        return max(
            detections,
            key=lambda detection: (
                self._area(detection[0]),
                detection[0][3],
                detection[1],
            ),
        )

    def _match_previous_target(
        self,
        detections: List[Detection],
        image_width: int,
        image_height: int,
    ) -> Optional[Detection]:
        if self._last_center_x is None or self._last_center_y is None:
            return self._choose_initial_target(detections)

        best_detection = None
        best_distance = float("inf")

        for detection in detections:
            bbox, _, _ = detection
            center_x, center_y = self._center(bbox)

            distance = math.sqrt(
                (center_x - self._last_center_x) ** 2
                + (center_y - self._last_center_y) ** 2
            )

            if distance < best_distance:
                best_distance = distance
                best_detection = detection

        image_diagonal = math.sqrt(image_width ** 2 + image_height ** 2)
        max_allowed_shift = image_diagonal * self.max_center_shift_ratio

        if best_distance > max_allowed_shift:
            return None

        return best_detection

    def _build_target_info(
        self,
        detection: Detection,
        image_height: int,
        reason: str,
    ) -> TargetInfo:
        bbox, score, class_id = detection
        _, _, _, bottom_y = bbox

        center_x, center_y = self._center(bbox)
        area = self._area(bbox)
        distance_state = self._distance_state(bottom_y, image_height)

        self._has_target = True
        self._last_center_x = center_x
        self._last_center_y = center_y

        return TargetInfo(
            found=True,
            bbox=bbox,
            center_x=center_x,
            center_y=center_y,
            bottom_y=bottom_y,
            area=area,
            score=float(score),
            class_id=int(class_id),
            distance_state=distance_state,
            reason=reason,
        )

    def _lost(self, reason: str) -> TargetInfo:
        return TargetInfo(
            found=False,
            bbox=None,
            center_x=None,
            center_y=None,
            bottom_y=None,
            area=0,
            score=0.0,
            class_id=None,
            distance_state=LOST,
            reason=reason,
        )

    def _build_no_target_reason(self, detections: List[Detection]) -> str:
        class_counts = {}

        for _, _, class_id in detections:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

        return f"no_valid_target_detection_classes={class_counts}"

    @staticmethod
    def _center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _area(bbox: Tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = bbox
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        return width * height

    @staticmethod
    def _distance_state(bottom_y: int, image_height: int) -> str:
        if image_height <= 0:
            return LOST

        bottom_ratio = bottom_y / float(image_height)

        # Stop only when the truck is really close.
        if bottom_ratio >= 0.65:
            return TOO_CLOSE

        # Slow down when close, but not too early.
        if bottom_ratio >= 0.52:
            return CLOSE

        # Normal following distance.
        if bottom_ratio >= 0.34:
            return GOOD

        return FAR