from typing import Tuple

MODEL_PATH = "tasks/object_detection/models/best.onnx"


def NUMBER_FRAMES_SKIPPED() -> int:
    return 1


def filter_by_classes(pred_class: int) -> bool:
    return True


def filter_by_scores(score: float) -> bool:
    return score >= 0.5


def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return w * h >= 500
