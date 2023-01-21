import cv2
import numpy as np


def format_image(image: np.ndarray) -> np.ndarray:
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def box_to_point(box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # todo limit (0, width) and (0, height) if needed
    return box[0:2], box[2:4]


def draw_faces(image: np.ndarray, boxes: np.ndarray, scores: list[float] = None, **kwargs):
    count = len(boxes)
    show_score = scores is not None
    for i in range(count):
        point_1, point_2 = box_to_point(boxes[i])
        score = scores[i] if show_score else None
        draw_face(image, point_1, point_2, score, **kwargs)


def draw_face(
        image: np.ndarray,
        point_1: np.ndarray,
        point_2: np.ndarray,
        score: float = None,
        score_pad: tuple[int, int] = (0, 5),
        color: tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
        font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.5,
        **kwargs,
):
    cv2.rectangle(image, point_1, point_2, color, thickness, **kwargs)
    if score is not None:
        position = point_1 - score_pad
        cv2.putText(image, f'{100 * score:.2f}', position, font_face, font_scale, color, thickness, **kwargs)
