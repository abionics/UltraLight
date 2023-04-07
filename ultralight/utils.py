import struct

import cv2
import numpy as np

from ultralight.types import BoxesType, ScoresType


def format_image(image: np.ndarray) -> np.ndarray:
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def box_to_point(box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return box[0:2], box[2:4]


def draw_faces(image: np.ndarray, boxes: BoxesType, scores: ScoresType = None, **kwargs):
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


def create_batched_version(input_filename: str, output_filename: str, batch_size: int | str = 'N'):
    import onnx
    model = onnx.load(input_filename)
    # Set dynamic batch size in inputs and outputs
    graph = model.graph
    for tensor in list(graph.input) + list(graph.value_info) + list(graph.output):
        tensor_dim_0 = tensor.type.tensor_type.shape.dim[0]
        if isinstance(batch_size, int):
            tensor_dim_0.dim_value = batch_size
        else:
            tensor_dim_0.dim_param = batch_size

    # Set dynamic batch size in reshapes (-1)
    for node in graph.node:
        if node.op_type != 'Reshape':
            continue
        for init in graph.initializer:
            # node.input[1] is expected to be a reshape
            if init.name != node.input[1]:
                continue
            # Shape is stored as a list of ints
            if len(init.int64_data) > 0:
                # This overwrites bias nodes' reshape shape but should be fine
                init.int64_data[0] = -1
            # Shape is stored as bytes
            elif len(init.raw_data) > 0:
                shape = bytearray(init.raw_data)
                struct.pack_into('q', shape, 0, -1)
                init.raw_data = bytes(shape)

    # Fix fucking bug (todo fix normally)
    for node in model.graph.node:
        for output in node.output:
            if output not in ('310', '311'):
                continue
            model.graph.output.extend([onnx.ValueInfoProto(name=output)])
    onnx.save(model, output_filename)


if __name__ == '__main__':
    # create_batched_version('../models/ultra_light_320.onnx', '../models/ultra_light_320_batched_64.onnx', 64)
    create_batched_version('../models/ultra_light_640.onnx', '../models/ultra_light_640_batched_4.onnx', 4)
