import os.path
from typing import Literal, Sequence

import cv2
import numpy as np
import onnxruntime as ort
import requests

MODEL_DEFAULT_PATH = '.models'
MODEL_URL_TEMPLATE = 'https://github.com/abionics/UltraLight/releases/download/v1.0.0/{}'

SizeType = Literal[320, 640]


class UltraLightDetector:
    IMAGE_MEAN = np.asarray([127, 127, 127])

    def __init__(
            self,
            size: SizeType = 640,
            path: str = None,
            providers: Sequence[str] = None,
            score_threshold: float = 0.9,
            iou_threshold: float = 0.3,
            keep_top_k: int = 1024,
            top_k: int = 256,
    ):
        """
        :param size: size of model input (640x480 or 320x240)
        :param path: path to ONNX file of model
        :param providers: ONNX providers (devices) as CPUExecutionProvider/CUDAExecutionProvider/CoreMLExecutionProvider
        :param score_threshold: confidence score threshold
        :param iou_threshold: intersection over union threshold
        :param top_k: keep K results before NMS, if k <= 0 than keep all the results
        :param top_k: keep K results after NMS, if k <= 0 than keep all the results
        """
        self._model_shape = (640, 480) if size == 640 else (320, 240)
        self._ort_session = self._create_ort_session(size, path, providers)
        ort_session_input = self._ort_session.get_inputs()[0]
        self._input_name = ort_session_input.name
        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold
        self._keep_top_k = keep_top_k
        self._top_k = top_k
        onnx_model_shape = tuple(ort_session_input.shape[3:1:-1])
        if onnx_model_shape != self._model_shape:
            raise Exception(
                f'ONNX model\'s shape is {onnx_model_shape}, while detector shape is {self._model_shape}! '
                'Change `size` param or use another model file',
            )
        if self._score_threshold < 0 or self._score_threshold > 1:
            raise Exception(f'Invalid score threshold: {self._score_threshold}, it should be in range [0..1]')
        if self._iou_threshold < 0 or self._iou_threshold > 1:
            raise Exception(f'Invalid iou threshold: {self._iou_threshold}, it should be in range [0..1]')

    @classmethod
    def _create_ort_session(cls, size: SizeType, path: str, providers: Sequence[str]) -> ort.InferenceSession:
        if path is None:
            path = cls._download_model(size)
        return ort.InferenceSession(path, providers=providers)

    @staticmethod
    def _download_model(size: SizeType) -> str:
        if not os.path.exists(MODEL_DEFAULT_PATH):
            os.mkdir(MODEL_DEFAULT_PATH)
        filename = f'ultra_light_{size}.onnx'
        path = os.path.join(MODEL_DEFAULT_PATH, filename)
        if os.path.exists(path):
            return path
        url = MODEL_URL_TEMPLATE.format(filename)
        print(f'Downloading model "{filename}" from {url}...')
        response = requests.get(url)
        print(f'Downloaded model "{filename}"')
        with open(path, 'wb') as file:
            file.write(response.content)
        print(f'Saved model "{filename}" to "{path}"')
        return path

    def detect(self, image: np.ndarray, fit_to_image: bool = False) -> tuple[np.ndarray, list[float]]:
        if image is None:
            raise Exception('Image is None')
        height, width, _ = image.shape
        image = self._preprocess_image(image)
        scores_out, boxes_out = self._ort_session.run(None, {self._input_name: image})
        return self._extract(boxes_out[0], scores_out[0], width, height, fit_to_image)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self._model_shape)
        image = (image - self.IMAGE_MEAN) / 128
        image = np.transpose(image, [2, 0, 1])
        return np.expand_dims(image, axis=0).astype(np.float32)

    def _extract(
            self,
            boxes: np.ndarray,
            scores: np.ndarray,
            width: int,
            height: int,
            fit_to_image: bool = False,
    ) -> tuple[np.ndarray, list[float]]:
        """
        Select boxes that contain human faces (function `predict` in original implementation)
        Parameters:
            boxes (N, 4): boxes array in corner-form
            scores (N, 2): scores array
            width: original image width
            height: original image height
            fit_to_image: should boxes fit to image borders
        Returns:
            boxes (K, 4): an array of boxes kept
            scores (K): a list of scores for each boxes
        """
        scores = scores[:, 1]
        mask = scores > self._score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        picked = self._hard_nms(boxes, scores)
        if len(picked) == 0:
            return np.asarray([]), list()
        boxes = boxes[picked]
        scores = scores[picked]
        boxes *= np.asarray([width, height, width, height])
        if fit_to_image:
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, width - 1)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, height - 1)
        return boxes.astype(np.int32), scores.tolist()

    def _hard_nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        """
        Perform hard non-maximum-suppression to filter out boxes with iou greater than threshold
        Parameters:
            boxes (N, 4): boxes array in corner-form
            scores (N, 2): scores array
        Returns:
            picked: a list of indexes of the kept boxes
        """
        picked = list()
        indexes = np.argsort(scores)
        indexes = indexes[-self._keep_top_k:]
        while len(indexes) > 0:
            current = indexes[-1]
            picked.append(current)
            if 0 < self._top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes]
            iou = self._iou_of(
                rest_boxes,
                np.expand_dims(current_box, axis=0),
            )
            indexes = indexes[iou <= self._iou_threshold]
        return picked

    def _iou_of(self, boxes0: np.ndarray, boxes1: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Compute intersection-over-union (Jaccard index) of boxes
        Parameters:
            boxes0 (N, 4): ground truth boxes
            boxes1 (N or 1, 4): predicted boxes
            eps: a small number to avoid 0 as denominator
        Returns:
            iou (N): IoU values
        """
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
        overlap_area = self._area_of(overlap_left_top, overlap_right_bottom)
        area0 = self._area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self._area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    @staticmethod
    def _area_of(left_top: np.ndarray, right_bottom: np.ndarray) -> np.ndarray:
        """
        Compute areas of rectangles by two corners
        Parameters:
            left_top (N, 2): left top corner
            right_bottom (N, 2): right bottom corner
        Returns:
            area (N): return the area
        """
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]
