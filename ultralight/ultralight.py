from typing import Sequence

import cv2
import numexpr as ne
import numpy as np
from onnxruntime import InferenceSession

from ultralight.exceptions import InvalidModelParamException, InvalidInputException
from ultralight.loader import UltraLightLoader
from ultralight.types import SizeType, BoxesType, ScoresType


class UltraLightDetector:

    def __init__(
            self,
            size: SizeType = 640,
            batched: bool = True,
            path: str = None,
            providers: Sequence[str] = None,
            score_threshold: float = 0.9,
            iou_threshold: float = 0.3,
            keep_top_k: int = 1024,
            top_k: int = 256,
    ):
        """
        :param size: size of model input (640x480 or 320x240)
        :param batched: should model use batched input
        :param path: path to ONNX file of model
        :param providers: ONNX providers (devices) as CPUExecutionProvider/CUDAExecutionProvider/CoreMLExecutionProvider
        :param score_threshold: confidence score threshold
        :param iou_threshold: intersection over union threshold
        :param top_k: keep K results before NMS, if k <= 0 than keep all the results
        :param top_k: keep K results after NMS, if k <= 0 than keep all the results
        """
        self._batched = batched
        self._model_shape = (640, 480) if size == 640 else (320, 240)
        self._session = self._create_session(size, batched, path, providers)
        session_input = self._session.get_inputs()[0]
        self._input_name = session_input.name
        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold
        self._keep_top_k = keep_top_k
        self._top_k = top_k
        onnx_model_shape = tuple(session_input.shape[3:1:-1])
        if onnx_model_shape != self._model_shape:
            raise InvalidModelParamException(
                f'ONNX model\'s shape is {onnx_model_shape}, while detector shape is {self._model_shape}! '
                'Change `size` param or use another model file',
            )
        if self._score_threshold < 0 or self._score_threshold > 1:
            raise InvalidModelParamException(
                f'Invalid score threshold: {self._score_threshold}, it should be in range [0..1]'
            )
        if self._iou_threshold < 0 or self._iou_threshold > 1:
            raise InvalidModelParamException(
                f'Invalid iou threshold: {self._iou_threshold}, it should be in range [0..1]'
            )

    @classmethod
    def _create_session(
            cls,
            size: SizeType,
            batched: bool,
            path: str,
            providers: Sequence[str],
    ) -> InferenceSession:
        if path is None:
            loader = UltraLightLoader()
            path = loader.load(size, batched)
        return InferenceSession(path, providers=providers)

    def detect_one(self, image: np.ndarray, fit_to_image: bool = False) -> tuple[BoxesType, ScoresType]:
        if image is None:
            raise InvalidInputException('Image is None')
        return self.detect_batch(np.asarray([image]), fit_to_image)[0]

    def detect_batch(
            self,
            images: Sequence[np.ndarray],
            fit_to_image: bool = False,
    ) -> list[tuple[BoxesType, ScoresType]]:
        if images is None or len(images) == 0:
            return list()
        for i, image in enumerate(images):
            if image is None:
                raise InvalidInputException(f'Image with index {i} is None')
        if len(images) > 1 and not self._batched:
            raise InvalidInputException(
                f'Current detector is not batched, but images count is more than 1 (count is {len(images)}), '
                f'use `batched=True` param in detector\'s constructor to fix this'
            )
        images_input = self._preprocess_images(images)
        outputs = self._session.run(None, {self._input_name: images_input})
        results = list()
        for image, scores, boxes, *_ in zip(images, *outputs):
            height, width, _ = image.shape
            boxes, scores = self._postprocess(boxes, scores, width, height, fit_to_image)
            results.append((boxes, scores))
        return results

    def _preprocess_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        images = np.asarray([
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for image in images
        ])
        images = np.asarray([
            cv2.resize(image, self._model_shape)
            for image in images
        ])
        images = images.astype(np.float32)  # noqa used in numexpr
        images = ne.evaluate('(images - 127) / 128')
        return np.transpose(images, [0, 3, 1, 2])

    def _postprocess(
            self,
            boxes: np.ndarray,
            scores: np.ndarray,
            width: int,
            height: int,
            fit_to_image: bool = False,
    ) -> tuple[BoxesType, ScoresType]:
        """
        Select boxes that contain human faces (function `predict` in original implementation)
        Parameters:
            boxes (N, 4): boxes array in corner-form
            scores (N, 2): scores array
            width: original image width
            height: original image height
            fit_to_image: should boxes fit to image borders
        Returns:
            faces (K, 4): an array of faces
            scores (K): an array of scores for each face
        """
        scores = scores[:, 1]
        mask = scores > self._score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        picked = self._hard_nms(boxes, scores)
        if len(picked) == 0:
            return np.asarray([]), np.asarray([], dtype=np.float32)
        boxes = boxes[picked]
        scores = scores[picked]
        boxes *= np.asarray([width, height, width, height])
        if fit_to_image:
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, width - 1)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, height - 1)
        return boxes.astype(np.int32), scores

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
