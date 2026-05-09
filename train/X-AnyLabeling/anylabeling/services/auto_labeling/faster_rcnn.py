import logging
import os

import cv2
import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .engines.build_onnx_engine import OnnxBaseModel


class FasterRCNN(Model):
    """Object detection model using Faster R-CNN"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "conf_threshold",
            "classes",
        ]
        widgets = [
            "button_run",
            "input_conf",
            "edit_conf",
            "toggle_preserve_existing_annotations",
        ]
        output_modes = {
            "rectangle": QCoreApplication.translate("Model", "Rectangle"),
        }
        default_output_mode = "rectangle"

    def __init__(self, model_config, on_message) -> None:
        super().__init__(model_config, on_message)
        model_name = self.config["type"]
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {model_name} model.",
                )
            )
        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.classes = self.config["classes"]
        self.input_shape = self.net.get_input_shape()[-2:]  # (H, W)
        self.conf_thres = self.config["conf_threshold"]
        self.replace = True

        # ImageNet normalization constants used by torchvision Faster R-CNN
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _to_valid_int(self, value):
        """Try converting ONNX dimension value to a positive int."""
        try:
            ivalue = int(value)
            if ivalue > 0:
                return ivalue
        except (TypeError, ValueError):
            pass
        return None

    def _normalize_outputs(self, outputs):
        """Normalize different Faster R-CNN ONNX outputs to boxes/labels/scores."""
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        arrays = []
        for out in outputs:
            arr = np.asarray(out)
            if arr.ndim >= 1 and arr.shape[0] == 1:
                arr = np.squeeze(arr, axis=0)
            arrays.append(arr)

        # Case A: [boxes(N,4), labels(N), scores(N)]
        if len(arrays) >= 3:
            box_cands = [a for a in arrays if a.ndim == 2 and a.shape[1] == 4]
            one_d = [a for a in arrays if a.ndim == 1]
            if box_cands and len(one_d) >= 2:
                boxes = box_cands[0]
                # Heuristic: scores are usually float in [0,1], labels are integer-like.
                a, b = one_d[0], one_d[1]
                a_score_like = np.mean((a >= 0) & (a <= 1))
                b_score_like = np.mean((b >= 0) & (b <= 1))
                if a_score_like >= b_score_like:
                    scores, labels = a, b
                else:
                    scores, labels = b, a
                return boxes, labels, scores

        # Case B: [boxes(N,4), class_scores(N,C)]
        if len(arrays) == 2:
            a, b = arrays

            # MMDetection end2end export: [dets(N,5), labels(N)]
            # dets columns are [x1, y1, x2, y2, score]
            if a.ndim == 2 and a.shape[1] == 5 and b.ndim in (1, 2):
                boxes = a[:, :4]
                scores = a[:, 4]
                labels = b.reshape(-1)
                return boxes, labels, scores
            if b.ndim == 2 and b.shape[1] == 5 and a.ndim in (1, 2):
                boxes = b[:, :4]
                scores = b[:, 4]
                labels = a.reshape(-1)
                return boxes, labels, scores

            if a.ndim == 2 and a.shape[1] == 4 and b.ndim == 2:
                boxes = a
                class_scores = b
            elif b.ndim == 2 and b.shape[1] == 4 and a.ndim == 2:
                boxes = b
                class_scores = a
            else:
                boxes = None
                class_scores = None

            if boxes is not None and class_scores is not None:
                if class_scores.shape[1] == 1:
                    # Single class score head
                    labels = np.zeros(class_scores.shape[0], dtype=np.int64)
                    scores = class_scores[:, 0]
                else:
                    # Multi-class score head. If one extra class exists, treat idx 0 as background.
                    if class_scores.shape[1] == len(self.classes) + 1:
                        scores = class_scores[:, 1:].max(axis=1)
                        labels = class_scores[:, 1:].argmax(axis=1) + 1
                    else:
                        scores = class_scores.max(axis=1)
                        labels = class_scores.argmax(axis=1)
                return boxes, labels, scores

        # Case C: single tensor [N, >=6] like [x1,y1,x2,y2,score,label]
        if len(arrays) == 1:
            det = arrays[0]
            if det.ndim == 2 and det.shape[1] >= 6:
                boxes = det[:, :4]
                c4 = det[:, 4]
                c5 = det[:, 5]
                c4_score_like = np.mean((c4 >= 0) & (c4 <= 1))
                c5_score_like = np.mean((c5 >= 0) & (c5 <= 1))
                if c4_score_like >= c5_score_like:
                    scores, labels = c4, c5
                else:
                    scores, labels = c5, c4
                return boxes, labels, scores

        shape_info = [tuple(a.shape) for a in arrays]
        raise ValueError(
            f"Unsupported Faster R-CNN output shapes: {shape_info}"
        )

    def set_auto_labeling_conf(self, value):
        """Set auto labeling confidence threshold"""
        if value > 0:
            self.conf_thres = value

    def set_auto_labeling_preserve_existing_annotations_state(self, state):
        """Toggle preservation of existing annotations based on checkbox state"""
        self.replace = not state

    def preprocess(self, input_image):
        """
        Resize, normalize and convert image to NCHW float32 tensor.

        Args:
            input_image (np.ndarray): RGB image in HWC uint8 format.

        Returns:
            blob (np.ndarray): NCHW float32 tensor ready for inference.
            scale_x (float): Width scale factor (input_w / orig_w).
            scale_y (float): Height scale factor (input_h / orig_h).
        """
        orig_h, orig_w = input_image.shape[:2]
        input_h_raw, input_w_raw = self.input_shape
        input_h = self._to_valid_int(input_h_raw)
        input_w = self._to_valid_int(input_w_raw)

        # Some Faster R-CNN ONNX models use dynamic dims like "height"/"width".
        # In that case, keep original image size to avoid invalid resize args.
        if input_h is None or input_w is None:
            input_h, input_w = orig_h, orig_w
            resized = input_image
        else:
            resized = cv2.resize(input_image, (int(input_w), int(input_h)))
        # Normalize to [0, 1] then apply ImageNet mean/std
        image = resized.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        # HWC → CHW → NCHW
        image = np.transpose(image, (2, 0, 1))
        blob = np.expand_dims(image, axis=0).astype(np.float32)

        scale_x = float(input_w) / float(orig_w)
        scale_y = float(input_h) / float(orig_h)
        return blob, scale_x, scale_y

    def postprocess(self, outputs, scale_x, scale_y, orig_w, orig_h):
        """
        Decode Faster R-CNN ONNX outputs into detection results.

        Expected outputs layout (torchvision / MMDetection style):
            outputs[0] – boxes  : (N, 4)  [x1, y1, x2, y2] in input-image space
            outputs[1] – labels : (N,)    integer class indices (0-based)
            outputs[2] – scores : (N,)    confidence scores in [0, 1]

        Args:
            outputs (list[np.ndarray]): Raw model outputs.
            scale_x (float): Width scale used in preprocessing.
            scale_y (float): Height scale used in preprocessing.
            orig_w (int): Original image width.
            orig_h (int): Original image height.

        Returns:
            list[dict]: Detected objects with keys x1, y1, x2, y2, label, score.
        """
        boxes, labels, scores = self._normalize_outputs(outputs)
        labels = np.asarray(labels).reshape(-1)

        # Determine label convention once: 0-based or 1-based
        one_based_labels = False
        if labels.size > 0:
            l_min = int(np.min(labels))
            l_max = int(np.max(labels))
            one_based_labels = l_min >= 1 and l_max <= len(self.classes)

        results = []
        for box, label_idx, score in zip(boxes, labels, scores):
            if float(score) < self.conf_thres:
                continue

            # Map coordinates back to original image space
            x1 = int(np.clip(box[0] / scale_x, 0, orig_w - 1))
            y1 = int(np.clip(box[1] / scale_y, 0, orig_h - 1))
            x2 = int(np.clip(box[2] / scale_x, 0, orig_w - 1))
            y2 = int(np.clip(box[3] / scale_y, 0, orig_h - 1))

            label_idx = int(label_idx)
            if one_based_labels:
                label_idx = label_idx - 1
            if label_idx < 0 or label_idx >= len(self.classes):
                continue
            label = str(self.classes[label_idx])

            results.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "label": label,
                    "score": float(score),
                }
            )
        return results

    def predict_shapes(self, image, image_path=None):
        """
        Run Faster R-CNN inference and return detected bounding-box shapes.

        Args:
            image: QImage passed from the labeling canvas.
            image_path (str, optional): Path to the image file.

        Returns:
            AutoLabelingResult: Contains a list of rectangle Shape objects.
        """
        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            return []

        orig_h, orig_w = image.shape[:2]
        blob, scale_x, scale_y = self.preprocess(image)

        # Faster R-CNN returns multiple outputs; set extract=False to get all
        outputs = self.net.get_ort_inference(blob, extract=False)

        results = self.postprocess(outputs, scale_x, scale_y, orig_w, orig_h)

        shapes = []
        for det in results:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            shape = Shape(
                label=det["label"],
                score=det["score"],
                shape_type="rectangle",
            )
            shape.add_point(QtCore.QPointF(x1, y1))
            shape.add_point(QtCore.QPointF(x2, y1))
            shape.add_point(QtCore.QPointF(x2, y2))
            shape.add_point(QtCore.QPointF(x1, y2))
            shapes.append(shape)

        return AutoLabelingResult(shapes, replace=self.replace)

    def unload(self):
        del self.net
