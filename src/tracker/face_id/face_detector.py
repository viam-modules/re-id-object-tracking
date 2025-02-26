import os
from typing import Dict, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch

from src.config.config import FaceIdConfig
from src.image.image import ImageObject
from src.tracker.track import Track
from src.tracker.utils import (
    get_cropped_tensor,
    pad_image_to_target_size,
    padded_to_original_coordinates,
    resize_for_padding,
    save_tensor,
)
from src.utils import resource_path

ULTRA_FACE_REPO = "ultraface"


class FaceDetectorModelConfig:
    def __init__(
        self,
        model_file: str,
        input_size: Tuple[int, int],
        repository: str = ULTRA_FACE_REPO,
        mean: float = 127,
        std: float = 128,
    ):
        self.model_file = model_file
        self.repository = repository
        self.mean = mean
        self.std = std
        self.input_size = input_size

    def get_model_path(self):
        """
        Returns model absolute path
        """
        relative_path = os.path.join(self.repository, self.model_file)
        return resource_path(relative_path)


ENCODERS_CONFIG: Dict[str, FaceDetectorModelConfig] = {
    "ultraface_version-RFB-320-int8": FaceDetectorModelConfig(
        model_file="version-RFB-320-int8.onnx", input_size=(240, 320)
    ),
}

OSNET_REPO = "osnet"


class FaceDetector:
    def __init__(
        self,
        cfg: FaceIdConfig,
        debug: bool = False,
    ):
        self.model_config = ENCODERS_CONFIG.get(cfg.detector.value)
        model_path = self.model_config.get_model_path()
        self.input_size = self.model_config.input_size
        providers = ["CPUExecutionProvider"]
        # self.device = torch.device()
        if torch.cuda.is_available():
            self.device_type = "cuda"
            providers.append("CUDAExecutionProvider")
        else:
            self.device_type = "cpu"
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.io_binding = self.session.io_binding()
        self.input_name = self.session.get_inputs()[0].name
        self.confidence_output_name = self.session.get_outputs()[
            0
        ].name  # Assuming this is confidences
        self.box_output_name = self.session.get_outputs()[1].name
        self.threshold = cfg.detector_threshold.value
        self.debug = debug
        self.margin = 0

    def extract_face_from_track(
        self, image_object: ImageObject, track: Track
    ) -> torch.Tensor:
        image_height, image_width = image_object.float32_tensor.shape[1:]
        x1, y1, x2, y2 = map(int, track.bbox)  # Ensure integer coordinates
        x1, y1 = max(0, x1), max(0, y1)  # Clip to image dimensions
        x2, y2 = min(image_width, x2), min(image_height, y2)

        cropped_image = image_object.float32_tensor[:, y1:y2, x1:x2]
        if self.debug:
            save_tensor(cropped_image, "should_be_a_person.png")
        return self.extract_face(cropped_image)

    def extract_face(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Returns the box images in the coordinate of the input
        """

        input_height, input_width = input.shape[1:]
        resized_image, new_height, new_width, _, _ = resize_for_padding(
            input, self.input_size
        )
        padded_image = pad_image_to_target_size(resized_image, self.input_size)
        input_height, input_width = padded_image.shape[2:]
        if self.debug:
            save_tensor(padded_image, "resized_image.png")
        padded_image = padded_image - 127

        # Divide every pixel value by 128
        torch_tensor = padded_image / 128

        # Ensure contiguous
        torch_tensor.contiguous()

        # Bind the input using data_ptr
        self.io_binding.bind_input(
            name=self.input_name,
            device_type=self.device_type,
            # device_id=torch_tensor.get_device(),
            device_id=0,  # GPU ID
            element_type=np.float32,  # Data type
            shape=torch_tensor.shape,
            buffer_ptr=torch_tensor.data_ptr(),  # Pointer to the tensor's memory
        )

        # Bind the output to let ONNX allocate it
        # Assuming this is boxes
        self.io_binding.bind_output(self.confidence_output_name)
        self.io_binding.bind_output(self.box_output_name)

        # Run the inference
        self.session.run_with_iobinding(self.io_binding)

        # Fetch outputs
        confidences = self.io_binding.copy_outputs_to_cpu()[0]
        boxes = self.io_binding.copy_outputs_to_cpu()[1]
        boxes, _, _ = predict(
            input_width, input_height, confidences, boxes, self.threshold
        )

        if len(boxes) != 1:
            return None
        x1_f, x2_f = boxes[0][0], boxes[0][2]
        y1_f, y2_f = boxes[0][1], boxes[0][3]
        original_x1, original_y1 = padded_to_original_coordinates(
            x1_f, y1_f, input.shape[1:], (new_height, new_width), self.input_size
        )
        original_x2, original_y2 = padded_to_original_coordinates(
            x2_f, y2_f, input.shape[1:], (new_height, new_width), self.input_size
        )

        face_tensor = get_cropped_tensor(
            input, original_y1, original_y2, original_x1, original_x2, self.margin
        )
        if self.debug:
            save_tensor(face_tensor, "should_be_a_face.png")
        return face_tensor


def area_of(left_top, right_bottom):
    """
    Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Perform hard non-maximum-supression to filter out boxes with iou greater
    than threshold
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
        picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def predict(
    width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1
):
    """
    Select boxes that contain human faces
    Args:
        width: original image width
        height: original image height
        confidences (N, 2): confidence array
        boxes (N, 4): boxes array in corner-form
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
        boxes (k, 4): an array of boxes kept (format: x1, y1, x2, y2)
        labels (k): an array of labels for each boxes kept
        probs (k): an array of probabilities for each boxes being in corresponding labels
    """
    boxes = boxes[0]
    confidences = confidences[0]
    # print(boxes)
    # print(confidences)

    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        # print(confidences.shape[1])
        probs = confidences[:, class_index]
        # print(probs)
        mask = probs > prob_threshold
        probs = probs[mask]

        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        # print(subset_boxes)
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(
            box_probs,
            iou_threshold=iou_threshold,
            top_k=top_k,
        )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return (
        picked_box_probs[:, :4].astype(np.int32),
        np.array(picked_labels),
        picked_box_probs[:, 4],
    )
