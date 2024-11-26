import torchvision
import torch


import os
from typing import List, Dict


from src.config.config import DetectorConfig
from src.utils import resource_path
from src.tracker.detector.detection import Detection
from src.tracker.detector.detector import Detector
from src.image.image import ImageObject
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)


class TorchvisionDetector(Detector):
    def __init__(self, cfg: DetectorConfig):
        super().__init__(cfg)
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
        self.categories = weights.meta["categories"]

        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights, box_score_thresh=cfg.threshold.value, num_classes=91
        )
        self.threshold = cfg.threshold.value  # TODO: do it in the detector super class
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.transform = weights.transforms()

    def detect(self, image: ImageObject, visualize: bool = False) -> List[Detection]:
        # image_tensor = (
        #     torch.from_numpy(image).permute(2, 0, 1).contiguous()
        # )
        preprocessed_image = self.transform(image.uint8_tensor)
        batch = [preprocessed_image]

        with torch.no_grad():
            detections = self.model(batch)[0]

        return self.post_process(detections)

    def post_process(self, input: Dict[str, torch.Tensor]) -> List[Detection]:
        """
        Post-process the output of a torchvision detection model to create a list of Detection objects,
        filtering only detections where the label is 1.

        :param input: The output from a torchvision detection model, which is a dictionary containing:
                    - 'boxes': Tensor of shape [N, 4], bounding boxes in [x1, y1, x2, y2] format.
                    - 'scores': Tensor of shape [N], confidence scores for each detection.
                    - 'labels': Tensor of shape [N], class indices for each detection.
        :return: A list of Detection objects where the label is 1.
        """
        detections = []

        boxes = input["boxes"]  # Tensor of shape [N, 4]
        scores = input["scores"]  # Tensor of shape [N]
        labels = input["labels"]  # Tensor of shape [N]

        # Iterate over the detections in the current image
        for i in range(len(boxes)):
            label_idx = labels[i].item()  # Class index as an int
            score = scores[i].item()
            if (
                label_idx == 1 and score > self.threshold
            ):  # Filter only detections with label 1
                bbox = list(
                    map(int, boxes[i].tolist())
                )  # [x1, y1, x2, y2] format as list of integerså
                score = score  # Confidence score as a float
                category = self.categories[label_idx]  # Get the category (class name)

                # Append the Detection object
                detections.append(Detection(bbox, score, category))

        return detections
