import datetime
import os
import ssl
from typing import Dict, List

import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
)

from src.config.config import DetectorConfig
from src.image.image import ImageObject
from src.tracker.detector.detection import Detection
from src.tracker.detector.detector import Detector
from src.tracker.utils import save_tensor

# # This is to download resnet weights at runtime.
# Bypasses SSL certificate verification to avoid SSL errors such as:
# URLError - <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate>
ssl._create_default_https_context = ssl._create_unverified_context


class TorchvisionDetector(Detector):
    def __init__(self, cfg: DetectorConfig):
        super().__init__(cfg)
        if cfg.model_name == "fasterrcnn_mobilenet_v3_large_320_fpn":
            weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
            self.model = fasterrcnn_mobilenet_v3_large_320_fpn(
                weights=weights,
                box_score_thresh=cfg.threshold,
                num_classes=91,
            )
        elif cfg.model_name == "fasterrcnn_mobilenet_v3_large_fpn":
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
            self.model = fasterrcnn_mobilenet_v3_large_fpn(
                weights=weights,
                box_score_thresh=cfg.threshold,
                num_classes=91,
            )

        else:
            raise ValueError(
                "Supported torchvision detector are 'fasterrcnn_mobilenet_v3_large_320_fpn' or  'fasterrcnn_mobilenet_v3_large_fpn'"
            )

        self.categories = weights.meta["categories"]
        self.threshold = cfg.threshold  # TODO: do it in the detector super class
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.transform = weights.transforms()

        # Debug configuration
        self._enable_debug_tools = cfg._enable_debug_tools
        if self._enable_debug_tools:
            if cfg._path_to_debug_directory is None:
                raise ValueError(
                    "path_to_debug_directory is not set but _enable_debug_tools is set to true. Please set it to a valid path."
                )
            self._path_to_debug_directory = cfg._path_to_debug_directory
            if not os.path.exists(self._path_to_debug_directory):
                os.makedirs(self._path_to_debug_directory)
            self._max_size_debug_directory = cfg._max_size_debug_directory

    def detect(self, image: ImageObject, visualize: bool = False) -> List[Detection]:
        preprocessed_image = self.transform(image.uint8_tensor)
        batch = [preprocessed_image]

        with torch.no_grad():
            output = self.model(batch)[0]

        detections = self.post_process(output)
        # Save image if persons were detected and saving is enabled
        if self._enable_debug_tools and detections:
            # Check if debug directory has space
            debug_files = [
                f
                for f in os.listdir(self._path_to_debug_directory)
                if f.endswith(".png")
            ]
            if len(debug_files) < self._max_size_debug_directory:
                # Create filename with timestamp and sorted bounding boxes with scores
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                boxes_str = "-".join(
                    f"{x1}_{y1}_{x2}_{y2}_{d.score:.2f}"
                    for d in detections
                    for x1, y1, x2, y2 in [d.bbox]
                )
                filename = f"{timestamp}_boxes_{boxes_str}.png"
                filepath = os.path.join(self._path_to_debug_directory, filename)

                save_tensor(image.float32_tensor, filepath)

        return detections

    def post_process(self, input: Dict[str, torch.Tensor]) -> List[Detection]:
        """
        Post-process the output of a torchvision detection model to create a list of Detection objects,
        filtering only detections where the label is 1.
        """
        detections = []

        boxes = input["boxes"]  # Tensor of shape [N, 4]
        scores = input["scores"]  # Tensor of shape [N]
        labels = input["labels"]  # Tensor of shape [N]

        # Iterate over the detections in the current image
        for i in range(len(boxes)):
            label_idx = labels[i].item()
            score = scores[i].item()
            if label_idx == 1 and score > self.threshold:
                bbox = list(map(int, boxes[i].tolist()))
                category = self.categories[label_idx]
                detections.append(Detection(bbox, score, category))

        return detections
