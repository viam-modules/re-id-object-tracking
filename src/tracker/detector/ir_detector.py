import torch.nn as nn
import torch
import os
from torchvision.models.detection.faster_rcnn import FasterRCNN
import logging
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from src.config.config import DetectorConfig
from src.image.image import ImageObject
from src.tracker.detector.detection import Detection
from src.tracker.detector.detector import Detector
from src.tracker.utils import save_tensor
from typing import List, Dict
import datetime

log = logging.getLogger(__name__)


class IRDetector(Detector):
    def __init__(self, cfg: DetectorConfig):
        super().__init__(cfg)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "/Users/isha.yerramilli-rao/re-id-ir-detector/best_model.pth"
        self.model = self._build_model(model_path)

        # setting params
        self.input_size = (600, 800)
        self.image_mean = [0.2618]
        self.image_std = [0.1905]
        self.box_nms_thresh = 0.5
        self.threshold = 0.85
        self.categories = ["background", "person"]
        self.model.to(self.device)
        self.model.eval()

        # transforms backbone and resizes/ normalizes images
        self.model.transform = SingleChannelRCNNTransform(
            min_size=self.input_size[0],  # can adjust
            max_size=self.input_size[1],
            image_mean=self.image_mean,
            image_std=self.image_std,
        )
        # Debug configuration from torchvision detector
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
    
    @property
    def detector_type(self) -> str:
        return "IR"

    def _build_model(self, model_path: str):
        # replicating model strcutre from training
        class FasterRCNNDetector(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            # Create backbone
            backbone = mobilenet_v3_large(weights=None)
            backbone.features[0][0] = replace_first_conv_to_1channel(
                backbone.features[0][0]
            )

            # Create FPN
            backbone_fpn = BackboneWithFPN(
                backbone.features,
                return_layers={
                    "4": "0",
                    "6": "1",
                    "12": "2",
                    "16": "3",
                },
                in_channels_list=[40, 40, 112, 960],
                out_channels=256,
            )

            # create FasterRCNN backbone
            self.model = FasterRCNN(
                backbone=backbone_fpn,
                num_classes=2,  # +1 for background
                box_nms_thresh=0.5,
            )

        wrapper_model = FasterRCNNDetector(self.model)

        if os.path.exists(model_path):
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            if "model_state_dict" in checkpoint:
                wrapper_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                wrapper_model.load_state_dict(checkpoint)
            print(f"Loaded IR model from {model_path}")
        else:
            raise FileNotFoundError(f"IR model file not found: {model_path}")

        return wrapper_model.model

    def detect(self, image: ImageObject, visualize: bool = False) -> List[Detection]:
        rgb_tensor = image.uint8_tensor  # shape: [3,h,w]
        single_channel_tensor = rgb_tensor[0:1]  # shape: [1,h,w]
        single_channel_tensor = single_channel_tensor.float() / 255.0

        preprocessed_image = self.model.transform([single_channel_tensor])
        # batch = [preprocessed_image]
        # getting the actual tensors
        processed_tensors = (preprocessed_image[0]).tensors
        self.resized_image = processed_tensors

        with torch.no_grad():
            output = self.model(processed_tensors)  # run inference

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
                self.visualize_ir_detections(detections, filepath)

        return detections

    # TODO: add post processing to check detections are in the right format
    def post_process(self, input: List[Dict[str, torch.Tensor]]) -> List[Detection]:
        """
        Post-process the output of a torchvision detection model to create a list of Detection objects,
        filtering only detections where the label is 1.
        """
        detections = []

        results = input[0]

        print(f"results: {results.keys()}")

        boxes = results["boxes"]  # Tensor of shape [N, 4]
        scores = results["scores"]  # Tensor of shape [N]
        labels = results["labels"]  # Tensor of shape [N]

        # Iterate over the detections in the current image
        for i in range(len(boxes)):
            label_idx = labels[i].item()
            score = scores[i].item()
            if score > self.threshold:
                bbox = list(map(int, boxes[i].tolist()))
                category = self.categories[label_idx]
                print(f"bbox: {bbox}, score: {score}, category: {category}")
                detections.append(Detection(bbox, score, category))

        return detections

    # def visualize_ir_detections(self, detections, output_path):
    #     """
    #     Visualize IR detections on the image and save to debug directory.

    #     Args:
    #         image_tensor: torch.Tensor of shape [C, H, W] (your single channel image)
    #         detections: List of Detection objects from your IR detector
    #         output_path: Path where to save the visualization
    #     """
    #     import matplotlib.pyplot as plt
    #     import matplotlib.patches as patches
    #     import numpy

    #     # take first channel and squeeze to just H, W

    #     img_np = self.resized_image[0].cpu().numpy().squeeze(0)

    #     fig, ax = plt.subplots(1, figsize=(12, 8))
    #     ax.imshow(img_np, cmap="gray")

    #     # Plot detected boxes
    #     if detections and len(detections) > 0:
    #         for detection in detections:
    #             x1, y1, x2, y2 = detection.bbox
    #             score = detection.score
    #             category = detection.category

    #             # Create rectangle patch
    #             rect = patches.Rectangle(
    #                 (x1, y1),
    #                 x2 - x1,
    #                 y2 - y1,
    #                 linewidth=2,
    #                 edgecolor="red",
    #                 facecolor="none",
    #             )
    #             ax.add_patch(rect)

    #             # Add text with score and category
    #             ax.text(
    #                 x1,
    #                 y1 - 5,
    #                 f"{category}: {score:.3f}",
    #                 color="red",
    #                 fontsize=10,
    #                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    #             )

    #     plt.title(f"IR Detections: {len(detections)} persons found")
    #     plt.axis("off")

    #     # Save the figure
    #     plt.savefig(output_path, dpi=150, bbox_inches="tight")
    #     plt.close()

    #     print(f"Saved detection visualization to: {output_path}")


class SingleChannelRCNNTransform(GeneralizedRCNNTransform):
    def __init__(self, min_size, max_size, image_mean, image_std):
        # Override mean and std for single channel
        image_mean = [image_mean[0]]
        image_std = [image_std[0]]
        super().__init__(min_size, max_size, image_mean, image_std)


def replace_first_conv_to_1channel(conv3: nn.Conv2d) -> nn.Conv2d:
    new_conv = nn.Conv2d(  # rebuilding first conv layer to accept 1 channel input
        in_channels=1,
        out_channels=conv3.out_channels,
        kernel_size=conv3.kernel_size,
        stride=conv3.stride,
        padding=conv3.padding,
        bias=(conv3.bias is not None),
    )
    # Custom init
    nn.init.kaiming_normal_(
        new_conv.weight, mode="fan_out", nonlinearity="relu"
    )  # fan out for preserving weight variance
    if new_conv.bias is not None:
        nn.init.zeros_(
            new_conv.bias
        )  # initializes bias to 0 as weights will be adjusted later
    return new_conv
