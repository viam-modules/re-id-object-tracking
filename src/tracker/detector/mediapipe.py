import os
from typing import List, Dict

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from src.config.config import DetectorConfig
from src.utils import resource_path
from src.tracker.detector.detection import Detection


class DetectorModelConfig:
    def __init__(self, path: str):
        self.path = path

    def get_model_path(self):
        """
        Returns model absolute path
        """
        relative_path = os.path.join("mediapipe", self.path)
        return resource_path(relative_path)


DETECTORS_CONFIG: Dict[str, DetectorModelConfig] = {
    "effDet0_int8": DetectorModelConfig(
        path="efficientdet_int8.tflite",
    ),
    "effDet0_fp32": DetectorModelConfig(
        path="efficientdet_fp32.tflite",
    ),
    "effDet2_fp32": DetectorModelConfig(
        path="efficientdet2_fp32.tflite",
    ),
}


class Detector:
    def __init__(self, cfg: DetectorConfig):
        """
        Initialize the Detector with a given model and options.

        :param model_path: Path to the TFLite model file.
        :param score_threshold: Minimum confidence score for detections.
        :param max_results: Maximum number of detection results to return.
        """

        model_name = cfg.model_name.value
        self.model_config = DETECTORS_CONFIG.get(model_name)

        BaseOptions = mp.tasks.BaseOptions
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = ObjectDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=self.model_config.get_model_path()
            ),
            max_results=cfg.max_results.value,
            running_mode=VisionRunningMode.IMAGE,
            score_threshold=cfg.threshold.value,
            category_allowlist=["person"],
        )
        self.detector = vision.ObjectDetector.create_from_options(self.options)

    def detect(self, image: np.ndarray, visualize: bool = False) -> List[Detection]:
        """
        Detect persons in the given image and return bounding boxes and scores.

        :param image: Image provided as a NumPy array.
        :return: List of tuples, where each tuple contains a bounding box [x1, y1, x2, y2] and a confidence score.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = self.detector.detect(mp_image)

        detections = []
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x1, y1 = bbox.origin_x, bbox.origin_y
            x2 = x1 + bbox.width
            y2 = y1 + bbox.height
            score = detection.categories[0].score  # category "0" is person
            category = detection.categories[0].category_name

            detections.append(
                (
                    Detection(
                        bbox=[x1, y1, x2, y2],
                        score=score,
                        category=category,
                    )
                )
            )

            if visualize:
                cv2.rectangle(
                    bgr_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )
                # Put the score on the top-left corner of the bounding box
                cv2.putText(
                    bgr_image,
                    f"{category}: {score:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        if visualize:
            cv2.imshow("Detections", bgr_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return detections
