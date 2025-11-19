import argparse
import asyncio

import torch

from src.config.config import ReIDObjetcTrackerConfig
from src.image.image import ImageObject
from src.tracker.detector.detector import get_detector
from tests.fake_camera import FakeCamera
from tests.utils import get_config

CAMERA_NAME = "fake-camera"
WORKING_CONFIG_DICT = {
    "camera_name": CAMERA_NAME,
    "path_to_database": "./something.db",
    "_start_background_loop": False,
    "detection_threshold": 0.1,
    "detector_model_name": "fasterrcnn_mobilenet_v3_large_fpn",
    "_enable_debug_tools": True,
    "_max_size_debug_directory": 7,
}


async def main(img_path, threshold, detector_model_name):
    cam = FakeCamera(CAMERA_NAME, img_path=img_path, use_ring_buffer=True)
    device = torch.device("cpu")

    # Getting detector configuration
    config_dict = WORKING_CONFIG_DICT.copy()
    config_dict["detection_threshold"] = threshold
    config_dict["detector_model_name"] = detector_model_name
    cfg = get_config(config_dict)
    person_detector_config = ReIDObjetcTrackerConfig(cfg).detector_config
    detector = get_detector(person_detector_config)

    number_of_detections = 0
    for i in range(cam.get_number_of_images()):
        imgs = await cam.get_images()
        if len(imgs) > 0:
            img_object = ImageObject(imgs[0], device)
            detections = detector.detect(img_object)
            if len(detections) > 0:
                print(f"Frame {i}: found {detections}")
                number_of_detections += 1

    print(
        f"Found {number_of_detections} detection(s) with threshold {threshold} using model {detector_model_name}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate person detection")
    parser.add_argument(
        "--img_path",
        type=str,
        help="Path to the image or directory of images",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="Detection threshold"
    )
    parser.add_argument(
        "--detector_model_name",
        type=str,
        default="fasterrcnn_mobilenet_v3_large_fpn",
        choices=[
            "fasterrcnn_mobilenet_v3_large_fpn",
            "fasterrcnn_mobilenet_v3_large_320_fpn",
        ],
        help="Name of the person detector model to use",
    )
    args = parser.parse_args()

    # Update the WORKING_CONFIG_DICT with the new model name
    config_dict = WORKING_CONFIG_DICT.copy()
    config_dict["detector_model_name"] = args.detector_model_name
    config_dict["detection_threshold"] = args.threshold

    asyncio.run(main(args.img_path, args.threshold, args.detector_model_name))
