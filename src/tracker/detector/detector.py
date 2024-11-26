from abc import ABC, abstractmethod
import numpy as np
from typing import List
from src.config.config import DetectorConfig
from src.tracker.detector.detection import Detection


class Detector(ABC):
    def __init__(self, cfg: DetectorConfig):
        """
        Initializes the detector with a DetectorConfig object that contains configuration for the model.
        """
        self.cfg = cfg

    @abstractmethod
    def detect(self, image: np.ndarray, visualize: bool = False) -> List[Detection]:
        """
        Abstract method to be implemented by specific detector classes. Each detector must implement
        this method to detect objects in the provided image.
        """


# Detector factory function based on configuration
def get_detector(cfg: DetectorConfig) -> Detector:
    """
    Factory function to return the correct detector based on the configuration.
    """
    # Delay the import of TorchvisionDetector to avoid circular imports
    if cfg.model_name.value == "fasterrcnn_mobilenet_v3_large_320_fpn":
        from src.tracker.detector.torchvision_detector import TorchvisionDetector

        return TorchvisionDetector(cfg)
    else:
        raise ValueError(f"Model {cfg.model_name.value} is not supported.")
