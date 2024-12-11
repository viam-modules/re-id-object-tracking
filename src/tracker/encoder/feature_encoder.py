from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch

from src.config.config import FeatureEncoderConfig
from src.tracker.detector.detection import Detection


class FeatureEncoder(ABC):
    def __init__(self, cfg: FeatureEncoderConfig):
        """
        Initializes the detector with a DetectorConfig object that contains configuration for the model.
        """
        self.cfg = cfg

    @abstractmethod
    def compute_features(
        self, image: np.ndarray, detections: List[Detection]
    ) -> List[np.ndarray]:
        """
        Abstract method to be implemented by specific detector classes. Each detector must implement
        this method to detect objects in the provided image.
        """
        pass

    @abstractmethod
    def compute_distance(
        self, feature_vector_1: torch.Tensor, feature_vector_2: torch.Tensor
    ) -> float:
        pass


# Detector factory function based on configuration
def get_encoder(cfg: FeatureEncoderConfig) -> FeatureEncoder:
    """
    Factory function to return the correct detector based on the configuration.
    """
    # Delay the import of TorchvisionDetector to avoid circular imports
    model_name: str = cfg.feature_extractor_name.value
    if model_name.startswith("osnet_"):
        from src.tracker.encoder.os_net.os_net_encoder import OSNetFeatureEncoder

        return OSNetFeatureEncoder(cfg)
    else:
        raise ValueError(f"Model {cfg.model_name.value} is not supported.")
