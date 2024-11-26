import torch
import numpy as np
from src.config.config import FaceIdConfig
import onnxruntime as ort
import os
from src.utils import resource_path
from typing import Dict, Tuple
from src.tracker.utils import (
    resize_for_padding,
    pad_image_to_target_size,
)
from src.tracker.face_id.face_net.inception_resnet_v1 import InceptionResnetV1


class EncoderModelConfig:
    def __init__(
        self,
        model_file: str,
        repository: str,
        metric: str,
        input_shape: Tuple[int, int],
        mean: float = 0,
        std: float = 1,
    ):
        self.model_file = model_file
        self.repository = repository
        self.metric = metric
        self.mean = mean
        self.std = std
        self.input_shape = input_shape

    def get_model_path(self):
        """
        Returns model absolute path
        """
        relative_path = os.path.join(self.repository, self.model_file)
        return resource_path(relative_path)


ENCODERS_CONFIG: Dict[str, EncoderModelConfig] = {
    "facenet": EncoderModelConfig(
        model_file="20180402-114759-vggface2.pt",
        repository="face_net",
        metric="euclidean",
        input_shape=(112, 112),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
}


class FaceFeaturesExtractor:
    def __init__(self, cfg: FaceIdConfig):
        self.model_config = ENCODERS_CONFIG.get(cfg.feature_extractor.value)
        model_path = self.model_config.get_model_path()
        self.input_shape = self.model_config.input_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        face_recognizer = InceptionResnetV1(model_path=model_path).eval()
        self.face_recognizer = face_recognizer.to(self.device)
        self.face_recognizer.requires_grad_(False)

        def transform(face: torch.Tensor):
            return (face - 127.5) / 128.0

        self.transform = transform
        # self.input_size = self.model_config.input_size
        providers = ["CPUExecutionProvider"]
        if torch.cuda.is_available():
            providers.append("CUDAExecutionProvider")

    def get_embedding(self, face: torch.Tensor) -> torch.Tensor:
        resized_image, _, _, _, _ = resize_for_padding(face, self.input_shape)
        padded_image = pad_image_to_target_size(resized_image, self.input_shape)
        torch_tensor = self.transform(padded_image)  # normalize
        embed = self.face_recognizer(torch_tensor)
        return embed[0]

    def compute_distance(self, feature_vector_1, feature_vector_2, metric="cosine"):
        """
        Compute pairwise distances between feature vectors using PyTorch.

        :param feature_vector_1: First feature vector (PyTorch tensor).
        :param feature_vector_2: Second feature vector (PyTorch tensor).
        :param metric: The distance metric to use ('euclidean', 'cosine', 'manhattan').
        :return: Computed distance.
        """
        if metric == "euclidean":
            distance = torch.norm(feature_vector_1 - feature_vector_2, p=2)
        elif metric == "cosine":
            distance = 1 - torch.nn.functional.cosine_similarity(
                feature_vector_1.unsqueeze(0), feature_vector_2.unsqueeze(0)
            )
        elif metric == "manhattan":
            distance = torch.sum(torch.abs(feature_vector_1 - feature_vector_2))
        else:
            raise ValueError(f"Unsupported metric '{metric}'")
        return distance.cpu().item()
