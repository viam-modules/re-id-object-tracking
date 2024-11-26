import torch
import numpy as np
from src.config.config import FaceIdConfig
import onnxruntime as ort
import os
from src.utils import resource_path
from typing import Dict, Tuple
from src.tracker.utils import (
    save_tensor,
    resize_for_padding,
    pad_image_to_target_size,
    padded_to_original_coordinates,
    get_cropped_tensor,
)
from scipy.spatial.distance import cityblock, cosine, euclidean


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
    "sface": EncoderModelConfig(
        model_file="face_recognition_sface_2021dec.onnx",
        repository="sface",
        metric="euclidean",
        input_shape=(112, 112),
        mean=1,
        std=0.5,
    ),
}


class FaceFeaturesExtractor:
    def __init__(
        self,
        cfg: FaceIdConfig,
        debug: bool = False,
    ):
        self.debug = debug
        self.model_config = ENCODERS_CONFIG.get(cfg.feature_extractor.value)
        model_path = self.model_config.get_model_path()
        self.input_shape = self.model_config.input_shape

        # self.input_size = self.model_config.input_size
        providers = ["CPUExecutionProvider"]
        if torch.cuda.is_available():
            providers.append("CUDAExecutionProvider")
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.io_binding = self.session.io_binding()

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def get_embedding(self, face: torch.Tensor) -> torch.Tensor:
        resized_image, new_height, new_width, target_height, target_width = (
            resize_for_padding(face, self.input_shape)
        )
        padded_image = pad_image_to_target_size(resized_image, self.input_shape)
        if self.debug:
            save_tensor(padded_image, "resized_face_image.png")
        # torch_tensor
        # resized_image = resized_image - self.model_config.mean

        # torch_tensor = padded_image - 127.5
        # torch_tensor = torch_tensor / 255
        # torch_tensor = resized_image.contiguous()
        torch_tensor = padded_image[:, [2, 1, 0], :, :]
        torch_tensor = padded_image

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        # Bind the input using data_ptr
        self.io_binding.bind_input(
            name=self.input_name,
            device_type=device_type,
            # device_id=torch_tensor.get_device(),
            device_id=0,  # GPU ID
            element_type=np.float32,  # Data type
            shape=torch_tensor.shape,
            buffer_ptr=torch_tensor.data_ptr(),  # Pointer to the tensor's memory
        )

        # Bind the output to let ONNX allocate it
        # Assuming this is boxes
        self.io_binding.bind_output(self.output_name)

        # Run the inference
        self.session.run_with_iobinding(self.io_binding)
        embed = self.io_binding.copy_outputs_to_cpu()[0]
        return embed / np.linalg.norm(embed)

    def compute_distance(self, feature_vector_1, feature_vector_2):
        """
        Compute pairwise distances (Euclidean) between feature vectors.

        :param features: A list of feature vectors.
        :return: A dictionary with pairwise distances between the features.
        """

        if self.model_config.metric == "euclidean":
            distance = euclidean(feature_vector_1, feature_vector_2)
        elif self.model_config.metric == "cosine":
            distance = cosine(feature_vector_1, feature_vector_2)
        elif self.model_config.metric == "manhattan":
            distance = cityblock(feature_vector_1, feature_vector_2)
        else:
            raise ValueError(f"Unsupported metric '{self.model_config.metric }'")

        return (distance - self.model_config.mean) / self.model_config.std
