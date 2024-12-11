import os
import os.path as osp
import pickle

# not useful
import sys
from collections import OrderedDict
from functools import partial
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms as T
from scipy.spatial.distance import cityblock, cosine, euclidean
from viam.logging import getLogger

from src.config.config import FeatureEncoderConfig
from src.image.image import ImageObject
from src.tracker.detector.detection import Detection
from src.tracker.encoder.feature_encoder import FeatureEncoder
from src.tracker.encoder.os_net.osnet import osnet_ain_x1_0
from src.tracker.utils import (
    pad_image_to_target_size,
    resize_for_padding,
)
from src.utils import resource_path

LOGGER = getLogger(__name__)


class EncoderModelConfig:
    def __init__(
        self,
        model_file: str,
        repository: str,
        metric: str,
        mean: float = 0,
        std: float = 1,
    ):
        self.model_file = model_file
        self.repository = repository
        self.metric = metric
        self.mean = mean
        self.std = std

    def get_model_path(self):
        """
        Returns model absolute path
        """
        relative_path = os.path.join(self.repository, self.model_file)
        return resource_path(relative_path)


ENCODERS_CONFIG: Dict[str, EncoderModelConfig] = {
    "osnet_x0_25": EncoderModelConfig(
        model_file="osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        repository="torchreid",
        metric="euclidean",
        mean=10,
        std=(35 - 10),
    ),
    "osnet_ain_x1_0": EncoderModelConfig(
        model_file="osnet_ain_ms_d_c.pth.tar",
        repository="torchreid",
        metric="cosine",
        mean=0,
        std=1,
    ),
    "model_2": None,
}

OSNET_REPO = "osnet"


class OSNetFeatureEncoder(FeatureEncoder):
    def __init__(self, cfg: FeatureEncoderConfig):
        super().__init__(cfg)
        """
        Initialize the FeatureEncoder with a feature extractor model.

        :param model_name: The name of the model to use for feature extraction.
        :param model_path: The path to the pre-trained model file.
        :param device: The device to run the model on ('cpu' or 'cuda').
        """

        if cfg.device.value == "cuda":
            if not torch.cuda.is_available():
                LOGGER.warning(
                    "'feature_encoder_device' is set to 'cuda' but cuda is not avalaible. using cpu instead"
                )
                use_gpu = False
                self.device = torch.device("cpu")
            else:
                use_gpu = True
                self.device = torch.device("cuda")

        else:  # cfg.device = "cpu"
            use_gpu = False
            self.device = torch.device("cpu")

        self.model_config = ENCODERS_CONFIG.get(cfg.feature_extractor_name.value)

        model_name = "osnet_ain_x1_0"  # cfg.feature_extractor_name.value
        if model_name == "osnet_ain_x1_0":
            self.input_shape = (256, 128)
            model = osnet_ain_x1_0(
                num_classes=1000, loss="softmax", pretrained=False, use_gpu=use_gpu
            )
            model.eval()
            model_path = resource_path(
                os.path.join(OSNET_REPO, "osnet_ain_ms_d_c.pth.tar")
            )
            load_pretrained_weights(model, model_path)
            self.model = model.to(self.device)
        else:
            raise ValueError("only osnet ain now")

        ##preprocessing
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]

        def gpu_compatible_transforms(tensor: torch.Tensor):
            normalize = T.Normalize(mean=pixel_mean, std=pixel_std)
            return normalize(tensor)

        self.preprocess = gpu_compatible_transforms

    def compute_features(
        self, img: ImageObject, detections: List[Detection]
    ) -> List[np.ndarray]:
        device = img.float32_tensor.device
        image_height, image_width = img.float32_tensor.shape[
            1:
        ]  # Assuming CxHxW format

        # Stack all bounding boxes into a tensor (x1, y1, x2, y2)
        bboxes = torch.tensor(
            [[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]] for d in detections],
            device=device,
        )

        # Crop and resize images
        cropped_images = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)  # Ensure integer coordinates
            x1, y1 = max(0, x1), max(0, y1)  # Clip to image dimensions
            x2, y2 = min(image_width, x2), min(image_height, y2)

            cropped_image = img.float32_tensor[
                :, y1:y2, x1:x2
            ]  # Crop image (CxH_cropxW_crop)

            # Ensure the cropped region is valid
            if cropped_image.numel() == 0:
                raise ValueError(f"Invalid crop region: {bbox}")

            # Resize the cropped image

            resized_image, new_height, new_width, _, _ = resize_for_padding(
                cropped_image, self.input_shape
            )
            padded_image = pad_image_to_target_size(resized_image, self.input_shape)
            cropped_images.append(padded_image[0])

        # Stack all resized images into a batch
        cropped_batch = torch.stack(
            cropped_images, dim=0
        )  # Resulting shape: (B, C, H, W)

        # Normalize the batch
        cropped_batch = self.preprocess(cropped_batch)

        # Extract features
        with torch.no_grad():
            res = self.model(cropped_batch)
        return res

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


def load_checkpoint(fpath):
    r"""Loads checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> from torchreid.utils import load_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")
    fpath = osp.abspath(osp.expanduser(fpath))
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else "cpu"
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location)
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
