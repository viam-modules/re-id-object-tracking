# pylint: disable=consider-using-dict-items

"""
This module provides an Identifier class
that has an extractor and an encoder to compute and compare
face embeddings.
"""

import math
import os

import numpy as np
from PIL import Image
from viam.logging import getLogger

# from src.distance import cosine_distance, distance_norm_l1, distance_norm_l2
# from src.encoder import Encoder
# from src.tracker.face_id.extractor import FaceDetector
# from src.models import utils
# from src.utils import check_ir, dist_to_conf_sigmoid
from src.tracker.track import Track
from src.config.config import FaceIdConfig
from typing import Tuple
from src.image.image import ImageObject
from src.tracker.face_id.face_detector import FaceDetector

# from src.tracker.face_id.face_features_extractor import FaceFeaturesExtractor
from src.tracker.face_id.face_net.facenet_extractor import FaceFeaturesExtractor
import torch
from src.tracker.utils import save_tensor

LOGGER = getLogger(__name__)


class FaceIdentifier:
    """
    A class to identify known faces by computing and comparing embeddings to known embeddings.

    Attributes:
        model_name (str): The name of the face recognition model.
        extractor (Extractor): The extractor object for extracting faces from images.
        encoder (Encoder): The encoder object for computing face embeddings.
        picture_directory (str): The directory containing images of known faces.
        known_embeddings (dict): A dictionary of known face embeddings.
        distance (callable): The distance metric function for comparing embeddings.
        identification_threshold (float): The threshold for identifying a face as known.
        sigmoid_steepness (float): The steepness of the sigmoid function for confidence calculation.
        debug (bool): If True, enables debug mode.

    Methods:
        compute_known_embeddings():
            Computes embeddings for known faces from the picture directory.
        get_detections(img):
            Computes face detections and identifications in the input image.
        compare_face_to_known_faces(face, is_ir, unknown_label="unknown"):
            Encodes the face, calculates its distances with known faces, and returns the best match and the confidence. # pylint: disable=line-too-long
    """

    def __init__(
        self,
        cfg: FaceIdConfig,
        debug: bool = False,
    ):
        self.debug = debug
        self.detector: FaceDetector = FaceDetector(cfg)
        self.feature_extractor: FaceFeaturesExtractor = FaceFeaturesExtractor(cfg)
        self.cfg = cfg
        self.known_embeddings = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_known_embeddings()

    def get_match(self, img: ImageObject, track: Track) -> Tuple[str, float]:
        """
        Take a track and return a match if match is found.
        """
        face = self.detector.extract_face_from_track(img, track)
        if face is None:
            return None, None
        embedding = self.feature_extractor.get_embedding(face)
        return self.find_match(embedding)

    def find_match(self, embedding: torch.Tensor):
        best_label = None
        best_confidence = None
        min_cos_dist = float("inf")  # Initialize with a very large value

        for label, target_embeddings in self.known_embeddings.items():
            for target_embed in target_embeddings:
                cos_dist = self.feature_extractor.compute_distance(
                    embedding, target_embed, "cosine"
                )
                euc_dist = self.feature_extractor.compute_distance(
                    embedding, target_embed, "euclidean"
                )

                # Check if distances meet the thresholds
                if (
                    cos_dist < self.cfg.cosine_id_threshold.value
                    and euc_dist <= self.cfg.euclidean_id_threshold.value
                ):
                    # Update the best match if the cosine distance is smaller
                    if cos_dist < min_cos_dist:
                        min_cos_dist = cos_dist
                        best_label = label
                        best_confidence = 1 - cos_dist

        return best_label, best_confidence

    def compute_known_embeddings(self):
        """
        Computes embeddings for known faces from the picture directory.
        """

        path_to_known_faces = self.cfg.path_to_known_faces.value
        all_entries = os.listdir(path_to_known_faces)
        directories = [
            entry
            for entry in all_entries
            if os.path.isdir(os.path.join(path_to_known_faces, entry))
        ]
        for directory in directories:
            label_path = os.path.join(path_to_known_faces, directory)
            embeddings = []
            for file in os.listdir(label_path):
                if (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
                ):
                    im = Image.open(label_path + "/" + file).convert(
                        "RGB"
                    )  # convert in RGB because png are RGBA
                    img_array = np.array(im)
                    uint8_tensor = (
                        torch.from_numpy(img_array).permute(2, 0, 1).contiguous()
                    )
                    float32_tensor = uint8_tensor.to(dtype=torch.float32)

                    float32_tensor = float32_tensor.to(self.device)
                    face = self.detector.extract_face(float32_tensor)
                    if self.debug:
                        save_tensor(face, f"{directory}.jpeg")
                    # TODO: check if there is only one face here
                    embed = self.feature_extractor.get_embedding(face)
                    embeddings.append(embed)
                else:
                    LOGGER.warning(
                        "Ignoring unsupported file type: %s. Only .jpg, .jpeg, and .png files are supported.",  # pylint: disable=line-too-long
                        file,
                    )

            self.known_embeddings[directory] = embeddings
