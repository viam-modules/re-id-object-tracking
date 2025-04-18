from typing import Optional

import numpy as np
import torch
from viam.services.vision import Detection


class Track:
    def __init__(
        self,
        track_id,
        bbox,
        feature_vector,
        distance,
        label=None,
        is_candidate: bool = False,
    ):
        """
        Initialize a track with a unique ID, bounding box, and re-id feature vector.

        :param track_id: Unique identifier for the track.
        :param bbox: Bounding box coordinates [x1, y1, x2, y2].
        :param feature_vector: a CUDA torch Tensor. Feature vector for re-id matching.
        """
        self.track_id = track_id
        self.bbox = np.array(bbox)
        self.predicted_bbox = np.array(bbox)
        self.feature_vector = feature_vector
        self.age = 0  # Time since the last update
        self.history = [np.array(bbox)]  # Stores past bounding boxes for this track
        self.velocity = np.array([0, 0, 0, 0])  # Initial velocity (no motion)
        self.distance = distance

        self.label = label

        self.label_from_reid: Optional[str] = None
        self.conf_from_reid = 0

        self.label_from_faceid: Optional[str] = None
        self.conf_from_faceid: Optional[str] = None

        self.persistence: int = 0
        self.is_candidate: bool = is_candidate
        self._is_detected: bool = True

    def __eq__(self, other) -> bool:
        """
        To test serialization, so just on unique id, bbox, feature vector
        """
        if not isinstance(other, Track):
            return False
        return (
            self.track_id == other.track_id
            and np.array_equal(self.bbox, other.bbox)
            and np.array_equal(self.feature_vector, other.feature_vector)
        )

    def update(self, bbox, feature_vector: torch.Tensor, distance):
        """
        Update the track with a new bounding box and feature vector.
        Also updates the velocity based on the difference between the last and current bbox.

        :param bbox: New bounding box coordinates.
        :param feature_vector: New feature vector.
        """
        self.distance = distance
        bbox = np.array(bbox)
        self.velocity = bbox - self.bbox  # Update velocity
        self.bbox = bbox
        self.feature_vector = feature_vector
        self.age = 0
        self.history.append(bbox)
        self.predicted_bbox = self.predict()

    def predict(self):
        """
        Predict the next position based on the current velocity and last known position.

        :return: Predicted bounding box coordinates.
        """
        predicted_bbox = self.bbox + self.velocity
        return predicted_bbox

    def change_track_id(self, new_track_id: str):
        self.track_id = new_track_id

    def increment_persistence(self):
        self.persistence += 1

    def get_persistence(self):
        return self.persistence

    def increment_age(self):
        """
        Increment the age and time since update for the track.
        If not updated, the prediction is updated using the velocity.
        """
        self.age += 1
        self.bbox = self.predict()  # Update the bbox with the predicted one

    def iou(self, bbox):
        """
        Calculate Intersection over Union (IoU) between this track's bbox and another bbox.

        :param bbox: Bounding box to compare with.

            (x1, y1) --------------+
                |                  |
                |                  |
                |                  |
                +-------------- (x2, y2)

        :return: IoU score.
        """
        x1_t, y1_t, x2_t, y2_t = self.predicted_bbox
        x1_o, y1_o, x2_o, y2_o = bbox

        # Determine the coordinates of the intersection rectangle
        x1_inter = max(x1_t, x1_o)
        y1_inter = max(y1_t, y1_o)
        x2_inter = min(x2_t, x2_o)
        y2_inter = min(y2_t, y2_o)

        # Compute the area of intersection
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        # Compute the area of both the prediction and ground-truth rectangles
        track_area = (x2_t - x1_t) * (y2_t - y1_t)
        other_area = (x2_o - x1_o) * (y2_o - y1_o)

        # Compute the Intersection over Union (IoU)
        union_area = track_area + other_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0

        return iou

    def feature_distance(self, feature_vector):
        """
        Calculate the distance between this track's feature vector and another feature vector.

        :param feature_vector: Feature vector to compare with.
        :return: Distance (e.g., Euclidean distance).
        """
        return np.linalg.norm(self.feature_vector - feature_vector)

    def get_detection(
        self,
        crop_region,
        min_persistence=None,
        original_image_width=None,
        original_image_height=None,
    ) -> Detection:
        if self.is_candidate:
            if min_persistence is None:
                raise ValueError(
                    "Need to pass persistence in argument to get track candidate"
                )
            class_name = self._get_label(min_persistence)
        else:
            class_name = self.track_id
            label = self._get_label(min_persistence)
            if label != self.track_id:
                class_name += f"  (label: {label})"

        # Convert bbox from cropped coordinates to original image coordinates
        x_min, y_min, x_max, y_max = self.bbox

        if crop_region:
            # Adjust coordinates based on crop region
            x_offset = int(crop_region.get("x1_rel", 0.0) * original_image_width)
            y_offset = int(crop_region.get("y1_rel", 0.0) * original_image_height)

            # Convert back to original image coordinates
            x_min = min(original_image_width - 1, x_min + x_offset)
            y_min = min(original_image_height - 1, y_min + y_offset)
            x_max = min(original_image_width - 1, x_max + x_offset)
            y_max = min(original_image_height - 1, y_max + y_offset)

        return Detection(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            confidence=1 - self.distance,
            class_name=class_name,
        )

    def serialize(self):
        return (
            self.track_id,
            self.bbox.tobytes(),
            self.feature_vector.cpu().detach().numpy().tobytes(),
        )

    def add_label(self, label_from_reid):
        self.label_from_reid = label_from_reid

    def get_embedding(self):
        return self.feature_vector

    def relabel(self, new_label):
        self.label = new_label

    def relabel_reid_label(self, label: str):
        self.label_from_reid = label

    def relabel_faceid_label(self, label: str):
        self.label_from_faceid = label

    def has_label(self) -> bool:
        return self.label is not None

    def _get_label(self, min_persistence=None):
        if self.is_candidate:
            return self.progress_bar(self.persistence, min_persistence)
        if self.label is not None:
            return self.label
        if self.label_from_faceid is not None:
            return self.label_from_faceid
        if self.label_from_reid is not None:
            return self.label_from_reid
        return self.track_id

    def progress_bar(self, current_progress, min_persistence):
        """
        Generates an emoji-based progress bar.

        :param current_progress: The current progress level (integer)
        :param max_persistence: The maximum progress level
        :return: A string with the emoji progress bar
        """
        progress_char = " *"  # Solid square
        empty_char = " ."

        # Ensure current_progress does not exceed max_persistence
        if self.persistence > min_persistence:
            current_progress = min_persistence

        # Create the progress bar
        progress_bar = progress_char * current_progress + empty_char * (
            min_persistence - current_progress
        )

        return "tracking" + f"  [{progress_bar}]"

    def set_is_detected(self):
        self._is_detected = True

    def unset_is_detected(self):
        self._is_detected = False

    def is_detected(self):
        return self._is_detected

    def get_all(self):
        answer = {
            "face_id_label": self.label_from_faceid,
            "face_id_conf": self.conf_from_faceid,
            "re_id_label": self.label_from_reid,
            "re_id_conf": self.conf_from_reid,
            "manual_label": self.label,
        }
        return answer
