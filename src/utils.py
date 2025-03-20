# pylint: disable=missing-module-docstring
import logging
import os
import sys
from io import BytesIO
from typing import Union

import numpy as np
from PIL import Image
from tabulate import tabulate
from viam.logging import getLogger
from viam.media.video import CameraMimeType, ViamImage

LOGGER = getLogger(__name__)
SUPPORTED_IMAGE_TYPE = [
    CameraMimeType.JPEG,
    CameraMimeType.PNG,
    CameraMimeType.VIAM_RGBA,
]
LIBRARY_SUPPORTED_FORMATS = ["JPEG", "PNG", "VIAM_RGBA"]


def decode_image(image: Union[Image.Image, ViamImage]) -> np.ndarray:
    """decode image to BGR numpy array

    Args:
        raw_image (Union[Image.Image, RawImage])

    Returns:
        np.ndarray: RGB numpy array
    """

    if isinstance(image, ViamImage):
        if image.mime_type not in SUPPORTED_IMAGE_TYPE:
            LOGGER.error(
                "Unsupported image type: %s. Supported types are %s.",
                image.mime_type,
                SUPPORTED_IMAGE_TYPE,
            )

            raise ValueError(f"Unsupported image type: {image.mime_type}.")

        image = Image.open(BytesIO(image.data), formats=LIBRARY_SUPPORTED_FORMATS)

    # No matter what it was originally, `image` now has type Image.Image.
    res = image.convert("RGB")
    return np.array(res)


def resource_path(relative_path):
    """
    Get the absolute path to a resource file, considering different environments.

    Args:
        relative_path (str): The relative path to the resource file.

    Returns:
        str: The absolute path to the resource file.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS  # pylint: disable=protected-access,no-member
    except Exception:  # pylint: disable=broad-exception-caught
        base_path = os.path.abspath(".")

    return os.path.join(base_path, "src", "models", relative_path)


def log_tracks_info(updated_tracks_ids, new_tracks_ids, lost_tracks_ids):
    # Creating the table data
    table_data = [
        ["Updated Tracks", ", ".join(updated_tracks_ids)],
        ["New Tracks", ", ".join(new_tracks_ids)],
        ["Lost Tracks", ", ".join(lost_tracks_ids)],
    ]

    table_message = tabulate(
        table_data, headers=["Category", "Track IDs"], tablefmt="grid"
    )
    LOGGER.info("\n\n" + table_message + "\n\n")


def log_cost_matrix(cost_matrix, track_ids, iteration_number):
    detection_headers = [f"detection_{i+1}" for i in range(cost_matrix.shape[1])]

    table_data = [[track_id] + list(row)
                  for track_id, row in zip(track_ids, cost_matrix)]

    table_message = tabulate(
        table_data, headers=["Track ID"] + detection_headers, tablefmt="grid"
    )

    logger = logging.getLogger("TrackDetectionLogger")
    logger.info(f"Iteration number: {iteration_number}")
    logger.info(table_message + "\n\n")
