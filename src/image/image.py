from viam.media.video import ViamImage
from typing import Dict, List
import numpy as np
import torch
from PIL import Image
import io


def get_tensor_from_np_array(np_array: np.ndarray) -> torch.Tensor:
    """
    returns an RGB tensor
    """
    uint8_tensor = (
        torch.from_numpy(np_array).permute(2, 0, 1).contiguous()
    )  # -> to (C, H, W)
    float32_tensor = uint8_tensor.to(dtype=torch.float32)
    return uint8_tensor, float32_tensor


class ImageObject:
    def __init__(self, viam_image: ViamImage, pil_image: Image = None, device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if pil_image is not None:
            self.pil_image = pil_image
        if viam_image is not None:
            self.viam_image = viam_image
            self.pil_image = Image.open(io.BytesIO(viam_image.data)).convert(
                "RGB"
            )  # -> in (H, W, C)
        self.np_array = np.array(self.pil_image, dtype=np.uint8)
        uint8_tensor, float32_tensor = get_tensor_from_np_array(self.np_array)
        self.uint8_tensor = uint8_tensor.to(self.device)
        self.float32_tensor = float32_tensor.to(self.device)

    # def add_encoding(mime_type: CameraMimeType, bytes):
    #     self.encodings[mime_type] = bytes

    # def get_cropped_tensor():
    #     pass

    # def get_viam_image():
    #     pass

    # def get_np_array(self):
    #     if self.np_array is not None:

    #     if self.encodings:

    # def get_tensor(self):
    #     if self.tensor is not None:
    #         return self.tensor
    #     else:

    # def get_pil_image(self):
    #     if self.pil_image is not None:
    #         return
    #     return Image.open(io.BytesIO(image_bytes)).convert("RGB")
