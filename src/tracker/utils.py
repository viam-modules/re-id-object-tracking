import torch
import torch.nn.functional as F
from PIL import Image


def save_tensor(tensor: torch.Tensor, path):
    if tensor.is_cuda:
        tensor = tensor.cpu().detach()
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dtype != torch.uint8:
        tensor = tensor.to(dtype=torch.uint8).contiguous()

    # for PIL
    hwc_tensor = tensor.permute(1, 2, 0)
    # Convert to NumPy array
    array = hwc_tensor.numpy()

    # Create a PIL Image
    image = Image.fromarray(array)
    image.save(path)


def resize_for_padding(input_tensor, target_size):
    # Get the original dimensions
    height, width = input_tensor.shape[1:]
    if height == 0 or width == 0:
        raise ValueError(f"got input {input_tensor}")
    target_height, target_width = target_size

    # Calculate scaling factors to maintain aspect ratio
    scale_h = target_height / height
    scale_w = target_width / width
    scale = min(scale_h, scale_w)  # Use the smaller scale to preserve aspect ratio

    # Calculate new height and width
    new_height = int(height * scale)
    new_width = int(width * scale)

    # Resize the image while preserving aspect ratio
    resized_image = F.interpolate(
        input_tensor.unsqueeze(0),  # Add batch dimension for resizing
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )

    # Now we need to pad the image to the target size
    return resized_image, new_height, new_width, target_height, target_width


def pad_image_to_target_size(resized_image, target_size):
    _, _, h, w = resized_image.shape
    target_height, target_width = target_size

    # Calculate padding for each side
    pad_h = target_height - h
    pad_w = target_width - w

    # Pad evenly or add extra pixels on one side if necessary
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Pad the image
    padded_image = F.pad(
        resized_image,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0,
    )

    return padded_image


def padded_to_original_coordinates(
    padded_x, padded_y, original_size, resized_size, target_size
):
    """
    Converts coordinates from the padded image back to the original image coordinates.

    Parameters:
        padded_x (float): The x-coordinate in the padded image.
        padded_y (float): The y-coordinate in the padded image.
        original_size (tuple): The original image size (height, width).
        resized_size (tuple): The resized image size (height, width) after resizing.
        target_size (tuple): The target size of the padded image (height, width).

    Returns:
        (original_x, original_y): The coordinates in the original image's coordinate system.
    """
    original_height, original_width = original_size
    resized_height, resized_width = resized_size
    target_height, target_width = target_size

    # Calculate the padding for each dimension
    pad_top = (target_height - resized_height) // 2
    pad_left = (target_width - resized_width) // 2

    # Remove padding from the padded image coordinates
    unpadded_x = padded_x - pad_left
    unpadded_y = padded_y - pad_top

    # Scale the coordinates back to the original size
    scale_x = original_width / resized_width
    scale_y = original_height / resized_height

    original_x = unpadded_x * scale_x
    original_y = unpadded_y * scale_y

    return int(original_x), int(original_y)


def get_cropped_tensor(
    input_tensor, original_y1, original_y2, original_x1, original_x2, margin
):
    """
    Crops a tensor with the specified coordinates and adds a margin, ensuring the coordinates
    remain within valid bounds of the tensor.

    Parameters:
        input_tensor (torch.Tensor): The input tensor to crop (shape: [channels, height, width]).
        original_y1 (int): The top boundary of the crop.
        original_y2 (int): The bottom boundary of the crop.
        original_x1 (int): The left boundary of the crop.
        original_x2 (int): The right boundary of the crop.
        margin (int): The margin to add around the crop.

    Returns:
        torch.Tensor: The cropped tensor with the added margin.
    """
    # Get the height and width of the input tensor
    _, height, width = input_tensor.shape

    # Calculate the new crop coordinates with the margin
    y1 = max(0, original_y1 - margin)
    y2 = min(height, original_y2 + margin)
    x1 = max(0, original_x1 - margin)
    x2 = min(width, original_x2 + margin)

    # Crop the tensor
    cropped_tensor = input_tensor[:, y1:y2, x1:x2]

    return cropped_tensor
