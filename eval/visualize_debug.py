from pathlib import Path

import cv2


def draw_detection(image, bbox, confidence):
    """Draw a single bounding box with its confidence score on the image."""
    x1, y1, x2, y2 = map(int, bbox)

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw confidence text with background
    text = f"{confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Get text size for background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    # Draw background rectangle for text
    cv2.rectangle(
        image, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1
    )

    # Draw text
    cv2.putText(image, text, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)

    return image


def process_debug_images(debug_dir):
    """Process all debug images in the directory."""
    debug_path = Path(debug_dir)

    # Create output directory
    output_dir = debug_path / "visualized"
    output_dir.mkdir(exist_ok=True)

    # Process each PNG file
    for img_path in debug_path.glob("*.png"):
        # Parse filename to get boxes and confidences
        filename = img_path.stem  # Get filename without extension
        # Skip the timestamp part and get the boxes part
        boxes_part = filename.split("boxes_")[1]
        # Split into individual box-confidence pairs
        box_strings = boxes_part.split("-")

        # Read image
        image = cv2.imread(str(img_path))

        # Process each box
        for box_str in box_strings:
            # Parse box coordinates and confidence
            x1, y1, x2, y2, conf = map(float, box_str.split("_"))
            bbox = [x1, y1, x2, y2]

            # Draw detection on image
            image = draw_detection(image, bbox, conf)

        # Save output image
        output_path = output_dir / f"viz_{img_path.name}"
        cv2.imwrite(str(output_path), image)
        print(f"Processed: {img_path.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize debug images with bounding boxes"
    )
    parser.add_argument(
        "debug_dir", type=str, help="Path to the debug directory containing images"
    )

    args = parser.parse_args()

    process_debug_images(args.debug_dir)
