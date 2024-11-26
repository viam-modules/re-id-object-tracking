class Detection:
    def __init__(self, bbox, score, category):
        """
        Initialize a detection with a bounding box, confidence score, and category.

        :param bbox: Bounding box [x1, y1, x2, y2].
        :param score: Confidence score for the detection.
        :param category: Category of the detected object as a string.
        """
        self.bbox = bbox
        self.score = score
        self.category = category

    def iou(self, other):
        """
        Calculate the Intersection over Union (IoU) between this detection's bounding box
        and another detection's bounding box.

        :param other: Another Detection instance.
        :return: IoU as a float value.
        """
        x1_self, y1_self, x2_self, y2_self = self.bbox
        x1_other, y1_other, x2_other, y2_other = other.bbox

        # Determine the coordinates of the intersection rectangle
        x1_inter = max(x1_self, x1_other)
        y1_inter = max(y1_self, y1_other)
        x2_inter = min(x2_self, x2_other)
        y2_inter = min(y2_self, y2_other)

        # Compute the area of intersection
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

        # Compute the area of both bounding boxes
        self_area = (x2_self - x1_self) * (y2_self - y1_self)
        other_area = (x2_other - x1_other) * (y2_other - y1_other)

        # Compute the Intersection over Union (IoU)
        union_area = self_area + other_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0

        return iou

    def __repr__(self):
        """
        Return a string representation of the detection for easy printing.
        """
        return f"Detection(bbox={self.bbox}, score={self.score:.2f}, category='{self.category}')"
