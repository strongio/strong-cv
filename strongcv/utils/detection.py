from typing import Optional, Tuple, Dict

import os
import numpy as np


def get_detection_mask(detections: Dict, img_size: Tuple):
    """Create a binary detection mask

    Args:
        detections (Dict): Dictionary of detections (usually of the same frame).
        img_size (Tuple): Size of the image.

    Returns:
        mask: Binary array
    """
    assert len(img_size) >= 2
    mask = np.ones(img_size).astype(np.uint8)
    for detection_id, detection in detections.items():
        x0, y0, x1, y1 = detection["bbox"]
        assert x1 > x0
        assert y1 > y0
        mask[y0:y1, x0:x1] = 0
    return mask
