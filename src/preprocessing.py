"""Image preprocessing shared by the classical pipeline and the demo."""

import cv2
import numpy as np

IMAGE_SIZE = (224, 224)
BLUR_KERNEL = (5, 5)


def preprocess_image(image: np.ndarray, use_blur: bool = True) -> np.ndarray:
    """Resize a BGR image to 224x224 and optionally apply the notebook's Gaussian blur."""
    if image is None or image.size == 0:
        raise ValueError("Citra kosong atau tidak valid.")
    resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    if use_blur:
        return cv2.GaussianBlur(resized, BLUR_KERNEL, 0)
    return resized