"""Saturation-channel Otsu segmentation with notebook-matched morphology."""

import cv2
import numpy as np

MORPH_KERNEL_SIZE = (7, 7)
MIN_FOREGROUND_RATIO = 0.05
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)


def saturation_channel(image_bgr: np.ndarray) -> np.ndarray:
    """Return the HSV saturation channel for an OpenCV BGR image."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Citra kosong atau tidak valid.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[..., 1]


def segment_food(image_bgr: np.ndarray, use_segmentation: bool = True) -> np.ndarray:
    """Return a uint8 {0,255} food mask using Otsu, opening, and closing."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Citra kosong atau tidak valid.")
    if not use_segmentation:
        return np.full(image_bgr.shape[:2], 255, dtype=np.uint8)

    saturation = saturation_channel(image_bgr)
    _, mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, KERNEL)
    if cv2.countNonZero(closed) / closed.size < MIN_FOREGROUND_RATIO:
        return np.full_like(closed, 255)
    return closed


def segmentation_stages(image_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Expose the intermediate stages shown in the Streamlit demo."""
    saturation = saturation_channel(image_bgr)
    _, otsu_mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    opening = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, KERNEL)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, KERNEL)
    if cv2.countNonZero(closing) / closing.size < MIN_FOREGROUND_RATIO:
        closing = np.full_like(closing, 255)
    result = cv2.bitwise_and(image_bgr, image_bgr, mask=closing)
    return {
        "saturation": saturation,
        "otsu_mask": otsu_mask,
        "opening": opening,
        "closing": closing,
        "segmented": result,
    }