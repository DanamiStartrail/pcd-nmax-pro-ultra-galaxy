"""HSV histogram and masked GLCM features matching the experiment notebook."""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

HSV_BINS = (8, 8, 8)
GLCM_LEVELS = 32
GLCM_PROPERTIES = ("contrast", "homogeneity", "energy", "correlation")
GLCM_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)


def color_histogram(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a normalized 512-bin HSV histogram under the supplied mask."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv], [0, 1, 2], mask, HSV_BINS, [0, 180, 0, 256, 0, 256]
    ).flatten()
    return histogram / (histogram.sum() + 1e-12)


def glcm_features(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return mean contrast, homogeneity, energy, and correlation over four angles."""
    if mask.shape != image_bgr.shape[:2]:
        raise ValueError("Dimensi mask harus sama dengan dimensi citra.")
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise ValueError("Mask kosong; fitur GLCM tidak dapat dihitung.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    gray_crop = gray[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]
    quantized = (
        (gray_crop.astype(np.float32) / 256 * (GLCM_LEVELS - 1)).astype(np.uint8)
        + 1
    )
    quantized[mask_crop == 0] = 0
    matrix = graycomatrix(
        quantized,
        distances=[1],
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS + 1,
        symmetric=True,
        normed=False,
    )
    matrix = matrix[1:, 1:, :, :].astype(np.float64)
    totals = matrix.sum(axis=(0, 1), keepdims=True)
    totals[totals == 0] = 1
    matrix /= totals
    return np.asarray(
        [graycoprops(matrix, prop).mean() for prop in GLCM_PROPERTIES],
        dtype=np.float64,
    )


def extract_features(
    image_bgr: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (HSV histogram, four-element GLCM feature vector)."""
    return color_histogram(image_bgr, mask), glcm_features(image_bgr, mask)
