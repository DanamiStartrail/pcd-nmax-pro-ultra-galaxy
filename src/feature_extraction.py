from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import graycomatrix, graycoprops

from .preprocessing import IMG_SIZE, bgr_to_rgb, preprocess, read_img
from .segmentation import segment, segment_steps

H_BINS = (8, 8, 8)  # Histogram HSV 3D -> 512 bin
GLCM_LEVELS = 32
GLCM_PROPS = ["contrast", "homogeneity", "energy", "correlation"]


def color_hist(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Ekstraksi histogram warna HSV 3D ternormalisasi (512 dimensi) pada area masker."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], mask, H_BINS, [0, 180, 0, 256, 0, 256]).flatten()
    return h / (h.sum() + 1e-12)


def glcm_feats(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Ekstraksi 4 fitur tekstur GLCM (32 level, 4 sudut) khusus piksel di dalam masker."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        return np.zeros(len(GLCM_PROPS), dtype=np.float64)

    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    g, m = g[y0:y1, x0:x1], mask[y0:y1, x0:x1]

    # Level 1..GLCM_LEVELS untuk objek makanan; 0 untuk area latar belakang
    q = (g.astype(np.float32) / 256.0 * (GLCM_LEVELS - 1)).astype(np.uint8) + 1
    q[m == 0] = 0

    P = graycomatrix(
        q,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=GLCM_LEVELS + 1,
        symmetric=True,
        normed=False,
    )
    P = P[1:, 1:, :, :].astype(np.float64)  # Buang level 0 (luar masker)
    s = P.sum(axis=(0, 1), keepdims=True)
    s[s == 0] = 1
    P = P / s

    return np.array([graycoprops(P, p).mean() for p in GLCM_PROPS], dtype=np.float64)


def extract(
    path: str | Path,
    use_blur: bool = True,
    use_seg: bool = True,
    return_mask: bool = False,
):
    """Ekstraksi fitur dari path file citra (sesuai fungsi extract di notebook)."""
    img = preprocess(read_img(path), use_blur=use_blur)
    mask = segment(img, use_seg=use_seg)
    out = (color_hist(img, mask), glcm_feats(img, mask))
    return (out + (mask,)) if return_mask else out


def extract_all(paths: list[str | Path], n_jobs: int = -1, **kw):
    """Ekstraksi fitur paralel untuk daftar path citra."""
    res = Parallel(n_jobs=n_jobs)(delayed(extract)(p, **kw) for p in paths)
    return np.stack([r[0] for r in res]), np.stack([r[1] for r in res])


def extract_from_bgr(
    img_bgr: np.ndarray, use_blur: bool = True, use_seg: bool = True
) -> dict:
    """Pipeline lengkap Model A untuk 1 citra BGR (mengembalikan tahapan visual & fitur)."""
    resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    blurred = preprocess(img_bgr, use_blur=use_blur)
    seg_info = segment_steps(blurred, use_seg=use_seg)
    mask_final = seg_info["mask_final"]

    segmented_bgr = cv2.bitwise_and(blurred, blurred, mask=mask_final)
    hist_vec = color_hist(blurred, mask_final)
    glcm_vec = glcm_feats(blurred, mask_final)

    glcm_dict = {
        "Contrast": round(float(glcm_vec[0]), 4),
        "Homogeneity": round(float(glcm_vec[1]), 4),
        "Energy": round(float(glcm_vec[2]), 4),
        "Correlation": round(float(glcm_vec[3]), 4),
    }

    return {
        "resized_rgb": bgr_to_rgb(resized),
        "blurred_rgb": bgr_to_rgb(blurred),
        "s_channel": seg_info["s_channel"],
        "mask_otsu": seg_info["mask_otsu"],
        "mask_final": mask_final,
        "fallback_triggered": seg_info["fallback_triggered"],
        "segmented_rgb": bgr_to_rgb(segmented_bgr),
        "hist_features": hist_vec,
        "glcm_vector": glcm_vec,
        "glcm_features": glcm_dict,
    }
