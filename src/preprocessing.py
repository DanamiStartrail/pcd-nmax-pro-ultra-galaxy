from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMG_SIZE = 224


def read_img(path: str | Path) -> np.ndarray:
    """Membaca citra dari path ke format BGR uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Gagal membaca citra dari path: {path}")
    return img


def pil_to_bgr(image_pil: Image.Image) -> np.ndarray:
    """Mengonversi objek PIL Image menjadi array NumPy BGR (format OpenCV)."""
    rgb = np.array(image_pil.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Mengonversi array BGR menjadi RGB untuk ditampilkan di Streamlit/Matplotlib."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def preprocess(img_bgr: np.ndarray, use_blur: bool = True) -> np.ndarray:
    """Tahap 1 Model A: Resize ke 224x224 dan Gaussian Blur (5x5)."""
    resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    if use_blur:
        return cv2.GaussianBlur(resized, (5, 5), 0)
    return resized
