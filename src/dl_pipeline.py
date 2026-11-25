import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .preprocessing import IMG_SIZE

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_B_PATH = ROOT_DIR / "models" / "model_b_mobilenetv2.keras"

# Urutan kelas alfabetis default (10 kelas makanan Indonesia) jika model_a.joblib belum dimuat
DEFAULT_CLASSES = [
    "ayam_goreng",
    "bakso",
    "gado_gado",
    "gudeg",
    "ikan_goreng",
    "nasi_goreng",
    "rendang",
    "sate",
    "soto",
    "tempe_tahu",
]


def load_model_b(model_path: str | Path = DEFAULT_MODEL_B_PATH):
    """Memuat bobot Model B (MobileNetV2 fine-tuning .keras / .h5)."""
    path = Path(model_path)
    if not path.exists():
        return None

    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False)


def preprocess_for_mobilenet(image_pil: Image.Image) -> np.ndarray:
    """Menyiapkan batch (1, 224, 224, 3) float32 RGB [0..255].
    Catatan: layer preprocess_input sudah tertanam di dalam arsitektur model_b_mobilenetv2.keras.
    """
    rgb = np.array(image_pil.convert("RGB"))
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_BILINEAR)
    return resized[None, ...].astype("float32")


def predict_model_b(
    image_pil: Image.Image, model, class_names: list[str] | None = None
) -> dict:
    """Menjalankan inferensi Model B (MobileNetV2) pada 1 citra PIL."""
    t0 = time.perf_counter()
    batch = preprocess_for_mobilenet(image_pil)
    probs = model.predict(batch, verbose=0)[0]
    elapsed = time.perf_counter() - t0

    labels = class_names if class_names is not None else DEFAULT_CLASSES
    if len(labels) != len(probs):
        labels = [f"Kelas_{i}" for i in range(len(probs))]

    top_idx = int(np.argmax(probs))
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {"Kelas": labels[i], "Probabilitas (%)": round(float(probs[i]) * 100, 2)}
        for i in top3_idx
    ]

    return {
        "pred_class": labels[top_idx],
        "confidence": float(probs[top_idx]),
        "top3": top3,
        "all_probs": dict(zip(labels, [float(p) for p in probs])),
        "inference_time": elapsed,
    }
