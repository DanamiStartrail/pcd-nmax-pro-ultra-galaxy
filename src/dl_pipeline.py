"""Lazy TensorFlow/Keras inference pipeline for the notebook's MobileNetV2 model."""

from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "model_b_mobilenetv2.keras"
IMAGE_SIZE = (224, 224)


def preprocess_for_mobilenet(image: Image.Image) -> np.ndarray:
    """Return an RGB float32 batch in [0,255]; preprocessing is embedded in the model."""
    resized = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)[None, ...]


def load_mobilenet_model(model_path: Path = DEFAULT_MODEL_PATH):
    """Load a Keras artifact lazily; return None if model weights are not available."""
    path = Path(model_path)
    if not path.is_file():
        return None
    import tensorflow as tf

    return tf.keras.models.load_model(path)


def predict_mobilenet(model, image: Image.Image, classes: list[str]) -> dict:
    """Return ranked class probabilities and wall-clock inference duration."""
    started = time.perf_counter()
    batch = preprocess_for_mobilenet(image)
    probabilities = np.asarray(model.predict(batch, verbose=0))[0]
    elapsed = time.perf_counter() - started
    if probabilities.size != len(classes):
        raise ValueError("Jumlah output model tidak cocok dengan daftar kelas.")
    order = np.argsort(probabilities)[::-1][:3]
    return {
        "class": classes[int(order[0])],
        "confidence": float(probabilities[order[0]]),
        "top3": [(classes[int(index)], float(probabilities[index])) for index in order],
        "seconds": elapsed,
    }
