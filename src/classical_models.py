"""Loading and inference helpers for notebook-trained classical models."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"


class PrototypeClassifier:
    """Notebook-compatible class prototype classifier for HSV and GLCM features."""

    def __init__(self, w_grid: tuple[float, ...] = (0.3, 0.5, 0.7)):
        self.w_grid = w_grid
        self.w: float | None = None

    def fit(self, histograms: np.ndarray, textures: np.ndarray, labels: np.ndarray):
        self.mu = textures.mean(axis=0)
        self.sd = textures.std(axis=0) + 1e-9
        z_textures = (textures - self.mu) / self.sd
        self.classes = np.unique(labels)
        self.prototype_histograms = np.stack(
            [histograms[labels == label].mean(axis=0) for label in self.classes]
        )
        self.prototype_textures = np.stack(
            [z_textures[labels == label].mean(axis=0) for label in self.classes]
        )
        return self

    def _distances(self, histograms: np.ndarray, textures: np.ndarray):
        bhattacharyya_coefficient = np.sqrt(histograms) @ np.sqrt(
            self.prototype_histograms
        ).T
        histogram_distance = np.sqrt(
            np.clip(1 - bhattacharyya_coefficient, 0, None)
        )
        texture_distance = cdist(
            (textures - self.mu) / self.sd, self.prototype_textures
        )
        return histogram_distance, texture_distance

    def tune_w(
        self, histograms: np.ndarray, textures: np.ndarray, labels: np.ndarray
    ) -> dict[float, float]:
        """Select the color/texture weight with highest validation accuracy."""
        color_distance, texture_distance = self._distances(histograms, textures)
        scores = {
            weight: accuracy_score(
                labels,
                self.classes[
                    (weight * color_distance + (1 - weight) * texture_distance).argmin(axis=1)
                ],
            )
            for weight in self.w_grid
        }
        self.w = max(scores, key=scores.get)
        return scores

    def predict(
        self, histograms: np.ndarray, textures: np.ndarray, w: float | None = None
    ):
        weight = self.w if w is None else w
        if weight is None:
            raise ValueError("Bobot gabungan belum ditentukan; setel `w` terlebih dahulu.")
        if not 0 <= weight <= 1:
            raise ValueError("Bobot warna harus berada pada rentang 0 sampai 1.")
        color_distance, texture_distance = self._distances(histograms, textures)
        return self.classes[
            (weight * color_distance + (1 - weight) * texture_distance).argmin(axis=1)
        ]


def load_classical_models(model_dir: Path = MODEL_DIR) -> dict[str, Any] | None:
    """Load model_a.joblib, returning None when optional weights are not installed."""
    artifact = Path(model_dir) / "model_a.joblib"
    if not artifact.is_file():
        return None
    payload = joblib.load(artifact)
    if not isinstance(payload, dict) or not {
        "proto", "scaler", "svm", "classes"
    } <= payload.keys():
        raise ValueError(f"Format artefak model klasik tidak valid: {artifact}")
    return payload


def load_baseline(model_dir: Path = MODEL_DIR) -> Any | None:
    """Load the optional notebook-trained k-NN baseline."""
    artifact = Path(model_dir) / "baseline_knn.joblib"
    return joblib.load(artifact) if artifact.is_file() else None


def svm_feature_vector(histogram: np.ndarray, texture: np.ndarray) -> np.ndarray:
    """Build the notebook feature vector: square-root HSV bins followed by GLCM."""
    return np.hstack([np.sqrt(histogram), texture])


def predict_model_a(
    payload: dict[str, Any], histogram: np.ndarray, texture: np.ndarray
) -> dict[str, str]:
    """Return predictions for prototype S1 and SVM S2 from a notebook bundle."""
    classes = np.asarray(payload["classes"])
    proto = payload["proto"]
    s1 = proto.predict(histogram[None, :], texture[None, :])
    svm_features = svm_feature_vector(histogram, texture)[None, :]
    s2 = payload["svm"].predict(payload["scaler"].transform(svm_features))
    return {
        "Skenario 1 (Prototipe)": str(classes[int(s1[0])]),
        "Skenario 2 (SVM)": str(classes[int(s2[0])]),
    }


def predict_baseline(model: Any, histogram: np.ndarray, classes: list[str]) -> str:
    """Predict with notebook's global HSV histogram k-NN baseline."""
    prediction = model.predict(np.sqrt(histogram[None, :]))[0]
    return str(classes[int(prediction)])
