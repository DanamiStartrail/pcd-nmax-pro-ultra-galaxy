"""Classification and segmentation metrics plus experiment ablation definitions."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ABLATIONS = {
    "Ablasi 1: tanpa praproses": {"use_blur": False, "use_segmentation": True},
    "Ablasi 2: tanpa segmentasi": {"use_blur": True, "use_segmentation": False},
    "Ablasi 3: tanpa fitur warna": {"use_color": False, "use_texture": True},
    "Ablasi 4: tanpa fitur tekstur": {"use_color": True, "use_texture": False},
}


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Return accuracy and macro F1 as fractions."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def segmentation_metrics(
    predicted_mask: np.ndarray, ground_truth_mask: np.ndarray
) -> dict[str, float]:
    """Compute binary IoU and Dice after converting both masks to boolean."""
    predicted = np.asarray(predicted_mask) > 0
    truth = np.asarray(ground_truth_mask) > 0
    if predicted.shape != truth.shape:
        raise ValueError("Dimensi mask prediksi dan ground truth harus sama.")
    intersection = np.logical_and(predicted, truth).sum()
    union = np.logical_or(predicted, truth).sum()
    return {
        "iou": float(intersection / (union + 1e-9)),
        "dice": float(2 * intersection / (predicted.sum() + truth.sum() + 1e-9)),
    }