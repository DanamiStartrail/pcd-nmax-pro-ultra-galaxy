from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .preprocessing import IMG_SIZE, preprocess, read_img
from .segmentation import segment


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Menghitung metrik utama evaluasi klasifikasi (Akurasi & Macro F1-Score)."""
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def compute_iou_dice(gt_mask: np.ndarray, pred_mask: np.ndarray) -> tuple[float, float]:
    """Menghitung skor Intersection over Union (IoU) dan Dice Coefficient."""
    gt = gt_mask > 0
    pr = pred_mask > 0
    inter = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    iou = float(inter / (union + 1e-9))
    dice = float((2.0 * inter) / (gt.sum() + pr.sum() + 1e-9))
    return iou, dice


def seg_scores(
    image_paths: list[str | Path], mask_dir: str | Path | None, n_max: int = 100
) -> dict | None:
    """Evaluasi segmentasi HSV-Otsu terhadap masker anotasi ground-truth."""
    if mask_dir is None:
        return None

    mask_root = Path(mask_dir)
    ious, dices = [], []
    for p in image_paths:
        mp = mask_root / (Path(p).stem + ".png")
        if not mp.exists():
            continue
        gt = (
            cv2.resize(
                cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE),
                (IMG_SIZE, IMG_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )
            > 127
        )
        pr = segment(preprocess(read_img(p))) > 0
        iou, dice = compute_iou_dice(gt, pr)
        ious.append(iou)
        dices.append(dice)
        if len(ious) >= n_max:
            break

    if not ious:
        return None

    return {
        "n_samples": len(ious),
        "iou_mean": float(np.mean(ious)),
        "dice_mean": float(np.mean(dices)),
    }
