import cv2
import numpy as np

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def segment_steps(img_bgr: np.ndarray, use_seg: bool = True) -> dict:
    """Mengembalikan tahapan segmentasi HSV-Otsu & morfologi untuk visualisasi UI."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[..., 1]

    if not use_seg:
        full_mask = np.full(img_bgr.shape[:2], 255, dtype=np.uint8)
        return {
            "s_channel": s_channel,
            "mask_otsu": full_mask,
            "mask_final": full_mask,
            "fallback_triggered": False,
        }

    _, mask_otsu = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_open = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, KERNEL)
    mask_final = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, KERNEL)

    fallback = False
    if mask_final.mean() / 255.0 < 0.05:
        mask_final = np.full_like(mask_final, 255)
        fallback = True

    return {
        "s_channel": s_channel,
        "mask_otsu": mask_otsu,
        "mask_final": mask_final,
        "fallback_triggered": fallback,
    }


def segment(img_bgr: np.ndarray, use_seg: bool = True) -> np.ndarray:
    """Otsu pada kanal saturasi HSV + morfologi (opening lalu closing). Return mask uint8 {0,255}."""
    return segment_steps(img_bgr, use_seg=use_seg)["mask_final"]
