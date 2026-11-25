import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .evaluation import metrics

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_A_PATH = ROOT_DIR / "models" / "model_a.joblib"


class PrototypeClassifier:
    """Skenario 1 Model A: d = w * Bhattacharyya(hist HSV) + (1 - w) * Euclidean(GLCM z-score)."""

    def __init__(self, w_grid=(0.3, 0.5, 0.7)):
        self.w_grid = w_grid
        self.w = None

    def fit(self, H: np.ndarray, G: np.ndarray, y: np.ndarray):
        self.mu, self.sd = G.mean(0), G.std(0) + 1e-9
        Z = (G - self.mu) / self.sd
        self.classes = np.unique(y)
        self.PH = np.stack([H[y == c].mean(0) for c in self.classes])
        self.PG = np.stack([Z[y == c].mean(0) for c in self.classes])
        return self

    def _dists(self, H: np.ndarray, G: np.ndarray):
        bc = np.sqrt(H) @ np.sqrt(self.PH).T
        db = np.sqrt(np.clip(1 - bc, 0, None))  # Bhattacharyya distance
        de = cdist((G - self.mu) / self.sd, self.PG)  # Euclidean pada GLCM z-score
        return db, de

    def combined_distance(self, H: np.ndarray, G: np.ndarray, w: float | None = None):
        w = self.w if w is None else w
        db, de = self._dists(H, G)
        return w * db + (1 - w) * de

    def tune_w(self, H: np.ndarray, G: np.ndarray, y: np.ndarray):
        db, de = self._dists(H, G)
        scores = {
            w: accuracy_score(y, self.classes[(w * db + (1 - w) * de).argmin(1)])
            for w in self.w_grid
        }
        self.w = max(scores, key=scores.get)
        return scores

    def predict(self, H: np.ndarray, G: np.ndarray, w: float | None = None):
        dists = self.combined_distance(H, G, w=w)
        return self.classes[dists.argmin(1)]


def feat_svm(H: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Gabungan akar histogram HSV (Hellinger) + 4 nilai GLCM."""
    return np.hstack([np.sqrt(H), G])


def fit_svm(Htr, Gtr, ytr, Hva, Gva, yva, seed: int = 42):
    """Pencarian hyperparameter terbaik SVM RBF pada data validasi."""
    sc = StandardScaler().fit(feat_svm(Htr, Gtr))
    Xtr, Xva = sc.transform(feat_svm(Htr, Gtr)), sc.transform(feat_svm(Hva, Gva))
    best, best_acc, params = None, -1.0, (1, "scale")
    for C in [1, 10, 100]:
        for g in ["scale", 1e-3, 1e-2]:
            m = SVC(C=C, gamma=g, kernel="rbf", random_state=seed).fit(Xtr, ytr)
            a = accuracy_score(yva, m.predict(Xva))
            if a > best_acc:
                best, best_acc, params = m, a, (C, g)
    return sc, best, params


def run_baseline(feats: dict, y_splits: dict):
    """Baseline: histogram warna citra utuh (Hellinger) + k-NN."""
    Htr, _ = feats["train"]
    Hva, _ = feats["val"]
    Hte, _ = feats["test"]
    ytr, yva, yte = y_splits["train"], y_splits["val"], y_splits["test"]

    best_k, best_a = 1, -1.0
    for k in [1, 3, 5, 7, 9]:
        a = accuracy_score(
            yva,
            KNeighborsClassifier(k, metric="euclidean")
            .fit(np.sqrt(Htr), ytr)
            .predict(np.sqrt(Hva)),
        )
        if a > best_a:
            best_k, best_a = k, a

    knn = KNeighborsClassifier(best_k).fit(np.sqrt(Htr), ytr)
    return metrics(yte, knn.predict(np.sqrt(Hte))), best_k


def load_model_a(model_path: str | Path = DEFAULT_MODEL_A_PATH) -> dict | None:
    """Memuat model_a.joblib ('proto', 'scaler', 'svm', 'classes')."""
    path = Path(model_path)
    if not path.exists():
        return None

    # Pastikan PrototypeClassifier tersedia di __main__ saat unpickle dari notebook
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "PrototypeClassifier"):
        setattr(main_mod, "PrototypeClassifier", PrototypeClassifier)

    return joblib.load(path)


def predict_model_a(hist_vec: np.ndarray, glcm_vec: np.ndarray, artifact: dict) -> dict:
    """Inferensi 1 citra untuk Model A Skenario 1 (Prototipe) dan Skenario 2 (SVM)."""
    t0 = time.perf_counter()
    classes = artifact["classes"]
    proto: PrototypeClassifier = artifact["proto"]
    scaler: StandardScaler = artifact["scaler"]
    svm: SVC = artifact["svm"]

    H = hist_vec[None, :]
    G = glcm_vec[None, :]

    # Skenario 1: Prototipe + jarak gabungan
    dists = proto.combined_distance(H, G)[0]
    s1_idx = int(np.argmin(dists))
    s1_class = classes[s1_idx]
    # Konversi jarak ke skor kemiripan relatif (softmax negatif jarak)
    inv_scores = np.exp(-dists * 5.0)
    s1_probs = inv_scores / inv_scores.sum()
    top3_s1_idx = np.argsort(s1_probs)[::-1][:3]
    top3_s1 = [
        {"Kelas": classes[i], "Skor Kemiripan (%)": round(float(s1_probs[i]) * 100, 2)}
        for i in top3_s1_idx
    ]

    # Skenario 2: SVM multi-kelas
    X_svm = scaler.transform(feat_svm(H, G))
    s2_idx = int(svm.predict(X_svm)[0])
    s2_class = classes[s2_idx]

    top3_s2 = []
    if hasattr(svm, "decision_function"):
        dec = svm.decision_function(X_svm)[0]
        exp_dec = np.exp(dec - np.max(dec))
        s2_probs = exp_dec / exp_dec.sum()
        top3_s2_idx = np.argsort(s2_probs)[::-1][:3]
        top3_s2 = [
            {
                "Kelas": classes[i],
                "Skor Relatif (%)": round(float(s2_probs[i]) * 100, 2),
            }
            for i in top3_s2_idx
        ]

    elapsed = time.perf_counter() - t0
    return {
        "s1_class": s1_class,
        "s1_distance": round(float(dists[s1_idx]), 4),
        "s1_weight_w": float(proto.w),
        "s1_top3": top3_s1,
        "s2_class": s2_class,
        "s2_top3": top3_s2,
        "inference_time": elapsed,
    }
