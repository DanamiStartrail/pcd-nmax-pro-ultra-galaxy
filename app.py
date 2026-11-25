"""Streamlit demo for the NMAX food classification research project."""

from pathlib import Path
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.classical_models import load_classical_models, predict_model_a
from src.dl_pipeline import load_mobilenet_model, predict_mobilenet
from src.feature_extraction import extract_features
from src.preprocessing import preprocess_image
from src.segmentation import segment_food, segmentation_stages

ROOT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT_DIR / "data" / "samples"
MODEL_DIR = ROOT_DIR / "models"
SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png"}


@st.cache_resource
def cached_classical_models(model_dir: str):
    return load_classical_models(Path(model_dir))


@st.cache_resource
def cached_mobilenet_model(model_path: str):
    return load_mobilenet_model(Path(model_path))


def available_samples() -> list[Path]:
    return sorted(
        path for path in SAMPLES_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGES
    )


def load_rgb_image(uploaded_file, sample_path: Path | None) -> tuple[Image.Image, str]:
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        return image, uploaded_file.name
    if sample_path is not None:
        with Image.open(sample_path) as source:
            return source.convert("RGB"), sample_path.name
    raise ValueError("Pilih atau unggah sebuah citra.")


def discover_classes(model_bundle) -> list[str]:
    if model_bundle is not None:
        return [str(label) for label in model_bundle["classes"]]
    train_dir = ROOT_DIR / "data" / "train"
    excluded = {"pizza", "burger", "french fries", "french_fries", "frenchfries"}
    return sorted(
        path.name for path in train_dir.iterdir()
        if path.is_dir() and path.name.lower() not in excluded
    ) if train_dir.is_dir() else []


def plot_hsv_histogram(histogram: np.ndarray):
    figure, axis = plt.subplots(figsize=(9, 2.8))
    axis.bar(np.arange(histogram.size), histogram, width=1.0, color="#2878b5")
    axis.set(title="Histogram HSV 3D (512 bin)", xlabel="Indeks bin", ylabel="Proporsi")
    figure.tight_layout()
    return figure


def render_classical(image_rgb: np.ndarray, model_bundle, classes: list[str]):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    resized = preprocess_image(image_bgr, use_blur=False)
    processed = preprocess_image(image_bgr, use_blur=True)
    stages = segmentation_stages(processed)
    histogram, texture = extract_features(processed, stages["closing"])

    st.subheader("Tahapan pengolahan citra")
    stage_images = [
        ("Resize 224×224", resized),
        ("Gaussian Blur 5×5", processed),
        ("Kanal S HSV", stages["saturation"]),
        ("Mask Otsu", stages["otsu_mask"]),
        ("Morfologi Opening", stages["opening"]),
        ("Morfologi Closing", stages["closing"]),
        ("Hasil segmentasi", cv2.cvtColor(stages["segmented"], cv2.COLOR_BGR2RGB)),
    ]
    columns = st.columns(4)
    for index, (title, stage_image) in enumerate(stage_images):
        display_image = (
            cv2.cvtColor(stage_image, cv2.COLOR_BGR2RGB)
            if stage_image.ndim == 3 and title != "Hasil segmentasi"
            else stage_image
        )
        columns[index % 4].image(display_image, caption=title, use_container_width=True)

    st.subheader("Fitur warna dan tekstur")
    left, right = st.columns([1.5, 1])
    with left:
        st.pyplot(plot_hsv_histogram(histogram), clear_figure=True)
    with right:
        st.dataframe(
            pd.DataFrame({"Fitur GLCM": ["Contrast", "Homogeneity", "Energy", "Correlation"],
                          "Nilai": texture}),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Prediksi Model A")
    if model_bundle is None:
        st.info("Bobot `models/model_a.joblib` belum tersedia; prediksi klasik aktif setelah artefak ditambahkan.")
    else:
        predictions = predict_model_a(model_bundle, histogram, texture)
        for scenario, label in predictions.items():
            st.metric(scenario, label)


def render_deep_learning(image: Image.Image, classes: list[str]):
    model_path = MODEL_DIR / "model_b_mobilenetv2.keras"
    model = cached_mobilenet_model(str(model_path))
    if model is None:
        st.info("Bobot `models/model_b_mobilenetv2.keras` belum tersedia; tambahkan model Keras hasil fine-tuning untuk mengaktifkan prediksi.")
        return None
    if not classes:
        st.info("Daftar kelas belum tersedia. Letakkan dataset pada `data/train/<kelas>/` atau tambahkan artefak Model A.")
        return None

    result = predict_mobilenet(model, image, classes)
    left, right = st.columns(2)
    left.metric("Prediksi utama", result["class"])
    right.metric("Keyakinan", f"{result['confidence']:.1%}")
    st.caption(f"Waktu inferensi: {result['seconds'] * 1000:.1f} ms")
    top3 = pd.DataFrame(result["top3"], columns=["Kelas", "Probabilitas"])
    st.bar_chart(top3.set_index("Kelas"))
    st.dataframe(
        top3.assign(Probabilitas=top3["Probabilitas"].map(lambda value: f"{value:.2%}")),
        hide_index=True,
        use_container_width=True,
    )
    return result


def render_research_summary():
    rows = [
        ("Baseline (HSV utuh + k-NN)", "Akurasi / Macro F1"),
        ("Model A - Skenario 1 (prototipe + jarak)", "Akurasi / Macro F1"),
        ("Model A - Skenario 2 (SVM)", "Akurasi / Macro F1"),
        ("Ablasi 1 - tanpa praproses (S1 dan S2)", "Akurasi / Macro F1"),
        ("Ablasi 2 - tanpa segmentasi (S1 dan S2)", "Akurasi / Macro F1"),
        ("Ablasi 3 - tanpa fitur warna (S1 dan S2)", "Akurasi / Macro F1"),
        ("Ablasi 4 - tanpa fitur tekstur (S1 dan S2)", "Akurasi / Macro F1"),
        ("Evaluasi segmentasi", "IoU / Dice"),
        ("Model B - MobileNetV2 fine-tuning", "Akurasi / Macro F1"),
        ("Inferensi Model A / Model B", "detik per citra"),
    ]
    frame = pd.DataFrame(rows, columns=["Skenario", "Metrik"])
    frame["Hasil"] = "Tidak tercatat pada output notebook"
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.info(
        "Notebook yang disertakan berhenti pada FileNotFoundError saat memuat dataset; "
        "sel evaluasi belum dijalankan dan tidak menyimpan angka metrik. Tabel ini sengaja "
        "tidak mengisi atau mengarang hasil."
    )


def main():
    st.set_page_config(page_title="PCD NMAX | Klasifikasi Makanan", page_icon="🍲", layout="wide")
    st.title("Klasifikasi Makanan Indonesia di Kantin")
    st.caption("Komparasi Pengolahan Citra Klasik (Model A) dan MobileNetV2 (Model B) — Kelompok NMAX")

    uploaded_file = st.file_uploader("Unggah citra makanan", type=["jpg", "jpeg", "png"])
    samples = available_samples()
    selected_sample = None
    if uploaded_file is None and samples:
        selected_name = st.selectbox("Atau pilih citra dari data/samples/", [path.name for path in samples])
        selected_sample = next(path for path in samples if path.name == selected_name)

    if uploaded_file is None and selected_sample is None:
        st.info("Unggah citra untuk memulai. Contoh yang tersedia di `data/samples/` akan muncul otomatis.")
        image_rgb = None
        image_pil = None
    else:
        image_pil, image_name = load_rgb_image(uploaded_file, selected_sample)
        image_rgb = np.asarray(image_pil)
        st.image(image_rgb, caption=image_name, width=350)

    model_bundle = cached_classical_models(str(MODEL_DIR))
    classes = discover_classes(model_bundle)
    tab_a, tab_b, tab_compare, tab_research = st.tabs(
        ["Model A · PCD Klasik", "Model B · MobileNetV2", "Perbandingan", "Ringkasan Riset"]
    )

    with tab_a:
        if image_rgb is None:
            st.info("Pilih citra terlebih dahulu untuk melihat pemrosesan klasik.")
        else:
            render_classical(image_rgb, model_bundle, classes)

    with tab_b:
        if image_pil is None:
            st.info("Pilih citra terlebih dahulu untuk menjalankan MobileNetV2.")
        else:
            render_deep_learning(image_pil, classes)

    with tab_compare:
        if image_rgb is None:
            st.info("Pilih citra terlebih dahulu untuk membandingkan model.")
        else:
            classical_result = None
            if model_bundle is not None:
                started = time.perf_counter()
                bgr = preprocess_image(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                mask = segment_food(bgr)
                histogram, texture = extract_features(bgr, mask)
                classical_result = predict_model_a(model_bundle, histogram, texture)
                classical_seconds = time.perf_counter() - started
            deep_model_available = (MODEL_DIR / "model_b_mobilenetv2.keras").is_file()
            dl_result = None
            if deep_model_available:
                model = cached_mobilenet_model(str(MODEL_DIR / "model_b_mobilenetv2.keras"))
                if model is not None and classes:
                    dl_result = predict_mobilenet(model, image_pil, classes)
            if classical_result is None or dl_result is None:
                st.info("Perbandingan prediksi membutuhkan bobot Model A dan Model B. Silakan tempatkan artefak pada folder `models/`.")
            else:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Model A · Klasik")
                    st.write(classical_result)
                    st.caption(f"Inferensi: {classical_seconds * 1000:.1f} ms")
                with col_b:
                    st.subheader("Model B · MobileNetV2")
                    st.metric("Prediksi", dl_result["class"])
                    st.metric("Keyakinan", f"{dl_result['confidence']:.1%}")
                    st.caption(f"Inferensi: {dl_result['seconds'] * 1000:.1f} ms")

    with tab_research:
        render_research_summary()


if __name__ == "__main__":
    main()