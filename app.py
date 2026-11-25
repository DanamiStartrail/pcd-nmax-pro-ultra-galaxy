from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.classical_models import load_model_a, predict_model_a
from src.dl_pipeline import DEFAULT_CLASSES, load_model_b, predict_model_b
from src.feature_extraction import extract_from_bgr
from src.preprocessing import IMG_SIZE, pil_to_bgr

ROOT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT_DIR / "data" / "samples"
ASSETS_DIR = ROOT_DIR / "assets"

st.set_page_config(
    page_title="Klasifikasi Makanan Indonesia — Kelompok NMAX",
    page_icon="🍲",
    layout="wide",
)


@st.cache_resource
def get_trained_models():
    """Memuat bobot Model A (Prototipe & SVM) dan Model B (MobileNetV2) sekali di memori."""
    artifact_a = load_model_a()
    model_b = load_model_b()
    return artifact_a, model_b


artifact_a, model_b = get_trained_models()
class_names = (
    list(artifact_a["classes"])
    if artifact_a is not None and "classes" in artifact_a
    else DEFAULT_CLASSES
)

# --- HEADER UTAMA ---
st.title("🍲 Klasifikasi Jenis Makanan Indonesia di Kantin")
st.markdown(
    "**Tugas Besar Pengolahan Citra Digital (IF25-40410) — Kelompok NMAX**  \n"
    "Komparasi metode **Pengolahan Citra Klasik (Model A: HSV-Otsu, Morfologi, Histogram HSV & GLCM)** "
    "dan **Deep Learning (Model B: MobileNetV2 Fine-Tuning)**."
)

# --- SIDEBAR: INPUT CITRA & PARAMETER INTERAKTIF ---
st.sidebar.header("📥 Input Citra Uji")
input_mode = st.sidebar.radio(
    "Sumber Citra:",
    ["Unggah Citra Sendiri", "Gunakan Sampel Bawaan"],
)

image_pil = None
image_label = ""

if input_mode == "Unggah Citra Sendiri":
    uploaded_file = st.sidebar.file_uploader(
        "Pilih foto makanan (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        image_label = uploaded_file.name
else:
    sample_files = (
        sorted(
            [
                p
                for p in SAMPLES_DIR.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )
        if SAMPLES_DIR.exists()
        else []
    )
    if sample_files:
        selected_sample = st.sidebar.selectbox(
            "Pilih citra sampel:",
            sample_files,
            format_func=lambda p: p.name,
        )
        image_pil = Image.open(selected_sample)
        image_label = selected_sample.name
    else:
        st.sidebar.info(
            "Folder `data/samples/` belum berisi gambar. "
            "Silakan unggah citra melalui opsi **Unggah Citra Sendiri**."
        )

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Parameter Pipeline Model A")
use_blur = st.sidebar.checkbox("Aktifkan Gaussian Blur (5×5)", value=True)
use_seg = st.sidebar.checkbox("Aktifkan Segmentasi HSV-Otsu & Morfologi", value=True)
st.sidebar.caption(
    "Matikan salah satu opsi di atas untuk mensimulasikan skenario Ablasi 1 atau Ablasi 2 secara langsung."
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Kelompok NMAX — Teknik Informatika ITERA**  \n"
    "Ghama • Rafiq • Alief • Abi • Mandanta"
)


# --- FUNGSI TAMPILAN PER TAB ---
def render_model_a_tab(res_a: dict, pred_a: dict | None):
    st.subheader("1. Tahapan Pra-pemrosesan & Segmentasi (Model A)")

    if res_a["fallback_triggered"]:
        st.info(
            "Area segmentasi objek terlalu kecil (<5% piksel), sehingga sistem otomatis "
            "menggunakan masker penuh sesuai aturan fallback pada eksperimen."
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(
            res_a["resized_rgb"],
            caption=f"1A. Resize ({IMG_SIZE}×{IMG_SIZE})",
            use_container_width=True,
        )
    with c2:
        blur_caption = (
            "1B. Gaussian Blur (5×5)"
            if use_blur
            else "1B. Tanpa Gaussian Blur (Ablasi 1)"
        )
        st.image(
            res_a["blurred_rgb"],
            caption=blur_caption,
            use_container_width=True,
        )
    with c3:
        st.image(
            res_a["s_channel"],
            caption="2A. Kanal Saturasi (S) HSV",
            use_container_width=True,
            clamp=True,
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        st.image(
            res_a["mask_otsu"],
            caption="2B. Otsu Thresholding (Kanal S)",
            use_container_width=True,
        )
    with c5:
        seg_caption = (
            "3. Masker Morfologi (Open + Close 7×7)"
            if use_seg
            else "3. Masker Penuh Tanpa Segmentasi (Ablasi 2)"
        )
        st.image(
            res_a["mask_final"],
            caption=seg_caption,
            use_container_width=True,
        )
    with c6:
        st.image(
            res_a["segmented_rgb"],
            caption="Hasil Akhir Region of Interest (ROI)",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("2. Ekstraksi Fitur Warna (HSV) & Tekstur (GLCM)")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Contrast", res_a["glcm_features"]["Contrast"])
    m2.metric("Homogeneity", res_a["glcm_features"]["Homogeneity"])
    m3.metric("Energy", res_a["glcm_features"]["Energy"])
    m4.metric("Correlation", res_a["glcm_features"]["Correlation"])

    with st.expander(
        "Lihat Distribusi Histogram Warna HSV 3D (512-bin ternormalisasi)"
    ):
        st.bar_chart(res_a["hist_features"])

    st.markdown("---")
    st.subheader("3. Hasil Klasifikasi Model A")

    if pred_a is None:
        st.info(
            "Berkas bobot `models/model_a.joblib` belum ditemukan. "
            "Letakkan hasil ekspor dari notebook ke folder `models/` untuk menampilkan prediksi kelas."
        )
        return

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(
            f"**Skenario 1: Prototipe Kelas + Jarak Gabungan (w = {pred_a['s1_weight_w']})**"
        )
        st.metric("Prediksi Kelas (S1)", pred_a["s1_class"])
        st.caption(f"Jarak gabungan terdekat: `{pred_a['s1_distance']}`")
        st.dataframe(
            pd.DataFrame(pred_a["s1_top3"]), use_container_width=True, hide_index=True
        )

    with col_s2:
        st.markdown("**Skenario 2: Support Vector Machine (SVM RBF)**")
        st.metric("Prediksi Kelas (S2)", pred_a["s2_class"])
        st.caption(
            f"Waktu inferensi klasifikasi: `{pred_a['inference_time'] * 1000:.2f} ms`"
        )
        if pred_a["s2_top3"]:
            st.dataframe(
                pd.DataFrame(pred_a["s2_top3"]),
                use_container_width=True,
                hide_index=True,
            )


def render_model_b_tab(pred_b: dict | None):
    st.subheader("Hasil Inferensi Model B (MobileNetV2 Fine-Tuning)")

    if pred_b is None:
        st.info(
            "Berkas bobot `models/model_b_mobilenetv2.keras` belum ditemukan. "
            "Letakkan hasil ekspor dari notebook ke folder `models/` untuk mengaktifkan inferensi MobileNetV2."
        )
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediksi Utama", pred_b["pred_class"])
    c2.metric("Tingkat Keyakinan (Confidence)", f"{pred_b['confidence'] * 100:.2f}%")
    c3.metric("Waktu Inferensi", f"{pred_b['inference_time'] * 1000:.1f} ms")

    col_tbl, col_chart = st.columns([1, 1.3])
    with col_tbl:
        st.markdown("#### Top-3 Prediksi Kelas")
        st.dataframe(
            pd.DataFrame(pred_b["top3"]), use_container_width=True, hide_index=True
        )

    with col_chart:
        st.markdown("#### Distribusi Probabilitas Seluruh Kelas")
        prob_df = pd.DataFrame(
            {
                "Kelas": list(pred_b["all_probs"].keys()),
                "Probabilitas (%)": [
                    round(v * 100, 2) for v in pred_b["all_probs"].values()
                ],
            }
        ).set_index("Kelas")
        st.bar_chart(prob_df)


def render_comparison_tab(res_a: dict, pred_a: dict | None, pred_b: dict | None):
    st.subheader("Perbandingan Langsung: Model A (PCD Klasik) vs Model B (MobileNetV2)")

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(
            res_a["segmented_rgb"],
            caption="Masukan Fitur Model A (ROI Hasil Segmentasi HSV-Otsu)",
            use_container_width=True,
        )
    with col_img2:
        st.image(
            res_a["resized_rgb"],
            caption=f"Masukan Model B (Citra RGB Utuh {IMG_SIZE}×{IMG_SIZE})",
            use_container_width=True,
        )

    st.markdown("#### Ringkasan Prediksi & Kecepatan Inferensi")
    summary_rows = [
        {
            "Metode": "Model A — Skenario 1 (Prototipe + Jarak)",
            "Prediksi Kelas": pred_a["s1_class"] if pred_a else "Bobot belum dimuat",
            "Skor / Jarak": f"Jarak = {pred_a['s1_distance']}" if pred_a else "-",
            "Waktu Proses (ms)": f"{pred_a['inference_time'] * 1000:.2f}"
            if pred_a
            else "-",
        },
        {
            "Metode": "Model A — Skenario 2 (SVM RBF)",
            "Prediksi Kelas": pred_a["s2_class"] if pred_a else "Bobot belum dimuat",
            "Skor / Jarak": (
                f"{pred_a['s2_top3'][0]['Skor Relatif (%)']}%"
                if pred_a and pred_a["s2_top3"]
                else "-"
            ),
            "Waktu Proses (ms)": f"{pred_a['inference_time'] * 1000:.2f}"
            if pred_a
            else "-",
        },
        {
            "Metode": "Model B — MobileNetV2 (Fine-Tuning)",
            "Prediksi Kelas": pred_b["pred_class"] if pred_b else "Bobot belum dimuat",
            "Skor / Jarak": f"{pred_b['confidence'] * 100:.2f}%" if pred_b else "-",
            "Waktu Proses (ms)": f"{pred_b['inference_time'] * 1000:.2f}"
            if pred_b
            else "-",
        },
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


def render_evaluation_tab():
    st.subheader("Ringkasan Skenario Eksperimen & Evaluasi Kuantitatif")
    st.markdown(
        "Evaluasi dilakukan pada dataset makanan Indonesia (10 kelas dari Kaggle "
        "`rizkyyk/dataset-food-classification` setelah membuang 3 kelas makanan Barat)."
    )

    # Muat otomatis tabel hasil jika file hasil_perbandingan.csv tersedia di root atau data/
    csv_candidates = [
        ROOT_DIR / "hasil_perbandingan.csv",
        ROOT_DIR / "data" / "hasil_perbandingan.csv",
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)

    if csv_path is not None:
        df_eval = pd.read_csv(csv_path)
        st.dataframe(df_eval, use_container_width=True, hide_index=True)
    else:
        st.info(
            "Menampilkan kerangka skenario eksperimen dari `notebooks/Tubes_PCD_NMAX.ipynb`. "
            "Simpan berkas `hasil_perbandingan.csv` dari notebook ke folder root untuk memuat angka akhir secara otomatis."
        )
        placeholder_rows = [
            {
                "Skenario": "Baseline (Histogram HSV Utuh + k-NN)",
                "Akurasi (%)": "-",
                "Macro F1-Score": "-",
                "Keterangan": "Acuan dasar tanpa segmentasi & tekstur",
            },
            {
                "Skenario": "Model A — S1 Prototipe (Blur + Seg + HSV + GLCM)",
                "Akurasi (%)": "-",
                "Macro F1-Score": "-",
                "Keterangan": "Jarak Bhattacharyya + Euclidean z-score",
            },
            {
                "Skenario": "Model A — S2 SVM RBF (Blur + Seg + HSV + GLCM)",
                "Akurasi (%)": "-",
                "Macro F1-Score": "-",
                "Keterangan": "Klasifikasi non-linear fitur gabungan",
            },
            {
                "Skenario": "Ablasi 1: Tanpa Praproses (Tanpa Gaussian Blur)",
                "Akurasi (%)": "-",
                "Macro F1-Score": "-",
                "Keterangan": "Mengukur pengaruh reduksi derau awal",
            },
            {
                "Skenario": "Ablasi 2: Tanpa Segmentasi (Citra Utuh)",
                "Akurasi (%)": "-",
                "Macro F1-Score": "-",
                "Keterangan": "Mengukur kontribusi isolasi ROI makanan",
            },
            {
                "Skenario": "Ablasi 3: Tanpa Fitur Warna (Hanya Tekstur GLCM)",
                "Akurasi (%)": "-",
                "Macro F1-Score": "-",
                "Keterangan": "Mengukur kontribusi fitur warna HSV",
            },
            {
                "Skenario": "Ablasi 4: Tanpa Fitur Tekstur (Hanya Warna HSV)",
                "Akurasi (%)": "-",
                "Macro F1-Score": "-",
                "Keterangan": "Mengukur kontribusi fitur tekstur GLCM",
            },
            {
                "Skenario": "Model B — MobileNetV2 (Transfer Learning & Fine-Tuning)",
                "Akurasi (%)": "-",
                "Macro F1-Score": "-",
                "Keterangan": "Pembanding Deep Learning (224×224)",
            },
        ]
        st.dataframe(
            pd.DataFrame(placeholder_rows), use_container_width=True, hide_index=True
        )

    cm_candidates = [
        ROOT_DIR / "confusion_matrix.png",
        ASSETS_DIR / "confusion_matrix.png",
    ]
    cm_path = next((p for p in cm_candidates if p.exists()), None)
    if cm_path is not None:
        st.markdown("#### Confusion Matrix Pengujian")
        st.image(str(cm_path), use_container_width=True)


# --- EKSEKUSI UTAMA ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔬 Model A (PCD Klasik)",
        "🧠 Model B (MobileNetV2)",
        "⚖️ Perbandingan A vs B",
        "📊 Evaluasi & Ablasi",
    ]
)

if image_pil is not None:
    st.caption(
        f"Citra aktif: **{image_label}** ({image_pil.size[0]}×{image_pil.size[1]} piksel)"
    )
    img_bgr = pil_to_bgr(image_pil)

    with st.spinner("Memproses citra pada Model A dan Model B..."):
        res_a = extract_from_bgr(img_bgr, use_blur=use_blur, use_seg=use_seg)
        pred_a = (
            predict_model_a(res_a["hist_features"], res_a["glcm_vector"], artifact_a)
            if artifact_a is not None
            else None
        )
        pred_b = (
            predict_model_b(image_pil, model_b, class_names=class_names)
            if model_b is not None
            else None
        )

    with tab1:
        render_model_a_tab(res_a, pred_a)
    with tab2:
        render_model_b_tab(pred_b)
    with tab3:
        render_comparison_tab(res_a, pred_a, pred_b)
    with tab4:
        render_evaluation_tab()
else:
    with tab1:
        st.info(
            "👈 Unggah foto makanan atau pilih sampel pada sidebar untuk melihat tahapan Model A."
        )
    with tab2:
        st.info(
            "👈 Unggah foto makanan atau pilih sampel pada sidebar untuk melihat prediksi Model B."
        )
    with tab3:
        st.info(
            "👈 Unggah foto makanan atau pilih sampel pada sidebar untuk membandingkan kedua model."
        )
    with tab4:
        render_evaluation_tab()
