# 🍲 Klasifikasi Jenis Makanan Indonesia di Kantin (Kelompok NMAX)

<!-- BADGES -->

[![Streamlit App](https://img.shields.io/badge/Demo_Aplikasi-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://dant-latihan-aplication-engineer.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

Repositori utama Tugas Besar mata kuliah **Pengolahan Citra Digital (IF25-40410)** — Kelas RA, Program Studi Teknik Informatika, Institut Teknologi Sumatera (ITERA). Proyek ini mengomparasikan pendekatan **Pengolahan Citra Klasik (Model A)** dan **Deep Learning (Model B — MobileNetV2)** dalam mengenali jenis makanan Indonesia.
[Simulasi Aplikasi Demo Streamlit](https://nmax-demo-aplication.streamlit.app/)

---

## 👥 Anggota Kelompok NMAX (Topik 3)

| No  |     NIM     | Nama Lengkap                  | Peran                         |
| :-: | :---------: | :---------------------------- | :---------------------------- |
|  1  | `123140182` | Muhammad Ghama Al Fajri       | **Project Lead**              |
|  2  | `123140197` | Muhammad Rafiq Ridho          | **Data Engineer & Annotator** |
|  3  | `123140184` | Mohd. Musyaffa Alief Athallah | **Computer Vision Engineer**  |
|  4  | `123140192` | Abi Sholihan                  | **Evaluation & QA Analyst**   |
|  5  | `124140147` | Mandanta Guru Singa           | **Application Engineer**      |

---

## 🔬 Ringkasan Metodologi & Skenario Eksperimen

1. **Dataset:** Menggunakan dataset makanan Indonesia dari Kaggle (`rizkyyk/dataset-food-classification`) dengan total 10 kelas makanan Indonesia (3 kelas makanan Barat dikecualikan) yang dibagi ke dalam `train`, `val`, dan `test`.
2. **Baseline:** Ekstraksi histogram warna HSV global pada citra utuh tanpa _Gaussian blur_, tanpa segmentasi, dan tanpa fitur tekstur GLCM, diklasifikasikan menggunakan _k-NN_ (jarak Hellinger/Bhattacharyya).
3. **Model A (Pengolahan Citra Klasik — 4 Tahap):**
    - **Pra-pemrosesan:** _Resize_ ke $224 \times 224$ piksel dan _Gaussian Blur_ ($5 \times 5$).
    - **Segmentasi & Morfologi:** Konversi ruang warna HSV, _Otsu thresholding_ pada kanal Saturasi (`S`), serta operasi morfologi _Opening–Closing_ dengan kernel elips $7 \times 7$.
    - **Ekstraksi Fitur:** Histogram warna HSV 3D ($8 \times 8 \times 8 = 512$ bin) dan 4 fitur tekstur GLCM (_contrast, homogeneity, energy, correlation_) pada area objek makanan.
    - **Klasifikasi:** Diuji dalam dua skenario, yaitu **Skenario 1** (Prototipe rata-rata kelas + jarak gabungan Bhattacharyya & Euclidean _z-score_ GLCM) dan **Skenario 2** (Support Vector Machine / SVM RBF multi-kelas).
4. **Studi Ablasi (Ablasi 1–4):** Pengujian kontribusi tiap komponen dengan menghilangkan tahap _Gaussian blur_, segmentasi, fitur warna HSV, atau fitur tekstur GLCM secara bergantian.
5. **Model B (Deep Learning — Pembanding):** _Transfer learning_ dan _fine-tuning_ arsitektur **MobileNetV2** (_pretrained_ ImageNet) pada citra RGB $224 \times 224$ piksel.

---

## 📁 Struktur Repositori

```text
pcd-nmax-pro-ultra-galaxy/
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                          # Antarmuka utama aplikasi produksi Streamlit
├── assets/                         # Dokumentasi visual & grafik hasil evaluasi
├── data/
│   ├── README.md                   # Panduan struktur & sumber dataset
│   └── samples/                    # Sampel citra uji untuk demo cepat Streamlit
├── models/
│   ├── README.md                   # Dokumentasi berkas bobot model terlatih
│   ├── model_a.joblib              # Bobot Model A (Prototipe, Scaler, SVM, Kelas)
│   └── model_b_mobilenetv2.keras   # Bobot Model B (MobileNetV2 fine-tuning)
├── notebooks/
│   └── Tubes_PCD_NMAX.ipynb        # Jupyter Notebook eksperimen lengkap tim
└── src/
    ├── __init__.py
    ├── preprocessing.py            # Resize & Gaussian Blur
    ├── segmentation.py             # HSV-Otsu & Morfologi Opening-Closing
    ├── feature_extraction.py       # Histogram HSV 3D & 4 fitur GLCM
    ├── classical_models.py         # Inferensi Baseline, Model A S1 & S2
    ├── dl_pipeline.py              # Inferensi Model B (MobileNetV2)
    └── evaluation.py               # Metrik klasifikasi & IoU/Dice segmentasi

```

---

## 🚀 Cara Menjalankan Proyek

### A. Menjalankan Aplikasi Demo (Streamlit) Secara Lokal

1. **Kloning repositori:**

```bash
git clone [https://github.com/DanamiStartrail/pcd-nmax-pro-ultra-galaxy.git](https://github.com/DanamiStartrail/pcd-nmax-pro-ultra-galaxy.git)
cd pcd-nmax-pro-ultra-galaxy

```

2. **Buat dan aktifkan _virtual environment_:**

```bash
python3 -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows

```

3. **Pasang pustaka yang dibutuhkan:**

```bash
pip install -r requirements.txt

```

4. **Pastikan berkas bobot model tersedia** di folder `models/` (`model_a.joblib` dan `model_b_mobilenetv2.keras`), lalu jalankan aplikasi:

```bash
streamlit run app.py

```

Buka alamat `http://localhost:8501` pada peramban (_browser_).

### B. Mereproduksi Eksperimen Notebook

1. Unduh dataset dari Kaggle (`rizkyyk/dataset-food-classification`) dan susun sesuai panduan pada [`data/README.md`](https://www.google.com/search?q=data/README.md).
2. Buka `notebooks/Tubes_PCD_NMAX.ipynb` menggunakan Google Colab (disarankan runtime T4 GPU) atau Jupyter/VS Code.
3. Sesuaikan variabel `DATA_DIR` pada sel konfigurasi awal, lalu jalankan seluruh sel secara berurutan.
