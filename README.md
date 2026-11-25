# Komparasi Metode Pengolahan Citra Klasik dan Deep Learning dalam Mengenali Jenis Makanan Indonesia di Kantin

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-models-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-MobileNetV2-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-demo-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://dant-latihan-aplication-engineer.streamlit.app/)

**Mata kuliah:** Pengolahan Citra Digital (IF25-40410), Kelas RA, Teknik Informatika ITERA<br>
**Semester:** Ganjil 2026/2027 · **Kelompok:** NMAX · **Topik:** 3

## Latar Belakang

Proyek ini membandingkan pengenalan jenis makanan kantin menggunakan fitur pengolahan citra klasik dengan transfer learning. Model A mengolah citra melalui resize, Gaussian blur, segmentasi Otsu pada saturasi HSV, histogram warna HSV dan tekstur GLCM. Model B menggunakan MobileNetV2 pretrained ImageNet yang dilatih dengan augmentasi dan fine-tuning. Keduanya dirancang untuk mengevaluasi pertukaran antara pipeline fitur yang dapat diinspeksi dan representasi deep learning.

## Anggota Kelompok NMAX

| NIM | Nama | Peran |
|---|---|---|
| 123140182 | Muhammad Ghama Al Fajri | Project Lead |
| 123140197 | Muhammad Rafiq Ridho | Data Engineer & Annotator |
| 123140184 | Mohd. Musyaffa Alief Athallah | Computer Vision Engineer |
| 123140192 | Abi Sholihan | Evaluation & QA Analyst |
| 124140147 | Mandanta Guru Singa | Application Engineer |

## Dataset

Dataset yang digunakan adalah [`rizkyyk/dataset-food-classification`](https://www.kaggle.com/datasets/rizkyyk/dataset-food-classification), dilaporkan berisi **4.990 citra pada 10 kelas**. Notebook mengecualikan folder `pizza`, `burger`, `french fries` (termasuk variasi penamaannya), lalu menyimpulkan kelas dari subfolder train. Nama kelas dan jumlah citra tiap split tidak tersimpan sebagai output notebook; untuk reproduksi gunakan susunan:

```text
DATA_DIR/
├── train/<kelas>/*.{jpg,jpeg,png}
├── val/<kelas>/*.{jpg,jpeg,png}
└── test/<kelas>/*.{jpg,jpeg,png}
```

Lihat [`data/README.md`](data/README.md) untuk petunjuk dataset dan `PCD_DATA_DIR`.

## Metodologi Eksperimen

1. **Praproses:** resize 224×224 dengan `INTER_AREA`; Gaussian blur opsional menggunakan kernel 5×5 dan sigma 0.
2. **Segmentasi:** Otsu pada kanal S HSV, morfologi opening lalu closing menggunakan kernel elips 7×7. Jika foreground kurang dari 5%, notebook memakai seluruh citra.
3. **Fitur:** histogram HSV 3D berukuran 8×8×8 (512 bin), dinormalisasi; GLCM grayscale 32 level, jarak 1, empat sudut (0°, 45°, 90°, 135°), simetris dan dirata-ratakan untuk contrast, homogeneity, energy, correlation.
4. **Baseline:** histogram HSV citra utuh dengan k-NN Euclidean; `k` dipilih dari 1, 3, 5, 7, 9 menggunakan validasi.
5. **Model A Skenario 1:** prototipe rata-rata per kelas; gabungan jarak Bhattacharyya histogram dan Euclidean GLCM z-score. Bobot warna dipilih dari 0,3/0,5/0,7 pada validasi.
6. **Model A Skenario 2:** SVM RBF dengan histogram akar (Hellinger) dan GLCM; scaler dan parameter `C`/`gamma` dipilih menggunakan validasi.
7. **Ablasi:** tanpa blur, tanpa segmentasi, tanpa fitur warna, serta tanpa fitur tekstur; masing-masing dievaluasi pada S1 dan S2.
8. **Evaluasi segmentasi:** IoU dan Dice pada mask uji beranotasi manual (maksimum 100; dilewati jika `MASK_DIR` tidak diatur).
9. **Model B:** MobileNetV2 pretrained, masukan 224×224, augmentasi flip/rotasi/zoom/contrast; head dilatih hingga 8 epoch pada Adam 1e-3 dengan backbone dibekukan, lalu 30 layer teratas fine-tuning hingga 12 epoch pada Adam 1e-5. Early stopping dan ReduceLROnPlateau mengikuti notebook.

> **Catatan fidelitas:** baseline di notebook adalah k-NN Euclidean, bukan jarak Bhattacharyya. Ukuran masukan notebook adalah 224×224 (bukan 256×256).

## Hasil Evaluasi

Notebook terlampir hanya menyimpan traceback `FileNotFoundError` ketika membaca `/dataset/train`. Sel ekstraksi, pelatihan, dan evaluasi belum dijalankan; karena itu file tersebut **tidak menyediakan angka aktual** untuk akurasi, Macro F1, IoU, Dice, hasil ablasi, waktu inferensi, maupun nilai 10 nama kelas. Angka tidak direka pada ringkasan ini atau pada aplikasi. Jalankan notebook pada dataset yang benar untuk menghasilkan dan mencatat hasil.

| Eksperimen | Akurasi | Macro F1 | IoU / Dice | Waktu inferensi |
|---|---:|---:|---:|---:|
| Baseline (HSV utuh + k-NN) | Belum tersedia | Belum tersedia | — | — |
| Model A S1 (prototipe) | Belum tersedia | Belum tersedia | — | Belum tersedia |
| Model A S2 (SVM) | Belum tersedia | Belum tersedia | — | — |
| Ablasi 1–4 (S1 dan S2) | Belum tersedia | Belum tersedia | — | — |
| Segmentasi Otsu | — | — | Belum tersedia | — |
| Model B MobileNetV2 | Belum tersedia | Belum tersedia | — | Belum tersedia |

## Struktur Repositori

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── app.py
├── assets/
├── data/
│   ├── README.md
│   └── samples/
├── models/
│   └── README.md
├── notebooks/
│   └── Tubes_PCD_NMAX.ipynb
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── segmentation.py
    ├── feature_extraction.py
    ├── classical_models.py
    ├── dl_pipeline.py
    └── evaluation.py
```

## Reproduksi Eksperimen

1. Siapkan Python 3.10+ dan lingkungan virtual, lalu pasang dependensi:

   ```bash
   python -m venv .venv
   # Linux/macOS:
   source .venv/bin/activate
   # Windows PowerShell:
   # .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Unduh dan ekstrak dataset ke `data/` atau tentukan folder lain. Sebelum membuka Jupyter, set `PCD_DATA_DIR` ke folder yang berisi direktori split; contoh Linux/macOS:

   ```bash
   export PCD_DATA_DIR="/path/ke/dataset"
   jupyter notebook notebooks/Tubes_PCD_NMAX.ipynb
   ```

   PowerShell:

   ```powershell
   $env:PCD_DATA_DIR = "D:\path\ke\dataset"
   jupyter notebook notebooks/Tubes_PCD_NMAX.ipynb
   ```

3. Jalankan sel secara berurutan. Atur `MASK_DIR` di notebook bila tersedia anotasi segmentasi. Hasil evaluasi dan ekspor model akan dibuat oleh notebook setelah seluruh tahap berhasil.

## Menjalankan Demo Streamlit

Letakkan artefak model sesuai [`models/README.md`](models/README.md), lalu jalankan:

```bash
streamlit run app.py
```

Buka `http://localhost:8501`. Demo menerima citra unggahan atau contoh dalam `data/samples/`; tanpa bobot model, antarmuka tetap dapat menampilkan tahapan pemrosesan klasik namun prediksi model terkait tidak aktif.

**Live Demo:** [PCD NMAX di Streamlit](https://dant-latihan-aplication-engineer.streamlit.app/) — deployment dapat tidak menyediakan bobot model dan bergantung pada konfigurasi hosting.
