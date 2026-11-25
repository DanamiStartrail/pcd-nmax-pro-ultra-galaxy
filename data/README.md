# 📂 Dokumentasi Dataset

## Sumber Dataset

- **Nama Dataset:** Dataset Food Classification (Indonesian Food)
- **Sumber:** Kaggle (`rizkyyk/dataset-food-classification`)
- **Cakupan Kelas:** 10 kelas makanan Indonesia (3 kelas makanan Barat yaitu `pizza`, `burger`, dan `french fries` dikecualikan pada tahap pra-pemrosesan data).

## Struktur Direktori untuk Reproduksi Notebook

Dataset mentah berukuran besar tidak diunggah ke repositori GitHub. Untuk menjalankan ulang `notebooks/Tubes_PCD_NMAX.ipynb`, susun direktori lokal (atau di Google Drive) dengan struktur berikut dan arahkan variabel `DATA_DIR` ke folder induknya:

```text
DATA_DIR/
├── train/
│   ├── <nama_kelas_1>/*.jpg
│   └── ...
├── val/
│   ├── <nama_kelas_1>/*.jpg
│   └── ...
└── test/
    ├── <nama_kelas_1>/*.jpg
    └── ...

```

## Folder `data/samples/`

Berisi beberapa contoh citra uji berukuran kecil yang digunakan oleh aplikasi Streamlit (`app.py`) untuk fitur pengujian cepat tanpa unggah manual.
