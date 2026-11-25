# Artefak Model

Tempatkan artefak hasil pelatihan di direktori ini. Berkas bobot tidak dilacak Git; unduh atau hasilkan secara lokal.

| Nama berkas | Format | Isi |
|---|---|---|
| `model_a.joblib` | Joblib | Dictionary berisi `proto`, `scaler`, `svm`, dan `classes`, sesuai ekspor notebook |
| `model_b_mobilenetv2.keras` | Keras | Model MobileNetV2 hasil fine-tuning, termasuk preprocessing dan kepala klasifikasi |
| `baseline_knn.joblib` | Joblib (opsional) | k-NN baseline dan label kelasnya jika baseline hendak digunakan sebagai artefak aplikasi |

Notebook menghasilkan `model_a.joblib` dan `model_b_mobilenetv2.keras` pada direktori kerja saat sel ekspor dijalankan; pindahkan keduanya ke folder ini. Aplikasi tetap dapat dibuka tanpa bobot, namun inferensi yang memerlukan model terkait tidak tersedia sampai artefaknya dipasang.
