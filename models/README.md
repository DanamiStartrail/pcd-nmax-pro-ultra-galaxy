# 🧠 Dokumentasi Bobot Model Terlatih

Folder ini menyimpan berkas artefak model hasil pelatihan dari `notebooks/Tubes_PCD_NMAX.ipynb` yang digunakan oleh aplikasi Streamlit (`app.py`) untuk melakukan inferensi secara langsung:

1. **`model_a.joblib`**
    - Dihasilkan oleh `joblib.dump(...)` pada sel terakhir notebook.
    - Berisi kamus (_dictionary_) Python dengan kunci: `proto` (`PrototypeClassifier` Skenario 1), `scaler` (`StandardScaler` untuk fitur gabungan HSV + GLCM), `svm` (model `SVC` RBF terbaik Skenario 2), dan `classes` (daftar urutan nama 10 kelas).
2. **`model_b_mobilenetv2.keras`**
    - Dihasilkan oleh `model_b.save('model_b_mobilenetv2.keras')` pada sel terakhir notebook.
    - Berisi arsitektur dan bobot hasil _fine-tuning_ **MobileNetV2** dengan ukuran masukan citra RGB `224×224` piksel.

````

---

### 2. Unggah Susunan Baru ke GitHub (`git push`)

Jalankan perintah berikut di terminal (dari folder root `pcd-nmax-pro-ultra-galaxy`) agar susunan repositori dan kode baru sudah tampil di GitHub sebelum kamu mengirim tautannya ke tim:

```bash
git add .
git commit -m "Restructure main project repository and modularize Model A & Model B pipelines"
git push origin main

````
