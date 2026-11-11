# 🖼️ PCD NMAX

<!-- BADGES / TOOL HEADERS -->
[![Deployment](https://img.shields.io/badge/deployment-Streamlit_Cloud-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pillow](https://img.shields.io/badge/Pillow-PIL-000000?style=flat-square&logo=python&logoColor=white)](https://python-pillow.org/)

---

## 📌 Deskripsi Proyek
Aplikasi interaktif berbasis web untuk pengolahan citra digital yang dirancang untuk mendukung demonstrasi proyek Tugas Besar Pengolahan Citra Digital (IF25-40410). Aplikasi ini menyediakan antarmuka pemrosesan citra secara langsung (*real-time*) berbasis Streamlit.

## 👥 Anggota Kelompok NMAX
| No | NIM | Nama | Peran |
|:--:|:---:|:----|:------|
| 1 | 123140182 | Muhammad Ghama Al-Fajri | **Project Lead** |
| 2 | 123140197 | Muhammad Rafiq Ridho | **Data Engineer & Annotator** |
| 3 | 123140184 | Mohd.Musyaffa Alief Athallah | **Computer Vision Engineer** |
| 4 | 123140192 | Abi Sholihan | **Evaluation & QA Analyst** |
| 5 | 124140147 | Mandanta Guru Singa | **Application Engineer** |

---

## 🛠️ Modul & Tech Stack
- **Bahasa**: Python 3.10+
- **Framework UI**: Streamlit
- **Pengolahan Citra**: OpenCV (Headless), Pillow, NumPy
- **Deployment**: Streamlit Community Cloud

---

## 📁 Struktur Repositori
```text
pcd-nmax-pro-ultra-galaxy/
├── .gitignore          # File abaikan Git
├── README.md           # Dokumentasi utama proyek
├── requirements.txt    # Daftar dependensi pustaka Python
├── app.py              # Berkas utama UI Streamlit
├── src/                # Kode sumber algoritma PCD (modular)
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   └── evaluation.py
├── data/               # Data citra masukan & ground truth
├── notebooks/          # Jupyter Notebook percobaan
└── assets/             # Tangkapan layar & aset visual
```

## 🚀 Cara Menjalankan Aplikasi Lokal

### 1. Kloning Repositori

```bash
git clone [https://github.com/DanamiStartrail/pcd-nmax-pro-ultra-galaxy.git](https://github.com/DanamiStartrail/pcd-nmax-pro-ultra-galaxy.git)
cd pcd-nmax-pro-ultra-galaxy

```

### 2. Buat & Aktifkan Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Untuk Linux/macOS
# venv\Scripts\activate   # Untuk Windows

```

### 3. Pasang Dependensi

```bash
pip install -r requirements.txt

```

### 4. Jalankan Aplikasi Streamlit

```bash
streamlit run app.py

```

Aplikasi dapat dibuka di browser lokal pada alamat `http://localhost:8501`.
