# Dataset

Dataset sumber: [`rizkyyk/dataset-food-classification`](https://www.kaggle.com/datasets/rizkyyk/dataset-food-classification), dengan total **4.990 citra dan 10 kelas** sebagaimana informasi dataset proyek. Notebook mengecualikan folder `pizza`, `burger`, dan `french fries` (termasuk variasi nama `french_fries`/`frenchfries`) lalu mengambil nama kelas dari direktori `train`; nama folder dataset harus konsisten pada seluruh split.

Ekstrak dataset sehingga struktur direktori menjadi:

```text
data/
├── train/<kelas>/*.{jpg,jpeg,png}
├── val/<kelas>/*.{jpg,jpeg,png}
└── test/<kelas>/*.{jpg,jpeg,png}
```

Untuk memakai lokasi lain, atur variabel lingkungan `PCD_DATA_DIR` ke direktori yang berisi folder `train`, `val`, dan `test` sebelum menjalankan notebook. Dataset tidak disertakan di Git karena ukurannya.

Folder `samples/` disediakan untuk citra contoh yang boleh ditampilkan di aplikasi. Masukkan citra milik proyek sendiri; jangan meletakkan dataset lengkap di sana.
