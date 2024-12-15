# Tugas Besar 2 - Implementasi Algoritma Pembelajaran Mesin

<p align="center">
  <img src="cat.png" alt="Cat" width="600" />
</p>


## Deskripsi Singkat
Repository ini berisi implementasi algoritma pembelajaran mesin, yaitu **K-Nearest Neighbors (KNN)**, **Gaussian Naive-Bayes**, dan **ID3**. Selain implementasi, juga dilakukan perbandingan performa antara algoritma yang dibuat sendiri dengan pustaka *Scikit-learn* menggunakan metrik **F1 Macro Score**. Proses *cleaning*, *preprocessing*, dan validasi model turut disertakan dalam proyek ini dan dapat dilihat pada notebook.

---

## Fitur Utama
- Implementasi algoritma **KNN**, **Gaussian Naive-Bayes**, dan **ID3**.
- Metode validasi **K-Fold Cross Validation** dengan 5 fold.
- Tahap *data cleaning* dan *preprocessing* yang meliputi:
  - Penanganan *missing values*.
  - Penanganan *outliers*.
  - Feature scaling dan encoding.
  - Normalisasi dengan **Power Transformer Yeo-Johnson**.
  - Reduksi dimensi menggunakan **PCA**.
- Perbandingan hasil implementasi algoritma dengan pustaka *Scikit-learn*.
- Evaluasi performa model menggunakan **F1 Macro Score**.

---

## Struktur Repository
```
Tubes2-AI/
├── doc/
│   └── Tubes2_13522013_13522074_13522101.pdf
├── dataset/              
├── src/                   
│   ├── knn.py           
│   ├── gaussian_nb.py  
│   ├── id3.py             
│   └── notebook.ipynb
├── submission/
├── requirements.txt
└── README.md              
```

---

## Cara Setup dan Menjalankan Program

### **1. Clone Repository**
Clone repository ini dengan menggunakan command berikut:
```bash
git clone https://github.com/b33rk/Tubes2-AI.git
cd Tubes2-AI
```

### **2. Buat Virtual Environment (Opsional)**
Disarankan menggunakan *virtual environment* untuk mengelola pustaka Python.
```bash
python -m venv env
source env/bin/activate    # Untuk Linux/MacOS
env\Scripts\activate       # Untuk Windows
```

### **3. Install Dependencies**
Instal semua library yang dibutuhkan menggunakan `requirements.txt`.
```bash
pip install -r requirements.txt
```

### **4. Jalankan Program**
Jalankan file `notebook.ipynb` dengan menekan tombol "Run All".

---

## Pembagian Tugas
| **Kegiatan**                       | **Nama (NIM)**                 |
|------------------------------------|--------------------------------|
| Implementasi Model KNN             | Abdullah Mubarak (13522101)    |
| Implementasi Model Gaussian Naive-Bayes | Muhammad Naufal Aulia (13522074) |
| Implementasi Model ID3             | Denise Felicia Tiowanni (13522013) |
| Data Cleaning dan Preprocessing    | Denise Felicia Tiowanni (13522013) |
|                                    | Muhammad Naufal Aulia (13522074) |
|                                    | Abdullah Mubarak (13522101)    |
| Laporan                           | Denise Felicia Tiowanni (13522013) |
|                                    | Muhammad Naufal Aulia (13522074) |
|                                    | Abdullah Mubarak (13522101)    |
