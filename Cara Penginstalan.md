# Cara Penginstalan DiabetesRisk AI

## 🚀 Cara Instalasi

### 1. Prasyarat Sistem
- Python 3.8 atau lebih tinggi
- Git (untuk cloning repository)
- Koneksi internet (untuk download dependencies)

### 2. Clone Repository
```bash
git clone https://github.com/muhammadhanifarafi/project_program_kelompok3_kecerdasan_buatan_if501.git
cd project_program_kelompok3_kecerdasan_buatan_if501
```

### 3. Buat Virtual Environment
```bash
# Untuk Windows
python -m venv venv
venv\Scripts\activate

# Untuk Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Pastikan Dataset Tersedia
Pastikan file `diabetes.csv` sudah ada di folder project. Jika belum, download dari Kaggle (Pima Indians Diabetes Dataset) dan letakkan di folder yang sama.

### 6. Jalankan Aplikasi
```bash
streamlit run app.py
```

Aplikasi akan otomatis terbuka di browser pada `http://localhost:8501`

## 📁 Struktur Project
```
project_program_kelompok3_kecerdasan_buatan_if501/
├── app.py                 # File utama aplikasi Streamlit
├── requirements.txt       # Dependencies Python
├── diabetes.csv          # Dataset diabetes (Pima Indians)
├── config.yaml           # Konfigurasi (jika ada)
├── Cara Penginstalan.md  # File ini
└── .devcontainer/        # Konfigurasi dev container
```

## 🛠️ Dependencies Utama
- **streamlit**: Framework web application
- **pandas**: Manipulasi data
- **numpy**: Komputasi numerik
- **scikit-learn**: Machine learning (Random Forest)
- **imbalanced-learn**: SMOTE untuk balancing data
- **matplotlib & seaborn**: Visualisasi data

## 🎯 Fitur Aplikasi
1. **Beranda & Dataset**: Eksplorasi data medis pasien
2. **Pelatihan Model**: Evaluasi performa model Random Forest
3. **Prediksi Diagnosa**: Simulasi diagnosa pasien dengan input parameter klinis

## 📊 Metrik Evaluasi
- Akurasi Model
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve
- Feature Importance

## 🔧 Troubleshooting

### Masalah Umum:
1. **File tidak ditemukan**: Pastikan `diabetes.csv` ada di folder yang sama
2. **Module not found**: Pastikan virtual environment aktif dan dependencies terinstall
3. **Port conflict**: Jika port 8501 digunakan, gunakan `streamlit run app.py --server.port 8502`

### Commands Tambahan:
```bash
# Cek versi Python
python --version

# Cek installed packages
pip list

# Update dependencies
pip install --upgrade -r requirements.txt
```
