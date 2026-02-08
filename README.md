# 🏥 DiabetesRisk AI: Sistem Deteksi Dini Diabetes

## 🎯 Fitur Utama

- **📊 Eksplorasi Data**: Analisis dataset Pima Indians Diabetes
- **🤖 Pelatihan Model**: Evaluasi performa Random Forest dengan SMOTE
- **🔍 Prediksi Diagnosa**: Simulasi diagnosa pasien real-time
- **📈 Visualisasi**: Confusion Matrix, ROC Curve, Feature Importance

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/muhammadhanifarafi/project_program_kelompok3_kecerdasan_buatan_if501.git
cd project_program_kelompok3_kecerdasan_buatan_if501

# Setup environment
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run aplikasi
streamlit run app.py
```

## 🛠️ Teknologi

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: Scikit-learn (Random Forest)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Balancing**: SMOTE (imbalanced-learn)

## 📁 Struktur Project

```
├── app.py                 # Aplikasi utama Streamlit
├── requirements.txt       # Dependencies
├── diabetes.csv          # Dataset
├── Cara Penginstalan.md  # Panduan instalasi
├── README.md            # File ini
└── .devcontainer/       # Dev container config
```

## 🧪 Dataset

Menggunakan **Pima Indians Diabetes Dataset** dengan 8 fitur medis:
1. Pregnancies - Jumlah kehamilan
2. Glucose - Kadar glukosa darah
3. BloodPressure - Tekanan darah diastolik
4. SkinThickness - Ketebalan kulit
5. Insulin - Level insulin serum
6. BMI - Indeks massa tubuh
7. DiabetesPedigreeFunction - Riwayat diabetes keluarga
8. Age - Usia

## 📖 Cara Penggunaan

1. Buka aplikasi di browser
2. Pilih menu "Beranda & Dataset" untuk eksplorasi data
3. Pilih menu "Pelatihan Model" untuk melihat performa
4. Pilih menu "Prediksi Diagnosa" untuk simulasi pasien
5. Input parameter klinis dan klik "Analisis Risiko Diabetes"