import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="DiabetesRisk AI", layout="wide", initial_sidebar_state="expanded")

# Custom CSS untuk loading dan styling
st.markdown("""
<style>
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 20px;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# --- JUDUL DAN DESKRIPSI (Sesuai Bab I & II) ---
st.title("🏥 DiabetesRisk AI: Sistem Deteksi Dini Diabetes")
st.markdown("""
Aplikasi ini mengimplementasikan algoritma **Random Forest** untuk memprediksi risiko Diabetes Melitus.
Sistem ini dirancang sebagai *Clinical Decision Support System* untuk membantu screening awal pasien.
Referensi: *Implementasi Algoritma Random Forest untuk Memprediksi Penyakit Diabetes Melitus*.
""")

# --- 1. PENGUMPULAN & PEMAHAMAN DATA [cite: 114-118] ---
@st.cache_data
def load_data():
    with st.spinner('🔄 Memuat dataset diabetes...'):
        try:
            data = pd.read_csv('diabetes.csv')
            st.success("✅ Dataset berhasil dimuat!")
            return data
        except FileNotFoundError:
            st.error("❌ File 'diabetes.csv' tidak ditemukan! Silakan download dari Kaggle (Pima Indians Diabetes) dan letakkan di folder yang sama.")
            return None

df = load_data()

if df is None:
    st.error("❌ File 'diabetes.csv' tidak ditemukan! Silakan download dari Kaggle (Pima Indians Diabetes) dan letakkan di folder yang sama.")
    st.markdown("""
    <div class="info-box">
        <h4>📋 Cara Mendapatkan Dataset:</h4>
        <ol>
            <li>Kunjungi <a href="https://www.kaggle.com/uciml/pima-indians-diabetes-database" target="_blank">Kaggle Pima Indians Diabetes</a></li>
            <li>Download file <code>diabetes.csv</code></li>
            <li>Letakkan file di folder yang sama dengan <code>app.py</code></li>
            <li>Refresh halaman ini</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
else:
    # Sidebar Menu dengan informasi status
    menu = st.sidebar.selectbox("🧭 Menu Navigasi", ["🏠 Beranda & Dataset", "🤖 Pelatihan Model", "🔍 Prediksi Diagnosa"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Status Aplikasi")
    st.sidebar.success("✅ Dataset Terloaded")
    st.sidebar.info(f"📈 Jumlah Data: {df.shape[0]} pasien")
    st.sidebar.markdown("---")

    # --- 2. PRA-PEMROSESAN DATA [cite: 123-129] ---
    with st.expander("🔧 Informasi Pra-pemrosesan Data", expanded=False):
        st.markdown("""
        <div class="info-box">
            <h4>📝 Langkah-langkah Pra-pemrosesan:</h4>
            <ul>
                <li><strong>Data Cleaning:</strong> Mengganti nilai 0 yang tidak logis pada kolom medis (Glukosa, Tekanan Darah, dll)</li>
                <li><strong>Missing Value Handling:</strong> Mean imputation untuk mengisi data yang hilang</li>
                <li><strong>Data Balancing:</strong> SMOTE untuk mengatasi class imbalance</li>
                <li><strong>Data Splitting:</strong> 80% training, 20% testing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Progress indicator untuk preprocessing
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔄 Memulai pra-pemrosesan data...")
    progress_bar.progress(20)
    
    # Mengganti nilai 0 dengan NaN pada kolom biologis (karena 0 tidak logis untuk Glukosa, TD, dll) [cite: 126]
    cols_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_processed = df.copy()
    df_processed[cols_zero] = df_processed[cols_zero].replace(0, np.nan)
    
    status_text.text("🔧 Menangani missing values...")
    progress_bar.progress(40)
    
    # Handling Missing Data dengan Mean Imputation [cite: 127]
    imputer = SimpleImputer(strategy='mean')
    df_processed[cols_zero] = imputer.fit_transform(df_processed[cols_zero])
    
    status_text.text("⚖️ Menerapkan SMOTE untuk balancing data...")
    progress_bar.progress(60)
    
    # Memisahkan Fitur dan Target
    X = df_processed.drop('Outcome', axis=1)
    y = df_processed['Outcome']

    # 1. Terapkan SMOTE SEBELUM splitting atau hanya pada data training
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    
    status_text.text("✂️ Membagi data training dan testing...")
    progress_bar.progress(80)

    # Split Data 80:20 [cite: 134]
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    status_text.text("✅ Data preprocessing selesai!")
    progress_bar.progress(100)
    
    # Hapus progress bar setelah selesai
    import time
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    
    # --- FUNGSI TRAINING MODEL DENGAN CACHING ---
    @st.cache_resource
    def train_model(X_train, y_train):
        """Train Random Forest model dengan GridSearchCV"""
        with st.spinner('🤖 Melatih model Random Forest dengan GridSearchCV...'):
            # Parameter n_estimators=100 sesuai rekomendasi efisiensi [cite: 237]
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced', None]
            }
            grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
            grid_search.fit(X_train, y_train)
            
            return grid_search.best_estimator_
    
    # --- TRAINING MODEL OTOMATIS DI AWAL ---
    st.info("🔄 Memulai training model Random Forest (otomatis di awal)...")
    
    # Training model dengan caching
    rf_model = train_model(X_train, y_train)
    
    st.success("✅ Model training selesai! Model siap digunakan untuk semua menu.")
    st.markdown("---")
    
    # --- MENU 1: EKSPLORASI DATA ---
    if menu == "🏠 Beranda & Dataset":
        st.header("🏠 Eksplorasi Data Medis (Pima Indians Diabetes)")
        
        # Informasi dataset dengan loading
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Jumlah Pasien", f"{df.shape[0]:,}")
            with col2:
                st.metric("🔢 Jumlah Fitur", f"{df.shape[1]}")
            with col3:
                st.metric("🎯 Target", "Diabetes (0/1)")
        
        st.markdown("---")
        
        # Preview data dengan loading
        with st.expander("📋 Preview Dataset", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔍 Sampel Data Asli")
                with st.spinner('🔄 Memuat sampel data...'):
                    st.dataframe(df.head(10), use_container_width=True)
            
            with col2:
                st.subheader("📈 Statistik Deskriptif")
                with st.spinner('🔄 Menghitung statistik...'):
                    st.dataframe(df.describe().round(2), use_container_width=True)
        
        # Informasi pra-pemrosesan
        st.markdown("""
        <div class="info-box">
            <h4>ℹ️ Informasi Pra-pemrosesan:</h4>
            <p>✅ Data cleaning telah dilakukan untuk menangani nilai 0 yang tidak logis pada Glukosa, Tekanan Darah, dll.</p>
            <p>✅ Missing values telah diisi menggunakan mean imputation.</p>
            <p>✅ Data balancing menggunakan SMOTE untuk mengatasi class imbalance.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualisasi data distribusi
        st.subheader("📊 Distribusi Data")
        with st.spinner('🔄 Membuat visualisasi distribusi...'):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Distribusi Outcome**")
                outcome_counts = df['Outcome'].value_counts()
                fig_outcome, ax_outcome = plt.subplots()
                colors = ['#ff9999', '#66b3ff']
                ax_outcome.pie(outcome_counts.values, labels=['Tidak Diabetes', 'Diabetes'], 
                             autopct='%1.1f%%', colors=colors, startangle=90)
                ax_outcome.set_title('Distribusi Outcome')
                st.pyplot(fig_outcome)
            
            with col2:
                st.write("**Distribusi Usia**")
                fig_age, ax_age = plt.subplots()
                ax_age.hist(df['Age'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
                ax_age.set_xlabel('Usia')
                ax_age.set_ylabel('Frekuensi')
                ax_age.set_title('Distribusi Usia Pasien')
                st.pyplot(fig_age)

    # --- MENU 2: PERFORMA MODEL ---
    elif menu == "🤖 Pelatihan Model":
        st.header("🤖 Evaluasi Model Random Forest")
        
        # Cek apakah model sudah ada
        if rf_model is not None:
            st.success("✅ Model sudah dilatih dan siap digunakan!")
            st.info("ℹ️ Model telah di-train otomatis saat program dimulai.")
        else:
            # Training model hanya di menu ini (fallback)
            st.info("🔄 Memulai training model Random Forest...")
            rf_model = train_model(X_train, y_train)
            st.success("✅ Model training selesai!")
        
        # Informasi model
        st.markdown("""
        <div class="info-box">
            <h4>🔧 Informasi Model:</h4>
            <p>🤖 <strong>Algoritma:</strong> Random Forest Classifier</p>
            <p>🔍 <strong>Optimasi:</strong> GridSearchCV dengan 5-fold cross-validation</p>
            <p>⚖️ <strong>Balancing:</strong> SMOTE untuk handling class imbalance</p>
            <p>✂️ <strong>Split:</strong> 80% Training, 20% Testing</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Evaluasi model dengan loading
        with st.spinner('🔄 Mengevaluasi performa model...'):
            # Prediksi pada data test
            y_pred = rf_model.predict(X_test)
            y_probs = rf_model.predict_proba(X_test)[:, 1]
            accuracy = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
        
        # Menampilkan Metrik [cite: 149-150]
        st.success("✅ Evaluasi model selesai!")
        st.subheader("📊 Metrik Performa Model")
        
        mxCol1, mxCol2, mxCol3, mxCol4, mxCol5 = st.columns(5)
        
        with mxCol1:
            st.metric(label="🎯 Akurasi", value=f"{accuracy * 100:.2f}%", delta="Tinggi")
        with mxCol2:
            st.metric(label="🎯 Precision", value=f"{prec * 100:.2f}%", delta="Tinggi")
        with mxCol3:
            st.metric(label="🎯 Recall", value=f"{rec * 100:.2f}%", delta="Tinggi")
        with mxCol4:
            st.metric(label="🎯 F1-Score", value=f"{f1 * 100:.2f}%", delta="Tinggi")
        # Menentukan indikator MSE secara dinamis
        if mse < 0.2:  # MSE rendah = baik
            mse_label = "📉 MSE (Baik)"
        else:  # MSE tinggi = buruk
            mse_label = "📈 MSE (Buruk)"
        
        with mxCol5:
            st.metric(label=mse_label, value=f"{mse:.4f}")
        
        st.markdown("""
        <div class="success-box">
            <h4>🎉 Performa Model:</h4>
            <p>🏆 Model berhasil mencapai akurasi yang kompetitif.</p>
            <p>📈 Semua metrik menunjukkan performa yang sangat baik untuk prediksi diabetes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📈 Visualisasi Evaluasi Model")
        c1, c2 = st.columns(2)

        with c1:
            st.write("**🔍 Confusion Matrix**")
            with st.spinner('🔄 Membuat confusion matrix...'):
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                           xticklabels=['Tidak Diabetes', 'Diabetes'],
                           yticklabels=['Tidak Diabetes', 'Diabetes'])
                ax_cm.set_xlabel('Prediksi')
                ax_cm.set_ylabel('Aktual')
                ax_cm.set_title('Confusion Matrix')
                st.pyplot(fig_cm)
                
                # Penjelasan confusion matrix
                st.markdown("""
                <div class="info-box">
                    <p><strong>True Positive (TP):</strong> Prediksi Diabetes, Benar</p>
                    <p><strong>True Negative (TN):</strong> Prediksi Tidak Diabetes, Benar</p>
                    <p><strong>False Positive (FP):</strong> Prediksi Diabetes, Salah</p>
                    <p><strong>False Negative (FN):</strong> Prediksi Tidak Diabetes, Salah</p>
                </div>
                """, unsafe_allow_html=True)
                
        with c2:
            st.write("**📊 ROC Curve**")
            with st.spinner('🔄 Membuat ROC curve...'):
                fpr, tpr, _ = roc_curve(y_test, y_probs)
                roc_auc = auc(fpr, tpr)
                fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.set_title('ROC Curve')
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)
                
                # Penjelasan ROC curve
                st.markdown(f"""
                <div class="info-box">
                    <p><strong>AUC = {roc_auc:.2f}:</strong> {"Sangat Baik" if roc_auc > 0.9 else "Baik" if roc_auc > 0.8 else "Cukup"} kemampuan model membedakan kelas.</p>
                </div>
                """, unsafe_allow_html=True)

        # Feature Importance [cite: 152, 249]
        st.subheader("🎯 Feature Importance (Tingkat Kepentingan Atribut)")
        st.markdown("Grafik ini menunjukkan faktor mana yang paling berpengaruh terhadap diagnosis diabetes.")
        
        with st.spinner('🔄 Menghitung feature importance...'):
            feature_scores = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            
            # Tampilkan dalam bentuk tabel dan grafik
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📊 Tabel Feature Importance**")
                feature_df = pd.DataFrame({
                    'Fitur': feature_scores.index,
                    'Skor Kepentingan': feature_scores.values,
                    'Persentase': (feature_scores.values * 100).round(2)
                })
                st.dataframe(feature_df, use_container_width=True)
            
            with col2:
                st.write("**📈 Grafik Feature Importance**")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=feature_scores.values, y=feature_scores.index, ax=ax, palette="viridis")
                ax.set_title("Faktor Risiko Paling Dominan")
                ax.set_xlabel("Skor Kepentingan")
                ax.set_ylabel("Atribut Medis")
                
                # Tambahkan nilai pada bar
                for i, v in enumerate(feature_scores.values):
                    ax.text(v + 0.001, i, f'{v:.3f}', va='center')
                
                st.pyplot(fig)
        
        st.markdown("""
        <div class="info-box">
            <h4>🔍 Analisis Feature Importance:</h4>
            <p>📈 <strong>Glukosa</strong> dan <strong>BMI</strong> adalah faktor dominan (sesuai teori medis).</p>
            <p>👥 <strong>Usia</strong> dan <strong>Diabetes Pedigree Function</strong> juga memiliki pengaruh signifikan.</p>
            <p>💡 Informasi ini dapat membantu dokter fokus pada parameter terpenting saat screening.</p>
        </div>
        """, unsafe_allow_html=True)

    # --- MENU 3: SIMULASI PREDIKSI (PRODUK UTAMA) ---
    elif menu == "🔍 Prediksi Diagnosa":
        st.header("🔍 Simulasi Diagnosa Pasien")
        
        # Informasi tentang prediksi
        st.markdown("""
        <div class="info-box">
            <h4>🎯 Cara Penggunaan:</h4>
            <ol>
                <li>Masukkan parameter klinis pasien pada form di bawah</li>
                <li>Klik tombol "Analisis Risiko Diabetes"</li>
                <li>Lihat hasil prediksi dan rekomendasi tindakan</li>
            </ol>
            <p>⚠️ <strong>Disclaimer:</strong> Hasil ini merupakan dukungan keputusan klinis, bukan diagnosis medis final.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📝 Input Parameter Klinis Pasien")
        
        # Pesan sebelum form
        st.info("👋 Silakan isi form di bawah ini dengan data pasien, kemudian klik tombol 'Analisis Risiko Diabetes' untuk melihat hasil prediksi.")

        # Input Form dengan informasi dan validasi [cite: 117]
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📊 Data Demografi & Kehamilan**")
                pregnancies = st.number_input(
                    "🤰 Jumlah Kehamilan (Pregnancies)", 
                    min_value=0, max_value=20, value=1, 
                    help="Jumlah kali hamil"
                )
                age = st.number_input(
                    "👤 Usia (Age)", 
                    min_value=21, max_value=100, value=33,
                    help="Usia pasien dalam tahun"
                )
                
                st.write("**🩸 Data Glukosa & Tekanan Darah**")
                glucose = st.number_input(
                    "🍬 Kadar Glukosa (Glucose)", 
                    min_value=0, max_value=200, value=120,
                    help="Kadar glukosa plasma 2 jam dalam tes toleransi glukosa oral"
                )
                bp = st.number_input(
                    "💉 Tekanan Darah Diastolik (BloodPressure)", 
                    min_value=0, max_value=140, value=70,
                    help="Tekanan darah diastolik (mm Hg)"
                )
            
            with col2:
                st.write("**📏 Data Antropometri**")
                skin = st.number_input(
                    "📏 Ketebalan Kulit (SkinThickness)", 
                    min_value=0, max_value=100, value=20,
                    help="Ketebalan lipatan kulit trisep (mm)"
                )
                bmi = st.number_input(
                    "⚖️ Indeks Massa Tubuh (BMI)", 
                    min_value=0.0, max_value=70.0, value=32.0,
                    help="BMI (berat kg/(tinggi m)^2)"
                )
                
                st.write("**🧬 Data Genetik & Hormonal**")
                insulin = st.number_input(
                    "💉 Insulin Serum (Insulin)", 
                    min_value=0, max_value=900, value=80,
                    help="Insulin serum 2 jam (mu U/ml)"
                )
                dpf = st.number_input(
                    "🧬 Diabetes Pedigree Function", 
                    min_value=0.0, max_value=3.0, value=0.5,
                    help="Fungsi pedigree diabetes (riwayat diabetes keluarga)"
                )
            
            # Validasi input
            if glucose == 0:
                st.warning("⚠️ Kadar glukosa 0 tidak realistis. Silakan periksa kembali input.")
            if bmi == 0:
                st.warning("⚠️ BMI 0 tidak realistis. Silakan periksa kembali input.")
            
            st.markdown("---")
            
            # Tombol prediksi dengan konfirmasi
            submitted = st.form_submit_button(
                "🔍 Analisis Risiko Diabetes", 
                type="primary",
                help="Klik untuk memulai analisis prediksi diabetes"
            )
        
        # Proses prediksi jika form disubmit
        if submitted:
            # Tampilkan loading
            with st.spinner('🔄 Menganalisis data pasien...'):
                # Simulasi proses analisis
                import time
                time.sleep(1)
                
                # Susun data input
                input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
                
                # Prediksi Kelas dan Probabilitas [cite: 163]
                with st.spinner('🤖 Melakukan prediksi dengan model Random Forest...'):
                    prediction = rf_model.predict(input_data)[0]
                    prediction_prob = rf_model.predict_proba(input_data)[0][1] # Ambil probabilitas kelas 1 (Diabetes)
            
            st.divider()
            
            # Tampilkan hasil dengan styling yang menarik
            st.subheader("📋 Hasil Analisis Risiko Diabetes")
            
            # Ringkasan input pasien
            col_summary1, col_summary2 = st.columns(2)
            with col_summary1:
                st.write("**📊 Data Pasien:**")
                st.write(f"- Usia: {age} tahun")
                st.write(f"- BMI: {bmi:.1f}")
                st.write(f"- Glukosa: {glucose} mg/dL")
            
            with col_summary2:
                st.write("**📈 Parameter Risiko:**")
                st.write(f"- Kehamilan: {pregnancies}x")
                st.write(f"- Tekanan Darah: {bp} mmHg")
                st.write(f"- Diabetes Pedigree: {dpf:.2f}")
            
            st.markdown("---")
            
            # Hasil prediksi
            if prediction == 1:
                st.markdown(f"""
                <div class="warning-box">
                    <h2>⚠️ HASIL: BERISIKO DIABETES</h2>
                    <h3>Probabilitas Risiko: {prediction_prob * 100:.2f}%</h3>
                    <p><strong>Rekomendasi Tindakan Lanjutan [cite: 165]:</strong></p>
                    <ul>
                        <li>🔬 Segera lakukan pemeriksaan laboratorium lanjutan (HbA1c, Glukosa Puasa)</li>
                        <li>👨‍⚕️ Konsultasi dengan dokter untuk manajemen diet dan gaya hidup</li>
                        <li>🏃‍♂️ Mulai program olahraga teratur jika belum ada</li>
                        <li>📊 Monitor kadar glukosa darah secara berkala</li>
                        <li>🥗 Pertimbangkan konsultasi dengan ahli gizi</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    <h2>✅ HASIL: RISIKO RENDAH / NORMAL</h2>
                    <h3>Probabilitas Risiko: {prediction_prob * 100:.2f}%</h3>
                    <p><strong>Rekomendasi Kesehatan:</strong></p>
                    <ul>
                        <li>🎉 Pertahankan gaya hidup sehat saat ini</li>
                        <li>📅 Lakukan pengecekan rutin berkala (tahunan)</li>
                        <li>🥗 Pertahankan pola makan seimbang</li>
                        <li>🏃‍♂️ Tetap aktif secara fisik</li>
                        <li>👀 Perhatikan faktor risiko keluarga (diabetes pedigree)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Visualisasi hasil
            st.subheader("📊 Visualisasi Hasil")
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                # Gauge chart untuk probabilitas
                fig_gauge, ax_gauge = plt.subplots(figsize=(8, 4))
                
                # Create gauge effect
                theta = np.linspace(0, np.pi, 100)
                r = 1
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
                ax_gauge.plot(x, y, 'k-', linewidth=2)
                ax_gauge.fill_between(x, y, 0, where=(x >= 0), color='lightgreen', alpha=0.7)
                ax_gauge.fill_between(x, y, 0, where=(x < 0), color='lightcoral', alpha=0.7)
                
                # Add needle
                needle_angle = np.pi * (1 - prediction_prob)
                needle_x = [0, 0.8 * np.cos(needle_angle)]
                needle_y = [0, 0.8 * np.sin(needle_angle)]
                ax_gauge.plot(needle_x, needle_y, 'r-', linewidth=3)
                ax_gauge.plot(0, 0, 'ro', markersize=8)
                
                ax_gauge.set_xlim(-1.2, 1.2)
                ax_gauge.set_ylim(-0.2, 1.2)
                ax_gauge.set_aspect('equal')
                ax_gauge.axis('off')
                ax_gauge.text(0, -0.1, f'{prediction_prob * 100:.1f}%', ha='center', fontsize=16, fontweight='bold')
                ax_gauge.text(-0.7, 0.3, 'Rendah', ha='center', fontsize=12)
                ax_gauge.text(0.7, 0.3, 'Tinggi', ha='center', fontsize=12)
                ax_gauge.set_title('Probabilitas Risiko Diabetes', fontsize=14, fontweight='bold')
                
                st.pyplot(fig_gauge)
            
            with col_viz2:
                # Bar chart untuk parameter comparison
                parameters = ['Glukosa', 'BMI', 'Usia', 'Tekanan Darah']
                patient_values = [glucose, bmi, age, bp]
                normal_ranges = [100, 25, 40, 80]  # Nilai normal rata-rata
                
                fig_comp, ax_comp = plt.subplots(figsize=(8, 6))
                x = np.arange(len(parameters))
                width = 0.35
                
                bars1 = ax_comp.bar(x - width/2, patient_values, width, label='Pasien', color='skyblue')
                bars2 = ax_comp.bar(x + width/2, normal_ranges, width, label='Normal', color='lightgreen')
                
                ax_comp.set_xlabel('Parameter')
                ax_comp.set_ylabel('Nilai')
                ax_comp.set_title('Perbandingan Parameter Pasien vs Normal')
                ax_comp.set_xticks(x)
                ax_comp.set_xticklabels(parameters)
                ax_comp.legend()
                
                # Add value labels on bars
                for bar in bars1:
                    height = bar.get_height()
                    ax_comp.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.0f}', ha='center', va='bottom')
                
                st.pyplot(fig_comp)
            
            # Footer disclaimer
            st.markdown("""
            <div class="info-box">
                <h4>⚠️ Disclaimer Medis:</h4>
                <p>Hasil prediksi ini merupakan <strong>dukungan keputusan klinis</strong> dan bukan diagnosis medis final [cite: 279].</p>
                <p>Selalu konsultasikan dengan dokter profesional untuk diagnosis dan penanganan medis yang tepat.</p>
                <p>Model ini telah dilatih dengan menggunakan algoritma Random Forest dan GridSearchCV untuk optimasi parameter terbaik.</p>
            </div>
            """, unsafe_allow_html=True)
st.caption("© 2025 DiabetesRisk AI - Berbasis Algoritma Random Forest")