# CLVS-ML: Analisis Komparatif Algoritma Machine Learning untuk Sistem Penilaian Kredit Konsumer

Eksperimen komparatif tiga algoritma machine learning — **Logistic Regression**, **Random Forest**, dan **XGBoost** — untuk klasifikasi multiclass penilaian kredit konsumer menggunakan data sintetik 10.000 nasabah Indonesia.

---

## Ringkasan Hasil

| Model | Accuracy | F1-Macro | Precision-Macro | Recall-Macro | Waktu Training |
|---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.9085 | 0.8851 | 0.9020 | 0.8706 | 0.3s |
| Random Forest | 0.9080 | 0.8813 | **0.9146** | 0.8566 | 3.4s |
| **XGBoost** | **0.9085** | **0.8907** | 0.9069 | **0.8768** | 3.6s |

XGBoost memberikan performa terbaik pada Accuracy, F1-Macro, dan Recall-Macro, sementara Random Forest unggul pada Precision-Macro.

---

## Deskripsi Proyek

### Latar Belakang

Penilaian kredit (*credit scoring*) merupakan proses evaluasi statistik yang digunakan lembaga keuangan untuk menilai kelayakan kredit calon debitur. Penelitian ini membandingkan kinerja tiga algoritma machine learning dalam memprediksi kualitas kredit nasabah ke dalam tiga kelas berdasarkan klasifikasi OJK:

| Kelas | Deskripsi | Distribusi |
|---|---|:---:|
| **Lancar** | Pembayaran tepat waktu, tidak ada tunggakan signifikan | 55.7% (5,573) |
| **Perhatian** | Tunggakan 1–90 hari, memerlukan pemantauan intensif | 34.7% (3,473) |
| **Macet** | Tunggakan >90 hari, risiko gagal bayar tinggi | 9.5% (954) |

### Dataset

Dataset sintetik yang dihasilkan secara algoritmik dengan distribusi statistik yang merepresentasikan karakteristik kredit nasabah Indonesia:

- **10.000 observasi** dengan **36 fitur prediktor** dan **1 variabel target**
- **23 fitur numerik** + **13 fitur kategorikal**
- Tidak ada missing values — dataset bersih
- Total memori: 9,226.9 KB

Fitur dikelompokkan ke dalam tiga kategori:

- **Demografis** — usia (21–65 tahun), jenis kelamin, status pernikahan, pendidikan (SMA/Diploma/S1/S2), jumlah tanggungan, lokasi (urban/rural)
- **Finansial** — penghasilan bulanan (rata-rata Rp 6.68 juta), DTI ratio, LTI ratio, saldo tabungan, nilai aset, angsuran bulanan
- **Riwayat Kredit** — jumlah kredit sebelumnya, jumlah tunggakan, hari tunggakan terlama, riwayat kredit (Baik/Cukup/Buruk)

---

## Metodologi

### Pra-pemrosesan Data

1. **LabelEncoder** untuk 13 fitur kategorikal → integer encoding
2. **LabelEncoder** untuk target → integer encoding (Lancar=0, Macet=1, Perhatian=2)
3. **Stratified Train/Test Split** 80/20 → 8,000 training, 2,000 testing
4. **StandardScaler** khusus untuk Logistic Regression saja (model tree-based tidak memerlukan scaling)

### Konfigurasi Model

| Model | Parameter Kunci |
|---|---|
| **Logistic Regression** | `max_iter=1000`, `solver=lbfgs`, `multi_class=auto` + StandardScaler |
| **Random Forest** | `n_estimators=200`, `max_depth=8`, `n_jobs=-1` |
| **XGBoost** | `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`, `early_stopping_rounds=20` |

XGBoost menggunakan early stopping dan berhenti pada **iterasi ke-172** dari 500, dengan best validation mlogloss **0.24000**.

---

## Hasil Detail

### F1-Score per Kelas

| Model | Lancar | Macet | Perhatian |
|---|:---:|:---:|:---:|
| Logistic Regression | 0.9436 | 0.8444 | 0.8672 |
| Random Forest | 0.9439 | 0.8328 | 0.8673 |
| XGBoost | 0.9404 | **0.8643** | **0.8675** |

### Confusion Matrix — XGBoost

| Aktual \ Prediksi | Lancar | Macet | Perhatian |
|---|:---:|:---:|:---:|
| **Lancar** | 1,065 | 0 | 49 |
| **Macet** | 1 | 156 | 34 |
| **Perhatian** | 85 | 14 | 596 |

Total prediksi benar: **1,817 / 2,000** (Akurasi: 0.9085)

### Top 10 Feature Importance — XGBoost (Gain)

| Rank | Fitur | Importance |
|:---:|---|:---:|
| 1 | DTI Ratio | 0.2179 |
| 2 | Hari Tunggakan Terlama | 0.1674 |
| 3 | Jumlah Tunggakan | 0.0924 |
| 4 | Riwayat Kredit | 0.0689 |
| 5 | Saldo Tabungan | 0.0336 |
| 6 | Penghasilan Bulanan | 0.0320 |
| 7 | Total Penghasilan | 0.0192 |
| 8 | Kredit Sebelumnya | 0.0172 |
| 9 | Level Pekerjaan | 0.0137 |
| 10 | LTI Ratio | 0.0136 |

Faktor perilaku pembayaran (tunggakan, riwayat kredit) dan rasio keuangan (DTI, LTI) memiliki daya prediksi jauh lebih tinggi dibandingkan faktor demografis.
---

## Cara Menjalankan

### Prasyarat

- Python 3.10+
- Google Colab (direkomendasikan) atau Jupyter Notebook

### Instalasi Dependensi

```bash
pip install numpy pandas scikit-learn xgboost altair vegafusion vl-convert-python joblib
```

### Menjalankan Notebook

1. Buka `CLVS_ML_Credit_Scoring_lab.ipynb` di Google Colab
2. Jalankan semua cell secara berurutan — notebook akan:
   - Generate dataset sintetik 10,000 nasabah
   - Melakukan eksplorasi data (EDA) dengan visualisasi Altair
   - Melatih tiga model (LR, RF, XGBoost)
   - Menampilkan benchmarking dan evaluasi detail
   - Menyimpan artefak model ke folder `models_xgb/`

---

## Dependensi Utama

| Library | Versi | Kegunaan |
|---|---|---|
| numpy | 2.0.2 | Komputasi numerik |
| pandas | 2.2.2 | Manipulasi data |
| scikit-learn | — | Preprocessing, LR, RF, metrik evaluasi |
| xgboost | 3.2.0 | Model XGBoost |
| altair | 5.5.0 | Visualisasi interaktif |
| joblib | — | Serialisasi model |

---

## Temuan Utama

1. **Ketiga model mencapai akurasi >90%**, mengonfirmasi efektivitas machine learning untuk credit scoring
2. **XGBoost unggul pada F1-Macro (0.8907)** — metrik paling relevan untuk klasifikasi imbalanced multiclass
3. **Random Forest unggul pada Precision-Macro (0.9146)** — lebih konservatif dalam prediksi positif
4. **DTI Ratio (21.79%)** menjadi fitur paling prediktif, diikuti Hari Tunggakan Terlama (16.74%)
5. **Early stopping efektif** — XGBoost konvergen di iterasi 172/500, mencegah overfitting

---

## Penulis

**Ade Saputra** — 2026

---

## Lisensi

Proyek ini dibuat untuk keperluan riset akademik.
