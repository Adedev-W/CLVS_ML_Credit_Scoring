# CLVS-ML: Sistem Penilaian Kredit berbasis Machine Learning

**Analisis Komparatif Algoritma ML untuk Klasifikasi Risiko Pinjaman Konsumer Indonesia**

---

## Tentang Riset Ini

Proyek ini adalah riset pribadi yang mengeksplorasi penerapan *machine learning* untuk masalah **credit scoring** di konteks pinjaman konsumer Indonesia. Ide dasarnya sederhana: bisakah kita membangun model yang mampu mengklasifikasikan risiko kredit nasabah secara otomatis, hanya dari data demografis dan keuangan mereka?

Karena data kredit nyata sulit didapat (dan sensitif), saya membuat **dataset sintetik 10.000 nasabah** dengan distribusi statistik yang realistis — didesain agar mencerminkan pola kredit yang sesungguhnya terjadi di perbankan Indonesia.

---

## Pertanyaan Riset

1. Seberapa akurat model ML dalam mengklasifikasikan risiko kredit ke 3 kelas (Lancar / Perhatian / Macet)?
2. Algoritma mana yang paling efektif untuk data tabular jenis ini — Logistic Regression, Random Forest, atau XGBoost?
3. Fitur apa yang paling menentukan keputusan kredit menurut model?

---

## Dataset Sintetik

Dataset dibuat dari nol dengan korelasi realistis yang ditanamkan secara eksplisit:

| Properti | Nilai |
|---|---|
| Jumlah baris | 10.000 nasabah |
| Jumlah fitur | 36 kolom (23 numerik + 13 kategorikal) |
| Target | `status_kredit` (3 kelas) |
| Seed | 42 (reproducible) |

### Distribusi Kelas Target

| Kelas | Proporsi | Deskripsi |
|---|---|---|
| **Lancar** | ~55% | Bayar tepat waktu, tidak ada tunggakan |
| **Perhatian** | ~35% | Tunggakan 1–90 hari |
| **Macet** | ~10% | Tunggakan >90 hari, risiko gagal bayar |

### Korelasi yang Ditanamkan (by design)

- DTI tinggi + riwayat buruk → cenderung **Macet**
- Penghasilan rendah + tunggakan banyak → cenderung **Macet**
- Tabungan tinggi + riwayat baik → cenderung **Lancar**

Ini bukan kebetulan — distribusi per segmen dirancang eksplisit agar model punya "sesuatu untuk dipelajari".

---

## Model yang Dibandingkan

| Model | Preprocessing | Catatan |
|---|---|---|
| Logistic Regression | StandardScaler wajib | Baseline klasik, interpretable |
| Random Forest | Tidak perlu scaling | Ensemble tree, robust terhadap outlier |
| **XGBoost** | Tidak perlu scaling | *State-of-the-art* untuk data tabular |

---

## Struktur Proyek

```
ClVS-ML-pinjaman/
│
├── CLVS_ML_Credit_Scoring.ipynb   ← Notebook utama (Google Colab)
├── README.md                       ← File ini
│
├── credit_scoring/
│   ├── generate_synthetic_10k.py  ← Generator dataset sintetik
│   └── train_xgboost.py           ← Pipeline training standalone
│
├── synthetic_credit_10k.csv        ← Dataset (digenerate saat run Cell 5)
│
└── models_xgb/                     ← Artefak model (digenerate saat run Cell 19)
    ├── xgb_credit.ubj
    ├── xgb_encoders.joblib
    └── xgb_target_encoder.joblib
```

---

## Cara Menjalankan

### Di Google Colab (direkomendasikan)

1. Upload `CLVS_ML_Credit_Scoring.ipynb` ke Google Colab
2. Jalankan semua cell dari atas ke bawah (**Runtime → Run all**)
3. Cell 2 akan menginstall dependensi yang belum ada secara otomatis
4. Tidak perlu upload file apapun — dataset digenerate langsung di Cell 5

### Di Lokal

```bash
# Clone / download proyek
cd ClVS-ML-pinjaman

# Install dependensi
pip install xgboost altair scikit-learn pandas numpy joblib

# Jalankan generator data dulu
python credit_scoring/generate_synthetic_10k.py

# Lalu training
python credit_scoring/train_xgboost.py
```

---

## Dependensi

| Library | Versi | Kegunaan |
|---|---|---|
| `xgboost` | 2.x | Model utama |
| `scikit-learn` | 1.6.1 | LR, RF, preprocessing, metrik |
| `pandas` | 2.2.x | Manipulasi data |
| `numpy` | — | Komputasi numerik |
| `altair` | ≥5.0 | Visualisasi riset |
| `joblib` | — | Simpan/load model |

---

## Isi Notebook (19 Cells)

| Cell | Konten |
|---|---|
| 1 | Cover, abstrak, metadata |
| 2 | Instalasi library |
| 3 | Import, konfigurasi, tema Altair |
| 4 | Pendahuluan & latar belakang |
| 5 | **Generasi data sintetik** → `synthetic_credit_10k.csv` |
| 6 | Header eksplorasi dataset |
| 7 | Statistik deskriptif + Chart 1 (donut) + Chart 2 (heatmap korelasi) |
| 8 | Chart 3 (boxplot DTI) + Chart 4 (bar riwayat) + Chart 5 (scatter) |
| 9 | Header pra-pemrosesan |
| 10 | Encoding + Train/Test Split + Chart 6 |
| 11 | Header pelatihan |
| 12 | **Training XGBoost** (early stopping) |
| 13 | **Training LR & RF** + tabel metrik |
| 14 | Header benchmarking |
| 15 | Chart 7 (grouped bar benchmarking) + Chart 8 (F1 heatmap) |
| 16 | Evaluasi detail XGBoost + Chart 9 (confusion matrix) + Chart 10 (loss) |
| 17 | Chart 11 (feature importance) |
| 18 | Kesimpulan & interpretasi |
| 19 | **Simpan artefak** ke `models_xgb/` |

---

## Visualisasi yang Dihasilkan

| # | Chart | Tipe |
|---|---|---|
| 1 | Distribusi kelas target | Donut chart |
| 2 | Korelasi fitur numerik | Heatmap |
| 3 | Distribusi DTI per kelas | Boxplot |
| 4 | Riwayat kredit per kelas | Grouped bar |
| 5 | Penghasilan vs DTI | Scatter plot |
| 6 | Distribusi train/test | Grouped bar |
| 7 | Perbandingan semua model | Grouped bar |
| 8 | F1-score per kelas per model | Heatmap |
| 9 | Confusion matrix XGBoost | Heatmap + label |
| 10 | Training loss per round | Line chart |
| 11 | Top 20 feature importance | Horizontal bar |

---

## Catatan Teknis

### XGBoost 2.x — Breaking Changes

Beberapa perubahan API penting yang sudah diperbaiki di notebook ini:

```python
# SALAH di XGBoost 2.x:
XGBClassifier(use_label_encoder=False)           # parameter dihapus
model.fit(..., callbacks=[EarlyStopping(...)])    # callbacks dihapus dari fit()
model.fit(..., early_stopping_rounds=20)          # dipindah ke constructor

# BENAR di XGBoost 2.x:
XGBClassifier(early_stopping_rounds=20)           # di constructor
model.fit(..., eval_set=[...], verbose=False)     # fit() hanya terima eval_set
```

### scikit-learn 1.6.x

```python
# SALAH di sklearn 1.6:
LogisticRegression(multi_class='multinomial')    # parameter dihapus

# BENAR — solver lbfgs handle multiclass secara default:
LogisticRegression(max_iter=1000)
```

---

## Keterbatasan & Rencana ke Depan

**Keterbatasan saat ini:**
- Dataset sepenuhnya sintetik — perlu validasi dengan data nyata
- Distribusi mungkin tidak merefleksikan kondisi pasar kredit terkini
- Tidak ada feature engineering yang intensif
- Belum ada hyperparameter tuning (grid/random search)

**Yang bisa dikembangkan:**
- [ ] Tambah SHAP values untuk interpretabilitas per-instance
- [ ] Coba model tambahan: LightGBM, CatBoost
- [ ] Implementasi oversampling (SMOTE) untuk kelas Macet yang minoritas
- [ ] Hyperparameter tuning dengan Optuna
- [ ] Validasi silang (cross-validation) yang lebih robust
- [ ] Simulasi dengan data kredit publik (bila tersedia)

---

## Lisensi

Riset pribadi — bebas digunakan untuk referensi dan pembelajaran.

---

*Dibuat dengan Python, XGBoost, scikit-learn, dan Altair.*
