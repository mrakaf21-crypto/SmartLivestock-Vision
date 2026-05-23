# 🐄 SmartLivestock Vision (SLV)
**Sistem Monitoring Dimensi, Bobot & Keseragaman Ternak Berbasis AI**

---

## 📋 Overview

SLV adalah sistem Computer Vision yang mampu:
- Mendeteksi sapi mainan (skala **1:48 s/d 1:65**) melalui kamera
- Mengukur **panjang** dan **tinggi** secara otomatis
- Mengestimasi **lingkar dada** dan **bobot hidup** (skala sapi asli)
- Menampilkan **Uniformity Index** keseragaman kelompok
- Dashboard interaktif real-time

---

## 🚀 Cara Menjalankan (Step by Step)

### STEP 0 — Install Dependensi
```bash
pip install -r requirements.txt
```

---

### STEP 1 — Kalibrasi Kamera *(Wajib pertama kali)*

1. Siapkan **penggaris fisik**
2. Letakkan di posisi yang sama dengan tempat sapi mainan akan diuji
3. Jalankan:
```bash
python src/calibration.py
```
4. Tekan **SPASI** untuk freeze frame
5. Klik **2 titik** pada penggaris
6. Masukkan jarak aslinya dalam mm
7. Kalibrasi tersimpan otomatis ke `src/config.json`

---

### STEP 2 — Pengumpulan Dataset & Labeling *(Untuk custom AI)*

> ⚡ **SKIP step ini dulu** kalau mau langsung demo.  
> Sistem bisa jalan tanpa custom model pakai mode OpenCV Contour.

Kalau mau akurasi lebih tinggi:

1. Foto sapi mainan **30-50 foto** dari arah **samping (side-view)**
2. Buka **[Roboflow](https://roboflow.com)** → Create Project → Upload foto
3. Label semua sapi dengan class: `sapi`
4. Export → **YOLOv8 Format**
5. Taruh isi folder ekspor ke `dataset/`

---

### STEP 3 — Training AI *(Opsional, skip jika pakai contour mode)*

```bash
# CPU only (kompatibel semua laptop)
python src/train.py --epochs 50 --batch 8 --device cpu

# Pakai GPU NVIDIA (lebih cepat, laptop gaming/workstation)
python src/train.py --epochs 50 --batch 16 --device 0
```

Model tersimpan otomatis ke `models/best.pt`

---

### STEP 4 — Jalankan Dashboard

```bash
streamlit run src/dashboard.py
```

Browser otomatis terbuka di `http://localhost:8501`

---

## 📐 Spesifikasi Sapi Mainan

| Parameter | Mainan | Skala | Sapi Asli (Est.) |
|-----------|--------|-------|-----------------|
| Panjang   | 31 mm  | 1:48  | ~150 cm         |
| Tinggi    | 19 mm  | 1:63  | ~120 cm         |
| Bobot     | ~30 g  | —     | ~400-450 kg     |

---

## 🛠️ Struktur Folder

```
SLV/
├── src/
│   ├── calibration.py   # Kalibrasi kamera
│   ├── detection.py     # Logika AI deteksi
│   ├── measurement.py   # Rumus matematika
│   ├── train.py         # Training model YOLO
│   ├── dashboard.py     # Dashboard Streamlit ← MAIN APP
│   └── config.json      # Hasil kalibrasi (auto-generated)
├── dataset/             # Dataset foto sapi mainan
│   ├── images/
│   └── labels/
├── models/
│   └── best.pt          # Model hasil training (auto-generated)
├── results/             # Grafik hasil training
├── assets/              # Logo, gambar tambahan
└── requirements.txt
```

---

## 🧮 Rumus yang Digunakan

### Estimasi Lingkar Dada
```
Lingkar Dada = (0.81 × Panjang) + (0.62 × Tinggi) + 12.4
```
*(Pendekatan regresi morfometri, rata-rata sapi Bali & PO Indonesia)*

### Estimasi Bobot — Schoorl (Dimodifikasi)
```
Bobot (kg) = (Lingkar Dada + 22)² / 100
```

### Keseragaman (Uniformity Index)
```
CV (%) = (Std Deviasi / Rata-rata) × 100
Uniformity Index = max(0, 100 - CV × 2)
```

| CV       | Status                   |
|----------|--------------------------|
| < 5%     | 🟢 Sangat Seragam        |
| 5–10%    | 🟡 Seragam               |
| 10–15%   | 🟠 Cukup Seragam         |
| > 15%    | 🔴 Tidak Seragam         |

---

## 💻 Requirements

- Python 3.8+
- Windows 10/11
- RAM minimal 4GB
- Kamera laptop (built-in)
- GPU NVIDIA *(opsional, untuk training lebih cepat)*

---

## ❓ FAQ

**Q: Kenapa pakai sapi mainan?**  
A: Ini adalah *Proof of Concept* (PoC). Algoritma yang terbukti akurat di skala miniatur akan tetap akurat di kandang asli dengan mengganti kamera ke CCTV resolusi tinggi.

**Q: Kamera tidak terbuka?**  
A: Coba ubah `camera_index` di sidebar dashboard dari 0 ke 1 atau 2.

**Q: Akurasi rendah tanpa custom model?**  
A: Lakukan Step 2-3 untuk training model khusus sapi mainan lo.

---

*SmartLivestock Vision — Proyek Mata Kuliah Smart Farming*
