"""
============================================================
SLV - SmartLivestock Vision
STEP 3: Training Model YOLO Custom (FIXED PATH)
============================================================
Jalankan SETELAH lo sudah punya dataset dari Roboflow.

Cara pakai:
1. Ekspor dataset dari Roboflow → format "YOLOv8"
2. Taruh folder hasil ekspor di: dataset/
3. Jalankan: python src/train.py
4. Model best.pt akan tersimpan di: models/best.pt
============================================================
"""

import os
import sys
import shutil

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics belum terinstall.")
    print("Jalankan: pip install ultralytics")
    sys.exit(1)

# ── Path ─────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
DATA_YAML   = os.path.join(DATASET_DIR, "data.yaml")

# ══════════════════════════════════════════════════════════
# TEMPLATE YAML (Sudah disesuaikan dengan struktur Roboflow)
# ══════════════════════════════════════════════════════════
YAML_TEMPLATE = f"""# SLV Dataset Configuration
path: {DATASET_DIR}
train: train/images
val:   valid/images
test:  test/images  # opsional

nc: 1              # jumlah kelas (cukup 1: sapi)
names:
  - sapi
"""

def check_dataset():
    """Cek apakah struktur dataset sudah sesuai bawaan Roboflow."""
    required = [
        os.path.join(DATASET_DIR, "train", "images"),
        os.path.join(DATASET_DIR, "valid", "images"),
        os.path.join(DATASET_DIR, "train", "labels"),
        os.path.join(DATASET_DIR, "valid", "labels"),
    ]
    missing = [p for p in required if not os.path.isdir(p)]
    return missing

def create_yaml():
    """Buat file data.yaml otomatis dengan konfigurasi baru."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(DATA_YAML, "w") as f:
        f.write(YAML_TEMPLATE)
    print(f"✅ data.yaml baru berhasil dibuat di: {DATA_YAML}")

def train(
    epochs    : int   = 50,
    img_size  : int   = 640,
    batch     : int   = 8,
    model_size: str   = "n",     # n=nano, s=small, m=medium
    device    : str   = "cpu",   # "cpu" atau "0" (GPU NVIDIA)
):
    print("\n" + "="*55)
    print("  SLV - Training Model YOLO Custom")
    print("="*55)

    # Cek dataset
    missing = check_dataset()
    if missing:
        print("\n⚠️  Struktur dataset belum lengkap!")
        print("Folder yang dibutuhkan:")
        for m in missing:
            print(f"  - {m}")
        print("\nPetunjuk:")
        print("  Pastikan isi file zip dari Roboflow sudah di-extract langsung di dalam folder dataset/")
        return

    # Paksa buat ulang YAML yang baru biar path-nya terupdate
    create_yaml()

    # Load model base (nano = paling ringan, cocok CPU)
    base_model = f"yolov8{model_size}.pt"
    print(f"\n📦 Base Model  : {base_model}")
    print(f"🖼️  Dataset     : {DATASET_DIR}")
    print(f"🔁 Epochs      : {epochs}")
    print(f"📐 Image Size  : {img_size}")
    print(f"📦 Batch Size  : {batch}")
    print(f"💻 Device      : {device}")
    print("\nMemulai training...\n")

    model = YOLO(base_model)
    results = model.train(
        data    = DATA_YAML,
        epochs  = epochs,
        imgsz   = img_size,
        batch   = batch,
        device  = device,
        name    = "slv_model",
        project = os.path.join(BASE_DIR, "runs"),
        patience= 20,         # early stopping
        cache   = False,      # hemat RAM
        workers = 2,          # thread minimal
        verbose = True,
    )

    # Copy best.pt ke folder models/
    run_dir  = results.save_dir
    best_src = os.path.join(run_dir, "weights", "best.pt")
    best_dst = os.path.join(MODEL_DIR, "best.pt")

    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(best_src):
        shutil.copy2(best_src, best_dst)
        print("\n" + "="*55)
        print(f"✅ Training selesai!")
        print(f"✅ Model disimpan di: {best_dst}")
        print("✅ Sekarang jalankan: streamlit run src/dashboard.py")
        print("="*55 + "\n")
    else:
        print(f"[WARNING] best.pt tidak ditemukan di {best_src}")
        print(f"Cek folder runs/ untuk hasil training manual.")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SLV - Train YOLO Model")
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--batch",   type=int,   default=8)
    parser.add_argument("--device",  type=str,   default="cpu",
                        help="'cpu' atau '0' untuk GPU NVIDIA")
    parser.add_argument("--size",    type=str,   default="n",
                        choices=["n","s","m"],
                        help="Model size: n=nano, s=small, m=medium")
    args = parser.parse_args()

    train(
        epochs     = args.epochs,
        batch      = args.batch,
        device     = args.device,
        model_size = args.size,
    )