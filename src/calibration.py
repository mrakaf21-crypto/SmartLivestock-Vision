"""
============================================================
SLV - SmartLivestock Vision
STEP 1: Kalibrasi Kamera
============================================================
Jalankan file ini PERTAMA KALI sebelum apapun.
Fungsi: Menentukan berapa mm per 1 pixel di kamera lo.

Cara pakai:
1. Siapkan penggaris fisik
2. Taruh penggaris di depan kamera, di posisi yang SAMA
   dengan tempat sapi mainan akan diletakkan
3. Jalankan: python src/calibration.py
4. Klik titik KIRI penggaris, lalu titik KANAN penggaris
   (misal: dari angka 0 sampai angka 30)
5. Masukkan jarak aslinya dalam mm
6. File config.py akan otomatis tersimpan
============================================================
"""

import cv2
import json
import os

# ── State kalibrasi ──────────────────────────────────────
points = []
frame_copy = None

def click_event(event, x, y, flags, param):
    global points, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        points.append((x, y))
        cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(frame_copy, f"P{len(points)}", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(points) == 2:
            cv2.line(frame_copy, points[0], points[1], (0, 255, 255), 2)
        cv2.imshow("Kalibrasi SLV", frame_copy)

def run_calibration():
    global points, frame_copy

    print("\n" + "="*55)
    print("  SLV - SmartLivestock Vision | MODE KALIBRASI")
    print("="*55)
    print("  1. Siapkan penggaris di depan kamera")
    print("  2. Tekan SPASI untuk ambil frame")
    print("  3. Klik 2 titik di penggaris (misal: 0mm → 30mm)")
    print("  4. Masukkan jarak aslinya dalam mm")
    print("  Tekan Q untuk keluar tanpa menyimpan")
    print("="*55 + "\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak terdeteksi!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Kamera aktif. Tekan SPASI untuk freeze frame...")
    frozen = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal baca frame dari kamera.")
            break

        if not frozen:
            frame_copy = frame.copy()
            cv2.putText(frame_copy, "Siapkan penggaris | SPASI = Freeze | Q = Keluar",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Kalibrasi SLV", frame_copy)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Kalibrasi dibatalkan.")
            break

        if key == ord(' ') and not frozen:
            frozen = True
            frame_copy = frame.copy()
            cv2.putText(frame_copy,
                        "Klik 2 titik pada penggaris (titik awal & akhir)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Kalibrasi SLV", frame_copy)
            cv2.setMouseCallback("Kalibrasi SLV", click_event)
            print("Frame dibekukan. Klik 2 titik pada penggaris...")

        if frozen and len(points) == 2:
            pixel_dist = ((points[1][0] - points[0][0])**2 +
                          (points[1][1] - points[0][1])**2) ** 0.5
            print(f"\nJarak pixel antara 2 titik: {pixel_dist:.2f} px")
            real_mm = float(input("Masukkan jarak asli dalam mm (misal: 30): "))

            mm_per_pixel = real_mm / pixel_dist

            # Skala sapi mainan ke sapi asli
            # Mainan: P=31mm, T=19mm | Asli: P≈1500mm, T≈1200mm (Sapi Bali)
            toy_length_mm   = 31.0
            toy_height_mm   = 19.0
            real_length_mm  = 1500.0   # Rata-rata sapi lokal dewasa
            real_height_mm  = 1200.0
            scale_length    = real_length_mm / toy_length_mm
            scale_height    = real_height_mm / toy_height_mm
            scale_weight    = ((scale_length + scale_height) / 2) ** 3

            config = {
                "mm_per_pixel"   : round(mm_per_pixel, 6),
                "toy_length_mm"  : toy_length_mm,
                "toy_height_mm"  : toy_height_mm,
                "real_length_mm" : real_length_mm,
                "real_height_mm" : real_height_mm,
                "scale_length"   : round(scale_length, 4),
                "scale_height"   : round(scale_height, 4),
                "scale_weight"   : round(scale_weight, 4),
                "camera_index"   : 0
            }

            os.makedirs("src", exist_ok=True)
            with open("src/config.json", "w") as f:
                json.dump(config, f, indent=4)

            print("\n" + "="*55)
            print(f"  ✅ Kalibrasi berhasil disimpan ke src/config.json")
            print(f"  mm per pixel  : {mm_per_pixel:.4f} mm/px")
            print(f"  Skala panjang : 1 : {scale_length:.1f}")
            print(f"  Skala tinggi  : 1 : {scale_height:.1f}")
            print(f"  Skala bobot   : 1 : {scale_weight:.0f}")
            print("="*55 + "\n")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_calibration()
