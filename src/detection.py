"""
============================================================
SLV - SmartLivestock Vision
MODULE: detection.py (STRICT ID LIMITER - MAX 5 COWS)
============================================================
Modul deteksi objek menggunakan YOLOv8 + Tracker ByteTrack
dengan pembatasan ketat maksimal 5 ID (Anti ID Melompat).
============================================================
"""

import cv2
import os
import json
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics tidak terinstall. Jalankan: pip install ultralytics")

from measurement import (
    pixels_to_mm, toy_to_real_length, toy_to_real_height,
    estimate_girth_cm, estimate_weight_schoorl,
    estimate_weight_winter, get_livestock_status
)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
CUSTOM_MODEL = os.path.join(MODEL_DIR, "best.pt")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {"mm_per_pixel": 0.15, "camera_index": 0}

COLORS = [
    (0, 255, 0),    # Sapi 1: Hijau
    (0, 200, 255),  # Sapi 2: Kuning-Biru
    (255, 100, 0),  # Sapi 3: Oranye
    (180, 0, 255),  # Sapi 4: Ungu
    (0, 255, 180),  # Sapi 5: Cyan
]

class LivestockDetector:
    def __init__(self):
        self.model  = None
        self.mode   = "contour"
        self.config = load_config()
        self._load_model()
        
        # ── MANAGEMENT ID KETAT (MAKSIMAL 5 COWS) ──
        self.id_map = {} # Menyimpan pasangan {raw_id_yolo: clean_id_1_sampai_5}
        self.available_slots = [1, 2, 3, 4, 5] # Slot ID yang boleh dipakai

    def _load_model(self):
        if not YOLO_AVAILABLE:
            print("[DETECTOR] Mode: OpenCV Contour (YOLO tidak tersedia)")
            self.mode = "contour"
            return

        if os.path.exists(CUSTOM_MODEL):
            print(f"[DETECTOR] Memuat custom model secara eksklusif: {CUSTOM_MODEL}")
            try:
                self.model = YOLO(CUSTOM_MODEL)
                self.mode  = "custom"
            except Exception as e:
                print(f"[DETECTOR] Gagal load custom model, fallback ke contour: {e}")
                self.mode = "contour"
        else:
            print(f"[WARNING] File {CUSTOM_MODEL} tidak ditemukan! Menggunakan mode Contour.")
            self.mode = "contour"

    def detect(self, frame: np.ndarray, selected_id: int = None) -> dict:
        if self.mode == "custom":
            return self._detect_yolo(frame, selected_id)
        else:
            return self._detect_contour(frame, selected_id)

    def _detect_yolo(self, frame, selected_id):
        annotated = frame.copy()
        detections = []

        h, w = frame.shape[:2]
        input_frame = cv2.resize(frame, (640, int(640 * h / w)))

        # Proses tracking menggunakan ByteTrack bawaan YOLO
        results = self.model.track(input_frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        scale_x = w / 640
        scale_y = h / (640 * h / w)

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            
            # Ambil semua raw ID dari YOLO yang aktif di frame detik ini
            active_raw_ids = [int(box.id[0]) for box in results[0].boxes if box.id is not None]
            
            # Bersihkan kamus ID map lama yang sudah tidak terdeteksi di kamera agar slot ID (1-5) bisa dipakai sapi lain
            for old_raw_id in list(self.id_map.keys()):
                if old_raw_id not in active_raw_ids:
                    # Kembalikan nomor slot ke antrean agar bisa dipakai kembali
                    released_slot = self.id_map[old_raw_id]
                    if released_slot not in self.available_slots:
                        self.available_slots.append(released_slot)
                        self.available_slots.sort() # Urutkan biar rapi dari kecil
                    del self.id_map[old_raw_id]

            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                
                if conf < 0.80:
                    continue

                if box.id is not None:
                    raw_id = int(box.id[0])
                    
                    # Logika Registrasi Slot Terbatas (Maksimal 5)
                    if raw_id not in self.id_map:
                        if self.available_slots:
                            # Ambil slot nomor terkecil yang masih kosong (1-5)
                            self.id_map[raw_id] = self.available_slots.pop(0)
                        else:
                            # Jika 5 slot sudah penuh terisi, abaikan deteksi objek ke-6 dst.
                            continue
                    
                    cow_id = self.id_map[raw_id]
                else:
                    continue

                # Batasan tameng terakhir: jika ada kebocoran angka di atas 5, potong paksa!
                if cow_id > 5:
                    continue

                x1 = int(x1 * scale_x); y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x); y2 = int(y2 * scale_y)

                # Hitung spesifikasi morfometri fisik sapi
                detection = self._calc_metrics(cow_id, x1, y1, x2, y2, conf)
                if detection is None:
                    continue

                detections.append(detection)
                color = COLORS[(cow_id - 1) % len(COLORS)]
                is_sel = (selected_id == cow_id)

                # Gambar kotak Bounding Box tracking
                thickness = 3 if is_sel else 2
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

                # Label teks dijamin maksimal hanya memunculkan angka Sapi #1 sampai Sapi #5
                label = f"Sapi #{cow_id}" + (" [DIPILIH]" if is_sel else "")
                cv2.putText(annotated, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(annotated, (cx, cy), 4, color, -1)

                if is_sel:
                    self._draw_dimension_lines(annotated, x1, y1, x2, y2, color)

        return {"detections": detections, "annotated": annotated}

    def _detect_contour(self, frame, selected_id):
        annotated = frame.copy()
        detections = []

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.THRESH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (frame.shape[0] * frame.shape[1]) * 0.002
        valid = [(cv2.contourArea(c), c) for c in contours if cv2.contourArea(c) > min_area]
        valid.sort(key=lambda x: -x[0])
        valid = valid[:5] # Batasi kontur maksimal 5 objek

        for i, (area, cnt) in enumerate(valid):
            x, y, w, h = cv2.boundingRect(cnt)
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            cow_id = i + 1 

            detection = self._calc_metrics(cow_id, x1, y1, x2, y2, conf=0.85)
            if detection is None:
                continue

            detections.append(detection)
            color  = COLORS[(cow_id - 1) % len(COLORS)]
            is_sel = (selected_id == cow_id)

            thickness = 3 if is_sel else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            label = f"Sapi #{cow_id}" + (" [DIPILIH]" if is_sel else "")
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if is_sel:
                self._draw_dimension_lines(annotated, x1, y1, x2, y2, color)

        return {"detections": detections, "annotated": annotated}

    def _calc_metrics(self, cow_id, x1, y1, x2, y2, conf):
        w_px = x2 - x1
        h_px = y2 - y1

        toy_p_mm = pixels_to_mm(w_px)
        toy_t_mm = pixels_to_mm(h_px)

        real_p_cm = toy_to_real_length(toy_p_mm)
        real_t_cm = toy_to_real_height(toy_t_mm)
        
        if real_p_cm > 280.0 or real_p_cm < 50.0 or real_t_cm > 200.0 or real_t_cm < 40.0:
            return None

        girth_cm  = estimate_girth_cm(real_p_cm, real_t_cm)
        weight_s  = estimate_weight_schoorl(girth_cm)
        weight_w  = estimate_weight_winter(real_p_cm, girth_cm)
        weight    = (weight_s + weight_w) / 2
        status    = get_livestock_status(weight)

        return {
            "id"          : cow_id,
            "bbox"        : (x1, y1, x2, y2),
            "confidence"  : round(conf, 2),
            "toy_panjang" : round(toy_p_mm, 1),
            "toy_tinggi"  : round(toy_t_mm, 1),
            "panjang_cm"  : round(real_p_cm, 1),
            "tinggi_cm"   : round(real_t_cm, 1),
            "lingkar_cm"  : girth_cm,
            "bobot_kg"    : round(weight, 1),
            "status"      : status,
        }

    def _draw_dimension_lines(self, frame, x1, y1, x2, y2, color):
        cy = (y1 + y2) // 2
        cv2.arrowedLine(frame, (x1, cy), (x2, cy), (255, 255, 255), 1, tipLength=0.05)
        cx = (x1 + x2) // 2
        cv2.arrowedLine(frame, (cx, y1), (cx, y2), (255, 255, 255), 1, tipLength=0.05)