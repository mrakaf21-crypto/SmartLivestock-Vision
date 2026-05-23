"""
============================================================
SLV - SmartLivestock Vision
MAIN APP: dashboard.py (CLEAN & STREAMLINED + FIXED DUPLICATE ID)
============================================================
Jalankan dengan:  streamlit run src/dashboard.py
============================================================
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
import sys
import time
from datetime import datetime

# Tambahkan path supaya import modul lokal bisa jalan
sys.path.insert(0, os.path.dirname(__file__))

from measurement import calculate_uniformity, get_livestock_status
from detection import LivestockDetector

# ── Config Halaman ────────────────────────────────────────
st.set_page_config(
    page_title="SmartLivestock Vision",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* Background utama */
    .stApp { background-color: #0f1117; }

    /* Card metrik */
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #262b3e);
        border: 1px solid #3a4060;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
        text-align: center;
    }

    /* Header app */
    .app-header {
        background: linear-gradient(90deg, #0f3460, #16213e);
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 20px;
        border-left: 4px solid #64ffda;
    }
    .app-title {
        color: #64ffda;
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0;
    }
    .app-subtitle {
        color: #8892b0;
        font-size: 0.85rem;
        margin-top: 4px;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────
def init_state():
    defaults = {
        "detector"      : None,
        "camera_active" : False,
        "detections"    : [],
        "history"       : [],       # list of weight per detection
        "cv_trend"      : [],       # Menyimpan log trend CV [{Waktu, CV}]
        "frame_count"   : 0,
        "skip_frame"    : 3,        # proses setiap N frame (hemat CPU)
        "annotated"     : None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Load Konfigurasi ──────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <p class="app-title">🐄 SmartLivestock Vision (SLV)</p>
    <p class="app-subtitle">
        Sistem Monitoring Dimensi, Bobot & Keseragaman Ternak Berbasis AI • Real-time
    </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Kontrol Sistem")
    st.divider()

    cam_index = st.selectbox("📷 Index Kamera", [0, 1, 2], help="0 = kamera bawaan laptop")
    skip = st.slider("🔄 Skip Frame (hemat CPU)", min_value=1, max_value=10, value=3)
    st.session_state["skip_frame"] = skip
    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("▶ START", use_container_width=True, type="primary"):
            st.session_state["camera_active"] = True
            if st.session_state["detector"] is None:
                with st.spinner("Memuat model AI..."):
                    st.session_state["detector"] = LivestockDetector()
            st.success("Kamera aktif!")

    with col_b:
        if st.button("⏹ STOP", use_container_width=True):
            st.session_state["camera_active"] = False
            st.info("Kamera dihentikan.")

    st.divider()
    input_mode = st.radio("📁 Mode Input", ["Kamera Live", "Upload Video", "Upload Foto"])

    uploaded_file = None
    if input_mode == "Upload Video":
        uploaded_file = st.file_uploader("Upload Video (.mp4, .avi)", type=["mp4", "avi", "mov"])
    elif input_mode == "Upload Foto":
        uploaded_file = st.file_uploader("Upload Foto (.jpg, .png)", type=["jpg", "jpeg", "png"])

    st.divider()
    cfg = load_config()
    if cfg:
        st.markdown("**📐 Kalibrasi Aktif**")
        st.code(f"mm/pixel : {cfg.get('mm_per_pixel', 'N/A')}\nSkala P  : 1 : {cfg.get('scale_length', 'N/A')}\nSkala T  : 1 : {cfg.get('scale_height', 'N/A')}")
    else:
        st.warning("⚠️ Kalibrasi belum dilakukan!\nJalankan: python src/calibration.py")

    st.divider()
    if st.button("🗑️ Reset Semua Data", use_container_width=True):
        st.session_state["history"] = []
        st.session_state["detections"] = []
        st.session_state["cv_trend"] = []
        st.rerun()

# ══════════════════════════════════════════════════════════
# LAYOUT UTAMA (LEBAR PENUH)
# ══════════════════════════════════════════════════════════
st.markdown("#### 📹 Live Feed Kamera AI")
video_placeholder = st.empty()
status_placeholder = st.empty()

st.divider()
col_list, col_uni = st.columns([2, 3], gap="medium")

with col_list:
    st.markdown("#### 📋 Daftar Sapi Terdeteksi Aktual")
    list_placeholder = st.empty()

with col_uni:
    st.markdown("#### 📊 Analisis Keseragaman Kelompok")
    uni_placeholder = st.empty()

st.divider()
col_hist, col_chart = st.columns([1, 1], gap="medium")

with col_hist:
    st.markdown("#### 📈 Riwayat Deteksi & Spesifikasi Fisik")
    history_placeholder = st.empty()

with col_chart:
    st.markdown("#### 📉 Grafik Tren Keseragaman Kandang (CV)")
    chart_placeholder = st.empty()

# ══════════════════════════════════════════════════════════
# FUNGSI PROSES FRAME
# ══════════════════════════════════════════════════════════
def process_frame(frame: np.ndarray):
    fc = st.session_state["frame_count"]
    st.session_state["frame_count"] = fc + 1

    if fc % st.session_state["skip_frame"] != 0:
        return st.session_state.get("annotated", frame)

    det = st.session_state["detector"]
    if det is None: return frame

    result = det.detect(frame, selected_id=None)
    st.session_state["detections"] = result["detections"]
    st.session_state["annotated"]  = result["annotated"]

    for d in result["detections"]:
        st.session_state["history"].append({
            "Waktu"      : datetime.now().strftime("%H:%M:%S"),
            "ID Sapi"    : f"#{d['id']}",
            "Panjang(cm)": d["panjang_cm"],
            "Tinggi(cm)" : d["tinggi_cm"],
            "Lingkar(cm)": d["lingkar_cm"],
            "Bobot(kg)"  : d["bobot_kg"],
            "Status"     : d["status"]["label"],
        })
        if len(st.session_state["history"]) > 200:
            st.session_state["history"] = st.session_state["history"][-200:]

    return result["annotated"]

# ══════════════════════════════════════════════════════════
# FUNGSI UPDATE DASHBOARD & GRAFIK TREN
# ══════════════════════════════════════════════════════════
def update_side_panels():
    detections = st.session_state["detections"]
    history    = st.session_state["history"]
    cv_trend   = st.session_state["cv_trend"]
    current_time = datetime.now().strftime("%H:%M:%S")
    unique_ms = int(time.time() * 1000) # Generator ID unik per milidetik

    # 1. Render Daftar Sapi
    with list_placeholder.container():
        if detections:
            for d in detections:
                st.markdown(f"🔹 **Sapi #{d['id']}** — `{d['bobot_kg']} kg` — {d['status']['label']}")
        else:
            st.markdown("<p style='color:#8892b0;'>Belum ada sapi terdeteksi di meja.</p>", unsafe_allow_html=True)

    # 2. Render Keseragaman & Rekam Tren CV ke Memori
    weights = [d["bobot_kg"] for d in detections] if detections else []
    with uni_placeholder.container():
        uni = calculate_uniformity(weights)
        if uni["count"] < 2:
            st.info(uni["status"])
        else:
            st.markdown(f"##### 📊 Status Kelompok: {uni['status']}")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1: st.metric(label="Rata-rata Bobot", value=f"{uni['mean']} kg")
            with m_col2: st.metric(label="Standar Deviasi", value=f"±{uni['std']} kg")
            with m_col3: st.metric(label="Koefisien Variasi (CV)", value=f"{uni['cv']}%")
            st.progress(int(uni["uniformity"]) / 100, text=f"Index Keseragaman: {uni['uniformity']}%")
            st.caption(f"Rentang Data Kelompok: Min {uni['min']} kg | Max {uni['max']} kg ({uni['count']} Sapi Aktif)")

            if not cv_trend or cv_trend[-1]["Waktu"] != current_time:
                cv_trend.append({"Waktu": current_time, "Koefisien Variasi (%)": float(uni["cv"])})
                if len(cv_trend) > 40: cv_trend.pop(0)
                st.session_state["cv_trend"] = cv_trend

    # 3. Render Tabel Riwayat & Mengamankan Key Komponen Tombol
    with history_placeholder.container():
        if history:
            df_full = pd.DataFrame(history)
            st.markdown(f"🔢 Total Data: `{len(df_full)}` | 👑 Terberat: `{df_full['Bobot(kg)'].max()} kg`")
            st.dataframe(df_full[-30:].iloc[::-1], use_container_width=True, height=160)
            
            c1, c2 = st.columns(2)
            with c1:
                # FIXED CRITICAL BUG: Menyisipkan key unik dinamis agar tidak kena Duplicate ID error
                st.download_button(
                    "⬇️ Export CSV", 
                    data=df_full.to_csv(index=False), 
                    file_name="SLV_data.csv", 
                    mime="text/csv", 
                    key=f"dl_btn_{unique_ms}", 
                    use_container_width=True
                )
            with c2:
                if st.button("🗑️ Hapus Baris Terakhir", key=f"del_btn_{unique_ms}", use_container_width=True):
                    if st.session_state["history"]:
                        st.session_state["history"].pop()
                        if st.session_state["cv_trend"]: st.session_state["cv_trend"].pop()
                        st.success("Baris terakhir sukses dihapus!")
                        time.sleep(0.3)
                        st.rerun()
        else:
            st.markdown("<p style='color:#8892b0;'>Belum ada riwayat data.</p>", unsafe_allow_html=True)

    # 4. Menggambar Line Chart Tren Keseragaman CV
    with chart_placeholder.container():
        if cv_trend and len(cv_trend) >= 2:
            df_chart = pd.DataFrame(cv_trend)
            st.line_chart(df_chart.set_index("Waktu"), y="Koefisien Variasi (%)", height=210)
        else:
            st.info("💡 Menunggu data terkumpul untuk memplot grafik tren variasi kandang...")

# ── Mode Execution ────────────────────────────────────────
if input_mode == "Upload Foto" and uploaded_file:
    if st.session_state["detector"] is None: st.session_state["detector"] = LivestockDetector()
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    annotated = process_frame(frame)
    with video_placeholder: st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    update_side_panels()

elif input_mode == "Upload Video" and uploaded_file:
    if st.session_state["detector"] is None: st.session_state["detector"] = LivestockDetector()
    tfile = f"/tmp/slv_video_{int(time.time())}.mp4"
    with open(tfile, "wb") as f: f.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        annotated = process_frame(frame)
        with video_placeholder: st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        update_side_panels()
        time.sleep(0.03)
    cap.release()

elif input_mode == "Kamera Live":
    if st.session_state["camera_active"]:
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            stop_cam = st.button("⏹ Stop Kamera", key="stop_cam_live")
            fps_display = status_placeholder.empty()

            while st.session_state["camera_active"] and not stop_cam:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret: break

                annotated = process_frame(frame)
                with video_placeholder: st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                update_side_panels()
                fps = 1 / (time.time() - t0 + 1e-9)
                fps_display.markdown(f"<p style='color:#64ffda; font-size:0.8rem;'>FPS: {fps:.1f}</p>", unsafe_allow_html=True)
            cap.release()
            if stop_cam:
                st.session_state["camera_active"] = False
                st.rerun()
    else:
        with video_placeholder:
            st.markdown("""
            <div style="background:#1a1f2e; border-radius:12px; padding:80px 20px; text-align:center; border: 1px dashed #3a4060; min-height:350px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
                <div style="font-size:4rem;">🐄</div>
                <div style="color:#64ffda; font-size:1.2rem; font-weight:700; margin:12px 0;">SmartLivestock Vision</div>
                <div style="color:#8892b0;">Tekan <b>▶ START</b> di sidebar untuk mengaktifkan kamera</div>
            </div>
            """, unsafe_allow_html=True)
        update_side_panels()