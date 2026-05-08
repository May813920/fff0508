import streamlit as st
import streamlit.components.v1 as components

import cv2
import math
import av
import time
import threading
import base64

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from pathlib import Path
from streamlit_autorefresh import st_autorefresh

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="長照睡姿固定過久警報系統",
    page_icon="🛌",
    layout="wide"
)

# =========================
# Custom style
# =========================
st.markdown("""
<style>

.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1f3c88;
    margin-bottom: 0.2rem;
}

.sub-text {
    color: #5f6b7a;
    font-size: 1rem;
    margin-bottom: 1.2rem;
}

.metric-card {
    background-color: #f8fbff;
    border: 1px solid #dfe8f3;
    border-radius: 14px;
    padding: 16px 20px;
    text-align: center;
}

.metric-label {
    font-size: 0.95rem;
    color: #6b7280;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1f3c88;
}

.alert-box {
    background-color: #fff1f2;
    border: 1px solid #fda4af;
    color: #b91c1c;
    border-radius: 12px;
    padding: 16px;
    font-size: 1.05rem;
    font-weight: 600;
}

.normal-box {
    background-color: #f0fdf4;
    border: 1px solid #86efac;
    color: #166534;
    border-radius: 12px;
    padding: 16px;
    font-size: 1.05rem;
    font-weight: 600;
}

.sound-box {
    background-color: #fff7ed;
    border: 1px solid #fdba74;
    color: #9a3412;
    border-radius: 12px;
    padding: 12px;
    font-size: 1rem;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# Title
# =========================
st.markdown(
    '<div class="main-title">🛌 長照睡姿固定過久警報系統</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-text">使用影像分析臥床姿勢變化，協助照護員及早發現長時間未翻身狀況。</div>',
    unsafe_allow_html=True
)

# =========================
# Load YOLO model
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n-pose.pt")

model = load_model()

# =========================
# Shared state
# =========================
class AppState:

    def __init__(self):

        self.lock = threading.Lock()

        self.current_posture = "無人躺著"
        self.last_posture = "無人躺著"

        self.start_time = time.time()
        self.duration = 0.0

        self.alarm = False
        self.alarm_acknowledged = False

        self.monitoring = False

if "shared_state" not in st.session_state:
    st.session_state.shared_state = AppState()

shared_state = st.session_state.shared_state

if "sound_enabled" not in st.session_state:
    st.session_state.sound_enabled = False

if "test_alarm_sound" not in st.session_state:
    st.session_state.test_alarm_sound = False

# =========================
# Helper
# =========================
def to_xy(point):
    return float(point[0]), float(point[1])

def dist(p1, p2):
    x1, y1 = to_xy(p1)
    x2, y2 = to_xy(p2)

    return math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )

# =========================
# 姿勢分類
# =========================
def classify_posture(results):

    current_posture = "無人躺著"

    if results is None or len(results) == 0:
        return current_posture

    if results[0].keypoints is None:
        return current_posture

    if results[0].keypoints.xy is None or len(results[0].keypoints.xy) == 0:
        return current_posture

    if results[0].keypoints.conf is None or len(results[0].keypoints.conf) == 0:
        return current_posture

    kps = results[0].keypoints.xy[0]
    conf = results[0].keypoints.conf[0]

    if len(kps) < 13:
        return current_posture

    if float(conf.max()) <= 0.5:
        return current_posture

    shoulder_width = dist(kps[5], kps[6])

    torso_length = (
        dist(kps[5], kps[11]) +
        dist(kps[6], kps[12])
    ) / 2

    is_side = (
        (float(conf[5]) < 0.4 or float(conf[6]) < 0.4)
        or
        (
            torso_length > 0 and
            (shoulder_width / torso_length) < 0.5
        )
    )

    if is_side:

        if (float(conf[4]) + float(conf[6])) > (float(conf[3]) + float(conf[5])) + 0.2:
            current_posture = "左側躺"

        elif (float(conf[3]) + float(conf[5])) > (float(conf[4]) + float(conf[6])) + 0.2:
            current_posture = "右側躺"

        else:
            if dist(kps[0], kps[3]) < dist(kps[0], kps[4]):
                current_posture = "右側躺"
            else:
                current_posture = "左側躺"

    else:
        current_posture = "仰躺"

    return current_posture

# =========================
# Alarm sound
# =========================
def render_loop_alarm():

    audio_file = Path("alarm.mp3")

    if not audio_file.exists():
        st.warning("⚠️ 找不到 alarm.mp3，請確認 alarm.mp3 有放在 app.py 同一層。")
        return

    audio_bytes = audio_file.read_bytes()
    b64 = base64.b64encode(audio_bytes).decode()

    st.markdown(
        """
        <div class="sound-box">
            🔊 警報聲已觸發。若瀏覽器沒有自動播放，請手動按下方播放器的播放鍵。
        </div>
        """,
        unsafe_allow_html=True
    )

    audio_html = f"""
    <audio id="alarmAudio" controls autoplay loop style="width: 100%;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        你的瀏覽器不支援音訊播放。
    </audio>

    <script>
        const audio = document.getElementById("alarmAudio");
        audio.volume = 1.0;

        function tryPlayAlarm() {{
            const playPromise = audio.play();

            if (playPromise !== undefined) {{
                playPromise.catch(function(error) {{
                    console.log("Autoplay was blocked by the browser.");
                }});
            }}
        }}

        tryPlayAlarm();

        document.addEventListener("click", function() {{
            tryPlayAlarm();
        }}, {{ once: true }});
    </script>
    """

    components.html(audio_html, height=90)

# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ 分析設定")

alarm_threshold = st.sidebar.slider(
    "同姿勢維持幾秒觸發警報",
    min_value=3,
    max_value=60,
    value=10,
    step=1
)

# =========================
# Enable sound
# =========================
if st.sidebar.button("🔊 啟用並測試警報聲"):

    st.session_state.sound_enabled = True
    st.session_state.test_alarm_sound = True

    st.sidebar.success("警報聲已啟用，請在右側確認聲音是否能播放")

st.sidebar.markdown("---")

# =========================
# Start button
# =========================
if st.sidebar.button("▶️ Start"):

    with shared_state.lock:

        shared_state.monitoring = True

        shared_state.start_time = time.time()

        shared_state.duration = 0.0

        shared_state.alarm = False

        shared_state.alarm_acknowledged = False

        shared_state.last_posture = shared_state.current_posture

# =========================
# Stop button
# =========================
if st.sidebar.button("⏹ Stop"):

    with shared_state.lock:

        shared_state.monitoring = False

        shared_state.duration = 0.0

        shared_state.alarm = False

        shared_state.alarm_acknowledged = False

        shared_state.current_posture = "無人躺著"

        shared_state.last_posture = "無人躺著"

    st.session_state.test_alarm_sound = False

st.sidebar.markdown("---")

st.sidebar.info(
    "按下 Start 後開始監測；Stop 會停止並重新計算。"
)

# 每秒刷新
st_autorefresh(interval=1000, key="refresh")

# =========================
# Video Processor
# =========================
class PoseVideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        try:
            results = model(img, verbose=False)
            current_posture = classify_posture(results)
            annotated = results[0].plot()

        except Exception as e:
            current_posture = "偵測錯誤"
            annotated = img.copy()

            cv2.putText(
                annotated,
                f"Detection error: {str(e)[:80]}",
                (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        now = time.time()

        with shared_state.lock:

            # Start 後才開始計算
            if shared_state.monitoring:

                # 同姿勢
                if current_posture == shared_state.last_posture:

                    shared_state.duration = (
                        now - shared_state.start_time
                    )

                # 姿勢改變
                else:

                    shared_state.last_posture = current_posture

                    shared_state.current_posture = current_posture

                    shared_state.start_time = now

                    shared_state.duration = 0.0

                    shared_state.alarm = False

                    shared_state.alarm_acknowledged = False

                # Alarm 判定
                if (
                    shared_state.duration >= alarm_threshold
                    and current_posture != "無人躺著"
                    and current_posture != "偵測錯誤"
                    and not shared_state.alarm_acknowledged
                ):

                    shared_state.alarm = True

                else:

                    if (
                        current_posture == "無人躺著"
                        or current_posture == "偵測錯誤"
                        or shared_state.alarm_acknowledged
                    ):
                        shared_state.alarm = False

                shared_state.current_posture = current_posture

            else:

                shared_state.duration = 0.0

                shared_state.alarm = False

        # =========================
        # YOLO 畫圖
        # =========================
        with shared_state.lock:

            monitor_text = (
                "Monitoring"
                if shared_state.monitoring
                else "Stopped"
            )

            posture_map = {
                "無人躺著": "No person",
                "左側躺": "Left side",
                "右側躺": "Right side",
                "仰躺": "Supine",
                "偵測錯誤": "Error"
            }

            posture_en = posture_map.get(
                shared_state.current_posture,
                "Unknown"
            )

            info_text = (
                f"{monitor_text} | "
                f"Posture: {posture_en} | "
                f"Time: {int(shared_state.duration)} sec"
            )

            cv2.rectangle(
                annotated,
                (20, 20),
                (900, 70),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                annotated,
                info_text,
                (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Alarm 畫面
            if shared_state.alarm:

                cv2.rectangle(
                    annotated,
                    (0, 0),
                    (annotated.shape[1], annotated.shape[0]),
                    (0, 0, 255),
                    10
                )

                cv2.putText(
                    annotated,
                    "ALARM",
                    (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    4,
                    cv2.LINE_AA
                )

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )

# =========================
# Layout
# =========================
left_col, right_col = st.columns([1.15, 1.4])

# =========================
# Webcam
# =========================
with left_col:

    st.subheader("1. 即時影像監測")

    st.info("請允許瀏覽器開啟相機。若出現找不到相機，請確認相機權限或改用有鏡頭的裝置。")

    webrtc_streamer(
        key="pose-monitor",

        mode=WebRtcMode.SENDRECV,

        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]}
            ]
        },

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        video_processor_factory=PoseVideoProcessor,

        async_processing=True,
    )

# =========================
# Right Panel
# =========================
with right_col:

    st.subheader("2. 摘要資訊")

    with shared_state.lock:

        posture_now = shared_state.current_posture

        duration_now = int(shared_state.duration)

        alarm_now = shared_state.alarm

        monitoring_now = shared_state.monitoring

    c1, c2, c3 = st.columns(3)

    # 姿勢
    with c1:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">目前姿勢</div>
            <div class="metric-value">{posture_now}</div>
        </div>
        """, unsafe_allow_html=True)

    # 秒數
    with c2:

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">持續時間</div>
            <div class="metric-value">{duration_now} 秒</div>
        </div>
        """, unsafe_allow_html=True)

    # 狀態
    with c3:

        system_text = (
            "監測中"
            if monitoring_now
            else "停止"
        )

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">系統狀態</div>
            <div class="metric-value">{system_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # 測試警報聲
    # =========================
    if st.session_state.test_alarm_sound:

        st.subheader("🔔 警報聲測試")

        render_loop_alarm()

        if st.button("停止測試警報聲"):
            st.session_state.test_alarm_sound = False
            st.rerun()

    # =========================
    # Alarm 區
    # =========================
    st.subheader("3. 警報摘要")

    if alarm_now:

        st.markdown(f"""
        <div class="alert-box">
            🚨 偵測到姿勢持續超過 {alarm_threshold} 秒，
            請協助翻身。
        </div>
        """, unsafe_allow_html=True)

        # Alarm 聲
        render_loop_alarm()

        # 確認按鈕
        if st.button("✅ 確認此資訊", type="primary"):

            with shared_state.lock:

                shared_state.alarm_acknowledged = True

                shared_state.alarm = False

            st.rerun()

    else:

        st.markdown("""
        <div class="normal-box">
            ✅ 目前尚未觸發警報
        </div>
        """, unsafe_allow_html=True)
