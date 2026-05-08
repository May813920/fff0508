import streamlit as st
import streamlit.components.v1 as components

import cv2
import math
import av
import time
import threading
import base64
from pathlib import Path

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
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
    padding: 14px;
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

if "preheat_audio" not in st.session_state:
    st.session_state.preheat_audio = False


# =========================
# Helper functions
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

    result = results[0]

    if result.keypoints is None:
        return current_posture

    if result.keypoints.xy is None or len(result.keypoints.xy) == 0:
        return current_posture

    if result.keypoints.conf is None or len(result.keypoints.conf) == 0:
        return current_posture

    kps = result.keypoints.xy[0]
    conf = result.keypoints.conf[0]

    if len(kps) < 13:
        return current_posture

    if float(conf.max()) <= 0.5:
        return current_posture

    shoulder_width = dist(kps[5], kps[6])

    torso_length = (
        dist(kps[5], kps[11]) +
        dist(kps[6], kps[12])
    ) / 2

    left_shoulder_conf = float(conf[5])
    right_shoulder_conf = float(conf[6])

    is_side = (
        left_shoulder_conf < 0.4
        or right_shoulder_conf < 0.4
        or (
            torso_length > 0
            and (shoulder_width / torso_length) < 0.5
        )
    )

    if is_side:
        left_ear_conf = float(conf[3])
        right_ear_conf = float(conf[4])

        if (right_ear_conf + right_shoulder_conf) > (
            left_ear_conf + left_shoulder_conf
        ) + 0.2:
            current_posture = "左側躺"

        elif (left_ear_conf + left_shoulder_conf) > (
            right_ear_conf + right_shoulder_conf
        ) + 0.2:
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
# 讀取音效
# =========================
def get_alarm_audio_base64():
    audio_file = Path("alarm.mp3")

    if not audio_file.exists():
        return None

    audio_bytes = audio_file.read_bytes()
    return base64.b64encode(audio_bytes).decode()


# =========================
# Start 後音訊預熱
# =========================
def render_audio_preheat():
    b64 = get_alarm_audio_base64()

    if b64 is None:
        st.warning("⚠️ 找不到 alarm.mp3，請確認 alarm.mp3 有放在 app.py 同一層。")
        return

    st.markdown(
        """
        <div class="sound-box">
            🔊 音訊預熱：系統會嘗試短暫播放警報聲，讓瀏覽器允許後續警報自動播放。
            如果沒有成功，請按下方「手動啟用警報聲」。
        </div>
        """,
        unsafe_allow_html=True
    )

    preheat_html = f"""
    <div style="margin-top: 10px;">
        <audio id="preheatAudio">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>

        <button onclick="manualPreheat()" 
            style="
                background-color:#2563eb;
                color:white;
                border:none;
                border-radius:10px;
                padding:12px 20px;
                font-size:18px;
                font-weight:700;
                cursor:pointer;
                margin-right:10px;
            ">
            🔊 手動啟用警報聲
        </button>

        <span id="preheatStatus" style="font-size:16px; color:#166534; font-weight:700;"></span>

        <script>
            const preheatAudio = document.getElementById("preheatAudio");
            const preheatStatus = document.getElementById("preheatStatus");

            function doPreheat() {{
                preheatAudio.volume = 0.15;
                preheatAudio.currentTime = 0;

                preheatAudio.play().then(() => {{
                    preheatStatus.innerText = "音訊已預熱";
                    setTimeout(() => {{
                        preheatAudio.pause();
                        preheatAudio.currentTime = 0;
                        preheatAudio.volume = 1.0;
                    }}, 300);
                }}).catch((error) => {{
                    preheatStatus.innerText = "自動預熱被瀏覽器阻擋，請按左側按鈕";
                    console.log("Audio preheat blocked:", error);
                }});
            }}

            function manualPreheat() {{
                doPreheat();
            }}

            doPreheat();
        </script>
    </div>
    """

    components.html(preheat_html, height=95)


# =========================
# 警報聲播放
# =========================
def render_loop_alarm(autoplay=True):
    b64 = get_alarm_audio_base64()

    if b64 is None:
        st.warning("⚠️ 找不到 alarm.mp3，請確認 alarm.mp3 有放在 app.py 同一層。")
        return

    st.markdown(
        """
        <div class="sound-box">
            🔊 警報聲已觸發。系統會嘗試自動播放；若沒有聲音，請按「播放警報聲」。
        </div>
        """,
        unsafe_allow_html=True
    )

    auto_play_script = """
        alarmAudio.play().catch(function(error) {
            alarmStatus.innerText = "自動播放被瀏覽器阻擋，請按播放按鈕";
            console.log("Autoplay blocked:", error);
        });
    """ if autoplay else ""

    alarm_html = f"""
    <div style="margin-top: 12px;">
        <audio id="alarmAudio" loop>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            你的瀏覽器不支援音訊播放。
        </audio>

        <button onclick="playAlarm()" 
            style="
                background-color:#dc2626;
                color:white;
                border:none;
                border-radius:10px;
                padding:12px 20px;
                font-size:18px;
                font-weight:700;
                cursor:pointer;
                margin-right:10px;
            ">
            🔊 播放警報聲
        </button>

        <button onclick="stopAlarm()" 
            style="
                background-color:#4b5563;
                color:white;
                border:none;
                border-radius:10px;
                padding:12px 20px;
                font-size:18px;
                font-weight:700;
                cursor:pointer;
                margin-right:10px;
            ">
            ⏹ 停止警報聲
        </button>

        <span id="alarmStatus" style="font-size:16px; color:#b91c1c; font-weight:700;"></span>

        <script>
            const alarmAudio = document.getElementById("alarmAudio");
            const alarmStatus = document.getElementById("alarmStatus");

            function playAlarm() {{
                alarmAudio.currentTime = 0;
                alarmAudio.play().then(() => {{
                    alarmStatus.innerText = "警報聲播放中";
                }}).catch((error) => {{
                    alarmStatus.innerText = "播放失敗，請再按一次";
                    console.log("Play failed:", error);
                }});
            }}

            function stopAlarm() {{
                alarmAudio.pause();
                alarmAudio.currentTime = 0;
                alarmStatus.innerText = "警報聲已停止";
            }}

            {auto_play_script}
        </script>
    </div>
    """

    components.html(alarm_html, height=110)


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

if st.sidebar.button("🔔 測試警報聲"):
    st.session_state.test_alarm_sound = True
    st.session_state.sound_enabled = True

st.sidebar.markdown("---")


# =========================
# Start button
# =========================
if st.sidebar.button("▶️ Start 啟動監測"):
    with shared_state.lock:
        shared_state.monitoring = True
        shared_state.start_time = time.time()
        shared_state.duration = 0.0
        shared_state.alarm = False
        shared_state.alarm_acknowledged = False
        shared_state.last_posture = shared_state.current_posture

    st.session_state.sound_enabled = True
    st.session_state.preheat_audio = True
    st.toast("系統已啟動，正在嘗試開啟攝影機與預熱警報聲")


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
    st.session_state.preheat_audio = False

st.sidebar.markdown("---")

st.sidebar.info(
    "按下 Start 後開始監測，系統會嘗試開啟攝影機並預熱警報音訊。"
)


# =========================
# Autorefresh
# =========================
# 這裡維持每秒刷新，所以警報觸發後右側資訊仍會繼續更新。
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
            if shared_state.monitoring:
                if current_posture == shared_state.last_posture:
                    shared_state.duration = now - shared_state.start_time

                else:
                    shared_state.last_posture = current_posture
                    shared_state.current_posture = current_posture
                    shared_state.start_time = now
                    shared_state.duration = 0.0
                    shared_state.alarm = False
                    shared_state.alarm_acknowledged = False

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
                shared_state.current_posture = current_posture

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
# 讀取目前監測狀態
# =========================
with shared_state.lock:
    monitoring_for_webrtc = shared_state.monitoring


# =========================
# Layout
# =========================
left_col, right_col = st.columns([1.15, 1.4])


# =========================
# Webcam
# =========================
with left_col:
    st.subheader("1. 即時影像監測")

    st.info(
        "按左側 Start 後，系統會嘗試啟動攝影機。若瀏覽器跳出權限視窗，請按允許。"
    )

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
        desired_playing=monitoring_for_webrtc,
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

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">目前姿勢</div>
            <div class="metric-value">{posture_now}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">持續時間</div>
            <div class="metric-value">{duration_now} 秒</div>
        </div>
        """, unsafe_allow_html=True)

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
    # Start 後音訊預熱
    # =========================
    if monitoring_now and st.session_state.preheat_audio and not alarm_now:
        st.subheader("🔊 音訊預熱")
        render_audio_preheat()


    # =========================
    # 測試警報聲
    # =========================
    if st.session_state.test_alarm_sound:
        st.subheader("🔔 警報聲測試")
        render_loop_alarm(autoplay=True)

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

        render_loop_alarm(autoplay=True)

        if st.button("✅ 確認此資訊", type="primary"):
            with shared_state.lock:
                shared_state.alarm_acknowledged = True
                shared_state.alarm = False
                shared_state.start_time = time.time()
                shared_state.duration = 0.0
                shared_state.last_posture = shared_state.current_posture

            st.session_state.preheat_audio = False
            st.rerun()

    else:
        st.markdown("""
        <div class="normal-box">
            ✅ 目前尚未觸發警報
        </div>
        """, unsafe_allow_html=True)
