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

.record-box {
    background-color: #eef6ff;
    border: 1px solid #93c5fd;
    color: #1e3a8a;
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
    '<div class="sub-text">使用即時影像分析臥床姿勢變化，協助照護員及早發現長時間未翻身狀況。</div>',
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

        # 降低即時偵測負擔用
        self.frame_count = 0
        self.last_detect_time = 0
        self.last_annotated = None

        # 錄影相關
        self.recording = False
        self.record_writer = None
        self.record_path = None
        self.record_size = None


if "shared_state" not in st.session_state:
    st.session_state.shared_state = AppState()

shared_state = st.session_state.shared_state


# =========================
# Session state
# =========================
if "sound_enabled" not in st.session_state:
    st.session_state.sound_enabled = False

if "test_alarm_sound" not in st.session_state:
    st.session_state.test_alarm_sound = False

if "recorded_video_path" not in st.session_state:
    st.session_state.recorded_video_path = None

# 警報時間設定，放在 session_state，避免每次重新執行後被重設
if "alarm_minutes" not in st.session_state:
    st.session_state.alarm_minutes = 0

if "alarm_seconds" not in st.session_state:
    st.session_state.alarm_seconds = 10

if "alarm_threshold" not in st.session_state:
    st.session_state.alarm_threshold = 10

if "alarm_time_text" not in st.session_state:
    st.session_state.alarm_time_text = "10 秒"


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


def resize_frame(img, max_width=640):
    """
    手機鏡頭解析度可能很高，先縮小再偵測，避免 Railway 跑太慢。
    """
    h, w = img.shape[:2]

    if w > max_width:
        scale = max_width / w
        new_h = int(h * scale)
        img = cv2.resize(img, (max_width, new_h))

    return img


def format_alarm_time(minutes, seconds):
    """
    將警報時間轉成好看的文字。
    如果秒數是 0，就不顯示秒數。
    """
    if minutes > 0 and seconds > 0:
        return f"{minutes} 分 {seconds} 秒"

    elif minutes > 0 and seconds == 0:
        return f"{minutes} 分"

    elif minutes == 0 and seconds > 0:
        return f"{seconds} 秒"

    else:
        return "1 秒"


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
            🔊 警報聲已觸發。若瀏覽器沒有自動播放，請按下方「播放警報聲」按鈕。
        </div>
        """,
        unsafe_allow_html=True
    )

    audio_html = f"""
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
            ">
            ⏹ 停止警報聲
        </button>

        <script>
            const audio = document.getElementById("alarmAudio");

            function playAlarm() {{
                audio.currentTime = 0;
                audio.play();
            }}

            function stopAlarm() {{
                audio.pause();
                audio.currentTime = 0;
            }}

            audio.play().catch(function(error) {{
                console.log("Autoplay was blocked by browser.");
            }});
        </script>
    </div>
    """

    components.html(audio_html, height=100)


# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ 分析設定")

# =========================
# Alarm time setting form
# =========================
st.sidebar.subheader("⏰ 警報時間設定")

with st.sidebar.form("alarm_time_form"):
    new_alarm_minutes = st.number_input(
        "分鐘",
        min_value=0,
        max_value=120,
        value=st.session_state.alarm_minutes,
        step=1
    )

    new_alarm_seconds = st.number_input(
        "秒",
        min_value=0,
        max_value=59,
        value=st.session_state.alarm_seconds,
        step=1
    )

    apply_alarm_time = st.form_submit_button("套用警報時間")

if apply_alarm_time:
    new_threshold = new_alarm_minutes * 60 + new_alarm_seconds

    if new_threshold <= 0:
        new_threshold = 1
        new_alarm_minutes = 0
        new_alarm_seconds = 1
        st.sidebar.warning("警報時間不能是 0，系統已自動改為 1 秒。")

    st.session_state.alarm_minutes = new_alarm_minutes
    st.session_state.alarm_seconds = new_alarm_seconds
    st.session_state.alarm_threshold = new_threshold
    st.session_state.alarm_time_text = format_alarm_time(
        new_alarm_minutes,
        new_alarm_seconds
    )

alarm_threshold = st.session_state.alarm_threshold
alarm_time_text = st.session_state.alarm_time_text

st.sidebar.info(
    f"目前設定：{alarm_time_text} 後觸發警報"
)

st.sidebar.markdown("---")


# =========================
# Sound buttons
# =========================
if st.sidebar.button("🔊 啟用警報聲"):
    st.session_state.sound_enabled = True
    st.sidebar.success("警報聲已啟用")

if st.sidebar.button("🔔 測試警報聲"):
    st.session_state.sound_enabled = True
    st.session_state.test_alarm_sound = True

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


# =========================
# Recording buttons
# =========================
st.sidebar.subheader("🎥 錄影功能")

if st.sidebar.button("⏺ 開始錄影"):
    with shared_state.lock:
        shared_state.recording = True
        shared_state.record_path = f"/tmp/pose_record_{int(time.time())}.mp4"
        shared_state.record_writer = None
        shared_state.record_size = None

    st.session_state.recorded_video_path = None
    st.sidebar.success("已開始錄影")

if st.sidebar.button("⏹ 停止錄影"):
    with shared_state.lock:
        shared_state.recording = False

        if shared_state.record_writer is not None:
            shared_state.record_writer.release()
            shared_state.record_writer = None

        st.session_state.recorded_video_path = shared_state.record_path

    st.sidebar.success("已停止錄影，可在右側下載影片")

with shared_state.lock:
    recording_now = shared_state.recording

if recording_now:
    st.sidebar.warning("🔴 錄影中")
else:
    st.sidebar.info("目前未錄影")

st.sidebar.markdown("---")

st.sidebar.info(
    "建議先設定警報時間，再按 Start 與開始錄影。錄影中不要調整側邊欄設定，避免鏡頭重新連線。"
)


# =========================
# Video Processor
# =========================
class PoseVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # 降低解析度，避免手機鏡頭畫面太大造成延遲
        img = resize_frame(img, max_width=640)

        try:
            results = model(img, verbose=False, imgsz=640)
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

            record_text = " | REC" if shared_state.recording else ""

            info_text = (
                f"{monitor_text}{record_text} | "
                f"Posture: {posture_en} | "
                f"Time: {int(shared_state.duration)} sec"
            )

            cv2.rectangle(
                annotated,
                (20, 20),
                (min(900, annotated.shape[1] - 20), 70),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                annotated,
                info_text,
                (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            if shared_state.recording:
                cv2.circle(
                    annotated,
                    (30, 95),
                    10,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    annotated,
                    "REC",
                    (50, 103),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
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
                    (30, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    4,
                    cv2.LINE_AA
                )

            # =========================
            # 錄影：寫入偵測後畫面
            # =========================
            if shared_state.recording:
                h, w = annotated.shape[:2]

                if (
                    shared_state.record_writer is None
                    or shared_state.record_size != (w, h)
                ):
                    shared_state.record_size = (w, h)

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    shared_state.record_writer = cv2.VideoWriter(
                        shared_state.record_path,
                        fourcc,
                        10.0,
                        (w, h)
                    )

                if shared_state.record_writer is not None:
                    shared_state.record_writer.write(annotated)

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

    st.info("請允許瀏覽器開啟相機，按下 Start 後開始監測。")

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
            "video": {
                "width": {"ideal": 640, "max": 960},
                "height": {"ideal": 480, "max": 720},
                "frameRate": {"ideal": 10, "max": 15},
            },
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
        alarm_now = shared_state.alarm
        monitoring_now = shared_state.monitoring

    if (
        shared_state.monitoring
        and shared_state.current_posture != "無人躺著"
        and shared_state.current_posture != "偵測錯誤"
    ):
        duration_now = int(time.time() - shared_state.start_time)
        shared_state.duration = duration_now
    else:
        duration_now = int(shared_state.duration)

    if (
        shared_state.monitoring
        and shared_state.current_posture != "無人躺著"
        and shared_state.current_posture != "偵測錯誤"
        and duration_now >= alarm_threshold
        and not shared_state.alarm_acknowledged
    ):
        shared_state.alarm = True
        alarm_now = True

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
        system_text = "錄影中" if recording_now else ("監測中" if monitoring_now else "停止")

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
            🚨 偵測到姿勢持續超過 {alarm_time_text}，
            請協助翻身。
        </div>
        """, unsafe_allow_html=True)

        render_loop_alarm()

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

    # =========================
    # Recording download
    # =========================
    st.markdown("---")
    st.subheader("4. 錄影下載")

    recorded_path = st.session_state.get("recorded_video_path")

    if (
        recorded_path is not None
        and Path(recorded_path).exists()
        and Path(recorded_path).stat().st_size > 0
    ):
        st.markdown(
            """
            <div class="record-box">
                ✅ 錄影完成，可以下載影片。
            </div>
            """,
            unsafe_allow_html=True
        )

        with open(recorded_path, "rb") as f:
            st.download_button(
                label="📥 下載錄影影片",
                data=f,
                file_name="pose_record.mp4",
                mime="video/mp4"
            )

    else:
        st.info("尚未產生錄影檔。請先按「開始錄影」，再按「停止錄影」。")


    st.markdown("---")

    st.subheader("5. 使用說明")

    st.markdown(
        """
        1. 開啟網頁後，請允許瀏覽器使用相機。  
        2. 先設定警報時間，按 **套用警報時間**。  
        3. 按左側 **Start** 開始即時姿勢監測。  
        4. 若要錄影，按左側 **⏺ 開始錄影**。  
        5. 錄影時畫面會顯示 **REC**。  
        6. 錄影中請不要調整側邊欄設定，避免相機重新連線。  
        7. 按 **⏹ 停止錄影** 後，右側會出現下載按鈕。  
        8. 若同一姿勢維持超過設定時間，會觸發警報。  
        9. 若使用 iPhone，警報聲可能需要手動按「播放警報聲」才會響。
        """
    )
