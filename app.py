import streamlit as st
import streamlit.components.v1 as components

import cv2
import math
import av
import time
import threading
import base64
import tempfile
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageOps
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
st.markdown(
    """
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

.result-box {
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
""",
    unsafe_allow_html=True,
)


# =========================
# Title
# =========================
st.markdown(
    '<div class="main-title">🛌 長照睡姿固定過久警報系統</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-text">使用影像分析臥床姿勢變化，協助照護員及早發現長時間未翻身狀況。</div>',
    unsafe_allow_html=True,
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

        # 降低即時偵測負擔
        self.last_detect_time = 0.0
        self.last_annotated = None


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
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def resize_frame(img, max_width=640):
    h, w = img.shape[:2]

    if w > max_width:
        scale = max_width / w
        new_h = int(h * scale)
        img = cv2.resize(img, (max_width, new_h))

    return img


def format_alarm_time(minutes, seconds):
    if minutes > 0 and seconds > 0:
        return f"{minutes} 分 {seconds} 秒"
    if minutes > 0 and seconds == 0:
        return f"{minutes} 分"
    if minutes == 0 and seconds > 0:
        return f"{seconds} 秒"
    return "1 秒"


def pil_to_bgr(image_pil):
    image_pil = ImageOps.exif_transpose(image_pil)
    image_pil = image_pil.convert("RGB")
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr


def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


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

    # 如果偵測不到人，可以把 0.5 改成 0.3
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
# 共用偵測函式
# =========================
def detect_posture_on_bgr(img_bgr, draw_overlay=True, overlay_text=None):
    img_bgr = resize_frame(img_bgr, max_width=640)

    try:
        results = model(img_bgr, verbose=False, imgsz=640)
        posture = classify_posture(results)
        annotated = results[0].plot()

    except Exception as e:
        posture = "偵測錯誤"
        annotated = img_bgr.copy()

        cv2.putText(
            annotated,
            f"Detection error: {str(e)[:80]}",
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if draw_overlay:
        if overlay_text is None:
            overlay_text = f"Posture: {posture}"

        cv2.rectangle(
            annotated,
            (20, 20),
            (min(900, annotated.shape[1] - 20), 70),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            annotated,
            overlay_text,
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated, posture


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
        unsafe_allow_html=True,
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

camera_choice = st.sidebar.radio(
    "即時鏡頭選擇",
    ["前鏡頭", "後鏡頭"],
    index=1,
)

facing_mode = "user" if camera_choice == "前鏡頭" else "environment"

st.sidebar.subheader("⏰ 警報時間設定")

with st.sidebar.form("alarm_time_form"):
    new_alarm_minutes = st.number_input(
        "分鐘",
        min_value=0,
        max_value=120,
        value=st.session_state.alarm_minutes,
        step=1,
    )

    new_alarm_seconds = st.number_input(
        "秒",
        min_value=0,
        max_value=59,
        value=st.session_state.alarm_seconds,
        step=1,
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
        new_alarm_seconds,
    )

alarm_threshold = st.session_state.alarm_threshold
alarm_time_text = st.session_state.alarm_time_text

st.sidebar.info(f"目前設定：{alarm_time_text} 後觸發警報")

st.sidebar.markdown("---")

if st.sidebar.button("🔊 啟用警報聲"):
    st.session_state.sound_enabled = True
    st.sidebar.success("警報聲已啟用")

if st.sidebar.button("🔔 測試警報聲"):
    st.session_state.sound_enabled = True
    st.session_state.test_alarm_sound = True

st.sidebar.markdown("---")

if st.sidebar.button("▶️ Start"):
    with shared_state.lock:
        shared_state.monitoring = True
        shared_state.start_time = time.time()
        shared_state.duration = 0.0
        shared_state.alarm = False
        shared_state.alarm_acknowledged = False
        shared_state.last_posture = shared_state.current_posture

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
    "即時鏡頭需要按畫面中的 START 才會開啟相機；側邊欄 Start 是開始監測計時。"
)


# 每秒刷新右側資訊
st_autorefresh(interval=1000, key="refresh")


# =========================
# 即時影像更新狀態
# =========================
def update_live_state(current_posture, threshold):
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
                shared_state.duration >= threshold
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


def draw_live_overlay(annotated):
    with shared_state.lock:
        monitor_text = "Monitoring" if shared_state.monitoring else "Stopped"

        posture_map = {
            "無人躺著": "No person",
            "左側躺": "Left side",
            "右側躺": "Right side",
            "仰躺": "Supine",
            "偵測錯誤": "Error",
        }

        posture_en = posture_map.get(shared_state.current_posture, "Unknown")

        info_text = (
            f"{monitor_text} | "
            f"Posture: {posture_en} | "
            f"Time: {int(shared_state.duration)} sec"
        )

        cv2.rectangle(
            annotated,
            (20, 20),
            (min(900, annotated.shape[1] - 20), 70),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            annotated,
            info_text,
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if shared_state.alarm:
            cv2.rectangle(
                annotated,
                (0, 0),
                (annotated.shape[1], annotated.shape[0]),
                (0, 0, 255),
                10,
            )

            cv2.putText(
                annotated,
                "ALARM",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                4,
                cv2.LINE_AA,
            )

    return annotated


# =========================
# WebRTC Video Processor
# =========================
class PoseVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = resize_frame(img, max_width=480)

        now = time.time()

        with shared_state.lock:
            should_detect = (now - shared_state.last_detect_time) >= 1.0

        if should_detect:
            annotated, current_posture = detect_posture_on_bgr(
                img,
                draw_overlay=False,
            )

            update_live_state(current_posture, alarm_threshold)
            annotated = draw_live_overlay(annotated)

            with shared_state.lock:
                shared_state.last_detect_time = now
                shared_state.last_annotated = annotated.copy()

        else:
            with shared_state.lock:
                if shared_state.last_annotated is not None:
                    annotated = shared_state.last_annotated.copy()
                else:
                    annotated = img.copy()

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24",
        )


# =========================
# Layout
# =========================
left_col, right_col = st.columns([1.2, 1.35])


# =========================
# Left Panel
# =========================
with left_col:
    st.subheader("1. 影像來源")

    tab_live, tab_camera, tab_image, tab_video = st.tabs(
        ["📷 即時鏡頭", "📸 拍照偵測", "🖼️ 上傳圖片", "🎞️ 上傳影片"]
    )

    # =========================
    # Live camera
    # =========================
    with tab_live:
        st.info(
            f"目前鏡頭設定：{camera_choice}。請先按下方 WebRTC 元件的 START，再按側邊欄 Start 開始監測。"
        )

        webrtc_streamer(
            key=f"pose-monitor-{facing_mode}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:stun3.l.google.com:19302"]},
                    {"urls": ["stun:stun4.l.google.com:19302"]},
                ]
            },
            media_stream_constraints={
                "video": {
                    "facingMode": {"ideal": facing_mode},
                    "width": {"ideal": 480, "max": 640},
                    "height": {"ideal": 360, "max": 480},
                    "frameRate": {"ideal": 5, "max": 8},
                },
                "audio": False,
            },
            video_processor_factory=PoseVideoProcessor,
            async_processing=True,
        )

    # =========================
    # Camera input fallback
    # =========================
    with tab_camera:
        st.markdown("### 📸 拍照偵測")
        st.caption("如果即時鏡頭跑不出來，可以用這個功能拍一張照片後分析。")

        camera_img = st.camera_input("請拍攝病床畫面")

        if camera_img is not None:
            image_pil = Image.open(camera_img)
            image_bgr = pil_to_bgr(image_pil)

            if st.button("分析拍照影像", type="primary"):
                with st.spinner("正在分析..."):
                    annotated_bgr, posture = detect_posture_on_bgr(
                        image_bgr,
                        draw_overlay=True,
                    )

                st.markdown(
                    f"""
                    <div class="result-box">
                        拍照偵測結果：{posture}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.image(
                    bgr_to_rgb(annotated_bgr),
                    caption=f"拍照偵測結果：{posture}",
                    use_container_width=True,
                )

    # =========================
    # Upload image
    # =========================
    with tab_image:
        st.markdown("### 🖼️ 上傳圖片偵測")
        st.caption("支援 JPG、JPEG、PNG。")

        uploaded_image = st.file_uploader(
            "請上傳病床圖片",
            type=["jpg", "jpeg", "png"],
            key="image_uploader",
        )

        if uploaded_image is not None:
            image_pil = Image.open(uploaded_image)
            image_bgr = pil_to_bgr(image_pil)

            st.image(
                bgr_to_rgb(resize_frame(image_bgr.copy(), max_width=640)),
                caption="原始圖片",
                use_container_width=True,
            )

            if st.button("開始分析圖片", type="primary"):
                with st.spinner("正在分析圖片..."):
                    annotated_bgr, posture = detect_posture_on_bgr(
                        image_bgr,
                        draw_overlay=True,
                    )

                st.markdown(
                    f"""
                    <div class="result-box">
                        圖片偵測結果：{posture}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.image(
                    bgr_to_rgb(annotated_bgr),
                    caption=f"圖片偵測結果：{posture}",
                    use_container_width=True,
                )

    # =========================
    # Upload video
    # =========================
    with tab_video:
        st.markdown("### 🎞️ 上傳影片偵測")
        st.caption("支援 MP4、MOV、AVI。系統會抽幀分析影片中的姿勢。")

        uploaded_video = st.file_uploader(
            "請上傳影片",
            type=["mp4", "mov", "avi"],
            key="video_uploader",
        )

        video_sample_interval = st.slider(
            "影片每隔幾幀分析一次",
            min_value=5,
            max_value=60,
            value=15,
            step=5,
        )

        max_analyze_frames = st.slider(
            "最多分析幾個抽樣畫面",
            min_value=10,
            max_value=200,
            value=60,
            step=10,
        )

        if uploaded_video is not None:
            video_bytes = uploaded_video.getvalue()
            st.video(video_bytes)

            if st.button("開始分析影片", type="primary"):
                suffix = Path(uploaded_video.name).suffix

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
                    temp_video.write(video_bytes)
                    temp_video_path = temp_video.name

                cap = cv2.VideoCapture(temp_video_path)

                if not cap.isOpened():
                    st.error("影片讀取失敗，請確認影片格式是否正確。")
                    Path(temp_video_path).unlink(missing_ok=True)

                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames <= 0:
                        total_frames = 1

                    progress = st.progress(0)
                    status_text = st.empty()

                    posture_list = []
                    sample_images = []

                    frame_idx = 0
                    analyzed_count = 0

                    while True:
                        ret, frame = cap.read()

                        if not ret:
                            break

                        if frame_idx % video_sample_interval == 0:
                            annotated_bgr, posture = detect_posture_on_bgr(
                                frame,
                                draw_overlay=True,
                                overlay_text=f"Frame {frame_idx}",
                            )

                            annotated_bgr, posture = detect_posture_on_bgr(
                                frame,
                                draw_overlay=True,
                                overlay_text=f"Frame {frame_idx} | Posture: {posture}",
                            )

                            posture_list.append(posture)

                            if len(sample_images) < 6:
                                sample_images.append(
                                    (frame_idx, posture, bgr_to_rgb(annotated_bgr))
                                )

                            analyzed_count += 1

                            status_text.write(
                                f"正在分析：第 {frame_idx} 幀，已分析 {analyzed_count} 張抽樣畫面"
                            )

                            if analyzed_count >= max_analyze_frames:
                                break

                        frame_idx += 1
                        progress.progress(min(frame_idx / total_frames, 1.0))

                    cap.release()
                    Path(temp_video_path).unlink(missing_ok=True)
                    progress.progress(1.0)

                    if len(posture_list) == 0:
                        st.warning("影片中沒有成功分析到畫面。")
                    else:
                        posture_counter = Counter(posture_list)
                        most_common_posture, most_common_count = posture_counter.most_common(1)[0]

                        st.markdown(
                            f"""
                            <div class="result-box">
                                影片主要姿勢：{most_common_posture}<br>
                                已分析抽樣畫面：{len(posture_list)} 張
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("#### 影片姿勢統計")

                        for posture_name in ["仰躺", "左側躺", "右側躺", "無人躺著", "偵測錯誤"]:
                            st.write(f"{posture_name}：{posture_counter.get(posture_name, 0)} 張")

                        st.markdown("#### 抽樣偵測畫面")

                        for idx, posture, img_rgb in sample_images:
                            st.image(
                                img_rgb,
                                caption=f"第 {idx} 幀：{posture}",
                                use_container_width=True,
                            )


# =========================
# Right Panel
# =========================
with right_col:
    st.subheader("2. 即時鏡頭摘要資訊")
    st.caption("此區只顯示即時鏡頭監測結果；圖片、拍照與影片結果會顯示在各自的分析區。")

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
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">目前姿勢</div>
                <div class="metric-value">{posture_now}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">持續時間</div>
                <div class="metric-value">{duration_now} 秒</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        system_text = "監測中" if monitoring_now else "停止"

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">系統狀態</div>
                <div class="metric-value">{system_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.test_alarm_sound:
        st.subheader("🔔 警報聲測試")
        render_loop_alarm()

        if st.button("停止測試警報聲"):
            st.session_state.test_alarm_sound = False
            st.rerun()

    st.subheader("3. 警報摘要")

    if alarm_now:
        st.markdown(
            f"""
            <div class="alert-box">
                🚨 偵測到姿勢持續超過 {alarm_time_text}，
                請協助翻身。
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_loop_alarm()

        if st.button("✅ 確認此資訊", type="primary"):
            with shared_state.lock:
                shared_state.alarm_acknowledged = True
                shared_state.alarm = False

            st.rerun()

    else:
        st.markdown(
            """
            <div class="normal-box">
                ✅ 目前尚未觸發警報
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.subheader("4. 使用說明")

    st.markdown(
        """
        1. **即時鏡頭**：請先按 WebRTC 畫面中的 START，允許相機，再按側邊欄 Start 開始監測。  
        2. **拍照偵測**：如果即時鏡頭跑不出來，請改用拍照偵測。  
        3. **上傳圖片**：可上傳病床圖片偵測仰躺、左側躺、右側躺。  
        4. **上傳影片**：可抽幀分析影片中的主要姿勢。  
        5. iPhone 上若警報聲沒有自動播放，請手動按「播放警報聲」。  
        """
    )
