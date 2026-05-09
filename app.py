import os
import cv2
import math
import time
import base64
import threading
from io import BytesIO

import numpy as np
import qrcode
from ultralytics import YOLO

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware


# =========================
# FastAPI app
# =========================
app = FastAPI(title="長照睡姿固定過久警報系統")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Load YOLO model
# =========================
model = YOLO("yolov8n-pose.pt")


# =========================
# Shared state
# =========================
class AppState:
    def __init__(self):
        self.lock = threading.Lock()

        self.current_posture = "尚未偵測"
        self.last_posture = "尚未偵測"

        self.start_time = time.time()
        self.duration = 0.0

        self.alarm = False
        self.alarm_acknowledged = False

        self.monitoring = False
        self.alarm_threshold = 10

        self.latest_image_base64 = ""
        self.last_update_time = ""


state = AppState()


# =========================
# Helper functions
# =========================
def get_base_url(request: Request):
    """
    Railway 通常會透過反向代理轉發，所以用 header 判斷公開網址。
    如果你有在 Railway 設定 PUBLIC_APP_URL，也會優先使用。
    """
    public_url = os.getenv("PUBLIC_APP_URL", "").strip()

    if public_url:
        return public_url.rstrip("/")

    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("host", "")

    return f"{proto}://{host}".rstrip("/")


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
# Process image
# =========================
def process_image_frame(img):
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

    with state.lock:
        if state.monitoring:
            if current_posture == state.last_posture:
                state.duration = now - state.start_time

            else:
                state.last_posture = current_posture
                state.current_posture = current_posture
                state.start_time = now
                state.duration = 0.0
                state.alarm = False
                state.alarm_acknowledged = False

            if (
                state.duration >= state.alarm_threshold
                and current_posture != "無人躺著"
                and current_posture != "偵測錯誤"
                and not state.alarm_acknowledged
            ):
                state.alarm = True

            else:
                if (
                    current_posture == "無人躺著"
                    or current_posture == "偵測錯誤"
                    or state.alarm_acknowledged
                ):
                    state.alarm = False

            state.current_posture = current_posture

        else:
            state.duration = 0.0
            state.alarm = False
            state.current_posture = current_posture

        monitor_text = "Monitoring" if state.monitoring else "Stopped"

        posture_map = {
            "無人躺著": "No person",
            "左側躺": "Left side",
            "右側躺": "Right side",
            "仰躺": "Supine",
            "偵測錯誤": "Error",
            "尚未偵測": "Not detected"
        }

        posture_en = posture_map.get(state.current_posture, "Unknown")

        info_text = (
            f"{monitor_text} | "
            f"Posture: {posture_en} | "
            f"Time: {int(state.duration)} sec"
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
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        if state.alarm:
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
                (30, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.7,
                (0, 0, 255),
                4,
                cv2.LINE_AA
            )

    return annotated


def image_to_base64(img):
    ok, buffer = cv2.imencode(".jpg", img)

    if not ok:
        return ""

    return base64.b64encode(buffer).decode("utf-8")


def generate_qr_png(url):
    qr = qrcode.QRCode(
        version=1,
        box_size=8,
        border=2
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer.getvalue()


# =========================
# Routes
# =========================
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <meta http-equiv="refresh" content="0; url=/dashboard">
        </head>
        <body>
            <p>Redirecting to dashboard...</p>
        </body>
    </html>
    """


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    base_url = get_base_url(request)
    camera_url = f"{base_url}/camera"

    return f"""
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
        <meta charset="UTF-8">
        <title>長照睡姿固定過久警報系統 Dashboard</title>
        <style>
            body {{
                margin: 0;
                font-family: Arial, "Microsoft JhengHei", sans-serif;
                background: #f5f7fb;
                color: #1f2937;
            }}

            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 24px;
            }}

            .title {{
                font-size: 32px;
                font-weight: 800;
                color: #1f3c88;
                margin-bottom: 6px;
            }}

            .subtitle {{
                color: #64748b;
                margin-bottom: 24px;
            }}

            .grid {{
                display: grid;
                grid-template-columns: 1.2fr 0.8fr;
                gap: 24px;
            }}

            .card {{
                background: white;
                border-radius: 18px;
                padding: 20px;
                box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
                border: 1px solid #e5e7eb;
            }}

            .video-box {{
                width: 100%;
                background: #111827;
                border-radius: 16px;
                min-height: 420px;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }}

            #latestImage {{
                max-width: 100%;
                width: 100%;
                border-radius: 16px;
            }}

            .metrics {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 12px;
                margin-bottom: 18px;
            }}

            .metric {{
                background: #f8fbff;
                border: 1px solid #dfe8f3;
                border-radius: 14px;
                padding: 16px;
                text-align: center;
            }}

            .metric-label {{
                color: #64748b;
                font-size: 14px;
                margin-bottom: 8px;
            }}

            .metric-value {{
                color: #1f3c88;
                font-size: 24px;
                font-weight: 800;
            }}

            button {{
                border: none;
                border-radius: 12px;
                padding: 12px 18px;
                font-size: 16px;
                font-weight: 700;
                cursor: pointer;
                margin: 4px;
            }}

            .start {{
                background: #2563eb;
                color: white;
            }}

            .stop {{
                background: #475569;
                color: white;
            }}

            .ack {{
                background: #16a34a;
                color: white;
            }}

            .alert {{
                background: #fff1f2;
                border: 1px solid #fda4af;
                color: #b91c1c;
                border-radius: 12px;
                padding: 16px;
                font-size: 18px;
                font-weight: 700;
                margin-top: 14px;
            }}

            .normal {{
                background: #f0fdf4;
                border: 1px solid #86efac;
                color: #166534;
                border-radius: 12px;
                padding: 16px;
                font-size: 18px;
                font-weight: 700;
                margin-top: 14px;
            }}

            .qr {{
                width: 220px;
                border: 1px solid #e5e7eb;
                border-radius: 14px;
                padding: 8px;
                background: white;
            }}

            input {{
                padding: 10px;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
                font-size: 16px;
                width: 80px;
            }}

            .hint {{
                color: #64748b;
                font-size: 14px;
                line-height: 1.6;
            }}

            @media (max-width: 900px) {{
                .grid {{
                    grid-template-columns: 1fr;
                }}

                .metrics {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>

    <body>
        <div class="container">
            <div class="title">🛌 長照睡姿固定過久警報系統</div>
            <div class="subtitle">電腦端 Dashboard：顯示 iPhone 傳回的畫面與姿勢偵測結果</div>

            <div class="grid">
                <div class="card">
                    <h2>1. 即時影像</h2>
                    <div class="video-box">
                        <img id="latestImage" src="" alt="尚未收到手機影像">
                    </div>
                    <p class="hint">請用 iPhone 掃描右側 QR Code，進入攝影機頁面後按「開始傳送」。</p>
                </div>

                <div class="card">
                    <h2>2. 手機攝影機連線</h2>
                    <img class="qr" src="/qr.png" alt="QR Code">
                    <p class="hint">
                        掃描後會開啟：<br>
                        <b>{camera_url}</b>
                    </p>

                    <hr>

                    <h2>3. 控制面板</h2>

                    <label>警報秒數：</label>
                    <input id="thresholdInput" type="number" min="3" max="120" value="10">
                    <button onclick="updateThreshold()" class="start">更新秒數</button>

                    <br><br>

                    <button onclick="startMonitoring()" class="start">▶️ Start</button>
                    <button onclick="stopMonitoring()" class="stop">⏹ Stop</button>
                    <button onclick="ackAlarm()" class="ack">✅ 確認警報</button>

                    <hr>

                    <h2>4. 摘要資訊</h2>

                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">目前姿勢</div>
                            <div id="posture" class="metric-value">尚未偵測</div>
                        </div>

                        <div class="metric">
                            <div class="metric-label">持續時間</div>
                            <div id="duration" class="metric-value">0 秒</div>
                        </div>

                        <div class="metric">
                            <div class="metric-label">系統狀態</div>
                            <div id="monitoring" class="metric-value">停止</div>
                        </div>
                    </div>

                    <div id="alarmBox" class="normal">✅ 目前尚未觸發警報</div>

                    <audio id="alarmAudio" src="/alarm.mp3" loop></audio>
                </div>
            </div>
        </div>

        <script>
            async function fetchState() {{
                try {{
                    const res = await fetch("/state");
                    const data = await res.json();

                    document.getElementById("posture").innerText = data.current_posture;
                    document.getElementById("duration").innerText = data.duration + " 秒";
                    document.getElementById("monitoring").innerText = data.monitoring ? "監測中" : "停止";
                    document.getElementById("thresholdInput").value = data.alarm_threshold;

                    if (data.latest_image_base64) {{
                        document.getElementById("latestImage").src =
                            "data:image/jpeg;base64," + data.latest_image_base64;
                    }}

                    const alarmBox = document.getElementById("alarmBox");
                    const alarmAudio = document.getElementById("alarmAudio");

                    if (data.alarm) {{
                        alarmBox.className = "alert";
                        alarmBox.innerText = "🚨 偵測到同一姿勢維持過久，請協助翻身";

                        alarmAudio.play().catch(() => {{
                            console.log("Autoplay blocked.");
                        }});
                    }} else {{
                        alarmBox.className = "normal";
                        alarmBox.innerText = "✅ 目前尚未觸發警報";
                        alarmAudio.pause();
                        alarmAudio.currentTime = 0;
                    }}

                }} catch (err) {{
                    console.log(err);
                }}
            }}

            async function startMonitoring() {{
                await fetch("/control", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{ action: "start" }})
                }});
                fetchState();
            }}

            async function stopMonitoring() {{
                await fetch("/control", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{ action: "stop" }})
                }});
                fetchState();
            }}

            async function ackAlarm() {{
                await fetch("/control", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{ action: "ack" }})
                }});
                fetchState();
            }}

            async function updateThreshold() {{
                const threshold = parseInt(document.getElementById("thresholdInput").value);

                await fetch("/control", {{
                    method: "POST",
                    headers: {{ "Content-Type": "application/json" }},
                    body: JSON.stringify({{
                        action: "threshold",
                        alarm_threshold: threshold
                    }})
                }});

                fetchState();
            }}

            setInterval(fetchState, 1000);
            fetchState();
        </script>
    </body>
    </html>
    """


@app.get("/camera", response_class=HTMLResponse)
async def camera_page():
    return """
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
        <meta charset="UTF-8">
        <title>手機攝影機端</title>
        <style>
            body {
                margin: 0;
                font-family: Arial, "Microsoft JhengHei", sans-serif;
                background: #0f172a;
                color: white;
                text-align: center;
            }

            .container {
                padding: 20px;
            }

            h1 {
                font-size: 28px;
                margin-bottom: 8px;
            }

            p {
                color: #cbd5e1;
                line-height: 1.6;
            }

            video {
                width: 100%;
                max-width: 520px;
                border-radius: 18px;
                background: black;
                margin-top: 16px;
            }

            button {
                border: none;
                border-radius: 14px;
                padding: 14px 22px;
                font-size: 18px;
                font-weight: 800;
                cursor: pointer;
                margin: 8px;
            }

            .start {
                background: #22c55e;
                color: white;
            }

            .stop {
                background: #ef4444;
                color: white;
            }

            .status {
                margin-top: 16px;
                padding: 14px;
                border-radius: 12px;
                background: #1e293b;
                color: #e2e8f0;
            }
        </style>
    </head>

    <body>
        <div class="container">
            <h1>📱 手機攝影機端</h1>
            <p>
                請允許使用相機，並將手機後鏡頭對準病床或模擬畫面。<br>
                這個頁面會定時把影像傳回電腦 Dashboard。
            </p>

            <video id="video" autoplay playsinline muted></video>

            <br>

            <button class="start" onclick="startCamera()">▶️ 開始傳送</button>
            <button class="stop" onclick="stopCamera()">⏹ 停止傳送</button>

            <div id="status" class="status">尚未開始</div>

            <canvas id="canvas" style="display:none;"></canvas>
        </div>

        <script>
            let video = document.getElementById("video");
            let canvas = document.getElementById("canvas");
            let statusBox = document.getElementById("status");

            let stream = null;
            let sending = false;
            let timer = null;

            async function startCamera() {
                try {
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: {
                            facingMode: { ideal: "environment" },
                            width: { ideal: 640, max: 960 },
                            height: { ideal: 480, max: 720 },
                            frameRate: { ideal: 10, max: 15 }
                        },
                        audio: false
                    });

                    video.srcObject = stream;
                    sending = true;

                    statusBox.innerText = "✅ 已開啟相機，正在傳送影像到 Dashboard";

                    timer = setInterval(captureAndSend, 800);

                } catch (err) {
                    statusBox.innerText = "❌ 無法開啟相機：" + err;
                }
            }

            function stopCamera() {
                sending = false;

                if (timer) {
                    clearInterval(timer);
                    timer = null;
                }

                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                }

                video.srcObject = null;
                statusBox.innerText = "已停止傳送";
            }

            async function captureAndSend() {
                if (!sending || !video.videoWidth) {
                    return;
                }

                const maxWidth = 640;
                const scale = maxWidth / video.videoWidth;
                const width = maxWidth;
                const height = Math.round(video.videoHeight * scale);

                canvas.width = width;
                canvas.height = height;

                const ctx = canvas.getContext("2d");
                ctx.drawImage(video, 0, 0, width, height);

                canvas.toBlob(async function(blob) {
                    if (!blob) return;

                    const formData = new FormData();
                    formData.append("file", blob, "frame.jpg");

                    try {
                        const res = await fetch("/upload", {
                            method: "POST",
                            body: formData
                        });

                        if (res.ok) {
                            statusBox.innerText = "✅ 影像傳送中：" + new Date().toLocaleTimeString();
                        } else {
                            statusBox.innerText = "⚠️ 傳送失敗";
                        }

                    } catch (err) {
                        statusBox.innerText = "❌ 傳送錯誤：" + err;
                    }

                }, "image/jpeg", 0.7);
            }
        </script>
    </body>
    </html>
    """


@app.get("/qr.png")
async def qr_png(request: Request):
    base_url = get_base_url(request)
    camera_url = f"{base_url}/camera"

    png = generate_qr_png(camera_url)

    return Response(content=png, media_type="image/png")


@app.post("/upload")
async def upload_frame(file: UploadFile = File(...)):
    image_bytes = await file.read()

    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(
            {"ok": False, "message": "無法讀取影像"},
            status_code=400
        )

    annotated = process_image_frame(img)
    img_b64 = image_to_base64(annotated)

    with state.lock:
        state.latest_image_base64 = img_b64
        state.last_update_time = time.strftime("%Y-%m-%d %H:%M:%S")

    return {"ok": True}


@app.get("/state")
async def get_state():
    with state.lock:
        return {
            "current_posture": state.current_posture,
            "last_posture": state.last_posture,
            "duration": int(state.duration),
            "alarm": state.alarm,
            "alarm_acknowledged": state.alarm_acknowledged,
            "monitoring": state.monitoring,
            "alarm_threshold": state.alarm_threshold,
            "latest_image_base64": state.latest_image_base64,
            "last_update_time": state.last_update_time,
        }


@app.post("/control")
async def control(payload: dict):
    action = payload.get("action", "")

    with state.lock:
        if action == "start":
            state.monitoring = True
            state.start_time = time.time()
            state.duration = 0.0
            state.alarm = False
            state.alarm_acknowledged = False
            state.last_posture = state.current_posture

        elif action == "stop":
            state.monitoring = False
            state.duration = 0.0
            state.alarm = False
            state.alarm_acknowledged = False
            state.current_posture = "尚未偵測"
            state.last_posture = "尚未偵測"

        elif action == "ack":
            state.alarm_acknowledged = True
            state.alarm = False

        elif action == "threshold":
            threshold = int(payload.get("alarm_threshold", 10))
            threshold = max(3, min(120, threshold))
            state.alarm_threshold = threshold

    return {"ok": True}


@app.get("/alarm.mp3")
async def alarm_mp3():
    path = "alarm.mp3"

    if os.path.exists(path):
        return FileResponse(path, media_type="audio/mpeg")

    return Response(status_code=404)
