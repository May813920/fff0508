FROM python:3.11-slim

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# 先升級 pip
RUN pip install --no-cache-dir --upgrade pip

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway 會自動提供 PORT，不要固定只用 8501
EXPOSE 8080

CMD streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8080} --server.headless=true
