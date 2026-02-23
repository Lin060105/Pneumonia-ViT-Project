# 1. 使用官方輕量級的 Python 3.10 環境作為基底
FROM python:3.10-slim

# 2. 設定容器內的工作目錄
WORKDIR /app

# 3. 安裝 OpenCV 影像處理底層所需的系統依賴套件
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. 先複製 requirements.txt (利用 Docker 快取機制加速未來構建)
COPY requirements.txt .

# 5. 安裝專案所需的所有 Python 套件
RUN pip install --no-cache-dir -r requirements.txt

# 6. 把專案所有的程式碼與模型權重複製到容器內
COPY . .

# 7. 宣告 Streamlit 預設使用的通訊埠
EXPOSE 8501

# 8. 容器啟動時要執行的終極指令 (啟動我們的醫療 App)
CMD ["streamlit", "run", "app_binary.py", "--server.port=8501", "--server.address=0.0.0.0"]