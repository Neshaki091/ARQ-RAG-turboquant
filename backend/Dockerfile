# 1. Sử dụng Python 3.12 làm base
FROM python:3.12-slim

# 2. Cài đặt các công cụ hệ thống cần thiết (C++, Rust, Git)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Cài đặt Rust (Cần thiết để build TurboQuant Native)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 4. Thiết lập thư mục làm việc
WORKDIR /app

# 5. Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy toàn bộ code Backend vào container
COPY . .

# 7. Cài đặt TurboQuant Engine (Biên dịch phần Rust)
RUN cd TQ_engine_lib && pip install -e .

# 8. Tạo các thư mục data cần thiết
RUN mkdir -p data/user_indexes data/uploads data/raw_system

# 9. Mở cổng 7860 (Cổng mặc định của HF Spaces)
EXPOSE 7860

# 10. Chạy ứng dụng bằng uvicorn
# Lưu ý: HF Spaces yêu cầu bind vào 0.0.0.0 và cổng 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
