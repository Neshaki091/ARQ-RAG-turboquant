# Hướng dẫn Cài đặt TurboQuant Backend trên Linux

Tài liệu này hướng dẫn cách thiết lập môi trường và chạy Backend của dự án TurboQuant (ARQ-RAG) trên các hệ điều hành Linux (Ubuntu/Debian được khuyến nghị).

## 1. Yêu cầu hệ thống
- **OS:** Ubuntu 22.04 LTS hoặc mới hơn (hoặc các distro tương đương).
- **RAM:** Tối thiểu 4GB (Khuyến nghị 8GB+ để xử lý vector).
- **CPU:** Hỗ trợ tập lệnh SIMD (AVX2/AVX-512) để đạt hiệu suất tối đa.
- **Internet:** Cần thiết để tải thư viện và mô hình AI.

---

## 2. Cài đặt các thành phần hệ thống

Mở terminal và chạy các lệnh sau:

### Bước 2.1: Cập nhật hệ thống và cài đặt công cụ Build
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential curl git pkg-config libssl-dev python3-pip python3-venv
```

### Bước 2.2: Cài đặt Rust (Bắt buộc cho TurboQuant Native Core)
TurboQuant sử dụng Rust để xử lý tính toán tốc độ cao.
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Kích hoạt Rust cho session hiện tại
source $HOME/.cargo/env
```
Kiểm tra cài đặt: `cargo --version`

---

## 3. Thiết lập mã nguồn và Môi trường ảo

### Bước 3.1: Di chuyển vào thư mục backend
Giả sử bạn đã clone code về máy:
```bash
cd /path/to/DEMO_ARQ_RAG/backend
```

### Bước 3.2: Tạo và kích hoạt môi trường ảo (Virtual Environment)
```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 3.3: Cài đặt các thư viện Python
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*Lưu ý: Nếu bạn dùng GPU NVIDIA, hãy cài đặt PyTorch phiên bản hỗ trợ CUDA.*

---

## 4. Cấu hình Biến môi trường

Tạo file `.env` nếu chưa có (dựa trên file mẫu hoặc `.env` hiện tại):
```bash
cp .env.example .env  # Nếu có file example
# Hoặc tự tạo file .env và điền các API KEY (GROQ_API_KEY, v.v.)
nano .env
```

---

## 5. Chạy Backend

### Khởi chạy chế độ Development
Khi chạy lần đầu, hệ thống sẽ tự động biên dịch thư viện Rust thành file `.so`. Quá trình này mất khoảng 1-2 phút.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Kiểm tra trạng thái
Mở trình duyệt hoặc dùng `curl` truy cập: `http://localhost:8000/docs` để xem Swagger UI.

---

## 6. Lưu ý quan trọng cho Linux

1.  **Quyền truy cập thư mục:** Đảm bảo user có quyền ghi vào thư mục `data/` để lưu index:
    ```bash
    mkdir -p data/user_indexes data/uploads data/raw_system
    chmod -R 755 data
    ```
2.  **Swap Memory:** Nếu RAM máy ảo thấp (dưới 4GB), bạn nên tạo thêm Swap để tránh bị lỗi `Out of Memory` khi biên dịch Rust hoặc nạp model.
3.  **TurboQuant Native:** File `TQ_engine_lib/tq_native_lib.so` sẽ được tự động tạo ra. Đừng lo lắng nếu bạn thấy log thông báo "Compiling Rust SIMD core" ở lần chạy đầu tiên.

---

## 7. Sử dụng Docker (Cách nhanh nhất)
Nếu bạn đã cài Docker và Docker Compose:
```bash
docker build -t arq-rag-backend .
docker run -p 8000:7860 --env-file .env arq-rag-backend
```
