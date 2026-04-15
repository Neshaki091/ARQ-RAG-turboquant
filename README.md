# ARQ-RAG (TurboQuant) Research Pipeline

Hệ thống RAG (Retrieval-Augmented Generation) hiệu năng cao, tối ưu hóa cho nghiên cứu khoa học và luận văn tốt nghiệp. Dự án tập trung vào việc áp dụng các kỹ thuật nén Vector (Quantization) tiên tiến như **PQ**, **SQ8** và mô hình đề xuất **ARQ-RAG (TurboQuant)** với cơ chế bù sai số **QJL (Query-dependent Joint Lattice)**.

## 🚀 Tính năng trọng tâm

- **5 Chế độ RAG Độc lập**: 
    - `Standard RAG`: Truy vấn vector thô không nén.
    - `Adaptive RAG`: Tự động điều chỉnh độ rộng tìm kiếm theo độ phức tạp câu hỏi.
    - `PQ RAG`: Sử dụng nén Product Quantization.
    - `SQ8 RAG`: Sử dụng nén Scalar Quantization 8-bit.
    - `ARQ-RAG (TurboQuant)`: Mô hình đề xuất sử dụng ADC Reranking và QJL để khôi phục độ chính xác sau khi nén.
- **Hệ thống Crawler thông minh**: Tự động thu thập bài báo từ arXiv bằng Python script.
- **Dashboard Giám sát thời gian thực**: Theo dõi RAM, Latency và log hệ thống trực quan.
- **Pipeline Benchmark Tự động**: Tích hợp lọc ngữ cảnh và đánh giá **RAGAS Full Metrics** (qua Gemini 3.1 Flash).
- **Safe Mode cho API**: Cơ chế delay 90s và giới hạn Payload (6000 tokens) giúp hoạt động bền bỉ với các model có giới hạn TPM thấp.

## 🏗️ Kiến trúc Hệ thống

- **Backend**: Python (FastAPI) - Xử lý logic RAG, Quantization bằng NumPy, quản lý dữ liệu với Qdrant và Supabase.
- **Frontend**: Next.js (TypeScript/TailwindCSS) - Giao diện Dashboard phong cách hiện đại, hỗ trợ stream chat và render Markdown LaTeX.
- **Cơ sở dữ liệu**: 
    - **Qdrant**: Vector Database lưu trữ các bộ nén khác nhau.
    - **Supabase**: Lưu trữ metadata bài báo và tệp kết quả Benchmark.

## 🛠️ Hướng dẫn Cài đặt & Chạy

### 1. Yêu cầu hệ thống
- Docker & Docker Compose
- Groq API Key (Dùng cho LLM)
- Gemini API Key (Dùng cho RAGAS Evaluation)
- Supabase Project (URL & Key)

### 2. Cấu hình biến môi trường
Tạo tệp `.env` tại thư mục gốc:
```env
GROQ_API_KEY=your_key
GEMINI_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_SERVICE_ROLE_KEY=your_key
SECRET_KEY=demo123
```

### 3. Khởi chạy bằng Docker
```powershell
docker-compose up --build -d
```
Hệ thống sẽ chạy tại:
- Frontend: `http://localhost:3000`
- API Backend: `http://localhost:8000`

## 📊 Quy trình Nghiên cứu (Workflow)

1. **Cào dữ liệu (Crawl)**: Sử dụng nút "Run Crawl" để lấy dữ liệu từ arXiv vào Database.
2. **Xử lý dữ liệu (Auto Embedding)**: Hệ thống tự động Chunking, Embedding và đồng bộ hóa 5 loại nén vector vào Qdrant.
3. **Thử nghiệm (Chat)**: Chat trực tiếp với từng model để cảm nhận độ chính xác và tốc độ.
4. **Đánh giá (Benchmark)**: Chọn số lượng câu hỏi (20/30/40) và nhấn "Research". Hệ thống sẽ chạy RAGAS và xuất file Excel cumulative kết quả.

## 📄 Ghi chú kỹ thuật
Dự án triển khai các thuật toán nén Vector thủ công (Manual Quantization) bằng **NumPy** thay vì dùng thư viện có sẵn để đảm bảo tính minh bạch cho các báo cáo trong luận văn. Cơ chế **TurboQuant** sử dụng tích vô hướng trực tiếp (ADC) giúp đạt tốc độ vượt trội so với tìm kiếm vector thô.

---
*Phát triển bởi: Nhóm nghiên cứu ARQ-RAG*
