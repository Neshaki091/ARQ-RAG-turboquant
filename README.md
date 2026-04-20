# ARQ-RAG: Adaptive Residual Quantization for Efficient RAG Benchmarking

[![Project Status: Active](https://img.shields.io/badge/Project%20Status-Active-brightgreen.svg)](https://github.com/Neshaki091/ARQ-RAG-turboquant)
[![Framework: Next.js](https://img.shields.io/badge/Frontend-Next.js%2014-black)](https://nextjs.org/)
[![Backend: FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Vector DB: Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red)](https://qdrant.tech/)
[Link demo](https://arq-rag-turboquant.vercel.app/)

## 🌟 Tổng quan
Đồ án này tập trung vào việc **tìm kiếm giải pháp tối ưu hóa tăng cường truy xuất lượng tử trực tuyến (Online Quantization-Enhanced Retrieval Optimization)** trong hệ thống RAG. 

Mục tiêu cốt lõi là giải quyết bài toán cân bằng giữa **Tốc độ truy xuất (Speed)** và **Độ chính xác ngữ cảnh (Contextual Accuracy)**. Bằng cách triển khai kỹ thuật lượng tử hóa thặng dư thích ứng (ARQ) dựa trên công nghệ TurboQuant, dự án minh chứng một giải pháp cho phép tìm kiếm dữ liệu lớn với độ trễ cực thấp trong khi vẫn đảm bảo chất lượng phản hồi của mô hình ngôn ngữ lớn (LLM).

Dự án này phục vụ cho việc thực nghiệm và chứng minh ưu điểm của ARQ so với các phương pháp Baseline phổ biến như Product Quantization (PQ) hay Scalar Quantization (SQ8).

## 🚀 Các tính năng nổi bật
- **Hybrid Cloud Architecture**: Lưu trữ mã nén trên Qdrant Cloud nhưng tính toán tìm kiếm (Scoring) trực tiếp tại RAM Local Backend.
- **Native RAM Engine**: Công cụ tìm kiếm viết bằng NumPy tối ưu hóa cho 5 mô hình (Raw, Adaptive, PQ, SQ8, ARQ).
- **Automated Benchmark Loop**: Tự động chạy hàng trăm query test và đánh giá bằng **RAGAS** (Faithfulness, Relevancy, Precision).
- **Interactive Dashboard**: Theo dõi Latency và Memory Usage thời gian thực qua giao diện Next.js hiện đại.

## 🛠 Bộ 5 Mô hình So sánh
1.  **RAG-RAW**: Sử dụng vector Float32 nguyên bản (Upper Bound).
2.  **Adaptive-RAG**: Kỹ thuật Matryoshka Embeddings linh hoạt.
3.  **Manual-PQ**: Product Quantization (32 subspaces).
4.  **Manual-SQ8**: Scalar Quantization (8-bit integer).
5.  **ARQ-RAG (TurboQuant)**: Sản phẩm nghiên cứu chính, sử dụng cơ chế ADC (Asymmetric Distance Computation).

## 📦 Cài đặt nhanh

### Yêu cầu hệ thống
- Docker & Docker Compose
- Python 3.10+
- Node.js 18+

### Cấu hình biến môi trường (.env)
Tạo file `.env` tại dự án gốc với các thông tin:
```env
# Cloud Config
QDRANT_CLOUD_URL=your_qdrant_url
QDRANT_CLOUD_API_KEY=your_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_key

# LLM Config
GOOGLE_API_KEY=your_gemini_key_for_eval
GOOGLE_API_KEY_2=your_gemini_key_for_gen
```

### Chạy hệ thống
```bash
# 1. Khởi động Backend & Frontend
docker-compose up -d --build

# 2. Đồng bộ mã nén từ Cloud
python scripts/cloud/re_quantize.py
```

## 📖 Tài liệu chuyên sâu
Để hiểu rõ hơn về luồng dữ liệu và thiết kế hệ thống, vui lòng tham khảo:
- [Kiến trúc hệ thống (ARCHITECTURE.md)](./ARCHITECTURE.md)
- [Thuật toán TurboQuant (Research Paper Reference)](./docs/theory.md) (nếu có)

## 📊 Chạy Thực nghiệm Hệ thống (Super Benchmark)

Để tự động chạy toàn bộ tập câu hỏi mẫu (483 câu) qua cả 5 mô hình (Raw, Adaptive, PQ, SQ8, ARQ) và thu thập chỉ số RAM/Latency vào Database, hãy đảm bảo Backend đang chạy và thực hiện lệnh:

```powershell
python legacy/scripts/run_all_models.py
```

*Lưu ý: Script này sẽ tự động xoay tua API Key để tránh giới hạn Rate Limit và lưu kết quả trực tiếp vào bảng `benchmarks` trên Supabase.*

---
*Tài liệu hướng dẫn triển khai dự án ARQ-RAG (TurboQuant).*

# PHÁT TRIỂN BỞI 2 SINH VIÊN:
- Huỳnh Công Luyện 
- Nguyễn Đình Mạnh

# MỤC TIÊU:
- Tìm kiếm giải pháp tối ưu hóa tăng cường truy xuất lượng tử trực tuyến (Online Quantization-Enhanced Retrieval Optimization) trong hệ thống RAG.
