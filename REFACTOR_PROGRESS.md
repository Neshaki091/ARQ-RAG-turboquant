# 🛠️ REFACTOR PROGRESS: ARQ-RAG MODULARIZATION

**Ngày cập nhật**: 2026-04-14
**Trạng thái**: Tạm ngưng (Pause) - Sẵn sàng resume.

## 📌 Tổng quan bối cảnh
Dự án đã được chuyển đổi từ cấu trúc tập trung sang cấu trúc **Modular tuyệt đối**.
- **Shared**: Chứa các Driver (Supabase, Qdrant, Embedding).
- **Models**: Chứa 5 thư mục mô hình độc lập. Mỗi mô hình có `quantization.py` (Toán), `builder.py` (Index), và `handler.py` (Chat).

## ✅ Các hạng mục đã hoàn thành
- [x] Triển khai cấu trúc thư mục mới.
- [x] Di chuyển logic Quantization về từng Model.
- [x] Viết lại `VectorStoreManager` để hỗ trợ nạp dữ liệu động (Modular).
- [x] Cập nhật `ChatService` để điều phối thông qua các `ModelHandler`.
- [x] Nâng cấp `BenchmarkManager` hỗ trợ 1000 câu hỏi và lưu Excel tích lũy.

## ⚠️ Lint Errors tồn đọng (Cần fix khi Resume)
| File | Vấn đề | Độ ưu tiên |
| :--- | :--- | :--- |
| `models/*/handler.py` | Import "numpy" & "langchain" không phân giải được (Thiếu env) | Cao |
| `shared/embed.py` | Import "httpx" & "tqdm" không phân giải được | Trung bình |
| `benchmark.py` | Import "pandas" không phân giải được | Cao |

## 🚀 Hướng dẫn Resume (Dành cho AI)
Khi người dùng yêu cầu tiếp tục, hãy thực hiện các bước:
1. Đọc lại file `REFACTOR_PROGRESS.md` này để nắm list file.
2. Kiểm tra lại môi trường Python để fix các lỗi Import (Lint errors) trên.
3. Hoàn thiện file `backend/data/benchmark_queries.json` với 1000 câu hỏi thực tế.
4. Chạy xác thực cuối cùng bằng một lượt Benchmark (Batch 0-10).

---
*Ghi chú: Mọi thuật toán PQ, SQ8, ARQ hiện đã nằm đúng nơi quy định.*
