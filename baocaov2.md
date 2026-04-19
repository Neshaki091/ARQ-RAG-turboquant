# 📑 BÁO CÁO THAY ĐỔI HỆ THỐNG ARQ-RAG (v2)
**Ngày**: 17/04/2026
**Trạng thái**: Hệ thống đã tối ưu hóa hoàn toàn cho Benchmark 500 câu.

---

## 🚀 1. Nâng cấp hạ tầng Mô hình (Mixed-Model Strategy)
Chúng ta đã chuyển đổi từ kiến trúc tập trung sang mô hình phối hợp đa nền tảng để triệt tiêu các lỗi 413 (Payload Too Large) và 429 (Rate Limit):

| Vai trò | Mô hình đảm nhiệm | Lý do thay đổi |
| :--- | :--- | :--- |
| **Generator (Câu trả lời)** | `gemini-3.1-flash-lite-preview` | Tận dụng Context Window 1M+ tokens, xử lý mượt mà chế độ EXTREME. |
| **Judge (Giám khảo Evaluation (DeepEval + TruLens))** | `gemma-4-31b-it` | Đảm bảo tính khắt khe, học thuật khi chấm điểm bài nghiên cứu. |
| **Q-Gen (Sinh câu hỏi)** | `gemma-4-31b-it` | Đạt chất lượng câu hỏi nghiên cứu cao nhất (Benchmark Ground Truth). |
| **Analyzer (Phân tích)** | `llama-3.1-8b-instant` | Phản hồi cực nhanh (<1s) để phân loại độ khó câu hỏi ngay lập tức. |

---

## 🛠️ 2. Tối ưu hóa Pipeline & Tốc độ
Để tuân thủ các hạn mức (Rate Limits) mới của Google/Groq, các thông số sau đã được cấu hình lại:

*   **Tốc độ Benchmark**: Thiết lập `asyncio.sleep(30)` giây giữa mỗi câu hỏi (Respect 15 RPM).
*   **Tốc độ Sinh câu hỏi**: Thiết lập `asyncio.sleep(10)` giây giữa mỗi câu hỏi.
*   **Mục tiêu Benchmark**: Điều chỉnh từ 1000 câu về **510 câu** (85 câu/chủ đề) để khớp với hạn mức **500 RPD** của Gemini.
*   **Context Window**: Nới lỏng hạn mức từ 32k lên **120,000 ký tự** (Cho phép đọc trọn vẹn 30 chunks chất lượng cao).

---

## 📂 3. Cập nhật Dữ liệu & Crawler
*   **Granular Chunking**: Giảm `chunk_size` từ 800 từ xuống **400 từ** để tăng độ chính xác của các "mảnh ghép" thông tin.
*   **Crawler Integrity**: Sửa logic `crawl_paper.py`. Hệ thống hiện tại **chỉ lưu Metadata vào Database khi và chỉ khi file PDF đã được tải và upload lên Storage thành công**.
*   **Timeout & Retry**: Tăng timeout tải PDF lên 60s và thêm cơ chế nghỉ 60s khi gặp lỗi 429 từ ArXiv.

---

## ⚠️ 4. Hướng dẫn vận hành (Next Steps)
Hệ thống hiện đã ở trạng thái ổn định nhất. Để bắt đầu đợt nghiên cứu mới, bạn cần:
1.  **Làm mới dữ liệu**: Nhấn `Purge Data` và chạy lại `Auto Pipeline` trên Dashboard để phân tách lại tài liệu theo chuẩn 400 từ mới.
2.  **Sinh bộ đề**: Chạy script `scripts/generate_benchmark_queries.py` để có bộ 510 câu hỏi Ground Truth chất lượng cao từ Gemma 4.
3.  **Benchmark**: Chạy thử nghiệm 3-5 câu để kiểm tra log, sau đó có thể chạy toàn bộ Batch 500 câu.

---
*Báo cáo được tạo tự động bởi Antigravity AI Assistant.*
