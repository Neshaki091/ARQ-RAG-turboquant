# Kế hoạch: Phân cấp vai trò Mixed-Model & Quản lý Rate Limit

Chúng ta sẽ thiết lập một "đội hình" AI tối ưu, tận dụng tối đa hạn mức của từng mô hình để hoàn thành bộ 1000 câu hỏi Benchmark.

## User Review Required

> [!IMPORTANT]
> **Cảnh báo về hạn mức RPD (Requests Per Day)**:
> - `gemini-3.1-flash-lite-preview` có hạn mức **500 RPD**. Do đó, bạn **KHÔNG THỂ** chạy xong 1000 câu benchmark trong 1 ngày (vì mỗi câu cần 1 lượt sinh trả lời).
> - **Giải pháp**: Chia benchmark thành 2-3 ngày, hoặc tôi sẽ thiết lập cơ chế "Dự phòng" sang mô hình khác khi hết lượt.
> - **Tốc độ**: Với 15 RPM, benchmark 1000 câu sẽ mất khoảng **1 tiếng 10 phút**.

## Phân vai chính thức (Model Matrix)

| Vai trò | Mô hình | Nền tảng | Hạn mức chính |
| :--- | :--- | :--- | :--- |
| **Giám khảo (Judge)** | `gemma-4-31b-it` | Google | 15 RPM \| 1500 RPD |
| **Sinh câu hỏi (Q-Gen)** | `gemma-4-31b-it` | Google | 15 RPM \| 1500 RPD |
| **Phân tích (Analyzer)** | `llama-3.1-8b-instant` | Groq | 30 RPM \| 14400 RPD |
| **Sinh câu trả lời** | `gemini-3.1-flash-lite-preview`| Google | 15 RPM \| 500 RPD |

## Proposed Changes

### 1. Shared Components
#### [MODIFY] [ragas_eval.py](file:///f:/IT%20project/DoAn/DEMO_ARQ_RAG/backend/shared/ragas_eval.py)
*   Đổi Judge về `gemma-4-31b-it`.

#### [MODIFY] [query_analyzer.py](file:///f:/IT%20project/DoAn/DEMO_ARQ_RAG/backend/shared/query_analyzer.py)
*   Xác nhận sử dụng `llama-3.1-8b-instant` (Đã xong).

### 2. Scripts & Benchmark
#### [MODIFY] [generate_benchmark_queries.py](file:///f:/IT%20project/DoAn/DEMO_ARQ_RAG/backend/scripts/generate_benchmark_queries.py)
*   Chuyển từ `ChatGroq` sang `ChatGoogleGenerativeAI`.
*   Sử dụng model `gemma-4-31b-it`.
*   Tăng `asyncio.sleep` lên **4.5 giây** (Để đảm bảo không vượt quá 15 RPM).

#### [MODIFY] [benchmark.py](file:///f:/IT%20project/DoAn/DEMO_ARQ_RAG/backend/benchmark.py)
*   Giảm `asyncio.sleep` từ 75s xuống **5.0 giây**. Vì Gemini có TPM cao (250K), chúng ta không cần chờ lâu cho token, nhưng cần chờ để tuân thủ 15 RPM.

## Open Questions

- Bạn có muốn tôi tích hợp cơ chế **Batching tự động** (mỗi ngày chạy một ít) hay sẽ tự tay nhấn Run từng đợt để quản lý hạn mức 500 RPD của Gemini?
- Bạn có muốn nới lỏng `MAX_CONTEXT_CHARS` cho Gemma 4 (Giám khảo) không? *Đề xuất*: Giữ 100k là đủ dùng.

## Verification Plan

- Chạy thử lệnh sinh câu hỏi: Xác nhận Gemma 4 hoạt động.
- Chạy Chat thử nghiệm: Xác nhận Gemini 3.1 Flash Lite phản hồi.
