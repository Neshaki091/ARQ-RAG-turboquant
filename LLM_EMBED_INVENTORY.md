# ARQ-RAG Model Inventory

Tài liệu này tổng hợp toàn bộ các mô hình Ngôn ngữ lớn (LLM) và mô hình Nhúng (Embedding) đang được sử dụng trong hệ thống ARQ-RAG cho mục đích nghiên cứu và thực nghiệm.

## 1. Hệ thống LLM (Large Language Models)

| Chức năng | Mô hình | Nền tảng | Vai trò |
| :--- | :--- | :--- | :--- |
| **Hội thoại (Chat)** | `gemma-4-26b-it` | Google Cloud API | Sinh câu trả lời cuối cùng cho người dùng dựa trên ngữ cảnh đã nén. |
| **Sinh bộ đề (Gen Query)** | `gemini-3.1-flash-lite-preview` | Google Cloud API | Trích xuất tri thức từ PDF để tạo câu hỏi và Ground Truth (Batch 5 q/req). |
| **Phân tích (Analyzer)** | `llama-3.1-8b-instant` | Groq Engine | Phân tích độ phức tạp của câu hỏi để điều chỉnh tham số nén Adaptive. |

## 2. Hệ thống Embedding (Vectorization)

| Chức năng | Mô hình | Triển khai | Ghi chú |
| :--- | :--- | :--- | :--- |
| **Nhúng dữ liệu (Local)** | `nomic-embed-text` | Ollama | Sử dụng trong quá trình Ingest PDF và tính toán tương đồng tại máy local. |
| **Nhúng dữ liệu (Cloud)** | `nomic-embed-text-v1.5` | SentenceTransformer | Tích hợp trực tiếp vào container Hugging Face để đảm bảo tính độc lập (Inference-only). |

## 3. Quản lý API Keys
- **GOOGLE_API_KEY**: Sử dụng cho toàn bộ dòng mô hình Gemma.
- **GROQ_API_KEY**: Sử dụng cho mô hình Llama (Analyzer).
- **SUPABASE**: Quản lý Cache và lưu trữ kết quả Benchmark.

---
*Cập nhật lần cuối: 19/04/2026*
