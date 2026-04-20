# Hướng Dẫn Chi Tiết: Tái Tạo Dự Án ARQ-RAG (TurboQuant) Từ Đầu

Dự án **ARQ-RAG** là một hệ thống nghiên cứu chuyên sâu về các thuật toán Vector Quantization (Nén Vector) trong Retrieval-Augmented Generation (RAG). Dưới đây là cẩm nang phân tích kiến trúc và từng bước hướng dẫn giúp bạn tự xây dựng lại một hệ thống tương tự từ con số 0.

## 1. Chọn Lọc Công Nghệ (Tech Stack)

Để mô phỏng lại dự án này với độ chính xác cao và xử lý trơn tru, bạn cần chuẩn bị các công cụ sau:

### Backend & AI
*   **Framework Core:** `FastAPI` (tối ưu cho API bất đồng bộ và tốc độ cao).
*   **Vector Database:** `Qdrant` (hỗ trợ lưu Float32, PQ, Scalar Quantization).
*   **Database & Storage:** `Supabase` (Lưu lịch sử, caching và lưu file model/PDF).
*   **Local Embedding:** `Ollama` chạy mô hình `nomic-embed-text-v1.5` (chiều vector: 768).
*   **LLM Inference:** `Groq API` với mô hình `llama-3.3-70b-versatile` (tốc độ sinh văn bản cực nhanh) thông qua `LangChain`.
*   **Data Processing:** `numpy`, `pandas`, `PyMuPDF` (xử lý PDF), `faiss-cpu` (toán học cụm).

### Frontend
*   **Framework:** `Next.js 15` (App Router) + `React 19`.
*   **Styling:** `Tailwind CSS 4`, `lucide-react`, `framer-motion` (animations).
*   **Render Text/Math:** Tiện ích `react-markdown` kết hợp `rehype-katex` và `remark-math` (giúp hiển thị công thức toán học từ PDF mượt mà).

### Hạ tầng (DevOps)
*   **Quản lý Container:** `Docker & Docker Compose` (gom nhóm Backend, Qdrant, Ollama, Cloudflare Tunnel).

---

## 2. Tổ Chức Cấu Trúc Thư Mục (Modular Architecture)

Điểm cốt lõi của ARQ-RAG nằm ở tính **Modular Tuyệt Đối**. Bạn cần tạo dự án chứa `backend`, `frontend` và `docker-compose.yml`.

```text
ARQ_RAG_CLONE/
├── backend/
│   ├── models/             # Code lõi của 5 thuật toán
│   │   ├── arq_rag/        # Reranking = ADC + QJL (TurboQuant)
│   │   ├── rag_adaptive/   # Tư duy: tự chỉnh Top-K dựa theo độ khó câu hỏi
│   │   ├── rag_pq/         # Product Quantization (Nén cụm)
│   │   ├── rag_raw/        # Float32 (Baseline chuẩn không nén)
│   │   └── rag_sq8/        # Scalar Quantization (Nén 8-bit)
│   ├── shared/             # Code dùng chung (tránh lặp code)
│   │   ├── embed.py        # Giao tiếp với Ollama Nomic
│   │   ├── query_analyzer.py # Phân loại câu hỏi bằng LLM + Cache Supabase
│   │   ├── supabase_client.py # Driver cho Supabase
│   │   └── vector_store.py    # Driver truy suất Qdrant
│   ├── main.py             # Router FastAPI (Entry-point)
│   └── chat_service.py     # Component Điều phối truy vấn (Dispatching)
├── frontend/               # Next.js Code Dashboard
└── docker-compose.yml      # Cấu hình chứa container và biến môi trường
```

---

## 3. Các Bước Triển Khai Thực Tế

### Bước 1: Khởi tạo Hạ Tầng với Docker Compose
Lắp ráp các Service nền tảng trước để hệ thống liên kết với nhau:
1.  Khai báo **Qdrant**: Cấu hình volume gắn kết thư mục nội bộ máy Host ra ngoài để không mất Dữ liệu Index khi container sập/khởi động lại.
2.  Khai báo **Ollama**: Thiết lập hỗ trợ Card Màn Hình (`deploy.resources.reservations.devices`) và khi chạy có thể tự động tải Model `nomic-embed-text`.
3.  Khai báo **Backend**: Tham chiếu các file môi trường (SUPABASE, QDRANT, GOOGLE_API_KEY, GROQ_API_KEY).

### Bước 2: Thiết lập Tầng Shared (Dùng chung)
*   Tạo lớp tương tác cho Qdrant (quản lý 5 Collections tên khác nhau tương ứng biến thể: `vector_raw`, `vector_pq`, v.v.).
*   Tạo lớp `SupabaseManager` kết nối đến PostgreSQL/Storage để tải Model Weights (`centroids.pkl`) thay vì lưu file nặng trong Github.
*   Viết Class tính toán đoạn văn bản nhúng với độ dài Chunk có logic overlapping (MAX_CHUNK_CHARS: 1000).

### Bước 3: Thuật Toán Lõi - ADC Reranking (Trái tim của TurboQuant)
Phương pháp tính tích vô hướng tự tinh chỉnh - *Asymmetric Distance Computation*:
1.  **Chuyển câu hỏi sang Vector:** Query từ người dùng phải duy trì độ chính xác cao ở mức **Float32 Khôn nén**.
2.  **Lọc Ứng Viên Thô:** Gọi API tới Qdrant lấy giới hạn (vd: limit=40 bản ghi bị nén).
3.  **Thuật toán Reranking (Phân Hạng Lại):** Tại `backend/models/arq_rag/handler.py`, sử dụng Code `numpy` nhân vô hướng giữa Mảng Câu hỏi với Vector Nén. Bù trừ độ sai lệch qua mảng `qjl_batch` và `gamma_batch` từ công thức TurboQuant.
4.  Cắt xén để lấy `top_k` (vd: Top 10 tốt nhất) để gửi cho Bộ sinh văn bản (Generation LLM).

### Bước 4: Tạo "Bộ Não" Điều Phối (Query Analyzer)
Để hệ thống có tính "Thích Ứng" (Adaptive), bạn cần cơ chế Phân tích:
*   Người yêu cầu đặt câu hỏi -> LLM kiểm tra đánh giá: *"Đơn Giản hay Phức Tạp?"*.
*   Dựa trên phản hồi: 
    *   **Đơn giản (SIMPLE):** Tìm kiếm ít lại (Limit: 20, Top_K: 5)
    *   **Phức tạp (COMPLEX):** Đòi hỏi mở rộng ngữ cảnh (Limit: 80, Top_K: 20 hoặc 30).
*   **Mẹo:** Để tăng tốc độ suy luận, cần tính năng Cache các câu hỏi đã đánh giá vào Supabase nhằm bỏ qua LLM cho những phản hồi tương đồng về sau.

### Bước 5: Viết Data Ingestion Pipeline (Thu thập và Lập Chỉ Mục Dữ liệu)
*   Xây dựng mô-đun Crawler lấy hàng ngàn tài liệu PDF khoa học (như trên arXiv).
*   Dùng `PyMuPDF` giải mã file, làm sạch bảng biểu/công thức.
*   Phân cắt văn bản dựa trên thuật toán (Chunking). Lồng văn bản với Model `nomic-embed-text` thành chuỗi Vector.
*   Lấy Vector, thực hiện thuật toán nén Vector (Quantizer) đối với các bảng cần nén và nhồi lên 5 Collection tương ứng trong Qdrant.

### Bước 6: Xây Dựng Frontend (Research Dashboard)
Giao diện không chỉ để Chat, mà để đánh giá và so sánh thực chứng các Model. Do đó Dashboad phải có:
*   Trình Quản Lý Model (Dropdown Menu chuyển RAW, ARQ, ADAPTIVE...).
*   Thông kê hiển thị tốc độ `latency` mili-giây, thông số Retrieval, Reranking tách bạch.
*   Trường Chat hỗ trợ Markdown, LaTeX và có dạng trả lại Real-time Streaming từ Backend Server.

---

## 4. Những Điểm Tối Ưu Tốc Độ Quan Trọng (Best Practices)

1.  **Dùng Numpy Tối Đa:** Tốc độ ADC Reranking cực kì nhạy cảm. Không dùng vòng lặp `for` thủ công trong Python. Phải tận dụng `np.dot()` hoặc các lệnh thao tác mảng của thư viện C/C++ ngầm bên dưới Numpy.
2.  **Context Cắt Tỉa (Pruning Context):** Trong Model Generator LLM, giới hạn tối đa `MAX_CONTEXT_CHARS` cho chuỗi ngữ cảnh. Nếu vượt quá (ví dụ 120.000 ký tự), thì phải chủ động bẫy logic `[:120000]` và đánh dấu báo cắt tỉa cho LLM.
3.  **Prompt Instruction Quyết Đoán:** Vì mô phỏng học thuật nên Prompt cho AI cần dứt khoát: *"Chỉ trả lời trọng tâm, không chào hỏi. Sử dụng LaTeX cho công thức toán."* và yêu cầu luôn được prefix với `[ARQ-RAG]`.

Quy trình này sẽ thiết lập nền móng vững chắc giúp bạn tùy biến, làm chủ thuật toán Vector Compression trong công nghệ LLM RAG - Cân bằng tuyệt đối giữa Accuracy và Latency.
