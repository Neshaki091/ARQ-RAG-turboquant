# 🚀 BÁO CÁO TIẾN ĐỘ DỰ ÁN ARQ-RAG (TURBOQUANT)

Tài liệu này tổng hợp các hạng mục đã hoàn thành và phân chia vai trò cho đội ngũ phát triển (2 Full Stack Developers) để tiếp tục vận hành và tối ưu hệ thống.

---

## 📅 Tổng quan các hạng mục đã hoàn thành (Progress Report)

Cho đến thời điểm hiện tại, dự án đã đi qua các cột mốc quan trọng sau:

### 1. Kiến trúc Modular hóa (Core Backend)
- **Cấu trúc lại toàn bộ Backend**: Chuyển đổi từ code tập trung sang mô hình **Modular tuyệt đối**. Mỗi mô hình (Raw, PQ, SQ8, Adaptive, ARQ) hiện có thư mục riêng với logic `quantization`, `builder` (lập chỉ mục) và `handler` (xử lý truy vấn) độc lập.
- **Shared Utilities**: Xây dựng bộ Driver dùng chung cho `Supabase`, `Qdrant` và `Embedding` giúp tối ưu hóa tái sử dụng code.

### 2. Thuật toán Nén và Tìm kiếm (Algorithms)
- **Triển khai thành công 5 biến thể RAG**:
    - **RAW**: Baseline chuẩn không nén.
    - **PQ (Product Quantization)**: Nén vector theo cụm.
    - **SQ8 (Scalar Quantization)**: Nén vector 8-bit.
    - **Adaptive**: Tự động điều chỉnh tham số theo độ khó câu hỏi.
    - **ARQ (TurboQuant)**: Thuật toán độc quyền sử dụng nén Residual-based (QJL) giúp sắp xếp lại (Reranking) cực nhanh.

### 3. Hệ thống Đánh giá Tự động (Benchmarking Pipeline)
- **Pipeline Benchmark quy mô lớn**: Hỗ trợ chạy thử nghiệm trên bộ dữ liệu 1000 câu hỏi.
- **Xuất báo cáo Excel**: Tự động lưu trữ kết quả và upload lên Supabase Storage để theo dõi lâu dài.

### 4. Giao diện Nghiên cứu (Research Dashboard)
- **Real-time Monitoring**: Theo dõi RAM, CPU và tiến độ chạy Pipeline ngay trên web.
- **Streaming Chat**: Giao diện hội thoại hỗ trợ Markdown, LaTeX và phản hồi theo thời gian thực (Zero Latency).
- **So sánh trực quan**: Cho phép người dùng chuyển đổi giữa 5 mô hình để thấy sự khác biệt về độ trễ và độ chính xác.

### 5. Tự động hóa Dữ liệu (Ingestion & Crawling)
- **arXiv Crawler**: Hệ thống tự động cào hàng ngàn bài báo từ arXiv dựa trên từ khóa nghiên cứu.
- **Supabase Integration**: Lưu trữ PDF và metadata bài báo một cách bền vững trên Cloud.

---

## 🤝 Phân chia Công việc (Team Collaboration)

Dự án được thực hiện bởi 2 thành viên Full Stack với các vai trò chuyên biệt:

### 1. Developer 1: Huỳnh Công Luyện
*   **Mô hình phụ trách**: **ARQ-RAG (TurboQuant)**, **RAG-SQ8 (Int8)**, **Adaptive-RAG**.
*   **Trọng tâm**: Nghiên cứu và triển khai các thuật toán nén nâng cao (Residual-based), cơ chế bù sai số QJL, và logic điều phối truy vấn linh hoạt (Query Analyzer).
*   **Hạ tầng**: Quản lý Reranking và Asymmetric Distance Computation (ADC).

### 2. Developer 2: Nguyễn Đình Mạnh
*   **Mô hình phụ trách**: **RAG-RAW (Baseline)**, **RAG-PQ (Product Quantization)**.
*   **Trọng tâm**: Xây dựng bộ khung Baseline chuẩn, triển khai thuật toán nén PQ và **Nghiên cứu lý thuyết thuật toán TurboQuant (Residual-based)**.
*   **Hạ tầng**: Quản lý lưu trữ Supabase và tích hợp Frontend Dashboard.

## 📏 Quy ước Nghiên cứu (Research Conventions)

Dự án tuân thủ nghiêm ngặt các quy chuẩn để đảm bảo tính khách quan của số liệu:
- **Kiến trúc ADC (Asymmetric Distance Computation)**: Chỉ nén Vector trong Database, giữ nguyên định dạng Floating Point cho Vector truy vấn (Query) để đảm bảo độ chính xác.
- **Quy trình Benchmark**: 
    - Chạy tuần tự từng model để tránh tranh chấp RAM và tài nguyên GPU/LLM.
- **Tính thống nhất**: 100% các model chạy trên cùng một Dataset, cùng Embedding Model và sử dụng cùng một bộ Query Set để so sánh công bằng.
- **TurboQuant Definition**: Kết hợp nén Scalar Quantization với kỹ thuật bù sai số QJL (Residual-based) để đạt hiệu năng tương đương vector gốc với dung lượng nén gấp nhiều lần.

## 📄 Ghi chú kỹ thuật
### 🧑‍💻 Developer 1: SQ8 & ARQ-RAG (TurboQuant)
*Tập trung nén sâu, tối ưu hóa bộ nhớ và thuật toán Reranking độc quyền.*

1.  **Phụ trách Model**:
    - **RAG-SQ8 (Scalar Quantization)**: Nén vector 8-bit để tiết kiệm RAM.
    - **ARQ-RAG (TurboQuant)**: Thuật toán nén Residual (QJL) kết hợp Reranking tốc độ cao.
2.  **Nhiệm vụ trọng tâm**:
    - Cải thiện tốc độ tính toán ADC (Asymmetric Distance Computation) cho ARQ.
    - Quản lý hiệu năng RAM và CPU của các tầng nén cao cấp.
    - Tối ưu hóa UI/UX cho Dashboard để hiển thị các chỉ số so sánh hiệu năng nén (Latency/Accuracy Tradeoff).
    - Quản lý hạ tầng (Docker) và tích hợp hệ thống (Supabase).

---

## 📏 Quy ước kỹ thuật và Thực nghiệm (Project Conventions)

Để đảm bảo tính nhất quán trong nghiên cứu, toàn bộ hệ thống tuân thủ các quy tắc sau:

### 1. Quy ước nén Vector (Quantization)
| Thành phần | RAG-RAW | Adaptive | RAG-PQ | RAG-SQ8 | ARQ-RAG |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vector DB** | Float32 | Float32 | PQ | Int8 | TurboQuant |
| **Collection** | `vector_raw` | `vector_raw` | `vector_pq` | `vector_sq8` | `vector_arq` |
| **Cơ chế** | Standard | Adaptive | ADC | ADC | ADC + QJL |

### 2. Nguyên tắc thực nghiệm công bằng (Golden Rules)
- **Cùng Dataset**: 100% các model chạy trên cùng một bộ tài liệu PDF.
- **Cùng Embedding**: Chỉ sử dụng duy nhất mô hình `nomic-embed-text` cho tất cả các vector DB.
- **Cùng Query Set**: Mọi lượt so sánh benchmark phải thực hiện trên cùng một bộ câu hỏi.
- **Chạy tuần tự (Sequential)**: Để đảm bảo đo RAM chính xác, các model benchmark được chạy lần lượt, giải phóng bộ nhớ sau mỗi lượt.

### 3. Phương thức tính toán ADC
Hệ thống tuân thủ nguyên tắc **Asymmetric Distance Computation**: 
- **Query**: Luôn giữ ở định dạng **Float32** (không nén).
- **Database**: Dữ liệu đã được nén (**Quantized**).
- Tính toán khoảng cách trực tiếp giữa Float32 và Quantized để tối ưu tốc độ mà không mất quá nhiều độ chính xác.

### 4. Quy ước Benchmark (Protocol)
- **Cấu trúc Test**: 
    - 10 queries/lượt (batch).
    - **Tổng cộng 500 queries duy nhất/model.**
    - Duy nhất 1 bộ Ground Truth chung cho cả 5 model (500 câu).
    - **Tổng cộng 50 lượt chạy/lượt batch (5 models x 10 câu).**
    - **Tổng cộng 2500 lượt đánh giá toàn hệ thống (50 batches).**
    - **Thời gian chờ (Safe Delay)**: 75 giây giữa các query.
- **Chỉ số hệ thống (System Metrics)**:
    - **RAM**: Đo bằng MB (Peak memory usage) qua `psutil`.
    - **Latency**: Đo bằng ms (End-to-end) từ lúc nhận query đến khi trả câu trả lời.
    - **Storage**: Kích thước tệp tin thực tế của Vector Collection.
    - Faithfulness (Tính trung thực).
    - Answer Relevance (Độ liên quan câu trả lời).
    - Context Precision & Recall (Độ chính xác và đầy đủ của ngữ cảnh).

---

## ⚙️ Cấu hình Hệ thống & Mô hình (Technical Configurations)

Dưới đây là bảng thông số chuẩn được áp dụng cho từng mô hình trong hệ thống:

### 1. Thông số chung (Shared Config)
*   **TARGET_DIMENSIONS**: 768 (Nomic Embed v1.5)
*   **SIMILARITY_THRESHOLD**: 20 (Keyword-based relevance score)
*   **MAX_CONTEXT_CHARS**: 24,000 ký tự (Giới hạn ngữ cảnh gửi đến LLM)
*   **API_DELAY**: 75 giây (TPM Safe Mode delay giữa các requests)
*   **BATCH_SIZE**: 10 câu hỏi/model (Tổng 50 runs mỗi đợt benchmark)

### 2. Cấu hình mô hình ngôn ngữ (LLM Judge & Chat)
*   **LLM Chính (Inference Engine)**: `llama-3.3-70b-versatile`
    *   *Nền tảng*: Groq Cloud API (Tối ưu hóa Latency).
    *   *Nhiệm vụ*: Answer Generation, Query Analysis.
*   **Embedding Model**: `nomic-embed-text-v1.5`
    *   *Nền tảng*: Ollama Local (Hỗ trợ context length lên tới 8192).
*   **Tham số Generation**: `temperature = 0` (Đảm bảo kết quả benchmark có tính lặp lại tốt).

### 3. Cấu hình Chia nhỏ văn bản (Chunking Config)
*   **MAX_CHUNK_CHARS**: 1.000 ký tự
*   **MIN_CHUNK_CHARS**: 100 ký tự
*   **OVERLAP_CHARS**: 200 ký tự
*   **Embedding Model**: `nomic-embed-text`

### 4. Thông số đặc thù từng Model (Model-specific Config)

| Model | TOP_K_CHUNKS | SEARCH_LIMIT | Quantization | Đặc điểm chính |
| :--- | :--- | :--- | :--- | :--- |
| **RAG-RAW** | 15 | 40 | Float32 | Baseline chuẩn |
| **RAG-Adaptive** | 5 - 30 (Linh hoạt) | 20 - 80 | Float32 | Tự động chỉnh Top-K |
| **RAG-PQ** | 15 | 40 | PQ (256 Centroids) | Nén vector theo cụm |
| **RAG-SQ8** | 15 | 40 | Int8 (8-bit) | Nén Scalar min-max |
| **ARQ-RAG** | 5 - 30 (Linh hoạt) | 20 - 80 | TurboQuant (~4-bit) | Bù sai số Residual |

### 4. Hằng số Kỹ thuật Nâng cao (Advanced Constants)
*   **Search Metric**: `Dot Product / Inner Product` (Tối ưu cho hướng của vector).
*   **Hệ thống Reranking**: Sử dụng **ADC (Asymmetric Distance Computation)**.
    *   Query Vector: `Float32` (giữ nguyên độ chính xác cao).
    *   Database Vector: `Quantized` (Nén để tiết kiệm hiệu năng).
*   **Phân bổ Crawler**:
    *   `TARGET_TOTAL`: 1,100 tài liệu gốc.
    *   `MAX_PER_QUERY`: 300 kết quả từ arXiv API.
    *   `CRAWL_DELAY`: 2.0 giây (Tránh bị chặn bởi server arXiv).
*   **Tên Collection chuẩn (Qdrant)**:
    *   `vector_raw`
    *   `vector_pq`
    *   `vector_sq8`
    *   `vector_arq`

---

## 📈 Roadmap Tiếp theo

1.  **Fixing Bug**: Xử lý triệt để lỗi `/purge-data` và ổn định luồng upload Supabase.
2.  **Dọn dẹp (Cleanup)**: Loại bỏ các mã nguồn rác và file test thừa (đã liệt kê ở phần trước).
3.  **Mở rộng bộ testset**: Hoàn thiện 1000 câu hỏi Ground Truth chất lượng cao.
4.  **Viết tài liệu kỹ thuật**: Chuẩn bị báo cáo đồ án dựa trên các kết quả đo đạc từ Dashboard.
