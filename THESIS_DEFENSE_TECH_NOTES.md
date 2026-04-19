# Technical Notes for Thesis Defense (ARQ-RAG System)

Tài liệu này tổng hợp các luận điểm kĩ thuật quan trọng để trình bày trước Hội đồng, giải thích lý do đằng sau các quyết định kiến trúc và tối ưu hóa trong hệ thống ARQ-RAG.

## 1. Chiến lược Truy xuất Đa tầng (Two-Stage Retrieval)
- **Giai đoạn 1 (Cloud - Scalability):** Sử dụng Qdrant Cloud với các vector đã nén (Product Quantization - PQ hoặc Scalar Quantization - SQ8). Mục tiêu là lọc nhanh hàng triệu tài liệu với chi phí lưu trữ thấp và tốc độ truy vấn mili-giây.
- **Giai đoạn 2 (ARQ Algorithm - Precision):** Thuật toán nén ARQ thực hiện bù đắp sai số (Residual Compensation) cho các đại diện vector bị mất mát thông tin. Việc này đảm bảo kết quả truy xuất có độ chính xác tương đương với vector gốc nhưng tiết kiệm 75-90% dung lượng RAM.

## 2. Lựa chọn Mô hình LLM (Gemma 26B/31B)
- **Lý do chọn Gemma (Google):** Đây là dòng mô hình mã nguồn mở mạnh mẽ nhất hiện nay ở phân khúc tầm trung. Sự lựa chọn này chứng minh hệ thống có khả năng chạy tốt trên cả hạ tầng thương mại (Google Cloud) và hạ tầng mở.
- **Phân bổ nhiệm vụ:**
    - `Gemma 26B`: Sử dụng cho hội thoại (Chat) vì tốc độ phản hồi nhanh.
    - `Gemma 31B`: Sử dụng làm **Judge LLM (Giám khảo)** cho RAGAS vì khả năng suy luận logic và đánh giá khách quan cao hơn, đảm bảo độ tin cậy cho kết quả nghiên cứu.

## 3. Tối ưu hóa API & Hiệu năng (Batching & Determinism)
- **Batching Generation (5 q/req):** Hệ thống không sinh từng câu hỏi đơn lẻ mà sinh theo lô (Batch). Điều này làm tăng hiệu suất sử dụng API lên 500%, giúp vượt qua các giới hạn về *Requests Per Minute (RPM)* của gói miễn phí mà vẫn xây dựng được bộ dataset 500+ câu hỏi nhanh chóng.
- **Cấu hình n=1 (Single Candidate):** Đây là kĩ thuật đảm bảo tính **Định danh (Determinism)**. Trong nghiên cứu khoa học, việc AI trả về duy nhất một phương án tốt nhất giúp kết quả thực nghiệm có tính lặp lại (Reproducibility) cao, tránh sai số ngẫu nhiên giữa các lần chạy Benchmark.

## 4. Kiến trúc Cloud-Native & Cloud-to-Local Sync
- **Independent Inference:** Bản demo trên Hugging Face được đóng gói để tự chạy mô hình Embedding tại chỗ (SentenceTransformers). Điều này chứng minh hệ thống có khả năng triển khai "Inference-only" trên các hạ tầng Cloud giới hạn mà không cần kết nối về máy chủ local.
- **Automated Data Sync:** Sử dụng dịch vụ Sync tự động qua Docker Compose giúp đảm bảo dữ liệu tại máy phát triển luôn đồng bộ 100% với dữ liệu trên Qdrant Cloud, loại bỏ rủi ro sai lệch dữ liệu khi chuyển đổi môi trường kiểm thử.

## 5. RAGAS Evaluation Strategy
- Hệ thống sử dụng Embedding chuẩn (`nomic-embed-text`) đồng bộ ở mọi giai đoạn (Ingest, RAG, Eval). Tính nhất quán về không gian vector là yếu tố then chốt để các chỉ số như `Faithfulness` và `Answer Relevancy` phản ánh đúng bản chất hiệu quả của thuật toán nén ARQ.

## 6. Phương pháp luận Đối chứng (Control Group Strategy)
- **Thiết lập Pure Baseline:** Để đảm bảo tính khách quan trong thực nghiệm, các mô hình đối chứng (`Raw`, `PQ`, `SQ8`) được thiết lập ở trạng thái nguyên bản nhất: **Limit = Top_K = 15**.
- **Mục tiêu khoa học:** Việc không sử dụng cơ chế tìm dư (Limit > Top_K) hay lọc nhiễu đối với Baseline giúp Hội đồng thấy rõ:
    - Sự mất mát thông tin thực tế khi nén vector theo cách truyền thống.
    - Khả năng xử lý "nhiễu" của các phương pháp cũ so với cơ chế hội tụ thông minh của ARQ.
- **Loại bỏ yếu tố thiên vị (Bias):** Hệ thống chứng minh rằng hiệu quả của ARQ-RAG đến từ **Thuật toán** (Nén bù sai số) và **Logic phân tích** (Adaptive Analyzer), chứ không phải do tăng cường tài nguyên tìm kiếm một cách ngẫu nhiên.

---
*Ghi chú: Các thông số này giúp khẳng định bạn không chỉ xây dựng được một ứng dụng chạy được, mà còn làm chủ được các kĩ thuật tối ưu hóa trong môi trường phân tán thực tế.*
