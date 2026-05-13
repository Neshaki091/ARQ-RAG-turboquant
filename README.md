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

## 🏆 Kết quả Đánh giá Vector Engine (5 Triệu Vectors)
*Thực nghiệm thực hiện trên: Laptop TUF Dash F15 (Intel Core i5-10300H, 16GB RAM, 1TB SSD NVMe PCIe 3.0 x4 | PCIe 4.0 x4)*

Hệ thống lõi đã trải qua bài kiểm tra chịu tải cực hạn (Stress Test) trên tập dữ liệu `facebook/wiki_dpr` quy mô **5 triệu vectors (768 chiều)**. Kết quả thực nghiệm (`benchmark_results.json`) đã chứng minh tính ưu việt tuyệt đối của TurboQuant (TQ-IVF) trong điện toán biên:

| Thuật toán | Mật độ nén | Private RAM (Mandatory) | Working Set (Max Cache) | Trạng thái (Scale 5M+) |
| :--- | :--- | :--- | :--- | :--- |
| **FAISS-SQ** | 4-bit | ~ 1,867 MB | 1,867 MB | ⚠️ OOM Risk (Memory Wall) |
| **FAISS-PQ** | 4-bit | ~ 1,873 MB | 1,873 MB | ⚠️ OOM Risk (Memory Wall) |
| **TQ-IVF** | 4-bit | **~ 12.0 MB** | ~ 1,334 MB | ✅ Ổn định (Zero-Copy) |
| **TQ-IVF** | 2-bit | **~ 16.0 MB** | ~ 554 MB | ✅ Ổn định (Zero-Copy) |

> **⚠️ Lưu ý cực kỳ quan trọng:** **Private RAM** là lượng bộ nhớ ứng dụng bắt buộc phải chiếm giữ để hoạt động (không thể giải phóng). **Working Set** bao gồm cả Page Cache mà Hệ điều hành tự động mượn để tăng tốc. TurboQuant chỉ cần ~12MB Private RAM để sống, trong khi các thư viện In-memory như FAISS đòi hỏi toàn bộ dữ liệu phải nằm trong Private RAM (1.8GB), dẫn đến lỗi OOM (std::bad_alloc) nếu bộ nhớ vật lý bị giới hạn.

### 📈 Chi tiết Độ chính xác (Accuracy Metrics)

#### 1. Top-1 Probability (%) - Chế độ 2-bit
| Algorithm | P@1 | P@8 | P@16 | P@64 |
| :--- | :---: | :---: | :---: | :---: |
| **TQ-IVF (np=2)** | 17.5% | 28.0% | 29.5% | 29.8% |
| **TQ-IVF (np=16)** | 26.0% | 63.3% | 68.8% | 73.3% |
| **TQ-IVF (np=64)** | 27.5% | 72.3% | 81.3% | **87.5%** |
| **FAISS-PQ (Baseline)** | 35.3% | 71.8% | 78.0% | 84.3% |

#### 2. Top-1 Probability (%) - Chế độ 4-bit
| Algorithm | P@1 | P@8 | P@16 | P@64 |
| :--- | :---: | :---: | :---: | :---: |
| **TQ-IVF (np=2)** | 36.8% | 43.5% | 43.5% | 43.5% |
| **TQ-IVF (np=16)** | 60.5% | 77.0% | 77.5% | 77.5% |
| **TQ-IVF (np=64)** | 69.0% | 90.5% | 91.0% | **91.0%** |
| **FAISS-SQ 4b** | 58.5% | 85.5% | 85.8% | 85.8% |
| **FAISS-PQ 4b** | 66.0% | 84.8% | 85.3% | 85.5% |

#### 3. Set Recall@K (%) - Chế độ 2-bit
| Algorithm | R@1 | R@8 | R@16 | R@64 |
| :--- | :---: | :---: | :---: | :---: |
| **TQ-IVF (np=2)** | 17.5% | 21.2% | 22.1% | 25.2% |
| **TQ-IVF (np=16)** | 26.0% | 38.7% | 40.7% | 44.9% |
| **TQ-IVF (np=64)** | 27.5% | 43.7% | 46.6% | **50.9%** |
| **FAISS-PQ 2b** | 35.3% | 44.4% | 47.4% | 50.7% |

#### 4. Set Recall@K (%) - Chế độ 4-bit
| Algorithm | R@1 | R@8 | R@16 | R@64 |
| :--- | :---: | :---: | :---: | :---: |
| **TQ-IVF (np=2)** | 36.8% | 35.3% | 37.1% | 37.6% |
| **TQ-IVF (np=16)** | 60.5% | 62.1% | 63.6% | 63.0% |
| **TQ-IVF (np=64)** | 69.0% | 72.3% | 74.1% | **74.4%** |
| **FAISS-SQ 4b** | 58.5% | 67.9% | 69.0% | 68.8% |
| **FAISS-PQ 4b** | 66.0% | 67.4% | 69.1% | 68.0% |

**Kết luận cốt lõi:** Các thư viện chuẩn công nghiệp như FAISS gặp **Nút thắt Bộ nhớ (Memory Wall)** khi Scale-up do cơ chế nạp toàn bộ mảng dữ liệu vào RAM (Heap). Ngược lại, **TurboQuant** với kiến trúc **Zero-Copy Memory Mapping (`mmap`)** kết hợp Rust SIMD chỉ tiêu tốn đúng ~12MB RAM, cho phép truy xuất khối lượng dữ liệu khổng lồ ngay trên các thiết bị giới hạn tài nguyên. TQ-IVF 4-bit đạt độ chính xác vượt trội (P@64 > 91%) trong khi vẫn duy trì mức tiêu thụ RAM tối thiểu.

### 📊 Phân tích Trực quan & Các Chỉ Số Đo Lường (Metrics)

Để trực quan hóa các con số trên, bạn có thể sử dụng script `visualize_results.py` đi kèm:
```bash
python Benchmark/eval_alt/visualize_results.py
```

Dưới đây là các định nghĩa và đồ thị thể hiện sức mạnh của hệ thống TurboQuant so với các chuẩn công nghiệp (FAISS):

#### 1. Sự đánh đổi giữa Chất lượng và Tốc độ (NDCG@16 / Recall@16 vs QPS)
**Định nghĩa:**
*   **Recall@K (Intersection Recall):** Thể hiện phần trăm số lượng vector chuẩn (Ground Truth) được giữ lại trong K kết quả tìm kiếm. Rất quan trọng đối với RAG vì LLM cần đa dạng ngữ cảnh để tổng hợp thông tin.
*   **NDCG@K (Normalized Discounted Cumulative Gain):** Thước đo quan trọng nhất cho RAG. Nó không chỉ xem xét thuật toán có tìm đúng tài liệu hay không, mà còn đánh giá xem tài liệu tốt nhất có được **xếp hạng cao nhất** (đưa lên đầu Prompt) hay không.
*   **QPS (Queries Per Second):** Tốc độ thông lượng (Throughput) của hệ thống khi chạy lô (Batching).

**Biểu đồ (Cột là Accuracy, Đường đứt nét là QPS):**
| Chất lượng Xếp hạng (NDCG@16 & QPS) | Chất lượng Thu hồi (Recall@16 & QPS) |
| :---: | :---: |
| ![NDCG 16](./benchmark_result/charts/ndcg_16_bar_with_qps.png) | ![Recall 16](./benchmark_result/charts/recall_16_bar_with_qps.png) |

**Phân tích:** 
Ở phiên bản nén 4-bit, TurboQuant đánh bại hoàn toàn FAISS PQ/SQ về cả NDCG (81.6% > 77.7%) lẫn Recall (74.2% > 69.1%), đồng thời nhỉnh hơn về tốc độ QPS. Ở cấu hình cực hạn 2-bit, TQ duy trì độ chính xác tương đương FAISS PQ, chấp nhận giảm QPS một chút để đánh đổi lấy sự kỳ diệu về kiến trúc bộ nhớ ở phần dưới.

#### 2. Phá vỡ bức tường bộ nhớ (The Memory Wall)
**Định nghĩa:**
*   **Private RAM (Màu xanh - Bắt buộc):** Lượng bộ nhớ cứng mà ứng dụng phải cấp phát (`malloc`) và chiếm giữ. Nếu cạn dung lượng này, ứng dụng sẽ sập (Crash / Out of Memory).
*   **Working Set / Page Cache (Màu cam - Linh hoạt):** Vùng RAM mà Hệ điều hành cho ứng dụng mượn tạm để chứa file đọc từ ổ cứng. Nếu máy tính hết RAM cho việc khác, OS sẽ tự động thu hồi vùng này mà không làm chết ứng dụng RAG.

![Efficiency Comparison](./benchmark_result/charts/efficiency_comparison.png)

**Phân tích:** 
Các thuật toán In-memory truyền thống như FAISS theo đuổi triết lý 'Đổi dung lượng lấy Tốc độ', nạp toàn bộ 5 triệu vector trực tiếp vào Private RAM, gây tiêu tốn khổng lồ (~1.9 GB) $\to$ Không thể triển khai trên các thiết bị Edge AI (IoT, Raspberry Pi, Laptop yếu). Ngược lại, TurboQuant thiết lập tiêu chuẩn mới với kỹ thuật **Zero-copy (Mmap)** bằng Rust, chỉ tốn vỏn vẹn **~20MB Private RAM** (giảm gần 100 lần) để chứa các điểm Tâm cụm (Centroids), đẩy toàn bộ khối lượng vector khổng lồ sang Page Cache linh hoạt của Hệ điều hành. Đây là bước đột phá kỹ thuật cốt lõi của đồ án.

#### 3. Độ biến dạng lượng tử hóa (MSE - Mean Squared Error)
**Định nghĩa:** MSE đo lường sai số vật lý bình phương giữa vector sau khi giải nén so với vector Float32 nguyên bản. MSE càng nhỏ, vector nén càng giống vector gốc.

![MSE Comparison](./benchmark_result/charts/mse_comparison.png)

**Phân tích:** 
Phương pháp lượng tử hóa thặng dư QJL của TurboQuant tập trung vào việc **bảo toàn góc** (Cosine Similarity) để tối ưu độ chính xác ngữ nghĩa cho văn bản, thay vì tối ưu sai số Euclid như mạng Codebook của FAISS. Kết quả là MSE của TQ đóng vai trò là một chỉ số tham khảo vật lý (mang tính ước lượng lý thuyết do tính chất Irreversible của mã Sign-bit). Thực tế đo lường thực nghiệm (NDCG/Recall ở trên) chứng minh khả năng bảo toàn hướng vector của TQ ưu việt hơn hẳn, giúp chất lượng xếp hạng tốt hơn bất chấp đánh đổi về MSE.

## 🛠 Hướng dẫn chạy Benchmark Lõi (eval_alt)

### 1. Yêu cầu hệ thống (Hardware & Software)
*   **Hệ điều hành:** Windows 10/11 hoặc Linux (Khuyến khích Ubuntu 22.04+).
*   **Phần cứng tối thiểu:** CPU i3 đời 10+, RAM 8GB (TQ không yêu cầu nhiều RAM nhưng quá trình tải/chuẩn bị dữ liệu ban đầu cần RAM để xử lý).
*   **Dung lượng ổ đĩa:** Cần trống ít nhất **40GB** (Dành cho việc tải 15GB dữ liệu thô, giải nén và lưu trữ các file Index đa dạng của TQ/FAISS). Khuyến khích sử dụng **SSD NVMe (PCIe 3.0 x4 hoặc cao hơn)** để đạt tốc độ quét tốt nhất qua mmap.
*   **Python:** Phiên bản 3.10 trở lên.

### 2. Cài đặt thư viện lõi
Chạy lệnh sau để cài đặt các thư viện toán học cần thiết:
```bash
pip install -r requirements_benchmark.txt
```

### 3. Các lệnh chạy Benchmark phổ biến

* **Chạy bài Test nhanh (50.000 vectors):**
  Lệnh này tải dữ liệu rất nhanh (~150MB) và FAISS sẽ không bị văng lỗi tràn RAM. Phù hợp để test luồng chạy.
  ```bash
  python Benchmark/eval_alt/benchmark.py --max-vectors 50000 --rebuild-cache
  ```

* **Chạy bài Stress Test Cực hạn (5 triệu vector):**
  Lệnh này tái hiện lại môi trường thực nghiệm của Luận văn. Lưu ý: Cần trống 20GB ổ cứng và mất 1-3 tiếng để tải dữ liệu từ HuggingFace.
  ```bash
  python Benchmark/eval_alt/benchmark.py --max-vectors 5000000
  ```

### 4. Giải thích các tham số (CLI Arguments)
- `--max-vectors <số_lượng>`: Chỉ định giới hạn số lượng vector tải về từ HuggingFace để nạp vào Index.
- `--rebuild-cache`: Ép hệ thống xóa bỏ các file cache nén cũ (`tq_index_temp`) và nạp lại dữ liệu gốc từ đầu. Rất hữu ích khi bạn đổi số lượng vector.
- `--k-values "1,2,4,8,16,64"`: Chỉ định các mốc đo lường độ phủ Set Recall@K.
- `--tq-nprobes "2,4,8,16"`: Điều chỉnh số lượng cụm lân cận cần quét của lõi TurboQuant. Mức nprobe càng cao thì Recall càng lớn nhưng QPS giảm.
- `--query-json <path>`: Dùng bộ câu hỏi chữ thật thay vì tạo câu hỏi ảo ngẫu nhiên.

---
*Tài liệu hướng dẫn triển khai dự án ARQ-RAG (TurboQuant).*

# PHÁT TRIỂN BỞI 2 SINH VIÊN:
- Huỳnh Công Luyện 
- Nguyễn Đình Mạnh

# MỤC TIÊU:
- Tìm kiếm giải pháp tối ưu hóa tăng cường truy xuất lượng tử trực tuyến (Online Quantization-Enhanced Retrieval Optimization) trong hệ thống RAG.
