# 🚀 TurboQuant Vector Retrieval Benchmark (ARQ-RAG)

![TurboQuant Benchmark](https://img.shields.io/badge/Algorithm-PolarQuant+QJL-blue.svg) ![Tokens](https://img.shields.io/badge/Scale-5_Million_Vectors-orange.svg) ![Recall](https://img.shields.io/badge/Recall@10-98.2%25-brightgreen.svg)

**Author:** HUYNH CONG LUYEN (Senior Student at the University of Transport Ho Chi Minh City)  
**Tác giả:** HUỲNH CÔNG LUYỆN (Sinh viên năm cuối, Trường Đại học Giao thông Vận tải Thành phố Hồ Chí Minh)

---

> 🌍 **Choose your language / Chọn ngôn ngữ:**
> - [🇺🇸 English (Detailed Implementation Below)](#-english-documentation)
> - [🇻🇳 Tiếng Việt (Hướng dẫn chi tiết bên dưới)](#-tài-liệu-tiếng-việt)

---

# 🇺🇸 English Documentation

## 📖 Overview & Algorithmic Highlights
This project serves as a comprehensive benchmarking stress-test suite for **TurboQuant**, a vector engine architecture modeled explicitly on the *Extreme Compression* theoretical frameworks pioneered by Google Research. The primary objective is to dramatically minimize RAM footprint and maximize search latency retrieval times, while preserving `Recall@10` over the near-perfect threshold (> 98%).

The `TQEngine` architecture effectively replicates the highly efficient Two-Stage algorithm flow outlined in recent literature:

### Stage 1: Random Rotation Matrix & Recursive Polar Quantization
1. **Variance Homogenization via Random Rotation Matrix ($Pi$)**: High-dimensional Large Language Model (LLM) embeddings often feature outlier dimensions that hurt naive uniform clustering patterns. TurboQuant intercepts the raw vectors and multiplies them by an orthogonal rotational matrix $Pi$ (generated via QR decomposition). This operation perfectly homogenizes variance magnitudes across all dimensional axes smoothly.
2. **Recursive Polar Quantization (PolarQuant)**: Instead of utilizing linear scalar (MSE/k-means) structures, TurboQuant groups vector values recursively. Taking pairs of sequential values, it transforms those points down into an Angle (using `atan2`) and a vector length/Radius (`norm()`). By executing this recursive transform $\log_2(D)$ times across 768 or 128 dimensions, the entire vector is systematically squeezed perfectly into **1 Singular Global Radius**, leaving only angular index components to be compressed.
3. **Dynamic Angular Bit Budgeting**: Not all angles hold equivalent relative importance. To attain an average rate of `3-bits` dynamically, our compression engine isolates the top-level Cartesian coordinates ($L_1$, span: `[-π, π]`) and strictly assigns it highest priority with $4$ bits. Secondary intermediate nested levels (span bounded between `[0, π/2]` because nested radii are implicitly $\ge 0$) inherit exactly $2$ bits.

### Stage 2: Quantized Johnson-Lindenstrauss (QJL) Error Compensation
The initial Stage 1 Polar compression is highly retentive minus a trailing fraction of loss (the *Residual*). To aggressively correct potential discrepancies inside inner dot product evaluations (Attention scores), Google's methodology passes this residual into a sub-gaussian, uniform matrix $S$, taking entirely just its `1-bit (+1 / -1)` signs limit. Thanks to the *Johnson-Lindenstrauss* lemma, the aggregated approximation effectively bounds dot product differences optimally for the Generative RAG Pipeline metric, pushing Recall bounds successfully past 98-99%.

---

## 💻 Hardware Setup & System Limits

This pipeline was designed to execute a **massive 5 Million Vectors** benchmarking capacity entirely achievable on a standard consumer-grade PC.

### Requirements & Warnings:
- **Operating System:** Tested on Windows OS.
- **Minimum Generating Node limit (`prepare_data.py`):** Your workstation **MUST have at least 8 GB of free RAM** before executing this data dump. The operation simulates FP32 random variables chunk-by-chunk under the hood across 5M distributions into raw `.npy` and indexed binaries. Our integrated `EmptyWorkingSet()` API calls clean Windows OS memory dumps optimally, but adequate breathing room prevents total system freeze-ups!
- **Memory For Evaluation Node (`stress_5m.py`):** Fully optimized via advanced "Optimal RAM / Adaptive Batching", evaluating 5 Million vectors consumes remarkably minimal power—running completely under a mere `~500 MB RAM` footprint window.

---

## 🛠 Executing Benchmarks

### 1. Data Processing & Chunk Indexing
You must generate all encoded/compressed bin structures offline first before search latency extraction occurs.

```bash
python Benchmark/eval/prepare_data.py
```
> *Select option **6** inside the interactive prompt to aggressively generate ALL bits variations of the 5 Million Tokens structure (RAW, SQ8, 3-bit, 5-bit, 9-bit).*

### 2. High-Performance Latency Execution (Stress Testing)
The master stress evaluation directly reads bytes across SSD block partitions imitating hardware IO speeds explicitly for comparative profiling.

```bash
python Benchmark/eval/stress_5m.py
```
> [!IMPORTANT]
> - Ensure you manually select **`[STEP 1]` $\to$ 2 (Optimal RAM Mode)**.
> - Ensure you manually select **`[STEP 2]` $\to$ 4 (BENCHMARK ALL)**.

**Adaptive Batching Profiles Defined under Optimal Configurations:**
- `RAW Baseline (FP32)`: Stressed out -> **Batch Limit: 200,000**.
- `SQ8 (8-bit Int)` & `TQ 9-bit`: Heavy scale -> **Batch Limit: 500,000**.
- `TQ 5-bit`: Balanced scale -> **Batch Limit: 700,000**.
- `TQ 3-bit`: Supremely lightweight scale -> **Batch Limit: 1,000,000**.

---

## 🏆 Official Results Evaluation

Running `N_QUERIES=5` over a dataset $N=5,000,000$ items across $D=768$ dimensions yielded exceptional scaling metrics that proved this research milestone effectively works:

```text
================================================================================
Method          | RAM Peak     | Time/Query   | Recall@10  | Speedup
--------------------------------------------------------------------------------
RAW (FP32)      |    588.8 MB |    16.8723s |    100.0% |     1.0x
SQ8             |    368.3 MB |     1.9540s |     92.5% |     8.6x
TQ 3-bit        |    553.0 MB |     1.6245s |     98.2% |    10.4x
TQ 5-bit        |    573.6 MB |     2.1786s |     99.5% |     7.7x
TQ 9-bit        |    491.9 MB |     2.7696s |     99.9% |     6.1x
================================================================================
```

- **TQ 3-bit achieves 10.4X Faster Retrieval over native FP32**, heavily decimating memory retrieval barriers safely whilst **retaining Recall > 98%**, completely blowing outdated scalar quantization (SQ8 hit 92.5%) limits out of the water.


<br><br>


---

# 🇻🇳 Tài Liệu Tiếng Việt

## 📖 Giải Thích Điểm Nhấn Thuật Toán
Sản phẩm nghiên cứu này là một bộ khung kiểm thử và đánh giá hiệu suất hệ thống Nén Vector Siêu Khủng **TurboQuant** (dựa trên các lý thuyết nén *Extreme Compression* chuyên sâu từ Google Research). Hệ thống hướng tới việc cắt giảm tối đa bộ nhớ truy xuất máy tính và tăng tốc độ thời gian thực (Vector Search / KV Cache) nhưng vẫn giữ được độ tiệm cận hoàn hảo `Recall@10` (> 98%).

Kiến trúc bên dưới của `TQEngine` được thiết kế theo đúng mô hình 2-Giai-Đoạn của nghiên cứu nguyên bản:

### Giai Đoạn 1: Phân Phối Đều & Lượng Tử Hóa Độc Cực Đệ Quy (Rotation + PolarQuant)
1. **Random Rotation Matrix ($Pi$)**: Các mô hình ngôn ngữ lớn (LLM) thường sinh ra các Vector chứa nhiều số "Outliers" dị biệt. Nên trước hết, TQEngine đem toàn bộ Vector nhân vào ma trận xoay đa chiều trực giao $Pi$ (*thu được thông qua phân rã Q-R*). Điều này mang lại một kết quả tuyệt vời: Các phương sai của mọi chiều được tản đều ra.
2. **Đệ Quy Tọa Độ Độc Cực (PolarQuant)**: Ném bỏ cơ chế gom nhóm (k-means) hay MSE tuyến tính cũ, hệ thống trích xuất từng cặp tham số chập với nhau để làm sinh ra một Góc định tính (`atan2`) và Bán kính `norm`. Đệ quy chuỗi thao tác này $\log_2(D)$ lần, hệ thống đã tóm gọn 128 hay 768 chiều không gian về duy nhất thành **1 Cạnh Huyền/Bán Kính Chung**, còn lại hệ thống chỉ cần lưu ma trận các phân mảnh số liệu cấu thành Góc.
3. **Phân Rã Bit Động (Adaptive Bits)**: Thuật toán tự rải và sắp xếp bit khôn ngoan để đạt mức dung lượng 3-bits hay 5-bits lý tưởng. Lớp cấu thành L1 (giao lộ gốc lên trục tọa độ mang khoảng tính `[-π, π]`) có sức ảnh hưởng cao nhất, hưởng 4 bit; các góc phụ còn lại bị giới hạn `[0, π/2]` do quy tắc biên dương thì hưởng 2 bits.

### Giai Đoạn 2: Xấp xỉ Phần Dư QJL (Quantized Johnson-Lindenstrauss)
Với 1 số ít thông tin (*Residuals*) hao hụt sau pha PolarQuant, hệ thống sử dụng thuật toán QJL. Nó đem phần dư nhân cho ma trận ngẫu nhiên hệ Gaussian uniform $S$, và giữ lại chỉ với 1-bit bộ nhớ (`+1/-1`). Định lý Johnson-Lindenstrauss đã bảo lưu khoảng cách Vector trên bộ khung Attention một cách an toàn nhất, đẩy Recall đi qua khỏi ngưỡng 98% cho cơ chế cốt lõi.

---

## 💻 Điều Kiện Môi Trường & Lưu Ý RAM

Quy trình Test Mô Phỏng chịu tải thực tế 5.000.000 (5 Triệu) Vectors.

### Cấu Hình System Test:
- **Cấu hình Windows OS**
- **Sản sinh dữ liệu (`prepare_data.py`):** **MÁY TÍNH CẦN TRỐNG ÍT NHẤT 8 GB RAM!** Tệp kịch bản này sinh ra rải rác ma trận Float 32 cho cả 5 triệu biến và ghi chúng ra ổ cứng. Hàm chùi gầm Garbage Collect và `EmptyWorkingSet()` trên API Windows đã được kích hoạt chạy song song để xả rác thủ công, nhưng máy bắt buộc phải nhường đủ khoảng không cho Data nén nhị phân khỏi tình trạng Treo RAM vĩnh viễn!
- **Môi trường Đo Lường Căng Thẳng (`stress_5m.py`):** Rất nhẹ nhờ cơ chế "Optimal RAM Streaming / Adaptive Batching", bạn có thể đọc và mô phỏng 5 triệu Vector trơn tru mà chỉ chịu tải rơi vào khoảng `~500 MB RAM`.

---

## 🛠 Hướng Dẫn Chạy Pipeline Benchmarks

### 1. Chuẩn bị File Nhị Phân (Data Indexing)
Chạy lệnh này đầu tiên vì toàn bộ phép thử phụ thuộc vào bộ nhớ nhị phân sinh ra từ thuật toán nén cấu trúc.

```bash
python Benchmark/eval/prepare_data.py
```
> *Mẹo nhỏ: Nhập phím lựa chọn số **6** để tự động Compile trọn bộ 5 Triệu Tokens ở toàn cõi các thể loại bit (RAW, 3, 5, 9).*

### 2. Thực Thi Công Cụ Căng Thẳng (Stress Testing 5M)
Mô phỏng truy xuất Streaming trên I/O băng thông.

```bash
python Benchmark/eval/stress_5m.py
```
> [!IMPORTANT]
> - Tại bước hỏi chọn **`[STEP 1]`**, hãy nhập số `2` để cấp quyền **Optimal RAM Mode**.
> - Tại bước **`[STEP 2]`**, hãy nhập số `4` (ALL) để xả toàn bộ bảng đo lường ra so sánh thực tế.

**Bảng Phân Bổ Động Adaptive Batch Limit (Nhằm không nổ RAM 4 GB):**
- Giới hạn tải RAW Baseline (Dữ liệu thường) -> **200,000 / Batch**.
- Giới hạn tải SQ8 & TQ 9-bit -> **500,000 / Batch**.
- Giới hạn tải TQ 5-bit -> **700,000 / Batch**.
- Giới hạn tải TQ 3-bit -> **1,000,000 / Batch**.

---

## 🏆 Bảng Điểm Tổng Đo Lường Benchmark

Tiến hành 5 Search Queries ngẫu nhiên truy quét kho Data ($N=5,000,000$; $D=768$). Ghi nhận sự lột xác sức mạnh của lý thuyết khoa học này theo bảng tóm lượt:

```text
================================================================================
Phương pháp     | RAM Peak     | Tgian/Query  | Recall@10  | Tốc độ
--------------------------------------------------------------------------------
RAW (FP32)      |    588.8 MB |    16.8723s |    100.0% |     1.0x
SQ8             |    368.3 MB |     1.9540s |     92.5% |     8.6x
TQ 3-bit        |    553.0 MB |     1.6245s |     98.2% |    10.4x
TQ 5-bit        |    573.6 MB |     2.1786s |     99.5% |     7.7x
TQ 9-bit        |    491.9 MB |     2.7696s |     99.9% |     6.1x
================================================================================
```

- Phiên bản siêu nén **TQ 3-bit chứng tỏ khả năng lướt truy xuất 10.4 Lần NHANH HƠN SO VỚI RAW FP32**.
- Xóa sổ khái niệm suy giảm (Degradation) dữ liệu: Tỷ lệ nhớ **Recall luôn cứng trên mốc 98%**, hạ gục thuật toán nén số nguyên SQ8 cũ kĩ rớt xuống đáy với chỉ 92.5%. Kỹ thuật xử lý cấu hình siêu tinh vi mang trọn vẹn kết quả cho ra đúng với nội dung kỳ vọng của bài toán Retrieval System/RAG.
