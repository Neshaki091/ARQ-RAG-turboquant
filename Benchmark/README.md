# 🚀 TurboQuant: Hệ Thống Truy Vấn Vector Hiệu Năng Cao Cho RAG

## 📝 Giới thiệu hệ thống
**TurboQuant (TQ)** là một công cụ tối ưu hóa truy vấn vector (Vector Search Engine) được thiết kế chuyên biệt cho các hệ thống RAG (Retrieval-Augmented Generation) và KV-Cache của mô hình ngôn ngữ lớn (LLM). Hệ thống sử dụng thuật toán nén dữ liệu **hai giai đoạn (Two-stage Quantization)** kết hợp giữa **MSE (Mean Squared Error)** và **QJL (Quantized Johnson-Lindenstrauss)** để đạt được tỷ lệ nén cực cao mà vẫn giữ được độ chính xác gần như tương đương với dữ liệu gốc (FP32).

### ✨ Đặc điểm nổi bật:
*   **Siêu nén**: Hỗ trợ các mức nén linh hoạt 3-bit, 5-bit và 9-bit.
*   **Độ chính xác vượt trội**: Recall@10 đạt >98% ngay cả ở mức nén 3-bit, cao hơn hẳn so với SQ8 truyền thống.
*   **Tối ưu phần cứng**: Cơ chế nén bit-packing giúp tiết kiệm băng thông GPU/CPU và bộ nhớ RAM.
*   **An toàn bộ nhớ**: Tích hợp chế độ **Micro-batching** và **Adaptive Sharding**, cho phép xử lý 5 triệu vector 768 chiều chỉ với tối đa **4GB RAM**.

---

## 🛠️ Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu
Sử dụng script `prepare_data.py` để sinh dữ liệu thử nghiệm (5 triệu vector thực tế). Script này được thiết kế để không vượt quá 4GB RAM trong quá trình nén.
```bash
python eval/prepare_data.py
```
*Lựa chọn Option 6 để sinh toàn bộ các bản nén 3/5/9-bit.*

### 2. Chạy phân tích hiệu năng (Benchmarking)
Sử dụng công cụ `stress_5m.py` để thực hiện kiểm tra áp lực và đo đạc các chỉ số:
```bash
python eval/stress_5m.py
```
**Quy trình vận hành:**
*   **Bước 1**: Chọn chế độ xử lý.
    *   `Standard Mode`: Chế độ tiêu chuẩn (Batch cố định).
    *   `Optimal RAM Mode`: Chế độ tối ưu RAM (Tự động điều chỉnh Batch size để bảo vệ hệ thống).
*   **Bước 2**: Chọn biến thể cần so sánh hoặc chọn `4` để xuất bảng tổng hợp toàn bộ.

---

## 📊 Kết quả Benchmark thực tế (Massive 5M Vectors)

Dưới đây là kết quả đo đạc trực tiếp trên hệ thống với 5 triệu vector chiều 768.

### Chế độ 1: Standard Mode (Fixed Batch = 500k)
Cung cấp tốc độ cao hơn nhưng tiêu tốn RAM đỉnh (Peak) lớn hơn đối với dữ liệu RAW.

| Phương pháp | RAM Peak | Thời gian/Query | Recall@10 | Tốc độ |
| :--- | :--- | :--- | :--- | :--- |
| **RAW (FP32)** | **1469.3 MB** | 16.8512s | 100.0% | 1.0x |
| **SQ8** | 366.8 MB | 1.8356s | 92.5% | 9.2x |
| **TQ 3-bit** | **372.2 MB** | **0.8731s** | **98.2%** | **19.3x** |
| **TQ 5-bit** | 238.8 MB | 1.3782s | 99.5% | 12.2x |
| **TQ 9-bit** | 238.8 MB | 1.3911s | 99.9% | 12.1x |

### Chế độ 2: Optimal RAM Mode (Adaptive Batching)
Tối ưu tuyệt đối cho máy có cấu hình 4GB RAM bằng cách chia nhỏ Shard nạp vào bộ nhớ.

| Phương pháp | RAM Peak | Thời gian/Query | Recall@10 | Tốc độ |
| :--- | :--- | :--- | :--- | :--- |
| **RAW (FP32)** | **295.3 MB** | 17.0405s | 100.0% | 1.0x |
| **SQ8** | 367.6 MB | 2.0429s | 92.5% | 8.3x |
| **TQ 3-bit** | **373.0 MB** | **0.9798s** | **98.2%** | **17.4x** |
| **TQ 5-bit** | 332.7 MB | 1.3189s | 99.5% | 12.9x |
| **TQ 9-bit** | 239.6 MB | 1.3778s | 99.9% | 12.4x |

---

## 💡 Kết luận

1.  **Về Tốc độ**: TurboQuant 3-bit mang lại sự bứt phá vượt bậc với tốc độ truy vấn nhanh gấp **17x - 19x** lần so với dữ liệu gốc, đồng thời nhanh gấp đôi so với SQ8.
2.  **Về Bộ nhớ**: Ở chế độ tối ưu, hệ thống giảm mức chiếm dụng RAM đỉnh của dữ liệu RAW từ ~1.5GB xuống chỉ còn **~300MB** mà không ảnh hưởng đáng kể đến hiệu năng.
3.  **Về Độ chính xác**: Các biến thể TQ vượt xa SQ8 về độ chính xác (Recall), giữ vững mức **>98%** ngay cả ở điều kiện nén khắt khe nhất (3-bit).
4.  **Tính ứng dụng**: TurboQuant là giải pháp lý tưởng cho việc triển khai RAG trên các thiết bị Edge hoặc các hệ thống Server có tài nguyên RAM hạn chế nhưng yêu cầu lượng Token xử lý lớn.

---
*Dự án thực hiện cho mục đích nghiên cứu và đồ án tốt nghiệp.*
