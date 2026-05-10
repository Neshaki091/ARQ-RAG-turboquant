# TurboQuant (TQ) Engine
**High-Performance Vector Search Middleware with SQ+QJL Quantization & IVF Support**

TurboQuant là một công cụ tìm kiếm vector mật độ cao (Dense Vector Search) được tối ưu hóa đặc biệt cho các môi trường hạn chế về tài nguyên (RAM) nhưng yêu cầu tốc độ truy vấn cực nhanh và độ chính xác cao. Dự án này triển khai kiến trúc **SQ+QJL (Scalar Quantization + Quantized Johnson-Lindenstrauss)** kết hợp với **Inverted File Index (IVF)**.

## 🚀 Tính năng nổi bật
- **Siêu nén (Ultra-Compression):** Hỗ trợ nén vector xuống 2-bit hoặc 4-bit mà vẫn duy trì Recall@10 > 90%.
- **Tự động Bootstrapping:** Tự động kiểm tra và biên dịch lõi Rust SIMD (AVX2/FMA) ngay lần đầu khởi chạy.
- **Cơ chế Out-of-Core:** Khả năng tìm kiếm trên 5 triệu vector chỉ với **~230MB RAM** (so với 1GB+ của các phương pháp truyền thống).
- **IVF Accelerated:** Tăng tốc tìm kiếm gấp 20+ lần thông qua phân cụm dữ liệu thông minh.
- **Dynamic Operations:** Hỗ trợ Thêm (`add`) và Xóa (`delete`) vector thời gian thực.
- **Hardware Acceleration:** Tự động sử dụng NVIDIA GPU (CUDA) để tăng tốc phép xoay ma trận và huấn luyện K-Means.


## 📊 Kết quả thực nghiệm (Stress Test 5M Vectors)
Dựa trên báo cáo `benchmark_report_alt.md` thực hiện trên tập dữ liệu 5,000,000 vectors (768-dim):

| Phương pháp | QPS (Queries/Sec) | RAM Tiêu thụ | Tốc độ (Speedup) | Độ chính xác (Recall@1) |
| :--- | :--- | :--- | :--- | :--- |
| **RAW (Float32)** | 0.06 | 932 MB | 1.0x | 100% |
| **SQ 4-bit** | 0.88 | 991 MB | 15.9x | 83% |
| **PQ 4-bit** | 0.59 | 865 MB | 10.6x | 81% |
| **TQ-IVF 4-bit (Ours)** | **1.17** | **236 MB** | **21.1x** | **90%** |

> **Nhận xét:** TurboQuant IVF không chỉ nhanh nhất mà còn là phương pháp tiết kiệm bộ nhớ nhất, trong khi vẫn giữ được độ chính xác (Recall) vượt trội hơn SQ và PQ.

## 🛠 Hướng dẫn sử dụng

### 1. Cấu hình môi trường
TurboQuant được thiết kế để chạy trực tiếp từ thư mục dự án (**Local Priority**).
- Yêu cầu: **Python 3.10+** và **Rust/Cargo** (để biên dịch lõi SIMD).
- Đảm bảo bạn đặt thư mục `TQ_engine_lib` cùng cấp với các file script của bạn.

### 2. Khởi tạo và Lập chỉ mục (Indexing)
```python
from TQ_engine_lib import TurboQuant
import torch

# Khởi tạo Engine (4-bit, có sử dụng IVF)
dim = 768
engine = TurboQuant(
    dim=dim, 
    bits=4, 
    use_ivf=True, 
    ivf_nlist=256,   # Số lượng cụm (clusters)
    ivf_nprobe=32    # Số lượng cụm sẽ quét khi tìm kiếm
)

# Giả sử bạn có 1 triệu vector (Tensor hoặc Numpy)
vectors = torch.randn(1000000, dim)
vectors = torch.nn.functional.normalize(vectors, dim=-1)

# Lập chỉ mục (Tự động thực hiện SQ+QJL và IVF Clustering)
engine.index(vectors)

# Lưu chỉ mục xuống đĩa (để sử dụng sau này)
engine.save_index("data/my_index")
```

### 3. Tìm kiếm (Search)
```python
# Load chỉ mục đã lưu
engine.load_index("data/my_index")

# Thực hiện truy vấn Top-K (Tự động lọc các ID đã xóa)
query = torch.randn(dim)
top_indices, top_scores = engine.search(query, top_k=10)

print(f"Top 10 Indices: {top_indices}")

### 4. Thêm & Xóa thời gian thực (Dynamic)
```python
# Xóa một vector khỏi kết quả (Soft Delete)
engine.delete(vector_id=5)

# Thêm một vector mới vào cụm gần nhất (Incremental Add)
new_vec = torch.randn(1, dim)
engine.add(new_vec, vector_id=9999)

# Tìm kiếm lại - Kết quả sẽ cập nhật ngay lập tức
ids, scores = engine.search(new_vec, top_k=1)
# ids[0] sẽ là 9999
```


## ⚠️ Lưu ý quan trọng
1. **Local Usage:** Luôn gọi thư viện từ thư mục dự án. Tránh cài đặt qua `pip install .` để đảm bảo logic Tự động-Bootstrapping hoạt động chính xác với mã nguồn mới nhất.
2. **Auto-Cleanup:** Sau khi biên dịch thành công, thư viện sẽ tự động xóa các file rác (`target/`, `Cargo.lock`) để tiết kiệm dung lượng.
3. **SIMD Requirements:** TurboQuant yêu cầu CPU hỗ trợ tập lệnh AVX2 và FMA để đạt được tốc độ như trong báo cáo.

## 📂 Cấu trúc thư mục thư viện
- `TQ_engine_lib/`: Thư mục chính của Engine.
- `TQ_engine_lib/core/`: Mã nguồn Rust SIMD.
- `eval_alt/`: Các script benchmark và đánh giá độ chính xác.
