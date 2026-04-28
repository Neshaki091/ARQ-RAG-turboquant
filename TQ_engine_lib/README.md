# TurboQuant (TQ) Native Library

TurboQuant là một công cụ tìm kiếm vector hiệu năng cực cao, sử dụng cơ chế nén **SQ+QJL (Scalar Quantization + Quick Johnson-Lindenstrauss)** kết hợp với tối ưu hóa **Rust SIMD (AVX2/FMA)**. 

Thư viện này được thiết kế để đạt được tốc độ tìm kiếm nhanh hơn gấp 10-50 lần so với tính toán RAW (Float32) trong khi vẫn duy trì độ chính xác (Recall) vượt trội so với các phương pháp SQ/PQ truyền thống.

## 🚀 Tính năng nổi bật
- **Siêu nén**: Hỗ trợ chế độ 2-bit và 4-bit (mỗi chiều vector chỉ tốn 2 hoặc 4 bits).
- **Tối ưu RAM**: Giảm dung lượng bộ nhớ lên tới 8-16 lần.
- **Tốc độ SIMD**: Lõi tính toán được viết bằng Rust, tối ưu hóa tận dụng tập lệnh AVX2 của CPU.
- **Dễ sử dụng**: Cung cấp High-level API tương tự các thư viện Vector DB chuyên nghiệp.

## 📦 Cài đặt
Trước khi sử dụng, hãy đảm bảo bạn đã cài đặt các thư viện phụ trợ:

```bash
pip install -r TQ_engine_lib/requirements.txt
```

*Lưu ý: Thư viện Native (.pyd) đã được biên dịch sẵn cho Windows x64.*

## 💡 Hướng dẫn sử dụng nhanh

Sử dụng lớp `TurboQuant` để thực hiện Indexing và Search một cách đơn giản nhất:

```python
import torch
from TQ_engine_lib import TurboQuant

# 1. Khởi tạo (dim: số chiều, bits: 2 hoặc 4)
dim = 768
tq = TurboQuant(dim=dim, bits=4)

# 2. Tạo dữ liệu giả lập và Indexing
# vectors: Tensor (N, dim)
vectors = torch.randn(100000, dim)
tq.index(vectors)

# 3. Tìm kiếm (Search)
query = torch.randn(dim)
indices, scores = tq.search(query, top_k=10)

print(f"Top 10 Indices: {indices}")
print(f"Scores: {scores}")
```

## 🛠️ Cấu hình chi tiết
TurboQuant hỗ trợ hai chế độ nén chính:
- **2-bit Mode**: (1-bit SQ + 1-bit QJL). Cực kỳ tiết kiệm RAM, tốc độ nhanh nhất.
- **4-bit Mode**: (3-bit SQ + 1-bit QJL). Cân bằng hoàn hảo giữa độ chính xác (Recall@10 > 95%) và tốc độ.

## 📊 Benchmarking
Để kiểm tra hiệu năng và độ chính xác trên máy của bạn, hãy sử dụng bộ script trong thư mục `eval_alt`:

```bash
# Chạy toàn bộ Stress Test (5M vectors) và Recall Test
python eval_alt/run_full_benchmarks.py
```
Kết quả sẽ được xuất ra file `benchmark_report_alt.md`.

## 🏗️ Cấu trúc thư mục
- `quantizer.py`: Lõi quản lý lượng tử hóa.
- `codebook.py`: Quản lý centroids và thuật toán Max-Lloyd.
- `tq_native.pyd`: Thư viện tính toán SIMD (Rust compiled).
- `core/`: Mã nguồn Rust (dành cho việc phát triển/biên dịch lại).

---
**TurboQuant - High-Performance Vector Search Engine**
*Thesis Project Optimization for Large-Scale RAG Systems.*
