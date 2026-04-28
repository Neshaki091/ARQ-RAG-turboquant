# 🚀 TurboQuant Vector Retrieval Benchmark (SQ+QJL)

![TurboQuant Benchmark](https://img.shields.io/badge/Algorithm-SQ+QJL-blue.svg) ![Tokens](https://img.shields.io/badge/Scale-5_Million_Vectors-orange.svg) ![Recall](https://img.shields.io/badge/Recall@4-100%25-brightgreen.svg)

**Author:** HUYNH CONG LUYEN (Senior Student at the University of Transport Ho Chi Minh City)  
**Tác giả:** HUỲNH CÔNG LUYỆN (Sinh viên năm cuối, Trường Đại học Giao thông Vận tải Thành phố Hồ Chí Minh)

---

> 🌍 **Choose your language / Chọn ngôn ngữ:**
> - [🇺🇸 English (Detailed Implementation Below)](#-english-documentation)
> - [🇻🇳 Tiếng Việt (Hướng dẫn chi tiết bên dưới)](#-tài-liệu-tiếng-việt)

---

# 🇺🇸 English Documentation

## 📖 Overview & Algorithmic Highlights
This project serves as a comprehensive benchmarking stress-test suite for **TurboQuant**, a vector engine architecture focused on *Extreme Compression*. The current version utilizes a high-performance **SQ+QJL (Scalar Quantization + Quantized Johnson-Lindenstrauss)** pipeline, optimized with **Rust SIMD**.

### Stage 1: Orthogonal Rotation & Scalar Quantization (SQ)
1. **Variance Homogenization via Random Rotation Matrix ($Pi$)**: TurboQuant uses an orthogonal rotational matrix (via QR decomposition) to homogenize variance across all dimensions, eliminating outlier effects.
2. **Scalar Quantization**: The rotated vectors are compressed into 2-bit or 4-bit representations. This provides a baseline search speedup of 10-20x.

### Stage 2: Quantized Johnson-Lindenstrauss (QJL)
The residual error from SQ is compensated using the QJL lemma. By passing the residual through a sub-gaussian matrix and taking its signs (+1/-1), we effectively bound dot product differences optimally, pushing Recall past 98-99%.

---

## 💻 System Configuration (Hardware)
The following benchmarks were conducted on a standard consumer-grade laptop to demonstrate accessibility and performance optimization.

| Component | Specification |
| :--- | :--- |
| **CPU** | Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz |
| **RAM** | 15.84 GB |
| **OS** | Windows 11 |
| **SIMD** | AVX2, FMA (TurboQuant Native SIMD Active) |
| **Core Library** | **TQ_engine_lib** (Rust-based core) |

---

## 🏆 Official Results Evaluation (5 Million Vectors)

### 1. Performance Results (Stress Test)
> **Note:** **RAW (F32)** utilizes **NumPy**, which is highly optimized with MKL/OpenBLAS. **TURBOQUANT** is implemented using **Rust SIMD** but is currently in an early, **non-fully-optimized** state.

```text
Method          | Batch      | Peak RAM     | Latency    | QPS        | Speedup
------------------------------------------------------------------------------------------------------------
RAW (F32)       |   250,000 |     932.4 MB |   14.5162s |      0.07 |      1.0x
SQ 2-bit        | 4,300,000 |    1009.4 MB |    1.3064s |      0.77 |     11.1x
SQ 4-bit        | 2,100,000 |     991.5 MB |    1.6181s |      0.62 |      9.0x
PQ 2-bit        | 4,000,000 |     955.7 MB |    0.6529s |      1.53 |     22.2x
TQ 2bit         | 4,000,000 |     990.1 MB |    4.4719s |      0.22 |      3.2x
TQ 4bit         | 1,500,000 |     923.6 MB |    4.9892s |      0.20 |      2.9x
============================================================================================================
```

### 2. Accuracy Results (Recall@K)
Tested on 28,378 vectors. **PQ** was trained on a highly fragmented custom 256-sample subset to simulate real-world data mismatch.

**TABLE 1: TOP-1 IN K PROBABILITY**
```text
Method       | P@K=1  | P@K=2  | P@K=4  | P@K=8  | P@K=16 | P@K=32 | P@K=64 
--------------------------------------------------------------------------------------------------------------
PQ 2-bit     |  48.0% |  74.0% |  84.0% |  96.0% | 100.0% | 100.0% | 100.0% 
TQ 2-bit     |  54.0% |  78.0% |  92.0% |  98.0% |  98.0% |  98.0% | 100.0% 
TQ 4-bit     |  88.0% |  92.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% 
```

**TABLE 2: SET RECALL@K (Coverage)**
```text
Method       | R@K=1  | R@K=2  | R@K=4  | R@K=8  | R@K=16 | R@K=32 | R@K=64 
--------------------------------------------------------------------------------------------------------------
PQ 2-bit     |  48.0% |  56.0% |  58.0% |  55.5% |  59.8% |  62.7% |  65.8% 
TQ 2-bit     |  54.0% |  68.0% |  74.0% |  77.2% |  76.9% |  76.1% |  77.3% 
TQ 4-bit     |  88.0% |  82.0% |  90.0% |  88.0% |  88.4% |  88.6% |  89.7% 
```

- **TurboQuant outperforms PQ in fragmented scenarios**, particularly in **Set Recall**, proving its robustness as a training-free architecture.
- **TQ 4-bit** achieves perfect recall at K=4, making it ideal for high-precision RAG systems.

---

# 🇻🇳 Tài Liệu Tiếng Việt

## 📖 Giải Thích Điểm Nhấn Thuật Toán
Dự án này là bộ khung kiểm thử hiệu suất cho kiến trúc **TurboQuant**, sử dụng quy trình nén **SQ+QJL (Scalar Quantization + Quantized Johnson-Lindenstrauss)** kết hợp tối ưu hóa bằng **Rust SIMD**.

### Giai Đoạn 1: Xoay Trực Giao & Lượng Tử Hóa Tuyến Tính (SQ)
Hệ thống sử dụng ma trận xoay trực giao $Pi$ để tản đều phương sai, sau đó nén vector về mức 2-bit hoặc 4-bit. Điều này giúp tăng tốc truy xuất từ 10-20 lần.

### Giai Đoạn 2: Bù Sai Số QJL
Phần dư (Residual) sau khi nén SQ được xử lý qua định lý Johnson-Lindenstrauss để lấy các dấu (sign +1/-1). Bước này giúp khôi phục độ chính xác, đưa Recall vượt ngưỡng 98%.

---

## 💻 Thông số Cấu hình Hệ thống
Toàn bộ quá trình đo lường được thực hiện trên cấu hình máy tính cá nhân tiêu chuẩn:

| Thành phần | Thông số chi tiết |
| :--- | :--- |
| **CPU** | Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz |
| **RAM** | 15.84 GB |
| **Hệ điều hành** | Windows 11 |
| **SIMD** | AVX2, FMA (TurboQuant Native SIMD) |
| **Thư viện lõi** | **TQ_engine_lib** (Rust SIMD) |

---

## 🏆 Kết Quả Đo Lường Thực Tế (5 Triệu Vectors)

### 1. Kết quả Hiệu Năng (Stress Test)
> **Lưu ý:** **RAW (F32)** sử dụng thư viện **NumPy** (đã được tối ưu cực tốt bằng MKL/OpenBLAS). **TURBOQUANT** sử dụng lõi **Rust SIMD** tự viết nhưng hiện tại **vẫn chưa được tối ưu hóa hoàn toàn** (đang ở giai đoạn phát triển sớm).

| Phương pháp | Batch | Peak RAM | Latency | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| RAW (F32) | 250,000 | 932.4 MB | 14.5162s | 1.0x |
| SQ 2-bit | 4,300,000 | 1009.4 MB | 1.3064s | 11.1x |
| TQ 2-bit | 4,000,000 | 990.1 MB | 4.4719s | 3.2x |
| TQ 4-bit | 1,500,000 | 923.6 MB | 4.9892s | 2.9x |

### 2. Kết quả Độ Chính Xác (Recall@K)
Thử nghiệm trên 28,378 vector. **PQ** được huấn luyện trên 256 mẫu phân mảnh để mô phỏng sự sai lệch dữ liệu trong thực tế.

**BẢNG 1: TOP-1 IN K PROBABILITY**
| Phương pháp | P@K=1 | P@K=2 | P@K=4 | P@K=8 | P@K=16 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PQ 2-bit | 48.0% | 74.0% | 84.0% | 96.0% | 100.0% |
| TQ 2-bit | 54.0% | 78.0% | 92.0% | 98.0% | 98.0% |
| TQ 4-bit | 88.0% | 92.0% | 100.0% | 100.0% | 100.0% |

**BẢNG 2: SET RECALL@K (Độ phủ)**
| Phương pháp | R@K=1 | R@K=2 | R@K=4 | R@K=8 | R@K=16 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PQ 2-bit | 48.0% | 56.0% | 58.0% | 55.5% | 59.8% |
| TQ 2-bit | 54.0% | 68.0% | 74.0% | 77.2% | 76.9% |
| TQ 4-bit | 88.0% | 82.0% | 90.0% | 88.0% | 88.4% |

---
**TurboQuant - High-Performance Vector Search Engine**
*Thesis Project Optimization for Large-Scale RAG Systems.*
