# TurboQuant Performance & Accuracy Report (TQ_engine_lib)
**Generated at:** 2026-04-29 02:54:00

## 1. System Configuration
| Component | Specification |
| :--- | :--- |
| **CPU** | Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz |
| **RAM** | 15.84 GB |
| **OS** | Windows 11 |
| **SIMD** | AVX2, FMA (TurboQuant Native SIMD Active - via TQ_engine_lib) |
| **Core Library** | **TQ_engine_lib** |
| **Python** | 3.13.13 |
| **PyTorch** | 2.11.0+cpu |

## 2. Benchmark Parameters
- **Total Vectors:** 5,000,000 (Stress Test)
- **Dimension:** 768
- **Batch Sizes (TQ):** 4M (2bit), 1.5M (4bit)
- **Batch Sizes (SQ):** 4.3M (2bit), 2.1M (4bit)
- **Queries:** 5 queries per iteration (Stress), 50 queries (Recall)

## 3. Performance Results (Stress Test 5M)
```text
TurboQuant Native (Rust SIMD) Active - SQ+QJL Mode

Benchmarking RAW (Float32) - Batch: 250000...
Benchmarking SQ 2-bit - Batch: 4300000...
Benchmarking SQ 4-bit - Batch: 2100000...
Benchmarking PQ 2-bit - Batch: 4000000...
Benchmarking PQ 4-bit - Batch: 1750000...
Benchmarking TQ 2bit - Batch: 4000000...
Benchmarking TQ 4bit - Batch: 1500000...

============================================================================================================
Method          | Batch      | Peak RAM     | Latency    | QPS        | Speedup
------------------------------------------------------------------------------------------------------------
RAW (F32)       |   250,000 |     932.0 MB |   14.4578s |      0.07 |      1.0x
SQ 2-bit        | 4,300,000 |    1009.0 MB |    1.1919s |      0.84 |     12.1x
SQ 4-bit        | 2,100,000 |     991.0 MB |    1.5146s |      0.66 |      9.5x
PQ 2-bit        | 4,000,000 |     955.2 MB |    0.5578s |      1.79 |     25.9x
PQ 4-bit        | 1,750,000 |     865.9 MB |    1.4316s |      0.70 |     10.1x
TQ 2bit         | 4,000,000 |     990.3 MB |    5.3666s |      0.19 |      2.7x
TQ 4bit         | 1,500,000 |     923.8 MB |    6.1341s |      0.16 |      2.4x
============================================================================================================
```

## 4. Accuracy Results (Recall@K)
```text
TurboQuant Native (Rust SIMD) Active - SQ+QJL Mode

===============================================================================================
TURBOQUANT COMPREHENSIVE RECALL: SQ vs PQ vs TQ (via TQ_engine_lib)
===============================================================================================
Dataset: 28378 vectors x 768d | 50 queries

Computing Ground Truth...
  ---- SQ 2-bit...
  PQ 2-bit/dim (M=192, sub_dim=4)...
  ---- TQ 2-bit (SQ+QJL Native)...
  ---- SQ 4-bit...
  PQ 4-bit/dim (M=384, sub_dim=2)...
  ---- TQ 4-bit (SQ+QJL Native)...

=========================================================================================================
Method       | R1@1  | R1@2  | R1@4  | R1@8  | R1@16 | R1@32 | R1@64 |      QPS
---------------------------------------------------------------------------------------------------------
SQ 2-bit     |  2.0% |  8.0% | 14.0% | 20.0% | 28.0% | 32.0% | 50.0% |    302.7
PQ 2-bit     | 74.0% | 92.0% | 98.0% | 100.0% | 100.0% | 100.0% | 100.0% |    290.5
TQ 2-bit     | 54.0% | 78.0% | 92.0% | 98.0% | 98.0% | 98.0% | 100.0% |     18.6
SQ 4-bit     | 82.0% | 94.0% | 98.0% | 100.0% | 100.0% | 100.0% | 100.0% |    267.3
PQ 4-bit     | 94.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |    282.0
TQ 4-bit     | 88.0% | 92.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |     19.1

=========================================================================================================
(*) PQ (Product Quantization) configured to match the same bits per dimension.
(*) Note: PQ search is simulated via full reconstruction in this script.
```

## 5. Execution Summary
- **Stress Test Duration:** 157.00s
- **Recall Test Duration:** 132.78s
- **Total Time:** 289.79s
- **Status:** All tests completed successfully.
