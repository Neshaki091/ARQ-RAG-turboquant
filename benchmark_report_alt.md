# TurboQuant Performance & Accuracy Report (TQ_engine_lib)
**Generated at:** 2026-04-29 03:50:50

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
- **PQ Configuration:** Custom training (10@start, 50@1k, 100@10k, 50@13k, 46@end | 20 iterations). M=96 (2b), M=192 (4b).

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
RAW (F32)       |   250,000 |     932.4 MB |   14.5162s |      0.07 |      1.0x
SQ 2-bit        | 4,300,000 |    1009.4 MB |    1.3064s |      0.77 |     11.1x
SQ 4-bit        | 2,100,000 |     991.5 MB |    1.6181s |      0.62 |      9.0x
PQ 2-bit        | 4,000,000 |     955.7 MB |    0.6529s |      1.53 |     22.2x
PQ 4-bit        | 1,750,000 |     866.0 MB |    1.6191s |      0.62 |      9.0x
TQ 2bit         | 4,000,000 |     990.1 MB |    4.4719s |      0.22 |      3.2x
TQ 4bit         | 1,500,000 |     923.6 MB |    4.9892s |      0.20 |      2.9x
============================================================================================================
```

## 4. Accuracy Results (Recall@K)
```text
TurboQuant Native (Rust SIMD) Active - SQ+QJL Mode

===============================================================================================
TURBOQUANT COMPREHENSIVE RECALL: SQ vs PQ vs TQ (via TQ_engine_lib)
===============================================================================================
Config: PQ trained on Highly Fragmented 256 samples (10@start, 50@1k, 100@10k, 50@13k, 46@end).
===============================================================================================
Dataset: 28378 vectors x 768d | 50 queries

Computing Ground Truth...
  ---- SQ 2-bit...
  PQ 2-bit/dim (M=96, sub_dim=8 | Highly Fragmented Training: 256 samples, 20 iter)...
  ---- TQ 2-bit (SQ+QJL Native)...
  ---- SQ 4-bit...
  PQ 4-bit/dim (M=192, sub_dim=4 | Highly Fragmented Training: 256 samples, 20 iter)...
  ---- TQ 4-bit (SQ+QJL Native)...

==============================================================================================================
TABLE 1: TOP-1 IN K PROBABILITY (Is the true best result within predicted Top-K?)
==============================================================================================================
Method       | P@K=1  | P@K=2  | P@K=4  | P@K=8  | P@K=16 | P@K=32 | P@K=64 |  
--------------------------------------------------------------------------------------------------------------
SQ 2-bit     |   2.0% |   8.0% |  14.0% |  20.0% |  28.0% |  32.0% |  50.0% |   
PQ 2-bit     |  48.0% |  74.0% |  84.0% |  96.0% | 100.0% | 100.0% | 100.0% |    
TQ 2-bit     |  54.0% |  78.0% |  92.0% |  98.0% |  98.0% |  98.0% | 100.0% |    
SQ 4-bit     |  82.0% |  94.0% |  98.0% | 100.0% | 100.0% | 100.0% | 100.0% |    
PQ 4-bit     |  78.0% |  90.0% |  96.0% |  98.0% | 100.0% | 100.0% | 100.0% |    
TQ 4-bit     |  88.0% |  92.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |     

==============================================================================================================
TABLE 2: SET RECALL@K (Percentage of actual Top-K items found in predicted Top-K)
==============================================================================================================
Method       | R@K=1  | R@K=2  | R@K=4  | R@K=8  | R@K=16 | R@K=32 | R@K=64 |      
--------------------------------------------------------------------------------------------------------------
SQ 2-bit     |   2.0% |   6.0% |   8.5% |  10.0% |  12.4% |  13.4% |  14.9% |   
PQ 2-bit     |  48.0% |  56.0% |  58.0% |  55.5% |  59.8% |  62.7% |  65.8% |    
TQ 2-bit     |  54.0% |  68.0% |  74.0% |  77.2% |  76.9% |  76.1% |  77.3% |     
SQ 4-bit     |  82.0% |  76.0% |  79.5% |  77.2% |  77.9% |  79.9% |  81.6% |    
PQ 4-bit     |  78.0% |  79.0% |  80.0% |  76.2% |  78.2% |  78.6% |  80.8% |    
TQ 4-bit     |  88.0% |  82.0% |  90.0% |  88.0% |  88.4% |  88.6% |  89.7% |     

==============================================================================================================
(*) PQ trained on custom fragmented 256 samples (10@start, 50@1k, 100@10k, 50@13k, 46@end).
(*) PQ configuration: M=96 for 2-bit, M=192 for 4-bit.
```

## 5. Execution Summary
- **Stress Test Duration:** 90.15s
- **Recall Test Duration:** 28.82s
- **Total Time:** 118.97s
- **Status:** All tests completed successfully.
