CHƯƠNG 4: QUY TRÌNH THỰC NGHIỆM KIỂM CHỨNG

# 1\. BẢNG QUY ƯỚC XÂY DỰNG 5 MÔ HÌNH (FINAL)

## Kiến trúc & Quantization

| **Thành phần** | **RAG-RAW** | **RAG-Adaptive** | **RAG-PQ** | **RAG-SQ8** | **ARQ-RAG (TurboQuant)** |
| -------------- | ----------- | ---------------- | ---------- | ----------- | ------------------------ |
| Vector DB      | Float32     | Float32          | PQ         | Int8        | TurboQuant (≈3-4 bit)    |
| Collection     | vector_raw  | vector_raw       | vector_pq  | vector_sq8  | vector_arq               |
| Query vector   | Float32     | Float32          | Float32    | Float32     | Float32                  |
| Quant DB       | ❌          | ❌               | ✅         | ✅          | ✅                       |
| Quant Query    | ❌          | ❌               | ❌         | ❌          | ❌ (**ADC**)             |
| KV-cache       | ❌          | ❌               | ❌         | ❌          | ✅ (TurboQuant)          |
| Adaptive Top-K | ❌          | ✅               | ❌         | ❌          | ✅                       |
| Reranking      | ❌          | ✅               | ❌         | ❌          | ✅                       |
| Search space   | Float       | Float            | PQ space   | Int8 space  | Quant space (ADC)        |

# 2\. QUY ƯỚC TRIỂN KHAI CHUNG (BẮT BUỘC)

## 🔹 Dataset & Embedding (shared)

docs = load_dataset()

embeddings = embed_all(docs) # dùng chung cho 5 model

👉 Quy định:

- ✔ 1 dataset duy nhất
- ✔ 1 embedding model duy nhất
- ✔ Không re-embed theo từng model

## 🔹 Collections (chuẩn)

vector_raw → Float32 (RAW, Adaptive)

vector_pq → PQ encoded

vector_sq8 → Int8

vector_arq → TurboQuant

# 3\. PIPELINE CHUẨN TỪNG MODEL

## RAG-RAW

Query → Embed → Search(Float32) → LLM

## RAG-Adaptive

Query → Analyze → Dynamic Top-K → Search → Rerank → LLM

## RAG-PQ (ADC)

Query(Float32) → Search (q · PQ(x)) → LLM

## RAG-SQ8 (ADC)

Query(Float32) → Search (q · Q_int8(x)) →LLM

## ARQ-RAG (TurboQuant FULL)

┌──────────────┐

Query ────────▶ │ Embedding │ (float)

└──────┬───────┘

↓

┌──────────────┐

│ Adaptive │

│ Routing │

└──────┬───────┘

↓

┌────────────────────────────┐

│ Search (ADC) │

│ q · TQ(x) │

└────────────┬───────────────┘

↓

Retrieve documents

↓

┌────────────────────────────┐

│ TurboQuant KV-cache │

│ (includes QJL) │

└────────────┬───────────────┘

↓

LLM

# 4\. QUY ƯỚC QUANTIZATION (CHUẨN HỌC THUẬT)

## Asymmetric Distance Computation (ADC)

score(q, x) ≈ q · Q(x)

| **Thành phần**  | **Trạng thái** |
| --------------- | -------------- |
| Query           | Float          |
| Database vector | Quantized      |

## TurboQuant

TurboQuant = Scalar Quantization + QJL (residual)

👉 Đặc điểm:

- Unbiased inner product estimator
- Nén: ~3-4 bit
- Dùng cho:
  - Vector DB
  - KV-cache

# 5\. QUY ƯỚC ĐẦU RA BENCHMARK

## 5.1 System Metrics

| **Metric**       | **Mô tả**                         | **Cách đo** |
| ---------------- | --------------------------------- | ----------- |
| **RAM (MB)**     | Peak memory usage (DB + KV-cache) | psutil      |
| **Storage (MB)** | Kích thước collection             | file size   |
| **Latency (ms)** | Query → Answer                    | time        |

### Code chuẩn

import psutil, time

start = time.time()

\# run query

latency = (time.time() - start) \* 1000

memory = psutil.Process().memory_info().rss / 1024\*\*2

## 5.2 Evaluation Metrics

<div class="joplin-table-wrapper"><table><tbody><tr><th><p><strong>Metric</strong></p></th><th><p><strong>Ý nghĩa</strong></p></th><th rowspan="5"><h2>Output chuẩn (JSON)</h2><p>{</p><p>"model": "ARQ-RAG",</p><p>"ram_mb": 780,</p><p>"storage_mb": 110,</p><p>"latency_ms": 52,</p><p>"ragas": {</p><p>"faithfulness": 0.93,</p><p>"answer_relevance": 0.90,</p><p>"context_precision": 0.92,</p><p>"context_recall": 0.88</p><p>}}</p></th></tr><tr><td><p>Faithfulness</p></td><td><p>Không hallucination</p></td></tr><tr><td><p>Answer Relevance</p></td><td><p>Đúng câu hỏi</p></td></tr><tr><td><p>Context Precision</p></td><td><p>Lấy đúng chunk</p></td></tr><tr><td><p>Context Recall</p></td><td><p>Lấy đủ thông tin</p></td></tr></tbody></table></div>

# 6\. BẢNG SO SÁNH KẾT QUẢ CUỐI

| **Model** | **RAM ↓**   | **Storage ↓** | **Latency ↓** | **Faithfulness ↑** | **Precision ↑** |
| --------- | ----------- | ------------- | ------------- | ------------------ | --------------- |
| RAW       | ❌ cao      | ❌ cao        | ❌ cao        | ✅ cao             | ✅ cao          |
| Adaptive  | ❌ cao      | ❌ cao        | ❌ rất cao    | ✅ rất cao         | ✅ rất cao      |
| PQ        | ✅ thấp     | ✅ thấp       | ✅ nhanh      | ❌ giảm            | ❌ giảm         |
| SQ8       | ✅ trung    | ✅ trung      | ✅ nhanh      | ⚠️ nhẹ             | ⚠️ nhẹ          |
| ARQ       | ✅ rất thấp | ✅ rất thấp   | ⚠️ trung      | ✅ cao             | ✅ cao          |

# 7\. QUY ƯỚC BENCHMARK (ĐỂ KHÔNG BỊ SAI)

✔ Cùng dataset  
✔ Cùng embedding  
✔ Cùng query set (~500 câu)  
✔ Chạy tuần tự (tránh OOM 16GB)  
✔ Reset memory mỗi model  
✔ Không thay đổi hyperparameter giữa model

# KẾT LUẬN CUỐI

👉 Pipeline chuẩn của bạn:

RAW → Adaptive → PQ/SQ8 → ARQ (TurboQuant full)

👉 Và điểm mấu chốt:

\- Không quant query (ADC)

\- TurboQuant = DB + KV-cache

\- QJL nằm trong TurboQuant

1\. MỤC TIÊU THỰC NGHIỆM

Đánh giá 5 mô hình:

RAG-RAW  
RAG-Adaptive  
RAG-PQ  
RAG-SQ8  
ARQ-RAG (TurboQuant)

Theo 2 nhóm:

- ⚙️ **System Performance**: RAM, Storage, Latency
- 🧠 **Quality**: Evaluation

# 🧠 2. QUY ƯỚC DỮ LIỆU

| **Thành phần**   | **Quy ước**                  |
| ---------------- | ---------------------------- |
| Dataset          | Cố định (vd: 1000 documents) |
| Embedding model  | Dùng chung cho tất cả        |
| Vector dimension | Cố định                      |
| Không re-embed   | ✔ bắt buộc                   |

# 🧪 3. QUY ƯỚC TEST

## 📦 Cấu trúc test

50 Test Sets  
mỗi Test Set = 50 runs (10 queries x 5 models)

👉 Tổng:

500 queries / model  
2500 queries toàn hệ thống

## 🔹 Yêu cầu test

- Câu hỏi đa dạng (dễ / trung bình / khó)
- Có ground truth (để tính Evaluation)
- Dùng **cùng bộ câu hỏi cho tất cả model** (Tổng 500 câu)

# ⚙️ 4. QUY TRÌNH CHẠY THỰC NGHIỆM

FOR mỗi batch (50):  
    FOR mỗi model (5):  
        load model + database  
        FOR mỗi unique query (10):  
            run query  
            đo latency  
            đo RAM  
            tính Evaluation (query-level)  
            lưu kết quả  
            đợi 75s (TPM Safe)
        unload model (giải phóng RAM)

# 📊 5. QUY ƯỚC ĐO METRICS

## 🔥 5.1 Query-level (QUAN TRỌNG NHẤT)

👉 Mỗi query phải lưu:

| **Metric**        |
| ----------------- |
| Latency (ms)      |
| RAM (MB)          |
| Faithfulness      |
| Answer Relevance  |
| Context Precision |
| Context Recall    |

## 🔥 5.2 TestSet-level

Từ 20 queries:

| **Metric**  | **Cách tính** |
| ----------- | ------------- |
| Peak RAM    | max(RAM)      |
| Max Latency | max(latency)  |
| Evaluation       | trung bình    |

## 🔥 5.3 Batch-level (MỚI)
- 10 queries x 5 models = 50 runs.
- Ước tính thời gian: ~85 phút/batch (với delay 75s).
- Tổng số batch cần chạy: 50 batches.

## 🔥 5.3 Model-level

Từ 10 test sets:

| **Metric**  | **Cách tính** |
| ----------- | ------------- |
| Avg RAM     | trung bình    |
| Avg Latency | trung bình    |
| Avg Evaluation   | trung bình    |

# 📦 6. OUTPUT FILE (EXCEL)

## 📄 Sheet 1: Query_Level (1000 dòng)

| Model | TestSet | QueryID | Latency | RAM | Faithfulness | Answer Relevance | Context Precision | Context Recall |

## 📄 Sheet 2: TestSet_Level (50 dòng)

| Model | TestSet | Peak RAM | Max Latency | Avg Faithfulness | Avg Precision | Avg Recall |

## 📄 Sheet 3: Summary

| Model | Avg RAM | Avg Latency | Avg Faithfulness | Avg Precision |

# ⚠️ 7. QUY ƯỚC QUANTIZATION (PHẢI ĐÚNG)

## 🔑 ADC (bắt buộc)

Query: Float32  
Database: Quantized

👉 Không quant query

## 🔑 TurboQuant

- Dùng cho:
  - Vector DB
  - KV-cache
- Không áp dụng cho query
- Bao gồm QJL bên trong

# 🚨 8. QUY TẮC ĐẢM BẢO CÔNG BẰNG

## ❌ Không được

- Thay đổi dataset giữa các model
- Thay đổi embedding
- Chạy song song (RAM sai)
- Bỏ qua query-level Evaluation

## ✅ Bắt buộc

- Chạy **tuần tự từng model**
- Reset RAM sau mỗi model
- Cùng query cho tất cả model
- Log đầy đủ từng query