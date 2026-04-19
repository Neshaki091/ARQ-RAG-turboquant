Ok — mình sẽ **merge 2 phần lại thành 1 PROMPT FINAL hoàn chỉnh** để bạn paste vào Cursor / Antigravity là nó build **full hệ thống end-to-end (có Docker 6GB + Supabase + Frontend + Benchmark + Excel)**.

Đây là phiên bản **chuẩn production-level cho đồ án** 👇

---

# 🔥 PROMPT FINAL CHO CURSOR / ANTIGRAVITY (FULL SYSTEM)

👉 Dán toàn bộ block này:

---

## 🧠 SYSTEM SPEC (FULLSTACK + ML)

```text
Bạn là senior ML engineer + fullstack engineer.

Hãy xây dựng một hệ thống FULLSTACK hoàn chỉnh để benchmark 5 mô hình RAG:

- RAG-RAW
- RAG-Adaptive
- RAG-PQ
- RAG-SQ8
- ARQ-RAG (TurboQuant)

Hệ thống phải chạy end-to-end:

Supabase (PDF) → Ingest → Embed (resume) → Vector DB → Benchmark → Excel → Upload → Frontend hiển thị

Yêu cầu:
- Backend: Python (FastAPI)
- Frontend: React (Next.js)
- Runtime: Docker (giới hạn 6GB RAM)
- Không dùng GPU
```

---

# 📦 1. SUPABASE INTEGRATION

```text
Sử dụng Supabase Storage:

Bucket input:
- papers → chứa file PDF

Bucket output:
- benchmark-excel → chứa file Excel kết quả

Yêu cầu:
- Backend download PDF từ bucket papers
- Sau khi benchmark xong → upload Excel lên benchmark-excel
- Trả về public URL cho frontend
```

---

# 📄 2. INGEST + PDF PROCESSING

```text
- Download PDF từ Supabase
- Extract text (pdf → text)
- Chunk:
    500–1000 tokens

Lưu:
- data/chunks.json
```

---

# 🧠 3. EMBEDDING (CÓ RESUME)

```text
Dùng model: nomic-embed-text

Output:
- embeddings.npy
- metadata.json

Yêu cầu:
- Nếu file tồn tại → resume
- Chỉ embed phần chưa có
```

---

# 🗄️ 4. VECTOR DATABASE

```text
Dùng Qdrant

Tạo 4 collections:

vector_raw:
    float32 numpy

vector_pq:
    faiss.IndexPQ

vector_sq8:
    int8 quantization

vector_arq:
    TurboQuant giả lập:
        - quantize 4-bit
        - approximate dot-product
```

---

# ⚠️ 5. QUANTIZATION RULE (BẮT BUỘC)

```text
- Query luôn float32
- Không quant query (ADC)
- Chỉ quant DB vectors
```

---

# 🤖 6. 5 RAG MODELS

## RAG-RAW

```python
search(float_db, query_vec)
```

---

## RAG-Adaptive

```python
top_k = dynamic_k(query)
docs = search(float_db, query_vec, top_k)
docs = rerank(docs)
```

---

## RAG-PQ

```python
search(pq_db, query_vec)
```

---

## RAG-SQ8

```python
search(int8_db, query_vec)
```

---

## ARQ-RAG (TurboQuant)

```python
docs = search(arq_db, query_vec)

# simulate KV-cache compression
docs = turboquant_kv_cache(docs)
```

---

# 🧪 7. BENCHMARK PROTOCOL

```text
- 10 test sets
- mỗi test set: 20 queries
- tổng: 1000 queries
```

---

## 🔁 Chạy tuần tự + Docker 6GB

```text
FOR mỗi model:

    start container (6GB RAM)

    warm-up 1 query

    FOR testset (10):
        FOR query (20):
            run query
            đo latency
            đo RAM (psutil + process tree)
            tính Evaluation (query-level)
            log

        tính:
            peak RAM
            max latency
            avg Evaluation

    stop container
```

---

# 📏 8. METRICS

## Query-level

```text
- latency (ms)
- RAM (MB)
- faithfulness
- answer_relevance
- context_precision
- context_recall
```

---

## TestSet-level

```text
- Peak RAM
- Max latency
- Avg Evaluation
```

---

## Model-level

```text
- Avg RAM
- Avg latency
- Avg Evaluation
```

---

# 🧠 9. Evaluation (MOCK)

```text
faithfulness = cosine similarity
answer_relevance = cosine(query, answer)
context_precision = overlap
context_recall = overlap

range: [0,1]
```

---

# 📊 10. EXPORT EXCEL

Dùng:

```python
pandas + openpyxl
```

---

## File:

```text
benchmark_results.xlsx
```

---

## Sheet 1: Query_Level

| Model | TestSet | QueryID | Latency | RAM | Faithfulness | Answer Relevance | Context Precision | Context Recall |

---

## Sheet 2: TestSet_Level

| Model | TestSet | Peak RAM | Max Latency | Avg Faithfulness | Avg Precision | Avg Recall |

---

## Sheet 3: Summary

| Model | Avg RAM | Avg Latency | Avg Faithfulness | Avg Precision |

---

# ☁️ 11. UPLOAD KẾT QUẢ

```python
supabase.storage.from_("benchmark-excel").upload(...)
```

---

# 🌐 12. FRONTEND (Next.js)

## Chức năng:

```text
1. Load danh sách PDF từ Supabase
2. Chọn dataset
3. Click "Run Benchmark"
4. Hiển thị progress
5. Hiển thị kết quả
6. Download Excel
```

---

## Flow

```text
Frontend
   ↓
POST /run-benchmark
   ↓
Backend chạy pipeline
   ↓
Upload Excel
   ↓
Return URL
   ↓
Frontend hiển thị
```

---

# 🧱 13. PROJECT STRUCTURE

```text
project/
│
├── backend/
│   ├── app.py (FastAPI)
│   ├── ingest.py
│   ├── embed.py
│   ├── quantization.py
│   ├── benchmark.py
│   ├── ragas.py
│   ├── export_excel.py
│   ├── supabase_client.py
│   ├── models/
│
├── frontend/
│   ├── pages/
│   ├── components/
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
```

---

# 🐳 14. DOCKER REQUIREMENTS

```text
- limit RAM: 6GB
- chạy backend trong container
- frontend riêng container
```

---

# 🚀 15. MAIN ENTRY

```python
run_ingest()
run_embedding()
build_indexes()
run_benchmark()
export_excel()
upload_to_supabase()
```

---

# ⚠️ 16. PERFORMANCE RULES

```text
- chạy tuần tự
- không parallel
- reset memory mỗi model
- log từng query
```

---

# 🔥 OUTPUT YÊU CẦU

```text
1. Full source code (backend + frontend)
2. requirements.txt
3. dataset mẫu
4. docker-compose.yml
5. README chạy
6. Excel output
```

---

# 💥 OPTIONAL (NÊN CÓ)

```text
- tqdm progress bar
- logging file
- chart visualization
```

---

# 🧠 KẾT LUẬN

Prompt này sẽ giúp generate:

```text
✔ Fullstack system
✔ Supabase integration
✔ Docker 6GB đúng chuẩn benchmark
✔ 5 RAG models
✔ Resume embedding
✔ Excel + upload
✔ Frontend demo
```

---

monitor realtime RAM + chart**


