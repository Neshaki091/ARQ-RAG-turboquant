# ARQ-RAG TurboQuant

> **Hệ thống nghiên cứu chuyên sâu** về Vector Quantization trong RAG — cân bằng tuyệt đối giữa **Accuracy** và **Latency**.

## Kiến trúc 5 Model

| Model | Thuật toán | Mô tả |
|-------|-----------|-------|
| `raw` | Float32 | Baseline chuẩn, độ chính xác cao nhất |
| `pq` | **Product Quantization** | Nén 384x, ADC reranking siêu nhanh |
| `sq8` | Scalar Quantization 8-bit | Nén 4x, đơn giản, ổn định |
| `arq` | ADC + QJL TurboQuant | Reranking nâng cao với Johnson-Lindenstrauss |
| `adaptive` | Dynamic Top-K | Tự điều chỉnh K dựa theo độ phức tạp câu hỏi |

## Tech Stack

- **Backend**: FastAPI + Qdrant + Ollama (`nomic-embed-text-v1.5`) + Groq LLM
- **Frontend**: Next.js 15 (App Router) + Tailwind CSS 4
- **Infra**: Docker Compose + Cloudflare Tunnel
- **PQ Core**: faiss-cpu K-Means + numpy ADC lookup table

## Các endpoints chính

- `GET /health` : Trạng thái hệ thống và Qdrant
- `GET /version` : Thông tin phiên bản (v1.0.0)
- `GET /models` : Danh sách 5 models hỗ trợ
- `POST /chat` : Xử lý chat cơ bản (kèm benchmark metrics)
- `POST /chat/stream` : Chat streaming qua SSE theo thời gian thực

## Cài đặt

```bash
# 1. Clone và cấu hình môi trường
cp .env.example .env
# Điền đầy đủ SUPABASE_URL, GROQ_API_KEY, v.v.

# 2. Khởi động hạ tầng
docker compose up -d

# 3. Ingest dữ liệu
docker compose exec backend python scripts/ingest_local.py --pdf-dir ./data/pdfs

# 4. Huấn luyện Product Quantizer
docker compose exec backend python scripts/train_pq.py

# 5. Chạy frontend
cd frontend && npm install && npm run dev
```

## Thuật toán RAG-PQ (Product Quantization)

```
Query (Float32, 768-dim)
  ↓ chia M=8 sub-vectors (96-dim)
  ↓ compute distance đến K=256 centroids mỗi sub-space
  → ADC Table (8×256) — tính 1 lần cho mỗi query

Stored Vectors (PQ-encoded: 8 bytes/vector thay vì 3072 bytes)
  ↓ tra ADC table theo centroid index
  → approximate distance trong O(M) thay vì O(D)
  → Tiết kiệm 384x bộ nhớ, tốc độ reranking cực nhanh
```
