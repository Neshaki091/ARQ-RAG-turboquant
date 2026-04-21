"""
main.py
=======
FastAPI Application — Entry point của ARQ-RAG backend.

Endpoints:
  GET  /health          — Health check + collections status
  POST /chat            — Chat (non-streaming, đầy đủ metrics)
  POST /chat/stream     — Chat streaming (SSE, Server-Sent Events)
  GET  /models          — Danh sách models hỗ trợ
  POST /ingest/file     — Upload và ingest 1 PDF file
  GET  /collections     — Trạng thái 5 Qdrant collections
"""

import logging
import os
import uuid
from typing import AsyncIterator, Literal, Optional, Dict, Any

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from chat_service import ChatDispatcher, SUPPORTED_MODELS

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI App ────────────────────────────────────────────────────────
app = FastAPI(
    title="ARQ-RAG TurboQuant API",
    description=(
        "Hệ thống nghiên cứu Vector Quantization trong RAG. "
        "5 model: raw, pq, sq8, arq, adaptive."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — cho phép Next.js frontend kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production: thay bằng domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton Dispatcher ───────────────────────────────────────────────
dispatcher: Optional[ChatDispatcher] = None

def check_required_env():
    """Kiểm tra các biến môi trường bắt buộc."""
    required = [
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "GROQ_API_KEY",
    ]
    missing = [env for env in required if not os.getenv(env)]
    if missing:
        msg = f"❌ Thiếu biến môi trường bắt buộc: {', '.join(missing)}"
        logger.error(msg)
        # Trong thực tế có thể dùng raise SystemExit(1) nếu muốn dừng app hoàn toàn
    return missing

@app.on_event("startup")
async def startup_event():
    global dispatcher
    logger.info("🚀 ARQ-RAG Backend đang khởi động...")
    
    # Kiểm tra env
    missing = check_required_env()
    if missing:
        logger.warning("⚠️  Cảnh báo: Một số tính năng sẽ không hoạt động do thiếu credentials.")

    try:
        dispatcher = ChatDispatcher()
        logger.info("✅ Backend sẵn sàng!")
    except Exception as e:
        logger.error(
            f"⚠️  Không thể khởi tạo ChatDispatcher: {e}\n"
            "Kiểm tra SUPABASE_URL, SUPABASE_KEY trong .env\n"
            "Backend chạy nhưng /chat sẽ trả lỗi cho đến khi credentials được cấu hình."
        )
        # KHÔNG raise — để FastAPI vẫn start, /health vẫn trả lời được


# ── Pydantic Models ────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000, description="Câu hỏi của người dùng")
    model: str = Field(
        default="pq",
        description=f"Tên model: {SUPPORTED_MODELS}",
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=50, description="Số kết quả sau reranking")
    limit: Optional[int] = Field(default=None, ge=1, le=200, description="Số candidates từ Qdrant")
    session_id: Optional[str] = Field(default=None, description="ID phiên chat (UUID)")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Bộ lọc metadata cho Qdrant (ví dụ: {'source': 'book1'})")


class ChatResponse(BaseModel):
    answer: str
    model: str
    session_id: str
    metrics: dict
    sources: list


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Health check — kiểm tra backend và Qdrant collections."""
    if dispatcher is None:
        return {
            "status": "degraded",
            "message": "ChatDispatcher chưa khởi tạo. Kiểm tra SUPABASE_URL và SUPABASE_KEY trong .env",
            "version": "1.0.0",
        }
    try:
        status = dispatcher.get_collections_status()
        return {
            "status": "ok",
            "version": "1.0.0",
            **status,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service không khả dụng: {e}")

@app.get("/version", tags=["System"])
async def get_version():
    """Lấy thông tin phiên bản backend."""
    return {
        "version": "1.0.0",
        "description": "ARQ-RAG TurboQuant API",
        "author": "Neshaki091",
        "status": "live"
    }


@app.get("/models", tags=["System"])
async def list_models():
    """Liệt kê tất cả models được hỗ trợ và mô tả ngắn."""
    return {
        "models": [
            {"name": "raw",      "description": "Float32 baseline — không nén, không reranking"},
            {"name": "pq",       "description": "Product Quantization — ADC reranking, nén 384x"},
            {"name": "sq8",      "description": "Scalar Quantization 8-bit — nén 4x"},
            {"name": "arq",      "description": "TurboQuant — ADC + QJL combined reranking"},
            {"name": "adaptive", "description": "Adaptive RAG — tự điều chỉnh top-k theo query complexity"},
        ],
        "default": "pq",
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """
    Chat non-streaming — trả về full answer + metrics sau khi hoàn thành.
    Phù hợp cho benchmark và so sánh latency.
    """
    if dispatcher is None:
        raise HTTPException(status_code=503, detail="Backend chưa sẵn sàng")

    session_id = req.session_id or str(uuid.uuid4())

    try:
        result = dispatcher.chat(
            query=req.query,
            model=req.model,
            top_k=req.top_k,
            limit=req.limit,
            session_id=session_id,
            filters=req.filters,
        )
        return ChatResponse(
            answer=result["answer"],
            model=result["model"],
            session_id=result.get("session_id", session_id),
            metrics=result.get("metrics", {}),
            sources=result.get("sources", []),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Lỗi chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi server nội bộ")


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(req: ChatRequest):
    """
    Chat streaming — SSE (Server-Sent Events).
    Yield từng token LLM ngay khi có → trải nghiệm real-time.

    Frontend nhận:
      data: <token>\\n\\n   (mỗi chunk)
      data: [DONE]\\n\\n    (khi kết thúc)
    """
    if dispatcher is None:
        raise HTTPException(status_code=503, detail="Backend chưa sẵn sàng")

    session_id = req.session_id or str(uuid.uuid4())

    async def token_generator() -> AsyncIterator[str]:
        try:
            for token in dispatcher.chat_stream(
                query=req.query,
                model=req.model,
                top_k=req.top_k,
                limit=req.limit,
                session_id=session_id,
                filters=req.filters,
            ):
                # SSE format: data: <content>\n\n
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )


@app.get("/collections", tags=["System"])
async def collections_status():
    """Trả về trạng thái 5 Qdrant collections."""
    if dispatcher is None:
        raise HTTPException(status_code=503, detail="Backend chưa sẵn sàng")
    return dispatcher.get_collections_status()


# ── Dev server ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
