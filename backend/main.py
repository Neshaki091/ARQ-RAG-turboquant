from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import psutil

# Tắt log thừa từ Google GenAI SDK
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("langchain_google_genai").setLevel(logging.WARNING)
import os
import json
import time
import numpy as np
from ingest import IngestionManager
from shared.embed import EmbeddingManager
from benchmark import BenchmarkManager
from export_excel import export_to_excel
from shared.supabase_client import SupabaseManager
from shared.vector_store import VectorStoreManager
import httpx

from collections import deque
ui_log_queue = deque(maxlen=100)

class UILogHandler(logging.Handler):
    def emit(self, record):
        # Chỉ lọc duy nhất log /status để tránh tràn giao diện do polling
        if "/status" in str(record.msg): return
        
        # 🟢 YÊU CẦU: Chỉ hiện log stream (ChatService) khi có LỖI (ERROR)
        if record.name == "ChatService" and record.levelno < logging.ERROR:
            return
            
        msg = self.format(record)
        ui_log_queue.append(msg)

ui_handler = UILogHandler()
ui_handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))

# ── Cấu hình Logging cho tất cả model retrieval ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-15s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
# Đảm bảo tất cả các hoạt động chính đều được ghi nhận vào UI
_TARGET_LOGGERS = [
    "RAG-RAW", "RAG-PQ", "RAG-SQ8", "RAG-Adaptive", "ARQ-RAG", 
    "SharedRAG", "VectorStore", "Ingest", "Embedding", "Benchmark", 
    "ChatService", "Supabase", "uvicorn", "uvicorn.access", "httpx"
]
for _logger_name in _TARGET_LOGGERS:
    logger = logging.getLogger(_logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(ui_handler)

class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/status" not in record.getMessage()

app = FastAPI(title="ARQ-RAG Benchmarking API")

@app.on_event("startup")
async def startup_event():
    # Filter out /status logs from uvicorn access logs
    logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Trạng thái hệ thống (Cải tiến) ---

# Global State
state = {
    "status": "IDLE", # IDLE, INGESTING, EMBEDDING, INDEXING, BENCHMARKING, COMPLETED
    "progress": 0,
    "ingest_current": 0,
    "ingest_total": 0,
    "embed_current": 0,
    "embed_total": 0,
    "benchmark_cursor": 0,
    "last_error": None,
    "excel_url": None,
    "last_latency": 0
}

class IngestRequest(BaseModel):
    num_files: int = 5

class PurgeRequest(BaseModel):
    secret_key: str
    target: str = "all" # "all", "vector", "pdf"

class BenchmarkRequest(BaseModel):
    batch_size: int = 10

import psutil

@app.get("/status")
async def get_status():
    process = psutil.Process(os.getpid())
    process_ram_mb = process.memory_info().rss / (1024 * 1024)
    virtual_mem = psutil.virtual_memory()
    sys_ram_mb = virtual_mem.used / (1024 * 1024)
    return {
        **state,
        "ram_usage": round(process_ram_mb, 2),
        "sys_ram_usage": round(sys_ram_mb, 2),
        "logs": list(ui_log_queue)
    }

@app.get("/pdfs")
async def list_pdfs():
    sm = SupabaseManager()
    files = sm.list_files("papers")
    return {"files": files}

@app.post("/run-ingest")
async def run_ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    if state["status"] not in ["IDLE", "COMPLETED"]:
        raise HTTPException(status_code=400, detail="Hệ thống đang bận")
    
    state["status"] = "INGESTING"
    state["progress"] = 0
    
    def process():
        try:
            im = IngestionManager()
            num_processed = im.process_n_files(req.num_files)
            state["status"] = "IDLE"
            state["progress"] = 100
        except Exception as e:
            state["status"] = "IDLE"
            state["last_error"] = str(e)

    background_tasks.add_task(process)
    return {"message": "Bắt đầu trích xuất PDF..."}

@app.post("/run-embed")
async def run_embed(background_tasks: BackgroundTasks):
    if state["status"] not in ["IDLE", "COMPLETED"]:
        raise HTTPException(status_code=400, detail="Hệ thống đang bận")
    
    state["status"] = "EMBEDDING"
    state["progress"] = 0
    
    def process():
        try:
            em = EmbeddingManager()
            chunks, embeddings = em.run_embedding()
            
            # Đã nâng cấp: Sử dụng IngestionManager để đồng bộ hóa mô hình (Modular)
            if chunks and embeddings is not None:
                state["status"] = "INDEXING"
                state["progress"] = 80
                im = IngestionManager()
                im.sync_to_qdrant(chunks, embeddings)
            
            state["status"] = "IDLE"
            state["progress"] = 100
        except Exception as e:
            state["status"] = "IDLE"
            state["last_error"] = str(e)
            logger.error(f"Embed Error: {e}")

    background_tasks.add_task(process)
    return {"message": "Bắt đầu tạo embeddings và đồng bộ hóa Qdrant..."}

@app.post("/purge-data")
async def purge_data(req: PurgeRequest):
    correct_secret = os.getenv("SECRET_KEY", "demo123")
    
    if req.secret_key != correct_secret:
        raise HTTPException(status_code=403, detail="Sai mã Secret Key!")
    
    try:
        target = req.target.lower()
        sm = SupabaseManager()
        vm = VectorStoreManager()

        # --- KHỐI XÓA VECTOR ---
        if target in ["all", "vector"]:
            logger.info("🗑️ Đang xóa Vector Database và các tệp tính toán...")
            vm.delete_all_collections()
            
            files_to_delete = [
                "data/embeddings.npy",
                "data/centroids.npy",
                "data/chunks.json",
                "data/metadata.json"
            ]
            for f in files_to_delete:
                if os.path.exists(f):
                    os.remove(f)
                    logger.info(f"   🗑️ Đã xóa tệp: {f}")
            
            # Reset phần liên quan đến benchmark và embed trong state
            state["embed_current"] = 0
            state["embed_total"] = 0
            state["benchmark_cursor"] = 0
            state["excel_url"] = None

        # --- KHỐI XÓA PDF & RECORDS ---
        if target in ["all", "pdf"]:
            logger.info("🗑️ Đang xóa PDF Storage và Database Records...")
            sm.clear_bucket("papers")
            sm.clear_bucket("benchmark-excel") # Xóa luôn excel cũ vì data gốc đã mất
            sm.clear_database_table("papers")
            sm.clear_database_table("benchmark_queries")
            
            # Xóa metadata local
            metadata_dir = "document/metadata"
            if os.path.exists(metadata_dir):
                import shutil
                shutil.rmtree(metadata_dir)
                os.makedirs(metadata_dir)
                logger.info("   🗑️ Đã dọn sạch metadata địa phương.")

            # Reset phần liên quan đến ingest
            state["ingest_current"] = 0
            state["ingest_total"] = 0
            state["status"] = "IDLE"
            state["progress"] = 0

        # Nếu xóa tất cả, reset toàn bộ state
        if target == "all":
            state.update({
                "status": "IDLE",
                "progress": 0,
                "ingest_current": 0,
                "ingest_total": 0,
                "embed_current": 0,
                "embed_total": 0,
                "benchmark_cursor": 0,
                "last_error": None,
                "excel_url": None,
                "last_latency": 0
            })
            return {"message": "Hệ thống đã được làm sạch HOÀN TOÀN!"}
        
        return {"message": f"Đã hoàn thành dọn dẹp mục tiêu: {target.upper()}"}

    except Exception as e:
        logger.error(f"Purge Error: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa dữ liệu: {str(e)}")

@app.post("/run-benchmark")
async def run_benchmark(req: BenchmarkRequest, background_tasks: BackgroundTasks):
    if state["status"] not in ["IDLE", "COMPLETED"]:
        raise HTTPException(status_code=400, detail="Hệ thống đang bận")
    
    state["status"] = "BENCHMARKING"
    state["progress"] = 0
    
    def process():
        import asyncio
        from chat_service import ChatService
        try:
            # Load Data
            if not os.path.exists("data/chunks.json") or not os.path.exists("data/embeddings.npy"):
                raise Exception("Thiếu dữ liệu: Hãy chạy Ingest và Embed trước.")
            
            with open("data/chunks.json", "r", encoding="utf-8") as f:
                chunks = json.load(f)
            embeddings = np.load("data/embeddings.npy")
            
            bm = BenchmarkManager(embeddings, chunks)
            cs = ChatService()
            
            batch_size = req.batch_size if req else 20
            start_idx = state.get("benchmark_cursor", 0)
            end_idx = start_idx + batch_size
            
            # Since process is synchronous thread run by BackgroundTasks, we need asyncio loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            batch_file, cumulative_file = loop.run_until_complete(bm.run_batch(cs, start_idx=start_idx, end_idx=end_idx))
            
            # Update cursor and state
            state["benchmark_cursor"] = end_idx
            
            # Upload the cumulative results to Supabase
            sm = SupabaseManager()
            filename = f"benchmark_cumulative_{start_idx}_to_{end_idx}_{int(time.time())}.xlsx"
            sm.upload_file("benchmark-excel", filename, cumulative_file)
            state["excel_url"] = sm.get_public_url("benchmark-excel", filename)
            
            state["status"] = "COMPLETED"
            state["progress"] = 100
        except Exception as e:
            state["status"] = "IDLE"
            state["last_error"] = str(e)
            logger.error(f"Benchmark Error: {e}")

    background_tasks.add_task(process)
    return {"message": "Bắt đầu chạy benchmark..."}

from chat_service import ChatService

class ChatRequest(BaseModel):
    query: str
    model: str = "groq"
    collection: str = "vector_arq"


@app.post("/chat-stream")
async def chat_stream(req: ChatRequest):
    try:
        cs = ChatService()
        return StreamingResponse(
            cs.chat_stream(req.query, req.model, req.collection), 
            media_type="application/x-ndjson"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BenchmarkQueryRequest(BaseModel):
    query: str
    model: str = "gemma-4-26b-it"
    collection: str = "vector_arq"
    google_api_key: str = None

@app.post("/api/benchmark/query")
async def benchmark_query(req: BenchmarkQueryRequest):
    """
    Endpoint chuyên dụng cho thực nghiệm phòng thí nghiệm.
    Đo lường chính xác RAM và Latency của tiến trình Docker.
    """
    import os
    import time
    import psutil
    from chat_service import ChatService
    
    # 1. Đo RAM trước khi xử lý
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 * 1024)
    start_time = time.time()
    
    # 2. Tạm thời nạp API Key được chỉ định cho yêu cầu này (Xoay tua)
    original_key = os.getenv("GOOGLE_API_KEY")
    if req.google_api_key:
        os.environ["GOOGLE_API_KEY"] = req.google_api_key
    
    try:
        cs = ChatService()
        # Chuyển đổi stream thành phản hồi đầy đủ
        full_answer = ""
        contexts = []
        
        async for chunk in cs.chat_stream(req.query, "google", req.collection):
            data = json.loads(chunk)
            if data["type"] == "text":
                full_answer += data["content"]
            elif data["type"] == "context":
                contexts = data["content"]
        
        # 3. Đo RAM sau khi xử lý & Latency
        end_time = time.time()
        end_mem = process.memory_info().rss / (1024 * 1024)
        peak_ram = max(0, end_mem - start_mem)
        latency = (end_time - start_time) * 1000
        
        return {
            "answer": full_answer,
            "contexts": contexts,
            "latency_ms": round(latency, 2),
            "peak_ram_mb": round(peak_ram, 2),
            "total_ram_mb": round(end_mem, 2),
            "cpu_percent": psutil.cpu_percent()
        }
        
    except Exception as e:
        logger.error(f"Benchmark Query Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Khôi phục Key ban đầu
        if original_key:
            os.environ["GOOGLE_API_KEY"] = original_key

@app.post("/run-auto-pipeline")
async def run_auto_pipeline(req: IngestRequest, background_tasks: BackgroundTasks):
    if state["status"] not in ["IDLE", "COMPLETED"]:
        raise HTTPException(status_code=400, detail="Hệ thống đang bận")
    
    state["status"] = "INGESTING"
    state["progress"] = 0
    
    def process():
        try:
            # Callbacks
            def on_ingest(curr, total):
                state["ingest_current"] = curr
                state["ingest_total"] = total
            
            def on_embed(curr, total):
                state["embed_current"] = curr
                state["embed_total"] = total

            # Step 1: Ingest/Chunk
            im = IngestionManager()
            state["status"] = "INGESTING"
            state["progress"] = 10
            im.process_n_files(req.num_files, on_progress=on_ingest)
            
            # Step 2: Embed
            state["status"] = "EMBEDDING"
            state["progress"] = 40
            em = EmbeddingManager()
            chunks, embeddings = em.run_embedding(on_progress=on_embed)
            
            # Step 3: Vector Store Ingest (Qdrant) - Modular Sync
            if chunks and embeddings is not None:
                state["status"] = "INDEXING"
                state["progress"] = 70
                im.sync_to_qdrant(chunks, embeddings)
            
            state["status"] = "COMPLETED"
            state["progress"] = 100
        except Exception as e:
            state["status"] = "IDLE"
            state["last_error"] = str(e)
            logger.error(f"Pipeline Error: {e}")

    background_tasks.add_task(process)
    return {"message": "Bắt đầu quy trình tự động (Chunk -> Embed -> Qdrant)..."}

@app.post("/run-generate-testset")
async def run_generate_testset(background_tasks: BackgroundTasks):
    if state["status"] not in ["IDLE", "COMPLETED"]:
        raise HTTPException(status_code=400, detail="Hệ thống đang bận")
    
    state["status"] = "GENERATING_TESTSET"
    state["progress"] = 0
    
    def process():
        import subprocess
        try:
            logger.info("🚀 Đang sinh bộ câu hỏi Ground Truth...")
            result = subprocess.run(["python", "scripts/generate_benchmark_queries.py"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ Đã sinh xong bộ testset nhân tạo!")
                if result.stdout: logger.info(f"Output: {result.stdout}")
            else:
                logger.error(f"❌ Lỗi sinh testset: {result.stderr}")
            state["status"] = "IDLE"
            state["progress"] = 100
        except Exception as e:
            state["status"] = "IDLE"
            state["last_error"] = str(e)
            logger.error(f"Generate Testset error: {e}")

    background_tasks.add_task(process)
    return {"message": "Bắt đầu tạo câu hỏi chuẩn (Ground Truth)..."}

if __name__ == "__main__":
    import uvicorn
    import time
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=True)
