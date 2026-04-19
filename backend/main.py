from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging
import psutil
import os
import json
import time
import numpy as np
import httpx
from collections import deque

from ingest import IngestionManager
from shared.embed import EmbeddingManager
from shared.supabase_client import SupabaseManager
from shared.vector_store import VectorStoreManager

# Tắt log thừa từ Google GenAI SDK
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("langchain_google_genai").setLevel(logging.WARNING)

ui_log_queue = deque(maxlen=100)

class UILogHandler(logging.Handler):
    def emit(self, record):
        if "/status" in str(record.msg): return
        if record.name == "ChatService" and record.levelno < logging.ERROR:
            return
        msg = self.format(record)
        ui_log_queue.append(msg)

ui_handler = UILogHandler()
ui_handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-15s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

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
    logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="https?://.*", # Linh hoạt cho tất cả các domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
state = {
    "status": "IDLE", 
    "progress": 0,
    "ingest_current": 0,
    "ingest_total": 0,
    "embed_current": 0,
    "embed_total": 0,
    "last_error": None,
    "last_latency": 0,
    "benchmark_running": False,
    "benchmark_model": None
}

class IngestRequest(BaseModel):
    num_files: int = 5

class PurgeRequest(BaseModel):
    secret_key: str
    target: str = "all" 

class BenchmarkRequest(BaseModel):
    batch_size: int = 10
    model: str = "vector_arq"

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
            im.process_n_files(req.num_files)
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
            if chunks and embeddings is not None:
                state["status"] = "INDEXING"
                im = IngestionManager()
                im.sync_to_qdrant(chunks, embeddings)
            state["status"] = "IDLE"
            state["progress"] = 100
        except Exception as e:
            state["status"] = "IDLE"
            state["last_error"] = str(e)
    background_tasks.add_task(process)
    return {"message": "Bắt đầu tạo embeddings..."}

@app.post("/purge-data")
async def purge_data(req: PurgeRequest):
    correct_secret = os.getenv("SECRET_KEY", "demo123")
    if req.secret_key != correct_secret:
        raise HTTPException(status_code=403, detail="Sai mã Secret Key!")
    try:
        target = req.target.lower()
        sm = SupabaseManager()
        vm = VectorStoreManager()
        if target in ["all", "vector"]:
            vm.delete_all_collections()
            for f in ["data/embeddings.npy", "data/centroids.npy", "data/chunks.json", "data/metadata.json"]:
                if os.path.exists(f): os.remove(f)
        if target in ["all", "pdf"]:
            sm.clear_bucket("papers")
            sm.clear_database_table("papers")
            sm.clear_database_table("benchmark_queries")
        state.update({"status": "IDLE", "progress": 0})
        return {"message": "Đã dọn dẹp dữ liệu!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/benchmark/run-test")
async def run_benchmark_ui(req: BenchmarkRequest, background_tasks: BackgroundTasks):
    if state["benchmark_running"]:
        raise HTTPException(status_code=400, detail="Hệ thống đang bận")
    state["status"] = "BENCHMARKING"
    state["benchmark_running"] = True
    state["benchmark_model"] = req.model
    state["progress"] = 0
    def process():
        try:
            sm = SupabaseManager()
            queries = sm.get_benchmark_queries()
            if not queries: raise Exception("Không tìm thấy đề thi.")
            test_subset = queries[:req.batch_size]
            for i, q in enumerate(test_subset):
                if not state["benchmark_running"]: break
                with httpx.Client(timeout=120.0) as client:
                    resp = client.post("http://localhost:8000/api/benchmark/query", 
                                     json={"query": q["question"], "collection": req.model})
                    if resp.status_code == 200:
                        res = resp.json()
                        sm.supabase.table("benchmarks").insert({
                            "model_name": req.model, "question": q["question"],
                            "answer": res["answer"], "contexts": res["contexts"],
                            "ground_truth": q.get("ground_truth", ""),
                            "latency_ms": res["latency_ms"], "peak_ram_mb": res["peak_ram_mb"],
                            "topic": q.get("topic", "General")
                        }).execute()
                state["progress"] = int(((i+1)/len(test_subset))*100)
            state["status"] = "COMPLETED"
        except Exception as e:
            state["status"] = "IDLE"
            state["last_error"] = str(e)
        finally:
            state["benchmark_running"] = False
    background_tasks.add_task(process)
    return {"message": "Bắt đầu benchmark..."}

@app.get("/api/benchmark/history")
async def get_benchmark_history(model: str = "all"):
    sm = SupabaseManager()
    try:
        query = sm.supabase.table("benchmarks").select("*").order("created_at", desc=True)
        if model != "all": query = query.eq("model_name", model)
        res = query.limit(100).execute()
        return {"results": res.data or []}
    except: return {"results": []}

@app.delete("/api/benchmark/clear")
async def clear_benchmark_history(model: str = "all"):
    sm = SupabaseManager()
    try:
        query = sm.supabase.table("benchmarks").delete()
        if model != "all": query = query.eq("model_name", model)
        else: query = query.neq("id", -1)
        query.execute()
        return {"message": "Đã xóa lịch sử."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from chat_service import ChatService
class ChatRequest(BaseModel):
    query: str
    model: str = "groq"
    collection: str = "vector_arq"

@app.post("/chat-stream")
async def chat_stream(req: ChatRequest):
    try:
        cs = ChatService()
        return StreamingResponse(cs.chat_stream(req.query, req.model, req.collection), media_type="application/x-ndjson")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BenchmarkQueryRequest(BaseModel):
    query: str
    collection: str = "vector_arq"

@app.post("/api/benchmark/query")
async def benchmark_query(req: BenchmarkQueryRequest):
    import time
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 * 1024)
    start_time = time.time()
    try:
        cs = ChatService()
        full_answer = ""
        contexts = []
        async for chunk in cs.chat_stream(req.query, "google", req.collection):
            data = json.loads(chunk)
            if data["type"] == "text": full_answer += data["content"]
            elif data["type"] == "context": contexts = data["content"]
        latency = (time.time() - start_time) * 1000
        peak_ram = max(0, (process.memory_info().rss / (1024 * 1024)) - start_mem)
        return {
            "answer": full_answer, "contexts": contexts,
            "latency_ms": round(latency, 2), "peak_ram_mb": round(peak_ram, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run-auto-pipeline")
async def run_auto_pipeline(req: IngestRequest, background_tasks: BackgroundTasks):
    state["status"] = "INGESTING"
    state["progress"] = 0
    def process():
        try:
            IngestionManager().process_n_files(req.num_files)
            chunks, embeddings = EmbeddingManager().run_embedding()
            if chunks and embeddings is not None:
                IngestionManager().sync_to_qdrant(chunks, embeddings)
            state["status"] = "COMPLETED"
            state["progress"] = 100
        except Exception as e:
            state["status"] = "IDLE"
            state["last_error"] = str(e)
    background_tasks.add_task(process)
    return {"message": "Bắt đầu pipeline..."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
