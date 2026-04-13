from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
import json
import numpy as np
from ingest import IngestionManager
from embed import EmbeddingManager
from benchmark import BenchmarkManager
from export_excel import export_to_excel
from supabase_client import SupabaseManager
from vector_store import VectorStoreManager

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

# Global State
state = {
    "status": "IDLE", # IDLE, INGESTING, EMBEDDING, INDEXING, BENCHMARKING, COMPLETED
    "progress": 0,
    "ingest_current": 0,
    "ingest_total": 0,
    "embed_current": 0,
    "embed_total": 0,
    "last_error": None,
    "excel_url": None
}

class IngestRequest(BaseModel):
    num_files: int = 5

class PurgeRequest(BaseModel):
    secret_key: str

@app.get("/status")
async def get_status():
    return state

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
            
            # Đã thêm: Đồng bộ hóa với Qdrant ngay sau khi nhúng thành công
            if chunks and embeddings is not None:
                state["status"] = "INDEXING"
                state["progress"] = 80
                vm = VectorStoreManager()
                vm.initialize_collections()
                vm.upsert_data(chunks, embeddings)
            
            state["status"] = "IDLE"
            state["progress"] = 100
        except Exception as e:
            state["status"] = "IDLE"
            state["last_error"] = str(e)
            print(f"Embed Error: {e}")

    background_tasks.add_task(process)
    return {"message": "Bắt đầu tạo embeddings và đồng bộ hóa Qdrant..."}

@app.post("/purge-data")
async def purge_data(req: PurgeRequest):
    # Lấy secret từ env (khớp với tệp .env của user)
    correct_secret = os.getenv("SECRET_KEY", "demo123")
    
    if req.secret_key != correct_secret:
        raise HTTPException(status_code=403, detail="Sai mã Secret Key!")
    
    try:
        # 1. Xóa Qdrant collections
        vm = VectorStoreManager()
        vm.delete_all_collections()
        
        # 2. Xóa các tệp data phục vụ tính toán (giữ lại chunks và metadata)
        files_to_delete = [
            "data/embeddings.npy",
            "data/centroids.npy"
        ]
        
        for f in files_to_delete:
            if os.path.exists(f):
                os.remove(f)
                print(f"Đã xóa tệp: {f}")
        
        # 3. Reset state
        global state
        state = {
            "status": "IDLE",
            "progress": 0,
            "ingest_current": 0,
            "ingest_total": 0,
            "embed_current": 0,
            "embed_total": 0,
            "last_error": None,
            "excel_url": None
        }
        
        return {"message": "Hệ thống đã được làm sạch hoàn toàn!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa dữ liệu: {str(e)}")

@app.post("/run-benchmark")
async def run_benchmark(background_tasks: BackgroundTasks):
    if state["status"] not in ["IDLE", "COMPLETED"]:
        raise HTTPException(status_code=400, detail="Hệ thống đang bận")
    
    state["status"] = "BENCHMARKING"
    state["progress"] = 0
    
    def process():
        try:
            # Load Data
            if not os.path.exists("data/chunks.json") or not os.path.exists("data/embeddings.npy"):
                raise Exception("Thiếu dữ liệu: Hãy chạy Ingest và Embed trước.")
            
            with open("data/chunks.json", "r", encoding="utf-8") as f:
                chunks = json.load(f)
            embeddings = np.load("data/embeddings.npy")
            
            bm = BenchmarkManager(embeddings, chunks)
            results = bm.run_benchmark(num_test_sets=2, queries_per_set=5) # Giảm số lượng để demo nhanh
            
            # Export
            file_path = export_to_excel(results)
            
            # Upload to Supabase
            sm = SupabaseManager()
            filename = f"benchmark_{int(time.time())}.xlsx"
            sm.upload_file("benchmark-excel", filename, file_path)
            state["excel_url"] = sm.get_public_url("benchmark-excel", filename)
            
            state["status"] = "COMPLETED"
            state["progress"] = 100
        except Exception as e:
            state["status"] = "IDLE"
            state["last_error"] = str(e)

    background_tasks.add_task(process)
    return {"message": "Bắt đầu chạy benchmark..."}

from chat_service import ChatService

class ChatRequest(BaseModel):
    query: str
    model: str = "gemini"
    collection: str = "vector_arq"

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        cs = ChatService()
        result = await cs.chat(req.query, req.model, req.collection)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
            
            # Step 3: Vector Store Ingest (Qdrant)
            if chunks and embeddings is not None:
                state["status"] = "INDEXING"
                state["progress"] = 70
                vm = VectorStoreManager()
                vm.initialize_collections()
                vm.upsert_data(chunks, embeddings)
            
            state["status"] = "COMPLETED"
            state["progress"] = 100
        except Exception as e:
            state["status"] = "IDLE"
            state["last_error"] = str(e)
            print(f"Pipeline Error: {e}")

    background_tasks.add_task(process)
    return {"message": "Bắt đầu quy trình tự động (Chunk -> Embed -> Qdrant)..."}

if __name__ == "__main__":
    import uvicorn
    import time # Cần time trong process()
    # Để hiện log các endpoint khác, giữ access_log=True (mặc định)
    # Lọc log /status được thực hiện trong startup_event()
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=True)
