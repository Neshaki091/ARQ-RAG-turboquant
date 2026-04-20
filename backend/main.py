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
import threading

def sanitize_string(s):
    """Xóa bỏ các ký tự null (\u0000) có thể gây lỗi cho Postgres/Supabase."""
    if isinstance(s, str):
        return s.replace("\u0000", "").replace("\x00", "")
    return s

from shared.supabase_client import SupabaseManager
from shared.vector_store import VectorStoreManager

# Tắt log thừa từ Google GenAI SDK
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("langchain_google_genai").setLevel(logging.WARNING)

# Ẩn log httpx spam (polling liên tục, không có giá trị debug)
_SUPPRESS_PATTERNS = ("storage/v1/object/list", "rest/v1/benchmarks?select")
class _SuppressHttpxSpam(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not any(p in msg for p in _SUPPRESS_PATTERNS)

logging.getLogger("httpx").addFilter(_SuppressHttpxSpam())

# Ẩn log uvicorn access cho các endpoint polling liên tục
_SUPPRESS_ROUTES = ("/api/benchmark/history", "/pdfs", "/api/system/metrics")
class _SuppressUvicornPolling(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not any(r in msg for r in _SUPPRESS_ROUTES)

logging.getLogger("uvicorn.access").addFilter(_SuppressUvicornPolling())

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
    "RAG-RAW", "RAG-Adaptive", "ARQ-RAG", 
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

class MemoryTracker:
    def __init__(self, pid: int):
        self.process = psutil.Process(pid)
        self.peak_mem = 0
        self.start_mem = self.process.memory_info().rss / (1024 * 1024)
        self._stop_event = threading.Event()  # Dùng Event để báo hiệu dừng thread an toàn

    def track(self):
        # Chạy đến khi nhận được tín hiệu dừng từ stop_event
        while not self._stop_event.is_set():
            try:
                curr = self.process.memory_info().rss / (1024 * 1024)
                if curr > self.peak_mem:
                    self.peak_mem = curr
                self._stop_event.wait(timeout=0.05)  # Chờ 50ms hoặc đến khi nhận tín hiệu
            except Exception:
                break

    def stop(self):
        """Gửi tín hiệu cho thread giám sát biết cần dừng lại."""
        self._stop_event.set()

    @property
    def peak_delta_mb(self) -> float:
        """Trả về lượng RAM tăng thêm so với thời điểm bắt đầu (MB)."""
        return max(0.0, self.peak_mem - self.start_mem)

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
    "last_error": None,
    "last_latency": 0,
    "benchmark_running": False,
    "benchmark_model": None
}


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
    
    # Lấy trạng thái của Native Engine (Cache info)
    from shared.native_engine import NativeEngine
    engine = NativeEngine()
    engine_status = {
        "current_group": engine.current_group,
        "is_cached": len(engine.cache) > 0,
        "num_points": len(engine.cache.get("ids", []))
    }

    return {
        **state,
        "ram_usage": round(process_ram_mb, 2),
        "sys_ram_usage": round(sys_ram_mb, 2),
        "engine": engine_status,
        "logs": list(ui_log_queue)
    }

@app.get("/pdfs")
async def list_pdfs():
    sm = SupabaseManager()
    files = sm.list_files("papers")
    return {"files": files}


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
        PROGRESS_FILE = "backend/data/benchmark_progress.json"
        
        # Load progress file (tạo mới nếu chưa có)
        try:
            os.makedirs("backend/data", exist_ok=True)
            if os.path.exists(PROGRESS_FILE):
                with open(PROGRESS_FILE, "r") as f:
                    progress_data = json.load(f)
            else:
                progress_data = {}
        except Exception:
            progress_data = {}

        done_ids = set(progress_data.get(req.model, []))
        print(f"[Benchmark] Model={req.model} | Đã chạy: {len(done_ids)} câu | File: {PROGRESS_FILE}")

        try:
            sm = SupabaseManager()
            queries = sm.get_benchmark_queries()
            if not queries: raise Exception("Không tìm thấy đề thi.")

            # Lọc bỏ câu đã chạy (dùng _id)
            remaining = [q for q in queries if q.get("_id") not in done_ids]
            test_subset = remaining[:req.batch_size]
            print(f"[Benchmark] Tổng={len(queries)} | Còn lại={len(remaining)} | Batch={len(test_subset)}")

            for i, q in enumerate(test_subset):
                if not state["benchmark_running"]: break
                
                with httpx.Client(timeout=180.0) as client:
                    resp = client.post("http://localhost:8000/api/benchmark/query", 
                                     json={"query": q["question"], "collection": req.model})
                    
                    if resp.status_code != 200:
                        raise Exception(f"Benchmark Failed: Query '{q['question'][:50]}...' trả về lỗi {resp.status_code}")
                        
                    res = resp.json()
                    raw_ctx = res.get("contexts", [])
                    contexts_val = [c[:500] for c in raw_ctx[:3]] if isinstance(raw_ctx, list) else []

                    try:
                        sm.supabase.table("benchmarks").insert({
                            "model_name": req.model,
                            "question": sanitize_string(q["question"]),
                            "answer": sanitize_string(res["answer"]),
                            "contexts": [sanitize_string(c) for c in contexts_val],
                            "ground_truth": sanitize_string(q.get("ground_truth", "")),
                            "latency_ms": res["latency_ms"],
                            "base_ram_mb": res["base_ram_mb"],
                            "peak_ram_mb": res["peak_ram_mb"],
                            "total_ram_mb": res["total_ram_mb"],
                            "retrieval_latency_ms": res.get("retrieval_latency_ms", 0),
                            "topic": q.get("topic", "General")
                        }).execute()

                        # Ghi _id vào progress file sau khi insert thành công
                        q_id = q.get("_id")
                        if q_id is not None:
                            done_ids.add(q_id)
                            progress_data[req.model] = list(done_ids)
                            with open(PROGRESS_FILE, "w") as f:
                                json.dump(progress_data, f, indent=2)

                    except Exception as db_err:
                        # Log chi tiết lỗi từ Supabase/PostgREST
                        error_msg = str(db_err)
                        print(f"❌ [Benchmark Insert Error] Detail: {error_msg}")
                        if "400" in error_msg or "Bad Request" in error_msg:
                            print(f"   Payload sent: model={req.model}, question={q.get('question','')}[:50]..., answer_len={len(res.get('answer',''))}")
                        raise

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

class EvaluateRequest(BaseModel):
    model_name: str          # Tên collection cần chấm: "vector_raw", "vector_pq", ...
    batch_size: int = 10     # 10 câu/batch × 3 contexts × 3 metrics ≈ 1,350 TPM (≤ 6,000 TPM Groq)

@app.post("/api/benchmark/evaluate")
async def evaluate_with_ragas(req: EvaluateRequest, background_tasks: BackgroundTasks):
    """
    Đánh giá RAGAS toàn bộ câu trả lời của 1 model từ bảng 'benchmarks'.
    Chia thành các batch nhỏ (batch_size), chạy RAGAS từng batch,
    tổng hợp điểm trung bình → lưu 1 dòng vào bảng 'ragas_results'.
    
    Dùng GOOGLE_API_KEY (key 1) — tách biệt hoàn toàn với generation.
    Metrics: faithfulness, answer_relevancy, context_precision.
    """
    def run_evaluation():
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
            from datasets import Dataset
            import math

            logger.info(f"[RAGAS] ===== BẮT ĐẦU ĐÁNH GIÁ MODEL: {req.model_name} =====")

            eval_api_key = os.getenv("GOOGLE_API_KEY")  # Key 1 — tách quota với generation
            eval_llm = ChatGoogleGenerativeAI(
                model="gemini-3.1-flash-lite-preview",
                google_api_key=eval_api_key,
                temperature=0,
            )
            eval_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=eval_api_key,
            )

            ragas_llm   = LangchainLLMWrapper(eval_llm)
            ragas_embed = LangchainEmbeddingsWrapper(eval_embeddings)
            metrics     = [faithfulness, answer_relevancy, context_precision]

            # Lấy TOÀN BỘ câu trả lời của model từ Supabase
            sm = SupabaseManager()
            rows = sm.supabase.table("benchmarks") \
                .select("question, answer, contexts, ground_truth") \
                .eq("model_name", req.model_name) \
                .neq("answer", "") \
                .execute().data or []

            if not rows:
                logger.warning(f"[RAGAS] Không tìm thấy dữ liệu cho model '{req.model_name}'.")
                return

            total = len(rows)
            n_batches = math.ceil(total / req.batch_size)
            logger.info(f"[RAGAS] Tổng: {total} câu | Batch size: {req.batch_size} | Số batch: {n_batches}")

            # Tích lũy điểm qua từng batch
            all_faithfulness       = []
            all_answer_relevancy   = []
            all_context_precision  = []

            # Ước tính số API calls để user nắm trước
            MAX_CONTEXTS_PER_SAMPLE = 3  # 3 chunks/câu × ~1,500 tokens = ~4,500 tokens/batch
            est_calls = total * (2 + 1 + MAX_CONTEXTS_PER_SAMPLE)
            logger.info(f"[RAGAS] Ước tính API calls: {est_calls} "
                        f"(faithfulness×2 + relevancy×1 + precision×{MAX_CONTEXTS_PER_SAMPLE} / câu)")
            logger.info(f"[RAGAS] Giới hạn Gemini free tier: 1,500 RPD → "
                        f"{'⚠️ CÓ THỂ BỊ RATE LIMIT' if est_calls > 1500 else '✅ Trong giới hạn'}")

            for b in range(n_batches):
                batch = rows[b * req.batch_size : (b + 1) * req.batch_size]
                logger.info(f"[RAGAS] Đang chạy Batch {b+1}/{n_batches} ({len(batch)} câu)...")

                try:
                    dataset = Dataset.from_dict({
                        "question":     [r["question"] for r in batch],
                        "answer":       [r["answer"] or "" for r in batch],
                        # Giới hạn tối đa 5 chunks/câu → giảm API calls context_precision
                        "contexts":     [
                            (r["contexts"][:MAX_CONTEXTS_PER_SAMPLE]
                             if isinstance(r["contexts"], list) else
                             ([r["contexts"]] if r["contexts"] else [""]))
                            for r in batch
                        ],
                        "ground_truth": [r.get("ground_truth") or "" for r in batch],
                    })

                    result = ragas_evaluate(
                        dataset,
                        metrics=metrics,
                        llm=ragas_llm,
                        embeddings=ragas_embed,
                    )

                    df = result.to_pandas()
                    all_faithfulness      += df["faithfulness"].dropna().tolist()
                    all_answer_relevancy  += df["answer_relevancy"].dropna().tolist()
                    all_context_precision += df["context_precision"].dropna().tolist()

                    logger.info(f"[RAGAS] Batch {b+1} xong | "
                                f"faith={df['faithfulness'].mean():.4f} | "
                                f"rel={df['answer_relevancy'].mean():.4f} | "
                                f"prec={df['context_precision'].mean():.4f}")

                except Exception as batch_err:
                    logger.error(f"[RAGAS] Lỗi ở Batch {b+1}: {batch_err}")
                    continue  # Vẫn tiếp tục batch tiếp theo

                # Delay giữa các batch để tôn trọng TPM rate limit (Groq: 6,000 TPM)
                if b < n_batches - 1:
                    import time as _time
                    delay_s = 90  # 90s đảm bảo chủ ngựa tốc độ token trong giới hạn
                    logger.info(f"[RAGAS] Chờ {delay_s}s để rải token load (TPM safety)...")
                    _time.sleep(delay_s)

            # Tính trung bình tổng hợp
            if not all_faithfulness:
                logger.error("[RAGAS] Không có điểm nào được tính. Dừng lại.")
                return

            avg_faith  = round(float(np.mean(all_faithfulness)), 4)
            avg_rel    = round(float(np.mean(all_answer_relevancy)), 4)
            avg_prec   = round(float(np.mean(all_context_precision)), 4)
            avg_ragas  = round(float(np.mean([avg_faith, avg_rel, avg_prec])), 4)

            # Lưu 1 dòng tổng kết vào bảng ragas_results
            sm.supabase.table("ragas_results").insert({
                "model_name":        req.model_name,
                "total_evaluated":   len(all_faithfulness),
                "batch_size":        req.batch_size,
                "faithfulness":      avg_faith,
                "answer_relevancy":  avg_rel,
                "context_precision": avg_prec,
                "ragas_score":       avg_ragas,
            }).execute()

            logger.info(f"[RAGAS] ===== HOÀN THÀNH =====")
            logger.info(f"[RAGAS] Model: {req.model_name} | "
                        f"Faithfulness={avg_faith} | "
                        f"Answer Relevancy={avg_rel} | "
                        f"Context Precision={avg_prec} | "
                        f"RAGAS Score={avg_ragas}")

        except Exception as e:
            logger.error(f"[RAGAS] Lỗi nghiêm trọng: {str(e)}")

    background_tasks.add_task(run_evaluation)
    total_batches = f"~{(483 // req.batch_size) + 1}"
    return {
        "message": f"Bắt đầu đánh giá RAGAS cho '{req.model_name}' | "
                   f"Batch size={req.batch_size} | Ước tính {total_batches} batch. "
                   f"Kết quả sẽ lưu vào bảng 'ragas_results'."
    }

@app.get("/api/benchmark/ragas-results")
async def get_ragas_results():
    """Lấy toàn bộ kết quả RAGAS từ bảng ragas_results để so sánh các model."""
    sm = SupabaseManager()
    try:
        res = sm.supabase.table("ragas_results") \
            .select("*") \
            .order("created_at", desc=True) \
            .execute()
        return {"results": res.data or []}
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
    tracker = MemoryTracker(os.getpid())
    monitor_thread = threading.Thread(target=tracker.track, daemon=True)
    monitor_thread.start()

    start_time = time.time()
    try:
        cs = ChatService()
        full_answer = ""
        contexts = []
        retrieval_lat = 0
        rerank_lat = 0
        async for chunk in cs.chat_stream(req.query, "google", req.collection):
            data = json.loads(chunk)
            # chat_service yield ra type "final" chứa toàn bộ kết quả
            if data["type"] == "final":
                full_answer = data.get("answer", "")
                contexts = data.get("contexts", [])
                # Chỉ lấy retrieval_latency_ms — trọng tâm đồ án là tối ưu hóa truy xuất
                retrieval_lat = data.get("retrieval_latency_ms", 0)
                print(f"DEBUG: retrieval_lat={retrieval_lat}ms")
            # Fallback: nếu handler yield từng token riêng
            elif data["type"] == "text":
                full_answer += data["content"]
            elif data["type"] == "context":
                contexts = data["content"]

        latency = (time.time() - start_time) * 1000
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tracker.stop()        # Báo hiệu thread giám sát dừng qua Event
        monitor_thread.join() # Chờ thread kết thúc sạch sẽ

    base_ram   = round(tracker.start_mem, 2)      # RAM nền trước khi bắt đầu xử lý
    peak_delta = round(tracker.peak_delta_mb, 2)  # RAM tăng thêm trong quá trình xử lý
    total_ram  = round(base_ram + peak_delta, 2)  # Tổng RAM thực tế để serve 1 query

    return {
        "answer": full_answer,
        "contexts": contexts,
        "latency_ms": round(latency, 2),
        "base_ram_mb": base_ram,
        "peak_ram_mb": peak_delta,
        "total_ram_mb": total_ram,
        "retrieval_latency_ms": retrieval_lat
    }



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
