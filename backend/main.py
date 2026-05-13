from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
import psutil
import sys
import shutil
import time
import torch
from dotenv import load_dotenv
from groq import AsyncGroq
from itertools import cycle
import asyncio

# Add local backend directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Nạp các biến môi trường từ file .env chính xác trong thư mục backend
env_path = os.path.join(backend_dir, ".env")
load_dotenv(dotenv_path=env_path)

# Khởi tạo danh sách các Groq Client để xoay vòng (Tránh Rate Limit)
groq_keys = [os.getenv(k) for k in os.environ if k.startswith("GROQ_API_KEY") and os.getenv(k)]
if not groq_keys:
    # Fallback nếu không tìm thấy key theo list, thử lấy key mặc định
    default_key = os.getenv("GROQ_API_KEY")
    if default_key:
        groq_keys = [default_key]

if groq_keys:
    print(f"LOG: Initialized {len(groq_keys)} Groq API Keys for rotation.")
    groq_clients_pool = cycle([AsyncGroq(api_key=k) for k in groq_keys])
else:
    print("WARNING: No GROQ_API_KEY found. AI features will be disabled.")
    groq_clients_pool = None

def get_groq_client():
    """Lấy client tiếp theo trong vòng lặp"""
    return next(groq_clients_pool) if groq_clients_pool else None

from services.ingestion_service import ingestion_service
from services.metadata_service import metadata_service
from services.tq_service import tq_service
from services.rerank_service import rerank_service
from services.import_service import import_service
from services.auth_service import auth_service
from services.sync_service import sync_service
from itertools import cycle
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, BackgroundTasks
from fastapi.concurrency import run_in_threadpool

batcher = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global batcher
    print("LOG: Starting DynamicBatcher...")
    batcher = DynamicBatcher(batch_window=0.5, max_batch_size=32)
    yield

app = FastAPI(
    lifespan=lifespan,
    title="DEMO ARQ-RAG API", 
    version="1.0.0",
    redirect_slashes=False
)

# --- DYNAMIC BATCHER FOR SIMULATION ---
# Cấu hình n_probe cho các chế độ tìm kiếm
MODE_CONFIG = {
    "ultrafast": 16,
    "fast": 32,
    "balance": 64,
    "accuracy": 64, # accuracy dùng n_probe 64 nhưng có thêm rerank
    "adaptive": 32
}

def get_search_params(mode: str, difficulty: str):
    """Xác định n_probe và use_rerank dựa trên mode và độ khó (Adaptive)"""
    mode = mode.lower()
    diff = difficulty.upper()
    
    if mode == "adaptive":
        if diff == "EASY":
            return 32, False # Tương đương FAST
        elif diff == "AVERAGE":
            return 64, False # Tương đương BALANCE
        else:
            return 64, True  # Tương đương ACCURACY (HARD/EXTRA)
    
    # Các chế độ cố định
    n_probe = MODE_CONFIG.get(mode, 64)
    use_rerank = (mode == "accuracy")
    return n_probe, use_rerank

class DynamicBatcher:
    def __init__(self, batch_window=1.0, max_batch_size=16):
        self.queue = []
        self.batch_window = batch_window
        self.max_batch_size = max_batch_size
        self.lock = asyncio.Lock()
        self.flush_event = asyncio.Event()
        asyncio.create_task(self._worker())

    async def add_query(self, query_data):
        future = asyncio.get_event_loop().create_future()
        async with self.lock:
            self.queue.append((query_data, future))
            # Nếu đủ 16 câu, kích hoạt xử lý ngay
            if len(self.queue) >= self.max_batch_size:
                self.flush_event.set()
        return await future
    async def _worker(self):
        while True:
            # Nếu hàng đợi trống, đợi item đầu tiên
            while not self.queue:
                await asyncio.sleep(0.05)
            
            # Đợi tối đa batch_window hoặc cho đến khi đủ 16 câu
            try:
                await asyncio.wait_for(self.flush_event.wait(), timeout=self.batch_window)
            except asyncio.TimeoutError:
                pass
            
            async with self.lock:
                if not self.queue:
                    continue
                batch = self.queue[:self.max_batch_size]
                self.queue = self.queue[self.max_batch_size:]
                self.flush_event.clear()
            
            # Xử lý batch
            asyncio.create_task(self.process_batch(batch))

    async def process_batch(self, batch):
        try:
            queries = [item[0]['query'] for item in batch]
            user_id = batch[0][0]['user_id']
            session_id = batch[0][0]['session_id']
            scope = batch[0][0].get('scope', 'both')
            mode_val = batch[0][0].get('mode', 'balance').lower()
            
            # --- ĐO THỜI GIAN EMBEDDING ---
            embed_start = time.time()
            query_vectors = await run_in_threadpool(ingestion_service.get_embeddings, queries)
            embed_latency = (time.time() - embed_start) * 1000 / len(queries) # Trung bình cho mỗi câu
            
            # --- AI ANALYSIS (BATCH PROMPT OPTIMIZED) ---
            # Gom toàn bộ queries vào 1 lần gọi Groq duy nhất để tránh Rate Limit và tăng tốc x10
            translated_queries, complexities = await analyze_queries_batch(queries)

            # Lấy allowed_ids từ session đầu tiên
            allowed_ids = metadata_service.get_ids_by_session(user_id, session_id)

            # Lấy n_probe và rerank dựa trên mode
            # Lấy n_probe và rerank dựa trên mode (đã có mode_val từ trên)
            mode = mode_val
            
            # --- ĐỊNH NGHĨA K DỰA TRÊN ĐỘ KHÓ ---
            K_MAP = {"EASY": 3, "AVERAGE": 5, "HARD": 10, "EXTRA": 15}
            
            # --- ĐO THỜI GIAN SEARCH ---
            search_start = time.time()
            
            # Vì search_batch chạy cho cả batch, ta phải chọn n_probe cao nhất trong batch nếu dùng adaptive
            # Hoặc đơn giản là dùng n_probe của mode nếu không phải adaptive
            max_n_probe = 0
            any_rerank = False
            
            batch_params = []
            for diff in complexities:
                np, rr = get_search_params(mode, diff)
                batch_params.append((np, rr))
                max_n_probe = max(max_n_probe, np)
                if rr: any_rerank = True

            # Nếu dùng rerank, ta lấy 4 * K candidates (lấy 40 cho an toàn)
            results = await run_in_threadpool(tq_service.search_batch, 
                query_vectors, 
                user_id=user_id,
                top_k=40 if any_rerank else 15, 
                scope=scope,
                allowed_ids=allowed_ids,
                n_probe=max_n_probe,
                use_rerank=any_rerank
            )
            search_latency = (time.time() - search_start) * 1000 / len(queries)
            
            # --- TỐI ƯU HÓA: FETCH METADATA BẰNG BATCH (CHỈ 1-2 LẦN TRUY VẤN DB) ---
            all_result_ids = []
            for r_list in results:
                for r in r_list:
                    all_result_ids.append(r['id'])
            
            # Lấy toàn bộ chunk metadata một lần duy nhất
            all_chunks_raw = metadata_service.get_by_ids(
                list(set(all_result_ids)), 
                user_id=user_id, 
                session_id=session_id, 
                scope=scope
            )
            # Tạo map để truy xuất nhanh O(1)
            chunks_map = {c.payload['id']: c.payload for c in all_chunks_raw}

            for i, item in enumerate(batch):
                diff = complexities[i]
                target_k = K_MAP.get(diff, 5)
                n_probe_val, use_rerank_val = batch_params[i]
                
                # Lấy candidates từ map đã fetch sẵn
                candidates = []
                for res in results[i]:
                    c_payload = chunks_map.get(res['id'])
                    if c_payload:
                        candidates.append({
                            "id": res['id'],
                            "text": c_payload.get('text', ''),
                            "source": c_payload.get('source', ''),
                            "score": res['score']
                        })
                
                # --- THỰC HIỆN RERANK NẾU CẦN ---
                final_results = candidates
                if use_rerank_val and candidates:
                    final_results = await run_in_threadpool(
                        rerank_service.rerank, 
                        queries[i], 
                        candidates, 
                        target_k
                    )
                else:
                    final_results = candidates[:target_k]
                
                # Format lại cho frontend
                chunk_details = [{
                    "text": r['text'][:200],
                    "source": r['source']
                } for r in final_results]
                
                item[1].set_result({
                    "chunks_found": len(final_results),
                    "chunks": chunk_details,
                    "complexity": diff,
                    "embed_latency": round(embed_latency, 2),
                    "search_latency": round(search_latency, 2),
                    "batch_mode": True
                })
        except Exception as e:
            for item in batch:
                if not item[1].done():
                    item[1].set_exception(e)

# Batcher will be initialized in lifespan

# Enable CORS - Cho phép Vercel truy cập
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Bạn có thể thay bằng ["https://your-app.vercel.app"] để bảo mật hơn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sử dụng đường dẫn tuyệt đối dựa trên vị trí file main.py để tránh nhầm lẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Tự động nạp chỉ mục cũ và nạp Model ngay khi khởi động
print("LOG: Initializing Backend Services...")

# 1. Tải dữ liệu từ Cloud về (nếu đang chạy trên HF)
from services.sync_service import sync_service
sync_service.download_from_hub()

# 2. TQ Service nạp System Index tự động khi khởi tạo
print("LOG: Backend Services Initialized.")

# Nạp model Multilingual E5 Base ngay lập tức để tránh latency câu đầu tiên
print("LOG: Pre-loading Multilingual E5 Base Embedding Model...")
ingestion_service._lazy_load_model()
print("SUCCESS: Embedding Model ready.")


@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "TurboQuant ARQ-RAG Backend is running",
        "version": "1.0.0"
    }

@app.post("/worker/rebuild")
async def trigger_worker(
    background_tasks: BackgroundTasks,
    limit: int = 100000, 
    secret: str = Form(...)
):
    # Kiểm tra quyền Admin (ADMIN_SECRET cấu hình trong HF Secrets)
    if secret != os.getenv("ADMIN_SECRET", "admin123"):
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    from scripts.hf_worker import run_worker
    background_tasks.add_task(run_worker, limit=limit)
    return {"status": "started", "message": f"Worker is rebuilding {limit} chunks in background."}

def get_ai_response(prompt: str):
    """Sử dụng mô hình gpt-oss-120b qua OpenRouter để đạt chất lượng cao nhất"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "Lỗi: Chưa cấu hình OPENROUTER_API_KEY trong .env"
        
    import requests
    import json
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "gpt-oss-120b", # Theo yêu cầu của người dùng
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )
        res_json = response.json()
        if "choices" in res_json:
            return res_json["choices"][0]["message"]["content"]
        else:
            return f"Lỗi OpenRouter: {json.dumps(res_json)}"
    except Exception as e:
        return f"Lỗi kết nối OpenRouter: {str(e)}"

async def analyze_query(query: str):
    """Gộp dịch thuật và phân loại độ khó vào 1 lần gọi duy nhất để tăng tốc 2x"""
    if not groq_keys:
        return query, "AVERAGE"
    prompt = f'[MANDATORY] Translate this into ENGLISH: "{query}" | Classify: EASY, AVERAGE, HARD, EXTRA. ONLY return: [TRANSLATION] | [DIFFICULTY]'
    try:
        chat_completion = await get_groq_client().chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=100,
            temperature=0
        )
        res = chat_completion.choices[0].message.content.strip()
        print(f"DEBUG: AI Raw Response: {res}")
        
        if " | " in res:
            parts = res.split(" | ")
            trans = parts[0].replace("Translation:", "").replace("TRANSLATION:", "").strip().strip('"')
            diff = parts[1].replace("Difficulty:", "").replace("DIFFICULTY:", "").strip().upper()
            return trans, diff
        return query, "AVERAGE"
    except Exception as e:
        print(f"ERROR Groq: {e}")
        return query, "AVERAGE"

async def analyze_queries_batch(queries: list[str]):
    """Gom toàn bộ danh sách câu hỏi vào 1 lần gọi Groq duy nhất (JSON Mode)"""
    if not groq_keys or not queries:
        return queries, ["AVERAGE"] * len(queries)
        
    # Tạo danh sách đánh số để AI dễ phân biệt
    numbered_queries = "\n".join([f"{i+1}. {q}" for i, q in enumerate(queries)])
    
    prompt = f"""
    Analyze the following list of queries. For each query:
    1. Translate it into English.
    2. Classify difficulty as EASY, AVERAGE, HARD, or EXTRA.
    
    Return ONLY a JSON object with a key "results" containing a list of objects:
    {{"results": [{{"translation": "...", "difficulty": "..."}}, ...]}}
    
    Queries:
    {numbered_queries}
    """
    
    try:
        chat_completion = await get_groq_client().chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            response_format={ "type": "json_object" },
            temperature=0
        )
        res_text = chat_completion.choices[0].message.content
        import json
        data = json.loads(res_text)
        results = data.get("results", [])
        
        # Nếu AI trả về thiếu kết quả, điền mặc định
        trans_list = []
        diff_list = []
        for i in range(len(queries)):
            if i < len(results):
                trans_list.append(results[i].get("translation", queries[i]))
                diff_list.append(results[i].get("difficulty", "AVERAGE").upper())
            else:
                trans_list.append(queries[i])
                diff_list.append("AVERAGE")
        
        return trans_list, diff_list
    except Exception as e:
        print(f"ERROR Batch Groq: {e}")
        return queries, ["AVERAGE"] * len(queries)

class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: Optional[str] = None
    messages_batch: Optional[list[str]] = None
    mode: str = "balance" # "ultrafast", "fast", "balance", "accuracy", "adaptive"
    scope: str = "both" # "user", "system", "both"
    stream: bool = False
    session_id: str = "default"
    session_title: str = None # Để đặt tiêu đề khi tạo session mới

class SessionTitleRequest(BaseModel):
    title: str

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = auth_service.decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = metadata_service.get_user(payload.get("sub"))
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

@app.post("/register")
async def register(request: RegisterRequest):
    hashed = auth_service.hash_password(request.password)
    user_id = metadata_service.add_user(request.username, hashed)
    if not user_id:
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"status": "success", "user_id": user_id}

@app.post("/login")
async def login(request: LoginRequest):
    user = metadata_service.get_user(request.username)
    if not user or not auth_service.verify_password(request.password, user['password']):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    token = auth_service.create_access_token({"sub": user['username'], "id": user['id']})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/me")
async def get_me(current_user = Depends(get_current_user)):
    user_dict = dict(current_user)
    return {"id": user_dict['id'], "username": user_dict['username'], "role": user_dict.get('role', 'user')}

def check_admin(user = Depends(get_current_user)):
    user_dict = dict(user)
    if user_dict.get('role') != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    return user_dict

@app.get("/admin/users")
async def admin_list_users(admin = Depends(check_admin)):
    return {"users": metadata_service.list_all_users()}

@app.get("/admin/system/chunks")
async def admin_get_system_chunks(offset: int = 0, limit: int = 100, admin = Depends(check_admin)):
    chunks = metadata_service.get_all_chunks(user_id=-1, limit=limit, offset=offset)
    total = metadata_service.get_count(user_id=-1)
    return {"chunks": chunks, "total": total, "offset": offset, "limit": limit}

@app.get("/system/stats")
async def get_system_stats():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # Working Set (RSS on Linux/Mac, WorkingSet on Windows)
    working_set = mem_info.rss / (1024 * 1024) # MB
    cpu_usage = process.cpu_percent(interval=None) # %
    return {
        "memory_mb": round(working_set, 2),
        "cpu_percent": cpu_usage,
        "uptime": round(time.time() - start_time, 2)
    }

# Biến global lưu thời gian khởi động
start_time = time.time()

@app.get("/benchmark/queries")
async def get_benchmark_queries():
    import json
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "queries", "benchmark_queries_400.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
    return []

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    session_id: str = Form("default"),
    current_user = Depends(get_current_user)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # 1. Lưu file tạm thời lên đĩa
        file_path = os.path.join(UPLOAD_DIR, f"{current_user['id']}_{int(time.time())}_{file.filename}")
        with open(file_path, "wb") as buffer:
            import shutil
            shutil.copyfileobj(file.file, buffer)

        # 2. Chạy tác vụ nặng trong thread pool với ĐƯỜNG DẪN file
        await run_in_threadpool(ingestion_service.process_pdf, file_path, file.filename, user_id=current_user['id'], session_id=session_id)
        
        # Đồng bộ ngầm lên Cloud
        background_tasks.add_task(sync_service.sync_to_hub, str(current_user['id']))
        
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents(session_id: str = None, current_user = Depends(get_current_user)):
    return {"documents": metadata_service.list_documents(user_id=current_user['id'], session_id=session_id)}

@app.get("/documents/{filename}/chunks")
async def get_document_chunks(filename: str, current_user = Depends(get_current_user)):
    # Nếu là admin, cho phép xem bất kỳ file nào
    # Nếu là user, chỉ cho xem file của chính mình
    target_user_id = current_user['id']
    if current_user.get('role') == 'admin':
        # Logic đơn giản: Admin có thể gửi query param ?user_id=X để xem của user khác
        pass 

    chunks = metadata_service.get_chunks_by_filename(filename, target_user_id)
    return {"filename": filename, "chunks": chunks}

@app.delete("/documents/{filename}")
async def delete_document(
    background_tasks: BackgroundTasks,
    filename: str, 
    current_user = Depends(get_current_user)
):
    try:
        await run_in_threadpool(ingestion_service.delete_document, filename, user_id=current_user['id'])
        
        # Đồng bộ ngầm việc xóa lên Cloud
        background_tasks.add_task(sync_service.sync_to_hub, str(current_user['id']))
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- SESSION ENDPOINTS ---

@app.get("/sessions")
async def get_sessions(current_user = Depends(get_current_user)):
    return {"sessions": metadata_service.list_sessions(current_user['id'])}

@app.post("/sessions")
async def create_session(request: dict, current_user = Depends(get_current_user)):
    session_id = request.get("session_id")
    title = request.get("title", "New Chat")
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
    metadata_service.create_session(session_id, current_user['id'], title)
    return {"status": "success", "session_id": session_id}

@app.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str, current_user = Depends(get_current_user)):
    messages = metadata_service.get_session_messages(session_id, current_user['id'])
    return {"messages": messages}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, current_user = Depends(get_current_user)):
    metadata_service.delete_session(session_id, current_user['id'])
    return {"status": "success"}

@app.put("/sessions/{session_id}/title")
async def update_session_title(session_id: str, request: SessionTitleRequest, current_user = Depends(get_current_user)):
    metadata_service.update_session_title(session_id, current_user['id'], request.title)
    return {"status": "success"}

@app.post("/import-precomputed")
async def import_precomputed_data(user_id: int = -1):
    try:
        count = import_service.import_precomputed(user_id=user_id)
        return {"status": "success", "count": count, "target": "system" if user_id == -1 else f"user_{user_id}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cleanup")
async def cleanup_indexes():
    try:
        tq_service.cleanup_old_versions()
        return {"status": "success", "message": "Đã dọn dẹp các phiên bản cũ"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import StreamingResponse
import json

@app.post("/chat")
async def chat_arq(request: ChatRequest, current_user = Depends(get_current_user)):
    user_id = current_user['id']
    
    # --- BATCH PROCESSING FOR SIMULATION ---
    # Nếu có messages_batch, xử lý tất cả cùng lúc bằng Batcher
    if request.messages_batch:
        try:
            tasks = [batcher.add_query({
                "query": msg,
                "user_id": user_id,
                "session_id": request.session_id,
                "scope": request.scope
            }) for msg in request.messages_batch]
            
            batch_results = await asyncio.gather(*tasks)
            return {
                "batch_results": batch_results,
                "batch_mode": True
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # --- DYNAMIC BATCHING FOR SINGLE REQUEST (SIMULATION MODE) ---
    if not request.stream:
        try:
            # Gửi vào bộ gom lô 50ms
            batch_result = await batcher.add_query({
                "query": request.message,
                "user_id": user_id,
                "session_id": request.session_id,
                "scope": request.scope
            })
            
            return {
                "answer": f"Simulation: Found {batch_result['chunks_found']} chunks",
                "chunks_count": batch_result['chunks_found'],
                "chunks": batch_result['chunks'],
                "complexity": batch_result['complexity'],
                "embed_latency": batch_result.get('embed_latency', 0),
                "search_latency": batch_result.get('search_latency', 0),
                "batch_mode": True
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    query = request.message
    mode = request.mode.lower()
    start_time = time.time()
    
    # 1 & 2. Dịch câu hỏi và Phân loại độ khó (Gộp 1 lần gọi duy nhất)
    translated_query, complexity = await run_in_threadpool(analyze_query, query)
    
    # Tính toán top_k động
    base_top_k = 5
    if complexity == "EASY":
        dynamic_top_k = base_top_k
    elif complexity == "AVERAGE":
        dynamic_top_k = base_top_k + 5
    else: # HARD / EXTRA
        dynamic_top_k = base_top_k + 10

    # 3. Xác định thông số Search Profile
    nprobe = 64
    use_cross_encoder = False
    
    if mode == "adaptive":
        if complexity == "EASY": mode = "ultrafast"
        elif complexity == "AVERAGE": mode = "balance"
        else: mode = "accuracy"

    if mode == "ultrafast":
        nprobe = 64
    elif mode == "fast":
        nprobe = 128
    elif mode == "balance":
        nprobe = 256
    elif mode == "accuracy":
        nprobe = 256
        use_cross_encoder = True

    # 4. Embedding & Retrieval
    try:
        query_vector = await run_in_threadpool(ingestion_service.get_embeddings, [translated_query], True)
        query_vector = query_vector[0]
        
        # Nếu dùng Accuracy, lấy nhiều ứng viên hơn để rerank
        retrieve_k = dynamic_top_k * 5 if use_cross_encoder else dynamic_top_k
        
        # 3.2. Fetch Allowed IDs for User Scope (Session-level isolation)
        allowed_ids = None
        if request.scope in ["user", "both"]:
            # Lấy danh sách ID thuộc về User và Session hiện tại
            allowed_ids = metadata_service.get_ids_by_session(current_user['id'], request.session_id)

        tq_results = await run_in_threadpool(tq_service.search, 
            query_vector, 
            top_k=retrieve_k, 
            user_id=current_user['id'],
            scope=request.scope,
            allowed_ids=allowed_ids
        )
        
        # Hydrate metadata - Dựa trên Source để tránh trùng ID giữa 2 DB
        hydrated_results = []
        for res in tq_results:
            source = res.get('source', 'system')
            
            if source == 'system':
                chunk = metadata_service.get_chunk(res['id'], user_id=-1)
            else: # source == 'user'
                chunk = metadata_service.get_chunk(res['id'], user_id=current_user['id'])
                # Kiểm tra session_id
                if chunk and chunk.get('session_id') != request.session_id:
                    chunk = None

            if chunk:
                from types import SimpleNamespace
                hydrated_results.append(SimpleNamespace(id=res['id'], score=res['score'], payload=chunk))

        # 5. Reranking (nếu cần) (Heavy Task)
        if use_cross_encoder and hydrated_results:
            from services.rerank_service import rerank_service
            final_results = await run_in_threadpool(rerank_service.rerank, translated_query, hydrated_results, dynamic_top_k)
        else:
            final_results = hydrated_results[:dynamic_top_k]

        context_parts = []
        for res in final_results:
            text = res.payload.get('text', '')
            source = res.payload.get('filename', 'Unknown')
            context_parts.append(f"[Source: {source}]: {text}")
            
        context = "\n\n".join(context_parts)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

    # 6. LLM Generation
    prompt = f"CÂU HỎI: {query}\n\nNGỮ CẢNH:\n{context}\n\nTRẢ LỜI:"
    
    # 7. Save User Message to History
    metadata_service.add_message(request.session_id, current_user['id'], "user", query)
    # Cập nhật title nếu là session mới
    if request.session_title:
        metadata_service.update_session_title(request.session_id, current_user['id'], request.session_title)
    
    display_latency = (time.time() - start_time) * 1000
    
    async def stream_generator():
        full_ai_response = ""
        # 1. Gửi Metadata ngay lập tức
        source_list = list(set([r.payload.get('source', 'Unknown Document') for r in final_results]))
        source_list = [s for s in source_list if s and s != 'Unknown']
        
        meta = {
            "mode": mode,
            "latency": f"{display_latency:.2f}ms",
            "complexity": complexity,
            "sources": source_list
        }
        yield json.dumps(meta) + "\n--META_END--\n"

        # 2. Stream tokens từ OpenRouter gpt-oss-120b
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                yield "Lỗi: Chưa cấu hình OPENROUTER_API_KEY trong .env"
                return

            import httpx
            
            system_instruction = (
                "Bạn là một chuyên gia phân tích dữ liệu và trợ lý AI cao cấp. "
                "Nhiệm vụ của bạn là sử dụng NGỮ CẢNH được cung cấp để trả lời câu hỏi một cách chi tiết, chuyên sâu và dễ hiểu. "
                "Hãy diễn giải vấn đề một cách logic, có cấu trúc (sử dụng bullet points nếu cần). "
                "Nếu không tìm thấy thông tin trong ngữ cảnh, hãy trả lời: 'Xin lỗi, tôi không tìm thấy thông tin này trong bộ tài liệu nghiên cứu của bạn.' "
                "Hãy luôn giữ phong thái chuyên nghiệp và khách quan."
            )

            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST",
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-oss-120b",
                        "stream": True,
                        "messages": [
                            {"role": "system", "content": system_instruction},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1
                    },
                    timeout=60.0
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                content = data["choices"][0]["delta"].get("content", "")
                                if content:
                                    full_ai_response += content
                                    yield content
                            except:
                                continue
            
            # 8. Save Assistant Message to History after streaming completes
            if full_ai_response:
                metadata_service.add_message(request.session_id, current_user['id'], "assistant", full_ai_response)

        except Exception as e:
            yield f"\nERROR: Lỗi Stream OpenRouter: {str(e)}"

    return StreamingResponse(
        stream_generator(), 
        media_type="text/plain",
        headers={"X-Accel-Buffering": "no"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
