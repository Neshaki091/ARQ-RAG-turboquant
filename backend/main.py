from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import psutil
import sys
import shutil
import time
import torch
from dotenv import load_dotenv
from groq import Groq
from itertools import cycle
import asyncio

# Add local backend directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Nạp các biến môi trường từ file .env chính xác trong thư mục backend
env_path = os.path.join(backend_dir, ".env")
load_dotenv(dotenv_path=env_path)

# Khởi tạo Groq Client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

from backend.services.ingestion_service import ingestion_service
from backend.services.metadata_service import metadata_service
from backend.services.tq_service import tq_service
from backend.services.import_service import import_service
from backend.services.auth_service import auth_service
from backend.services.sync_service import sync_service
from itertools import cycle
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, BackgroundTasks

app = FastAPI(
    title="DEMO ARQ-RAG API", 
    version="1.0.0",
    redirect_slashes=False
)

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
# TQ Service nạp System Index tự động khi khởi tạo
print("LOG: Backend Services Initialized.")

# Nạp model DPR ngay lập tức để tránh latency câu đầu tiên
print("LOG: Pre-loading Embedding Model...")
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
    
    from backend.scripts.hf_worker import run_worker
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

def analyze_query(query: str):
    """Gộp dịch thuật và phân loại độ khó vào 1 lần gọi duy nhất để tăng tốc 2x"""
    if not os.getenv("GROQ_API_KEY"):
        return query, "AVERAGE"
    prompt = f'[MANDATORY] Translate this into ENGLISH: "{query}" | Classify: EASY, AVERAGE, HARD, EXTRA. ONLY return: [TRANSLATION] | [DIFFICULTY]'
    try:
        chat_completion = groq_client.chat.completions.create(
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
            print(f"DEBUG: Translated: '{trans}' | Complexity: {diff}")
            return trans, diff
            
        return res.strip().strip('"'), "AVERAGE"
    except Exception as e:
        print(f"DEBUG: Analyze Error: {e}")
        return query, "AVERAGE"


        


        


class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    mode: str = "balance" # "ultrafast", "fast", "balance", "accuracy", "adaptive"
    scope: str = "both" # "user", "system", "both"
    stream: bool = False
    session_id: str = "default"

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

@app.get("/")
async def root():
    return {"message": "Welcome to DEMO ARQ-RAG API", "status": "online"}

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

@app.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    session_id: str = Form("default"),
    current_user = Depends(get_current_user)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file_path = os.path.join(UPLOAD_DIR, f"{current_user['id']}_{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        num_chunks = ingestion_service.process_pdf(file_path, file.filename, user_id=current_user['id'], session_id=session_id)
        
        # Đồng bộ ngầm lên Cloud
        background_tasks.add_task(sync_service.sync_to_hub, str(current_user['id']))
        
        return {"filename": file.filename, "chunks": num_chunks, "status": "success"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents(current_user = Depends(get_current_user)):
    return {"documents": metadata_service.list_documents(user_id=current_user['id'])}

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
        ingestion_service.delete_document(filename, user_id=current_user['id'])
        
        # Đồng bộ ngầm việc xóa lên Cloud
        background_tasks.add_task(sync_service.sync_to_hub, str(current_user['id']))
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

class BatchChatRequest(BaseModel):
    messages: list[str]
    mode: str = "balance"
    scope: str = "both"

@app.post("/chat/batch")
async def chat_batch_arq(request: BatchChatRequest, current_user = Depends(get_current_user)):
    """
    Xử lý hàng loạt câu hỏi cùng lúc để tận dụng Double Batching.
    """
    import traceback
    try:
        if not request.messages:
            return {"results": []}
            
        mode = request.mode
        
        # 1. Embedding cho toàn bộ batch câu hỏi
        try:
            query_vectors = ingestion_service.get_embeddings(request.messages)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")
            
        # Chuyển sang Tensor float32 (quan trọng)
        queries_t = torch.from_numpy(query_vectors).float()
        
        start_time = time.time()
        target_top_k = 10
        
        # 2. Double Batching Retrieval
        if mode == "ultrafast":
            ranked = tq_service.search_batch(query_vectors, top_k=target_top_k, bits=4, nprobe=32, rerank_mult=1)
        elif mode == "fast":
            ranked = tq_service.search_batch(query_vectors, top_k=target_top_k, bits=4, nprobe=32, rerank_mult=6)
        elif mode == "balance":
            ranked = tq_service.search_batch(query_vectors, top_k=target_top_k, bits=4, nprobe=64, rerank_mult=6)
        elif mode == "accuracy":
            ranked = tq_service.search_batch(query_vectors, top_k=target_top_k, bits=4, nprobe=64, rerank_mult=10)
        elif mode == "adaptive":
            # Batch adaptive đơn giản hóa: dùng Balance cho toàn bộ batch
            ranked = tq_service.search_batch(query_vectors, top_k=target_top_k, bits=4, nprobe=64, rerank_mult=6)
        else: # Default
            ranked = tq_service.search_batch(query_vectors, top_k=target_top_k, bits=4, nprobe=64, rerank_mult=6)

        results_list = []
        for ranked_items in ranked:
            tq_ids = [item["id"] for item in ranked_items]
            search_results = metadata_service.get_by_ids(tq_ids, user_id=current_user['id'], scope=request.scope)
            results_list.append(search_results)

        retrieval_latency = (time.time() - start_time) * 1000
        
        # 3. Tạo câu trả lời hàng loạt bằng Qwen Local
        import asyncio
        semaphore = asyncio.Semaphore(4)
        loop = asyncio.get_running_loop()
        
        async def get_single_answer(query, results):
            async with semaphore:
                if not results: return "Không tìm thấy tài liệu liên quan."
                context = "\n".join([r.payload.get('text', '')[:500] for r in results[:2]])
                prompt = f"Dựa vào context sau, trả lời cực ngắn gọn (1 câu) cho câu hỏi: {query}\nContext: {context}"
                try:
                    return await loop.run_in_executor(None, get_ai_response, prompt)
                except Exception as e:
                    return f"Lỗi tạo câu trả lời: {str(e)}"

        answer_tasks = [get_single_answer(request.messages[i], results_list[i]) for i in range(len(request.messages))]
        answers = await asyncio.gather(*answer_tasks)

        total_latency = (time.time() - start_time) * 1000
        
        return {
            "retrieval_latency": f"{retrieval_latency:.2f}ms",
            "total_latency": f"{total_latency/1000:.2f}s",
            "throughput": f"{len(request.messages) / (retrieval_latency/1000):.2f} queries/sec",
            "results": results_list,
            "answers": answers
        }
    except Exception as e:
        print("ERROR: CRITICAL ERROR IN /chat/batch:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_arq(request: ChatRequest, current_user = Depends(get_current_user)):
    query = request.message
    mode = request.mode.lower()
    start_time = time.time()
    
    # 1 & 2. Dịch câu hỏi và Phân loại độ khó (Gộp 1 lần gọi duy nhất)
    translated_query, complexity = analyze_query(query)
    
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
        query_vector = ingestion_service.get_embeddings([translated_query], is_query=True)[0]
        
        # Nếu dùng Accuracy, lấy nhiều ứng viên hơn để rerank
        retrieve_k = dynamic_top_k * 5 if use_cross_encoder else dynamic_top_k
        
        # 3.2. Fetch Allowed IDs for User Scope (Session-level isolation)
        allowed_ids = None
        if request.scope in ["user", "both"]:
            # Lấy danh sách ID thuộc về User và Session hiện tại
            allowed_ids = metadata_service.get_ids_by_session(current_user['id'], request.session_id)

        tq_results = tq_service.search(
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

        # 5. Reranking (nếu cần)
        if use_cross_encoder and hydrated_results:
            from backend.services.rerank_service import rerank_service
            final_results = rerank_service.rerank(translated_query, hydrated_results, dynamic_top_k)
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
    
    display_latency = (time.time() - start_time) * 1000
    
    async def stream_generator():
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
                                    yield content
                            except:
                                continue

        except Exception as e:
            yield f"\nERROR: Lỗi Stream OpenRouter: {str(e)}"

    return StreamingResponse(
        stream_generator(), 
        media_type="text/plain",
        headers={"X-Accel-Buffering": "no"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
