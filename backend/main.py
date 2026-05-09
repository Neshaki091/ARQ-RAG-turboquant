from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys
import shutil
import time
import torch
from dotenv import load_dotenv
from groq import Groq
from itertools import cycle
import asyncio

# Nạp các biến môi trường từ file .env
load_dotenv()

# Khởi tạo Groq Client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Add root to path to import TQ_engine_lib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.ingestion_service import ingestion_service
from backend.services.metadata_service import metadata_service
from backend.services.tq_service import tq_service
from backend.services.import_service import import_service
import google.generativeai as genai
from itertools import cycle

app = FastAPI(
    title="DEMO ARQ-RAG API", 
    version="1.0.0",
    redirect_slashes=False
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Tự động nạp chỉ mục cũ nếu có khi khởi động
print("📂 Checking for existing TurboQuant indexes...")
if tq_service.load():
    print("✅ Existing indexes loaded successfully.")
else:
    print("ℹ️ No existing indexes found. Please upload or import data.")

# Cấu hình Gemini 3.1 Flash-Lite
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite-preview"
api_keys = [os.getenv("GOOGLE_API_KEY_1"), os.getenv("GOOGLE_API_KEY_2")]
api_keys = [k for k in api_keys if k] # Lọc các key hợp lệ
key_pool = cycle(api_keys)

def get_ai_response(prompt: str):
    """Gọi Qwen Local qua Ollama để đạt tốc độ nhanh nhất"""
    import ollama
    try:
        response = ollama.generate(model="qwen2.5:0.5b", prompt=prompt)
        return response['response']
    except Exception as e:
        return f"Lỗi Ollama: {str(e)}. Hãy chắc chắn bạn đã chạy 'ollama pull qwen2.5:0.5b'"

def get_question_complexity(query: str) -> str:
    """Sử dụng Groq (Llama 3) để phân loại độ khó câu hỏi cực nhanh"""
    if not os.getenv("GROQ_API_KEY"):
        return "AVERAGE"
        
    prompt = f"""Phân loại độ khó câu hỏi nghiên cứu AI: '{query}'. 
Chỉ trả về 1 từ: EASY, AVERAGE, HARD, EXTRA.

- EASY: Định nghĩa cơ bản, hỏi đáp thông thường.
- AVERAGE: Giải thích mối quan hệ, quy trình đơn giản.
- HARD: Chứa thuật ngữ chuyên sâu (VD: ACS, G-RAG, Subgraph, Quantization), yêu cầu phân tích học thuật.
- EXTRA: So sánh nhiều phương pháp, yêu cầu tổng hợp từ nhiều nguồn dữ liệu phức tạp.

PHÂN LOẠI:"""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.2-1b-preview",
            max_tokens=10,
            temperature=0
        )
        res = chat_completion.choices[0].message.content.strip().upper()
        for level in ["EASY", "AVERAGE", "HARD", "EXTRA"]:
            if level in res: return level
        return "AVERAGE"
    except:
        return "AVERAGE"

class ChatRequest(BaseModel):
    message: str
    mode: str = "fast" # "raw", "fast", "ultra"
    stream: bool = False

@app.get("/")
async def root():
    return {"message": "Welcome to DEMO ARQ-RAG API", "status": "online"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        num_chunks = ingestion_service.process_pdf(file_path, file.filename)
        return {"filename": file.filename, "chunks": num_chunks, "status": "success"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    return {"documents": metadata_service.list_documents()}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    try:
        ingestion_service.delete_document(filename)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/import-precomputed")
async def import_precomputed_data():
    try:
        count = import_service.import_precomputed()
        return {"status": "success", "count": count}
    except Exception as e:
        import traceback
        traceback.print_exc() # In lỗi chi tiết ra console
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
    mode: str = "fast"

@app.post("/chat/batch")
async def chat_batch_arq(request: BatchChatRequest):
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
        if mode == "raw":
            results_list = []
            for i in range(len(request.messages)):
                results_list.append(tq_service.search_raw(query_vectors[i], top_k=target_top_k))
        else:
            bits = 4 if mode == "fast" else 2
            rerank_mult = 6 if bits == 4 else 10
            ranked = tq_service.search_batch(query_vectors, top_k=target_top_k, bits=bits, rerank_mult=rerank_mult)

            results_list = []
            for ranked_items in ranked:
                tq_ids = [item["id"] for item in ranked_items]
                search_results = metadata_service.get_by_ids(tq_ids)
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
        print("❌ CRITICAL ERROR IN /chat/batch:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_arq(request: ChatRequest):
    query = request.message
    mode = request.mode
    
    # 1. Phân loại độ khó (Groq Adaptive)
    c_start = time.time()
    complexity = get_question_complexity(query)
    c_time = (time.time() - c_start) * 1000
    print(f"🧠 Complexity: {complexity} ({c_time:.2f}ms) | Mode: {mode}")
    
    # Xác định Top-K dựa trên độ khó
    top_k_map = {
        "EASY": 5,
        "AVERAGE": 10,
        "HARD": 15,
        "EXTRA": 35
    }
    target_top_k = top_k_map.get(complexity, 10)

    # 2. Embedding
    e_start = time.time()
    try:
        query_vector = ingestion_service.get_embeddings([query])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")
    e_time = (time.time() - e_start) * 1000
    print(f"⏱️ Embedding: {e_time:.2f}ms")
    
    # 3. Retrieval
    start_time = time.time()
    
    if mode == "raw":
        tq_results = tq_service.search_raw(query_vector, top_k=target_top_k)
    elif mode == "ultra":
        tq_results = tq_service.search(query_vector, top_k=target_top_k, bits=2, rerank_mult=10)
    elif mode == "adaptive":
        # === ARQ-RAG ADAPTIVE ROUTING CORE ===
        if complexity == "EASY":
            tq_results = tq_service.search(query_vector, top_k=5, bits=4, rerank_mult=1)  # Không Rerank, tốc độ tối đa
        elif complexity == "AVERAGE":
            tq_results = tq_service.search(query_vector, top_k=10, bits=4, rerank_mult=6) # Rerank tiêu chuẩn
        elif complexity == "HARD":
            tq_results = tq_service.search(query_vector, top_k=15, bits=2, rerank_mult=10) # 2-bit siêu nén + Full Rerank để tăng độ phủ
        else: # EXTRA
            tq_results = tq_service.search_raw(query_vector, top_k=35) # Dùng Float32 gốc quét cạn
    else: # "fast"
        tq_results = tq_service.search(query_vector, top_k=target_top_k, bits=4, rerank_mult=6)

    # Hiển thị tốc độ thực tế 100% của Engine
    display_latency = (time.time() - start_time) * 1000
    print(f"⏱️ Retrieval: {display_latency:.2f}ms")
    
    tq_ids = [r['id'] for r in tq_results] if tq_results else []
    search_results = metadata_service.get_by_ids(tq_ids)
    
    context_parts = []
    for r in search_results:
        text = r.payload.get('text', '')
        source = r.payload.get('source', 'Unknown Document')
        page = r.payload.get('page', '?')
        context_parts.append(f"[{source} - p.{page}]: {text}")
        
    context = "\n\n".join(context_parts)
    
    # 3. Setup Prompt (Dữ liệu thuần túy, luật nằm ở System Instruction)
    prompt = f"CÂU HỎI: {query}\n\nNGỮ CẢNH:\n{context}\n\nTRẢ LỜI:"
    
    async def stream_generator():
        # 1. Gửi Metadata ngay lập tức
        source_list = list(set([r.payload.get('source', 'Unknown Document') for r in search_results]))
        source_list = [s for s in source_list if s and s != 'Unknown']
        
        meta = {
            "mode": mode,
            "latency": f"{display_latency:.2f}ms",
            "complexity": complexity,
            "sources": source_list
        }
        yield json.dumps(meta) + "\n--META_END--\n"

        # 2. Stream tokens từ Gemini (Sử dụng ASYNC để nhanh nhất)
        try:
            # Thử từng API Key trong pool
            for _ in range(len(api_keys)):
                current_key = next(key_pool)
                try:
                    genai.configure(api_key=current_key)
                    # Sử dụng cấu hình generation tối ưu cho tốc độ
                    generation_config = {
                        "temperature": 0.1,
                        "top_p": 0.95,
                        "max_output_tokens": 2048,
                        "response_mime_type": "text/plain",
                    }
                    # Đưa các ràng buộc vào System Instruction (Gemini hỗ trợ rất tốt)
                    system_instruction = (
                        "Bạn là một trợ lý AI chuyên nghiệp. CHỈ trả lời dựa trên Ngữ cảnh được cung cấp. "
                        "KHÔNG giải thích quá trình tìm kiếm. KHÔNG lặp lại yêu cầu. "
                        "Nếu không thấy thông tin, CHỈ trả lời đúng câu: 'Xin lỗi, tôi không tìm thấy thông tin này trong bộ tài liệu nghiên cứu của bạn.' "
                        "Tuyệt đối không đưa ra bất kỳ lý lẽ hay phân tích nào khác."
                    )
                    model = genai.GenerativeModel(
                        model_name='gemini-3.1-flash-lite-preview',
                        generation_config=generation_config,
                        system_instruction=system_instruction
                    )                    
                    # Gọi Async Stream
                    response = await model.generate_content_async(prompt, stream=True)
                    async for chunk in response:
                        try:
                            if chunk.candidates[0].content.parts:
                                text = chunk.text
                                if text:
                                    yield text
                                    # Giải phóng event loop để đẩy dữ liệu đi ngay
                                    await asyncio.sleep(0)
                        except (AttributeError, IndexError, ValueError):
                            continue
                    return 
                except Exception as e:
                    print(f"⚠️ Gemini Key Error: {e}")
                    continue
            yield "\n❌ Lỗi: Tất cả API Keys đều thất bại."
        except Exception as e:
            yield f"\n❌ Lỗi Stream Gemini: {str(e)}"

    return StreamingResponse(
        stream_generator(), 
        media_type="text/plain",
        headers={"X-Accel-Buffering": "no"}
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
