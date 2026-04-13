import os
import json
import time
import numpy as np
from typing import List, Dict
from embed import EmbeddingManager
from vector_store import VectorStoreManager
from ragas_eval import RagasEvaluator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

class ChatService:
    def __init__(self):
        self.embed_manager = EmbeddingManager()
        self.vector_manager = VectorStoreManager()
        self.ragas_evaluator = RagasEvaluator()
        self.method_labels = {
            "vector_raw": "[RAG-RAW]",
            "vector_adaptive": "[RAG-Adaptive]",
            "vector_pq": "[RAG-PQ]",
            "vector_sq8": "[RAG-SQ8]",
            "vector_arq": "[ARQ-RAG]"
        }
        
    def get_llm(self, model_type: str):
        if model_type == "gemini":
            return ChatGoogleGenerativeAI(
                model="gemini-3.1-flash-lite-preview",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0
            )
        else:
            # Dùng Qwen 2.5 3B qua Ollama container
            return ChatOllama(
                model="qwen2.5:3b",
                base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
                temperature=0
            )

    def _extract_text(self, content):
        """Chuyển đổi nội dung phản hồi từ LLM (có thể là list/dict) thành chuỗi văn bản thuần."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Gemini qua LangChain có thể trả về list các dictionary (parts)
            text_parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
                else:
                    text_parts.append(str(part))
            return "".join(text_parts)
        return str(content)

    def _rewrite_query(self, query: str) -> str:
        """Sử dụng Gemini để chuyển đổi câu hỏi Tiếng Việt sang Tiếng Anh chuyên ngành phục vụ retrieval."""
        try:
            llm = self.get_llm("gemini")
            prompt = f"""You are a specialized RAG Query Rewriter. 
Translate the following Vietnamese research question into a precise English search query.
Focus on maintaining all technical terms (like VBL, FBL, DMDT, ARQ, MIMO, etc.) exactly as they are.
The goal is to provide a query that will match well with English scientific papers.

Vietnamese Question: {query}
English Search Query:"""
            
            messages = [HumanMessage(content=prompt)]
            response = llm.invoke(messages)
            rewritten = self._extract_text(response.content).strip()
            print(f"Rewritten Query: {rewritten}")
            return rewritten
        except Exception as e:
            print(f"Lỗi rewrite query: {e}")
            return query

    async def chat(self, query: str, model_name: str, collection_name: str):
        # 1. Sử dụng Query gốc (Tránh Translation Noise)
        search_query = query
        # Tự động bỏ qua rewrite query để giữ đúng ngữ nghĩa chuyên môn Tiếng Việt
        # if any(ord(c) > 127 for c in query): 
        #    search_query = self._rewrite_query(query)

        # 2. Embed search query
        query_vector = np.array(self.embed_manager.get_embedding(search_query))
        
        # 3. Retrieval (Adaptive Limit)
        limit = 15
        if collection_name == "vector_arq":
            limit = 80 # Tăng Recall để TurboQuant xử lý được các rank thấp
        
        search_results = self.vector_manager.search(collection_name, query_vector, limit=limit)
        
        # 3. Reranking (Adaptive logic cho ARQ using ADC Direct Scoring)
        final_contexts = []
        source_hits = []
        
        if collection_name == "vector_arq":
            from quantization import QuantizationManager
            qm = QuantizationManager()
            
            # Trích xuất dữ liệu hàng loạt để tính điểm Batch
            idx_batch = np.array([hit.payload["idx"] for hit in search_results])
            qjl_batch = np.array([hit.payload["qjl"] for hit in search_results])
            gamma_batch = np.array([hit.payload["gamma"] for hit in search_results])
            orig_norms = np.array([hit.payload.get("orig_norm", 1.0) for hit in search_results])
            
            # Batch Reranking (ADC Direct Scoring với chuẩn hóa ngược)
            scores = qm.tq_prod.compute_score_batch(query_vector, idx_batch, qjl_batch, gamma_batch, orig_norms=orig_norms)
            
            # Sắp xếp và lấy Top 5
            refined_results = []
            for i, score in enumerate(scores):
                refined_results.append((score, search_results[i]))
                
            refined_results.sort(key=lambda x: x[0], reverse=True)
            top_hits = [x[1] for x in refined_results[:5]]
            
            final_contexts = [hit.payload["content"] for hit in top_hits]
            source_hits = top_hits
        else:
            final_contexts = [hit.payload["content"] for hit in search_results]
            source_hits = search_results

        # 4. Generate Answer
        context_text = "\n\n".join(final_contexts)
        system_prompt = f"""Bạn là một trợ lý nghiên cứu AI chuyên nghiệp. 
Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp sau đây. 
Nếu thông tin không có trong ngữ cảnh, hãy nói là bạn không biết, đừng tự bịa câu trả lời.

NGỮ CẢNH:
{context_text}"""

        llm = self.get_llm(model_name)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        start_time = time.time()
        response = llm.invoke(messages)
        answer = self._extract_text(response.content)
        # Tự động gắn nhãn theo mô hình đã chọn
        label = self.method_labels.get(collection_name, "[RAG]")
        if not answer.startswith(label):
            answer = f"{label} {answer}"
            
        latency = time.time() - start_time
        
        # 5. Ragas Evaluation
        scores = self.ragas_evaluator.evaluate(query, final_contexts, answer)
        
        return {
            "answer": answer,
            "sources": [
                {"file": hit.payload["file"], "content": hit.payload["content"]} 
                for hit in source_hits[:5]
            ],
            "scores": scores,
            "latency": round(latency, 2),
            "method": collection_name
        }

    async def chat_stream(self, query: str, model_name: str, collection_name: str):
        """Hàm stream tiến trình xử lý từng bước cho Frontend."""
        # 1. Sử dụng Query gốc (Tránh Translation Noise)
        search_query = query
        # Vô hiệu hóa bước dịch sang Tiếng Anh
        # yield json.dumps({"type": "status", "message": "🌐 Đang chuyển đổi truy vấn sang Tiếng Anh chuyên ngành..."}) + "\n"
        # search_query = self._rewrite_query(query)

        # 2. Embed query
        yield json.dumps({"type": "status", "message": "🔍 Đang nhúng câu hỏi bằng nomic-embed-text..."}) + "\n"
        query_vector = np.array(self.embed_manager.get_embedding(search_query))
        
        # 3. Retrieval (Adaptive Limit)
        limit = 15
        if collection_name == "vector_arq":
            limit = 80
        
        search_results = self.vector_manager.search(collection_name, query_vector, limit=limit)
        
        # 3. Reranking
        final_contexts = []
        source_hits = []
        
        if collection_name == "vector_arq":
            yield json.dumps({"type": "status", "message": "⚡ Đang xử lý ARQ Adaptive Reranking (Batch ADC Score)..."}) + "\n"
            from quantization import QuantizationManager
            qm = QuantizationManager()
            
            # Batch data extraction
            idx_batch = np.array([hit.payload["idx"] for hit in search_results])
            qjl_batch = np.array([hit.payload["qjl"] for hit in search_results])
            gamma_batch = np.array([hit.payload["gamma"] for hit in search_results])
            orig_norms = np.array([hit.payload.get("orig_norm", 1.0) for hit in search_results])
            
            # Batch Reranking
            scores = qm.tq_prod.compute_score_batch(query_vector, idx_batch, qjl_batch, gamma_batch, orig_norms=orig_norms)
            
            refined_results = []
            for i, score in enumerate(scores):
                refined_results.append((score, search_results[i]))
            
            refined_results.sort(key=lambda x: x[0], reverse=True)
            top_hits = [x[1] for x in refined_results[:5]]
            # Debug log rank 1
            print(f"Top 1 Score: {refined_results[0][0]} | File: {top_hits[0].payload['file']}")
            final_contexts = [hit.payload["content"] for hit in top_hits]
            source_hits = top_hits
        else:
            final_contexts = [hit.payload["content"] for hit in search_results]
            source_hits = search_results

        # 4. Generate Answer
        yield json.dumps({"type": "status", "message": "💎 Đang tổng hợp câu trả lời từ ngữ cảnh..."}) + "\n"
        context_text = "\n\n".join(final_contexts)
        system_prompt = f"""Bạn là một trợ lý nghiên cứu AI chuyên nghiệp. 
Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp sau đây. 
Nếu thông tin không có trong ngữ cảnh, hãy nói là bạn không biết, đừng tự bịa câu trả lời.

NGỮ CẢNH:
{context_text}"""
        
        llm = self.get_llm(model_name)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        start_time = time.time()
        response = llm.invoke(messages)
        answer = self._extract_text(response.content)
        # Tự động gắn nhãn theo mô hình đã chọn
        label = self.method_labels.get(collection_name, "[RAG]")
        if not answer.startswith(label):
            answer = f"{label} {answer}"
            
        latency = time.time() - start_time
        
        # 5. Ragas Evaluation
        yield json.dumps({"type": "status", "message": "📊 Đang đánh giá chất lượng phản hồi (RAGAS)..."}) + "\n"
        scores = self.ragas_evaluator.evaluate(query, final_contexts, answer)
        
        final_result = {
            "type": "final",
            "answer": answer,
            "sources": [
                {"file": hit.payload["file"], "content": hit.payload["content"]} 
                for hit in source_hits[:5]
            ],
            "scores": scores,
            "latency": round(latency, 2),
            "method": collection_name
        }
        yield json.dumps(final_result) + "\n"
