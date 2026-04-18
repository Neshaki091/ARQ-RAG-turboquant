import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from .supabase_client import SupabaseManager

class QueryAnalyzer:
    def __init__(self):
        self.sm = SupabaseManager()
        self.llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
            max_retries=3,
            request_timeout=30
        )

    def analyze(self, query: str) -> dict:
        """
        Phân tích câu hỏi và trả về cấu hình RAG tương ứng.
        Sử dụng Supabase Cache để tối ưu hóa.
        """
        # 1. Kiểm tra Cache từ Supabase
        cache_data = self.sm.get_query_cache(query)
        
        if cache_data:
            print(f"Cache Hit: '{query}' -> Cấu hình: {cache_data}")
            label = cache_data
        else:
            print(f"Cache Miss: Đang phân loại '{query}' bằng llama-3.1-8b-instant...")
            label = self._classify_with_llm(query)
            self.sm.set_query_cache(query, label)
        
        # 2. Map label -> RAG Parameters
        config_map = {
            "EASY": {"limit": 20, "top_k": 5, "desc": "Tra cứu nhanh"},
            "NORMAL": {"limit": 50, "top_k": 10, "desc": "Giải thích chi tiết"},
            "HARD": {"limit": 80, "top_k": 20, "desc": "Chứng minh/Tổng hợp phức tạp"},
            "EXTREME": {"limit": 120, "top_k": 30, "desc": "Nghiên cứu đa tài liệu"}
        }
        
        config = config_map.get(label, config_map["NORMAL"])
        limit_val = config["limit"]
        top_k_val = config["top_k"]

        return {
            "complexity": label,
            "limit": limit_val,
            "top_k": top_k_val,
            "label": f"🧠 Chế độ: {label} ({config['desc']} | Scan={limit_val}, Focus={top_k_val})"
        }

    def _classify_with_llm(self, query: str) -> str:
        """Sử dụng Llama 3.1 8B để phân loại câu hỏi vào 4 nhãn: EASY, NORMAL, HARD, EXTREME."""
        try:
            prompt = f"""You are a RAG Query Analyzer. Categorize the following research question into one of four levels:
- EASY: Simple definition, short lookup, or factual info.
- NORMAL: Detailed explanation, procedure, or single-concept analysis.
- HARD: Mathematical proof, complex comparison, or multi-paper synthesis.
- EXTREME: Holistic review across many papers, complex architectural comparison, or deep methodology derivation.

Question: {query}
Respond with ONLY ONE WORD: EASY, NORMAL, HARD, or EXTREME. Do not write anything else."""
            
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            
            res_text = response.content.strip().upper()
            if "EXTREME" in res_text: return "EXTREME"
            if "HARD" in res_text: return "HARD"
            if "NORMAL" in res_text: return "NORMAL"
            if "EASY" in res_text: return "EASY"
            return "NORMAL"
        except Exception as e:
            print(f"Lỗi khi phân loại LLM: {e}")
            return "NORMAL"
