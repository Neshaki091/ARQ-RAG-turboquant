import ollama
from ollama import AsyncClient
import json

class OllamaService:
    def __init__(self, model: str = "qwen2.5:1.5b"):
        self.model = model
        self.client = AsyncClient()

    async def chat(self, messages):
        response = await self.client.chat(model=self.model, messages=messages)
        return response['message']['content']

    async def analyze_query(self, query: str):
        """
        Analyze if the query needs a global scan or specific retrieval.
        Returns a JSON with strategy.
        """
        prompt = f"""
        Phân tích câu hỏi người dùng sau đây và quyết định chiến lược tìm kiếm:
        Câu hỏi: "{query}"
        
        Trả về JSON với các trường:
        - "strategy": "global" (nếu câu hỏi bao quát, so sánh, tổng hợp) hoặc "local" (nếu câu hỏi chi tiết về một sự kiện, thông số cụ thể).
        - "reason": Lý do ngắn gọn.
        - "keywords": Danh sách từ khóa quan trọng.
        
        JSON:
        """
        response = await self.client.generate(model=self.model, prompt=prompt)
        try:
            # Clean response to get only JSON
            text = response['response']
            start = text.find('{')
            end = text.rfind('}') + 1
            return json.loads(text[start:end])
        except:
            return {"strategy": "local", "reason": "default fallback", "keywords": []}

# Global instance
ollama_service = OllamaService()
