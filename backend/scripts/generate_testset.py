import os
import json
import random
import logging
import asyncio
from typing import List, Dict
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestsetGenerator")

load_dotenv()

class TestsetGenerator:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
            
        self.llm = ChatGroq(
            model_name=self.model_name,
            api_key=self.api_key,
            temperature=0.7 # Tăng nhiệt độ để câu hỏi đa dạng hơn
        )
        
        self.chunks_path = "backend/data/chunks.json"
        self.output_path = "backend/data/benchmark_queries_synthetic.json"

    def load_chunks(self) -> List[Dict]:
        if not os.path.exists(self.chunks_path):
            logger.error(f"Chunks file not found at {self.chunks_path}")
            return []
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def generate_qa(self, chunk_content: str) -> Dict:
        prompt = f"""
Dựa trên đoạn văn bản (chunk) sau đây từ nghiên cứu về 'MIMO Multihop Networks với ARQ', hãy tạo một câu hỏi nghiên cứu cụ thể và một câu trả lời chuẩn (ground truth) tương ứng.

Yêu cầu:
1. Câu hỏi phải mang tính học thuật, kỹ thuật và đi sâu vào chi tiết của đoạn văn.
2. Câu trả lời chuẩn (ground truth) phải chính xác, đầy đủ và hoàn toàn dựa trên nội dung đoạn văn cung cấp.
3. Câu trả lời nên trình bày rõ ràng, súc tích nhưng đầy đủ ý.
4. KHÔNG được bịa đặt thông tin không có trong đoạn văn.

Định dạng đầu ra là JSON duy nhất:
{{
    "question": "Câu hỏi ở đây?",
    "ground_truth": "Câu trả lời chuẩn ở đây."
}}

Đoạn văn:
\"\"\"{chunk_content}\"\"\"
"""
        try:
            response = await self.llm.ainvoke(prompt)
            # Trích xuất JSON từ phản hồi (giả định LLM trả về JSON sạch hoặc trong block code)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error generating QA: {e}")
            return None

    async def run(self, num_samples: int = 5):
        logger.info(f"🚀 Bắt đầu sinh {num_samples} câu hỏi thử nghiệm...")
        chunks = self.load_chunks()
        if not chunks:
            return

        # Chỉ chọn các chunk có nội dung đủ dài (> 200 ký tự) để đảm bảo chất lượng
        valid_chunks = [c for c in chunks if len(c.get('content', '')) > 200]
        if not valid_chunks:
            valid_chunks = chunks
            
        selected_chunks = random.sample(valid_chunks, min(num_samples, len(valid_chunks)))
        results = []

        for i, chunk in enumerate(selected_chunks):
            logger.info(f"📝 Đang xử lý chunk {i+1}/{num_samples} (ID: {chunk.get('chunk_id')})...")
            qa = await self.generate_qa(chunk['content'])
            if qa:
                qa["source_chunk"] = chunk.get('chunk_id')
                results.append(qa)
                logger.info(f"✅ Đã sinh xong: {qa['question'][:50]}...")
            
        # Lưu kết quả
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        logger.info(f"🏁 Đã lưu {len(results)} câu hỏi vào: {self.output_path}")

if __name__ == "__main__":
    generator = TestsetGenerator()
    asyncio.run(generator.run(num_samples=5)) # Chạy thử 5 câu mẫu
