import os
import json
import asyncio
import logging
import random
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Đảm bảo có thể import module từ thư mục cha
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shared.supabase_client import SupabaseManager

load_dotenv()

# Thiết lập Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CloudBenchmark")

class CloudBenchmarkGenerator:
    def __init__(self):
        self.sm = SupabaseManager()
        self.qdrant_url = os.getenv("QDRANT_CLOUD_URL")
        self.qdrant_key = os.getenv("QDRANT_CLOUD_API_KEY")
        
        if not self.qdrant_url or not self.qdrant_key:
            raise ValueError("❌ Thiếu QDRANT_CLOUD_URL hoặc QDRANT_CLOUD_API_KEY trong môi trường.")
            
        self.qdrant = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_key)
        
        # Tự động tạo Index cho trường 'topic'
        try:
            self.qdrant.create_payload_index(
                collection_name="vector_raw",
                field_name="topic",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info("✅ Đã đảm bảo Payload Index cho 'topic'.")
        except Exception:
            pass
            
        # SIÊU NÂNG CẤP: Sử dụng Gemini 3.1 Flash-Lite Preview (Thế hệ mới nhất 2026)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview",
            temperature=0.4,
            max_output_tokens=4096
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an elite research evaluator. Based on the provided research paper fragments (chunks), "
                       "generate EXACTLY 10 high-quality, distinct technical questions and their precise ground truth answers. "
                       "Each question must be challenging and technical. "
                       "Format your response as a valid JSON array of objects: "
                       '[ {{"question": "Q1...", "ground_truth": "A1..."}}, {{"question": "Q2...", "ground_truth": "A2..."}}, ... ]'),
            ("human", "Topic: {topic}\n\nContext Chunks:\n{context}")
        ])
        
        self.chain = self.prompt | self.llm

    def get_random_chunks_from_qdrant(self, topic: str, limit: int = 70):
        """Lấy ngẫu nhiên các đoạn văn bản từ Qdrant."""
        try:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="topic", match=models.MatchValue(value=topic))]
            )
            
            res, _ = self.qdrant.scroll(
                collection_name="vector_raw",
                scroll_filter=query_filter,
                limit=limit * 2,
                with_payload=True
            )
            
            if not res:
                # Fallback dữ liệu cũ
                res, _ = self.qdrant.scroll(collection_name="vector_raw", limit=limit, with_payload=True)
                
            if not res: return []
                
            selected = random.sample(res, min(limit, len(res)))
            return [{"content": p.payload.get("content"), "file": p.payload.get("file")} for p in selected]
            
        except Exception as e:
            logger.error(f"❌ Lỗi Qdrant: {e}")
            return []

    async def run(self, target_per_topic=85):
        TOPICS = ["TurboQuant", "ARQ", "PQ", "RAG", "ML_Optimization", "LLM_Inference"]
        
        # CỐ ĐỊNH: Nạp 100 chunk mỗi lần theo yêu cầu để đạt Coverage tối đa
        chunks_per_request = 100 
        logger.info(f"🚀 Chế độ Coverage Cao: Nạp {chunks_per_request} chunks/lần sinh.")

        existing_queries = self.sm.get_benchmark_queries()
        topic_counts = {t: 0 for t in TOPICS}
        for q in existing_queries:
            t = q.get("topic", "General")
            if t in topic_counts: topic_counts[t] += 1
            
        logger.info("🔥 BẮT ĐẦU PIPELINE CAO CẤP (100 CHUNKS - GEMINI 3.1 FLASH LITE)")

        for topic in TOPICS:
            current_count = topic_counts.get(topic, 0)
            if current_count >= target_per_topic:
                logger.info(f"✅ [{topic}] đã đủ {current_count} câu.")
                continue

            needed = target_per_topic - current_count
            needed_requests = (needed + 9) // 10
            
            logger.info(f"🔥 [{topic}]: Cần {needed} câu. Sinh trong {needed_requests} lượt...")

            for r in range(needed_requests):
                try:
                    sample_chunks = self.get_random_chunks_from_qdrant(topic, limit=chunks_per_request)
                    if not sample_chunks: break

                    context_text = "\n\n---\n\n".join([c["content"] for c in sample_chunks])
                    source_files = list(set(c["file"] for c in sample_chunks))

                    response = await self.chain.ainvoke({"topic": topic, "context": context_text})
                    
                    # Xử lý content nếu là list
                    content = response.content
                    if isinstance(content, list):
                        content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
                    
                    # Trích xuất JSON
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    
                    qa_list = json.loads(content.strip())
                    
                    if isinstance(qa_list, list):
                        for data in qa_list:
                            self.sm.save_single_benchmark_query(
                                question=data["question"],
                                ground_truth=data["ground_truth"],
                                topic=topic,
                                source_files=source_files
                            )
                        logger.info(f"   [{topic}] Lượt {r+1}/{needed_requests} thành công. (+10 câu)")
                    
                    # ĐIỀU CHỈNH: 30 giây chờ để bảo vệ giới hạn 250K TPM (vì nạp tới 100 chunks)
                    await asyncio.sleep(30)

                except Exception as e:
                    logger.error(f"❌ Lỗi vòng lặp {topic}: {e}")
                    await asyncio.sleep(40)

        logger.info("🎉 HOÀN THÀNH PIPELINE CHẤT LƯỢNG CAO!")

if __name__ == "__main__":
    generator = CloudBenchmarkGenerator()
    asyncio.run(generator.run())
