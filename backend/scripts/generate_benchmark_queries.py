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
        
        # Tự động tạo Index cho trường 'topic' để tránh lỗi 400 Bad Request
        try:
            self.qdrant.create_payload_index(
                collection_name="vector_raw",
                field_name="topic",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            logger.info("✅ Đã đảm bảo Payload Index cho 'topic' trên Qdrant.")
        except Exception:
            pass # Index có thể đã tồn tại 
        
        # Khởi tạo LLM
        # Lưu ý: Trên GitHub Actions, thư viện sẽ tự động lấy GOOGLE_API_KEY từ môi trường
        self.llm = ChatGoogleGenerativeAI(
            model="gemma-4-31b-it",
            temperature=0.3,
            max_output_tokens=2048
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an elite research evaluator. Based on the provided research paper fragments (chunks), "
                       "generate EXACTLY 5 high-quality, distinct technical questions and their precise ground truth answers. "
                       "Each question must be challenging and technical. "
                       "Format your response as a valid JSON array of objects: "
                       '[{"question": "Q1...", "ground_truth": "A1..."}, {"question": "Q2...", "ground_truth": "A2..."}, ...]'),
            ("human", "Topic: {topic}\n\nContext Chunks:\n{context}")
        ])
        
        self.chain = self.prompt | self.llm

    def get_random_chunks_from_qdrant(self, topic: str, limit: int = 5):
        """Lấy ngẫu nhiên các đoạn văn bản từ Qdrant theo Topic."""
        try:
            # Sử dụng Filter để lọc theo Topic
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="topic",
                        match=models.MatchValue(value=topic)
                    )
                ]
            )
            
            # Scroll lấy dữ liệu ngẫu nhiên (sử dụng offset ngẫu nhiên nếu cần, nhưng đơn giản nhất là scroll)
            res, _ = self.qdrant.scroll(
                collection_name="vector_raw",
                scroll_filter=query_filter,
                limit=limit * 2, # Lấy dư ra một chút để chọn ngẫu nhiên
                with_payload=True,
                with_vectors=False
            )
            
            if not res:
                # FALLBACK: Nếu không tìm thấy chunk theo Topic (do dữ liệu cũ chưa gắn nhãn)
                # Ta sẽ lấy ngẫu nhiên các chunk bất kỳ để AI vẫn có dữ liệu làm việc
                logger.warning(f"🔍 Nhóm [{topic}] chưa có dữ liệu gắn nhãn. Đang lấy dữ liệu ngẫu nhiên từ kho 28k chunks...")
                res, _ = self.qdrant.scroll(
                    collection_name="vector_raw",
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
                
            if not res:
                return []
                
            selected = random.sample(res, min(limit, len(res)))
            return [{
                "content": p.payload.get("content"),
                "file": p.payload.get("file")
            } for p in selected]
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi truy vấn Qdrant cho topic {topic}: {e}")
            return []

    async def run(self, target_per_topic=85):
        TOPICS = ["TurboQuant", "ARQ", "PQ", "RAG", "ML_Optimization", "LLM_Inference"]
        
        # 1. Tính toán số lượng chunk nạp vào dựa trên tổng số dữ liệu
        try:
            collection_info = self.qdrant.get_collection("vector_raw")
            total_chunks = collection_info.points_count
            # Công thức: Tổng số chunk / 500 (Mục tiêu 500 câu hỏi)
            # Đảm bảo tối thiểu 5 chunk và tối đa 100 chunk để tránh quá tải prompt
            chunks_per_request = max(5, min(100, total_chunks // 500))
            logger.info(f"📊 Hệ thống phát hiện {total_chunks} chunks. Tối ưu: nạp {chunks_per_request} chunks/lần sinh.")
        except Exception as e:
            logger.warning(f"⚠️ Không lấy được số lượng chunk, dùng mặc định 10. Lỗi: {e}")
            chunks_per_request = 10

        # 2. Kiểm tra trạng thái hiện tại từ Supabase
        existing_queries = self.sm.get_benchmark_queries()
        topic_counts = {}
        for q in existing_queries:
            t = q.get("topic", "General")
            topic_counts[t] = topic_counts.get(t, 0) + 1
            
        logger.info(f"📊 Bắt đầu quy trình Cloud Benchmark. Mục tiêu: {target_per_topic} câu/topic.")

        for topic in TOPICS:
            current_count = topic_counts.get(topic, 0)
            if current_count >= target_per_topic:
                logger.info(f"✅ Nhóm [{topic}] đã đủ {current_count} câu. Bỏ qua.")
                continue

            needed = target_per_topic - current_count
            needed_requests = (needed + 4) // 5
            
            logger.info(f"🚀 Nhóm [{topic}]: Cần thêm {needed} câu. Thực hiện {needed_requests} lượt gọi AI...")

            for r in range(needed_requests):
                try:
                    # 1. Lấy ngữ cảnh trực tiếp từ Qdrant với số lượng chunk đã tính toán
                    sample_chunks = self.get_random_chunks_from_qdrant(topic, limit=chunks_per_request)
                    if not sample_chunks:
                        logger.warning(f"⚠️ Không tìm thấy chunk nào trên Qdrant cho topic [{topic}]. Có thể cần Ingest dữ liệu mới.")
                        break

                    context_text = "\n\n---\n\n".join([c["content"] for c in sample_chunks])
                    source_files = list(set(c["file"] for c in sample_chunks))

                    # 2. Gọi AI sinh câu hỏi
                    response = await self.chain.ainvoke({"topic": topic, "context": context_text})
                    
                    # 3. Parse và Lưu
                    content = response.content
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
                        logger.info(f"   [{topic}] Batch {r+1}/{needed_requests} thành công.")
                    
                    await asyncio.sleep(7) # An toàn tuyệt đối cho giới hạn 15 RPM của Gemma 4

                except Exception as e:
                    logger.error(f"❌ Lỗi vòng lặp sinh {topic}: {e}")
                    await asyncio.sleep(10)

        logger.info("🎉 HOÀN THÀNH QUY TRÌNH SINH GROUND TRUTH CLOUD NATIVE!")

if __name__ == "__main__":
    generator = CloudBenchmarkGenerator()
    asyncio.run(generator.run())
