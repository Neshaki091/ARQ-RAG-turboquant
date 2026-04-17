import os
import json
import asyncio
import logging
import random
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import sys

# Đảm bảo có thể import module từ thư mục cha
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from shared.supabase_client import SupabaseManager

load_dotenv()

# Thiết lập Logger để hiện lên UI Dashboard
logger = logging.getLogger("Benchmark")

async def generate_benchmark_topic_based():
    sm = SupabaseManager()
    
    # 1. Tải danh sách Chunks
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "chunks.json")
    
    if not os.path.exists(data_path):
        logger.error(f"❌ Không tìm thấy {data_path}. Vui lòng chạy Ingest trước.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    # 2. Lấy danh sách bài báo đã được Embed thành công từ Database
    papers = sm.get_all_papers()
    embedded_paper_ids = {p["id"] for p in papers if p.get("is_embedded")}
    
    if not embedded_paper_ids:
        logger.warning("⚠️ Không tìm thấy bài báo nào có trạng thái 'is_embedded = True'.")
        logger.info("Gợi ý: Cần chạy quy trình Ingest/Embed cho các PDF đã cào trước khi sinh bộ đề.")
        return

    # Lọc chunks chỉ lấy từ các bài báo đã embed
    valid_chunks = [c for c in all_chunks if c.get("file", "").split("_")[0] in embedded_paper_ids]
    
    if not valid_chunks:
        logger.error("❌ Không có chunk nào từ các bài báo đã embed.")
        return

    # 3. Phân nhóm Chunks theo Topic
    # Nếu chunk chưa có nhãn topic (do chưa chạy Ingest mới), ta tra cứu từ Database
    topic_map = {p["id"]: p["topic"] for p in papers}
    
    chunks_by_topic = {}
    for c in valid_chunks:
        arxiv_id = c.get("file", "").split("_")[0]
        topic = c.get("topic") or topic_map.get(arxiv_id, "General")
        if topic not in chunks_by_topic:
            chunks_by_topic[topic] = []
        chunks_by_topic[topic].append(c)

    logger.info(f"📊 Đã sẵn sàng dữ liệu cho {len(chunks_by_topic)} nhóm chủ đề.")

    # 4. Khởi tạo LLM (Groq Llama 3.3 70B)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an elite research evaluator. Based on the provided research paper fragments (chunks), "
                   "generate ONE high-quality technical question and its precise ground truth answer. "
                   "The question must be challenging and suitable for academic benchmarking. "
                   "Focus on the methodology, results, or specific technical innovations mentioned. "
                   "Format your response as a valid JSON object: "
                   '{{"question": "...", "ground_truth": "..."}}'),
        ("human", "Topic: {topic}\n\nContext Chunks:\n{context}")
    ])

    chain = prompt | llm

    # 5. Vòng lặp sinh theo từng Topic
    TOPICS = ["TurboQuant", "ARQ", "PQ", "RAG", "ML_Optimization", "LLM_Inference"]
    TARGET_PER_TOPIC = 100

    # Lấy thống kê hiện tại từ Supabase
    existing_queries = sm.get_benchmark_queries()
    topic_counts = {}
    for q in existing_queries:
        t = q.get("topic", "General")
        topic_counts[t] = topic_counts.get(t, 0) + 1

    for topic in TOPICS:
        current_count = topic_counts.get(topic, 0)
        if current_count >= TARGET_PER_TOPIC:
            logger.info(f"✅ Nhóm [{topic}] đã đủ {current_count} câu hỏi. Bỏ qua.")
            continue

        topic_chunks = chunks_by_topic.get(topic, [])
        if not topic_chunks:
            logger.warning(f"⚠️ Nhóm [{topic}] không có dữ liệu (chưa cào bài báo nào?).")
            continue

        needed = TARGET_PER_TOPIC - current_count
        logger.info(f"🚀 Bắt đầu sinh thêm {needed} câu hỏi cho nhóm [{topic}]...")

        for i in range(needed):
            try:
                # Chọn 2-3 chunk ngẫu nhiên từ nhóm để làm ngữ cảnh sinh 1 câu
                # (Đảm bảo tính đa dạng thay vì chỉ dùng 1 chunk)
                sample_chunks = random.sample(topic_chunks, min(2, len(topic_chunks)))
                context_text = "\n\n---\n\n".join([c["content"] for c in sample_chunks])
                source_files = list(set(c["file"] for c in sample_chunks))

                # Gọi AI
                response = await chain.ainvoke({"topic": topic, "context": context_text})
                
                # Parse JSON
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                data = json.loads(content.strip())
                
                if "question" in data and "ground_truth" in data:
                    # Lưu trực tiếp lên Supabase (Sinh 1 Lưu 1)
                    sm.save_single_benchmark_query(
                        question=data["question"],
                        ground_truth=data["ground_truth"],
                        topic=topic,
                        source_files=source_files
                    )
                    logger.info(f"   [{topic}] Tiến độ: {current_count + i + 1}/{TARGET_PER_TOPIC}")
                
                # Rate limit safety (Groq free tier has TPM limits)
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"❌ Lỗi khi sinh câu hỏi nhóm {topic}: {e}")
                await asyncio.sleep(5)

    logger.info("🎉 HOÀN THÀNH TOÀN BỘ QUY TRÌNH SINH GROUND TRUTH!")

if __name__ == "__main__":
    asyncio.run(generate_benchmark_topic_based())
