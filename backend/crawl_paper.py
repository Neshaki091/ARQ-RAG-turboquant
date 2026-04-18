import os
import time
import asyncio
import httpx
import re
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
import xml.etree.ElementTree as ET
import logging

# Thiết lập Logging cho UI
logger = logging.getLogger("Crawler")
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Không tìm thấy SUPABASE_URL hoặc SUPABASE_SERVICE_ROLE_KEY!")
    exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
METADATA_DIR = Path(__file__).parent / "document" / "metadata"

TARGET_TOTAL = 1100
QUERIES = [
    {"name": 'TurboQuant', "query": 'all:"TurboQuant" OR all:"PolarQuant" OR all:"Quantized Johnson-Lindenstrauss"'},
    {"name": 'ARQ', "query": 'all:"Asynchronous RAG" OR all:"ARQ" OR all:"Adaptive Reranking"'},
    {"name": 'PQ', "query": 'all:"Product Quantization" OR all:"Vector Quantization" OR all:"Product quantization for nearest neighbor search"'},
    {"name": 'RAG', "query": 'all:"Retrieval-Augmented Generation" OR all:"RAG system" OR all:"Knowledge-intensive NLP"'},
    {"name": 'ML_Optimization', "query": 'all:"Machine Learning Optimization" OR all:"Model Compression" OR all:"Vector Database"'},
    {"name": 'LLM_Inference', "query": 'all:"Large Language Model Inference" OR all:"Efficient LLM" OR all:"KV Cache"'}
]

def clean_filename(s: str) -> str:
    return re.sub(r'[\/\\?%*:|"<>]', '-', s).strip()

def extract_id(url: str) -> str:
    # Trích xuất ID từ URL dạng http://arxiv.org/abs/2203.08381v1 -> 2203.08381v1
    if not url: return "unknown"
    return url.split('/')[-1]

async def crawl_arxiv():
    logger.info(f"🔍 Bắt đầu cào ~{TARGET_TOTAL} bài báo từ arXiv...")
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Đảm bảo bucket 'papers' tồn tại
    try:
        buckets = supabase.storage.list_buckets()
        if not any(b.name == 'papers' for b in buckets):
            logger.info("⚠️ Bucket 'papers' chưa tồn tại. Đang tạo...")
            supabase.storage.create_bucket('papers', options={'public': True})
    except Exception as e:
        logger.warning(f"⚠️ Cảnh báo khi kiểm tra bucket: {e}")

    total_saved = 0
    saved_ids_in_session = set()
    max_per_query = 300
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ArXivCrawler/2.0; +https://github.com/your-repo)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }

    # 1. Lấy danh sách ID đã tồn tại trong DB để tránh tải trùng
    try:
        res_db = supabase.table("papers").select("id").execute()
        existing_ids_in_db = {row['id'] for row in res_db.data}
        logger.info(f"💾 Database hiện có {len(existing_ids_in_db)} bài. Sẽ tự động bỏ qua các bài này.")
    except Exception as e:
        existing_ids_in_db = set()
        logger.warning(f"⚠️ Không thể kiểm tra ID từ DB: {e}")

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True, headers=headers) as client:
        for q in QUERIES:
            if total_saved >= TARGET_TOTAL: break
            
            logger.info(f"\n⏳ Đang tìm kiếm chủ đề: [{q['name']}]...")
            count_for_topic = 0
            
            # 2. Phân trang: Mỗi lần lấy 100 kết quả, lặp lại cho đến max_per_query
            for start_index in range(0, max_per_query, 100):
                if total_saved >= TARGET_TOTAL: break
                
                logger.info(f"   📑 Trang {start_index // 100 + 1} (start={start_index})...")
                encoded_query = urllib.parse.quote(q['query'])
                api_url = f"https://export.arxiv.org/api/query?search_query={encoded_query}&start={start_index}&max_results=100&sortBy=relevance&sortOrder=descending"
                
                try:
                    res = await client.get(api_url)
                    if res.status_code == 429:
                        logger.warning("   ⚠️ Rate limit API (429). Nghỉ 60s...")
                        await asyncio.sleep(60)
                        res = await client.get(api_url)
                    
                    res.raise_for_status()
                    root = ET.fromstring(res.text)
                    entries = root.findall('{http://www.w3.org/2005/Atom}entry')
                    
                    if not entries:
                        logger.info("   ℹ️ Không còn kết quả cho chủ đề này.")
                        break

                    for entry in entries:
                        if total_saved >= TARGET_TOTAL: break
                        
                        ns = {'atom': 'http://www.w3.org/2005/Atom'}
                        
                        id_elem = entry.find('atom:id', ns)
                        link_id = id_elem.text.strip() if id_elem is not None else ""
                        arxiv_id = extract_id(link_id)
                        
                        # KIỂM TRA TRÙNG LẶP
                        if arxiv_id in existing_ids_in_db or arxiv_id in saved_ids_in_session:
                            continue

                        title_elem = entry.find('atom:title', ns)
                        title = title_elem.text.replace('\n', ' ').strip() if title_elem is not None else 'Untitled'
                        
                        summary_elem = entry.find('atom:summary', ns)
                        summary = summary_elem.text.replace('\n', ' ').strip() if summary_elem is not None else ''
                        
                        published_elem = entry.find('atom:published', ns)
                        published = published_elem.text.split('T')[0] if published_elem is not None else 'Unknown'
                        
                        authors = [a.find('atom:name', ns).text.strip() for a in entry.findall('atom:author', ns) if a.find('atom:name', ns) is not None]

                        # 3. Tải PDF (Dùng link trực tiếp, ArXiv thường redirect .pdf về không đuôi)
                        pdf_url = f"https://export.arxiv.org/pdf/{arxiv_id}"
                        safe_title = clean_filename(title)[:50]
                        pdf_filename = f"{arxiv_id}_{safe_title}.pdf"
                        supabase_url_link = f"{SUPABASE_URL}/storage/v1/object/public/papers/{pdf_filename}"

                        logger.info(f"      📥 Đang tải: {arxiv_id}...")
                        
                        try:
                            # Nghỉ ngẫu nhiên từ 8-12 giây (để đạt tiến độ ~1100 bài / 3 tiếng)
                            import random
                            delay = random.uniform(8, 12)
                            await asyncio.sleep(delay)
                            
                            pdf_res = await client.get(pdf_url, timeout=45.0)
                            
                            # Xử lý Rate Limit khi tải PDF
                            if pdf_res.status_code == 429:
                                logger.warning("      ⚠️ Bị chặn (429) khi tải PDF. Nghỉ 120s để 'xả'...")
                                await asyncio.sleep(120)
                                pdf_res = await client.get(pdf_url, timeout=45.0)

                            if pdf_res.status_code == 200:
                                # 4. Upload lên Supabase Storage
                                upload_res = supabase.storage.from_("papers").upload(
                                    path=pdf_filename,
                                    file=pdf_res.content,
                                    file_options={"content-type": "application/pdf", "upsert": "true"}
                                )
                                
                                if hasattr(upload_res, 'error') and upload_res.error:
                                    logger.error(f"      ❌ Lỗi Storage: {upload_res.error}")
                                    continue

                                # 5. Lưu Metadata vào Database
                                try:
                                    supabase.table("papers").upsert({
                                        "id": arxiv_id,
                                        "title": title,
                                        "topic": q['name'],
                                        "url": pdf_url,
                                        "is_embedded": False
                                    }).execute()
                                    
                                    # Lưu Metadata TXT cục bộ làm backup
                                    txt_content = f"Tiêu đề: {title}\nTác giả: {', '.join(authors)}\nNgày: {published}\nLink: {pdf_url}\nChủ đề: {q['name']}\n\nAbstract: {summary}"
                                    with open(METADATA_DIR / f"{arxiv_id}.txt", "w", encoding="utf-8") as f:
                                        f.write(txt_content)

                                    saved_ids_in_session.add(arxiv_id)
                                    count_for_topic += 1
                                    total_saved += 1
                                    logger.info(f"      ✅ [OK] Đã lưu {arxiv_id}. Tổng mới: {total_saved}")
                                    
                                except Exception as db_err:
                                    logger.error(f"      ❌ Lỗi Database: {db_err}")
                            else:
                                logger.error(f"      ❌ PDF Error {pdf_res.status_code} cho {arxiv_id}")
                                
                        except Exception as e:
                            logger.error(f"      ❌ Lỗi kết nối PDF {arxiv_id}: {e}")

                except Exception as e:
                    logger.error(f"❌ Lỗi API ArXiv tại start={start_index}: {e}")
                    break
            
            logger.info(f"🏁 Đã xong chủ đề [{q['name']}]. Cào được {count_for_topic} bài mới.")

    logger.info(f"\n🎉 HOÀN THÀNH! Tổng cộng đã thêm mới {total_saved} bài báo.")

if __name__ == "__main__":
    asyncio.run(crawl_arxiv())
