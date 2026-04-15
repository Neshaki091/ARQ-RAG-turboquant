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

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Không tìm thấy SUPABASE_URL hoặc SUPABASE_SERVICE_ROLE_KEY!")
    exit(1)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
OUTPUT_DIR = Path(__file__).parent / "document"
METADATA_DIR = OUTPUT_DIR / "metadata"

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
    return url.split('/')[-1] if url else "unknown"

async def crawl_arxiv():
    print(f"🔍 Bắt đầu cào ~{TARGET_TOTAL} bài báo từ arXiv (PYTHON VERSION)...\n")
    
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    
    total_saved = 0
    saved_ids = set()
    max_per_query = 300
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for q in QUERIES:
            if total_saved >= TARGET_TOTAL: break
            
            print(f"\n⏳ Đang tìm kiếm chủ đề: [{q['name']}]...")
            encoded_query = urllib.parse.quote(q['query'])
            api_url = f"http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_per_query}&sortBy=relevance&sortOrder=descending"
            
            try:
                res = await client.get(api_url)
                res.raise_for_status()
                
                # Parse XML Atom Feed
                root = ET.fromstring(res.text)
                entries = root.findall('{http://www.w3.org/2005/Atom}entry')
                
                count_for_query = 0
                for entry in entries:
                    if total_saved >= TARGET_TOTAL: break
                    
                    # Extract fields (Safe extraction)
                    ns = {'atom': 'http://www.w3.org/2005/Atom'}
                    
                    title_elem = entry.find('atom:title', ns)
                    title = title_elem.text.replace('\n', ' ').strip() if title_elem is not None and title_elem.text else 'Untitled'
                    
                    summary_elem = entry.find('atom:summary', ns)
                    summary = summary_elem.text.replace('\n', ' ').strip() if summary_elem is not None and summary_elem.text else ''
                    
                    id_elem = entry.find('atom:id', ns)
                    link_id = id_elem.text.strip() if id_elem is not None and id_elem.text else "unknown"
                    
                    published_elem = entry.find('atom:published', ns)
                    published = published_elem.text.split('T')[0] if published_elem is not None and published_elem.text else 'Unknown'
                    
                    authors = []
                    for author in entry.findall('atom:author', ns):
                        name_elem = author.find('atom:name', ns)
                        if name_elem is not None and name_elem.text:
                            authors.append(name_elem.text.strip())
                    
                    arxiv_id = extract_id(link_id)
                    if arxiv_id in saved_ids: continue
                    saved_ids.add(arxiv_id)
                    
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    base_filename = f"{arxiv_id}_{clean_filename(title)[:50]}"
                    pdf_filename = f"{base_filename}.pdf"
                    
                    # Get Public URL from Supabase
                    supabase_url_link = f"{SUPABASE_URL}/storage/v1/object/public/papers/{pdf_filename}"
                    
                    print(f" 📥 Đang tải PDF: {arxiv_id} ({count_for_query + 1}) - {title[:30]}...")
                    
                    try:
                        pdf_res = await client.get(pdf_url)
                        if pdf_res.status_code == 200:
                            # Upload to Supabase
                            print(f"   ⬆️ Đang Upload PDF lên Supabase Storage...")
                            supabase.storage.from_("papers").upload(
                                path=pdf_filename,
                                file=pdf_res.content,
                                file_options={"content-type": "application/pdf", "x-upsert": "true"}
                            )
                            
                            # Save Metadata TXT
                            txt_content = f"""Tiêu đề bài báo: {title}
Tác giả: {', '.join(authors)}
Ngày xuất bản: {published}
Nền tảng: arXiv
Link gốc: {link_id}
Link PDF: {pdf_url}
Link Lưu Trữ (Supabase): {supabase_url_link}
Chủ đề Crawler: {q['name']}

--- TÓM TẮT (ABSTRACT) ---
{summary}
"""
                            txt_path = METADATA_DIR / f"{base_filename}.txt"
                            with open(txt_path, "w", encoding="utf-8") as f:
                                f.write(txt_content)
                                
                            print(f"   ✅ Đã xử lý xong bài báo.")
                            count_for_query += 1
                            total_saved += 1
                        else:
                            print(f"   ❌ Bỏ qua PDF (lỗi HTTP {pdf_res.status_code})")
                            
                    except Exception as e:
                        print(f"   ❌ Lỗi khi tải/upload: {e}")
                    
                    # Delay 2s to avoid ban
                    await asyncio.sleep(2.0)
                
                print(f"✅ Đã cào xong {count_for_query} bài báo cho [{q['name']}]. Tổng: {total_saved}/{TARGET_TOTAL}")
                
            except Exception as e:
                print(f"❌ Lỗi API khi tìm [{q['name']}]: {e}")

    print(f"\n🎉 HOÀN THÀNH TOÀN BỘ! Đã tải {total_saved} bài báo.")

if __name__ == "__main__":
    asyncio.run(crawl_arxiv())
