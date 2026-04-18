import os
from supabase import create_client, Client
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger("Supabase")

class SupabaseManager:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        # Thử lấy từ SUPABASE_KEY (do docker mapping) hoặc trực tiếp từ SERVICE_ROLE_KEY
        key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            logger.warning("Supabase URL or Key not found in environment variables")
        self.supabase: Client = create_client(url, key) if url and key else None

    def list_files(self, bucket: str = "papers"):
        if not self.supabase: return []
        try:
            all_files = []
            offset = 0
            limit = 100
            while True:
                res = self.supabase.storage.from_(bucket).list(options={
                    'limit': limit,
                    'offset': offset,
                    'sortBy': {'column': 'name', 'order': 'asc'}
                })
                if not res:
                    break
                
                names = [f['name'] for f in res if f['name'] != '.emptyFolderPlaceholder']
                all_files.extend(names)
                
                if len(res) < limit:
                    break
                offset += limit
            
            return all_files
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách file từ storage: {e}")
            return []

    # --- System Control (For GitHub Actions Crawler) ---

    def set_stop_signal(self, should_stop: bool = True):
        """Đặt tín hiệu dừng cho crawler."""
        if not self.supabase: return
        try:
            self.supabase.table("system_config").upsert({
                "key": "crawler_stop_signal",
                "value": str(should_stop).lower()
            }).execute()
            logger.info(f"🛑 Tín hiệu dừng Crawler đã được đặt thành: {should_stop}")
        except Exception as e:
            logger.error(f"Lỗi khi đặt stop signal: {e} (Có thể bạn chưa tạo bảng system_config)")

    def check_stop_signal(self) -> bool:
        """Kiểm tra tín hiệu dừng."""
        if not self.supabase: return False
        try:
            res = self.supabase.table("system_config").select("value").eq("key", "crawler_stop_signal").execute()
            if res.data and len(res.data) > 0:
                return res.data[0]["value"] == "true"
        except Exception as e:
            # Nếu lỗi (như thiếu bảng), mặc định không dừng
            pass
        return False

    # --- Papers Management ---

    def upsert_paper(self, paper_id: str, title: str, topic: str, url: str):
        """Lưu hoặc cập nhật thông tin bài báo đã cào (Khớp với Schema mới nhất: id, url, topic)."""
        if not self.supabase: return
        try:
            self.supabase.table("papers").upsert({
                "id": paper_id,
                "title": title,
                "topic": topic,
                "url": url,
                "is_embedded": False
            }).execute()
            logger.info(f"   📊 Metadata đã đồng bộ lên Database.")
        except Exception as e:
            logger.error(f"Lỗi khi lưu bài báo vào Supabase: {e}")

    def update_paper_embedded_status(self, paper_id: str, status: bool = True):
        """Cập nhật trạng thái đã embed của bài báo."""
        if not self.supabase: return
        try:
            self.supabase.table("papers").update({"is_embedded": status}).eq("id", paper_id).execute()
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật trạng thái embed: {e}")

    def get_paper_metadata(self, paper_id: str):
        """Lấy metadata của bài báo từ Database."""
        if not self.supabase: return None
        try:
            res = self.supabase.table("papers").select("*").eq("id", paper_id).execute()
            if res.data and len(res.data) > 0:
                return res.data[0]
        except Exception as e:
            logger.error(f"Lỗi khi đọc metadata bài báo: {e}")
        return None

    def get_all_papers(self):
        """Lấy danh sách toàn bộ bài báo."""
        if not self.supabase: return []
        try:
            res = self.supabase.table("papers").select("*").execute()
            return res.data if res.data else []
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách bài báo: {e}")
            return []

    def download_file(self, bucket: str, filename: str, destination: str):
        if not self.supabase: 
            logger.info(f"Mocking download: {filename}")
            return
        
        res = self.supabase.storage.from_(bucket).download(filename)
        with open(destination, "wb") as f:
            f.write(res)

    def get_file_content(self, bucket: str, filename: str):
        if not self.supabase:
            return None
        return self.supabase.storage.from_(bucket).download(filename)

    def upload_file(self, bucket: str, filename: str, file_path: str):
        if not self.supabase:
            logger.info(f"Mocking upload: {filename}")
            return
        
        with open(file_path, "rb") as f:
            # Sử dụng upsert: True để ghi đè nếu tồn tại
            self.supabase.storage.from_(bucket).upload(
                path=filename, 
                file=f, 
                file_options={"content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "upsert": True}
            )

    def clear_bucket(self, bucket: str):
        """Xóa toàn bộ file trong bucket (Dùng vòng lặp để vượt giới hạn 100 file/lần của API)."""
        if not self.supabase: return
        try:
            total_deleted = 0
            while True:
                # Lấy danh sách 100 file (limit mặc định của Supabase là 100)
                files = self.supabase.storage.from_(bucket).list(options={'limit': 100})
                
                # Lọc ra danh sách tên file hợp lệ (bỏ qua placeholder)
                file_names = [f['name'] for f in files if f['name'] != '.emptyFolderPlaceholder']
                
                if not file_names:
                    break
                
                logger.info(f"🗑️ Đang xóa batch {len(file_names)} file trong bucket '{bucket}'...")
                self.supabase.storage.from_(bucket).remove(file_names)
                total_deleted += len(file_names)
                
                # Nghỉ ngắn giữa các batch để tránh bị rate limit
                import time
                time.sleep(0.5)
            
            if total_deleted > 0:
                logger.info(f"✅ Đã dọn sạch tổng cộng {total_deleted} file trong bucket '{bucket}'.")
        except Exception as e:
            logger.error(f"❌ Lỗi khi dọn dẹp bucket {bucket}: {e}")

    def get_public_url(self, bucket: str, filename: str):
        if not self.supabase:
            return "https://example.com/mock-url.xlsx"
        return self.supabase.storage.from_(bucket).get_public_url(filename)

    def clear_database_table(self, table_name: str):
        """Xóa toàn bộ hàng trong một bảng (cấu hình Supabase cần cho phép DELETE không filter)."""
        if not self.supabase: return
        try:
            # Sử dụng cột phù hợp để lách luật xóa (id cho papers, question cho benchmark)
            pk = "id" if table_name == "papers" else "question"
            self.supabase.table(table_name).delete().neq(pk, "none_existent_id").execute()
            logger.info(f"✅ Đã xóa toàn bộ dữ liệu trong bảng: {table_name}")
        except Exception as e:
            logger.error(f"Lỗi khi xóa bảng {table_name}: {e}")

    # --- Query Classification Cache ---

    def get_query_cache(self, query_text: str):
        """Tra cứu kết quả phân loại từ Supabase."""
        if not self.supabase: return None
        try:
            res = self.supabase.table("query_cache").select("complexity").eq("query_text", query_text).execute()
            if res.data and len(res.data) > 0:
                return res.data[0]["complexity"]
        except Exception as e:
            logger.error(f"Lỗi khi đọc query_cache: {e}")
        return None

    def set_query_cache(self, query_text: str, complexity: str):
        """Lưu kết quả phân loại vào Supabase."""
        if not self.supabase: return
        try:
            self.supabase.table("query_cache").upsert({
                "query_text": query_text,
                "complexity": complexity
            }).execute()
        except Exception as e:
            logger.error(f"Lỗi khi ghi query_cache: {e}")

    # --- Benchmark Queries (Ground Truth) ---
    def get_benchmark_queries(self):
        """Lấy danh sách câu hỏi ground_truth từ Supabase."""
        if not self.supabase: return []
        try:
            res = self.supabase.table("benchmark_queries").select("question, ground_truth, topic, source_files").execute()
            return res.data if res.data else []
        except Exception as e:
            logger.error(f"Lỗi khi đọc benchmark_queries: {e}")
            return []

    def save_benchmark_queries(self, queries: list):
        """Lưu danh sách câu hỏi ground_truth lên Supabase (Hỗ trợ Bulk Upsert)."""
        if not self.supabase: return
        try:
            valid_queries = []
            for q in queries:
                if q.get("question") and q.get("ground_truth"):
                    valid_queries.append({
                        "question": q["question"],
                        "ground_truth": q["ground_truth"],
                        "topic": q.get("topic"),
                        "source_files": q.get("source_files", [])
                    })
            
            if valid_queries:
                self.supabase.table("benchmark_queries").upsert(valid_queries).execute()
                logger.info(f"Đã đồng bộ {len(valid_queries)} câu hỏi lên Supabase Database!")
        except Exception as e:
            logger.error(f"Lỗi khi đồng bộ benchmark_queries: {e}")

    def save_single_benchmark_query(self, question: str, ground_truth: str, topic: str, source_files: list):
        """Lưu một câu hỏi ground_truth duy nhất - Dùng cho Stream Generation."""
        if not self.supabase: return
        try:
            self.supabase.table("benchmark_queries").upsert({
                "question": question,
                "ground_truth": ground_truth,
                "topic": topic,
                "source_files": source_files
            }).execute()
            logger.info(f"✅ Đã lưu câu hỏi mới lên Supabase (Topic: {topic})")
        except Exception as e:
            logger.error(f"Lỗi khi lưu câu hỏi đơn lẻ: {e}")
