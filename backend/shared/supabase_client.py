import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseManager:
    def __init__(self):
        url = os.getenv("SUPABASE_URL")
        # Thử lấy từ SUPABASE_KEY (do docker mapping) hoặc trực tiếp từ SERVICE_ROLE_KEY
        key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            print("WARNING: Supabase URL or Key not found in environment variables")
        self.supabase: Client = create_client(url, key) if url and key else None

    def list_files(self, bucket: str = "papers"):
        if not self.supabase: return []
        res = self.supabase.storage.from_(bucket).list()
        return [f['name'] for f in res if f['name'] != '.emptyFolderPlaceholder']

    def download_file(self, bucket: str, filename: str, destination: str):
        if not self.supabase: 
            print(f"Mocking download: {filename}")
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
            print(f"Mocking upload: {filename}")
            return
        
        with open(file_path, "rb") as f:
            # Sử dụng upsert: True để ghi đè nếu tồn tại
            self.supabase.storage.from_(bucket).upload(
                path=filename, 
                file=f, 
                file_options={"content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "upsert": True}
            )

    def clear_bucket(self, bucket: str):
        """Xóa toàn bộ file trong bucket."""
        if not self.supabase: return
        try:
            files = self.supabase.storage.from_(bucket).list()
            file_names = [f['name'] for f in files if f['name'] != '.emptyFolderPlaceholder']
            if file_names:
                print(f"Đang xóa {len(file_names)} file trong bucket {bucket}...")
                self.supabase.storage.from_(bucket).remove(file_names)
        except Exception as e:
            print(f"Lỗi khi xóa bucket {bucket}: {e}")

    def get_public_url(self, bucket: str, filename: str):
        if not self.supabase:
            return "https://example.com/mock-url.xlsx"
        return self.supabase.storage.from_(bucket).get_public_url(filename)

    # --- Query Classification Cache ---

    def get_query_cache(self, query_text: str):
        """Tra cứu kết quả phân loại từ Supabase."""
        if not self.supabase: return None
        try:
            res = self.supabase.table("query_cache").select("complexity").eq("query_text", query_text).execute()
            if res.data and len(res.data) > 0:
                return res.data[0]["complexity"]
        except Exception as e:
            print(f"Lỗi khi đọc query_cache: {e}")
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
            print(f"Lỗi khi ghi query_cache: {e}")
