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
            self.supabase.storage.from_(bucket).upload(filename, f, {"content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"})

    def get_public_url(self, bucket: str, filename: str):
        if not self.supabase:
            return "https://example.com/mock-url.xlsx"
        return self.supabase.storage.from_(bucket).get_public_url(filename)
