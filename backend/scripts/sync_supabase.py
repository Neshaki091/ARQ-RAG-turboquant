import os
from dotenv import load_dotenv
from supabase import create_client, Client
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME = "papers"
DOWNLOAD_DIR = "data/uploads"

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Error: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not found in .env")
    exit(1)

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def download_file(file_info):
    """Function to download a single file, used by ThreadPoolExecutor"""
    file_name = file_info['name']
    
    # Skip directories or non-pdf files if any
    if not file_name or file_name.endswith('/') or not file_name.lower().endswith('.pdf'):
        return None
        
    dest_path = os.path.join(DOWNLOAD_DIR, file_name)
    
    # Skip if already exists
    if os.path.exists(dest_path):
        return f"Skipped: {file_name}"
        
    try:
        res = supabase.storage.from_(BUCKET_NAME).download(file_name)
        with open(dest_path, "wb") as f:
            f.write(res)
        return f"Downloaded: {file_name}"
    except Exception as e:
        return f"Error {file_name}: {e}"

def download_all_papers():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    print(f"🔍 Listing files in bucket: {BUCKET_NAME}...")
    
    all_files = []
    limit = 100
    offset = 0
    
    while True:
        res = supabase.storage.from_(BUCKET_NAME).list(
            path="",
            options={
                "limit": limit,
                "offset": offset,
                "sortBy": {"column": "name", "order": "asc"}
            }
        )
        
        if not res:
            break
            
        all_files.extend(res)
        if len(res) < limit:
            break
            
        offset += limit
        print(f"   Found {len(all_files)} files so far...")

    total_files = len(all_files)
    print(f"✅ Total files to check: {total_files}")
    
    # Use ThreadPoolExecutor for concurrent downloads
    # Setting max_workers to 10 for a good balance between speed and stability
    print(f"🚀 Starting concurrent download (10 workers)...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Wrap with tqdm for progress bar
        list(tqdm(executor.map(download_file, all_files), total=total_files, desc="Syncing papers"))

if __name__ == "__main__":
    download_all_papers()
    print("\n✨ Sync completed!")
