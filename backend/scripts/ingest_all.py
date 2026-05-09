import os
import sys

# Add root to sys.path to allow importing backend services
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.services.ingestion_service import ingestion_service
from backend.services.metadata_service import metadata_service
from tqdm import tqdm

def ingest_all():
    upload_dir = "data/uploads"
    if not os.path.exists(upload_dir):
        print(f"❌ Directory not found: {upload_dir}")
        return

    # Get all PDF files
    files = [f for f in os.listdir(upload_dir) if f.lower().endswith('.pdf')]
    
    # Filter out already indexed files to avoid duplicates
    existing_docs = set(metadata_service.list_documents())
    files_to_process = [f for f in files if f not in existing_docs]
    
    if not files_to_process:
        print("✅ All documents are already indexed.")
        return

    print(f"📦 Found {len(files)} files ({len(files_to_process)} new). Starting ingestion...")
    print("⚠️ This process uses Ollama for embeddings and may take some time.")
    
    for file_name in tqdm(files_to_process, desc="Ingesting PDFs"):
        file_path = os.path.join(upload_dir, file_name)
        try:
            # process_pdf will handle chunking, embedding, and indexing in TQ + Local Metadata
            num_chunks = ingestion_service.process_pdf(file_path, file_name)
        except Exception as e:
            print(f"\n❌ Error processing {file_name}: {e}")

    print(f"\n✨ Successfully ingested new documents!")
    print(f"📊 Total documents now in system: {len(metadata_service.list_documents())}")

if __name__ == "__main__":
    ingest_all()
