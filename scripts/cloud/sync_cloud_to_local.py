import os
import time
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger("Sync-Cloud-To-Local")

def sync():
    # 1. Kết nối Source (Cloud)
    cloud_url = os.getenv("QDRANT_CLOUD_URL")
    cloud_api_key = os.getenv("QDRANT_CLOUD_API_KEY")
    
    if not cloud_url or not cloud_api_key:
        logger.error("❌ Thiếu cấu hình QDRANT_CLOUD_URL hoặc API_KEY trong .env")
        return

    cloud_client = QdrantClient(url=cloud_url, api_key=cloud_api_key)
    
    # 2. Kết nối Destination (Local Docker)
    # Lưu ý: Trong docker network, địa chỉ là 'qdrant', ở ngoài là 'localhost'
    local_url = os.getenv("QDRANT_LOCAL_URL", "http://qdrant:6333")
    local_client = QdrantClient(url=local_url)

    collections = ["vector_raw", "vector_sq8", "vector_pq", "vector_arq"]
    
    logger.info("🚀 Bắt đầu quá trình đồng bộ dữ liệu từ Cloud về Local...")

    for coll_name in collections:
        try:
            # Kiểm tra collection trên Cloud
            cloud_info = cloud_client.get_collection(coll_name)
            logger.info(f"📦 Đang xử lý collection: {coll_name} ({cloud_info.points_count} points)")

            # Đảm bảo collection tồn tại ở Local
            try:
                local_client.get_collection(coll_name)
            except:
                logger.info(f"   🆕 Tạo mới collection {coll_name} ở local...")
                vector_size = 768 # Mặc định cho nomic-embed-text
                local_client.create_collection(
                    collection_name=coll_name,
                    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
                )

            # Cuộn (Scroll) dữ liệu từ Cloud và Upsert vào Local
            offset = None
            total_synced = 0
            batch_size = 500
            
            while True:
                response = cloud_client.scroll(
                    collection_name=coll_name,
                    limit=batch_size,
                    with_payload=True,
                    with_vectors=True,
                    offset=offset
                )
                points, next_offset = response
                
                if not points:
                    break
                
                # Upsert vào local
                local_client.upsert(
                    collection_name=coll_name,
                    points=points
                )
                
                total_synced += len(points)
                offset = next_offset
                logger.info(f"   ✅ Đã sync {total_synced}/{cloud_info.points_count} points cho {coll_name}...")
                
                if not offset:
                    break
            
            logger.info(f"✨ Hoàn thành đồng bộ collection: {coll_name}")

        except Exception as e:
            logger.error(f"❌ Lỗi khi đồng bộ {coll_name}: {e}")

if __name__ == "__main__":
    sync()
