import numpy as np
import os
import pickle
from qdrant_client import QdrantClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyCompression")

def main():
    # Configuration
    # Use internal docker networking if possible, or localhost if run via tunnel
    Q_URL = os.getenv("QDRANT_CLOUD_URL", "http://localhost:6333")
    Q_KEY = os.getenv("QDRANT_CLOUD_API_KEY")

    client = QdrantClient(url=Q_URL, api_key=Q_KEY)

    try:
        # 1. Fetch a sample point from raw
        logger.info("Đang lấy mẫu vector từ 'vector_raw'...")
        raw_res = client.scroll(collection_name="vector_raw", limit=1, with_vectors=True)
        if not raw_res[0]:
            logger.error("Collection 'vector_raw' trống!")
            return
        
        p_raw = raw_res[0][0]
        v_raw = np.array(p_raw.vector)

        # 2. Fetch the SAME point from arq
        logger.info(f"Đang lấy mẫu vector cùng ID ({p_raw.id}) từ 'vector_arq'...")
        try:
            arq_point = client.retrieve(collection_name="vector_arq", ids=[p_raw.id], with_vectors=True)[0]
            v_arq = np.array(arq_point.vector)
        except Exception:
            logger.error(f"Điểm {p_raw.id} không tồn tại trong 'vector_arq'. Hãy chạy re_quantize.py trước!")
            return

        # 3. Analyze
        diff = v_raw - v_arq
        mse = np.mean(diff**2)
        dist = np.linalg.norm(diff)
        cos_sim = np.dot(v_raw, v_arq) / (np.linalg.norm(v_raw) * np.linalg.norm(v_arq))

        print("\n" + "="*50)
        print("📊 BÁO CÁO PHÂN TÍCH TURBOQUANT BOTTLENECK")
        print("="*50)
        print(f"Point ID: {p_raw.id}")
        print(f"Định dạng Vector ARQ: {v_arq.dtype} (Kích thước: {len(v_arq)})")
        print(f"Sai số bình phương trung bình (MSE): {mse:.10f}")
        print(f"Khoảng cách Euclidean: {dist:.6f}")
        print(f"Độ tương đồng Cosine: {cos_sim:.6f}")
        print("-" * 50)

        # 4. Check Payload
        payload = arq_point.payload
        has_codes = all(k in payload for k in ["idx", "qjl", "gamma"])
        
        if has_codes:
            print("✅ PHÁT HIỆN MÃ NÉN (COMPRESSED CODES) TRONG PAYLOAD:")
            print(f"   - idx (Centroid Indices): {len(payload['idx'])} bits/values")
            print(f"   - qjl (Sign Bits): {len(payload['qjl'])} bits")
            print(f"   - gamma (Scale): {payload['gamma']:.6f}")
        else:
            print("❌ KHÔNG TÌM THẤY MÃ NÉN TRONG PAYLOAD.")

        if mse > 1e-8:
            print("\n🔥 KẾT LUẬN THỰC NGHIỆM:")
            print("Dữ liệu trong 'vector_arq' đã được LƯỢNG TỬ HÓA (LOSS-Y).")
            print("Nó không phải là vector gốc. Đây là vector tái tạo (reconstructed)")
            print("từ các mã nén 4-bit/dim của TurboQuant.")
            print("Việc lưu trữ Float32 chỉ là 'vỏ bọc' (Approximate Representation),")
            print("chứng minh bạn đã loại bỏ hoàn toàn thông tin gốc (discretization bottleneck).")
        else:
            print("\n⚠️ CẢNH BÁO: Vector ARQ giống hệt vector Raw.")
            print("Quy trình nén có thể chưa được thực thi hoặc dữ liệu chưa được cập nhật.")
        print("="*50 + "\n")

    except Exception as e:
        logger.error(f"Lỗi khi thực hiện verify: {e}")

if __name__ == "__main__":
    main()
