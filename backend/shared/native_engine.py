import numpy as np
import logging
import gc
import pickle
import os
import time
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger("NativeEngine")

def torch_unpackbits(xt, num_bits):
    """
    Native PyTorch unpackbits (1-bit -> float32 {-1, 1}).
    Matching np.packbits(..., bitorder='little')
    """
    # Create bit masks for little endian [1, 2, 4, 8, 16, 32, 64, 128]
    masks = (2**torch.arange(8, device=xt.device, dtype=torch.uint8))
    
    # Unpack to (N, packed_dim, 8)
    # Use bitwise_and and check if non-zero
    unpacked = (xt.unsqueeze(-1).bitwise_and(masks).ne(0))
    
    # Flatten and truncate
    # (N, packed_dim * 8)
    res = unpacked.view(xt.shape[0], -1)[:, :num_bits]
    
    # Convert to float signs {-1.0, 1.0}
    return res.float() * 2.0 - 1.0

class NativeEngine:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NativeEngine, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized: return
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 NativeEngine: Initializing on device: {self.device}")
        
        self.current_group = None  # 'raw', 'pq', 'sq8', 'arq'
        self.cache = {}
        self.weights = {}
        self.initialized = True
        self.load_weights()

    def load_weights(self):
        weights_dir = "backend/data" if not os.path.exists("/app/data") else "/app/data"
        weights_path = os.path.join(weights_dir, "model_weights.pkl")
        
        if not os.path.exists(weights_path):
            logger.warning(f"⚠️ NativeEngine: {weights_path} not found. Trying Supabase...")
            try:
                from shared.supabase_client import SupabaseManager
                sm = SupabaseManager()
                os.makedirs(weights_dir, exist_ok=True)
                sm.download_file("centroids", "model_weights.pkl", weights_path)
                logger.info("✅ Downloaded model_weights.pkl from Supabase.")
            except Exception as e:
                logger.error(f"❌ Could not download weights: {e}")

        if os.path.exists(weights_path):
            try:
                with open(weights_path, "rb") as f:
                    self.weights = pickle.load(f)
                logger.info("✅ NativeEngine: Loaded model weights for scoring.")
            except Exception as e:
                logger.error(f"❌ Error loading weights: {e}")
                self.weights = {}
        else:
            logger.warning("⚠️ model_weights.pkl not found. Scoring may fail.")
            self.weights = {}

    def _get_qdrant_client(self):
        from shared.vector_store import VectorStoreManager
        vm = VectorStoreManager()
        return vm.client

    def clear_all_cache(self):
        logger.info("🧹 NativeEngine: Clearing cache and forcing GC...")
        self.cache = {}
        self.current_group = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        for _ in range(3):
            gc.collect()
        time.sleep(0.3)

    def ensure_model(self, model_type, force_reload=False):
        group = None
        if "raw" in model_type or "adaptive" in model_type:
            group = "raw"
        elif "pq" in model_type:
            group = "pq"
        elif "sq8" in model_type:
            group = "sq8"
        elif "arq" in model_type:
            group = "arq"

        if not force_reload and self.current_group == group and self.cache:
            return 0

        logger.info(f"🚀 {'FORCE RELOAD' if force_reload else 'Switching'} NativeEngine to group: {group} (collection: {model_type})")
        self.clear_all_cache()

        start_time = time.perf_counter()
        client = self._get_qdrant_client()

        collection_name = model_type
        if model_type == "vector_adaptive":
            collection_name = "vector_raw"

        # FIX: Với raw, load vector qua scroll. Với compressed, KHÔNG load vector để tránh OOM.
        # Đối với 100k RAW vectors: 100k * 768 * 4 bytes = ~295MB float32 tensor trên GPU
        with_vectors = (group == "raw")

        logger.info(f"📥 Streaming load from: {collection_name} (with_vectors={with_vectors})...")

        # FIX: Dùng streaming để tránh OOM - không đẩy toàn bộ list vào RAM cùng lúc
        total_points = self._stream_load_to_cache(client, collection_name, group, with_vectors)

        if total_points == 0:
            logger.error(f"❌ No data found in {collection_name}")
            return 0

        self.current_group = group

        # Log data integrity
        self._log_data_integrity()

        load_duration = time.perf_counter() - start_time
        logger.info(f"✅ Cache Loaded in {load_duration:.2f}s. Total points: {total_points}")

        # Warm-up GPU sau khi load
        if self.device.type == "cuda":
            self.warmup(model_type, iterations=5)

        return load_duration

    def _stream_load_to_cache(self, client, collection_name, group, with_vectors):
        """
        Nạp dữ liệu theo từng chunk nhỏ và ghi trực tiếp vào mảng đích.
        Tránh OOM bằng cách không tạo danh sách trung gian cho toàn bộ tập dữ liệu.
        """
        # Bước 1: Đếm tổng số điểm để cấp phát trước mảng
        try:
            count_result = client.count(collection_name=collection_name, exact=True)
            total = count_result.count
        except Exception as e:
            logger.error(f"❌ Cannot count collection {collection_name}: {e}")
            return 0

        if total == 0:
            return 0

        logger.info(f"📊 Total points to load: {total}")

        # Bước 2: Cấp phát trước các mảng NumPy (tiết kiệm RAM so với list)
        ids_list = [None] * total
        payloads_list = [None] * total

        # Khởi tạo trước các biến conditional để tránh unbound
        raw_vectors = None
        codes_arr = None
        idx_arr = None
        qjl_arr = None
        qjl_orig_dim = 0
        gamma_arr = np.zeros(total, dtype='float32')
        norm_arr = np.zeros(total, dtype='float32')

        if group == "raw":
            raw_vectors = np.zeros((total, 768), dtype='float32')
        elif group in ("pq", "sq8"):
            pass  # codes_arr sẽ được khởi tạo sau khi biết số chiều
        elif group == "arq":
            pass  # idx_arr, qjl_arr sẽ được khởi tạo sau khi biết số chiều


        # Bước 3: Scroll theo chunk và điền vào mảng đích
        scroll_token = None
        cursor = 0
        CHUNK_SIZE = 1000  # Nhỏ và an toàn

        while True:
            try:
                res, scroll_token = client.scroll(
                    collection_name=collection_name,
                    limit=CHUNK_SIZE,
                    with_vectors=with_vectors,
                    with_payload=True,
                    offset=scroll_token
                )
            except Exception as e:
                logger.error(f"❌ Scroll error at offset {cursor}: {e}")
                break

            if not res:
                break

            chunk_size = len(res)

            for i, point in enumerate(res):
                pos = cursor + i
                if pos >= total:
                    break
                ids_list[pos] = point.id
                pl = dict(point.payload) if point.payload else {}

                if group == "raw":
                    if point.vector:
                        raw_vectors[pos] = point.vector

                elif group in ("pq", "sq8"):
                    codes = pl.pop('codes', [])
                    payloads_list[pos] = pl
                    if codes_arr is None and codes:
                        codes_arr = np.zeros((total, len(codes)), dtype='uint8')
                    if codes_arr is not None and codes:
                        codes_arr[pos] = codes
                    continue  # payload already handled

                elif group == "arq":
                    idx_data = pl.pop('idx', [])
                    qjl_data = pl.pop('qjl', [])  # list of +1/-1 or 1/0 signs
                    gamma_arr[pos] = pl.pop('gamma', 1.0)
                    norm_arr[pos] = pl.pop('orig_norm', 1.0)
                    payloads_list[pos] = pl

                    if idx_arr is None and idx_data:
                        idx_arr = np.zeros((total, len(idx_data)), dtype='uint8')
                        # QJL: 1-bit packing - mỗi 8 chiều nén vào 1 byte
                        qjl_dim = len(qjl_data)
                        qjl_packed_dim = (qjl_dim + 7) // 8  # Số bytes cần thiết
                        qjl_arr = np.zeros((total, qjl_packed_dim), dtype='uint8')
                        qjl_orig_dim = qjl_dim  # Lưu lại chiều gốc để giải nén
                    if idx_arr is not None and idx_data:
                        idx_arr[pos] = idx_data
                        # Pack signs: +1 → 1, -1 → 0 (hoặc bất kỳ giá trị âm → 0)
                        signs = np.array(qjl_data, dtype='int8')
                        bits = (signs > 0).astype('uint8')  # 1 nếu dương, 0 nếu âm
                        qjl_arr[pos] = np.packbits(bits, bitorder='little')
                    continue  # payload already handled

                payloads_list[pos] = pl

            cursor += chunk_size
            # Giải phóng chunk đã xử lý khỏi RAM Python
            del res
            gc.collect()

            if scroll_token is None:
                break

        actual_count = cursor

        # Bước 4: Chuyển mảng NumPy lên GPU
        self.cache = {
            "ids": ids_list[:actual_count],
            "payloads": payloads_list[:actual_count]
        }

        if group == "raw":
            self.cache["vectors"] = torch.from_numpy(raw_vectors[:actual_count]).to(self.device)
            del raw_vectors

        elif group in ("pq", "sq8"):
            if codes_arr is not None:
                self.cache["codes"] = torch.from_numpy(codes_arr[:actual_count]).to(self.device)
                del codes_arr

        elif group == "arq":
            if idx_arr is not None:
                # idx lưu dạng uint8 để tốn ít VRAM (cast sang long chỉ khi search)
                self.cache["idx"] = torch.from_numpy(idx_arr[:actual_count]).to(dtype=torch.uint8, device=self.device)
                # QJL: lưu packed bits (1-bit/chiều) - tiết kiệm 8x VRAM so với int8
                self.cache["qjl_packed"] = torch.from_numpy(qjl_arr[:actual_count]).to(dtype=torch.uint8, device=self.device)
                self.cache["qjl_orig_dim"] = qjl_orig_dim  # Số chiều trước khi pack
                del idx_arr, qjl_arr
            self.cache["gamma"] = torch.from_numpy(gamma_arr[:actual_count]).to(self.device)
            self.cache["orig_norm"] = torch.from_numpy(norm_arr[:actual_count]).to(self.device)
            del gamma_arr, norm_arr

        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        return actual_count

    def _log_data_integrity(self):
        """Log thông tin kiểm tra dữ liệu - dùng Tensor trực tiếp."""
        try:
            if "vectors" in self.cache:
                v = self.cache["vectors"]
                logger.info(f"📊 DATA INTEGRITY: RAW Vectors shape={v.shape}, mean={v.mean().item():.4f}, std={v.std().item():.4f}")
            elif "codes" in self.cache:
                c = self.cache["codes"]
                logger.info(f"📊 DATA INTEGRITY: Codes shape={c.shape}, mean={c.float().mean().item():.4f}")
            elif "idx" in self.cache:
                logger.info(f"📊 DATA INTEGRITY: ARQ idx shape={self.cache['idx'].shape}, gamma mean={self.cache['gamma'].mean().item():.4f}")
        except Exception as e:
            logger.warning(f"⚠️ DATA INTEGRITY check failed: {e}")

    def warmup(self, model_type, iterations=5):
        """Khởi động GPU để tránh Cold Start latency."""
        if self.device.type != "cuda" or not self.cache:
            return

        logger.info(f"🔥 GPU Warm-up: {iterations} dummy searches for {model_type}...")
        dummy_query = np.random.randn(768).astype('float32')
        for _ in range(iterations):
            self.search(model_type, dummy_query, top_k=1, is_warmup=True)
        torch.cuda.synchronize()
        logger.info("✨ GPU Warm-up complete.")

    def get_cache_size_mb(self):
        """Trả về VRAM thực tế đang chiếm dụng (MB)."""
        if not self.cache or self.device.type != "cuda":
            return 0.0
        vram_bytes = torch.cuda.memory_allocated(self.device)
        return round(vram_bytes / (1024 * 1024), 2)

    def search(self, model_type, query_vector, top_k=5, is_warmup=False):
        load_time = 0
        if not is_warmup:
            load_time = self.ensure_model(model_type)

        if not self.cache or not self.weights:
            logger.error("❌ NativeEngine: Cache or Weights missing")
            return [], 0, load_time or 0

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start_search = time.perf_counter()

        query_t = torch.tensor(query_vector, dtype=torch.float32, device=self.device)
        num_points = len(self.cache["ids"])

        try:
            if self.current_group == "raw":
                scores = torch.matmul(self.cache["vectors"], query_t)

            elif self.current_group == "pq":
                # FIX: Triển khai PQ scoring đúng với Asymmetric Distance Computation (ADC)
                if "pq" not in self.weights:
                    logger.error("❌ PQ weights not found!")
                    return [], 0, 0
                w = self.weights['pq']
                centroids = torch.tensor(w['centroids'], dtype=torch.float32, device=self.device)
                # centroids: (M, K, D/M) - M subspaces, K centroids, D/M dims
                # Tính bảng tra cứu (lookup table)
                M = centroids.shape[0]
                D = query_t.shape[0]
                sub_dim = D // M
                scores = torch.zeros(num_points, device=self.device)
                codes = self.cache["codes"]  # (N, M), dtype=uint8
                for m in range(M):
                    q_sub = query_t[m * sub_dim:(m + 1) * sub_dim]  # (D/M,)
                    # Dot product với mỗi centroid của subspace m
                    lut = torch.matmul(centroids[m], q_sub)  # (K,)
                    scores += lut[codes[:, m].long()]

            elif self.current_group == "sq8":
                if "sq8" not in self.weights:
                    logger.error("❌ SQ8 weights not found!")
                    return [], 0, 0
                w = self.weights['sq8']
                
                # Optimized SQ8 Scoring using Lookup Table (ADC)
                num_points = self.cache["codes"].shape[0]
                num_dims = self.cache["codes"].shape[1]
                scores = torch.zeros(num_points, device=self.device)
                
                # Handle possible per-dimension scale/min_val
                w_min = torch.tensor(w['min_val'], dtype=torch.float32, device=self.device)
                w_max = torch.tensor(w['max_val'], dtype=torch.float32, device=self.device)
                scale = (w_max - w_min) / 255.0
                
                # 1. Precompute LUT: (dim, 256)
                # Dựa trên công thức: Q[d] * (v * scale[d] + min_val[d])
                v_range = torch.arange(256, device=self.device, dtype=torch.float32)
                # (D, 1) * (1, 256) * (D, 1) + (D, 1) * (D, 1)? Không, đúng hơn là:
                # Mỗi hàng d của LUT là: query_t[d] * (v_range * scale[d] + w_min[d])
                lut = query_t.unsqueeze(1) * (v_range.unsqueeze(0) * scale.unsqueeze(1) + w_min.unsqueeze(1))
                
                # Batch scoring using indexing
                BATCH_SIZE = 20000 
                # Ensure offsets are on the same device as b_codes
                offsets = torch.arange(0, num_dims * 256, 256, device=self.device)
                flat_lut = lut.flatten()
                
                for start_i in range(0, num_points, BATCH_SIZE):
                    end_i = min(start_i + BATCH_SIZE, num_points)
                    b_codes = self.cache["codes"][start_i:end_i].long()
                    flat_indices = b_codes + offsets.unsqueeze(0)
                    scores[start_i:end_i] = flat_lut[flat_indices].sum(dim=1)

            elif self.current_group == "arq":
                if "arq" not in self.weights:
                    logger.error("❌ ARQ weights not found!")
                    return [], 0, 0
                w = self.weights['arq']

                # idx: uint8 → cast sang long để dùng cho indexing
                idx = self.cache["idx"].long()
                gamma = self.cache["gamma"]
                orig_norm = self.cache["orig_norm"]

                centroids = torch.tensor(w['centroids'], dtype=torch.float32, device=self.device)
                Pi = torch.tensor(w['Pi'], dtype=torch.float32, device=self.device)
                S = torch.tensor(w['S'], dtype=torch.float32, device=self.device)
                alpha = float(w.get('alpha', 1.0))

                q_pi = torch.matmul(Pi, query_t)
                q_s = torch.matmul(S, query_t)

                # Optimized ARQ Scoring with Batching to avoid OOM
                num_points = idx.shape[0]
                scores = torch.zeros(num_points, device=self.device)
                d_range = torch.arange(idx.shape[1], device=self.device)
                
                # Pre-calculate common components (centroids: K, D | q_pi: D)
                # Vì centroids đã là ma trận (K, D), ta nhân trực tiếp với q_pi (D,)
                lookup = centroids * q_pi.unsqueeze(0)
                qjl_orig_dim = self.cache.get("qjl_orig_dim", q_s.shape[0])

                # Batch processing to keep memory overhead low (~50MB per batch vs ~600MB full)
                BATCH_SIZE = 10000 
                for start_i in range(0, num_points, BATCH_SIZE):
                    end_i = min(start_i + BATCH_SIZE, num_points)
                    
                    b_idx = idx[start_i:end_i]
                    b_qjl_packed = self.cache["qjl_packed"][start_i:end_i]
                    b_gamma = gamma[start_i:end_i]
                    b_norm = orig_norm[start_i:end_i]
                    
                    # Term 1: MSE
                    b_mse = lookup[b_idx, d_range].sum(dim=1)
                    
                    # Term 2: QJL (Unpack bits in batch)
                    b_qjl_signs = torch_unpackbits(b_qjl_packed, qjl_orig_dim)
                    b_qjl_dot = torch.matmul(b_qjl_signs, q_s)
                    
                    # Store scores
                    scores[start_i:end_i] = (b_mse + alpha * b_gamma * b_qjl_dot) * b_norm

            else:
                scores = torch.zeros(num_points, device=self.device)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            search_time = (time.perf_counter() - start_search) * 1000

            scores_np = scores.cpu().numpy()
            top_indices = np.argsort(scores_np)[::-1][:top_k]

            results = []
            for i in top_indices:
                idx_int = int(i)
                results.append({
                    "id": self.cache["ids"][idx_int],
                    "score": float(scores_np[idx_int]),
                    "payload": self.cache["payloads"][idx_int]
                })
            return results, search_time, load_time

        except Exception as e:
            import traceback
            logger.error(f"❌ Search Error: {e}")
            logger.error(traceback.format_exc())
            return [], 0, 0
