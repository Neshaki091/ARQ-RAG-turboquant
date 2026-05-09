import os
import torch
import numpy as np
import time
import shutil
from TQ_engine_lib.quantizer import TQEngine

class TQService:
    def __init__(self, dim: int = 768, use_ivf: bool = True):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.vector_path = os.path.join(self.data_dir, "Vector", "nomic_768_raw.npy")
        self._raw_vectors_mm = None
        
        print(f"🔧 Initializing TQ Engines. Base data dir: {self.data_dir}")
        # ivf_nprobe=0 => auto-nprobe based on list density (balanced ANN throughput/recall).
        self.engine_4bit = TQEngine(dim=dim, bits=4, use_ivf=use_ivf, ivf_nprobe=0)
        self.engine_2bit = TQEngine(dim=dim, bits=2, use_ivf=use_ivf, ivf_nprobe=0)
        self._configure_balanced_profile()
        
        # Đường dẫn mặc định (sẽ được cập nhật khi load/save)
        self.index_path_4 = os.path.join(self.data_dir, "tq_index_4bit_latest")
        self.index_path_2 = os.path.join(self.data_dir, "tq_index_2bit_latest")

    def _configure_balanced_profile(self):
        # Balanced profile close to FAISS latency/quality envelope (with rerank enabled in search path).
        self.engine_4bit.ivf_target_candidates = 20000
        self.engine_4bit.ivf_nprobe_min = 2
        self.engine_4bit.ivf_nprobe_max = 64

        self.engine_2bit.ivf_target_candidates = 26000
        self.engine_2bit.ivf_nprobe_min = 2
        self.engine_2bit.ivf_nprobe_max = 96

    def _get_raw_vectors(self):
        if self._raw_vectors_mm is None and os.path.exists(self.vector_path):
            self._raw_vectors_mm = np.load(self.vector_path, mmap_mode='r')
        return self._raw_vectors_mm

    def _rerank_exact(self, query_vector: np.ndarray, candidate_ids: np.ndarray, top_k: int):
        if candidate_ids.size == 0:
            return []
        raw = self._get_raw_vectors()
        if raw is None:
            return [{"id": int(i), "score": 0.0} for i in candidate_ids[:top_k]]
        candidates = np.asarray(raw[candidate_ids], dtype=np.float32)
        scores = np.dot(candidates, np.asarray(query_vector, dtype=np.float32))
        keep = min(top_k, scores.shape[0])
        order = np.argpartition(scores, -keep)[-keep:]
        order = order[np.argsort(scores[order])[::-1]]
        return [{"id": int(candidate_ids[i]), "score": float(scores[i])} for i in order]
        
    def index_vectors(self, vectors: np.ndarray):
        vectors_t = torch.from_numpy(vectors).float()
        print("⚡ Quantizing to 4-bit...")
        self.engine_4bit.index(vectors_t)
        print("⚡ Quantizing to 2-bit...")
        self.engine_2bit.index(vectors_t)
        
    def add_vectors(self, vectors: np.ndarray, start_id: int):
        vectors_t = torch.from_numpy(vectors).float()
        for i, v in enumerate(vectors_t):
            self.engine_4bit.add(v, start_id + i)
            self.engine_2bit.add(v, start_id + i)
        
        self.engine_4bit.merge_dynamic_shards()
        self.engine_2bit.merge_dynamic_shards()
        self.save()

    def search(self, query_vector: np.ndarray, top_k: int = 10, bits: int = 4, rerank_mult: int = 8):
        query_t = torch.from_numpy(query_vector).float()
        engine = self.engine_4bit if bits == 4 else self.engine_2bit
        retrieve_k = max(top_k, top_k * max(1, int(rerank_mult)))
        ids, scores = engine.search(query_t, top_k=retrieve_k)
        
        if ids is None or len(ids) == 0:
            return []
        candidate_ids = np.array(ids.tolist(), dtype=np.int64)
        candidate_ids = candidate_ids[candidate_ids >= 0]
        return self._rerank_exact(query_vector, candidate_ids, top_k)

    def search_batch(self, query_vectors: np.ndarray, top_k: int = 10, bits: int = 4, rerank_mult: int = 8):
        engine = self.engine_4bit if bits == 4 else self.engine_2bit
        queries_t = torch.from_numpy(np.asarray(query_vectors, dtype=np.float32)).float()
        retrieve_k = max(top_k, top_k * max(1, int(rerank_mult)))
        batch_results = engine.search_batch(queries_t, top_k=retrieve_k)
        final = []
        for qi, (ids, _scores) in enumerate(batch_results):
            candidate_ids = np.array(ids.tolist(), dtype=np.int64)
            candidate_ids = candidate_ids[candidate_ids >= 0]
            final.append(self._rerank_exact(query_vectors[qi], candidate_ids, top_k))
        return final

    def save(self):
        # Sử dụng timestamp để tránh lỗi file locked trên Windows
        ts = int(time.time())
        new_path_4 = os.path.join(self.data_dir, f"tq_index_4bit_{ts}")
        new_path_2 = os.path.join(self.data_dir, f"tq_index_2bit_{ts}")
        
        print(f"💾 Saving 4-bit index to NEW path: {new_path_4}")
        self.engine_4bit.save_index(new_path_4)
        
        print(f"💾 Saving 2-bit index to NEW path: {new_path_2}")
        self.engine_2bit.save_index(new_path_2)
        
        # Cập nhật config latest
        self.index_path_4 = new_path_4
        self.index_path_2 = new_path_2
        self._save_latest_paths()
        
        # Tự động dọn dẹp các bản cũ để tiết kiệm ổ đĩa
        self.cleanup_old_versions(ts)

    def cleanup_old_versions(self, current_ts=None):
        """Xóa các thư mục index cũ hơn phiên bản hiện tại"""
        print("🧹 Cleaning up old index versions...")
        for folder in os.listdir(self.data_dir):
            if folder.startswith("tq_index_") and str(current_ts) not in folder:
                full_path = os.path.join(self.data_dir, folder)
                try:
                    shutil.rmtree(full_path)
                    print(f"🗑️ Deleted old version: {folder}")
                except Exception:
                    # Nếu vẫn bị khóa thì bỏ qua, lần sau xóa tiếp
                    pass

    def _save_latest_paths(self):
        config_path = os.path.join(self.data_dir, "index_config.json")
        import json
        with open(config_path, "w") as f:
            json.dump({
                "path_4": self.index_path_4,
                "path_2": self.index_path_2
            }, f)

    def load(self):
        config_path = os.path.join(self.data_dir, "index_config.json")
        if not os.path.exists(config_path):
            return False
            
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
            
        self.index_path_4 = config.get("path_4")
        self.index_path_2 = config.get("path_2")
        
        loaded = False
        success = False
        # Load 4-bit
        if self.index_path_4:
            try:
                self.engine_4bit.load_index(self.index_path_4)
                print(f"Successfully loaded 4-bit index from: {self.index_path_4}")
                success = True
            except Exception as e:
                print(f"⚠️ Warning: Could not load 4-bit index: {e}")

        # Load 2-bit
        if self.index_path_2:
            try:
                self.engine_2bit.load_index(self.index_path_2)
                print(f"Successfully loaded 2-bit index from: {self.index_path_2}")
                success = True
            except Exception as e:
                print(f"⚠️ Warning: Could not load 2-bit index: {e}")
        return success

    def search_raw(self, query_vector, top_k=10):
        """Tìm kiếm vét cạn theo Batch (Dành cho 5M+ vector trên 4GB RAM)"""
        import numpy as np
        vector_path = os.path.join(self.data_dir, "Vector", "nomic_768_raw.npy")
        if not os.path.exists(vector_path):
            return []

        # Dùng mmap để truy cập file mà không chiếm RAM
        mmap_vectors = np.load(vector_path, mmap_mode='r')
        total_vectors = mmap_vectors.shape[0]
        
        # Công thức Batch Size tùy chỉnh theo yêu cầu
        ram_gb = 4.0
        highest_unit = 10**(len(str(total_vectors)) - 1) # Ví dụ 28378 -> 10000
        
        # Logic: 0.3 * N * (H*100) / N / (RAM * (RAM/0.4))
        numerator = 0.3 * total_vectors * (highest_unit * 100) / total_vectors
        denominator = ram_gb * (ram_gb / 0.4)
        batch_size = int(numerator / denominator) + 1
        
        print(f"📊 Calculated Batch Size: {batch_size} (N={total_vectors}, H={highest_unit})")
        
        all_top_ids = []
        all_top_scores = []
        
        norm_query = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        
        # Duyệt qua từng Batch để tiết kiệm RAM và tối ưu CPU cache
        for start_idx in range(0, total_vectors, batch_size):
            end_idx = min(start_idx + batch_size, total_vectors)
            
            # Chỉ nạp đúng 1 Batch vào RAM tại thời điểm tính toán
            batch_vectors = mmap_vectors[start_idx:end_idx]
            
            # Dot product (Cosine Similarity)
            scores = np.dot(batch_vectors, norm_query)
            
            # Lấy Top-K của riêng Batch này
            batch_top_k = min(top_k, len(scores))
            batch_indices = np.argsort(scores)[-batch_top_k:][::-1]
            
            for idx in batch_indices:
                all_top_ids.append(start_idx + idx)
                all_top_scores.append(scores[idx])
                
            # Duy trì chỉ Top-K tốt nhất toàn cục sau mỗi Batch để tránh list bị phình to
            if len(all_top_ids) > top_k * 2:
                global_indices = np.argsort(all_top_scores)[-top_k:][::-1]
                all_top_ids = [all_top_ids[i] for i in global_indices]
                all_top_scores = [all_top_scores[i] for i in global_indices]

        # Kết quả cuối cùng sau khi quét qua toàn bộ 5M vector (mô phỏng)
        final_indices = np.argsort(all_top_scores)[-top_k:][::-1]
        return [{"id": int(all_top_ids[i]), "score": float(all_top_scores[i])} for i in final_indices]

# Global instance
tq_service = TQService()
