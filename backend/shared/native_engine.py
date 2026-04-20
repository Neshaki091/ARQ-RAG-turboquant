import numpy as np
import logging
import gc
import pickle
import os
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger("NativeEngine")

class NativeEngine:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NativeEngine, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized: return
        self.current_group = None # 'raw', 'pq', 'sq8', 'arq'
        self.cache = {} 
        self.weights = {}
        self.initialized = True
        self.load_weights()

    def load_weights(self):
        weights_dir = "backend/data"
        weights_path = os.path.join(weights_dir, "model_weights.pkl")
        
        # 1. Tự động tải từ Cloud nếu chưa có local
        if not os.path.exists(weights_path):
            logger.info(f"⚠️ NativeEngine: {weights_path} không tồn tại. Đang tự động tải từ Supabase...")
            try:
                from shared.supabase_client import SupabaseManager
                sm = SupabaseManager()
                os.makedirs(weights_dir, exist_ok=True)
                # Tải file từ bucket 'centroids'
                sm.download_file("centroids", "model_weights.pkl", weights_path)
                logger.info("✅ NativeEngine: Đã tải thành công model_weights.pkl từ bucket 'centroids'")
            except Exception as e:
                logger.error(f"❌ NativeEngine: Không thể tự động tải weights: {e}")

        # 2. Đọc file weights vào RAM
        if os.path.exists(weights_path):
            try:
                with open(weights_path, "rb") as f:
                    self.weights = pickle.load(f)
                logger.info("✅ NativeEngine: Loaded model weights for scoring.")
            except Exception as e:
                logger.error(f"❌ NativeEngine: Error loading weights: {e}")
                self.weights = {}
        else:
            logger.warning("⚠️ NativeEngine: Vẫn không tìm thấy model_weights.pkl. Quá trình tính toán có thể bị lỗi.")
            self.weights = {}

    def _get_qdrant_client(self):
        from shared.vector_store import VectorStoreManager
        vm = VectorStoreManager()
        return vm.client

    def ensure_model(self, model_type):
        group = None
        if model_type in ["vector_raw", "vector_adaptive"]:
            group = "raw"
        elif model_type == "vector_pq":
            group = "pq"
        elif model_type == "vector_sq8":
            group = "sq8"
        elif model_type == "vector_arq":
            group = "arq"
        
        if self.current_group == group and self.cache:
            return 0

        logger.info(f"🚀 Switching Native Engine cache to group: {group} (Requested: {model_type})")
        
        self.cache = {}
        gc.collect()
        time.sleep(0.1)

        start_time = time.time()
        client = self._get_qdrant_client()
        
        collection_name = model_type
        if model_type == "vector_adaptive": collection_name = "vector_raw"
        
        logger.info(f"📥 Loading all points from Qdrant collection: {collection_name}...")
        
        all_points = []
        scroll_token = None
        while True:
            res, scroll_token = client.scroll(
                collection_name=collection_name,
                limit=2000,
                with_vectors=True,
                with_payload=True,
                offset=scroll_token
            )
            if not res: break
            all_points.extend(res)
            if scroll_token is None: break
        
        if not all_points:
            logger.error(f"❌ No data found in {collection_name}")
            return 0

        self._process_points_to_cache(group, all_points)
        self.current_group = group
        
        load_duration = time.time() - start_time
        logger.info(f"✅ Cache Loaded in {load_duration:.2f}s. Total points: {len(all_points)}")
        return load_duration

    def _process_points_to_cache(self, group, points):
        self.cache = {
            "ids": [p.id for p in points],
            "payloads": [p.payload for p in points]
        }
        
        if group == "raw":
            self.cache["vectors"] = np.array([p.vector for p in points], dtype='float32')
        elif group == "pq":
            self.cache["codes"] = np.array([p.payload['codes'] for p in points], dtype='uint8')
        elif group == "sq8":
            self.cache["codes"] = np.array([p.payload['codes'] for p in points], dtype='uint8')
        elif group == "arq":
            self.cache["idx"] = np.array([p.payload['idx'] for p in points], dtype='int32')
            self.cache["qjl"] = np.array([p.payload['qjl'] for p in points], dtype='int8')
            self.cache["gamma"] = np.array([p.payload['gamma'] for p in points], dtype='float32')
            self.cache["orig_norm"] = np.array([p.payload.get('orig_norm', 1.0) for p in points], dtype='float32')

    def search(self, model_type, query_vector, top_k=5):
        load_time = self.ensure_model(model_type)
        
        if not self.cache or not self.weights:
            logger.error(f"❌ NativeEngine: Cache or Weights missing")
            return [], 0, load_time or 0

        start_search = time.time()
        query_vector = np.asarray(query_vector, dtype='float32')
        num_points = len(self.cache["ids"])
        scores = np.zeros(num_points)

        try:
            if self.current_group == "raw":
                vectors = self.cache["vectors"]
                scores = np.dot(vectors, query_vector)
                
            elif self.current_group == "pq":
                if "pq" not in self.weights: return [], 0, 0
                centroids = self.weights['pq']['centroids']
                m = len(centroids)
                ds = 768 // m
                codes = self.cache["codes"]
                total_dist = np.zeros(num_points)
                for i in range(m):
                    q_part = query_vector[i*ds : (i+1)*ds]
                    diff = centroids[i] - q_part
                    dist_table = np.sum(diff**2, axis=1)
                    total_dist += dist_table[codes[:, i]]
                scores = -total_dist
                
            elif self.current_group == "sq8":
                if "sq8" not in self.weights: return [], 0, 0
                w = self.weights['sq8']
                codes = self.cache["codes"].astype(float)
                X_approx = (codes / 255.0) * (w['max_val'] - w['min_val']) + w['min_val']
                scores = np.dot(X_approx, query_vector)

            elif self.current_group == "arq":
                if "arq" not in self.weights: return [], 0, 0
                w = self.weights['arq']
                idx = self.cache["idx"]
                qjl = self.cache["qjl"].astype(float)
                gamma = self.cache["gamma"]
                orig_norm = self.cache["orig_norm"]

                q_pi = np.dot(w['Pi'], query_vector)
                mse_scores = np.dot(w['centroids'][idx], q_pi)
                q_s = np.dot(w['S'], query_vector)
                qjl_dot = np.dot(qjl, q_s)
                scores = (mse_scores + w['alpha'] * gamma * qjl_dot) * orig_norm

            search_time = (time.time() - start_search) * 1000
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for i in top_indices:
                idx_int = int(i)
                results.append({
                    "id": self.cache["ids"][idx_int],
                    "score": float(scores[idx_int]),
                    "payload": self.cache["payloads"][idx_int]
                })
            return results, search_time, load_time

        except Exception as e:
            logger.error(f"❌ Search Error: {e}")
            return [], 0, 0
