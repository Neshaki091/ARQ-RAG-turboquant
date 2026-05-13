import os
import sys
import torch
import numpy as np
import time
import shutil
import json
import threading
from collections import OrderedDict
from TQ_engine_lib.quantizer import TQEngine
from .metadata_service import metadata_service
from .rerank_service import rerank_service

class UserEngineManager:
    """Quản lý các Engine của User theo cơ chế Online Session (LRU Cache)"""
    def __init__(self, dim=768, max_active_sessions=100):
        self.dim = dim
        self.max_active_sessions = max_active_sessions
        # Dùng OrderedDict để làm LRU Cache
        self.engines = OrderedDict() 
        self.lock = threading.Lock()
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.user_index_root = os.path.join(self.data_dir, "user_indexes")
        os.makedirs(self.user_index_root, exist_ok=True)

    def get_engine(self, user_id: int) -> TQEngine:
        with self.lock:
            # Nếu đã có trong cache, đẩy lên đầu (vừa mới dùng)
            if user_id in self.engines:
                self.engines.move_to_end(user_id)
                return self.engines[user_id]
            
            # Nếu vượt quá giới hạn session, xóa Engine cũ nhất
            if len(self.engines) >= self.max_active_sessions:
                oldest_user_id, _ = self.engines.popitem(last=False)
                print(f"SESSION: Evicting user {oldest_user_id} to free memory.")

            # Tạo mới hoặc load từ đĩa
            print(f"SESSION: Loading index for user {user_id}...")
            engine = TQEngine(dim=self.dim, bits=4, use_ivf=True, ivf_nlist=16)
            
            # Đường dẫn index của riêng user này
            user_path = os.path.join(self.user_index_root, f"user_{user_id}")
            if os.path.exists(user_path):
                try:
                    engine.load_index(user_path)
                except Exception as e:
                    print(f"WARNING: Could not load index for user {user_id}: {e}")
            
            self.engines[user_id] = engine
            return engine

    def save_user_engine(self, user_id: int):
        with self.lock:
            if user_id in self.engines:
                user_path = os.path.join(self.user_index_root, f"user_{user_id}")
                self.engines[user_id].save_index(user_path)
                print(f"SESSION: Saved index for user {user_id}")

class TQService:
    def __init__(self, dim: int = 768):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        
        # Quản lý hàng nghìn User Index bằng Manager
        self.user_manager = UserEngineManager(dim=dim, max_active_sessions=100)
        
        # System Engine (Wikipedia): Giữ cố định vì dùng chung cho tất cả
        self.system_engine = TQEngine(dim=dim, bits=4, use_ivf=True, ivf_nlist=4096)
        self.system_engine.ivf_nprobe = 256
        self.system_loaded = False
        self.lock = threading.Lock()
        
        # Load System Index ngay khi khởi động
        self.load_system_index()

    def load_system_index(self):
        # Bộ nhớ hệ thống (Wikipedia) - Tên cố định để tránh bị xóa nhầm
        system_path = os.path.join(self.data_dir, "tq_index_4bit_np4096_system")
        
        if os.path.exists(system_path):
            try:
                self.system_engine.load_index(system_path)
                self.system_loaded = True
                print(f"SUCCESS: System Memory Loaded: {system_path}")
            except Exception as e:
                print(f"WARNING: Could not load system memory: {e}")
        else:
            print(f"WARNING: System index not found at {system_path}")

    def add_vectors(self, vectors: np.ndarray, start_id: int, user_id: int):
        vectors_t = torch.from_numpy(vectors).float()
        engine = self.user_manager.get_engine(user_id)
        
        with self.lock:
            # Khởi tạo hoặc thêm mới
            if not hasattr(engine, 'current_ivf_data') or engine.current_ivf_data is None:
                # Tự động điều chỉnh n_list cho user mới nếu ít data
                original_nlist = engine.ivf_nlist
                engine.ivf_nlist = min(original_nlist, len(vectors_t))
                engine.index(vectors_t)
                engine.ivf_nlist = original_nlist
            else:
                for i, v in enumerate(vectors_t):
                    engine.add(v, start_id + i)
            
            engine.merge_dynamic_shards()
        self.user_manager.save_user_engine(user_id)

    def search(self, query_vector: np.ndarray, user_id: int, top_k: int = 10, scope: str = "both", allowed_ids: list = None, n_probe: int = 64, use_rerank: bool = False):
        query_t = torch.from_numpy(query_vector).float()
        all_results = []
        
        # 1. Search User Data (Chỉ nạp khi cần)
        if scope in ["user", "both"]:
            engine = self.user_manager.get_engine(user_id)
            if hasattr(engine, 'current_ivf_data') and engine.current_ivf_data is not None:
                # Truyền allowed_ids vào engine search
                ids, scores = engine.search(query_t, top_k=top_k, n_probe=16, allowed_ids=allowed_ids)
                if ids is not None:
                    for i, s in zip(ids.tolist(), scores.tolist()):
                        if int(i) >= 0:
                            all_results.append({"id": int(i), "score": float(s), "source": "user"})

        # 2. Search System Data
        if scope in ["system", "both"] and self.system_loaded:
            # Nếu dùng rerank, ta lấy 4 * top_k candidates
            k_for_search = top_k * 4 if use_rerank else top_k
            ids, scores = self.system_engine.search(query_t, top_k=k_for_search, n_probe=n_probe)
            if ids is not None:
                for i, s in zip(ids.tolist(), scores.tolist()):
                    if int(i) >= 0:
                        all_results.append({"id": int(i), "score": float(s), "source": "system"})

        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Hydrate results with text from metadata_service
        hydrated_results = []
        for r in all_results:
            c = metadata_service.get_chunk(r['id'])
            if c:
                r['text'] = c.get('text', '')
                r['source'] = c.get('source', '')
                hydrated_results.append(r)

        # 3. Perform Rerank if enabled
        if use_rerank and hydrated_results:
            # Truyền query text vào (cần convert từ vector nếu query_vector được truyền vào)
            # Lưu ý: Ở đây search nhận query_vector, ta cần query_text cho rerank_service
            # Ta giả định query_text sẽ được truyền vào hoặc xử lý ở main.py
            # Để tối ưu, ta sẽ thực hiện rerank ở main.py vì ở đó có sẵn query_text
            pass 

        return hydrated_results[:top_k]

    def search_batch(self, query_vectors: np.ndarray, user_id: int, top_k: int = 10, scope: str = "both", allowed_ids: list = None, n_probe: int = 64, use_rerank: bool = False):
        query_t = torch.from_numpy(query_vectors).float()
        batch_results = []
        
        # 1. Search User Data
        user_batch_raw = None
        if scope in ["user", "both"]:
            engine = self.user_manager.get_engine(user_id)
            if hasattr(engine, 'current_ivf_data') and engine.current_ivf_data is not None:
                # User data thường ít, n_probe=16 là đủ
                user_batch_raw = engine.search_batch(query_t, top_k=top_k, n_probe=16, allowed_ids=allowed_ids)
        
        # 2. Search System Data
        sys_batch_raw = None
        if scope in ["system", "both"] and self.system_loaded:
            # Dùng n_probe truyền từ frontend (Ultrafast=16, Balance=128, v.v.)
            sys_batch_raw = self.system_engine.search_batch(query_t, top_k=top_k, n_probe=n_probe)

        # Merge results for each query in batch
        for b in range(len(query_vectors)):
            b_results = []
            
            # Xử lý kết quả từ User Index cho query b
            if user_batch_raw and b < len(user_batch_raw):
                ids, scores = user_batch_raw[b]
                for i, s in zip(ids.tolist(), scores.tolist()):
                    if int(i) >= 0:
                        b_results.append({"id": int(i), "score": float(s), "source": "user"})
            
            # Xử lý kết quả từ System Index cho query b
            if sys_batch_raw and b < len(sys_batch_raw):
                ids, scores = sys_batch_raw[b]
                for i, s in zip(ids.tolist(), scores.tolist()):
                    if int(i) >= 0:
                        b_results.append({"id": int(i), "score": float(s), "source": "system"})
            
            # Sắp xếp và lấy Top-K cho query b
            b_results.sort(key=lambda x: x['score'], reverse=True)
            batch_results.append(b_results[:top_k])
            
        return batch_results

    def delete_vector(self, vector_id: int, user_id: int):
        engine = self.user_manager.get_engine(user_id)
        engine.delete(vector_id)
        self.user_manager.save_user_engine(user_id)

# Global instance
tq_service = TQService()
