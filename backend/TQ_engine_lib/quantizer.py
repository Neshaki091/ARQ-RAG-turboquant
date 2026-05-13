import torch
import numpy as np
import math
import os
import gc
import json
from typing import Union, List, Optional, Tuple, Sequence
from dataclasses import dataclass
from .codebook import ScalarQuantizer
from .rotation import get_orthogonal_matrix, rotate_forward, rotate_backward
from .tq_bridge import tq_native

@dataclass
class ProdQuantized:
    sq_codes: np.ndarray
    qjl_signs: np.ndarray
    norms: np.ndarray
    centroids: np.ndarray
    dim: int
    sq_bits: int
    total_bits: int
    qjl_scale: float
    rot_op: np.ndarray
    res_norms: np.ndarray

@dataclass
class IVFData:
    coarse_centroids: torch.Tensor
    pq_data: ProdQuantized
    vector_ids: np.ndarray
    list_offsets: np.ndarray
    n_list: int
    n_probe: int

class TQEngine:
    def __init__(self, dim: int = 768, bits: int = 4, device: str = None, use_ivf: bool = False, ivf_nlist: int = 1024, ivf_nprobe: int = 32):
        if bits not in [2, 4]:
            raise ValueError(f"TurboQuant currently only supports 2-bit (1+1) and 4-bit (3+1) configurations. Received: {bits}")
            
        self.dim = dim
        self.bits = bits
        self.sq_bits = bits - 1
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_ivf = use_ivf
        self.ivf_nlist = ivf_nlist
        # If ivf_nprobe <= 0, we will auto-tune per index based on list sizes (target scan budget).
        self.ivf_nprobe = int(ivf_nprobe)

        # Auto-nprobe tuning: target number of candidates scanned per query (ANN-only).
        # Higher => better recall, lower => higher QPS. Tune this to match your latency budget.
        self.ivf_target_candidates = 20000
        self.ivf_nprobe_min = 1
        self.ivf_nprobe_max = None  # None => up to nlist

        # --- NEW: Real-time Dynamic Storage ---
        self.deleted_ids = set()
        self.dynamic_shards = {}  # {cluster_idx: [ (id, ProdQuantized_single), ... ]}
        self.current_ivf_data = None # Will hold the indexed IVFData
        self.raw_vectors = None      # Memory-mapped raw vectors for Reranking

        # 1. Initialize Scalar Quantizer (Stage 1)
        self.sq_quantizer = ScalarQuantizer(dim=dim, bits=self.sq_bits, device=self.device)

        # 2. Pure TurboQuant: Exact Dimension Orthogonal Rotation
        self.rot_op_t = get_orthogonal_matrix(dim, device=self.device)
        self.rot_op_np = self.rot_op_t.cpu().numpy().astype(np.float32)

        # 3. Calculate Dynamic Scale Placeholder
        self.qjl_scale = 1.0 

    def _auto_nprobe(self) -> int:
        """
        Choose nprobe based on IVF list size distribution to keep scan budget stable.

        Uses median list size (robust to skew). This is ANN-only tuning; rerank can use higher retrieve_k instead.
        """
        ivf = self.current_ivf_data
        if ivf is None or not isinstance(ivf, IVFData):
            return max(1, self.ivf_nprobe)

        offsets = np.asarray(ivf.list_offsets, dtype=np.int64)
        if offsets.ndim != 1 or offsets.size < 2:
            return max(1, self.ivf_nprobe)

        counts = offsets[1:] - offsets[:-1]
        if counts.size == 0:
            return max(1, self.ivf_nprobe)

        med = float(np.median(counts))
        if med <= 0:
            med = float(np.mean(counts)) if float(np.mean(counts)) > 0 else 1.0

        target = max(1, int(self.ivf_target_candidates))
        nprobe = int(np.ceil(target / med))

        nlist = int(ivf.n_list) if hasattr(ivf, "n_list") else int(counts.size)
        nprobe = max(self.ivf_nprobe_min, nprobe)
        if self.ivf_nprobe_max is not None:
            nprobe = min(int(self.ivf_nprobe_max), nprobe)
        nprobe = min(nlist, nprobe)
        return max(1, nprobe)

    def _train_kmeans(self, x_sample: torch.Tensor, n_list: int, iters: int = 10):
        """Train Coarse Centroids trong không gian GỐC (RAW)."""
        N, D = x_sample.shape
        # Khởi tạo centroids ngẫu nhiên từ mẫu
        indices = torch.randperm(N)[:n_list]
        centroids = x_sample[indices].clone()
        
        print(f"  Training IVF K-Means in RAW space ({n_list} clusters)...")
        for i in range(iters):
            # Tính assignments theo batch
            assignments = torch.zeros(N, dtype=torch.long, device=self.device)
            chunk_size = 8192
            for s in range(0, N, chunk_size):
                e = min(s + chunk_size, N)
                batch = x_sample[s:e]
                
                # Dùng Inner Product để gán cụm (DPR đặc thù)
                scores = torch.mm(batch, centroids.t())
                assignments[s:e] = scores.argmax(dim=1)
                del scores, batch
            
            # Cập nhật centroids
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(n_list, 1, device=self.device)
            ones = torch.ones(N, 1, device=self.device)
            new_centroids.index_add_(0, assignments, x_sample)
            counts.index_add_(0, assignments, ones)
            counts = torch.clamp(counts, min=1.0)
            centroids = new_centroids / counts

            # IMPORTANT: For cosine/IP retrieval on L2-normalized vectors, normalize centroids.
            # Without this, dot-product routing is biased by centroid norms and recall can drop
            # at low nprobe (especially when increasing nlist).
            centroids = centroids / (torch.norm(centroids, dim=1, keepdim=True) + 1e-12)
            
            if i % 10 == 0:
                print(f"    Iteration {i}/{iters} done")
            gc.collect()
            
        return centroids

    def bind_raw_data(self, npy_path: str):
        """
        Gắn file vector gốc (Float32) dưới dạng memory-map để phục vụ Reranking.
        """
        if os.path.exists(npy_path):
            self.raw_vectors = np.load(npy_path, mmap_mode='r')
            print(f"[LINK] TurboQuant: Raw data bound for Reranking (mmap): {npy_path}")
        else:
            print(f"[WARNING] Warning: Raw data file not found: {npy_path}")

    def index(self, x: Union[torch.Tensor, np.ndarray], online_clustering: bool = False, save_path: str = None):
        """
        Xây dựng chỉ mục IVF.
        Centroids nằm ở RAW space. Residuals nằm ở ROTATED space.
        """
        # Accept both numpy and torch inputs.
        # Note: IVF indexing uses numpy ops (np.dot, memmap write). For torch input, convert once.
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            x_np = np.asarray(x, dtype=np.float32)

        N, D = x_np.shape
        self.dim = D
        
        if self.use_ivf:
            # 1. Training IVF Centroids (RAW SPACE)
            n_sample = min(N, 65536)
            indices = np.random.choice(N, n_sample, replace=False)
            x_sample = torch.from_numpy(x_np[indices].copy()).to(self.device)

            # [TỐI ƯU CÁCH 4]: Huấn luyện ma trận xoay bằng ITQ thay vì Random
            from .rotation import train_optimized_rotation
            self.rot_op_t = train_optimized_rotation(x_sample, self.dim)
            self.rot_op_np = self.rot_op_t.cpu().numpy().astype(np.float32)

            # KHÔNG xoay mẫu trước khi train kmeans
            coarse_centroids = self._train_kmeans(x_sample, self.ivf_nlist, iters=30)
            del x_sample
            
            # 2. Gán cụm (Assignments) trong RAW SPACE
            assignments = np.zeros(N, dtype=np.int32)
            chunk_size = 10000 
            coarse_centroids_np = coarse_centroids.cpu().numpy().astype(np.float32)
            
            print(f"TurboQuant: Assignments in RAW space for {N:,} vectors...")
            for i in range(0, N, chunk_size):
                e = min(i + chunk_size, N)
                batch_np = x_np[i:e].copy()
                
                # Dùng Inner Product để gán cụm (tốt cho DPR)
                scores = np.dot(batch_np, coarse_centroids_np.T)
                assignments[i:e] = np.argmax(scores, axis=1)
                
                if i % 500000 == 0:
                    print(f"    Assigned {i:,}/{N:,} vectors...")
                
                # Giải phóng bộ nhớ chunk
                del batch_np, scores
                if i % 500000 == 0:
                    gc.collect()

            # 2.5. Tính QJL scale theo đúng không gian nén: ROTATE(residual = x - centroid)
            # Lý do: IVF đang nén residual; scale lấy từ raw-rotated có thể lệch mạnh (đặc biệt với SQ 3-bit / 4b).
            # Ước lượng bằng sample nhỏ để không tốn RAM.
            with torch.no_grad():
                n_scale = min(N, 65536)
                scale_idx = np.random.choice(N, n_scale, replace=False)
                x_scale = torch.from_numpy(np.asarray(x_np[scale_idx], dtype=np.float32)).to(self.device)
                c_idx = torch.from_numpy(assignments[scale_idx]).to(self.device).long()
                c = coarse_centroids.index_select(0, c_idx)
                residual = x_scale - c
                residual_rot = rotate_forward(residual, self.rot_op_t)
                self.qjl_scale = float(torch.mean(torch.abs(residual_rot)).item())
                print(f"  Dynamic QJL Scale (rotated residual) calculated: {self.qjl_scale:.6f}")
                del x_scale, c_idx, c, residual, residual_rot

            # 3. Chuẩn bị file lưu trữ trên đĩa (Streaming Mode)
            index_dir = save_path if save_path else "tq_index_temp"
            os.makedirs(index_dir, exist_ok=True)
            
            # Tính toán offsets
            counts = np.bincount(assignments, minlength=self.ivf_nlist)
            offsets = np.zeros(self.ivf_nlist + 1, dtype=np.int64)
            offsets[1:] = np.cumsum(counts)
            
            # Tạo các memmap file (Pre-allocation với kích thước nén chính xác)
            # TQ-2b: SQ=1bit -> packed 8:1 (96 bytes). QJL=1bit -> packed 8:1 (96 bytes)
            # TQ-4b: SQ=3bit -> packed 2:1 (384 bytes). QJL=1bit -> packed 8:1 (96 bytes)
            
            sq_packed_dim = self.dim
            if self.sq_bits == 1: sq_packed_dim = self.dim // 8
            elif self.sq_bits == 3: sq_packed_dim = self.dim // 2
            
            qjl_packed_dim = self.dim // 8
            
            # Tạo các memmap file (Dùng open_memmap để ghi đúng NPY Header)
            from numpy.lib.format import open_memmap
            f_sq = open_memmap(os.path.join(index_dir, "sq_codes.npy"), mode='w+', dtype=np.uint8, shape=(N, sq_packed_dim))
            f_signs = open_memmap(os.path.join(index_dir, "qjl_signs.npy"), mode='w+', dtype=np.uint8, shape=(N, qjl_packed_dim))
            f_norms = open_memmap(os.path.join(index_dir, "norms.npy"), mode='w+', dtype=np.float32, shape=(N,))
            f_res_norms = open_memmap(os.path.join(index_dir, "res_norms.npy"), mode='w+', dtype=np.float32, shape=(N,))
            f_ids = open_memmap(os.path.join(index_dir, "vector_ids.npy"), mode='w+', dtype=np.int64, shape=(N,))

            # Lưu sẵn Centroids và Metadata
            dummy_pq = self._quantize_flat(torch.zeros((1, self.dim), device=self.device))
            np.save(os.path.join(index_dir, "coarse_centroids.npy"), coarse_centroids.cpu().numpy())
            np.save(os.path.join(index_dir, "rot_op.npy"), self.rot_op_np)
            np.save(os.path.join(index_dir, "sq_centroids.npy"), dummy_pq.centroids)
            np.save(os.path.join(index_dir, "list_offsets.npy"), offsets.astype(np.int32))
            
            with open(os.path.join(index_dir, "metadata.json"), "w", encoding='utf-8') as f:
                json.dump({
                    "dim": int(self.dim), "bits": int(self.bits), "qjl_scale": float(self.qjl_scale),
                    "n_list": int(self.ivf_nlist), "n_probe": int(self.ivf_nprobe), "deleted_ids": []
                }, f, indent=2)

            print(f"TurboQuant: Streaming compression to {index_dir}...")
            with torch.no_grad():
                for c_idx in range(self.ivf_nlist):
                    idx_in_cluster = np.where(assignments == c_idx)[0]
                    count = len(idx_in_cluster)
                    if count > 0:
                        # Lấy vector thô
                        cluster_x = torch.from_numpy(x_np[idx_in_cluster].copy()).to(self.device)
                        # Lấy centroid tương ứng
                        centroid = coarse_centroids[c_idx].unsqueeze(0)
                        
                        # Nén Residual (x - centroid)
                        cluster_pq = self._quantize_flat(cluster_x, online_clustering, centroid=centroid)
                        
                        start_pos, end_pos = offsets[c_idx], offsets[c_idx+1]
                        f_sq[start_pos:end_pos] = cluster_pq.sq_codes
                        f_signs[start_pos:end_pos] = cluster_pq.qjl_signs
                        f_norms[start_pos:end_pos] = cluster_pq.norms
                        f_res_norms[start_pos:end_pos] = cluster_pq.res_norms
                        f_ids[start_pos:end_pos] = idx_in_cluster
                        del cluster_x, cluster_pq
                    
                    if c_idx % 200 == 0:
                        f_sq.flush(); f_signs.flush(); f_norms.flush(); f_res_norms.flush(); f_ids.flush()
                        gc.collect()
                        print(f"    Cluster {c_idx}/{self.ivf_nlist} done (saved)...")

            # Finalize
            f_sq.flush(); f_signs.flush(); f_norms.flush(); f_res_norms.flush(); f_ids.flush()
            
            flat_pq = ProdQuantized(
                sq_codes=f_sq, qjl_signs=f_signs, norms=f_norms,
                centroids=dummy_pq.centroids, dim=self.dim, sq_bits=self.sq_bits, total_bits=self.bits,
                qjl_scale=self.qjl_scale, rot_op=self.rot_op_np, res_norms=f_res_norms
            )
            self.current_ivf_data = IVFData(
                coarse_centroids=coarse_centroids, pq_data=flat_pq, vector_ids=f_ids,
                list_offsets=offsets.astype(np.int32), n_list=self.ivf_nlist, n_probe=self.ivf_nprobe
            )
            return self.current_ivf_data
        else:
            # Flat mode
            return self._quantize_flat(torch.from_numpy(np.array(x, dtype=np.float32)).to(self.device), online_clustering)

    def add(self, vector: torch.Tensor, vector_id: int):
        """
        Thêm 1 vector mới vào hệ thống theo thời gian thực.
        """
        if self.current_ivf_data is None:
            raise ValueError("Cần gọi index() hoặc load_index() trước khi add().")
            
        if vector.device.type != self.device:
            vector = vector.to(self.device)
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
            
        # 1. Tìm cụm gần nhất trong không gian RAW
        scores = torch.mm(vector, self.current_ivf_data.coarse_centroids.t())
        c_idx = scores.argmax(dim=1).item()
        centroid = self.current_ivf_data.coarse_centroids[c_idx].unsqueeze(0)
        
        # 2. Nén Residual (vector - centroid)
        pq_single = self._quantize_flat(vector, online_clustering=False, centroid=centroid)
        
        # 3. Đưa vào dynamic shards
        if c_idx not in self.dynamic_shards:
            self.dynamic_shards[c_idx] = []
        self.dynamic_shards[c_idx].append((vector_id, pq_single))
        
        # Xóa khỏi danh sách deleted nêú có (trường hợp ghi đè)
        if vector_id in self.deleted_ids:
            self.deleted_ids.remove(vector_id)

    def merge_dynamic_shards(self):
        """
        Gộp toàn bộ dữ liệu từ dynamic_shards vào cấu trúc IVF chính để tối ưu hóa tìm kiếm SIMD.
        """
        ivf = self.current_ivf_data
        if not isinstance(ivf, IVFData) or not self.dynamic_shards:
            return

        print(f"[MERGE] TurboQuant: Merging {sum(len(v) for v in self.dynamic_shards.values())} new vectors into IVF...")

        # 1. Thu thập dữ liệu mới và gộp mảng
        new_total_size = len(ivf.vector_ids) + sum(len(v) for v in self.dynamic_shards.values())
        
        updated_sq_codes = torch.zeros((new_total_size, ivf.pq_data.sq_codes.shape[1]), dtype=torch.uint8)
        updated_qjl_signs = torch.zeros((new_total_size, ivf.pq_data.qjl_signs.shape[1]), dtype=torch.int8)
        updated_norms = torch.zeros(new_total_size)
        updated_res_norms = torch.zeros(new_total_size)
        updated_vector_ids = np.zeros(new_total_size, dtype=np.int64)
        updated_offsets = [0]

        curr_pos = 0
        for c_idx in range(ivf.n_list):
            # Thêm dữ liệu cũ của cụm này
            old_start = ivf.list_offsets[c_idx]
            old_end = ivf.list_offsets[c_idx+1]
            old_size = old_end - old_start
            
            if old_size > 0:
                updated_sq_codes[curr_pos:curr_pos+old_size] = torch.from_numpy(ivf.pq_data.sq_codes[old_start:old_end])
                updated_qjl_signs[curr_pos:curr_pos+old_size] = torch.from_numpy(ivf.pq_data.qjl_signs[old_start:old_end])
                updated_norms[curr_pos:curr_pos+old_size] = torch.from_numpy(ivf.pq_data.norms[old_start:old_end])
                updated_res_norms[curr_pos:curr_pos+old_size] = torch.from_numpy(ivf.pq_data.res_norms[old_start:old_end])
                updated_vector_ids[curr_pos:curr_pos+old_size] = ivf.vector_ids[old_start:old_end]
                curr_pos += old_size
            
            # Thêm dữ liệu mới của cụm này
            if c_idx in self.dynamic_shards:
                for vid, dpq in self.dynamic_shards[c_idx]:
                    # Đảm bảo chuyển đổi từ numpy sang torch nếu cần
                    updated_sq_codes[curr_pos] = torch.from_numpy(dpq.sq_codes) if isinstance(dpq.sq_codes, np.ndarray) else dpq.sq_codes
                    updated_qjl_signs[curr_pos] = torch.from_numpy(dpq.qjl_signs) if isinstance(dpq.qjl_signs, np.ndarray) else dpq.qjl_signs
                    updated_norms[curr_pos] = float(dpq.norms[0]) if isinstance(dpq.norms, np.ndarray) else float(dpq.norms)
                    updated_res_norms[curr_pos] = float(dpq.res_norms[0]) if isinstance(dpq.res_norms, np.ndarray) else float(dpq.res_norms)
                    updated_vector_ids[curr_pos] = vid
                    curr_pos += 1
                
            updated_offsets.append(curr_pos)

        # 3. Cập nhật lại state
        ivf.pq_data.sq_codes = updated_sq_codes
        ivf.pq_data.qjl_signs = updated_qjl_signs
        ivf.pq_data.norms = updated_norms
        ivf.pq_data.res_norms = updated_res_norms
        ivf.vector_ids = updated_vector_ids
        ivf.list_offsets = torch.tensor(updated_offsets, dtype=torch.long)
        
        # 4. Clear dynamic shards
        self.dynamic_shards.clear()
        print(f"[SUCCESS] Successfully merged into IVF index. New size: {new_total_size}")

    def delete(self, vector_id: int):
        """
        Xóa mềm vector khỏi kết quả tìm kiếm.
        """
        self.deleted_ids.add(vector_id)


    def _quantize_flat(self, x: torch.Tensor, online_clustering: bool = False, x_rot: torch.Tensor = None, centroid: torch.Tensor = None) -> ProdQuantized:
        """
        Nén bộ dữ liệu vector x (N, D) sang định dạng TurboQuant (SQ+QJL).
        Nếu có centroid, sẽ nén residual (x - centroid).
        """
        if x.device.type != self.device:
            x = x.to(self.device)
            
        # 1. TÍNH RESIDUAL (Nếu có centroid)
        if centroid is not None:
            if centroid.device.type != self.device:
                centroid = centroid.to(self.device)
            x_target = x - centroid
        else:
            x_target = x

        # 2. ROTATE (Chỉ xoay nếu chưa có x_rot)
        if x_rot is None:
            x_rot = rotate_forward(x_target, self.rot_op_t)

        # 3. Extract Norms (Dùng norm của vector gốc để phục vụ tái cấu trúc khoảng cách)
        norms = torch.norm(x, dim=-1)
        res_norms = torch.norm(x_target, dim=-1)
        
        # 4. Stage 1+2 (SQ + QJL) — prefer Rust for speed
        if online_clustering:
            self.sq_quantizer.fit(x_rot)

        # Ensure contiguous float32 for Rust bridge
        x_rot_np = np.ascontiguousarray(x_rot.detach().cpu().numpy(), dtype=np.float32)
        sq_centroids_np = np.ascontiguousarray(self.sq_quantizer.centroids.detach().cpu().numpy(), dtype=np.float32)

        try:
            sq_codes_packed, qjl_signs_packed, res_norms_np = tq_native.tq_quantize_rotated(
                x_rot_np,
                sq_centroids_np,
                int(self.sq_bits),
            )
            sq_codes_np = np.asarray(sq_codes_packed, dtype=np.uint8)
            qjl_signs = np.asarray(qjl_signs_packed, dtype=np.uint8)
            res_norms_np = np.asarray(res_norms_np, dtype=np.float32)
        except Exception:
            # Fallback to torch/python path (correctness first)
            sq_q = self.sq_quantizer.quantize(x_rot)
            x_hat_1 = self.sq_quantizer.reconstruct(sq_q.indices)
            residual = x_rot - x_hat_1
            res_norms_np = torch.norm(residual, dim=-1).detach().cpu().numpy().astype(np.float32)
            signs = (residual > 0).to(torch.uint8).cpu().numpy()
            qjl_signs = np.packbits(signs, axis=-1, bitorder='little').astype(np.uint8)
            sq_codes_np = sq_q.indices.cpu().numpy().astype(np.uint8)
        
        return ProdQuantized(
            sq_codes=sq_codes_np,
            qjl_signs=qjl_signs.astype(np.uint8),
            norms=norms.cpu().numpy().astype(np.float32),
            centroids=self.sq_quantizer.centroids.cpu().numpy().astype(np.float32),
            dim=self.dim,
            sq_bits=self.sq_bits,
            total_bits=self.bits,
                    qjl_scale=self.qjl_scale,
                    rot_op=self.rot_op_np,
                    res_norms=res_norms_np
                )

    def search_batch(self, queries: torch.Tensor, top_k: int = 100, n_probe: int = None, allowed_ids: Optional[List[int]] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        ivf = self.current_ivf_data
        if ivf is None:
            raise ValueError("No data indexed. Call index() first.")
            
        if queries.device.type != self.device:
            queries = queries.to(self.device)
            
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)
            
        num_queries = queries.shape[0]
        
        # 1. Query Rotation (Batch)
        q_rot = rotate_forward(queries, self.rot_op_t)
        q_rot_np = np.ascontiguousarray(q_rot.cpu().numpy(), dtype=np.float32)
        queries_np = np.ascontiguousarray(queries.cpu().numpy() if isinstance(queries, torch.Tensor) else queries, dtype=np.float32)
        
        pq = ivf.pq_data

        # Auto-tune nprobe if requested (ivf_nprobe <= 0)
        if n_probe is not None:
            nprobe = int(n_probe)
        else:
            nprobe = self._auto_nprobe() if self.ivf_nprobe <= 0 else int(self.ivf_nprobe)
        
        # 2. Gọi True Online Batch Scan từ Rust (Core Engine)
        scores, indices = tq_native.tq_ivf_online_scan(
            queries_np, # Dùng cho coarse search (Stage 1: RAW space)
            np.ascontiguousarray(pq.sq_codes, dtype=np.uint8),
            np.ascontiguousarray(pq.centroids, dtype=np.float32),
            np.ascontiguousarray(pq.norms, dtype=np.float32),
            np.ascontiguousarray(pq.qjl_signs, dtype=np.uint8),
            np.ascontiguousarray(pq.res_norms, dtype=np.float32),
            q_rot_np,   # Dùng cho fine scan (Stage 2: ROTATED space)
            np.ascontiguousarray(ivf.list_offsets, dtype=np.int32),
            np.ascontiguousarray(ivf.coarse_centroids.cpu().numpy() if isinstance(ivf.coarse_centroids, torch.Tensor) else ivf.coarse_centroids, dtype=np.float32),
            int(nprobe),
            float(pq.qjl_scale),
            int(self.dim),
            int(self.sq_bits),
            int(top_k if allowed_ids is None else top_k * 10) # Lấy nhiều hơn nếu có filter để bù đắp
        )
        
        # 3. Map local IDs to global IDs and package results
        results = []
        global_ids = ivf.vector_ids
        allowed_set = set(allowed_ids) if allowed_ids is not None else None

        for i in range(num_queries):
            valid_mask = indices[i] != -1
            q_indices = indices[i][valid_mask]
            q_scores = scores[i][valid_mask]
            
            # Lấy global IDs
            q_global_ids = global_ids[q_indices]
            
            # Lọc theo allowed_ids nếu có
            if allowed_set is not None:
                mask = np.isin(q_global_ids, list(allowed_set))
                q_global_ids = q_global_ids[mask]
                q_scores = q_scores[mask]
                
                # Cắt bớt về đúng top_k sau khi lọc
                q_global_ids = q_global_ids[:top_k]
                q_scores = q_scores[:top_k]
            
            final_ids = torch.from_numpy(q_global_ids.copy()).to(self.device)
            final_scores = torch.from_numpy(q_scores.copy()).to(self.device)
            results.append((final_ids, final_scores))
            
        return results

    def _search_batch_mmap(self, q_rot_np, top_k):
        """Phương án dự phòng dùng mmap (vẫn rất nhanh nhờ Rust Online Scan)"""
        ivf = self.current_ivf_data
        pq = ivf.pq_data
        
        scores, indices = tq_native.tq_ivf_online_scan(
            q_rot_np,
            np.ascontiguousarray(pq.sq_codes, dtype=np.uint8),
            np.ascontiguousarray(pq.centroids, dtype=np.float32),
            np.ascontiguousarray(pq.norms, dtype=np.float32),
            np.ascontiguousarray(pq.qjl_signs, dtype=np.uint8),
            np.ascontiguousarray(pq.res_norms, dtype=np.float32),
            q_rot_np,
            np.ascontiguousarray(ivf.list_offsets, dtype=np.int32),
            np.ascontiguousarray(ivf.coarse_centroids.cpu().numpy() if isinstance(ivf.coarse_centroids, torch.Tensor) else ivf.coarse_centroids, dtype=np.float32),
            int(self.ivf_nprobe),
            float(pq.qjl_scale),
            int(self.dim),
            int(self.sq_bits),
            int(top_k)
        )
        
        results = []
        global_ids = ivf.vector_ids
        for i in range(q_rot_np.shape[0]):
            valid_mask = indices[i] != -1
            q_indices = indices[i][valid_mask]
            q_scores = scores[i][valid_mask]
            results.append((torch.from_numpy(global_ids[q_indices].copy()).to(self.device), torch.from_numpy(q_scores.copy()).to(self.device)))
            
        return results


    def search(self, query: torch.Tensor, top_k: int = 100, n_probe: int = None, allowed_ids: Optional[List[int]] = None) -> tuple[torch.Tensor, torch.Tensor]:
        ivf = self.current_ivf_data
        if ivf is None:
            raise ValueError("No data indexed. Call index() first.")
            
        if query.device.type != self.device:
            query = query.to(self.device)
            
        if isinstance(ivf, IVFData):
            # Tối ưu: Sử dụng search_batch để kích hoạt tq_ivf_online_scan (Rust-native)
            if query.dim() == 1:
                query = query.unsqueeze(0)
            
            results = self.search_batch(query, top_k=top_k, n_probe=n_probe, allowed_ids=allowed_ids)
            return results[0] 
        else:
            return self._native_cosine_search_flat(query, ivf, top_k, allowed_ids=allowed_ids)

    def _native_cosine_search_flat(self, query: torch.Tensor, pq: ProdQuantized, top_k: int = 100, allowed_ids: list = None) -> tuple[torch.Tensor, torch.Tensor]:
        if query.device.type != self.device:
            query = query.to(self.device)
            
        if query.dim() == 1:
            query = query.unsqueeze(0)
            
        # 1. Query Rotation
        q_rot = rotate_forward(query, self.rot_op_t).squeeze(0)
        q_np = q_rot.cpu().numpy().astype(np.float32)
        query_1d = np.array(q_np, dtype=np.float32, order='C')

        # 2. Tính toán Batch Size
        total_vectors = pq.sq_codes.shape[0]
        ram_gb = 4.0
        h = 10**(len(str(total_vectors)) - 1)
        raw_batch_size = int((0.3 * total_vectors * (h * 100) / total_vectors) / (ram_gb * (ram_gb / 0.4))) + 1
        compression_ratio = 32 // self.bits
        tq_batch_size = raw_batch_size * compression_ratio
        
        all_scores = []
        for start_idx in range(0, total_vectors, tq_batch_size):
            end_idx = min(start_idx + tq_batch_size, total_vectors)
            sq_batch = np.ascontiguousarray(pq.sq_codes[start_idx:end_idx], dtype=np.uint8)
            qjl_batch = np.ascontiguousarray(pq.qjl_signs[start_idx:end_idx], dtype=np.uint8)
            norms_batch = np.ascontiguousarray(pq.norms[start_idx:end_idx], dtype=np.float32)
            res_norms_batch = np.ascontiguousarray(pq.res_norms[start_idx:end_idx], dtype=np.float32)
            centroids_1d = np.ascontiguousarray(pq.centroids, dtype=np.float32)
            
            batch_scores = tq_native.tq_scan(
                query_1d, sq_batch, centroids_1d, norms_batch,
                qjl_batch, res_norms_batch, query_1d,
                float(pq.qjl_scale), int(self.dim), int(self.sq_bits)
            )
            all_scores.append(batch_scores)
        
        final_scores = np.concatenate(all_scores)
        
        # 3. Lọc ID nếu cần
        if allowed_ids is not None:
            allowed_set = set(allowed_ids)
            filtered_indices = []
            filtered_scores = []
            
            # Lưu ý: Ở chế độ Flat, index chính là ID (giả định) hoặc lấy từ pq_data if exists
            # Trong TQEngine, flat index giả định index 0..N là global ID nếu không có vector_ids
            for i, s in enumerate(final_scores):
                if i in allowed_set and i not in self.deleted_ids:
                    filtered_indices.append(i)
                    filtered_scores.append(s)
            
            if not filtered_indices:
                return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float32)
                
            scores_t = torch.tensor(filtered_scores, dtype=torch.float32)
            indices_t = torch.tensor(filtered_indices, dtype=torch.long)
            
            top_k = min(top_k, len(scores_t))
            top_scores, top_indices_rel = torch.topk(scores_t, top_k)
            return indices_t[top_indices_rel], top_scores
        else:
            scores_t = torch.from_numpy(final_scores).view(-1)
            # Lọc deleted_ids
            if self.deleted_ids:
                for did in self.deleted_ids:
                    if did < len(scores_t): scores_t[did] = -1e9
            
            top_scores, top_indices = torch.topk(scores_t, min(top_k, len(scores_t)))
            return top_indices, top_scores

    def save_index(self, path: str):
        """
        Lưu toàn bộ chỉ mục và metadata. 
        Trong chế độ Streaming, hàm này chỉ cập nhật Metadata và các mảng nhỏ (centroids).
        """
        import gc
        import time
        from pathlib import Path
        
        save_path = Path(path).resolve()
        if self.current_ivf_data is None:
            raise ValueError("Không có dữ liệu để lưu. Hãy gọi index() trước.")
            
        gc.collect()
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        
        ivf = self.current_ivf_data
        pq = ivf.pq_data
        
        def to_np(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy()
            return obj

        print(f"TurboQuant: Finalizing metadata in {save_path}...")
        
        files_to_save = {}
        # Chỉ lưu các mảng lớn nếu chúng chưa phải là memmap (không ở chế độ streaming)
        if not isinstance(pq.sq_codes, np.memmap):
            files_to_save["sq_codes.npy"] = to_np(pq.sq_codes)
            files_to_save["qjl_signs.npy"] = to_np(pq.qjl_signs)
            files_to_save["norms.npy"] = to_np(pq.norms)
            files_to_save["res_norms.npy"] = to_np(pq.res_norms)
            files_to_save["vector_ids.npy"] = to_np(ivf.vector_ids)
            files_to_save["list_offsets.npy"] = to_np(ivf.list_offsets)

        # Các metadata nhỏ luôn cần lưu/cập nhật
        files_to_save["coarse_centroids.npy"] = to_np(ivf.coarse_centroids)
        files_to_save["rot_op.npy"] = to_np(pq.rot_op)
        files_to_save["sq_centroids.npy"] = to_np(pq.centroids)

        for filename, data in files_to_save.items():
            np.save(str(save_path / filename), data)

        meta = {
            "dim": int(self.dim),
            "bits": int(self.bits),
            "qjl_scale": float(self.qjl_scale),
            "n_list": int(ivf.n_list),
            "n_probe": int(ivf.n_probe),
            "deleted_ids": list(self.deleted_ids)
        }
        with open(str(save_path / "metadata.json"), "w", encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
            
        print(f"Successfully finalized TurboQuant index at: {save_path}")

        for filename, data in files_to_save.items():
            success = False
            for i in range(3):
                try:
                    np.save(str(save_path / filename), data)
                    success = True
                    break
                except OSError as e:
                    if i == 2: raise e
                    time.sleep(0.5)
            if not success:
                print(f"❌ Failed to save {filename} after retries.")

        # 4. Lưu Metadata
        meta = {
            "dim": int(self.dim),
            "bits": int(self.bits),
            "qjl_scale": float(self.qjl_scale),
            "n_list": int(ivf.n_list),
            "n_probe": int(ivf.n_probe),
            "deleted_ids": list(self.deleted_ids)
        }
        
        with open(str(save_path / "metadata.json"), "w", encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
            
        print(f"Successfully saved TurboQuant index to: {save_path}")

    def load_index(self, path: str):
        """
        Tải lại chỉ mục từ thư mục chỉ định với cơ chế ngắt kết nối file (Lock-free).
        """
        from pathlib import Path
        load_path = Path(path).resolve()
        
        if not load_path.exists():
            raise FileNotFoundError(f"Thư mục index không tồn tại: {load_path}")
            
        # 1. Load Metadata (JSON) - Tự động đóng sau khi đọc
        with open(str(load_path / "metadata.json"), "r", encoding='utf-8') as f:
            meta = json.load(f)
            
        self.dim = meta["dim"]
        self.bits = meta["bits"]
        self.sq_bits = self.bits - 1
        self.qjl_scale = meta["qjl_scale"]
        self.ivf_nlist = meta["n_list"]
        self.ivf_nprobe = meta["n_probe"]
        self.deleted_ids = set(meta["deleted_ids"])
        
        # 2. Load mảng dữ liệu (Mmap cho mảng lớn, RAM cho mảng nhỏ để tránh lock file trên Windows)
        coarse_centroids = torch.from_numpy(np.load(os.path.join(path, "coarse_centroids.npy"))).to(self.device)
        sq_codes = np.load(os.path.join(path, "sq_codes.npy"), mmap_mode='r')
        qjl_signs = np.load(os.path.join(path, "qjl_signs.npy"), mmap_mode='r')
        
        # Các file này nạp thẳng vào RAM để tránh lỗi lock file [Errno 22] khi ghi đè trên Windows
        norms = np.load(os.path.join(path, "norms.npy"))
        res_norms = np.load(os.path.join(path, "res_norms.npy"))
        vector_ids = np.load(os.path.join(path, "vector_ids.npy"))
        list_offsets = np.load(os.path.join(path, "list_offsets.npy"))
        
        # Các file nhỏ này hay bị lỗi [Errno 22] trên Windows nếu để mmap='r'
        rot_op = np.load(os.path.join(path, "rot_op.npy"))
        sq_centroids = np.load(os.path.join(path, "sq_centroids.npy"))
        
        self.rot_op_np = rot_op
        self.rot_op_t = torch.from_numpy(rot_op).to(self.device)
        self.index_path = path # Lưu lại để dùng cho Streaming
        
        # 3. Rebuild DataClasses
        pq_data = ProdQuantized(
            sq_codes=sq_codes,
            qjl_signs=qjl_signs,
            norms=norms,
            centroids=sq_centroids,
            dim=self.dim,
            sq_bits=self.sq_bits,
            total_bits=self.bits,
            qjl_scale=self.qjl_scale,
            rot_op=rot_op,
            res_norms=res_norms
        )
        
        self.current_ivf_data = IVFData(
            coarse_centroids=coarse_centroids,
            pq_data=pq_data,
            vector_ids=vector_ids,
            list_offsets=list_offsets,
            n_list=self.ivf_nlist,
            n_probe=self.ivf_nprobe
        )
        
        print(f"Successfully loaded TurboQuant index from: {path} ({len(vector_ids)} vectors)")

