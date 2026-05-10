"""
Benchmark suite: load pre-embedded DPR from Hugging Face (`facebook/wiki_dpr`) -> SQ / PQ / FAISS / TQ.
Based on benchmark_recall.py (measure recall & QPS) and streaming logic from stress_5m.py
to limit RAM usage (~2GB per phase, configurable via --max-ram-gb).

Requires: torch, numpy, faiss, scikit-learn, psutil
Optional: datasets (to stream/download embeddings from Hugging Face)

Default: Load 5M DPR 768d vectors from `facebook/wiki_dpr` (pre-embedded), NO re-embedding.
"""

from __future__ import annotations

import argparse
import gc
import heapq
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import faiss
from sklearn.cluster import MiniBatchKMeans

# Đường dẫn Benchmark để load TQ_engine_lib nội bộ
benchmark_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, benchmark_dir)

from TQ_engine_lib.quantizer import TQEngine
from TQ_engine_lib.tq_bridge import tq_native
from TQ_engine_lib.rotation import rotate_forward

try:
    import psutil
except ImportError:
    psutil = None

try:
    from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
except ImportError:
    DPRQuestionEncoder = None

# --- Metric (giữ nguyên benchmark_recall) ---


def measure_accuracy(predicted_indices, ground_truth_indices, k_values):
    metrics = {"top1_in_k": {}, "set_recall": {}}
    gt_top1 = ground_truth_indices[0]

    for k in k_values:
        pred_set = set(predicted_indices[:k])
        metrics["top1_in_k"][k] = 1.0 if gt_top1 in pred_set else 0.0
        gt_set = set(ground_truth_indices[:k])
        metrics["set_recall"][k] = len(pred_set.intersection(gt_set)) / k if k > 0 else 0.0

    return metrics


import ctypes

def get_ram_info():
    """Trả về (Private_MB, WorkingSet_MB)"""
    if psutil is None:
        return 0.0, 0.0
    p = psutil.Process(os.getpid())
    info = p.memory_info()
    # Private Bytes là RAM thực sự ứng dụng chiếm giữ
    # RSS (Working Set) là RAM bao gồm cả Page Cache (mmap)
    return info.private / (1024 * 1024), info.rss / (1024 * 1024)

def rss_mb():
    # Giữ lại hàm này để tương thích, trả về RSS
    _, rss = get_ram_info()
    return rss

def empty_working_set():
    """Ép Windows thu hồi lại toàn bộ RAM 'mượn' (Page Cache)"""
    if os.name == 'nt':
        try:
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            # Giảm Working Set xuống mức tối thiểu (-1, -1)
            ctypes.windll.kernel32.SetProcessWorkingSetSize(handle, -1, -1)
        except:
            pass

def enforce_memory_limit(limit_mb: int):
    """Thiết lập giới hạn RAM cứng thực thụ cho tiến trình hiện tại trên Windows"""
    if os.name == 'nt' and limit_mb > 0:
        try:
            limit_bytes = limit_mb * 1024 * 1024
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            
            # Các cờ ép buộc giới hạn cứng:
            # QUOTA_LIMITS_HARDWS_MIN_ENABLE = 0x2
            # QUOTA_LIMITS_HARDWS_MAX_ENABLE = 0x4
            flags = 0x2 | 0x4 
            
            # Sử dụng SetProcessWorkingSetSizeEx (yêu cầu quyền truy cập phù hợp)
            res = ctypes.windll.kernel32.SetProcessWorkingSetSizeEx(
                handle, 
                1024 * 1024, # Tối thiểu 1MB
                limit_bytes, # Tối đa X MB
                flags
            )
            if res != 0:
                print(f"--- [!!!] ĐÃ THIẾT LẬP GIỚI HẠN RAM CỨNG: {limit_mb} MB ---")
            else:
                print("--- [!] Cảnh báo: Windows từ chối áp dụng giới hạn cứng. ---")
        except Exception as e:
            print(f"Không thể thiết lập giới hạn RAM: {e}")


def max_vectors_for_ram(dim: int, max_ram_gb: float, reserve: float = 0.72) -> int:
    """Ước lượng số vector float32 corpus an toàn trong max_ram_gb (heuristic)."""
    bytes_budget = max_ram_gb * (1024**3) * reserve
    per = dim * 4
    return max(1024, int(bytes_budget // per))


# --- HF pre-embedded DPR ---


def _load_hf_stream(dataset_name: str, dataset_config: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Install datasets: pip install datasets") from e

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    kwargs = {"split": split, "streaming": True}
    if token:
        kwargs["token"] = token
    try:
        if dataset_config:
            return load_dataset(dataset_name, dataset_config, **kwargs)
        return load_dataset(dataset_name, **kwargs)
    except TypeError:
        # datasets phiên bản cũ dùng use_auth_token
        if token:
            kwargs["use_auth_token"] = token
            kwargs.pop("token", None)
        if dataset_config:
            return load_dataset(dataset_name, dataset_config, **kwargs)
        return load_dataset(dataset_name, **kwargs)
    except RuntimeError as e:
        # datasets>=4 có thể chặn dataset script (wiki_dpr.py)
        if "Dataset scripts are no longer supported" not in str(e):
            raise
        parquet_glob = _wiki_dpr_parquet_glob(dataset_name, dataset_config)
        print(f"datasets script blocked, falling back to parquet stream: {parquet_glob}")
        parquet_kwargs = {"split": split, "streaming": True, "data_files": parquet_glob}
        if token:
            parquet_kwargs["token"] = token
        try:
            return load_dataset("parquet", **parquet_kwargs)
        except TypeError:
            if token:
                parquet_kwargs["use_auth_token"] = token
                parquet_kwargs.pop("token", None)
            return load_dataset("parquet", **parquet_kwargs)


def _wiki_dpr_parquet_glob(dataset_name: str, dataset_config: str) -> str:
    """
    Map config -> parquet path cho facebook/wiki_dpr.
    Ví dụ:
      - psgs_w100.multiset.compressed -> data/psgs_w100/multiset/*.parquet
      - psgs_w100.nq -> data/psgs_w100/nq/*.parquet
    """
    if dataset_name != "facebook/wiki_dpr":
        raise RuntimeError(
            f"Parquet mapping not supported for dataset '{dataset_name}'. "
            "Use facebook/wiki_dpr or add custom mapping."
        )
    cfg = (dataset_config or "psgs_w100.multiset").strip()
    cfg = cfg.replace(".compressed", "")
    cfg = cfg.replace(".dummy", "/dummy")
    if "." in cfg:
        parts = cfg.split(".")
        if len(parts) >= 2:
            cfg = f"{parts[0]}/{parts[1]}"
    cfg = cfg.replace("//", "/")
    return f"hf://datasets/{dataset_name}/data/{cfg}/*.parquet"


def _to_vec(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        return None
    return arr


def _find_vector(row: Dict[str, Any], candidates: Sequence[str]) -> Optional[np.ndarray]:
    for key in candidates:
        if key in row:
            v = _to_vec(row[key])
            if v is not None:
                return v
    # fallback: tìm field list/ndarray 1 chiều đầu tiên
    for _, val in row.items():
        v = _to_vec(val)
        if v is not None:
            return v
    return None


def _get_real_queries(json_path: str, dim: int) -> Optional[np.ndarray]:
    """Tải text từ JSON và dùng DPR Question Encoder để embed"""
    if not os.path.exists(json_path):
        print(f"⚠️ Không tìm thấy file JSON: {json_path}")
        return None
    
    if DPRQuestionEncoder is None:
        print("⚠️ Chưa cài transformers. Cần cài 'pip install transformers' để embed text.")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    if not questions:
        return None
    
    print(f"🧠 Đang nạp DPR Question Encoder để embed {len(questions)} câu hỏi...")
    model_name = "facebook/dpr-question_encoder-single-nq-base"
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
    model = DPRQuestionEncoder.from_pretrained(model_name)
    model.eval()

    queries = []
    with torch.no_grad():
        for q_text in questions:
            inputs = tokenizer(q_text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            # DPR Question Encoder trả về pooled output (CLSToken)
            v = outputs.pooler_output.cpu().numpy()[0]
            queries.append(v)
    
    return np.array(queries, dtype=np.float32)


def build_corpus_from_hf_preembedded(
    cache_path: str,
    dataset_name: str,
    dataset_config: str,
    split: str,
    target_vectors: int,
    vector_fields: Sequence[str],
) -> Tuple[np.memmap, int]:
    """
    Stream embeddings đã có sẵn trên HF và ghi trực tiếp vào .npy memmap.
    Không encode lại.
    """
    if target_vectors <= 0:
        raise RuntimeError("target_vectors phải > 0")
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp_path = cache_path + ".tmp.npy"
    stream = _load_hf_stream(dataset_name, dataset_config, split)

    mm: Optional[np.memmap] = None
    dim: Optional[int] = None
    n = 0
    BATCH_HF = 1000  # Đọc 1000 dòng/lần, giải phóng RAM sau mỗi batch

    for batch in stream.iter(batch_size=BATCH_HF):
        # batch là dict of lists: {"embeddings": [[...], [...], ...], ...}
        vecs_raw = None
        for field in vector_fields:
            if field in batch and batch[field] is not None:
                vecs_raw = batch[field]
                break
        if vecs_raw is None:
            # fallback: tìm field có dạng list-of-list số
            for key, val in batch.items():
                if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                    try:
                        test = np.asarray(val[0], dtype=np.float32)
                        if test.ndim == 1 and test.size > 1:
                            vecs_raw = val
                            break
                    except (ValueError, TypeError):
                        continue
        if vecs_raw is None:
            continue

        # Chuyển thành numpy array 2D
        vecs_np = np.asarray(vecs_raw, dtype=np.float32)
        if vecs_np.ndim == 1:
            continue
        if vecs_np.ndim != 2:
            continue

        if mm is None:
            dim = int(vecs_np.shape[1])
            mm = np.lib.format.open_memmap(
                tmp_path,
                mode="w+",
                dtype=np.float32,
                shape=(target_vectors, dim),
            )
            print(f"Detected embedding field dim={dim}. Start streaming {target_vectors:,} vectors...")

        # Ghi từng batch vào memmap
        valid = vecs_np[vecs_np.shape[1] == dim] if vecs_np.shape[1] != dim else vecs_np
        can_write = min(len(valid), target_vectors - n)
        if can_write <= 0:
            break
        mm[n : n + can_write] = valid[:can_write]
        n += can_write

        # Giải phóng hoàn toàn batch khỏi RAM
        del batch, vecs_raw, vecs_np, valid
        gc.collect()

        if n % 100_000 < BATCH_HF:
            mm.flush()
            print(f"  loaded {n:,}/{target_vectors:,} vectors | RSS ~ {rss_mb():.0f} MB")
        if n >= target_vectors:
            break

    if mm is None or dim is None:
        raise RuntimeError("Không tìm thấy cột embedding phù hợp trong dataset.")
    mm.flush()
    del mm
    gc.collect()

    if n < target_vectors:
        print(f"Warning: chỉ tải được {n:,} vectors (ít hơn target {target_vectors:,}).")
        full_mm = np.load(tmp_path, mmap_mode="r")
        arr = np.asarray(full_mm[:n], dtype=np.float32)
        np.save(cache_path, arr)
        del full_mm, arr
        gc.collect()
        os.remove(tmp_path)
        mm2 = np.load(cache_path, mmap_mode="r")
        return mm2, int(mm2.shape[1])

    if os.path.exists(cache_path):
        os.remove(cache_path)
    os.replace(tmp_path, cache_path)
    mm2 = np.load(cache_path, mmap_mode="r")
    return mm2, int(mm2.shape[1])


def build_query_vectors_from_corpus(
    corpus_mm: np.memmap,
    num_queries: int,
    seed: int,
    noise_std: float,
) -> torch.Tensor:
    """Sinh query từ chính embedding đã có, tránh encode lại."""
    n_total = corpus_mm.shape[0]
    if num_queries <= 0 or num_queries > n_total:
        raise RuntimeError(f"num_queries={num_queries} không hợp lệ với corpus size={n_total}")
    rng = np.random.default_rng(seed)
    q_idx = rng.choice(n_total, size=num_queries, replace=False)
    q_np = np.asarray(corpus_mm[q_idx], dtype=np.float32).copy()
    if noise_std > 0:
        q_np += rng.normal(0.0, noise_std, size=q_np.shape).astype(np.float32)
    q_t = torch.from_numpy(q_np).float()
    return F.normalize(q_t, dim=-1)


def ground_truth_streaming(
    queries_t: torch.Tensor,
    corpus_mm: np.memmap,
    k_max: int,
    chunk_rows: int,
) -> List[List[int]]:
    """Top-k inner product trên corpus đã L2-normalize; quét theo chunk để tiết kiệm RAM."""
    print(f"  GT Brute-force: Scanning {corpus_mm.shape[0]:,} vectors...")
    n, dim = corpus_mm.shape
    num_queries = queries_t.shape[0]
    heaps = [[] for _ in range(num_queries)]
    
    for s in range(0, n, chunk_rows):
        e = min(s + chunk_rows, n)
        chunk = np.asarray(corpus_mm[s:e], dtype=np.float32)
        chunk = chunk / (np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-12)
        scores = chunk @ queries_t.cpu().numpy().T
        
        for qi in range(num_queries):
            q_scores = scores[:, qi]
            for j, val in enumerate(q_scores):
                gid = s + j
                if len(heaps[qi]) < k_max:
                    heapq.heappush(heaps[qi], (float(val), gid))
                elif val > heaps[qi][0][0]:
                    heapq.heapreplace(heaps[qi], (float(val), gid))
        
        if s % 1_000_000 == 0 and s > 0:
            print(f"    {s:,} / {n:,} done...")

    gt = []
    for qi in range(num_queries):
        top = sorted(heaps[qi], key=lambda x: -x[0])
        gt.append([idx for _, idx in top[:k_max]])
    return gt


# --- Evaluators (benchmark_recall + FAISS SQ) ---


# --- Batch Quantizers & Evaluators ---

def quantize_sq_to_disk(corpus_mm: np.memmap, bits: int, out_path: str):
    """Nén SQ và lưu xuống đĩa (memmap)."""
    n, dim = corpus_mm.shape
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # 1. Tính min/max (quét 1 vòng)
    v_min, v_max = float('inf'), float('-inf')
    chunk_size = 100000
    print(f"  SQ-{bits}b: Computing min/max...")
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        chunk = corpus_mm[s:e]
        v_min = min(v_min, np.min(chunk))
        v_max = max(v_max, np.max(chunk))
    
    n_levels = 2**bits - 1
    scale = (v_max - v_min) / n_levels
    
    # 2. Ghi codes
    codes_mm = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.uint8, shape=(n, dim))
    print(f"  SQ-{bits}b: Quantizing {n:,} vectors...")
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        chunk = corpus_mm[s:e]
        codes_mm[s:e] = np.clip((chunk - v_min) / scale, 0, n_levels).astype(np.uint8)
        if s % 500000 == 0:
            print(f"    {s:,} done...")
    codes_mm.flush()
    return {"v_min": float(v_min), "v_max": float(v_max), "scale": float(scale)}

def evaluate_sq_batch(codes_mm, queries_t, ground_truth, k_values, meta):
    n, dim = codes_mm.shape
    v_min, v_max, scale = meta["v_min"], meta["v_max"], meta["scale"]
    queries_np = queries_t.cpu().numpy().astype(np.float32) # [num_queries, dim]
    num_queries = len(queries_np)
    max_k = max(k_values)
    
    # Heaps để lưu Top-K cho từng query
    heaps = [[] for _ in range(num_queries)]
    
    start = time.perf_counter()
    chunk_size = 50000 # Rất nhỏ để an toàn RAM
    
    print(f"  SQ-Batch: Scanning {n:,} vectors for {num_queries} queries...")
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        # 1. Dequantize chunk (Float32)
        batch_vecs = codes_mm[s:e].astype(np.float32)
        batch_vecs *= scale
        batch_vecs += v_min
        
        # 2. Tính scores cho TẤT CẢ queries cùng lúc
        # [num_queries, dim] @ [dim, batch_size] -> [num_queries, batch_size]
        all_scores = queries_np @ batch_vecs.T
        
        # 3. Cập nhật Top-K cho từng query
        for qi in range(num_queries):
            q_scores = all_scores[qi]
            q_heap = heaps[qi]
            for j in range(len(q_scores)):
                val = float(q_scores[j])
                gid = s + j
                if len(q_heap) < max_k:
                    heapq.heappush(q_heap, (val, gid))
                elif val > q_heap[0][0]:
                    heapq.heapreplace(q_heap, (val, gid))
        
        if s % 1000000 == 0 and s > 0:
            print(f"    {s:,} / {n:,} vectors done...")
        del batch_vecs, all_scores
        gc.collect()

    # Tính Accuracy
    all_top1 = {k: [] for k in k_values}
    all_recall = {k: [] for k in k_values}
    for qi in range(num_queries):
        top_indices = [idx for val, idx in sorted(heaps[qi], key=lambda x: -x[0])]
        m = measure_accuracy(top_indices, ground_truth[qi], k_values)
        for k in k_values:
            all_top1[k].append(m["top1_in_k"][k])
            all_recall[k].append(m["set_recall"][k])

    qps = num_queries / (time.perf_counter() - start)
    return {k: np.mean(all_top1[k])*100 for k in k_values}, {k: np.mean(all_recall[k])*100 for k in k_values}, qps

def evaluate_tq_batch(engine, queries_t, ground_truth, k_values):
    """Evaluate TQ dùng Native Batch Search (SIMD Rust) như trong stress_5m.py"""
    all_top1 = {k: [] for k in k_values}
    all_recall = {k: [] for k in k_values}
    max_k = max(k_values)
    
    ivf = engine.current_ivf_data
    pq = ivf.pq_data
    dim = engine.dim
    num_queries = len(queries_t)
    
    # 1. Rotate toàn bộ queries một lần (Batch Rotation)
    q_rot = rotate_forward(queries_t, engine.rot_op_t).cpu().numpy().astype(np.float32)
    
    start = time.perf_counter()
    # 2. Bắn loạt (Native Batch Scan trong Rust)
    # Đây là hàm mạnh nhất, quét toàn bộ queries qua toàn bộ IVF shards trong 1 lần gọi
    _, batch_indices = tq_native.tq_ivf_online_scan(
        q_rot,
        np.ascontiguousarray(pq.sq_codes, dtype=np.uint8),
        np.ascontiguousarray(pq.centroids, dtype=np.float32),
        np.ascontiguousarray(pq.norms, dtype=np.float32),
        np.ascontiguousarray(pq.qjl_signs, dtype=np.uint8),
        np.ascontiguousarray(pq.res_norms, dtype=np.float32),
        q_rot,
        np.ascontiguousarray(ivf.list_offsets, dtype=np.int32),
        np.ascontiguousarray(ivf.coarse_centroids.cpu().numpy(), dtype=np.float32),
        int(ivf.n_probe),
        float(pq.qjl_scale),
        int(dim),
        int(engine.sq_bits),
        int(max_k)
    )
    duration = time.perf_counter() - start
    
    # 3. Tính toán Accuracy
    global_vector_ids = ivf.vector_ids
    for i in range(num_queries):
        # Lấy ID thực tế từ kết quả index của Rust
        top_indices = [global_vector_ids[idx] for idx in batch_indices[i] if idx != -1]
        m = measure_accuracy(top_indices, ground_truth[i], k_values)
        for k in k_values:
            all_top1[k].append(m["top1_in_k"][k])
            all_recall[k].append(m["set_recall"][k])
            
    qps = num_queries / duration
    return {k: np.mean(all_top1[k])*100 for k in k_values}, {k: np.mean(all_recall[k])*100 for k in k_values}, qps


def evaluate_tq_native(vectors, queries, ground_truth, k_values, bits=4):
    print(f"  ---- TQ Flat {bits}-bit (Rust batch scan)...")
    dim = vectors.shape[1]
    num_queries = len(queries)
    n_vectors = len(vectors)

    engine = TQEngine(dim=dim, bits=bits, device="cpu", use_ivf=False)
    engine.index(vectors)

    ivf = engine.current_ivf_data
    pq = ivf.pq_data
    max_k = max(k_values)

    q_rot = rotate_forward(queries, engine.rot_op_t).cpu().numpy().astype(np.float32)

    start = time.perf_counter()
    scores_flat = tq_native.tq_batch_scan(
        q_rot,
        np.ascontiguousarray(pq.sq_codes, dtype=np.uint8),
        np.ascontiguousarray(pq.centroids, dtype=np.float32),
        np.ascontiguousarray(pq.norms, dtype=np.float32),
        np.ascontiguousarray(pq.qjl_signs, dtype=np.uint8),
        np.ascontiguousarray(pq.res_norms, dtype=np.float32),
        q_rot,
        float(pq.qjl_scale),
        int(dim),
        int(engine.sq_bits),
    )

    scores_tensor = torch.from_numpy(scores_flat).reshape(num_queries, n_vectors)
    _, batch_indices = torch.topk(scores_tensor, max_k, dim=1)
    duration = time.perf_counter() - start

    all_top1 = {k: [] for k in k_values}
    all_recall = {k: [] for k in k_values}

    for i in range(num_queries):
        top_indices = batch_indices[i].tolist()
        m = measure_accuracy(top_indices, ground_truth[i], k_values)
        for k in k_values:
            all_top1[k].append(m["top1_in_k"][k])
            all_recall[k].append(m["set_recall"][k])

    qps = num_queries / duration
    m_top1 = {k: np.mean(all_top1[k]) * 100 for k in k_values}
    m_recall = {k: np.mean(all_recall[k]) * 100 for k in k_values}
    return m_top1, m_recall, qps


def evaluate_tq_ivf(
    vectors,
    queries,
    ground_truth,
    k_values,
    bits: int,
    n_probe: int,
    rerank: bool = False,
):
    label = f"TQ-IVF nprobe={n_probe} {bits}-bit" + ("+Rerank" if rerank else "")
    print(f"  ---- {label}...")
    dim = vectors.shape[1]

    engine = TQEngine(
        dim=dim,
        bits=bits,
        device="cpu",
        use_ivf=True,
        ivf_nlist=256,
        ivf_nprobe=n_probe,
    )
    engine.index(vectors, online_clustering=False)

    ivf = engine.current_ivf_data
    ivf.n_probe = n_probe
    pq = ivf.pq_data
    max_k = max(k_values)

    retrieve_k = max_k * 10 if rerank else max_k

    q_rot = rotate_forward(queries, engine.rot_op_t).cpu().numpy().astype(np.float32)

    start = time.perf_counter()
    _, batch_indices = tq_native.tq_ivf_online_scan(
        q_rot,
        np.ascontiguousarray(pq.sq_codes, dtype=np.uint8),
        np.ascontiguousarray(pq.centroids, dtype=np.float32),
        np.ascontiguousarray(pq.norms, dtype=np.float32),
        np.ascontiguousarray(pq.qjl_signs, dtype=np.uint8),
        np.ascontiguousarray(pq.res_norms, dtype=np.float32),
        q_rot,
        np.ascontiguousarray(ivf.list_offsets, dtype=np.int32),
        np.ascontiguousarray(ivf.coarse_centroids.cpu().numpy(), dtype=np.float32),
        int(ivf.n_probe),
        float(pq.qjl_scale),
        int(dim),
        int(engine.sq_bits),
        int(retrieve_k),
    )

    all_top1 = {k: [] for k in k_values}
    all_recall = {k: [] for k in k_values}
    global_vector_ids = ivf.vector_ids

    for i in range(len(queries)):
        candidate_ids = [global_vector_ids[idx] for idx in batch_indices[i] if idx != -1]

        if rerank and len(candidate_ids) > 0:
            candidates_raw = vectors[candidate_ids]
            exact_scores = torch.mm(queries[i : i + 1], candidates_raw.t()).view(-1)
            _, final_idx = torch.topk(exact_scores, min(max_k, len(exact_scores)))
            top_indices = [candidate_ids[j] for j in final_idx.tolist()]
        else:
            top_indices = candidate_ids[:max_k]

        m = measure_accuracy(top_indices, ground_truth[i], k_values)
        for k in k_values:
            all_top1[k].append(m["top1_in_k"][k])
            all_recall[k].append(m["set_recall"][k])

    qps = len(queries) / (time.perf_counter() - start)
    m_top1 = {k: np.mean(all_top1[k]) * 100 for k in k_values}
    m_recall = {k: np.mean(all_recall[k]) * 100 for k in k_values}
    return m_top1, m_recall, qps


def load_vectors_torch(corpus_mm: np.memmap, chunk_rows: int) -> torch.Tensor:
    """Nạp toàn bộ corpus từ memmap vào torch (một lần); caller cần giới hạn max_vectors."""
    n = corpus_mm.shape[0]
    parts = []
    for s in range(0, n, chunk_rows):
        e = min(s + chunk_rows, n)
        parts.append(np.asarray(corpus_mm[s:e], dtype=np.float32))
    stacked = np.concatenate(parts, axis=0)
    t = torch.from_numpy(stacked).float()
    return F.normalize(t, dim=-1)


def run(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.hard_limit_mb > 0:
        enforce_memory_limit(args.hard_limit_mb)
        
    k_values = [int(x) for x in args.k_values.split(",")]
    nprobe_list = [int(x) for x in args.tq_nprobes.split(",")]
    tq_rerank_mult = max(1, int(args.tq_rerank_mult))
    tq_nlists = [int(x) for x in args.tq_nlists.split(",")]

    data_dir = os.path.join(project_root, "Benchmark", "data", "wiki_benchmark")
    os.makedirs(data_dir, exist_ok=True)
    cache_npy = os.path.join(data_dir, "corpus_embedded.npy")
    meta_path = os.path.join(data_dir, "corpus_meta.json")

    if args.max_vectors < args.min_vectors:
        raise SystemExit(f"max_vectors ({args.max_vectors}) too low; need at least {args.min_vectors} vectors.")

    print(f"Target: Load {args.max_vectors:,} DPR vectors from HF (no re-embed)")

    # Check if normalized cache exists
    cache_norm_npy = cache_npy.replace(".npy", "_norm.npy")
    cache_norm_exists = os.path.exists(cache_norm_npy)
    
    # Check if raw cache exists
    cache_exists = os.path.exists(cache_npy)
    
    if cache_norm_exists:
        print("Normalized cache found. Bypassing raw data download.")
        cache_exists = True  # Bỏ qua download nếu đã có bản chuẩn hóa
    elif cache_exists:
        existing_mm = np.load(cache_npy, mmap_mode='r')
        if existing_mm.shape[0] != args.max_vectors:
            print(f"Cache has {existing_mm.shape[0]:,} vectors, but you requested {args.max_vectors:,}.")
            print("   Auto-enabling --rebuild-cache to download correct count.")
            args.rebuild_cache = True
        del existing_mm

    if (args.rebuild_cache and not cache_norm_exists) or not cache_exists:
        corpus_mm, dim = build_corpus_from_hf_preembedded(
            cache_npy,
            dataset_name=args.hf_dataset,
            dataset_config=args.hf_config,
            split=args.hf_split,
            target_vectors=args.max_vectors,
            vector_fields=[x.strip() for x in args.hf_vector_fields.split(",") if x.strip()],
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dim": dim,
                    "n": int(corpus_mm.shape[0]),
                    "dataset": args.hf_dataset,
                    "config": args.hf_config,
                    "split": args.hf_split,
                },
                f,
                indent=2,
            )
        del corpus_mm
        gc.collect()

    cache_norm_npy = cache_npy.replace(".npy", "_norm.npy")
    
    if os.path.exists(cache_norm_npy):
        print(f"Normalized corpus found. Loading directly...")
        corpus_mm = np.load(cache_norm_npy, mmap_mode="r")
        n_total, dim = corpus_mm.shape
    else:
        corpus_mm_raw = np.load(cache_npy, mmap_mode="r")
        n_total, dim = corpus_mm_raw.shape
        
        # CHUẨN HÓA TOÀN BỘ DỮ LIỆU ĐỂ DOT PRODUCT == COSINE SIMILARITY
        print(f"Normalizing 5M vectors for consistent benchmark...")
        print("Creating normalized corpus memmap...")
        norm_mm = np.lib.format.open_memmap(cache_norm_npy, mode='w+', dtype='float32', shape=(n_total, dim))
        for i in range(0, n_total, 100000):
            chunk = corpus_mm_raw[i:i+100000].copy()
            norms = np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-10
            norm_mm[i:i+100000] = chunk / norms
            if i % 1000000 == 0: print(f"  Normalized {i:,} vectors...")
        norm_mm.flush()
        del norm_mm
        del corpus_mm_raw
        gc.collect()
        corpus_mm = np.load(cache_norm_npy, mmap_mode="r")
    
    corpus_mm = np.load(cache_norm_npy, mmap_mode="r")
    norm_check = np.linalg.norm(corpus_mm[0])
    print(f"Normalized Corpus: {n_total:,} x {dim} | Norm[0]: {norm_check:.4f} | RSS ~ {rss_mb():.0f} MB")

    actual_num_queries = args.num_queries
    if args.query_json and os.path.exists(args.query_json):
        with open(args.query_json, 'r', encoding='utf-8') as f:
            actual_num_queries = len(json.load(f))
            args.num_queries = actual_num_queries

    suffix = "_real" if args.query_json else ""
    gt_path = os.path.join(data_dir, f"gt_indices_{actual_num_queries}q_{n_total}v{suffix}.npy")
    q_path = os.path.join(data_dir, f"queries_{actual_num_queries}q{suffix}.pt")

    # Force regeneration for normalized consistency
    is_normalized_run = os.path.exists(gt_path.replace(".npy", ".txt"))
    
    if not args.rebuild_cache and not args.rebuild_gt and os.path.exists(gt_path) and os.path.exists(q_path) and is_normalized_run:
        print(f"Loading Normalized Ground Truth ({n_total:,}v) & Queries from cache...")
        queries_t = torch.load(q_path, weights_only=True)
        ground_truth = np.load(gt_path)
    else:
        queries_np = None
        if args.query_json:
            queries_np = _get_real_queries(args.query_json, dim)
            if queries_np is not None:
                # Chuẩn hóa queries để Dot Product == Cosine
                q_norms = np.linalg.norm(queries_np, axis=1, keepdims=True) + 1e-10
                queries_np = queries_np / q_norms
                print(f"✅ Loaded and Normalized {len(queries_np)} real queries from JSON.")
        
        if queries_np is None:
            print(f"Generating {args.num_queries} random queries from corpus...")
            indices = np.random.choice(n_total, args.num_queries, replace=False)
            queries_np = corpus_mm[indices].copy()
        
        actual_num_queries = len(queries_np)
        queries_t = torch.from_numpy(queries_np).float()
        
        # Calculate Ground Truth using normalized dot product
        print(f"Calculating exact Ground Truth for {actual_num_queries} queries on 5M vectors...")
        ground_truth = np.zeros((actual_num_queries, 100), dtype=np.int32)
        for i in range(actual_num_queries):
            q = queries_np[i]
            # Quét Dot Product trên toàn bộ 5M vectors (mmap)
            scores = np.dot(corpus_mm, q) 
            ground_truth[i] = np.argsort(scores)[-100:][::-1]
            if i % 10 == 0: print(f"  GT Progress: {i}/{actual_num_queries}...")
        
        # Save cache
        np.save(gt_path, ground_truth)
        torch.save(queries_t, q_path)
        with open(gt_path.replace(".npy", ".txt"), 'w') as f: f.write("Normalized")

    # --- PHASE 1: PREPARATION (Indexing everything first) ---
    print("\n" + "="*30)
    print("PHASE 1: PREPARING ALL INDICES")
    print("="*30)
    
    bit_modes = [2, 4]
    for bits in bit_modes:
        # TQ Preparation (sweep nlist)
        for nlist in tq_nlists:
            tq_path = os.path.join(data_dir, f"tq_index_{bits}b_nl{nlist}")
            if args.rebuild_cache or not os.path.exists(os.path.join(tq_path, "metadata.json")):
                print(f"Building TQ-{bits}b index (nlist={nlist})...")
                engine = TQEngine(dim=dim, bits=bits, device="cpu", use_ivf=True, ivf_nlist=nlist)
                engine.index(corpus_mm, save_path=tq_path)
                engine.save_index(tq_path)
                del engine
                gc.collect()
            else:
                print(f"TQ-{bits}b index (nlist={nlist}) already exists. Skipping.")

        # SQ Preparation (FAISS IVF-SQ In-Memory)
        # NOTE: "FAISS-SQ 2b" is misleading because FAISS ScalarQuantizer doesn't provide true 2-bit;
        # the old code was falling back to QT_4bit. We skip SQ at 2b to keep comparisons clean.
        if bits != 2:
            sq_faiss_path = os.path.join(data_dir, f"sq_faiss_{bits}b.index")
            if args.rebuild_cache or not os.path.exists(sq_faiss_path):
                print(f"Building FAISS IVF-SQ-{bits}b (In-Memory)...")
                try:
                    if bits == 4:
                        qtype = faiss.ScalarQuantizer.QT_4bit
                    elif bits == 8:
                        qtype = faiss.ScalarQuantizer.QT_8bit
                    else:
                        print(f"⚠️ FAISS SQ không hỗ trợ {bits}b chính thức. Dùng tạm QT_4bit cho so sánh.")
                        qtype = faiss.ScalarQuantizer.QT_4bit
                    quantizer = faiss.IndexFlatIP(dim)
                    sq_index = faiss.IndexIVFScalarQuantizer(quantizer, dim, 1024, qtype, faiss.METRIC_INNER_PRODUCT)
                    sq_index.train(corpus_mm[:65536].copy())
                    
                    # Nạp chunk cực nhỏ để tránh bad_alloc
                    sq_chunk = 5000
                    for i in range(0, n_total, sq_chunk):
                        actual_end = min(i + sq_chunk, n_total)
                        sq_index.add(corpus_mm[i:actual_end].copy())
                        if i % 500000 == 0:
                            print(f"    SQ added {i:,}/{n_total:,} vectors...")
                    
                    faiss.write_index(sq_index, sq_faiss_path)
                    del sq_index; gc.collect()
                except MemoryError:
                    print("SKIPPING FAISS-SQ: Memory limit reached (5M vectors requires >2GB RAM in FAISS).")
            else:
                print(f"FAISS IVF-SQ-{bits}b index already exists. Skipping.")

        # PQ Preparation (FAISS In-Memory)
        pq_cache_path = os.path.join(data_dir, f"pq_index_{bits}b.faiss")
        if args.rebuild_cache or not os.path.exists(pq_cache_path):
            print(f"Building FAISS PQ-{bits}b (In-Memory)...")
            try:
                m = dim // 2 if bits == 4 else dim // 4
                quantizer = faiss.IndexFlatIP(dim)
                pq_index = faiss.IndexIVFPQ(quantizer, dim, 1024, m, 8)
                pq_index.metric_type = faiss.METRIC_INNER_PRODUCT
                pq_index.cp.ntrial = 20
                pq_index.train(corpus_mm[:65536].copy())
                
                pq_chunk = 10000
                for i in range(0, n_total, pq_chunk):
                    actual_end = min(i + pq_chunk, n_total)
                    pq_index.add(corpus_mm[i:actual_end].copy())
                    if i % 500000 == 0:
                        print(f"    PQ added {i:,}/{n_total:,} vectors...")
                faiss.write_index(pq_index, pq_cache_path)
                del pq_index; gc.collect()
            except MemoryError:
                print("SKIPPING FAISS-PQ: Memory limit reached.")
        else:
            print(f"FAISS PQ-{bits}b index already exists. Skipping.")

    print("\n" + "="*30)
    print("PHASE 2: EVALUATING ALL MODELS")
    print("="*30)
    
    results: List[Dict[str, Any]] = []
    for bits in bit_modes:
        for nlist in tq_nlists:
            tq_path = os.path.join(data_dir, f"tq_index_{bits}b_nl{nlist}")
            if os.path.exists(tq_path):
                # Lưu mốc RAM trước khi nạp mô hình
                priv_before, rss_before = get_ram_info()
                
                engine = TQEngine(dim=dim, bits=bits, device="cpu", use_ivf=True, ivf_nlist=nlist)
                engine.load_index(tq_path)
                
                for nprobe in nprobe_list:
                    engine.ivf_nprobe = nprobe
                    print(f"Evaluating TQ-IVF {bits}b (nlist={nlist}, nprobe={nprobe}, rerank_mult={tq_rerank_mult})...")
                    start_t = time.time()
                    retrieve_k = max(k_values) * tq_rerank_mult
                    tq_results = engine.search_batch(queries_t, top_k=retrieve_k)
                    qps = len(queries_t) / (time.time() - start_t)
                    
                    I = np.zeros((len(queries_t), max(k_values)), dtype=np.int64)
                    for i, (ids, _) in enumerate(tq_results):
                        candidate_ids = ids.cpu().numpy().astype(np.int64, copy=False)
                        candidate_ids = candidate_ids[candidate_ids >= 0]
                        if candidate_ids.size == 0:
                            continue

                        if tq_rerank_mult > 1 and candidate_ids.size > max(k_values):
                            candidates_raw = torch.from_numpy(np.asarray(corpus_mm[candidate_ids], dtype=np.float32))
                            exact_scores = torch.mv(candidates_raw, queries_t[i].cpu())
                            keep = min(max(k_values), exact_scores.numel())
                            _, top_idx = torch.topk(exact_scores, keep)
                            final_ids = candidate_ids[top_idx.numpy()]
                            I[i, :len(final_ids)] = final_ids
                        else:
                            keep = min(max(k_values), candidate_ids.size)
                            I[i, :keep] = candidate_ids[:keep]
                    
                    m_t, m_r = {}, {}
                    for k in k_values:
                        acc, rec = 0, 0
                        for i in range(len(queries_t)):
                            pred = I[i, :k]
                            if ground_truth[i, 0] in pred: acc += 1
                            gt_set, pred_set = set(ground_truth[i, :k]), set(pred)
                            rec += len(gt_set.intersection(pred_set)) / k
                        m_t[k], m_r[k] = (acc / len(queries_t)) * 100, (rec / len(queries_t)) * 100

                    priv_now, rss_now = get_ram_info()
                    ram_priv_inc = max(0.0, priv_now - priv_before)

                    rr_tag = f" rr{x}" if (x := tq_rerank_mult) > 1 else ""
                    results.append({
                        "label": f"TQ-IVF nl{nlist} np{nprobe} {bits}b{rr_tag}", 
                        "top1": m_t, "recall": m_r, "qps": qps, 
                        "priv_mb": ram_priv_inc, "rss_mb": rss_now
                    })
                    
                    # Dọn dẹp Page Cache ngay sau mỗi nprobe để lượt sau đo chính xác
                    empty_working_set() 
                
                del engine
                gc.collect()
                empty_working_set()

        # 2. FAISS SQ Evaluation (IVF-SQ)
        if bits != 2:
            sq_faiss_path = os.path.join(data_dir, f"sq_faiss_{bits}b.index")
            if os.path.exists(sq_faiss_path):
                print(f"Evaluating FAISS SQ {bits}b (IVF)...")
                try:
                    gc.collect()
                    priv_before, rss_before = get_ram_info()
                    sq_faiss_index = faiss.read_index(sq_faiss_path)
                    
                    priv_now, _ = get_ram_info()
                    ram_sq = max(0.0, priv_now - priv_before)
                    
                    sq_faiss_index.nprobe = 64
                    
                    start_sq = time.time()
                    D, I = sq_faiss_index.search(queries_t.cpu().numpy(), max(k_values))
                    qps_sq = len(queries_t) / (time.time() - start_sq)
                    
                    m_t_sq, m_r_sq = {}, {}
                    for k in k_values:
                        acc, rec = 0, 0
                        for i in range(len(queries_t)):
                            pred = I[i, :k]
                            if ground_truth[i, 0] in pred: acc += 1
                            gt_set, pred_set = set(ground_truth[i, :k]), set(pred)
                            rec += len(gt_set.intersection(pred_set)) / k
                        m_t_sq[k], m_r_sq[k] = (acc / len(queries_t)) * 100, (rec / len(queries_t)) * 100

                    priv_sq, _ = get_ram_info()
                    ram_sq = priv_sq - priv_before
                    
                    _, rss_after_sq = get_ram_info()
                        
                    results.append({
                        "label": f"FAISS-SQ {bits}b", 
                        "top1": m_t_sq, "recall": m_r_sq, "qps": qps_sq, 
                        "priv_mb": ram_sq, "rss_mb": rss_after_sq
                    })
                    del sq_faiss_index; gc.collect()
                    empty_working_set() # Dọn dẹp Cache
                except Exception as e:
                    print(f"Error evaluating FAISS SQ: {e}")
            else:
                print(f"FAISS SQ index {sq_faiss_path} not found. Skipping evaluation.")

        # 3. FAISS PQ Evaluation (IVF)
        pq_cache_path = os.path.join(data_dir, f"pq_index_{bits}b.faiss")
        if os.path.exists(pq_cache_path):
            print(f"Evaluating FAISS PQ {bits}b (IVF)...")
            try:
                gc.collect()
                priv_before, rss_before = get_ram_info()
                pq_index = faiss.read_index(pq_cache_path)
                
                priv_now, _ = get_ram_info()
                ram_pq = max(0.0, priv_now - priv_before)
                
                pq_index.nprobe = 64
                start_pq = time.time()
                D, I = pq_index.search(queries_t.cpu().numpy(), max(k_values))
                qps_pq = len(queries_t) / (time.time() - start_pq)
                
                m_t_pq, m_r_pq = {}, {}
                for k in k_values:
                    acc, rec = 0, 0
                    for i in range(len(queries_t)):
                        pred = I[i, :k]
                        if ground_truth[i, 0] in pred: acc += 1
                        gt_set, pred_set = set(ground_truth[i, :k]), set(pred)
                        rec += len(gt_set.intersection(pred_set)) / k
                    m_t_pq[k], m_r_pq[k] = (acc / len(queries_t)) * 100, (rec / len(queries_t)) * 100

                priv_pq, _ = get_ram_info()
                ram_pq = priv_pq - priv_before
                
                _, rss_after_pq = get_ram_info()
                    
                results.append({
                    "label": f"FAISS-PQ {bits}b", 
                    "top1": m_t_pq, "recall": m_r_pq, "qps": qps_pq, 
                    "priv_mb": ram_pq, "rss_mb": rss_after_pq
                })
                del pq_index; gc.collect()
                empty_working_set() # Dọn dẹp Cache
            except Exception as e:
                print(f"Error evaluating FAISS PQ: {e}")
        else:
            print(f"FAISS PQ index {pq_cache_path} not found. Skipping evaluation.")

    hdr = " | ".join([f"P@K={k:<2}" for k in k_values])
    print("\n" + "=" * 110)
    print("TOP-1 IN K (%)")
    print("=" * 110)
    print(f"{'Method':<28} | {hdr} | {'QPS':>8}")
    print("-" * 110)
    for res in results:
        cols = " | ".join([f"{res['top1'][k]:5.1f}%" for k in k_values])
        print(f"{res['label']:<28} | {cols} | {res['qps']:>8.1f}")

    hdr2 = " | ".join([f"R@K={k:<2}" for k in k_values])
    print("\n" + "=" * 125)
    print("SET RECALL@K (%)")
    print("=" * 125)
    print(f"{'Method':<28} | {hdr2} | {'QPS':>8} | {'Private(MB)':>11} | {'WorkSet(MB)':>11}")
    print("-" * 145)
    for res in results:
        cols = " | ".join([f"{res['recall'][k]:5.1f}%" for k in k_values])
        print(f"{res['label']:<28} | {cols} | {res['qps']:>8.1f} | {res.get('priv_mb', 0):>11.1f} | {res.get('rss_mb', 0):>11.1f}")

    out_dir = os.path.join(project_root, "benchmark_result")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "benchmark_results.json")
    serializable = []
    for r in results:
        serializable.append(
            {
                "label": r["label"],
                "qps": r["qps"],
                "top1": {str(k): float(v) for k, v in r["top1"].items()},
                "recall": {str(k): float(v) for k, v in r["recall"].items()},
                "ram_mb": r.get("ram_mb", 0.0),
                "ram_cache_mb": r.get("ram_cache_mb", 0.0),
            }
        )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"k_values": k_values, "results": serializable}, f, indent=2)
    print(f"Saved results to JSON: {out_json}")
    return results


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark pre-embedded DPR + FAISS/SQ/PQ/TQ")
    p.add_argument("--max-ram-gb", type=float, default=2.0, help="Heuristic giới hạn corpus (float32)")
    p.add_argument("--dim-hint", type=int, default=768, help="Chiều DPR base (768)")
    p.add_argument("--max-vectors", type=int, default=5_000_000, help="Số vector cần tải từ HF")
    p.add_argument("--eval-max-vectors", type=int, default=0, help="Giới hạn benchmark in-memory (0 = auto theo RAM)")
    p.add_argument("--min-vectors", type=int, default=4096, help="Tối thiểu vector để benchmark ổn định")
    p.add_argument("--num-queries", type=int, default=100)
    p.add_argument("--k-values", type=str, default="1,2,4,8,16,32,64")
    p.add_argument("--tq-nprobes", type=str, default="2,4,8,16,32,64", help="TQ-IVF nprobe sweep")
    p.add_argument("--tq-nlists", type=str, default="4096,8192", help="TQ-IVF nlist sweep (needs rebuilt TQ indexes).")
    p.add_argument(
        "--tq-rerank-mult",
        type=int,
        default=10,
        help="Shortlist factor cho rerank exact của TQ-IVF (1 = tắt; mặc định 10 = lấy 10×K rồi rerank về K).",
    )
    p.add_argument(
        "--hf-dataset",
        type=str,
        default="facebook/wiki_dpr",
    )
    p.add_argument(
        "--hf-config",
        type=str,
        default="psgs_w100.multiset.compressed",
    )
    p.add_argument("--hf-split", type=str, default="train")
    p.add_argument(
        "--hf-vector-fields",
        type=str,
        default="embeddings,embedding,vector,dpr,ctx_embedding",
        help="Danh sách cột embedding ưu tiên, phân tách bằng dấu phẩy.",
    )
    p.add_argument("--query-seed", type=int, default=42, help="Seed chọn query vectors từ corpus")
    p.add_argument(
        "--query-noise-std",
        type=float,
        default=0.01,
        help="Nhiễu Gaussian thêm vào query lấy từ corpus (0 để dùng y hệt vector gốc)",
    )
    p.add_argument("--gt-chunk-rows", type=int, default=8192, help="Chunk khi quét ground truth")
    p.add_argument("--load-chunk-rows", type=int, default=65536, help="Chunk khi nạp memmap -> torch")
    p.add_argument("--rebuild-cache", action="store_true", help="Bỏ cache và tải lại từ HF dataset")
    p.add_argument("--rebuild-gt", action="store_true", help="Chỉ tính toán lại Ground Truth cho Queries")
    p.add_argument("--query-json", type=str, default="", help="Path to JSON file with text questions to embed.")
    p.add_argument("--hard-limit-mb", type=int, default=0, help="Ép giới hạn RAM cứng (chỉ Windows). VD: 512")
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())
