import os
import sys
import subprocess
import shutil
import time
import torch

def should_rebuild(core_dir, binary_path):
    """Kiểm tra xem code Rust có mới hơn file binary hiện tại không."""
    if not os.path.exists(binary_path):
        return True
    
    binary_time = os.path.getmtime(binary_path)
    src_dir = os.path.join(core_dir, "src")
    
    if not os.path.exists(src_dir):
        return False
        
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith(".rs"):
                if os.path.getmtime(os.path.join(root, f)) > binary_time:
                    return True
    return False

def build_native_core(force=False):
    """Tự động biên dịch lõi Rust SIMD nếu thiếu hoặc code đã thay đổi."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    core_dir = os.path.join(base_dir, "core")
    binary_name = "tq_native_lib.pyd" if os.name == "nt" else "tq_native_lib.so"
    binary_path = os.path.join(base_dir, binary_name)
    
    if not os.path.exists(core_dir):
        return False

    if not force and not should_rebuild(core_dir, binary_path):
        return True

    print(f"--- TurboQuant: {'Code changed! ' if os.path.exists(binary_path) else ''}Compiling Rust SIMD core ---")
    
    # 1. Check Cargo
    try:
        subprocess.run(["cargo", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("TurboQuant Error: 'cargo' not found. Please install Rust.")
        return False

    # 2. Build
    try:
        subprocess.run(["cargo", "build", "--release"], cwd=core_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"TurboQuant Error: Build failed: {e}")
        return False

    # 3. Copy & Cleanup
    target_dir = os.path.join(core_dir, "target", "release")
    search_ext = ".dll" if os.name == "nt" else ".so"
    
    found_lib = None
    for f in os.listdir(target_dir):
        if f.startswith("tq_native_lib") and f.endswith(search_ext):
            found_lib = f
            break
    
    if found_lib:
        shutil.copy2(os.path.join(target_dir, found_lib), binary_path)
        print(f"TurboQuant: Build successful -> {binary_name}")
        
        # Cleanup
        try:
            shutil.rmtree(os.path.join(core_dir, "target"), ignore_errors=True)
            if os.path.exists(os.path.join(core_dir, "Cargo.lock")):
                os.remove(os.path.join(core_dir, "Cargo.lock"))
        except: pass
        return True
    
    return False

# --- AUTO-INIT PHASE ---
# Đảm bảo thư viện luôn mới nhất trước khi import
build_native_core()

try:
    from . import tq_native_lib
except ImportError:
    print("TurboQuant Warning: Native SIMD mode disabled (Build failed or file missing).")

from .quantizer import TQEngine, ProdQuantized
from .codebook import ScalarQuantizer

__version__ = "0.4.1"

class TurboQuant:
    """
    High-level API for TurboQuant Vector Search.
    
    Hỗ trợ tự động biên dịch và nén SQ+QJL với IVF.
    """
    def __init__(self, dim: int, bits: int = 4, device: str = None, 
                 use_ivf: bool = False, ivf_nlist: int = 1024, ivf_nprobe: int = 32):
        self.engine = TQEngine(dim=dim, bits=bits, device=device, 
                               use_ivf=use_ivf, ivf_nlist=ivf_nlist, ivf_nprobe=ivf_nprobe)
        self.pq_data = None

    def index(self, vectors: torch.Tensor, online_clustering: bool = False):
        """Lập chỉ mục dữ liệu."""
        self.pq_data = self.engine.quantize(vectors, online_clustering=online_clustering)
        print(f"TurboQuant: Đã lập chỉ mục {vectors.shape[0]} vectors.")

    def search(self, query: torch.Tensor, top_k: int = 10):
        """Tìm kiếm Top-K."""
        if self.pq_data is None:
            raise ValueError("Index trống. Vui lòng gọi .index() trước.")
        return self.engine.native_cosine_search(query, self.pq_data, top_k=top_k)

    def save_index(self, directory: str, prefix: str):
        """Lưu chỉ mục xuống đĩa."""
        import os
        import numpy as np
        from .quantizer import IVFData, ProdQuantized
        
        os.makedirs(directory, exist_ok=True)
        if self.pq_data is None: raise ValueError("Index trống.")
            
        is_ivf = isinstance(self.pq_data, IVFData)
        config = {"dim": self.engine.dim, "bits": self.engine.bits, "use_ivf": is_ivf}
        
        if is_ivf:
            config["ivf_nlist"] = self.pq_data.n_list
            config["ivf_nprobe"] = self.pq_data.n_probe
            np.savez(os.path.join(directory, f"{prefix}_ivf_meta.npz"), 
                     coarse_centroids=self.pq_data.coarse_centroids.cpu().numpy(),
                     list_offsets=self.pq_data.list_offsets,
                     vector_ids=self.pq_data.vector_ids, **config)
            pq = self.pq_data.pq_data
        else:
            np.savez(os.path.join(directory, f"{prefix}_meta.npz"), **config)
            pq = self.pq_data
            
        np.save(os.path.join(directory, f"{prefix}_sq_codes.npy"), pq.sq_codes)
        np.save(os.path.join(directory, f"{prefix}_qjl_signs.npy"), pq.qjl_signs)
        np.save(os.path.join(directory, f"{prefix}_norms.npy"), pq.norms)
        np.save(os.path.join(directory, f"{prefix}_res_norms.npy"), pq.res_norms)
        
        np.savez(os.path.join(directory, f"{prefix}_pq_meta.npz"),
                 centroids=pq.centroids, dim=pq.dim, sq_bits=pq.sq_bits,
                 total_bits=pq.total_bits, qjl_scale=pq.qjl_scale, rot_op=pq.rot_op)
        print(f"TurboQuant: Đã lưu tại {directory} với prefix '{prefix}'.")

    def load_index(self, directory: str, prefix: str):
        """Tải chỉ mục từ đĩa."""
        import os
        import numpy as np
        from .quantizer import IVFData, ProdQuantized
        
        meta_ivf_path = os.path.join(directory, f"{prefix}_ivf_meta.npz")
        meta_flat_path = os.path.join(directory, f"{prefix}_meta.npz")
        is_ivf = os.path.exists(meta_ivf_path)
        
        if is_ivf:
            meta = np.load(meta_ivf_path)
            self.engine.use_ivf = True
            self.engine.ivf_nlist = int(meta["ivf_nlist"])
            self.engine.ivf_nprobe = int(meta["ivf_nprobe"])
        else:
            meta = np.load(meta_flat_path)
            self.engine.use_ivf = False
            
        self.engine.dim = int(meta["dim"])
        self.engine.bits = int(meta["bits"])
        pq_meta = np.load(os.path.join(directory, f"{prefix}_pq_meta.npz"))
        
        pq = ProdQuantized(
            sq_codes=np.load(os.path.join(directory, f"{prefix}_sq_codes.npy"), mmap_mode='r'),
            qjl_signs=np.load(os.path.join(directory, f"{prefix}_qjl_signs.npy"), mmap_mode='r'),
            norms=np.load(os.path.join(directory, f"{prefix}_norms.npy"), mmap_mode='r'),
            res_norms=np.load(os.path.join(directory, f"{prefix}_res_norms.npy"), mmap_mode='r'),
            centroids=pq_meta["centroids"], dim=int(pq_meta["dim"]),
            sq_bits=int(pq_meta["sq_bits"]), total_bits=int(pq_meta["total_bits"]),
            qjl_scale=float(pq_meta["qjl_scale"]), rot_op=pq_meta["rot_op"]
        )
        
        if is_ivf:
            self.pq_data = IVFData(
                coarse_centroids=torch.from_numpy(meta["coarse_centroids"]).to(self.engine.device),
                pq_data=pq, vector_ids=meta["vector_ids"], list_offsets=meta["list_offsets"],
                n_list=self.engine.ivf_nlist, n_probe=self.engine.ivf_nprobe)
        else:
            self.pq_data = pq
        print(f"TurboQuant: Đã tải chỉ mục từ {directory}.")
