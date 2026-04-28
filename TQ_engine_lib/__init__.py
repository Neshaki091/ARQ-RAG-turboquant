import torch
from .quantizer import TQEngine, ProdQuantized
from .codebook import ScalarQuantizer
try:
    from . import tq_native_lib
except ImportError:
    pass

__version__ = "0.3.0"

class TurboQuant:
    """
    High-level API for TurboQuant Vector Search.
    
    Usage:
        tq = TurboQuant(dim=768, bits=4)
        tq.index(vectors)
        indices, scores = tq.search(query, top_k=10)
    """
    def __init__(self, dim: int, bits: int = 4, device: str = None):
        self.engine = TQEngine(dim=dim, bits=bits, device=device)
        self.pq_data = None

    def index(self, vectors: torch.Tensor, online_clustering: bool = False):
        """Build the index from vectors."""
        self.pq_data = self.engine.quantize(vectors, online_clustering=online_clustering)
        print(f"TurboQuant: Indexed {vectors.shape[0]} vectors.")

    def search(self, query: torch.Tensor, top_k: int = 10):
        """Search the index for the given query."""
        if self.pq_data is None:
            raise ValueError("Index is empty. Call .index() first.")
        return self.engine.native_cosine_search(query, self.pq_data, top_k=top_k)
