from qdrant_client.http import models

class AdaptiveBuilder:
    def __init__(self, dimension=768):
        self.dimension = dimension

    def get_storage_config(self):
        """Standard config with no quantization (adaptive logic is search-side)."""
        return {
            "quantization": None,
            "hnsw": None
        }

    def build_index(self, embeddings):
        """Returns nothing extra for payload."""
        return None
