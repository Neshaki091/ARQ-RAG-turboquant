import numpy as np

class ManualSQ8:
    def __init__(self, d):
        self.d = d
        self.min_val = None
        self.max_val = None

    def train(self, X):
        self.min_val = np.min(X, axis=0)
        self.max_val = np.max(X, axis=0) + 1e-10

    def quantize(self, X):
        X_scaled = (X - self.min_val) / (self.max_val - self.min_val)
        return (X_scaled * 255).astype(np.uint8)

    def compute_scores(self, query, codes):
        q_scaled = (query - self.min_val) / (self.max_val - self.min_val)
        q_int = (q_scaled * 255).astype(np.float32)
        diffs = codes.astype(np.float32) - q_int
        dist = np.linalg.norm(diffs, axis=1)
        return -dist
