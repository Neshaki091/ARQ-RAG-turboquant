import numpy as np
import faiss
import os

class TurboQuantMSE:
    def __init__(self, d, b, random_state=None):
        self.d = d
        self.b = b
        self.num_centroids = 2 ** b
        if random_state is None:
            random_state = np.random.RandomState(42)
        H = random_state.randn(d, d)
        Q, _ = np.linalg.qr(H)
        self.Pi = Q 
        self.centroids = np.zeros(self.num_centroids)

    def quantize_batch(self, X):
        Y = np.dot(X, self.Pi.T)
        diffs = np.abs(Y[:, :, np.newaxis] - self.centroids[np.newaxis, np.newaxis, :])
        idx = np.argmin(diffs, axis=2)
        return idx

    def dequantize_batch(self, idx):
        Y_tilde = self.centroids[idx]
        X_tilde = np.dot(Y_tilde, self.Pi)
        return X_tilde

class TurboQuantProd:
    def __init__(self, d, b, random_state=None):
        self.d = d
        self.b = b
        if random_state is None:
            random_state = np.random.RandomState(42)
        self.tq_mse = TurboQuantMSE(d, b - 1, random_state=random_state)
        self.S = random_state.randn(d, d)
        self.alpha = np.sqrt(np.pi / 2.0) / d

    def quantize_batch(self, X):
        idx = self.tq_mse.quantize_batch(X)
        X_tilde_mse = self.tq_mse.dequantize_batch(idx)
        R = X - X_tilde_mse
        gamma = np.linalg.norm(R, axis=1, ord=2) 
        TR = np.dot(R, self.S.T)
        qjl = np.sign(TR).astype(np.int8)
        qjl[qjl == 0] = 1 
        return idx, qjl, gamma

    def compute_score_batch(self, query, idx_batch, qjl_batch, gamma_batch, orig_norms=None):
        q_pi = np.dot(self.tq_mse.Pi, query)
        q_s = np.dot(self.S, query)
        mse_scores = np.dot(self.tq_mse.centroids[idx_batch], q_pi)
        qjl_dot = np.dot(qjl_batch.astype(float), q_s)
        qjl_scores = self.alpha * gamma_batch * qjl_dot
        scores = mse_scores + qjl_scores
        if orig_norms is not None:
            scores = scores * orig_norms
        return scores

    def reconstruct_batch(self, idx, qjl, gamma):
        """Tái tạo vector xấp xỉ từ các mã nén ARQ."""
        X_mse = self.tq_mse.dequantize_batch(idx)
        # Residual approximation: R ~ alpha * gamma * (qjl * S)
        R_approx = self.alpha * gamma[:, np.newaxis] * np.dot(qjl.astype(float), self.S)
        return X_mse + R_approx

