import numpy as np
import faiss
import os

class TurboQuantMSE:
    """
    Giai đoạn 1: TurboQuant tối ưu hóa Lỗi bình phương trung bình (MSE)
    Dựa trên Algorithm 1: TurboQuant_mse từ Google Paper
    """
    def __init__(self, d, b, random_state=None):
        self.d = d
        self.b = b
        self.num_centroids = 2 ** b
        
        if random_state is None:
            random_state = np.random.RandomState(42)
        
        # 1. Tạo ma trận xoay trực giao Pi (d x d) qua QR decomposition
        H = random_state.randn(d, d)
        Q, _ = np.linalg.qr(H)
        self.Pi = Q 
        
        # 2. Centroids sẽ được huấn luyện bằng Lloyd-Max (1D K-Means)
        # Khởi tạo mặc định, sẽ được cập nhật sau khi gọi train()
        self.centroids = np.zeros(self.num_centroids)

    def quantize_batch(self, X):
        """ Nén hàng loạt: Y = X * Pi^T """
        Y = np.dot(X, self.Pi.T)
        
        # Tìm index của centroid gần nhất cho từng tọa độ: idx_j = argmin |y_j - c_k|
        # Broadingcasting để tính toán nhanh cho ma trận
        diffs = np.abs(Y[:, :, np.newaxis] - self.centroids[np.newaxis, np.newaxis, :])
        idx = np.argmin(diffs, axis=2)
        return idx # Trả về ma trận (N, d) chứa các index

    def dequantize_batch(self, idx):
        """ Giải nén hàng loạt: X_tilde = Y_tilde * Pi """
        Y_tilde = self.centroids[idx]
        X_tilde = np.dot(Y_tilde, self.Pi)
        return X_tilde

class TurboQuantProd:
    """
    Giai đoạn 2: TurboQuant tối ưu hóa Tích vô hướng (Inner Product)
    Kết hợp MSE với QJL để bảo toàn tích vô hướng (ADC)
    """
    def __init__(self, d, b, random_state=None):
        self.d = d
        self.b = b
        if random_state is None:
            random_state = np.random.RandomState(42)
        
        # 1. Khởi tạo MSE part với (b-1) bits
        self.tq_mse = TurboQuantMSE(d, b - 1, random_state=random_state)
        
        # 2. Khởi tạo ma trận ngẫu nhiên S cho QJL
        self.S = random_state.randn(d, d)
        self.alpha = np.sqrt(np.pi / 2.0) / d

    def quantize_batch(self, X):
        """ 
        Nén (Quantization) theo chuẩn Google Paper:
        Nhận vào X đã được CHUẨN HÓA L2.
        """
        """ Nén (Quantization) theo chuẩn Google Paper """
        # Nén phần MSE
        idx = self.tq_mse.quantize_batch(X)
        X_tilde_mse = self.tq_mse.dequantize_batch(idx)
        
        # Tính phần dư (Residuals)
        R = X - X_tilde_mse
        gamma = np.linalg.norm(R, axis=1, ord=2) 
        
        # Biến đổi QJL lên phần dư: sign(S * R^T)
        TR = np.dot(R, self.S.T)
        qjl = np.sign(TR).astype(np.int8)
        qjl[qjl == 0] = 1 
        
        return idx, qjl, gamma

    def compute_score_direct(self, query, idx, qjl, gamma):
        """ 
        TÍNH ĐIỂM TRỰC TIẾP TRÊN MÃ NÉN (ADC - Asymmetric Distance Computation)
        Công thức: score = dot(q_pi, centroids[idx]) + alpha * gamma * dot(q_s, qjl)
        """
        # Biến đổi query trước (Pre-transform)
        q_pi = np.dot(self.tq_mse.Pi, query)
        q_s = np.dot(self.S, query)
        
        # 1. Tính Dot Product trên phần MSE
        mse_score = np.dot(self.tq_mse.centroids[idx], q_pi)
        
        # 2. Tính Dot Product trên phần QJL
        qjl_dot = np.dot(qjl.astype(float), q_s)
        qjl_score = self.alpha * gamma * qjl_dot
        
        return mse_score + qjl_score

    def compute_score_batch(self, query, idx_batch, qjl_batch, gamma_batch, orig_norms=None):
        """ Tính điểm hàng loạt (Batch ADC) """
        q_pi = np.dot(self.tq_mse.Pi, query)
        q_s = np.dot(self.S, query)
        
        # MSE component
        mse_scores = np.dot(self.tq_mse.centroids[idx_batch], q_pi)
        
        # QJL component
        qjl_dot = np.dot(qjl_batch.astype(float), q_s)
        qjl_scores = self.alpha * gamma_batch * qjl_dot
        
        scores = mse_scores + qjl_scores
        
        # Nhân ngược lại với Original Norm để khôi phục Dot Product thực
        if orig_norms is not None:
            scores = scores * orig_norms
            
        return scores

class QuantizationManager:
    def __init__(self, dimension=768):
        self.dimension = dimension
        # Khởi tạo TurboQuantProd với bit-width b=4 (chuẩn phổ biến)
        self.tq_prod = TurboQuantProd(d=dimension, b=4)
        
        # Tự động nạp Centroids nếu đã tồn tại
        centroids_path = "backend/data/centroids.npy"
        if os.path.exists(centroids_path):
            try:
                loaded_centroids = np.load(centroids_path)
                self.tq_prod.tq_mse.centroids = loaded_centroids
                print(f"Loaded existing centroids from {centroids_path}")
            except Exception as e:
                print(f"Error loading centroids: {e}")

    def build_raw(self, embeddings):
        return embeddings.astype('float32')

    def build_pq(self, embeddings, m=32, nbits=8):
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexPQ(self.dimension, m, nbits)
        index.train(embeddings.astype('float32'))
        index.add(embeddings.astype('float32'))
        return index

    def build_sq8(self, embeddings):
        index = faiss.IndexScalarQuantizer(self.dimension, faiss.ScalarQuantizer.QT_8bit)
        index.train(embeddings.astype('float32'))
        index.add(embeddings.astype('float32'))
        return index

    def train_centroids(self, embeddings):
        """ 
        Học Centroids từ tập dữ liệu (1D K-Means) theo nguyên lý Lloyd-Max 
        """
        X = embeddings.astype('float32')
        # 1. Chuẩn hóa L2
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / (norms + 1e-10)
        
        # 2. Xoay vector
        Y = np.dot(X_norm, self.tq_prod.tq_mse.Pi.T)
        
        # 3. Dàn phẳng và chạy 1D K-Means
        y_flat = Y.flatten().reshape(-1, 1)
        
        # Sử dụng FAISS để chạy K-Means 1 chiều cực nhanh
        n_centroids = self.tq_prod.tq_mse.num_centroids
        kmeans = faiss.Kmeans(d=1, k=n_centroids, niter=20, verbose=False)
        kmeans.train(y_flat)
        
        # 4. Cập nhật Centroids (Sắp xếp tăng dần để bám sát phân phối)
        new_centroids = np.sort(kmeans.centroids.flatten())
        self.tq_prod.tq_mse.centroids = new_centroids
        print(f"Lloyd-Max Centroids Trained: {new_centroids}")
        
        # Lưu lại để dùng sau
        os.makedirs("backend/data", exist_ok=True)
        np.save("backend/data/centroids.npy", new_centroids)

    def build_arq(self, embeddings):
        """ 
        TurboQuant Lloyd-Max: Trả về kết quả nén kèm Original Norm
        """
        X = embeddings.astype('float32')
        orig_norms = np.linalg.norm(X, axis=1)
        # Chuẩn hóa trước khi nén
        X_norm = X / (orig_norms[:, np.newaxis] + 1e-10)
        
        idx, qjl, gamma = self.tq_prod.quantize_batch(X_norm)
        
        return {
            "idx": idx,
            "qjl": qjl,
            "gamma": gamma,
            "orig_norm": orig_norms,
            "method": "TurboQuant Lloyd-Max (Optimal MSE)"
        }
