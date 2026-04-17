import numpy as np
import faiss

class ManualPQ:
    def __init__(self, d, m, nbits=8):
        self.d = d
        self.m = m
        self.nbits = nbits
        self.k = 2 ** nbits
        self.ds = d // m
        self.centroids = []

    def train(self, X):
        X = X.astype('float32')
        n_data = X.shape[0]
        
        # Điều chỉnh k linh hoạt: không được vượt quá số lượng điểm dữ liệu
        k_actual = min(self.k, n_data)
        if k_actual < 1: return # Không có dữ liệu để huấn luyện
        
        self.centroids = []
        for i in range(self.m):
            sub_X = X[:, i*self.ds : (i+1)*self.ds]
            # Faiss Kmeans cần ít nhất k điểm
            if n_data >= k_actual and k_actual >= 2:
                kmeans = faiss.Kmeans(d=self.ds, k=k_actual, niter=20, verbose=False)
                kmeans.train(sub_X)
                self.centroids.append(kmeans.centroids)
            else:
                # Fallback: Sử dụng chính các điểm dữ liệu làm centroids nếu quá ít
                padding = np.zeros((max(0, k_actual - n_data), self.ds), dtype='float32')
                fallback_centroids = np.vstack([sub_X, padding]) if n_data > 0 else np.zeros((k_actual, self.ds), dtype='float32')
                self.centroids.append(fallback_centroids)

    def quantize(self, X):
        N = X.shape[0]
        codes = np.zeros((N, self.m), dtype=np.uint8)
        for i in range(self.m):
            sub_X = X[:, i*self.ds : (i+1)*self.ds]
            diffs = np.linalg.norm(sub_X[:, np.newaxis, :] - self.centroids[i][np.newaxis, :, :], axis=2)
            codes[:, i] = np.argmin(diffs, axis=1)
        return codes

    def compute_adc_scores(self, query, codes):
        q_tilde = query.astype('float32')
        total_scores = np.zeros(codes.shape[0])
        for i in range(self.m):
            sub_q = q_tilde[i*self.ds : (i+1)*self.ds]
            dist_table = np.linalg.norm(self.centroids[i] - sub_q, axis=1)
            total_scores += dist_table[codes[:, i]]
        return -total_scores
