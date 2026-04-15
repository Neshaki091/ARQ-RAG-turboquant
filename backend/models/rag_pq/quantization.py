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
        for i in range(self.m):
            sub_X = X[:, i*self.ds : (i+1)*self.ds]
            kmeans = faiss.Kmeans(d=self.ds, k=self.k, niter=20, verbose=False)
            kmeans.train(sub_X)
            self.centroids.append(kmeans.centroids)

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
