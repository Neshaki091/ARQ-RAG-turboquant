import numpy as np

class TurboQuantMSE:
    """
    Giai đoạn 1: TurboQuant tối ưu hóa Lỗi bình phương trung bình (MSE)
    Dựa trên Algorithm 1: TurboQuant_mse
    """
    def __init__(self, d, b):
        self.d = d
        self.b = b
        self.num_centroids = 2 ** b
        
        # 1. Tạo ma trận xoay ngẫu nhiên Pi (d x d)
        # Sử dụng phân rã QR để đảm bảo Pi là ma trận trực giao (Orthogonal/Rotation Matrix)
        H = np.random.randn(d, d)
        Q, R_mat = np.linalg.qr(H)
        self.Pi = Q 
        
        # 2. Tìm các tâm cụm (centroids) tối ưu
        # Trong thực tế, các tâm cụm này được giải sẵn bằng bài toán k-means 1 chiều liên tục 
        # trên phân phối Beta cho từng bit-width và được lưu cứng (hardcoded).
        # Ở đây ta dùng hàm mô phỏng lấy các giá trị phân bố đều trong [-1, 1].
        self.centroids = self._get_optimal_centroids()

    def _get_optimal_centroids(self):
        # Mô phỏng tập codebook c_1, c_2, ..., c_{2^b}
        return np.linspace(-1, 1, self.num_centroids)

    def quantize(self, x):
        """ Hàm Nén (Quantization) """
        # Phép xoay ngẫu nhiên: y = Pi * x
        y = np.dot(self.Pi, x)
        
        # Tìm index của centroid gần nhất cho từng tọa độ: idx_j = argmin |y_j - c_k|
        # Dùng broadcasting để tính khoảng cách
        diffs = np.abs(y[:, np.newaxis] - self.centroids)
        idx = np.argmin(diffs, axis=1)
        
        return idx # Trả về mảng các số nguyên (cần b bits để lưu trữ mỗi số)

    def dequantize(self, idx):
        """ Hàm Giải nén (Dequantization) """
        # Khôi phục tọa độ: y_tilde_j = c_{idx_j}
        y_tilde = self.centroids[idx]
        
        # Xoay ngược lại để lấy vector giải nén: x_tilde = Pi^T * y_tilde
        x_tilde = np.dot(self.Pi.T, y_tilde)
        
        return x_tilde


class TurboQuantProd:
    """
    Giai đoạn 2: TurboQuant tối ưu hóa Tích vô hướng (Inner Product)
    Dựa trên Algorithm 2: TurboQuant_prod kết hợp QJL
    """
    def __init__(self, d, b):
        self.d = d
        self.b = b
        
        # 1. Khởi tạo TurboQuant_mse với độ rộng bit là (b - 1)
        self.tq_mse = TurboQuantMSE(d, b - 1)
        
        # 2. Khởi tạo ma trận ngẫu nhiên S cho thuật toán QJL
        # S có phân phối chuẩn N(0, 1) kích thước d x d
        self.S = np.random.randn(d, d)

    def quantize(self, x):
        """ Hàm Nén (Quantization) """
        # 1. Nén bằng TurboQuant_mse
        idx = self.tq_mse.quantize(x)
        x_tilde_mse = self.tq_mse.dequantize(idx)
        
        # 2. Tính vector phần dư (residual error) và chuẩn L2 (gamma)
        r = x - x_tilde_mse
        gamma = np.linalg.norm(r, ord=2)
        
        # 3. Áp dụng biến đổi QJL 1-bit lên phần dư: qjl = sign(S * r)
        transformed_r = np.dot(self.S, r)
        qjl = np.sign(transformed_r)
        qjl[qjl == 0] = 1  # Xử lý trường hợp hiếm khi giá trị = 0
        
        # Mã nén hoàn chỉnh gồm: Chỉ số MSE, Vector nhị phân QJL, và Chuẩn phần dư
        return idx, qjl, gamma

    def dequantize(self, idx, qjl, gamma):
        """ Hàm Giải nén (Dequantization) """
        # 1. Giải nén phần MSE
        x_tilde_mse = self.tq_mse.dequantize(idx)
        
        # 2. Giải nén phần QJL
        # Công thức: x_tilde_qjl = (sqrt(pi/2) / d) * gamma * S^T * qjl
        scaling_factor = np.sqrt(np.pi / 2.0) / self.d
        x_tilde_qjl = scaling_factor * gamma * np.dot(self.S.T, qjl)
        
        # 3. Cộng gộp để ra vector giải nén toàn bộ
        x_tilde = x_tilde_mse + x_tilde_qjl
        
        return x_tilde

# === VÍ DỤ SỬ DỤNG ===
if __name__ == "__main__":
    d = 128     # Số chiều của vector
    b = 4       # Tổng số bit lượng tử hóa mong muốn (b bits)
    
    # Tạo 1 vector ngẫu nhiên x trên siêu cầu đơn vị (unit hypersphere)
    x = np.random.randn(d)
    x = x / np.linalg.norm(x)
    
    # Khởi tạo mô hình
    tq_prod = TurboQuantProd(d=d, b=b)
    
    # Thực hiện Nén (Quantize)
    idx, qjl, gamma = tq_prod.quantize(x)
    
    # Thực hiện Giải nén (Dequantize)
    x_reconstructed = tq_prod.dequantize(idx, qjl, gamma)
    
    print(f"Vector gốc (5 phần tử đầu): {x[:5]}")
    print(f"Vector giải nén (5 phần tử đầu): {x_reconstructed[:5]}")
    print(f"Bảo toàn Tích vô hướng (Tự nhân với chính nó): {np.dot(x, x_reconstructed):.4f}")
    print(f"Sai số L2 (Norm Error): {np.linalg.norm(x - x_reconstructed):.4f}")