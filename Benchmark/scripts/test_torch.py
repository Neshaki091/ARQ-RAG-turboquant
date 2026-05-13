import time
print("Bắt đầu nạp Torch...")
start = time.time()
import torch
print(f"Nạp xong Torch trong {time.time() - start:.2f} giây")
print(f"Phiên bản Torch: {torch.__version__}")
