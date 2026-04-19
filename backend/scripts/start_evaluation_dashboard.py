import os
from trulens_eval import Tru
import logging

# Tắt log thừa
logging.getLogger("trulens_eval").setLevel(logging.WARNING)

def main():
    print("🚀 Đang khởi động TruLens Dashboard...")
    print("📊 Bảng điều khiển này sẽ giúp bạn theo dõi RAG Triad (Groundedness, Relevance).")
    
    tru = Tru()
    
    # Khởi động dashboard trên port 8501 (mặc định)
    # Nếu chạy trong Docker, cần export cổng này ra ngoài.
    tru.run_dashboard()
    
    print("\n✅ Dashboard đã sẵn sàng tại địa chỉ: http://localhost:8501")
    print("Nhấn Ctrl+C để dừng Dashboard.")

if __name__ == "__main__":
    main()
