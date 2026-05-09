import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Thiết lập style cho biểu đồ
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)

# Đường dẫn file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BENCHMARK_RESULT_DIR = os.path.join(PROJECT_ROOT, "benchmark_result")
JSON_PATH = os.path.join(BENCHMARK_RESULT_DIR, "benchmark_results.json")
OUTPUT_DIR = os.path.join(BENCHMARK_RESULT_DIR, "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_accuracy(data, bit_mode, metric_type):
    """
    Vẽ biểu đồ đường cho Top-1 hoặc Recall (2-bit hoặc 4-bit)
    metric_type: 'top1' hoặc 'recall'
    bit_mode: '2b' hoặc '4b'
    """
    k_values = data["k_values"]
    plt.figure(figsize=(10, 6))
    
    # Lọc các kết quả theo bit_mode
    # Ưu tiên lấy nlist=4096 để biểu đồ không bị rối, hoặc lấy đại diện
    results = [r for r in data["results"] if r["label"].endswith(bit_mode)]
    
    # Chỉ lấy FAISS và một vài mẫu nprobe của TQ để biểu đồ rõ ràng
    targets = ["FAISS", "np2", "np16", "np64"]
    
    for res in results:
        label = res["label"]
        # Kiểm tra xem có phải target không
        if not any(t in label for t in targets):
            continue
            
        y_values = [res[metric_type][str(k)] for k in k_values]
        
        # Tạo style cho đường kẻ
        linestyle = '-'
        marker = 'o'
        if "FAISS" in label:
            linestyle = '--'
            marker = 's'
            color = 'red'
        else:
            color = None # Để seaborn tự chọn
            
        plt.plot(k_values, y_values, label=label, marker=marker, linestyle=linestyle, linewidth=2)

    title_map = {"top1": "Top-1 Probability", "recall": "Set Recall@K"}
    plt.title(f"{title_map[metric_type]} Comparison ({bit_mode} Mode)", fontsize=14, fontweight='bold')
    plt.xlabel("K (Number of neighbors)", fontsize=12)
    plt.ylabel(f"{title_map[metric_type]} (%)", fontsize=12)
    plt.xticks(k_values)
    plt.ylim(0, 105)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    file_name = f"{metric_type}_{bit_mode}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, file_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {file_name}")

def plot_efficiency(data):
    """
    Vẽ biểu đồ so sánh QPS và Private RAM (Bar chart + Line)
    """
    labels = []
    qps_values = []
    priv_values = []
    rss_values = []
    
    # 1. Lấy tất cả các mẫu FAISS có trong JSON
    faiss_results = [r for r in data["results"] if "FAISS" in r["label"]]
    for r in faiss_results:
        labels.append(r["label"])
        qps_values.append(r["qps"])
        priv_values.append(r.get("priv_mb", 0.0))
        rss_values.append(r.get("rss_mb", 0.0))

    # 2. Lấy các mẫu TQ tiêu biểu (np=64)
    tq_samples = [r for r in data["results"] if "np64" in r["label"] and "nl4096" in r["label"]]
    for r in tq_samples:
        labels.append(r["label"].replace("nl4096 ", ""))
        qps_values.append(r["qps"])
        priv_values.append(r.get("priv_mb", 0.0))
        rss_values.append(r.get("rss_mb", 0.0))

    # Vẽ biểu đồ Dual Axis cho QPS và Private RAM
    fig, ax1 = plt.subplots(figsize=(14, 8))
    color_qps = 'tab:blue'
    ax1.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Throughput (QPS)', color=color_qps, fontsize=12, fontweight='bold')
    bars = ax1.bar(labels, qps_values, color=color_qps, alpha=0.6, width=0.5, label='QPS')
    ax1.tick_params(axis='y', labelcolor=color_qps)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5, f'{height:.1f}', ha='center', va='bottom', color=color_qps, fontweight='bold')

    ax2 = ax1.twinx()
    color_priv = 'tab:green'
    ax2.set_ylabel('Mandatory Private RAM (MB)', color=color_priv, fontsize=12, fontweight='bold')
    ax2.plot(labels, priv_values, color=color_priv, marker='D', markersize=10, linewidth=3, label='Private RAM')
    ax2.tick_params(axis='y', labelcolor=color_priv)
    for i, txt in enumerate(priv_values):
        ax2.annotate(f'{txt:.1f} MB', (labels[i], priv_values[i]), textcoords="offset points", xytext=(0,10), ha='center', color=color_priv, fontweight='bold')

    plt.title("QPS vs Private Memory Usage (The 'Memory Wall' Proof)", fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=15)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "efficiency_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Vẽ biểu đồ phụ so sánh Private vs Working Set (RSS)
    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, priv_values, width, label='Private (Mandatory)', color='tab:green', alpha=0.8)
    plt.bar(x + width/2, rss_values, width, label='Working Set (Inc. Cache)', color='tab:orange', alpha=0.8)
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Architecture: Private Allocation vs. OS Page Cache')
    plt.xticks(x, labels, rotation=15)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "ram_architecture_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved efficiency and architecture charts.")

if __name__ == "__main__":
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found.")
    else:
        results_data = load_data()
        
        # Vẽ 4 biểu đồ Accuracy
        plot_accuracy(results_data, "2b", "top1")
        plot_accuracy(results_data, "4b", "top1")
        plot_accuracy(results_data, "2b", "recall")
        plot_accuracy(results_data, "4b", "recall")
        
        # Vẽ biểu đồ Efficiency
        plot_efficiency(results_data)
        
        print(f"\nSuccess! All 5 charts are saved in: {OUTPUT_DIR}")
