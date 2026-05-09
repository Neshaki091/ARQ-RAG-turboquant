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
DATA_DIR = os.path.join("Benchmark", "data", "wiki_benchmark")
JSON_PATH = os.path.join(DATA_DIR, "benchmark_results.json")
OUTPUT_DIR = os.path.join(DATA_DIR, "charts")
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
    Vẽ biểu đồ so sánh QPS và RAM (Bar chart)
    """
    # Chọn ra các mẫu tiêu biểu để so sánh
    # TQ-IVF 2b np64, TQ-IVF 4b np64, FAISS-PQ 2b
    labels = []
    qps_values = []
    ram_values = []
    
    # Lấy FAISS PQ 2b (mẫu duy nhất FAISS chạy được)
    faiss_pq = [r for r in data["results"] if "FAISS-PQ 2b" in r["label"]]
    if faiss_pq:
        labels.append("FAISS-PQ 2b")
        qps_values.append(faiss_pq[0]["qps"])
        ram_values.append(faiss_pq[0]["ram_mb"])
        
    # Lấy TQ 2b np64
    tq_2b = [r for r in data["results"] if "TQ-IVF nl4096 np64 2b" in r["label"]]
    if tq_2b:
        labels.append("TQ-IVF 2b")
        qps_values.append(tq_2b[0]["qps"])
        ram_values.append(tq_2b[0]["ram_mb"])

    # Lấy TQ 4b np64
    tq_4b = [r for r in data["results"] if "TQ-IVF nl4096 np64 4b" in r["label"]]
    if tq_4b:
        labels.append("TQ-IVF 4b")
        qps_values.append(tq_4b[0]["qps"])
        ram_values.append(tq_4b[0]["ram_mb"])

    # Vẽ biểu đồ Dual Axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_qps = 'tab:blue'
    ax1.set_xlabel('Model Configuration')
    ax1.set_ylabel('Throughput (QPS)', color=color_qps, fontsize=12, fontweight='bold')
    bars = ax1.bar(labels, qps_values, color=color_qps, alpha=0.6, width=0.4, label='QPS')
    ax1.tick_params(axis='y', labelcolor=color_qps)
    
    # Thêm số liệu trên đầu cột QPS
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5, f'{height:.1f}', ha='center', va='bottom', color=color_qps, fontweight='bold')

    ax2 = ax1.twinx()
    color_ram = 'tab:red'
    ax2.set_ylabel('Peak RAM (MB)', color=color_ram, fontsize=12, fontweight='bold')
    ax2.plot(labels, ram_values, color=color_ram, marker='D', markersize=10, linewidth=3, label='RAM')
    ax2.tick_params(axis='y', labelcolor=color_ram)
    
    # Thêm số liệu cho RAM
    for i, txt in enumerate(ram_values):
        ax2.annotate(f'{txt:.1f} MB', (labels[i], ram_values[i]), textcoords="offset points", xytext=(0,10), ha='center', color=color_ram, fontweight='bold')

    plt.title("QPS vs Memory Usage Comparison (5M Vectors)", fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    file_name = "efficiency_comparison.png"
    plt.savefig(os.path.join(OUTPUT_DIR, file_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {file_name}")

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
