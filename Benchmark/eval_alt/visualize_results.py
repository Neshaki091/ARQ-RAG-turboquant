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
    
    # Lấy toàn bộ các kết quả khớp với bit_mode
    results = [r for r in data["results"] if r["label"].endswith(bit_mode)]
    
    import re
    filtered_results = []
    for res in results:
        label = res["label"]
        if "FAISS" in label:
            filtered_results.append(res)
        elif "TQ-IVF" in label and "nl4096" in label:
            match = re.search(r'np(\d+)', label)
            if match:
                np_val = int(match.group(1))
                if 16 <= np_val <= 64:
                    filtered_results.append(res)
    
    for res in filtered_results:
        label = res["label"]
        # Đổi tên label cho TQ để hiển thị rõ ràng hơn
        if "TQ-IVF" in label:
            match = re.search(r'np(\d+)', label)
            if match:
                label = f"TQ 4096 nprobe {match.group(1)}"
            
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
    
    # 1. Lấy các mẫu FAISS 4-bit
    faiss_results = [r for r in data["results"] if "FAISS" in r["label"] and "4b" in r["label"]]
    for r in faiss_results:
        labels.append(r["label"])
        qps_values.append(r["qps"])
        priv_values.append(r.get("priv_mb", 0.0))
        rss_values.append(r.get("rss_mb", 0.0))

    # 2. Lấy các mẫu TQ 4-bit với nlist=4096 và nprobe trong khoảng [16, 64]
    import re
    tq_samples = []
    for r in data["results"]:
        label = r["label"]
        if "TQ-IVF" in label and "nl4096" in label and "4b" in label:
            match = re.search(r'np(\d+)', label)
            if match:
                np_val = int(match.group(1))
                if 16 <= np_val <= 64:
                    tq_samples.append(r)
    
    # Sắp xếp theo nprobe để biểu đồ đẹp hơn
    tq_samples.sort(key=lambda x: int(re.search(r'np(\d+)', x["label"]).group(1)))
    
    for r in tq_samples:
        # Format lại label: TQ 4096 nprobe X
        match = re.search(r'np(\d+)', r["label"])
        clean_label = f"TQ 4096 nprobe {match.group(1)}" if match else r["label"]
        labels.append(clean_label)
        qps_values.append(r["qps"])
        priv_values.append(r.get("priv_mb", 0.0))
        rss_values.append(r.get("rss_mb", 0.0))

    # Vẽ biểu đồ Dual Axis cho QPS và RAM (Private vs Working Set)
    fig, ax1 = plt.subplots(figsize=(14, 8))
    color_qps = 'tab:blue'
    ax1.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Throughput (QPS)', color=color_qps, fontsize=12, fontweight='bold')
    bars = ax1.bar(labels, qps_values, color=color_qps, alpha=0.4, width=0.5, label='QPS')
    ax1.tick_params(axis='y', labelcolor=color_qps)
    
    # Ghi số QPS lên cột
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5, f'{height:.1f}', ha='center', va='bottom', color=color_qps, fontweight='bold')

    ax2 = ax1.twinx()
    
    # Đường 1: Private RAM (RAM thực sự cần thiết)
    ax2.plot(labels, priv_values, color='tab:green', marker='D', markersize=10, linewidth=3, label='Private RAM (Mandatory)')
    
    # Đường 2: Working Set (RAM bao gồm cả Cache)
    ax2.plot(labels, rss_values, color='tab:orange', marker='s', markersize=8, linewidth=2, linestyle='--', label='Working Set (Inc. Page Cache)')
    
    ax2.set_ylabel('Memory Usage (MB)', color='black', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Ghi chú cho Private RAM
    for i, txt in enumerate(priv_values):
        ax2.annotate(f'{txt:.1f}', (labels[i], priv_values[i]), textcoords="offset points", xytext=(-15,10), ha='center', color='tab:green', fontweight='bold')

    # Ghi chú cho Working Set
    for i, txt in enumerate(rss_values):
        ax2.annotate(f'{txt:.0f}', (labels[i], rss_values[i]), textcoords="offset points", xytext=(15,-15), ha='center', color='tab:orange', fontweight='bold')

    plt.title("QPS vs Memory Architecture (The 'Memory Wall' Proof)", fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=15)
    
    # Thêm Legend cho cả 2 trục
    lines, labels_leg = ax1.get_legend_handles_labels()
    lines2, labels2_leg = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels_leg + labels2_leg, loc='upper left')

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
        
        # Vẽ biểu đồ Accuracy (Chỉ 4-bit theo yêu cầu)
        plot_accuracy(results_data, "4b", "top1")
        plot_accuracy(results_data, "4b", "recall")
        
        # Vẽ biểu đồ Efficiency
        plot_efficiency(results_data)
        
        print(f"\nSuccess! All charts are saved in: {OUTPUT_DIR}")
