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

    title_map = {
        "top1": "Top-1 Probability", 
        "recall": "Set Recall@K",
        "ndcg": "NDCG@K (Ranking Quality)"
    }
    plt.title(f"{title_map[metric_type]} Comparison ({bit_mode} Mode)", fontsize=14, fontweight='bold')
    plt.xlabel("K (Number of neighbors)", fontsize=12)
    plt.ylabel(f"Accuracy (%)", fontsize=12)
    plt.xticks(k_values)
    plt.ylim(0, 105)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    file_name = f"{metric_type}_{bit_mode}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, file_name), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {file_name}")

def plot_k16_comparison(data, metric_type):
    """
    Vẽ biểu đồ cột (Bar chart) cho K=16 (NDCG@16 hoặc Recall@16) 
    so sánh các mô hình cụ thể.
    """
    target_labels = [
        "TQ-IVF nl4096 np32 2b",
        "TQ-IVF nl4096 np64 2b",
        "TQ-IVF nl8192 np64 2b",
        "TQ-IVF nl8192 np128 2b",
        "FAISS-PQ 2b",
        "TQ-IVF nl4096 np32 4b",
        "TQ-IVF nl4096 np64 4b",
        "TQ-IVF nl8192 np64 4b",
        "TQ-IVF nl8192 np128 4b",
        "FAISS-SQ 4b",
        "FAISS-PQ 4b"
    ]
    
    results = []
    for r in data["results"]:
        if r["label"] in target_labels:
            results.append(r)
            
    # Sắp xếp theo đúng thứ tự của target_labels
    results.sort(key=lambda x: target_labels.index(x["label"]))
    
    labels = []
    values = []
    qps_values = []
    colors = []
    
    for r in results:
        labels.append(r["label"].replace("TQ-IVF ", "TQ ").replace("FAISS-", ""))
        values.append(r[metric_type].get("16", 0.0))
        qps_values.append(r["qps"])
        if "2b" in r["label"]:
            colors.append("lightblue" if "TQ" in r["label"] else "lightcoral")
        else:
            colors.append("dodgerblue" if "TQ" in r["label"] else "red")
            
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Vẽ cột Accuracy trên trục trái
    bars = ax1.bar(labels, values, color=colors, edgecolor='black', alpha=0.8)
    
    # Ghi giá trị phần trăm
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    title_name = "NDCG@16" if metric_type == "ndcg" else "Recall@16"
    ax1.set_ylabel(f"{title_name} (%)", fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(values) + 15)
    ax1.tick_params(axis='x', rotation=30, labelsize=10)
    
    # Vẽ đường QPS trên trục phải
    ax2 = ax1.twinx()
    ax2.plot(labels, qps_values, color='black', marker='D', markersize=8, linewidth=2, linestyle='--', label='QPS (Queries/sec)')
    ax2.set_ylabel('Throughput (QPS)', color='black', fontsize=12, fontweight='bold')
    
    # Ghi chú giá trị QPS
    for i, txt in enumerate(qps_values):
        ax2.annotate(f'{txt:.1f}', (labels[i], qps_values[i]), textcoords="offset points", xytext=(0,10), ha='center', color='black', fontweight='bold', fontsize=9)
        
    # Giới hạn trục QPS để đường không bị đè vào các thanh cột quá nhiều
    ax2.set_ylim(0, max(qps_values) * 1.5)

    plt.title(f"{title_name} and QPS Comparison Across Architectures", fontsize=16, fontweight='bold')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='TQ 2-bit (Acc)'),
        Patch(facecolor='lightcoral', edgecolor='black', label='FAISS 2-bit (Acc)'),
        Patch(facecolor='dodgerblue', edgecolor='black', label='TQ 4-bit (Acc)'),
        Patch(facecolor='red', edgecolor='black', label='FAISS 4-bit (Acc)'),
        Line2D([0], [0], color='black', marker='D', linestyle='--', label='QPS')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    fig.tight_layout()
    file_name = f"{metric_type}_16_bar_with_qps.png"
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
    
    target_labels = [
        "TQ-IVF nl4096 np32 2b",
        "TQ-IVF nl4096 np64 2b",
        "TQ-IVF nl8192 np64 2b",
        "TQ-IVF nl8192 np128 2b",
        "FAISS-PQ 2b",
        "TQ-IVF nl4096 np32 4b",
        "TQ-IVF nl4096 np64 4b",
        "TQ-IVF nl8192 np64 4b",
        "TQ-IVF nl8192 np128 4b",
        "FAISS-SQ 4b",
        "FAISS-PQ 4b"
    ]
    
    results = []
    for r in data["results"]:
        if r["label"] in target_labels:
            results.append(r)
            
    # Sắp xếp theo order
    results.sort(key=lambda x: target_labels.index(x["label"]))
    
    for r in results:
        labels.append(r["label"].replace("TQ-IVF ", "TQ ").replace("FAISS-", ""))
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

    # Giới hạn trục Y trái để chữ không bị cắt
    ax1.set_ylim(0, max(qps_values) * 1.15)
    
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

    # Giới hạn trục Y phải
    ax2.set_ylim(0, max(rss_values) * 1.2)

    plt.title("QPS vs Memory Architecture (The 'Memory Wall' Proof)", fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    
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

def plot_mse(data):
    """
    Vẽ biểu đồ cột so sánh MSE (Độ biến dạng/lượng tử hóa)
    Càng thấp càng tốt.
    """
    labels = []
    mse_values = []
    colors = []
    
    # Lọc các kết quả MSE tiêu biểu
    # MSE của TQ chỉ phụ thuộc vào nlist và bit, không phụ thuộc nprobe, nên lấy nprobe tùy ý
    target_labels = [
        "TQ-IVF nl4096 np64 2b",
        "TQ-IVF nl8192 np128 2b",
        "FAISS-PQ 2b",
        "TQ-IVF nl4096 np64 4b",
        "TQ-IVF nl8192 np128 4b",
        "FAISS-SQ 4b",
        "FAISS-PQ 4b"
    ]
    
    targets = []
    for r in data["results"]:
        if r["label"] in target_labels and r.get("mse", 0) > 0:
            targets.append(r)
            
    # Sắp xếp theo order trong target_labels
    targets.sort(key=lambda x: target_labels.index(x["label"]))
    
    for r in targets:
        mse = r.get("mse", 0.0)
        label = r["label"]
        
        # Đổi tên cho gọn
        if "TQ-IVF nl4096" in label:
            short_label = f"TQ nl4096 {'2b' if '2b' in label else '4b'}"
        elif "TQ-IVF nl8192" in label:
            short_label = f"TQ nl8192 {'2b' if '2b' in label else '4b'}"
        else:
            short_label = label.replace("FAISS-", "")
            
        if "TQ" in label:
            colors.append("skyblue" if "2b" in label else "dodgerblue")
        else:
            colors.append("salmon" if "2b" in label else "red")
            
        labels.append(short_label)
        mse_values.append(mse)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, mse_values, color=colors, alpha=0.8, edgecolor='black')
    
    # Ghi giá trị lên đầu cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.yscale('log') # Dùng log scale vì 2b và 4b chênh lệch rất lớn
    plt.title("Quantization Distortion (MSE) - Lower is Better", fontsize=16, fontweight='bold')
    plt.ylabel("Mean Squared Error (Log Scale)", fontsize=12)
    plt.xticks(rotation=20, ha='right', fontsize=11)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "mse_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved MSE comparison chart.")

if __name__ == "__main__":
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found.")
    else:
        results_data = load_data()
        
        # Vẽ biểu đồ Accuracy cho cả 2b và 4b
        for mode in ["2b", "4b"]:
            plot_accuracy(results_data, mode, "top1")
            plot_accuracy(results_data, mode, "recall")
            plot_accuracy(results_data, mode, "ndcg")
            
        # Vẽ biểu đồ Bar Chart cho riêng K=16
        plot_k16_comparison(results_data, "ndcg")
        plot_k16_comparison(results_data, "recall")
        
        # Vẽ biểu đồ Distortion (Chỉ số mới)
        plot_mse(results_data)
        
        # Vẽ biểu đồ Efficiency
        plot_efficiency(results_data)
        
        print(f"\nSuccess! All charts are saved in: {OUTPUT_DIR}")
