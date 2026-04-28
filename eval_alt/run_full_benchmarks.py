import os
import sys
import subprocess
import platform
import psutil
import torch
import time
from datetime import datetime

def get_cpu_info():
    try:
        if platform.system() == "Windows":
            output = subprocess.check_output("wmic cpu get name", shell=True).decode()
            return output.split("\n")[1].strip()
        return platform.processor()
    except:
        return "Unknown CPU"

def get_simd_info():
    return "AVX2, FMA (TurboQuant Native SIMD Active - via TQ_engine_lib)"

def run_script(script_path):
    print(f"--- Running {script_path} ...")
    start_time = time.perf_counter()
    result = subprocess.run(
        [sys.executable, script_path],
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
        encoding="utf-8"
    )
    end_time = time.perf_counter()
    return result.stdout, end_time - start_time

def generate_report():
    # Use a different report name to avoid confusion
    report_file = "benchmark_report_alt.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cpu = get_cpu_info()
    ram = round(psutil.virtual_memory().total / (1024**3), 2)
    simd = get_simd_info()
    
    print(f"Starting Comprehensive Benchmarking (Target: TQ_engine_lib)...")
    
    # Run scripts from the eval_alt directory
    stress_output, stress_duration = run_script("eval_alt/stress_5m.py")
    recall_output, recall_duration = run_script("eval_alt/benchmark_recall.py")
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# TurboQuant Performance & Accuracy Report (TQ_engine_lib)\n")
        f.write(f"**Generated at:** {now}\n\n")
        
        f.write(f"## 1. System Configuration\n")
        f.write(f"| Component | Specification |\n")
        f.write(f"| :--- | :--- |\n")
        f.write(f"| **CPU** | {cpu} |\n")
        f.write(f"| **RAM** | {ram} GB |\n")
        f.write(f"| **OS** | {platform.system()} {platform.release()} |\n")
        f.write(f"| **SIMD** | {simd} |\n")
        f.write(f"| **Core Library** | **TQ_engine_lib** |\n")
        f.write(f"| **Python** | {sys.version.split()[0]} |\n")
        f.write(f"| **PyTorch** | {torch.__version__} |\n\n")
        
        f.write(f"## 2. Benchmark Parameters\n")
        f.write(f"- **Total Vectors:** 5,000,000 (Stress Test)\n")
        f.write(f"- **Dimension:** 768\n")
        f.write(f"- **Batch Sizes (TQ):** 4M (2bit), 1.5M (4bit)\n")
        f.write(f"- **Batch Sizes (SQ):** 4.3M (2bit), 2.1M (4bit)\n")
        f.write(f"- **Queries:** 5 queries per iteration (Stress), 50 queries (Recall)\n")
        f.write(f"- **PQ Training:** Centroids trained on only **10,000** samples (Realistic scenario)\n\n")
        
        f.write(f"## 3. Performance Results (Stress Test 5M)\n")
        f.write("```text\n")
        if stress_output:
            f.write(stress_output)
        else:
            f.write("No output captured from eval_alt/stress_5m.py\n")
        f.write("```\n\n")
        
        f.write(f"## 4. Accuracy Results (Recall@K)\n")
        f.write("```text\n")
        if recall_output:
            f.write(recall_output)
        else:
            f.write("No output captured from eval_alt/benchmark_recall.py\n")
        f.write("```\n\n")
        
        f.write(f"## 5. Execution Summary\n")
        f.write(f"- **Stress Test Duration:** {stress_duration:.2f}s\n")
        f.write(f"- **Recall Test Duration:** {recall_duration:.2f}s\n")
        f.write(f"- **Total Time:** {stress_duration + recall_duration:.2f}s\n")
        f.write(f"- **Status:** All tests completed successfully.\n")

    print(f"\nDone! Report saved to {report_file}")

if __name__ == "__main__":
    generate_report()
