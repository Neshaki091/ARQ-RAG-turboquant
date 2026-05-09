import json
import csv
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert benchmark_results.json to CSV for Excel.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.join(script_dir, "..", "data", "wiki_benchmark", "benchmark_results.json")
    default_out = os.path.join(script_dir, "..", "data", "wiki_benchmark", "benchmark_results.csv")
    
    parser.add_argument("--input", default=default_in, help="Path to input JSON file")
    parser.add_argument("--output", default=default_out, help="Path to output CSV file")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        print(f"Error: Could not find input file {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    k_values = data.get("k_values", [])
    results = data.get("results", [])

    if not results:
        print("No results found in JSON.")
        return

    # Chuẩn bị header cho file CSV
    headers = ["Model", "RAM (MB)", "QPS"]
    
    # Thêm cột cho Top1@K
    for k in k_values:
        headers.append(f"Top1@{k}")
        
    # Thêm cột cho Set_Recall@K
    for k in k_values:
        headers.append(f"Set_Recall@{k}")

    print(f"Writing data to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        # Dùng tab delimiter để copy-paste trực tiếp vào Excel dễ dàng, hoặc lưu thành .csv dùng dấu phẩy
        writer = csv.writer(f, delimiter=',')
        writer.writerow(headers)

        for res in results:
            row = [
                res.get("label", "N/A"),
                f"{res.get('ram_mb', 0):.2f}",
                f"{res.get('qps', 0):.2f}"
            ]

            # Điền giá trị Top1
            top1_data = res.get("top1", {})
            for k in k_values:
                val = top1_data.get(str(k), 0.0)
                row.append(f"{val:.2f}")

            # Điền giá trị Recall
            recall_data = res.get("recall", {})
            for k in k_values:
                val = recall_data.get(str(k), 0.0)
                row.append(f"{val:.2f}")

            writer.writerow(row)

    print(f"Done! Successfully created CSV file at:")
    print(output_path)
    print("Bạn có thể mở file .csv này trực tiếp bằng Excel, hoặc Data -> From Text/CSV trong Excel.")

if __name__ == "__main__":
    main()
