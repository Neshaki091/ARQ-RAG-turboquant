import pandas as pd
import os

def export_to_excel(results, output_file="benchmark_results.xlsx"):
    """
    Xuất kết quả sang Excel với 3 Sheet: Query_Level, TestSet_Level, Summary.
    `results` là danh sách các dict chứa dữ liệu thô.
    """
    df_raw = pd.DataFrame(results)
    
    # Sheet 1: Query_Level
    # Cấu trúc: Model | TestSet | QueryID | Latency | RAM | Faithfulness | Answer Relevance | Context Precision | Context Recall
    query_level = df_raw.copy()
    
    # Sheet 2: TestSet_Level
    # Cấu trúc: Model | TestSet | Peak RAM | Max Latency | Avg Faithfulness | Avg Precision | Avg Recall
    testset_level = df_raw.groupby(['Model', 'TestSet']).agg({
        'RAM': 'max',
        'Latency': 'max',
        'Faithfulness': 'mean',
        'Context Precision': 'mean',
        'Context Recall': 'mean'
    }).reset_index()
    testset_level.columns = ['Model', 'TestSet', 'Peak RAM', 'Max Latency', 'Avg Faithfulness', 'Avg Precision', 'Avg Recall']
    
    # Sheet 3: Summary
    # Cấu trúc: Model | Avg RAM | Avg Latency | Avg Faithfulness | Avg Precision
    summary = df_raw.groupby('Model').agg({
        'RAM': 'mean',
        'Latency': 'mean',
        'Faithfulness': 'mean',
        'Context Precision': 'mean'
    }).reset_index()
    summary.columns = ['Model', 'Avg RAM', 'Avg Latency', 'Avg Faithfulness', 'Avg Precision']
    
    # Xuất ra file
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        query_level.to_excel(writer, sheet_name='Query_Level', index=False)
        testset_level.to_excel(writer, sheet_name='TestSet_Level', index=False)
        summary.to_excel(writer, sheet_name='Summary', index=False)
        
    print(f"Đã xuất báo cáo: {output_file}")
    return output_file
