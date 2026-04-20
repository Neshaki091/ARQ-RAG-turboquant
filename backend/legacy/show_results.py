from shared.supabase_client import SupabaseManager
from collections import defaultdict

sm = SupabaseManager()
res = sm.supabase.table('benchmarks').select('model_name,latency_ms,retrieval_latency_ms,peak_ram_mb,total_ram_mb').order('created_at', desc=True).limit(60).execute()

data = defaultdict(lambda: {'latency': [], 'retrieval': [], 'peak_ram': [], 'total_ram': []})
for r in res.data:
    m = r['model_name']
    data[m]['latency'].append(r['latency_ms'] or 0)
    data[m]['retrieval'].append(r['retrieval_latency_ms'] or 0)
    data[m]['peak_ram'].append(r['peak_ram_mb'] or 0)
    data[m]['total_ram'].append(r['total_ram_mb'] or 0)

print('Model            | Count | Retrieval(ms) | Latency(ms) | PeakRAM(MB) | TotalRAM(MB)')
print('-' * 95)
order = ['vector_raw', 'vector_pq', 'vector_sq8', 'vector_adaptive', 'vector_arq']
for m in order:
    d = data[m]
    if not d['latency']:
        continue
    n = len(d['latency'])
    avg_ret = sum(d['retrieval']) / n
    avg_lat = sum(d['latency']) / n
    avg_pr = sum(d['peak_ram']) / n
    avg_tr = sum(d['total_ram']) / n
    print(f'{m:<16} | {n:>5} | {avg_ret:>13.2f} | {avg_lat:>11.2f} | {avg_pr:>11.2f} | {avg_tr:>12.2f}')
