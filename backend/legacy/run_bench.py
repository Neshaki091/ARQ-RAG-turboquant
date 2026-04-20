import httpx
import time

models = ['vector_raw', 'vector_pq', 'vector_sq8', 'vector_adaptive', 'vector_arq']
base = 'http://localhost:8000'

for m in models:
    print(f'==> Bat dau: {m}', flush=True)
    r = httpx.post(f'{base}/api/benchmark/run-test', json={'batch_size': 10, 'model': m}, timeout=10)
    print(f'    Trigger: {r.status_code} {r.text}', flush=True)

    # Cho den khi benchmark_running = False
    for _ in range(200):
        time.sleep(5)
        try:
            s = httpx.get(f'{base}/status', timeout=15).json()
        except Exception:
            print(f'    [{m}] status timeout, thu lai...', flush=True)
            continue
        prog = s.get('progress', 0)
        status = s.get('status', '?')
        running = s.get('benchmark_running', True)
        print(f'    [{m}] status={status} | progress={prog}% | running={running}', flush=True)
        if not running:
            break

    print(f'==> Xong: {m}', flush=True)
    time.sleep(3)

print('=== ALL DONE ===', flush=True)
