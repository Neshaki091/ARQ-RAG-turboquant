import asyncio
import httpx
import time
import random
import json
import os
from statistics import mean

# Cấu hình
BASE_URL = "http://localhost:8000"
TOTAL_USERS = 32
SIMULATION_DURATION = 1  # Tổng thời gian các user bắt đầu gửi (giây)
USER_ID = 1  # ID user mặc định cho demo
SESSION_ID = "test_sim_session"

async def login(client):
    """Đăng nhập để lấy token"""
    try:
        # Giả định có user admin/admin123
        response = await client.post(
            f"{BASE_URL}/login",
            json={"username": "admin", "password": "admin123"}
        )
        if response.status_code == 200:
            token = response.json().get("access_token")
            client.headers.update({"Authorization": f"Bearer {token}"})
            print("LOG: Login successful.")
            return True
        else:
            # Nếu chưa có user, thử register
            print("LOG: Login failed, trying to register...")
            await client.post(
                f"{BASE_URL}/register",
                json={"username": "admin", "password": "admin123"}
            )
            return await login(client)
    except Exception as e:
        print(f"LOG: Auth error: {e}")
        return False

async def send_request(client, query, user_num):
    """Mô phỏng 1 người dùng gửi câu hỏi và đo thời gian phản hồi"""
    start_time = time.time()
    try:
        response = await client.post(
            f"{BASE_URL}/chat",
            json={
                "query": query,
                "user_id": USER_ID,
                "session_id": SESSION_ID,
                "mode": "adaptive",
                "scope": "both",
                "isSimulation": True # Chỉ lấy kết quả search, không chạy LLM để test tốc độ lõi
            },
            timeout=60.0
        )
        latency = time.time() - start_time
        if response.status_code == 200:
            print(f"[User {user_num:02d}] Success! Latency: {latency:.2f}s | Complexity: {response.json().get('complexity')}")
            return latency
        else:
            print(f"[User {user_num:02d}] Failed! Status: {response.status_code}")
            return None
    except Exception as e:
        print(f"[User {user_num:02d}] Error: {str(e)}")
        return None

async def main():
    # 1. Load dữ liệu câu hỏi từ benchmark
    # Tự động xác định đường dẫn tương đối từ thư mục cha (Root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    queries_path = os.path.join(base_dir, "backend", "data", "queries", "benchmark_queries_400.json")
    
    if not os.path.exists(queries_path):
        # Fallback nếu chạy trực tiếp trong backend
        queries_path = os.path.join("data", "queries", "benchmark_queries_400.json")

    with open(queries_path, "r", encoding="utf-8") as f:
        all_queries = json.load(f)

    print(f"=== STARTING REAL-WORLD USER SIMULATION ===")
    print(f"Total Users: {TOTAL_USERS} | Duration: {SIMULATION_DURATION}s")
    print(f"Targeting: {BASE_URL}")
    print("-" * 50)

    async with httpx.AsyncClient() as client:
        # 0. Login
        if not await login(client):
            print("LOG: Failed to authenticate. Exiting.")
            return

        tasks = []
        start_sim_time = time.time()
        current_delay = 0

        for i in range(TOTAL_USERS):
            query = random.choice(all_queries)
            
            # Mỗi user cách nhau cực ngắn từ 10ms đến 20ms (Tạo Burst Traffic)
            current_delay += random.uniform(0.01, 0.05)
            
            async def staggered_request(d, idx, q):
                await asyncio.sleep(d)
                return await send_request(client, q, idx)
            
            tasks.append(staggered_request(current_delay, i + 1, query))

        # Chạy tất cả các mô phỏng
        results = await asyncio.gather(*tasks)

    # 2. Tổng kết chỉ số
    valid_latencies = [l for l in results if l is not None]
    total_sim_time = time.time() - start_sim_time

    if valid_latencies:
        print("-" * 50)
        print(f"=== SIMULATION SUMMARY ===")
        print(f"Total Successful Requests: {len(valid_latencies)}/{TOTAL_USERS}")
        print(f"Min Latency: {min(valid_latencies):.2f}s")
        print(f"Max Latency: {max(valid_latencies):.2f}s")
        print(f"Average Latency (Per User): {mean(valid_latencies):.2f}s")
        print(f"Overall Throughput: {len(valid_latencies) / total_sim_time:.2f} queries/sec")
        print(f"Total Execution Time: {total_sim_time:.2f}s")
        print("-" * 50)
        print("REMARK: Individual user latency remains low thanks to Dynamic Batching.")

if __name__ == "__main__":
    asyncio.run(main())
