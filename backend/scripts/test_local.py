import requests
import os
import time

BASE_URL = "http://localhost:8000"
TEST_USER = "tester_01"
TEST_PASS = "password123"
SESSION_ID = "session_test_99"

def test_backend():
    print("TEST: Starting Local Backend Test...")
    
    # 1. Check Root
    try:
        res = requests.get(f"{BASE_URL}/")
        print(f"OK: Root Health: {res.json()}")
    except:
        print("FAIL: Server is NOT running. Please start it with uvicorn first!")
        return

    # 2. Register
    requests.post(f"{BASE_URL}/register", json={"username": TEST_USER, "password": TEST_PASS})
    
    # 3. Login
    login_res = requests.post(f"{BASE_URL}/login", json={"username": TEST_USER, "password": TEST_PASS})
    if login_res.status_code != 200:
        print("FAIL: Login failed")
        return
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("OK: Login successful, token received.")

    # 4. Upload (Tạo một file text giả lập PDF hoặc gửi text)
    # Ở đây tôi giả định bạn có 1 file PDF test. Nếu không, tôi sẽ bỏ qua bước này 
    # và test bằng cách hỏi Wiki (System scope)
    
    # 5. Chat Test (Sử dụng mode Ultrafast để test tốc độ)
    print(f"LOG: Testing Chat (Mode: Ultrafast, Session: {SESSION_ID})...")
    chat_payload = {
        "message": "What is TurboQuant?",
        "mode": "ultrafast",
        "scope": "both",
        "session_id": SESSION_ID
    }
    
    start = time.time()
    chat_res = requests.post(f"{BASE_URL}/chat", json=chat_payload, headers=headers)
    end = time.time()
    
    if chat_res.status_code == 200:
        print(f"OK: Chat response started in {end-start:.2f}s")
        print("--- STREAMING RESPONSE ---")
        full_text = ""
        for line in chat_res.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if "--META_END--" in decoded_line:
                    meta_part = decoded_line.split("--META_END--")[0]
                    print(f"METADATA: {meta_part}")
                else:
                    print(decoded_line, end="", flush=True)
                    full_text += decoded_line
        print("\n--------------------------")
        print(f"OK: Full answer received.")
    else:
        print(f"FAIL: Chat failed: {chat_res.text}")

if __name__ == "__main__":
    test_backend()
