import requests
import json
import time
import sys
import io

# Đảm bảo in được tiếng Việt trên terminal Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE_URL = "http://localhost:8000"

def test():
    print(f"[*] Testing Backend at {BASE_URL}...")
    
    # 1. Root check
    res = requests.get(f"{BASE_URL}/")
    print(f"[+] Root: {res.json()}")
    
    # 2. Register
    user_data = {"username": "testuser_" + str(int(time.time())), "password": "testpass123"}
    res = requests.post(f"{BASE_URL}/register", json=user_data)
    print(f"[+] Register: {res.json()}")
    
    # 3. Login
    res = requests.post(f"{BASE_URL}/login", json=user_data)
    login_res = res.json()
    print(f"[+] Login successful")
    token = login_res["access_token"]
    
    # 4. Chat (RAG Search)
    headers = {"Authorization": f"Bearer {token}"}
    chat_data = {
        "message": "DNS là gì và nó hoạt động như thế nào?",
        "mode": "balance",
        "scope": "system"
    }
    
    print(f"[*] Sending RAG query to 5 million vectors...")
    start = time.time()
    
    with requests.post(f"{BASE_URL}/chat", json=chat_data, headers=headers, stream=True) as r:
        metadata_buffer = ""
        metadata_received = False
        full_response = ""
        
        with open("ai_response.txt", "w", encoding="utf-8") as f:
            for line in r.iter_lines():
                if not line: continue
                decoded = line.decode('utf-8')
                
                if not metadata_received:
                    if "--META_END--" in decoded:
                        metadata_buffer += decoded.split("--META_END--")[0]
                        try:
                            meta = json.loads(metadata_buffer.strip())
                            print(f"[+] Metadata: Latency={meta['latency']}, Complexity={meta['complexity']}")
                            print(f"[+] Sources Found: {len(meta['sources'])} documents")
                        except Exception as e:
                            print(f"[!] Meta Parse Error: {e}")
                        metadata_received = True
                    else:
                        metadata_buffer += decoded
                else:
                    full_response += decoded
                    f.write(decoded)
                    # Print a bit to console
                    if len(full_response) < 500:
                        print(decoded, end="", flush=True)
        
    duration = time.time() - start
    print(f"\n\n[+] Chat complete in {duration:.2f}s")
    print("-" * 50)
    print(f"Full response saved to ai_response.txt")

if __name__ == "__main__":
    try:
        test()
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
