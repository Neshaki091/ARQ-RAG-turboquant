import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_history():
    print("[*] Testing Chat History Feature...")
    
    # 1. Login
    user_data = {"username": "testuser_history", "password": "testpass123"}
    requests.post(f"{BASE_URL}/register", json=user_data)
    res = requests.post(f"{BASE_URL}/login", json=user_data)
    token = res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Create a specific session
    session_id = f"session_{int(time.time())}"
    requests.post(f"{BASE_URL}/sessions", json={"session_id": session_id, "title": "Test History Session"}, headers=headers)
    
    # 3. Send a Chat message
    print(f"[*] Sending question to session {session_id}...")
    chat_data = {
        "message": "Xin chào, bạn có thể giúp tôi được không?",
        "session_id": session_id,
        "scope": "system"
    }
    
    # Wait for full response
    with requests.post(f"{BASE_URL}/chat", json=chat_data, headers=headers, stream=True) as r:
        for line in r.iter_lines(): pass # Just consume the stream
    
    print("[+] AI response finished.")
    
    # 4. Verify History
    print("[*] Verifying history in Database...")
    res = requests.get(f"{BASE_URL}/sessions/{session_id}/messages", headers=headers)
    history = res.json()["messages"]
    
    print("-" * 30)
    for msg in history:
        print(f"[{msg['role'].upper()}]: {msg['content'][:100]}...")
    print("-" * 30)
    
    if len(history) >= 2:
        print("[SUCCESS] Chat History is working!")
    else:
        print("[FAILED] Chat History was not saved.")

if __name__ == "__main__":
    test_history()
