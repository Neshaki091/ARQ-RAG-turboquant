import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def test_groq():
    api_key = os.getenv("GROQ_API_KEY")
    model_name = "llama-3.3-70b-versatile"
    print(f"Testing model: {model_name}")
    print(f"API Key prefix: {api_key[:10]}...")
    
    try:
        llm = ChatGroq(
            model_name=model_name,
            api_key=api_key,
            temperature=0
        )
        response = llm.invoke("Hi")
        print(f"Success: {response.content}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_groq()
