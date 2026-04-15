import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

async def debug():
    print(f"Testing model: models/gemini-3.1-flash-lite-preview")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: No API key found")
        return
        
    try:
        llm = ChatGoogleGenerativeAI(model="models/gemini-3.1-flash-lite-preview", google_api_key=api_key)
        print("Model initialized. Invoking...")
        res = await llm.ainvoke("Generate a single technical question about MIMO-ARQ in wireless communication.")
        print(f"Success! Response content: {res.content}")
    except Exception as e:
        print(f"Failed! Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug())
