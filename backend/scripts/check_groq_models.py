import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv("backend/.env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
valid_models = client.models.list()
for m in valid_models.data:
    print(f"- {m.id}")
