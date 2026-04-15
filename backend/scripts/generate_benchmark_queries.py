import os
import json
import time
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

load_dotenv()

async def generate_questions():
    # 1. Load data
    data_path = "backend/data/chunks.json"
    output_path = "backend/data/benchmark_queries.json"
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 2. Check existing queries to resume
    existing_queries = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                existing_queries = json.load(f)
            except:
                existing_queries = []
    
    if len(existing_queries) >= 1000:
        print(f"Benchmark queries already have {len(existing_queries)} questions. Done.")
        return

    print(f"Current questions: {len(existing_queries)}. Generating more to reach 1000...")

    # 3. Setup LLM
    llm = ChatGoogleGenerativeAI(model="models/gemini-3.1-flash-lite-preview", google_api_key=os.getenv("GOOGLE_API_KEY"))

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant specializing in Wireless Communication and MIMO-ARQ. "
                   "Based on the following document chunk, generate {num_questions} diverse and high-quality technical questions. "
                   "The questions should be a mix of English and Vietnamese. "
                   "Format your response as a JSON list of strings."),
        ("human", "Document content: {content}")
    ])

    chain = prompt | llm

    # 4. Generate loop
    target_count = 1000
    queries = existing_queries
    
    # Calculate how many questions per chunk on average
    remaining = target_count - len(queries)
    questions_per_chunk = max(3, (remaining // len(chunks)) + 1)

    pbar = tqdm(total=target_count)
    pbar.update(len(queries))

    for chunk in chunks:
        if len(queries) >= target_count:
            break
            
        content = chunk.get("content", "")
        if len(content) < 100:
            continue

        try:
            num_to_gen = min(questions_per_chunk, target_count - len(queries))
            response = await chain.ainvoke({"content": content, "num_questions": num_to_gen})
            
            # Extract JSON list from response
            text = response.content
            # Handle if text is a list of blocks (unlikely but safe)
            if isinstance(text, list):
                text = " ".join([str(item) for item in text])
                
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            cleaned_text = text.strip()
            # If the user included extra text before/after the list, try to find the [ ]
            if "[" in cleaned_text and "]" in cleaned_text:
                start = cleaned_text.find("[")
                end = cleaned_text.rfind("]") + 1
                cleaned_text = cleaned_text[start:end]

            new_questions = json.loads(cleaned_text)
            
            if isinstance(new_questions, list):
                for q in new_questions:
                    if len(queries) < target_count:
                        queries.append(q)
                        pbar.update(1)
            
            # Save progress incrementally
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(queries, f, ensure_ascii=False, indent=4)
                
            # Rate limiting / polite delay
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"Error processing chunk: {e}")
            await asyncio.sleep(2)

    pbar.close()
    print(f"Finished! Total questions in {output_path}: {len(queries)}")

if __name__ == "__main__":
    asyncio.run(generate_questions())
