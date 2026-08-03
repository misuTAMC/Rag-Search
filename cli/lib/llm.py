import os
import re
import time
from time import sleep
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()
from lib.search_utils import PROMPTS_PATH
api_key=os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set in the environment variables.")


# from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
# MODEL='google/gemma-4-31b-it:free'
# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=api_key)

# messages: list[ChatCompletionMessageParam] = [
#     {
#         "role": "user",
#         "content": "Hello, how are you?",
#     }
# ]
# response=client.chat.completions.create(
#     model=MODEL,
#     messages=messages
# )
client = genai.Client()


#* ==================== CLI COMMAND FUNCTIONS ====================
def generate_content(prompt: str, query: str) -> str:
    if "{query}" in prompt:
        prompt = prompt.format(query=query)
        
    client = genai.Client()
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
            ),
        )
        
        if response.text:
            return response.text.strip()
        return ""
    except Exception as e:
        print(f"Error : {e}")
        return ""
    
def correct_spelling(query:str)->str:
    # client = OpenAI(
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=api_key)

    with open(PROMPTS_PATH/"spelling.md", "r", encoding="utf-8") as f:
        prompt_template = f.read()
    
    return generate_content(prompt_template,query)

def rewrite_query(query:str)->str:
    # client = OpenAI(
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=api_key)

    with open(PROMPTS_PATH/"rewrite.md", "r", encoding="utf-8") as f:
        prompt_template = f.read()
    
    return generate_content(prompt_template,query)
def expand_query(query:str)->str:
    # client = OpenAI(
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=api_key)

    with open(PROMPTS_PATH/"expand.md", "r", encoding="utf-8") as f:
        prompt_template = f.read()
    
    return generate_content(prompt_template,query)

def rerank_results(query: str, results: list[dict]) -> list[dict]:
    with open(PROMPTS_PATH / 'rerank_score.md', 'r', encoding='utf-8') as f:
        prompt_template = f.read()
        
    for idx, result in enumerate(results):
        formatted_prompt = prompt_template.format(
            query=query,
            title=result.get('title', ''),
            document=result.get('document', '')
        )
        
        if idx > 0:
            sleep(12) 
            
        try:
            llm_response = generate_content(formatted_prompt, query)
            
            #extract diem so rerank tu llm_response
            match = re.search(r'\d+\.?\d*', llm_response)
            rerank_score = float(match.group()) if match else 0.0
        except Exception as e:
            print(f"Error processing result with title '{result.get('title', '')}': {e}")
            rerank_score = 0.0
            
        result['rerank_score'] = rerank_score
        
    results.sort(key=lambda x: x.get('rerank_score', 0.0), reverse=True)
    return results

        
        