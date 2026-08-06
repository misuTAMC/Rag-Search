import json
import os
import re
import time
from time import sleep
from dotenv import load_dotenv

from google import genai
from google.genai import types
from openai import OpenAI  

load_dotenv()
from lib.search_utils import PROMPTS_PATH

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")

# Khởi tạo các Clients tương ứng
gemini_client = genai.Client() if GEMINI_KEY else None
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY
) if OPENROUTER_KEY else None


# * ==================== ENGINE HELPER FUNCTIONS ====================

def call_gemini(prompt: str) -> str:
    if not gemini_client:
        return ""
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return response.text.strip() if response.text else ""
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return ""


def call_openrouter(prompt: str, model_name: str = "google/gemma-4-26b-a4b-it:free") -> str:
    if not openrouter_client:
        return ""
    try:
        response = openrouter_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip() if response.choices[0].message.content else ""
    except Exception as e:
        print(f"Error calling OpenRouter API: {e}")
        return ""


def generate_content_hybrid(prompt: str, query: str, prefer_engine: str = "gemini",**args) -> str:
    if "{query}" in prompt:
        prompt = prompt.format(query=query,**args)

    if prefer_engine == "gemini":
        res = call_gemini(prompt)
        if res: return res
        # Nếu sập Quota, move  sang OpenRouter 
        return call_openrouter(prompt)
        
    else:
        res = call_openrouter(prompt)
        if res: return res
        # Nếu OpenRouter nghẽn call Gemini 
        return call_gemini(prompt)


# * ==================== CORE CLI FUNCTIONS ====================

def correct_spelling(query: str) -> str:
    with open(PROMPTS_PATH / "spelling.md", "r", encoding="utf-8") as f:
        prompt_template = f.read()
    enhanced = generate_content_hybrid(prompt_template, query, prefer_engine="openrouter")
    return enhanced if enhanced else query  #sập cả 2, dùng lại câu query gốc


def rewrite_query(query: str) -> str:
    with open(PROMPTS_PATH / "rewrite.md", "r", encoding="utf-8") as f:
        prompt_template = f.read()
    return generate_content_hybrid(prompt_template, query, prefer_engine="openrouter")


def expand_query(query: str) -> str:
    with open(PROMPTS_PATH / "expand.md", "r", encoding="utf-8") as f:
        prompt_template = f.read()
    return generate_content_hybrid(prompt_template, query, prefer_engine="openrouter")


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
            llm_response = generate_content_hybrid(formatted_prompt, query, prefer_engine="gemini")
            match = re.search(r'\d+\.?\d*', llm_response)
            rerank_score = float(match.group()) if match else 0.0
        except Exception as e:
            rerank_score = 0.0
            
        result['rerank_score'] = rerank_score
        
    results.sort(key=lambda x: x.get('rerank_score', 0.0), reverse=True)
    return results


def batch_rerank_results(query: str, doc_list_strings: list[str]):
    with open(PROMPTS_PATH / 'batch_rerank_score.md', 'r', encoding='utf-8') as f:
        prompt_template = f.read()
            
        formatted_prompt = prompt_template.format(
            query=query,
            doc_list_str="\n".join(doc_list_strings)
        )
    try:
        llm_response = generate_content_hybrid(formatted_prompt, query, prefer_engine="gemini")
        
        clean_response = llm_response.strip()
        clean_response = clean_response.replace("```json", "").replace("```", "").strip()
            
        reranked_ids = json.loads(clean_response)
        return reranked_ids
        
    except Exception as e:
        print(f"Error during batch reranking: {e}")
        return []
def rag_answer(query: str, results_list: list[dict]) -> str:
    if not results_list:
        return ""
        
    with open(PROMPTS_PATH / "rag.md", "r", encoding='utf-8') as f:
        prompt_template = f.read()
        
    doc_strings = []
    for doc in results_list:
        doc_strings.append(f"Title: {doc.get('title', '')} - Description: {doc.get('document', '')}")
    docs_combined_str = "\n".join(doc_strings)
    
    formatted_prompt = prompt_template.format(query=query, docs=docs_combined_str)
    try:
        llm_response = generate_content_hybrid(
            formatted_prompt,
            query,
            prefer_engine="gemini"
        )
        
        return llm_response.strip()
    except Exception as e:
        print(f"Error during LLM Calling answers: {e}")
        return ""

def llm_summarization(query:str,results_list:list[dict])->str:
    if not results_list:
        return ""
    with open(PROMPTS_PATH/"summarization.md","r",encoding="utf-8") as f:
        prompt_template=f.read()
    doc_strings=[]
    for doc in results_list:
            doc_strings.append(f"Title: {doc.get('title', '')} - Description: {doc.get('document', '')}")
    docs_combined_str = "\n".join(doc_strings)
    formatted_prompt = prompt_template.format(query=query, results=docs_combined_str)
    try:
        llm_response = generate_content_hybrid(
            formatted_prompt,
            query,
            prefer_engine="gemini"
        )
        
        return llm_response.strip()
    except Exception as e:
        print(f"Error during LLM Summarization : {e}")
        return ""
def llm_citations(query:str,results_list:list[dict])->str:
    if not results_list:
        return "The results are empty!"
    
    with open(PROMPTS_PATH/"citations.md",'r',encoding='utf-8') as f:
        prompt_template=f.read()
    doc_strings=[]
    for doc in results_list:
        doc_strings.append(f"Title: {doc.get('title', '')} - Description: {doc.get('document', '')}")
    docs_combined_str = "\n".join(doc_strings)
    formatted_prompt=prompt_template.format(
        query=query,
        documents=docs_combined_str
    )
    try:
        llm_response=generate_content_hybrid(
            prompt=formatted_prompt,
            query=query,
            prefer_engine='gemini'
        )
        
        return llm_response.strip()
    except Exception as e:
        print(f"Error during LLM Citations : {e}")
        return ""
        
def llm_qa(query:str,results_list:list[dict])->str:
    if not results_list:
        return "The results are empty!"
    
    with open(PROMPTS_PATH/"qa.md",'r',encoding='utf-8') as f:
        prompt_template=f.read()
    doc_strings=[]
    for doc in results_list:
        doc_strings.append(f"Title: {doc.get('title', '')} - Description: {doc.get('document', '')}")
    docs_combined_str = "\n".join(doc_strings)
    print(docs_combined_str)
    formatted_prompt=prompt_template.format(
        query=query,
        context=docs_combined_str
    )
    try:
        llm_response=generate_content_hybrid(
            prompt=formatted_prompt,
            query=query,
            prefer_engine='gemini'
        )
        
        return llm_response.strip()
    except Exception as e:
        print(f"Error during LLM Question-Answering : {e}")
        return ""
    
def llm_judge_results(query:str,results_list:list[dict])->list[int]:
    
    if not results_list:
        return []
    with open(PROMPTS_PATH / "llm_judge.md", "r", encoding="utf-8") as f:
        prompt_template = f.read()
    formatted_results_strings = []
    for res in results_list:
        formatted_results_strings.append(f"Title: {res.get('title', '')} - Document: {res.get('document', '')}")

    formatted_results_combined = "\n".join(formatted_results_strings)

    try:
        llm_response = generate_content_hybrid(
            prompt_template, 
            query, 
            prefer_engine="gemini", 
            formatted_results=formatted_results_combined
        )
        
        clean_response = llm_response.strip()
        if clean_response.startswith("```"):
            clean_response = clean_response.replace("```json", "").replace("```", "").strip()
            
        scores = json.loads(clean_response)
        
        if isinstance(scores, list) and len(scores) == len(results_list):
            return [int(s) for s in scores]
        
        return [0] * len(results_list)
        
    except Exception as e:
        print(f"Error during LLM Judge evaluation process: {e}")
        return [0] * len(results_list)

                