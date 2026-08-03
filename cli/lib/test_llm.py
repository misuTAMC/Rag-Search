import os
from dotenv import load_dotenv
load_dotenv()

api_key=os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is not set in the environment variables.")


from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key)

messages: list[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "Hello, how are you?",
    }
]
response=client.chat.completions.create(
    model="openai/gpt-oss-20b:free",
    messages=messages
)


if __name__ == "__main__":
    print(response.choices[0].message.content)
    if response.usage:
        print(response.usage.prompt_tokens)
        print(response.usage.completion_tokens)
        
        
        
#* ==================== CLI COMMAND FUNCTIONS ====================

def get_spelling_correction(query:str)->str:
    api_key=os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return query
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key)

    
    prompt=f"""Fix any spelling errors in the user-provided movie search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query unchanged.
    Output only the final query text, nothing else.
    User query: "{query}"
    """
    
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    
    try:
        response=client.chat.completions.create(
            model="openai/gpt-oss-20b:free",
            messages=messages
        )
        enhanced_query=response.choices[0].message.content
        if enhanced_query is None:
            return query
        return enhanced_query
    except Exception as e:
        print(f"Error during spelling correction: {e}")
        return query