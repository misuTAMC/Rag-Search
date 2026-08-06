import argparse
import base64
import mimetypes
import os
from openai import OpenAI
from lib.search_utils import PROMPTS_PATH
from openai.types.chat import ChatCompletionUserMessageParam
def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal Query Rewriting using OpenRouter")
    parser.add_argument("--image", type=str, required=True, help="The path to an image file")
    parser.add_argument("--query", type=str, required=True, help="A text query to rewrite based on the image")
    args = parser.parse_args()
  
    
    api_key=os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable is not set.")
        return
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        
    #determine the MIME type of the image file, defaulting to "image/jpeg":
    mime,_=mimetypes.guess_type(args.image)
    mime=mime or "image/jpeg"
    
    with open(args.image,"rb") as image_file:
        img=image_file.read()
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    with open(PROMPTS_PATH/"multimodal.md","r",encoding="utf-8") as f:
        system_prompt=f.read()
    
    data_url=f"data:{mime};base64,{base64.b64encode(img).decode()}"
    
    messages: list[ChatCompletionUserMessageParam] = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": system_prompt.strip()},
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": args.query.strip()},
        ],
    }
    ]
    
    try:
        
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=messages
        )

        
        content = response.choices[0].message.content
        content = response.choices[0].message.content or ""
        print(f"Rewritten query: {content.strip()}")

        if response.usage is not None:
            print(f"Total tokens:    {response.usage.total_tokens}")

    except Exception as e:
        print(f"Error during API call to OpenRouter: {e}")
    
        

if __name__ == "__main__":
    main()