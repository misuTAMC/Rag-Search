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
# response=client.chat.completions.create(
#     model="openai/gpt-oss-20b:free",
#     messages=messages
# )
# print(response.choices[0].message.content)
# print(response.usage)

