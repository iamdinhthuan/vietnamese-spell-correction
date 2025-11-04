
from openai import OpenAI
from pprint import pprint

BASE_URL = "https://mkp-api.fptcloud.com"
API_KEY = "sk-V0llMcFLP9uBabyW9AM-4g"
MODEL_NAME = "gpt-oss-120b"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": "hello world"
        }
    ],
    temperature=1,
    max_tokens=1024,
    presence_penalty=0,
    frequency_penalty=0
)

pprint(response.to_dict())