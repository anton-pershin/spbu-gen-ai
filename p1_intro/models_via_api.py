# explain OpenAI REST API
# use it via requests
# use it via openai
# use it via litellm

import json

import requests
from openai import OpenAI
import litellm

from api_keys import CAILA_API_KEY 

def run_requests(messages: list[dict[str, str]]) -> None:
    r = requests.post(
        "https://caila.io/api/adapters/openai/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CAILA_API_KEY}" 
        },
        data=json.dumps({
            "model": "just-ai/openai-proxy/o3-mini",
            "messages": messages,
        }),
    )

    response_json = json.loads(r.text)
    print(response_json["choices"][0]["message"]["content"])


def run_openai(messages: list[dict[str, str]]) -> None:
    client = OpenAI(
        api_key=CAILA_API_KEY,
        base_url="https://caila.io/api/adapters/openai"
    )
    completion = client.chat.completions.create(
        messages=messages,
        model="just-ai/openai-proxy/o3-mini",
    )
    print(completion.choices[0].message.content)


def run_litellm(messages: list[dict[str, str]]) -> None:
    response = litellm.completion(
        model="openai/just-ai/openai-proxy/o3-mini",
        api_key=CAILA_API_KEY,
        api_base="https://caila.io/api/adapters/openai",
        messages=messages,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    system_prompt = 'You are a helpful assistant. Each message you write should end with the sentence "Thank you for your attention to this matter"'
    user_prompt = "Write the Fibonacci sequence generator"
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    print("### Generation via requests ###\n\n")
    run_requests(messages)

    print("\n\n### Generation via openai ###\n\n")
    run_openai(messages)
    
    print("\n\n### Generation via litellm ###\n\n")
    run_litellm(messages)

