from typing import List, Dict
import requests
import os


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "mistralai/mistral-7b-instruct"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def build_prompt(question: str, contexts: List[Dict]) -> str:

    context_text = "\n\n".join(
        f"[Document: {c['document']} | Chunk {c['chunk_id']}]\n{c['text']}"
        for c in contexts
    )

    prompt = f"""
You are a question answering system.

Answer the question ONLY using the information provided in the context below.
If the answer cannot be found in the context, respond exactly with:
"I don't know based on the provided context."

Context:
{context_text}

Question:
{question}

Answer:
"""
    return prompt.strip()


def generate_answer(prompt: str) -> str:

    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"].strip()
