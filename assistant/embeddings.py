from __future__ import annotations  # Forward type hints (Python 3.11+)
from dataclasses import dataclass

import requests  # For REST calls to Ollama

# Optional OpenAI client (v1)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

import logging
# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from config import SETTINGS  # Settings: API_KEY, MODEL_EMBED, BASE_URL, etc.
except Exception:
    from .config import SETTINGS


@dataclass
class EmbedResult:
    """Simple wrapper for embedding vectors."""
    vectors: "list[list[float]]"

def generate_completion(
    prompt: str,
    context: str = "",
    temperature: "float" = 0.8,
    max_tokens: "int | None" = None,
) -> str:
    """
    Generate a response to a user query using OpenAI's chat completion API.

    Args:
        prompt: The user's query or instruction.
        context: Additional context (retrieved text from DB) to guide the answer.
        temperature: Sampling temperature (higher = more creative, lower = more focused).
        max_tokens: Optional max tokens for the response.

    Returns:
        The generated response text.
    """
    try:
        # Generate response for the user's query
        client = OpenAI(
            base_url=SETTINGS.BASE_URL,
            api_key=SETTINGS.API_KEY,
            # timeout=1200,  # Timeout in seconds for all requests
        )
        response = client.chat.completions.create(
            model=SETTINGS.MODEL_CHAT,  # e.g., "gpt-4o"
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Use the provided context when answering. "
                        "For that you get additional information from a database. It's always a piece of text. "
                        "Please consider this text in your answer. Give a detailed answer."
                    ),
                },
                {
                    "role": "assistant",
                    "content": f"Context (from database): {context}" if context else "",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            # max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.exception(f"[Error generating completion: {e}]")
        try:
            r = requests.post(
                url=f"{SETTINGS.BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {SETTINGS.API_KEY}"},
                json={
                    "model": SETTINGS.MODEL_CHAT,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant. Use the provided context when answering. "
                                "For that you get additional information from a database. It's always a piece of text. "
                                "Please consider this text in your answer. Give a detailed answer."
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": f"Context (from database): {context}" if context else "",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                },
                # timeout=120  # seconds
            )
            return r.json().get("choices", [{"message": {"content": ""}}])[0]["message"]["content"] or ""  # parse the JSON body
        except Exception as e:
            logger.exception(f"[Error generating completion: {e}]")
            return f"[Error generating completion: {e}]"


def generate_embedding(
    texts: "str | list[str]",
) -> list[float]:
    """
    Generate embeddings for one or more texts using:
    - OpenAI API (official Python client)
    - Ollama API (HTTP POST)

    Returns a single embedding if input is single string.
    """
    # Ensure we always have a list of strings
    if isinstance(texts, str):
        texts = [texts]

    # --------------------
    # Ollama embedding via REST
    # --------------------
    ## we'll use REST requests for Ollama
    # Generate embedding for the user's query
    if SETTINGS.LLM_PROVIDER in ["OLLAMA", "HF"]:
        results = []
        for t in texts:
            payload = {
                "prompt": t,  # as str single query, The text to embed
                "model": SETTINGS.OLLAMA_MODEL_EMBED,  # Ollama model
            }
            # POST request to Ollama embeddings endpoint
            r = requests.post(
                # url=f"{SETTINGS.BASE_URL}/chat/completions",
                url=f"{SETTINGS.OLLAMA_BASE_URL.replace("v1", '')}api/embeddings",  # Ollama API endpoint
                json=payload,
                # timeout=60  # Optional: consider adding a timeout
            )
            r.raise_for_status()  # Raise exception if HTTP error
            emb = r.json().get("embedding", [])
            if not emb:
                raise ValueError(f"No embedding returned for text: {t}")
            results.append(emb)
            break  # Only process the first text? Consider removing if multiple texts
        return results[0]  # Return single embedding (first text)
    
    # --------------------
    # OpenAI embeddings client
    # --------------------
    # Generate embedding for the user's query
    client = OpenAI(
        base_url=SETTINGS.BASE_URL,
        api_key=SETTINGS.API_KEY,
    )
    resp = client.embeddings.create(
        input=texts,  # Can be a list of strings or str 
        model=SETTINGS.MODEL_EMBED,
    )
    # Return embedding for first input text
    return resp.data[0].embedding