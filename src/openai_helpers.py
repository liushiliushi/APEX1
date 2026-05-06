import time
from typing import Mapping
import openai
from openai import OpenAI
import tiktoken
import os
import json
from dotenv import load_dotenv

# Force reload from .env file, overriding system environment variables
load_dotenv(override=True)
encoding = tiktoken.get_encoding("cl100k_base")

# Global token tracker
_token_tracker = {
    'calls': 0,
    'prompt_tokens': 0,
    'completion_tokens': 0,
    'total_tokens': 0,
}

def get_token_stats() -> dict:
    """Return current cumulative token usage stats."""
    return dict(_token_tracker)

def reset_token_stats():
    """Reset token tracker to zero."""
    for k in _token_tracker:
        _token_tracker[k] = 0

def chat_completion_with_retries(model: str, sys_prompt: str, prompt: str, max_retries: int = 5, retry_interval_sec: int = 20, top_logprobs=0, response_format=None, **kwargs) -> Mapping:

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key,
                    base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))

    for n_attempts_remaining in range(max_retries, 0, -1):
        try:
            # 构建基础参数
            create_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                **kwargs
            }
            
            # 添加logprobs参数
            if top_logprobs > 0:
                create_params["logprobs"] = True
                create_params["top_logprobs"] = top_logprobs
            
            # 添加response_format参数
            if response_format:
                create_params["response_format"] = response_format
            
            res = client.chat.completions.create(**create_params)

            # Track token usage
            _token_tracker['calls'] += 1
            if res and hasattr(res, 'usage') and res.usage:
                _token_tracker['prompt_tokens'] += getattr(res.usage, 'prompt_tokens', 0) or 0
                _token_tracker['completion_tokens'] += getattr(res.usage, 'completion_tokens', 0) or 0
                _token_tracker['total_tokens'] += getattr(res.usage, 'total_tokens', 0) or 0

            return res

        except (
            openai.RateLimitError,
            openai.APIError,
            openai.OpenAIError,
            json.JSONDecodeError,
            ) as e:
            print(e)
            print(f"Hit openai.error exception. Waiting {retry_interval_sec} seconds for retry... ({n_attempts_remaining - 1} attempts remaining)", flush=True)
            time.sleep(retry_interval_sec)
    return {}
def truncate_text(text, max_tokens):
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        print(f"WARNING: Maximum token length exceeded ({len(tokens)} > {max_tokens})")
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    return text

def claude_completion_with_retries(model: str, sys_prompt: str, prompt: str, max_retries: int = 5, retry_interval_sec: int = 20, **kwargs) -> str:
    """
    Call Claude model through OpenRouter with retries.
    Returns the text content directly (for non-JSON responses).

    Args:
        model: Claude model name (e.g., "anthropic/claude-3.5-sonnet")
        sys_prompt: System prompt
        prompt: User prompt
        max_retries: Maximum retry attempts
        retry_interval_sec: Seconds between retries
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        str: Response text content, or empty string if failed
    """
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))

    for n_attempts_remaining in range(max_retries, 0, -1):
        try:
            create_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                **kwargs
            }

            res = client.chat.completions.create(**create_params)

            if res and hasattr(res, 'choices') and res.choices:
                return res.choices[0].message.content.strip()
            else:
                print("[Warning] No valid response from Claude")
                return ""

        except (
            openai.RateLimitError,
            openai.APIError,
            openai.OpenAIError,
            json.JSONDecodeError,
        ) as e:
            print(f"Claude API error: {e}")
            print(f"Waiting {retry_interval_sec} seconds for retry... ({n_attempts_remaining - 1} attempts remaining)", flush=True)
            time.sleep(retry_interval_sec)

    print("Failed to get Claude response after all retries")
    return ""

def get_embedding_with_retries(text: str, model: str = "text-embedding-ada-002", max_retries: int = 5, retry_interval_sec: int = 20, max_tokens: int = 8000):
    """
    Get text embedding using OpenAI API with retries.

    Args:
        text: Text to embed
        model: OpenAI embedding model to use (default: text-embedding-ada-002)
        max_retries: Maximum number of retry attempts
        retry_interval_sec: Seconds to wait between retries
        max_tokens: Maximum number of tokens (text will be truncated if exceeded)

    Returns:
        numpy array of embedding vector, or None if failed
    """
    import numpy as np

    # Truncate text if too long
    text = truncate_text(text, max_tokens)

    api_key = os.getenv("OPENAI_API_KEY2")
    if not api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        return None

    # Use direct OpenAI API (not through openrouter)
    client = OpenAI(api_key=api_key)

    for n_attempts_remaining in range(max_retries, 0, -1):
        try:
            response = client.embeddings.create(
                model=model,
                input=text
            )

            # Extract embedding vector
            embedding = np.array(response.data[0].embedding, dtype=np.float32)

            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except (
            openai.RateLimitError,
            openai.APIError,
            openai.OpenAIError,
        ) as e:
            print(f"OpenAI embedding error: {e}")
            print(f"Waiting {retry_interval_sec} seconds for retry... ({n_attempts_remaining - 1} attempts remaining)", flush=True)
            time.sleep(retry_interval_sec)

    print("Failed to get embedding after all retries")
    return None

