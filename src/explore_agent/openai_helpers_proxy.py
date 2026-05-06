"""Proxy for openai_helpers — provides simplified interfaces for the explore_agent package."""
import json
from ..openai_helpers import chat_completion_with_retries


def chat_completion(model: str, sys_prompt: str, prompt: str,
                    max_tokens: int = 4000, temperature: float = 0.4,
                    response_format=None) -> str:
    """Call LLM and return raw text content, or empty string on failure."""
    kwargs = {
        'max_tokens': max_tokens,
        'temperature': temperature,
    }
    if response_format:
        kwargs['response_format'] = response_format

    response = chat_completion_with_retries(
        model=model,
        sys_prompt=sys_prompt,
        prompt=prompt,
        **kwargs,
    )

    if response and hasattr(response, 'choices') and response.choices and response.choices[0].message:
        content = response.choices[0].message.content
        if content is not None:
            return content.strip()
    return ""


def parse_json_response(raw: str) -> dict:
    """Robustly parse JSON from LLM output (handles markdown code blocks)."""
    if not raw:
        return None

    # Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Extract from markdown code block
    for prefix in ('```json', '```'):
        if prefix in raw:
            start = raw.index(prefix) + len(prefix)
            end = raw.index('```', start) if '```' in raw[start:] else len(raw)
            try:
                return json.loads(raw[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass
    return None
