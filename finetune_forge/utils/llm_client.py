# finetune_forge/utils/llm_client.py

import json
import logging
from typing import Any

from anthropic import Anthropic

logger = logging.getLogger(__name__)

_client: Anthropic | None = None


def get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


def _strip_json_fence(content: str) -> str:
    """Remove a ```json / ``` markdown fence if the model wrapped its output."""
    clean = content.strip()
    if clean.startswith("```"):
        parts = clean.split("```")
        # parts[1] is the fenced body; fall back to the original if malformed.
        if len(parts) >= 2:
            clean = parts[1].strip()
            if clean.startswith("json"):
                clean = clean[len("json"):].strip()
    return clean.strip()


def call_llm(
    prompt: str,
    system: str = "",
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 2048,
    expect_json: bool = False,
) -> str | dict[str, Any]:
    """
    Call Claude API. If expect_json=True, parses and returns a dict.
    Raises ValueError on JSON parse failure.
    """
    client = get_client()
    messages = [{"role": "user", "content": prompt}]

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    content = response.content[0].text

    if expect_json:
        clean = _strip_json_fence(content)
        try:
            return json.loads(clean)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {content}")
            raise ValueError(f"LLM did not return valid JSON: {e}") from e

    return content
