"""AI client — Claude Opus 4.6 only, no Gemini fallback."""

import json
import logging
import re

from anthropic import AsyncAnthropic

from config.settings import settings

logger = logging.getLogger(__name__)

_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client


def _extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


async def generate_json(
    model: str,
    system_instruction: str,
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> dict:
    client = _get_client()
    full_system = f"{system_instruction}\n\nRespond ONLY with valid JSON. No markdown, no explanation."

    try:
        resp = await client.messages.create(
            model=model,
            system=full_system,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.content[0].text
        cleaned = _extract_json(text)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error from {model}: {e}\nRaw: {text[:500]}")
        return {}
    except Exception as e:
        logger.error(f"AI client error ({model}): {e}")
        return {}
