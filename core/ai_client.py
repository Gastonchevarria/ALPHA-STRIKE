"""AI client — Gemini API with JSON-only responses."""
from __future__ import annotations

import asyncio
import json
import logging
import re

from config.settings import settings

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> str:
    """Extract JSON object from text, stripping markdown wrappers."""
    # Try ```json ... ``` first
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    # Try raw { ... }
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
    """Generate JSON via Gemini API."""
    from google import genai
    from google.genai import types

    api_key = settings.GEMINI_API_KEY
    if not api_key:
        logger.error("GEMINI_API_KEY is not set — AI calls disabled")
        return {}

    client = genai.Client(api_key=api_key)

    try:
        resp = await asyncio.to_thread(
            client.models.generate_content,
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction + "\n\nRespond ONLY with valid JSON. No markdown, no explanation.",
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )
        text = resp.text
        cleaned = _extract_json(text)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error from {model}: {e}\nRaw: {text[:500]}")
        return {}
    except Exception as e:
        logger.error(f"AI client error ({model}): {e}")
        return {}
