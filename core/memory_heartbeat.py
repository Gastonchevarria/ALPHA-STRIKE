"""Memory consolidation heartbeat + nightly reflection."""

import asyncio
import logging
from datetime import datetime, timezone

from config.settings import settings
from core.ai_client import generate_json
from core.memory_tiers import add_memory, get_recent, promote

logger = logging.getLogger(__name__)


class MemoryHeartbeat:
    """Periodic memory consolidation and nightly reflection."""

    def __init__(self):
        self.last_consolidation: str | None = None
        self.last_reflection: str | None = None

    async def consolidate_short_to_mid(self):
        """Take recent short-term memories and consolidate to mid-tier."""
        entries = get_recent("short", limit=5)
        if len(entries) < 3:
            return

        combined = " | ".join(e["content"] for e in entries[-3:])
        add_memory(
            "mid",
            f"[AUTO-CONSOLIDATED] {combined[:300]}",
            tags=["auto_consolidated"],
        )
        self.last_consolidation = datetime.now(timezone.utc).isoformat()
        logger.info("Memory heartbeat: consolidated short → mid")

    async def nightly_reflection(self, strategies_summary: str = ""):
        """Daily reflection: analyze mid-tier and promote patterns to long-term."""
        if not settings.GEMINI_API_KEY:
            return

        entries = get_recent("mid", limit=20)
        if len(entries) < 5:
            return

        entries_text = "\n".join(f"- {e['content']}" for e in entries)

        prompt = (
            f"MID-TERM MEMORIES:\n{entries_text}\n\n"
            f"STRATEGIES SUMMARY:\n{strategies_summary}\n\n"
            "Identify patterns worth promoting to long-term memory."
        )

        system = """Analyze these trading memories and identify structural patterns.
Output ONLY valid JSON:
{
  "patterns_found": [
    {
      "pattern": "<description>",
      "confidence": 0.0-1.0,
      "promote": true|false,
      "tags": ["pattern_type"]
    }
  ],
  "top3_strategies": "<60 words summary of best performers>"
}
Rules:
- Only promote patterns seen in 3+ observations
- Confidence >= 0.75 to promote
- Maximum 3 promotions per reflection
"""

        try:
            analysis = await generate_json(
                model=settings.BRAIN_MODEL,
                system_instruction=system,
                prompt=prompt,
                temperature=0.2,
                max_tokens=512,
            )

            if not analysis:
                return

            for p in analysis.get("patterns_found", []):
                if p.get("promote") and p.get("confidence", 0) >= 0.75:
                    promote("mid", f"[PATTERN] {p['pattern']}", tags=p.get("tags", []))

            top3 = analysis.get("top3_strategies", "")
            if top3:
                add_memory("long", f"[NIGHTLY] {top3}", tags=["nightly_reflection"])

            self.last_reflection = datetime.now(timezone.utc).isoformat()
            logger.info("Nightly reflection completed")

        except Exception as e:
            logger.error(f"Nightly reflection error: {e}")
