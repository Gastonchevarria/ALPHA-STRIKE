"""Tournament Coordinator — Gemini brain for confidence adjustment."""

import json
import logging
from datetime import datetime, timezone

from config.settings import settings
from core.ai_client import generate_json
from core.memory_tiers import add_memory, get_all_context

logger = logging.getLogger(__name__)


class TournamentCoordinator:
    """The Brain: analyzes tournament and adjusts strategy confidence multipliers."""

    _SYSTEM = """You are TournamentCoordinator for Agent GOD 2 — a 13-strategy multi-pair crypto futures tournament.
Strategies trade 6 pairs (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT) with 10-50x leverage.
Some strategies are in PAPER mode, some may be in SHADOW or LIVE.

Output ONLY valid JSON:
{
  "market_assessment": "<30 words>",
  "hot_strategies": ["<id>"],
  "cold_strategies": ["<id>"],
  "regime_recommendation": "TRENDING_UP|TRENDING_DOWN|RANGING|VOLATILE",
  "confidence_adjustments": { "<strategy_id>": 1.0 },
  "pair_recommendations": { "<pair>": "active|cautious|avoid" },
  "promotion_recommendations": [{"id": "<strategy_id>", "action": "promote|demote|hold", "reason": "<30 words>"}],
  "key_insight": "<60 words>",
  "memory_worthy": true,
  "risk_level": "LOW|MEDIUM|HIGH|EXTREME"
}
Rules:
- confidence > 1.1 only if PF > 1.3 AND WR > 55% AND trades >= 20
- confidence < 0.8 if WR < 35% OR last 5 trades all SL
- All confidence values: 0.5–1.5 range
- promotion_recommendations: only recommend 'promote' if strategy meets PAPER graduation criteria
- LIVE strategies: recommend 'demote' if recent performance deteriorating
"""

    def __init__(self):
        self.last_run: str | None = None
        self.last_analysis: dict | None = None
        self._multipliers: dict[str, float] = {}

    def get_multiplier(self, strategy_id: str) -> float:
        return self._multipliers.get(strategy_id, 1.0)

    async def run(
        self,
        tournament_status: dict,
        regimes: dict,
        correlation: dict,
    ) -> dict | None:
        if not settings.GEMINI_API_KEY:
            return None

        memory_ctx = get_all_context(max_per_tier=5)

        prompt = (
            f"TOURNAMENT STATUS:\n{json.dumps(tournament_status, indent=2)}\n\n"
            f"MARKET REGIMES:\n{json.dumps(regimes, indent=2)}\n\n"
            f"CORRELATION:\n{json.dumps(correlation, indent=2)}\n\n"
            f"MEMORY:\n{memory_ctx}"
        )

        try:
            analysis = await generate_json(
                model=settings.BRAIN_MODEL,
                system_instruction=self._SYSTEM,
                prompt=prompt,
                temperature=0.15,
                max_tokens=1024,
            )

            if not analysis:
                return None

            # Apply clamped multipliers
            for sid, mult in analysis.get("confidence_adjustments", {}).items():
                self._multipliers[sid] = max(0.5, min(1.5, float(mult)))

            # Memory promotion
            if analysis.get("memory_worthy"):
                insight = analysis.get("key_insight", "")
                add_memory("long", f"[COORDINATOR] {insight}", tags=["coordinator"])

            self.last_run = datetime.now(timezone.utc).isoformat()
            self.last_analysis = analysis
            return analysis

        except Exception as e:
            logger.error(f"Coordinator error: {e}")
            return None
