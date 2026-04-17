"""Per-trade AI post-mortem analysis using Claude Opus 4.6."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from core.ai_client import generate_json
from core.memory_tiers import add_memory

logger = logging.getLogger(__name__)

_TRADES_FILE = Path("learnings/trades.jsonl")


class StrategySelfTrainer:
    """Analyzes each closed trade and adjusts strategy parameters."""

    _SYSTEM = """You are the SelfTrainer of a multi-pair crypto futures trading bot (Agent GOD 2).
Analyze this closed trade and extract actionable learnings.

Output ONLY valid JSON:
{
  "outcome_assessment": "correct_call|premature_exit|wrong_direction|good_risk_mgmt",
  "market_condition": "trending_up|trending_down|ranging|volatile",
  "signal_quality": 0.0-1.0,
  "pair_selection_quality": "optimal|acceptable|poor",
  "key_lesson": "<60 words>",
  "param_adjustments": {
    "tp_pct": 0.005,
    "sl_pct": 0.003,
    "margin_pct": 0.08
  },
  "promote_to_memory": true|false,
  "memory_tier": "short|mid|long",
  "memory_content": "<60 words>"
}
Rules:
- param_adjustments: only suggest if clearly needed. Max ±2% per iteration.
- promote_to_memory=true only for non-obvious insights
- long tier only for structural lessons validated by 5+ trades
"""

    def __init__(self, strategy_id: str, enabled: bool = True):
        self.strategy_id = strategy_id
        self.enabled = enabled

    async def analyze(
        self,
        trade: dict,
        signals: dict,
        market_context: str,
        ltm,
    ) -> dict | None:
        if not self.enabled or not settings.ANTHROPIC_API_KEY:
            return None

        prompt = (
            f"STRATEGY: {self.strategy_id}\n"
            f"TRADE:\n{json.dumps(trade, indent=2)}\n\n"
            f"SIGNALS:\n{json.dumps(signals, indent=2)}\n\n"
            f"CONTEXT: {market_context}\n\n"
            f"CURRENT PARAMS:\n{json.dumps(ltm.all_params(), indent=2)}"
        )

        try:
            analysis = await generate_json(
                model=settings.EXEC_MODEL,
                system_instruction=self._SYSTEM,
                prompt=prompt,
                temperature=0.2,
                max_tokens=512,
            )

            if not analysis:
                return None

            # Apply parameter adjustments with safety rails
            for k, v in analysis.get("param_adjustments", {}).items():
                current = ltm.get_param(k)
                if current is not None:
                    safe_v = max(current - 0.02, min(current + 0.02, float(v)))
                    ltm.set_param(k, round(safe_v, 4), source="self_trainer")

            # Persist trade to JSONL
            _TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "strategy_id": self.strategy_id,
                "direction": trade.get("direction"),
                "pair": trade.get("pair", ""),
                "outcome": trade.get("reason"),
                "pnl_net": trade.get("pnl_net"),
                "price": trade.get("price"),
                "analysis": analysis,
            }
            with open(_TRADES_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")

            # Memory promotion
            if analysis.get("promote_to_memory"):
                tier = analysis.get("memory_tier", "short")
                content = analysis.get("memory_content", analysis.get("key_lesson", ""))
                add_memory(tier, f"[{self.strategy_id}] {content}", tags=["trade_lesson", self.strategy_id])

            return analysis

        except Exception as e:
            logger.error(f"SelfTrainer error for {self.strategy_id}: {e}")
            return None
