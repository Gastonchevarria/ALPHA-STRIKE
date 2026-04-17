"""Auto-pause/eliminate strategies based on 24h performance."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

_STATE_FILE = Path("learnings/eliminator_state.json")


class StrategyEliminator:
    """Evaluates strategies and pauses/eliminates poor performers."""

    def __init__(self, strategies: list, initial_balance: float = 1000.0):
        self.strategies = strategies
        self.initial_balance = initial_balance
        self._state: dict[str, dict] = {}
        self._load_state()

    def _load_state(self):
        if _STATE_FILE.exists():
            try:
                self._state = json.loads(_STATE_FILE.read_text())
            except Exception:
                self._state = {}

    def _save_state(self):
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(self._state, indent=2))

    def evaluate_all(self) -> list[dict]:
        """Evaluate all strategies and return list of actions taken."""
        actions = []
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=24)

        for strat in self.strategies:
            sid = strat.cfg.id
            state = self._state.setdefault(sid, {
                "status": "ACTIVE",
                "pause_count": 0,
                "paused_until": None,
                "history": [],
            })

            # Check if pause expired
            if state["status"] == "PAUSED" and state.get("paused_until"):
                until = datetime.fromisoformat(state["paused_until"])
                if now >= until:
                    state["status"] = "ACTIVE"
                    strat.is_paused = False
                    actions.append({"id": sid, "action": "UNPAUSE", "reason": "Pause expired"})

            if state["status"] in ("ELIMINATED", "PAUSED"):
                continue

            # Calculate 24h PnL
            closed = [
                t for t in strat.trade_log
                if t.get("action") == "CLOSE"
                and datetime.fromisoformat(t["ts"]) >= cutoff
            ]

            if len(closed) < settings.ELIMINATOR_MIN_TRADES:
                continue

            pnl_24h = sum(t.get("pnl_net", 0) for t in closed)
            pnl_pct = pnl_24h / max(self.initial_balance, 1)

            if pnl_pct <= settings.ELIMINATOR_THRESHOLD_PCT:
                state["pause_count"] = state.get("pause_count", 0) + 1
                reason = f"24h PnL ${pnl_24h:.2f} ({pnl_pct:.1%}) <= {settings.ELIMINATOR_THRESHOLD_PCT:.1%}"

                if state["pause_count"] >= settings.ELIMINATOR_MAX_PAUSES:
                    state["status"] = "ELIMINATED"
                    strat.is_eliminated = True
                    strat.is_paused = False

                    # If live, demote first
                    if strat.phase == "LIVE":
                        strat.phase = "PAPER"
                        strat.live_balance = 0.0

                    actions.append({"id": sid, "action": "ELIMINATE", "reason": reason})
                else:
                    state["status"] = "PAUSED"
                    state["paused_until"] = (now + timedelta(hours=settings.ELIMINATOR_PAUSE_HOURS)).isoformat()
                    strat.is_paused = True
                    actions.append({
                        "id": sid,
                        "action": "PAUSE",
                        "reason": reason,
                        "until": state["paused_until"],
                        "pause_count": state["pause_count"],
                    })

                state["history"].append({"ts": now.isoformat(), "action": actions[-1]["action"], "reason": reason})

        self._save_state()
        return actions

    def reactivate(self, strategy_id: str) -> dict:
        """Force reactivate a paused/eliminated strategy."""
        state = self._state.get(strategy_id)
        if not state:
            return {"error": f"Strategy {strategy_id} not found"}

        state["status"] = "ACTIVE"
        state["pause_count"] = 0
        state["paused_until"] = None

        for strat in self.strategies:
            if strat.cfg.id == strategy_id:
                strat.is_paused = False
                strat.is_eliminated = False
                break

        self._save_state()
        return {"id": strategy_id, "action": "REACTIVATED"}

    def full_status(self) -> dict:
        return self._state
