"""Promotion Manager — Paper → Shadow → Live graduation pipeline."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

_STATE_FILE = Path("learnings/promotion_state.json")


class PromotionManager:
    """Manages strategy promotion through PAPER → SHADOW → LIVE pipeline."""

    def __init__(self, strategies: list):
        self.strategies = strategies
        self._state: dict[str, dict] = {}
        self._history: list[dict] = []
        self._load_state()

    def _load_state(self):
        if _STATE_FILE.exists():
            try:
                data = json.loads(_STATE_FILE.read_text())
                self._state = data.get("strategies", {})
                self._history = data.get("history", [])
            except Exception:
                pass

    def _save_state(self):
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {"strategies": self._state, "history": self._history}
        _STATE_FILE.write_text(json.dumps(data, indent=2))

    def check_promotions(self, coordinator_recommendations: list | None = None) -> list[dict]:
        """Check all strategies for promotion eligibility."""
        actions = []
        now = datetime.now(timezone.utc)

        for strat in self.strategies:
            sid = strat.cfg.id
            state = self._state.setdefault(sid, {
                "phase": "PAPER",
                "shadow_start": None,
                "shadow_errors": 0,
                "live_trades": 0,
                "live_scale": settings.PROMOTION_LIVE_INITIAL_PCT,
                "first_active": now.isoformat(),
            })

            if strat.is_eliminated:
                continue

            current_phase = state["phase"]

            # PAPER → SHADOW check
            if current_phase == "PAPER":
                stats = strat.stats()
                days_active = (now - datetime.fromisoformat(state["first_active"])).days

                if (
                    stats["trades"] >= settings.PROMOTION_MIN_TRADES
                    and stats["win_rate"] >= settings.PROMOTION_MIN_WR * 100
                    and stats["profit_factor"] >= settings.PROMOTION_MIN_PF
                    and stats["max_drawdown"] < settings.PROMOTION_MAX_DD
                    and days_active >= settings.PROMOTION_MIN_DAYS
                    and not strat.is_paused
                ):
                    # Check coordinator approval
                    approved = True
                    if coordinator_recommendations:
                        for rec in coordinator_recommendations:
                            if rec.get("id") == sid and rec.get("action") == "demote":
                                approved = False
                                break

                    if approved:
                        state["phase"] = "SHADOW"
                        state["shadow_start"] = now.isoformat()
                        state["shadow_errors"] = 0
                        strat.phase = "SHADOW"
                        action = {"id": sid, "action": "PROMOTE_TO_SHADOW", "reason": "Met all PAPER criteria"}
                        actions.append(action)
                        self._history.append({**action, "ts": now.isoformat()})
                        logger.info(f"PROMOTION: {sid} → SHADOW")

            # SHADOW → LIVE check
            elif current_phase == "SHADOW":
                shadow_start = datetime.fromisoformat(state["shadow_start"])
                hours_in_shadow = (now - shadow_start).total_seconds() / 3600

                if state.get("shadow_errors", 0) >= 3:
                    # Demote back to PAPER
                    state["phase"] = "PAPER"
                    strat.phase = "PAPER"
                    action = {"id": sid, "action": "DEMOTE_TO_PAPER", "reason": "Too many shadow errors"}
                    actions.append(action)
                    self._history.append({**action, "ts": now.isoformat()})
                    logger.warning(f"DEMOTION: {sid} → PAPER (shadow errors)")

                elif hours_in_shadow >= settings.PROMOTION_SHADOW_HOURS:
                    state["phase"] = "LIVE"
                    state["live_scale"] = settings.PROMOTION_LIVE_INITIAL_PCT
                    state["live_trades"] = 0
                    strat.phase = "LIVE"
                    strat.live_balance = strat.balance * settings.PROMOTION_LIVE_INITIAL_PCT
                    action = {"id": sid, "action": "PROMOTE_TO_LIVE", "reason": f"Shadow passed ({hours_in_shadow:.0f}h)"}
                    actions.append(action)
                    self._history.append({**action, "ts": now.isoformat()})
                    logger.info(f"PROMOTION: {sid} → LIVE (${strat.live_balance:.2f})")

            # LIVE scaling check
            elif current_phase == "LIVE":
                state["live_trades"] = state.get("live_trades", 0)

                # Scale up every 50 trades
                if state["live_trades"] >= 50:
                    current_scale = state.get("live_scale", settings.PROMOTION_LIVE_INITIAL_PCT)
                    new_scale = min(current_scale + settings.PROMOTION_LIVE_SCALE_STEP, settings.PROMOTION_LIVE_MAX_PCT)
                    if new_scale > current_scale:
                        state["live_scale"] = new_scale
                        strat.live_balance = strat.balance * new_scale
                        state["live_trades"] = 0
                        logger.info(f"LIVE SCALE UP: {sid} → {new_scale:.0%}")

                # Demotion check: >10% loss in live
                if strat.live_balance > 0:
                    initial_live = strat.balance * settings.PROMOTION_LIVE_INITIAL_PCT
                    if initial_live > 0:
                        live_pnl_pct = (strat.live_balance - initial_live) / initial_live
                        if live_pnl_pct < -0.10:
                            state["phase"] = "PAPER"
                            state["live_scale"] = settings.PROMOTION_LIVE_INITIAL_PCT
                            strat.phase = "PAPER"
                            strat.live_balance = 0.0
                            action = {"id": sid, "action": "DEMOTE_TO_PAPER", "reason": f"Live loss {live_pnl_pct:.1%}"}
                            actions.append(action)
                            self._history.append({**action, "ts": now.isoformat()})
                            logger.warning(f"DEMOTION: {sid} → PAPER (live loss)")

        self._save_state()
        return actions

    def demote(self, strategy_id: str) -> dict:
        """Force demotion of a strategy to PAPER."""
        state = self._state.get(strategy_id)
        if not state:
            return {"error": "Strategy not found"}

        state["phase"] = "PAPER"
        state["live_scale"] = settings.PROMOTION_LIVE_INITIAL_PCT

        for strat in self.strategies:
            if strat.cfg.id == strategy_id:
                strat.phase = "PAPER"
                strat.live_balance = 0.0
                break

        self._history.append({
            "id": strategy_id, "action": "MANUAL_DEMOTE",
            "reason": "Manual demotion", "ts": datetime.now(timezone.utc).isoformat(),
        })
        self._save_state()
        return {"id": strategy_id, "action": "DEMOTED_TO_PAPER"}

    def force_promote(self, strategy_id: str) -> dict:
        """Force promotion to next phase."""
        state = self._state.get(strategy_id)
        if not state:
            return {"error": "Strategy not found"}

        current = state["phase"]
        if current == "PAPER":
            state["phase"] = "SHADOW"
            state["shadow_start"] = datetime.now(timezone.utc).isoformat()
            next_phase = "SHADOW"
        elif current == "SHADOW":
            state["phase"] = "LIVE"
            state["live_scale"] = settings.PROMOTION_LIVE_INITIAL_PCT
            next_phase = "LIVE"
        else:
            return {"error": "Already in LIVE"}

        for strat in self.strategies:
            if strat.cfg.id == strategy_id:
                strat.phase = next_phase
                if next_phase == "LIVE":
                    strat.live_balance = strat.balance * settings.PROMOTION_LIVE_INITIAL_PCT
                break

        self._history.append({
            "id": strategy_id, "action": f"FORCE_PROMOTE_TO_{next_phase}",
            "reason": "Manual promotion", "ts": datetime.now(timezone.utc).isoformat(),
        })
        self._save_state()
        return {"id": strategy_id, "action": f"PROMOTED_TO_{next_phase}"}

    def get_pipeline(self) -> dict:
        return {
            "strategies": self._state,
            "history": self._history[-20:],
        }

    def record_shadow_error(self, strategy_id: str):
        if strategy_id in self._state:
            self._state[strategy_id]["shadow_errors"] = self._state[strategy_id].get("shadow_errors", 0) + 1
            self._save_state()

    def record_live_trade(self, strategy_id: str):
        if strategy_id in self._state:
            self._state[strategy_id]["live_trades"] = self._state[strategy_id].get("live_trades", 0) + 1
            self._save_state()
