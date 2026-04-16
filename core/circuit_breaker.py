"""Dual-level circuit breaker: paper (lenient) + live (aggressive)."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_FILE = Path("learnings/circuit_breaker.json")


class CircuitBreaker:
    def __init__(
        self,
        initial_paper_total: float,
        paper_threshold: float = -0.12,
        live_threshold: float = -0.05,
        max_concurrent_live: int = 3,
        max_live_capital_pct: float = 0.30,
    ):
        self.initial_paper_total = initial_paper_total
        self.paper_threshold = paper_threshold
        self.live_threshold = live_threshold
        self.max_concurrent_live = max_concurrent_live
        self.max_live_capital_pct = max_live_capital_pct

        self.paper_triggered = False
        self.live_triggered = False
        self.paper_halt_until: datetime | None = None
        self.live_halt_until: datetime | None = None
        self.paper_reason = ""
        self.live_reason = ""

        self.live_daily_pnl = 0.0
        self.live_daily_reset: datetime = datetime.now(timezone.utc)
        self.live_total_capital = 0.0

        self._load_state()

    def check_paper(self, current_paper_total: float) -> bool:
        now = datetime.now(timezone.utc)
        if self.paper_triggered and self.paper_halt_until:
            if now >= self.paper_halt_until:
                self.paper_triggered = False
                self.paper_reason = ""
                self.paper_halt_until = None
            else:
                return True

        pct = (current_paper_total - self.initial_paper_total) / max(self.initial_paper_total, 1)
        if pct <= self.paper_threshold:
            self.paper_triggered = True
            self.paper_halt_until = now + timedelta(hours=24)
            self.paper_reason = f"Paper portfolio {pct:.1%} <= {self.paper_threshold:.1%}"
            logger.warning(f"PAPER CIRCUIT BREAKER TRIPPED: {self.paper_reason}")
            self._save_state()
            return True
        return False

    def check_live(self, live_pnl_delta: float = 0.0, live_capital: float = 0.0) -> bool:
        now = datetime.now(timezone.utc)

        if (now - self.live_daily_reset).total_seconds() > 86400:
            self.live_daily_pnl = 0.0
            self.live_daily_reset = now

        if self.live_triggered and self.live_halt_until:
            if now >= self.live_halt_until:
                self.live_triggered = False
                self.live_reason = ""
                self.live_halt_until = None
            else:
                return True

        self.live_daily_pnl += live_pnl_delta
        self.live_total_capital = live_capital

        if live_capital > 0:
            daily_pct = self.live_daily_pnl / max(live_capital, 1)
            if daily_pct <= self.live_threshold:
                self.live_triggered = True
                self.live_halt_until = now + timedelta(hours=24)
                self.live_reason = f"Live daily PnL {daily_pct:.1%} <= {self.live_threshold:.1%}"
                logger.critical(f"LIVE CIRCUIT BREAKER TRIPPED: {self.live_reason}")
                self._save_state()
                return True
        return False

    def can_open_live(self, current_live_positions: int, position_capital: float) -> tuple[bool, str]:
        if self.live_triggered:
            return False, "Live circuit breaker is triggered"
        if current_live_positions >= self.max_concurrent_live:
            return False, f"Max concurrent live positions ({self.max_concurrent_live}) reached"
        if self.live_total_capital > 0:
            usage = position_capital / max(self.live_total_capital, 1)
            if usage > self.max_live_capital_pct:
                return False, f"Would exceed max live capital ({self.max_live_capital_pct:.0%})"
        return True, "OK"

    def reset_paper(self):
        self.paper_triggered = False
        self.paper_reason = ""
        self.paper_halt_until = None
        self._save_state()

    def reset_live(self):
        self.live_triggered = False
        self.live_reason = ""
        self.live_halt_until = None
        self.live_daily_pnl = 0.0
        self._save_state()

    def status(self) -> dict:
        return {
            "paper": {
                "triggered": self.paper_triggered,
                "reason": self.paper_reason,
                "halt_until": self.paper_halt_until.isoformat() if self.paper_halt_until else None,
            },
            "live": {
                "triggered": self.live_triggered,
                "reason": self.live_reason,
                "halt_until": self.live_halt_until.isoformat() if self.live_halt_until else None,
                "daily_pnl": round(self.live_daily_pnl, 2),
            },
        }

    def _save_state(self):
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(self.status(), indent=2))

    def _load_state(self):
        if _STATE_FILE.exists():
            try:
                data = json.loads(_STATE_FILE.read_text())
                self.paper_triggered = data.get("paper", {}).get("triggered", False)
                self.live_triggered = data.get("live", {}).get("triggered", False)
            except Exception:
                pass
