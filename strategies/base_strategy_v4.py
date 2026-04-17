"""Base strategy V4 — multi-pair, promotion-aware, Kelly sizing."""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.memory_tiers import add_memory

logger = logging.getLogger(__name__)

_PARAMS_DIR = Path("data/strategy_params")


class LTMParamStore:
    """Per-strategy persistent parameter tuning."""

    def __init__(self, strategy_id: str):
        self.strategy_id = strategy_id
        _PARAMS_DIR.mkdir(parents=True, exist_ok=True)
        self._path = _PARAMS_DIR / f"{strategy_id}_params.json"
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                return {"trades": 0, "params": {}}
        return {"trades": 0, "params": {}}

    def _save(self):
        self._path.write_text(json.dumps(self._data, indent=2))

    def get_param(self, key: str, default=None):
        return self._data.get("params", {}).get(key, default)

    def set_param(self, key: str, value, source: str = "auto"):
        self._data.setdefault("params", {})[key] = value
        self._save()

    def increment_trades(self):
        self._data["trades"] = self._data.get("trades", 0) + 1
        self._save()

    def total_trades(self) -> int:
        return self._data.get("trades", 0)

    def all_params(self) -> dict:
        return self._data.get("params", {})


@dataclass
class StrategyConfig:
    id: str
    name: str
    timeframe: str
    leverage: int
    margin_pct: float
    tp_pct: float
    sl_pct: float
    cron_expr: dict
    regime_filter: list[str]
    description: str
    timeout_minutes: int = 120
    kelly_enabled: bool = True


@dataclass
class TradeSignal:
    direction: str  # "LONG" | "SHORT" | "HOLD"
    execute: bool
    confidence: float
    signals: dict
    reason: str
    pair: str = ""


class BaseStrategyV4(ABC):
    """Multi-pair, promotion-aware base strategy."""

    KELLY_MIN_TRADES = 20
    KELLY_MAX_SCALE = 2.0
    KELLY_MIN_SCALE = 0.5
    FEE_RATE = 0.0008  # Taker fee round-trip

    def __init__(
        self,
        config: StrategyConfig,
        initial_balance: float = 1000.0,
        api_key: str = "",
        exec_model: str = "claude-opus-4-6",
        self_trainer_enabled: bool = True,
    ):
        self.cfg = config
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.position: str | None = None
        self.trade_log: list[dict] = []
        self.cycle_count = 0
        self.skip_count = 0

        # Promotion state
        self.phase = "PAPER"  # PAPER | SHADOW | LIVE
        self.live_balance = 0.0
        self.peak_balance = initial_balance

        # Eliminator state
        self.is_paused = False
        self.is_eliminated = False
        self.confidence_multiplier = 1.0

        # Position state
        self._entry_price = 0.0
        self._entry_dir = ""
        self._entry_pair = ""
        self._entry_sig: dict = {}
        self._entry_time: datetime | None = None
        self._entry_margin = 0.0

        # TP/SL absolute prices
        self.tp_abs = 0.0
        self.sl_abs = 0.0

        # Cooldown per pair: {pair: datetime_until}
        self._pair_cooldown: dict[str, datetime] = {}

        # LTM param store
        self.trainer = LTMParamStore(config.id)

        # Self-trainer (initialized externally by runner)
        self._self_trainer = None

    @abstractmethod
    async def evaluate(self, pair: str, df) -> TradeSignal:
        """Evaluate a single pair and return a trade signal."""
        ...

    def _effective_margin(self) -> float:
        """Calculate margin with optional Kelly sizing."""
        base = round(self.balance * self.cfg.margin_pct, 2)

        if not self.cfg.kelly_enabled:
            return base

        closed = [t for t in self.trade_log if t.get("action") == "CLOSE"]
        if len(closed) < self.KELLY_MIN_TRADES:
            return base

        wins = [t for t in closed if t.get("pnl_net", 0) > 0]
        losses = [t for t in closed if t.get("pnl_net", 0) < 0]

        if not wins or not losses:
            return base

        wr = len(wins) / len(closed)
        avg_win = sum(t["pnl_net"] for t in wins) / len(wins)
        avg_loss = abs(sum(t["pnl_net"] for t in losses) / len(losses))

        b = avg_win / max(avg_loss, 0.01)
        kelly = wr - (1 - wr) / b
        qk = max(0.0, kelly * 0.25)

        scale = self.KELLY_MIN_SCALE + (qk / 0.25) * (self.KELLY_MAX_SCALE - self.KELLY_MIN_SCALE)
        scale = max(self.KELLY_MIN_SCALE, min(self.KELLY_MAX_SCALE, scale))

        return round(self.balance * self.cfg.margin_pct * scale, 2)

    def is_pair_on_cooldown(self, pair: str) -> bool:
        """Check if a pair is on cooldown for this strategy."""
        if pair in self._pair_cooldown:
            if datetime.now(timezone.utc) < self._pair_cooldown[pair]:
                return True
            del self._pair_cooldown[pair]
        return False

    def open_position(self, direction: str, price: float, pair: str, signals: dict):
        """Open a new position."""
        margin = self._effective_margin()
        notional = margin * self.cfg.leverage
        mult = 1 if direction == "LONG" else -1

        self.tp_abs = price * (1 + self.cfg.tp_pct * mult)
        self.sl_abs = price * (1 - self.cfg.sl_pct * mult)
        self.position = direction

        self._entry_price = price
        self._entry_dir = direction
        self._entry_pair = pair
        self._entry_sig = signals
        self._entry_time = datetime.now(timezone.utc)
        self._entry_margin = margin

        self._log("OPEN", direction, price, None, None, signals, margin, pair)

    def check_exit(self, price: float) -> str | None:
        """Check if position should be closed. Returns reason or None."""
        if not self.position:
            return None

        if self.position == "LONG":
            if price >= self.tp_abs:
                return "TP"
            if price <= self.sl_abs:
                return "SL"
        else:
            if price <= self.tp_abs:
                return "TP"
            if price >= self.sl_abs:
                return "SL"

        if self._entry_time:
            elapsed_min = (datetime.now(timezone.utc) - self._entry_time).total_seconds() / 60
            if elapsed_min >= self.cfg.timeout_minutes:
                return "TIMEOUT"
        return None

    def close_position(self, price: float, reason: str) -> tuple[float, dict]:
        """Close position and return (pnl, trade_record)."""
        margin = self._entry_margin or self._effective_margin()
        notional = margin * self.cfg.leverage
        fee_rt = notional * self.FEE_RATE
        mult = 1 if self._entry_dir == "LONG" else -1
        price_pct = (price - self._entry_price) / self._entry_price * mult
        pnl = notional * price_pct - fee_rt

        self.balance += pnl
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        now = datetime.now(timezone.utc)
        record = {
            "ts": now.isoformat(),
            "action": "CLOSE",
            "direction": self._entry_dir,
            "pair": self._entry_pair,
            "reason": reason,
            "pnl_net": round(pnl, 2),
            "price": price,
            "entry_price": self._entry_price,
            "balance": round(self.balance, 2),
            "signals": self._entry_sig,
        }

        self._log("CLOSE", self._entry_dir, price, pnl, reason, self._entry_sig, margin, self._entry_pair)
        self.trainer.increment_trades()

        # Set cooldown on pair
        cooldown_candles = 3
        candle_minutes = {"1m": 1, "5m": 5, "15m": 15}.get(self.cfg.timeframe, 5)
        self._pair_cooldown[self._entry_pair] = now + timedelta(minutes=cooldown_candles * candle_minutes)

        self.position = None
        self._entry_time = None
        return pnl, record

    def _log(self, action, direction, price, pnl, reason, signals, margin, pair=""):
        self.trade_log.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "direction": direction,
            "pair": pair,
            "price": price,
            "pnl_net": round(pnl, 2) if pnl is not None else None,
            "reason": reason,
            "signals": signals,
            "margin": round(margin, 2),
            "balance": round(self.balance, 2),
        })

    def stats(self) -> dict:
        """Return strategy statistics."""
        closed = [t for t in self.trade_log if t.get("action") == "CLOSE"]
        wins = [t for t in closed if t.get("pnl_net", 0) > 0]
        losses = [t for t in closed if t.get("pnl_net", 0) < 0]
        gross_win = sum(t["pnl_net"] for t in wins)
        gross_loss = abs(sum(t["pnl_net"] for t in losses)) or 1e-9

        drawdown = 0.0
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - self.balance) / self.peak_balance

        return {
            "id": self.cfg.id,
            "name": self.cfg.name,
            "timeframe": self.cfg.timeframe,
            "leverage": self.cfg.leverage,
            "phase": self.phase,
            "balance": round(self.balance, 2),
            "live_balance": round(self.live_balance, 2),
            "trades": len(closed),
            "win_rate": round(len(wins) / max(len(closed), 1) * 100, 1),
            "profit_factor": round(gross_win / gross_loss, 3),
            "max_drawdown": round(drawdown, 4),
            "is_paused": self.is_paused,
            "is_eliminated": self.is_eliminated,
            "conf_multiplier": round(self.confidence_multiplier, 2),
            "ready_to_rank": len(closed) >= 20,
            "open_position": {
                "direction": self.position,
                "pair": self._entry_pair,
                "entry_price": self._entry_price,
                "tp": self.tp_abs,
                "sl": self.sl_abs,
                "entry_time": self._entry_time.isoformat() if self._entry_time else None,
            } if self.position else None,
            "cycle_count": self.cycle_count,
            "skip_count": self.skip_count,
        }

    async def trigger_self_trainer(self, trade: dict, regime: str):
        """Trigger self-trainer analysis after a trade close."""
        if self._self_trainer:
            await self._self_trainer.analyze(
                trade,
                trade.get("signals", {}),
                f"Regime: {regime}, Pair: {trade.get('pair', 'unknown')}",
                self.trainer,
            )
