"""Aggregate performance tracking across all strategies."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_PERF_FILE = Path("learnings/performance.json")
_TRADES_FILE = Path("learnings/trades.jsonl")


def _load_perf() -> dict:
    if _PERF_FILE.exists():
        try:
            return json.loads(_PERF_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_perf(data: dict):
    _PERF_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PERF_FILE.write_text(json.dumps(data, indent=2))


def _compute_profit_factor() -> float:
    if not _TRADES_FILE.exists():
        return 0.0
    gross_win = 0.0
    gross_loss = 0.0
    try:
        for line in _TRADES_FILE.read_text().strip().split("\n"):
            if not line:
                continue
            trade = json.loads(line)
            pnl = trade.get("pnl_net", 0)
            if pnl > 0:
                gross_win += pnl
            else:
                gross_loss += abs(pnl)
    except Exception:
        pass
    return round(gross_win / max(gross_loss, 1e-9), 3)


def update(
    direction: str,
    outcome: str,
    pnl_net: float,
    price: float,
    current_balance: float,
    starting_balance: float,
    strategy_id: str = "",
    pair: str = "",
) -> dict:
    p = _load_perf()
    now = datetime.now(timezone.utc).isoformat()

    if "start_date" not in p:
        p["start_date"] = now[:10]
        p["starting_balance_usd"] = starting_balance

    p["current_balance_usd"] = current_balance
    p["total_trades"] = p.get("total_trades", 0) + 1

    if outcome in ("TP", "TP_MOONBAG"):
        p["winning_trades"] = p.get("winning_trades", 0) + 1
    elif outcome == "SL":
        p["losing_trades"] = p.get("losing_trades", 0) + 1

    total = p.get("total_trades", 1)
    wins = p.get("winning_trades", 0)
    p["win_rate"] = round(wins / max(total, 1) * 100, 1)
    p["total_pnl_usd"] = round(p.get("total_pnl_usd", 0) + pnl_net, 2)

    trade_record = {
        "ts": now, "direction": direction, "outcome": outcome,
        "pnl_net": round(pnl_net, 2), "price": price,
        "strategy_id": strategy_id, "pair": pair,
    }

    best = p.get("best_trade")
    if not best or pnl_net > best.get("pnl_net", 0):
        p["best_trade"] = trade_record

    worst = p.get("worst_trade")
    if not worst or pnl_net < worst.get("pnl_net", 0):
        p["worst_trade"] = trade_record

    p["profit_factor"] = _compute_profit_factor()
    _save_perf(p)
    return p


def get() -> dict:
    return _load_perf()
