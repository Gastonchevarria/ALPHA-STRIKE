"""Position and risk management for Agent GOD 2."""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    direction: str
    entry_price: float
    entry_time: datetime
    notional: float
    margin: float
    leverage: int
    tp_price: float
    sl_price: float

    def unrealised_pnl(self, price: float) -> float:
        mult = 1 if self.direction == "LONG" else -1
        pct = (price - self.entry_price) / self.entry_price * mult
        fee = self.notional * 0.0008
        return self.notional * pct - fee

    def should_tp(self, price: float) -> bool:
        if self.direction == "LONG":
            return price >= self.tp_price
        return price <= self.tp_price

    def should_sl(self, price: float) -> bool:
        if self.direction == "LONG":
            return price <= self.sl_price
        return price >= self.sl_price

    def distance_to_tp(self, price: float) -> float:
        return abs(self.tp_price - price) / price * 100

    def distance_to_sl(self, price: float) -> float:
        return abs(self.sl_price - price) / price * 100
