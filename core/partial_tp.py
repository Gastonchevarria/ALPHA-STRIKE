"""Partial take-profit system (disabled by default, available for live)."""

from dataclasses import dataclass


@dataclass
class PartialTPConfig:
    tp1_pct: float = 0.25
    tp2_pct: float = 0.50
    tp3_pct: float = 0.75
    moonbag_trail: float = 0.60
    enabled: bool = False


@dataclass
class PartialTPState:
    entry_price: float = 0.0
    direction: str = ""
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    peak_pnl: float = 0.0
    closed_pct: float = 0.0

    def check(self, current_pnl: float, tp_target: float, config: PartialTPConfig):
        if not config.enabled:
            return None

        if current_pnl > self.peak_pnl:
            self.peak_pnl = current_pnl

        if not self.tp1_hit and current_pnl >= tp_target * 0.5:
            self.tp1_hit = True
            self.closed_pct = config.tp1_pct
            return ("PARTIAL_CLOSE", config.tp1_pct)

        if not self.tp2_hit and current_pnl >= tp_target * 0.75:
            self.tp2_hit = True
            self.closed_pct = config.tp2_pct
            return ("PARTIAL_CLOSE", config.tp2_pct - config.tp1_pct)

        if not self.tp3_hit and current_pnl >= tp_target:
            self.tp3_hit = True
            self.closed_pct = config.tp3_pct
            return ("PARTIAL_CLOSE", config.tp3_pct - config.tp2_pct)

        if self.tp3_hit and self.peak_pnl > 0:
            if current_pnl < self.peak_pnl * config.moonbag_trail:
                return ("CLOSE_ALL", 1.0 - self.closed_pct)

        return None
