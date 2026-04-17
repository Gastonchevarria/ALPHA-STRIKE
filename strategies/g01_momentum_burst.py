"""G-01 Momentum Burst — 1m ROC + Volume Spike detection."""

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-01",
    name="Momentum Burst",
    timeframe="1m",
    leverage=40,
    margin_pct=0.08,
    tp_pct=0.0035,
    sl_pct=0.0020,
    cron_expr={"minute": "*/2"},
    regime_filter=["ANY"],
    description="Detects sudden momentum explosions using ROC + volume spike.",
    timeout_minutes=30,
)


class G01MomentumBurst(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "1m", limit=20)

        df.ta.roc(length=3, append=True)
        vol_ma = df["volume"].rolling(10).mean()
        df["vol_ratio"] = df["volume"] / vol_ma.replace(0, 1)

        signals = {}
        r = df.iloc[-1]
        roc = r.get("ROC_3", 0)
        vol_ratio = r.get("vol_ratio", 0)
        signals = {"roc_3": round(roc, 4), "vol_ratio": round(vol_ratio, 2)}

        # Check 3 consecutive candles with positive/negative ROC
        last3_roc = [df.iloc[i].get("ROC_3", 0) for i in range(-3, 0)]
        all_positive = all(r > 0.1 for r in last3_roc)
        all_negative = all(r < -0.1 for r in last3_roc)

        if vol_ratio < 2.0:
            return TradeSignal("HOLD", False, 0.3, signals, "Volume too low", pair)

        if all_positive and roc > 0.15:
            conf = min(0.92, 0.70 + roc * 0.5 + (vol_ratio - 2) * 0.05)
            return TradeSignal("LONG", True, conf, signals, f"Momentum burst UP, ROC={roc:.2f}", pair)

        if all_negative and roc < -0.15:
            conf = min(0.92, 0.70 + abs(roc) * 0.5 + (vol_ratio - 2) * 0.05)
            return TradeSignal("SHORT", True, conf, signals, f"Momentum burst DOWN, ROC={roc:.2f}", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No momentum burst detected", pair)
