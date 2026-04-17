"""G-09 ATR Breakout Rider — ATR expansion detection with trailing stop."""

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-09",
    name="ATR Breakout Rider",
    timeframe="5m",
    leverage=30,
    margin_pct=0.08,
    tp_pct=0.0055,
    sl_pct=0.003,
    cron_expr={"minute": "*/5"},
    regime_filter=["VOLATILE"],
    description="Detects ATR expansion for breakout riding with ATR trailing stop.",
    timeout_minutes=120,
)


class G09ATRBreakout(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)
        self._trailing_atr = 0.0

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "5m", limit=30)

        df.ta.atr(length=14, append=True)
        df.ta.ema(length=20, append=True)

        r = df.iloc[-1]
        atr_col = next((c for c in df.columns if c.startswith("ATR")), None)
        if not atr_col:
            return TradeSignal("HOLD", False, 0.3, {}, "ATR unavailable", pair)

        atr = r[atr_col]
        atr_avg = df[atr_col].rolling(20).mean().iloc[-1]
        atr_ratio = atr / max(atr_avg, 1e-9)
        ema20 = r.get("EMA_20", r["close"])
        price = r["close"]

        signals = {"atr": round(atr, 2), "atr_ratio": round(atr_ratio, 2), "ema20": round(ema20, 2)}

        if atr_ratio < 1.5:
            return TradeSignal("HOLD", False, 0.35, signals, "ATR not expanding enough", pair)

        self._trailing_atr = atr * 1.5

        if price > ema20:
            conf = min(0.88, 0.72 + (atr_ratio - 1.5) * 0.1)
            return TradeSignal("LONG", True, conf, signals, f"ATR breakout UP, ratio={atr_ratio:.1f}x", pair)

        if price < ema20:
            conf = min(0.88, 0.72 + (atr_ratio - 1.5) * 0.1)
            return TradeSignal("SHORT", True, conf, signals, f"ATR breakout DOWN, ratio={atr_ratio:.1f}x", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "ATR expanding but no direction", pair)
