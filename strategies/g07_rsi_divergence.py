"""G-07 RSI Divergence Hunter — classic RSI divergences on 15m."""

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-07",
    name="RSI Divergence Hunter",
    timeframe="15m",
    leverage=20,
    margin_pct=0.08,
    tp_pct=0.007,
    sl_pct=0.004,
    cron_expr={"minute": "*/15"},
    regime_filter=["ANY"],
    description="Detects classic bullish/bearish RSI divergences with pivot confirmation.",
    timeout_minutes=240,
)


def _find_pivots(series, window=5):
    """Find local highs and lows in a series."""
    highs = []
    lows = []
    for i in range(window, len(series) - window):
        if series.iloc[i] == max(series.iloc[i - window:i + window + 1]):
            highs.append((i, series.iloc[i]))
        if series.iloc[i] == min(series.iloc[i - window:i + window + 1]):
            lows.append((i, series.iloc[i]))
    return highs, lows


class G07RSIDivergence(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "15m", limit=60)

        df.ta.rsi(length=14, append=True)
        rsi_col = "RSI_14"

        if rsi_col not in df.columns:
            return TradeSignal("HOLD", False, 0.3, {}, "RSI unavailable", pair)

        price_highs, price_lows = _find_pivots(df["close"], window=3)
        rsi_highs, rsi_lows = _find_pivots(df[rsi_col].dropna(), window=3)

        signals = {"rsi": round(df[rsi_col].iloc[-1], 2), "price_lows": len(price_lows), "price_highs": len(price_highs)}

        # Bullish divergence: price lower low + RSI higher low
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            p1, p2 = price_lows[-2], price_lows[-1]
            r1, r2 = rsi_lows[-2], rsi_lows[-1]
            if p2[1] < p1[1] and r2[1] > r1[1]:
                conf = min(0.88, 0.75 + (r2[1] - r1[1]) * 0.003)
                return TradeSignal("LONG", True, conf, signals, "Bullish RSI divergence", pair)

        # Bearish divergence: price higher high + RSI lower high
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            p1, p2 = price_highs[-2], price_highs[-1]
            r1, r2 = rsi_highs[-2], rsi_highs[-1]
            if p2[1] > p1[1] and r2[1] < r1[1]:
                conf = min(0.88, 0.75 + (r1[1] - r2[1]) * 0.003)
                return TradeSignal("SHORT", True, conf, signals, "Bearish RSI divergence", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No RSI divergence detected", pair)
