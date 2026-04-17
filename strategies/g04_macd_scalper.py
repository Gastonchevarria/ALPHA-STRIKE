"""G-04 MACD Scalper — histogram zero-line crosses + EMA confirmation."""

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-04",
    name="MACD Scalper",
    timeframe="5m",
    leverage=30,
    margin_pct=0.08,
    tp_pct=0.005,
    sl_pct=0.003,
    cron_expr={"minute": "*/5"},
    regime_filter=["TRENDING_UP", "TRENDING_DOWN"],
    description="MACD histogram zero-cross + EMA(9) direction + ADX filter.",
    timeout_minutes=120,
)


class G04MACDScalper(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "5m", limit=50)

        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.ema(length=9, append=True)
        df.ta.adx(length=14, append=True)

        r = df.iloc[-1]
        p = df.iloc[-2]

        hist_col = next((c for c in df.columns if "MACDh" in c), None)
        hist = r.get(hist_col, 0) if hist_col else 0
        prev_hist = p.get(hist_col, 0) if hist_col else 0
        ema9 = r.get("EMA_9", 0)
        adx = r.get("ADX_14", 0)
        price = r["close"]

        signals = {"macd_hist": round(hist, 4), "ema9": round(ema9, 2), "adx": round(adx, 2)}

        if adx < 20:
            return TradeSignal("HOLD", False, 0.3, signals, "ADX too low", pair)

        # Histogram crosses above zero + price above EMA9
        if prev_hist <= 0 < hist and price > ema9:
            conf = min(0.88, 0.72 + abs(hist) * 50 + (adx - 20) * 0.003)
            return TradeSignal("LONG", True, conf, signals, f"MACD hist cross up, ADX={adx:.0f}", pair)

        # Histogram crosses below zero + price below EMA9
        if prev_hist >= 0 > hist and price < ema9:
            conf = min(0.88, 0.72 + abs(hist) * 50 + (adx - 20) * 0.003)
            return TradeSignal("SHORT", True, conf, signals, f"MACD hist cross down, ADX={adx:.0f}", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No MACD cross", pair)
