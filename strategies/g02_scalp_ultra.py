"""G-02 Scalp Ultra — evolved S-02, RSI extremes + BB touch + spread filter."""

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-02",
    name="Scalp Ultra",
    timeframe="1m",
    leverage=35,
    margin_pct=0.08,
    tp_pct=0.0030,
    sl_pct=0.0018,
    cron_expr={"minute": "*/1"},
    regime_filter=["RANGING", "VOLATILE"],
    description="RSI(7) extremes + volume + BB touch. Evolution of S-02.",
    timeout_minutes=30,
)


class G02ScalpUltra(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "1m", limit=30)

        df.ta.rsi(length=7, append=True)
        df.ta.bbands(length=20, std=2, append=True)

        r = df.iloc[-1]
        rsi = r.get("RSI_7", 50)
        vol_ratio = r["volume"] / max(df["volume"].rolling(20).mean().iloc[-1], 1e-9)

        bbl_col = next((c for c in df.columns if c.startswith("BBL")), None)
        bbu_col = next((c for c in df.columns if c.startswith("BBU")), None)
        price = r["close"]

        touching_lower = bbl_col and price <= r[bbl_col] * 1.001
        touching_upper = bbu_col and price >= r[bbu_col] * 0.999

        # Spread filter: reject if high-low > 0.1% of close
        spread = (r["high"] - r["low"]) / max(price, 1) * 100
        signals = {"rsi_7": round(rsi, 2), "vol_ratio": round(vol_ratio, 2), "spread": round(spread, 4)}

        if vol_ratio < 1.2:
            return TradeSignal("HOLD", False, 0.3, signals, "Volume too low", pair)

        if spread > 0.1:
            return TradeSignal("HOLD", False, 0.3, signals, "Spread too wide", pair)

        if rsi < 25 and touching_lower:
            conf = min(0.92, 0.70 + (30 - rsi) * 0.01 + (vol_ratio - 1) * 0.05)
            return TradeSignal("LONG", True, conf, signals, f"RSI oversold + BB lower touch", pair)

        if rsi > 75 and touching_upper:
            conf = min(0.92, 0.70 + (rsi - 70) * 0.01 + (vol_ratio - 1) * 0.05)
            return TradeSignal("SHORT", True, conf, signals, f"RSI overbought + BB upper touch", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "RSI not extreme enough", pair)
