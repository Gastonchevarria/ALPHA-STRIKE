"""G-08 VWAP Sniper — evolved SA-4, VWAP bounce with dynamic σ bands."""

import numpy as np
import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-08",
    name="VWAP Sniper",
    timeframe="5m",
    leverage=30,
    margin_pct=0.08,
    tp_pct=0.005,
    sl_pct=0.0028,
    cron_expr={"minute": "*/5"},
    regime_filter=["RANGING"],
    description="VWAP bounce with ±1σ/±2σ dynamic TP targets. Evolution of SA-4.",
    timeout_minutes=120,
)


class G08VWAPSniper(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "5m", limit=50)

        df.ta.rsi(length=14, append=True)

        # Calculate VWAP and standard deviation bands
        typical = (df["high"] + df["low"] + df["close"]) / 3
        cum_vol = df["volume"].cumsum()
        cum_tp_vol = (typical * df["volume"]).cumsum()
        vwap = cum_tp_vol / cum_vol.replace(0, 1)

        # Standard deviation bands
        df["vwap"] = vwap
        sq_diff = ((typical - vwap) ** 2 * df["volume"]).cumsum()
        std = np.sqrt(sq_diff / cum_vol.replace(0, 1))
        df["vwap_1up"] = vwap + std
        df["vwap_1dn"] = vwap - std
        df["vwap_2up"] = vwap + 2 * std
        df["vwap_2dn"] = vwap - 2 * std

        r = df.iloc[-1]
        price = r["close"]
        rsi = r.get("RSI_14", 50)
        vwap_val = r["vwap"]
        dist_to_vwap = (price - vwap_val) / vwap_val * 100

        signals = {
            "vwap": round(vwap_val, 2),
            "dist_pct": round(dist_to_vwap, 4),
            "rsi": round(rsi, 2),
        }

        # Long: price touches VWAP from above, RSI not overbought
        if abs(dist_to_vwap) < 0.05 and price > vwap_val and rsi > 40 and rsi < 60:
            conf = min(0.88, 0.72 + (1 - abs(dist_to_vwap)) * 0.1)
            return TradeSignal("LONG", True, conf, signals, "VWAP bounce LONG", pair)

        # Short: price touches VWAP from below, RSI not oversold
        if abs(dist_to_vwap) < 0.05 and price < vwap_val and rsi > 40 and rsi < 60:
            conf = min(0.88, 0.72 + (1 - abs(dist_to_vwap)) * 0.1)
            return TradeSignal("SHORT", True, conf, signals, "VWAP rejection SHORT", pair)

        # Reversal at 2σ band
        if price <= r["vwap_2dn"] and rsi < 35:
            conf = 0.85
            return TradeSignal("LONG", True, conf, signals, "VWAP -2σ reversal LONG", pair)

        if price >= r["vwap_2up"] and rsi > 65:
            conf = 0.85
            return TradeSignal("SHORT", True, conf, signals, "VWAP +2σ reversal SHORT", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No VWAP signal", pair)
