"""G-05 Stochastic Reversal — %K/%D crosses at extreme zones."""

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-05",
    name="Stochastic Reversal",
    timeframe="5m",
    leverage=25,
    margin_pct=0.08,
    tp_pct=0.0055,
    sl_pct=0.003,
    cron_expr={"minute": "*/5"},
    regime_filter=["RANGING"],
    description="Stochastic %K/%D crosses in oversold/overbought zones.",
    timeout_minutes=120,
)


class G05StochasticReversal(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "5m", limit=30)

        df.ta.stoch(k=14, d=3, append=True)

        k_col = next((c for c in df.columns if c.startswith("STOCHk")), None)
        d_col = next((c for c in df.columns if c.startswith("STOCHd")), None)

        if not k_col or not d_col:
            return TradeSignal("HOLD", False, 0.3, {}, "Stochastic data unavailable", pair)

        r = df.iloc[-1]
        p = df.iloc[-2]
        k = r[k_col]
        d = r[d_col]
        prev_k = p[k_col]
        prev_d = p[d_col]

        # Candle confirmation
        candle_bullish = r["close"] > r["open"]
        candle_bearish = r["close"] < r["open"]

        signals = {"stoch_k": round(k, 2), "stoch_d": round(d, 2)}

        # Bullish: K crosses above D in oversold zone
        if k < 20 and prev_k <= prev_d and k > d and candle_bullish:
            conf = min(0.88, 0.72 + (20 - k) * 0.005)
            return TradeSignal("LONG", True, conf, signals, f"Stoch bullish cross in oversold, K={k:.0f}", pair)

        # Bearish: K crosses below D in overbought zone
        if k > 80 and prev_k >= prev_d and k < d and candle_bearish:
            conf = min(0.88, 0.72 + (k - 80) * 0.005)
            return TradeSignal("SHORT", True, conf, signals, f"Stoch bearish cross in overbought, K={k:.0f}", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No stochastic reversal signal", pair)
