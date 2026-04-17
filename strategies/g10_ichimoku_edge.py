"""G-10 Ichimoku Cloud Edge — Kumo breakout/bounce with Chikou confirmation."""

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-10",
    name="Ichimoku Cloud Edge",
    timeframe="15m",
    leverage=20,
    margin_pct=0.08,
    tp_pct=0.0065,
    sl_pct=0.0035,
    cron_expr={"minute": "*/15"},
    regime_filter=["TRENDING_UP", "TRENDING_DOWN"],
    description="Kumo breakout and bounce with Tenkan/Kijun/Chikou confirmation.",
    timeout_minutes=240,
)


class G10IchimokuEdge(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "15m", limit=80)

        df.ta.ichimoku(append=True)

        # Find Ichimoku columns
        tenkan = next((c for c in df.columns if "ISA" in c or "TENKAN" in c.upper()), None)
        kijun = next((c for c in df.columns if "ISB" in c or "KIJUN" in c.upper()), None)
        span_a = next((c for c in df.columns if "ISA" in c), None)
        span_b = next((c for c in df.columns if "ISB" in c), None)

        if not span_a or not span_b:
            return TradeSignal("HOLD", False, 0.3, {}, "Ichimoku data unavailable", pair)

        r = df.iloc[-1]
        price = r["close"]
        sa = r.get(span_a, 0)
        sb = r.get(span_b, 0)
        cloud_top = max(sa, sb)
        cloud_bottom = min(sa, sb)

        # Chikou span = current close vs price 26 periods ago
        chikou_bullish = len(df) > 26 and price > df.iloc[-27]["close"]
        chikou_bearish = len(df) > 26 and price < df.iloc[-27]["close"]

        # Tenkan > Kijun
        tk_bullish = tenkan and kijun and r.get(tenkan, 0) > r.get(kijun, 0)
        tk_bearish = tenkan and kijun and r.get(tenkan, 0) < r.get(kijun, 0)

        signals = {
            "price": price,
            "cloud_top": round(cloud_top, 2),
            "cloud_bottom": round(cloud_bottom, 2),
            "chikou_bullish": chikou_bullish,
            "tk_bullish": tk_bullish,
        }

        # Kumo breakout UP
        if price > cloud_top and tk_bullish and chikou_bullish:
            dist = (price - cloud_top) / cloud_top * 100
            conf = min(0.90, 0.76 + dist * 0.1)
            return TradeSignal("LONG", True, conf, signals, "Kumo breakout UP + TK + Chikou", pair)

        # Kumo breakout DOWN
        if price < cloud_bottom and tk_bearish and chikou_bearish:
            dist = (cloud_bottom - price) / cloud_bottom * 100
            conf = min(0.90, 0.76 + dist * 0.1)
            return TradeSignal("SHORT", True, conf, signals, "Kumo breakout DOWN + TK + Chikou", pair)

        # Kumo bounce: price near cloud edge in trend direction
        if abs(price - cloud_top) / cloud_top < 0.002 and tk_bullish:
            conf = 0.75
            return TradeSignal("LONG", True, conf, signals, "Kumo bounce at cloud top", pair)

        if abs(price - cloud_bottom) / cloud_bottom < 0.002 and tk_bearish:
            conf = 0.75
            return TradeSignal("SHORT", True, conf, signals, "Kumo bounce at cloud bottom", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No Ichimoku signal", pair)
