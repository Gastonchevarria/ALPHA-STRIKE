"""G-13 Volume Delta Sniper — divergence between price and cumulative volume delta."""

from core.data_fetcher import get_taker_volume
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-13",
    name="Volume Delta Sniper",
    timeframe="1m",
    leverage=50,
    margin_pct=0.08,
    tp_pct=0.0045,
    sl_pct=0.0025,
    cron_expr={"minute": "*/2"},
    regime_filter=["ANY"],
    description="Detects hidden accumulation/distribution via volume delta divergence.",
    timeout_minutes=30,
)


class G13VolumeDeltaSniper(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        tv = await get_taker_volume(pair, "1m", limit=10)

        # Cumulative volume delta over last 5 candles
        last5 = tv.tail(5)
        delta = (last5["taker_buy_vol"] - last5["taker_sell_vol"]).sum()
        price_change = (last5.iloc[-1]["close"] - last5.iloc[0]["open"]) / last5.iloc[0]["open"] * 100

        signals = {
            "cum_delta": round(delta, 2),
            "price_change_pct": round(price_change, 4),
        }

        # Divergence: price drops but delta positive = hidden accumulation
        if price_change < -0.05 and delta > 0:
            strength = min(abs(delta) / max(last5["volume"].mean(), 1), 1.0)
            conf = min(0.90, 0.72 + strength * 0.15)
            return TradeSignal("LONG", True, conf, signals, f"Hidden accumulation, delta={delta:.0f}", pair)

        # Divergence: price rises but delta negative = distribution
        if price_change > 0.05 and delta < 0:
            strength = min(abs(delta) / max(last5["volume"].mean(), 1), 1.0)
            conf = min(0.90, 0.72 + strength * 0.15)
            return TradeSignal("SHORT", True, conf, signals, f"Distribution detected, delta={delta:.0f}", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No volume delta divergence", pair)
