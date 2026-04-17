"""G-03 Order Flow Imbalance — taker buy/sell volume ratio."""

from core.data_fetcher import get_taker_volume
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-03",
    name="Order Flow Imbalance",
    timeframe="1m",
    leverage=45,
    margin_pct=0.08,
    tp_pct=0.0040,
    sl_pct=0.0022,
    cron_expr={"minute": "*/2"},
    regime_filter=["ANY"],
    description="Analyzes taker buy/sell volume ratio for directional imbalance.",
    timeout_minutes=30,
)


class G03OrderFlowImbalance(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        tv = await get_taker_volume(pair, "1m", limit=10)

        r = tv.iloc[-1]
        buy_ratio = r.get("buy_ratio", 0.5)
        candle_dir = 1 if r["close"] > r["open"] else -1
        avg_buy_ratio = tv["buy_ratio"].mean()

        signals = {
            "buy_ratio": round(buy_ratio, 4),
            "avg_buy_ratio": round(avg_buy_ratio, 4),
            "candle_dir": candle_dir,
        }

        if buy_ratio > 0.65 and candle_dir > 0:
            conf = min(0.90, 0.70 + (buy_ratio - 0.65) * 2)
            return TradeSignal("LONG", True, conf, signals, f"Buy imbalance {buy_ratio:.0%}", pair)

        if buy_ratio < 0.35 and candle_dir < 0:
            sell_ratio = 1 - buy_ratio
            conf = min(0.90, 0.70 + (sell_ratio - 0.65) * 2)
            return TradeSignal("SHORT", True, conf, signals, f"Sell imbalance {sell_ratio:.0%}", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No significant imbalance", pair)
