"""G-12 Cross-Pair Divergence — GOD 2 exclusive, mean reversion on correlation breaks."""

from core.correlation_engine import get_correlation_matrix
from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-12",
    name="Cross-Pair Divergence",
    timeframe="5m",
    leverage=25,
    margin_pct=0.08,
    tp_pct=0.006,
    sl_pct=0.0032,
    cron_expr={"minute": "*/5"},
    regime_filter=["ANY"],
    description="GOD 2 exclusive: mean reversion when altcoin diverges from BTC.",
    timeout_minutes=120,
)


class G12CrossPairDivergence(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        # This strategy only trades altcoins, not BTC itself
        if pair == "BTCUSDT":
            return TradeSignal("HOLD", False, 0.0, {}, "G-12 only trades altcoins", pair)

        corr_data = get_correlation_matrix()
        matrix = corr_data.get("matrix", {})

        if "BTCUSDT" not in matrix or pair not in matrix.get("BTCUSDT", {}):
            return TradeSignal("HOLD", False, 0.3, {}, "Correlation data unavailable", pair)

        btc_corr = matrix["BTCUSDT"].get(pair, 0.5)

        # Get recent returns for both
        btc_df = await fetch_ohlcv("BTCUSDT", "5m", limit=5)
        alt_df = await fetch_ohlcv(pair, "5m", limit=5) if df is None else df

        btc_ret = (btc_df.iloc[-1]["close"] - btc_df.iloc[0]["open"]) / btc_df.iloc[0]["open"] * 100
        alt_ret = (alt_df.iloc[-1]["close"] - alt_df.iloc[0]["open"]) / alt_df.iloc[0]["open"] * 100
        divergence = alt_ret - btc_ret

        signals = {
            "btc_corr": round(btc_corr, 4),
            "btc_ret": round(btc_ret, 4),
            "alt_ret": round(alt_ret, 4),
            "divergence": round(divergence, 4),
        }

        # Only trade when correlation is normally high but currently diverging
        if btc_corr < 0.6:
            return TradeSignal("HOLD", False, 0.3, signals, "Correlation too low to trade divergence", pair)

        # Altcoin lagging BTC significantly → expect mean reversion (altcoin catches up)
        if btc_ret > 0.3 and divergence < -0.5:
            conf = min(0.88, 0.72 + abs(divergence) * 0.1 + btc_corr * 0.1)
            return TradeSignal("LONG", True, conf, signals, f"{pair} lagging BTC by {divergence:.2f}%", pair)

        # Altcoin overperforming BTC significantly → expect mean reversion (altcoin pulls back)
        if btc_ret < -0.3 and divergence > 0.5:
            conf = min(0.88, 0.72 + abs(divergence) * 0.1 + btc_corr * 0.1)
            return TradeSignal("SHORT", True, conf, signals, f"{pair} overperforming BTC by {divergence:.2f}%", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No significant cross-pair divergence", pair)
