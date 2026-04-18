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
    description="GOD 2 exclusive: mean reversion when altcoin diverges from anchor pair.",
    timeout_minutes=120,
)

# Anchor pair: the most liquid pair in the tournament, used as reference for divergences
ANCHOR_PAIR = "ORDIUSDT"


class G12CrossPairDivergence(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        # This strategy doesn't trade the anchor pair itself
        if pair == ANCHOR_PAIR:
            return TradeSignal("HOLD", False, 0.0, {}, f"G-12 only trades pairs other than anchor {ANCHOR_PAIR}", pair)

        corr_data = get_correlation_matrix()
        matrix = corr_data.get("matrix", {})

        if ANCHOR_PAIR not in matrix or pair not in matrix.get(ANCHOR_PAIR, {}):
            return TradeSignal("HOLD", False, 0.3, {}, "Correlation data unavailable", pair)

        anchor_corr = matrix[ANCHOR_PAIR].get(pair, 0.5)

        # Get recent returns for both
        anchor_df = await fetch_ohlcv(ANCHOR_PAIR, "5m", limit=5)
        alt_df = await fetch_ohlcv(pair, "5m", limit=5) if df is None else df

        anchor_ret = (anchor_df.iloc[-1]["close"] - anchor_df.iloc[0]["open"]) / anchor_df.iloc[0]["open"] * 100
        alt_ret = (alt_df.iloc[-1]["close"] - alt_df.iloc[0]["open"]) / alt_df.iloc[0]["open"] * 100
        divergence = alt_ret - anchor_ret

        signals = {
            "anchor_pair": ANCHOR_PAIR,
            "anchor_corr": round(anchor_corr, 4),
            "anchor_ret": round(anchor_ret, 4),
            "alt_ret": round(alt_ret, 4),
            "divergence": round(divergence, 4),
        }

        # Only trade when correlation is normally high but currently diverging
        if anchor_corr < 0.6:
            return TradeSignal("HOLD", False, 0.3, signals, "Correlation too low to trade divergence", pair)

        # Altcoin lagging anchor significantly → expect mean reversion (altcoin catches up)
        if anchor_ret > 0.3 and divergence < -0.5:
            conf = min(0.88, 0.72 + abs(divergence) * 0.1 + anchor_corr * 0.1)
            return TradeSignal("LONG", True, conf, signals, f"{pair} lagging {ANCHOR_PAIR} by {divergence:.2f}%", pair)

        # Altcoin overperforming anchor significantly → expect mean reversion (altcoin pulls back)
        if anchor_ret < -0.3 and divergence > 0.5:
            conf = min(0.88, 0.72 + abs(divergence) * 0.1 + anchor_corr * 0.1)
            return TradeSignal("SHORT", True, conf, signals, f"{pair} overperforming {ANCHOR_PAIR} by {divergence:.2f}%", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No significant cross-pair divergence", pair)
