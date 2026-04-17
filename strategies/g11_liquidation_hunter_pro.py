"""G-11 Liquidation Hunter Pro — evolved SA-2, OI drops + price cascade."""

from core.data_fetcher import fetch_ohlcv, get_open_interest
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-11",
    name="Liquidation Hunter Pro",
    timeframe="5m",
    leverage=40,
    margin_pct=0.08,
    tp_pct=0.005,
    sl_pct=0.0025,
    cron_expr={"minute": "*/5"},
    regime_filter=["VOLATILE"],
    description="Detects liquidation cascades via OI drops + violent price moves.",
    timeout_minutes=120,
)


class G11LiquidationHunterPro(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)
        self._prev_oi: dict[str, float] = {}

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "5m", limit=10)

        current_oi = await get_open_interest(pair)
        if current_oi is None:
            return TradeSignal("HOLD", False, 0.3, {}, "OI data unavailable", pair)

        prev_oi = self._prev_oi.get(pair, current_oi)
        oi_change_pct = (current_oi - prev_oi) / max(prev_oi, 1) * 100
        self._prev_oi[pair] = current_oi

        # Price movement over last 5 candles
        last5 = df.tail(5)
        price_change = (last5.iloc[-1]["close"] - last5.iloc[0]["open"]) / last5.iloc[0]["open"] * 100
        vol_spike = last5.iloc[-1]["volume"] / max(last5["volume"].mean(), 1e-9)

        signals = {
            "oi_change_pct": round(oi_change_pct, 2),
            "price_change_pct": round(price_change, 4),
            "vol_spike": round(vol_spike, 2),
        }

        # Liquidation cascade: OI drops sharply + violent price move
        if oi_change_pct < -3 and vol_spike > 1.5:
            if price_change > 0.3:
                conf = min(0.90, 0.74 + abs(oi_change_pct) * 0.02 + vol_spike * 0.03)
                return TradeSignal("LONG", True, conf, signals, f"Liq cascade UP, OI drop {oi_change_pct:.1f}%", pair)
            if price_change < -0.3:
                conf = min(0.90, 0.74 + abs(oi_change_pct) * 0.02 + vol_spike * 0.03)
                return TradeSignal("SHORT", True, conf, signals, f"Liq cascade DOWN, OI drop {oi_change_pct:.1f}%", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No liquidation cascade detected", pair)
