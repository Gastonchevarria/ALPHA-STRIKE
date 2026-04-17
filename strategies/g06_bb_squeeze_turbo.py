"""G-06 BB Squeeze Turbo — evolved SB-8, Keltner + volume confirmation."""

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-06",
    name="BB Squeeze Turbo",
    timeframe="5m",
    leverage=35,
    margin_pct=0.08,
    tp_pct=0.006,
    sl_pct=0.003,
    cron_expr={"minute": "*/5"},
    regime_filter=["ANY"],
    description="Bollinger squeeze + Keltner breakout + volume. Evolution of SB-8.",
    timeout_minutes=120,
)


class G06BBSqueezeTurbo(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "5m", limit=60)

        df.ta.bbands(length=20, std=2, append=True)
        df.ta.kc(length=20, scalar=1.5, append=True)

        bbu_col = next((c for c in df.columns if c.startswith("BBU")), None)
        bbl_col = next((c for c in df.columns if c.startswith("BBL")), None)
        bbm_col = next((c for c in df.columns if c.startswith("BBM")), None)
        kcu_col = next((c for c in df.columns if c.startswith("KCU")), None)
        kcl_col = next((c for c in df.columns if c.startswith("KCL")), None)

        if not all([bbu_col, bbl_col, bbm_col, kcu_col, kcl_col]):
            return TradeSignal("HOLD", False, 0.3, {}, "Indicator data unavailable", pair)

        r = df.iloc[-1]
        p = df.iloc[-2]

        # Squeeze: BB inside KC
        bb_upper, bb_lower = r[bbu_col], r[bbl_col]
        kc_upper, kc_lower = r[kcu_col], r[kcl_col]
        prev_bb_upper, prev_bb_lower = p[bbu_col], p[bbl_col]
        prev_kc_upper, prev_kc_lower = p[kcu_col], p[kcl_col]

        was_squeezed = prev_bb_upper < prev_kc_upper and prev_bb_lower > prev_kc_lower
        is_released = bb_upper >= kc_upper or bb_lower <= kc_lower

        # Bandwidth percentile
        df["bb_width"] = (df[bbu_col] - df[bbl_col]) / df[bbm_col].replace(0, 1)
        width = df["bb_width"].iloc[-1]
        pctl = (df["bb_width"].iloc[-50:] < width).sum() / 50

        vol_ratio = r["volume"] / max(df["volume"].rolling(20).mean().iloc[-1], 1e-9)
        price = r["close"]

        signals = {
            "was_squeezed": was_squeezed,
            "is_released": is_released,
            "width_pctl": round(pctl, 2),
            "vol_ratio": round(vol_ratio, 2),
        }

        if not was_squeezed or not is_released:
            return TradeSignal("HOLD", False, 0.4, signals, "No squeeze release", pair)

        if vol_ratio < 1.3:
            return TradeSignal("HOLD", False, 0.4, signals, "Volume too low for squeeze breakout", pair)

        if price > kc_upper:
            conf = min(0.90, 0.74 + (1 - pctl) * 0.2 + (vol_ratio - 1) * 0.05)
            return TradeSignal("LONG", True, conf, signals, f"BB squeeze breakout UP", pair)

        if price < kc_lower:
            conf = min(0.90, 0.74 + (1 - pctl) * 0.2 + (vol_ratio - 1) * 0.05)
            return TradeSignal("SHORT", True, conf, signals, f"BB squeeze breakout DOWN", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "Squeeze released but no clear direction", pair)
