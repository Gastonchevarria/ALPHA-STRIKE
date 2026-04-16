"""Per-pair market regime detection with caching."""

import logging
import time

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv

logger = logging.getLogger(__name__)

_regime_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 300


def get_cached_regime(symbol: str = "BTCUSDT") -> dict:
    if symbol in _regime_cache:
        ts, regime = _regime_cache[symbol]
        if time.time() - ts < _CACHE_TTL:
            return regime
    return {"regime": "UNKNOWN", "symbol": symbol, "signals": {}}


async def detect_regime(symbol: str = "BTCUSDT") -> dict:
    try:
        df = await fetch_ohlcv(symbol, "15m", limit=60)

        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.bbands(length=20, std=2, append=True)

        r = df.iloc[-1]
        ema20 = r.get("EMA_20", 0)
        ema50 = r.get("EMA_50", 0)
        adx = r.get("ADX_14", 0)
        atr = r.get("ATRr_14", 0)

        bbu_col = next((c for c in df.columns if c.startswith("BBU")), None)
        bbl_col = next((c for c in df.columns if c.startswith("BBL")), None)
        bbm_col = next((c for c in df.columns if c.startswith("BBM")), None)

        bb_expansion = 0.0
        if bbu_col and bbl_col and bbm_col:
            df["bb_width"] = (df[bbu_col] - df[bbl_col]) / df[bbm_col].replace(0, 1)
            bb_expansion = df["bb_width"].iloc[-1] / max(df["bb_width"].iloc[-20:].mean(), 1e-9)

        atr_pct = atr / max(r["close"], 1) * 100 if atr else 0

        if ema20 > ema50 and adx > 25:
            regime = "TRENDING_UP"
        elif ema20 < ema50 and adx > 25:
            regime = "TRENDING_DOWN"
        elif atr_pct > 0.5 or bb_expansion > 1.5:
            regime = "VOLATILE"
        elif adx < 20:
            regime = "RANGING"
        else:
            regime = "RANGING"

        result = {
            "regime": regime,
            "symbol": symbol,
            "updated_at": time.time(),
            "signals": {
                "ema_delta": round(ema20 - ema50, 2),
                "adx": round(adx, 2),
                "atr_pct": round(atr_pct, 4),
                "bb_expansion": round(bb_expansion, 3),
            },
        }

        _regime_cache[symbol] = (time.time(), result)
        return result

    except Exception as e:
        logger.error(f"Regime detection failed for {symbol}: {e}")
        return get_cached_regime(symbol)


async def detect_all_regimes(pairs: list[str]) -> dict[str, dict]:
    import asyncio
    tasks = [detect_regime(p) for p in pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    regimes = {}
    for pair, result in zip(pairs, results):
        if isinstance(result, dict):
            regimes[pair] = result
        else:
            regimes[pair] = get_cached_regime(pair)
    return regimes


def strategy_matches_regime(regime_filter: list[str], regime: dict) -> bool:
    if "ANY" in regime_filter:
        return True
    return regime.get("regime", "UNKNOWN") in regime_filter
