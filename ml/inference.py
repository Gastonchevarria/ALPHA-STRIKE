"""Runtime inference API used by tournament runner."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from config.settings import settings
from core.data_fetcher import fetch_ohlcv, get_taker_volume
from ml.ev_model import predict_ev
from ml.features import compute_features_from_df, enrich_cross_pair
from ml.regime_classifier import predict_regime
from ml.volatility_predictor import predict_volatility

logger = logging.getLogger(__name__)

_cache: dict[str, tuple[float, Any]] = {}
_TTL = settings.ML_INFERENCE_CACHE_SECONDS


def _cached(key: str):
    if key in _cache:
        ts, val = _cache[key]
        if time.time() - ts < _TTL:
            return val
    return None


def _set_cache(key: str, val: Any):
    _cache[key] = (time.time(), val)


async def _get_features(pair: str, timeframe: str = "5m") -> dict:
    key = f"features:{pair}:{timeframe}"
    cached = _cached(key)
    if cached is not None:
        return cached

    df = await fetch_ohlcv(pair, timeframe, limit=100)
    try:
        tv = await get_taker_volume(pair, timeframe, limit=len(df))
        df["taker_buy_base"] = tv["taker_buy_vol"].values
    except Exception:
        df["taker_buy_base"] = df["volume"] / 2

    features = compute_features_from_df(df)
    _set_cache(key, features)
    return features


async def get_regime_prediction(pair: str) -> dict:
    if not settings.ML_ENABLED:
        return {"regime": "UNKNOWN", "confidence": 0.0}

    key = f"regime:{pair}"
    cached = _cached(key)
    if cached is not None:
        return cached

    features = await _get_features(pair, "5m")
    result = predict_regime(features)
    _set_cache(key, result)
    return result


async def get_expected_value(strategy_id: str, pair: str) -> dict:
    if not settings.ML_ENABLED:
        return {"expected_pnl_usd": None, "confidence": 0.0}

    key = f"ev:{strategy_id}:{pair}"
    cached = _cached(key)
    if cached is not None:
        return cached

    features = await _get_features(pair, "5m")
    result = predict_ev(strategy_id, features)
    _set_cache(key, result)
    return result


async def get_volatility(pair: str) -> dict:
    if not settings.ML_ENABLED:
        return {"predicted_vol_pct": 0.0, "vol_ratio": 1.0}

    key = f"vol:{pair}"
    cached = _cached(key)
    if cached is not None:
        return cached

    features = await _get_features(pair, "5m")
    result = predict_volatility(pair, features)
    _set_cache(key, result)
    return result


def invalidate_cache():
    """Called after retraining to clear cached predictions."""
    _cache.clear()
    logger.info("ML inference cache invalidated")
