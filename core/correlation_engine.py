"""Cross-pair correlation engine for divergence detection."""

import asyncio
import logging
import time

import numpy as np
import pandas as pd

from core.data_fetcher import fetch_ohlcv

logger = logging.getLogger(__name__)

_correlation_cache: dict | None = None
_cache_ts: float = 0
_CACHE_TTL = 300


async def compute_correlation_matrix(
    pairs: list[str],
    interval: str = "5m",
    periods: int = 20,
) -> dict:
    global _correlation_cache, _cache_ts

    now = time.time()
    if _correlation_cache and now - _cache_ts < _CACHE_TTL:
        return _correlation_cache

    tasks = [fetch_ohlcv(p, interval, limit=periods + 5) for p in pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    returns = {}
    for pair, result in zip(pairs, results):
        if isinstance(result, pd.DataFrame) and len(result) >= periods:
            pct = result["close"].pct_change().dropna().tail(periods)
            returns[pair] = pct.values
        else:
            logger.warning(f"Correlation: skipping {pair}, insufficient data")

    if len(returns) < 2:
        return {"matrix": {}, "divergence_index": 1.0, "breaks": []}

    pair_names = list(returns.keys())
    n = len(pair_names)
    matrix = {}
    correlations = []

    for i in range(n):
        matrix[pair_names[i]] = {}
        for j in range(n):
            if i == j:
                matrix[pair_names[i]][pair_names[j]] = 1.0
            else:
                r1 = returns[pair_names[i]]
                r2 = returns[pair_names[j]]
                min_len = min(len(r1), len(r2))
                if min_len > 2:
                    corr = float(np.corrcoef(r1[:min_len], r2[:min_len])[0, 1])
                    if np.isnan(corr):
                        corr = 0.0
                else:
                    corr = 0.0
                matrix[pair_names[i]][pair_names[j]] = round(corr, 4)
                if i < j:
                    correlations.append(corr)

    avg_corr = float(np.mean(correlations)) if correlations else 0.0
    std_corr = float(np.std(correlations)) if len(correlations) > 1 else 0.0

    breaks = []
    for i in range(n):
        for j in range(i + 1, n):
            corr = matrix[pair_names[i]][pair_names[j]]
            if avg_corr - corr > 2 * std_corr and std_corr > 0.05:
                breaks.append({
                    "pair_a": pair_names[i],
                    "pair_b": pair_names[j],
                    "correlation": corr,
                    "avg": round(avg_corr, 4),
                    "deviation": round((avg_corr - corr) / max(std_corr, 0.01), 2),
                })

    result = {
        "matrix": matrix,
        "avg_correlation": round(avg_corr, 4),
        "divergence_index": round(1.0 - avg_corr, 4),
        "breaks": breaks,
        "updated_at": time.time(),
    }

    _correlation_cache = result
    _cache_ts = now
    return result


def get_correlation_matrix() -> dict:
    if _correlation_cache:
        return _correlation_cache
    return {"matrix": {}, "divergence_index": 0.0, "breaks": []}


def get_divergence_index() -> float:
    if _correlation_cache:
        return _correlation_cache.get("divergence_index", 0.0)
    return 0.0
