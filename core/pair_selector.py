"""Parallel multi-pair evaluation and selection."""

import asyncio
import logging

from core.data_fetcher import fetch_ohlcv
from core.market_regime import get_cached_regime, strategy_matches_regime

logger = logging.getLogger(__name__)

# Max strategies with open position on same pair
MAX_EXPOSURE_PER_PAIR = 4


async def select_best_pair(
    strategy,
    pairs: list[str],
    all_strategies: list,
    min_confidence: float = 0.72,
) -> dict | None:
    """Evaluate all pairs in parallel and return best signal.

    Returns: {pair, signal, confidence} or None if no valid signal.
    """

    # Check exposure limits per pair
    pair_exposure = {}
    for s in all_strategies:
        if s.position and s._entry_pair:
            pair_exposure[s._entry_pair] = pair_exposure.get(s._entry_pair, 0) + 1

    eligible_pairs = []
    for pair in pairs:
        if strategy.is_pair_on_cooldown(pair):
            continue
        if pair_exposure.get(pair, 0) >= MAX_EXPOSURE_PER_PAIR:
            continue
        # Check regime match for this specific pair
        regime = get_cached_regime(pair)
        if not strategy_matches_regime(strategy.cfg.regime_filter, regime):
            continue
        eligible_pairs.append(pair)

    if not eligible_pairs:
        return None

    # Evaluate all eligible pairs in parallel
    async def eval_pair(pair):
        try:
            df = await fetch_ohlcv(pair, strategy.cfg.timeframe, limit=60)
            signal = await strategy.evaluate(pair, df)
            return {"pair": pair, "signal": signal}
        except Exception as e:
            logger.warning(f"Eval failed for {strategy.cfg.id} on {pair}: {e}")
            return None

    tasks = [eval_pair(p) for p in eligible_pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter valid signals
    valid = []
    for r in results:
        if isinstance(r, dict) and r is not None:
            sig = r["signal"]
            if sig.execute and sig.direction != "HOLD":
                effective_conf = sig.confidence * strategy.confidence_multiplier
                if effective_conf >= min_confidence:
                    valid.append({
                        "pair": r["pair"],
                        "signal": sig,
                        "effective_confidence": effective_conf,
                    })

    if not valid:
        return None

    # Rank by effective confidence
    valid.sort(key=lambda x: x["effective_confidence"], reverse=True)
    return valid[0]
