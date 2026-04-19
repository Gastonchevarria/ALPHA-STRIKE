"""Orchestrates full retraining of all ML models."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from config.settings import settings
from ml.data_loader import update_incremental
from ml.ev_model import train_ev_model
from ml.inference import invalidate_cache
from ml.regime_classifier import train_regime_classifier
from ml.volatility_predictor import train_volatility_predictor

logger = logging.getLogger(__name__)

STRATEGY_IDS = [f"G-{i:02d}" for i in range(1, 14)]


async def run_full_training() -> dict:
    """Full retraining pipeline. Returns summary of all models trained."""
    start = datetime.now(timezone.utc)
    summary = {"started_at": start.isoformat(), "models": {}}

    pairs = settings.pairs_list
    logger.info(f"=== Starting full ML retraining for pairs: {pairs} ===")

    # 1. Update historical data
    logger.info("Step 1/4: Updating historical data")
    pair_dfs = {}
    for pair in pairs:
        try:
            df = await update_incremental(pair, "5m")
            pair_dfs[pair] = df
            logger.info(f"  {pair}: {len(df)} rows")
        except Exception as e:
            logger.error(f"  Failed to update {pair}: {e}")

    if len(pair_dfs) < 2:
        summary["error"] = "Insufficient data updated"
        return summary

    # 2. Train Regime Classifier
    logger.info("Step 2/4: Training Regime Classifier")
    try:
        metrics = train_regime_classifier(pair_dfs)
        summary["models"]["regime_classifier"] = {"status": "ok", "metrics": metrics}
    except Exception as e:
        logger.error(f"Regime classifier failed: {e}")
        summary["models"]["regime_classifier"] = {"status": "failed", "error": str(e)}

    # 3. Train Volatility Predictor
    logger.info("Step 3/4: Training Volatility Predictor")
    try:
        metrics = train_volatility_predictor(pair_dfs)
        summary["models"]["volatility_predictor"] = {"status": "ok", "metrics": metrics}
    except Exception as e:
        logger.error(f"Volatility predictor failed: {e}")
        summary["models"]["volatility_predictor"] = {"status": "failed", "error": str(e)}

    # 4. Train EV models (one per strategy)
    logger.info("Step 4/4: Training EV models per strategy")
    for sid in STRATEGY_IDS:
        try:
            metrics = train_ev_model(sid, pair_dfs)
            summary["models"][f"ev_{sid}"] = {"status": "ok", "metrics": metrics}
        except Exception as e:
            logger.warning(f"EV model for {sid} failed: {e}")
            summary["models"][f"ev_{sid}"] = {"status": "failed", "error": str(e)}

    invalidate_cache()

    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    summary["duration_seconds"] = (datetime.now(timezone.utc) - start).total_seconds()

    logger.info(f"=== Retraining complete in {summary['duration_seconds']:.0f}s ===")
    return summary


async def initial_setup():
    """First-time setup: backfill data + train all models."""
    from ml.data_loader import initial_backfill
    logger.info("=== Initial ML setup ===")
    await initial_backfill(settings.pairs_list, timeframes=["5m"])
    return await run_full_training()
