"""Expected-Value Model — one XGBoost regressor per strategy."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

from config.settings import settings
from ml.features import FEATURE_NAMES
from ml.labels import generate_ev_labels_from_trades
from ml.model_store import load_model, save_model
from ml.validation import validate_regressor

logger = logging.getLogger(__name__)

_TRADES_FILE = Path("learnings/trades.jsonl")


def _model_name(strategy_id: str) -> str:
    return f"ev_{strategy_id.replace('-', '')}"


def _model_factory() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )


def _load_trades() -> pd.DataFrame:
    if not _TRADES_FILE.exists():
        return pd.DataFrame(columns=["ts", "strategy_id", "pair", "pnl_net"])

    rows = []
    for line in _TRADES_FILE.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            rows.append({
                "ts": rec.get("ts"),
                "strategy_id": rec.get("strategy_id"),
                "pair": rec.get("pair", ""),
                "pnl_net": rec.get("pnl_net", 0.0),
            })
        except json.JSONDecodeError:
            continue

    return pd.DataFrame(rows)


def prepare_ev_training_data(
    strategy_id: str,
    pair_dfs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.Series]:
    """For a strategy, build (features at entry time, PnL) pairs from trades.jsonl."""
    from ml.features import compute_features_from_df

    trades_df = _load_trades()
    strat_trades = generate_ev_labels_from_trades(trades_df, strategy_id)

    if len(strat_trades) < settings.ML_MIN_TRADES_FOR_EV_MODEL:
        raise ValueError(f"Strategy {strategy_id} has only {len(strat_trades)} trades, need >= {settings.ML_MIN_TRADES_FOR_EV_MODEL}")

    all_X, all_y = [], []
    for _, trade in strat_trades.iterrows():
        pair = trade["pair"]
        ts = trade["ts"]
        pnl = trade["pnl_net"]

        if pair not in pair_dfs:
            continue

        df = pair_dfs[pair]
        df_ts = pd.to_datetime(df["ts"], utc=True)
        matches = (df_ts - ts).abs().idxmin()
        idx = int(matches)

        if idx < 60:
            continue

        try:
            feats = compute_features_from_df(df, target_index=idx)
            all_X.append(feats)
            all_y.append(pnl)
        except Exception:
            continue

    if not all_X:
        raise ValueError(f"Could not extract features for any trades of {strategy_id}")

    X = pd.DataFrame(all_X)[FEATURE_NAMES]
    y = pd.Series(all_y)
    return X, y


def train_ev_model(strategy_id: str, pair_dfs: dict[str, pd.DataFrame]) -> dict:
    try:
        X, y = prepare_ev_training_data(strategy_id, pair_dfs)
    except ValueError as e:
        logger.warning(f"Skipping EV model for {strategy_id}: {e}")
        return {"skipped": True, "reason": str(e)}

    logger.info(f"Training EV model for {strategy_id} on {len(X)} trades")

    if len(X) < 100:
        # Not enough data for walk-forward; use simple train/test split
        split = int(len(X) * 0.8)
        model = _model_factory()
        model.fit(X.iloc[:split], y.iloc[:split])
        from sklearn.metrics import mean_absolute_error, r2_score
        preds = model.predict(X.iloc[split:])
        metrics = {
            "mae_mean": float(mean_absolute_error(y.iloc[split:], preds)),
            "r2_mean": float(r2_score(y.iloc[split:], preds)),
            "n_splits": 1,
        }
    else:
        metrics = validate_regressor(_model_factory, X, y, n_splits=3)

    final_model = _model_factory()
    final_model.fit(X, y)

    metadata = {
        "strategy_id": strategy_id,
        "metrics": metrics,
        "n_train": len(X),
        "pnl_mean": float(y.mean()),
        "pnl_std": float(y.std()),
    }
    save_model(_model_name(strategy_id), final_model, metadata)
    return metrics


def predict_ev(strategy_id: str, features: dict) -> dict:
    model = load_model(_model_name(strategy_id))
    if model is None:
        return {"expected_pnl_usd": None, "confidence": 0.0, "n_samples_trained_on": 0}

    from ml.model_store import load_metadata
    meta = load_metadata(_model_name(strategy_id))

    X = pd.DataFrame([{k: features.get(k, 0.0) for k in FEATURE_NAMES}])
    pred = float(model.predict(X)[0])

    return {
        "expected_pnl_usd": round(pred, 2),
        "confidence": 0.6,  # placeholder; could use prediction variance
        "n_samples_trained_on": meta.get("n_train", 0),
    }
