"""Volatility Predictor — predicts realized volatility over next 30 min."""
from __future__ import annotations

import logging

import pandas as pd
from xgboost import XGBRegressor

from ml.features import FEATURE_NAMES, extract_features_batch
from ml.labels import generate_volatility_labels
from ml.model_store import load_model, save_model
from ml.validation import validate_regressor

logger = logging.getLogger(__name__)

MODEL_NAME = "volatility_predictor"

# All pairs one-hot encoded. Must match settings.pairs_list.
_PAIR_FEATURES = [f"pair_{p}" for p in ["WIFUSDT", "1000BONKUSDT", "1000PEPEUSDT", "1000SHIBUSDT", "SUIUSDT", "ORDIUSDT"]]


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


def _add_pair_onehot(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    out = df.copy()
    for p in _PAIR_FEATURES:
        out[p] = 1.0 if f"pair_{pair}" == p else 0.0
    return out


def prepare_training_data(pair_dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
    all_X, all_y = [], []
    for pair, df in pair_dfs.items():
        features_df = extract_features_batch(df, start_idx=60)
        vol_labels = generate_volatility_labels(df, horizon_bars=30)

        # Align on ts
        aligned = features_df.merge(
            df[["ts"]].assign(_label=vol_labels.values),
            left_on="_ts", right_on="ts", how="left",
        )
        valid = aligned.dropna(subset=["_label"])

        X = valid[FEATURE_NAMES].copy()
        X = _add_pair_onehot(X, pair)
        y = valid["_label"]

        all_X.append(X)
        all_y.append(y)

    if not all_X:
        raise RuntimeError("No training data")

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)
    return X, y


def train_volatility_predictor(pair_dfs: dict[str, pd.DataFrame]) -> dict:
    X, y = prepare_training_data(pair_dfs)
    logger.info(f"Training volatility predictor on {len(X)} samples")

    metrics = validate_regressor(_model_factory, X, y, n_splits=5)
    logger.info(f"Validation: MAE={metrics['mae_mean']:.6f}, R²={metrics['r2_mean']:.3f}")

    final_model = _model_factory()
    final_model.fit(X, y)

    # Compute baseline volatility per pair
    baselines = {}
    for pair, df in pair_dfs.items():
        returns = df["close"].pct_change().dropna()
        baselines[pair] = float(returns.rolling(30).std().mean())

    metadata = {
        "metrics": metrics,
        "n_train": len(X),
        "feature_names": FEATURE_NAMES + _PAIR_FEATURES,
        "baselines": baselines,
    }
    save_model(MODEL_NAME, final_model, metadata)
    return metrics


def predict_volatility(pair: str, features: dict) -> dict:
    model = load_model(MODEL_NAME)
    if model is None:
        return {"predicted_vol_pct": 0.0, "baseline_vol_pct": 0.0, "vol_ratio": 1.0}

    from ml.model_store import load_metadata
    meta = load_metadata(MODEL_NAME)
    baseline = meta.get("baselines", {}).get(pair, 0.001)

    row = {k: features.get(k, 0.0) for k in FEATURE_NAMES}
    for p in _PAIR_FEATURES:
        row[p] = 1.0 if f"pair_{pair}" == p else 0.0
    X = pd.DataFrame([row])

    pred = float(model.predict(X)[0])
    pred = max(pred, 1e-6)

    return {
        "predicted_vol_pct": pred,
        "baseline_vol_pct": baseline,
        "vol_ratio": round(pred / max(baseline, 1e-9), 3),
    }
