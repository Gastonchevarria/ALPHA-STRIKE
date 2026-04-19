"""Regime Classifier — predicts regime over next 15 minutes."""
from __future__ import annotations

import logging

import pandas as pd
from xgboost import XGBClassifier

from ml.features import FEATURE_NAMES, compute_features_from_df, extract_features_batch
from ml.labels import generate_regime_labels
from ml.model_store import load_metadata, load_model, save_model
from ml.validation import validate_classifier

logger = logging.getLogger(__name__)

MODEL_NAME = "regime_classifier"
LABEL_MAP = {0: "RANGING", 1: "TRENDING_UP", 2: "TRENDING_DOWN", 3: "VOLATILE"}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def _model_factory() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )


def prepare_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features + labels from a single pair's OHLCV DataFrame."""
    features_df = extract_features_batch(df, start_idx=60)
    labels = generate_regime_labels(df, horizon_bars=15)

    # Align features with labels by index
    features_df = features_df.set_index(features_df["_ts"].apply(lambda x: df[df["ts"] == x].index[0] if any(df["ts"] == x) else -1))
    features_df = features_df[features_df.index >= 0]

    labels_aligned = labels.loc[features_df.index]
    mask = labels_aligned >= 0

    X = features_df.loc[mask, FEATURE_NAMES]
    y = labels_aligned[mask]

    return X.reset_index(drop=True), y.reset_index(drop=True)


def train_regime_classifier(
    all_pair_dfs: dict[str, pd.DataFrame],
) -> dict:
    """Train a single global classifier from data of all pairs."""
    all_X, all_y = [], []
    for pair, df in all_pair_dfs.items():
        try:
            X, y = prepare_training_data(df)
            all_X.append(X)
            all_y.append(y)
            logger.info(f"Prepared {len(X)} samples from {pair}")
        except Exception as e:
            logger.error(f"Failed to prepare {pair}: {e}")

    if not all_X:
        raise RuntimeError("No training data")

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)

    logger.info(f"Total training data: {len(X)} samples, class dist: {y.value_counts().to_dict()}")

    metrics = validate_classifier(_model_factory, X, y, n_splits=5)
    logger.info(f"Walk-forward validation: accuracy={metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}")

    # Train final model on all data
    final_model = _model_factory()
    final_model.fit(X, y)

    metadata = {
        "metrics": metrics,
        "n_train": len(X),
        "feature_names": FEATURE_NAMES,
        "class_distribution": y.value_counts().to_dict(),
    }
    save_model(MODEL_NAME, final_model, metadata)

    return metrics


def predict_regime(features: dict) -> dict:
    """Predict regime for a single observation."""
    model = load_model(MODEL_NAME)
    if model is None:
        return {"regime": "UNKNOWN", "confidence": 0.0, "probabilities": {}}

    X = pd.DataFrame([{k: features.get(k, 0.0) for k in FEATURE_NAMES}])
    probs = model.predict_proba(X)[0]
    pred_idx = int(probs.argmax())

    return {
        "regime": LABEL_MAP[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)},
    }
