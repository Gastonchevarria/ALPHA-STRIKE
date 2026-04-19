"""Walk-forward validation for time-series ML."""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def walk_forward_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_fraction: float = 0.1,
) -> list[tuple[pd.Index, pd.Index]]:
    """Generate walk-forward train/test splits.

    Each split: train on [0, test_start), test on [test_start, test_end)
    Test windows move forward through the data.
    """
    n = len(df)
    test_size = int(n * test_fraction)
    min_train = n - test_size * n_splits

    if min_train < n * 0.4:
        raise ValueError(f"Not enough data for {n_splits} splits")

    splits = []
    for i in range(n_splits):
        test_start = min_train + i * test_size
        test_end = min(test_start + test_size, n)
        train_idx = df.index[:test_start]
        test_idx = df.index[test_start:test_end]
        splits.append((train_idx, test_idx))

    return splits


def validate_classifier(
    model_factory: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict:
    """Walk-forward validation for a classifier. Returns aggregated metrics."""
    accuracies = []
    for train_idx, test_idx in walk_forward_split(X, n_splits=n_splits):
        model = model_factory()
        model.fit(X.loc[train_idx], y.loc[train_idx])
        preds = model.predict(X.loc[test_idx])
        acc = accuracy_score(y.loc[test_idx], preds)
        accuracies.append(acc)

    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "n_splits": n_splits,
        "per_split_accuracy": [float(a) for a in accuracies],
    }


def validate_regressor(
    model_factory: Callable,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> dict:
    """Walk-forward validation for a regressor. Returns MAE and R^2."""
    maes, r2s = [], []
    for train_idx, test_idx in walk_forward_split(X, n_splits=n_splits):
        model = model_factory()
        model.fit(X.loc[train_idx], y.loc[train_idx])
        preds = model.predict(X.loc[test_idx])
        maes.append(mean_absolute_error(y.loc[test_idx], preds))
        r2s.append(r2_score(y.loc[test_idx], preds))

    return {
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "n_splits": n_splits,
    }
