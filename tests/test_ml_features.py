"""Unit tests for feature extraction - especially leakage checks."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone


def make_sample_df(n=200):
    """Create deterministic OHLCV sample data."""
    np.random.seed(42)
    ts = [datetime.now(timezone.utc) - timedelta(minutes=(n - i)) for i in range(n)]
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "ts": ts,
        "open": close - np.random.uniform(0, 0.3, n),
        "high": close + np.abs(np.random.randn(n) * 0.5),
        "low": close - np.abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.uniform(100, 1000, n),
        "taker_buy_base": np.random.uniform(40, 600, n),
    })
    return df


def test_features_returns_39_fields():
    from ml.features import compute_features_from_df, FEATURE_NAMES
    df = make_sample_df()
    feats = compute_features_from_df(df, target_index=150)
    assert len(feats) == len(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert name in feats, f"Missing feature: {name}"


def test_features_no_future_leakage():
    """Features at index i should NOT change if we remove rows after i."""
    from ml.features import compute_features_from_df
    df = make_sample_df()

    feats_full = compute_features_from_df(df, target_index=100)
    feats_truncated = compute_features_from_df(df.iloc[:101].copy(), target_index=100)

    for key in feats_full:
        assert abs(feats_full[key] - feats_truncated[key]) < 1e-9, f"Leakage detected in {key}"


def test_features_handle_short_history():
    """Should raise ValueError if not enough history."""
    from ml.features import compute_features_from_df
    df = make_sample_df(n=70)
    with pytest.raises(ValueError):
        compute_features_from_df(df, target_index=30)


def test_features_no_nan_or_inf():
    from ml.features import compute_features_from_df
    df = make_sample_df()
    feats = compute_features_from_df(df, target_index=150)
    for k, v in feats.items():
        assert np.isfinite(v), f"Feature {k} is not finite: {v}"


def test_features_batch_extraction():
    from ml.features import extract_features_batch
    df = make_sample_df(n=150)
    feats_df = extract_features_batch(df, start_idx=60)
    assert len(feats_df) == 150 - 60
    assert "rsi_14" in feats_df.columns
    assert "_ts" in feats_df.columns
