"""Tests for label generation — verifying no look-ahead bias for regimes."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone


def make_trending_df(n=200):
    ts = [datetime.now(timezone.utc) - timedelta(minutes=(n - i)) for i in range(n)]
    # Strong uptrend
    close = 100 + np.linspace(0, 20, n) + np.random.randn(n) * 0.1
    df = pd.DataFrame({
        "ts": ts, "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": np.ones(n) * 100,
    })
    return df


def test_regime_labels_correct_count():
    from ml.labels import generate_regime_labels
    df = make_trending_df(n=100)
    labels = generate_regime_labels(df, horizon_bars=15)
    assert len(labels) == len(df)
    # Last horizon_bars positions should be -1 (insufficient future data)
    assert (labels.iloc[-15:] == -1).all()


def test_regime_labels_trending_up():
    from ml.labels import generate_regime_labels
    df = make_trending_df(n=200)
    labels = generate_regime_labels(df, horizon_bars=15)
    valid_labels = labels[labels >= 0]
    # Strong uptrend should produce mostly label=1 (TRENDING_UP)
    assert (valid_labels == 1).mean() > 0.3, f"Got {(valid_labels == 1).mean()} TRENDING_UP"


def test_volatility_labels_finite():
    from ml.labels import generate_volatility_labels
    np.random.seed(42)
    n = 200
    ts = [datetime.now(timezone.utc) - timedelta(minutes=(n - i)) for i in range(n)]
    close = 100 + np.cumsum(np.random.randn(n) * 0.3)
    df = pd.DataFrame({"ts": ts, "close": close})

    labels = generate_volatility_labels(df, horizon_bars=30)
    valid = labels.dropna()
    assert len(valid) == n - 30
    assert (valid > 0).all()
    assert np.isfinite(valid).all()


def test_ev_labels_from_trades_empty():
    from ml.labels import generate_ev_labels_from_trades
    df = pd.DataFrame({"strategy_id": ["G-02"], "ts": ["2026-04-01"], "pnl_net": [1.0], "pair": ["BTCUSDT"]})
    result = generate_ev_labels_from_trades(df, "G-01")
    assert len(result) == 0


def test_ev_labels_from_trades_filters():
    from ml.labels import generate_ev_labels_from_trades
    df = pd.DataFrame({
        "strategy_id": ["G-01", "G-02", "G-01"],
        "ts": ["2026-04-01", "2026-04-02", "2026-04-03"],
        "pnl_net": [1.0, 2.0, 3.0],
        "pair": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    })
    result = generate_ev_labels_from_trades(df, "G-01")
    assert len(result) == 2
    assert list(result["pnl_net"]) == [1.0, 3.0]
