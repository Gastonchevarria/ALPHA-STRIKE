"""Unit tests for the ML inference API."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, AsyncMock
from datetime import datetime, timedelta, timezone


def make_sample_ohlcv(n=100):
    """Create minimal OHLCV DataFrame for testing."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.3)
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
        "volume": np.random.uniform(100, 500, n),
    })


def test_regime_prediction_no_model():
    """When no model is trained, should return UNKNOWN."""
    from ml.regime_classifier import predict_regime
    result = predict_regime({"rsi_14": 50.0, "adx": 25.0})
    assert result["regime"] == "UNKNOWN"
    assert result["confidence"] == 0.0


def test_volatility_prediction_no_model():
    """When no model is trained, should return neutral vol_ratio."""
    from ml.volatility_predictor import predict_volatility
    result = predict_volatility("SUIUSDT", {"rsi_14": 50.0})
    assert result["vol_ratio"] == 1.0


def test_ev_prediction_no_model():
    """When no model is trained, should return None expected_pnl."""
    from ml.ev_model import predict_ev
    result = predict_ev("G-01", {"rsi_14": 50.0})
    assert result["expected_pnl_usd"] is None
    assert result["n_samples_trained_on"] == 0


def test_inference_cache():
    """Cache should return same value within TTL."""
    from ml.inference import _cached, _set_cache, invalidate_cache

    invalidate_cache()
    assert _cached("test:key") is None

    _set_cache("test:key", {"value": 42})
    assert _cached("test:key") == {"value": 42}

    invalidate_cache()
    assert _cached("test:key") is None


def test_model_store_list_empty():
    """Model store should list models even if none exist."""
    from ml.model_store import list_models
    models = list_models()
    assert isinstance(models, list)
