"""Shared test fixtures for Agent GOD 2."""

import pytest


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for strategy tests."""
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    n = 100
    base_price = 65000.0
    prices = base_price + np.cumsum(np.random.randn(n) * 50)

    df = pd.DataFrame({
        "open": prices,
        "high": prices + np.abs(np.random.randn(n) * 30),
        "low": prices - np.abs(np.random.randn(n) * 30),
        "close": prices + np.random.randn(n) * 20,
        "volume": np.random.uniform(100, 1000, n),
    })
    return df


@pytest.fixture
def mock_settings():
    """Return settings with test defaults."""
    from config.settings import Settings
    return Settings(
        GEMINI_API_KEY="test-key",
        MODE="paper",
        INITIAL_BALANCE=1000.0,
        PORT=9999,
    )
