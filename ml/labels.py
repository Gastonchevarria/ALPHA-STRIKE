"""Label generation for ML models. Uses future data, so only for training."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_regime_labels(df: pd.DataFrame, horizon_bars: int = 15) -> pd.Series:
    """For each bar, label the regime over [t, t+horizon_bars].

    Labels:
    - 0: RANGING
    - 1: TRENDING_UP
    - 2: TRENDING_DOWN
    - 3: VOLATILE

    Uses the same rules as the runtime regime detector but applied to future window.
    """
    import pandas_ta as ta

    labels = pd.Series(index=df.index, dtype="int32")

    # Precompute indicators on full series
    df = df.copy()
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.atr(length=14, append=True)

    for i in range(len(df)):
        future_end = min(i + horizon_bars, len(df) - 1)
        if future_end - i < horizon_bars:
            labels.iloc[i] = -1  # insufficient future data
            continue

        # Compute future-window aggregates
        window = df.iloc[i:future_end + 1]
        ema20_end = window["EMA_20"].iloc[-1]
        ema50_end = window["EMA_50"].iloc[-1]
        adx_mean = window["ADX_14"].mean()
        atr_mean = window["ATRr_14"].mean()
        close_mean = window["close"].mean()
        atr_pct = atr_mean / max(close_mean, 1e-9) * 100

        if pd.isna(ema20_end) or pd.isna(ema50_end) or pd.isna(adx_mean):
            labels.iloc[i] = -1
            continue

        if ema20_end > ema50_end and adx_mean > 25:
            labels.iloc[i] = 1  # TRENDING_UP
        elif ema20_end < ema50_end and adx_mean > 25:
            labels.iloc[i] = 2  # TRENDING_DOWN
        elif atr_pct > 0.5:
            labels.iloc[i] = 3  # VOLATILE
        else:
            labels.iloc[i] = 0  # RANGING

    return labels


def generate_volatility_labels(df: pd.DataFrame, horizon_bars: int = 30) -> pd.Series:
    """For each bar at time t, label = std of 1-bar returns over [t, t+horizon_bars]."""
    returns = df["close"].pct_change()
    labels = pd.Series(index=df.index, dtype="float64")

    for i in range(len(df) - horizon_bars):
        future_returns = returns.iloc[i + 1:i + 1 + horizon_bars]
        if len(future_returns) >= horizon_bars // 2:
            labels.iloc[i] = float(future_returns.std())
        else:
            labels.iloc[i] = np.nan

    labels.iloc[len(df) - horizon_bars:] = np.nan
    return labels


def generate_ev_labels_from_trades(
    trades_df: pd.DataFrame,
    strategy_id: str,
) -> pd.DataFrame:
    """From historical trades of a specific strategy, create training examples.

    Returns DataFrame with columns: ts (entry time), pnl_net (label), pair.
    Features must be extracted separately at each entry time using features.py.
    """
    filtered = trades_df[trades_df["strategy_id"] == strategy_id].copy()
    if len(filtered) == 0:
        return pd.DataFrame(columns=["ts", "pnl_net", "pair"])

    filtered["ts"] = pd.to_datetime(filtered["ts"], utc=True)
    return filtered[["ts", "pnl_net", "pair"]].reset_index(drop=True)
