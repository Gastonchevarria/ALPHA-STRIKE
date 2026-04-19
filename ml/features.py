"""Feature extraction for ML models. 39 features per observation."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    # Price-based (12)
    "return_1m", "return_5m", "return_15m", "return_1h",
    "volatility_5m", "volatility_30m",
    "price_z_20",
    "distance_from_ema20", "distance_from_ema50", "ema_spread",
    "high_low_range_5m",
    # Momentum (8)
    "rsi_7", "rsi_14", "rsi_14_change_5m",
    "macd_hist", "macd_hist_change",
    "stoch_k", "stoch_d", "adx",
    # Volume (6)
    "volume_ratio_5m", "volume_ratio_20m",
    "taker_buy_ratio", "taker_buy_ratio_5m_avg",
    "volume_delta_5m", "dollar_volume_1m",
    # Bollinger/Volatility (5)
    "bb_position", "bb_width", "atr_pct", "atr_ratio", "keltner_position",
    # Time-based (4)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # Cross-pair (4) — filled externally
    "anchor_correlation", "anchor_return_divergence",
    "market_divergence_index", "relative_volume_rank",
]


def compute_features_from_df(
    df: pd.DataFrame,
    target_index: Optional[int] = None,
) -> dict:
    """Compute features for a single observation at the given index.

    df must have columns: ts, open, high, low, close, volume, taker_buy_base.
    If target_index is None, computes for the last row.

    NEVER uses data beyond target_index (no future leakage).
    """
    if target_index is None:
        target_index = len(df) - 1
    if target_index < 60:
        raise ValueError(f"Need at least 60 prior rows, got index {target_index}")

    # Slice up to and including target_index (past + present, no future)
    past = df.iloc[:target_index + 1].copy()

    close = past["close"]
    high = past["high"]
    low = past["low"]
    volume = past["volume"]
    taker_buy = past.get("taker_buy_base", volume / 2)

    features = {}

    # --- Price-based ---
    features["return_1m"] = float(close.pct_change(1).iloc[-1]) if len(close) > 1 else 0.0
    features["return_5m"] = float(close.pct_change(5).iloc[-1]) if len(close) > 5 else 0.0
    features["return_15m"] = float(close.pct_change(15).iloc[-1]) if len(close) > 15 else 0.0
    features["return_1h"] = float(close.pct_change(60).iloc[-1]) if len(close) > 60 else 0.0

    returns_1m = close.pct_change(1).dropna()
    features["volatility_5m"] = float(returns_1m.tail(5).std()) if len(returns_1m) >= 5 else 0.0
    features["volatility_30m"] = float(returns_1m.tail(30).std()) if len(returns_1m) >= 30 else 0.0

    mean_20 = close.tail(20).mean()
    std_20 = close.tail(20).std()
    features["price_z_20"] = float((close.iloc[-1] - mean_20) / max(std_20, 1e-9))

    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1] if len(close) >= 50 else ema20
    features["distance_from_ema20"] = float((close.iloc[-1] - ema20) / max(ema20, 1e-9))
    features["distance_from_ema50"] = float((close.iloc[-1] - ema50) / max(ema50, 1e-9))
    features["ema_spread"] = float((ema20 - ema50) / max(ema50, 1e-9))

    hl_range = (high.tail(5).max() - low.tail(5).min()) / max(close.iloc[-1], 1e-9)
    features["high_low_range_5m"] = float(hl_range)

    # --- Momentum ---
    try:
        rsi_7 = past.ta.rsi(length=7).iloc[-1] if len(close) >= 7 else 50.0
        rsi_14 = past.ta.rsi(length=14).iloc[-1] if len(close) >= 14 else 50.0
        rsi_14_prev = past.ta.rsi(length=14).iloc[-6] if len(close) >= 20 else rsi_14
    except Exception:
        rsi_7 = rsi_14 = rsi_14_prev = 50.0
    features["rsi_7"] = float(rsi_7 if not pd.isna(rsi_7) else 50.0)
    features["rsi_14"] = float(rsi_14 if not pd.isna(rsi_14) else 50.0)
    features["rsi_14_change_5m"] = float(rsi_14 - rsi_14_prev) if not pd.isna(rsi_14_prev) else 0.0

    try:
        macd = past.ta.macd(fast=12, slow=26, signal=9)
        hist_col = [c for c in macd.columns if "MACDh" in c][0]
        macd_hist = macd[hist_col].iloc[-1]
        macd_hist_prev = macd[hist_col].iloc[-2] if len(macd) > 1 else macd_hist
    except Exception:
        macd_hist = macd_hist_prev = 0.0
    features["macd_hist"] = float(macd_hist if not pd.isna(macd_hist) else 0.0)
    features["macd_hist_change"] = float((macd_hist - macd_hist_prev) if not pd.isna(macd_hist_prev) else 0.0)

    try:
        stoch = past.ta.stoch(k=14, d=3)
        k_col = [c for c in stoch.columns if "STOCHk" in c][0]
        d_col = [c for c in stoch.columns if "STOCHd" in c][0]
        stoch_k = stoch[k_col].iloc[-1]
        stoch_d = stoch[d_col].iloc[-1]
    except Exception:
        stoch_k = stoch_d = 50.0
    features["stoch_k"] = float(stoch_k if not pd.isna(stoch_k) else 50.0)
    features["stoch_d"] = float(stoch_d if not pd.isna(stoch_d) else 50.0)

    try:
        adx = past.ta.adx(length=14)
        adx_col = [c for c in adx.columns if "ADX" in c][0]
        adx_val = adx[adx_col].iloc[-1]
    except Exception:
        adx_val = 20.0
    features["adx"] = float(adx_val if not pd.isna(adx_val) else 20.0)

    # --- Volume ---
    vol_ma5 = volume.tail(5).mean()
    vol_ma20 = volume.tail(20).mean()
    features["volume_ratio_5m"] = float(volume.iloc[-1] / max(vol_ma5, 1e-9))
    features["volume_ratio_20m"] = float(volume.iloc[-1] / max(vol_ma20, 1e-9))

    features["taker_buy_ratio"] = float(taker_buy.iloc[-1] / max(volume.iloc[-1], 1e-9))
    features["taker_buy_ratio_5m_avg"] = float(taker_buy.tail(5).sum() / max(volume.tail(5).sum(), 1e-9))

    delta = (taker_buy - (volume - taker_buy)).tail(5).sum()
    features["volume_delta_5m"] = float(delta)
    features["dollar_volume_1m"] = float(close.iloc[-1] * volume.iloc[-1])

    # --- Bollinger/Volatility ---
    try:
        bb = past.ta.bbands(length=20, std=2)
        bbu = [c for c in bb.columns if c.startswith("BBU")][0]
        bbl = [c for c in bb.columns if c.startswith("BBL")][0]
        bbm = [c for c in bb.columns if c.startswith("BBM")][0]
        bb_upper = bb[bbu].iloc[-1]
        bb_lower = bb[bbl].iloc[-1]
        bb_mid = bb[bbm].iloc[-1]
        bb_std = (bb_upper - bb_lower) / 4  # std ≈ range/4
        features["bb_position"] = float((close.iloc[-1] - bb_mid) / max(bb_std, 1e-9))
        features["bb_width"] = float((bb_upper - bb_lower) / max(bb_mid, 1e-9))
    except Exception:
        features["bb_position"] = 0.0
        features["bb_width"] = 0.01

    try:
        atr = past.ta.atr(length=14)
        atr_col = [c for c in atr.columns if "ATR" in c][0] if hasattr(atr, "columns") else None
        if atr_col:
            atr_val = atr[atr_col].iloc[-1]
            atr_avg = atr[atr_col].tail(20).mean()
        else:
            atr_val = atr.iloc[-1] if hasattr(atr, "iloc") else 0.0
            atr_avg = atr.tail(20).mean() if hasattr(atr, "tail") else atr_val
    except Exception:
        atr_val = atr_avg = 0.0
    features["atr_pct"] = float(atr_val / max(close.iloc[-1], 1e-9))
    features["atr_ratio"] = float(atr_val / max(atr_avg, 1e-9)) if atr_avg else 1.0

    try:
        kc = past.ta.kc(length=20, scalar=1.5)
        kcu = [c for c in kc.columns if c.startswith("KCU")][0]
        kcl = [c for c in kc.columns if c.startswith("KCL")][0]
        kc_upper = kc[kcu].iloc[-1]
        kc_lower = kc[kcl].iloc[-1]
        kc_mid = (kc_upper + kc_lower) / 2
        kc_width = (kc_upper - kc_lower)
        features["keltner_position"] = float((close.iloc[-1] - kc_mid) / max(kc_width, 1e-9))
    except Exception:
        features["keltner_position"] = 0.0

    # --- Time-based ---
    ts = past["ts"].iloc[-1] if "ts" in past.columns else datetime.now(timezone.utc)
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    hour = ts.hour
    dow = ts.weekday()
    features["hour_sin"] = float(np.sin(2 * np.pi * hour / 24))
    features["hour_cos"] = float(np.cos(2 * np.pi * hour / 24))
    features["dow_sin"] = float(np.sin(2 * np.pi * dow / 7))
    features["dow_cos"] = float(np.cos(2 * np.pi * dow / 7))

    # Cross-pair features filled externally (via enrich_cross_pair)
    features["anchor_correlation"] = 0.5
    features["anchor_return_divergence"] = 0.0
    features["market_divergence_index"] = 0.5
    features["relative_volume_rank"] = 0.5

    # Replace any NaN/inf with safe defaults
    for k, v in features.items():
        if not np.isfinite(v):
            features[k] = 0.0

    return features


def enrich_cross_pair(
    features: dict,
    pair: str,
    anchor_pair: str,
    all_pair_data: dict[str, pd.DataFrame],
    target_index: int,
) -> dict:
    """Add the 4 cross-pair features using data from all pairs."""
    if pair == anchor_pair or anchor_pair not in all_pair_data:
        return features

    anchor_df = all_pair_data[anchor_pair]
    pair_df = all_pair_data.get(pair)
    if pair_df is None or target_index >= len(anchor_df):
        return features

    # Align on ts
    anchor_window = anchor_df.iloc[max(0, target_index - 20):target_index + 1]
    pair_window = pair_df.iloc[max(0, target_index - 20):target_index + 1]

    if len(anchor_window) < 10 or len(pair_window) < 10:
        return features

    anchor_returns = anchor_window["close"].pct_change().dropna()
    pair_returns = pair_window["close"].pct_change().dropna()
    n = min(len(anchor_returns), len(pair_returns))
    if n >= 5:
        features["anchor_correlation"] = float(np.corrcoef(anchor_returns.tail(n), pair_returns.tail(n))[0, 1])

    anchor_ret_5m = (anchor_window["close"].iloc[-1] - anchor_window["close"].iloc[-6]) / anchor_window["close"].iloc[-6] if len(anchor_window) >= 6 else 0.0
    pair_ret_5m = (pair_window["close"].iloc[-1] - pair_window["close"].iloc[-6]) / pair_window["close"].iloc[-6] if len(pair_window) >= 6 else 0.0
    features["anchor_return_divergence"] = float(pair_ret_5m - anchor_ret_5m)

    # Market divergence = 1 - avg correlation across all pairs
    # Relative volume rank = pair's volume rank 1-N normalized 0-1
    volumes = []
    for p, df in all_pair_data.items():
        if target_index < len(df):
            volumes.append((p, df.iloc[target_index]["volume"]))
    if volumes:
        volumes.sort(key=lambda x: x[1])
        ranks = {p: (i + 1) / len(volumes) for i, (p, _) in enumerate(volumes)}
        features["relative_volume_rank"] = float(ranks.get(pair, 0.5))

    # market_divergence_index is expensive per-row; use 0.5 placeholder (updated by runtime correlation engine)
    features["market_divergence_index"] = 0.5

    return features


def extract_features_batch(
    df: pd.DataFrame,
    start_idx: int = 60,
) -> pd.DataFrame:
    """Extract features for all rows in df (for training)."""
    rows = []
    for i in range(start_idx, len(df)):
        try:
            feats = compute_features_from_df(df, target_index=i)
            feats["_ts"] = df["ts"].iloc[i]
            rows.append(feats)
        except Exception as e:
            logger.warning(f"Feature extraction failed at index {i}: {e}")

    features_df = pd.DataFrame(rows)
    return features_df
