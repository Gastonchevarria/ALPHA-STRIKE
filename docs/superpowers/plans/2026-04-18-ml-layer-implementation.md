# Agent GOD 2 — ML Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a machine learning augmentation layer (3 models: Regime Classifier, per-strategy Expected-Value, Volatility Predictor) for Agent GOD 2, running in the same Cloud Run container.

**Architecture:** New `ml/` package with XGBoost-based models trained on historical Binance OHLCV data, walk-forward validated, retrained weekly. Inference called inline from the tournament runner to filter signals and adjust TP/SL dynamically.

**Tech Stack:** Python 3.12, XGBoost 2.1.3, scikit-learn, pandas, pyarrow, google-cloud-storage, integrated with existing FastAPI + APScheduler.

**Spec:** `docs/superpowers/specs/2026-04-18-ml-layer-design.md`

---

## File Map

### Create (new files)
| File | Responsibility |
|:-----|:---------------|
| `ml/__init__.py` | Package init |
| `ml/data_loader.py` | Download historical OHLCV from Binance, parquet storage |
| `ml/features.py` | Feature engineering (39 features per observation) |
| `ml/labels.py` | Label generation for each model (no future leakage) |
| `ml/regime_classifier.py` | Model 1: Regime prediction (XGBoost classifier) |
| `ml/ev_model.py` | Model 2: Per-strategy expected value (13 XGBoost regressors) |
| `ml/volatility_predictor.py` | Model 3: Volatility prediction (XGBoost regressor) |
| `ml/training_pipeline.py` | Orchestrates training of all 15 models |
| `ml/model_store.py` | Load/save models to disk + GCS |
| `ml/inference.py` | Runtime inference API with caching |
| `ml/validation.py` | Walk-forward cross-validation utilities |
| `ml/retrain_scheduler.py` | APScheduler job for weekly retraining |
| `tests/test_ml_features.py` | Unit tests for feature extraction (leakage checks) |
| `tests/test_ml_labels.py` | Unit tests for label generation |
| `tests/test_ml_inference.py` | Unit tests for inference API |

### Modify (existing files)
| File | Change |
|:-----|:-------|
| `config/settings.py` | Add ML_* config variables |
| `requirements.txt` | Add xgboost, scikit-learn, joblib, pyarrow, google-cloud-storage |
| `scheduler/tournament_runner_god2.py` | Add inference calls + weekly retrain job |
| `strategies/base_strategy_v4.py` | Add `open_position_with_custom_params` method |
| `main_god2.py` | Add 6 new `/ml/*` endpoints |
| `static/dashboard.html` | Add 7th view "ML Insights" |

---

## Task 1: Dependencies + Config

**Files:**
- Modify: `requirements.txt`
- Modify: `config/settings.py`

- [ ] **Step 1: Add ML dependencies to requirements.txt**

Append to the existing `requirements.txt`:

```
xgboost==2.1.3
scikit-learn==1.5.2
joblib==1.4.2
pyarrow==17.0.0
google-cloud-storage==2.18.2
```

- [ ] **Step 2: Install locally**

```bash
cd /Users/gastonchevarria/Alpha/agent-god-2
python3 -m pip install xgboost==2.1.3 scikit-learn==1.5.2 joblib==1.4.2 pyarrow==17.0.0 google-cloud-storage==2.18.2
```

- [ ] **Step 3: Add ML settings to config/settings.py**

Insert these fields inside the `Settings` class, after the existing Memory section:

```python
    # ML Layer
    ML_ENABLED: bool = True
    ML_MIN_EV_USD: float = 0.50
    ML_REGIME_CONFIDENCE_THRESHOLD: float = 0.70
    ML_VOL_RATIO_MIN: float = 0.70
    ML_VOL_RATIO_MAX: float = 1.50
    ML_RETRAIN_DAY: int = 6  # Sunday (0=Mon, 6=Sun)
    ML_RETRAIN_HOUR_UTC: int = 3
    ML_HISTORICAL_DAYS: int = 90
    ML_GCS_BUCKET: str = "agent-god-2-data"
    ML_INFERENCE_CACHE_SECONDS: int = 30
    ML_MIN_TRADES_FOR_EV_MODEL: int = 50
```

- [ ] **Step 4: Verify imports**

```bash
python3 -c "import xgboost; import sklearn; import joblib; import pyarrow; from google.cloud import storage; print('All ML deps imported OK')"
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt config/settings.py
git commit -m "feat: add ML layer dependencies and config"
```

---

## Task 2: Data Loader (Historical OHLCV)

**Files:**
- Create: `ml/__init__.py` (empty)
- Create: `ml/data_loader.py`

- [ ] **Step 1: Create ml/__init__.py**

Empty file.

- [ ] **Step 2: Implement ml/data_loader.py**

```python
"""Historical OHLCV data loader for ML training."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

_DATA_DIR = Path("data/ml_historical")
_BINANCE_FUTURES = "https://fapi.binance.com"
_BATCH_SIZE = 1000  # max candles per request


def _parquet_path(pair: str, timeframe: str) -> Path:
    return _DATA_DIR / f"{pair}_{timeframe}.parquet"


async def _fetch_klines_batch(
    pair: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
) -> list:
    """Fetch one batch of klines from Binance Futures."""
    params = {
        "symbol": pair,
        "interval": timeframe,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": _BATCH_SIZE,
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{_BINANCE_FUTURES}/fapi/v1/klines", params=params)
        if resp.status_code != 200:
            raise RuntimeError(f"Binance API error {resp.status_code}: {resp.text}")
        return resp.json()


def _klines_to_df(raw: list) -> pd.DataFrame:
    cols = [
        "ts", "open", "high", "low", "close", "volume",
        "close_ts", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["close_ts"] = pd.to_datetime(df["close_ts"], unit="ms", utc=True)
    df["trades"] = df["trades"].astype(int)
    return df[["ts", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]]


async def download_historical(
    pair: str,
    timeframe: str,
    days_back: int = 90,
) -> pd.DataFrame:
    """Download full history and return as DataFrame."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)

    # Timeframe to milliseconds per candle
    tf_ms = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "1h": 3_600_000,
    }
    step_ms = tf_ms[timeframe] * _BATCH_SIZE

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    all_data = []
    cursor = start_ms
    while cursor < end_ms:
        batch_end = min(cursor + step_ms, end_ms)
        logger.info(f"Fetching {pair} {timeframe} from {datetime.fromtimestamp(cursor/1000, tz=timezone.utc)} ({len(all_data)} rows so far)")
        try:
            batch = await _fetch_klines_batch(pair, timeframe, cursor, batch_end)
            if not batch:
                break
            all_data.extend(batch)
            cursor = batch[-1][0] + tf_ms[timeframe]  # next candle after last fetched
            await asyncio.sleep(0.15)  # rate limit: ~400 req/min, well under 1200 limit
        except Exception as e:
            logger.error(f"Failed batch {pair} {timeframe} at {cursor}: {e}")
            await asyncio.sleep(2)
            continue

    df = _klines_to_df(all_data)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df


async def update_incremental(pair: str, timeframe: str) -> pd.DataFrame:
    """Update existing parquet with latest data (last 48h) and return full DataFrame."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _parquet_path(pair, timeframe)

    if not path.exists():
        logger.info(f"No existing data for {pair} {timeframe}, doing full backfill")
        df = await download_historical(pair, timeframe, days_back=90)
        df.to_parquet(path, engine="pyarrow", compression="snappy")
        return df

    existing = pd.read_parquet(path)
    last_ts = existing["ts"].max()

    # Fetch from (last_ts - 2h) to now, to handle any corrections
    start = last_ts - timedelta(hours=2)
    now = datetime.now(timezone.utc)
    hours_to_fetch = max(2, (now - start).total_seconds() / 3600)
    days_back = hours_to_fetch / 24 + 0.1

    new_data = await download_historical(pair, timeframe, days_back=days_back)

    combined = pd.concat([existing, new_data])
    combined = combined.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)

    # Trim old data beyond retention window
    cutoff = datetime.now(timezone.utc) - timedelta(days=180)
    combined = combined[combined["ts"] >= cutoff].reset_index(drop=True)

    combined.to_parquet(path, engine="pyarrow", compression="snappy")
    logger.info(f"Updated {pair} {timeframe}: {len(combined)} total rows (added {len(combined) - len(existing)})")
    return combined


def load_cached(pair: str, timeframe: str) -> pd.DataFrame | None:
    """Load cached parquet file, or None if missing."""
    path = _parquet_path(pair, timeframe)
    if not path.exists():
        return None
    return pd.read_parquet(path)


async def initial_backfill(pairs: list[str], timeframes: list[str] | None = None):
    """Run once: download 90 days of history for all pair × timeframe combinations."""
    if timeframes is None:
        timeframes = ["1m", "5m", "15m"]
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        for tf in timeframes:
            path = _parquet_path(pair, tf)
            if path.exists():
                logger.info(f"Skipping {pair} {tf}, already exists")
                continue
            df = await download_historical(pair, tf, days_back=90)
            df.to_parquet(path, engine="pyarrow", compression="snappy")
            logger.info(f"Saved {pair} {tf}: {len(df)} rows")
```

- [ ] **Step 3: Test the loader**

```bash
python3 -c "
import asyncio
from ml.data_loader import download_historical

async def main():
    df = await download_historical('SUIUSDT', '5m', days_back=1)
    print(f'Rows: {len(df)}')
    print(df.head())
    print(df.dtypes)

asyncio.run(main())
"
```

Expected: ~288 rows (1 day × 288 5m candles).

- [ ] **Step 4: Commit**

```bash
git add ml/__init__.py ml/data_loader.py
git commit -m "feat(ml): add historical OHLCV data loader with parquet storage"
```

---

## Task 3: Feature Engineering

**Files:**
- Create: `ml/features.py`
- Create: `tests/test_ml_features.py`

- [ ] **Step 1: Implement ml/features.py**

```python
"""Feature extraction for ML models. 39 features per observation."""

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pandas_ta as ta

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
    target_index: int | None = None,
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
```

- [ ] **Step 2: Create tests/test_ml_features.py**

```python
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
```

- [ ] **Step 3: Run tests**

```bash
cd /Users/gastonchevarria/Alpha/agent-god-2
python3 -m pytest tests/test_ml_features.py -v
```

Expected: 5/5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add ml/features.py tests/test_ml_features.py
git commit -m "feat(ml): add feature engineering with 39 features and leakage tests"
```

---

## Task 4: Label Generation

**Files:**
- Create: `ml/labels.py`
- Create: `tests/test_ml_labels.py`

- [ ] **Step 1: Implement ml/labels.py**

```python
"""Label generation for ML models. Uses future data, so only for training."""

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
```

- [ ] **Step 2: Create tests/test_ml_labels.py**

```python
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
```

- [ ] **Step 3: Run tests**

```bash
python3 -m pytest tests/test_ml_labels.py -v
```

Expected: 5/5 pass.

- [ ] **Step 4: Commit**

```bash
git add ml/labels.py tests/test_ml_labels.py
git commit -m "feat(ml): add label generation for regimes, volatility, and EV"
```

---

## Task 5: Walk-Forward Validation Utility

**Files:**
- Create: `ml/validation.py`

- [ ] **Step 1: Implement ml/validation.py**

```python
"""Walk-forward validation for time-series ML."""

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
```

- [ ] **Step 2: Commit**

```bash
git add ml/validation.py
git commit -m "feat(ml): add walk-forward validation utilities"
```

---

## Task 6: Model Store (Persistence)

**Files:**
- Create: `ml/model_store.py`

- [ ] **Step 1: Implement ml/model_store.py**

```python
"""Load/save ML models to disk + optional Google Cloud Storage."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib

from config.settings import settings

logger = logging.getLogger(__name__)

_MODELS_DIR = Path("models")
_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _model_path(name: str) -> Path:
    return _MODELS_DIR / f"{name}.joblib"


def _meta_path(name: str) -> Path:
    return _MODELS_DIR / f"{name}.meta.json"


def save_model(name: str, model, metadata: dict | None = None):
    """Save a model locally and optionally to GCS."""
    path = _model_path(name)
    joblib.dump(model, path)

    meta = metadata or {}
    meta["saved_at"] = datetime.now(timezone.utc).isoformat()
    meta["model_name"] = name
    _meta_path(name).write_text(json.dumps(meta, indent=2))

    logger.info(f"Saved model {name} to {path}")

    # Optional: upload to GCS if configured
    try:
        _upload_to_gcs(name)
    except Exception as e:
        logger.warning(f"GCS upload skipped for {name}: {e}")


def load_model(name: str):
    """Load a model from disk. Returns None if missing."""
    path = _model_path(name)
    if not path.exists():
        # Try downloading from GCS
        try:
            _download_from_gcs(name)
        except Exception as e:
            logger.info(f"Model {name} not available locally or in GCS: {e}")
            return None

    if not path.exists():
        return None

    return joblib.load(path)


def load_metadata(name: str) -> dict:
    """Load model metadata."""
    path = _meta_path(name)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def list_models() -> list[dict]:
    """List all available models with metadata."""
    models = []
    for f in _MODELS_DIR.glob("*.joblib"):
        name = f.stem
        models.append({
            "name": name,
            "path": str(f),
            "size_kb": round(f.stat().st_size / 1024, 1),
            "metadata": load_metadata(name),
        })
    return models


def _upload_to_gcs(name: str):
    """Upload model and metadata to GCS bucket."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(settings.ML_GCS_BUCKET)

    for ext in ["joblib", "meta.json"]:
        local = _MODELS_DIR / f"{name}.{ext}"
        if local.exists():
            blob = bucket.blob(f"models/{name}.{ext}")
            blob.upload_from_filename(str(local))


def _download_from_gcs(name: str):
    """Download model and metadata from GCS bucket."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(settings.ML_GCS_BUCKET)

    for ext in ["joblib", "meta.json"]:
        blob = bucket.blob(f"models/{name}.{ext}")
        local = _MODELS_DIR / f"{name}.{ext}"
        if blob.exists():
            blob.download_to_filename(str(local))
```

- [ ] **Step 2: Commit**

```bash
git add ml/model_store.py
git commit -m "feat(ml): add model persistence with optional GCS backup"
```

---

## Task 7: Regime Classifier

**Files:**
- Create: `ml/regime_classifier.py`

- [ ] **Step 1: Implement ml/regime_classifier.py**

```python
"""Regime Classifier — predicts regime over next 15 minutes."""

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
```

- [ ] **Step 2: Commit**

```bash
git add ml/regime_classifier.py
git commit -m "feat(ml): add regime classifier with XGBoost"
```

---

## Task 8: Volatility Predictor

**Files:**
- Create: `ml/volatility_predictor.py`

- [ ] **Step 1: Implement ml/volatility_predictor.py**

```python
"""Volatility Predictor — predicts realized volatility over next 30 min."""

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
```

- [ ] **Step 2: Commit**

```bash
git add ml/volatility_predictor.py
git commit -m "feat(ml): add volatility predictor with XGBoost"
```

---

## Task 9: Expected-Value Model (per strategy)

**Files:**
- Create: `ml/ev_model.py`

- [ ] **Step 1: Implement ml/ev_model.py**

```python
"""Expected-Value Model — one XGBoost regressor per strategy."""

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
```

- [ ] **Step 2: Commit**

```bash
git add ml/ev_model.py
git commit -m "feat(ml): add per-strategy EV model"
```

---

## Task 10: Inference API (with caching)

**Files:**
- Create: `ml/inference.py`

- [ ] **Step 1: Implement ml/inference.py**

```python
"""Runtime inference API used by tournament runner."""

import asyncio
import logging
import time
from typing import Any

from config.settings import settings
from core.data_fetcher import fetch_ohlcv, get_taker_volume
from ml.ev_model import predict_ev
from ml.features import compute_features_from_df, enrich_cross_pair
from ml.regime_classifier import predict_regime
from ml.volatility_predictor import predict_volatility

logger = logging.getLogger(__name__)

_cache: dict[str, tuple[float, Any]] = {}
_TTL = settings.ML_INFERENCE_CACHE_SECONDS


def _cached(key: str):
    if key in _cache:
        ts, val = _cache[key]
        if time.time() - ts < _TTL:
            return val
    return None


def _set_cache(key: str, val: Any):
    _cache[key] = (time.time(), val)


async def _get_features(pair: str, timeframe: str = "5m") -> dict:
    key = f"features:{pair}:{timeframe}"
    cached = _cached(key)
    if cached is not None:
        return cached

    df = await fetch_ohlcv(pair, timeframe, limit=100)
    try:
        tv = await get_taker_volume(pair, timeframe, limit=len(df))
        df["taker_buy_base"] = tv["taker_buy_vol"].values
    except Exception:
        df["taker_buy_base"] = df["volume"] / 2

    features = compute_features_from_df(df)
    _set_cache(key, features)
    return features


async def get_regime_prediction(pair: str) -> dict:
    if not settings.ML_ENABLED:
        return {"regime": "UNKNOWN", "confidence": 0.0}

    key = f"regime:{pair}"
    cached = _cached(key)
    if cached is not None:
        return cached

    features = await _get_features(pair, "5m")
    result = predict_regime(features)
    _set_cache(key, result)
    return result


async def get_expected_value(strategy_id: str, pair: str) -> dict:
    if not settings.ML_ENABLED:
        return {"expected_pnl_usd": None, "confidence": 0.0}

    key = f"ev:{strategy_id}:{pair}"
    cached = _cached(key)
    if cached is not None:
        return cached

    features = await _get_features(pair, "5m")
    result = predict_ev(strategy_id, features)
    _set_cache(key, result)
    return result


async def get_volatility(pair: str) -> dict:
    if not settings.ML_ENABLED:
        return {"predicted_vol_pct": 0.0, "vol_ratio": 1.0}

    key = f"vol:{pair}"
    cached = _cached(key)
    if cached is not None:
        return cached

    features = await _get_features(pair, "5m")
    result = predict_volatility(pair, features)
    _set_cache(key, result)
    return result


def invalidate_cache():
    """Called after retraining to clear cached predictions."""
    _cache.clear()
    logger.info("ML inference cache invalidated")
```

- [ ] **Step 2: Commit**

```bash
git add ml/inference.py
git commit -m "feat(ml): add inference API with 30s caching"
```

---

## Task 11: Training Pipeline (Orchestrator)

**Files:**
- Create: `ml/training_pipeline.py`

- [ ] **Step 1: Implement ml/training_pipeline.py**

```python
"""Orchestrates full retraining of all ML models."""

import asyncio
import logging
from datetime import datetime, timezone

from config.settings import settings
from ml.data_loader import update_incremental
from ml.ev_model import train_ev_model
from ml.inference import invalidate_cache
from ml.regime_classifier import train_regime_classifier
from ml.volatility_predictor import train_volatility_predictor

logger = logging.getLogger(__name__)

STRATEGY_IDS = [f"G-{i:02d}" for i in range(1, 14)]


async def run_full_training() -> dict:
    """Full retraining pipeline. Returns summary of all models trained."""
    start = datetime.now(timezone.utc)
    summary = {"started_at": start.isoformat(), "models": {}}

    pairs = settings.pairs_list
    logger.info(f"=== Starting full ML retraining for pairs: {pairs} ===")

    # 1. Update historical data
    logger.info("Step 1/4: Updating historical data")
    pair_dfs = {}
    for pair in pairs:
        try:
            df = await update_incremental(pair, "5m")
            pair_dfs[pair] = df
            logger.info(f"  {pair}: {len(df)} rows")
        except Exception as e:
            logger.error(f"  Failed to update {pair}: {e}")

    if len(pair_dfs) < 2:
        summary["error"] = "Insufficient data updated"
        return summary

    # 2. Train Regime Classifier
    logger.info("Step 2/4: Training Regime Classifier")
    try:
        metrics = train_regime_classifier(pair_dfs)
        summary["models"]["regime_classifier"] = {"status": "ok", "metrics": metrics}
    except Exception as e:
        logger.error(f"Regime classifier failed: {e}")
        summary["models"]["regime_classifier"] = {"status": "failed", "error": str(e)}

    # 3. Train Volatility Predictor
    logger.info("Step 3/4: Training Volatility Predictor")
    try:
        metrics = train_volatility_predictor(pair_dfs)
        summary["models"]["volatility_predictor"] = {"status": "ok", "metrics": metrics}
    except Exception as e:
        logger.error(f"Volatility predictor failed: {e}")
        summary["models"]["volatility_predictor"] = {"status": "failed", "error": str(e)}

    # 4. Train EV models (one per strategy)
    logger.info("Step 4/4: Training EV models per strategy")
    for sid in STRATEGY_IDS:
        try:
            metrics = train_ev_model(sid, pair_dfs)
            summary["models"][f"ev_{sid}"] = {"status": "ok", "metrics": metrics}
        except Exception as e:
            logger.warning(f"EV model for {sid} failed: {e}")
            summary["models"][f"ev_{sid}"] = {"status": "failed", "error": str(e)}

    invalidate_cache()

    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    summary["duration_seconds"] = (datetime.now(timezone.utc) - start).total_seconds()

    logger.info(f"=== Retraining complete in {summary['duration_seconds']:.0f}s ===")
    return summary


async def initial_setup():
    """First-time setup: backfill data + train all models."""
    from ml.data_loader import initial_backfill
    logger.info("=== Initial ML setup ===")
    await initial_backfill(settings.pairs_list, timeframes=["5m"])
    return await run_full_training()
```

- [ ] **Step 2: Commit**

```bash
git add ml/training_pipeline.py
git commit -m "feat(ml): add training pipeline orchestrator"
```

---

## Task 12: Integration with Tournament Runner

**Files:**
- Modify: `strategies/base_strategy_v4.py`
- Modify: `scheduler/tournament_runner_god2.py`

- [ ] **Step 1: Add `open_position_with_custom_params` to BaseStrategyV4**

Add this method to `BaseStrategyV4` in `strategies/base_strategy_v4.py`, right after the existing `open_position` method:

```python
    def open_position_with_custom_params(
        self,
        direction: str,
        price: float,
        pair: str,
        signals: dict,
        tp_pct: float | None = None,
        sl_pct: float | None = None,
    ):
        """Same as open_position but allows runtime TP/SL override (used by ML layer)."""
        effective_tp = tp_pct if tp_pct is not None else self.cfg.tp_pct
        effective_sl = sl_pct if sl_pct is not None else self.cfg.sl_pct

        margin = self._effective_margin()
        notional = margin * self.cfg.leverage
        mult = 1 if direction == "LONG" else -1

        self.tp_abs = price * (1 + effective_tp * mult)
        self.sl_abs = price * (1 - effective_sl * mult)
        self.position = direction

        self._entry_price = price
        self._entry_dir = direction
        self._entry_pair = pair
        self._entry_sig = {**signals, "ml_tp_pct": effective_tp, "ml_sl_pct": effective_sl}
        from datetime import datetime, timezone
        self._entry_time = datetime.now(timezone.utc)
        self._entry_margin = margin

        self._log("OPEN", direction, price, None, None, self._entry_sig, margin, pair)
```

- [ ] **Step 2: Integrate ML calls in TournamentRunnerGOD2._run_strategy**

In `scheduler/tournament_runner_god2.py`, add these imports at the top:

```python
from config.settings import settings
from ml import inference as ml_inference
```

Then modify `_run_strategy` to use ML predictions. Find the existing `_run_strategy` method and replace the block that opens positions. Specifically, replace:

```python
        if result:
            pair = result["pair"]
            signal = result["signal"]
            price = await get_current_price(pair)
            strat.open_position(signal.direction, price, pair, signal.signals)
            logger.info(
                f"[{strat.cfg.id}] OPEN {signal.direction} {pair} @ {price:.2f} "
                f"(conf={result['effective_confidence']:.2f})"
            )
```

With:

```python
        if result:
            pair = result["pair"]
            signal = result["signal"]

            # ML checks (if enabled)
            if settings.ML_ENABLED:
                # 1. ML regime check
                ml_regime = await ml_inference.get_regime_prediction(pair)
                if ml_regime["confidence"] >= settings.ML_REGIME_CONFIDENCE_THRESHOLD:
                    if "ANY" not in strat.cfg.regime_filter and ml_regime["regime"] not in strat.cfg.regime_filter:
                        strat.skip_count += 1
                        logger.info(f"[{strat.cfg.id}] ML regime mismatch: predicted {ml_regime['regime']}, filter {strat.cfg.regime_filter}")
                        return

                # 2. EV check
                ev = await ml_inference.get_expected_value(strat.cfg.id, pair)
                if ev.get("expected_pnl_usd") is not None and ev.get("n_samples_trained_on", 0) >= settings.ML_MIN_TRADES_FOR_EV_MODEL:
                    if ev["expected_pnl_usd"] < settings.ML_MIN_EV_USD:
                        strat.skip_count += 1
                        logger.info(f"[{strat.cfg.id}] ML EV too low: ${ev['expected_pnl_usd']:.2f} < ${settings.ML_MIN_EV_USD}")
                        return

                # 3. Volatility adjustment
                vol_pred = await ml_inference.get_volatility(pair)
                vol_ratio = max(settings.ML_VOL_RATIO_MIN, min(settings.ML_VOL_RATIO_MAX, vol_pred.get("vol_ratio", 1.0)))
                adjusted_tp = strat.cfg.tp_pct * vol_ratio
                adjusted_sl = strat.cfg.sl_pct * vol_ratio
            else:
                adjusted_tp = strat.cfg.tp_pct
                adjusted_sl = strat.cfg.sl_pct

            price = await get_current_price(pair)
            strat.open_position_with_custom_params(
                signal.direction, price, pair, signal.signals,
                tp_pct=adjusted_tp, sl_pct=adjusted_sl,
            )
            logger.info(
                f"[{strat.cfg.id}] OPEN {signal.direction} {pair} @ {price:.2f} "
                f"(conf={result['effective_confidence']:.2f}, tp={adjusted_tp*100:.2f}%, sl={adjusted_sl*100:.2f}%)"
            )
```

- [ ] **Step 3: Add weekly retrain scheduler job**

In `scheduler/tournament_runner_god2.py`, inside the `start` method, add:

```python
        # ML retraining job (weekly)
        if settings.ML_ENABLED:
            self.scheduler.add_job(
                self._run_ml_retrain,
                "cron",
                day_of_week=settings.ML_RETRAIN_DAY,
                hour=settings.ML_RETRAIN_HOUR_UTC,
                minute=0,
                id="ml_retrain",
            )
```

Add the new method inside the class:

```python
    async def _run_ml_retrain(self):
        """Weekly ML model retraining."""
        logger.info("=== Weekly ML retrain triggered ===")
        try:
            from ml.training_pipeline import run_full_training
            summary = await run_full_training()
            logger.info(f"ML retrain summary: {summary}")
            from core.memory_tiers import add_memory
            add_memory(
                "long",
                f"[ML RETRAIN] Completed in {summary.get('duration_seconds', 0):.0f}s. Models: {list(summary.get('models', {}).keys())}",
                tags=["ml_retrain"],
            )
        except Exception as e:
            logger.error(f"ML retrain failed: {e}")
```

- [ ] **Step 4: Run existing tests to ensure no regression**

```bash
cd /Users/gastonchevarria/Alpha/agent-god-2
python3 -m pytest tests/ -v --tb=short
```

Expected: all pre-existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add strategies/base_strategy_v4.py scheduler/tournament_runner_god2.py
git commit -m "feat(ml): integrate ML inference + retrain into tournament runner"
```

---

## Task 13: API Endpoints

**Files:**
- Modify: `main_god2.py`

- [ ] **Step 1: Add ML endpoints to main_god2.py**

Add the following router section at the end of `main_god2.py`, just before the last line:

```python
# --- ML Endpoints ---

@router_god2.get("/ml/status")
async def ml_status():
    from ml.model_store import list_models
    from config.settings import settings
    return {
        "enabled": settings.ML_ENABLED,
        "models": list_models(),
        "cache_ttl_seconds": settings.ML_INFERENCE_CACHE_SECONDS,
    }


@router_god2.get("/ml/regime/{pair}")
async def ml_regime(pair: str):
    from ml.inference import get_regime_prediction
    return await get_regime_prediction(pair)


@router_god2.get("/ml/ev/{strategy_id}/{pair}")
async def ml_ev(strategy_id: str, pair: str):
    from ml.inference import get_expected_value
    return await get_expected_value(strategy_id, pair)


@router_god2.get("/ml/volatility/{pair}")
async def ml_volatility(pair: str):
    from ml.inference import get_volatility
    return await get_volatility(pair)


@router_god2.post("/ml/retrain")
async def ml_retrain_trigger(runner=Depends(get_runner)):
    import asyncio
    asyncio.create_task(runner._run_ml_retrain())
    return {"status": "retrain_started"}


@router_god2.get("/ml/training-history")
async def ml_training_history():
    from core.memory_tiers import get_recent
    entries = get_recent("long", limit=50)
    return [e for e in entries if "ml_retrain" in e.get("tags", [])]
```

- [ ] **Step 2: Test the endpoints**

Start the server and test:

```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 9090 &
sleep 3
curl -s http://localhost:9090/ml/status | python3 -m json.tool
curl -s http://localhost:9090/ml/regime/SUIUSDT | python3 -m json.tool
kill %1
```

- [ ] **Step 3: Commit**

```bash
git add main_god2.py
git commit -m "feat(ml): add ML status and inference API endpoints"
```

---

## Task 14: Dashboard "ML Insights" View

**Files:**
- Modify: `static/dashboard.html`

- [ ] **Step 1: Add the 7th view to dashboard.html**

Locate the sidebar navigation in `static/dashboard.html` and add a new entry after the Live Monitor:

```html
            <li data-view="ml">ML Insights</li>
```

Locate the main view area and add a new view container:

```html
<div id="view-ml" class="view hidden">
  <h2>ML Insights</h2>
  <div class="card">
    <h3>Regime Predictions</h3>
    <div id="ml-regimes" class="grid grid-3"></div>
  </div>
  <div class="card mt-16">
    <h3>Expected Value by Strategy</h3>
    <table class="data-table">
      <thead><tr><th>Strategy</th><th>Pair</th><th>Expected PnL</th><th>Samples</th></tr></thead>
      <tbody id="ml-ev-body"></tbody>
    </table>
  </div>
  <div class="card mt-16">
    <h3>Volatility Predictions</h3>
    <table class="data-table">
      <thead><tr><th>Pair</th><th>Predicted Vol</th><th>Baseline</th><th>Ratio</th></tr></thead>
      <tbody id="ml-vol-body"></tbody>
    </table>
  </div>
  <div class="card mt-16">
    <h3>Training History</h3>
    <div id="ml-training-history"></div>
  </div>
</div>
```

Add JavaScript to fetch and render ML data (paste at the end of the main `<script>` section):

```javascript
async function fetchMLView() {
  const pairs = ['WIFUSDT', '1000BONKUSDT', '1000PEPEUSDT', '1000SHIBUSDT', 'SUIUSDT', 'ORDIUSDT'];

  // Regime predictions
  const regimesHtml = [];
  for (const pair of pairs) {
    const data = await apiFetch(`/ml/regime/${pair}`);
    if (data) {
      const regime = data.regime || 'UNKNOWN';
      const conf = (data.confidence * 100).toFixed(0);
      const color = regime === 'TRENDING_UP' ? 'text-green' : regime === 'TRENDING_DOWN' ? 'text-red' : regime === 'VOLATILE' ? 'text-yellow' : 'text-dim';
      regimesHtml.push(`<div class="stat-card">
        <div class="text-dim">${pair.replace('USDT','').replace('1000','')}</div>
        <div class="${color}" style="font-weight:600;font-size:1.1rem">${regime}</div>
        <div class="text-dim" style="font-size:.8rem">Confidence: ${conf}%</div>
      </div>`);
    }
  }
  document.getElementById('ml-regimes').innerHTML = regimesHtml.join('');

  // EV predictions for active strategies (one row per strategy on its best pair)
  const evRows = [];
  for (const s of leaderboardData) {
    const defaultPair = 'SUIUSDT';
    const ev = await apiFetch(`/ml/ev/${s.id}/${defaultPair}`);
    if (ev) {
      const pnl = ev.expected_pnl_usd;
      const samples = ev.n_samples_trained_on || 0;
      const pnlTxt = pnl === null ? '--' : (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
      const pnlClass = pnl === null ? 'text-dim' : (pnl > 0 ? 'text-green' : 'text-red');
      evRows.push(`<tr><td>${s.id} ${s.name}</td><td>${defaultPair}</td><td class="${pnlClass}">${pnlTxt}</td><td>${samples}</td></tr>`);
    }
  }
  document.getElementById('ml-ev-body').innerHTML = evRows.join('') || '<tr><td colspan="4" class="text-dim text-center">No EV models trained yet</td></tr>';

  // Volatility
  const volRows = [];
  for (const pair of pairs) {
    const vol = await apiFetch(`/ml/volatility/${pair}`);
    if (vol) {
      const pred = (vol.predicted_vol_pct * 100).toFixed(3);
      const base = (vol.baseline_vol_pct * 100).toFixed(3);
      const ratio = vol.vol_ratio || 1;
      const ratioClass = ratio > 1.1 ? 'text-red' : ratio < 0.9 ? 'text-green' : 'text-dim';
      volRows.push(`<tr><td>${pair}</td><td>${pred}%</td><td>${base}%</td><td class="${ratioClass}">${ratio.toFixed(2)}x</td></tr>`);
    }
  }
  document.getElementById('ml-vol-body').innerHTML = volRows.join('') || '<tr><td colspan="4" class="text-dim text-center">No volatility model trained yet</td></tr>';

  // Training history
  const history = await apiFetch('/ml/training-history');
  if (history && history.length) {
    document.getElementById('ml-training-history').innerHTML = history.map(h => 
      `<div style="padding:8px;border-bottom:1px solid #334155"><code>${h.ts.slice(0,19)}</code> — ${h.content}</div>`
    ).join('');
  } else {
    document.getElementById('ml-training-history').innerHTML = '<div class="text-dim">No training history yet</div>';
  }
}
```

Register the view in the view switching code. Find where other views are listed and add:

```javascript
if (view === 'ml') fetchMLView();
```

- [ ] **Step 2: Commit**

```bash
git add static/dashboard.html
git commit -m "feat(ml): add ML Insights view to dashboard"
```

---

## Task 15: Initial Model Training (one-time setup)

**Files:**
- None (operational step)

- [ ] **Step 1: Run initial backfill and training**

Create and run this one-time script:

```bash
cd /Users/gastonchevarria/Alpha/agent-god-2
python3 -c "
import asyncio
from ml.training_pipeline import initial_setup

async def main():
    summary = await initial_setup()
    print('=== INITIAL SETUP COMPLETE ===')
    for name, info in summary.get('models', {}).items():
        print(f'  {name}: {info.get(\"status\")} — metrics: {info.get(\"metrics\")}')

asyncio.run(main())
"
```

Expected duration: 30-60 minutes (data download + training of 15 models).

- [ ] **Step 2: Verify models exist**

```bash
ls -lh models/
```

Expected: `regime_classifier.joblib`, `volatility_predictor.joblib`, and several `ev_G*.joblib` files (only those strategies with >= 50 trades).

- [ ] **Step 3: Commit the training data**

```bash
git add data/ml_historical/
git commit -m "data(ml): initial historical OHLCV backfill (90 days)"
```

Note: models are NOT committed (they're large and regenerable). They're persisted in GCS bucket for Cloud Run instances.

---

## Task 16: Deploy to Cloud Run

- [ ] **Step 1: Create GCS bucket for model persistence**

```bash
gcloud storage buckets create gs://agent-god-2-data \
  --project=proyecto001-490716 \
  --location=us-central1 \
  --uniform-bucket-level-access
```

- [ ] **Step 2: Grant Cloud Run service account access**

```bash
PROJECT_ID=proyecto001-490716
SA=$(gcloud run services describe agent-god-2 --project $PROJECT_ID --region us-central1 --format='value(spec.template.spec.serviceAccountName)')
if [ -z "$SA" ]; then
  SA="${PROJECT_ID}-compute@developer.gserviceaccount.com"
fi
gcloud storage buckets add-iam-policy-binding gs://agent-god-2-data \
  --member="serviceAccount:$SA" \
  --role="roles/storage.objectAdmin"
```

- [ ] **Step 3: Upload models to GCS**

```bash
gcloud storage cp models/*.joblib gs://agent-god-2-data/models/
gcloud storage cp models/*.meta.json gs://agent-god-2-data/models/
```

- [ ] **Step 4: Redeploy**

```bash
cd /Users/gastonchevarria/Alpha/agent-god-2
ANTHROPIC_KEY=$(grep "^ANTHROPIC_API_KEY=" .env | cut -d'=' -f2)
gcloud run deploy agent-god-2 \
  --source . \
  --project proyecto001-490716 \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --min-instances 1 \
  --max-instances 1 \
  --set-env-vars "^@^ANTHROPIC_API_KEY=${ANTHROPIC_KEY}@MODE=paper@LOG_LEVEL=INFO@PAIRS=WIFUSDT,1000BONKUSDT,1000PEPEUSDT,1000SHIBUSDT,SUIUSDT,ORDIUSDT@ML_ENABLED=true" \
  --port 9090
```

Note: memory increased to 2Gi and CPU to 2 to accommodate ML inference.

- [ ] **Step 5: Verify deployment**

```bash
curl -s https://agent-god-2-656958691766.us-central1.run.app/ml/status | python3 -m json.tool
curl -s https://agent-god-2-656958691766.us-central1.run.app/ml/regime/SUIUSDT | python3 -m json.tool
curl -s https://agent-god-2-656958691766.us-central1.run.app/ml/volatility/ORDIUSDT | python3 -m json.tool
```

Expected: All return valid JSON with predictions.

---

## Task 17: Shadow Mode Monitoring

Before enabling ML-driven decisions, run in "shadow mode" for 7 days to validate predictions don't break existing flows.

- [ ] **Step 1: Set shadow mode**

```bash
gcloud run services update agent-god-2 \
  --project proyecto001-490716 \
  --region us-central1 \
  --update-env-vars "ML_ENABLED=false"
```

This computes ML predictions (for monitoring) but doesn't use them in decisions.

- [ ] **Step 2: After 7 days, check metrics**

```bash
curl -s https://agent-god-2-656958691766.us-central1.run.app/ml/training-history | python3 -m json.tool
curl -s https://agent-god-2-656958691766.us-central1.run.app/tournament/leaderboard | python3 -m json.tool
```

Compare predicted regimes vs actual outcomes over the 7-day window. If prediction quality looks reasonable:

```bash
gcloud run services update agent-god-2 \
  --project proyecto001-490716 \
  --region us-central1 \
  --update-env-vars "ML_ENABLED=true"
```

---

## Task 18: Success Metrics Tracking

After 30 days of ML-enabled operation, compare to pre-ML baseline:

- [ ] **Step 1: Pull 30-day metrics**

```bash
curl -s https://agent-god-2-656958691766.us-central1.run.app/tournament/portfolio > /tmp/ml_30d.json
curl -s https://agent-god-2-656958691766.us-central1.run.app/tournament/leaderboard > /tmp/ml_30d_lb.json
```

Manually compare to the 30-day window before enabling ML:

| Metric | Before ML | After ML | Target |
|:-------|:----------|:---------|:-------|
| Portfolio Sharpe | ? | ? | +20% |
| Win Rate (avg) | ? | ? | +3 pts |
| Profit Factor (avg) | ? | ? | +15% |
| Max Drawdown | ? | ? | not worse |
| Eliminated strategies | ? | ? | not more |

- [ ] **Step 2: Decision**

If 3+ metrics improve and safety metrics don't degrade: ML layer stays enabled.
If metrics are flat or worse: disable `ML_ENABLED`, investigate, iterate.

---

## Appendix: Troubleshooting

**"Models directory empty on Cloud Run"**
- Check GCS permissions: `gcloud storage buckets describe gs://agent-god-2-data`
- Verify service account has `storage.objectAdmin` role on the bucket
- Check logs: `gcloud logging read "resource.type=cloud_run_revision" --limit 20 --project proyecto001-490716`

**"Features contain NaN"**
- Run feature extraction tests: `pytest tests/test_ml_features.py -v`
- Verify input OHLCV data has no gaps: check parquet files are non-empty

**"EV model skipped for most strategies"**
- Strategies need >= 50 closed trades before EV model trains
- Wait until strategies accumulate history, or lower `ML_MIN_TRADES_FOR_EV_MODEL`

**"Regime classifier accuracy is <60%"**
- Model might be underfitting — increase `n_estimators` to 300
- Or features may be insufficient — review feature importance with `model.feature_importances_`
- Most likely cause: labels are noisy because regime rules are inherently fuzzy

**"Inference too slow (>100ms)"**
- Check cache is working: inspect `_cache` in `ml/inference.py`
- Reduce `ML_INFERENCE_CACHE_SECONDS` if data is stale but increase if latency is the issue
- Profile with `cProfile` to find hotspot

**"Retraining takes >2 hours"**
- Reduce `ML_HISTORICAL_DAYS` from 90 to 60
- Reduce `n_estimators` in model factories from 200 to 100
- Consider moving retraining to a separate Cloud Run Job instead of inline
