# Agent GOD 2 — ML Layer Design Specification

**Date**: 2026-04-18
**Project**: ML augmentation layer for Agent GOD 2
**Location**: `/Users/gastonchevarria/Alpha/agent-god-2/ml/`
**Status**: Design approved, pending implementation

---

## 1. Overview

The ML Layer is a set of three machine learning models that augment Agent GOD 2's 13 strategies without replacing them. The goal is to improve the signal-to-noise ratio of trade decisions by adding data-driven filters and dynamic parameter adjustments.

**Scope**: This is NOT an attempt to "predict price." Markets are non-stationary and near-martingale — predicting direction with >60% accuracy is not possible without data leakage. Instead, we predict **tractable** quantities that improve trade quality:

1. **Market regime** (much easier than direction)
2. **Expected PnL per strategy** (conditional on current state)
3. **Realized volatility** (for dynamic TP/SL)

**Non-goals**:
- Replace the 13 strategies (they stay)
- Replace the Claude Opus Coordinator (it uses ML predictions as input)
- Predict price direction (low ROI, high risk of overfitting)

---

## 2. The Three Models

### 2.1 Regime Classifier
**Task**: Given current market features for a pair, predict which regime it will be in over the next 15 minutes.

**Output classes**: `TRENDING_UP`, `TRENDING_DOWN`, `RANGING`, `VOLATILE`

**Why**: The current regime detector uses a simple EMA + ADX heuristic. An ML model trained on historical data can identify regime transitions 15-30 minutes earlier.

**Target accuracy**: 70-75% (vs ~60% for the rule-based detector)

**Algorithm**: XGBoost classifier (multi-class, softmax output)

**Training frequency**: Weekly retraining on rolling 90-day window

### 2.2 Expected-Value Model (per strategy)
**Task**: Given current market features, predict expected PnL (in USD) if strategy X opens a position right now.

**Output**: Continuous value (USD), one model per strategy (13 models)

**Why**: Replace the fixed 0.72 confidence threshold with a dynamic threshold based on expected EV. If EV < $1 (fees cost more than edge) → skip. If EV > $10 → higher conviction.

**Target metric**: Mean Absolute Error < $3 per trade

**Algorithm**: XGBoost regressor (one per strategy)

**Training frequency**: Weekly retraining; each strategy trained on its own trade history

### 2.3 Volatility Predictor
**Task**: Given current market features, predict realized volatility (standard deviation of returns) over the next N minutes.

**Output**: Continuous value (% volatility), one value per pair × timeframe

**Why**: Static TP (e.g. 0.35%) doesn't make sense in all conditions. If predicted vol for next 30 min is 0.1%, a 0.35% TP is too wide. If predicted vol is 1.5%, a 0.35% TP fills in seconds.

**Target metric**: R² > 0.40 (volatility is partially predictable)

**Algorithm**: XGBoost regressor (single global model with pair as a feature)

**Training frequency**: Weekly retraining on rolling 60-day window

---

## 3. Architecture

### 3.1 Integration Point

The ML Layer lives inside the Agent GOD 2 container as a new `ml/` package. It's NOT a separate microservice because:
- Model inference is fast (<50ms per prediction)
- Scaling requirements match the main bot
- Simpler deployment, no network overhead

```
agent-god-2/
├── ml/                             # NEW
│   ├── __init__.py
│   ├── data_loader.py              # Download/cache historical OHLCV
│   ├── features.py                 # Feature engineering (shared)
│   ├── regime_classifier.py        # Model 1: regime prediction
│   ├── ev_model.py                 # Model 2: per-strategy EV
│   ├── volatility_predictor.py     # Model 3: vol prediction
│   ├── training_pipeline.py        # Orchestrates training for all 3
│   ├── model_store.py              # Load/save models from disk/GCS
│   ├── inference.py                # Runtime inference API (used by runner)
│   └── retrain_scheduler.py        # Weekly retrain job
├── data/
│   └── ml_historical/              # Parquet files of OHLCV history
│       ├── WIFUSDT_1m.parquet
│       ├── WIFUSDT_5m.parquet
│       └── ...
├── models/                         # NEW (saved model artifacts)
│   ├── regime_classifier.joblib
│   ├── volatility_predictor.joblib
│   ├── ev_G01.joblib
│   ├── ev_G02.joblib
│   └── ...
└── ml_requirements.txt             # Additional deps for ML
```

### 3.2 Inference Flow

When a strategy is about to decide whether to open a position:

```
1. Strategy evaluates signals (existing flow)
2. Pair Selector picks best pair
3. NEW: inference.get_regime_prediction(pair) → {regime, confidence}
4. NEW: inference.get_expected_value(strategy_id, features) → ev_usd
5. NEW: inference.get_volatility(pair, horizon=30min) → predicted_vol
6. Runner decides:
   - If predicted regime conflicts with strategy's regime_filter → skip
   - If EV < min_ev_threshold → skip
   - Dynamic TP = base_tp * (predicted_vol / baseline_vol)
7. Open position with adjusted parameters
```

### 3.3 Training Flow

Runs weekly (Sunday 03:00 UTC):

```
1. Download new OHLCV data for all 6 pairs (incremental)
2. Update parquet files (append new data, drop old data > 180 days)
3. Extract features for entire dataset
4. Train Regime Classifier (90-day window)
5. For each of 13 strategies: train EV model (using trades.jsonl)
6. Train Volatility Predictor (60-day window)
7. Run validation (time-based holdout)
8. If new model beats current on validation → swap
9. Otherwise keep current model, log warning
10. Upload new models to GCS bucket for persistence
```

---

## 4. Feature Engineering

All three models share the same feature extraction pipeline. Features are computed on-demand at inference time and cached for 30 seconds.

### 4.1 Features per Pair (at a given timestamp)

**Price-based (12 features)**:
- `return_1m`, `return_5m`, `return_15m`, `return_1h`: log returns
- `volatility_5m`, `volatility_30m`: rolling std of 1m returns
- `price_z_20`: z-score of close vs 20-period mean
- `distance_from_ema20`: (close - ema20) / ema20
- `distance_from_ema50`: (close - ema50) / ema50
- `ema_spread`: (ema20 - ema50) / ema50
- `high_low_range_5m`: (max_high - min_low) / close over 5 bars

**Momentum (8 features)**:
- `rsi_7`, `rsi_14`: RSI at two windows
- `rsi_14_change_5m`: delta in RSI over 5 minutes
- `macd_hist`: MACD histogram value
- `macd_hist_change`: delta in hist
- `stoch_k`, `stoch_d`: stochastic values
- `adx`: ADX indicator

**Volume (6 features)**:
- `volume_ratio_5m`, `volume_ratio_20m`: current vol / rolling avg
- `taker_buy_ratio`: buy volume / total volume
- `taker_buy_ratio_5m_avg`: rolling average
- `volume_delta_5m`: sum of (buy - sell) vol over 5 bars
- `dollar_volume_1m`: close * volume

**Bollinger/Volatility (5 features)**:
- `bb_position`: (close - BB_mid) / BB_std
- `bb_width`: (BB_upper - BB_lower) / BB_mid
- `atr_pct`: ATR / close
- `atr_ratio`: current ATR / 20-period avg ATR
- `keltner_position`: (close - KC_mid) / KC_width

**Time-based (4 features)**:
- `hour_sin`, `hour_cos`: cyclical encoding of hour of day (UTC)
- `dow_sin`, `dow_cos`: cyclical encoding of day of week

**Cross-pair (4 features)**:
- `anchor_correlation`: 20-period correlation with ORDIUSDT
- `anchor_return_divergence`: own 5m return - ORDIUSDT's 5m return
- `market_divergence_index`: avg pairwise correlation across all 6 pairs (from correlation_engine)
- `relative_volume_rank`: this pair's volume vs other 5 pairs (1-6 rank normalized)

**Total: 39 features**

### 4.2 Feature Pipeline Contract

```python
class FeatureExtractor:
    async def extract(self, pair: str, timestamp: datetime | None = None) -> dict:
        """Returns dict of 39 features. If timestamp=None, uses current time."""
        ...
    
    async def extract_batch(self, pair: str, start_ts, end_ts) -> pd.DataFrame:
        """Used by training pipeline for historical feature extraction."""
        ...
```

---

## 5. Data Pipeline

### 5.1 Historical Data Download

Binance Futures provides historical klines via REST API. No cost, rate-limited to 1200 req/min.

**Strategy**:
- Initial backfill: 180 days of 1m data per pair = ~260k rows/pair
- Incremental updates: last 48h, appended to existing parquet
- Storage: Google Cloud Storage bucket `gs://agent-god-2-data/historical/`
- Local cache: `data/ml_historical/{pair}_{timeframe}.parquet`

**Rate limit management**:
- 1000 rows per request (max)
- 6 pairs × 3 timeframes × ~260 requests for initial backfill = 4,680 requests
- Spaced 100ms apart = ~8 minutes total
- Incremental updates: ~6 requests per run = negligible

### 5.2 Training Data Preparation

```python
def prepare_training_data(
    pair: str,
    timeframe: str,
    days_back: int,
    target_column: str,  # "regime" | "ev_G01" | "volatility_30m"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns (X, y) where:
    - X: DataFrame of features (no leakage)
    - y: Series of labels (future-looking, but shifted properly)
    """
```

**Critical**: Labels are computed from FUTURE data but features are from PAST data. Example for regime label:
- At time T, label = regime over [T, T+15min]
- At time T, features use data from [T-N, T]
- NEVER use data from T+1 onwards as features

### 5.3 Walk-forward Validation

Time series data CANNOT use random train/test split. We use walk-forward:

```
Train: [Day 1 - Day 60]    Validate: [Day 61 - Day 67]    (week 1)
Train: [Day 2 - Day 61]    Validate: [Day 62 - Day 68]    (week 2)
Train: [Day 3 - Day 62]    Validate: [Day 63 - Day 69]    (week 3)
...
```

Reports metrics averaged across all validation windows.

---

## 6. Model Specifications

### 6.1 Regime Classifier

```python
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
)
```

**Label encoding**:
- 0: RANGING
- 1: TRENDING_UP
- 2: TRENDING_DOWN
- 3: VOLATILE

**Label generation** (from historical data):
- For each bar at time T, look at [T, T+15 bars]
- Compute EMA20/EMA50/ADX/ATR for that window
- Apply the existing regime detection logic to get ground truth regime
- This is the label

**Output interface**:
```python
predict_regime(features: dict) -> {
    "regime": "TRENDING_UP",
    "confidence": 0.78,
    "probabilities": {"RANGING": 0.05, "TRENDING_UP": 0.78, "TRENDING_DOWN": 0.02, "VOLATILE": 0.15}
}
```

### 6.2 Expected-Value Model (per strategy)

```python
model = XGBRegressor(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    objective="reg:squarederror",
    random_state=42,
)
```

**Label**: Historical PnL (in USD) of trades that strategy X took when features matched current state.

**Training data source**: `learnings/trades.jsonl` — all closed trades with their entry features and PnL.

**Minimum data requirement**: 50 closed trades per strategy. Below that, model falls back to "no prediction" and runner uses default 0.72 threshold.

**Output interface**:
```python
predict_ev(strategy_id: str, features: dict) -> {
    "expected_pnl_usd": 3.25,
    "confidence": 0.65,  # inverse of prediction variance
    "n_samples_trained_on": 127,
}
```

**Usage**: Replace fixed confidence threshold with dynamic threshold:
- If predicted EV < $0.50 → skip (doesn't cover fees)
- If predicted EV > $5.00 → take position with normal size
- If predicted EV > $15.00 → take position with 1.3x size

### 6.3 Volatility Predictor

```python
model = XGBRegressor(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=42,
)
```

**Label**: Standard deviation of 1-minute returns over next 30 minutes:
```python
y[t] = std(returns[t:t+30])
```

**Features**: All 39 features + `pair_id` as categorical (one-hot encoded or embedded).

**Output interface**:
```python
predict_volatility(pair: str, features: dict, horizon_minutes: int = 30) -> {
    "predicted_vol_pct": 0.0042,  # 0.42%
    "baseline_vol_pct": 0.0035,   # 20-day avg
    "vol_ratio": 1.2,             # predicted / baseline
}
```

**Usage**: Adjust TP/SL dynamically:
```python
adjusted_tp = strategy.base_tp * min(max(vol_ratio, 0.7), 1.5)  # clamp 0.7-1.5x
adjusted_sl = strategy.base_sl * min(max(vol_ratio, 0.7), 1.5)
```

---

## 7. Retraining Pipeline

### 7.1 Schedule
- **Cron**: Sunday 03:00 UTC (low-activity time)
- **Runner**: APScheduler job in existing `tournament_runner_god2.py`
- **Duration**: ~45 minutes for full retrain of all 14 models (1 regime + 13 EV + 1 vol = 15 total)

### 7.2 Process

```
1. Log start event to memory system
2. Download incremental data (last 48h) → update parquet files
3. Load full historical dataset (90 days)
4. Extract features for full dataset
5. For each model:
   a. Prepare X, y with walk-forward splits
   b. Train new model on latest 85 days, validate on last 5 days
   c. Load current production model
   d. Compute baseline metrics (current model on same validation set)
   e. If new model beats baseline by >2% → promote to production
   f. If new model is worse → keep current, log alert
   g. Save model to disk and GCS
6. Invalidate inference cache
7. Log completion summary to memory system (long tier)
8. Send push notification with training summary
```

### 7.3 Model Promotion Criteria

| Model | Promotion Threshold |
|:------|:--------------------|
| Regime Classifier | New accuracy > current + 2% |
| EV Model (per strategy) | New MAE < current MAE * 0.95 (5% improvement) |
| Volatility Predictor | New R² > current R² + 0.02 |

If new model fails, keep the old one. Alert is logged for human review.

---

## 8. Integration with Existing System

### 8.1 Changes to `tournament_runner_god2.py`

Add inference calls in the strategy evaluation cycle:

```python
async def _run_strategy(self, strat):
    # ... existing checks ...
    
    # NEW: Get ML predictions for each candidate pair
    result = await select_best_pair(strat, self.pairs, self.strategies, min_confidence=settings.MIN_CONFIDENCE)
    
    if result:
        pair = result["pair"]
        signal = result["signal"]
        
        # NEW: Check ML regime prediction
        ml_regime = await inference.get_regime_prediction(pair)
        if ml_regime["confidence"] > 0.7:  # high-confidence prediction
            if ml_regime["regime"] not in strat.cfg.regime_filter:
                strat.skip_count += 1
                logger.info(f"[{strat.cfg.id}] ML regime mismatch: {ml_regime['regime']}")
                return
        
        # NEW: Check expected value
        features = await ml_feature_extractor.extract(pair)
        ev = await inference.get_expected_value(strat.cfg.id, features)
        if ev["expected_pnl_usd"] < settings.ML_MIN_EV_USD and ev["n_samples_trained_on"] > 50:
            strat.skip_count += 1
            logger.info(f"[{strat.cfg.id}] ML EV too low: ${ev['expected_pnl_usd']:.2f}")
            return
        
        # NEW: Adjust TP/SL based on volatility prediction
        vol_pred = await inference.get_volatility(pair, features)
        vol_ratio = max(0.7, min(1.5, vol_pred["vol_ratio"]))
        adjusted_tp_pct = strat.cfg.tp_pct * vol_ratio
        adjusted_sl_pct = strat.cfg.sl_pct * vol_ratio
        
        # Use adjusted values when opening position
        strat.open_position_with_custom_params(
            signal.direction, price, pair, signal.signals,
            tp_pct=adjusted_tp_pct, sl_pct=adjusted_sl_pct
        )
```

### 8.2 Changes to `base_strategy_v4.py`

Add method to open position with runtime-adjusted TP/SL:

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
    # ... rest of logic using effective_tp/effective_sl
```

### 8.3 New Settings

Add to `config/settings.py`:

```python
# ML Layer
ML_ENABLED: bool = True
ML_MIN_EV_USD: float = 0.50
ML_REGIME_CONFIDENCE_THRESHOLD: float = 0.70
ML_VOL_RATIO_MIN: float = 0.70
ML_VOL_RATIO_MAX: float = 1.50
ML_RETRAIN_DAY: int = 6  # Sunday
ML_RETRAIN_HOUR_UTC: int = 3
ML_HISTORICAL_DAYS: int = 90
ML_GCS_BUCKET: str = "agent-god-2-data"
```

### 8.4 New API Endpoints

Add to `main_god2.py`:

```python
GET  /ml/status              # Model versions, last training time, metrics
GET  /ml/regime/{pair}       # Current regime prediction for a pair
GET  /ml/ev/{strategy_id}    # Current EV prediction for a strategy
GET  /ml/volatility/{pair}   # Current vol prediction for a pair
POST /ml/retrain             # Trigger manual retrain (admin)
GET  /ml/training-history    # Last 10 training runs with metrics
```

### 8.5 Dashboard Additions

Add a 7th view to the dashboard: **"ML Insights"**:
- Current regime predictions per pair with confidence bars
- EV predictions for all 13 strategies (sorted)
- Volatility predictions vs baselines
- Training history (accuracy over time)
- Model versions and last retrain timestamps

---

## 9. Dependencies

Add to `ml_requirements.txt`:

```
xgboost==2.1.3
scikit-learn==1.5.2
joblib==1.4.2
pyarrow==17.0.0
google-cloud-storage==2.18.2
```

Merge into main `requirements.txt` for deployment.

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|:-----|:-----------|
| Overfitting to training data | Walk-forward validation, regularization, held-out test set |
| Data leakage (future in features) | Strict feature pipeline that only uses past data; unit tests |
| Model drift in production | Weekly retraining, metrics monitoring, auto-fallback to rules |
| Inference latency slowing down strategies | 30s cache on predictions, <50ms inference budget |
| Insufficient trade data for EV model | Require 50+ trades minimum, else skip EV check |
| Training failures breaking bot | Training is isolated job, failures don't affect live inference |
| GCS costs | Monthly bucket cost < $1 at our data volume |
| Breaking existing flows | ML_ENABLED feature flag — can be disabled instantly |

---

## 11. Success Metrics

After 30 days of ML-enabled operation vs 30 days before:

- **Primary**: Portfolio Sharpe ratio increase ≥ 20%
- **Secondary**: Win rate increase of at least 3 percentage points
- **Secondary**: Profit factor increase ≥ 15%
- **Safety**: Number of eliminated strategies does NOT increase
- **Safety**: Max drawdown does NOT increase

If none of these improve, ML layer is disabled and design is re-evaluated.

---

## 12. Rollout Plan

1. **Week 1-2**: Build data pipeline + feature extraction + unit tests
2. **Week 2**: Train initial models offline, validate on held-out data
3. **Week 3**: Integrate with runner, keep `ML_ENABLED=False`
4. **Week 3**: Shadow mode — compute predictions but don't use them, log metrics
5. **Week 4**: Enable `ML_ENABLED=True` for top 3 ranked strategies only
6. **Week 5**: Full rollout if metrics are stable
7. **Week 6+**: Monitor, iterate on features, add new models as needed
