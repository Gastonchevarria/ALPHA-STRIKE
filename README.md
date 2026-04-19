<p align="center">
  <img src="https://img.shields.io/badge/⚡-ALPHA_STRIKE-00d2ff?style=for-the-badge&labelColor=0a0a1a&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJ3aGl0ZSI+PHBhdGggZD0iTTEyIDNMMiAyMWgyMEwxMiAzeiIvPjwvc3ZnPg==" alt="Alpha Strike"/>
</p>

<h1 align="center">⚡ ALPHA STRIKE</h1>
<h3 align="center">Autonomous AI Trading Engine — 13 Strategies × 6 Pairs × Zero Human Intervention</h3>

<p align="center">
  <img src="https://img.shields.io/badge/AI-Claude%20Opus%204.6-blueviolet?style=for-the-badge&logo=anthropic" alt="AI Model"/>
  <img src="https://img.shields.io/badge/ML-XGBoost-orange?style=for-the-badge" alt="ML"/>
  <img src="https://img.shields.io/badge/Exchange-Binance%20Futures-F0B90B?style=for-the-badge&logo=binance" alt="Exchange"/>
  <img src="https://img.shields.io/badge/Deploy-Cloud%20Run-4285F4?style=for-the-badge&logo=google-cloud" alt="Deploy"/>
  <img src="https://img.shields.io/badge/Python-3.12-green?style=for-the-badge&logo=python" alt="Python"/>
</p>

<p align="center">
  <b>A self-evolving crypto trading system that breeds, battles, and promotes strategies<br/>through a Darwinian tournament — powered by Claude Opus AI and a real-time XGBoost ML layer.</b>
</p>

---

## 🧬 What is Alpha Strike?

Alpha Strike is a **fully autonomous trading engine** that runs 13 independent strategies across 6 cryptocurrency pairs simultaneously on Binance USDT-M Futures. Strategies compete in a Darwin-style tournament where only the fittest survive, get promoted from paper to live trading, and continuously evolve through AI-powered self-training.

**No human intervention required.** The system trades, learns, adapts, and evolves 24/7.

```
Paper Trading → Shadow Mode → Live Execution → Capital Scaling
     ↑                                              ↓
     └──── Elimination ← Demotion ← Underperformance
```

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      ⚡ ALPHA STRIKE                         │
├───────────────┬────────────────┬─────────────────────────────┤
│  13 STRATEGIES │  ML LAYER      │  AI BRAIN (Claude Opus 4.6) │
│  ────────────  │  ───────────   │  ──────────────────────     │
│  G-01 Momentum │  Regime Class. │  Tournament Coordinator     │
│  G-02 Scalp    │  EV Predictor  │  Self-Trainer (per trade)   │
│  G-03 Orderflow│  Vol Predictor │  Memory Consolidation       │
│  G-04 MACD     │  39 Features   │  Nightly Reflection         │
│  G-05 Stoch    │  Walk-Forward  │  Confidence Multipliers     │
│  G-06 BB Sqz   │  XGBoost       │                             │
│  G-07 RSI Div  │                │                             │
│  G-08 VWAP     │                │                             │
│  G-09 ATR      │                │                             │
│  G-10 Ichimoku │                │                             │
│  G-11 Liq Hunt │                │                             │
│  G-12 Cross-P  │                │                             │
│  G-13 Vol Delta│                │                             │
├───────────────┴────────────────┴─────────────────────────────┤
│  CORE: Circuit Breaker │ Kelly Sizing │ Pair Selector │ Risk │
│  INFRA: FastAPI │ Premium Dashboard │ Binance REST │ 25+ API│
└──────────────────────────────────────────────────────────────┘
```

## 🎯 The 13 Gladiators

### ⚡ Ultra-Fast Block (1m timeframe)

| ID | Strategy | Leverage | TP/SL | Edge |
|----|----------|----------|-------|------|
| **G-01** | Momentum Burst | 40x | 0.35/0.20% | ROC + volume spike, 3 consecutive candles |
| **G-02** | Scalp Ultra | 35x | 0.30/0.18% | RSI(7) extremes + BB touch + spread filter |
| **G-03** | Orderflow Imbalance | 45x | 0.40/0.22% | Taker buy/sell ratio > 65% |
| **G-13** | Volume Delta Sniper | 50x | 0.45/0.25% | Price vs cumulative delta divergence |

### 🚀 Fast Block (5m timeframe)

| ID | Strategy | Leverage | TP/SL | Edge |
|----|----------|----------|-------|------|
| **G-04** | MACD Scalper | 30x | 0.50/0.30% | Histogram crosses zero + EMA9 + ADX>20 |
| **G-05** | Stochastic Reversal | 25x | 0.55/0.30% | %K crosses %D at extremes (<20/>80) |
| **G-06** | BB Squeeze Turbo | 35x | 0.60/0.30% | Bollinger inside Keltner + volume |
| **G-08** | VWAP Sniper | 30x | 0.50/0.28% | VWAP bounce with ±1σ/±2σ bands |
| **G-09** | ATR Breakout Rider | 30x | 0.55/0.30% | ATR expansion >1.5x + trailing stop |

### 🧠 Strategic Block (5m-15m timeframe)

| ID | Strategy | Leverage | TP/SL | Edge |
|----|----------|----------|-------|------|
| **G-07** | RSI Divergence Hunter | 20x | 0.70/0.40% | Classic divergences with 2+ pivots |
| **G-10** | Ichimoku Cloud Edge | 20x | 0.65/0.35% | Kumo breakout + Tenkan/Kijun + Chikou |
| **G-11** | Liq Hunter Pro | 40x | 0.50/0.25% | OI drop >3% + violent price action |
| **G-12** | Cross-Pair Divergence | 25x | 0.60/0.32% | **Exclusive**: mean reversion across pairs |

## 🧠 ML Augmentation Layer

Three XGBoost models enhance every trade decision in real-time:

| Model | Purpose | Features | Cycle |
|-------|---------|----------|-------|
| **Regime Classifier** | Predicts market regime (Trend Up/Down, Range, Volatile) | 39 technical + time features | Weekly |
| **EV Predictor** | Expected P&L per strategy × pair combination | Per-strategy historical perf | Weekly |
| **Volatility Predictor** | Adjusts TP/SL dynamically based on predicted vol | Realized vol + regime context | Weekly |

**Design Principles:**
- 🔒 **Zero future leakage** — Features computed strictly from past data
- 📊 **Walk-forward validation** — No look-ahead bias
- 🎯 **39-feature pipeline** — Price, momentum, volume, volatility, time
- ⚡ **30-second inference cache** — Sub-millisecond predictions

## 🏆 Tournament System

```
PAPER MODE (all strategies start here)
    │
    ▼ After 100+ trades, WR > 55%, PF > 1.5, DD < 15%, 7+ days
    │
SHADOW MODE (48h observation on testnet, zero risk)
    │
    ▼ No errors, sustained performance
    │
LIVE MODE (5% capital → auto-scales to 50% max)
    │
    ├── Circuit Breaker: Dual (Paper -12% / Live -5% daily)
    ├── Kelly Sizing: Quarter-Kelly, scale 0.5x–2.0x
    ├── Strategy Eliminator: -8% → pause 12h, 3 pauses → permanent kill
    └── Auto-Demotion: Back to paper if live performance degrades >10%
```

**Self-Evolution Loop:**
1. 🧬 **Self-Trainer** — Claude Opus analyzes every closed trade, adjusts TP/SL/margin within ±2% safety rails
2. 🧭 **Tournament Coordinator** — Opus reviews all 13 strategies every 2h, adjusts confidence multipliers
3. 🧠 **3-Tier Memory** — Short/Mid/Long-term with nightly AI reflection and pattern promotion
4. 🔗 **Correlation Engine** — Rolling 6×6 pair correlation matrix, detects breakdowns for G-12

## 📊 Premium Dashboard (6 Views)

Real-time web dashboard with live updates:

| View | Description |
|------|-------------|
| 🏆 **Leaderboard** | P&L, win rate, Sharpe, drawdown rankings |
| 📈 **Strategy Detail** | Per-strategy equity curve, trade log, pair distribution |
| 🗺️ **Pair Heatmap** | 6 pairs × 13 strategies performance matrix |
| 🔗 **Correlation** | Live 6×6 correlation matrix with breakdown alerts |
| 🚀 **Promotion Pipeline** | Kanban board: Paper → Shadow → Live |
| ⚡ **Live Monitor** | Real-time live positions, kill switch, circuit breaker |

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/gastonchevarria/agent-god-2.git
cd agent-god-2

# Configure
cp .env.example .env
# Edit .env with your API keys

# Install
pip install -r requirements.txt

# Run locally (paper mode by default)
uvicorn main:app --host 0.0.0.0 --port 9090

# Deploy to Cloud Run
gcloud run deploy alpha-strike \
  --source . \
  --region us-central1 \
  --memory 1Gi --cpu 1 \
  --min-instances 1 --max-instances 1 \
  --port 9090
```

## 🔧 Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Claude Opus 4.6 (brain + self-trainer) | Required |
| `BINANCE_API_KEY` | Binance Futures API key | Required for live |
| `BINANCE_API_SECRET` | Binance Futures secret | Required for live |
| `PAIRS` | Trading pairs (comma-separated) | `BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT` |
| `MODE` | `paper` or `live` | `paper` |
| `ML_ENABLED` | Enable ML augmentation | `true` |

## 📁 Project Structure

```
alpha-strike/
├── strategies/                  # 13 trading strategies (G-01 to G-13)
│   ├── base_strategy_v4.py          # Base: multi-pair, Kelly, promotion-aware
│   ├── g01_momentum_burst.py        # 1m  40x  ROC + vol spike
│   ├── g02_scalp_ultra.py           # 1m  35x  RSI + BB
│   ├── ...
│   └── g13_volume_delta_sniper.py   # 1m  50x  Volume delta
│
├── core/                        # Engine components
│   ├── ai_client.py                 # Claude Opus 4.6 wrapper
│   ├── circuit_breaker.py           # Dual risk protection
│   ├── correlation_engine.py        # 6×6 pair correlation
│   ├── promotion_manager.py         # Paper → Shadow → Live
│   ├── self_trainer.py              # Per-trade AI analysis
│   ├── tournament_coordinator.py    # Brain (every 2h)
│   ├── memory_tiers.py              # 3-tier AI memory
│   ├── pair_selector.py             # Parallel 6-pair evaluation
│   ├── live_executor.py             # Real Binance Futures execution
│   └── strategy_eliminator.py       # Auto-pause/kill
│
├── ml/                          # Machine Learning layer
│   ├── features.py                  # 39-feature extraction
│   ├── regime_classifier.py         # Market regime prediction
│   ├── ev_model.py                  # Expected value per strategy
│   ├── volatility_predictor.py      # Dynamic TP/SL adjustment
│   ├── inference.py                 # Cached runtime API
│   └── training_pipeline.py         # Weekly retraining
│
├── scheduler/                   # Tournament orchestrator
├── static/                      # Premium dashboard (6 views)
├── config/                      # Settings & env management
├── tests/                       # Unit & integration tests (11+)
├── main.py                      # FastAPI app (port 9090)
├── main_god2.py                 # API router (25+ endpoints)
├── Dockerfile                   # Cloud Run deployment
└── requirements.txt             # Dependencies
```

## 🛡️ Risk Management

| Protection | Paper | Live |
|-----------|-------|------|
| **Circuit Breaker** | -12% daily → halt 24h | -5% daily → halt immediately |
| **Strategy Eliminator** | -8% in 24h → pause 12h | Same + auto-demote to paper |
| **Max Exposure/Pair** | 4 strategies max | 3 live positions max |
| **Capital Limit** | $1,000/strategy | 30% of total capital |
| **Kill Switch** | `POST /tournament/pause` | `POST /live/halt` |

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred through the use of this software. **Never trade with money you can't afford to lose.**

---

<p align="center">
  <b>Built with ⚡ by <a href="https://github.com/gastonchevarria">@gastonchevarria</a></b><br/>
  <sub>Powered by Claude Opus 4.6 • XGBoost • Binance Futures • FastAPI</sub>
</p>
