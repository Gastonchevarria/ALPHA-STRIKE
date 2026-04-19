# Agent GOD 2 — Design Specification

**Date**: 2026-04-16
**Project**: Multi-Agent Reinforcement Learning BTC/Crypto Futures Trading Bot v2
**Location**: `/Users/gastonchevarria/Alpha/agent-god-2/`
**Port**: 9090
**Status**: Design approved, pending implementation

---

## 1. Overview

Agent GOD 2 is a completely independent evolution of the original marl-btc-trader (Quantum Prime GOD V100). It is a multi-agent tournament-based trading system with 13 aggressive strategies operating across 6 cryptocurrency pairs, with a promotion pipeline that graduates winning strategies from paper trading to live Binance Futures execution.

### Key Differentiators from Original

- **Multi-pair**: 6 pairs (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT) vs original's single BTC
- **Faster strategies**: 1m-15m timeframes vs original's 1h-centric approach
- **Higher leverage**: 20-50x range vs original's mostly 25-30x
- **Promotion to Live**: Paper → Shadow → Live pipeline (original is paper-only)
- **Correlation Engine**: Cross-pair divergence detection (impossible in single-pair system)
- **Claude Opus 4.6 everywhere**: Brain AND exec model (original uses Sonnet for exec)
- **Premium Dashboard**: 6 views with heatmaps, equity curves, correlation matrix, live monitor

---

## 2. Architecture

### Directory Structure

```
agent-god-2/
├── config/
│   └── settings.py                     # Pydantic settings (port 9090)
├── core/
│   ├── ai_client.py                    # Claude Opus 4.6 routing
│   ├── market_regime.py                # Per-pair regime detection (each pair has its own regime)
│   ├── pair_selector.py                # Parallel 6-pair evaluation
│   ├── correlation_engine.py           # Cross-pair correlation & divergence
│   ├── tournament_coordinator.py       # Brain (Claude Opus 4.6, every 2h)
│   ├── strategy_eliminator.py          # Auto-pause/eliminate (-8% in 24h)
│   ├── promotion_manager.py            # Paper → Shadow → Live graduation
│   ├── self_trainer.py                 # Per-trade AI analysis (Opus 4.6)
│   ├── memory_tiers.py                 # Short/mid/long memory with decay
│   ├── memory_heartbeat.py             # Consolidation every 20min
│   ├── data_fetcher.py                 # Binance API with smart cache
│   ├── risk_manager.py                 # Position, margin, liquidation
│   ├── live_executor.py                # Real Binance Futures execution
│   ├── performance_tracker.py          # Stats & profit factor
│   ├── circuit_breaker.py              # Dual-level CB (paper + live)
│   ├── partial_tp.py                   # Partial take-profit
│   └── learnings_logger.py             # File-based persistence
├── strategies/
│   ├── base_strategy_v4.py             # Multi-pair, promotion-aware base
│   ├── g01_momentum_burst.py
│   ├── g02_scalp_ultra.py
│   ├── g03_orderflow_imbalance.py
│   ├── g04_macd_scalper.py
│   ├── g05_stochastic_reversal.py
│   ├── g06_bb_squeeze_turbo.py
│   ├── g07_rsi_divergence.py
│   ├── g08_vwap_sniper.py
│   ├── g09_atr_breakout.py
│   ├── g10_ichimoku_edge.py
│   ├── g11_liquidation_hunter_pro.py
│   ├── g12_cross_pair_divergence.py
│   └── g13_volume_delta_sniper.py
├── scheduler/
│   └── tournament_runner_god2.py
├── static/
│   └── dashboard.html
├── tests/
├── data/
│   └── strategy_params/
├── learnings/
├── main.py
├── main_god2.py
├── requirements.txt
└── .env.example
```

### Execution Flow

1. `main.py` starts FastAPI on port 9090 with lifespan
2. Lifespan creates `TournamentRunnerGOD2` with 13 strategies
3. APScheduler schedules per-strategy crons + background jobs
4. Each strategy cycle:
   - Check paused/eliminated → skip if yes
   - Check circuit breaker → skip if triggered
   - Check regime match → skip if mismatch
   - **Pair Selector**: `asyncio.gather()` evaluates all 6 pairs in parallel
   - Filter by MIN_CONFIDENCE, rank by confidence × volatility
   - If position open: check exit (TP/SL/TIMEOUT)
   - If no position: open with best pair+signal
   - If strategy is LIVE: route through `live_executor` instead of paper
5. Background jobs: regime detection, eliminator, coordinator, memory heartbeat, promotion checks

---

## 3. The 13 Strategies

All strategies inherit from `BaseStrategyV4`, start with $1000 balance, and evaluate 6 pairs in parallel.

**Common rules:**
- Each strategy can hold **one position at a time** (across all pairs). Must close before opening another.
- Default **margin_pct**: 8% of balance (adjusted by Kelly after 20 trades)
- **Auto-close timeout**: 30 minutes for 1m strategies, 2 hours for 5m, 4 hours for 15m
- **Fee model**: 0.0004 maker / 0.0008 taker (market orders = taker)

### Block: Ultra-Fast (1m) — 4 strategies

#### G-01 Momentum Burst
- **Timeframe**: 1m | **Leverage**: 40x | **TP**: 0.35% | **SL**: 0.20%
- **Cron**: `*/2` (every 2 minutes)
- **Logic**: Detects sudden momentum explosions using Rate of Change (ROC) + Volume Spike. If ROC exceeds threshold for 3 consecutive candles + volume is 2x average → enter in momentum direction. Selects the pair with highest ROC at that moment.
- **Regime**: ANY (momentum can happen in any regime)

#### G-02 Scalp Ultra (evolution of S-02)
- **Timeframe**: 1m | **Leverage**: 35x | **TP**: 0.30% | **SL**: 0.18%
- **Cron**: `*/1` (every minute)
- **Logic**: RSI(7) at extremes (<25 LONG, >75 SHORT) + volume confirmation + Bollinger Band touch. Improvement over original: adds spread filter (avoids high-spread entries), multi-pair evaluation to choose the pair with most extreme RSI.
- **Regime**: RANGING, VOLATILE

#### G-03 Order Flow Imbalance
- **Timeframe**: 1m | **Leverage**: 45x | **TP**: 0.40% | **SL**: 0.22%
- **Cron**: `*/2`
- **Logic**: Analyzes taker buy/sell volume ratio from Binance API. Imbalance >65% buy → LONG, >65% sell → SHORT. Confirmation: current candle must close in imbalance direction. The pair with highest imbalance wins the signal.
- **Regime**: ANY

#### G-13 Volume Delta Sniper
- **Timeframe**: 1m | **Leverage**: 50x | **TP**: 0.45% | **SL**: 0.25%
- **Cron**: `*/2`
- **Logic**: Calculates cumulative volume delta (buy vol - sell vol) over last 5 candles of 1m. Divergence detection: price drops but delta rises = LONG (hidden accumulation). Price rises but delta falls = SHORT (distribution). Most aggressive leverage (50x) but with the most sophisticated detection logic.
- **Regime**: ANY

### Block: Fast (5m) — 5 strategies

#### G-04 MACD Scalper
- **Timeframe**: 5m | **Leverage**: 30x | **TP**: 0.50% | **SL**: 0.30%
- **Cron**: `*/5`
- **Logic**: MACD histogram crosses zero line + EMA(9) direction confirmation. Filters by ADX > 20 (needs minimum trend). Evaluates 6 pairs, prioritizes the one with most pronounced histogram.
- **Regime**: TRENDING_UP, TRENDING_DOWN

#### G-05 Stochastic Reversal
- **Timeframe**: 5m | **Leverage**: 25x | **TP**: 0.55% | **SL**: 0.30%
- **Cron**: `*/5`
- **Logic**: Stochastic %K crosses %D in oversold zone (<20) → LONG | overbought zone (>80) → SHORT. Confirmation: next candle must close in signal direction. Works best in RANGING, penalized in strong TRENDING.
- **Regime**: RANGING

#### G-06 BB Squeeze Turbo (evolution of SB-8)
- **Timeframe**: 5m | **Leverage**: 35x | **TP**: 0.60% | **SL**: 0.30%
- **Cron**: `*/5`
- **Logic**: Detects Bollinger squeeze (bandwidth < 10th percentile of last 50 candles). On squeeze release: direction determined by Keltner Channel breakout. Improvement: adds volume confirmation and multi-pair evaluation (pair with tightest squeeze wins).
- **Regime**: ANY

#### G-08 VWAP Sniper (evolution of SA-4)
- **Timeframe**: 5m | **Leverage**: 30x | **TP**: 0.50% | **SL**: 0.28%
- **Cron**: `*/5`
- **Logic**: Price touches VWAP from above with RSI > 40 → LONG bounce. From below with RSI < 60 → SHORT rejection. Improvement: adds standard deviation bands (±1σ, ±2σ) as dynamic TP targets instead of fixed TP.
- **Regime**: RANGING

#### G-09 ATR Breakout Rider
- **Timeframe**: 5m | **Leverage**: 30x | **TP**: 0.55% | **SL**: 0.30%
- **Cron**: `*/5`
- **Logic**: Detects ATR expansion (current ATR > 1.5x 20-period average ATR). Direction determined by candle close relative to EMA(20). Trailing stop based on ATR: SL moves 1.5x ATR behind price. Ideal for sudden volatility moments.
- **Regime**: VOLATILE

### Block: Strategic (5m-15m) — 4 strategies

#### G-07 RSI Divergence Hunter
- **Timeframe**: 15m | **Leverage**: 20x | **TP**: 0.70% | **SL**: 0.40%
- **Cron**: `*/15`
- **Logic**: Detects classic divergences: price makes lower low but RSI makes higher low → LONG. Price makes higher high but RSI makes lower high → SHORT. Requires minimum 2 pivots for divergence confirmation.
- **Regime**: ANY

#### G-10 Ichimoku Cloud Edge
- **Timeframe**: 15m | **Leverage**: 20x | **TP**: 0.65% | **SL**: 0.35%
- **Cron**: `*/15`
- **Logic**: Kumo breakout: price crosses cloud with Tenkan > Kijun → strong LONG. Kumo bounce: price touches cloud edge and bounces in trend direction. Chikou span confirms (must be above/below price from 26 periods ago).
- **Regime**: TRENDING_UP, TRENDING_DOWN

#### G-11 Liquidation Hunter Pro (evolution of SA-2)
- **Timeframe**: 5m | **Leverage**: 40x | **TP**: 0.50% | **SL**: 0.25%
- **Cron**: `*/5`
- **Logic**: Detects massive liquidation zones using open interest changes + price action. When OI drops abruptly (>3% in 5 candles) + violent price move = liquidation cascade. Enters in cascade direction to ride momentum. Multi-pair enables detection in altcoins (more violent cascades).
- **Regime**: VOLATILE

#### G-12 Cross-Pair Divergence
- **Timeframe**: 5m | **Leverage**: 25x | **TP**: 0.60% | **SL**: 0.32%
- **Cron**: `*/5`
- **Logic**: GOD 2 exclusive strategy. Measures rolling correlation (20 periods) between BTC and each altcoin. When correlation breaks (altcoin diverges from BTC) → mean reversion trade. Example: BTC up 1%, SOL down 0.5% when normal correlation is 0.85 → LONG SOL. Relies on `correlation_engine` data.
- **Regime**: ANY

### Strategy Summary Table

| ID | Name | TF | Leverage | TP% | SL% | Cron | Type |
|----|------|----|----------|-----|-----|------|------|
| G-01 | Momentum Burst | 1m | 40x | 0.35 | 0.20 | */2 | NEW |
| G-02 | Scalp Ultra | 1m | 35x | 0.30 | 0.18 | */1 | EVOLVED |
| G-03 | Order Flow Imbalance | 1m | 45x | 0.40 | 0.22 | */2 | NEW |
| G-04 | MACD Scalper | 5m | 30x | 0.50 | 0.30 | */5 | NEW |
| G-05 | Stochastic Reversal | 5m | 25x | 0.55 | 0.30 | */5 | NEW |
| G-06 | BB Squeeze Turbo | 5m | 35x | 0.60 | 0.30 | */5 | EVOLVED |
| G-07 | RSI Divergence | 15m | 20x | 0.70 | 0.40 | */15 | NEW |
| G-08 | VWAP Sniper | 5m | 30x | 0.50 | 0.28 | */5 | EVOLVED |
| G-09 | ATR Breakout | 5m | 30x | 0.55 | 0.30 | */5 | NEW |
| G-10 | Ichimoku Edge | 15m | 20x | 0.65 | 0.35 | */15 | NEW |
| G-11 | Liq Hunter Pro | 5m | 40x | 0.50 | 0.25 | */5 | EVOLVED |
| G-12 | Cross-Pair Div | 5m | 25x | 0.60 | 0.32 | */5 | NEW |
| G-13 | Vol Delta Sniper | 1m | 50x | 0.45 | 0.25 | */2 | NEW |

---

## 4. Promotion System (Paper → Live)

### Pipeline

```
PAPER (all start here, $1000 each)
    │
    ▼  meets promotion criteria
SHADOW (real execution on testnet, 0 capital, 48h)
    │
    ▼  48h with < 3 execution errors
LIVE (real money, controlled capital)
```

### Phase 1: PAPER

All strategies begin here with $1000 simulated balance.

**Criteria for promotion to SHADOW:**
- Minimum 100 closed trades
- Win Rate >= 55%
- Profit Factor >= 1.5
- Maximum drawdown < 15% from peak
- Active for at least 7 days without being eliminated
- Tournament Coordinator (Claude Opus) must approve the promotion with analysis

### Phase 2: SHADOW

Executes real orders on Binance Testnet to verify execution logic works.

- Duration: 48 hours
- Verifies: execution latency, slippage, partial fills, API errors
- If more than 3 execution errors → returns to PAPER
- Zero capital at risk

### Phase 3: LIVE

Real money on Binance Futures.

- **Initial capital**: 5% of the strategy's paper balance (e.g., $1,500 paper → $75 real)
- **Gradual scaling**: every 50 successful live trades, capital increases by 5% (up to max 50% of paper balance)
- **Automatic demotion**: if strategy drops >10% in live → returns to PAPER, loses all promotion progress
- **Manual kill switch**: API endpoint for immediate demotion
- **Daily report**: Claude Opus generates daily report of all LIVE strategies

### Live Executor

`live_executor.py` handles real execution:
- Connection to Binance Futures API (not testnet)
- Market orders for speed (1m strategies can't wait for limit fills)
- Verifies available balance before each order
- Exhaustive logging of each order (request, response, fill price, fees)
- Rate limiting to not exceed Binance API limits
- Failsafe: if Binance API doesn't respond in 3s → cancel operation

### Live Security

- **Live circuit breaker**: if total LIVE strategies lose >5% of total live capital in 24h → immediate halt of ALL live + notification
- **Max concurrent live positions**: maximum 3 LIVE positions open simultaneously
- **Max live capital**: never more than 30% of total available capital in LIVE positions
- **Total separation**: paper and live balances are independent. Paper continues simulating even if live is halted

---

## 5. Core Systems

### Pair Selector (`core/pair_selector.py`)

Each strategy evaluation cycle:

1. `asyncio.gather()` → evaluate all 6 pairs in parallel
2. Each pair returns: `{pair, signal, confidence, indicators}`
3. Filter: discard signals with confidence < MIN_CONFIDENCE
4. Rank: sort by `confidence * current_volatility`
5. Result: best pair+signal wins, or HOLD if none pass filter

Rules:
- **Cooldown per pair**: after closing a position on a pair, can't reopen on same pair for 3 candles (prevents revenge trading)
- **Max exposure per pair**: maximum 4 strategies can have open position on same pair simultaneously (prevents concentration)

### Correlation Engine (`core/correlation_engine.py`)

Runs every 5 minutes:
- Rolling correlation matrix (20 periods) between all 6 pairs
- Break detection: when correlation drops below 2 standard deviations from rolling mean
- General divergence index: when average pairwise correlation drops below 0.4 = chaotic market → circuit breaker considers reducing exposure
- Data available to all strategies and the Coordinator via `get_correlation_matrix()` and `get_divergence_index()`

### Tournament Coordinator (improved)

- **Model**: Claude Opus 4.6
- **Frequency**: every 2 hours (more frequent than original's 4h due to faster strategies)
- **Additional inputs**: correlation data, per-pair metrics, promotion phase of each strategy
- **Additional output**: promotion/demotion recommendation for Promotion Manager
- **Enhanced prompt**: includes multi-pair context and phase awareness

### Strategy Eliminator (improved)

- Threshold: **-8% in 24h** (more aggressive than original's -10% in 48h)
- Pause duration: **12 hours** (shorter, gives strategies faster comeback)
- Max pauses before elimination: **3**
- **New**: if a LIVE strategy is eliminated → automatic demotion to PAPER first, then elimination

### Self-Trainer (improved)

- **Model**: Claude Opus 4.6 (upgraded from original's Sonnet)
- Post-trade analysis now includes: which pair was chosen, why other pairs were rejected, correlation state at time of trade
- More granular parameter adjustment: can adjust per-pair preference weights

### Circuit Breaker (dual-level)

- **Paper CB**: -12% of total paper portfolio → halt all paper (more permissive, it's simulation)
- **Live CB**: -5% of total live capital in 24h → halt all live immediately (very aggressive, protects real money)
- Manual reset via API for each level independently

### Data Fetcher (improved)

- **Smart cache**: if 3 strategies request BTC/USDT 1m in the same second → single Binance request
- **Prefetch**: at each cycle start, preloads data for all 6 pairs in all 3 timeframes (1m, 5m, 15m)
- **Rate limiter**: respects Binance's 1200 req/min limit
- **Additional metrics**: taker buy/sell volume (for G-03 and G-13), open interest (for G-11)

### Memory System

Same 3-tier architecture as original but with faster heartbeat:
- **Short-term**: 24h TTL, max 20 entries
- **Mid-term**: 7 days TTL, max 50 entries
- **Long-term**: permanent, max 100 entries
- **Heartbeat**: every 20 minutes (vs original's 30min)
- **Nightly reflection**: daily @ 02:00 UTC

---

## 6. Dashboard Premium

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  HEADER: Agent GOD 2 | Portfolio Total | Live Capital | BTC │
├──────────────┬──────────────────────────────────────────────┤
│  SIDEBAR     │  MAIN AREA (6 views)                         │
│  13 strats   │  - Overview (default)                        │
│  with phase  │  - Strategy Detail                           │
│  pills       │  - Pair Heatmap                              │
│  (PAPER/     │  - Correlation Matrix                        │
│  SHADOW/     │  - Promotion Pipeline                        │
│  LIVE)       │  - Live Monitor                              │
├──────────────┴──────────────────────────────────────────────┤
│  FOOTER: CB Status | Regime | Coordinator Last Update       │
└─────────────────────────────────────────────────────────────┘
```

### Views

1. **Overview**: Global equity curve, leaderboard by PF, open positions with TP/SL progress bars, daily/weekly/monthly P&L, per-strategy sparklines
2. **Strategy Detail**: Individual equity curve, trade log (pair, direction, PnL, duration), trades-by-pair donut chart, current parameters, pause/elimination history, promotion phase with progress
3. **Pair Heatmap**: 6 pairs × 13 strategies grid, color = cumulative PnL, click for detail, row/column totals
4. **Correlation Matrix**: 6×6 heatmap updating every 5min, visual alerts on correlation breaks, break history
5. **Promotion Pipeline**: Kanban (PAPER | SHADOW | LIVE), strategy cards with key metrics, progress bars toward promotion criteria, promotion/demotion history
6. **Live Monitor**: LIVE-only strategies, real-time P&L, executed orders with slippage tracking, capital used vs available, per-strategy kill switch button, Live CB proximity alert

### Technical

- Vanilla JS + Chart.js (consistent with original)
- Polling: 5s for open positions, 10s for rest
- Dark theme: green (profit/live), blue (paper), yellow (shadow), red (loss/alert)
- Responsive: desktop + tablet
- Browser push notifications for promotions and Live CB activation

---

## 7. API Endpoints

Port: 9090

### Tournament
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/tournament/status` | All 13 strategies status (includes phase) |
| GET | `/tournament/leaderboard` | Ranking by PF, WR, phase |
| GET | `/tournament/portfolio` | Total portfolio (paper + live separated) |
| GET | `/tournament/strategy/{id}` | Single strategy detail + trade log |
| GET | `/tournament/regime` | Current market regime per pair |
| GET | `/tournament/coordinator` | Last coordinator analysis |
| POST | `/tournament/pause` | Pause all |
| POST | `/tournament/resume` | Resume all |

### Pairs
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/pairs/correlation` | Current correlation matrix |
| GET | `/pairs/heatmap` | PnL by strategy×pair |
| GET | `/pairs/{symbol}/price` | Current price of a pair |

### Promotion
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/promotion/pipeline` | All phases status |
| POST | `/promotion/strategy/{id}/promote` | Force manual promotion |
| POST | `/promotion/strategy/{id}/demote` | Manual demotion (kill switch) |
| GET | `/promotion/history` | Promotion/demotion history |

### Live
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/live/positions` | Open LIVE positions |
| GET | `/live/orders` | Real order history |
| GET | `/live/capital` | Live capital used vs available |
| POST | `/live/halt` | Halt all live immediately |
| POST | `/live/resume` | Resume live |

### Eliminator & Circuit Breaker
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/tournament/eliminator` | Eliminator status |
| POST | `/tournament/strategy/{id}/reactivate` | Reactivate strategy |
| GET | `/tournament/circuit-breaker` | CB status (paper + live) |
| POST | `/tournament/circuit-breaker/reset/{level}` | Reset CB by level |

### Memory & Health
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/memory/{tier}` | Memory entries (short/mid/long) |
| GET | `/learnings` | Persisted learnings |
| GET | `/health` | Health check |
| GET | `/dashboard` | Redirect to dashboard HTML |

---

## 8. Configuration

```env
# AI
ANTHROPIC_API_KEY=...
BRAIN_MODEL=claude-opus-4-6
EXEC_MODEL=claude-opus-4-6

# Binance
BINANCE_API_KEY=...
BINANCE_SECRET=...
BINANCE_TESTNET=false

# Trading
PAIRS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT
MODE=paper
INITIAL_BALANCE=1000.0

# Tournament
TOURNAMENT_ENABLED=true
COORDINATOR_INTERVAL_HOURS=2

# Eliminator
ELIMINATOR_THRESHOLD_PCT=-0.08
ELIMINATOR_PAUSE_HOURS=12
ELIMINATOR_MAX_PAUSES=3

# Circuit Breaker
CB_PAPER_THRESHOLD=-0.12
CB_LIVE_THRESHOLD=-0.05
CB_LIVE_MAX_CONCURRENT=3
CB_LIVE_MAX_CAPITAL_PCT=0.30

# Promotion
PROMOTION_MIN_TRADES=100
PROMOTION_MIN_WR=0.55
PROMOTION_MIN_PF=1.5
PROMOTION_MAX_DD=0.15
PROMOTION_MIN_DAYS=7
PROMOTION_SHADOW_HOURS=48
PROMOTION_LIVE_INITIAL_PCT=0.05
PROMOTION_LIVE_SCALE_STEP=0.05
PROMOTION_LIVE_MAX_PCT=0.50

# Kelly
KELLY_ENABLED=true
KELLY_MIN_TRADES=20
KELLY_MAX_SCALE=2.0
KELLY_MIN_SCALE=0.5

# Memory
MEMORY_HEARTBEAT_INTERVAL_MIN=20
MEMORY_REFLECTION_HOUR_UTC=2

# Signal
MIN_CONFIDENCE=0.72

# Server
PORT=9090
LOG_LEVEL=INFO
```

---

## 9. Dependencies

```
fastapi==0.115.6
uvicorn[standard]==0.32.1
apscheduler==3.10.4
anthropic>=0.39.0
pydantic-settings==2.7.0
pandas>=2.2.3
pandas_ta>=0.4.71b0
numpy>=2.0.2
httpx==0.28.1
pytest==8.3.4
```

Note: Removed `google-genai` and `ccxt` from original. GOD 2 uses Claude Opus 4.6 exclusively (no Gemini fallback) and connects directly to Binance REST API via httpx (no ccxt overhead).

---

## 10. Key Design Decisions

1. **Claude Opus 4.6 for both brain and exec**: The original used Sonnet for self-trainer to save costs. GOD 2 prioritizes quality — every AI decision uses the best model available. Strategies that make it to LIVE will generate enough profit to justify the API cost.

2. **Multi-pair parallel evaluation**: Instead of iterating pairs sequentially, `asyncio.gather()` evaluates all 6 simultaneously. For 1m strategies running every minute, this is critical — sequential would miss signals.

3. **Promotion pipeline with Shadow phase**: Direct paper-to-live is dangerous. Shadow phase catches execution bugs (API errors, slippage, partial fills) with zero capital at risk. 48 hours is enough to catch most edge cases.

4. **Conservative live capital scaling**: Starting at 5% and scaling in 5% increments means a strategy needs to prove itself ~10 times before reaching meaningful capital. This is intentionally slow — real money demands patience.

5. **Dual circuit breaker**: Paper CB at -12% is lenient (simulation should explore). Live CB at -5% is aggressive (real money demands protection). They operate independently.

6. **No Gemini fallback**: The original had Gemini as fallback. GOD 2 drops this — if Anthropic API is down, strategies continue trading on their technical signals without AI analysis. The AI enhances but isn't required for basic operation.

7. **Faster eliminator (24h vs 48h)**: Strategies are faster (1m-15m vs 1h), so damage accumulates faster. -8% in 24h catches problems earlier than -10% in 48h.
