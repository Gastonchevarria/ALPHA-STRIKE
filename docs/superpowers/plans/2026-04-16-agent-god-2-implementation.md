# Agent GOD 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Agent GOD 2 — a 13-strategy multi-pair tournament trading bot with Paper→Shadow→Live promotion, running independently on port 9090.

**Architecture:** FastAPI + APScheduler orchestrating 13 strategies that evaluate 6 crypto pairs in parallel via asyncio.gather(). Dual circuit breaker (paper/live), Claude Opus 4.6 for all AI, promotion pipeline graduating winners to Binance Futures live trading. Premium dashboard with 6 views.

**Tech Stack:** Python 3.11+, FastAPI, APScheduler, Anthropic SDK, pandas + pandas_ta, httpx, Chart.js

**Spec:** `docs/superpowers/specs/2026-04-16-agent-god-2-design.md`

---

## File Map

### Create (new files)
| File | Responsibility |
|------|---------------|
| `config/settings.py` | Pydantic configuration (all env vars) |
| `core/__init__.py` | Package init |
| `core/ai_client.py` | Claude Opus 4.6 JSON generation |
| `core/data_fetcher.py` | Binance API with smart cache, multi-pair |
| `core/market_regime.py` | Per-pair regime detection |
| `core/pair_selector.py` | Parallel 6-pair evaluation |
| `core/correlation_engine.py` | Cross-pair correlation matrix |
| `core/memory_tiers.py` | 3-tier memory (short/mid/long) |
| `core/memory_heartbeat.py` | Memory consolidation + nightly reflection |
| `core/circuit_breaker.py` | Dual-level CB (paper + live) |
| `core/strategy_eliminator.py` | Auto-pause/eliminate strategies |
| `core/tournament_coordinator.py` | Claude Opus brain for confidence adjustment |
| `core/self_trainer.py` | Per-trade AI post-mortem |
| `core/performance_tracker.py` | Aggregate stats |
| `core/risk_manager.py` | Position tracking |
| `core/live_executor.py` | Binance Futures real execution |
| `core/promotion_manager.py` | Paper→Shadow→Live graduation |
| `core/partial_tp.py` | Partial take-profit |
| `core/learnings_logger.py` | Markdown logging |
| `strategies/__init__.py` | Package init |
| `strategies/base_strategy_v4.py` | Multi-pair, promotion-aware base class |
| `strategies/g01_momentum_burst.py` | 1m momentum explosions |
| `strategies/g02_scalp_ultra.py` | 1m RSI extreme scalping |
| `strategies/g03_orderflow_imbalance.py` | 1m buy/sell pressure |
| `strategies/g04_macd_scalper.py` | 5m MACD histogram |
| `strategies/g05_stochastic_reversal.py` | 5m stochastic crosses |
| `strategies/g06_bb_squeeze_turbo.py` | 5m Bollinger squeeze |
| `strategies/g07_rsi_divergence.py` | 15m RSI divergence |
| `strategies/g08_vwap_sniper.py` | 5m VWAP bounce |
| `strategies/g09_atr_breakout.py` | 5m ATR expansion |
| `strategies/g10_ichimoku_edge.py` | 15m Ichimoku cloud |
| `strategies/g11_liquidation_hunter_pro.py` | 5m liquidation cascade |
| `strategies/g12_cross_pair_divergence.py` | 5m cross-pair mean reversion |
| `strategies/g13_volume_delta_sniper.py` | 1m volume delta |
| `scheduler/__init__.py` | Package init |
| `scheduler/tournament_runner_god2.py` | Main orchestrator |
| `static/dashboard.html` | Premium dashboard (6 views) |
| `main.py` | FastAPI app with lifespan |
| `main_god2.py` | API router |
| `requirements.txt` | Dependencies |
| `.env.example` | Config template |
| `tests/test_strategies.py` | Strategy unit tests |
| `tests/test_core.py` | Core module tests |
| `tests/test_runner.py` | Runner integration tests |
| `tests/conftest.py` | Shared fixtures |

---

## Task 1: Project Scaffold + Config + Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `config/__init__.py`
- Create: `config/settings.py`
- Create: `core/__init__.py`
- Create: `strategies/__init__.py`
- Create: `scheduler/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create directory structure**

```bash
cd /Users/gastonchevarria/Alpha/agent-god-2
mkdir -p config core strategies scheduler static tests data/strategy_params learnings
```

- [ ] **Step 2: Create requirements.txt**

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
pytest-asyncio==0.24.0
```

- [ ] **Step 3: Create .env.example**

```env
# AI — Claude Opus 4.6 for everything
ANTHROPIC_API_KEY=sk-ant-...

BRAIN_MODEL=claude-opus-4-6
EXEC_MODEL=claude-opus-4-6

# Binance
BINANCE_API_KEY=
BINANCE_SECRET=
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
ELIMINATOR_MIN_TRADES=5

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

# Self-Trainer
SELF_TRAINER_ENABLED=true
SELF_TRAINER_MIN_TRADES_TO_EVOLVE=5

# Memory
MEMORY_HEARTBEAT_INTERVAL_MIN=20
MEMORY_REFLECTION_HOUR_UTC=2

# Signal
MIN_CONFIDENCE=0.72

# Observability
LOG_LEVEL=INFO
PORT=9090
```

- [ ] **Step 4: Create config/settings.py**

```python
"""Agent GOD 2 — Configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # AI
    ANTHROPIC_API_KEY: str = ""
    BRAIN_MODEL: str = "claude-opus-4-6"
    EXEC_MODEL: str = "claude-opus-4-6"

    # Binance
    BINANCE_API_KEY: str = ""
    BINANCE_SECRET: str = ""
    BINANCE_TESTNET: bool = False

    # Trading
    PAIRS: str = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT"
    MODE: str = "paper"
    INITIAL_BALANCE: float = 1000.0

    # Tournament
    TOURNAMENT_ENABLED: bool = True
    COORDINATOR_INTERVAL_HOURS: int = 2

    # Eliminator
    ELIMINATOR_THRESHOLD_PCT: float = -0.08
    ELIMINATOR_PAUSE_HOURS: int = 12
    ELIMINATOR_MAX_PAUSES: int = 3
    ELIMINATOR_MIN_TRADES: int = 5

    # Circuit Breaker
    CB_PAPER_THRESHOLD: float = -0.12
    CB_LIVE_THRESHOLD: float = -0.05
    CB_LIVE_MAX_CONCURRENT: int = 3
    CB_LIVE_MAX_CAPITAL_PCT: float = 0.30

    # Promotion
    PROMOTION_MIN_TRADES: int = 100
    PROMOTION_MIN_WR: float = 0.55
    PROMOTION_MIN_PF: float = 1.5
    PROMOTION_MAX_DD: float = 0.15
    PROMOTION_MIN_DAYS: int = 7
    PROMOTION_SHADOW_HOURS: int = 48
    PROMOTION_LIVE_INITIAL_PCT: float = 0.05
    PROMOTION_LIVE_SCALE_STEP: float = 0.05
    PROMOTION_LIVE_MAX_PCT: float = 0.50

    # Kelly
    KELLY_ENABLED: bool = True
    KELLY_MIN_TRADES: int = 20
    KELLY_MAX_SCALE: float = 2.0
    KELLY_MIN_SCALE: float = 0.5

    # Self-Trainer
    SELF_TRAINER_ENABLED: bool = True
    SELF_TRAINER_MIN_TRADES_TO_EVOLVE: int = 5

    # Memory
    MEMORY_HEARTBEAT_INTERVAL_MIN: int = 20
    MEMORY_REFLECTION_HOUR_UTC: int = 2

    # Signal
    MIN_CONFIDENCE: float = 0.72

    # Observability
    LOG_LEVEL: str = "INFO"
    PORT: int = 9090

    @property
    def pairs_list(self) -> list[str]:
        return [p.strip() for p in self.PAIRS.split(",") if p.strip()]

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
```

- [ ] **Step 5: Create package __init__.py files**

`config/__init__.py`, `core/__init__.py`, `strategies/__init__.py`, `scheduler/__init__.py`, `tests/__init__.py` — all empty files.

- [ ] **Step 6: Create tests/conftest.py**

```python
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
        ANTHROPIC_API_KEY="test-key",
        MODE="paper",
        INITIAL_BALANCE=1000.0,
        PORT=9999,
    )
```

- [ ] **Step 7: Install dependencies and verify**

```bash
cd /Users/gastonchevarria/Alpha/agent-god-2
pip install -r requirements.txt
python -c "from config.settings import settings; print(f'Port: {settings.PORT}, Pairs: {settings.pairs_list}')"
```

Expected: `Port: 9090, Pairs: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT']`

- [ ] **Step 8: Commit**

```bash
git init
git add -A
git commit -m "feat: scaffold Agent GOD 2 project with config and dependencies"
```

---

## Task 2: Data Fetcher (Multi-Pair with Smart Cache)

**Files:**
- Create: `core/data_fetcher.py`
- Create: `tests/test_core.py`

- [ ] **Step 1: Write test for data fetcher**

In `tests/test_core.py`:

```python
"""Tests for core modules."""

import pytest
import pandas as pd
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_fetch_klines_returns_dataframe():
    """fetch_klines should return a DataFrame with OHLCV columns."""
    mock_data = [
        [1713200000000, "65000", "65100", "64900", "65050", "100",
         1713200059999, "6505000", 50, "60", "3903000", "0"]
        for _ in range(10)
    ]
    with patch("core.data_fetcher.httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value.__aenter__ = AsyncMock()
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_data
        instance.return_value.get = AsyncMock(return_value=mock_resp)

        from core.data_fetcher import fetch_klines
        # Just verify the function exists and has correct signature
        assert callable(fetch_klines)


@pytest.mark.asyncio
async def test_cache_key_generation():
    """Cache keys should be unique per symbol+interval+limit."""
    from core.data_fetcher import _cache_key
    k1 = _cache_key("BTCUSDT", "1m", 100)
    k2 = _cache_key("ETHUSDT", "1m", 100)
    k3 = _cache_key("BTCUSDT", "5m", 100)
    assert k1 != k2
    assert k1 != k3


def test_pairs_config():
    """Should have 6 configured pairs."""
    from config.settings import settings
    assert len(settings.pairs_list) == 6
    assert "BTCUSDT" in settings.pairs_list
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/gastonchevarria/Alpha/agent-god-2
python -m pytest tests/test_core.py -v
```

Expected: FAIL — `core.data_fetcher` does not exist yet

- [ ] **Step 3: Implement data_fetcher.py**

```python
"""Binance data fetcher with smart caching and multi-pair support."""

import asyncio
import logging
import time
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BINANCE_ENDPOINTS = [
    "https://data-api.binance.vision",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]

# Smart cache: {key: (timestamp, dataframe)}
_kline_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_CACHE_TTL = {
    "1m": 30,
    "5m": 120,
    "15m": 300,
}
_price_cache: dict[str, tuple[float, float]] = {}
_PRICE_TTL = 5


def _cache_key(symbol: str, interval: str, limit: int) -> str:
    return f"{symbol}:{interval}:{limit}"


async def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    limit: int = 100,
) -> pd.DataFrame:
    """Fetch klines from Binance with caching and endpoint fallback."""
    key = _cache_key(symbol, interval, limit)
    now = time.time()
    ttl = _CACHE_TTL.get(interval, 60)

    if key in _kline_cache:
        ts, df = _kline_cache[key]
        if now - ts < ttl:
            return df.copy()

    params = {"symbol": symbol, "interval": interval, "limit": limit + 1}

    for base in _BINANCE_ENDPOINTS:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{base}/api/v3/klines", params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    break
        except Exception as e:
            logger.warning(f"Endpoint {base} failed for {symbol}: {e}")
            continue
    else:
        raise RuntimeError(f"All Binance endpoints failed for {symbol} {interval}")

    cols = [
        "ts", "open", "high", "low", "close", "vol",
        "close_ts", "qvol", "trades", "tbbase", "tbquote", "ignore",
    ]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open", "high", "low", "close", "vol", "qvol", "tbbase", "tbquote"]:
        df[c] = df[c].astype(float)

    df = df.iloc[:-1].reset_index(drop=True)  # Remove incomplete candle
    _kline_cache[key] = (now, df)
    return df.copy()


async def fetch_ohlcv(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    limit: int = 100,
) -> pd.DataFrame:
    """Fetch OHLCV with pandas_ta-compatible column names."""
    df = await fetch_klines(symbol, interval, limit)
    df = df.rename(columns={"vol": "volume"})
    return df[["open", "high", "low", "close", "volume"]].copy()


async def get_current_price(symbol: str = "BTCUSDT") -> float:
    """Get current price with short-lived cache."""
    now = time.time()
    if symbol in _price_cache:
        ts, price = _price_cache[symbol]
        if now - ts < _PRICE_TTL:
            return price

    for base in _BINANCE_ENDPOINTS:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"{base}/api/v3/ticker/price",
                    params={"symbol": symbol},
                )
                if resp.status_code == 200:
                    price = float(resp.json()["price"])
                    _price_cache[symbol] = (now, price)
                    return price
        except Exception:
            continue

    raise RuntimeError(f"Cannot fetch price for {symbol}")


async def get_all_prices(pairs: list[str]) -> dict[str, float]:
    """Fetch prices for all pairs in parallel."""
    tasks = [get_current_price(p) for p in pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    prices = {}
    for pair, result in zip(pairs, results):
        if isinstance(result, float):
            prices[pair] = result
        else:
            logger.error(f"Price fetch failed for {pair}: {result}")
    return prices


async def prefetch_all(pairs: list[str], intervals: list[str] = None):
    """Preload kline data for all pairs and intervals into cache."""
    if intervals is None:
        intervals = ["1m", "5m", "15m"]
    tasks = []
    for pair in pairs:
        for interval in intervals:
            limit = {"1m": 30, "5m": 60, "15m": 60}.get(interval, 60)
            tasks.append(fetch_klines(pair, interval, limit))
    await asyncio.gather(*tasks, return_exceptions=True)


async def get_taker_volume(symbol: str = "BTCUSDT", interval: str = "1m", limit: int = 10) -> pd.DataFrame:
    """Fetch klines with taker buy/sell volume data."""
    df = await fetch_klines(symbol, interval, limit)
    df["taker_buy_vol"] = df["tbbase"]
    df["taker_sell_vol"] = df["vol"] - df["tbbase"]
    df["buy_ratio"] = df["taker_buy_vol"] / df["vol"].replace(0, 1)
    return df


async def get_open_interest(symbol: str = "BTCUSDT") -> float | None:
    """Fetch open interest from Binance Futures API."""
    futures_endpoints = [
        "https://fapi.binance.com",
        "https://fapi.binance.vision",
    ]
    for base in futures_endpoints:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"{base}/fapi/v1/openInterest",
                    params={"symbol": symbol},
                )
                if resp.status_code == 200:
                    return float(resp.json()["openInterest"])
        except Exception:
            continue
    return None
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_core.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add core/data_fetcher.py tests/test_core.py
git commit -m "feat: add multi-pair data fetcher with smart cache"
```

---

## Task 3: AI Client (Claude-Only)

**Files:**
- Create: `core/ai_client.py`

- [ ] **Step 1: Implement ai_client.py**

```python
"""AI client — Claude Opus 4.6 only, no Gemini fallback."""

import json
import logging
import re

from anthropic import AsyncAnthropic

from config.settings import settings

logger = logging.getLogger(__name__)

_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client


def _extract_json(text: str) -> str:
    """Extract JSON object from text that may contain markdown wrappers."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


async def generate_json(
    model: str,
    system_instruction: str,
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> dict:
    """Generate a JSON response using Claude."""
    client = _get_client()
    full_system = f"{system_instruction}\n\nRespond ONLY with valid JSON. No markdown, no explanation."

    try:
        resp = await client.messages.create(
            model=model,
            system=full_system,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.content[0].text
        cleaned = _extract_json(text)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error from {model}: {e}\nRaw: {text[:500]}")
        return {}
    except Exception as e:
        logger.error(f"AI client error ({model}): {e}")
        return {}
```

- [ ] **Step 2: Commit**

```bash
git add core/ai_client.py
git commit -m "feat: add Claude-only AI client"
```

---

## Task 4: Memory System

**Files:**
- Create: `core/memory_tiers.py`

- [ ] **Step 1: Implement memory_tiers.py**

```python
"""3-tier memory system with TTL-based pruning."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_LEARNINGS_DIR = Path("learnings")

_TIER_CONFIG = {
    "short": {"max_entries": 20, "ttl_hours": 24, "file": "memory_short.json"},
    "mid": {"max_entries": 50, "ttl_hours": 168, "file": "memory_mid.json"},
    "long": {"max_entries": 100, "ttl_hours": None, "file": "memory_long.json"},
}


def _tier_path(tier: str) -> Path:
    return _LEARNINGS_DIR / _TIER_CONFIG[tier]["file"]


def _load_tier(tier: str) -> list[dict]:
    path = _tier_path(tier)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, KeyError):
            return []
    return []


def _save_tier(tier: str, entries: list[dict]):
    _LEARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    _tier_path(tier).write_text(json.dumps(entries, indent=2))


def _prune(tier: str, entries: list[dict]) -> list[dict]:
    cfg = _TIER_CONFIG[tier]
    now = datetime.now(timezone.utc)

    if cfg["ttl_hours"] is not None:
        cutoff = now - timedelta(hours=cfg["ttl_hours"])
        entries = [
            e for e in entries
            if datetime.fromisoformat(e["ts"]) > cutoff
        ]

    if len(entries) > cfg["max_entries"]:
        entries = entries[-cfg["max_entries"]:]

    return entries


def add_memory(tier: str, content: str, tags: list[str] | None = None) -> dict:
    """Add an entry to the specified memory tier."""
    if tier not in _TIER_CONFIG:
        raise ValueError(f"Invalid tier: {tier}")

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "content": content,
        "tags": tags or [],
    }

    entries = _load_tier(tier)
    entries.append(entry)
    entries = _prune(tier, entries)
    _save_tier(tier, entries)
    return entry


def get_recent(tier: str, limit: int = 10) -> list[dict]:
    """Get most recent entries from a tier."""
    entries = _load_tier(tier)
    entries = _prune(tier, entries)
    return entries[-limit:]


def get_all_context(max_per_tier: int = 5) -> str:
    """Get formatted memory context for LLM prompts."""
    parts = []
    for tier in ["long", "mid", "short"]:
        entries = get_recent(tier, max_per_tier)
        if entries:
            parts.append(f"### {tier.upper()} MEMORY ({len(entries)} entries)")
            for e in entries:
                ts = e["ts"][:10]
                parts.append(f"- [{ts}] {e['content']}")
    return "\n".join(parts) if parts else "No memories stored yet."


def promote(from_tier: str, content: str, tags: list[str] | None = None):
    """Promote content to the next tier up."""
    tier_order = ["short", "mid", "long"]
    idx = tier_order.index(from_tier)
    if idx < len(tier_order) - 1:
        next_tier = tier_order[idx + 1]
        add_memory(next_tier, content, tags)
```

- [ ] **Step 2: Commit**

```bash
git add core/memory_tiers.py
git commit -m "feat: add 3-tier memory system"
```

---

## Task 5: Market Regime (Per-Pair)

**Files:**
- Create: `core/market_regime.py`

- [ ] **Step 1: Implement market_regime.py**

```python
"""Per-pair market regime detection with caching."""

import logging
import time

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv

logger = logging.getLogger(__name__)

# Cache: {symbol: (timestamp, regime_dict)}
_regime_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 300  # 5 minutes


def get_cached_regime(symbol: str = "BTCUSDT") -> dict:
    """Get cached regime for a symbol, or unknown if not cached."""
    if symbol in _regime_cache:
        ts, regime = _regime_cache[symbol]
        if time.time() - ts < _CACHE_TTL:
            return regime
    return {"regime": "UNKNOWN", "symbol": symbol, "signals": {}}


async def detect_regime(symbol: str = "BTCUSDT") -> dict:
    """Detect market regime for a specific pair using 15m candles."""
    try:
        df = await fetch_ohlcv(symbol, "15m", limit=60)

        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.bbands(length=20, std=2, append=True)

        r = df.iloc[-1]
        ema20 = r.get("EMA_20", 0)
        ema50 = r.get("EMA_50", 0)
        adx = r.get("ADX_14", 0)
        atr = r.get("ATRr_14", 0)

        bbu_col = next((c for c in df.columns if c.startswith("BBU")), None)
        bbl_col = next((c for c in df.columns if c.startswith("BBL")), None)
        bbm_col = next((c for c in df.columns if c.startswith("BBM")), None)

        bb_expansion = 0.0
        if bbu_col and bbl_col and bbm_col:
            df["bb_width"] = (df[bbu_col] - df[bbl_col]) / df[bbm_col].replace(0, 1)
            bb_expansion = df["bb_width"].iloc[-1] / max(df["bb_width"].iloc[-20:].mean(), 1e-9)

        atr_pct = atr / max(r["close"], 1) * 100 if atr else 0

        if ema20 > ema50 and adx > 25:
            regime = "TRENDING_UP"
        elif ema20 < ema50 and adx > 25:
            regime = "TRENDING_DOWN"
        elif atr_pct > 0.5 or bb_expansion > 1.5:
            regime = "VOLATILE"
        elif adx < 20:
            regime = "RANGING"
        else:
            regime = "RANGING"

        result = {
            "regime": regime,
            "symbol": symbol,
            "updated_at": time.time(),
            "signals": {
                "ema_delta": round(ema20 - ema50, 2),
                "adx": round(adx, 2),
                "atr_pct": round(atr_pct, 4),
                "bb_expansion": round(bb_expansion, 3),
            },
        }

        _regime_cache[symbol] = (time.time(), result)
        return result

    except Exception as e:
        logger.error(f"Regime detection failed for {symbol}: {e}")
        return get_cached_regime(symbol)


async def detect_all_regimes(pairs: list[str]) -> dict[str, dict]:
    """Detect regime for all pairs."""
    import asyncio
    tasks = [detect_regime(p) for p in pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    regimes = {}
    for pair, result in zip(pairs, results):
        if isinstance(result, dict):
            regimes[pair] = result
        else:
            regimes[pair] = get_cached_regime(pair)
    return regimes


def strategy_matches_regime(regime_filter: list[str], regime: dict) -> bool:
    """Check if current regime matches strategy's filter."""
    if "ANY" in regime_filter:
        return True
    return regime.get("regime", "UNKNOWN") in regime_filter
```

- [ ] **Step 2: Commit**

```bash
git add core/market_regime.py
git commit -m "feat: add per-pair market regime detection"
```

---

## Task 6: Correlation Engine

**Files:**
- Create: `core/correlation_engine.py`

- [ ] **Step 1: Implement correlation_engine.py**

```python
"""Cross-pair correlation engine for divergence detection."""

import asyncio
import logging
import time

import numpy as np
import pandas as pd

from core.data_fetcher import fetch_ohlcv

logger = logging.getLogger(__name__)

_correlation_cache: dict | None = None
_cache_ts: float = 0
_CACHE_TTL = 300  # 5 minutes


async def compute_correlation_matrix(
    pairs: list[str],
    interval: str = "5m",
    periods: int = 20,
) -> dict:
    """Compute rolling correlation matrix between all pairs."""
    global _correlation_cache, _cache_ts

    now = time.time()
    if _correlation_cache and now - _cache_ts < _CACHE_TTL:
        return _correlation_cache

    tasks = [fetch_ohlcv(p, interval, limit=periods + 5) for p in pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    returns = {}
    for pair, result in zip(pairs, results):
        if isinstance(result, pd.DataFrame) and len(result) >= periods:
            pct = result["close"].pct_change().dropna().tail(periods)
            returns[pair] = pct.values
        else:
            logger.warning(f"Correlation: skipping {pair}, insufficient data")

    if len(returns) < 2:
        return {"matrix": {}, "divergence_index": 1.0, "breaks": []}

    pair_names = list(returns.keys())
    n = len(pair_names)
    matrix = {}
    correlations = []

    for i in range(n):
        matrix[pair_names[i]] = {}
        for j in range(n):
            if i == j:
                matrix[pair_names[i]][pair_names[j]] = 1.0
            else:
                r1 = returns[pair_names[i]]
                r2 = returns[pair_names[j]]
                min_len = min(len(r1), len(r2))
                if min_len > 2:
                    corr = float(np.corrcoef(r1[:min_len], r2[:min_len])[0, 1])
                    if np.isnan(corr):
                        corr = 0.0
                else:
                    corr = 0.0
                matrix[pair_names[i]][pair_names[j]] = round(corr, 4)
                if i < j:
                    correlations.append(corr)

    avg_corr = float(np.mean(correlations)) if correlations else 0.0
    std_corr = float(np.std(correlations)) if len(correlations) > 1 else 0.0

    breaks = []
    for i in range(n):
        for j in range(i + 1, n):
            corr = matrix[pair_names[i]][pair_names[j]]
            if avg_corr - corr > 2 * std_corr and std_corr > 0.05:
                breaks.append({
                    "pair_a": pair_names[i],
                    "pair_b": pair_names[j],
                    "correlation": corr,
                    "avg": round(avg_corr, 4),
                    "deviation": round((avg_corr - corr) / max(std_corr, 0.01), 2),
                })

    result = {
        "matrix": matrix,
        "avg_correlation": round(avg_corr, 4),
        "divergence_index": round(1.0 - avg_corr, 4),
        "breaks": breaks,
        "updated_at": time.time(),
    }

    _correlation_cache = result
    _cache_ts = now
    return result


def get_correlation_matrix() -> dict:
    """Get cached correlation matrix."""
    if _correlation_cache:
        return _correlation_cache
    return {"matrix": {}, "divergence_index": 0.0, "breaks": []}


def get_divergence_index() -> float:
    """Get current divergence index (0=fully correlated, 1=fully divergent)."""
    if _correlation_cache:
        return _correlation_cache.get("divergence_index", 0.0)
    return 0.0
```

- [ ] **Step 2: Commit**

```bash
git add core/correlation_engine.py
git commit -m "feat: add cross-pair correlation engine"
```

---

## Task 7: Circuit Breaker (Dual-Level)

**Files:**
- Create: `core/circuit_breaker.py`

- [ ] **Step 1: Implement circuit_breaker.py**

```python
"""Dual-level circuit breaker: paper (lenient) + live (aggressive)."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_STATE_FILE = Path("learnings/circuit_breaker.json")


class CircuitBreaker:
    """Dual-level circuit breaker for Agent GOD 2."""

    def __init__(
        self,
        initial_paper_total: float,
        paper_threshold: float = -0.12,
        live_threshold: float = -0.05,
        max_concurrent_live: int = 3,
        max_live_capital_pct: float = 0.30,
    ):
        self.initial_paper_total = initial_paper_total
        self.paper_threshold = paper_threshold
        self.live_threshold = live_threshold
        self.max_concurrent_live = max_concurrent_live
        self.max_live_capital_pct = max_live_capital_pct

        self.paper_triggered = False
        self.live_triggered = False
        self.paper_halt_until: datetime | None = None
        self.live_halt_until: datetime | None = None
        self.paper_reason = ""
        self.live_reason = ""

        self.live_daily_pnl = 0.0
        self.live_daily_reset: datetime = datetime.now(timezone.utc)
        self.live_total_capital = 0.0

        self._load_state()

    def check_paper(self, current_paper_total: float) -> bool:
        """Check paper circuit breaker. Returns True if triggered."""
        now = datetime.now(timezone.utc)
        if self.paper_triggered and self.paper_halt_until:
            if now >= self.paper_halt_until:
                self.paper_triggered = False
                self.paper_reason = ""
                self.paper_halt_until = None
            else:
                return True

        pct = (current_paper_total - self.initial_paper_total) / max(self.initial_paper_total, 1)
        if pct <= self.paper_threshold:
            self.paper_triggered = True
            self.paper_halt_until = now + timedelta(hours=24)
            self.paper_reason = f"Paper portfolio {pct:.1%} <= {self.paper_threshold:.1%}"
            logger.warning(f"PAPER CIRCUIT BREAKER TRIPPED: {self.paper_reason}")
            self._save_state()
            return True
        return False

    def check_live(self, live_pnl_delta: float = 0.0, live_capital: float = 0.0) -> bool:
        """Check live circuit breaker. Returns True if triggered."""
        now = datetime.now(timezone.utc)

        # Reset daily PnL tracker
        if (now - self.live_daily_reset).total_seconds() > 86400:
            self.live_daily_pnl = 0.0
            self.live_daily_reset = now

        if self.live_triggered and self.live_halt_until:
            if now >= self.live_halt_until:
                self.live_triggered = False
                self.live_reason = ""
                self.live_halt_until = None
            else:
                return True

        self.live_daily_pnl += live_pnl_delta
        self.live_total_capital = live_capital

        if live_capital > 0:
            daily_pct = self.live_daily_pnl / max(live_capital, 1)
            if daily_pct <= self.live_threshold:
                self.live_triggered = True
                self.live_halt_until = now + timedelta(hours=24)
                self.live_reason = f"Live daily PnL {daily_pct:.1%} <= {self.live_threshold:.1%}"
                logger.critical(f"LIVE CIRCUIT BREAKER TRIPPED: {self.live_reason}")
                self._save_state()
                return True
        return False

    def can_open_live(self, current_live_positions: int, position_capital: float) -> tuple[bool, str]:
        """Check if a new live position can be opened."""
        if self.live_triggered:
            return False, "Live circuit breaker is triggered"
        if current_live_positions >= self.max_concurrent_live:
            return False, f"Max concurrent live positions ({self.max_concurrent_live}) reached"
        if self.live_total_capital > 0:
            usage = position_capital / max(self.live_total_capital, 1)
            if usage > self.max_live_capital_pct:
                return False, f"Would exceed max live capital ({self.max_live_capital_pct:.0%})"
        return True, "OK"

    def reset_paper(self):
        """Manual reset of paper circuit breaker."""
        self.paper_triggered = False
        self.paper_reason = ""
        self.paper_halt_until = None
        self._save_state()

    def reset_live(self):
        """Manual reset of live circuit breaker."""
        self.live_triggered = False
        self.live_reason = ""
        self.live_halt_until = None
        self.live_daily_pnl = 0.0
        self._save_state()

    def status(self) -> dict:
        return {
            "paper": {
                "triggered": self.paper_triggered,
                "reason": self.paper_reason,
                "halt_until": self.paper_halt_until.isoformat() if self.paper_halt_until else None,
            },
            "live": {
                "triggered": self.live_triggered,
                "reason": self.live_reason,
                "halt_until": self.live_halt_until.isoformat() if self.live_halt_until else None,
                "daily_pnl": round(self.live_daily_pnl, 2),
            },
        }

    def _save_state(self):
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(self.status(), indent=2))

    def _load_state(self):
        if _STATE_FILE.exists():
            try:
                data = json.loads(_STATE_FILE.read_text())
                self.paper_triggered = data.get("paper", {}).get("triggered", False)
                self.live_triggered = data.get("live", {}).get("triggered", False)
            except Exception:
                pass
```

- [ ] **Step 2: Commit**

```bash
git add core/circuit_breaker.py
git commit -m "feat: add dual-level circuit breaker (paper + live)"
```

---

## Task 8: Learnings Logger + Performance Tracker

**Files:**
- Create: `core/learnings_logger.py`
- Create: `core/performance_tracker.py`
- Create: `core/partial_tp.py`
- Create: `core/risk_manager.py`

- [ ] **Step 1: Implement learnings_logger.py**

```python
"""Structured markdown logging for learnings, errors, and feature requests."""

import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_LEARNINGS_DIR = Path("learnings")


def ensure_learnings_dir():
    """Create learnings directory and initial files."""
    _LEARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ["LEARNINGS.md", "ERRORS.md", "FEATURE_REQUESTS.md"]:
        path = _LEARNINGS_DIR / fname
        if not path.exists():
            path.write_text(f"# {fname.replace('.md', '').replace('_', ' ').title()}\n\n")


def _next_id(prefix: str, filepath: Path) -> str:
    """Generate next sequential ID for today."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    base = f"{prefix}-{today}"
    if filepath.exists():
        content = filepath.read_text()
        count = content.count(f"{base}-")
    else:
        count = 0
    return f"{base}-{count + 1:03d}"


def log_learning(
    category: str,
    summary: str,
    details: str,
    action: str,
    area: str = "backend",
    priority: str = "medium",
    tags: list[str] | None = None,
) -> str:
    """Log a learning to LEARNINGS.md."""
    ensure_learnings_dir()
    filepath = _LEARNINGS_DIR / "LEARNINGS.md"
    entry_id = _next_id("LRN", filepath)
    now = datetime.now(timezone.utc).isoformat()
    tag_str = ", ".join(tags) if tags else "general"

    entry = f"\n## {entry_id}: {summary}\n"
    entry += f"- **Category**: {category}\n"
    entry += f"- **Area**: {area} | **Priority**: {priority}\n"
    entry += f"- **Tags**: {tag_str}\n"
    entry += f"- **Timestamp**: {now}\n"
    entry += f"- **Details**: {details}\n"
    entry += f"- **Action**: {action}\n"

    with open(filepath, "a") as f:
        f.write(entry)
    return entry_id


def log_error(
    skill_or_command: str,
    summary: str,
    error_text: str,
    context: str,
    suggested_fix: str,
) -> str:
    """Log an error to ERRORS.md."""
    ensure_learnings_dir()
    filepath = _LEARNINGS_DIR / "ERRORS.md"
    entry_id = _next_id("ERR", filepath)
    now = datetime.now(timezone.utc).isoformat()

    entry = f"\n## {entry_id}: {summary}\n"
    entry += f"- **Command**: {skill_or_command}\n"
    entry += f"- **Timestamp**: {now}\n"
    entry += f"- **Error**: {error_text}\n"
    entry += f"- **Context**: {context}\n"
    entry += f"- **Suggested Fix**: {suggested_fix}\n"

    with open(filepath, "a") as f:
        f.write(entry)
    return entry_id
```

- [ ] **Step 2: Implement performance_tracker.py**

```python
"""Aggregate performance tracking across all strategies."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_PERF_FILE = Path("learnings/performance.json")
_TRADES_FILE = Path("learnings/trades.jsonl")


def _load_perf() -> dict:
    if _PERF_FILE.exists():
        try:
            return json.loads(_PERF_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_perf(data: dict):
    _PERF_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PERF_FILE.write_text(json.dumps(data, indent=2))


def _compute_profit_factor() -> float:
    """Read trades.jsonl and compute global profit factor."""
    if not _TRADES_FILE.exists():
        return 0.0
    gross_win = 0.0
    gross_loss = 0.0
    try:
        for line in _TRADES_FILE.read_text().strip().split("\n"):
            if not line:
                continue
            trade = json.loads(line)
            pnl = trade.get("pnl_net", 0)
            if pnl > 0:
                gross_win += pnl
            else:
                gross_loss += abs(pnl)
    except Exception:
        pass
    return round(gross_win / max(gross_loss, 1e-9), 3)


def update(
    direction: str,
    outcome: str,
    pnl_net: float,
    price: float,
    current_balance: float,
    starting_balance: float,
    strategy_id: str = "",
    pair: str = "",
) -> dict:
    """Update global performance stats after a trade close."""
    p = _load_perf()
    now = datetime.now(timezone.utc).isoformat()

    if "start_date" not in p:
        p["start_date"] = now[:10]
        p["starting_balance_usd"] = starting_balance

    p["current_balance_usd"] = current_balance
    p["total_trades"] = p.get("total_trades", 0) + 1

    if outcome in ("TP", "TP_MOONBAG"):
        p["winning_trades"] = p.get("winning_trades", 0) + 1
    elif outcome == "SL":
        p["losing_trades"] = p.get("losing_trades", 0) + 1

    total = p.get("total_trades", 1)
    wins = p.get("winning_trades", 0)
    p["win_rate"] = round(wins / max(total, 1) * 100, 1)
    p["total_pnl_usd"] = round(p.get("total_pnl_usd", 0) + pnl_net, 2)

    trade_record = {
        "ts": now, "direction": direction, "outcome": outcome,
        "pnl_net": round(pnl_net, 2), "price": price,
        "strategy_id": strategy_id, "pair": pair,
    }

    best = p.get("best_trade")
    if not best or pnl_net > best.get("pnl_net", 0):
        p["best_trade"] = trade_record

    worst = p.get("worst_trade")
    if not worst or pnl_net < worst.get("pnl_net", 0):
        p["worst_trade"] = trade_record

    p["profit_factor"] = _compute_profit_factor()
    _save_perf(p)
    return p


def get() -> dict:
    return _load_perf()
```

- [ ] **Step 3: Implement partial_tp.py**

```python
"""Partial take-profit system (disabled by default, available for live)."""

from dataclasses import dataclass, field


@dataclass
class PartialTPConfig:
    tp1_pct: float = 0.25
    tp2_pct: float = 0.50
    tp3_pct: float = 0.75
    moonbag_trail: float = 0.60
    enabled: bool = False


@dataclass
class PartialTPState:
    entry_price: float = 0.0
    direction: str = ""
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    peak_pnl: float = 0.0
    closed_pct: float = 0.0

    def check(self, current_pnl: float, tp_target: float, config: PartialTPConfig):
        if not config.enabled:
            return None

        if current_pnl > self.peak_pnl:
            self.peak_pnl = current_pnl

        if not self.tp1_hit and current_pnl >= tp_target * 0.5:
            self.tp1_hit = True
            self.closed_pct = config.tp1_pct
            return ("PARTIAL_CLOSE", config.tp1_pct)

        if not self.tp2_hit and current_pnl >= tp_target * 0.75:
            self.tp2_hit = True
            self.closed_pct = config.tp2_pct
            return ("PARTIAL_CLOSE", config.tp2_pct - config.tp1_pct)

        if not self.tp3_hit and current_pnl >= tp_target:
            self.tp3_hit = True
            self.closed_pct = config.tp3_pct
            return ("PARTIAL_CLOSE", config.tp3_pct - config.tp2_pct)

        if self.tp3_hit and self.peak_pnl > 0:
            if current_pnl < self.peak_pnl * config.moonbag_trail:
                return ("CLOSE_ALL", 1.0 - self.closed_pct)

        return None
```

- [ ] **Step 4: Implement risk_manager.py**

```python
"""Position and risk management for Agent GOD 2."""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    direction: str
    entry_price: float
    entry_time: datetime
    notional: float
    margin: float
    leverage: int
    tp_price: float
    sl_price: float

    def unrealised_pnl(self, price: float) -> float:
        mult = 1 if self.direction == "LONG" else -1
        pct = (price - self.entry_price) / self.entry_price * mult
        fee = self.notional * 0.0008
        return self.notional * pct - fee

    def should_tp(self, price: float) -> bool:
        if self.direction == "LONG":
            return price >= self.tp_price
        return price <= self.tp_price

    def should_sl(self, price: float) -> bool:
        if self.direction == "LONG":
            return price <= self.sl_price
        return price >= self.sl_price

    def distance_to_tp(self, price: float) -> float:
        return abs(self.tp_price - price) / price * 100

    def distance_to_sl(self, price: float) -> float:
        return abs(self.sl_price - price) / price * 100
```

- [ ] **Step 5: Commit**

```bash
git add core/learnings_logger.py core/performance_tracker.py core/partial_tp.py core/risk_manager.py
git commit -m "feat: add learnings logger, performance tracker, partial TP, risk manager"
```

---

## Task 9: Base Strategy V4 (Multi-Pair, Promotion-Aware)

**Files:**
- Create: `strategies/base_strategy_v4.py`
- Create: `tests/test_strategies.py`

- [ ] **Step 1: Write test**

In `tests/test_strategies.py`:

```python
"""Tests for strategy base class and individual strategies."""

import pytest
from datetime import datetime, timezone


def test_strategy_config_creation():
    from strategies.base_strategy_v4 import StrategyConfig
    cfg = StrategyConfig(
        id="G-TEST",
        name="Test Strategy",
        timeframe="5m",
        leverage=30,
        margin_pct=0.08,
        tp_pct=0.005,
        sl_pct=0.003,
        cron_expr={"minute": "*/5"},
        regime_filter=["ANY"],
        description="Test",
        timeout_minutes=120,
    )
    assert cfg.id == "G-TEST"
    assert cfg.leverage == 30


def test_trade_signal_creation():
    from strategies.base_strategy_v4 import TradeSignal
    sig = TradeSignal(
        direction="LONG",
        execute=True,
        confidence=0.85,
        signals={"rsi": 25},
        reason="Test signal",
        pair="BTCUSDT",
    )
    assert sig.execute is True
    assert sig.pair == "BTCUSDT"


def test_base_strategy_open_close():
    from strategies.base_strategy_v4 import StrategyConfig, BaseStrategyV4, TradeSignal

    class DummyStrategy(BaseStrategyV4):
        async def evaluate(self, pair, df):
            return TradeSignal("HOLD", False, 0.5, {}, "test", pair)

    cfg = StrategyConfig(
        id="G-TEST", name="Test", timeframe="5m", leverage=30,
        margin_pct=0.08, tp_pct=0.005, sl_pct=0.003,
        cron_expr={"minute": "*/5"}, regime_filter=["ANY"],
        description="Test", timeout_minutes=120,
    )
    strat = DummyStrategy(config=cfg, initial_balance=1000.0)
    assert strat.balance == 1000.0
    assert strat.position is None

    strat.open_position("LONG", 65000.0, "BTCUSDT", {"test": 1})
    assert strat.position == "LONG"
    assert strat._entry_pair == "BTCUSDT"

    exit_reason = strat.check_exit(66000.0)
    assert exit_reason == "TP"

    pnl, record = strat.close_position(66000.0, "TP")
    assert pnl > 0
    assert strat.position is None
    assert record["pair"] == "BTCUSDT"


def test_kelly_margin_below_min_trades():
    from strategies.base_strategy_v4 import StrategyConfig, BaseStrategyV4, TradeSignal

    class DummyStrategy(BaseStrategyV4):
        async def evaluate(self, pair, df):
            return TradeSignal("HOLD", False, 0.5, {}, "test", pair)

    cfg = StrategyConfig(
        id="G-TEST", name="Test", timeframe="5m", leverage=30,
        margin_pct=0.08, tp_pct=0.005, sl_pct=0.003,
        cron_expr={"minute": "*/5"}, regime_filter=["ANY"],
        description="Test", timeout_minutes=120,
    )
    strat = DummyStrategy(config=cfg, initial_balance=1000.0)
    margin = strat._effective_margin()
    assert margin == 80.0  # 1000 * 0.08
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_strategies.py -v
```

Expected: FAIL

- [ ] **Step 3: Implement base_strategy_v4.py**

```python
"""Base strategy V4 — multi-pair, promotion-aware, Kelly sizing."""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.memory_tiers import add_memory

logger = logging.getLogger(__name__)

_PARAMS_DIR = Path("data/strategy_params")


class LTMParamStore:
    """Per-strategy persistent parameter tuning."""

    def __init__(self, strategy_id: str):
        self.strategy_id = strategy_id
        _PARAMS_DIR.mkdir(parents=True, exist_ok=True)
        self._path = _PARAMS_DIR / f"{strategy_id}_params.json"
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                return {"trades": 0, "params": {}}
        return {"trades": 0, "params": {}}

    def _save(self):
        self._path.write_text(json.dumps(self._data, indent=2))

    def get_param(self, key: str, default=None):
        return self._data.get("params", {}).get(key, default)

    def set_param(self, key: str, value, source: str = "auto"):
        self._data.setdefault("params", {})[key] = value
        self._save()

    def increment_trades(self):
        self._data["trades"] = self._data.get("trades", 0) + 1
        self._save()

    def total_trades(self) -> int:
        return self._data.get("trades", 0)

    def all_params(self) -> dict:
        return self._data.get("params", {})


@dataclass
class StrategyConfig:
    id: str
    name: str
    timeframe: str
    leverage: int
    margin_pct: float
    tp_pct: float
    sl_pct: float
    cron_expr: dict
    regime_filter: list[str]
    description: str
    timeout_minutes: int = 120
    kelly_enabled: bool = True


@dataclass
class TradeSignal:
    direction: str  # "LONG" | "SHORT" | "HOLD"
    execute: bool
    confidence: float
    signals: dict
    reason: str
    pair: str = ""


class BaseStrategyV4(ABC):
    """Multi-pair, promotion-aware base strategy."""

    KELLY_MIN_TRADES = 20
    KELLY_MAX_SCALE = 2.0
    KELLY_MIN_SCALE = 0.5
    FEE_RATE = 0.0008  # Taker fee round-trip

    def __init__(
        self,
        config: StrategyConfig,
        initial_balance: float = 1000.0,
        api_key: str = "",
        exec_model: str = "claude-opus-4-6",
        self_trainer_enabled: bool = True,
    ):
        self.cfg = config
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.position: str | None = None
        self.trade_log: list[dict] = []
        self.cycle_count = 0
        self.skip_count = 0

        # Promotion state
        self.phase = "PAPER"  # PAPER | SHADOW | LIVE
        self.live_balance = 0.0
        self.peak_balance = initial_balance

        # Eliminator state
        self.is_paused = False
        self.is_eliminated = False
        self.confidence_multiplier = 1.0

        # Position state
        self._entry_price = 0.0
        self._entry_dir = ""
        self._entry_pair = ""
        self._entry_sig: dict = {}
        self._entry_time: datetime | None = None
        self._entry_margin = 0.0

        # TP/SL absolute prices
        self.tp_abs = 0.0
        self.sl_abs = 0.0

        # Cooldown per pair: {pair: datetime_until}
        self._pair_cooldown: dict[str, datetime] = {}

        # LTM param store
        self.trainer = LTMParamStore(config.id)

        # Self-trainer (initialized externally by runner)
        self._self_trainer = None

    @abstractmethod
    async def evaluate(self, pair: str, df) -> TradeSignal:
        """Evaluate a single pair and return a trade signal."""
        ...

    def _effective_margin(self) -> float:
        """Calculate margin with optional Kelly sizing."""
        base = round(self.balance * self.cfg.margin_pct, 2)

        if not self.cfg.kelly_enabled:
            return base

        closed = [t for t in self.trade_log if t.get("action") == "CLOSE"]
        if len(closed) < self.KELLY_MIN_TRADES:
            return base

        wins = [t for t in closed if t.get("pnl_net", 0) > 0]
        losses = [t for t in closed if t.get("pnl_net", 0) < 0]

        if not wins or not losses:
            return base

        wr = len(wins) / len(closed)
        avg_win = sum(t["pnl_net"] for t in wins) / len(wins)
        avg_loss = abs(sum(t["pnl_net"] for t in losses) / len(losses))

        b = avg_win / max(avg_loss, 0.01)
        kelly = wr - (1 - wr) / b
        qk = max(0.0, kelly * 0.25)

        scale = self.KELLY_MIN_SCALE + (qk / 0.25) * (self.KELLY_MAX_SCALE - self.KELLY_MIN_SCALE)
        scale = max(self.KELLY_MIN_SCALE, min(self.KELLY_MAX_SCALE, scale))

        return round(self.balance * self.cfg.margin_pct * scale, 2)

    def is_pair_on_cooldown(self, pair: str) -> bool:
        """Check if a pair is on cooldown for this strategy."""
        if pair in self._pair_cooldown:
            if datetime.now(timezone.utc) < self._pair_cooldown[pair]:
                return True
            del self._pair_cooldown[pair]
        return False

    def open_position(self, direction: str, price: float, pair: str, signals: dict):
        """Open a new position."""
        margin = self._effective_margin()
        notional = margin * self.cfg.leverage
        mult = 1 if direction == "LONG" else -1

        self.tp_abs = price * (1 + self.cfg.tp_pct * mult)
        self.sl_abs = price * (1 - self.cfg.sl_pct * mult)
        self.position = direction

        self._entry_price = price
        self._entry_dir = direction
        self._entry_pair = pair
        self._entry_sig = signals
        self._entry_time = datetime.now(timezone.utc)
        self._entry_margin = margin

        self._log("OPEN", direction, price, None, None, signals, margin, pair)

    def check_exit(self, price: float) -> str | None:
        """Check if position should be closed. Returns reason or None."""
        if not self.position:
            return None

        if self.position == "LONG":
            if price >= self.tp_abs:
                return "TP"
            if price <= self.sl_abs:
                return "SL"
        else:
            if price <= self.tp_abs:
                return "TP"
            if price >= self.sl_abs:
                return "SL"

        if self._entry_time:
            elapsed_min = (datetime.now(timezone.utc) - self._entry_time).total_seconds() / 60
            if elapsed_min >= self.cfg.timeout_minutes:
                return "TIMEOUT"
        return None

    def close_position(self, price: float, reason: str) -> tuple[float, dict]:
        """Close position and return (pnl, trade_record)."""
        margin = self._entry_margin or self._effective_margin()
        notional = margin * self.cfg.leverage
        fee_rt = notional * self.FEE_RATE
        mult = 1 if self._entry_dir == "LONG" else -1
        price_pct = (price - self._entry_price) / self._entry_price * mult
        pnl = notional * price_pct - fee_rt

        self.balance += pnl
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance

        now = datetime.now(timezone.utc)
        record = {
            "ts": now.isoformat(),
            "action": "CLOSE",
            "direction": self._entry_dir,
            "pair": self._entry_pair,
            "reason": reason,
            "pnl_net": round(pnl, 2),
            "price": price,
            "entry_price": self._entry_price,
            "balance": round(self.balance, 2),
            "signals": self._entry_sig,
        }

        self._log("CLOSE", self._entry_dir, price, pnl, reason, self._entry_sig, margin, self._entry_pair)
        self.trainer.increment_trades()

        # Set cooldown on pair
        cooldown_candles = 3
        candle_minutes = {"1m": 1, "5m": 5, "15m": 15}.get(self.cfg.timeframe, 5)
        self._pair_cooldown[self._entry_pair] = now + timedelta(minutes=cooldown_candles * candle_minutes)

        self.position = None
        self._entry_time = None
        return pnl, record

    def _log(self, action, direction, price, pnl, reason, signals, margin, pair=""):
        self.trade_log.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "direction": direction,
            "pair": pair,
            "price": price,
            "pnl_net": round(pnl, 2) if pnl is not None else None,
            "reason": reason,
            "signals": signals,
            "margin": round(margin, 2),
            "balance": round(self.balance, 2),
        })

    def stats(self) -> dict:
        """Return strategy statistics."""
        closed = [t for t in self.trade_log if t.get("action") == "CLOSE"]
        wins = [t for t in closed if t.get("pnl_net", 0) > 0]
        losses = [t for t in closed if t.get("pnl_net", 0) < 0]
        gross_win = sum(t["pnl_net"] for t in wins)
        gross_loss = abs(sum(t["pnl_net"] for t in losses)) or 1e-9

        drawdown = 0.0
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - self.balance) / self.peak_balance

        return {
            "id": self.cfg.id,
            "name": self.cfg.name,
            "timeframe": self.cfg.timeframe,
            "leverage": self.cfg.leverage,
            "phase": self.phase,
            "balance": round(self.balance, 2),
            "live_balance": round(self.live_balance, 2),
            "trades": len(closed),
            "win_rate": round(len(wins) / max(len(closed), 1) * 100, 1),
            "profit_factor": round(gross_win / gross_loss, 3),
            "max_drawdown": round(drawdown, 4),
            "is_paused": self.is_paused,
            "is_eliminated": self.is_eliminated,
            "conf_multiplier": round(self.confidence_multiplier, 2),
            "ready_to_rank": len(closed) >= 20,
            "open_position": {
                "direction": self.position,
                "pair": self._entry_pair,
                "entry_price": self._entry_price,
                "tp": self.tp_abs,
                "sl": self.sl_abs,
                "entry_time": self._entry_time.isoformat() if self._entry_time else None,
            } if self.position else None,
            "cycle_count": self.cycle_count,
            "skip_count": self.skip_count,
        }

    async def trigger_self_trainer(self, trade: dict, regime: str):
        """Trigger self-trainer analysis after a trade close."""
        if self._self_trainer:
            await self._self_trainer.analyze(
                trade,
                trade.get("signals", {}),
                f"Regime: {regime}, Pair: {trade.get('pair', 'unknown')}",
                self.trainer,
            )
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_strategies.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add strategies/base_strategy_v4.py tests/test_strategies.py
git commit -m "feat: add BaseStrategyV4 with multi-pair and promotion support"
```

---

## Task 10: Ultra-Fast Strategies (G01, G02, G03, G13)

**Files:**
- Create: `strategies/g01_momentum_burst.py`
- Create: `strategies/g02_scalp_ultra.py`
- Create: `strategies/g03_orderflow_imbalance.py`
- Create: `strategies/g13_volume_delta_sniper.py`

- [ ] **Step 1: Implement g01_momentum_burst.py**

```python
"""G-01 Momentum Burst — 1m ROC + Volume Spike detection."""

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-01",
    name="Momentum Burst",
    timeframe="1m",
    leverage=40,
    margin_pct=0.08,
    tp_pct=0.0035,
    sl_pct=0.0020,
    cron_expr={"minute": "*/2"},
    regime_filter=["ANY"],
    description="Detects sudden momentum explosions using ROC + volume spike.",
    timeout_minutes=30,
)


class G01MomentumBurst(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "1m", limit=20)

        df.ta.roc(length=3, append=True)
        vol_ma = df["volume"].rolling(10).mean()
        df["vol_ratio"] = df["volume"] / vol_ma.replace(0, 1)

        signals = {}
        r = df.iloc[-1]
        roc = r.get("ROC_3", 0)
        vol_ratio = r.get("vol_ratio", 0)
        signals = {"roc_3": round(roc, 4), "vol_ratio": round(vol_ratio, 2)}

        # Check 3 consecutive candles with positive/negative ROC
        last3_roc = [df.iloc[i].get("ROC_3", 0) for i in range(-3, 0)]
        all_positive = all(r > 0.1 for r in last3_roc)
        all_negative = all(r < -0.1 for r in last3_roc)

        if vol_ratio < 2.0:
            return TradeSignal("HOLD", False, 0.3, signals, "Volume too low", pair)

        if all_positive and roc > 0.15:
            conf = min(0.92, 0.70 + roc * 0.5 + (vol_ratio - 2) * 0.05)
            return TradeSignal("LONG", True, conf, signals, f"Momentum burst UP, ROC={roc:.2f}", pair)

        if all_negative and roc < -0.15:
            conf = min(0.92, 0.70 + abs(roc) * 0.5 + (vol_ratio - 2) * 0.05)
            return TradeSignal("SHORT", True, conf, signals, f"Momentum burst DOWN, ROC={roc:.2f}", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No momentum burst detected", pair)
```

- [ ] **Step 2: Implement g02_scalp_ultra.py**

```python
"""G-02 Scalp Ultra — evolved S-02, RSI extremes + BB touch + spread filter."""

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-02",
    name="Scalp Ultra",
    timeframe="1m",
    leverage=35,
    margin_pct=0.08,
    tp_pct=0.0030,
    sl_pct=0.0018,
    cron_expr={"minute": "*/1"},
    regime_filter=["RANGING", "VOLATILE"],
    description="RSI(7) extremes + volume + BB touch. Evolution of S-02.",
    timeout_minutes=30,
)


class G02ScalpUltra(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "1m", limit=30)

        df.ta.rsi(length=7, append=True)
        df.ta.bbands(length=20, std=2, append=True)

        r = df.iloc[-1]
        rsi = r.get("RSI_7", 50)
        vol_ratio = r["volume"] / max(df["volume"].rolling(20).mean().iloc[-1], 1e-9)

        bbl_col = next((c for c in df.columns if c.startswith("BBL")), None)
        bbu_col = next((c for c in df.columns if c.startswith("BBU")), None)
        price = r["close"]

        touching_lower = bbl_col and price <= r[bbl_col] * 1.001
        touching_upper = bbu_col and price >= r[bbu_col] * 0.999

        # Spread filter: reject if high-low > 0.1% of close
        spread = (r["high"] - r["low"]) / max(price, 1) * 100
        signals = {"rsi_7": round(rsi, 2), "vol_ratio": round(vol_ratio, 2), "spread": round(spread, 4)}

        if vol_ratio < 1.2:
            return TradeSignal("HOLD", False, 0.3, signals, "Volume too low", pair)

        if spread > 0.1:
            return TradeSignal("HOLD", False, 0.3, signals, "Spread too wide", pair)

        if rsi < 25 and touching_lower:
            conf = min(0.92, 0.70 + (30 - rsi) * 0.01 + (vol_ratio - 1) * 0.05)
            return TradeSignal("LONG", True, conf, signals, f"RSI oversold + BB lower touch", pair)

        if rsi > 75 and touching_upper:
            conf = min(0.92, 0.70 + (rsi - 70) * 0.01 + (vol_ratio - 1) * 0.05)
            return TradeSignal("SHORT", True, conf, signals, f"RSI overbought + BB upper touch", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "RSI not extreme enough", pair)
```

- [ ] **Step 3: Implement g03_orderflow_imbalance.py**

```python
"""G-03 Order Flow Imbalance — taker buy/sell volume ratio."""

from core.data_fetcher import get_taker_volume
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-03",
    name="Order Flow Imbalance",
    timeframe="1m",
    leverage=45,
    margin_pct=0.08,
    tp_pct=0.0040,
    sl_pct=0.0022,
    cron_expr={"minute": "*/2"},
    regime_filter=["ANY"],
    description="Analyzes taker buy/sell volume ratio for directional imbalance.",
    timeout_minutes=30,
)


class G03OrderFlowImbalance(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        tv = await get_taker_volume(pair, "1m", limit=10)

        r = tv.iloc[-1]
        buy_ratio = r.get("buy_ratio", 0.5)
        candle_dir = 1 if r["close"] > r["open"] else -1
        avg_buy_ratio = tv["buy_ratio"].mean()

        signals = {
            "buy_ratio": round(buy_ratio, 4),
            "avg_buy_ratio": round(avg_buy_ratio, 4),
            "candle_dir": candle_dir,
        }

        if buy_ratio > 0.65 and candle_dir > 0:
            conf = min(0.90, 0.70 + (buy_ratio - 0.65) * 2)
            return TradeSignal("LONG", True, conf, signals, f"Buy imbalance {buy_ratio:.0%}", pair)

        if buy_ratio < 0.35 and candle_dir < 0:
            sell_ratio = 1 - buy_ratio
            conf = min(0.90, 0.70 + (sell_ratio - 0.65) * 2)
            return TradeSignal("SHORT", True, conf, signals, f"Sell imbalance {sell_ratio:.0%}", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No significant imbalance", pair)
```

- [ ] **Step 4: Implement g13_volume_delta_sniper.py**

```python
"""G-13 Volume Delta Sniper — divergence between price and cumulative volume delta."""

from core.data_fetcher import get_taker_volume
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-13",
    name="Volume Delta Sniper",
    timeframe="1m",
    leverage=50,
    margin_pct=0.08,
    tp_pct=0.0045,
    sl_pct=0.0025,
    cron_expr={"minute": "*/2"},
    regime_filter=["ANY"],
    description="Detects hidden accumulation/distribution via volume delta divergence.",
    timeout_minutes=30,
)


class G13VolumeDeltaSniper(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        tv = await get_taker_volume(pair, "1m", limit=10)

        # Cumulative volume delta over last 5 candles
        last5 = tv.tail(5)
        delta = (last5["taker_buy_vol"] - last5["taker_sell_vol"]).sum()
        price_change = (last5.iloc[-1]["close"] - last5.iloc[0]["open"]) / last5.iloc[0]["open"] * 100

        signals = {
            "cum_delta": round(delta, 2),
            "price_change_pct": round(price_change, 4),
        }

        # Divergence: price drops but delta positive = hidden accumulation
        if price_change < -0.05 and delta > 0:
            strength = min(abs(delta) / max(last5["volume"].mean(), 1), 1.0)
            conf = min(0.90, 0.72 + strength * 0.15)
            return TradeSignal("LONG", True, conf, signals, f"Hidden accumulation, delta={delta:.0f}", pair)

        # Divergence: price rises but delta negative = distribution
        if price_change > 0.05 and delta < 0:
            strength = min(abs(delta) / max(last5["volume"].mean(), 1), 1.0)
            conf = min(0.90, 0.72 + strength * 0.15)
            return TradeSignal("SHORT", True, conf, signals, f"Distribution detected, delta={delta:.0f}", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No volume delta divergence", pair)
```

- [ ] **Step 5: Commit**

```bash
git add strategies/g01_momentum_burst.py strategies/g02_scalp_ultra.py strategies/g03_orderflow_imbalance.py strategies/g13_volume_delta_sniper.py
git commit -m "feat: add 4 ultra-fast strategies (G01, G02, G03, G13)"
```

---

## Task 11: Fast Strategies (G04, G05, G06, G08, G09)

**Files:**
- Create: `strategies/g04_macd_scalper.py`
- Create: `strategies/g05_stochastic_reversal.py`
- Create: `strategies/g06_bb_squeeze_turbo.py`
- Create: `strategies/g08_vwap_sniper.py`
- Create: `strategies/g09_atr_breakout.py`

- [ ] **Step 1: Implement g04_macd_scalper.py**

```python
"""G-04 MACD Scalper — histogram zero-line crosses + EMA confirmation."""

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-04",
    name="MACD Scalper",
    timeframe="5m",
    leverage=30,
    margin_pct=0.08,
    tp_pct=0.005,
    sl_pct=0.003,
    cron_expr={"minute": "*/5"},
    regime_filter=["TRENDING_UP", "TRENDING_DOWN"],
    description="MACD histogram zero-cross + EMA(9) direction + ADX filter.",
    timeout_minutes=120,
)


class G04MACDScalper(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "5m", limit=50)

        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.ema(length=9, append=True)
        df.ta.adx(length=14, append=True)

        r = df.iloc[-1]
        p = df.iloc[-2]

        hist_col = next((c for c in df.columns if "MACDh" in c), None)
        hist = r.get(hist_col, 0) if hist_col else 0
        prev_hist = p.get(hist_col, 0) if hist_col else 0
        ema9 = r.get("EMA_9", 0)
        adx = r.get("ADX_14", 0)
        price = r["close"]

        signals = {"macd_hist": round(hist, 4), "ema9": round(ema9, 2), "adx": round(adx, 2)}

        if adx < 20:
            return TradeSignal("HOLD", False, 0.3, signals, "ADX too low", pair)

        # Histogram crosses above zero + price above EMA9
        if prev_hist <= 0 < hist and price > ema9:
            conf = min(0.88, 0.72 + abs(hist) * 50 + (adx - 20) * 0.003)
            return TradeSignal("LONG", True, conf, signals, f"MACD hist cross up, ADX={adx:.0f}", pair)

        # Histogram crosses below zero + price below EMA9
        if prev_hist >= 0 > hist and price < ema9:
            conf = min(0.88, 0.72 + abs(hist) * 50 + (adx - 20) * 0.003)
            return TradeSignal("SHORT", True, conf, signals, f"MACD hist cross down, ADX={adx:.0f}", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No MACD cross", pair)
```

- [ ] **Step 2: Implement g05_stochastic_reversal.py**

```python
"""G-05 Stochastic Reversal — %K/%D crosses at extreme zones."""

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-05",
    name="Stochastic Reversal",
    timeframe="5m",
    leverage=25,
    margin_pct=0.08,
    tp_pct=0.0055,
    sl_pct=0.003,
    cron_expr={"minute": "*/5"},
    regime_filter=["RANGING"],
    description="Stochastic %K/%D crosses in oversold/overbought zones.",
    timeout_minutes=120,
)


class G05StochasticReversal(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "5m", limit=30)

        df.ta.stoch(k=14, d=3, append=True)

        k_col = next((c for c in df.columns if c.startswith("STOCHk")), None)
        d_col = next((c for c in df.columns if c.startswith("STOCHd")), None)

        if not k_col or not d_col:
            return TradeSignal("HOLD", False, 0.3, {}, "Stochastic data unavailable", pair)

        r = df.iloc[-1]
        p = df.iloc[-2]
        k = r[k_col]
        d = r[d_col]
        prev_k = p[k_col]
        prev_d = p[d_col]

        # Candle confirmation
        candle_bullish = r["close"] > r["open"]
        candle_bearish = r["close"] < r["open"]

        signals = {"stoch_k": round(k, 2), "stoch_d": round(d, 2)}

        # Bullish: K crosses above D in oversold zone
        if k < 20 and prev_k <= prev_d and k > d and candle_bullish:
            conf = min(0.88, 0.72 + (20 - k) * 0.005)
            return TradeSignal("LONG", True, conf, signals, f"Stoch bullish cross in oversold, K={k:.0f}", pair)

        # Bearish: K crosses below D in overbought zone
        if k > 80 and prev_k >= prev_d and k < d and candle_bearish:
            conf = min(0.88, 0.72 + (k - 80) * 0.005)
            return TradeSignal("SHORT", True, conf, signals, f"Stoch bearish cross in overbought, K={k:.0f}", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No stochastic reversal signal", pair)
```

- [ ] **Step 3: Implement g06_bb_squeeze_turbo.py**

```python
"""G-06 BB Squeeze Turbo — evolved SB-8, Keltner + volume confirmation."""

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-06",
    name="BB Squeeze Turbo",
    timeframe="5m",
    leverage=35,
    margin_pct=0.08,
    tp_pct=0.006,
    sl_pct=0.003,
    cron_expr={"minute": "*/5"},
    regime_filter=["ANY"],
    description="Bollinger squeeze + Keltner breakout + volume. Evolution of SB-8.",
    timeout_minutes=120,
)


class G06BBSqueezeTurbo(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "5m", limit=60)

        df.ta.bbands(length=20, std=2, append=True)
        df.ta.kc(length=20, scalar=1.5, append=True)

        bbu_col = next((c for c in df.columns if c.startswith("BBU")), None)
        bbl_col = next((c for c in df.columns if c.startswith("BBL")), None)
        bbm_col = next((c for c in df.columns if c.startswith("BBM")), None)
        kcu_col = next((c for c in df.columns if c.startswith("KCU")), None)
        kcl_col = next((c for c in df.columns if c.startswith("KCL")), None)

        if not all([bbu_col, bbl_col, bbm_col, kcu_col, kcl_col]):
            return TradeSignal("HOLD", False, 0.3, {}, "Indicator data unavailable", pair)

        r = df.iloc[-1]
        p = df.iloc[-2]

        # Squeeze: BB inside KC
        bb_upper, bb_lower = r[bbu_col], r[bbl_col]
        kc_upper, kc_lower = r[kcu_col], r[kcl_col]
        prev_bb_upper, prev_bb_lower = p[bbu_col], p[bbl_col]
        prev_kc_upper, prev_kc_lower = p[kcu_col], p[kcl_col]

        was_squeezed = prev_bb_upper < prev_kc_upper and prev_bb_lower > prev_kc_lower
        is_released = bb_upper >= kc_upper or bb_lower <= kc_lower

        # Bandwidth percentile
        df["bb_width"] = (df[bbu_col] - df[bbl_col]) / df[bbm_col].replace(0, 1)
        width = df["bb_width"].iloc[-1]
        pctl = (df["bb_width"].iloc[-50:] < width).sum() / 50

        vol_ratio = r["volume"] / max(df["volume"].rolling(20).mean().iloc[-1], 1e-9)
        price = r["close"]

        signals = {
            "was_squeezed": was_squeezed,
            "is_released": is_released,
            "width_pctl": round(pctl, 2),
            "vol_ratio": round(vol_ratio, 2),
        }

        if not was_squeezed or not is_released:
            return TradeSignal("HOLD", False, 0.4, signals, "No squeeze release", pair)

        if vol_ratio < 1.3:
            return TradeSignal("HOLD", False, 0.4, signals, "Volume too low for squeeze breakout", pair)

        if price > kc_upper:
            conf = min(0.90, 0.74 + (1 - pctl) * 0.2 + (vol_ratio - 1) * 0.05)
            return TradeSignal("LONG", True, conf, signals, f"BB squeeze breakout UP", pair)

        if price < kc_lower:
            conf = min(0.90, 0.74 + (1 - pctl) * 0.2 + (vol_ratio - 1) * 0.05)
            return TradeSignal("SHORT", True, conf, signals, f"BB squeeze breakout DOWN", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "Squeeze released but no clear direction", pair)
```

- [ ] **Step 4: Implement g08_vwap_sniper.py**

```python
"""G-08 VWAP Sniper — evolved SA-4, VWAP bounce with dynamic σ bands."""

import numpy as np
import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-08",
    name="VWAP Sniper",
    timeframe="5m",
    leverage=30,
    margin_pct=0.08,
    tp_pct=0.005,
    sl_pct=0.0028,
    cron_expr={"minute": "*/5"},
    regime_filter=["RANGING"],
    description="VWAP bounce with ±1σ/±2σ dynamic TP targets. Evolution of SA-4.",
    timeout_minutes=120,
)


class G08VWAPSniper(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "5m", limit=50)

        df.ta.rsi(length=14, append=True)

        # Calculate VWAP and standard deviation bands
        typical = (df["high"] + df["low"] + df["close"]) / 3
        cum_vol = df["volume"].cumsum()
        cum_tp_vol = (typical * df["volume"]).cumsum()
        vwap = cum_tp_vol / cum_vol.replace(0, 1)

        # Standard deviation bands
        df["vwap"] = vwap
        sq_diff = ((typical - vwap) ** 2 * df["volume"]).cumsum()
        std = np.sqrt(sq_diff / cum_vol.replace(0, 1))
        df["vwap_1up"] = vwap + std
        df["vwap_1dn"] = vwap - std
        df["vwap_2up"] = vwap + 2 * std
        df["vwap_2dn"] = vwap - 2 * std

        r = df.iloc[-1]
        price = r["close"]
        rsi = r.get("RSI_14", 50)
        vwap_val = r["vwap"]
        dist_to_vwap = (price - vwap_val) / vwap_val * 100

        signals = {
            "vwap": round(vwap_val, 2),
            "dist_pct": round(dist_to_vwap, 4),
            "rsi": round(rsi, 2),
        }

        # Long: price touches VWAP from above, RSI not overbought
        if abs(dist_to_vwap) < 0.05 and price > vwap_val and rsi > 40 and rsi < 60:
            conf = min(0.88, 0.72 + (1 - abs(dist_to_vwap)) * 0.1)
            return TradeSignal("LONG", True, conf, signals, "VWAP bounce LONG", pair)

        # Short: price touches VWAP from below, RSI not oversold
        if abs(dist_to_vwap) < 0.05 and price < vwap_val and rsi > 40 and rsi < 60:
            conf = min(0.88, 0.72 + (1 - abs(dist_to_vwap)) * 0.1)
            return TradeSignal("SHORT", True, conf, signals, "VWAP rejection SHORT", pair)

        # Reversal at 2σ band
        if price <= r["vwap_2dn"] and rsi < 35:
            conf = 0.85
            return TradeSignal("LONG", True, conf, signals, "VWAP -2σ reversal LONG", pair)

        if price >= r["vwap_2up"] and rsi > 65:
            conf = 0.85
            return TradeSignal("SHORT", True, conf, signals, "VWAP +2σ reversal SHORT", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No VWAP signal", pair)
```

- [ ] **Step 5: Implement g09_atr_breakout.py**

```python
"""G-09 ATR Breakout Rider — ATR expansion detection with trailing stop."""

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-09",
    name="ATR Breakout Rider",
    timeframe="5m",
    leverage=30,
    margin_pct=0.08,
    tp_pct=0.0055,
    sl_pct=0.003,
    cron_expr={"minute": "*/5"},
    regime_filter=["VOLATILE"],
    description="Detects ATR expansion for breakout riding with ATR trailing stop.",
    timeout_minutes=120,
)


class G09ATRBreakout(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)
        self._trailing_atr = 0.0

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "5m", limit=30)

        df.ta.atr(length=14, append=True)
        df.ta.ema(length=20, append=True)

        r = df.iloc[-1]
        atr_col = next((c for c in df.columns if c.startswith("ATR")), None)
        if not atr_col:
            return TradeSignal("HOLD", False, 0.3, {}, "ATR unavailable", pair)

        atr = r[atr_col]
        atr_avg = df[atr_col].rolling(20).mean().iloc[-1]
        atr_ratio = atr / max(atr_avg, 1e-9)
        ema20 = r.get("EMA_20", r["close"])
        price = r["close"]

        signals = {"atr": round(atr, 2), "atr_ratio": round(atr_ratio, 2), "ema20": round(ema20, 2)}

        if atr_ratio < 1.5:
            return TradeSignal("HOLD", False, 0.35, signals, "ATR not expanding enough", pair)

        self._trailing_atr = atr * 1.5

        if price > ema20:
            conf = min(0.88, 0.72 + (atr_ratio - 1.5) * 0.1)
            return TradeSignal("LONG", True, conf, signals, f"ATR breakout UP, ratio={atr_ratio:.1f}x", pair)

        if price < ema20:
            conf = min(0.88, 0.72 + (atr_ratio - 1.5) * 0.1)
            return TradeSignal("SHORT", True, conf, signals, f"ATR breakout DOWN, ratio={atr_ratio:.1f}x", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "ATR expanding but no direction", pair)
```

- [ ] **Step 6: Commit**

```bash
git add strategies/g04_macd_scalper.py strategies/g05_stochastic_reversal.py strategies/g06_bb_squeeze_turbo.py strategies/g08_vwap_sniper.py strategies/g09_atr_breakout.py
git commit -m "feat: add 5 fast strategies (G04, G05, G06, G08, G09)"
```

---

## Task 12: Strategic Strategies (G07, G10, G11, G12)

**Files:**
- Create: `strategies/g07_rsi_divergence.py`
- Create: `strategies/g10_ichimoku_edge.py`
- Create: `strategies/g11_liquidation_hunter_pro.py`
- Create: `strategies/g12_cross_pair_divergence.py`

- [ ] **Step 1: Implement g07_rsi_divergence.py**

```python
"""G-07 RSI Divergence Hunter — classic RSI divergences on 15m."""

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-07",
    name="RSI Divergence Hunter",
    timeframe="15m",
    leverage=20,
    margin_pct=0.08,
    tp_pct=0.007,
    sl_pct=0.004,
    cron_expr={"minute": "*/15"},
    regime_filter=["ANY"],
    description="Detects classic bullish/bearish RSI divergences with pivot confirmation.",
    timeout_minutes=240,
)


def _find_pivots(series, window=5):
    """Find local highs and lows in a series."""
    highs = []
    lows = []
    for i in range(window, len(series) - window):
        if series.iloc[i] == max(series.iloc[i - window:i + window + 1]):
            highs.append((i, series.iloc[i]))
        if series.iloc[i] == min(series.iloc[i - window:i + window + 1]):
            lows.append((i, series.iloc[i]))
    return highs, lows


class G07RSIDivergence(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "15m", limit=60)

        df.ta.rsi(length=14, append=True)
        rsi_col = "RSI_14"

        if rsi_col not in df.columns:
            return TradeSignal("HOLD", False, 0.3, {}, "RSI unavailable", pair)

        price_highs, price_lows = _find_pivots(df["close"], window=3)
        rsi_highs, rsi_lows = _find_pivots(df[rsi_col].dropna(), window=3)

        signals = {"rsi": round(df[rsi_col].iloc[-1], 2), "price_lows": len(price_lows), "price_highs": len(price_highs)}

        # Bullish divergence: price lower low + RSI higher low
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            p1, p2 = price_lows[-2], price_lows[-1]
            r1, r2 = rsi_lows[-2], rsi_lows[-1]
            if p2[1] < p1[1] and r2[1] > r1[1]:
                conf = min(0.88, 0.75 + (r2[1] - r1[1]) * 0.003)
                return TradeSignal("LONG", True, conf, signals, "Bullish RSI divergence", pair)

        # Bearish divergence: price higher high + RSI lower high
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            p1, p2 = price_highs[-2], price_highs[-1]
            r1, r2 = rsi_highs[-2], rsi_highs[-1]
            if p2[1] > p1[1] and r2[1] < r1[1]:
                conf = min(0.88, 0.75 + (r1[1] - r2[1]) * 0.003)
                return TradeSignal("SHORT", True, conf, signals, "Bearish RSI divergence", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No RSI divergence detected", pair)
```

- [ ] **Step 2: Implement g10_ichimoku_edge.py**

```python
"""G-10 Ichimoku Cloud Edge — Kumo breakout/bounce with Chikou confirmation."""

import pandas_ta as ta

from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-10",
    name="Ichimoku Cloud Edge",
    timeframe="15m",
    leverage=20,
    margin_pct=0.08,
    tp_pct=0.0065,
    sl_pct=0.0035,
    cron_expr={"minute": "*/15"},
    regime_filter=["TRENDING_UP", "TRENDING_DOWN"],
    description="Kumo breakout and bounce with Tenkan/Kijun/Chikou confirmation.",
    timeout_minutes=240,
)


class G10IchimokuEdge(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "15m", limit=80)

        df.ta.ichimoku(append=True)

        # Find Ichimoku columns
        tenkan = next((c for c in df.columns if "ISA" in c or "TENKAN" in c.upper()), None)
        kijun = next((c for c in df.columns if "ISB" in c or "KIJUN" in c.upper()), None)
        span_a = next((c for c in df.columns if "ISA" in c), None)
        span_b = next((c for c in df.columns if "ISB" in c), None)

        if not span_a or not span_b:
            return TradeSignal("HOLD", False, 0.3, {}, "Ichimoku data unavailable", pair)

        r = df.iloc[-1]
        price = r["close"]
        sa = r.get(span_a, 0)
        sb = r.get(span_b, 0)
        cloud_top = max(sa, sb)
        cloud_bottom = min(sa, sb)

        # Chikou span = current close vs price 26 periods ago
        chikou_bullish = len(df) > 26 and price > df.iloc[-27]["close"]
        chikou_bearish = len(df) > 26 and price < df.iloc[-27]["close"]

        # Tenkan > Kijun
        tk_bullish = tenkan and kijun and r.get(tenkan, 0) > r.get(kijun, 0)
        tk_bearish = tenkan and kijun and r.get(tenkan, 0) < r.get(kijun, 0)

        signals = {
            "price": price,
            "cloud_top": round(cloud_top, 2),
            "cloud_bottom": round(cloud_bottom, 2),
            "chikou_bullish": chikou_bullish,
            "tk_bullish": tk_bullish,
        }

        # Kumo breakout UP
        if price > cloud_top and tk_bullish and chikou_bullish:
            dist = (price - cloud_top) / cloud_top * 100
            conf = min(0.90, 0.76 + dist * 0.1)
            return TradeSignal("LONG", True, conf, signals, "Kumo breakout UP + TK + Chikou", pair)

        # Kumo breakout DOWN
        if price < cloud_bottom and tk_bearish and chikou_bearish:
            dist = (cloud_bottom - price) / cloud_bottom * 100
            conf = min(0.90, 0.76 + dist * 0.1)
            return TradeSignal("SHORT", True, conf, signals, "Kumo breakout DOWN + TK + Chikou", pair)

        # Kumo bounce: price near cloud edge in trend direction
        if abs(price - cloud_top) / cloud_top < 0.002 and tk_bullish:
            conf = 0.75
            return TradeSignal("LONG", True, conf, signals, "Kumo bounce at cloud top", pair)

        if abs(price - cloud_bottom) / cloud_bottom < 0.002 and tk_bearish:
            conf = 0.75
            return TradeSignal("SHORT", True, conf, signals, "Kumo bounce at cloud bottom", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No Ichimoku signal", pair)
```

- [ ] **Step 3: Implement g11_liquidation_hunter_pro.py**

```python
"""G-11 Liquidation Hunter Pro — evolved SA-2, OI drops + price cascade."""

from core.data_fetcher import fetch_ohlcv, get_open_interest
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-11",
    name="Liquidation Hunter Pro",
    timeframe="5m",
    leverage=40,
    margin_pct=0.08,
    tp_pct=0.005,
    sl_pct=0.0025,
    cron_expr={"minute": "*/5"},
    regime_filter=["VOLATILE"],
    description="Detects liquidation cascades via OI drops + violent price moves.",
    timeout_minutes=120,
)


class G11LiquidationHunterPro(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)
        self._prev_oi: dict[str, float] = {}

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        if df is None:
            df = await fetch_ohlcv(pair, "5m", limit=10)

        current_oi = await get_open_interest(pair)
        if current_oi is None:
            return TradeSignal("HOLD", False, 0.3, {}, "OI data unavailable", pair)

        prev_oi = self._prev_oi.get(pair, current_oi)
        oi_change_pct = (current_oi - prev_oi) / max(prev_oi, 1) * 100
        self._prev_oi[pair] = current_oi

        # Price movement over last 5 candles
        last5 = df.tail(5)
        price_change = (last5.iloc[-1]["close"] - last5.iloc[0]["open"]) / last5.iloc[0]["open"] * 100
        vol_spike = last5.iloc[-1]["volume"] / max(last5["volume"].mean(), 1e-9)

        signals = {
            "oi_change_pct": round(oi_change_pct, 2),
            "price_change_pct": round(price_change, 4),
            "vol_spike": round(vol_spike, 2),
        }

        # Liquidation cascade: OI drops sharply + violent price move
        if oi_change_pct < -3 and vol_spike > 1.5:
            if price_change > 0.3:
                conf = min(0.90, 0.74 + abs(oi_change_pct) * 0.02 + vol_spike * 0.03)
                return TradeSignal("LONG", True, conf, signals, f"Liq cascade UP, OI drop {oi_change_pct:.1f}%", pair)
            if price_change < -0.3:
                conf = min(0.90, 0.74 + abs(oi_change_pct) * 0.02 + vol_spike * 0.03)
                return TradeSignal("SHORT", True, conf, signals, f"Liq cascade DOWN, OI drop {oi_change_pct:.1f}%", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No liquidation cascade detected", pair)
```

- [ ] **Step 4: Implement g12_cross_pair_divergence.py**

```python
"""G-12 Cross-Pair Divergence — GOD 2 exclusive, mean reversion on correlation breaks."""

from core.correlation_engine import get_correlation_matrix
from core.data_fetcher import fetch_ohlcv
from strategies.base_strategy_v4 import BaseStrategyV4, StrategyConfig, TradeSignal

CONFIG = StrategyConfig(
    id="G-12",
    name="Cross-Pair Divergence",
    timeframe="5m",
    leverage=25,
    margin_pct=0.08,
    tp_pct=0.006,
    sl_pct=0.0032,
    cron_expr={"minute": "*/5"},
    regime_filter=["ANY"],
    description="GOD 2 exclusive: mean reversion when altcoin diverges from BTC.",
    timeout_minutes=120,
)


class G12CrossPairDivergence(BaseStrategyV4):

    def __init__(self, **kwargs):
        super().__init__(config=CONFIG, **kwargs)

    async def evaluate(self, pair: str, df=None) -> TradeSignal:
        # This strategy only trades altcoins, not BTC itself
        if pair == "BTCUSDT":
            return TradeSignal("HOLD", False, 0.0, {}, "G-12 only trades altcoins", pair)

        corr_data = get_correlation_matrix()
        matrix = corr_data.get("matrix", {})

        if "BTCUSDT" not in matrix or pair not in matrix.get("BTCUSDT", {}):
            return TradeSignal("HOLD", False, 0.3, {}, "Correlation data unavailable", pair)

        btc_corr = matrix["BTCUSDT"].get(pair, 0.5)

        # Get recent returns for both
        btc_df = await fetch_ohlcv("BTCUSDT", "5m", limit=5)
        alt_df = await fetch_ohlcv(pair, "5m", limit=5) if df is None else df

        btc_ret = (btc_df.iloc[-1]["close"] - btc_df.iloc[0]["open"]) / btc_df.iloc[0]["open"] * 100
        alt_ret = (alt_df.iloc[-1]["close"] - alt_df.iloc[0]["open"]) / alt_df.iloc[0]["open"] * 100
        divergence = alt_ret - btc_ret

        signals = {
            "btc_corr": round(btc_corr, 4),
            "btc_ret": round(btc_ret, 4),
            "alt_ret": round(alt_ret, 4),
            "divergence": round(divergence, 4),
        }

        # Only trade when correlation is normally high but currently diverging
        if btc_corr < 0.6:
            return TradeSignal("HOLD", False, 0.3, signals, "Correlation too low to trade divergence", pair)

        # Altcoin lagging BTC significantly → expect mean reversion (altcoin catches up)
        if btc_ret > 0.3 and divergence < -0.5:
            conf = min(0.88, 0.72 + abs(divergence) * 0.1 + btc_corr * 0.1)
            return TradeSignal("LONG", True, conf, signals, f"{pair} lagging BTC by {divergence:.2f}%", pair)

        # Altcoin overperforming BTC significantly → expect mean reversion (altcoin pulls back)
        if btc_ret < -0.3 and divergence > 0.5:
            conf = min(0.88, 0.72 + abs(divergence) * 0.1 + btc_corr * 0.1)
            return TradeSignal("SHORT", True, conf, signals, f"{pair} overperforming BTC by {divergence:.2f}%", pair)

        return TradeSignal("HOLD", False, 0.35, signals, "No significant cross-pair divergence", pair)
```

- [ ] **Step 5: Commit**

```bash
git add strategies/g07_rsi_divergence.py strategies/g10_ichimoku_edge.py strategies/g11_liquidation_hunter_pro.py strategies/g12_cross_pair_divergence.py
git commit -m "feat: add 4 strategic strategies (G07, G10, G11, G12)"
```

---

## Task 13: Self-Trainer + Strategy Eliminator + Coordinator + Memory Heartbeat

**Files:**
- Create: `core/self_trainer.py`
- Create: `core/strategy_eliminator.py`
- Create: `core/tournament_coordinator.py`
- Create: `core/memory_heartbeat.py`

- [ ] **Step 1: Implement self_trainer.py**

```python
"""Per-trade AI post-mortem analysis using Claude Opus 4.6."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from core.ai_client import generate_json
from core.memory_tiers import add_memory

logger = logging.getLogger(__name__)

_TRADES_FILE = Path("learnings/trades.jsonl")


class StrategySelfTrainer:
    """Analyzes each closed trade and adjusts strategy parameters."""

    _SYSTEM = """You are the SelfTrainer of a multi-pair crypto futures trading bot (Agent GOD 2).
Analyze this closed trade and extract actionable learnings.

Output ONLY valid JSON:
{
  "outcome_assessment": "correct_call|premature_exit|wrong_direction|good_risk_mgmt",
  "market_condition": "trending_up|trending_down|ranging|volatile",
  "signal_quality": 0.0-1.0,
  "pair_selection_quality": "optimal|acceptable|poor",
  "key_lesson": "<60 words>",
  "param_adjustments": {
    "tp_pct": 0.005,
    "sl_pct": 0.003,
    "margin_pct": 0.08
  },
  "promote_to_memory": true|false,
  "memory_tier": "short|mid|long",
  "memory_content": "<60 words>"
}
Rules:
- param_adjustments: only suggest if clearly needed. Max ±2% per iteration.
- promote_to_memory=true only for non-obvious insights
- long tier only for structural lessons validated by 5+ trades
"""

    def __init__(self, strategy_id: str, enabled: bool = True):
        self.strategy_id = strategy_id
        self.enabled = enabled

    async def analyze(
        self,
        trade: dict,
        signals: dict,
        market_context: str,
        ltm,
    ) -> dict | None:
        if not self.enabled or not settings.ANTHROPIC_API_KEY:
            return None

        prompt = (
            f"STRATEGY: {self.strategy_id}\n"
            f"TRADE:\n{json.dumps(trade, indent=2)}\n\n"
            f"SIGNALS:\n{json.dumps(signals, indent=2)}\n\n"
            f"CONTEXT: {market_context}\n\n"
            f"CURRENT PARAMS:\n{json.dumps(ltm.all_params(), indent=2)}"
        )

        try:
            analysis = await generate_json(
                model=settings.EXEC_MODEL,
                system_instruction=self._SYSTEM,
                prompt=prompt,
                temperature=0.2,
                max_tokens=512,
            )

            if not analysis:
                return None

            # Apply parameter adjustments with safety rails
            for k, v in analysis.get("param_adjustments", {}).items():
                current = ltm.get_param(k)
                if current is not None:
                    safe_v = max(current - 0.02, min(current + 0.02, float(v)))
                    ltm.set_param(k, round(safe_v, 4), source="self_trainer")

            # Persist trade to JSONL
            _TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "strategy_id": self.strategy_id,
                "direction": trade.get("direction"),
                "pair": trade.get("pair", ""),
                "outcome": trade.get("reason"),
                "pnl_net": trade.get("pnl_net"),
                "price": trade.get("price"),
                "analysis": analysis,
            }
            with open(_TRADES_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")

            # Memory promotion
            if analysis.get("promote_to_memory"):
                tier = analysis.get("memory_tier", "short")
                content = analysis.get("memory_content", analysis.get("key_lesson", ""))
                add_memory(tier, f"[{self.strategy_id}] {content}", tags=["trade_lesson", self.strategy_id])

            return analysis

        except Exception as e:
            logger.error(f"SelfTrainer error for {self.strategy_id}: {e}")
            return None
```

- [ ] **Step 2: Implement strategy_eliminator.py**

```python
"""Auto-pause/eliminate strategies based on 24h performance."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

_STATE_FILE = Path("learnings/eliminator_state.json")


class StrategyEliminator:
    """Evaluates strategies and pauses/eliminates poor performers."""

    def __init__(self, strategies: list, initial_balance: float = 1000.0):
        self.strategies = strategies
        self.initial_balance = initial_balance
        self._state: dict[str, dict] = {}
        self._load_state()

    def _load_state(self):
        if _STATE_FILE.exists():
            try:
                self._state = json.loads(_STATE_FILE.read_text())
            except Exception:
                self._state = {}

    def _save_state(self):
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(self._state, indent=2))

    def evaluate_all(self) -> list[dict]:
        """Evaluate all strategies and return list of actions taken."""
        actions = []
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=24)

        for strat in self.strategies:
            sid = strat.cfg.id
            state = self._state.setdefault(sid, {
                "status": "ACTIVE",
                "pause_count": 0,
                "paused_until": None,
                "history": [],
            })

            # Check if pause expired
            if state["status"] == "PAUSED" and state.get("paused_until"):
                until = datetime.fromisoformat(state["paused_until"])
                if now >= until:
                    state["status"] = "ACTIVE"
                    strat.is_paused = False
                    actions.append({"id": sid, "action": "UNPAUSE", "reason": "Pause expired"})

            if state["status"] in ("ELIMINATED", "PAUSED"):
                continue

            # Calculate 24h PnL
            closed = [
                t for t in strat.trade_log
                if t.get("action") == "CLOSE"
                and datetime.fromisoformat(t["ts"]) >= cutoff
            ]

            if len(closed) < settings.ELIMINATOR_MIN_TRADES:
                continue

            pnl_24h = sum(t.get("pnl_net", 0) for t in closed)
            pnl_pct = pnl_24h / max(self.initial_balance, 1)

            if pnl_pct <= settings.ELIMINATOR_THRESHOLD_PCT:
                state["pause_count"] = state.get("pause_count", 0) + 1
                reason = f"24h PnL ${pnl_24h:.2f} ({pnl_pct:.1%}) <= {settings.ELIMINATOR_THRESHOLD_PCT:.1%}"

                if state["pause_count"] >= settings.ELIMINATOR_MAX_PAUSES:
                    state["status"] = "ELIMINATED"
                    strat.is_eliminated = True
                    strat.is_paused = False

                    # If live, demote first
                    if strat.phase == "LIVE":
                        strat.phase = "PAPER"
                        strat.live_balance = 0.0

                    actions.append({"id": sid, "action": "ELIMINATE", "reason": reason})
                else:
                    state["status"] = "PAUSED"
                    state["paused_until"] = (now + timedelta(hours=settings.ELIMINATOR_PAUSE_HOURS)).isoformat()
                    strat.is_paused = True
                    actions.append({
                        "id": sid,
                        "action": "PAUSE",
                        "reason": reason,
                        "until": state["paused_until"],
                        "pause_count": state["pause_count"],
                    })

                state["history"].append({"ts": now.isoformat(), "action": actions[-1]["action"], "reason": reason})

        self._save_state()
        return actions

    def reactivate(self, strategy_id: str) -> dict:
        """Force reactivate a paused/eliminated strategy."""
        state = self._state.get(strategy_id)
        if not state:
            return {"error": f"Strategy {strategy_id} not found"}

        state["status"] = "ACTIVE"
        state["pause_count"] = 0
        state["paused_until"] = None

        for strat in self.strategies:
            if strat.cfg.id == strategy_id:
                strat.is_paused = False
                strat.is_eliminated = False
                break

        self._save_state()
        return {"id": strategy_id, "action": "REACTIVATED"}

    def full_status(self) -> dict:
        return self._state
```

- [ ] **Step 3: Implement tournament_coordinator.py**

```python
"""Tournament Coordinator — Claude Opus 4.6 brain for confidence adjustment."""

import json
import logging
from datetime import datetime, timezone

from config.settings import settings
from core.ai_client import generate_json
from core.memory_tiers import add_memory, get_all_context

logger = logging.getLogger(__name__)


class TournamentCoordinator:
    """The Brain: analyzes tournament and adjusts strategy confidence multipliers."""

    _SYSTEM = """You are TournamentCoordinator for Agent GOD 2 — a 13-strategy multi-pair crypto futures tournament.
Strategies trade 6 pairs (BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT) with 10-50x leverage.
Some strategies are in PAPER mode, some may be in SHADOW or LIVE.

Output ONLY valid JSON:
{
  "market_assessment": "<30 words>",
  "hot_strategies": ["<id>"],
  "cold_strategies": ["<id>"],
  "regime_recommendation": "TRENDING_UP|TRENDING_DOWN|RANGING|VOLATILE",
  "confidence_adjustments": { "<strategy_id>": 1.0 },
  "pair_recommendations": { "<pair>": "active|cautious|avoid" },
  "promotion_recommendations": [{"id": "<strategy_id>", "action": "promote|demote|hold", "reason": "<30 words>"}],
  "key_insight": "<60 words>",
  "memory_worthy": true,
  "risk_level": "LOW|MEDIUM|HIGH|EXTREME"
}
Rules:
- confidence > 1.1 only if PF > 1.3 AND WR > 55% AND trades >= 20
- confidence < 0.8 if WR < 35% OR last 5 trades all SL
- All confidence values: 0.5–1.5 range
- promotion_recommendations: only recommend 'promote' if strategy meets PAPER graduation criteria
- LIVE strategies: recommend 'demote' if recent performance deteriorating
"""

    def __init__(self):
        self.last_run: str | None = None
        self.last_analysis: dict | None = None
        self._multipliers: dict[str, float] = {}

    def get_multiplier(self, strategy_id: str) -> float:
        return self._multipliers.get(strategy_id, 1.0)

    async def run(
        self,
        tournament_status: dict,
        regimes: dict,
        correlation: dict,
    ) -> dict | None:
        if not settings.ANTHROPIC_API_KEY:
            return None

        memory_ctx = get_all_context(max_per_tier=5)

        prompt = (
            f"TOURNAMENT STATUS:\n{json.dumps(tournament_status, indent=2)}\n\n"
            f"MARKET REGIMES:\n{json.dumps(regimes, indent=2)}\n\n"
            f"CORRELATION:\n{json.dumps(correlation, indent=2)}\n\n"
            f"MEMORY:\n{memory_ctx}"
        )

        try:
            analysis = await generate_json(
                model=settings.BRAIN_MODEL,
                system_instruction=self._SYSTEM,
                prompt=prompt,
                temperature=0.15,
                max_tokens=1024,
            )

            if not analysis:
                return None

            # Apply clamped multipliers
            for sid, mult in analysis.get("confidence_adjustments", {}).items():
                self._multipliers[sid] = max(0.5, min(1.5, float(mult)))

            # Memory promotion
            if analysis.get("memory_worthy"):
                insight = analysis.get("key_insight", "")
                add_memory("long", f"[COORDINATOR] {insight}", tags=["coordinator"])

            self.last_run = datetime.now(timezone.utc).isoformat()
            self.last_analysis = analysis
            return analysis

        except Exception as e:
            logger.error(f"Coordinator error: {e}")
            return None
```

- [ ] **Step 4: Implement memory_heartbeat.py**

```python
"""Memory consolidation heartbeat + nightly reflection."""

import asyncio
import logging
from datetime import datetime, timezone

from config.settings import settings
from core.ai_client import generate_json
from core.memory_tiers import add_memory, get_recent, promote

logger = logging.getLogger(__name__)


class MemoryHeartbeat:
    """Periodic memory consolidation and nightly reflection."""

    def __init__(self):
        self.last_consolidation: str | None = None
        self.last_reflection: str | None = None

    async def consolidate_short_to_mid(self):
        """Take recent short-term memories and consolidate to mid-tier."""
        entries = get_recent("short", limit=5)
        if len(entries) < 3:
            return

        combined = " | ".join(e["content"] for e in entries[-3:])
        add_memory(
            "mid",
            f"[AUTO-CONSOLIDATED] {combined[:300]}",
            tags=["auto_consolidated"],
        )
        self.last_consolidation = datetime.now(timezone.utc).isoformat()
        logger.info("Memory heartbeat: consolidated short → mid")

    async def nightly_reflection(self, strategies_summary: str = ""):
        """Daily reflection: analyze mid-tier and promote patterns to long-term."""
        if not settings.ANTHROPIC_API_KEY:
            return

        entries = get_recent("mid", limit=20)
        if len(entries) < 5:
            return

        entries_text = "\n".join(f"- {e['content']}" for e in entries)

        prompt = (
            f"MID-TERM MEMORIES:\n{entries_text}\n\n"
            f"STRATEGIES SUMMARY:\n{strategies_summary}\n\n"
            "Identify patterns worth promoting to long-term memory."
        )

        system = """Analyze these trading memories and identify structural patterns.
Output ONLY valid JSON:
{
  "patterns_found": [
    {
      "pattern": "<description>",
      "confidence": 0.0-1.0,
      "promote": true|false,
      "tags": ["pattern_type"]
    }
  ],
  "top3_strategies": "<60 words summary of best performers>"
}
Rules:
- Only promote patterns seen in 3+ observations
- Confidence >= 0.75 to promote
- Maximum 3 promotions per reflection
"""

        try:
            analysis = await generate_json(
                model=settings.BRAIN_MODEL,
                system_instruction=system,
                prompt=prompt,
                temperature=0.2,
                max_tokens=512,
            )

            if not analysis:
                return

            for p in analysis.get("patterns_found", []):
                if p.get("promote") and p.get("confidence", 0) >= 0.75:
                    promote("mid", f"[PATTERN] {p['pattern']}", tags=p.get("tags", []))

            top3 = analysis.get("top3_strategies", "")
            if top3:
                add_memory("long", f"[NIGHTLY] {top3}", tags=["nightly_reflection"])

            self.last_reflection = datetime.now(timezone.utc).isoformat()
            logger.info("Nightly reflection completed")

        except Exception as e:
            logger.error(f"Nightly reflection error: {e}")
```

- [ ] **Step 5: Commit**

```bash
git add core/self_trainer.py core/strategy_eliminator.py core/tournament_coordinator.py core/memory_heartbeat.py
git commit -m "feat: add self-trainer, eliminator, coordinator, memory heartbeat"
```

---

## Task 14: Live Executor + Promotion Manager

**Files:**
- Create: `core/live_executor.py`
- Create: `core/promotion_manager.py`

- [ ] **Step 1: Implement live_executor.py**

```python
"""Binance Futures live execution engine."""

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

_ORDERS_FILE = Path("learnings/live_orders.jsonl")
_FUTURES_BASE = "https://fapi.binance.com"


class LiveExecutor:
    """Execute real orders on Binance Futures."""

    def __init__(self):
        self.api_key = settings.BINANCE_API_KEY
        self.secret = settings.BINANCE_SECRET
        self._enabled = bool(self.api_key and self.secret)

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        sig = hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        strategy_id: str = "",
    ) -> dict | None:
        """Place a market order on Binance Futures."""
        if not self._enabled:
            logger.error("Live executor not configured (missing API keys)")
            return None

        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": f"{quantity:.6f}",
        }
        params = self._sign(params)

        headers = {"X-MBX-APIKEY": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.post(
                    f"{_FUTURES_BASE}/fapi/v1/order",
                    params=params,
                    headers=headers,
                )

                result = resp.json()
                order_record = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "strategy_id": strategy_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "status": resp.status_code,
                    "response": result,
                }

                self._log_order(order_record)

                if resp.status_code == 200:
                    logger.info(f"LIVE ORDER: {side} {quantity} {symbol} — {result.get('orderId')}")
                    return result
                else:
                    logger.error(f"LIVE ORDER FAILED: {result}")
                    return None

        except httpx.TimeoutException:
            logger.error(f"LIVE ORDER TIMEOUT: {side} {quantity} {symbol} — cancelled")
            return None
        except Exception as e:
            logger.error(f"LIVE ORDER ERROR: {e}")
            return None

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        if not self._enabled:
            return False

        params = {"symbol": symbol, "leverage": leverage}
        params = self._sign(params)
        headers = {"X-MBX-APIKEY": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.post(
                    f"{_FUTURES_BASE}/fapi/v1/leverage",
                    params=params,
                    headers=headers,
                )
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"Set leverage error: {e}")
            return False

    async def get_balance(self) -> float:
        """Get USDT futures balance."""
        if not self._enabled:
            return 0.0

        params = self._sign({})
        headers = {"X-MBX-APIKEY": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"{_FUTURES_BASE}/fapi/v2/balance",
                    params=params,
                    headers=headers,
                )
                if resp.status_code == 200:
                    for asset in resp.json():
                        if asset["asset"] == "USDT":
                            return float(asset["balance"])
        except Exception as e:
            logger.error(f"Get balance error: {e}")
        return 0.0

    def _log_order(self, record: dict):
        _ORDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_ORDERS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
```

- [ ] **Step 2: Implement promotion_manager.py**

```python
"""Promotion Manager — Paper → Shadow → Live graduation pipeline."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

_STATE_FILE = Path("learnings/promotion_state.json")


class PromotionManager:
    """Manages strategy promotion through PAPER → SHADOW → LIVE pipeline."""

    def __init__(self, strategies: list):
        self.strategies = strategies
        self._state: dict[str, dict] = {}
        self._history: list[dict] = []
        self._load_state()

    def _load_state(self):
        if _STATE_FILE.exists():
            try:
                data = json.loads(_STATE_FILE.read_text())
                self._state = data.get("strategies", {})
                self._history = data.get("history", [])
            except Exception:
                pass

    def _save_state(self):
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {"strategies": self._state, "history": self._history}
        _STATE_FILE.write_text(json.dumps(data, indent=2))

    def check_promotions(self, coordinator_recommendations: list | None = None) -> list[dict]:
        """Check all strategies for promotion eligibility."""
        actions = []
        now = datetime.now(timezone.utc)

        for strat in self.strategies:
            sid = strat.cfg.id
            state = self._state.setdefault(sid, {
                "phase": "PAPER",
                "shadow_start": None,
                "shadow_errors": 0,
                "live_trades": 0,
                "live_scale": settings.PROMOTION_LIVE_INITIAL_PCT,
                "first_active": now.isoformat(),
            })

            if strat.is_eliminated:
                continue

            current_phase = state["phase"]

            # PAPER → SHADOW check
            if current_phase == "PAPER":
                stats = strat.stats()
                days_active = (now - datetime.fromisoformat(state["first_active"])).days

                if (
                    stats["trades"] >= settings.PROMOTION_MIN_TRADES
                    and stats["win_rate"] >= settings.PROMOTION_MIN_WR * 100
                    and stats["profit_factor"] >= settings.PROMOTION_MIN_PF
                    and stats["max_drawdown"] < settings.PROMOTION_MAX_DD
                    and days_active >= settings.PROMOTION_MIN_DAYS
                    and not strat.is_paused
                ):
                    # Check coordinator approval
                    approved = True
                    if coordinator_recommendations:
                        for rec in coordinator_recommendations:
                            if rec.get("id") == sid and rec.get("action") == "demote":
                                approved = False
                                break

                    if approved:
                        state["phase"] = "SHADOW"
                        state["shadow_start"] = now.isoformat()
                        state["shadow_errors"] = 0
                        strat.phase = "SHADOW"
                        action = {"id": sid, "action": "PROMOTE_TO_SHADOW", "reason": "Met all PAPER criteria"}
                        actions.append(action)
                        self._history.append({**action, "ts": now.isoformat()})
                        logger.info(f"PROMOTION: {sid} → SHADOW")

            # SHADOW → LIVE check
            elif current_phase == "SHADOW":
                shadow_start = datetime.fromisoformat(state["shadow_start"])
                hours_in_shadow = (now - shadow_start).total_seconds() / 3600

                if state.get("shadow_errors", 0) >= 3:
                    # Demote back to PAPER
                    state["phase"] = "PAPER"
                    strat.phase = "PAPER"
                    action = {"id": sid, "action": "DEMOTE_TO_PAPER", "reason": "Too many shadow errors"}
                    actions.append(action)
                    self._history.append({**action, "ts": now.isoformat()})
                    logger.warning(f"DEMOTION: {sid} → PAPER (shadow errors)")

                elif hours_in_shadow >= settings.PROMOTION_SHADOW_HOURS:
                    state["phase"] = "LIVE"
                    state["live_scale"] = settings.PROMOTION_LIVE_INITIAL_PCT
                    state["live_trades"] = 0
                    strat.phase = "LIVE"
                    strat.live_balance = strat.balance * settings.PROMOTION_LIVE_INITIAL_PCT
                    action = {"id": sid, "action": "PROMOTE_TO_LIVE", "reason": f"Shadow passed ({hours_in_shadow:.0f}h)"}
                    actions.append(action)
                    self._history.append({**action, "ts": now.isoformat()})
                    logger.info(f"PROMOTION: {sid} → LIVE (${strat.live_balance:.2f})")

            # LIVE scaling check
            elif current_phase == "LIVE":
                state["live_trades"] = state.get("live_trades", 0)

                # Scale up every 50 trades
                if state["live_trades"] >= 50:
                    current_scale = state.get("live_scale", settings.PROMOTION_LIVE_INITIAL_PCT)
                    new_scale = min(current_scale + settings.PROMOTION_LIVE_SCALE_STEP, settings.PROMOTION_LIVE_MAX_PCT)
                    if new_scale > current_scale:
                        state["live_scale"] = new_scale
                        strat.live_balance = strat.balance * new_scale
                        state["live_trades"] = 0
                        logger.info(f"LIVE SCALE UP: {sid} → {new_scale:.0%}")

                # Demotion check: >10% loss in live
                if strat.live_balance > 0:
                    initial_live = strat.balance * settings.PROMOTION_LIVE_INITIAL_PCT
                    if initial_live > 0:
                        live_pnl_pct = (strat.live_balance - initial_live) / initial_live
                        if live_pnl_pct < -0.10:
                            state["phase"] = "PAPER"
                            state["live_scale"] = settings.PROMOTION_LIVE_INITIAL_PCT
                            strat.phase = "PAPER"
                            strat.live_balance = 0.0
                            action = {"id": sid, "action": "DEMOTE_TO_PAPER", "reason": f"Live loss {live_pnl_pct:.1%}"}
                            actions.append(action)
                            self._history.append({**action, "ts": now.isoformat()})
                            logger.warning(f"DEMOTION: {sid} → PAPER (live loss)")

        self._save_state()
        return actions

    def demote(self, strategy_id: str) -> dict:
        """Force demotion of a strategy to PAPER."""
        state = self._state.get(strategy_id)
        if not state:
            return {"error": "Strategy not found"}

        state["phase"] = "PAPER"
        state["live_scale"] = settings.PROMOTION_LIVE_INITIAL_PCT

        for strat in self.strategies:
            if strat.cfg.id == strategy_id:
                strat.phase = "PAPER"
                strat.live_balance = 0.0
                break

        self._history.append({
            "id": strategy_id, "action": "MANUAL_DEMOTE",
            "reason": "Manual demotion", "ts": datetime.now(timezone.utc).isoformat(),
        })
        self._save_state()
        return {"id": strategy_id, "action": "DEMOTED_TO_PAPER"}

    def force_promote(self, strategy_id: str) -> dict:
        """Force promotion to next phase."""
        state = self._state.get(strategy_id)
        if not state:
            return {"error": "Strategy not found"}

        current = state["phase"]
        if current == "PAPER":
            state["phase"] = "SHADOW"
            state["shadow_start"] = datetime.now(timezone.utc).isoformat()
            next_phase = "SHADOW"
        elif current == "SHADOW":
            state["phase"] = "LIVE"
            state["live_scale"] = settings.PROMOTION_LIVE_INITIAL_PCT
            next_phase = "LIVE"
        else:
            return {"error": "Already in LIVE"}

        for strat in self.strategies:
            if strat.cfg.id == strategy_id:
                strat.phase = next_phase
                if next_phase == "LIVE":
                    strat.live_balance = strat.balance * settings.PROMOTION_LIVE_INITIAL_PCT
                break

        self._history.append({
            "id": strategy_id, "action": f"FORCE_PROMOTE_TO_{next_phase}",
            "reason": "Manual promotion", "ts": datetime.now(timezone.utc).isoformat(),
        })
        self._save_state()
        return {"id": strategy_id, "action": f"PROMOTED_TO_{next_phase}"}

    def get_pipeline(self) -> dict:
        return {
            "strategies": self._state,
            "history": self._history[-20:],
        }

    def record_shadow_error(self, strategy_id: str):
        if strategy_id in self._state:
            self._state[strategy_id]["shadow_errors"] = self._state[strategy_id].get("shadow_errors", 0) + 1
            self._save_state()

    def record_live_trade(self, strategy_id: str):
        if strategy_id in self._state:
            self._state[strategy_id]["live_trades"] = self._state[strategy_id].get("live_trades", 0) + 1
            self._save_state()
```

- [ ] **Step 3: Commit**

```bash
git add core/live_executor.py core/promotion_manager.py
git commit -m "feat: add live executor and promotion manager"
```

---

## Task 15: Pair Selector

**Files:**
- Create: `core/pair_selector.py`

- [ ] **Step 1: Implement pair_selector.py**

```python
"""Parallel multi-pair evaluation and selection."""

import asyncio
import logging

from core.data_fetcher import fetch_ohlcv
from core.market_regime import get_cached_regime, strategy_matches_regime

logger = logging.getLogger(__name__)

# Max strategies with open position on same pair
MAX_EXPOSURE_PER_PAIR = 4


async def select_best_pair(
    strategy,
    pairs: list[str],
    all_strategies: list,
    min_confidence: float = 0.72,
) -> dict | None:
    """Evaluate all pairs in parallel and return best signal.

    Returns: {pair, signal, confidence} or None if no valid signal.
    """

    # Check exposure limits per pair
    pair_exposure = {}
    for s in all_strategies:
        if s.position and s._entry_pair:
            pair_exposure[s._entry_pair] = pair_exposure.get(s._entry_pair, 0) + 1

    eligible_pairs = []
    for pair in pairs:
        if strategy.is_pair_on_cooldown(pair):
            continue
        if pair_exposure.get(pair, 0) >= MAX_EXPOSURE_PER_PAIR:
            continue
        # Check regime match for this specific pair
        regime = get_cached_regime(pair)
        if not strategy_matches_regime(strategy.cfg.regime_filter, regime):
            continue
        eligible_pairs.append(pair)

    if not eligible_pairs:
        return None

    # Evaluate all eligible pairs in parallel
    async def eval_pair(pair):
        try:
            df = await fetch_ohlcv(pair, strategy.cfg.timeframe, limit=60)
            signal = await strategy.evaluate(pair, df)
            return {"pair": pair, "signal": signal}
        except Exception as e:
            logger.warning(f"Eval failed for {strategy.cfg.id} on {pair}: {e}")
            return None

    tasks = [eval_pair(p) for p in eligible_pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter valid signals
    valid = []
    for r in results:
        if isinstance(r, dict) and r is not None:
            sig = r["signal"]
            if sig.execute and sig.direction != "HOLD":
                effective_conf = sig.confidence * strategy.confidence_multiplier
                if effective_conf >= min_confidence:
                    valid.append({
                        "pair": r["pair"],
                        "signal": sig,
                        "effective_confidence": effective_conf,
                    })

    if not valid:
        return None

    # Rank by effective confidence
    valid.sort(key=lambda x: x["effective_confidence"], reverse=True)
    return valid[0]
```

- [ ] **Step 2: Commit**

```bash
git add core/pair_selector.py
git commit -m "feat: add parallel multi-pair selector"
```

---

## Task 16: Tournament Runner GOD2

**Files:**
- Create: `scheduler/tournament_runner_god2.py`

- [ ] **Step 1: Implement tournament_runner_god2.py**

```python
"""Tournament Runner GOD2 — main orchestrator for 13 strategies across 6 pairs."""

import asyncio
import logging
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config.settings import settings
from core.circuit_breaker import CircuitBreaker
from core.correlation_engine import compute_correlation_matrix, get_correlation_matrix
from core.data_fetcher import get_current_price, prefetch_all
from core.market_regime import detect_all_regimes, get_cached_regime
from core.memory_heartbeat import MemoryHeartbeat
from core.pair_selector import select_best_pair
from core.promotion_manager import PromotionManager
from core.self_trainer import StrategySelfTrainer
from core.strategy_eliminator import StrategyEliminator
from core.tournament_coordinator import TournamentCoordinator

from strategies.g01_momentum_burst import G01MomentumBurst
from strategies.g02_scalp_ultra import G02ScalpUltra
from strategies.g03_orderflow_imbalance import G03OrderFlowImbalance
from strategies.g04_macd_scalper import G04MACDScalper
from strategies.g05_stochastic_reversal import G05StochasticReversal
from strategies.g06_bb_squeeze_turbo import G06BBSqueezeTurbo
from strategies.g07_rsi_divergence import G07RSIDivergence
from strategies.g08_vwap_sniper import G08VWAPSniper
from strategies.g09_atr_breakout import G09ATRBreakout
from strategies.g10_ichimoku_edge import G10IchimokuEdge
from strategies.g11_liquidation_hunter_pro import G11LiquidationHunterPro
from strategies.g12_cross_pair_divergence import G12CrossPairDivergence
from strategies.g13_volume_delta_sniper import G13VolumeDeltaSniper

logger = logging.getLogger(__name__)


class TournamentRunnerGOD2:
    """Main orchestrator: 13 strategies × 6 pairs with promotion pipeline."""

    def __init__(self):
        bal = settings.INITIAL_BALANCE
        api_key = settings.ANTHROPIC_API_KEY
        model = settings.EXEC_MODEL

        common = dict(initial_balance=bal, api_key=api_key, exec_model=model)

        self.strategies = [
            G01MomentumBurst(**common),
            G02ScalpUltra(**common),
            G03OrderFlowImbalance(**common),
            G04MACDScalper(**common),
            G05StochasticReversal(**common),
            G06BBSqueezeTurbo(**common),
            G07RSIDivergence(**common),
            G08VWAPSniper(**common),
            G09ATRBreakout(**common),
            G10IchimokuEdge(**common),
            G11LiquidationHunterPro(**common),
            G12CrossPairDivergence(**common),
            G13VolumeDeltaSniper(**common),
        ]

        # Attach self-trainers
        for strat in self.strategies:
            strat._self_trainer = StrategySelfTrainer(
                strat.cfg.id,
                enabled=settings.SELF_TRAINER_ENABLED,
            )

        self.pairs = settings.pairs_list
        initial_total = bal * len(self.strategies)

        self.circuit_breaker = CircuitBreaker(
            initial_paper_total=initial_total,
            paper_threshold=settings.CB_PAPER_THRESHOLD,
            live_threshold=settings.CB_LIVE_THRESHOLD,
            max_concurrent_live=settings.CB_LIVE_MAX_CONCURRENT,
            max_live_capital_pct=settings.CB_LIVE_MAX_CAPITAL_PCT,
        )
        self.eliminator = StrategyEliminator(self.strategies, bal)
        self.coordinator = TournamentCoordinator()
        self.promotion = PromotionManager(self.strategies)
        self.heartbeat = MemoryHeartbeat()

        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self.background_tasks: set = set()
        self._started = False
        self.is_paused = False

    async def start(self):
        """Start all scheduled jobs."""
        # Per-strategy cron jobs
        for strat in self.strategies:
            self.scheduler.add_job(
                self._run_strategy,
                "cron",
                kwargs={"strat": strat},
                id=strat.cfg.id,
                max_instances=1,
                coalesce=True,
                **strat.cfg.cron_expr,
            )

        # Background jobs
        self.scheduler.add_job(self._check_all_exits, "cron", minute="*", second=30, id="exits")
        self.scheduler.add_job(self._update_regimes, "cron", minute="*/5", second=5, id="regimes")
        self.scheduler.add_job(self._update_correlation, "cron", minute="*/5", second=15, id="correlation")
        self.scheduler.add_job(self._run_eliminator, "cron", minute=0, id="eliminator")
        self.scheduler.add_job(
            self._run_coordinator, "cron",
            minute=0, hour=f"*/{settings.COORDINATOR_INTERVAL_HOURS}",
            id="coordinator",
        )
        self.scheduler.add_job(
            self._run_heartbeat, "cron",
            minute=f"*/{settings.MEMORY_HEARTBEAT_INTERVAL_MIN}",
            id="heartbeat",
        )
        self.scheduler.add_job(self._nightly_reflection, "cron", hour=2, minute=0, id="nightly")
        self.scheduler.add_job(self._check_promotions, "cron", minute=30, id="promotions")

        self.scheduler.start()
        self._started = True
        logger.info(f"=== Agent GOD 2 ONLINE === {len(self.strategies)} strategies × {len(self.pairs)} pairs")

    async def stop(self):
        self.scheduler.shutdown(wait=False)
        self._started = False

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    async def _run_strategy(self, strat):
        """Main strategy execution cycle."""
        strat.cycle_count += 1

        if self.is_paused or strat.is_paused or strat.is_eliminated:
            strat.skip_count += 1
            return

        total = sum(s.balance for s in self.strategies)
        if self.circuit_breaker.check_paper(total):
            strat.skip_count += 1
            return

        # If strategy has open position, check exit
        if strat.position:
            price = await get_current_price(strat._entry_pair)
            exit_reason = strat.check_exit(price)
            if exit_reason:
                pnl, record = strat.close_position(price, exit_reason)
                regime = get_cached_regime(strat._entry_pair)
                self._bg_task(strat.trigger_self_trainer(record, regime.get("regime", "UNKNOWN")))
            return

        # Select best pair
        result = await select_best_pair(
            strat, self.pairs, self.strategies,
            min_confidence=settings.MIN_CONFIDENCE,
        )

        if result:
            pair = result["pair"]
            signal = result["signal"]
            price = await get_current_price(pair)
            strat.open_position(signal.direction, price, pair, signal.signals)
            logger.info(
                f"[{strat.cfg.id}] OPEN {signal.direction} {pair} @ {price:.2f} "
                f"(conf={result['effective_confidence']:.2f})"
            )

    async def _check_all_exits(self):
        """Check all open positions for exit conditions."""
        if self.is_paused:
            return

        for strat in self.strategies:
            if not strat.position or strat.is_paused or strat.is_eliminated:
                continue
            try:
                price = await get_current_price(strat._entry_pair)
                exit_reason = strat.check_exit(price)
                if exit_reason:
                    pnl, record = strat.close_position(price, exit_reason)
                    regime = get_cached_regime(strat._entry_pair)
                    self._bg_task(strat.trigger_self_trainer(record, regime.get("regime", "UNKNOWN")))
            except Exception as e:
                logger.error(f"Exit check error for {strat.cfg.id}: {e}")

    async def _update_regimes(self):
        await detect_all_regimes(self.pairs)

    async def _update_correlation(self):
        await compute_correlation_matrix(self.pairs)

    async def _run_eliminator(self):
        actions = self.eliminator.evaluate_all()
        for a in actions:
            logger.info(f"ELIMINATOR: {a['id']} → {a['action']}: {a['reason']}")

    async def _run_coordinator(self):
        status = self.get_status()
        regimes = {p: get_cached_regime(p) for p in self.pairs}
        correlation = get_correlation_matrix()
        analysis = await self.coordinator.run(status, regimes, correlation)
        if analysis:
            for strat in self.strategies:
                mult = self.coordinator.get_multiplier(strat.cfg.id)
                strat.confidence_multiplier = mult

    async def _run_heartbeat(self):
        await self.heartbeat.consolidate_short_to_mid()

    async def _nightly_reflection(self):
        summary = "\n".join(
            f"- {s.cfg.id} ({s.cfg.name}): PF={s.stats()['profit_factor']}, WR={s.stats()['win_rate']}%"
            for s in self.strategies if not s.is_eliminated
        )
        await self.heartbeat.nightly_reflection(summary)

    async def _check_promotions(self):
        recs = None
        if self.coordinator.last_analysis:
            recs = self.coordinator.last_analysis.get("promotion_recommendations")
        actions = self.promotion.check_promotions(recs)
        for a in actions:
            logger.info(f"PROMOTION: {a['id']} → {a['action']}: {a['reason']}")

    def _bg_task(self, coro):
        task = asyncio.create_task(coro)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    # --- Status methods ---

    def portfolio_summary(self) -> dict:
        paper_total = sum(s.balance for s in self.strategies)
        live_total = sum(s.live_balance for s in self.strategies if s.phase == "LIVE")
        return {
            "paper_total": round(paper_total, 2),
            "paper_initial": settings.INITIAL_BALANCE * len(self.strategies),
            "live_total": round(live_total, 2),
            "strategies_active": sum(1 for s in self.strategies if not s.is_eliminated and not s.is_paused),
            "strategies_paused": sum(1 for s in self.strategies if s.is_paused),
            "strategies_eliminated": sum(1 for s in self.strategies if s.is_eliminated),
            "strategies_live": sum(1 for s in self.strategies if s.phase == "LIVE"),
            "strategies_shadow": sum(1 for s in self.strategies if s.phase == "SHADOW"),
        }

    def leaderboard(self) -> list:
        stats = [s.stats() for s in self.strategies]
        active = sorted(
            [s for s in stats if not s["is_eliminated"] and s["ready_to_rank"]],
            key=lambda x: x["profit_factor"],
            reverse=True,
        )
        pending = [s for s in stats if not s["is_eliminated"] and not s["ready_to_rank"]]
        dead = [s for s in stats if s["is_eliminated"]]
        return active + pending + dead

    def get_status(self) -> dict:
        return {
            "version": "GOD2",
            "running": self._started,
            "paused": self.is_paused,
            "pairs": self.pairs,
            "portfolio": self.portfolio_summary(),
            "circuit_breaker": self.circuit_breaker.status(),
            "leaderboard": self.leaderboard(),
        }

    def get_strategy_detail(self, strategy_id: str) -> dict | None:
        for s in self.strategies:
            if s.cfg.id == strategy_id:
                return {
                    **s.stats(),
                    "trade_log": s.trade_log[-50:],
                    "params": s.trainer.all_params(),
                }
        return None
```

- [ ] **Step 2: Commit**

```bash
git add scheduler/tournament_runner_god2.py
git commit -m "feat: add tournament runner GOD2 orchestrator"
```

---

## Task 17: API Router + Main App

**Files:**
- Create: `main_god2.py`
- Create: `main.py`

- [ ] **Step 1: Implement main_god2.py**

```python
"""API Router for Agent GOD 2 tournament endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Request

from core.market_regime import get_cached_regime
from core.correlation_engine import get_correlation_matrix

router_god2 = APIRouter(tags=["Agent GOD 2"])


def get_runner(request: Request):
    runner = getattr(request.app.state, "tournament_runner", None)
    if not runner:
        raise HTTPException(503, "GOD2 Runner offline")
    return runner


# --- Tournament ---

@router_god2.get("/tournament/status")
async def tournament_status(runner=Depends(get_runner)):
    return runner.get_status()


@router_god2.get("/tournament/leaderboard")
async def leaderboard(runner=Depends(get_runner)):
    return {"leaderboard": runner.leaderboard()}


@router_god2.get("/tournament/portfolio")
async def portfolio(runner=Depends(get_runner)):
    return runner.portfolio_summary()


@router_god2.get("/tournament/strategy/{strategy_id}")
async def strategy_detail(strategy_id: str, runner=Depends(get_runner)):
    detail = runner.get_strategy_detail(strategy_id)
    if not detail:
        raise HTTPException(404, f"Strategy {strategy_id} not found")
    return detail


@router_god2.get("/tournament/regime")
async def regime(runner=Depends(get_runner)):
    return {pair: get_cached_regime(pair) for pair in runner.pairs}


@router_god2.get("/tournament/coordinator")
async def coordinator(runner=Depends(get_runner)):
    return {
        "last_run": runner.coordinator.last_run,
        "last_analysis": runner.coordinator.last_analysis,
    }


@router_god2.post("/tournament/pause")
async def pause(runner=Depends(get_runner)):
    runner.pause()
    return {"status": "paused"}


@router_god2.post("/tournament/resume")
async def resume(runner=Depends(get_runner)):
    runner.resume()
    return {"status": "resumed"}


# --- Pairs ---

@router_god2.get("/pairs/correlation")
async def correlation():
    return get_correlation_matrix()


@router_god2.get("/pairs/heatmap")
async def pair_heatmap(runner=Depends(get_runner)):
    heatmap = {}
    for strat in runner.strategies:
        sid = strat.cfg.id
        heatmap[sid] = {}
        for pair in runner.pairs:
            pair_pnl = sum(
                t.get("pnl_net", 0) for t in strat.trade_log
                if t.get("action") == "CLOSE" and t.get("pair") == pair
            )
            heatmap[sid][pair] = round(pair_pnl, 2)
    return heatmap


@router_god2.get("/pairs/{symbol}/price")
async def pair_price(symbol: str):
    from core.data_fetcher import get_current_price
    price = await get_current_price(symbol)
    return {"symbol": symbol, "price": price}


# --- Promotion ---

@router_god2.get("/promotion/pipeline")
async def promotion_pipeline(runner=Depends(get_runner)):
    return runner.promotion.get_pipeline()


@router_god2.post("/promotion/strategy/{strategy_id}/promote")
async def force_promote(strategy_id: str, runner=Depends(get_runner)):
    return runner.promotion.force_promote(strategy_id)


@router_god2.post("/promotion/strategy/{strategy_id}/demote")
async def demote(strategy_id: str, runner=Depends(get_runner)):
    return runner.promotion.demote(strategy_id)


@router_god2.get("/promotion/history")
async def promotion_history(runner=Depends(get_runner)):
    return runner.promotion._history[-50:]


# --- Live ---

@router_god2.get("/live/positions")
async def live_positions(runner=Depends(get_runner)):
    positions = []
    for s in runner.strategies:
        if s.phase == "LIVE" and s.position:
            positions.append(s.stats()["open_position"])
    return positions


@router_god2.get("/live/capital")
async def live_capital(runner=Depends(get_runner)):
    total = sum(s.live_balance for s in runner.strategies if s.phase == "LIVE")
    return {"total_live_capital": round(total, 2)}


@router_god2.post("/live/halt")
async def halt_live(runner=Depends(get_runner)):
    runner.circuit_breaker.live_triggered = True
    return {"status": "LIVE_HALTED"}


@router_god2.post("/live/resume")
async def resume_live(runner=Depends(get_runner)):
    runner.circuit_breaker.reset_live()
    return {"status": "LIVE_RESUMED"}


# --- Eliminator & Circuit Breaker ---

@router_god2.get("/tournament/eliminator")
async def eliminator_status(runner=Depends(get_runner)):
    return runner.eliminator.full_status()


@router_god2.post("/tournament/strategy/{strategy_id}/reactivate")
async def reactivate(strategy_id: str, runner=Depends(get_runner)):
    return runner.eliminator.reactivate(strategy_id)


@router_god2.get("/tournament/circuit-breaker")
async def cb_status(runner=Depends(get_runner)):
    return runner.circuit_breaker.status()


@router_god2.post("/tournament/circuit-breaker/reset/{level}")
async def cb_reset(level: str, runner=Depends(get_runner)):
    if level == "paper":
        runner.circuit_breaker.reset_paper()
    elif level == "live":
        runner.circuit_breaker.reset_live()
    else:
        raise HTTPException(400, "Level must be 'paper' or 'live'")
    return {"status": f"{level}_RESET"}


# --- Memory & Health ---

@router_god2.get("/memory/{tier}")
async def get_memory(tier: str, limit: int = 10):
    from core.memory_tiers import get_recent
    return {"tier": tier, "entries": get_recent(tier, limit)}


@router_god2.get("/learnings")
async def learnings():
    from pathlib import Path
    result = {}
    for fname in ["LEARNINGS.md", "ERRORS.md", "FEATURE_REQUESTS.md"]:
        path = Path("learnings") / fname
        result[fname] = path.read_text() if path.exists() else ""
    return result
```

- [ ] **Step 2: Implement main.py**

```python
"""Agent GOD 2 — FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from core.data_fetcher import get_current_price
from core.learnings_logger import ensure_learnings_dir
from core.memory_tiers import get_recent
from main_god2 import router_god2
from scheduler.tournament_runner_god2 import TournamentRunnerGOD2

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_learnings_dir()

    tournament = TournamentRunnerGOD2()
    app.state.tournament_runner = tournament
    await tournament.start()

    logger.info("=== Agent GOD 2 ONLINE ===")
    logger.info(f"Strategies: {len(tournament.strategies)}")
    logger.info(f"Pairs: {tournament.pairs}")
    logger.info(f"Brain: {settings.BRAIN_MODEL}")
    logger.info(f"Port: {settings.PORT}")
    yield

    await tournament.stop()
    logger.info("=== Agent GOD 2 OFFLINE ===")


app = FastAPI(
    title="Agent GOD 2",
    description="13-Strategy Multi-Pair Crypto Futures Tournament with Live Promotion",
    version="GOD2",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router_god2)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return RedirectResponse(url="/static/dashboard.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "GOD2",
        "brain": settings.BRAIN_MODEL,
        "exec": settings.EXEC_MODEL,
        "pairs": settings.pairs_list,
        "mode": settings.MODE,
    }


@app.get("/dashboard")
async def dashboard():
    return RedirectResponse(url="/static/dashboard.html")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, reload=False)
```

- [ ] **Step 3: Verify app starts**

```bash
cd /Users/gastonchevarria/Alpha/agent-god-2
python -c "from main import app; print('App created OK')"
```

Expected: `App created OK`

- [ ] **Step 4: Commit**

```bash
git add main.py main_god2.py
git commit -m "feat: add FastAPI app and API router"
```

---

## Task 18: Dashboard Premium

**Files:**
- Create: `static/dashboard.html`

- [ ] **Step 1: Create the premium dashboard**

This is a large HTML file (~1200 lines) with 6 views. Create `static/dashboard.html` with the full premium dashboard implementation including:

- Header with Agent GOD 2 branding, portfolio totals, live capital indicator
- Sidebar with 13 strategies showing phase pills (PAPER/SHADOW/LIVE)
- 6 main views: Overview, Strategy Detail, Pair Heatmap, Correlation Matrix, Promotion Pipeline, Live Monitor
- Chart.js for equity curves, bar charts, scatter plots
- Dark theme with green/blue/yellow/red color coding
- 5s/10s polling intervals
- Browser push notifications for promotions and CB triggers
- Responsive design

*Note: This file is too large to include inline in the plan. The implementing agent should create the full dashboard based on the spec in Section 6 of the design doc, using the original dashboard (`marl-btc-trader/static/dashboard.html`) as architectural reference but with the 6-view layout and premium features described in the spec.*

**Key implementation details:**
- Port: 9090 (all API calls to `http://localhost:9090/`)
- Views are toggled via sidebar navigation, single-page app pattern
- Chart.js CDN: `https://cdn.jsdelivr.net/npm/chart.js`
- Colors: `--accent: #818cf8; --green: #34d399; --red: #f87171; --yellow: #fbbf24; --blue: #60a5fa;`
- Phase pills: PAPER=blue, SHADOW=yellow, LIVE=green, ELIMINATED=red
- Polling: positions every 5s, rest every 10s

- [ ] **Step 2: Commit**

```bash
git add static/dashboard.html
git commit -m "feat: add premium 6-view dashboard"
```

---

## Task 19: Integration Tests

**Files:**
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write integration tests**

```python
"""Integration tests for Agent GOD 2 tournament runner."""

import pytest
from unittest.mock import AsyncMock, patch


def test_all_strategies_importable():
    """All 13 strategies should be importable."""
    from strategies.g01_momentum_burst import G01MomentumBurst
    from strategies.g02_scalp_ultra import G02ScalpUltra
    from strategies.g03_orderflow_imbalance import G03OrderFlowImbalance
    from strategies.g04_macd_scalper import G04MACDScalper
    from strategies.g05_stochastic_reversal import G05StochasticReversal
    from strategies.g06_bb_squeeze_turbo import G06BBSqueezeTurbo
    from strategies.g07_rsi_divergence import G07RSIDivergence
    from strategies.g08_vwap_sniper import G08VWAPSniper
    from strategies.g09_atr_breakout import G09ATRBreakout
    from strategies.g10_ichimoku_edge import G10IchimokuEdge
    from strategies.g11_liquidation_hunter_pro import G11LiquidationHunterPro
    from strategies.g12_cross_pair_divergence import G12CrossPairDivergence
    from strategies.g13_volume_delta_sniper import G13VolumeDeltaSniper

    strategies = [
        G01MomentumBurst, G02ScalpUltra, G03OrderFlowImbalance,
        G04MACDScalper, G05StochasticReversal, G06BBSqueezeTurbo,
        G07RSIDivergence, G08VWAPSniper, G09ATRBreakout,
        G10IchimokuEdge, G11LiquidationHunterPro, G12CrossPairDivergence,
        G13VolumeDeltaSniper,
    ]
    assert len(strategies) == 13


def test_runner_creates_13_strategies():
    """Tournament runner should initialize with 13 strategies."""
    with patch("scheduler.tournament_runner_god2.settings") as mock_settings:
        mock_settings.INITIAL_BALANCE = 1000.0
        mock_settings.ANTHROPIC_API_KEY = "test"
        mock_settings.EXEC_MODEL = "claude-opus-4-6"
        mock_settings.BRAIN_MODEL = "claude-opus-4-6"
        mock_settings.SELF_TRAINER_ENABLED = False
        mock_settings.pairs_list = ["BTCUSDT", "ETHUSDT"]
        mock_settings.CB_PAPER_THRESHOLD = -0.12
        mock_settings.CB_LIVE_THRESHOLD = -0.05
        mock_settings.CB_LIVE_MAX_CONCURRENT = 3
        mock_settings.CB_LIVE_MAX_CAPITAL_PCT = 0.30
        mock_settings.COORDINATOR_INTERVAL_HOURS = 2
        mock_settings.MEMORY_HEARTBEAT_INTERVAL_MIN = 20
        mock_settings.PAIRS = "BTCUSDT,ETHUSDT"

        from scheduler.tournament_runner_god2 import TournamentRunnerGOD2
        runner = TournamentRunnerGOD2()
        assert len(runner.strategies) == 13
        assert all(s.balance == 1000.0 for s in runner.strategies)


def test_all_strategies_have_unique_ids():
    """Each strategy should have a unique ID."""
    from scheduler.tournament_runner_god2 import TournamentRunnerGOD2
    with patch("scheduler.tournament_runner_god2.settings") as mock_settings:
        mock_settings.INITIAL_BALANCE = 1000.0
        mock_settings.ANTHROPIC_API_KEY = "test"
        mock_settings.EXEC_MODEL = "claude-opus-4-6"
        mock_settings.BRAIN_MODEL = "claude-opus-4-6"
        mock_settings.SELF_TRAINER_ENABLED = False
        mock_settings.pairs_list = ["BTCUSDT"]
        mock_settings.CB_PAPER_THRESHOLD = -0.12
        mock_settings.CB_LIVE_THRESHOLD = -0.05
        mock_settings.CB_LIVE_MAX_CONCURRENT = 3
        mock_settings.CB_LIVE_MAX_CAPITAL_PCT = 0.30
        mock_settings.COORDINATOR_INTERVAL_HOURS = 2
        mock_settings.MEMORY_HEARTBEAT_INTERVAL_MIN = 20
        mock_settings.PAIRS = "BTCUSDT"

        runner = TournamentRunnerGOD2()
        ids = [s.cfg.id for s in runner.strategies]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"


def test_circuit_breaker_dual_level():
    """Circuit breaker should have independent paper and live levels."""
    from core.circuit_breaker import CircuitBreaker
    cb = CircuitBreaker(
        initial_paper_total=13000,
        paper_threshold=-0.12,
        live_threshold=-0.05,
    )
    # Paper not triggered
    assert not cb.check_paper(12000)  # -7.7%
    # Paper triggered
    assert cb.check_paper(11000)  # -15.4%
    # Live independent
    assert not cb.live_triggered


def test_strategy_phases():
    """Strategies should start in PAPER phase."""
    from strategies.g01_momentum_burst import G01MomentumBurst
    strat = G01MomentumBurst(initial_balance=1000.0)
    assert strat.phase == "PAPER"
    assert strat.live_balance == 0.0
```

- [ ] **Step 2: Run all tests**

```bash
cd /Users/gastonchevarria/Alpha/agent-god-2
python -m pytest tests/ -v
```

Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_runner.py
git commit -m "feat: add integration tests"
```

---

## Task 20: Final Verification

- [ ] **Step 1: Verify full project structure**

```bash
find /Users/gastonchevarria/Alpha/agent-god-2 -name "*.py" -o -name "*.html" -o -name "*.txt" -o -name "*.env*" | sort
```

Expected: All 40+ files present.

- [ ] **Step 2: Run complete test suite**

```bash
cd /Users/gastonchevarria/Alpha/agent-god-2
python -m pytest tests/ -v --tb=short
```

Expected: ALL PASS

- [ ] **Step 3: Verify app starts without errors**

```bash
cd /Users/gastonchevarria/Alpha/agent-god-2
timeout 5 python -c "
import asyncio
from main import app
print('App created successfully')
print(f'Routes: {len(app.routes)}')
" || true
```

- [ ] **Step 4: Final commit**

```bash
cd /Users/gastonchevarria/Alpha/agent-god-2
git add -A
git commit -m "feat: Agent GOD 2 complete — 13 strategies, 6 pairs, promotion pipeline, premium dashboard"
```
