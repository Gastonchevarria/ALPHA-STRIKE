"""Historical OHLCV data loader for ML training."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

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


def load_cached(pair: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load cached parquet file, or None if missing."""
    path = _parquet_path(pair, timeframe)
    if not path.exists():
        return None
    return pd.read_parquet(path)


async def initial_backfill(pairs: list, timeframes: Optional[list] = None):
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
