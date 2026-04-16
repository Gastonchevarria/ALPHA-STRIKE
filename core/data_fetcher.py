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

    df = df.iloc[:-1].reset_index(drop=True)
    _kline_cache[key] = (now, df)
    return df.copy()


async def fetch_ohlcv(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    limit: int = 100,
) -> pd.DataFrame:
    df = await fetch_klines(symbol, interval, limit)
    df = df.rename(columns={"vol": "volume"})
    return df[["open", "high", "low", "close", "volume"]].copy()


async def get_current_price(symbol: str = "BTCUSDT") -> float:
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
    if intervals is None:
        intervals = ["1m", "5m", "15m"]
    tasks = []
    for pair in pairs:
        for interval in intervals:
            limit = {"1m": 30, "5m": 60, "15m": 60}.get(interval, 60)
            tasks.append(fetch_klines(pair, interval, limit))
    await asyncio.gather(*tasks, return_exceptions=True)


async def get_taker_volume(symbol: str = "BTCUSDT", interval: str = "1m", limit: int = 10) -> pd.DataFrame:
    df = await fetch_klines(symbol, interval, limit)
    df["taker_buy_vol"] = df["tbbase"]
    df["taker_sell_vol"] = df["vol"] - df["tbbase"]
    df["buy_ratio"] = df["taker_buy_vol"] / df["vol"].replace(0, 1)
    return df


async def get_open_interest(symbol: str = "BTCUSDT") -> float | None:
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
