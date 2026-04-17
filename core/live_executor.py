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
