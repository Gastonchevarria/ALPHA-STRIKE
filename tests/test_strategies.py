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
