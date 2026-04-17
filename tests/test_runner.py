"""Integration tests for Agent GOD 2 tournament runner."""

import pytest
from unittest.mock import patch


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
    from scheduler.tournament_runner_god2 import TournamentRunnerGOD2
    runner = TournamentRunnerGOD2()
    assert len(runner.strategies) == 13
    assert all(s.balance == 1000.0 for s in runner.strategies)


def test_all_strategies_have_unique_ids():
    """Each strategy should have a unique ID."""
    from scheduler.tournament_runner_god2 import TournamentRunnerGOD2
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
    assert not cb.check_paper(12000)  # -7.7%
    assert cb.check_paper(11000)  # -15.4%
    assert not cb.live_triggered


def test_strategy_phases():
    """Strategies should start in PAPER phase."""
    from strategies.g01_momentum_burst import G01MomentumBurst
    strat = G01MomentumBurst(initial_balance=1000.0)
    assert strat.phase == "PAPER"
    assert strat.live_balance == 0.0


def test_memory_tiers():
    """Memory system should support add and retrieve."""
    import tempfile
    import os
    from unittest.mock import patch
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("core.memory_tiers._LEARNINGS_DIR", Path(tmpdir)):
            from core.memory_tiers import add_memory, get_recent
            add_memory("short", "test memory", tags=["test"])
            entries = get_recent("short", limit=5)
            assert len(entries) == 1
            assert entries[0]["content"] == "test memory"


def test_app_creates_successfully():
    """FastAPI app should create without errors."""
    from main import app
    assert app.title == "Agent GOD 2"
    assert len(app.routes) > 20
