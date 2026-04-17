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
