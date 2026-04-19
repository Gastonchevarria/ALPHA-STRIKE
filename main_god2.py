"""API Router for Agent GOD 2 tournament endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Request, Header

from config.settings import settings
from core.market_regime import get_cached_regime
from core.correlation_engine import get_correlation_matrix

router_god2 = APIRouter(tags=["Agent GOD 2"])

async def verify_admin(x_api_key: str = Header(None)):
    if not settings.ADMIN_API_KEY:
        return  # Auth disable if no key is set yet
    if x_api_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized - Invalid API Key")


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


@router_god2.post("/tournament/pause", dependencies=[Depends(verify_admin)])
async def pause(runner=Depends(get_runner)):
    runner.pause()
    return {"status": "paused"}


@router_god2.post("/tournament/resume", dependencies=[Depends(verify_admin)])
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


@router_god2.post("/promotion/strategy/{strategy_id}/promote", dependencies=[Depends(verify_admin)])
async def force_promote(strategy_id: str, runner=Depends(get_runner)):
    return runner.promotion.force_promote(strategy_id)


@router_god2.post("/promotion/strategy/{strategy_id}/demote", dependencies=[Depends(verify_admin)])
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


@router_god2.post("/live/halt", dependencies=[Depends(verify_admin)])
async def halt_live(runner=Depends(get_runner)):
    runner.circuit_breaker.live_triggered = True
    return {"status": "LIVE_HALTED"}


@router_god2.post("/live/resume", dependencies=[Depends(verify_admin)])
async def resume_live(runner=Depends(get_runner)):
    runner.circuit_breaker.reset_live()
    return {"status": "LIVE_RESUMED"}


# --- Eliminator & Circuit Breaker ---

@router_god2.get("/tournament/eliminator")
async def eliminator_status(runner=Depends(get_runner)):
    return runner.eliminator.full_status()


@router_god2.post("/tournament/strategy/{strategy_id}/reactivate", dependencies=[Depends(verify_admin)])
async def reactivate(strategy_id: str, runner=Depends(get_runner)):
    return runner.eliminator.reactivate(strategy_id)


@router_god2.get("/tournament/circuit-breaker")
async def cb_status(runner=Depends(get_runner)):
    return runner.circuit_breaker.status()


@router_god2.post("/tournament/circuit-breaker/reset/{level}", dependencies=[Depends(verify_admin)])
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


# --- ML Endpoints ---

@router_god2.get("/ml/status")
async def ml_status():
    from ml.model_store import list_models
    from config.settings import settings
    return {
        "enabled": settings.ML_ENABLED,
        "models": list_models(),
        "cache_ttl_seconds": settings.ML_INFERENCE_CACHE_SECONDS,
    }


@router_god2.get("/ml/regime/{pair}")
async def ml_regime(pair: str):
    from ml.inference import get_regime_prediction
    return await get_regime_prediction(pair)


@router_god2.get("/ml/ev/{strategy_id}/{pair}")
async def ml_ev(strategy_id: str, pair: str):
    from ml.inference import get_expected_value
    return await get_expected_value(strategy_id, pair)


@router_god2.get("/ml/volatility/{pair}")
async def ml_volatility(pair: str):
    from ml.inference import get_volatility
    return await get_volatility(pair)


@router_god2.post("/ml/retrain", dependencies=[Depends(verify_admin)])
async def ml_retrain_trigger(runner=Depends(get_runner)):
    import asyncio
    asyncio.create_task(runner._run_ml_retrain())
    return {"status": "retrain_started"}


@router_god2.get("/ml/training-history")
async def ml_training_history():
    from core.memory_tiers import get_recent
    entries = get_recent("long", limit=50)
    return [e for e in entries if "ml_retrain" in e.get("tags", [])]

