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
