"""Agent GOD 2 — Configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # AI
    GEMINI_API_KEY: str = ""
    BRAIN_MODEL: str = "gemini-3.1-pro-preview"
    EXEC_MODEL: str = "gemini-3-flash-preview"

    # Binance
    BINANCE_API_KEY: str = ""
    BINANCE_SECRET: str = ""
    BINANCE_TESTNET: bool = False

    # Security
    ADMIN_API_KEY: str = ""

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

    # ML Layer
    ML_ENABLED: bool = True
    ML_MIN_EV_USD: float = 0.50
    ML_REGIME_CONFIDENCE_THRESHOLD: float = 0.70
    ML_VOL_RATIO_MIN: float = 0.70
    ML_VOL_RATIO_MAX: float = 1.50
    ML_RETRAIN_DAY: int = 6  # Sunday (0=Mon, 6=Sun)
    ML_RETRAIN_HOUR_UTC: int = 3
    ML_HISTORICAL_DAYS: int = 90
    ML_GCS_BUCKET: str = "agent-god-2-data"
    ML_INFERENCE_CACHE_SECONDS: int = 30
    ML_MIN_TRADES_FOR_EV_MODEL: int = 50

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
