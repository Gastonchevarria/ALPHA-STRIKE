# Agent GOD 2 — Contexto del Proyecto

## 1. Vision General
**Agent GOD 2** es la segunda generacion del sistema de trading algoritmico autonomo de la familia GOD. Evoluciona el concepto original (`marl-btc-trader`) con tres avances fundamentales: **multi-par** (6 pares vs 1), **estrategias mas rapidas** (1m-15m vs 1h), y un **pipeline de promocion a Live** (Paper → Shadow → Live en Binance Futures con dinero real).

Es un proyecto **completamente independiente** del original. No comparte codigo, directorio, ni puerto. Puede correr en paralelo sin conflictos.

**URL de Produccion**: `https://agent-god-2-656958691766.us-central1.run.app`
**Dashboard**: `https://agent-god-2-656958691766.us-central1.run.app/`
**Proyecto GCP**: `proyecto001-490716`
**Region**: `us-central1`
**Puerto**: 9090

---

## 2. Diferencias vs el Original (marl-btc-trader)

| Aspecto | Original (V100) | Agent GOD 2 |
|:--------|:----------------|:------------|
| Pares | BTC/USDT unicamente | 6 pares: BTC, ETH, SOL, BNB, XRP, DOGE |
| Timeframes | Centrado en 1h | Mix agresivo: 1m, 5m, 15m |
| Leverage | 25-30x | 20-50x (segun estrategia) |
| Modelo AI | Opus (brain) + Sonnet (exec) | Opus 4.6 para TODO |
| Live trading | Solo paper | Pipeline Paper → Shadow → Live |
| Correlacion | No existe | Cross-pair correlation engine |
| Dashboard | 1 vista basica | 6 vistas premium |
| Puerto | 8080 | 9090 |

---

## 3. Arquitectura del Cerebro (Claude Opus 4.6)
A diferencia del original que usa Sonnet para ejecucion, GOD 2 usa **Claude Opus 4.6 para todo**. No hay Gemini fallback — si Anthropic esta caido, las estrategias siguen operando con senales tecnicas sin IA.

### Capas de Inteligencia:
1. **Tournament Coordinator ("El Cerebro" - Opus 4.6)**:
   - Se ejecuta cada **2 horas** (mas frecuente que el original, porque las estrategias son mas rapidas).
   - Analiza: estado del torneo + regimenes por par + correlaciones + fases de promocion.
   - Ajusta multiplicadores de confianza (0.5x a 1.5x) + recomienda promociones/demotions.
2. **Self-Trainer ("El Analista" - Opus 4.6)**:
   - Post-mortem por trade incluyendo: par elegido, por que no eligio otro, estado de correlacion.
   - Ajuste de parametros con safety rails (max ±2% por iteracion).
3. **Memory Heartbeat**:
   - Consolidacion cada **20 minutos** (vs 30min del original).
   - Reflexion nocturna a las 02:00 UTC con promocion de patrones a long-term.

---

## 4. Sistema de Torneo

### Parametros
- **Capital**: 13 estrategias × $1,000 = **$13,000** paper total.
- **Confidence Threshold**: 0.72 minimo (despues del multiplicador).
- **Pair Selector**: Cada estrategia evalua los 6 pares en paralelo (`asyncio.gather()`), elige el mejor.
- **Cooldown por par**: 3 velas despues de cerrar una posicion en un par.
- **Max exposure por par**: Maximo 4 estrategias abiertas en el mismo par simultaneamente.

### Gestion de Riesgo
- **Circuit Breaker DUAL**:
  - Paper: -12% del portfolio → halt 24h (permisivo, es simulacion).
  - Live: -5% del capital live en 24h → halt inmediato (agresivo, protege dinero real).
- **Strategy Eliminator**: -8% en 24h → pausa 12h. 3 pausas = eliminacion permanente.
- **Kelly Sizing**: Quarter-Kelly despues de 20 trades, scale 0.5x-2.0x.
- **Timeouts por bloque**: 30min (1m strats), 2h (5m strats), 4h (15m strats).

---

## 5. Pipeline de Promocion (Paper → Shadow → Live)
Este es el sistema mas critico de GOD 2 — la pipeline que gradua estrategias a dinero real.

### Fase 1: PAPER (todas empiezan aqui)
Criterios para avanzar a SHADOW:
- Minimo **100 trades** cerrados
- Win Rate **>= 55%**
- Profit Factor **>= 1.5**
- Drawdown maximo **< 15%** desde peak
- Activa por al menos **7 dias** sin ser eliminada
- El Coordinator debe **aprobar** con analisis

### Fase 2: SHADOW (48 horas)
- Ejecuta ordenes reales en Binance Testnet (0 capital en riesgo).
- Verifica: latencia, slippage, fills parciales, errores de API.
- Si hay mas de 3 errores → regresa a PAPER.

### Fase 3: LIVE (dinero real)
- **Capital inicial**: 5% del balance paper de la estrategia.
- **Scaling**: cada 50 trades exitosos, capital sube 5% (hasta 50% del balance paper).
- **Demotion automatica**: si cae >10% en live → regresa a PAPER.
- **Kill switch manual**: `POST /promotion/strategy/{id}/demote`
- **Max concurrent live**: 3 posiciones LIVE simultaneas.
- **Max capital live**: 30% del capital total disponible.

---

## 6. Las 13 Estrategias

### Bloque Ultra-Rapido (1m) — 4 estrategias
| ID | Nombre | Leverage | TP/SL | Cron | Logica |
|:---|:-------|:---------|:------|:-----|:-------|
| **G-01** | Momentum Burst | 40x | 0.35/0.20% | */2 min | ROC + volume spike, 3 velas consecutivas |
| **G-02** | Scalp Ultra* | 35x | 0.30/0.18% | */1 min | RSI(7) extremos + BB touch + spread filter |
| **G-03** | Order Flow Imbalance | 45x | 0.40/0.22% | */2 min | Taker buy/sell ratio >65% |
| **G-13** | Volume Delta Sniper | 50x | 0.45/0.25% | */2 min | Divergencia precio vs delta acumulado |

### Bloque Rapido (5m) — 5 estrategias
| ID | Nombre | Leverage | TP/SL | Cron | Logica |
|:---|:-------|:---------|:------|:-----|:-------|
| **G-04** | MACD Scalper | 30x | 0.50/0.30% | */5 min | Histograma cruza cero + EMA9 + ADX>20 |
| **G-05** | Stochastic Reversal | 25x | 0.55/0.30% | */5 min | %K cruza %D en zonas extremas (<20/>80) |
| **G-06** | BB Squeeze Turbo* | 35x | 0.60/0.30% | */5 min | Bollinger inside Keltner + volumen |
| **G-08** | VWAP Sniper* | 30x | 0.50/0.28% | */5 min | VWAP bounce con bandas ±1σ/±2σ |
| **G-09** | ATR Breakout Rider | 30x | 0.55/0.30% | */5 min | ATR expansion >1.5x + trailing stop ATR |

### Bloque Estrategico (5m-15m) — 4 estrategias
| ID | Nombre | Leverage | TP/SL | Cron | Logica |
|:---|:-------|:---------|:------|:-----|:-------|
| **G-07** | RSI Divergence Hunter | 20x | 0.70/0.40% | */15 min | Divergencias clasicas con 2+ pivots |
| **G-10** | Ichimoku Cloud Edge | 20x | 0.65/0.35% | */15 min | Kumo breakout + Tenkan/Kijun + Chikou |
| **G-11** | Liq Hunter Pro* | 40x | 0.50/0.25% | */5 min | OI drop >3% + precio violento |
| **G-12** | Cross-Pair Divergence | 25x | 0.60/0.32% | */5 min | **Exclusiva GOD 2**: mean reversion entre pares |

*\* = Evolucion de una estrategia del original*

---

## 7. Correlation Engine (exclusivo GOD 2)
Modulo nuevo que no existe en el original (`core/correlation_engine.py`):
- Calcula **matriz de correlacion rolling** (20 periodos) entre los 6 pares cada 5 minutos.
- Detecta **quiebres de correlacion**: cuando correlacion cae >2σ del promedio.
- **Divergence Index**: promedio general de correlacion. Si cae < 0.4 = mercado caotico.
- Alimenta a la estrategia exclusiva **G-12 Cross-Pair Divergence**.
- Datos disponibles para el Coordinator y todas las estrategias.

---

## 8. Dashboard Premium (6 Vistas)
Interfaz en `static/dashboard.html` (~1,500 lineas):

1. **Overview** (default): Equity curve global, leaderboard, posiciones abiertas con barras TP/SL, KPIs.
2. **Strategy Detail** (click en sidebar): Equity curve individual, trade log, donut por par, parametros, progreso de promocion.
3. **Pair Heatmap**: Grid 6 pares × 13 estrategias, color = PnL acumulado.
4. **Correlation Matrix**: Heatmap 6×6, alertas visuales en quiebres.
5. **Promotion Pipeline**: Kanban PAPER|SHADOW|LIVE, progress bars hacia criterios.
6. **Live Monitor**: Solo estrategias LIVE, P&L real, kill switch, circuit breaker proximity.

**Colores**: accent #818cf8, green #34d399, red #f87171, yellow #fbbf24, blue #60a5fa.
**Polling**: 5s posiciones, 10s resto. Dark theme #0f172a.

---

## 9. API Endpoints (puerto 9090)

### Tournament
| Metodo | Path | Proposito |
|:-------|:-----|:----------|
| GET | `/tournament/status` | Estado de las 13 estrategias |
| GET | `/tournament/leaderboard` | Ranking por PF, WR, fase |
| GET | `/tournament/portfolio` | Portfolio total (paper + live) |
| GET | `/tournament/strategy/{id}` | Detalle + trade log |
| GET | `/tournament/regime` | Regimen por par |
| GET | `/tournament/coordinator` | Ultimo analisis del brain |
| POST | `/tournament/pause` | Pausar todo |
| POST | `/tournament/resume` | Resumir todo |

### Pares y Correlacion
| Metodo | Path | Proposito |
|:-------|:-----|:----------|
| GET | `/pairs/correlation` | Matriz de correlacion |
| GET | `/pairs/heatmap` | PnL por estrategia×par |
| GET | `/pairs/{symbol}/price` | Precio actual |

### Promocion
| Metodo | Path | Proposito |
|:-------|:-----|:----------|
| GET | `/promotion/pipeline` | Estado de fases |
| POST | `/promotion/strategy/{id}/promote` | Forzar promocion |
| POST | `/promotion/strategy/{id}/demote` | Kill switch |
| GET | `/promotion/history` | Historial |

### Live
| Metodo | Path | Proposito |
|:-------|:-----|:----------|
| GET | `/live/positions` | Posiciones LIVE abiertas |
| GET | `/live/capital` | Capital live |
| POST | `/live/halt` | Halt all live |
| POST | `/live/resume` | Resumir live |

### Otros
| Metodo | Path | Proposito |
|:-------|:-----|:----------|
| GET | `/tournament/eliminator` | Estado eliminador |
| GET | `/tournament/circuit-breaker` | CB status (paper + live) |
| POST | `/tournament/circuit-breaker/reset/{level}` | Reset CB |
| GET | `/memory/{tier}` | Memorias |
| GET | `/health` | Health check |

---

## 10. Stack Tecnologico & Deployment
- **Backend**: FastAPI (Python 3.12)
- **Scheduling**: APScheduler (crons asincronicos por estrategia)
- **AI**: `anthropic>=0.39.0` (Claude Opus 4.6 exclusivamente, sin Gemini)
- **Data**: `pandas>=2.2.3`, `pandas_ta>=0.4.71b0`, `numpy>=2.0.2`
- **HTTP**: `httpx==0.28.1` (sin ccxt, conexion directa a Binance REST)
- **Deployment**: Google Cloud Run (Docker)
- **Min/Max Instances**: 1/1 (siempre encendido, sin duplicacion)
- **Memory**: 1 GB | **CPU**: 1

### Desplegar cambios:
```bash
cd /Users/gastonchevarria/Alpha/agent-god-2
gcloud run deploy agent-god-2 \
  --source . \
  --project proyecto001-490716 \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi --cpu 1 \
  --min-instances 1 --max-instances 1 \
  --port 9090
```
Nota: `ANTHROPIC_API_KEY` se pasa via `--set-env-vars` (no es secret en este proyecto).

---

## 11. Estructura de Archivos
```
agent-god-2/
├── config/settings.py                  # Pydantic config (35+ parametros)
├── core/
│   ├── ai_client.py                    # Claude Opus 4.6 only
│   ├── data_fetcher.py                 # Binance API con smart cache multi-par
│   ├── market_regime.py                # Regimen PER-PAIR (no global)
│   ├── pair_selector.py                # Evaluacion paralela 6 pares
│   ├── correlation_engine.py           # Matriz de correlacion cross-pair
│   ├── tournament_coordinator.py       # Brain Opus (cada 2h)
│   ├── self_trainer.py                 # Post-mortem Opus por trade
│   ├── strategy_eliminator.py          # -8% en 24h → pause 12h
│   ├── promotion_manager.py            # Paper → Shadow → Live
│   ├── live_executor.py                # Ejecucion real Binance Futures
│   ├── circuit_breaker.py              # DUAL: paper -12% + live -5%
│   ├── memory_tiers.py                 # Short/mid/long
│   ├── memory_heartbeat.py             # Cada 20min + nightly
│   ├── performance_tracker.py          # Stats globales
│   ├── risk_manager.py                 # Position dataclass
│   ├── partial_tp.py                   # TP parcial (disabled by default)
│   └── learnings_logger.py             # Markdown logging
├── strategies/
│   ├── base_strategy_v4.py             # Base: multi-par, promotion-aware, Kelly
│   ├── g01_momentum_burst.py           # 1m  40x  ROC + vol spike
│   ├── g02_scalp_ultra.py              # 1m  35x  RSI + BB (evol S-02)
│   ├── g03_orderflow_imbalance.py      # 1m  45x  Taker ratio
│   ├── g04_macd_scalper.py             # 5m  30x  MACD hist cross
│   ├── g05_stochastic_reversal.py      # 5m  25x  Stoch K/D extremos
│   ├── g06_bb_squeeze_turbo.py         # 5m  35x  BB+KC squeeze (evol SB-8)
│   ├── g07_rsi_divergence.py           # 15m 20x  RSI divergence 2+ pivots
│   ├── g08_vwap_sniper.py              # 5m  30x  VWAP ±σ bands (evol SA-4)
│   ├── g09_atr_breakout.py             # 5m  30x  ATR expansion + trailing
│   ├── g10_ichimoku_edge.py            # 15m 20x  Kumo break + Chikou
│   ├── g11_liquidation_hunter_pro.py   # 5m  40x  OI drop cascade (evol SA-2)
│   ├── g12_cross_pair_divergence.py    # 5m  25x  Mean reversion cross-pair
│   └── g13_volume_delta_sniper.py      # 1m  50x  Volume delta divergence
├── scheduler/
│   └── tournament_runner_god2.py       # Orquestador: 13 strats × 6 pares
├── static/dashboard.html               # Dashboard premium 6 vistas (~1,500 lineas)
├── main.py                             # FastAPI app (puerto 9090)
├── main_god2.py                        # Router con 25+ endpoints
├── tests/                              # 11 tests (pytest)
├── Dockerfile                          # Deploy Cloud Run
├── requirements.txt                    # 11 dependencias
└── .env.example                        # Template de configuracion
```

---

## 12. Historial de Cambios
| Fecha | Commit | Descripcion |
|:------|:-------|:-----------|
| 2026-04-17 | `c08fd2f` | Feat: Dockerfile + .dockerignore para Cloud Run |
| 2026-04-16 | `fcda9fb` | Feat: integration tests (11 tests) |
| 2026-04-16 | `452e676` | Feat: dashboard premium 6 vistas (1,529 lineas) |
| 2026-04-16 | `26597b4` | Feat: FastAPI app + API router (25+ endpoints) |
| 2026-04-16 | `53dd4ed` | Feat: tournament runner GOD2 |
| 2026-04-16 | `50347f8` | Feat: parallel multi-pair selector |
| 2026-04-16 | `3eee6ff` | Feat: live executor + promotion manager |
| 2026-04-16 | `571852a` | Feat: intelligence layer (self-trainer, eliminator, coordinator, heartbeat) |
| 2026-04-16 | `193131e` | Feat: BaseStrategyV4 + 13 estrategias |
| 2026-04-16 | `517ce55` | Feat: 15 core modules |
| 2026-04-16 | `f4a2e3a` | Feat: scaffold inicial |

---

## 13. Para el Proximo Dev
Si estas continuando este proyecto, aqui va lo esencial:

1. **No toques** las estrategias ganadoras sin razon. El self-trainer ajusta parametros automaticamente.
2. **Para agregar una estrategia nueva**: crea `strategies/g14_nombre.py` heredando de `BaseStrategyV4`, implementa `evaluate(pair, df) -> TradeSignal`, y agregala a la lista en `tournament_runner_god2.py`.
3. **Para desplegar**: `gcloud run deploy agent-god-2 --source . --project proyecto001-490716 --region us-central1`
4. **Para monitorear**: abre el dashboard en la URL de produccion. La vista Overview te da el estado general.
5. **Promotion pipeline**: las estrategias se auto-promueven cuando cumplen criterios. Si queres forzar: `POST /promotion/strategy/G-01/promote`.
6. **Si algo falla en live**: `POST /live/halt` detiene TODAS las posiciones live inmediatamente.
7. **Logs**: `gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=agent-god-2" --project proyecto001-490716 --limit 50`
8. **El spec y plan completos** estan en `docs/superpowers/specs/` y `docs/superpowers/plans/`.
