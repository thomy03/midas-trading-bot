# MIDAS V8.1 — Architecture

## Full Pipeline Overview

```
╔══════════════════════════════════════════════════════════════════════╗
║                    MIDAS V8.1 — FULL ARCHITECTURE                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  EXTERNAL SOURCES           PROCESSING                OUTPUT         ║
║                                                                      ║
║  ┌──────────┐                                                        ║
║  │ X/Twitter │──Grok──┐                                              ║
║  └──────────┘         │                                              ║
║  ┌──────────┐         │   ┌────────────────┐                         ║
║  │  Reddit  │─Social──┼──▶│ Heat Detector  │                         ║
║  │StockTwits│         │   │ (score 0→1)    │                         ║
║  └──────────┘         │   └───────┬────────┘                         ║
║  ┌──────────┐         │           │                                  ║
║  │  Price/  │─yfinance┘           ▼                                  ║
║  │  Volume  │          ┌──────────────────────┐                      ║
║  └──────────┘          │  Attention Manager   │                      ║
║                        │  (focus, cooldown,   │                      ║
║  ┌──────────┐          │   region filtering)  │                      ║
║  │  News    │─Fetch─┐  └──────────┬───────────┘                      ║
║  │Headlines │       │             │                                  ║
║  └──────────┘       │  ┌──────────▼───────────┐                      ║
║  ┌──────────┐       │  │  ReasoningEngine     │                      ║
║  │  Market  │─VIX───┤  │  Tech(55%)+Funda(45%)│                      ║
║  │  Context │ SPY   │  │  + ML Gate           │                      ║
║  └──────────┘       │  │  Score: 0→100        │                      ║
║  ┌──────────┐       │  └──────────┬───────────┘                      ║
║  │  Trends  │─Disc──┘             │                                  ║
║  └──────────┘        ┌────────────┼─────────────┐                    ║
║                      ▼            ▼             ▼                    ║
║              ┌────────────┐ ┌──────────┐ ┌──────────────┐            ║
║              │Sector-Regime│ │V8 Intel  │ │Multi-Strategy│            ║
║              │Score Adj   │ │Orchestr. │ │Tracker       │            ║
║              │(±12 pts)   │ │(±15 pts) │ │(4 profiles)  │            ║
║              └─────┬──────┘ │[LLM only]│ └──────┬───────┘            ║
║                    │        └────┬─────┘        │                    ║
║                    └─────────────┼──────────────┘                    ║
║                           ┌──────▼──────────┐                        ║
║                           │ Risk Management │                        ║
║                           │ Defensive Mode  │                        ║
║                           │ Correlation Chk │                        ║
║                           │ Circuit Breakers│                        ║
║                           └──────┬──────────┘                        ║
║                           ┌──────▼──────────┐                        ║
║                           │  Paper Trader   │→ 4 Virtual Portfolios  ║
║                           │  (execution)    │→ Telegram Notifs       ║
║                           └─────────────────┘→ Decision Journal      ║
╚══════════════════════════════════════════════════════════════════════╝
```

## Docker Services

| Service | Container | Command | Role |
|---------|-----------|---------|------|
| `agent` | `midas-agent` | `python run_agent.py --mode live --paper` | LLM Agent (Grok + Gemini + ML) |
| `agent-nollm` | `midas-agent-nollm` | `python run_agent.py --mode live --paper` | Pure quant agent |
| `api` | `midas-api` | `uvicorn src.api.main:app` | Dashboard API |

Data isolation:
- **LLM**: `./data/`, `./logs/`, `data/multi_strategy_llm.json`
- **NoLLM**: `./data-nollm/`, `./logs-nollm/`, `data/multi_strategy_nollm.json`
- **Models**: `./models/` (shared, read-only)

## Main Loop (5-minute cycle)

```
run_agent.py → LiveLoop.start()
  ├── _main_loop (5 min)     → Trading Pipeline
  ├── _heat_loop (5 min)     → Heat Data Collection
  └── _health_loop (10 min)  → Strategy Health Check
```

### Trading Pipeline (per cycle)

1. **Regime Detection** — `regime_adapter.detect_regime()` → BULL/BEAR/RANGE/VOLATILE
2. **Session Check** — `_get_market_session()` → PRE_MARKET/REGULAR/AFTER_HOURS/CLOSED
3. **Guardrails** — Daily P&L, drawdown, position count, circuit breakers
4. **Attention Focus** — `attention_manager.update_focus()` — priority allocation with 4h cooldown
5. **Screening** — `ReasoningEngine.analyze()` on max 3 symbols — 4 pillars in parallel
6. **Signal Processing** — Multi-strategy eval, sector-regime adjustment, V8 Intel overlay, defensive/correlation checks
7. **Execution** — `PaperTrader.open_position()` with dynamic sizing + portfolio rotation if full

### Heat Loop

Sources: Grok Scanner (LLM only), Social Scanner (Reddit/StockTwits), Price Data

```
Heat Score = mention_velocity(30%) + sentiment_shift(20%)
           + volume_anomaly(25%) + price_momentum(25%)

HOT (≥0.7)     → Priority 1 in Attention Manager
WARMING (≥0.4)  → Priority 2
COLD (<0.4)     → Ignored
```

Decay: 15-min half-life. Max 200 tracked, 20 HOT.

### Health Loop

`strategy_evolver.check_health()` every 10 min:
- **healthy** → log OK
- **warning** → notification + suggest tweaks
- **critical** → PAUSE trading + alert

## Module Map

```
src/
├── agents/
│   ├── live_loop.py              # Main event-driven loop
│   ├── reasoning_engine.py       # 4-pillar scoring engine
│   ├── multi_strategy_tracker.py # 4 virtual portfolios per agent
│   ├── guardrails.py             # Risk limits
│   ├── regime_adapter.py         # Market regime detection
│   ├── sector_regime_scorer.py   # Sector × regime adjustment
│   ├── adaptive_scorer.py        # Weight learning from past trades
│   ├── adaptive_ml_gate.py       # Volatility-aware ML gate
│   ├── decision_journal.py       # Decision logging
│   ├── nightly_auditor.py        # Daily trade audit
│   └── pillars/
│       ├── technical_pillar.py   # EMA, MACD, ADX, RSI, Volume, BB
│       ├── fundamental_pillar.py # P/E, Growth, Margins, Health
│       ├── sentiment_pillar.py   # Disabled (weight=0)
│       ├── news_pillar.py        # Disabled (weight=0)
│       └── ml_pillar.py          # RandomForest confirmation gate
├── intelligence/
│   ├── intelligence_orchestrator.py  # V8 Gemini reasoning (LLM only)
│   ├── grok_scanner.py              # X/Twitter discovery (LLM only)
│   ├── heat_detector.py             # Symbol heat scoring
│   ├── attention_manager.py         # Focus allocation
│   ├── gemini_client.py             # Gemini API client
│   └── news_fetcher.py              # News aggregation
├── execution/
│   ├── paper_trader.py              # Paper trading engine
│   ├── defensive_manager.py         # 4-level defensive mode
│   ├── correlation_manager.py       # Diversification enforcement
│   ├── position_manager.py          # Trailing stops, exits
│   ├── position_sizer.py            # Kelly + vol + confidence sizing
│   ├── dynamic_stops.py             # ATR-based SL/TP
│   ├── portfolio_rotation.py        # Smart rotation on full portfolio
│   └── ibkr_executor.py             # IBKR order execution
├── data/
│   ├── market_data.py               # yfinance fetcher
│   ├── polygon_client.py            # Polygon.io client
│   └── fmp_client.py                # FMP client
├── learning/
│   ├── knowledge_engine.py          # Trap detection, lessons
│   └── feedback_loop.py             # Performance feedback
└── api/
    └── main.py                      # FastAPI dashboard
```

## V8 Intelligence Orchestrator (LLM Agent Only)

```
Grok Scanner → Autonomous Discover → Deep Dive → Chain of Thought
     +
News Fetcher + Trend Discovery + Market Context (VIX, SPY)
     ↓
_collect_intelligence() → _build_reasoning_prompt()
     ↓
Gemini Flash LLM → IntelligenceBrief (cached 15 min)
  .events[], .megatrends[], .macro_regime_bias, .portfolio_alerts
     ↓
get_symbol_adjustment(sym) = Σ events + Σ megatrends → capped ±15 pts
```

Grok daily budget: 50 API calls. Intel brief cache: 15 minutes.

---

*See also: [Scoring](SCORING.md) | [Risk Management](RISK.md)*
