# MIDAS V8.1 — Dual-Agent Trading System

> **Adaptive Quantitative Trading with A/B Testing & Optional LLM Intelligence**

## Overview

Midas V8.1 is a paper-trading system running **2 independent agents** (LLM vs NoLLM) across **4 strategy profiles** each, totaling **8 virtual portfolios** ($15K each). It screens ~300 stocks across CAC40, European, Nasdaq, and S&P500 universes.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│               MIDAS V8.1 — Docker Stack             │
│                                                     │
│  ┌──────────────────┐   ┌──────────────────┐       │
│  │  midas-agent     │   │  midas-agent-nollm│       │
│  │  (LLM Agent)     │   │  (Pure Quant)     │       │
│  │  DISABLE_LLM=false│   │  DISABLE_LLM=true │       │
│  │  2-4 GB RAM      │   │  1-2 GB RAM       │       │
│  └────────┬─────────┘   └─────────┬────────┘       │
│           └──────────┬────────────┘                 │
│                ┌─────▼──────┐                       │
│                │  midas-api │                       │
│                │  (FastAPI)  │                       │
│                │  Port 8000  │                       │
│                └────────────┘                       │
│  Network: midas-net (bridge)                        │
└─────────────────────────────────────────────────────┘
```

### 3 Docker Containers

| Service | Role | RAM |
|---------|------|-----|
| `midas-agent` | LLM agent — Grok (X/Twitter) + Gemini reasoning overlay ±15pts | 2-4 GB |
| `midas-agent-nollm` | Pure quantitative agent — no LLM calls | 1-2 GB |
| `midas-api` | FastAPI dashboard serving both agents' data | 512 MB |

## Scoring System

**Two active pillars** with weighted combination:

| Pillar | Weight | Description |
|--------|--------|-------------|
| **Technical** | 55% | EMA alignment, MACD, ADX, RSI, Stochastic, Volume, Bollinger, ATR |
| **Fundamental** | 45% | P/E, P/B, P/S, PEG, Revenue/Earnings Growth, Margins, ROE, D/E |

- **Sentiment & News pillars**: code present but **disabled** (weight=0). The Intelligence Orchestrator handles news/sentiment globally for the LLM agent.
- **ML Pillar**: weight=0 but acts as a **confirmation gate** — can block BUY signals (score < 40) or boost them (+5 pts if score ≥ 60).
- Internal score scale: -100 to +100 → display scale: 0 to 100.

### Decision Thresholds (display 0–100)

| Decision | Score |
|----------|-------|
| STRONG_BUY | ≥ 70 |
| BUY | 55 – 69 |
| HOLD | 40 – 54 |
| SELL | 25 – 39 |
| STRONG_SELL | < 25 |

## Dual-Agent A/B Testing

### LLM Agent
- Grok Scanner: autonomous X/Twitter discovery + deep-dive + chain-of-thought
- Gemini Intelligence Orchestrator: market context reasoning → **±15 pts** score overlay
- Heat Detection from social + price + Grok sources

### NoLLM Agent
- Pure quantitative: same scoring engine, no LLM overlay
- Heat detection from price + social only (no Grok)
- Baseline for measuring LLM value-add

### 4 Strategy Profiles (per agent)

| Profile | Min Score | Max Positions | Position Size | ML Gate |
|---------|-----------|---------------|---------------|---------|
| 🔴 Aggressive + ML | 70 | 10 | 8% base | ON (min 40) |
| 🟠 Aggressive No ML | 70 | 10 | 8% base | OFF |
| 🟢 Moderate + ML | 78 | 6 | 5% base | ON (min 50) |
| 🔵 Moderate No ML | 78 | 6 | 5% base | OFF |

**2 agents × 4 profiles = 8 virtual portfolios** ($15,000 each).

## Position Sizing

Score-based dynamic sizing:

| Score | Multiplier | Aggressive | Moderate |
|-------|-----------|------------|----------|
| ≥ 90 | ×2.0 | 16% | 10% |
| ≥ 85 | ×1.6 | 12.8% | 8% |
| ≥ 80 | ×1.2 | 9.6% | 6% |
| < 80 | ×1.0 | 8% | 5% |

## Risk Management

- **Stop-Loss**: ATR-based (1.5× aggressive, 2.0× moderate), clamped 2–10%
- **Take-Profit**: ATR-based (3.0× aggressive, 4.0× moderate), clamped 5–30%
- **Trailing Stop**: activates at +5% gain, trails at 3–12% from peak
- **Max Hold**: 30 days → auto-exit
- **Circuit Breakers**: 3% daily loss / 15% max drawdown → defensive mode
- **Defensive Manager**: 4 levels (NONE → CAUTIOUS → DEFENSIVE → MAXIMUM)
- **Correlation Manager**: max 25% sector, max 15% single stock, avg correlation < 0.70

## Market Regime

Detected via SPY + VIX:

| Regime | Condition | Position Impact |
|--------|-----------|-----------------|
| BULL | SPY > EMA50+3%, trend > 3%, VIX < 20 | Max 10%, score ×1.05 |
| RANGE | Default | Max 8%, score ×1.00 |
| BEAR | SPY < EMA50-3%, trend < -3% | Max 5%, score ×0.90 |
| VOLATILE | VIX > 30 or vol > 35% | Max 4%, score ×0.80–0.60 |

## Market Hours (Paris Time)

| Session | Hours | Markets |
|---------|-------|---------|
| Europe | 08:00 – 15:30 | CAC40, European stocks |
| Overlap | 15:30 – 17:30 | EU + US |
| US | 17:30 – 22:00 | Nasdaq, S&P500 |

## Universe

~300 stocks across:
- **CAC40** (40 French blue chips)
- **Europe** (major EU exchanges: .PA, .DE, .AS, .MI, .MC, .L)
- **Nasdaq** (top tech/growth US stocks)
- **S&P500** (US large caps)

Ticker files in `config/`: `cac40.json`, `europe.json`, `nasdaq.json`, `sp500.json`.

## Pipeline (5-minute cycle)

1. **Regime Detection** — SPY/VIX analysis
2. **Session Check** — EU/Overlap/US filtering
3. **Guardrails** — daily P&L, drawdown, circuit breakers
4. **Attention Focus** — priority: Manual > Hot > Warming > Watchlist > Discovery
5. **Screening** — max 3 symbols/cycle via ReasoningEngine (4 pillars)
6. **Signal Processing** — Multi-Strategy evaluation, Sector-Regime adjustment, V8 Intel overlay (LLM only), Defensive/Correlation checks
7. **Execution** — Paper Trader with dynamic position sizing

## Key Directories

```
/opt/midas/
├── run_agent.py              # Entry point
├── docker-compose.prod.yml   # 3-service Docker stack
├── config/strategies.py      # 4 strategy profiles
├── src/
│   ├── agents/               # Core engine, pillars, reasoning
│   ├── intelligence/         # Grok, Gemini, Heat, Attention
│   ├── execution/            # Paper trader, risk, sizing, stops
│   ├── data/                 # Market data fetchers
│   └── api/                  # FastAPI dashboard
├── data/                     # LLM agent data
├── data-nollm/               # NoLLM agent data
├── models/                   # Shared ML models (read-only)
└── docs/                     # Documentation
```

## Documentation

- [Architecture Details](docs/ARCHITECTURE.md)
- [Scoring System](docs/SCORING.md)
- [Risk Management](docs/RISK.md)

---

*Midas V8.1 — Last updated: 2026-02-11*
