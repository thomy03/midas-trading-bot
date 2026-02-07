# MIDAS Trading Bot V6.2

> **Adaptive Multi-Pillar Trading System with Machine Learning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MIDAS is an algorithmic trading system that combines **5 scoring pillars**, **market regime detection**, and **adaptive machine learning** to generate swing trading signals on US/EU equities.

## 🎯 Key Features

- **5-Pillar Scoring Engine**: Technical (25+ indicators), Fundamental, Sentiment (Grok/X), News, ML
- **Regime Detection**: Automatically adapts to BULL/BEAR/RANGE/VOLATILE markets
- **Adaptive ML Gate**: Volatility-based ML switching for optimal performance
- **40+ Technical Features**: Trend, momentum, volume, volatility indicators
- **Interactive Brokers Integration**: Paper and live trading support

## 📊 Backtest Results (10 years, 2015-2025)

| Metric | MIDAS ML-Enhanced | S&P 500 |
|--------|-------------------|---------|
| **CAGR** | 30.3% | 10-12% |
| **Sharpe Ratio** | 2.12 | 0.5-0.7 |
| **Max Drawdown** | -31.3% | -34% |
| **Win Rate** | 65.1% | - |
| **Trades** | 544 | - |

*Note: Backtests exclude transaction costs (~2-3% annual impact)*

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SCORING ENGINE                            │
├─────────────┬─────────────┬─────────────┬─────────────┬─────┤
│  TECHNICAL  │ FUNDAMENTAL │  SENTIMENT  │    NEWS     │ ML  │
│   22-30%    │   15-28%    │   12-22%    │   5-15%     │20-30│
├─────────────┴─────────────┴─────────────┴─────────────┴─────┤
│              REGIME-WEIGHTED AGGREGATION                     │
│         (BULL / BEAR / RANGE / VOLATILE)                     │
├─────────────────────────────────────────────────────────────┤
│                  ADAPTIVE ML GATE                            │
│    Vol > 3% → 5 Pillars only | Vol ≤ 3% → ML active         │
├─────────────────────────────────────────────────────────────┤
│                  FINAL SCORE (0-100)                         │
│                  Score ≥ 75 → BUY SIGNAL                     │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
src/
├── agents/
│   ├── pillars/           # 5 scoring pillars
│   │   ├── technical_pillar.py
│   │   ├── fundamental_pillar.py
│   │   ├── sentiment_pillar.py
│   │   ├── news_pillar.py
│   │   └── ml_pillar.py
│   ├── adaptive_ml_gate.py    # Volatility-based ML switching
│   ├── regime_adapter.py      # Market regime detection
│   ├── adaptive_scorer.py     # Score aggregation
│   └── live_loop.py           # Main trading loop
├── learning/
│   ├── knowledge_engine.py    # Learn from mistakes
│   ├── dynamic_influence_learner.py  # Discover influencers
│   └── smart_signal_learner.py       # Weak signal detection
├── brokers/
│   ├── ib_broker.py           # Interactive Brokers
│   └── paper_trader.py        # Paper trading
├── indicators/                # 30+ technical indicators
└── data/                      # Market data clients
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)
- API keys: Grok (xAI), Gemini, Alpha Vantage

### Installation

```bash
# Clone repository
git clone https://github.com/thomy03/midas-trading-bot.git
cd midas-trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Edit `.env` with your credentials:

```env
# Required
GROK_API_KEY=xai-your_key
GOOGLE_AI_API_KEY=your_gemini_key

# Optional
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Running

```bash
# Paper trading mode
python -m src.main --mode paper

# With Docker
docker-compose up -d
```

## 📈 Scoring Pillars

### Technical Pillar (22-30%)

Analyzes 25+ indicators across 4 categories:

| Category | Weight | Indicators |
|----------|--------|------------|
| Trend | 30% | EMA 20/50/200, MACD, ADX, Supertrend |
| Momentum | 25% | RSI, Stochastic, Williams %R, CCI, ROC |
| Volume | 25% | OBV, VWAP, Volume Ratio, CMF, MFI |
| Volatility | 20% | ATR, Bollinger Bands |

### Fundamental Pillar (15-28%)

- P/E ratio vs sector
- PEG ratio
- Debt/Equity
- Profit margins
- Revenue growth
- Free cash flow

### Sentiment Pillar (12-22%)

- X/Twitter analysis via Grok API
- StockTwits sentiment
- Dynamic influencer discovery (no hardcoded list)

### News Pillar (5-15%)

- Multi-source aggregation (Alpha Vantage, FMP, NewsAPI)
- Event detection (earnings, FDA approvals)
- LLM sentiment analysis

### ML Pillar (20-30%)

- 40 technical features
- Random Forest classifier
- Monthly retraining on trade history
- Market regime detection

## 🎛️ Regime Adaptation

The system detects market conditions and adapts:

| Regime | Detection | Adjustments |
|--------|-----------|-------------|
| **BULL** | SPY > EMA50 +3%, VIX < 20 | Momentum ↑, Small caps OK |
| **BEAR** | SPY < EMA50 -3% | Fundamentals ↑, Blue chips only |
| **RANGE** | Sideways | Balanced weights |
| **VOLATILE** | VIX > 30 | Mega caps only, tight stops |

## 🤖 Adaptive ML Gate (V6.2)

Volatility-based ML switching:

```
Volatility > 3%  →  5 Pillars only (ML disabled)
Volatility ≤ 3%  →  ML Gate active:
                    - ML confidence > 60% → BOOST (+5 pts)
                    - ML confidence < 40% → BLOCK (reject)
                    - Else → NEUTRAL (pass-through)
```

## 📊 API & Dashboard

```bash
# Start API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Endpoints
GET  /api/health          # Health check
GET  /api/signals         # Current signals
GET  /api/portfolio       # Portfolio status
POST /api/analyze/{symbol} # Analyze specific symbol
WS   /ws/signals          # Real-time signals
```

Dashboard: `http://localhost:3000` (if running webapp)

## ⚠️ Limitations

- **Long only**: No short selling (retail-focused design)
- **Data latency**: yfinance has 15-min delay
- **Backtest**: Transaction costs not included (estimate -2-3% CAGR)
- **Test coverage**: Needs improvement for production

## 📜 License

MIT License - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for market data
- [scikit-learn](https://scikit-learn.org/) for ML models
- [xAI Grok](https://x.ai/) for sentiment analysis

---

**Disclaimer**: This software is for educational purposes only. Trading involves significant risk of loss. Past performance does not guarantee future results.
