# MIDAS V6.2 - Technical Architecture

*Last updated: 2026-02-08*

## Scoring Engine Overview

MIDAS uses a 5-pillar scoring system with regime adaptation:

### Pillars

| Pillar | Weight Range | Source Files |
|--------|--------------|--------------|
| Technical | 22-30% | `src/agents/pillars/technical_pillar.py` |
| Fundamental | 15-28% | `src/agents/pillars/fundamental_pillar.py` |
| Sentiment | 12-22% | `src/agents/pillars/sentiment_pillar.py` |
| News | 5-15% | `src/agents/pillars/news_pillar.py` |
| ML | 20-30% | `src/agents/pillars/ml_pillar.py` |

### Technical Pillar Features (40+)

**Trend (30%):**
- EMA crossovers (20/50, 50/200)
- MACD histogram & signal
- ADX value & direction
- Supertrend signal
- Aroon oscillator
- Price vs EMA positions

**Momentum (25%):**
- RSI 14 + slope
- Stochastic K/D
- Williams %R
- CCI 20
- ROC 10
- Momentum 10

**Volume (25%):**
- Volume ratio 20d
- OBV trend & slope
- CMF 20
- MFI 14
- Volume breakout detection
- Price-volume trend

**Volatility (20%):**
- ATR %
- ATR ratio
- Bollinger Band width
- BB %B
- 20-day volatility

### Regime Detection

File: `src/agents/regime_adapter.py`

Detection logic:
- **BULL**: SPY > EMA50 +3%, VIX < 20
- **BEAR**: SPY < EMA50 -3%
- **VOLATILE**: VIX > 30 or Vol > 35%
- **RANGE**: Default

### Adaptive ML Gate

File: `src/agents/adaptive_ml_gate.py`

Volatility-based switching:
- Vol > 3% → 5 Pillars only
- Vol ≤ 3% → ML Gate active (boost/block/neutral)

## Learning Modules

### Knowledge Engine
- Learns from past mistakes
- Tracks 21 performance indicators
- Generates adaptive rules

### Dynamic Influence Learner
- No hardcoded influencers
- Discovers impact dynamically
- Updates on trade outcomes

### Smart Signal Learner
- Detects weak signals (mentions, events)
- Correlates with price movements

## Configuration

Pillar weights: `config/pillar_weights.json`

Adjustable per regime (BULL/BEAR/RANGE/VOLATILE).
