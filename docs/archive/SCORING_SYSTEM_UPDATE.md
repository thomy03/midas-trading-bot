# Midas Scoring System - BULL Regime Optimization Update

## Overview

This update addresses the underperformance of the BULL regime (53.5% win rate vs 61% in VOLATILE). The changes implement adaptive strategies specifically designed to capitalize on trending market conditions.

## Problem Analysis

### Why BULL was underperforming:
1. **Fixed stop losses** - Triggered on normal trend pullbacks, causing premature exits
2. **Conservative targets** - Left money on the table in strong trends  
3. **Equal weighting** - Didn't account for different signal importance in trends
4. **Pullback sensitivity** - Treated temporary dips as exit signals

## Solution Components

### 1. ATR-Based Trailing Stops (`bull_optimizer.py`)

**Before:** Fixed percentage stop (e.g., -5%)
**After:** Dynamic ATR trailing stop that adapts to volatility

```python
# Configuration
TrailingStopConfig(
    atr_multiplier=2.0,           # Stop at 2x ATR from peak
    min_profit_to_activate=0.02,  # Activate after 2% profit
    tighten_at_profit={           # Progressively tighten
        0.05: 1.8,  # At 5% profit → 1.8x ATR
        0.10: 1.5,  # At 10% profit → 1.5x ATR
        0.15: 1.2,  # At 15% profit → 1.2x ATR
    }
)
```

**Benefits:**
- Allows positions to breathe during normal volatility
- Protects profits progressively as they grow
- Adapts to each asset's volatility characteristics

### 2. Extended Targets in BULL (+20%)

**Before:** Same targets regardless of regime
**After:** +20% target extension when regime=BULL and trend is strong

```python
# Example
base_target = 1.08  # 8% target
if regime == "BULL" and trend_strength > 0.6:
    adjusted_target = 1.08 * 1.20 = 1.296  # ~10% target
```

**Rationale:** Strong trends often exceed initial targets. By extending targets in confirmed BULL regimes, we capture more upside.

### 3. Pullback Filtering

**Before:** Any significant dip triggered sell signal
**After:** Require confirmation before exiting on pullbacks

```python
PullbackFilter(
    tolerance=0.03,          # 3% pullback allowed
    confirmation_bars=2      # Need 2 bars below tolerance
)

# In BULL with strong trend, tolerance increases to 4.5%
```

**Why this helps:** In uptrends, 2-4% pullbacks are normal and often followed by continuation. This prevents selling into weakness that recovers.

### 4. Regime-Adjusted Pillar Weights

**Base Weights:**
| Pillar | Base | BULL | VOLATILE | BEAR |
|--------|------|------|----------|------|
| Technical | 0.25 | 0.22 | 0.30 | 0.22 |
| Fundamental | 0.20 | 0.15 | 0.18 | 0.28 |
| Sentiment | 0.15 | 0.12 | 0.17 | 0.18 |
| Momentum | 0.20 | **0.26** | 0.18 | 0.16 |
| Trend | 0.20 | **0.25** | 0.17 | 0.16 |

**BULL changes:**
- Momentum: 0.20 → 0.26 (+30%)
- Trend: 0.20 → 0.25 (+25%)
- Sentiment: 0.15 → 0.12 (-20%) - often lags in trends

### 5. Enhanced Hold Logic

New criteria for holding longer in BULL:
- If position profitable AND momentum > 0.6 → HOLD
- If trend strength > 0.6 AND profit > 2% → HOLD
- Extended max hold from 10 to 15 days in BULL

## Configuration File (`pillar_weights.json`)

```json
{
  "regime_adjustments": {
    "BULL": {
      "parameters": {
        "target_multiplier": 1.20,
        "trailing_stop_atr": 2.0,
        "max_hold_days": 15,
        "pullback_tolerance": 0.03,
        "min_trend_strength": 0.6,
        "momentum_threshold": 0.55
      }
    }
  }
}
```

## Expected Impact

Based on the optimizations:

| Metric | Before | Expected After |
|--------|--------|----------------|
| BULL Win Rate | 53.5% | 60-62% |
| BULL Avg Win | +6.2% | +7.5-8% |
| BULL Avg Loss | -4.1% | -4.5% (slightly wider stops) |
| BULL Profit Factor | 1.1 | 1.35-1.45 |

**Trade-offs:**
- Slightly larger individual losses (wider stops)
- Fewer trades (more filtering)
- Higher conviction on entries
- Better capture of full trend moves

## Integration Steps

### 1. Copy new files:
```bash
cp src/scoring/bull_optimizer.py /root/tradingbot-github/src/scoring/
cp src/scoring/adaptive_scorer_patch.py /root/tradingbot-github/src/scoring/
cp src/agents/reasoning_engine_patch.py /root/tradingbot-github/src/agents/
cp config/pillar_weights.json /root/tradingbot-github/config/
```

### 2. Update existing files:

**In `reasoning_engine.py`:**
```python
from .reasoning_engine_patch import ReasoningEngineBullPatch

class ReasoningEngine:
    def __init__(self):
        # ... existing code ...
        self.bull_patch = ReasoningEngineBullPatch()
```

**In `adaptive_scorer.py`:**
```python
from .adaptive_scorer_patch import AdaptiveScorerBullPatch

class AdaptiveScorer:
    def __init__(self):
        # ... existing code ...
        self.bull_patch = AdaptiveScorerBullPatch()
```

### 3. Run validation backtest:
```bash
python -m src.backtest.runner --start 2020-01-01 --end 2025-01-01 --regime-filter BULL
```

## Interactive Brokers Integration

New broker module added for live trading:

### Files:
- `src/brokers/ib_broker.py` - IB TWS/Gateway client
- `src/brokers/paper_trader.py` - Paper trading simulator
- `src/brokers/__init__.py` - Factory function

### Environment Variables:
```env
TRADING_MODE=paper  # or "live"
IB_HOST=127.0.0.1
IB_PORT=7497        # 7497=TWS paper, 7496=TWS live
IB_CLIENT_ID=1
```

### Usage:
```python
from brokers import get_broker, OrderSide, OrderType

broker = get_broker(mode="paper")  # or "live"
await broker.connect()

result = await broker.place_order(
    symbol="AAPL",
    side=OrderSide.BUY,
    quantity=10,
    order_type=OrderType.MARKET
)
```

## Monitoring

Key metrics to track after deployment:
1. Win rate by regime (target: BULL ≥ 60%)
2. Average hold time in BULL (expect increase from ~5 to ~8 days)
3. Trailing stop activation rate
4. Pullback filter saves (trades held that recovered)

---

*Update: 2026-02-05*
*Version: 2.0.0*
