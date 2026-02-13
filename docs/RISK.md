# MIDAS V8.1 — Risk Management

## Position Sizing

### Multi-Strategy Sizing (Primary — Paper Trading)

Score-based dynamic sizing per strategy profile:

| Profile | Base Size | Score ≥90 | Score ≥85 | Score ≥80 | < 80 |
|---------|-----------|-----------|-----------|-----------|------|
| Aggressive | 8% | 16% | 12.8% | 9.6% | 8% |
| Moderate | 5% | 10% | 8% | 6% | 5% |

Multipliers: ≥90 → ×2.0, ≥85 → ×1.6, ≥80 → ×1.2, <80 → ×1.0

### Paper Trader Sizing (risk-per-trade)

| Score | Risk/Trade | Max Position |
|-------|-----------|--------------|
| ≥ 90 | 1.5% | 10% |
| ≥ 85 | 1.2% | 8% |
| ≥ 80 | 1.0% | 6% |
| ≥ 75 | 0.7% | 5% |
| < 75 | 0.5% | 3% |

### Advanced Sizing (PositionSizer — Kelly)

For live trading with 20+ trade history:

```
size = base(Kelly/2) × vol_adj × confidence_mult × regime_mult
     clamped to [3%, 20%]
```

- **Kelly**: half-Kelly from win rate + avg win/loss ratio
- **Volatility**: `target_risk(1.5%) / ATR%`, clipped [0.5, 2.0]
- **Confidence**: ≥90 → ×1.5, ≥80 → ×1.2, ≥70 → ×1.0, <65 → ×0.6
- **Regime**: BULL ×1.0, RANGE ×0.8, BEAR ×0.6, VOLATILE ×0.5

---

## Stop-Loss

### ATR-Based (Dynamic Stops)

```
SL distance = ATR(14) × multiplier
```

| Strategy | ATR Multiplier |
|----------|---------------|
| Aggressive | 1.5× |
| Moderate | 2.0× |

Confidence adjusts: high confidence → tighter stop (1.5× ATR), low confidence → wider (2.25× ATR).

**Clamp**: 2% minimum, 10% maximum distance from entry.

### Regime-Imposed Stops

| Regime | Max Stop-Loss |
|--------|---------------|
| BULL | 8% |
| RANGE | 6% |
| BEAR | 5% |
| VOLATILE | 4% |

### Defensive Manager Stops

| Level | Trigger | Stop-Loss | Max Invested | Min Score | Max Positions |
|-------|---------|-----------|-------------|-----------|---------------|
| NONE | Normal | 8% | 100% | 75 | 20 |
| CAUTIOUS | Minor stress | 6% | 70% | 78 | 15 |
| DEFENSIVE | Significant stress | 4% | 30% | 85 | 8 |
| MAXIMUM | Severe stress | 3% | 15% | 90 | 5 |

---

## Take-Profit

### ATR-Based

| Strategy | ATR Multiplier | R:R Ratio |
|----------|---------------|-----------|
| Aggressive | 3.0× | 2:1 |
| Moderate | 4.0× | 2:1 |

Confidence boosts R:R from 1.5:1 (low) to 2.0:1 (high). **Clamp**: 5–30%.

---

## Trailing Stops

### Adaptive Position Manager
- **Activation**: after +5% unrealized gain
- **Trail distance**: 12% from highest price
- **Direction**: only moves up, never down

### Position Manager (agents)
- **Activation**: after +5% gain
- **Trail distance**: 3% from peak
- **Updated**: every scan cycle

---

## Circuit Breakers

| Condition | Action |
|-----------|--------|
| Daily loss ≥ 3% | Enter defensive mode |
| Max drawdown ≥ 15% | Enter defensive mode |
| 5 consecutive losses | Enter defensive mode |

---

## Defensive Manager

4-level progressive defense system triggered by consecutive losses, drawdown, or market stress:

1. **NONE** → Full trading
2. **CAUTIOUS** → Reduced exposure (70%), tighter stops, higher score threshold
3. **DEFENSIVE** → Minimal exposure (30%), very tight stops, only high-conviction trades
4. **MAXIMUM** → Near-halt (15%), strictest stops, only exceptional opportunities

---

## Correlation & Diversification

Enforced by `CorrelationManager`:

| Rule | Limit |
|------|-------|
| Max single sector exposure | 25% |
| Max single stock exposure | 15% |
| Max average correlation with existing positions | 0.70 |
| Correlation lookback | 60 days |

Only checked with ≥2 existing positions.

---

## Portfolio Rotation

When portfolio is full and a stronger signal appears:

- New signal must score **15+ points** above weakest position
- Weakest position must be held **≥3 days** (anti-churning)
- Winners > 10% are **protected** from rotation
- Max **2 rotations/day**

Rotation score factors: current score, P&L performance, holding duration, Grok sentiment.

---

## Exit Conditions Summary

A position is closed when ANY of these triggers:

| Condition | Source |
|-----------|--------|
| Price ≤ Stop-Loss | ATR-based, regime, or defensive stop |
| Price ≥ Take-Profit | ATR-based TP |
| Trailing stop hit | After +5% gain, trails 3–12% from peak |
| Score drops ≥10 pts | Periodic score review (24h) |
| Score < 40 absolute | Auto-sell |
| Decision → SELL/STRONG_SELL | Score re-evaluation |
| Grok sentiment < -0.6 | LLM agent only |
| Max hold 30 days | Time-based auto-exit |
| Portfolio rotation | Replaced by stronger signal |

---

## Max Trades & Screening Limits

| Limit | Value |
|-------|-------|
| Max screens per cycle | 3 |
| Max trades per day | 3 |
| Symbol cooldown | 4 hours (bypass if heat > 0.8) |
| Max positions (aggressive) | 10 |
| Max positions (moderate) | 6 |

---

*See also: [Architecture](ARCHITECTURE.md) | [Scoring](SCORING.md)*
