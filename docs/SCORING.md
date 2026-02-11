# MIDAS V8.1 — Scoring System

## Overview

Scores are computed by the **ReasoningEngine** using 2 active pillars (Technical 55%, Fundamental 45%) plus an ML confirmation gate. Internal scale is -100 to +100, converted to display 0–100 via `(internal + 100) / 2`.

## Pillar Weights

| Pillar | Weight | Status |
|--------|--------|--------|
| Technical | 0.55 | ✅ Active |
| Fundamental | 0.45 | ✅ Active |
| Sentiment | 0.00 | ⚠️ Disabled (orchestrator handles globally) |
| News | 0.00 | ⚠️ Disabled (orchestrator handles globally) |
| ML | 0.00 | Acts as **confirmation gate**, not weighted |

Weights can be adjusted by the **Adaptive Scorer** (learns from past trades, clamped 10–40% per pillar, learning rate 0.05, min 20 trades).

---

## Technical Pillar (55%)

### Internal Categories

| Category | Weight | Indicators |
|----------|--------|------------|
| Trend | 30% | EMA 20/50/200 alignment, MACD (12,26,9), ADX (14) |
| Momentum | 25% | RSI (14), Stochastic K/D (14,3) |
| Volume | 25% | Volume Ratio (20d), OBV (5d) |
| Volatility | 20% | ATR (14) ratio, Bollinger Bands (20, 2σ) |

### Trend (30%)

**EMA Alignment:**
- Price > EMA20 > EMA50 > EMA200 → **+80** (strong uptrend)
- Price < EMA20 < EMA50 < EMA200 → **-80** (strong downtrend)
- Partial alignment → ±30 to ±60

**MACD:** +70 (bullish strengthening) to -70 (bearish strengthening)

**ADX:** > 25 with +DI > -DI → up to +80; with -DI > +DI → down to -80; ≤ 25 → 0

### Momentum (25%)

**RSI (14):** < 30 → +60 (oversold opportunity), > 70 → -60 (overbought risk), trending ±30

**Stochastic:** < 20 → +50, > 80 → -50, crossovers ±20

### Volume (25%)

**Volume Ratio:** > 2× avg + price up → +70, + price down → -70

**OBV divergence:** OBV up + price down → +50 (accumulation), OBV down + price up → -50 (distribution)

### Volatility (20%)

**ATR ratio** (vs 20d SMA): > 1.5 → -20, < 0.7 → +20

**Bollinger:** Price below lower band → +50, above upper → -50

### Technical Score

```
score = trend(30%) + momentum(25%) + volume(25%) + volatility(20%)
```

Confidence = 0.5 + (indicator agreement × 0.5), range 0.5–1.0.

---

## Fundamental Pillar (45%)

### Internal Categories

| Category | Weight | Metrics |
|----------|--------|---------|
| Valuation | 30% | P/E (vs sector), P/B, P/S, PEG |
| Growth | 30% | Revenue Growth, Earnings Growth, EPS Forecast |
| Profitability | 25% | Profit Margin, ROE, Operating Margin |
| Financial Health | 15% | Debt/Equity, Current Ratio |

### Key Scoring Examples

**P/E** (compared to sector average — Tech=30, Finance=15, Energy=12, etc.):
- PE < sector × 0.6 → +70 (undervalued)
- PE > sector × 2.0 → -70 (very overvalued)

**Revenue Growth:** > 30% → +70, 15–30% → +50, < 0% → -50

**ROE:** > 20% → +60, < 0% → -40

**Debt/Equity:** < 30% → +50, > 150% → -50

Confidence: fixed 0.7 (fundamentals change slowly). Data quality = available_metrics / total_metrics.

---

## ML Confirmation Gate

The ML Pillar (RandomForestClassifier, 40+ features) has **weight = 0** but acts as a gate:

| ML Score (0–100) | Effect on BUY/STRONG_BUY |
|-------------------|--------------------------|
| < 40 | **BLOCKED** → downgraded to HOLD |
| 40–60 | Pass unchanged |
| ≥ 60 | **BONUS** +5 pts to final score |

### ML Features (40+)

- **Trend (10):** EMA crosses, MACD, ADX, Supertrend, Aroon, price vs EMAs
- **Momentum (10):** RSI, Stochastic, Williams %R, CCI, ROC, momentum
- **Volume (8):** Volume ratio, OBV, CMF, MFI, volume breakout
- **Volatility (6):** ATR%, BB width/%, 20d realized vol
- **Regime (6):** SPY vs EMA50, VIX level/percentile, sector momentum

Model: `RandomForestClassifier(n_estimators=100, max_depth=10)`. Min 50 trades to train. Falls back to heuristic scoring if no trained model.

### Adaptive ML Gate (V6.2)

Additional `GradientBoostingClassifier` gate, **active only in low volatility** (< 3%):

| Volatility | ML Confidence | Effect |
|------------|---------------|--------|
| > 3% | N/A | Pillars decide alone |
| ≤ 3% | > 0.6 | Score + 5 |
| ≤ 3% | < 0.4 | Score → **0** (blocked) |
| ≤ 3% | 0.4–0.6 | No change |

---

## Score Computation Pipeline

```
1. Fetch market context (SPY, VIX, regime)
2. Run 4 pillars in parallel
3. Run ML Pillar
4. Determine weights (adaptive or config)
5. Compute internal score:
   internal = Σ(pillar.score × weight × data_quality)
6. Apply regime multiplier:
   BULL ×1.05, RANGE ×1.00, BEAR ×0.90, CRASH ×0.60
7. Convert: display = (internal + 100) / 2
8. Apply Adaptive ML Gate (if low volatility)
9. Determine decision (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL)
10. Apply ML Confirmation Gate (block or boost)
11. Apply Knowledge Engine (trap detection, lessons learned)
12. Final score + confidence + reasoning
```

### Decision Thresholds

| Decision | Display Score |
|----------|---------------|
| STRONG_BUY | ≥ 70 |
| BUY | 55 – 69 |
| HOLD | 40 – 54 |
| SELL | 25 – 39 |
| STRONG_SELL | < 25 |

### Context Adjustments

| Context | Weight Adjustment |
|---------|-------------------|
| High Volatility | Tech ×1.2, Funda ×1.1 |
| Bear Market | Funda ×1.2 |
| Bull Market | (minor adjustments) |

Weights are renormalized to sum = 1.0 after adjustment.

### Sector-Regime Bonus (±12 pts max)

| Sector | BULL | BEAR | VOLATILE |
|--------|------|------|----------|
| Growth (Tech) | +5 | -5 | -7 |
| Defensive (Staples, Utilities) | -3 | +5 | +7 |
| Cyclical (Consumer, Industrial) | +3 | -3 | -5 |

Additional market-cap bonus: Mega caps get +3 in BEAR/VOLATILE, micro caps get +3 in BULL.

---

*See also: [Architecture](ARCHITECTURE.md) | [Risk Management](RISK.md)*
