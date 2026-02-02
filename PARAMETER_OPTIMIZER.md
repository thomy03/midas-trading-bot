# Parameter Optimizer - Learning Engine for TradingBot

## Overview

The **Parameter Optimizer** is an auto-learning engine that analyzes completed trades to:
1. **Identify winning/losing patterns** - Find combinations of indicators that predict success
2. **Suggest parameter adjustments** - Optimize thresholds, periods, and weights
3. **Track learning over time** - Build institutional memory of what works

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Parameter Optimizer                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│  │    Trade     │   │   Pattern    │   │  Parameter   │       │
│  │   Recorder   │──▶│   Analyzer   │──▶│   Suggester  │       │
│  └──────────────┘   └──────────────┘   └──────────────┘       │
│         │                  │                  │                │
│         ▼                  ▼                  ▼                │
│  ┌──────────────────────────────────────────────────┐         │
│  │              Data Storage Layer                   │         │
│  │  - trade_analysis_db.json (all trades)           │         │
│  │  - learned_patterns.json (patterns)              │         │
│  │  - parameter_history.json (changes)              │         │
│  └──────────────────────────────────────────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Nightly Auditor                             │
│  - Calls optimizer.optimize_from_trades()                       │
│  - Reviews recommendations                                      │
│  - Applies changes (with confidence thresholds)                │
└─────────────────────────────────────────────────────────────────┘
```

## Data Captured Per Trade

### Indicator Snapshot
Every completed trade captures a comprehensive snapshot of **ALL** indicators at entry:

#### Technical
| Category | Indicators |
|----------|------------|
| Trend | RSI, MACD (value, signal, histogram), EMA (20, 50, 200), ADX |
| Momentum | Stochastic K/D, Williams %R, CCI, ROC |
| Volume | Volume Ratio, OBV Trend, VWAP Position, CMF, MFI |
| Volatility | ATR, ATR%, Bollinger Position, BB Width |
| Structure | Distance to Support/Resistance |

#### Fundamental
- P/E Ratio, PEG Ratio
- Revenue Growth, Profit Margin
- Debt/Equity, ROE, Current Ratio

#### Sentiment
- Grok X/Twitter Score
- Overall Sentiment Score
- Social Volume (relative activity)
- Sources used

#### News
- News Score, News Count
- Earnings proximity flag
- Analyst update flag

#### Market Context
- Market Regime (bull/bear/range/volatile)
- SPY Trend
- VIX Level
- Sector Momentum

## Pattern Recognition

### Single Indicator Patterns
The optimizer analyzes performance across indicator ranges:

```python
# Example: RSI buckets
rsi_buckets = [
    (0, 30, "oversold"),      # RSI < 30
    (30, 50, "neutral_low"),   # RSI 30-50
    (50, 70, "neutral_high"),  # RSI 50-70
    (70, 100, "overbought")    # RSI > 70
]
```

For each bucket, it calculates:
- Win Rate
- Average Return
- Profit Factor
- Sample Size → Confidence Level

### Combination Patterns
More powerful multi-condition patterns:

| Pattern | Conditions | Typical Performance |
|---------|------------|---------------------|
| `rsi_oversold_volume_spike` | RSI < 30 AND Volume > 2x | High win rate on reversals |
| `bullish_ema_strong_trend` | Bullish EMA + ADX > 30 | Good momentum entries |
| `tech_sentiment_alignment` | Technical > 60 AND Sentiment > 60 | Confirmation trades |
| `high_conviction_low_vix` | Score > 70 AND VIX < 18 | Low-risk high conviction |

## Parameter Optimization

### Optimizable Parameters

| Parameter | Default | Min | Max | Step |
|-----------|---------|-----|-----|------|
| `rsi_oversold` | 30 | 20 | 40 | 5 |
| `rsi_overbought` | 70 | 60 | 80 | 5 |
| `volume_ratio_min` | 1.5 | 1.0 | 3.0 | 0.25 |
| `adx_threshold` | 25 | 15 | 35 | 5 |
| `min_confidence` | 55 | 40 | 80 | 5 |
| `min_combined_score` | 55 | 45 | 75 | 5 |
| `weight_technical` | 0.25 | 0.15 | 0.40 | 0.05 |
| `weight_fundamental` | 0.25 | 0.10 | 0.35 | 0.05 |
| `weight_sentiment` | 0.25 | 0.10 | 0.35 | 0.05 |
| `weight_news` | 0.25 | 0.05 | 0.25 | 0.05 |

### Confidence Levels

Changes are tagged with confidence based on sample size:
- **HIGH** (≥20 samples): Statistically significant, auto-apply
- **MEDIUM** (10-19 samples): Likely significant, review recommended
- **LOW** (<10 samples): Indicative only, manual review required

## Integration with NightlyAuditor

The optimizer integrates into the nightly audit cycle:

```python
# In nightly_auditor.py - run_audit_phase()

async def run_audit_phase(self):
    # 1. Existing audit logic
    await self.run_audit(trades)
    
    # 2. NEW: Parameter optimization
    from src.agents.parameter_optimizer import get_parameter_optimizer
    
    optimizer = get_parameter_optimizer()
    opt_report = await optimizer.optimize_from_trades(trades)
    
    # 3. Apply high-confidence changes
    if opt_report.recommended_changes:
        applied = optimizer.apply_optimizations(
            [c for c in opt_report.recommended_changes 
             if c.confidence in ['high', 'medium']]
        )
    
    # 4. Log results
    logger.info(f"Optimization: {len(opt_report.winning_patterns)} winning patterns")
    logger.info(f"Optimization: {len(opt_report.recommended_changes)} changes suggested")
```

## File Storage

| File | Purpose |
|------|---------|
| `data/auditor/trade_analysis_db.json` | All trades with full indicator snapshots |
| `data/auditor/learned_patterns.json` | Identified winning/losing patterns |
| `data/auditor/parameter_history.json` | History of parameter changes |
| `data/auditor/optimization_reports/` | Daily optimization reports |

## Usage Examples

### Recording a Trade
```python
from src.agents.parameter_optimizer import get_parameter_optimizer

optimizer = get_parameter_optimizer()

# After a trade completes
record = optimizer.record_trade(
    trade_data={
        'trade_id': 'TRD001',
        'symbol': 'NVDA',
        'entry_price': 140.0,
        'exit_price': 147.0,
        'pnl_percent': 5.0,
        'exit_type': 'take_profit'
    },
    pillar_scores={
        'technical': 72,
        'fundamental': 65,
        'sentiment': 78,
        'news': 55,
        'combined': 68,
        'confidence': 75
    },
    indicators={
        'rsi': 35,
        'volume_ratio': 2.3,
        'adx': 32,
        'ema_alignment': 'bullish',
        'vix_level': 16
    },
    weights={'technical': 0.25, 'fundamental': 0.25, 'sentiment': 0.25, 'news': 0.25}
)
```

### Running Optimization
```python
# Run full optimization
report = await optimizer.optimize_from_trades()

# Check results
print(f"Win Rate: {report.win_rate:.0%}")
print(f"Winning patterns: {len(report.winning_patterns)}")

# Review recommended changes
for change in report.recommended_changes:
    print(f"{change.parameter_name}: {change.current_value} → {change.proposed_value}")
    print(f"  Reason: {change.reason}")
    print(f"  Confidence: {change.confidence}")
```

### Getting Patterns
```python
# Get high-confidence patterns only
patterns = optimizer.get_patterns(min_confidence=OptimizationConfidence.MEDIUM)

for p in patterns:
    print(f"{p.description}: {p.win_rate:.0%} ({p.total_trades} trades)")
```

## Daily Report Format

```json
{
  "date": "2025-02-01T20:00:00",
  "trades_analyzed": 45,
  "total_pnl": 12.3,
  "win_rate": 0.62,
  "profit_factor": 1.85,
  
  "winning_patterns": [
    {
      "pattern_id": "rsi_oversold_volume_spike",
      "description": "RSI < 30 + Volume > 2x",
      "win_rate": 0.78,
      "total_trades": 23,
      "confidence": "high"
    }
  ],
  
  "recommended_changes": [
    {
      "parameter_name": "rsi_oversold",
      "current_value": 30,
      "proposed_value": 25,
      "reason": "RSI < 25 shows 75% win rate vs 60% at current",
      "confidence": "medium"
    }
  ],
  
  "recommended_weights": {
    "technical": 0.30,
    "fundamental": 0.20,
    "sentiment": 0.30,
    "news": 0.20
  }
}
```

## Safety & Constraints

1. **Parameter Bounds**: All parameters have hard min/max limits
2. **Confidence Gates**: Only HIGH/MEDIUM confidence changes auto-apply
3. **History Tracking**: All changes are logged for audit/rollback
4. **Gradual Changes**: Step sizes prevent drastic swings
5. **Sample Requirements**: Minimum 5 trades for any pattern recognition

## Evolution Path

Future enhancements planned:
- ML-based pattern discovery (beyond predefined combinations)
- A/B testing of parameter changes
- Regime-specific parameters (different settings for bull/bear markets)
- Cross-symbol pattern learning
- Real-time parameter adjustment during trading
