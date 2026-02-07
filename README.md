# Midas BULL Optimization + Interactive Brokers Module

## 🎯 Objective
Improve BULL regime win rate from 53.5% to 60%+ and add Interactive Brokers integration for live trading.

## 📁 Files Created

```
midas-bull-optimization/
├── src/
│   ├── brokers/
│   │   ├── __init__.py           # Broker factory
│   │   ├── ib_broker.py          # Interactive Brokers client
│   │   └── paper_trader.py       # Paper trading simulator
│   ├── scoring/
│   │   ├── bull_optimizer.py     # BULL optimization logic
│   │   └── adaptive_scorer_patch.py  # Scorer integration
│   ├── agents/
│   │   └── reasoning_engine_patch.py # Engine integration
│   └── live_loop_integration.py  # Live trading loop example
├── config/
│   └── pillar_weights.json       # Regime-specific weights
├── SCORING_SYSTEM_UPDATE.md      # Detailed documentation
├── README.md                     # This file
└── deploy.sh                     # Deployment script
```

## 🔧 Key Changes

### 1. ATR Trailing Stops
- Dynamic stops based on Average True Range
- Progressively tighten as profit increases
- Allows trends to run while protecting profits

### 2. Extended BULL Targets (+20%)
- Target multiplier of 1.20x in BULL regime
- Additional 10% boost for very strong trends
- Captures more upside in trending markets

### 3. Pullback Filtering
- 3% tolerance before exit signal (4.5% in strong trends)
- Requires 2-bar confirmation
- Prevents premature exits on normal retracements

### 4. Regime-Adjusted Weights
- BULL: Momentum 0.26, Trend 0.25 (boosted)
- BULL: Sentiment 0.12 (reduced - lags in trends)
- Extended max hold to 15 days in BULL

## 🚀 Deployment

### SSH Access Issue
**Note:** SSH access to VPS (46.225.58.233) failed with current key. The key at `/home/node/.openclaw/workspace/.ssh/id_ed25519` is not authorized for root.

**To resolve:**
1. Add the public key to VPS authorized_keys:
   ```bash
   # On VPS:
   echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPhXVNcv9tB2/8/ipf4rXk/r5aafOmtAVU+m37gemYQU thomy03@jarvis" >> ~/.ssh/authorized_keys
   ```
   
2. Or use password authentication to copy files

### Manual Deployment
```bash
# From local machine with VPS access:
cd /home/node/.openclaw/workspace/midas-bull-optimization

# Copy files to VPS
scp -r src/brokers root@46.225.58.233:/root/tradingbot-github/src/
scp -r src/scoring/bull_optimizer.py root@46.225.58.233:/root/tradingbot-github/src/scoring/
scp -r src/scoring/adaptive_scorer_patch.py root@46.225.58.233:/root/tradingbot-github/src/scoring/
scp -r src/agents/reasoning_engine_patch.py root@46.225.58.233:/root/tradingbot-github/src/agents/
scp config/pillar_weights.json root@46.225.58.233:/root/tradingbot-github/config/

# Install ib_insync
ssh root@46.225.58.233 "pip install ib_insync"
```

## 📊 Validation Backtest

Once deployed, run:
```bash
ssh root@46.225.58.233 "cd /root/tradingbot-github && python -m src.backtest.runner --start 2020-01-01 --end 2025-01-01"
```

Expected results:
- BULL win rate: 60%+ (vs 53.5% before)
- Improved profit factor in BULL regime
- Slightly longer average hold times

## 🔌 Interactive Brokers Setup

### Requirements
```bash
pip install ib_insync
```

### Configuration (.env)
```env
TRADING_MODE=paper     # "paper" or "live"
IB_HOST=127.0.0.1
IB_PORT=7497          # 7497=TWS paper, 7496=TWS live
IB_CLIENT_ID=1
IB_ACCOUNT=           # Leave empty for default
```

### IB TWS/Gateway Setup
1. Install IB Trader Workstation or IB Gateway
2. Enable API connections: Configure → API → Settings
   - Enable ActiveX and Socket Clients
   - Socket port: 7497 (paper) or 7496 (live)
   - Allow connections from localhost
3. Disable read-only mode for trading

### Usage Example
```python
from brokers import get_broker, OrderSide

# Paper trading
broker = get_broker(mode="paper")
await broker.connect()

# Place order
result = await broker.place_order(
    symbol="AAPL",
    side=OrderSide.BUY,
    quantity=10
)

# Get positions
positions = await broker.get_positions()
```

## ⚠️ Important Notes

1. **Backtest first** - Always validate changes with historical data before live trading
2. **Paper trade** - Test with paper trading before going live
3. **Monitor closely** - Watch the first few BULL regime trades carefully
4. **Risk management** - The wider trailing stops mean slightly larger potential losses per trade

## 📈 Monitoring

Track these metrics after deployment:
- Win rate by regime (target: BULL ≥ 60%)
- Average hold time in BULL (expect ~8 days)
- Trailing stop vs fixed stop exits
- Pullback recoveries (trades saved by filter)

---

*Created: 2026-02-05*
