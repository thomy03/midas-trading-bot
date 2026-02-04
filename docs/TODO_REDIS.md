# TODO: Redis Cache Integration

## Status
❌ Redis not currently available on VPS

## Proposed Architecture

### Why Redis?
1. **Faster cache access** - In-memory vs disk parquet files
2. **TTL management** - Native expiration support
3. **Shared cache** - Multiple workers can share data
4. **Pub/Sub** - Real-time notifications for price alerts

### Components to Cache

```
┌─────────────────────────────────────────────────────────────┐
│                      REDIS CACHE LAYERS                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  L1: HOT DATA (TTL: 5min)                                   │
│  ├── Current prices (realtime)                              │
│  ├── Active signals                                         │
│  └── Session state                                          │
│                                                              │
│  L2: WARM DATA (TTL: 4h)                                    │
│  ├── Stock info (marketCap, sector, etc.)                   │
│  ├── Ticker lists per market                                │
│  └── Screening results                                      │
│                                                              │
│  L3: COLD DATA (TTL: 24h)                                   │
│  ├── Historical OHLCV (1d interval)                         │
│  ├── Weekly OHLCV (1wk interval)                            │
│  └── Backtest results                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Schema

```python
# Prices
"price:{symbol}"              -> float       # TTL: 5min
"price:batch:{timestamp}"     -> dict        # TTL: 5min

# Stock Info
"info:{symbol}"               -> dict        # TTL: 4h
"marketcap:{symbol}"          -> float       # TTL: 24h

# Historical Data (serialized parquet or JSON)
"hist:{symbol}:{interval}:{period}" -> bytes # TTL: 24h

# Ticker Lists
"tickers:{market}"            -> list        # TTL: 4h
"tickers:all"                 -> list        # TTL: 4h

# Screening/Signals
"signals:active"              -> list        # TTL: 1h
"screening:{date}"            -> list        # TTL: 24h
```

### Implementation Steps

1. **Install Redis on VPS**
   ```bash
   apt install redis-server
   systemctl enable redis-server
   systemctl start redis-server
   ```

2. **Add redis dependency**
   ```
   redis>=4.5.0
   ```

3. **Create cache wrapper** (`src/utils/redis_cache.py`)
   ```python
   import redis
   import json
   import pandas as pd
   from typing import Optional, Any
   
   class RedisCache:
       def __init__(self, host='localhost', port=6379, db=0):
           self.client = redis.Redis(host=host, port=port, db=db)
       
       def get_price(self, symbol: str) -> Optional[float]:
           val = self.client.get(f"price:{symbol}")
           return float(val) if val else None
       
       def set_price(self, symbol: str, price: float, ttl: int = 300):
           self.client.setex(f"price:{symbol}", ttl, price)
       
       def get_historical(self, symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
           key = f"hist:{symbol}:{interval}:{period}"
           data = self.client.get(key)
           if data:
               return pd.read_json(data)
           return None
       
       def set_historical(self, symbol: str, interval: str, period: str, 
                          df: pd.DataFrame, ttl: int = 86400):
           key = f"hist:{symbol}:{interval}:{period}"
           self.client.setex(key, ttl, df.to_json())
   ```

4. **Integrate with MarketDataFetcher**
   - Add Redis as L1 cache before disk cache
   - Fallback: Redis → Disk → API

### docker-compose addition

```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

volumes:
  redis_data:
```

### Environment Variables

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=  # optional
```

---

## Priority: LOW-MEDIUM
Current disk cache with parquet is working well for nightly prefetch.
Redis would be more beneficial when:
- Running multiple bot instances
- Need real-time price updates
- Want pub/sub for alerts

## Estimated Effort: 4-6 hours
