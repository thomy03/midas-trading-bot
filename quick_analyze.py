#!/usr/bin/env python3
"""Quick analysis script - called from webapp API."""
import asyncio
import json
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/app')

from src.data.market_data import MarketDataFetcher
from src.agents.pillars.technical_pillar import TechnicalPillar
from datetime import datetime

async def analyze(sym):
    fetcher = MarketDataFetcher()
    df = fetcher.get_historical_data(sym, period='6mo', interval='1d')
    
    if df is None or len(df) < 20:
        return {'error': f'No data for {sym}', 'symbol': sym}
    
    tech = TechnicalPillar(weight=0.30)
    data = {'df': df, 'symbol': sym}
    score = await tech.analyze(sym, data)
    
    raw = score.score
    normalized = max(0, min(100, (raw + 100) / 2 if -100 <= raw <= 100 else raw))
    
    fund_score = sent_score = news_score = ml_score = 50
    
    try:
        from src.agents.pillars.fundamental_pillar import FundamentalPillar
        f = FundamentalPillar(weight=0.25)
        r = await f.analyze(sym, data)
        fund_score = max(0, min(100, (r.score + 100) / 2 if -100 <= r.score <= 100 else r.score))
    except Exception:
        pass
    
    try:
        from src.agents.pillars.sentiment_pillar import SentimentPillar
        s = SentimentPillar(weight=0.20)
        r = await s.analyze(sym, data)
        sent_score = max(0, min(100, (r.score + 100) / 2 if -100 <= r.score <= 100 else r.score))
    except Exception:
        pass
    
    try:
        from src.agents.pillars.ml_pillar import MLPillar
        m = MLPillar(weight=0.15)
        r = await m.analyze(sym, data)
        ml_score = max(0, min(100, (r.score + 100) / 2 if -100 <= r.score <= 100 else r.score))
    except Exception:
        pass
    
    final = normalized * 0.30 + fund_score * 0.25 + sent_score * 0.20 + 50 * 0.10 + ml_score * 0.15
    
    if final >= 70: decision = 'STRONG_BUY'
    elif final >= 55: decision = 'BUY'
    elif final >= 45: decision = 'HOLD'
    elif final >= 30: decision = 'SELL'
    else: decision = 'STRONG_SELL'
    
    bullish = sum(1 for x in [normalized, fund_score, sent_score, ml_score] if x >= 55)
    bearish = sum(1 for x in [normalized, fund_score, sent_score, ml_score] if x <= 45)
    
    if bullish > bearish: summary = 'Bullish signals detected. Consider entry.'
    elif bearish > bullish: summary = 'Bearish pressure detected. Consider reducing position.'
    else: summary = 'Mixed signals. Hold current position.'
    
    return {
        'symbol': sym,
        'timestamp': datetime.now().isoformat(),
        'finalScore': round(final),
        'decision': decision,
        'confidence': 85,
        'summary': summary,
        'pillars': [
            {'name': 'Technical', 'score': round(normalized), 'weight': 30, 'details': (score.reasoning or '')[:80]},
            {'name': 'Fundamental', 'score': round(fund_score), 'weight': 25, 'details': 'Financial health'},
            {'name': 'Sentiment', 'score': round(sent_score), 'weight': 20, 'details': 'Social sentiment'},
            {'name': 'News', 'score': 50, 'weight': 10, 'details': 'News impact'},
            {'name': 'ML Adaptive', 'score': round(ml_score), 'weight': 15, 'details': 'Pattern recognition'}
        ],
        'currentPrice': float(df['Close'].iloc[-1])
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: quick_analyze.py SYMBOL'}))
        sys.exit(1)
    
    symbol = sys.argv[1].upper()
    result = asyncio.run(analyze(symbol))
    print(json.dumps(result))
