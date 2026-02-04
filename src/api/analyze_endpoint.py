"""
Quick analysis endpoint using the ReasoningEngine pillars.
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import data fetcher
from src.data.market_data import MarketDataFetcher

# Import pillars
from src.agents.pillars.technical_pillar import TechnicalPillar
from src.agents.pillars.fundamental_pillar import FundamentalPillar
from src.agents.pillars.sentiment_pillar import SentimentPillar
from src.agents.pillars.news_pillar import NewsPillar

# Try ML pillar
try:
    from src.agents.pillars.ml_pillar import MLPillar
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


async def analyze_symbol(symbol: str) -> Dict[str, Any]:
    """
    Analyze a symbol using all available pillars.
    
    Returns a complete analysis with scores from each pillar.
    """
    symbol = symbol.upper().strip()
    logger.info(f"[ANALYZE] Starting analysis for {symbol}")
    
    # Initialize data fetcher
    fetcher = MarketDataFetcher()
    
    # Fetch OHLCV data
    try:
        df = await asyncio.to_thread(
            fetcher.get_daily_data, symbol, days=100
        )
        if df is None or len(df) < 20:
            return {
                "symbol": symbol,
                "error": f"Insufficient data for {symbol}",
                "timestamp": datetime.now().isoformat()
            }
        logger.info(f"[ANALYZE] Got {len(df)} data points for {symbol}")
    except Exception as e:
        logger.error(f"[ANALYZE] Failed to fetch data: {e}")
        return {
            "symbol": symbol,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
    
    # Prepare data dict for pillars
    data = {
        'df': df,
        'symbol': symbol,
        'current_price': float(df['Close'].iloc[-1]) if len(df) > 0 else 0
    }
    
    # Initialize pillars
    pillars = {
        'technical': TechnicalPillar(weight=0.30),
        'fundamental': FundamentalPillar(weight=0.25),
        'sentiment': SentimentPillar(weight=0.20),
        'news': NewsPillar(weight=0.10),
    }
    
    if ML_AVAILABLE:
        pillars['ml'] = MLPillar(weight=0.15)
    
    # Run all pillar analyses
    results = {}
    total_score = 0
    total_weight = 0
    
    for name, pillar in pillars.items():
        try:
            logger.info(f"[ANALYZE] Running {name} pillar...")
            score = await pillar.analyze(symbol, data)
            
            # Convert score to 0-100 scale if needed
            raw_score = score.score if hasattr(score, 'score') else 0
            # Normalize to 0-100 (from -100 to +100)
            normalized_score = (raw_score + 100) / 2 if raw_score < 0 or raw_score > 100 else raw_score
            normalized_score = max(0, min(100, normalized_score))
            
            weight = pillar.weight
            weighted_score = normalized_score * weight
            
            results[name] = {
                'score': round(normalized_score, 1),
                'weight': round(weight * 100),
                'weighted': round(weighted_score, 1),
                'signal': score.signal.value if hasattr(score, 'signal') else 'neutral',
                'reasoning': score.reasoning if hasattr(score, 'reasoning') else '',
                'confidence': round(score.confidence * 100) if hasattr(score, 'confidence') else 50
            }
            
            total_score += weighted_score
            total_weight += weight
            
            logger.info(f"[ANALYZE] {name}: {normalized_score:.1f} (weight {weight})")
            
        except Exception as e:
            logger.error(f"[ANALYZE] {name} pillar failed: {e}")
            results[name] = {
                'score': 50,
                'weight': round(pillar.weight * 100),
                'weighted': 25,
                'signal': 'neutral',
                'reasoning': f'Error: {str(e)}',
                'confidence': 0
            }
            total_score += 50 * pillar.weight
            total_weight += pillar.weight
    
    # Calculate final score (0-100)
    final_score = total_score / total_weight if total_weight > 0 else 50
    final_score = max(0, min(100, final_score))
    
    # Determine decision
    if final_score >= 70:
        decision = 'STRONG_BUY'
    elif final_score >= 55:
        decision = 'BUY'
    elif final_score >= 45:
        decision = 'HOLD'
    elif final_score >= 30:
        decision = 'SELL'
    else:
        decision = 'STRONG_SELL'
    
    # Calculate confidence
    confidences = [r.get('confidence', 50) for r in results.values()]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 50
    
    # Generate summary
    bullish_pillars = [k for k, v in results.items() if v['score'] >= 60]
    bearish_pillars = [k for k, v in results.items() if v['score'] <= 40]
    
    if len(bullish_pillars) > len(bearish_pillars):
        summary = f"Bullish signals from {', '.join(bullish_pillars)}. "
    elif len(bearish_pillars) > len(bullish_pillars):
        summary = f"Bearish pressure detected. Consider reducing position. "
    else:
        summary = f"Mixed signals. Hold current position. "
    
    return {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'finalScore': round(final_score),
        'decision': decision,
        'confidence': round(avg_confidence),
        'summary': summary.strip(),
        'pillars': [
            {
                'name': name.capitalize(),
                'score': v['score'],
                'weight': v['weight'],
                'details': v.get('reasoning', '')[:100]
            }
            for name, v in results.items()
        ],
        'currentPrice': data.get('current_price', 0)
    }


# Test function
if __name__ == '__main__':
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    result = asyncio.run(analyze_symbol(symbol))
    import json
    print(json.dumps(result, indent=2))
