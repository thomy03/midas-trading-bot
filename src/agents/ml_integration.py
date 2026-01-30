"""
ML Integration for TradingBot - Adds ML validation to trade signals

This integrates the ML Pillar as a "meta-validator" that can veto trades
predicted to have low probability of success.
"""

import sys
import os
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')

import importlib.util

# Load ML Pillar directly
spec = importlib.util.spec_from_file_location("ml_pillar", "/app/src/agents/pillars/ml_pillar.py")
ml_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ml_module)

MLPillar = ml_module.MLPillar
MLPrediction = ml_module.MLPrediction

# Singleton
_ml_pillar = None

def get_ml_validator():
    """Get ML validator singleton"""
    global _ml_pillar
    if _ml_pillar is None:
        _ml_pillar = MLPillar()
    return _ml_pillar


def validate_trade_ml(
    symbol: str,
    technical_score: float,
    fundamental_score: float,
    sentiment_score: float,
    news_score: float,
    combined_score: float,
    rsi: float = 50,
    volume_ratio: float = 1.0,
    market_regime: str = 'neutral',
    vix: float = 20
) -> dict:
    """
    Validate a potential trade using ML.
    
    Returns:
        dict with:
        - ml_score: 0-100 ML confidence score
        - ml_probability: 0-1 probability of winning trade
        - ml_veto: True if ML recommends NOT trading
        - ml_confidence: 'high', 'medium', 'low'
    """
    ml = get_ml_validator()
    
    # Build data dicts for ML pillar
    technical_data = {
        'score': technical_score,
        'rsi': rsi,
        'volume_ratio': volume_ratio,
        'above_ema': True,  # Assume bullish if signal generated
        'atr_percent': 2,
        'price_vs_52w_high': 0.85,
        'price_vs_52w_low': 1.3,
        'macd_signal': 'bullish'
    }
    
    fundamental_data = {
        'score': fundamental_score,
        'pe_ratio': 20,
        'revenue_growth': 15,
        'profit_margin': 12,
        'debt_to_equity': 0.5
    }
    
    sentiment_data = {
        'score': sentiment_score,
        'twitter': (sentiment_score - 50) / 50,  # Convert to -1 to 1
        'news': (news_score - 50) / 50
    }
    
    regime_map = {'bull': 1, 'bullish': 1, 'neutral': 0, 'bear': -1, 'bearish': -1, 'mixed': 0}
    market_data = {
        'regime': market_regime,
        'vix': vix,
        'sector_momentum': 5
    }
    
    # Get ML prediction
    prediction = ml.predict(
        symbol=symbol,
        technical_data=technical_data,
        fundamental_data=fundamental_data,
        sentiment_data=sentiment_data,
        market_data=market_data,
        combined_score=combined_score
    )
    
    return {
        'ml_score': prediction.score,
        'ml_probability': prediction.probability,
        'ml_veto': prediction.veto,
        'ml_confidence': prediction.confidence,
        'ml_model_version': prediction.model_version
    }


if __name__ == '__main__':
    # Test
    print("Testing ML Integration...")
    
    result = validate_trade_ml(
        symbol='NVDA',
        technical_score=75,
        fundamental_score=80,
        sentiment_score=70,
        news_score=65,
        combined_score=73,
        rsi=35,
        volume_ratio=1.5,
        market_regime='bullish',
        vix=18
    )
    
    print(f"\nTest NVDA (bullish signals):")
    print(f"  ML Score: {result['ml_score']:.1f}/100")
    print(f"  Win Probability: {result['ml_probability']*100:.1f}%")
    print(f"  ML Veto: {result['ml_veto']}")
    print(f"  Confidence: {result['ml_confidence']}")
    
    # Test bearish case
    result2 = validate_trade_ml(
        symbol='XYZ',
        technical_score=40,
        fundamental_score=35,
        sentiment_score=30,
        news_score=25,
        combined_score=35,
        rsi=75,
        volume_ratio=0.5,
        market_regime='bearish',
        vix=35
    )
    
    print(f"\nTest XYZ (bearish signals):")
    print(f"  ML Score: {result2['ml_score']:.1f}/100")
    print(f"  Win Probability: {result2['ml_probability']*100:.1f}%")
    print(f"  ML Veto: {result2['ml_veto']}")
    print(f"  Confidence: {result2['ml_confidence']}")
