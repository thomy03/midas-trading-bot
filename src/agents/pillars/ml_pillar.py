"""
ML Pillar - Machine Learning scoring pillar (proper integration)

Returns a PillarScore like other pillars for seamless integration
into the reasoning engine's weighted scoring system.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Any

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from .base import BasePillar, PillarScore, PillarSignal

logger = logging.getLogger(__name__)


class MLPillar(BasePillar):
    """
    Machine Learning pillar for trade scoring.
    
    Uses Gradient Boosting to predict trade success probability
    based on features from other pillars and market context.
    """
    
    MODEL_PATH = Path("/app/data/models/ml_pillar_model.joblib")
    SCALER_PATH = Path("/app/data/models/ml_pillar_scaler.joblib")
    
    FEATURE_COLUMNS = [
        'technical_score', 'rsi', 'rsi_oversold', 'rsi_overbought',
        'macd_signal', 'ema_trend', 'atr_percent', 'volume_ratio',
        'price_vs_52w_high', 'price_vs_52w_low',
        'fundamental_score', 'pe_ratio_norm', 'revenue_growth',
        'profit_margin', 'debt_to_equity',
        'sentiment_score', 'twitter_sentiment', 'news_sentiment',
        'market_regime', 'vix_level', 'sector_momentum',
        'combined_score'
    ]
    
    def __init__(self, weight: float = 0.15):
        super().__init__(weight)
        self.model = None
        self.scaler = None
        self.is_trained = False
        self._load_model()
    
    def get_name(self) -> str:
        """Return pillar name"""
        return "ml"
    
    def _load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            if self.MODEL_PATH.exists() and self.SCALER_PATH.exists():
                self.model = joblib.load(self.MODEL_PATH)
                self.scaler = joblib.load(self.SCALER_PATH)
                self.is_trained = True
                logger.info(f"[ML Pillar] Model loaded successfully")
                return True
        except Exception as e:
            logger.warning(f"[ML Pillar] Could not load model: {e}")
        return False
    
    def _extract_features(
        self,
        technical_score: float,
        fundamental_score: float,
        sentiment_score: float,
        news_score: float,
        market_context: Dict[str, Any]
    ) -> np.ndarray:
        """Extract feature vector from pillar scores and context"""
        
        ctx = market_context or {}
        
        features = {
            'technical_score': technical_score / 100,
            'fundamental_score': fundamental_score / 100,
            'sentiment_score': sentiment_score / 100,
            'combined_score': (technical_score * 0.35 + fundamental_score * 0.35 + 
                             sentiment_score * 0.20 + news_score * 0.10) / 100,
            'rsi': ctx.get('rsi', 50) / 100,
            'rsi_oversold': 1 if ctx.get('rsi', 50) < 30 else 0,
            'rsi_overbought': 1 if ctx.get('rsi', 50) > 70 else 0,
            'macd_signal': ctx.get('macd_signal', 0),
            'ema_trend': ctx.get('ema_trend', 0),
            'atr_percent': min(ctx.get('atr_percent', 2) / 10, 1),
            'volume_ratio': min(ctx.get('volume_ratio', 1) / 3, 1),
            'price_vs_52w_high': ctx.get('price_vs_52w_high', 0.85),
            'price_vs_52w_low': ctx.get('price_vs_52w_low', 1.2),
            'pe_ratio_norm': min(ctx.get('pe_ratio', 20) / 50, 1),
            'revenue_growth': np.tanh(ctx.get('revenue_growth', 0) / 50),
            'profit_margin': np.tanh(ctx.get('profit_margin', 0) / 30),
            'debt_to_equity': min(ctx.get('debt_to_equity', 0.5) / 2, 1),
            'twitter_sentiment': ctx.get('twitter_sentiment', 0),
            'news_sentiment': (news_score - 50) / 50,
            'market_regime': ctx.get('market_regime', 0),
            'vix_level': min(ctx.get('vix', 20) / 40, 1),
            'sector_momentum': np.tanh(ctx.get('sector_momentum', 0) / 10),
        }
        
        vector = [features.get(col, 0) for col in self.FEATURE_COLUMNS]
        return np.array(vector).reshape(1, -1)
    
    async def analyze(
        self,
        symbol: str,
        data: Dict[str, Any] = None,
        technical_score: float = 50,
        fundamental_score: float = 50,
        sentiment_score: float = 50,
        news_score: float = 50,
        market_context: Dict[str, Any] = None,
        **kwargs
    ) -> PillarScore:
        """
        Analyze using ML model and return a PillarScore.
        """
        # If model not available, return neutral score
        if not self.is_trained or self.model is None:
            return PillarScore.from_score(
                pillar_name="ml",
                score=0,
                reasoning="ML model not available - using neutral score",
                factors=[{"factor": "model_status", "value": "not_loaded"}],
                confidence=0.0,
                data_quality=0.0
            )
        
        try:
            features = self._extract_features(
                technical_score, fundamental_score,
                sentiment_score, news_score, market_context
            )
            
            features_scaled = self.scaler.transform(features)
            proba = self.model.predict_proba(features_scaled)[0]
            win_probability = proba[1] if len(proba) > 1 else proba[0]
            
            # Convert probability to score (-100 to +100)
            score = (win_probability - 0.5) * 200
            confidence = min(abs(win_probability - 0.5) * 2, 1.0)
            
            factors = [
                {"factor": "win_probability", "value": f"{win_probability:.1%}", "impact": "primary"},
                {"factor": "model_confidence", "value": f"{confidence:.1%}", "impact": "secondary"},
            ]
            
            if win_probability > 0.65:
                reasoning = f"ML predicts HIGH success probability ({win_probability:.0%})"
            elif win_probability > 0.5:
                reasoning = f"ML predicts MODERATE success probability ({win_probability:.0%})"
            elif win_probability > 0.35:
                reasoning = f"ML predicts CAUTIOUS outlook ({win_probability:.0%})"
            else:
                reasoning = f"ML predicts LOW success probability ({win_probability:.0%})"
            
            return PillarScore.from_score(
                pillar_name="ml",
                score=score,
                reasoning=reasoning,
                factors=factors,
                confidence=confidence,
                data_quality=1.0
            )
            
        except Exception as e:
            logger.error(f"[ML Pillar] Prediction error for {symbol}: {e}")
            return PillarScore.from_score(
                pillar_name="ml",
                score=0,
                reasoning=f"ML prediction error: {str(e)}",
                factors=[{"factor": "error", "value": str(e)}],
                confidence=0.0,
                data_quality=0.0
            )


_ml_pillar: Optional[MLPillar] = None


def get_ml_pillar() -> MLPillar:
    """Get or create ML Pillar singleton"""
    global _ml_pillar
    if _ml_pillar is None:
        _ml_pillar = MLPillar()
    return _ml_pillar
