"""
Adaptive Scorer Patch for BULL Optimization
============================================
Modifications to integrate BULL regime optimizations into the scoring system.

This patch enhances the existing AdaptiveScorer with:
1. Dynamic regime-based weight adjustments
2. Trend strength integration
3. Momentum boosting in BULL markets
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .bull_optimizer import (
    MarketRegime,
    BullOptimizedScorer,
    BullRegimeConfig,
    calculate_trend_strength
)


@dataclass
class ScoringResult:
    """Result of scoring calculation"""
    total_score: float
    pillar_scores: Dict[str, float]
    weights_used: Dict[str, float]
    regime: str
    trend_strength: float
    recommendation: str
    confidence: float


class AdaptiveScorerBullPatch:
    """
    Patch for AdaptiveScorer to optimize BULL regime performance
    
    Key changes:
    - Dynamic weight adjustment based on regime
    - Trend strength integration in scoring
    - Momentum amplification in BULL
    - Improved thresholds for BULL markets
    """
    
    def __init__(self, config_path: str = "config/pillar_weights.json"):
        self.config = self._load_config(config_path)
        self.bull_scorer = BullOptimizedScorer()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load pillar weights configuration"""
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        else:
            # Default config
            return {
                "base_weights": {
                    "technical": 0.25,
                    "fundamental": 0.20,
                    "sentiment": 0.15,
                    "momentum": 0.20,
                    "trend": 0.20
                },
                "scoring_thresholds": {
                    "strong_buy": 0.75,
                    "buy": 0.60,
                    "hold": 0.40,
                    "sell": 0.30,
                    "strong_sell": 0.20
                }
            }
    
    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """
        Get pillar weights adjusted for market regime
        
        In BULL: More weight on momentum and trend
        In VOLATILE: More weight on technical
        In BEAR: More weight on fundamentals
        """
        regime_config = self.config.get("regime_adjustments", {}).get(regime, {})
        
        if regime_config:
            # Use configured weights for this regime
            weights = {
                "technical": regime_config.get("technical", 0.25),
                "fundamental": regime_config.get("fundamental", 0.20),
                "sentiment": regime_config.get("sentiment", 0.15),
                "momentum": regime_config.get("momentum", 0.20),
                "trend": regime_config.get("trend", 0.20)
            }
        else:
            # Fall back to programmatic calculation
            market_regime = MarketRegime(regime) if regime in [r.value for r in MarketRegime] else MarketRegime.SIDEWAYS
            weights = self.bull_scorer.get_regime_weights(market_regime)
        
        # Normalize to sum to 1
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
    
    def calculate_enhanced_score(
        self,
        pillar_scores: Dict[str, float],
        regime: str,
        trend_strength: float = 0.5,
        momentum_score: float = 0.5
    ) -> ScoringResult:
        """
        Calculate total score with BULL optimizations
        
        Args:
            pillar_scores: Dict with scores for each pillar (0-1)
            regime: Current market regime
            trend_strength: Trend strength indicator (0-1)
            momentum_score: Momentum indicator (0-1)
            
        Returns:
            ScoringResult with detailed scoring information
        """
        # Get regime-adjusted weights
        weights = self.get_regime_weights(regime)
        
        # BULL-specific boosts
        if regime == "BULL" and trend_strength > 0.6:
            # Amplify momentum contribution in strong BULL trends
            momentum_boost = 1.0 + (trend_strength - 0.6) * 0.5  # Up to 20% boost
            if "momentum" in pillar_scores:
                pillar_scores["momentum"] = min(1.0, pillar_scores["momentum"] * momentum_boost)
            
            # Amplify trend contribution
            if "trend" in pillar_scores:
                pillar_scores["trend"] = min(1.0, pillar_scores["trend"] * (1 + (trend_strength - 0.6) * 0.3))
        
        # Calculate weighted score
        total_score = 0.0
        for pillar, score in pillar_scores.items():
            weight = weights.get(pillar, 0)
            total_score += score * weight
        
        # Confidence based on pillar agreement
        scores = list(pillar_scores.values())
        confidence = 1.0 - np.std(scores) if scores else 0.5
        
        # Get recommendation
        recommendation = self._get_recommendation(total_score, regime, trend_strength)
        
        return ScoringResult(
            total_score=total_score,
            pillar_scores=pillar_scores.copy(),
            weights_used=weights,
            regime=regime,
            trend_strength=trend_strength,
            recommendation=recommendation,
            confidence=confidence
        )
    
    def _get_recommendation(
        self, 
        score: float, 
        regime: str, 
        trend_strength: float
    ) -> str:
        """
        Get trading recommendation based on score and regime
        
        In BULL with strong trends, be more aggressive on buys
        """
        thresholds = self.config.get("scoring_thresholds", {})
        
        # Adjust thresholds for BULL regime
        if regime == "BULL" and trend_strength > 0.65:
            # Lower the buy threshold slightly in strong BULL
            buy_threshold = thresholds.get("buy", 0.60) - 0.05
            strong_buy_threshold = thresholds.get("strong_buy", 0.75) - 0.05
        else:
            buy_threshold = thresholds.get("buy", 0.60)
            strong_buy_threshold = thresholds.get("strong_buy", 0.75)
        
        if score >= strong_buy_threshold:
            return "STRONG_BUY"
        elif score >= buy_threshold:
            return "BUY"
        elif score >= thresholds.get("hold", 0.40):
            return "HOLD"
        elif score >= thresholds.get("sell", 0.30):
            return "SELL"
        else:
            return "STRONG_SELL"
    
    def get_regime_parameters(self, regime: str) -> Dict[str, Any]:
        """
        Get trading parameters for current regime
        
        Returns parameters like target multiplier, stop ATR, max hold days
        """
        regime_config = self.config.get("regime_adjustments", {}).get(regime, {})
        params = regime_config.get("parameters", {})
        
        # Defaults
        defaults = {
            "target_multiplier": 1.0,
            "trailing_stop_atr": 2.0,
            "max_hold_days": 10,
            "pullback_tolerance": 0.02,
            "min_trend_strength": 0.5
        }
        
        return {**defaults, **params}
    
    def should_enter_trade(
        self,
        score: float,
        regime: str,
        trend_strength: float,
        current_positions: int,
        max_positions: int = 10
    ) -> Tuple[bool, str]:
        """
        Determine if we should enter a new trade
        
        In BULL regime, be more willing to enter on momentum
        
        Returns:
            Tuple[should_enter, reason]
        """
        # Position limit check
        if current_positions >= max_positions:
            return False, "Max positions reached"
        
        # Get regime parameters
        params = self.get_regime_parameters(regime)
        min_trend = params.get("min_trend_strength", 0.5)
        
        # Basic score threshold
        thresholds = self.config.get("scoring_thresholds", {})
        buy_threshold = thresholds.get("buy", 0.60)
        
        # BULL regime adjustments
        if regime == "BULL":
            if trend_strength >= 0.7 and score >= buy_threshold - 0.08:
                return True, "BULL: Strong trend, moderate score"
            if score >= buy_threshold:
                return True, "BULL: Score above threshold"
        else:
            if score >= buy_threshold and trend_strength >= min_trend:
                return True, "Score and trend criteria met"
        
        return False, f"Score {score:.2f} below threshold"


# =========================================================================
# INTEGRATION EXAMPLE
# =========================================================================
"""
To integrate with existing adaptive_scorer.py:

1. Import the patch:
   from .adaptive_scorer_patch import AdaptiveScorerBullPatch

2. In AdaptiveScorer.__init__, add:
   self.bull_patch = AdaptiveScorerBullPatch()

3. Modify the calculate_score method:
   
   def calculate_score(self, symbol: str, data: dict) -> ScoringResult:
       # Calculate individual pillar scores (existing logic)
       pillar_scores = {
           'technical': self.technical_pillar.score(data),
           'fundamental': self.fundamental_pillar.score(data),
           'sentiment': self.sentiment_pillar.score(data),
           'momentum': self.momentum_pillar.score(data),
           'trend': self.trend_pillar.score(data),
       }
       
       # Get trend strength
       trend_strength = self._calculate_trend_strength(data)
       
       # Use enhanced scoring
       result = self.bull_patch.calculate_enhanced_score(
           pillar_scores=pillar_scores,
           regime=self.current_regime,
           trend_strength=trend_strength,
           momentum_score=pillar_scores.get('momentum', 0.5)
       )
       
       return result

4. Use regime parameters for position management:
   params = self.bull_patch.get_regime_parameters(self.current_regime)
   target = entry_price * (1 + base_target_pct * params['target_multiplier'])
   atr_stop = params['trailing_stop_atr']
"""
