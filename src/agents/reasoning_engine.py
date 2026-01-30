"""
Reasoning Engine - Multi-pillar decision making orchestrator.

Combines 4 pillars for balanced decision making:
- Technical (25%): Chart patterns, indicators
- Fundamental (25%): Company financials
- Sentiment (25%): Social media sentiment
- News (25%): Recent news and events

Each pillar contributes a score from -100 to +100.
The weighted combination produces the final decision.
"""

import asyncio
import pandas as pd
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging
import json

from .pillars import (
    BasePillar,
    PillarScore,
    PillarSignal,
    TechnicalPillar,
    get_technical_pillar,
    FundamentalPillar,
    get_fundamental_pillar,
)

# V5 - Adaptive Learning (optional import)
try:
    from .adaptive_scorer import get_adaptive_scorer, AdaptiveScorer, FeatureVector
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False
    AdaptiveScorer = None
    FeatureVector = None

from .pillars import (
    SentimentPillar,
    get_sentiment_pillar,
    NewsPillar,
    get_news_pillar,
)
# V5.3 - ML Pillar
try:
    from src.agents.pillars.ml_pillar import MLPillar, get_ml_pillar
    ML_PILLAR_AVAILABLE = True
except ImportError:
    ML_PILLAR_AVAILABLE = False
    MLPillar = None


from src.intelligence.market_context import (
    MarketContext,
    MarketRegime,
    VolatilityRegime,
    get_market_context,
)

logger = logging.getLogger(__name__)


class DecisionType(str, Enum):
    """Final decision type - V4.8: Thresholds on display scale (0-100)"""
    STRONG_BUY = "strong_buy"       # Score > 70 (display scale)
    BUY = "buy"                      # Score 55-70
    HOLD = "hold"                    # Score 40-55
    SELL = "sell"                    # Score 25-40
    STRONG_SELL = "strong_sell"      # Score < 25


@dataclass
class ReasoningResult:
    """Complete reasoning result combining all pillars"""
    symbol: str
    timestamp: str

    # Pillar scores
    technical_score: PillarScore
    fundamental_score: PillarScore
    sentiment_score: PillarScore
    news_score: PillarScore

    # Combined result
    total_score: float              # V4.8: 0 to 100 (display scale, unified)
    decision: DecisionType
    confidence: float               # 0 to 1

    # Reasoning
    reasoning_summary: str
    key_factors: List[Dict[str, Any]]
    risk_factors: List[str]

    # Weights used
    weights_used: Dict[str, float]

    # Market context
    market_context: Optional[MarketContext] = None
    position_size_adjustment: float = 1.0

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'pillar_scores': {
                'technical': self.technical_score.to_dict(),
                'fundamental': self.fundamental_score.to_dict(),
                'sentiment': self.sentiment_score.to_dict(),
                'news': self.news_score.to_dict(),
            },
            'total_score': self.total_score,
            'decision': self.decision.value,
            'confidence': self.confidence,
            'reasoning_summary': self.reasoning_summary,
            'key_factors': self.key_factors,
            'risk_factors': self.risk_factors,
            'weights_used': self.weights_used,
            'market_context': self.market_context.to_dict() if self.market_context else None,
            'position_size_adjustment': self.position_size_adjustment
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ReasoningConfig:
    """Configuration for the reasoning engine"""
    # Pillar weights (must sum to 1.0) - used as defaults if adaptive learning disabled
    technical_weight: float = 0.30
    fundamental_weight: float = 0.30
    sentiment_weight: float = 0.15
    news_weight: float = 0.10
    ml_weight: float = 0.15  # V5.3 - ML Pillar

    # V4.8: Decision thresholds on DISPLAY scale (0-100)
    # This matches the UI display for user clarity
    # Old internal scale was -100 to +100, now unified to 0-100
    strong_buy_threshold: float = 70   # Was 50 internal → (50+100)/2 = 75, adjusted to 70
    buy_threshold: float = 55          # Was 25 internal → 62.5, adjusted to match UI threshold
    sell_threshold: float = 40         # Was -25 internal → 37.5, adjusted to 40
    strong_sell_threshold: float = 25  # Was -50 internal → 25

    # Minimum confidence to act
    min_confidence: float = 0.5

    # Parallel execution
    parallel_pillars: bool = True

    # Market context
    use_market_context: bool = True
    adjust_for_regime: bool = True  # Adjust weights based on market regime

    # V5 - Adaptive Learning
    use_adaptive_weights: bool = True   # Use learned weights from trade history
    record_for_learning: bool = True    # Record analyses for future learning

    def validate(self):
        """Validate configuration"""
        total = (self.technical_weight + self.fundamental_weight +
                 self.sentiment_weight + self.news_weight + self.ml_weight)
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Pillar weights must sum to 1.0, got {total}")


class ReasoningEngine:
    """
    Multi-pillar reasoning engine for trading decisions.

    Orchestrates 4 analysis pillars and combines their outputs
    into a unified decision with full reasoning trail.
    """

    def __init__(self, config: Optional[ReasoningConfig] = None):
        self.config = config or ReasoningConfig()
        self.config.validate()

        # Pillars will be initialized on first use
        self._technical: Optional[TechnicalPillar] = None
        self._fundamental: Optional[FundamentalPillar] = None
        self._sentiment: Optional[SentimentPillar] = None
        self._news: Optional[NewsPillar] = None

        # V5 - Adaptive Learning
        self._adaptive_scorer: Optional[AdaptiveScorer] = None
        if ADAPTIVE_AVAILABLE and self.config.use_adaptive_weights:
            try:
                self._adaptive_scorer = get_adaptive_scorer()
                logger.info(f"[REASONING] Adaptive scoring enabled. Current weights: {self._adaptive_scorer.weights.to_dict()}")
            except Exception as e:
                logger.warning(f"[REASONING] Adaptive scoring unavailable: {e}")

        self._initialized = False

    async def initialize(self):
        """Initialize all pillars"""
        if self._initialized:
            return

        logger.info("Initializing ReasoningEngine pillars...")

        # Technical and Fundamental are sync singletons
        self._technical = get_technical_pillar()
        self._fundamental = get_fundamental_pillar()

        # Sentiment and News are async
        self._sentiment = await get_sentiment_pillar()
        self._news = await get_news_pillar()
        # V5.3 - ML Pillar
        self._ml = None
        if ML_PILLAR_AVAILABLE:
            try:
                self._ml = get_ml_pillar()
                logger.info("[REASONING] ML Pillar initialized")
            except Exception as e:
                logger.warning(f"[REASONING] ML Pillar not available: {e}")


        self._initialized = True
        logger.info("ReasoningEngine initialized")

    async def close(self):
        """Close all async pillar connections to prevent leaks"""
        logger.info("Closing ReasoningEngine pillars...")

        # Close async pillars (sentiment and news have aiohttp sessions)
        if self._sentiment:
            try:
                await self._sentiment.close()
            except Exception as e:
                logger.warning(f"Error closing sentiment pillar: {e}")

        if self._news:
            try:
                await self._news.close()
            except Exception as e:
                logger.warning(f"Error closing news pillar: {e}")

        self._initialized = False
        logger.info("ReasoningEngine closed")

    async def analyze(
        self,
        symbol: str,
        df: Optional[pd.DataFrame] = None,
        fundamentals: Optional[Dict] = None,
        sentiment_data: Optional[Dict] = None,
        news_data: Optional[Dict] = None
    ) -> ReasoningResult:
        """
        Perform multi-pillar analysis on a symbol.

        Args:
            symbol: Stock symbol to analyze
            df: OHLCV DataFrame (will fetch if not provided)
            fundamentals: Pre-fetched fundamental data
            sentiment_data: Pre-fetched sentiment data
            news_data: Pre-fetched news data

        Returns:
            ReasoningResult with complete analysis
        """
        await self.initialize()

        timestamp = datetime.now().isoformat()
        logger.info(f"Analyzing {symbol} with multi-pillar reasoning")

        # Fetch market context if enabled
        market_context = None
        if self.config.use_market_context:
            try:
                market_context = await get_market_context()
                logger.info(f"Market context: {market_context.regime.value}, VIX={market_context.vix_level:.1f}")
            except Exception as e:
                logger.warning(f"Failed to get market context: {e}")

        # Prepare data for each pillar
        technical_data = {'df': df}
        fundamental_data = {'fundamentals': fundamentals}
        sentiment_input = sentiment_data or {}
        news_input = {'news': news_data}

        # Run pillars (parallel or sequential)
        if self.config.parallel_pillars:
            pillar_results = await asyncio.gather(
                self._technical.analyze(symbol, technical_data),
                self._fundamental.analyze(symbol, fundamental_data),
                self._sentiment.analyze(symbol, sentiment_input),
                self._news.analyze(symbol, news_input),
                return_exceptions=True
            )

            # Handle any exceptions
            scores = []
            for i, result in enumerate(pillar_results):
                if isinstance(result, Exception):
                    logger.error(f"Pillar {i} failed: {result}")
                    scores.append(self._create_error_score(i))
                else:
                    scores.append(result)

            technical_score, fundamental_score, sentiment_score, news_score = scores

            # V5.3 - ML Pillar (calculated after other pillars)
            ml_score = None
            if self._ml and ML_PILLAR_AVAILABLE:
                try:
                    ml_score = await self._ml.analyze(
                        symbol=symbol,
                        technical_score=technical_score.score,
                        fundamental_score=fundamental_score.score,
                        sentiment_score=sentiment_score.score,
                        news_score=news_score.score,
                        market_context={'vix': market_context.vix_level if market_context else 20}
                    )
                    logger.debug(f"[ML] Score for {symbol}: {ml_score.score:.1f}")
                except Exception as e:
                    logger.warning(f"[REASONING] ML Pillar error: {e}")


        else:
            # Sequential execution
            technical_score = await self._technical.analyze(symbol, technical_data)
            fundamental_score = await self._fundamental.analyze(symbol, fundamental_data)
            sentiment_score = await self._sentiment.analyze(symbol, sentiment_input)
            news_score = await self._news.analyze(symbol, news_input)

        # Calculate weighted total (adjust for market context if enabled)
        # V5: Use adaptive weights if available
        if self._adaptive_scorer:
            adaptive_weights = self._adaptive_scorer.weights
            weights = {
                'technical': adaptive_weights.technical,
                'fundamental': adaptive_weights.fundamental,
                'sentiment': adaptive_weights.sentiment,
                'news': adaptive_weights.news,
                'ml': self.config.ml_weight
            }
            logger.debug(f"[REASONING] Using adaptive weights: {weights}")
        else:
            weights = {
                'technical': self.config.technical_weight,
                'fundamental': self.config.fundamental_weight,
                'sentiment': self.config.sentiment_weight,
                'news': self.config.news_weight,
                'ml': self.config.ml_weight
            }

        # Adjust weights based on market regime
        if market_context and self.config.adjust_for_regime:
            weights = self._adjust_weights_for_context(weights, market_context)

        # Calculate raw weighted score on internal scale (-100 to +100)
        internal_score = (
            technical_score.weighted_score(weights['technical']) +
            fundamental_score.weighted_score(weights['fundamental']) +
            sentiment_score.weighted_score(weights['sentiment']) +
            news_score.weighted_score(weights['news'])
        )

        # V5.3 - Add ML score
        if ml_score and hasattr(ml_score, 'weighted_score'):
            internal_score += ml_score.weighted_score(weights.get('ml', 0.15))


        # Apply market context score adjustment
        position_size_adj = 1.0
        if market_context:
            # Reduce score in bear/crash markets, boost in bull
            regime_adj = self._get_regime_score_adjustment(market_context.regime)
            internal_score = internal_score * regime_adj
            position_size_adj = market_context.position_size_multiplier

        # V4.8: Convert to DISPLAY scale (0-100) for unified scoring
        # Formula: display_score = (internal_score + 100) / 2
        # This ensures consistency between ReasoningEngine and UI display
        total_score = (internal_score + 100) / 2

        logger.info(f"[REASONING] {symbol}: Internal={internal_score:+.1f} → Display={total_score:.1f}/100")

        # Determine decision (now comparing display scale against display thresholds)
        decision = self._get_decision(total_score)

        # Calculate overall confidence
        confidence = self._calculate_overall_confidence([
            technical_score, fundamental_score, sentiment_score, news_score
        ])

        # Generate reasoning
        reasoning_summary = self._generate_summary(
            symbol, total_score, decision,
            technical_score, fundamental_score, sentiment_score, news_score
        )

        # Extract key factors
        key_factors = self._extract_key_factors([
            technical_score, fundamental_score, sentiment_score, news_score
        ])

        # Identify risks
        risk_factors = self._identify_risks([
            technical_score, fundamental_score, sentiment_score, news_score
        ])

        # Add market context risks
        if market_context:
            risk_factors.extend(self._get_market_context_risks(market_context))

        return ReasoningResult(
            symbol=symbol,
            timestamp=timestamp,
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            sentiment_score=sentiment_score,
            news_score=news_score,
            total_score=total_score,
            decision=decision,
            confidence=confidence,
            reasoning_summary=reasoning_summary,
            key_factors=key_factors,
            risk_factors=risk_factors,
            weights_used=weights,
            market_context=market_context,
            position_size_adjustment=position_size_adj
        )

    def _create_error_score(self, pillar_index: int) -> PillarScore:
        """Create a neutral score for a failed pillar"""
        names = ['Technical', 'Fundamental', 'Sentiment', 'News']
        return PillarScore.from_score(
            pillar_name=names[pillar_index],
            score=0,
            reasoning="Pillar analysis failed",
            data_quality=0.0
        )

    def _get_decision(self, total_score: float) -> DecisionType:
        """Determine decision based on total score (display scale 0-100)

        V4.8: Uses >= for consistency with webapp.py UI thresholds.
        """
        if total_score >= self.config.strong_buy_threshold:
            return DecisionType.STRONG_BUY
        elif total_score >= self.config.buy_threshold:
            return DecisionType.BUY
        elif total_score >= self.config.sell_threshold:
            return DecisionType.HOLD
        elif total_score >= self.config.strong_sell_threshold:
            return DecisionType.SELL
        else:
            return DecisionType.STRONG_SELL

    def _calculate_overall_confidence(self, scores: List[PillarScore]) -> float:
        """Calculate overall confidence from pillar confidences"""
        # Weighted average of confidences, adjusted by data quality
        total_weight = 0
        weighted_confidence = 0

        for score in scores:
            weight = score.data_quality
            weighted_confidence += score.confidence * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        base_confidence = weighted_confidence / total_weight

        # Check for agreement between pillars
        directions = [1 if s.score > 0 else -1 if s.score < 0 else 0 for s in scores]
        positive = sum(1 for d in directions if d > 0)
        negative = sum(1 for d in directions if d < 0)

        # Boost confidence if pillars agree
        if positive >= 3 or negative >= 3:
            agreement_bonus = 0.1
        elif positive >= 2 or negative >= 2:
            agreement_bonus = 0.05
        else:
            agreement_bonus = 0

        return min(1.0, base_confidence + agreement_bonus)

    def _generate_summary(
        self,
        symbol: str,
        total_score: float,
        decision: DecisionType,
        technical: PillarScore,
        fundamental: PillarScore,
        sentiment: PillarScore,
        news: PillarScore
    ) -> str:
        """Generate human-readable reasoning summary"""
        parts = []

        # Header - V4.8: Use display scale format (0-100)
        decision_text = decision.value.replace('_', ' ').upper()
        parts.append(f"=== {symbol} ANALYSIS: {decision_text} (Score: {total_score:.1f}/100) ===\n")

        # Pillar summary table
        parts.append("PILLAR BREAKDOWN:")
        pillars = [
            ('Technical', technical),
            ('Fundamental', fundamental),
            ('Sentiment', sentiment),
            ('News', news)
        ]

        for name, score in pillars:
            signal = score.signal.value.replace('_', ' ')
            quality = f"[{score.data_quality:.0%}]" if score.data_quality < 1 else ""
            parts.append(f"  {name:12}: {score.score:+6.1f} ({signal}) {quality}")

        # Key reasoning from each pillar
        parts.append("\nKEY INSIGHTS:")
        for name, score in pillars:
            if score.score != 0 and score.data_quality > 0.3:
                # Get first line of reasoning
                first_line = score.reasoning.split('\n')[0]
                parts.append(f"  [{name}] {first_line}")

        # Decision rationale
        parts.append(f"\nDECISION RATIONALE:")
        if decision in [DecisionType.STRONG_BUY, DecisionType.BUY]:
            bullish_pillars = [name for name, s in pillars if s.score > 20]
            parts.append(f"  Bullish signals from: {', '.join(bullish_pillars) or 'weighted combination'}")
        elif decision in [DecisionType.STRONG_SELL, DecisionType.SELL]:
            bearish_pillars = [name for name, s in pillars if s.score < -20]
            parts.append(f"  Bearish signals from: {', '.join(bearish_pillars) or 'weighted combination'}")
        else:
            parts.append("  Mixed signals suggest caution")

        return "\n".join(parts)

    def _extract_key_factors(self, scores: List[PillarScore]) -> List[Dict[str, Any]]:
        """Extract the most impactful factors across all pillars"""
        all_factors = []

        for score in scores:
            for factor in score.factors:
                factor_copy = factor.copy()
                factor_copy['pillar'] = score.pillar_name
                all_factors.append(factor_copy)

        # Sort by absolute impact
        all_factors.sort(key=lambda x: abs(x.get('score', 0)), reverse=True)

        return all_factors[:8]  # Top 8 factors

    def _identify_risks(self, scores: List[PillarScore]) -> List[str]:
        """Identify risk factors from the analysis"""
        risks = []

        for score in scores:
            # Low data quality is a risk
            if score.data_quality < 0.5:
                risks.append(f"Low data quality for {score.pillar_name} ({score.data_quality:.0%})")

            # Conflicting signals within a pillar
            if score.confidence < 0.6:
                risks.append(f"Low confidence in {score.pillar_name} analysis")

            # Specific risk factors from pillar reasoning
            if 'volatility' in score.reasoning.lower() and 'high' in score.reasoning.lower():
                risks.append("High market volatility detected")
            if 'overvalued' in score.reasoning.lower():
                risks.append("Potential overvaluation risk")
            if 'declining' in score.reasoning.lower() or 'negative' in score.reasoning.lower():
                if score.pillar_name == 'Fundamental':
                    risks.append("Declining fundamentals")

        # Check for pillar disagreement
        bullish = sum(1 for s in scores if s.score > 20)
        bearish = sum(1 for s in scores if s.score < -20)
        if bullish > 0 and bearish > 0:
            risks.append("Mixed signals across pillars - conflicting indicators")

        return list(set(risks))[:5]  # Unique risks, max 5

    def adjust_weights(
        self,
        technical: float = None,
        fundamental: float = None,
        sentiment: float = None,
        news: float = None
    ):
        """
        Adjust pillar weights dynamically.

        All provided weights will be normalized to sum to 1.0.
        """
        weights = {
            'technical': technical if technical is not None else self.config.technical_weight,
            'fundamental': fundamental if fundamental is not None else self.config.fundamental_weight,
            'sentiment': sentiment if sentiment is not None else self.config.sentiment_weight,
            'news': news if news is not None else self.config.news_weight
        }

        total = sum(weights.values())
        if total == 0:
            raise ValueError("At least one weight must be non-zero")

        # Normalize
        self.config.technical_weight = weights['technical'] / total
        self.config.fundamental_weight = weights['fundamental'] / total
        self.config.sentiment_weight = weights['sentiment'] / total
        self.config.news_weight = weights['news'] / total

        logger.info(f"Weights adjusted: T={self.config.technical_weight:.2f}, "
                    f"F={self.config.fundamental_weight:.2f}, "
                    f"S={self.config.sentiment_weight:.2f}, "
                    f"N={self.config.news_weight:.2f}")

    def _adjust_weights_for_context(
        self,
        weights: Dict[str, float],
        context: MarketContext
    ) -> Dict[str, float]:
        """
        Adjust pillar weights based on market context.

        - In high volatility: increase technical weight
        - In bear markets: increase fundamental weight (quality focus)
        - In bull markets: increase sentiment weight (momentum)
        """
        adjusted = weights.copy()

        # Volatility-based adjustments
        if context.volatility in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
            # High volatility: more weight on technicals and fundamentals
            adjusted['technical'] *= 1.2
            adjusted['fundamental'] *= 1.1
            adjusted['sentiment'] *= 0.8
            adjusted['news'] *= 0.9

        # Regime-based adjustments
        if context.regime in [MarketRegime.BEAR, MarketRegime.STRONG_BEAR, MarketRegime.CRASH]:
            # Bear market: focus on fundamentals and reduce sentiment
            adjusted['fundamental'] *= 1.2
            adjusted['sentiment'] *= 0.7
        elif context.regime in [MarketRegime.BULL, MarketRegime.STRONG_BULL]:
            # Bull market: momentum matters more
            adjusted['sentiment'] *= 1.1
            adjusted['news'] *= 1.1

        # Normalize to sum to 1.0
        total = sum(adjusted.values())
        return {k: v / total for k, v in adjusted.items()}

    def _get_regime_score_adjustment(self, regime: MarketRegime) -> float:
        """
        Get score adjustment multiplier based on market regime.

        In bear markets, we apply a dampening factor to buy signals.
        """
        adjustments = {
            MarketRegime.STRONG_BULL: 1.1,   # Boost signals
            MarketRegime.BULL: 1.05,
            MarketRegime.RANGE: 1.0,
            MarketRegime.BEAR: 0.9,          # Dampen buy signals
            MarketRegime.STRONG_BEAR: 0.8,
            MarketRegime.CRASH: 0.6          # Strong dampening
        }
        return adjustments.get(regime, 1.0)

    def _get_market_context_risks(self, context: MarketContext) -> List[str]:
        """Extract risk factors from market context"""
        risks = []

        # Regime risks
        if context.regime == MarketRegime.CRASH:
            risks.append("MARKET CRASH - extreme caution required")
        elif context.regime == MarketRegime.STRONG_BEAR:
            risks.append("Strong bear market - consider defensive positions")
        elif context.regime == MarketRegime.BEAR:
            risks.append("Bear market environment")

        # Volatility risks
        if context.volatility == VolatilityRegime.EXTREME:
            risks.append(f"Extreme volatility (VIX={context.vix_level:.1f})")
        elif context.volatility == VolatilityRegime.HIGH:
            risks.append(f"High volatility environment (VIX={context.vix_level:.1f})")

        # Position sizing warning
        if context.position_size_multiplier < 0.7:
            risks.append(f"Reduced position sizing recommended ({context.position_size_multiplier:.0%})")

        return risks


# Singleton
_engine_instance: Optional[ReasoningEngine] = None


async def get_reasoning_engine(config: Optional[ReasoningConfig] = None) -> ReasoningEngine:
    """Get or create the ReasoningEngine singleton"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ReasoningEngine(config)
        await _engine_instance.initialize()
    return _engine_instance
