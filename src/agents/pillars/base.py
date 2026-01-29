"""
Base Pillar - Abstract base class for all decision pillars.

Each pillar:
- Analyzes a specific aspect of the market
- Returns a score from -100 to +100
- Provides reasoning for the score
- Has a weight (default 25%)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PillarSignal(str, Enum):
    """Signal type from a pillar"""
    STRONG_BUY = "strong_buy"      # Score > 60
    BUY = "buy"                     # Score 30-60
    NEUTRAL = "neutral"             # Score -30 to 30
    SELL = "sell"                   # Score -60 to -30
    STRONG_SELL = "strong_sell"     # Score < -60


@dataclass
class PillarScore:
    """Result from a pillar analysis"""
    pillar_name: str
    score: float                    # -100 to +100
    signal: PillarSignal
    confidence: float               # 0.0 to 1.0

    # Reasoning
    reasoning: str                  # Human-readable explanation
    factors: List[Dict[str, Any]]   # Individual contributing factors

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data_quality: float = 1.0       # 0.0 to 1.0, reduces weight if data incomplete

    @classmethod
    def from_score(cls, pillar_name: str, score: float, reasoning: str,
                   factors: List[Dict[str, Any]] = None,
                   confidence: float = 0.8,
                   data_quality: float = 1.0) -> 'PillarScore':
        """Create a PillarScore from a raw score"""
        # Clamp score
        score = max(-100, min(100, score))

        # Determine signal
        if score > 60:
            signal = PillarSignal.STRONG_BUY
        elif score > 30:
            signal = PillarSignal.BUY
        elif score > -30:
            signal = PillarSignal.NEUTRAL
        elif score > -60:
            signal = PillarSignal.SELL
        else:
            signal = PillarSignal.STRONG_SELL

        return cls(
            pillar_name=pillar_name,
            score=score,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            factors=factors or [],
            data_quality=data_quality
        )

    def weighted_score(self, weight: float = 0.25) -> float:
        """Get the weighted score (accounting for data quality)"""
        return self.score * weight * self.data_quality

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pillar_name': self.pillar_name,
            'score': self.score,
            'signal': self.signal.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'factors': self.factors,
            'timestamp': self.timestamp,
            'data_quality': self.data_quality
        }


class BasePillar(ABC):
    """
    Abstract base class for all decision pillars.

    Each pillar must implement:
    - analyze(): Perform analysis and return a PillarScore
    - get_name(): Return the pillar name
    """

    def __init__(self, weight: float = 0.25):
        """
        Initialize the pillar.

        Args:
            weight: The weight of this pillar (default 25%)
        """
        self.weight = weight
        self._cache: Dict[str, PillarScore] = {}
        self._cache_ttl = 300  # 5 minutes default

    @abstractmethod
    def get_name(self) -> str:
        """Return the pillar name"""
        pass

    @abstractmethod
    async def analyze(
        self,
        symbol: str,
        data: Dict[str, Any]
    ) -> PillarScore:
        """
        Analyze a symbol and return a score.

        Args:
            symbol: The stock symbol to analyze
            data: Additional data (OHLCV, news, etc.)

        Returns:
            PillarScore with the analysis result
        """
        pass

    async def analyze_cached(
        self,
        symbol: str,
        data: Dict[str, Any],
        force_refresh: bool = False
    ) -> PillarScore:
        """
        Analyze with caching.

        Args:
            symbol: The stock symbol
            data: Additional data
            force_refresh: If True, bypass cache

        Returns:
            PillarScore (from cache if valid)
        """
        cache_key = f"{self.get_name()}:{symbol}"

        if not force_refresh and cache_key in self._cache:
            cached = self._cache[cache_key]
            cached_time = datetime.fromisoformat(cached.timestamp)
            age_seconds = (datetime.now() - cached_time).total_seconds()

            if age_seconds < self._cache_ttl:
                logger.debug(f"Using cached {self.get_name()} for {symbol}")
                return cached

        # Perform fresh analysis
        result = await self.analyze(symbol, data)
        self._cache[cache_key] = result
        return result

    def clear_cache(self, symbol: str = None):
        """Clear cache for a symbol or all"""
        if symbol:
            cache_key = f"{self.get_name()}:{symbol}"
            self._cache.pop(cache_key, None)
        else:
            self._cache.clear()

    def _create_score(
        self,
        score: float,
        reasoning: str,
        factors: List[Dict[str, Any]] = None,
        confidence: float = 0.8,
        data_quality: float = 1.0
    ) -> PillarScore:
        """Helper to create a PillarScore"""
        return PillarScore.from_score(
            pillar_name=self.get_name(),
            score=score,
            reasoning=reasoning,
            factors=factors,
            confidence=confidence,
            data_quality=data_quality
        )
