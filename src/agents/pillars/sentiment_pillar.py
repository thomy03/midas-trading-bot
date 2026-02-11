"""
Sentiment Pillar - DISABLED

Disabled - orchestrator handles sentiment via grok_scanner.
This pillar always returns a neutral score (50).
Weight is set to 0.00 in the scoring config.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .base import BasePillar, PillarScore

logger = logging.getLogger(__name__)


class SentimentPillar(BasePillar):
    """
    Sentiment analysis pillar - DISABLED.
    
    Disabled - orchestrator handles sentiment via grok_scanner.
    Returns neutral score to maintain pipeline compatibility.
    """

    def __init__(self, weight: float = 0.00, **kwargs):
        super().__init__(weight)

    def get_name(self) -> str:
        return "Sentiment"

    async def initialize(self):
        """No-op: pillar disabled."""
        pass

    async def close(self):
        """No-op: pillar disabled."""
        pass

    async def analyze(self, symbol: str, data: Dict[str, Any]) -> PillarScore:
        """
        Returns neutral score. Sentiment is handled by orchestrator via grok_scanner.
        """
        logger.debug(f"[SENTIMENT] {symbol}: Pillar disabled, returning neutral (50)")
        return self._create_score(
            score=50.0,
            reasoning=f"Sentiment pillar disabled for {symbol} - handled by orchestrator via grok_scanner",
            factors=[],
            confidence=0.5,
            data_quality=0.5
        )


# Singleton
_sentiment_pillar: Optional[SentimentPillar] = None


async def get_sentiment_pillar() -> SentimentPillar:
    """Get or create the SentimentPillar singleton"""
    global _sentiment_pillar
    if _sentiment_pillar is None:
        _sentiment_pillar = SentimentPillar()
        await _sentiment_pillar.initialize()
    return _sentiment_pillar
