"""
News Pillar - DISABLED

Disabled - orchestrator handles news reasoning.
This pillar always returns a neutral score (50).
Weight is set to 0.00 in the scoring config.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .base import BasePillar, PillarScore

logger = logging.getLogger(__name__)


class NewsPillar(BasePillar):
    """
    News analysis pillar - DISABLED.
    
    Disabled - orchestrator handles news reasoning.
    Returns neutral score to maintain pipeline compatibility.
    """

    def __init__(self, weight: float = 0.00, **kwargs):
        super().__init__(weight)

    def get_name(self) -> str:
        return "News"

    async def initialize(self):
        """No-op: pillar disabled."""
        pass

    async def close(self):
        """No-op: pillar disabled."""
        pass

    async def analyze(self, symbol: str, data: Dict[str, Any]) -> PillarScore:
        """
        Returns neutral score. News reasoning is handled by orchestrator.
        """
        logger.debug(f"[NEWS] {symbol}: Pillar disabled, returning neutral (50)")
        return self._create_score(
            score=50.0,
            reasoning=f"News pillar disabled for {symbol} - handled by orchestrator",
            factors=[],
            confidence=0.5,
            data_quality=0.5
        )


# Singleton
_news_pillar: Optional[NewsPillar] = None


async def get_news_pillar() -> NewsPillar:
    """Get or create the NewsPillar singleton"""
    global _news_pillar
    if _news_pillar is None:
        _news_pillar = NewsPillar()
        await _news_pillar.initialize()
    return _news_pillar
