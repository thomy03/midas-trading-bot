"""
Pillars Module - Les 4 piliers de décision du trading bot.

Chaque pilier contribue 25% au score final:
- Technical: Analyse technique via indicateurs
- Fundamental: Analyse fondamentale (P/E, croissance, etc.)
- Sentiment: Sentiment social (StockTwits, Reddit, X)
- News: Actualités et événements récents
"""

from .base import BasePillar, PillarScore, PillarSignal
from .adaptive_technical_pillar import AdaptiveTechnicalPillar as TechnicalPillar, get_adaptive_technical_pillar as get_technical_pillar
from .fundamental_pillar import FundamentalPillar, get_fundamental_pillar
from .sentiment_pillar import SentimentPillar, get_sentiment_pillar
from .news_pillar import NewsPillar, get_news_pillar

__all__ = [
    # Base
    'BasePillar',
    'PillarScore',
    'PillarSignal',

    # Pillars
    'TechnicalPillar',
    'get_technical_pillar',
    'FundamentalPillar',
    'get_fundamental_pillar',
    'SentimentPillar',
    'get_sentiment_pillar',
    'NewsPillar',
    'get_news_pillar',
]
