"""
Market Intelligence Package
Intégration LLM pour l'analyse de marché et la veille sectorielle.
"""

from .market_intelligence import (
    MarketIntelligence,
    OpenRouterClient,
    IntelligenceResult,
    SectorIntelligence,
    IntelligenceType,
    get_market_intelligence,
)

from .news_fetcher import (
    NewsFetcher,
    NewsArticle,
    ResearchPaper,
    NewsSource,
    get_news_fetcher,
)

from .trend_discovery import (
    TrendDiscovery,
    TrendDiscoveryScheduler,
    EmergingTrend,
    TrendReport,
    TrendStrength,
    TrendType,
    get_trend_discovery,
)

from .heat_detector import (
    HeatDetector,
    HeatEvent,
    SymbolHeat,
    HeatSnapshot,
    HeatConfig,
    get_heat_detector,
)

from .attention_manager import (
    AttentionManager,
    FocusTopic,
    AttentionBudget,
    AttentionConfig,
    get_attention_manager,
)

from .signal_correlator import (
    SignalCorrelator,
    SignalRecord,
    SourceWeight,
    CorrelationInsight,
    get_signal_correlator,
)

__all__ = [
    # Market Intelligence
    'MarketIntelligence',
    'OpenRouterClient',
    'IntelligenceResult',
    'SectorIntelligence',
    'IntelligenceType',
    'get_market_intelligence',
    # News Fetcher
    'NewsFetcher',
    'NewsArticle',
    'ResearchPaper',
    'NewsSource',
    'get_news_fetcher',
    # Trend Discovery
    'TrendDiscovery',
    'TrendDiscoveryScheduler',
    'EmergingTrend',
    'TrendReport',
    'TrendStrength',
    'TrendType',
    'get_trend_discovery',
    # Heat Detector
    'HeatDetector',
    'HeatEvent',
    'SymbolHeat',
    'HeatSnapshot',
    'HeatConfig',
    'get_heat_detector',
    # Attention Manager
    'AttentionManager',
    'FocusTopic',
    'AttentionBudget',
    'AttentionConfig',
    'get_attention_manager',
    # Signal Correlator
    'SignalCorrelator',
    'SignalRecord',
    'SourceWeight',
    'CorrelationInsight',
    'get_signal_correlator',
]

# V6 - Discovery Mode (Bull regime pépites)
try:
    from .discovery_mode import DiscoveryMode, get_discovery_mode, DiscoveryConfig
    DISCOVERY_MODE_AVAILABLE = True
except ImportError:
    DISCOVERY_MODE_AVAILABLE = False
    DiscoveryMode = None
