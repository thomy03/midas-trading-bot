"""
Dependency Injection Container for TradingBot Pillars

This module provides a simple Factory-based DI container for managing
the core components ("Pillars") of the trading system.

ARCHITECTURE:
------------
The trading system is built around 4 pillars:
1. DATA: Market data fetching (market_data_fetcher)
2. ANALYSIS: Technical analysis (ema_analyzer, rsi_analyzer)
3. STORAGE: Database operations (db_manager)
4. SCREENING: Signal generation (market_screener)

USAGE:
------
    from src.core.container import get_container

    # Get singleton container
    container = get_container()

    # Access components
    data = container.market_data
    screener = container.screener

    # Or with custom config
    container = Container(config={'capital': 50000})

BENEFITS:
---------
- Centralized dependency management
- Easy testing with mock injection
- Lazy initialization (components created on first access)
- Configuration override support
- Thread-safe singleton pattern

Author: Midas Architecture Agent
Created: 2025-02-04
"""

import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ContainerConfig:
    """Configuration for the DI container"""
    # Capital settings
    capital: float = 100000.0
    portfolio_file: str = 'data/portfolio.json'
    
    # Screener settings
    use_enhanced_detector: bool = True
    precision_mode: str = 'medium'
    
    # Database settings
    database_path: Optional[str] = None  # Uses default from settings if None
    
    # Max workers for parallel processing
    max_workers: int = 10


class Container:
    """
    Dependency Injection Container for TradingBot
    
    Provides lazy-loaded, cached access to core system components.
    Thread-safe with locking for initialization.
    
    Example:
        container = Container()
        screener = container.screener  # Lazy init on first access
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize container with optional config override
        
        Args:
            config: Optional dict to override default settings
        """
        self._config = ContainerConfig(**(config or {}))
        self._lock = threading.Lock()
        
        # Cached instances (lazy-loaded)
        self._market_data = None
        self._ema_analyzer = None
        self._db_manager = None
        self._screener = None
        self._news_fetcher = None
        self._trend_discovery = None
        self._position_sizer = None
        self._confidence_scorer = None
    
    # ==========================================
    # PILLAR 1: DATA
    # ==========================================
    
    @property
    def market_data(self):
        """Get market data fetcher (lazy-loaded)"""
        if self._market_data is None:
            with self._lock:
                if self._market_data is None:
                    from src.data.market_data import market_data_fetcher
                    self._market_data = market_data_fetcher
        return self._market_data
    
    # ==========================================
    # PILLAR 2: ANALYSIS
    # ==========================================
    
    @property
    def ema_analyzer(self):
        """Get EMA analyzer (lazy-loaded)"""
        if self._ema_analyzer is None:
            with self._lock:
                if self._ema_analyzer is None:
                    from src.indicators.ema_analyzer import ema_analyzer
                    self._ema_analyzer = ema_analyzer
        return self._ema_analyzer
    
    @property
    def news_fetcher(self):
        """Get news fetcher (lazy-loaded)"""
        if self._news_fetcher is None:
            with self._lock:
                if self._news_fetcher is None:
                    from src.intelligence import get_news_fetcher
                    self._news_fetcher = get_news_fetcher()
        return self._news_fetcher
    
    @property
    def trend_discovery(self):
        """Get trend discovery engine (lazy-loaded)"""
        if self._trend_discovery is None:
            with self._lock:
                if self._trend_discovery is None:
                    from src.intelligence.trend_discovery import get_trend_discovery
                    self._trend_discovery = get_trend_discovery()
        return self._trend_discovery
    
    # ==========================================
    # PILLAR 3: STORAGE
    # ==========================================
    
    @property
    def db_manager(self):
        """Get database manager (lazy-loaded)"""
        if self._db_manager is None:
            with self._lock:
                if self._db_manager is None:
                    from src.database.db_manager import db_manager
                    self._db_manager = db_manager
        return self._db_manager
    
    # ==========================================
    # PILLAR 4: SCREENING
    # ==========================================
    
    @property
    def screener(self):
        """Get market screener (lazy-loaded with config)"""
        if self._screener is None:
            with self._lock:
                if self._screener is None:
                    from src.screening.screener import MarketScreener
                    self._screener = MarketScreener(
                        use_enhanced_detector=self._config.use_enhanced_detector,
                        precision_mode=self._config.precision_mode,
                        total_capital=self._config.capital,
                        portfolio_file=self._config.portfolio_file
                    )
        return self._screener
    
    # ==========================================
    # UTILS
    # ==========================================
    
    @property
    def position_sizer(self):
        """Get position sizer (from screener)"""
        return self.screener.position_sizer
    
    @property
    def confidence_scorer(self):
        """Get confidence scorer (lazy-loaded)"""
        if self._confidence_scorer is None:
            with self._lock:
                if self._confidence_scorer is None:
                    from src.utils.confidence_scorer import confidence_scorer
                    self._confidence_scorer = confidence_scorer
        return self._confidence_scorer
    
    # ==========================================
    # FACTORY METHODS
    # ==========================================
    
    def create_screener(
        self,
        use_enhanced: bool = True,
        precision: str = 'medium',
        capital: float = None
    ):
        """
        Factory: Create a new screener instance with custom config
        
        Use this when you need a screener with different settings
        than the default singleton.
        
        Args:
            use_enhanced: Use enhanced RSI detector
            precision: 'high', 'medium', or 'low'
            capital: Trading capital (uses container default if None)
        
        Returns:
            New MarketScreener instance
        """
        from src.screening.screener import MarketScreener
        return MarketScreener(
            use_enhanced_detector=use_enhanced,
            precision_mode=precision,
            total_capital=capital or self._config.capital,
            portfolio_file=self._config.portfolio_file
        )
    
    def create_news_fetcher(self, provider: str = 'finnhub'):
        """
        Factory: Create a new news fetcher with specific provider
        
        Args:
            provider: News provider name
        
        Returns:
            NewsFetcher instance
        """
        from src.intelligence import NewsFetcher
        return NewsFetcher(provider=provider)
    
    # ==========================================
    # CONFIG & LIFECYCLE
    # ==========================================
    
    @property
    def config(self) -> ContainerConfig:
        """Get current configuration"""
        return self._config
    
    def update_capital(self, new_capital: float):
        """
        Update trading capital
        
        Note: This affects the screener's position sizer.
        
        Args:
            new_capital: New capital amount
        """
        self._config.capital = new_capital
        if self._screener is not None:
            self._screener.update_capital(new_capital)
    
    def reset(self):
        """
        Reset all cached instances
        
        Use this for testing or when config changes require
        full reinitialization.
        """
        with self._lock:
            self._market_data = None
            self._ema_analyzer = None
            self._db_manager = None
            self._screener = None
            self._news_fetcher = None
            self._trend_discovery = None
            self._position_sizer = None
            self._confidence_scorer = None


# ==========================================
# SINGLETON INSTANCE
# ==========================================

_container_instance: Optional[Container] = None
_container_lock = threading.Lock()


def get_container(config: Optional[Dict[str, Any]] = None) -> Container:
    """
    Get the global container singleton
    
    Args:
        config: Optional config override (only used on first call)
    
    Returns:
        Container singleton instance
    
    Example:
        container = get_container()
        screener = container.screener
    """
    global _container_instance
    
    if _container_instance is None:
        with _container_lock:
            if _container_instance is None:
                _container_instance = Container(config)
    
    return _container_instance


def reset_container():
    """Reset the global container (for testing)"""
    global _container_instance
    with _container_lock:
        if _container_instance is not None:
            _container_instance.reset()
        _container_instance = None
