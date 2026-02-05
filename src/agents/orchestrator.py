"""
Market Agent Orchestrator - Cerveau Central du Robot de Trading V4.1
Coordonne tous les modules: Discovery, Reasoning, Decision, Execution, Learning.

V4.1 Changes:
- ReasoningEngine (4 Pillars) instead of V3 screener
- TradeMemory (RAG) for learning from past trades
- Dual LLM: Gemini for intelligence, Grok for sentiment only
"""

import os
import sys
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

# Ajouter le chemin racine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.agents.state import StateManager, AgentPhase, MarketRegime, Position, TradeRecord, get_state_manager
from src.agents.guardrails import TradingGuardrails, TradeRequest, ValidationResult, TradeValidation, get_guardrails
from src.intelligence.analysis_store import AnalysisStore, get_analysis_store
from src.data.market_data import MarketDataFetcher

# V5.1 - Adaptive Position Manager & Portfolio Rotation
from src.execution.position_manager import AdaptivePositionManager, get_position_manager
from src.execution.portfolio_rotation import PortfolioRotationManager, get_rotation_manager

# V4.5 - Hybrid Screening (FMP pre-screening + IBKR validation)
try:
    from src.screening.hybrid_screener import HybridScreener, HybridScreenerConfig, get_hybrid_screener
    HYBRID_SCREENER_AVAILABLE = True
except ImportError:
    HYBRID_SCREENER_AVAILABLE = False
    HybridScreener = None

# V4.5 - IBKR Watchdog for automatic reconnection
try:
    from src.execution.ibkr_watchdog import IBKRWatchdog, WatchdogConfig
    IBKR_WATCHDOG_AVAILABLE = True
except ImportError:
    IBKR_WATCHDOG_AVAILABLE = False
    IBKRWatchdog = None

# V4.5 - Async Position Sizer with EUR/USD conversion
try:
    from src.utils.position_sizing import AsyncPositionSizer
    ASYNC_POSITION_SIZER_AVAILABLE = True
except ImportError:
    ASYNC_POSITION_SIZER_AVAILABLE = False
    AsyncPositionSizer = None

# V5 - Adaptive Learning
try:
    from src.agents.adaptive_scorer import get_adaptive_scorer, AdaptiveScorer, FeatureVector
    ADAPTIVE_LEARNING_AVAILABLE = True
except ImportError:
    ADAPTIVE_LEARNING_AVAILABLE = False
    AdaptiveScorer = None

# V5.1 - Shadow Tracking (Apprentissage Autonome)
try:
    from src.intelligence.shadow_tracker import ShadowTracker, get_shadow_tracker
    SHADOW_TRACKER_AVAILABLE = True
except ImportError:
    SHADOW_TRACKER_AVAILABLE = False
    ShadowTracker = None

# V5.2 - Indicator Discovery (Auto-découverte des indicateurs)
try:
    from src.intelligence.indicator_discovery import IndicatorDiscovery, get_indicator_discovery
    INDICATOR_DISCOVERY_AVAILABLE = True
except ImportError:
    INDICATOR_DISCOVERY_AVAILABLE = False
    IndicatorDiscovery = None

# V5.2 - Intuitive Reasoning (Raisonnement News → Secteur → Actions)
try:
    from src.intelligence.intuitive_reasoning import IntuitiveReasoning, get_intuitive_reasoning
    INTUITIVE_REASONING_AVAILABLE = True
except ImportError:
    INTUITIVE_REASONING_AVAILABLE = False
    IntuitiveReasoning = None

# V5.3 - ML Pillar for trade validation
try:
    from src.agents.ml_integration import validate_trade_ml, get_ml_validator
    ML_VALIDATOR_AVAILABLE = True
except ImportError:
    ML_VALIDATOR_AVAILABLE = False

# V6 - Discovery Mode (Bull regime pépites)
try:
    from src.intelligence.discovery_mode import DiscoveryMode, get_discovery_mode, DiscoveryConfig
    DISCOVERY_MODE_AVAILABLE = True
except ImportError:
    DISCOVERY_MODE_AVAILABLE = False
    DiscoveryMode = None


logger = logging.getLogger(__name__)


# =========================================================================
# CONFIGURATION
# =========================================================================

@dataclass
class OrchestratorConfig:
    """Configuration de l'orchestrateur"""
    # Capital
    initial_capital: float = 1500.0

    # Horaires (Eastern Time - NYSE)
    discovery_time: time = time(6, 0)      # 06:00 ET
    analysis_time: time = time(7, 0)       # 07:00 ET
    market_open: time = time(9, 30)        # 09:30 ET
    market_close: time = time(16, 0)       # 16:00 ET
    audit_time: time = time(20, 0)         # 20:00 ET

    # Limites
    max_focus_symbols: int = 100  # Augmenté de 30 à 100
    max_watchlist_size: int = 500  # Augmenté de 200 à 500
    scan_interval_minutes: int = 15        # Scan toutes les 15 min pendant le trading

    # LLM - Intelligence (Gemini 3 Flash via OpenRouter)
    llm_model: str = ""  # Read from OPENROUTER_MODEL env var
    openrouter_api_key: Optional[str] = None

    # Grok - Sentiment only (X/Twitter access)
    grok_api_key: Optional[str] = None

    # IBKR
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497                  # Paper: 7497, Live: 7496
    ibkr_client_id: int = 1

    # Données
    data_dir: str = "data"

    # RAG / Trade Memory
    memory_enabled: bool = True
    memory_db_path: str = "data/vector_store"
    similar_trades_k: int = 5              # Number of similar trades to retrieve


class MarketStatus(Enum):
    """Statut du marché"""
    PRE_MARKET = "pre_market"
    OPEN = "open"
    CLOSED = "closed"
    AFTER_HOURS = "after_hours"


# =========================================================================
# ORCHESTRATOR
# =========================================================================

class MarketAgent:
    """
    Agent principal de trading.
    Coordonne Discovery, Screening, Decision et Execution.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialise l'agent.

        Args:
            config: Configuration de l'orchestrateur
        """
        self.config = config or OrchestratorConfig()

        # Read LLM config from environment if not set
        if not self.config.llm_model:
            self.config.llm_model = os.getenv('OPENROUTER_MODEL', 'google/gemini-3-flash-preview')
        if not self.config.openrouter_api_key:
            self.config.openrouter_api_key = os.getenv('OPENROUTER_API_KEY', '')
        if not self.config.grok_api_key:
            self.config.grok_api_key = os.getenv('GROK_API_KEY', '')

        # Composants core
        self.state_manager = get_state_manager(self.config.data_dir)
        self.guardrails = get_guardrails(self.config.initial_capital)

        # V4.1 - ReasoningEngine (4 Pillars)
        self.reasoning_engine = None

        # V4.1 - RAG / Trade Memory
        self.trade_memory = None
        self.pattern_extractor = None

        # V4.1 - Strategy Composer (LLM-powered strategy creation)
        self.strategy_composer = None

        # V4.2 - Narrative Generator (human-readable reports)
        self.narrative_generator = None

        # V4.3 - Analysis Store (persistent storage for analyses)
        self.analysis_store: AnalysisStore = get_analysis_store()

        # Composants V4
        self.trend_discovery = None
        self.market_screener = None  # Fallback V3
        self.social_scanner = None
        self.grok_scanner = None
        self.position_manager = None
        self.rotation_manager = None
        self.stock_discovery = None

        # V6 - Discovery Mode (Bull regime)
        self.discovery_mode = None
        if DISCOVERY_MODE_AVAILABLE:
            self.discovery_mode = get_discovery_mode()
            logger.info("[ORCHESTRATOR] Discovery Mode initialized")
        self.ibkr_executor = None

        # V4.4 - Market Data Fetcher (pour fournir les donnees OHLCV aux pilliers)
        self.market_data_fetcher = MarketDataFetcher()

        # V4.5 - Hybrid Screener (FMP pre-screening + IBKR validation)
        self.hybrid_screener: Optional[HybridScreener] = None

        # V4.5 - IBKR Watchdog for automatic reconnection
        self.ibkr_watchdog: Optional[IBKRWatchdog] = None

        # V4.5 - Async Position Sizer with EUR/USD conversion
        self.position_sizer: Optional[AsyncPositionSizer] = None

        # V5 - Adaptive Learning
        self.adaptive_scorer: Optional[AdaptiveScorer] = None
        if ADAPTIVE_LEARNING_AVAILABLE:
            try:
                self.adaptive_scorer = get_adaptive_scorer()
                logger.info(f"[V5] Adaptive learning enabled. Current weights: {self.adaptive_scorer.weights.to_dict()}")
            except Exception as e:
                logger.warning(f"[V5] Adaptive learning unavailable: {e}")

        # V5.1 - Shadow Tracker (Paper Trading Automatique pour Apprentissage)
        self.shadow_tracker: Optional[ShadowTracker] = None

        # V5.2 - Indicator Discovery (Auto-découverte des indicateurs qui fonctionnent)
        self.indicator_discovery: Optional[IndicatorDiscovery] = None

        # V5.2 - Intuitive Reasoning (Raisonnement News → Secteur → Actions)
        self.intuitive_reasoning: Optional[IntuitiveReasoning] = None

        # État runtime
        self._running = False
        self._task: Optional[asyncio.Task] = None

        logger.info(f"MarketAgent V4.1 initialized: capital={self.config.initial_capital}€")

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    async def initialize(self):
        """Initialise tous les composants"""
        logger.info("Initializing MarketAgent V4.1 components...")

        # 0. ReasoningEngine (4 Pillars) - PRIMARY
        try:
            from src.agents.reasoning_engine import get_reasoning_engine
            self.reasoning_engine = await get_reasoning_engine()
            logger.info("ReasoningEngine (4 Pillars) initialized")
        except Exception as e:
            logger.warning(f"ReasoningEngine not available: {e}")

        # 0.1 Trade Memory (RAG)
        if self.config.memory_enabled:
            try:
                from src.intelligence.trade_memory import AdaptiveTradeMemory
                self.trade_memory = AdaptiveTradeMemory(
                    db_path=self.config.memory_db_path
                )
                logger.info("TradeMemory (RAG) initialized")
            except Exception as e:
                logger.warning(f"TradeMemory not available: {e}")

        # 0.2 Pattern Extractor
        if self.trade_memory:
            try:
                from src.intelligence.pattern_extractor import PatternExtractor
                self.pattern_extractor = PatternExtractor()
                logger.info("PatternExtractor initialized")
            except Exception as e:
                logger.warning(f"PatternExtractor not available: {e}")

        # 0.3 Strategy Composer (uses Gemini for strategy creation)
        try:
            from src.agents.strategy_composer import get_strategy_composer, ComposerConfig
            composer_config = ComposerConfig(
                openrouter_api_key=self.config.openrouter_api_key or os.getenv('OPENROUTER_API_KEY'),
                llm_model=self.config.llm_model
            )
            self.strategy_composer = await get_strategy_composer(composer_config)
            logger.info("StrategyComposer initialized (Gemini)")
        except Exception as e:
            logger.warning(f"StrategyComposer not available: {e}")

        # 0.4 Narrative Generator (human-readable analysis reports)
        # V4.9: Uses Gemini direct (GOOGLE_AI_API_KEY)
        try:
            from src.intelligence.narrative_generator import NarrativeGenerator
            self.narrative_generator = NarrativeGenerator(
                api_key=os.getenv('GOOGLE_AI_API_KEY'),
                model=os.getenv('GEMINI_MODEL', 'gemini-3-flash-preview')
            )
            await self.narrative_generator.initialize()
            logger.info("NarrativeGenerator initialized with Gemini")
        except Exception as e:
            logger.warning(f"NarrativeGenerator not available: {e}")

        # 1. Trend Discovery - V4.9: Uses Gemini direct
        try:
            from src.intelligence.trend_discovery import TrendDiscovery
            self.trend_discovery = TrendDiscovery(
                api_key=os.getenv('GOOGLE_AI_API_KEY'),
                model=os.getenv('GEMINI_MODEL', 'gemini-3-flash-preview'),
                data_dir=os.path.join(self.config.data_dir, 'trends')
            )
            await self.trend_discovery.initialize()
            logger.info("TrendDiscovery initialized with Gemini")
        except Exception as e:
            logger.warning(f"TrendDiscovery not available: {e}")

        # 2. Market Screener (FALLBACK V3 - only if ReasoningEngine fails)
        if not self.reasoning_engine:
            try:
                from src.screening.screener import market_screener
                self.market_screener = market_screener
                logger.info("MarketScreener V3 initialized (fallback)")
            except Exception as e:
                logger.warning(f"MarketScreener not available: {e}")

        # 3. Social Scanner (nouveau V4)
        try:
            from src.intelligence.social_scanner import SocialScanner
            self.social_scanner = SocialScanner(enable_stocktwits=False, enable_social=False)
            await self.social_scanner.initialize()
            logger.info("SocialScanner initialized")
        except Exception as e:
            logger.warning(f"SocialScanner not available: {e}")

        # 4. Grok Scanner (nouveau V4)
        try:
            from src.intelligence.grok_scanner import GrokScanner
            self.grok_scanner = GrokScanner(
                api_key=self.config.grok_api_key or os.getenv('GROK_API_KEY')
            )
            await self.grok_scanner.initialize()
            logger.info("GrokScanner initialized")
            # V5.1 - Initialize Position Manager
            self.position_manager = get_position_manager(grok_scanner=self.grok_scanner)
            self.rotation_manager = get_rotation_manager(
                position_manager=self.position_manager,
                grok_scanner=self.grok_scanner,
                max_positions=10
            )
            logger.info("AdaptivePositionManager & RotationManager initialized")
        except Exception as e:
            logger.warning(f"GrokScanner not available: {e}")

        # 5. Stock Discovery (nouveau V4)
        try:
            from src.intelligence.stock_discovery import StockDiscovery
            self.stock_discovery = StockDiscovery()
            logger.info("StockDiscovery initialized")
        except Exception as e:
            logger.warning(f"StockDiscovery not available: {e}")

        # 6. IBKR Executor (nouveau V4)
        try:
            from src.execution.ibkr_executor import IBKRExecutor
            self.ibkr_executor = IBKRExecutor(
                host=self.config.ibkr_host,
                port=self.config.ibkr_port,
                client_id=self.config.ibkr_client_id
            )
            # Ne pas connecter automatiquement - attendre le trading
            logger.info("IBKRExecutor initialized (not connected)")
        except Exception as e:
            logger.warning(f"IBKRExecutor not available: {e}")

        # 7. V4.5 - Hybrid Screener (FMP pre-screening + IBKR validation)
        if HYBRID_SCREENER_AVAILABLE:
            try:
                # Get IBKR client if connected
                ibkr_client = None
                if self.ibkr_executor and hasattr(self.ibkr_executor, 'ib'):
                    ibkr_client = self.ibkr_executor.ib

                self.hybrid_screener = await get_hybrid_screener(ibkr_client=ibkr_client)
                logger.info("HybridScreener initialized (FMP pre-screening + IBKR validation)")
            except Exception as e:
                logger.warning(f"HybridScreener not available: {e}")

        # 8. V4.5 - IBKR Watchdog for automatic reconnection
        if IBKR_WATCHDOG_AVAILABLE and self.ibkr_executor:
            try:
                ibkr_client = getattr(self.ibkr_executor, 'ib', None)
                if ibkr_client:
                    watchdog_config = WatchdogConfig(
                        host=self.config.ibkr_host,
                        port=self.config.ibkr_port,
                        client_id=self.config.ibkr_client_id,
                        max_daily_loss_pct=0.03  # 3% max daily loss
                    )
                    self.ibkr_watchdog = IBKRWatchdog(
                        ib=ibkr_client,
                        config=watchdog_config,
                        on_kill_switch=self._on_watchdog_kill_switch
                    )
                    logger.info("IBKRWatchdog initialized (auto-reconnect + kill switch)")
            except Exception as e:
                logger.warning(f"IBKRWatchdog not available: {e}")

        # 9. V4.5 - Async Position Sizer with EUR/USD conversion
        if ASYNC_POSITION_SIZER_AVAILABLE:
            try:
                # Get IBKR client for FX rates
                ibkr_client = None
                if self.ibkr_executor and hasattr(self.ibkr_executor, 'ib'):
                    ibkr_client = self.ibkr_executor.ib

                self.position_sizer = AsyncPositionSizer(
                    total_capital=self.config.initial_capital,
                    account_currency='EUR',  # European account
                    ibkr_client=ibkr_client,
                    max_position_pct=0.25
                )
                logger.info(f"AsyncPositionSizer initialized: {self.config.initial_capital} EUR")
            except Exception as e:
                logger.warning(f"AsyncPositionSizer not available: {e}")

        # 10. V5.1 - Shadow Tracker (Apprentissage Autonome)
        if SHADOW_TRACKER_AVAILABLE:
            try:
                self.shadow_tracker = await get_shadow_tracker()
                stats = self.shadow_tracker.get_statistics()
                logger.info(f"[V5.1] ShadowTracker initialized: {stats['total_signals']} signals tracked, "
                           f"Win Rate: {stats['win_rate']:.1f}%")
            except Exception as e:
                logger.warning(f"ShadowTracker not available: {e}")

        # 11. V5.2 - Indicator Discovery (Auto-découverte des indicateurs)
        if INDICATOR_DISCOVERY_AVAILABLE:
            try:
                self.indicator_discovery = await get_indicator_discovery()
                stats = self.indicator_discovery.get_statistics()
                logger.info(f"[V5.2] IndicatorDiscovery initialized: {stats['indicators_tracked']} indicators tracked")
            except Exception as e:
                logger.warning(f"IndicatorDiscovery not available: {e}")

        # 12. V5.2 - Intuitive Reasoning (News → Secteur → Actions)
        if INTUITIVE_REASONING_AVAILABLE:
            try:
                self.intuitive_reasoning = await get_intuitive_reasoning()
                stats = self.intuitive_reasoning.get_statistics()
                logger.info(f"[V5.2] IntuitiveReasoning initialized: {stats['active_implications']} active implications")
            except Exception as e:
                logger.warning(f"IntuitiveReasoning not available: {e}")

        logger.info("MarketAgent initialization complete")

    async def close(self):
        """Ferme tous les composants"""
        logger.info("Closing MarketAgent...")

        # V4.5 - Stop watchdog first
        if self.ibkr_watchdog:
            try:
                await self.ibkr_watchdog.stop()
            except Exception as e:
                logger.error(f"Error stopping watchdog: {e}")

        if self.narrative_generator:
            await self.narrative_generator.close()

        if self.trend_discovery:
            await self.trend_discovery.close()

        if self.social_scanner:
            await self.social_scanner.close()

        if self.grok_scanner:
            await self.grok_scanner.close()

        # V4.5 - Close reasoning engine (has async pillars with sessions)
        if self.reasoning_engine:
            try:
                await self.reasoning_engine.close()
            except Exception as e:
                logger.error(f"Error closing reasoning engine: {e}")

        # V4.5 - Close stock discovery (has aiohttp session)
        if self.stock_discovery:
            try:
                await self.stock_discovery.close()
            except Exception as e:
                logger.error(f"Error closing stock discovery: {e}")

        # V4.5 - Close hybrid screener (has FMP client with session)
        if self.hybrid_screener:
            try:
                await self.hybrid_screener.close()
            except Exception as e:
                logger.error(f"Error closing hybrid screener: {e}")

        if self.ibkr_executor:
            await self.ibkr_executor.disconnect()

        logger.info("MarketAgent closed")

    async def _on_watchdog_kill_switch(self, reason: str):
        """
        Callback when IBKR watchdog triggers kill switch.

        This is called when:
        - Connection failed 5 times consecutively
        - Daily loss exceeds 3%

        Args:
            reason: Reason for kill switch ('connection_failure', 'daily_loss_limit')
        """
        logger.critical(f"KILL SWITCH TRIGGERED BY WATCHDOG: {reason}")

        # Activate guardrails kill switch
        self.guardrails.activate_kill_switch(reason)

        # Log to state manager
        self.state_manager.log_error(f"Kill switch: {reason}")

        # Send notification if available
        # TODO: Integrate with notification_manager

    # =========================================================================
    # V4.5 - HYBRID SCREENING (FMP + IBKR)
    # =========================================================================

    async def run_fmp_prescreening(self) -> List[str]:
        """
        Run FMP pre-screening to get candidates before IBKR validation.

        This avoids IBKR pacing violations by pre-filtering with FMP.

        Returns:
            List of symbol strings with RSI breakout signals
        """
        if not self.hybrid_screener:
            logger.warning("HybridScreener not available - using existing watchlist")
            return self.state_manager.state.watchlist[:100]

        try:
            logger.info("=== FMP PRE-SCREENING STARTED ===")

            # Phase 1: FMP pre-screening (scans 3000 tickers)
            candidates = await self.hybrid_screener.run_prescreening()

            if not candidates:
                logger.warning("No candidates from FMP pre-screening")
                return []

            # Extract symbols
            symbols = [c.symbol for c in candidates]

            # Update watchlist with new candidates
            existing = set(self.state_manager.state.watchlist)
            new_symbols = [s for s in symbols if s not in existing]
            if new_symbols:
                self.state_manager.state.watchlist.extend(new_symbols[:self.config.max_watchlist_size])
                logger.info(f"Added {len(new_symbols)} new symbols to watchlist from FMP")

            # Set focus symbols for trading
            self.state_manager.state.focus_symbols = symbols[:self.config.max_focus_symbols]

            logger.info(f"FMP pre-screening complete: {len(symbols)} candidates, {len(new_symbols)} new")

            return symbols

        except Exception as e:
            logger.error(f"FMP pre-screening error: {e}")
            return []

    async def validate_with_ibkr(self, symbols: List[str]) -> List[str]:
        """
        Validate FMP candidates with IBKR official data.

        This is Phase 2 of hybrid screening - validates signals with real data.

        Args:
            symbols: Symbols from FMP pre-screening

        Returns:
            List of validated symbol strings
        """
        if not self.hybrid_screener:
            logger.warning("HybridScreener not available - returning unvalidated symbols")
            return symbols

        if not self.hybrid_screener.ibkr:
            logger.warning("IBKR not connected - skipping validation")
            return symbols

        try:
            logger.info(f"=== IBKR VALIDATION: {len(symbols)} candidates ===")

            # Convert to ScreeningCandidate format
            from src.screening.hybrid_screener import ScreeningCandidate
            candidates = [
                ScreeningCandidate(
                    symbol=s,
                    signal='BUY',
                    rsi_at_breakout=0,
                    strength='UNKNOWN',
                    source='FMP'
                )
                for s in symbols
            ]

            # Run IBKR validation
            validated = await self.hybrid_screener.run_ibkr_validation(candidates)

            validated_symbols = [c.symbol for c in validated]
            logger.info(f"IBKR validation complete: {len(validated_symbols)}/{len(symbols)} validated")

            return validated_symbols

        except Exception as e:
            logger.error(f"IBKR validation error: {e}")
            return symbols

    # =========================================================================
    # MARKET STATUS
    # =========================================================================

    def get_market_status(self) -> MarketStatus:
        """Retourne le statut actuel du marché"""
        now = datetime.now()
        current_time = now.time()

        # Weekend
        if now.weekday() >= 5:
            return MarketStatus.CLOSED

        if current_time < self.config.market_open:
            return MarketStatus.PRE_MARKET
        elif current_time < self.config.market_close:
            return MarketStatus.OPEN
        elif current_time < time(20, 0):
            return MarketStatus.AFTER_HOURS
        else:
            return MarketStatus.CLOSED

    # =========================================================================
    # PHASE: DISCOVERY (06:00)
    # =========================================================================

    async def run_discovery_phase(self) -> Dict:
        """
        Phase de découverte matinale.
        Scanne les réseaux sociaux et les news pour identifier les opportunités.

        Returns:
            Dict avec les résultats de la découverte
        """
        logger.info("=== DISCOVERY PHASE STARTED ===")
        self.state_manager.set_phase(AgentPhase.DISCOVERY)

        results = {
            "timestamp": datetime.now().isoformat(),
            "social_trending": [],
            "grok_insights": [],
            "discovered_stocks": [],
            "volume_anomalies": [],
            "watchlist": []
        }

        # 1. Scanner les réseaux sociaux (StockTwits, Reddit)
        if self.social_scanner:
            try:
                social_result = await self.social_scanner.full_scan()
                # SocialScanResult has trending_symbols List[TrendingSymbol] + get_top_symbols() helper
                if hasattr(social_result, 'get_top_symbols'):
                    results["social_trending"] = social_result.get_top_symbols(20)
                elif hasattr(social_result, 'trending_symbols') and social_result.trending_symbols:
                    results["social_trending"] = [t.symbol for t in social_result.trending_symbols[:20]]
                logger.info(f"Social scan: {len(results['social_trending'])} trending symbols")
            except Exception as e:
                logger.error(f"Social scan error: {e}")
                self.state_manager.log_error(f"Social scan failed: {e}")

        # 2. Scanner Grok (X/Twitter via xAI)
        if self.grok_scanner:
            try:
                grok_insights = await self.grok_scanner.search_financial_trends()
                results["grok_insights"] = []
                for insight in grok_insights:
                    # GrokInsight has symbols attribute
                    if hasattr(insight, 'to_dict'):
                        results["grok_insights"].append(insight.to_dict())
                    if hasattr(insight, 'symbols') and insight.symbols:
                        results["social_trending"].extend(insight.symbols)
                logger.info(f"Grok scan: {len(results['grok_insights'])} insights")
            except Exception as e:
                logger.error(f"Grok scan error: {e}")
                self.state_manager.log_error(f"Grok scan failed: {e}")

        # 3. V5.2 - Intuitive Reasoning: News générales → Secteurs → Actions
        # C'est ici qu'on analyse les news macro pour identifier les secteurs à surveiller
        news_derived_symbols = []
        if self.intuitive_reasoning:
            try:
                logger.info("Running IntuitiveReasoning: Analyzing macro news for sector implications...")
                implications = await self.intuitive_reasoning.daily_news_scan()

                # Récupérer les recommandations d'achat
                buy_recs = self.intuitive_reasoning.get_buy_recommendations(min_confidence=0.5)

                # Extraire les symboles à surveiller
                news_derived_symbols = self.intuitive_reasoning.get_symbols_to_watch()

                results["intuitive_reasoning"] = {
                    "implications_count": len(implications),
                    "buy_recommendations": len(buy_recs),
                    "symbols_derived": news_derived_symbols,
                    "top_sectors": list(set(impl.sector for impl in implications))[:5] if implications else []
                }

                # Log les implications clés
                for rec in buy_recs[:3]:  # Top 3 recommendations
                    logger.info(f"[INTUITIVE] BUY Signal: {rec['sector']} -> {rec['symbols'][:3]} (conf={rec['confidence']:.0%})")
                    logger.info(f"  Catalyst: {rec['catalyst'][:80]}...")

                logger.info(f"IntuitiveReasoning: {len(implications)} implications, {len(news_derived_symbols)} symbols derived from macro news")
            except Exception as e:
                logger.error(f"IntuitiveReasoning error: {e}")
                self.state_manager.log_error(f"IntuitiveReasoning failed: {e}")

        # 4. Découverte dynamique de stocks (volume/momentum)
        if self.stock_discovery:
            try:
                discovered = await self.stock_discovery.discover_trending()
                results["discovered_stocks"] = discovered
                logger.info(f"Stock discovery: {len(discovered)} new symbols")
            except Exception as e:
                logger.error(f"Stock discovery error: {e}")
                self.state_manager.log_error(f"Stock discovery failed: {e}")

        # 5. Scanner les anomalies de volume
        if self.trend_discovery:
            try:
                anomalies = await self.trend_discovery.detect_volume_anomalies()
                results["volume_anomalies"] = [s for s, _ in anomalies[:20]]
                logger.info(f"Volume anomalies: {len(results['volume_anomalies'])} detected")
            except Exception as e:
                logger.error(f"Volume anomaly detection error: {e}")

        # 6. V4.5 - FMP Pre-screening (RSI breakout candidates)
        # This is the PRIMARY source of candidates - avoids IBKR pacing violations
        fmp_candidates = []
        if self.hybrid_screener:
            try:
                logger.info("Running FMP pre-screening for RSI breakout candidates...")
                fmp_candidates = await self.run_fmp_prescreening()
                results["fmp_candidates"] = fmp_candidates
                logger.info(f"FMP pre-screening: {len(fmp_candidates)} RSI breakout candidates")
            except Exception as e:
                logger.error(f"FMP pre-screening error: {e}")
                self.state_manager.log_error(f"FMP pre-screening failed: {e}")

        # 7. Construire la watchlist dynamique
        all_symbols = set()
        all_symbols.update(results["social_trending"])
        all_symbols.update(results["discovered_stocks"])
        all_symbols.update(results["volume_anomalies"])

        # V5.2 - Ajouter les symboles dérivés des news macro (IntuitiveReasoning)
        if news_derived_symbols:
            all_symbols.update(news_derived_symbols)
            logger.info(f"Added {len(news_derived_symbols)} symbols from macro news analysis")

        # V4.5 - Prioritize FMP candidates (they have RSI signals)
        all_symbols.update(fmp_candidates)

        # Limiter la taille
        results["watchlist"] = list(all_symbols)[:self.config.max_watchlist_size]

        # V4.5 - Set focus symbols prioritizing FMP candidates
        if fmp_candidates:
            # FMP candidates first (they have confirmed signals)
            focus = fmp_candidates[:self.config.max_focus_symbols]
            # Fill remaining with other sources
            remaining_slots = self.config.max_focus_symbols - len(focus)
            if remaining_slots > 0:
                other_symbols = [s for s in results["watchlist"] if s not in focus]
                focus.extend(other_symbols[:remaining_slots])
        else:
            # FMP failed - use social/momentum sources from watchlist
            focus = list(all_symbols)[:self.config.max_focus_symbols]
            logger.warning(f"FMP unavailable - using {len(focus)} symbols from social sources")

        # Always set focus_symbols
        self.state_manager.state.focus_symbols = focus
        results["focus_symbols"] = focus

        # Mettre à jour l'état
        self.state_manager.update_watchlist(results["watchlist"])
        self.state_manager.update_last_scan()


        # === AUTO-GENERATE REPORT ===
        try:
            import subprocess
            import json as json_lib
            logger.info("Generating market analysis report...")
            # Save discovery results for report
            report_data_path = "/app/data/latest_scan.json"
            with open(report_data_path, "w") as f:
                json_lib.dump(results, f, indent=2, default=str)
            # Generate report
            proc = subprocess.run(
                ["python", "/app/src/intelligence/report_generator.py"],
                capture_output=True,
                text=True,
                timeout=120
            )
            if proc.returncode == 0:
                logger.info("Market analysis report generated successfully")
            else:
                logger.warning(f"Report generation warning: {proc.stderr[:200]}")
        except Exception as e:
            logger.error(f"Report generation error: {e}")

        logger.info(f"=== DISCOVERY COMPLETE: {len(results['watchlist'])} symbols in watchlist, {len(fmp_candidates)} FMP candidates ===")
        return results

    # =========================================================================
    # PHASE: ANALYSIS (07:00)
    # =========================================================================

    async def run_analysis_phase(self) -> Dict:
        """
        Phase d'analyse des tendances.
        Utilise le LLM pour identifier les narratifs et prioriser les symboles.

        Returns:
            Dict avec les résultats de l'analyse
        """
        logger.info("=== ANALYSIS PHASE STARTED ===")
        self.state_manager.set_phase(AgentPhase.ANALYSIS)

        results = {
            "timestamp": datetime.now().isoformat(),
            "trends": [],
            "narratives": [],
            "market_sentiment": 0.0,
            "focus_symbols": [],
            "market_regime": MarketRegime.UNKNOWN.value
        }

        # 1. Trend Discovery (analyse LLM)
        if self.trend_discovery:
            try:
                report = await self.trend_discovery.daily_scan()
                results["trends"] = [t.to_dict() for t in report.trends]
                results["narratives"] = report.narrative_updates
                results["market_sentiment"] = report.market_sentiment

                # Extraire les focus symbols
                focus = report.watchlist_additions[:self.config.max_focus_symbols]
                results["focus_symbols"] = focus

                # Déterminer le régime de marché
                if report.market_sentiment > 0.3:
                    regime = MarketRegime.BULL_STRONG
                    confidence = min(0.9, abs(report.market_sentiment))
                elif report.market_sentiment > 0.1:
                    regime = MarketRegime.BULL_WEAK
                    confidence = 0.6
                elif report.market_sentiment < -0.3:
                    regime = MarketRegime.BEAR_STRONG
                    confidence = min(0.9, abs(report.market_sentiment))
                elif report.market_sentiment < -0.1:
                    regime = MarketRegime.BEAR_WEAK
                    confidence = 0.6
                else:
                    regime = MarketRegime.RANGING
                    confidence = 0.5

                results["market_regime"] = regime.value
                self.state_manager.update_market_regime(regime, confidence)

                logger.info(f"Trend analysis: {len(results['trends'])} trends, "
                           f"sentiment={results['market_sentiment']:.2f}, "
                           f"regime={regime.value}")

            except Exception as e:
                logger.error(f"Trend analysis error: {e}")
                self.state_manager.log_error(f"Trend analysis failed: {e}")

        # 2. Mettre à jour l'état
        self.state_manager.update_focus_symbols(results["focus_symbols"])
        self.state_manager.update_trends(results["trends"])
        self.state_manager.update_narratives([n.get("name", "") for n in results["narratives"]])

        logger.info(f"=== ANALYSIS COMPLETE: {len(results['focus_symbols'])} focus symbols ===")
        return results

    # =========================================================================
    # PHASE: TRADING (09:30-16:00)
    # =========================================================================

    async def run_trading_scan(self, execute_trades: bool = True) -> Dict:
        """
        Un cycle de scan pendant les heures de trading.
        Détecte les signaux et exécute les trades (si execute_trades=True).

        Args:
            execute_trades: Si False, analyse seulement sans exécuter de trades.
                           Utile pour préparer l'ouverture du marché.

        Returns:
            Dict avec les résultats du scan
        """
        logger.info("--- Trading Scan Started ---")
        self.state_manager.set_phase(AgentPhase.TRADING)

        results = {
            "timestamp": datetime.now().isoformat(),
            "scanned_symbols": 0,
            "signals_found": 0,
            "trades_executed": 0,
            "trades_rejected": 0,
            "alerts": [],
            # V4.2 - Detailed scoring for transparency
            "analyzed_symbols": [],
            "scoring_details": {},  # symbol -> {total_score, technical, fundamental, sentiment, news}
            "rejected_symbols": {},  # symbol -> reason
            "guardrails_blocked": [],  # List of {symbol, reason}
            # V4.2 - Narrative reports (human-readable analysis)
            "narrative_reports": {}  # symbol -> markdown report
        }

        # Vérifier le kill switch
        kill_active, kill_reason = self.guardrails.is_kill_switch_active()
        if kill_active:
            logger.warning(f"Kill switch active: {kill_reason}")
            return results

        # 1. Récupérer les symboles à scanner
        focus_symbols = self.state_manager.state.focus_symbols
        if not focus_symbols:
            focus_symbols = self.state_manager.state.watchlist[:50]

        if not focus_symbols:
            logger.warning("No symbols to scan")
            return results

        # V4.3 - Filter out recently analyzed symbols to avoid redundant re-analysis
        original_count = len(focus_symbols)
        cooldown_hours = getattr(self.config, 'analysis_cooldown_hours', 1)  # Default 1 hour

        # Get symbols that haven't been analyzed recently
        fresh_symbols = self.analysis_store.get_unanalyzed_symbols(focus_symbols, hours=cooldown_hours)

        if len(fresh_symbols) < len(focus_symbols):
            skipped = original_count - len(fresh_symbols)
            logger.info(f"Skipping {skipped} recently analyzed symbols (cooldown: {cooldown_hours}h)")
            results["skipped_recent"] = skipped

        # If all symbols were recently analyzed, maybe get some that need refresh
        if not fresh_symbols:
            # Get symbols needing reanalysis (analyzed > 24h ago)
            stale_symbols = self.analysis_store.get_symbols_needing_reanalysis(hours=24)
            if stale_symbols:
                fresh_symbols = stale_symbols[:15]
                logger.info(f"All focus symbols recently analyzed. Refreshing {len(fresh_symbols)} stale symbols.")
            else:
                logger.info("All symbols recently analyzed. Waiting for cooldown.")
                results["all_symbols_recent"] = True
                return results

        focus_symbols = fresh_symbols
        results["scanned_symbols"] = len(focus_symbols)

        # 2. Scanner avec ReasoningEngine (4 Pillars) - PRIMARY
        if self.reasoning_engine:
            try:
                for symbol in focus_symbols:
                    try:
                        # Track analyzed symbol
                        results["analyzed_symbols"].append(symbol)

                        # V4.4: Fetch OHLCV data for technical analysis
                        logger.info(f"Fetching OHLCV data for {symbol}...")
                        df = self.market_data_fetcher.get_historical_data(
                            symbol=symbol,
                            period='6mo',  # 6 months for enough data points (>50 required)
                            interval='1d'
                        )
                        if df is None or len(df) < 50:
                            logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} rows (need 50+)")
                            results["rejected_symbols"][symbol] = f"Insufficient historical data ({len(df) if df is not None else 0} rows)"
                            continue

                        # Analyze with 4 Pillars - NOW WITH DATA
                        reasoning_result = await self.reasoning_engine.analyze(symbol, df=df)

                        # Store detailed scoring for transparency (V4.3 - enriched with reasoning)
                        # V4.4: Round all scores to 2 decimals to avoid floating point artifacts
                        results["scoring_details"][symbol] = {
                            "total_score": round(reasoning_result.total_score, 2),
                            "technical": round(reasoning_result.technical_score.score, 2) if reasoning_result.technical_score else 0,
                            "fundamental": round(reasoning_result.fundamental_score.score, 2) if reasoning_result.fundamental_score else 0,
                            "sentiment": round(reasoning_result.sentiment_score.score, 2) if reasoning_result.sentiment_score else 0,
                            "news": round(reasoning_result.news_score.score, 2) if reasoning_result.news_score else 0,
                            "decision": reasoning_result.decision.value,
                            "key_factors": reasoning_result.key_factors[:5] if reasoning_result.key_factors else [],
                            "risk_factors": reasoning_result.risk_factors[:5] if reasoning_result.risk_factors else [],
                            # V4.3 - Detailed pillar reasoning for decision tree display
                            # V4.4: All scores rounded to 2 decimals
                            # V4.8: Added data_quality for each pillar (indicates data reliability)
                            "pillar_details": {
                                "technical": {
                                    "score": round(reasoning_result.technical_score.score, 2) if reasoning_result.technical_score else 0,
                                    "reasoning": reasoning_result.technical_score.reasoning if reasoning_result.technical_score else "",
                                    "factors": reasoning_result.technical_score.factors[:5] if reasoning_result.technical_score and reasoning_result.technical_score.factors else [],
                                    "signal": reasoning_result.technical_score.signal.value if reasoning_result.technical_score else "neutral",
                                    "data_quality": reasoning_result.technical_score.data_quality if reasoning_result.technical_score else 0
                                },
                                "fundamental": {
                                    "score": round(reasoning_result.fundamental_score.score, 2) if reasoning_result.fundamental_score else 0,
                                    "reasoning": reasoning_result.fundamental_score.reasoning if reasoning_result.fundamental_score else "",
                                    "factors": reasoning_result.fundamental_score.factors[:5] if reasoning_result.fundamental_score and reasoning_result.fundamental_score.factors else [],
                                    "signal": reasoning_result.fundamental_score.signal.value if reasoning_result.fundamental_score else "neutral",
                                    "data_quality": reasoning_result.fundamental_score.data_quality if reasoning_result.fundamental_score else 0
                                },
                                "sentiment": {
                                    "score": round(reasoning_result.sentiment_score.score, 2) if reasoning_result.sentiment_score else 0,
                                    "reasoning": reasoning_result.sentiment_score.reasoning if reasoning_result.sentiment_score else "",
                                    "factors": reasoning_result.sentiment_score.factors[:5] if reasoning_result.sentiment_score and reasoning_result.sentiment_score.factors else [],
                                    "signal": reasoning_result.sentiment_score.signal.value if reasoning_result.sentiment_score else "neutral",
                                    "data_quality": reasoning_result.sentiment_score.data_quality if reasoning_result.sentiment_score else 0
                                },
                                "news": {
                                    "score": round(reasoning_result.news_score.score, 2) if reasoning_result.news_score else 0,
                                    "reasoning": reasoning_result.news_score.reasoning if reasoning_result.news_score else "",
                                    "factors": reasoning_result.news_score.factors[:5] if reasoning_result.news_score and reasoning_result.news_score.factors else [],
                                    "signal": reasoning_result.news_score.signal.value if reasoning_result.news_score else "neutral",
                                    "data_quality": reasoning_result.news_score.data_quality if reasoning_result.news_score else 0
                                }
                            },
                            "reasoning_summary": reasoning_result.reasoning_summary
                        }

                        # V4.3 - PERSIST ANALYSIS TO STORE
                        try:
                            # Get market regime if available
                            market_regime = None
                            if reasoning_result.market_context:
                                market_regime = reasoning_result.market_context.regime.value

                            # Save to persistent store (SQLite + JSON)
                            self.analysis_store.save_analysis(
                                symbol=symbol,
                                reasoning_result=reasoning_result,
                                market_regime=market_regime
                            )

                            # Add journey step with full reasoning
                            pillar_details = results["scoring_details"][symbol].get("pillar_details", {})
                            step_type = "analysis" if reasoning_result.decision.value in ["strong_buy", "buy"] else "rejected"

                            self.analysis_store.add_journey_step(
                                symbol=symbol,
                                step=step_type,
                                title=f"Score {round(reasoning_result.total_score, 1)}/100 - {reasoning_result.decision.value}",
                                reasoning=reasoning_result.reasoning_summary,
                                data={
                                    "total": round(reasoning_result.total_score, 2),
                                    "technical": round(reasoning_result.technical_score.score, 2) if reasoning_result.technical_score else 0,
                                    "fundamental": round(reasoning_result.fundamental_score.score, 2) if reasoning_result.fundamental_score else 0,
                                    "sentiment": round(reasoning_result.sentiment_score.score, 2) if reasoning_result.sentiment_score else 0,
                                    "news": round(reasoning_result.news_score.score, 2) if reasoning_result.news_score else 0,
                                    "decision": reasoning_result.decision.value,
                                    "pillar_details": pillar_details,
                                    "key_factors": reasoning_result.key_factors[:5] if reasoning_result.key_factors else [],
                                    "risk_factors": reasoning_result.risk_factors[:5] if reasoning_result.risk_factors else []
                                }
                            )
                            logger.info(f"Persisted analysis for {symbol}")
                        except Exception as e:
                            logger.error(f"Failed to persist analysis for {symbol}: {e}")

                        # Skip if not actionable - but record why
                        if reasoning_result.decision.value not in ["strong_buy", "buy"]:
                            decision_reason = f"Decision: {reasoning_result.decision.value} (score: {reasoning_result.total_score:.1f}/100)"
                            if reasoning_result.risk_factors:
                                decision_reason += f" | Risks: {', '.join(reasoning_result.risk_factors[:2])}"
                            results["rejected_symbols"][symbol] = decision_reason
                            continue

                        # Consult Trade Memory for similar past trades
                        memory_insight = None
                        if self.trade_memory:
                            try:
                                from src.intelligence.trade_memory import TradeContext
                                context = self._build_trade_context(symbol, reasoning_result)
                                similar_trades = self.trade_memory.find_similar(
                                    context=context,
                                    top_k=self.config.similar_trades_k,
                                    current_regime=context.market_regime
                                )
                                if similar_trades:
                                    # Calculate historical win rate for similar contexts
                                    wins = sum(1 for t in similar_trades if t.outcome.is_win)
                                    memory_insight = {
                                        "similar_count": len(similar_trades),
                                        "historical_win_rate": wins / len(similar_trades),
                                        "avg_pnl": sum(t.outcome.pnl_pct for t in similar_trades) / len(similar_trades),
                                        "avg_relevance": sum(t.relevance_score for t in similar_trades) / len(similar_trades)
                                    }
                                    logger.info(f"Memory insight for {symbol}: {memory_insight}")
                            except Exception as e:
                                logger.warning(f"Trade memory lookup failed: {e}")

                        # Convert to alert format
                        alert = self._reasoning_to_alert(symbol, reasoning_result, memory_insight)
                        results["alerts"].append(alert)
                        results["signals_found"] += 1

                        # V5.1 - Shadow Track EVERY BUY signal for autonomous learning
                        if self.shadow_tracker:
                            try:
                                # Get current price for entry
                                entry_price = alert.get("price", 0)
                                if entry_price > 0:
                                    await self.shadow_tracker.track_signal(
                                        symbol=symbol,
                                        signal_type=alert.get("signal", "BUY"),
                                        total_score=reasoning_result.total_score,
                                        technical_score=reasoning_result.technical_score.score if reasoning_result.technical_score else 0,
                                        fundamental_score=reasoning_result.fundamental_score.score if reasoning_result.fundamental_score else 0,
                                        sentiment_score=reasoning_result.sentiment_score.score if reasoning_result.sentiment_score else 0,
                                        news_score=reasoning_result.news_score.score if reasoning_result.news_score else 0,
                                        entry_price=entry_price,
                                        stop_loss=alert.get("stop_loss"),
                                        take_profit=entry_price * 1.10,  # +10% default TP
                                        key_factors=reasoning_result.key_factors[:5] if reasoning_result.key_factors else [],
                                        risk_factors=reasoning_result.risk_factors[:5] if reasoning_result.risk_factors else []
                                    )
                                    logger.info(f"[SHADOW] Auto-tracked {symbol} @ ${entry_price:.2f}")
                            except Exception as e:
                                logger.warning(f"Shadow tracking failed for {symbol}: {e}")

                        # V4.2 - Generate narrative report for actionable signals
                        if self.narrative_generator:
                            try:
                                # Collect social data for narrative
                                social_data = None
                                grok_data = None

                                if self.social_scanner:
                                    try:
                                        social_data = await self.social_scanner.get_symbol_details_for_narrative(symbol)
                                    except Exception:
                                        pass

                                if self.grok_scanner:
                                    try:
                                        grok_data = await self.grok_scanner.get_symbol_details_for_narrative(symbol)
                                    except Exception:
                                        pass

                                # Generate narrative
                                narrative = await self.narrative_generator.generate_analysis(
                                    symbol=symbol,
                                    social_data=social_data,
                                    grok_data=grok_data,
                                    technical_data={
                                        'rsi': reasoning_result.technical_score.factors[0].get('rsi', 50) if reasoning_result.technical_score and reasoning_result.technical_score.factors else 50,
                                        'ema_alignment': 'bullish' if reasoning_result.technical_score and reasoning_result.technical_score.score > 50 else 'neutral',
                                        'volume_ratio': 1.0,
                                        'trend': 'up' if reasoning_result.technical_score and reasoning_result.technical_score.score > 50 else 'sideways'
                                    },
                                    fundamental_data={
                                        'score': reasoning_result.fundamental_score.score if reasoning_result.fundamental_score else 0
                                    },
                                    news_data={
                                        'score': reasoning_result.news_score.score if reasoning_result.news_score else 0
                                    },
                                    reasoning_result=reasoning_result
                                )

                                # Store markdown report
                                results["narrative_reports"][symbol] = narrative.to_markdown()
                                logger.info(f"Narrative report generated for {symbol}")

                            except Exception as e:
                                logger.warning(f"Failed to generate narrative for {symbol}: {e}")

                        logger.info(f"Signal: {symbol} - {reasoning_result.decision.value} "
                                   f"(score={reasoning_result.total_score:.1f}, conf={reasoning_result.confidence:.2f})")

                    except Exception as e:
                        logger.warning(f"Failed to analyze {symbol}: {e}")
                        continue

                logger.info(f"ReasoningEngine found {results['signals_found']} signals")

                # 3. Process each alert (if execute_trades=True)
                if execute_trades:
                    for alert in results["alerts"]:
                        trade_result = await self._process_alert(alert)
                        if trade_result.get("executed"):
                            results["trades_executed"] += 1
                        elif trade_result.get("rejected"):
                            results["trades_rejected"] += 1
                            # Track guardrails blocks for transparency
                            reason = trade_result.get("reason", "Unknown")
                            if "guardrail" in reason.lower() or "kill" in reason.lower() or "limit" in reason.lower():
                                results["guardrails_blocked"].append({
                                    "symbol": alert.get("symbol"),
                                    "reason": reason
                                })
                else:
                    # V4.7: Analysis mode only - no trade execution
                    logger.info(f"Analysis mode: {len(results['alerts'])} signals prepared (no execution)")
                    results["analysis_only"] = True
                    results["pending_signals"] = len(results["alerts"])

            except Exception as e:
                logger.error(f"ReasoningEngine error: {e}")
                self.state_manager.log_error(f"Reasoning failed: {e}")

        # FALLBACK: V3 Screener if ReasoningEngine not available
        elif self.market_screener:
            try:
                alerts = self.market_screener.screen_multiple_stocks(focus_symbols)
                results["signals_found"] = len(alerts)
                results["alerts"] = alerts

                logger.info(f"V3 Screener found {len(alerts)} signals")

                if execute_trades:
                    for alert in alerts:
                        trade_result = await self._process_alert(alert)
                        if trade_result.get("executed"):
                            results["trades_executed"] += 1
                        elif trade_result.get("rejected"):
                            results["trades_rejected"] += 1
                else:
                    # V4.7: Analysis mode only
                    logger.info(f"Analysis mode: {len(alerts)} signals prepared (no execution)")
                    results["analysis_only"] = True
                    results["pending_signals"] = len(alerts)

            except Exception as e:
                logger.error(f"Screening error: {e}")
                self.state_manager.log_error(f"Screening failed: {e}")

        logger.info(f"--- Trading Scan Complete: {results['trades_executed']} executed, "
                   f"{results['trades_rejected']} rejected ---")
        return results

    def _build_trade_context(self, symbol: str, reasoning_result) -> 'TradeContext':
        """
        Build a TradeContext from ReasoningResult for memory lookup.

        Args:
            symbol: Stock symbol
            reasoning_result: Result from ReasoningEngine

        Returns:
            TradeContext for memory comparison
        """
        from src.intelligence.trade_memory import TradeContext

        # Extract market context
        market_ctx = reasoning_result.market_context
        regime = "unknown"
        vix = 20.0
        spy_trend = "sideways"

        if market_ctx:
            regime = market_ctx.regime.value if hasattr(market_ctx, 'regime') else "unknown"
            vix = market_ctx.vix_level if hasattr(market_ctx, 'vix_level') else 20.0
            spy_trend = market_ctx.spy_trend if hasattr(market_ctx, 'spy_trend') else "sideways"

        # Extract scores from pillars
        tech_score = reasoning_result.technical_score
        sent_score = reasoning_result.sentiment_score

        return TradeContext(
            symbol=symbol,
            timestamp=reasoning_result.timestamp,
            market_regime=regime,
            vix_level=vix,
            spy_trend=spy_trend,
            sector_momentum=0.0,  # TODO: Get from sector analyzer
            rsi=tech_score.factors[0].get('rsi', 50) if tech_score.factors else 50,
            ema_alignment="bullish" if tech_score.score > 0 else "bearish",
            volume_ratio=1.0,  # TODO: Get from technical pillar
            breakout_strength=tech_score.score / 100 if tech_score.score > 0 else None,
            social_sentiment=sent_score.factors[0].get('social', 0) if sent_score.factors else 0,
            grok_sentiment=sent_score.factors[0].get('grok', 0) if sent_score.factors else 0,
            stocktwits_sentiment=sent_score.factors[0].get('stocktwits', 0) if sent_score.factors else 0,
            heat_score=0.0,
            news_sentiment=reasoning_result.news_score.score / 100,
            catalyst_present=reasoning_result.news_score.score > 30,
            decision_score=reasoning_result.total_score,
            confidence=reasoning_result.confidence
        )

    def _reasoning_to_alert(self, symbol: str, reasoning_result, memory_insight: Optional[Dict] = None) -> Dict:
        """
        Convert ReasoningResult to alert format compatible with _process_alert.

        Args:
            symbol: Stock symbol
            reasoning_result: Result from ReasoningEngine
            memory_insight: Optional insight from TradeMemory

        Returns:
            Alert dict in standard format
        """
        # Map decision to signal
        decision_to_signal = {
            "strong_buy": "STRONG_BUY",
            "buy": "BUY",
            "hold": "HOLD",
            "sell": "SELL",
            "strong_sell": "STRONG_SELL"
        }
        signal = decision_to_signal.get(reasoning_result.decision.value, "HOLD")

        # Convert confidence (0-1) to score (0-100)
        confidence_score = int(reasoning_result.confidence * 100)

        # Adjust confidence based on memory insight
        if memory_insight:
            # Boost confidence if historical win rate is high
            if memory_insight.get("historical_win_rate", 0) > 0.6:
                confidence_score = min(100, confidence_score + 5)
            # Reduce if win rate is low
            elif memory_insight.get("historical_win_rate", 0) < 0.4:
                confidence_score = max(0, confidence_score - 10)

        # Get price (would need to fetch from data)
        price = 0.0  # Will be fetched in _process_alert if needed

        return {
            "symbol": symbol,
            "signal": signal,
            "confidence_score": confidence_score,
            "price": price,
            "stop_loss": None,  # TODO: Calculate from technical analysis
            "reasoning_summary": reasoning_result.reasoning_summary,
            "total_score": reasoning_result.total_score,
            "pillar_scores": {
                "technical": reasoning_result.technical_score.score,
                "fundamental": reasoning_result.fundamental_score.score,
                "sentiment": reasoning_result.sentiment_score.score,
                "news": reasoning_result.news_score.score
            },
            "key_factors": reasoning_result.key_factors[:3],
            "risk_factors": reasoning_result.risk_factors,
            "memory_insight": memory_insight,
            "timestamp": reasoning_result.timestamp
        }

    async def _process_alert(self, alert: Dict) -> Dict:
        """
        Traite une alerte du screener.
        Valide avec les guardrails et exécute si approuvé.

        Args:
            alert: Alerte du screener

        Returns:
            Dict avec le résultat du traitement
        """
        symbol = alert.get("symbol")
        signal = alert.get("signal", "")
        confidence = alert.get("confidence_score", 0)
        price = alert.get("price", 0)
        stop_loss = alert.get("stop_loss")

        result = {
            "symbol": symbol,
            "signal": signal,
            "executed": False,
            "rejected": False,
            "reason": ""
        }

        # Ignorer les signaux faibles
        if signal not in ["STRONG_BUY", "BUY"]:
            result["rejected"] = True
            result["reason"] = f"Signal trop faible: {signal} (requis: BUY ou STRONG_BUY)"
            return result

        # V5.3 - ML Pillar Validation (veto power)
        if 'ML_VALIDATOR_AVAILABLE' in dir() and ML_VALIDATOR_AVAILABLE:
            try:
                scoring = alert.get("scoring_details", {})
                ml_result = validate_trade_ml(
                    symbol=symbol,
                    technical_score=scoring.get("technical", 50),
                    fundamental_score=scoring.get("fundamental", 50),
                    sentiment_score=scoring.get("sentiment", 50),
                    news_score=scoring.get("news", 50),
                    combined_score=confidence,
                    rsi=scoring.get("rsi", 50),
                    market_regime=alert.get("market_regime", "neutral")
                )
                result["ml_score"] = ml_result.get("ml_score", 50)
                result["ml_probability"] = ml_result.get("ml_probability", 0.5)
                if ml_result.get("ml_veto", False):
                    result["rejected"] = True
                    result["reason"] = f"ML Veto: {ml_result['ml_probability']*100:.0f}% win prob (min 30%)"
                    logger.info(f"[ML] Trade {symbol} vetoed: {ml_result['ml_probability']*100:.1f}%")
                    return result
                logger.info(f"[ML] Trade {symbol} approved: {ml_result['ml_probability']*100:.1f}%")
            except Exception as e:
                logger.warning(f"[ML] Validation error for {symbol}: {e}")


        # Ignorer si confiance trop basse
        if confidence < 55:
            result["rejected"] = True
            result["reason"] = f"Score confiance trop bas: {confidence}/100 (seuil: 55)"
            return result

        # Calculer la taille de position
        capital = self.state_manager.state.current_capital
        position_value = capital * 0.05  # 5% par défaut
        quantity = int(position_value / price) if price > 0 else 0

        if quantity < 1:
            result["rejected"] = True
            result["reason"] = f"Position trop petite: prix ${price:.2f} > capital disponible pour 1 action"
            return result

        # Créer la demande de trade
        trade_request = TradeRequest(
            symbol=symbol,
            action="BUY",
            quantity=quantity,
            price=price,
            order_type="LIMIT",
            current_capital=capital,
            position_value=quantity * price,
            daily_pnl=self.guardrails._daily_pnl,
            current_drawdown=self.state_manager.get_drawdown(),
            stop_loss=stop_loss,
            thesis=f"{signal} signal, confidence {confidence}/100"
        )

        # Valider avec les guardrails
        validation = self.guardrails.validate_trade(trade_request)

        if not validation.is_allowed():
            # V5.2 - Check portfolio rotation
            if self.rotation_manager and "position" in validation.reason.lower():
                logger.info(f"Portfolio full, checking rotation for {symbol}...")
                try:
                    rotation = await self.rotation_manager.check_and_rotate(
                        new_signal_symbol=symbol,
                        new_signal_score=confidence,
                        executor=self.ibkr_executor
                    )
                    if rotation.should_rotate:
                        logger.info(f"Rotation OK: sold {rotation.sell_symbol}")
                        validation = self.guardrails.validate_trade(trade_request)
                    else:
                        logger.info(f"No rotation: {rotation.sell_reason}")
                except Exception as e:
                    logger.warning(f"Rotation check failed: {e}")

            if not validation.is_allowed():
                result["rejected"] = True
                result["reason"] = f"Guardrail: {validation.reason}"
                logger.info(f"Trade rejected for {symbol}: {validation.reason}")
                return result

        # Exécuter selon le niveau de validation
        if validation.status == TradeValidation.APPROVED:
            # Exécution autonome
            executed = await self._execute_trade(trade_request)
            result["executed"] = executed
            result["reason"] = "Autonomous execution" if executed else "Execution failed"

        elif validation.status == TradeValidation.NEEDS_NOTIFICATION:
            # TODO: Envoyer notification et attendre
            result["rejected"] = True
            result["reason"] = "Notification required (not implemented)"

        elif validation.status == TradeValidation.NEEDS_APPROVAL:
            # TODO: Attendre validation manuelle
            result["rejected"] = True
            result["reason"] = "Manual approval required"

        return result

    async def _execute_trade(self, trade: TradeRequest) -> bool:
        """
        Exécute un trade via IBKR.

        Args:
            trade: La demande de trade validée

        Returns:
            True si exécuté avec succès
        """
        if not self.ibkr_executor:
            logger.warning("IBKR executor not available - simulating trade")
            # Simuler le trade pour les tests
            self._simulate_trade(trade)
            return True

        try:
            # Connecter si nécessaire
            if not self.ibkr_executor.is_connected():
                await self.ibkr_executor.connect()

            # Exécuter l'ordre
            order_id = await self.ibkr_executor.place_order(
                symbol=trade.symbol,
                action=trade.action,
                quantity=trade.quantity,
                order_type=trade.order_type,
                limit_price=trade.price if trade.order_type == "LIMIT" else None
            )

            if order_id:
                # Enregistrer la position
                position = Position(
                    symbol=trade.symbol,
                    quantity=trade.quantity,
                    entry_price=trade.price,
                    entry_date=datetime.now().isoformat(),
                    stop_loss=trade.stop_loss,
                    thesis=trade.thesis
                )
                self.state_manager.add_position(position)

                # V5.1 - Track with Adaptive Position Manager
                if self.position_manager:
                    self.position_manager.add_position(
                        symbol=trade.symbol,
                        entry_price=trade.price,
                        quantity=trade.quantity,
                        entry_score=trade.confidence_score or 50
                    )
                    logger.info(f"[V5.1] Position tracked: {trade.symbol}")

                # Enregistrer dans les guardrails
                self.guardrails.record_trade(
                    trade.symbol, trade.action, trade.position_value
                )

                # V5 - Record for adaptive learning
                if self.adaptive_scorer and hasattr(trade, 'pillar_scores'):
                    try:
                        self.adaptive_scorer.record_entry(
                            trade_id=f"{trade.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                            symbol=trade.symbol,
                            entry_price=trade.price,
                            pillar_scores=trade.pillar_scores or {},
                            entry_score=trade.confidence_score or 50,
                            decision=trade.action
                        )
                        logger.info(f"[V5] Trade recorded for adaptive learning: {trade.symbol}")
                    except Exception as e:
                        logger.warning(f"[V5] Failed to record trade for learning: {e}")

                logger.info(f"Trade executed: {trade.action} {trade.quantity} {trade.symbol} @ {trade.price}")
                return True

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            self.state_manager.log_error(f"Trade execution failed: {e}")

        return False

    def _simulate_trade(self, trade: TradeRequest):
        """Simule un trade (pour tests sans IBKR)"""
        position = Position(
            symbol=trade.symbol,
            quantity=trade.quantity,
            entry_price=trade.price,
            entry_date=datetime.now().isoformat(),
            stop_loss=trade.stop_loss,
            thesis=trade.thesis
        )
        self.state_manager.add_position(position)

        # V5.1 - Track with Adaptive Position Manager
        if self.position_manager:
            self.position_manager.add_position(
                symbol=trade.symbol,
                entry_price=trade.price,
                quantity=trade.quantity,
                entry_score=trade.confidence_score or 50
            )
            logger.info(f"[V5.1] Position tracked: {trade.symbol}")

        self.guardrails.record_trade(
            trade.symbol, trade.action, trade.position_value
        )

        # V5 - Record for adaptive learning (simulated trades count too)
        if self.adaptive_scorer and hasattr(trade, 'pillar_scores'):
            try:
                self.adaptive_scorer.record_entry(
                    trade_id=f"{trade.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    symbol=trade.symbol,
                    entry_price=trade.price,
                    pillar_scores=trade.pillar_scores or {},
                    entry_score=trade.confidence_score or 50,
                    decision=trade.action
                )
                logger.info(f"[V5] Simulated trade recorded for learning: {trade.symbol}")
            except Exception as e:
                logger.warning(f"[V5] Failed to record simulated trade: {e}")

        logger.info(f"Trade SIMULATED: {trade.action} {trade.quantity} {trade.symbol} @ {trade.price}")

    # =========================================================================
    # PHASE: AUDIT (20:00)
    # =========================================================================

    async def run_audit_phase(self) -> Dict:
        """
        Phase d'audit nocturne.
        Analyse les trades du jour, stocke dans TradeMemory, et génère des règles apprises.

        Returns:
            Dict avec les résultats de l'audit
        """
        logger.info("=== AUDIT PHASE STARTED ===")
        self.state_manager.set_phase(AgentPhase.AUDIT)

        results = {
            "timestamp": datetime.now().isoformat(),
            "trades_analyzed": 0,
            "trades_stored": 0,
            "patterns_extracted": 0,
            "new_rules_generated": 0,
            "rules": [],
            "memory_cleanup": {},
            "summary": {}
        }

        # 1. Get closed trades from today
        closed_trades = self.state_manager.get_closed_trades_today()
        results["trades_analyzed"] = len(closed_trades)
        logger.info(f"Analyzing {len(closed_trades)} closed trades from today")

        # 2. Store each trade in TradeMemory (RAG)
        if self.trade_memory and closed_trades:
            from src.intelligence.trade_memory import TradeContext, TradeOutcome

            for trade in closed_trades:
                try:
                    # Build context from trade record
                    context = TradeContext(
                        symbol=trade.symbol,
                        timestamp=trade.entry_date,
                        market_regime=trade.metadata.get("market_regime", "unknown"),
                        vix_level=trade.metadata.get("vix_level", 20.0),
                        spy_trend=trade.metadata.get("spy_trend", "sideways"),
                        sector_momentum=trade.metadata.get("sector_momentum", 0.0),
                        rsi=trade.metadata.get("rsi", 50),
                        ema_alignment=trade.metadata.get("ema_alignment", "neutral"),
                        volume_ratio=trade.metadata.get("volume_ratio", 1.0),
                        breakout_strength=trade.metadata.get("breakout_strength"),
                        social_sentiment=trade.metadata.get("social_sentiment", 0),
                        grok_sentiment=trade.metadata.get("grok_sentiment", 0),
                        stocktwits_sentiment=trade.metadata.get("stocktwits_sentiment", 0),
                        heat_score=trade.metadata.get("heat_score", 0),
                        news_sentiment=trade.metadata.get("news_sentiment", 0),
                        catalyst_present=trade.metadata.get("catalyst_present", False),
                        decision_score=trade.metadata.get("decision_score", 0),
                        confidence=trade.metadata.get("confidence", 0)
                    )

                    # Build outcome
                    outcome = TradeOutcome(
                        pnl_pct=trade.pnl_pct,
                        hold_days=trade.hold_days,
                        exit_type=trade.exit_type,
                        max_drawdown=trade.metadata.get("max_drawdown", 0),
                        max_gain=trade.metadata.get("max_gain", 0)
                    )

                    # Store in memory
                    stored = self.trade_memory.store_trade(
                        trade_id=trade.id,
                        context=context,
                        outcome=outcome
                    )
                    if stored:
                        results["trades_stored"] += 1

                except Exception as e:
                    logger.warning(f"Failed to store trade {trade.id} in memory: {e}")

            logger.info(f"Stored {results['trades_stored']} trades in memory")

        # V5 - Update adaptive learning with closed trades
        if self.adaptive_scorer and closed_trades:
            adaptive_recorded = 0
            for trade in closed_trades:
                try:
                    # Find matching trade entry in adaptive scorer
                    # Use symbol + entry_date as identifier
                    trade_id_prefix = f"{trade.symbol}_"

                    # Record exit for learning
                    self.adaptive_scorer.record_exit(
                        trade_id=trade_id_prefix + trade.entry_date.replace('-', '').replace(':', '').replace('T', '')[:14],
                        exit_price=trade.exit_price or trade.entry_price,
                        exit_reason=trade.exit_type or "unknown"
                    )
                    adaptive_recorded += 1
                except Exception as e:
                    logger.debug(f"[V5] Could not record exit for {trade.symbol}: {e}")

            if adaptive_recorded > 0:
                logger.info(f"[V5] Recorded {adaptive_recorded} trade exits for adaptive learning")

            # Get and log current adaptive weights
            weights = self.adaptive_scorer.weights.to_dict()
            results["adaptive_weights"] = weights
            logger.info(f"[V5] Current adaptive weights: {weights}")

        # 3. Extract patterns from recent trades
        if self.pattern_extractor and self.trade_memory:
            try:
                # Get all recent trades from memory
                recent_trades = await self.trade_memory.get_recent_trades(days=30)

                if len(recent_trades) >= 10:  # Need enough data for pattern extraction
                    extraction_result = await self.pattern_extractor.run_extraction(recent_trades)

                    results["patterns_extracted"] = len(extraction_result.candidates)
                    results["new_rules_generated"] = len(extraction_result.rules)
                    results["rules"] = [r.to_dict() for r in extraction_result.rules]

                    logger.info(f"Extracted {results['patterns_extracted']} patterns, "
                               f"generated {results['new_rules_generated']} rules")

                    # Save rules to file
                    if extraction_result.rules:
                        await self._save_learned_rules(extraction_result.rules)

            except Exception as e:
                logger.warning(f"Pattern extraction failed: {e}")

        # 4. Memory cleanup - remove stale data
        if self.trade_memory:
            try:
                cleanup_result = await self.trade_memory.cleanup_stale_memory()
                results["memory_cleanup"] = cleanup_result
                logger.info(f"Memory cleanup: removed {sum(cleanup_result.values())} stale trades")
            except Exception as e:
                logger.warning(f"Memory cleanup failed: {e}")

        # 4.1 V5.1 - Shadow Tracker: Verify outcomes and learn
        if self.shadow_tracker:
            try:
                # Verify all pending shadow signals against actual prices
                shadow_stats = await self.shadow_tracker.verify_outcomes()
                results["shadow_tracking"] = shadow_stats
                logger.info(f"[SHADOW] Verified {shadow_stats.get('verified', 0)} signals: "
                           f"{shadow_stats.get('winners', 0)} winners, {shadow_stats.get('losers', 0)} losers")

                # Generate learning insights from completed signals
                insights = self.shadow_tracker.generate_learning_insights()
                if insights:
                    results["learning_insights"] = len(insights)
                    logger.info(f"[SHADOW] Generated {len(insights)} learning insights")

                    # Apply weight adjustments automatically if sufficient confidence
                    self.shadow_tracker.apply_weight_adjustments(max_change=0.03)
                    new_weights = self.shadow_tracker.pillar_weights
                    results["shadow_weights"] = new_weights
                    logger.info(f"[SHADOW] Updated pillar weights: {new_weights}")

                # Get overall stats
                overall_stats = self.shadow_tracker.get_statistics()
                results["shadow_overall"] = overall_stats
                logger.info(f"[SHADOW] Overall: {overall_stats['total_signals']} signals, "
                           f"Win Rate: {overall_stats['win_rate']:.1f}%, "
                           f"Profit Factor: {overall_stats['profit_factor']:.2f}")

            except Exception as e:
                logger.warning(f"Shadow tracker audit failed: {e}")

        # 4.5 LLM Analysis via Strategy Composer (if losing trades)
        results["strategy_improvements"] = []
        losing_trades = [t for t in closed_trades if t.pnl_pct < 0] if closed_trades else []

        if self.strategy_composer and len(losing_trades) >= 2:
            try:
                # Get current market context
                market_context = {}
                if self.trend_discovery:
                    try:
                        market_ctx = await self.trend_discovery.get_market_context()
                        market_context = market_ctx if isinstance(market_ctx, dict) else {}
                    except Exception:
                        pass

                # Identify failure patterns from losing trades
                failure_patterns = self._extract_failure_patterns(losing_trades)

                # Build performance data
                performance_data = {
                    "total_trades": len(closed_trades) if closed_trades else 0,
                    "win_rate": results["summary"].get("win_rate", 0) if results.get("summary") else 0,
                    "total_pnl": results["summary"].get("total_pnl_pct", 0) if results.get("summary") else 0,
                    "losing_trades": len(losing_trades)
                }

                # Ask LLM for strategy improvements
                if performance_data["win_rate"] < 0.5 or performance_data["total_pnl"] < 0:
                    logger.info("Win rate or P&L below target, requesting LLM strategy analysis...")

                    # Convert trades to dict format for LLM
                    trades_for_llm = [
                        {
                            "symbol": t.symbol,
                            "pnl": t.pnl_pct,
                            "hold_days": t.hold_days,
                            "exit_type": t.exit_type,
                            **t.metadata
                        }
                        for t in losing_trades[:5]
                    ]

                    # Compose new strategy suggestion
                    new_strategy = await self.strategy_composer.compose_strategy(
                        performance_data=performance_data,
                        failure_patterns=failure_patterns,
                        market_context=market_context
                    )

                    results["strategy_improvements"].append({
                        "strategy_name": new_strategy.name,
                        "description": new_strategy.description,
                        "reasoning": new_strategy.reasoning[:500],
                        "status": "proposed"
                    })

                    # Save proposed strategy
                    await self._save_proposed_strategy(new_strategy)

                    logger.info(f"LLM proposed new strategy: {new_strategy.name}")

            except Exception as e:
                logger.warning(f"Strategy Composer analysis failed: {e}")

        # 5. Generate summary
        if results["trades_analyzed"] > 0:
            wins = sum(1 for t in closed_trades if t.pnl_pct > 0)
            total_pnl = sum(t.pnl_pct for t in closed_trades)
            results["summary"] = {
                "trades_today": results["trades_analyzed"],
                "wins": wins,
                "losses": results["trades_analyzed"] - wins,
                "win_rate": wins / results["trades_analyzed"] if results["trades_analyzed"] > 0 else 0,
                "total_pnl_pct": total_pnl,
                "avg_pnl_pct": total_pnl / results["trades_analyzed"]
            }

        self.state_manager.update_last_audit()

        logger.info(f"=== AUDIT COMPLETE: {results['trades_stored']} stored, "
                   f"{results['new_rules_generated']} new rules ===")
        return results

    async def _save_learned_rules(self, rules: List) -> None:
        """Save learned rules to JSON file"""
        import json
        rules_file = os.path.join(self.config.data_dir, "learned_guidelines.json")

        existing_rules = []
        if os.path.exists(rules_file):
            try:
                with open(rules_file, 'r') as f:
                    existing_rules = json.load(f)
            except Exception:
                pass

        # Add new rules
        for rule in rules:
            rule_dict = rule.to_dict() if hasattr(rule, 'to_dict') else rule
            rule_dict["learned_at"] = datetime.now().isoformat()
            existing_rules.append(rule_dict)

        # Save
        with open(rules_file, 'w') as f:
            json.dump(existing_rules, f, indent=2)

        logger.info(f"Saved {len(rules)} new rules to {rules_file}")

    def _extract_failure_patterns(self, losing_trades: List) -> List[str]:
        """Extract common patterns from losing trades"""
        patterns = []

        if not losing_trades:
            return patterns

        # Analyze exit types
        exit_types = {}
        for t in losing_trades:
            exit_type = t.exit_type if hasattr(t, 'exit_type') else 'unknown'
            exit_types[exit_type] = exit_types.get(exit_type, 0) + 1

        most_common_exit = max(exit_types, key=exit_types.get) if exit_types else None
        if most_common_exit == "stop_loss":
            patterns.append("Majority of losses hit stop loss - consider wider stops or better entry timing")
        elif most_common_exit == "max_hold_period":
            patterns.append("Many trades expired at max hold - consider shorter timeframes or momentum confirmation")

        # Analyze holding periods
        avg_hold = sum(t.hold_days for t in losing_trades if hasattr(t, 'hold_days')) / len(losing_trades)
        if avg_hold < 3:
            patterns.append(f"Short average hold ({avg_hold:.1f} days) - entries may be too early")
        elif avg_hold > 30:
            patterns.append(f"Long average hold ({avg_hold:.1f} days) - consider trailing stops")

        # Analyze metadata for common factors
        high_vix_losses = sum(1 for t in losing_trades
                             if hasattr(t, 'metadata') and t.metadata.get('vix_level', 0) > 25)
        if high_vix_losses > len(losing_trades) * 0.6:
            patterns.append("High VIX correlation with losses - reduce position size in high volatility")

        # RSI patterns
        overbought_entries = sum(1 for t in losing_trades
                                 if hasattr(t, 'metadata') and t.metadata.get('rsi', 50) > 70)
        if overbought_entries > len(losing_trades) * 0.4:
            patterns.append("Many entries at overbought RSI - wait for pullbacks")

        return patterns

    async def _save_proposed_strategy(self, strategy) -> None:
        """Save a proposed strategy to JSON file"""
        import json
        strategies_file = os.path.join(self.config.data_dir, "proposed_strategies.json")

        existing = []
        if os.path.exists(strategies_file):
            try:
                with open(strategies_file, 'r') as f:
                    existing = json.load(f)
            except Exception:
                pass

        # Add new strategy
        strategy_dict = strategy.to_dict() if hasattr(strategy, 'to_dict') else strategy
        strategy_dict["proposed_at"] = datetime.now().isoformat()
        existing.append(strategy_dict)

        # Keep only last 10 proposed strategies
        existing = existing[-10:]

        with open(strategies_file, 'w') as f:
            json.dump(existing, f, indent=2)

        logger.info(f"Saved proposed strategy: {strategy.name}")

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    async def run_main_loop(self):
        """
        Boucle principale de l'agent.
        Exécute les phases selon l'heure.
        """
        logger.info("Starting MarketAgent main loop...")
        self._running = True

        while self._running:
            try:
                now = datetime.now()
                current_time = now.time()
                market_status = self.get_market_status()

                # Vérifier le kill switch
                kill_active, _ = self.guardrails.is_kill_switch_active()
                if kill_active:
                    self.state_manager.set_phase(AgentPhase.PAUSED)
                    await asyncio.sleep(60)  # Vérifier toutes les minutes
                    continue

                # Phases selon l'heure
                if current_time >= self.config.discovery_time and current_time < self.config.analysis_time:
                    if self.state_manager.get_phase() != AgentPhase.DISCOVERY:
                        await self.run_discovery_phase()

                elif current_time >= self.config.analysis_time and current_time < self.config.market_open:
                    if self.state_manager.get_phase() != AgentPhase.ANALYSIS:
                        await self.run_analysis_phase()

                elif market_status == MarketStatus.OPEN:
                    # Scan périodique pendant les heures de marché
                    await self.run_trading_scan()
                    await asyncio.sleep(self.config.scan_interval_minutes * 60)
                    continue

                elif current_time >= self.config.audit_time:
                    if self.state_manager.get_phase() != AgentPhase.AUDIT:
                        await self.run_audit_phase()

                else:
                    self.state_manager.set_phase(AgentPhase.IDLE)

                # Attendre avant le prochain cycle
                await asyncio.sleep(60)  # Vérifier toutes les minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                self.state_manager.log_error(f"Main loop error: {e}")
                await asyncio.sleep(60)

        logger.info("MarketAgent main loop stopped")

    async def start(self):
        """Démarre l'agent"""
        await self.initialize()
        self._task = asyncio.create_task(self.run_main_loop())
        logger.info("MarketAgent started")

    async def stop(self):
        """Arrête l'agent"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.close()
        logger.info("MarketAgent stopped")

    # =========================================================================
    # MANUAL COMMANDS
    # =========================================================================

    async def run_now(self, phase: str = "all") -> Dict:
        """
        Exécute une phase manuellement.

        Args:
            phase: 'discovery', 'analysis', 'trading', 'audit', ou 'all'

        Returns:
            Dict avec les résultats
        """
        results = {}

        if phase in ["discovery", "all"]:
            results["discovery"] = await self.run_discovery_phase()

        if phase in ["analysis", "all"]:
            results["analysis"] = await self.run_analysis_phase()

        if phase in ["trading", "all"]:
            results["trading"] = await self.run_trading_scan()

        if phase in ["audit", "all"]:
            results["audit"] = await self.run_audit_phase()

        return results

    def get_status(self) -> Dict:
        """Retourne le statut complet de l'agent V4.1"""
        return {
            "agent": {
                "version": "4.1",
                "phase": self.state_manager.get_phase().value,
                "running": self._running,
                "market_status": self.get_market_status().value
            },
            "state": self.state_manager.get_summary(),
            "guardrails": self.guardrails.get_status(),
            "components": {
                # V4.1 Core
                "reasoning_engine": self.reasoning_engine is not None,
                "trade_memory": self.trade_memory is not None,
                "pattern_extractor": self.pattern_extractor is not None,
                # Intelligence
                "trend_discovery": self.trend_discovery is not None,
                "social_scanner": self.social_scanner is not None,
                "grok_scanner": self.grok_scanner is not None,
                "stock_discovery": self.stock_discovery is not None,
                # Execution
                "ibkr_executor": self.ibkr_executor is not None,
                # Fallback
                "market_screener_v3": self.market_screener is not None
            },
            "config": {
                "llm_model": self.config.llm_model,
                "memory_enabled": self.config.memory_enabled
            }
        }


# =========================================================================
# FACTORY
# =========================================================================

_agent_instance: Optional[MarketAgent] = None


def get_market_agent(config: Optional[OrchestratorConfig] = None) -> MarketAgent:
    """Retourne l'instance singleton de l'agent"""
    global _agent_instance

    if _agent_instance is None:
        _agent_instance = MarketAgent(config)

    return _agent_instance


# =========================================================================
# CLI
# =========================================================================

async def main():
    """Point d'entrée CLI"""
    import argparse

    parser = argparse.ArgumentParser(description="TradingBot V4 - Market Agent")
    parser.add_argument("--run", choices=["discovery", "analysis", "trading", "audit", "all"],
                       help="Run a specific phase")
    parser.add_argument("--status", action="store_true", help="Show agent status")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon (main loop)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    agent = get_market_agent()

    if args.status:
        await agent.initialize()
        import json
        print(json.dumps(agent.get_status(), indent=2))

    elif args.run:
        await agent.initialize()
        results = await agent.run_now(args.run)
        import json
        print(json.dumps(results, indent=2, default=str))
        await agent.close()

    elif args.daemon:
        await agent.start()
        try:
            # Attendre indéfiniment
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            await agent.stop()

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
