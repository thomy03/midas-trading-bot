"""
Market Agent Orchestrator - Cerveau Central du Robot de Trading V4.1
Coordonne tous les modules: Discovery, Reasoning, Decision, Execution, Learning.

V4.1 Changes:
- ReasoningEngine (4 Pillars) instead of V3 screener
- TradeMemory (RAG) for learning from past trades
- Dual LLM: Gemini for intelligence, Grok for sentiment only

V8.2 Sprint 4: Refactored into 3 delegate modules:
- discovery_analysis.py: Discovery + Analysis phases
- execution_coordinator.py: Trading scan + alert processing + trade execution
- learning_feedback.py: Audit phase + pattern extraction + strategy improvement
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

# V8.2 Sprint 4 - Delegate modules
from src.agents.discovery_analysis import DiscoveryAnalysisCoordinator
from src.agents.execution_coordinator import ExecutionCoordinator
from src.agents.learning_feedback import LearningFeedbackCoordinator

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

# V5.2 - Indicator Discovery (Auto-decouverte des indicateurs)
try:
    from src.intelligence.indicator_discovery import IndicatorDiscovery, get_indicator_discovery
    INDICATOR_DISCOVERY_AVAILABLE = True
except ImportError:
    INDICATOR_DISCOVERY_AVAILABLE = False
    IndicatorDiscovery = None

# V5.2 - Intuitive Reasoning (Raisonnement News -> Secteur -> Actions)
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

# V6 - Discovery Mode (Bull regime pepites)
try:
    from src.intelligence.discovery_mode import DiscoveryMode, get_discovery_mode, DiscoveryConfig
    DISCOVERY_MODE_AVAILABLE = True
except ImportError:
    DISCOVERY_MODE_AVAILABLE = False
    DiscoveryMode = None

# V6.1 - Orchestrator Memory (decision tracking & accuracy)
try:
    from src.learning.orchestrator_memory import get_orchestrator_memory, OrchestratorMemory
    ORCHESTRATOR_MEMORY_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_MEMORY_AVAILABLE = False
    OrchestratorMemory = None


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
    max_focus_symbols: int = 100
    max_watchlist_size: int = 500
    scan_interval_minutes: int = 15

    # LLM - Intelligence (Gemini 3 Flash via OpenRouter)
    llm_model: str = ""
    openrouter_api_key: Optional[str] = None

    # Grok - Sentiment only (X/Twitter access)
    grok_api_key: Optional[str] = None

    # IBKR
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    ibkr_client_id: int = 1

    # Donnees
    data_dir: str = "data"

    # RAG / Trade Memory
    memory_enabled: bool = True
    memory_db_path: str = "data/vector_store"
    similar_trades_k: int = 5


class MarketStatus(Enum):
    """Statut du marche"""
    PRE_MARKET = "pre_market"
    OPEN = "open"
    CLOSED = "closed"
    AFTER_HOURS = "after_hours"


# =========================================================================
# ORCHESTRATOR (Facade)
# =========================================================================

class MarketAgent:
    """
    Agent principal de trading.
    Coordonne Discovery, Screening, Decision et Execution.

    V8.2 Sprint 4: Delegates phase logic to specialized modules:
    - _discovery: DiscoveryAnalysisCoordinator
    - _execution: ExecutionCoordinator
    - _learning: LearningFeedbackCoordinator
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()

        # Read LLM config from environment if not set
        if not self.config.llm_model:
            self.config.llm_model = os.getenv('OPENROUTER_MODEL', 'google/gemini-3-flash-preview')
        if not self.config.openrouter_api_key:
            self.config.openrouter_api_key = os.getenv('OPENROUTER_API_KEY', '')
        if not self.config.grok_api_key:
            self.config.grok_api_key = os.getenv('GROK_API_KEY', '')

        # Core components
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

        # Components V4
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

        # V4.4 - Market Data Fetcher
        self.market_data_fetcher = MarketDataFetcher()

        # V4.5 - Hybrid Screener
        self.hybrid_screener: Optional[HybridScreener] = None

        # V4.5 - IBKR Watchdog
        self.ibkr_watchdog: Optional[IBKRWatchdog] = None

        # V4.5 - Async Position Sizer
        self.position_sizer: Optional[AsyncPositionSizer] = None

        # V5 - Adaptive Learning
        self.adaptive_scorer: Optional[AdaptiveScorer] = None
        if ADAPTIVE_LEARNING_AVAILABLE:
            try:
                self.adaptive_scorer = get_adaptive_scorer()
                logger.info(f"[V5] Adaptive learning enabled. Current weights: {self.adaptive_scorer.weights.to_dict()}")
            except Exception as e:
                logger.warning(f"[V5] Adaptive learning unavailable: {e}")

        # V5.1 - Shadow Tracker
        self.shadow_tracker: Optional[ShadowTracker] = None

        # V5.2 - Indicator Discovery
        self.indicator_discovery: Optional[IndicatorDiscovery] = None

        # V5.2 - Intuitive Reasoning
        self.intuitive_reasoning: Optional[IntuitiveReasoning] = None

        # V6.1 - Orchestrator Memory
        self.orchestrator_memory: Optional[OrchestratorMemory] = None
        if ORCHESTRATOR_MEMORY_AVAILABLE:
            try:
                self.orchestrator_memory = get_orchestrator_memory()
                stats = self.orchestrator_memory.get_accuracy_stats(30)
                logger.info(f"[V6.1] OrchestratorMemory loaded: {stats['total_verified']} verified decisions, accuracy={stats['accuracy']}%")
            except Exception as e:
                logger.warning(f"OrchestratorMemory not available: {e}")

        # Runtime state
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # V8.2 Sprint 4 - Delegate coordinators
        self._discovery = DiscoveryAnalysisCoordinator(self)
        self._execution = ExecutionCoordinator(self)
        self._learning = LearningFeedbackCoordinator(self)

        logger.info(f"MarketAgent V4.1 initialized: capital={self.config.initial_capital}")

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
                self.trade_memory = AdaptiveTradeMemory(db_path=self.config.memory_db_path)
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

        # 0.3 Strategy Composer
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

        # 0.4 Narrative Generator
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

        # 1. Trend Discovery
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

        # 2. Market Screener (FALLBACK V3)
        if not self.reasoning_engine:
            try:
                from src.screening.screener import market_screener
                self.market_screener = market_screener
                logger.info("MarketScreener V3 initialized (fallback)")
            except Exception as e:
                logger.warning(f"MarketScreener not available: {e}")

        # 3. Grok Scanner
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

        # 4. Stock Discovery
        try:
            from src.intelligence.stock_discovery import StockDiscovery
            self.stock_discovery = StockDiscovery()
            logger.info("StockDiscovery initialized")
        except Exception as e:
            logger.warning(f"StockDiscovery not available: {e}")

        # 5. IBKR Executor
        try:
            from src.execution.ibkr_executor import IBKRExecutor
            self.ibkr_executor = IBKRExecutor(
                host=self.config.ibkr_host,
                port=self.config.ibkr_port,
                client_id=self.config.ibkr_client_id
            )
            logger.info("IBKRExecutor initialized (not connected)")
        except Exception as e:
            logger.warning(f"IBKRExecutor not available: {e}")

        # 6. Hybrid Screener
        if HYBRID_SCREENER_AVAILABLE:
            try:
                ibkr_client = None
                if self.ibkr_executor and hasattr(self.ibkr_executor, 'ib'):
                    ibkr_client = self.ibkr_executor.ib
                self.hybrid_screener = await get_hybrid_screener(ibkr_client=ibkr_client)
                logger.info("HybridScreener initialized (FMP pre-screening + IBKR validation)")
            except Exception as e:
                logger.warning(f"HybridScreener not available: {e}")

        # 7. IBKR Watchdog
        if IBKR_WATCHDOG_AVAILABLE and self.ibkr_executor:
            try:
                ibkr_client = getattr(self.ibkr_executor, 'ib', None)
                if ibkr_client:
                    watchdog_config = WatchdogConfig(
                        host=self.config.ibkr_host,
                        port=self.config.ibkr_port,
                        client_id=self.config.ibkr_client_id,
                        max_daily_loss_pct=0.03
                    )
                    self.ibkr_watchdog = IBKRWatchdog(
                        ib=ibkr_client,
                        config=watchdog_config,
                        on_kill_switch=self._on_watchdog_kill_switch
                    )
                    logger.info("IBKRWatchdog initialized (auto-reconnect + kill switch)")
            except Exception as e:
                logger.warning(f"IBKRWatchdog not available: {e}")

        # 8. Async Position Sizer
        if ASYNC_POSITION_SIZER_AVAILABLE:
            try:
                ibkr_client = None
                if self.ibkr_executor and hasattr(self.ibkr_executor, 'ib'):
                    ibkr_client = self.ibkr_executor.ib
                self.position_sizer = AsyncPositionSizer(
                    total_capital=self.config.initial_capital,
                    account_currency='EUR',
                    ibkr_client=ibkr_client,
                    max_position_pct=0.25
                )
                logger.info(f"AsyncPositionSizer initialized: {self.config.initial_capital} EUR")
            except Exception as e:
                logger.warning(f"AsyncPositionSizer not available: {e}")

        # 9. Shadow Tracker
        if SHADOW_TRACKER_AVAILABLE:
            try:
                self.shadow_tracker = await get_shadow_tracker()
                stats = self.shadow_tracker.get_statistics()
                logger.info(f"[V5.1] ShadowTracker initialized: {stats['total_signals']} signals tracked, "
                           f"Win Rate: {stats['win_rate']:.1f}%")
            except Exception as e:
                logger.warning(f"ShadowTracker not available: {e}")

        # 10. Indicator Discovery
        if INDICATOR_DISCOVERY_AVAILABLE:
            try:
                self.indicator_discovery = await get_indicator_discovery()
                stats = self.indicator_discovery.get_statistics()
                logger.info(f"[V5.2] IndicatorDiscovery initialized: {stats['indicators_tracked']} indicators tracked")
            except Exception as e:
                logger.warning(f"IndicatorDiscovery not available: {e}")

        # 11. Intuitive Reasoning
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

        if self.ibkr_watchdog:
            try:
                await self.ibkr_watchdog.stop()
            except Exception as e:
                logger.error(f"Error stopping watchdog: {e}")

        if self.narrative_generator:
            await self.narrative_generator.close()

        if self.trend_discovery:
            await self.trend_discovery.close()

        if self.grok_scanner:
            await self.grok_scanner.close()

        if self.reasoning_engine:
            try:
                await self.reasoning_engine.close()
            except Exception as e:
                logger.error(f"Error closing reasoning engine: {e}")

        if self.stock_discovery:
            try:
                await self.stock_discovery.close()
            except Exception as e:
                logger.error(f"Error closing stock discovery: {e}")

        if self.hybrid_screener:
            try:
                await self.hybrid_screener.close()
            except Exception as e:
                logger.error(f"Error closing hybrid screener: {e}")

        if self.ibkr_executor:
            await self.ibkr_executor.disconnect()

        logger.info("MarketAgent closed")

    async def _on_watchdog_kill_switch(self, reason: str):
        """Callback when IBKR watchdog triggers kill switch."""
        logger.critical(f"KILL SWITCH TRIGGERED BY WATCHDOG: {reason}")
        self.guardrails.activate_kill_switch(reason)
        self.state_manager.log_error(f"Kill switch: {reason}")

    # =========================================================================
    # MARKET STATUS
    # =========================================================================

    def get_market_status(self) -> MarketStatus:
        """Retourne le statut actuel du marche"""
        now = datetime.now()
        current_time = now.time()

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
    # DELEGATED PHASE METHODS
    # =========================================================================

    async def run_fmp_prescreening(self) -> List[str]:
        """Delegate to DiscoveryAnalysisCoordinator."""
        return await self._discovery.run_fmp_prescreening()

    async def validate_with_ibkr(self, symbols: List[str]) -> List[str]:
        """Delegate to DiscoveryAnalysisCoordinator."""
        return await self._discovery.validate_with_ibkr(symbols)

    async def run_discovery_phase(self) -> Dict:
        """Delegate to DiscoveryAnalysisCoordinator."""
        return await self._discovery.run_discovery_phase()

    async def run_analysis_phase(self) -> Dict:
        """Delegate to DiscoveryAnalysisCoordinator."""
        return await self._discovery.run_analysis_phase()

    async def run_trading_scan(self, execute_trades: bool = True) -> Dict:
        """Delegate to ExecutionCoordinator."""
        return await self._execution.run_trading_scan(execute_trades)

    async def run_audit_phase(self) -> Dict:
        """Delegate to LearningFeedbackCoordinator."""
        return await self._learning.run_audit_phase()

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    async def run_main_loop(self):
        """Boucle principale de l'agent. Execute les phases selon l'heure."""
        logger.info("Starting MarketAgent main loop...")
        self._running = True

        while self._running:
            try:
                now = datetime.now()
                current_time = now.time()
                market_status = self.get_market_status()

                kill_active, _ = self.guardrails.is_kill_switch_active()
                if kill_active:
                    self.state_manager.set_phase(AgentPhase.PAUSED)
                    await asyncio.sleep(60)
                    continue

                if current_time >= self.config.discovery_time and current_time < self.config.analysis_time:
                    if self.state_manager.get_phase() != AgentPhase.DISCOVERY:
                        await self.run_discovery_phase()

                elif current_time >= self.config.analysis_time and current_time < self.config.market_open:
                    if self.state_manager.get_phase() != AgentPhase.ANALYSIS:
                        await self.run_analysis_phase()

                elif market_status == MarketStatus.OPEN:
                    await self.run_trading_scan()
                    await asyncio.sleep(self.config.scan_interval_minutes * 60)
                    continue

                elif current_time >= self.config.audit_time:
                    if self.state_manager.get_phase() != AgentPhase.AUDIT:
                        await self.run_audit_phase()

                else:
                    self.state_manager.set_phase(AgentPhase.IDLE)

                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                self.state_manager.log_error(f"Main loop error: {e}")
                await asyncio.sleep(60)

        logger.info("MarketAgent main loop stopped")

    async def start(self):
        """Demarre l'agent"""
        await self.initialize()
        self._task = asyncio.create_task(self.run_main_loop())
        logger.info("MarketAgent started")

    async def stop(self):
        """Arrete l'agent"""
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
        """Execute une phase manuellement."""
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
                "reasoning_engine": self.reasoning_engine is not None,
                "trade_memory": self.trade_memory is not None,
                "pattern_extractor": self.pattern_extractor is not None,
                "trend_discovery": self.trend_discovery is not None,
                "social_scanner": self.social_scanner is not None,
                "grok_scanner": self.grok_scanner is not None,
                "stock_discovery": self.stock_discovery is not None,
                "ibkr_executor": self.ibkr_executor is not None,
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
    """Point d'entree CLI"""
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
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            await agent.stop()

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
