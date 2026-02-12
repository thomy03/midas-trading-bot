"""
Live Loop - Boucle principale event-driven du trading bot
Orchestre tous les composants en temps réel.
"""

from src.agents.regime_adapter import get_regime_adapter, MarketRegime
from src.intelligence.discovery_mode import get_discovery_mode, DiscoveryMode

import asyncio
import logging
import signal
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import pytz

from .guardrails import TradingGuardrails, get_guardrails, GuardrailViolation
from .state import AgentState, StateManager, get_state_manager
from .strategy_evolver import StrategyEvolver, get_strategy_evolver, TradeResult
from .decision_journal import DecisionJournal, get_decision_journal

from ..intelligence.attention_manager import AttentionManager, get_attention_manager
from ..execution.execution_bridge import get_execution_bridge
from ..execution.paper_trader import get_paper_trader, PaperTrader  # kept for multi-strategy tracker compat
# Portfolio tracking
try:
    from ..utils.portfolio_tracker import record_snapshot
    PORTFOLIO_TRACKER_AVAILABLE = True
except ImportError:
    PORTFOLIO_TRACKER_AVAILABLE = False
    record_snapshot = None

# V7 - Risk management integration
try:
    from ..execution.defensive_manager import DefensiveManager, get_defensive_manager
    from ..execution.correlation_manager import CorrelationManager, CorrelationConfig, PortfolioPosition, get_correlation_manager
    from ..execution.position_manager import AdaptivePositionManager, get_position_manager
    from ..execution.position_sizer import PositionSizer, get_position_sizer
    RISK_MANAGEMENT_AVAILABLE = True
except ImportError:
    RISK_MANAGEMENT_AVAILABLE = False
    DefensiveManager = None
    CorrelationManager = None
    AdaptivePositionManager = None
    PositionSizer = None

# V4 - Multi-pillar reasoning (4 pillars: technique, fondamental, sentiment, news)
try:
    from .reasoning_engine import ReasoningEngine, get_reasoning_engine, DecisionType
    REASONING_ENGINE_AVAILABLE = True
except ImportError:
    REASONING_ENGINE_AVAILABLE = False
    ReasoningEngine = None
    DecisionType = None

logger = logging.getLogger(__name__)

# V8.1 Sprint 1C: Signal tracker for feedback loop
try:
    from src.learning import signal_tracker as _signal_tracker
    SIGNAL_TRACKER_AVAILABLE = True
except ImportError:
    SIGNAL_TRACKER_AVAILABLE = False

# V8.1 Sprint 1C: Liquidity filter
try:
    from src.utils.liquidity_filter import check_liquidity as _check_liquidity
    LIQUIDITY_FILTER_AVAILABLE = True
except ImportError:
    LIQUIDITY_FILTER_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class MarketSession(Enum):
    """Sessions de marché"""
    PRE_MARKET = "pre_market"       # 04:00 - 09:30 ET
    REGULAR = "regular"              # 09:30 - 16:00 ET
    AFTER_HOURS = "after_hours"      # 16:00 - 20:00 ET
    CLOSED = "closed"


@dataclass
class LiveLoopConfig:
    """Configuration de la boucle live"""
    # Polling intervals (secondes)
    heat_poll_interval: int = 300      # Polling des sources de chaleur
    screening_interval: int = 300     # Screening (5 min)
    health_check_interval: int = 600  # Health check (10 min)

    # Timezone
    timezone: str = "Europe/Paris"

    # Heures de trading
    pre_market_start: time = time(8, 0)
    market_open: time = time(9, 0)
    market_close: time = time(17, 30)
    after_hours_end: time = time(22, 0)

    # Mode
    trade_pre_market: bool = False
    trade_after_hours: bool = False
    analyze_when_closed: bool = False  # Analyse meme hors marche
    paper_trading: bool = True

    # V7: Risk management
    enable_defensive_mode: bool = True
    enable_correlation_checks: bool = True
    enable_position_monitor: bool = True
    daily_loss_limit_pct: float = 0.03     # 3% daily loss -> pause
    max_drawdown_pct: float = 0.15         # 15% drawdown -> max defensive
    consecutive_loss_limit: int = 5        # 5 losses -> defensive mode

    # Limites
    max_concurrent_screens: int = 3
    max_heat_sources: int = 5

    # Notifications
    notification_callback: Optional[Callable] = None


@dataclass
class LoopMetrics:
    """Métriques de la boucle"""
    started_at: Optional[datetime] = None
    cycles_completed: int = 0
    screens_performed: int = 0
    signals_found: int = 0
    trades_executed: int = 0
    errors_count: int = 0
    last_cycle_time: Optional[datetime] = None
    avg_cycle_duration_ms: float = 0


# =============================================================================
# LIVE LOOP
# =============================================================================

class LiveLoop:
    """
    Boucle principale event-driven.

    Orchestre:
    - Heat detection (quoi est chaud maintenant)
    - Attention management (sur quoi se concentrer)
    - Screening (analyse technique)
    - Decision making (journalisé)
    - Strategy evolution (auto-apprentissage)
    """

    def __init__(self, config: Optional[LiveLoopConfig] = None):
        self.config = config or LiveLoopConfig()
        # V6.1: Regime-adaptive strategy
        self.regime_adapter = get_regime_adapter()
        self.current_regime = MarketRegime.RANGE
        self.discovery_mode = get_discovery_mode()

        # V7: Risk management components
        self.defensive_manager: Optional[DefensiveManager] = None
        self.correlation_manager: Optional[CorrelationManager] = None
        self.position_monitor: Optional[AdaptivePositionManager] = None
        self._position_monitor_task: Optional[asyncio.Task] = None
        self._consecutive_losses: int = 0

        # V8: Intelligence Orchestrator
        self._intelligence_orchestrator = None

        # Components (initialisés dans initialize())
        self.guardrails: object = None
        self.state: Optional[AgentState] = None
        self.attention_manager: Optional[AttentionManager] = None
        self.strategy_evolver: Optional[StrategyEvolver] = None
        self.decision_journal: Optional[DecisionJournal] = None

        # State
        self._running = False
        self._paused = False
        self._shutdown_event = asyncio.Event()
        self._current_session = MarketSession.CLOSED

        # Tasks
        self._main_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None

        # Metrics
        self._metrics = LoopMetrics()

        # Timezone
        self._tz = pytz.timezone(self.config.timezone)

        # Callbacks
        self._on_signal_callback: Optional[Callable] = None
        self._on_trade_callback: Optional[Callable] = None

        logger.info("LiveLoop created")

    async def initialize(self):
        """Initialise tous les composants"""
        logger.info("Initializing LiveLoop components...")

        # Core components
        self.guardrails = get_guardrails()
        self.state_manager = get_state_manager()
        self.state = self.state_manager.state

        # Intelligence components
        self.attention_manager = await get_attention_manager()

        # Learning components
        self.strategy_evolver = await get_strategy_evolver()
        self.decision_journal = await get_decision_journal()

        # Setup notification callbacks
        if self.config.notification_callback:
            self.strategy_evolver.set_notification_callback(
                self.config.notification_callback
            )

        # V7: Initialize risk management
        if RISK_MANAGEMENT_AVAILABLE:
            self.defensive_manager = get_defensive_manager()
            self.correlation_manager = get_correlation_manager()
            self.position_monitor = get_position_manager()
            logger.info("V7 Risk management initialized")

        # V8: Initialize Intelligence Orchestrator
        try:
            from src.intelligence.intelligence_orchestrator import IntelligenceOrchestrator
            from src.intelligence.gemini_client import GeminiClient
            from src.intelligence.market_context import get_market_context_analyzer

            gemini = GeminiClient()
            await gemini.initialize()

            # Gather available sources (graceful if some are missing)
            grok_scanner = None
            news_fetcher = None
            trend_discovery = None
            if os.environ.get("DISABLE_LLM", "false").lower() != "true":
                try:
                    from src.intelligence.grok_scanner import GrokScanner
                    grok_scanner = GrokScanner()
                    await grok_scanner.initialize()
                except Exception:
                    logger.debug("Grok scanner not available for V8 orchestrator")
            else:
                logger.info("LLM disabled - skipping Grok scanner for V8 orchestrator")
            try:
                from src.intelligence.news_fetcher import NewsFetcher
                news_fetcher = NewsFetcher()
            except Exception:
                logger.debug("News fetcher not available for V8 orchestrator")
            try:
                from src.intelligence.trend_discovery import TrendDiscovery
                trend_discovery = TrendDiscovery()
                await trend_discovery.initialize()
            except Exception:
                logger.debug("Trend discovery not available for V8 orchestrator")

            market_ctx = get_market_context_analyzer()

            # Portfolio symbols from attention manager or config
            portfolio_syms = []
            if hasattr(self, 'attention_manager') and self.attention_manager:
                try:
                    portfolio_syms = list(self.attention_manager.get_tracked_symbols())
                except Exception:
                    pass

            self._intelligence_orchestrator = IntelligenceOrchestrator(
                gemini_client=gemini,
                grok_scanner=grok_scanner,
                news_fetcher=news_fetcher,
                trend_discovery=trend_discovery,
                market_context=market_ctx,
                portfolio_symbols=portfolio_syms
            )
            logger.info("V8 Intelligence Orchestrator initialized")
        except Exception as e:
            logger.info("V8 Intelligence Orchestrator not available: %s", e)
            self._intelligence_orchestrator = None

        logger.info("LiveLoop initialized successfully")

    def set_signal_callback(self, callback: Callable):
        """Callback appelé quand un signal est trouvé"""
        self._on_signal_callback = callback

    def set_trade_callback(self, callback: Callable):
        """Callback appelé quand un trade est exécuté"""
        self._on_trade_callback = callback

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------

    async def start(self):
        """Démarre la boucle principale"""
        if self._running:
            logger.warning("LiveLoop already running")
            return

        self._running = True
        self._shutdown_event.clear()
        self._metrics.started_at = datetime.now()

        logger.info("🚀 Starting LiveLoop...")

        # Enregistrer les signal handlers
        self._setup_signal_handlers()

        # Démarrer les tâches parallèles
        self._main_task = asyncio.create_task(self._main_loop())
        self._health_task = asyncio.create_task(self._health_loop())

        # V7: Position monitor loop
        if self.config.enable_position_monitor and self.position_monitor:
            self._position_monitor_task = asyncio.create_task(self._position_monitor_loop())

        tasks = [self._main_task, self._health_task]
        if self._position_monitor_task:
            tasks.append(self._position_monitor_task)

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("LiveLoop tasks cancelled")
        finally:
            self._running = False
            logger.info("LiveLoop stopped")

    async def stop(self):
        """Arrête la boucle proprement"""
        logger.info("Stopping LiveLoop...")
        self._running = False
        self._shutdown_event.set()

        # Annuler les tâches
        for task in [self._main_task, self._health_task, self._position_monitor_task]:
            if task and not task.done():
                task.cancel()

        # Fermer les composants
        await self._cleanup()

    async def pause(self):
        """Met en pause le trading (continue le monitoring)"""
        self._paused = True
        logger.info("⏸️ LiveLoop paused (monitoring continues)")

        if self.config.notification_callback:
            await self.config.notification_callback(
                "⏸️ Trading en pause - Le monitoring continue"
            )

    async def resume(self):
        """Reprend le trading"""
        self._paused = False
        logger.info("▶️ LiveLoop resumed")

        if self.config.notification_callback:
            await self.config.notification_callback(
                "▶️ Trading repris"
            )

    def _setup_signal_handlers(self):
        """Configure les handlers de signal système"""
        try:
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
        except NotImplementedError:
            # Windows ne supporte pas add_signal_handler
            pass

    # -------------------------------------------------------------------------
    # MAIN TRADING LOOP
    # -------------------------------------------------------------------------

    async def _main_loop(self):
        """Boucle principale de trading"""
        logger.info("Main trading loop started")

        while self._running and not self._shutdown_event.is_set():
            cycle_start = datetime.now()

            # V6.1: Detect market regime and adapt strategy
            try:
                regime, confidence = self.regime_adapter.detect_regime()
                self.current_regime = regime
                cfg = self.regime_adapter.get_config(regime)
                logger.info(f"🎯 REGIME: {regime.value.upper()} | Small caps: {cfg.allow_small_caps} | Defensive: {cfg.prefer_defensive}")
                # V6.1: Activate Discovery Mode in BULL regime
                if regime.value == "bull" and self.discovery_mode:
                    logger.info("🔍 BULL REGIME - Discovery Mode ACTIVE (searching for hidden gems)")
            except Exception as e:
                logger.warning(f"Regime detection failed: {e}")

            try:
                # 1. Vérifier la session de marché
                self._current_session = self._get_market_session()

                # Allow analysis even when market closed (but no trading)
                if not self._should_trade() and self._current_session == MarketSession.CLOSED:
                    # Still do analysis for preparation
                    if self.config.analyze_when_closed:
                        pass  # Continue to analysis
                    else:
                        await asyncio.sleep(60)
                        continue
                elif not self._should_trade():
                    await asyncio.sleep(60)  # Attendre 1 minute
                    continue

                # 2. Vérifier les guardrails
                if not await self._check_guardrails():
                    logger.warning("Guardrails check failed, pausing...")
                    # Track portfolio value
                    if PORTFOLIO_TRACKER_AVAILABLE:
                        try:
                            record_snapshot()
                        except Exception as e:
                            logger.warning(f"Portfolio tracking failed: {e}")
                    await self.pause()
                    await asyncio.sleep(300)  # Attendre 5 min
                    continue

                # 3. Si en pause, juste monitorer
                if self._paused:
                    await asyncio.sleep(self.config.screening_interval)
                    continue

                # 4. Obtenir les symboles en focus
                focus_topics = await self.attention_manager.update_focus()

                if not focus_topics:
                    logger.debug("No focus topics, waiting...")
                    await asyncio.sleep(60)
                    continue

                # 5. Screener les symboles prioritaires
                symbols_to_screen = self.attention_manager.get_symbols_for_screening(
                    limit=self.config.max_concurrent_screens
                )

                if symbols_to_screen:
                    await self._screen_symbols(symbols_to_screen)

                # 6. Reset le cycle
                await self.attention_manager.reset_cycle()

                # Métriques
                self._metrics.cycles_completed += 1
                self._metrics.last_cycle_time = datetime.now()

                cycle_duration = (datetime.now() - cycle_start).total_seconds() * 1000
                self._update_avg_cycle_duration(cycle_duration)

            except GuardrailViolation as e:
                logger.error(f"Guardrail violation: {e}")
                await self.pause()
                await self._notify(f"🚨 Guardrail violation: {e}")

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self._metrics.errors_count += 1
                await asyncio.sleep(30)  # Court délai avant retry

            # Attendre avant le prochain cycle
            await asyncio.sleep(self.config.screening_interval)

        logger.info("Main trading loop ended")

    async def _screen_symbols(self, symbols: List[str]):
        """Screene une liste de symboles avec les 4 piliers"""
        
        # V4 - Use ReasoningEngine if available
        if REASONING_ENGINE_AVAILABLE:
            try:
                reasoning_engine = await get_reasoning_engine()
                
                for symbol in symbols:
                    try:
                        await self.attention_manager.mark_screened(symbol)
                        
                        # V8.1: Liquidity filter - skip illiquid symbols
                        if LIQUIDITY_FILTER_AVAILABLE:
                            try:
                                liq = _check_liquidity(symbol)
                                if not liq['liquid']:
                                    logger.debug(f"[LIQUIDITY] Skipping {symbol}: {liq.get('reason')}")
                                    continue
                            except Exception as e:
                                logger.debug(f"[LIQUIDITY] Error for {symbol}: {e}")
                        
                        # Analyse 4 piliers
                        result = await reasoning_engine.analyze(symbol)
                        
                        if result and result.decision in [DecisionType.BUY, DecisionType.STRONG_BUY]:
                            # Get current price for trade execution
                            try:
                                import yfinance as yf
                                ticker = yf.Ticker(symbol)
                                current_price = ticker.fast_info.get('lastPrice') or ticker.fast_info.get('regularMarketPrice', 0)
                            except Exception as e:
                                logger.warning(f"Could not get price for {symbol}: {e}")
                                current_price = 0
                            
                            alert = {
                                'symbol': symbol,
                                'confidence_score': result.total_score,
                                'confidence_signal': result.decision.value.upper(),
                                'current_price': current_price,
                                'pillar_technical': result.technical_score,
                                'pillar_fundamental': result.fundamental_score,
                                'pillar_sentiment': result.sentiment_score,
                                'pillar_news': result.news_score,
                                'reasoning': result.reasoning_summary,
                                'ml_score': result.ml_score,
                                'regime': self.current_regime.value if hasattr(self.current_regime, 'value') else '',
                                'key_factors': [str(f) for f in (result.key_factors or [])[:5]],
                                'timestamp': result.timestamp
                            }
                            self._metrics.signals_found += 1
                            await self.attention_manager.mark_signal_found(symbol)
                            await self._process_signal(alert)
                        
                        self._metrics.screens_performed += 1
                        
                    except Exception as e:
                        import traceback; logger.error(f"Error screening {symbol} with ReasoningEngine: {e}\n{traceback.format_exc()}")
                        
                return
            except Exception as e:
                logger.warning(f"ReasoningEngine failed, falling back to MarketScreener: {e}")
        
        # Fallback to simple screener
        from ..screening.screener import MarketScreener
        screener = MarketScreener()

        for symbol in symbols:
            try:
                # Marquer comme en cours de screening
                await self.attention_manager.mark_screened(symbol)

                # Obtenir les paramètres actuels du strategy evolver
                params = self.strategy_evolver.get_current_params()

                # Screener avec les paramètres ajustés
                alert = screener.screen_single_stock(symbol)

                if alert:
                    self._metrics.signals_found += 1
                    await self.attention_manager.mark_signal_found(symbol)
                    await self._process_signal(alert)

                self._metrics.screens_performed += 1

            except Exception as e:
                logger.error(f"Error screening {symbol}: {e}")

    async def _process_signal(self, alert: Dict):
        """Traite un signal détecté"""
        symbol = alert.get('symbol', 'UNKNOWN')
        confidence = alert.get('confidence_score', 0)
        signal_type = alert.get('confidence_signal', 'UNKNOWN')
        
        logger.info(f"📊 Signal detected: {symbol} - {signal_type} (score: {confidence})")
        
        # V8.1: Evaluate signal against all 4 strategy profiles
        try:
            from src.agents.multi_strategy_tracker import get_multi_tracker
            tracker = get_multi_tracker()
            pillar_scores = {
                "technical": float(getattr(alert.get("pillar_technical", 0), "score", alert.get("pillar_technical", 0)) or 0),
                "fundamental": float(getattr(alert.get("pillar_fundamental", 0), "score", alert.get("pillar_fundamental", 0)) or 0),
                "sentiment": float(getattr(alert.get("pillar_sentiment", 0), "score", alert.get("pillar_sentiment", 0)) or 0),
                "news": float(getattr(alert.get("pillar_news", 0), "score", alert.get("pillar_news", 0)) or 0),
                "ml": float(alert.get("ml_score", 0) or 0),
            }
            ml_score = float(alert.get("ml_score", 0) or 0)
            current_price = float(alert.get("current_price", 0) or 0)
            atr = float(alert.get("atr", current_price * 0.02) or current_price * 0.02)
            
            if current_price > 0:
                multi_results = tracker.evaluate_signal(
                    symbol=symbol,
                    total_score=confidence,
                    ml_score=ml_score,
                    pillar_scores=pillar_scores,
                    current_price=current_price,
                    atr=atr,
                    reasoning=str(alert.get("reasoning", ""))[:500],
                    regime=str(alert.get("regime", "")),
                    key_factors=[str(f) for f in alert.get("key_factors", [])[:5]],
                )
                accepted = [k for k, v in multi_results.items() if v.startswith("accepted")]
                if accepted:
                    logger.info(f"   🧪 Multi-strategy: {len(accepted)}/4 accepted: {multi_results}")
                else:
                    logger.info(f"   🧪 Multi-strategy: 0/4 accepted")
        except Exception as e:
            logger.warning(f"Multi-strategy tracker error: {e}")
        
        # Log detailed scores (4 piliers)
        if 'pillar_technical' in alert:
            logger.info(f"   📊 4 PILIERS: Tech={alert.get('pillar_technical',0)}/25, Fonda={alert.get('pillar_fundamental',0)}/25, Sentiment={alert.get('pillar_sentiment',0)}/25, News={alert.get('pillar_news',0)}/25")
        elif 'score_ema' in alert:
            # Fallback to old L1/L2 format
            logger.info(f"   📈 Piliers L1: EMA={alert.get('score_ema',0)}/20, Support={alert.get('score_support',0)}/20, RSI={alert.get('score_rsi',0)}/25, Fresh={alert.get('score_freshness',0)}/20, Vol={alert.get('score_volume',0)}/15")
            if 'l2_total_score' in alert:
                logger.info(f"   🏢 Piliers L2: Health={alert.get('l2_health_score',0)}/20, Context={alert.get('l2_context_score',0)}/10, Sentiment={alert.get('l2_sentiment_score',0)}/30 | ELITE={alert.get('l2_is_elite', False)}")

        # Filtrer par score minimum (BUY >= 55, STRONG_BUY >= 75)
        min_score = self.config.min_confidence_score if hasattr(self.config, 'min_confidence_score') else 55
        if confidence < min_score:
            logger.info(f"\u23ed\ufe0f Signal {symbol} skipped: score {confidence} < {min_score}")
            # V8.1: Track rejected signal for feedback loop
            if SIGNAL_TRACKER_AVAILABLE:
                try:
                    _signal_tracker.record({
                        'symbol': symbol, 'score': confidence,
                        'decision': 'REJECTED', 'regime': alert.get('regime', ''),
                        'sector': alert.get('sector', ''),
                        'rejection_reason': f'score {confidence} < {min_score}',
                        'key_factors': alert.get('key_factors', []),
                    })
                except Exception:
                    pass
            return
            
        # V8.1: Track taken signal for feedback loop
        if SIGNAL_TRACKER_AVAILABLE:
            try:
                _signal_tracker.record({
                    'symbol': symbol, 'score': confidence,
                    'decision': signal_type, 'regime': alert.get('regime', ''),
                    'sector': alert.get('sector', ''),
                    'key_factors': alert.get('key_factors', []),
                })
            except Exception:
                pass

        # Callback externe
        if self._on_signal_callback:
            await self._on_signal_callback(alert)

        # Vérifier si on peut trader
        if self._paused:
            logger.info(f"Trading paused, signal logged but not executed: {symbol}")
            return

        # Vérifier les guardrails
        try:
            await self.guardrails.validate_trade_request(alert)
        except GuardrailViolation as e:
            logger.warning(f"Trade blocked by guardrail: {e}")
            return

        # V7: Sector-regime score adjustment
        try:
            from src.agents.sector_regime_scorer import get_sector_regime_scorer
            regime_str = self.current_regime.value.upper() if hasattr(self.current_regime, 'value') else 'RANGE'
            sector_scorer = get_sector_regime_scorer()
            sector_adj = sector_scorer.get_adjustment(
                symbol, regime=regime_str,
                sector=alert.get('sector'),
                market_cap=alert.get('market_cap'),
            )
            if abs(sector_adj) > 0:
                confidence = max(0, min(100, confidence + sector_adj))
                alert['confidence_score'] = confidence
                logger.debug(f"[SECTOR] {symbol} score adjusted by {sector_adj:+.1f} -> {confidence:.1f}")
        except Exception as e:
            logger.debug(f"Sector scoring not available: {e}")

        # V8: Intelligence Orchestrator overlay
        if self._intelligence_orchestrator is not None:
            try:
                brief = await self._intelligence_orchestrator.get_brief()
                # Dump brief to JSON for dashboard API
                try:
                    import json as _json
                    _brief_path = os.path.join("data", "intelligence_brief.json")
                    with open(_brief_path, "w") as _bf:
                        _json.dump({
                            "reasoning_summary": brief.reasoning_summary if hasattr(brief, 'reasoning_summary') else "",
                            "portfolio_alerts": brief.portfolio_alerts[:5] if hasattr(brief, 'portfolio_alerts') else [],
                            "market_events": getattr(brief, 'market_events', [])[:10],
                            "megatrends": getattr(brief, 'megatrends', [])[:5],
                            "timestamp": datetime.now().isoformat(),
                        }, _bf, indent=2, default=str)
                except Exception:
                    pass
                intel_adj = self._intelligence_orchestrator.get_symbol_adjustment(symbol, brief)
                if intel_adj != 0:
                    old_conf = confidence
                    confidence = max(0, min(100, confidence + intel_adj))
                    alert['confidence_score'] = confidence
                    logger.info(
                        "V8 Intel: %s %+.1f pts (%.0f -> %.0f) | %s",
                        symbol, intel_adj, old_conf, confidence,
                        brief.reasoning_summary[:80] if brief.reasoning_summary else ''
                    )
                # Log portfolio alerts
                if hasattr(self, 'decision_journal') and self.decision_journal:
                    for pa in brief.portfolio_alerts[:3]:
                        try:
                            await self.decision_journal.log_decision(
                                symbol='PORTFOLIO', action='INTEL_ALERT',
                                signal_data={'alert': pa}
                            )
                        except Exception:
                            pass
            except Exception as e:
                logger.debug(f"Intelligence orchestrator not available: {e}")

        # V7: Defensive mode check
        if self.defensive_manager and self.config.enable_defensive_mode and RISK_MANAGEMENT_AVAILABLE:
            try:
                regime_str = self.current_regime.value.upper() if hasattr(self.current_regime, 'value') else 'RANGE'
                level = self.defensive_manager.get_defensive_level(
                    regime=regime_str,
                    consecutive_losses=self._consecutive_losses
                )
                allowed, reason = self.defensive_manager.should_enter_trade(
                    score=confidence,
                    current_positions=len(get_paper_trader().portfolio.positions) if hasattr(get_paper_trader(), 'portfolio') else 0
                )
                if not allowed:
                    logger.info(f"[DEFENSIVE] Trade {symbol} blocked: {reason}")
                    return
            except Exception as e:
                logger.warning(f"Defensive check error: {e}")

        # V7: Correlation check
        if self.correlation_manager and self.config.enable_correlation_checks and RISK_MANAGEMENT_AVAILABLE:
            try:
                paper_trader = get_paper_trader()
                if hasattr(paper_trader, 'portfolio') and paper_trader.portfolio.positions:
                    existing = [
                        PortfolioPosition(
                            symbol=p.symbol,
                            sector=getattr(p, 'sector', 'Unknown'),
                            value=p.quantity * (p.current_price if hasattr(p, 'current_price') and p.current_price else p.entry_price)
                        )
                        for p in paper_trader.portfolio.positions
                    ]
                    total_value = sum(p.value for p in existing) + paper_trader.portfolio.cash
                    allowed, reason = self.correlation_manager.check_new_position(
                        new_symbol=symbol,
                        existing_positions=existing,
                        total_portfolio_value=total_value
                    )
                    if not allowed:
                        logger.info(f"[CORRELATION] Trade {symbol} blocked: {reason}")
                        return
            except Exception as e:
                logger.warning(f"Correlation check error: {e}")

        # Journaliser la décision
        decision = await self.decision_journal.log_decision(
            symbol=symbol,
            action="BUY",
            signal_data=alert
        )

        # Paper trading: exécuter via PaperTrader
        if self.config.paper_trading:
            await self._execute_trade(alert, decision)
        else:
            logger.warning(f"Live trading not implemented, signal logged only: {symbol}")

    async def _execute_trade(self, alert: Dict, decision):
        """Exécute un trade (paper trading)"""
        symbol = alert.get('symbol')
        price = alert.get('current_price', 0)
        score = alert.get('confidence_score', 0)
        
        if not price or price <= 0:
            logger.warning(f"Cannot execute trade for {symbol}: no valid price")
            return
        
        # Get paper trader
        bridge = get_execution_bridge()
        
        # Extract pillar scores and reasoning from alert
        pillar_technical = alert.get('pillar_technical')
        pillar_fundamental = alert.get('pillar_fundamental')
        pillar_sentiment = alert.get('pillar_sentiment')
        pillar_news = alert.get('pillar_news')
        reasoning = alert.get('reasoning')
        
        # Convert PillarScore objects to floats if needed
        if hasattr(pillar_technical, 'score'):
            pillar_technical = pillar_technical.score
        if hasattr(pillar_fundamental, 'score'):
            pillar_fundamental = pillar_fundamental.score
        if hasattr(pillar_sentiment, 'score'):
            pillar_sentiment = pillar_sentiment.score
        if hasattr(pillar_news, 'score'):
            pillar_news = pillar_news.score
        
        # Fetch company info
        company_name = None
        sector = None
        industry = None
        try:
            import yfinance as yf
            ticker_info = yf.Ticker(symbol)
            info = ticker_info.info
            company_name = info.get('shortName') or info.get('longName')
            sector = info.get('sector')
            industry = info.get('industry')
        except Exception as e:
            logger.warning(f"Could not fetch company info for {symbol}: {e}")
        
        # Open position with full analysis
        position = bridge.open_position(
            symbol=symbol,
            price=price,
            score=score,
            decision_type=str(decision.decision.value) if hasattr(decision, 'decision') else "BUY",
            pillar_technical=pillar_technical,
            pillar_fundamental=pillar_fundamental,
            pillar_sentiment=pillar_sentiment,
            pillar_news=pillar_news,
            reasoning=reasoning,
            company_name=company_name,
            sector=sector,
            industry=industry
        )
        
        # V5.5 - PORTFOLIO ROTATION: If not enough cash, try to upgrade portfolio
        if position is None and score > 0:
            try:
                positions = paper_trader.portfolio.positions
                if positions:
                    weakest = min(positions, key=lambda p: p.score_at_entry or 0)
                    weakest_score = weakest.score_at_entry or 0
                    
                    if score > weakest_score + 5:
                        logger.info(f"🔄 ROTATION: {symbol} (score={score}) > {weakest.symbol} (score={weakest_score})")
                        
                        try:
                            import yfinance as yf
                            sell_ticker = yf.Ticker(weakest.symbol)
                            sell_price = sell_ticker.fast_info.get('lastPrice') or weakest.entry_price
                        except:
                            sell_price = weakest.entry_price
                        
                        sell_result = bridge.close_position(
                            symbol=weakest.symbol,
                            price=sell_price,
                            reason=f"rotation_upgrade_to_{symbol}"
                        )
                        
                        if sell_result:
                            logger.info(f"🔻 SOLD {weakest.symbol} @ ${sell_price:.2f} for rotation")
                            
                            position = bridge.open_position(
                                symbol=symbol,
                                price=price,
                                score=score,
                                decision_type=str(decision.decision.value) if hasattr(decision, 'decision') else "BUY",
                                pillar_technical=pillar_technical,
                                pillar_fundamental=pillar_fundamental,
                                pillar_sentiment=pillar_sentiment,
                                pillar_news=pillar_news,
                                reasoning=reasoning,
                                company_name=company_name,
                                sector=sector,
                                industry=industry
                            )
                            
                            if position:
                                logger.info(f"🔺 ROTATION COMPLETE: Bought {symbol} @ ${price:.2f}")
                    else:
                        logger.debug(f"No rotation: {symbol} ({score}) not better than {weakest.symbol} ({weakest_score})")
            except Exception as e:
                logger.warning(f"Rotation failed: {e}")
        
        if position:
            self._metrics.trades_executed += 1
            logger.info(f"📈 Paper trade executed: {symbol} x{position.quantity} @ {price:.2f}")
            
            # Notify
            if self.config.notification_callback:
                await self.config.notification_callback({
                    'type': 'trade',
                    'action': 'BUY',
                    'symbol': symbol,
                    'price': price,
                    'quantity': position.quantity,
                    'score': score
                })
        
        if self._on_trade_callback:
            await self._on_trade_callback(alert, decision)

    # -------------------------------------------------------------------------
    # V7: POSITION MONITOR LOOP
    # -------------------------------------------------------------------------

    async def _position_monitor_loop(self):
        """V7: Monitor open positions for exit signals every 60 seconds."""
        logger.info("V7 Position monitor loop started")
        while self._running and not self._shutdown_event.is_set():
            try:
                if self.position_monitor and self.position_monitor.positions:
                    for symbol in list(self.position_monitor.positions.keys()):
                        exit_signal = self.position_monitor.check_exit(symbol)
                        if exit_signal:
                            reason, details = exit_signal
                            logger.info(f"[MONITOR] Exit signal for {symbol}: {reason.value} - {details}")
                            await self.position_monitor.execute_exit(symbol, reason, details)
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
            await asyncio.sleep(60)
        logger.info("V7 Position monitor loop ended")

    # -------------------------------------------------------------------------
    # HEALTH CHECK LOOP
    # -------------------------------------------------------------------------

    async def _health_loop(self):
        """Boucle de vérification de santé"""
        logger.info("Health check loop started")

        while self._running and not self._shutdown_event.is_set():
            try:
                # Vérifier la santé de la stratégie
                health = await self.strategy_evolver.check_health()

                if health.status == "critical":
                    logger.warning(f"Strategy health CRITICAL: {health.issues}")
                    await self.pause()
                    await self._notify(
                        f"🚨 Santé stratégie CRITIQUE\n"
                        f"Win rate: {health.win_rate:.0%}\n"
                        f"Issues: {', '.join(health.issues)}"
                    )

                elif health.status == "warning":
                    logger.warning(f"Strategy health WARNING: {health.issues}")
                    await self._notify(
                        f"⚠️ Santé stratégie en baisse\n"
                        f"Win rate: {health.win_rate:.0%}\n"
                        f"Issues: {', '.join(health.issues)}"
                    )

                    # Proposer des ajustements
                    proposals = await self.strategy_evolver.propose_adjustments()
                    for proposal in proposals:
                        if proposal.adaptation_level.value == "tweak":
                            await self.strategy_evolver.auto_adjust(proposal)

                # Log périodique
                if health.status == "healthy":
                    logger.info(
                        f"Strategy health OK - Win rate: {health.win_rate:.0%}, "
                        f"Score: {health.health_score:.0f}/100"
                    )

            except Exception as e:
                logger.error(f"Error in health loop: {e}", exc_info=True)

            await asyncio.sleep(self.config.health_check_interval)

        logger.info("Health check loop ended")

    # -------------------------------------------------------------------------
    # MARKET SESSION
    # -------------------------------------------------------------------------

    def _get_market_session(self) -> MarketSession:
        """Détermine la session de marché actuelle"""
        now = datetime.now(self._tz)
        current_time = now.time()

        # Weekend
        if now.weekday() >= 5:
            return MarketSession.CLOSED

        # Pre-market
        if self.config.pre_market_start <= current_time < self.config.market_open:
            return MarketSession.PRE_MARKET

        # Regular hours
        if self.config.market_open <= current_time < self.config.market_close:
            return MarketSession.REGULAR

        # After hours
        if self.config.market_close <= current_time < self.config.after_hours_end:
            return MarketSession.AFTER_HOURS

        return MarketSession.CLOSED

    def _should_trade(self) -> bool:
        """Détermine si on doit trader maintenant"""
        session = self._current_session

        if session == MarketSession.CLOSED:
            return False

        if session == MarketSession.PRE_MARKET:
            return self.config.trade_pre_market

        if session == MarketSession.AFTER_HOURS:
            return self.config.trade_after_hours

        return True  # Regular hours

    # -------------------------------------------------------------------------
    # GUARDRAILS
    # -------------------------------------------------------------------------

    async def _check_guardrails(self) -> bool:
        """Vérifie tous les guardrails"""
        try:
            # Daily loss check
            daily_pnl = self.state.get_daily_pnl()
            await self.guardrails.check_daily_loss(daily_pnl)

            # Drawdown check
            drawdown = self.state.get_current_drawdown()
            await self.guardrails.check_drawdown(drawdown)

            # V7: Circuit breakers
            if daily_pnl < -(self.config.daily_loss_limit_pct * 100):
                logger.warning(f"[CIRCUIT BREAKER] Daily loss {daily_pnl:.1f}% exceeds limit")
                return False

            if drawdown > self.config.max_drawdown_pct * 100:
                if self.defensive_manager:
                    self.defensive_manager.get_defensive_level('VOLATILE', vix_level=40, drawdown_pct=drawdown/100)
                logger.warning(f"[CIRCUIT BREAKER] Drawdown {drawdown:.1f}% -> MAXIMUM defensive")

            # Position count
            position_count = self.state.get_open_positions_count()
            await self.guardrails.check_position_count(position_count)

            return True

        except GuardrailViolation as e:
            logger.error(f"Guardrail check failed: {e}")
            return False

    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------

    def _update_avg_cycle_duration(self, new_duration_ms: float):
        """Met à jour la durée moyenne de cycle"""
        if self._metrics.cycles_completed <= 1:
            self._metrics.avg_cycle_duration_ms = new_duration_ms
        else:
            # Moyenne mobile
            alpha = 0.1
            self._metrics.avg_cycle_duration_ms = (
                alpha * new_duration_ms +
                (1 - alpha) * self._metrics.avg_cycle_duration_ms
            )

    async def _notify(self, message: str):
        """Envoie une notification"""
        if self.config.notification_callback:
            await self.config.notification_callback(message)
        logger.info(f"Notification: {message}")

    async def _cleanup(self):
        """Nettoie les ressources"""
        logger.info("Cleaning up LiveLoop...")

        if self.attention_manager:
            await self.attention_manager.close()

        if self.strategy_evolver:
            await self.strategy_evolver.close()

        if self.decision_journal:
            await self.decision_journal.close()

        if self.state:
            self.state_manager.save()

        logger.info("LiveLoop cleanup complete")

    # -------------------------------------------------------------------------
    # GETTERS
    # -------------------------------------------------------------------------

    def get_metrics(self) -> LoopMetrics:
        """Retourne les métriques"""
        return self._metrics

    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut complet"""
        return {
            'running': self._running,
            'paused': self._paused,
            'session': self._current_session.value,
            'metrics': {
                'started_at': self._metrics.started_at.isoformat() if self._metrics.started_at else None,
                'cycles': self._metrics.cycles_completed,
                'screens': self._metrics.screens_performed,
                'signals': self._metrics.signals_found,
                'trades': self._metrics.trades_executed,
                'errors': self._metrics.errors_count,
                'avg_cycle_ms': round(self._metrics.avg_cycle_duration_ms, 1)
            },
            'focus_topics': [
                t.symbol for t in self.attention_manager.get_focus_topics(5)
            ] if self.attention_manager else []
        }

    def is_running(self) -> bool:
        """Vérifie si la boucle tourne"""
        return self._running

    def is_paused(self) -> bool:
        """Vérifie si le trading est en pause"""
        return self._paused


# =============================================================================
# FACTORY
# =============================================================================

_live_loop: Optional[LiveLoop] = None


async def get_live_loop(config: Optional[LiveLoopConfig] = None) -> LiveLoop:
    """Retourne l'instance singleton de la LiveLoop"""
    global _live_loop
    if _live_loop is None:
        _live_loop = LiveLoop(config)
        await _live_loop.initialize()
    return _live_loop


async def start_live_trading(
    paper_trading: bool = True,
    notification_callback: Optional[Callable] = None
):
    """Démarre le trading live (helper function)"""
    config = LiveLoopConfig(
        paper_trading=paper_trading,
        notification_callback=notification_callback
    )

    loop = await get_live_loop(config)
    await loop.start()
