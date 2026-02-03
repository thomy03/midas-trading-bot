"""
Live Loop - Boucle principale event-driven du trading bot
Orchestre tous les composants en temps réel.
"""

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

from ..intelligence.heat_detector import HeatDetector, get_heat_detector
from ..intelligence.attention_manager import AttentionManager, get_attention_manager

# V4 - Multi-pillar reasoning (4 pillars: technique, fondamental, sentiment, news)
try:
    from .reasoning_engine import ReasoningEngine, get_reasoning_engine, DecisionType
    REASONING_ENGINE_AVAILABLE = True
except ImportError:
    REASONING_ENGINE_AVAILABLE = False
    ReasoningEngine = None
    DecisionType = None

logger = logging.getLogger(__name__)


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
    heat_poll_interval: int = 60      # Polling des sources de chaleur
    screening_interval: int = 300     # Screening (5 min)
    health_check_interval: int = 600  # Health check (10 min)

    # Timezone
    timezone: str = "America/New_York"

    # Heures de trading
    pre_market_start: time = time(4, 0)
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)
    after_hours_end: time = time(20, 0)

    # Mode
    trade_pre_market: bool = False
    trade_after_hours: bool = False
    analyze_when_closed: bool = True  # Analyse meme hors marche
    paper_trading: bool = True

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

        # Components (initialisés dans initialize())
        self.guardrails: Optional[Guardrails] = None
        self.state: Optional[AgentState] = None
        self.heat_detector: Optional[HeatDetector] = None
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
        self._heat_task: Optional[asyncio.Task] = None
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
        self.heat_detector = await get_heat_detector()
        self.attention_manager = await get_attention_manager(self.heat_detector)

        # Learning components
        self.strategy_evolver = await get_strategy_evolver()
        self.decision_journal = await get_decision_journal()

        # Setup notification callbacks
        if self.config.notification_callback:
            self.strategy_evolver.set_notification_callback(
                self.config.notification_callback
            )

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
        self._heat_task = asyncio.create_task(self._heat_loop())
        self._health_task = asyncio.create_task(self._health_loop())

        try:
            await asyncio.gather(
                self._main_task,
                self._heat_task,
                self._health_task
            )
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
        for task in [self._main_task, self._heat_task, self._health_task]:
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
                        
                        # Analyse 4 piliers
                        result = await reasoning_engine.analyze(symbol)
                        
                        if result and result.decision in [DecisionType.BUY, DecisionType.STRONG_BUY]:
                            alert = {
                                'symbol': symbol,
                                'confidence_score': result.final_score,
                                'confidence_signal': result.decision.value.upper(),
                                'pillar_technical': result.technical_score,
                                'pillar_fundamental': result.fundamental_score,
                                'pillar_sentiment': result.sentiment_score,
                                'pillar_news': result.news_score,
                                'reasoning': result.reasoning,
                                'timestamp': result.timestamp
                            }
                            self._metrics.signals_found += 1
                            await self.attention_manager.mark_signal_found(symbol)
                            await self._process_signal(alert)
                        
                        self._metrics.screens_performed += 1
                        
                    except Exception as e:
                        logger.error(f"Error screening {symbol} with ReasoningEngine: {e}")
                        
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
            logger.info(f"⏭️ Signal {symbol} skipped: score {confidence} < {min_score}")
            return
            
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

        # Journaliser la décision
        decision = await self.decision_journal.log_decision(
            symbol=symbol,
            action="BUY",
            signal_data=alert
        )

        # Si le mode paper trading est désactivé, exécuter le trade
        if not self.config.paper_trading:
            await self._execute_trade(alert, decision)
        else:
            logger.info(f"📝 Paper trade logged: {symbol}")

    async def _execute_trade(self, alert: Dict, decision):
        """Exécute un trade réel"""
        # TODO: Intégrer avec IBKR executor
        symbol = alert.get('symbol')
        logger.info(f"🔔 Would execute trade: {symbol}")

        self._metrics.trades_executed += 1

        if self._on_trade_callback:
            await self._on_trade_callback(alert, decision)

    # -------------------------------------------------------------------------
    # HEAT DETECTION LOOP
    # -------------------------------------------------------------------------

    async def _heat_loop(self):
        """Boucle de détection de chaleur"""
        logger.info("Heat detection loop started")

        while self._running and not self._shutdown_event.is_set():
            try:
                # Seulement pendant les heures de marché étendue
                session = self._get_market_session()
                if session == MarketSession.CLOSED:
                    await asyncio.sleep(60)
                    continue

                # Collecter les données de différentes sources
                await self._collect_heat_data()

            except Exception as e:
                logger.error(f"Error in heat loop: {e}", exc_info=True)

            await asyncio.sleep(self.config.heat_poll_interval)

        logger.info("Heat detection loop ended")

    async def _collect_heat_data(self):
        """Collecte les données de chaleur de toutes les sources"""
        # Import des scanners
        try:
            from ..intelligence.grok_scanner import get_grok_scanner
            from ..intelligence.social_scanner import get_social_scanner

            # Grok (X/Twitter)
            grok_scanner = await get_grok_scanner()
            if grok_scanner:
                trends = await grok_scanner.search_financial_trends()
                if trends:
                    # Convert List[GrokInsight] to Dict[str, Dict] for ingest_grok_data
                    grok_data = {}
                    for insight in trends:
                        for symbol in insight.mentioned_symbols:
                            if symbol not in grok_data:
                                grok_data[symbol] = {
                                    'sentiment_score': insight.sentiment_score,
                                    'summary': insight.summary,
                                    'confidence': insight.confidence
                                }
                    if grok_data:
                        await self.heat_detector.ingest_grok_data(grok_data)

            # Social (Reddit, StockTwits)
            social_scanner = await get_social_scanner()
            if social_scanner:
                result = await social_scanner.full_scan()
                if result and result.trending_symbols:
                    # Convert trending_symbols to dict for ingest_reddit_data
                    social_data = {}
                    for ts in result.trending_symbols:
                        social_data[ts.symbol] = {
                            'count': ts.mention_count,
                            'sentiment': ts.avg_sentiment,
                            'posts': []
                        }
                    if social_data:
                        await self.heat_detector.ingest_reddit_data(social_data)

        except ImportError:
            logger.debug("Social scanners not available")
        except Exception as e:
            logger.warning(f"Error collecting heat data: {e}")

        # Données de prix/volume (toujours disponibles)
        try:
            price_data = await self._get_price_movements()
            if price_data:
                await self.heat_detector.ingest_price_data(price_data)
        except Exception as e:
            logger.warning(f"Error getting price data: {e}")

    async def _get_price_movements(self) -> Dict[str, Dict]:
        """Obtient les mouvements de prix récents"""
        # Symboles en focus
        focus_symbols = self.attention_manager.get_symbols_for_screening(limit=20)

        if not focus_symbols:
            return {}

        from ..data.market_data import MarketDataFetcher

        market_data = MarketDataFetcher()
        price_data = {}

        for symbol in focus_symbols:
            try:
                df = market_data.get_stock_data(symbol, period="5d", interval="1h")
                if df is not None and len(df) >= 2:
                    current_price = df['Close'].iloc[-1]
                    prev_price = df['Close'].iloc[-2]
                    change_pct = ((current_price - prev_price) / prev_price) * 100

                    avg_volume = df['Volume'].mean()
                    current_volume = df['Volume'].iloc[-1]
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

                    price_data[symbol] = {
                        'change_pct': change_pct,
                        'volume_ratio': volume_ratio,
                        'price': current_price
                    }
            except Exception as e:
                logger.debug(f"Error getting price for {symbol}: {e}")

        return price_data

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

        if self.heat_detector:
            await self.heat_detector.close()

        if self.attention_manager:
            await self.attention_manager.close()

        if self.strategy_evolver:
            await self.strategy_evolver.close()

        if self.decision_journal:
            await self.decision_journal.close()

        if self.state:
            await self.state.save()

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
            ] if self.attention_manager else [],
            'hot_symbols': [
                h.symbol for h in self.heat_detector.get_hot_symbols(5)
            ] if self.heat_detector else []
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
