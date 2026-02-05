"""
Attention Manager v2 - Avec Cooldown et Discovery Mode
Alloue l'attention du bot aux symboles les plus prometteurs.
Evite les analyses répétées et force la découverte de nouvelles opportunités.
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field

# V5.6 - Adaptive filtering
try:
    from src.intelligence.market_context import get_market_context, MarketRegime
    MARKET_CONTEXT_AVAILABLE = True
except ImportError:
    MARKET_CONTEXT_AVAILABLE = False
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import json
from pathlib import Path

from .heat_detector import HeatDetector, SymbolHeat, get_heat_detector
from src.utils.market_scheduler import MarketScheduler

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FocusTopic:
    """Topic sur lequel le bot se concentre"""
    symbol: str
    heat_score: float
    priority: int  # 1 = highest priority

    # Raison du focus
    focus_reason: str  # "hot", "warming", "manual", "watchlist", "discovery"

    # Timing
    focused_since: datetime
    last_checked: Optional[datetime] = None
    last_analyzed_at: Optional[datetime] = None  # NEW: Cooldown tracking

    # Actions
    screened: bool = False
    traded: bool = False

    # Performance sur ce focus
    checks_count: int = 0
    signals_found: int = 0


@dataclass
class AttentionBudget:
    """Budget d'attention disponible"""
    max_focus_topics: int = 10
    max_screening_per_cycle: int = 5
    max_trades_per_day: int = 3

    # Consommation actuelle
    current_focus_count: int = 0
    screens_this_cycle: int = 0
    trades_today: int = 0
    
    # Discovery stats
    symbols_discovered_today: int = 0


@dataclass
class AttentionConfig:
    """Configuration de l'Attention Manager"""
    # Limites de focus
    max_hot_focus: int = 5
    max_warming_focus: int = 10
    max_watchlist_focus: int = 5

    # Decay
    focus_timeout_minutes: int = 30
    cooling_threshold: float = 0.3

    # Priorites
    hot_priority: int = 1
    warming_priority: int = 2
    watchlist_priority: int = 3
    manual_priority: int = 0
    discovery_priority: int = 4  # NEW: Discovery symbols

    # NEW: Cooldown settings
    analysis_cooldown_hours: int = 4  # Don't re-analyze within X hours
    hot_bypass_threshold: float = 0.8  # Bypass cooldown if heat > this
    
    # NEW: Discovery settings
    discovery_batch_size: int = 10  # How many new symbols to discover per cycle
    force_discovery_after_cycles: int = 3  # Force discovery if no new symbols after X cycles


# =============================================================================
# ATTENTION MANAGER
# =============================================================================

class AttentionManager:
    """
    Gere dynamiquement l'attention du bot.
    V2: Avec cooldown intelligent et mode découverte.
    """

    def __init__(
        self,
        heat_detector: Optional[HeatDetector] = None,
        config: Optional[AttentionConfig] = None
    ):
        self.heat_detector = heat_detector
        self.config = config or AttentionConfig()

        # Topics en focus
        self._focus_topics: Dict[str, FocusTopic] = {}
        
        # V5.6 - Grok Priority: symbols found by Grok get screened FIRST
        self._grok_priority_symbols: List[str] = []
        self._grok_priority_updated: Optional[datetime] = None

        # Watchlist utilisateur
        self._watchlist: Set[str] = set()

        # Focus manuel
        self._manual_focus: Set[str] = set()

        # Budget d'attention
        self._budget = AttentionBudget()

        # Historique
        self._focus_history: List[Dict] = []
        
        # NEW: Analyzed today tracking (persisted)
        self._analyzed_today: Dict[str, datetime] = {}  # symbol -> last_analyzed_at
        self._cycles_without_discovery: int = 0

        # Lock pour thread safety
        self._lock = asyncio.Lock()

        # Persistence
        self._data_dir = Path("data/attention")
        self._data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("AttentionManager v2 initialized (with cooldown)")

    async def initialize(self):
        """Initialise le manager"""
        if self.heat_detector is None:
            self.heat_detector = await get_heat_detector()

        await self._load_watchlist()
        await self._load_analyzed_today()
        logger.info("AttentionManager ready")

    # -------------------------------------------------------------------------
    # COOLDOWN MANAGEMENT (NEW)
    # -------------------------------------------------------------------------
    
    def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in analysis cooldown"""
        if symbol not in self._analyzed_today:
            return False
        
        last_analyzed = self._analyzed_today[symbol]
        cooldown_delta = timedelta(hours=self.config.analysis_cooldown_hours)
        
        return datetime.now() - last_analyzed < cooldown_delta
    
    def _should_bypass_cooldown(self, symbol: str) -> bool:
        """Check if symbol should bypass cooldown (very hot)"""
        heat = self.heat_detector.get_symbol_heat(symbol)
        if heat and heat.heat_score >= self.config.hot_bypass_threshold:
            logger.info(f"{symbol} bypassing cooldown (heat={heat.heat_score:.2f})")
            return True
        return False
    
    def _get_cooldown_remaining(self, symbol: str) -> Optional[timedelta]:
        """Get remaining cooldown time for a symbol"""
        if symbol not in self._analyzed_today:
            return None
        
        last_analyzed = self._analyzed_today[symbol]
        cooldown_delta = timedelta(hours=self.config.analysis_cooldown_hours)
        remaining = (last_analyzed + cooldown_delta) - datetime.now()
        
        return remaining if remaining.total_seconds() > 0 else None

    # -------------------------------------------------------------------------
    # DISCOVERY MODE (NEW)
    # -------------------------------------------------------------------------
    
    def _get_unanalyzed_symbols(self, limit: int = 10) -> List[str]:
        """
        Get symbols from watchlist that haven't been analyzed today.
        Prioritizes by region (current market session).
        """
        # Filter by current market session
        session_symbols = [
            s for s in self._watchlist 
            if MarketScheduler.should_scan_symbol(s)
        ]
        
        # Remove already analyzed
        unanalyzed = [
            s for s in session_symbols 
            if s not in self._analyzed_today
        ]
        
        if not unanalyzed:
            # All symbols analyzed today, get oldest analyzed ones
            sorted_by_age = sorted(
                [(s, self._analyzed_today.get(s, datetime.min)) for s in session_symbols],
                key=lambda x: x[1]
            )
            unanalyzed = [s for s, _ in sorted_by_age[:limit * 2]]
        
        # Shuffle for variety
        random.shuffle(unanalyzed)
        
        logger.info(f"Discovery: {len(unanalyzed)} unanalyzed symbols available")
        return unanalyzed[:limit]
    
    async def trigger_discovery_scan(self) -> List[str]:
        """
        Trigger a discovery scan when regular symbols are in cooldown.
        Returns list of fresh symbols to analyze.
        """
        fresh_symbols = self._get_unanalyzed_symbols(self.config.discovery_batch_size)
        
        if fresh_symbols:
            logger.info(f"Discovery mode: scanning {len(fresh_symbols)} new symbols")
            self._budget.symbols_discovered_today += len(fresh_symbols)
            self._cycles_without_discovery = 0
            
            # Add them as discovery focus topics
            now = datetime.now()
            for symbol in fresh_symbols:
                if symbol not in self._focus_topics:
                    self._focus_topics[symbol] = FocusTopic(
                        symbol=symbol,
                        heat_score=0.1,  # Low heat, discovery mode
                        priority=self.config.discovery_priority,
                        focus_reason="discovery",
                        focused_since=now
                    )
        
        return fresh_symbols

    # -------------------------------------------------------------------------
    # FOCUS MANAGEMENT (UPDATED)
    # -------------------------------------------------------------------------

    async def update_focus(self) -> List[FocusTopic]:
        """Met a jour la liste des topics en focus."""
        async with self._lock:
            now = datetime.now()

            await self._cleanup_stale_focus()

            # Add manual focus
            for symbol in self._manual_focus:
                if symbol not in self._focus_topics:
                    self._focus_topics[symbol] = FocusTopic(
                        symbol=symbol,
                        heat_score=1.0,
                        priority=self.config.manual_priority,
                        focus_reason="manual",
                        focused_since=now
                    )

            # Add hot symbols
            hot_symbols = self.heat_detector.get_hot_symbols(self.config.max_hot_focus)
            for heat in hot_symbols:
                if heat.symbol not in self._focus_topics:
                    self._focus_topics[heat.symbol] = FocusTopic(
                        symbol=heat.symbol,
                        heat_score=heat.heat_score,
                        priority=self.config.hot_priority,
                        focus_reason="hot",
                        focused_since=now
                    )
                else:
                    self._focus_topics[heat.symbol].heat_score = heat.heat_score

            # Add warming symbols
            current_count = len(self._focus_topics)
            if current_count < self.config.max_hot_focus + self.config.max_warming_focus:
                warming = self.heat_detector.get_warming_symbols(self.config.max_warming_focus)
                for heat in warming:
                    if heat.symbol not in self._focus_topics:
                        self._focus_topics[heat.symbol] = FocusTopic(
                            symbol=heat.symbol,
                            heat_score=heat.heat_score,
                            priority=self.config.warming_priority,
                            focus_reason="warming",
                            focused_since=now
                        )

            # Add watchlist symbols
            current_count = len(self._focus_topics)
            max_total = (
                self.config.max_hot_focus +
                self.config.max_warming_focus +
                self.config.max_watchlist_focus
            )
            if current_count < max_total:
                session_filtered = [s for s in self._watchlist if MarketScheduler.should_scan_symbol(s)]
                for symbol in session_filtered[:self.config.max_watchlist_focus]:
                    if symbol not in self._focus_topics:
                        heat = self.heat_detector.get_symbol_heat(symbol)
                        heat_score = heat.heat_score if heat else 0.1
                        self._focus_topics[symbol] = FocusTopic(
                            symbol=symbol,
                            heat_score=heat_score,
                            priority=self.config.watchlist_priority,
                            focus_reason="watchlist",
                            focused_since=now
                        )

            sorted_topics = self._get_sorted_focus()
            self._budget.current_focus_count = len(sorted_topics)

            return sorted_topics

    async def _cleanup_stale_focus(self):
        """Retire les topics qui ne sont plus pertinents"""
        now = datetime.now()
        timeout = timedelta(minutes=self.config.focus_timeout_minutes)

        to_remove = []
        for symbol, topic in self._focus_topics.items():
            if topic.focus_reason == "manual":
                continue

            if topic.last_checked and now - topic.last_checked > timeout:
                to_remove.append(symbol)
                continue

            heat = self.heat_detector.get_symbol_heat(symbol)
            if heat is None or heat.heat_score < self.config.cooling_threshold:
                if symbol not in self._watchlist:
                    to_remove.append(symbol)

        for symbol in to_remove:
            logger.debug(f"Removing {symbol} from focus (stale/cooling)")
            del self._focus_topics[symbol]

    def _get_sorted_focus(self) -> List[FocusTopic]:
        """Retourne les topics tries par priorite puis heat"""
        topics = list(self._focus_topics.values())
        topics.sort(key=lambda t: (t.priority, -t.heat_score))
        return topics

    # -------------------------------------------------------------------------
    # QUERIES (UPDATED)
    # -------------------------------------------------------------------------

    def get_focus_topics(self, limit: int = 10) -> List[FocusTopic]:
        """Retourne les topics actuellement en focus"""
        topics = self._get_sorted_focus()
        return topics[:limit]

    def get_top_focus(self) -> Optional[FocusTopic]:
        """Retourne le topic avec la plus haute priorite"""
        topics = self._get_sorted_focus()
        return topics[0] if topics else None


    def set_grok_priority_symbols(self, symbols: List[str]):
        """
        V5.6 - Set symbols found by Grok as priority for screening.
        These will be screened FIRST before falling back to watchlist.
        """
        # Filter by market session
        session_filtered = [s for s in symbols if MarketScheduler.should_scan_symbol(s)]
        self._grok_priority_symbols = session_filtered
        self._grok_priority_updated = datetime.now()
        logger.info(f"🐦 GROK PRIORITY: {len(session_filtered)} symbols set for priority screening: {session_filtered[:10]}")
    

    def get_current_filters(self) -> dict:
        """
        V5.6 - Get adaptive market cap/volume filters based on current regime.
        """
        regime = "range"  # default
        region = "us"  # default
        
        # Get current market regime
        if MARKET_CONTEXT_AVAILABLE:
            try:
                import asyncio
                ctx = asyncio.get_event_loop().run_until_complete(get_market_context())
                if ctx:
                    regime = ctx.regime.value
            except:
                pass
        
        # Get current region from scheduler
        try:
            from src.utils.market_scheduler import MarketScheduler
            session = MarketScheduler.get_session_info()
            if session.get('region') == 'europe':
                region = 'eu'
        except:
            pass
        
        # Get adaptive filters
        from src.screening.hybrid_screener import ScreenerConfig
        filters = ScreenerConfig.get_adaptive_filters(regime=regime, region=region)
        
        logger.info(f"🎯 Adaptive filters: regime={regime}, region={region} → cap>=${filters['market_cap_min']/1e6:.0f}M, vol>{filters['volume_min']/1000:.0f}K")
        
        return filters

    def get_grok_priority_symbols(self) -> List[str]:
        """Get current Grok priority symbols (not in cooldown)."""
        available = []
        for symbol in self._grok_priority_symbols:
            if not self._is_in_cooldown(symbol):
                available.append(symbol)
        return available

    def get_symbols_for_screening(self, limit: int = 5) -> List[str]:
        """
        Retourne les symboles a screener dans ce cycle.
        V5.6: GROK PRIORITY - Screen Grok symbols FIRST, then fallback to watchlist.
        """
        result = []
        
        # STEP 1: Grok Priority Symbols (what's buzzing on X/Twitter)
        grok_available = self.get_grok_priority_symbols()
        if grok_available:
            result.extend(grok_available[:limit])
            logger.info(f"🐦 Screening GROK PRIORITY symbols: {result}")
            if len(result) >= limit:
                return result[:limit]
        
        # STEP 2: Hot symbols from heat detector
        topics = self._get_sorted_focus()

        # Filter by region
        session_info = MarketScheduler.get_session_info()
        logger.info(f"Market session: {session_info['region']} ({session_info['paris_time']} Paris)")
        
        filtered_topics = [
            t for t in topics 
            if MarketScheduler.should_scan_symbol(t.symbol)
        ]

        # NEW: Filter out symbols in cooldown (unless very hot)
        available = []
        skipped_cooldown = []
        
        for topic in filtered_topics:
            if self._is_in_cooldown(topic.symbol):
                if self._should_bypass_cooldown(topic.symbol):
                    available.append(topic)
                    logger.info(f"{topic.symbol}: cooldown bypassed (very hot)")
                else:
                    skipped_cooldown.append(topic.symbol)
                    remaining = self._get_cooldown_remaining(topic.symbol)
                    logger.debug(f"{topic.symbol}: skipped (cooldown {remaining})")
            else:
                available.append(topic)
        
        if skipped_cooldown:
            logger.info(f"Skipped {len(skipped_cooldown)} symbols in cooldown: {skipped_cooldown[:5]}...")

        # NEW: If not enough symbols, trigger discovery
        if len(available) < limit:
            self._cycles_without_discovery += 1
            
            if self._cycles_without_discovery >= self.config.force_discovery_after_cycles:
                logger.info(f"Not enough symbols available, triggering discovery mode")
                fresh = self._get_unanalyzed_symbols(limit - len(available))
                
                for symbol in fresh:
                    if symbol not in [t.symbol for t in available]:
                        # Create temporary focus topic for discovery
                        discovery_topic = FocusTopic(
                            symbol=symbol,
                            heat_score=0.1,
                            priority=self.config.discovery_priority,
                            focus_reason="discovery",
                            focused_since=datetime.now()
                        )
                        available.append(discovery_topic)
                        self._focus_topics[symbol] = discovery_topic

        # Sort: not screened first, then by priority/heat
        not_screened = [t for t in available if not t.screened]
        screened = [t for t in available if t.screened]
        
        # Only return not_screened (don't re-screen already screened this cycle)
        result = [t.symbol for t in not_screened[:limit]]
        
        if len(result) < limit and self._cycles_without_discovery >= 2:
            # Add some discovered symbols
            extra_discovery = self._get_unanalyzed_symbols(limit - len(result))
            result.extend(extra_discovery)
        
        logger.info(f"Symbols for screening: {result}")
        return result[:limit]

    def is_in_focus(self, symbol: str) -> bool:
        """Verifie si un symbole est en focus"""
        return symbol.upper() in self._focus_topics

    def get_focus_reason(self, symbol: str) -> Optional[str]:
        """Retourne la raison du focus pour un symbole"""
        topic = self._focus_topics.get(symbol.upper())
        return topic.focus_reason if topic else None

    # -------------------------------------------------------------------------
    # WATCHLIST & MANUAL
    # -------------------------------------------------------------------------

    def add_to_watchlist(self, symbol: str):
        self._watchlist.add(symbol.upper())
        logger.info(f"Added {symbol} to watchlist")

    def remove_from_watchlist(self, symbol: str):
        self._watchlist.discard(symbol.upper())
        logger.info(f"Removed {symbol} from watchlist")

    def get_watchlist(self) -> List[str]:
        return list(self._watchlist)

    def add_manual_focus(self, symbol: str):
        self._manual_focus.add(symbol.upper())
        logger.info(f"Added manual focus: {symbol}")

    def remove_manual_focus(self, symbol: str):
        self._manual_focus.discard(symbol.upper())
        if symbol.upper() in self._focus_topics:
            del self._focus_topics[symbol.upper()]
        logger.info(f"Removed manual focus: {symbol}")

    # -------------------------------------------------------------------------
    # TRACKING (UPDATED)
    # -------------------------------------------------------------------------

    async def mark_screened(self, symbol: str):
        """Marque un symbole comme screene avec timestamp"""
        symbol = symbol.upper()
        now = datetime.now()
        
        if symbol in self._focus_topics:
            topic = self._focus_topics[symbol]
            topic.screened = True
            topic.last_checked = now
            topic.last_analyzed_at = now  # NEW
            topic.checks_count += 1
        
        # NEW: Track in analyzed_today
        self._analyzed_today[symbol] = now
        self._budget.screens_this_cycle += 1
        
        logger.info(f"{symbol} marked as analyzed (cooldown starts)")
        
        # Persist periodically
        if self._budget.screens_this_cycle % 5 == 0:
            await self._save_analyzed_today()

    async def mark_signal_found(self, symbol: str):
        if symbol.upper() in self._focus_topics:
            self._focus_topics[symbol.upper()].signals_found += 1

    async def mark_traded(self, symbol: str):
        if symbol.upper() in self._focus_topics:
            self._focus_topics[symbol.upper()].traded = True
            self._budget.trades_today += 1

    async def reset_cycle(self):
        """Reset les compteurs de cycle (mais PAS le cooldown)"""
        self._budget.screens_this_cycle = 0

        # V2: Only reset screened flag, NOT the cooldown
        for topic in self._focus_topics.values():
            topic.screened = False
        
        # Don't reset _analyzed_today - that's the cooldown!

    async def reset_daily(self):
        """Reset les compteurs quotidiens"""
        self._budget.trades_today = 0
        self._budget.symbols_discovered_today = 0
        self._cycles_without_discovery = 0

        # NEW: Clear analyzed_today for new day
        self._analyzed_today.clear()

        # Save history
        self._focus_history.append({
            'date': datetime.now().isoformat(),
            'focus_count': len(self._focus_topics),
            'topics': [t.symbol for t in self._focus_topics.values()]
        })
        self._focus_history = self._focus_history[-30:]
        
        logger.info("Daily reset: cooldown cache cleared")

    # -------------------------------------------------------------------------
    # BUDGET
    # -------------------------------------------------------------------------

    def can_screen_more(self) -> bool:
        return self._budget.screens_this_cycle < self._budget.max_screening_per_cycle

    def can_trade_more(self) -> bool:
        return self._budget.trades_today < self._budget.max_trades_per_day

    def get_budget(self) -> AttentionBudget:
        return self._budget
    
    def get_cooldown_stats(self) -> Dict:
        """Get stats about current cooldown state"""
        in_cooldown = [s for s in self._analyzed_today if self._is_in_cooldown(s)]
        return {
            'symbols_analyzed_today': len(self._analyzed_today),
            'currently_in_cooldown': len(in_cooldown),
            'cooldown_hours': self.config.analysis_cooldown_hours,
            'symbols_discovered_today': self._budget.symbols_discovered_today
        }

    # -------------------------------------------------------------------------
    # PERSISTENCE (UPDATED)
    # -------------------------------------------------------------------------

    async def _load_watchlist(self):
        """Charge la watchlist depuis le disque + tickers de la session active"""
        path = self._data_dir / "watchlist.json"
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    loaded = set(data.get('symbols', []))
                    if loaded:
                        self._watchlist = self._watchlist.union(loaded)
            except Exception as e:
                logger.warning(f"Could not load watchlist: {e}")
        
        # Load region-specific tickers
        ticker_files = MarketScheduler.get_active_ticker_files()
        tickers_dir = self._data_dir.parent / "tickers"
        
        for ticker_file in ticker_files:
            ticker_path = tickers_dir / ticker_file
            if ticker_path.exists():
                try:
                    with open(ticker_path, 'r') as f:
                        data = json.load(f)
                        tickers = data.get('tickers', [])
                        for t in tickers:
                            symbol = t.get('symbol') if isinstance(t, dict) else t
                            if symbol:
                                self._watchlist.add(symbol.upper())
                    logger.info(f"Loaded {len(tickers)} symbols from {ticker_file}")
                except Exception as e:
                    logger.warning(f"Could not load {ticker_file}: {e}")
        
        logger.info(f"Total watchlist: {len(self._watchlist)} symbols")

    async def _load_analyzed_today(self):
        """Load today's analyzed symbols from disk"""
        path = self._data_dir / "analyzed_today.json"
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    # Only load if same day
                    saved_date = data.get('date', '')
                    today = datetime.now().strftime('%Y-%m-%d')
                    
                    if saved_date == today:
                        for symbol, ts in data.get('symbols', {}).items():
                            self._analyzed_today[symbol] = datetime.fromisoformat(ts)
                        logger.info(f"Loaded {len(self._analyzed_today)} analyzed symbols from today")
                    else:
                        logger.info("Analyzed cache is from yesterday, starting fresh")
            except Exception as e:
                logger.warning(f"Could not load analyzed_today: {e}")

    async def _save_analyzed_today(self):
        """Save today's analyzed symbols to disk"""
        path = self._data_dir / "analyzed_today.json"
        try:
            data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'symbols': {s: t.isoformat() for s, t in self._analyzed_today.items()}
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save analyzed_today: {e}")

    async def save_watchlist(self):
        path = self._data_dir / "watchlist.json"
        try:
            with open(path, 'w') as f:
                json.dump({
                    'symbols': list(self._watchlist),
                    'updated_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save watchlist: {e}")

    async def close(self):
        """Ferme proprement le manager"""
        await self.save_watchlist()
        await self._save_analyzed_today()
        logger.info("AttentionManager closed")


# =============================================================================
# FACTORY
# =============================================================================

_attention_manager: Optional[AttentionManager] = None


async def get_attention_manager(
    heat_detector: Optional[HeatDetector] = None,
    config: Optional[AttentionConfig] = None
) -> AttentionManager:
    """Retourne l'instance singleton de l'AttentionManager"""
    global _attention_manager
    if _attention_manager is None:
        _attention_manager = AttentionManager(heat_detector, config)
        await _attention_manager.initialize()
    return _attention_manager
