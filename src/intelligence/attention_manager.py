"""
Attention Manager v3 - Simplified (V8.1 cleanup)
Grok priority symbols first, then watchlist rotation.
No more HOT/WARMING/COLD heat levels.
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import json
from pathlib import Path

from src.utils.market_scheduler import MarketScheduler

logger = logging.getLogger(__name__)


@dataclass
class FocusTopic:
    """Topic sur lequel le bot se concentre"""
    symbol: str
    priority: int = 3
    focus_reason: str = "watchlist"  # "grok_priority", "watchlist", "manual", "discovery"
    focused_since: datetime = field(default_factory=datetime.now)
    last_checked: Optional[datetime] = None
    last_analyzed_at: Optional[datetime] = None
    screened: bool = False
    traded: bool = False
    checks_count: int = 0
    signals_found: int = 0
    heat_score: float = 0.0  # kept for API compat


@dataclass
class AttentionBudget:
    max_focus_topics: int = 10
    max_screening_per_cycle: int = 5
    max_trades_per_day: int = 3
    current_focus_count: int = 0
    screens_this_cycle: int = 0
    trades_today: int = 0
    symbols_discovered_today: int = 0


@dataclass
class AttentionConfig:
    max_watchlist_focus: int = 15
    focus_timeout_minutes: int = 30
    manual_priority: int = 0
    grok_priority: int = 1
    watchlist_priority: int = 3
    discovery_priority: int = 4
    analysis_cooldown_hours: int = 4
    discovery_batch_size: int = 0
    force_discovery_after_cycles: int = 999
    # Compat aliases
    max_hot_focus: int = 5
    max_warming_focus: int = 10
    cooling_threshold: float = 0.3
    hot_bypass_threshold: float = 0.8
    hot_priority: int = 1
    warming_priority: int = 2


class AttentionManager:
    """
    v3: Grok priority first, then watchlist rotation.
    No heat detector dependency.
    """

    def __init__(self, heat_detector=None, config: Optional[AttentionConfig] = None):
        # heat_detector param kept for API compat but ignored
        self.config = config or AttentionConfig()
        self._focus_topics: Dict[str, FocusTopic] = {}
        self._grok_priority_symbols: List[str] = []
        self._grok_priority_updated: Optional[datetime] = None
        self._watchlist: Set[str] = set()
        self._manual_focus: Set[str] = set()
        self._budget = AttentionBudget()
        self._focus_history: List[Dict] = []
        self._analyzed_today: Dict[str, datetime] = {}
        self._cycles_without_discovery: int = 0
        self._lock = asyncio.Lock()
        self._data_dir = Path("data/attention")
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Load priority symbols from file
        try:
            with open("/app/data/grok/priority_symbols.json", "r") as f:
                data = json.load(f)
                self._grok_priority_symbols = data.get("symbols", [])
                logger.info(f"Loaded {len(self._grok_priority_symbols)} priority symbols from file")
        except Exception as e:
            logger.debug(f"No priority file: {e}")

        logger.info("AttentionManager v3 initialized (no heat detector)")

    async def initialize(self):
        await self._load_watchlist()
        await self._load_analyzed_today()
        logger.info("AttentionManager ready")

    # -- Cooldown --
    def _is_in_cooldown(self, symbol: str) -> bool:
        if symbol not in self._analyzed_today:
            return False
        return datetime.now() - self._analyzed_today[symbol] < timedelta(hours=self.config.analysis_cooldown_hours)

    def _get_cooldown_remaining(self, symbol: str) -> Optional[timedelta]:
        if symbol not in self._analyzed_today:
            return None
        remaining = (self._analyzed_today[symbol] + timedelta(hours=self.config.analysis_cooldown_hours)) - datetime.now()
        return remaining if remaining.total_seconds() > 0 else None

    # -- Focus management --
    async def update_focus(self) -> List[FocusTopic]:
        async with self._lock:
            now = datetime.now()
            await self._cleanup_stale_focus()

            # Manual focus
            for symbol in self._manual_focus:
                if symbol not in self._focus_topics:
                    self._focus_topics[symbol] = FocusTopic(
                        symbol=symbol, priority=self.config.manual_priority,
                        focus_reason="manual", focused_since=now
                    )

            # Grok priority symbols
            for symbol in self._grok_priority_symbols:
                if symbol not in self._focus_topics and MarketScheduler.should_scan_symbol(symbol):
                    self._focus_topics[symbol] = FocusTopic(
                        symbol=symbol, priority=self.config.grok_priority,
                        focus_reason="grok_priority", focused_since=now
                    )

            # Watchlist rotation (fill remaining slots)
            current_count = len(self._focus_topics)
            max_total = self.config.max_watchlist_focus + len(self._grok_priority_symbols)
            if current_count < max_total:
                session_filtered = [s for s in self._watchlist if MarketScheduler.should_scan_symbol(s)]
                random.shuffle(session_filtered)
                for symbol in session_filtered[:self.config.max_watchlist_focus]:
                    if symbol not in self._focus_topics:
                        self._focus_topics[symbol] = FocusTopic(
                            symbol=symbol, priority=self.config.watchlist_priority,
                            focus_reason="watchlist", focused_since=now
                        )

            sorted_topics = self._get_sorted_focus()
            self._budget.current_focus_count = len(sorted_topics)
            return sorted_topics

    async def _cleanup_stale_focus(self):
        now = datetime.now()
        timeout = timedelta(minutes=self.config.focus_timeout_minutes)
        to_remove = []
        for symbol, topic in self._focus_topics.items():
            if topic.focus_reason == "manual":
                continue
            if topic.last_checked and now - topic.last_checked > timeout:
                to_remove.append(symbol)
        for symbol in to_remove:
            del self._focus_topics[symbol]

    def _get_sorted_focus(self) -> List[FocusTopic]:
        topics = list(self._focus_topics.values())
        topics.sort(key=lambda t: (t.priority, -t.checks_count))
        return topics

    # -- Queries --
    def get_focus_topics(self, limit: int = 10) -> List[FocusTopic]:
        return self._get_sorted_focus()[:limit]

    def get_top_focus(self) -> Optional[FocusTopic]:
        topics = self._get_sorted_focus()
        return topics[0] if topics else None

    def get_tracked_symbols(self) -> Set[str]:
        return set(self._focus_topics.keys()) | self._watchlist

    def set_grok_priority_symbols(self, symbols: List[str]):
        session_filtered = [s for s in symbols if MarketScheduler.should_scan_symbol(s)]
        self._grok_priority_symbols = session_filtered
        self._grok_priority_updated = datetime.now()
        logger.info(f"🐦 GROK PRIORITY: {len(session_filtered)} symbols: {session_filtered[:10]}")

    def get_grok_priority_symbols(self) -> List[str]:
        return [s for s in self._grok_priority_symbols if not self._is_in_cooldown(s)]

    def get_symbols_for_screening(self, limit: int = 5) -> List[str]:
        result = []

        # 1. Grok priority (not in cooldown)
        grok_available = self.get_grok_priority_symbols()
        if grok_available:
            result.extend(grok_available[:limit])
            logger.info(f"🐦 Screening GROK PRIORITY: {result}")
            if len(result) >= limit:
                return result[:limit]

        # 2. Focus topics (not screened, not in cooldown, region-filtered)
        session_info = MarketScheduler.get_session_info()
        logger.info(f"Market session: {session_info.get('region', 'unknown')} ({session_info.get('paris_time', '?')} Paris)")

        topics = self._get_sorted_focus()
        filtered = [t for t in topics if MarketScheduler.should_scan_symbol(t.symbol)]

        available = []
        skipped = []
        for topic in filtered:
            if topic.symbol in result:
                continue
            if self._is_in_cooldown(topic.symbol):
                skipped.append(topic.symbol)
            else:
                available.append(topic)

        if skipped:
            logger.info(f"Skipped {len(skipped)} in cooldown: {skipped[:5]}...")

        # Discovery if not enough
        if len(available) < (limit - len(result)):
            self._cycles_without_discovery += 1
            if self._cycles_without_discovery >= self.config.force_discovery_after_cycles:
                fresh = self._get_unanalyzed_symbols(limit - len(result) - len(available))
                for sym in fresh:
                    if sym not in [t.symbol for t in available] and sym not in result:
                        dt = FocusTopic(symbol=sym, priority=self.config.discovery_priority,
                                       focus_reason="discovery", focused_since=datetime.now())
                        available.append(dt)
                        self._focus_topics[sym] = dt

        not_screened = [t for t in available if not t.screened]
        result.extend([t.symbol for t in not_screened[:limit - len(result)]])

        if len(result) < limit and self._cycles_without_discovery >= 2:
            extra = self._get_unanalyzed_symbols(limit - len(result))
            result.extend(extra)

        logger.info(f"Symbols for screening: {result}")
        return result[:limit]

    def _get_unanalyzed_symbols(self, limit: int = 10) -> List[str]:
        session_symbols = [s for s in self._watchlist if MarketScheduler.should_scan_symbol(s)]
        unanalyzed = [s for s in session_symbols if s not in self._analyzed_today]
        if not unanalyzed:
            sorted_by_age = sorted(
                [(s, self._analyzed_today.get(s, datetime.min)) for s in session_symbols],
                key=lambda x: x[1]
            )
            unanalyzed = [s for s, _ in sorted_by_age[:limit * 2]]
        random.shuffle(unanalyzed)
        return unanalyzed[:limit]

    def is_in_focus(self, symbol: str) -> bool:
        return symbol.upper() in self._focus_topics

    def get_focus_reason(self, symbol: str) -> Optional[str]:
        topic = self._focus_topics.get(symbol.upper())
        return topic.focus_reason if topic else None

    def get_current_filters(self) -> dict:
        regime = "range"
        region = "us"
        try:
            session = MarketScheduler.get_session_info()
            if session.get("region") == "europe":
                region = "eu"
        except:
            pass
        from src.screening.hybrid_screener import ScreenerConfig
        filters = ScreenerConfig.get_adaptive_filters(regime=regime, region=region)
        logger.info(f"🎯 Adaptive filters: regime={regime}, region={region} → cap>=${filters[market_cap_min]/1e6:.0f}M, vol>{filters[volume_min]/1000:.0f}K")
        return filters

    # -- Watchlist & Manual --
    def add_to_watchlist(self, symbol: str):
        self._watchlist.add(symbol.upper())

    def remove_from_watchlist(self, symbol: str):
        self._watchlist.discard(symbol.upper())

    def get_watchlist(self) -> List[str]:
        return list(self._watchlist)

    def add_manual_focus(self, symbol: str):
        self._manual_focus.add(symbol.upper())

    def remove_manual_focus(self, symbol: str):
        self._manual_focus.discard(symbol.upper())
        self._focus_topics.pop(symbol.upper(), None)

    # -- Tracking --
    async def mark_screened(self, symbol: str):
        symbol = symbol.upper()
        now = datetime.now()
        if symbol in self._focus_topics:
            t = self._focus_topics[symbol]
            t.screened = True
            t.last_checked = now
            t.last_analyzed_at = now
            t.checks_count += 1
        self._analyzed_today[symbol] = now
        self._budget.screens_this_cycle += 1
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
        self._budget.screens_this_cycle = 0
        for topic in self._focus_topics.values():
            topic.screened = False

    async def reset_daily(self):
        self._budget.trades_today = 0
        self._budget.symbols_discovered_today = 0
        self._cycles_without_discovery = 0
        self._analyzed_today.clear()
        self._focus_history.append({
            "date": datetime.now().isoformat(),
            "focus_count": len(self._focus_topics),
            "topics": [t.symbol for t in self._focus_topics.values()]
        })
        self._focus_history = self._focus_history[-30:]

    # -- Budget --
    def can_screen_more(self) -> bool:
        return self._budget.screens_this_cycle < self._budget.max_screening_per_cycle

    def can_trade_more(self) -> bool:
        return self._budget.trades_today < self._budget.max_trades_per_day

    def get_budget(self) -> AttentionBudget:
        return self._budget

    def get_cooldown_stats(self) -> Dict:
        in_cooldown = [s for s in self._analyzed_today if self._is_in_cooldown(s)]
        return {
            "symbols_analyzed_today": len(self._analyzed_today),
            "currently_in_cooldown": len(in_cooldown),
            "cooldown_hours": self.config.analysis_cooldown_hours,
            "symbols_discovered_today": self._budget.symbols_discovered_today,
        }

    # -- Persistence --
    async def _load_watchlist(self):
        path = self._data_dir / "watchlist.json"
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    self._watchlist = self._watchlist.union(set(data.get("symbols", [])))
            except Exception as e:
                logger.warning(f"Could not load watchlist: {e}")

        ticker_files = MarketScheduler.get_active_ticker_files()
        tickers_dir = self._data_dir.parent / "tickers"
        for ticker_file in ticker_files:
            ticker_path = tickers_dir / ticker_file
            if ticker_path.exists():
                try:
                    with open(ticker_path, "r") as f:
                        data = json.load(f)
                        tickers = data.get("tickers", [])
                        for t in tickers:
                            symbol = t.get("symbol") if isinstance(t, dict) else t
                            if symbol:
                                self._watchlist.add(symbol.upper())
                    logger.info(f"Loaded {len(tickers)} symbols from {ticker_file}")
                except Exception as e:
                    logger.warning(f"Could not load {ticker_file}: {e}")
        logger.info(f"Total watchlist: {len(self._watchlist)} symbols")

    async def _load_analyzed_today(self):
        path = self._data_dir / "analyzed_today.json"
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    if data.get("date") == datetime.now().strftime("%Y-%m-%d"):
                        for symbol, ts in data.get("symbols", {}).items():
                            self._analyzed_today[symbol] = datetime.fromisoformat(ts)
                        logger.info(f"Loaded {len(self._analyzed_today)} analyzed symbols")
            except Exception as e:
                logger.warning(f"Could not load analyzed_today: {e}")

    async def _save_analyzed_today(self):
        path = self._data_dir / "analyzed_today.json"
        try:
            with open(path, "w") as f:
                json.dump({"date": datetime.now().strftime("%Y-%m-%d"),
                           "symbols": {s: t.isoformat() for s, t in self._analyzed_today.items()}}, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save analyzed_today: {e}")

    async def save_watchlist(self):
        path = self._data_dir / "watchlist.json"
        try:
            with open(path, "w") as f:
                json.dump({"symbols": list(self._watchlist), "updated_at": datetime.now().isoformat()}, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save watchlist: {e}")

    async def close(self):
        await self.save_watchlist()
        await self._save_analyzed_today()
        logger.info("AttentionManager closed")


_attention_manager: Optional[AttentionManager] = None


async def get_attention_manager(
    heat_detector=None,
    config: Optional[AttentionConfig] = None
) -> AttentionManager:
    global _attention_manager
    if _attention_manager is None:
        _attention_manager = AttentionManager(heat_detector, config)
        await _attention_manager.initialize()
    return _attention_manager
