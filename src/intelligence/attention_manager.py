"""
Attention Manager - Gestion dynamique du focus sur les topics chauds
Alloue l'attention du bot aux symboles les plus prometteurs.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from collections import defaultdict
import json
from pathlib import Path

from .heat_detector import HeatDetector, SymbolHeat, get_heat_detector

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
    focus_reason: str  # "hot", "warming", "manual", "watchlist"

    # Timing
    focused_since: datetime
    last_checked: Optional[datetime] = None

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


@dataclass
class AttentionConfig:
    """Configuration de l'Attention Manager"""
    # Limites de focus
    max_hot_focus: int = 5  # Max symboles "hot" a suivre
    max_warming_focus: int = 10  # Max symboles "warming"
    max_watchlist_focus: int = 5  # Max de la watchlist

    # Decay
    focus_timeout_minutes: int = 30  # Retirer du focus apres X min sans activite
    cooling_threshold: float = 0.3  # En dessous = retirer du focus

    # Priorites
    hot_priority: int = 1
    warming_priority: int = 2
    watchlist_priority: int = 3
    manual_priority: int = 0  # Highest


# =============================================================================
# ATTENTION MANAGER
# =============================================================================

class AttentionManager:
    """
    Gere dynamiquement l'attention du bot.

    Decide sur quels symboles se concentrer en fonction de:
    - Heat scores (symboles chauds)
    - Watchlist utilisateur
    - Demandes manuelles
    - Historique de performance
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

        # Watchlist utilisateur - chargée depuis UniverseScanner ou fichier
        self._watchlist: Set[str] = set()

        # Focus manuel (priorite max)
        self._manual_focus: Set[str] = set()

        # Budget d'attention
        self._budget = AttentionBudget()

        # Historique
        self._focus_history: List[Dict] = []

        # Lock pour thread safety
        self._lock = asyncio.Lock()

        # Persistence
        self._data_dir = Path("data/attention")
        self._data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("AttentionManager initialized")

    async def initialize(self):
        """Initialise le manager"""
        if self.heat_detector is None:
            self.heat_detector = await get_heat_detector()

        await self._load_watchlist()
        logger.info("AttentionManager ready")

    # -------------------------------------------------------------------------
    # FOCUS MANAGEMENT
    # -------------------------------------------------------------------------

    async def update_focus(self) -> List[FocusTopic]:
        """
        Met a jour la liste des topics en focus.
        Appeler a chaque cycle de la boucle live.
        """
        async with self._lock:
            now = datetime.now()

            # 1. Retirer les topics expires ou refroidis
            await self._cleanup_stale_focus()

            # 2. Ajouter les symboles manuels (priorite max)
            for symbol in self._manual_focus:
                if symbol not in self._focus_topics:
                    self._focus_topics[symbol] = FocusTopic(
                        symbol=symbol,
                        heat_score=1.0,
                        priority=self.config.manual_priority,
                        focus_reason="manual",
                        focused_since=now
                    )

            # 3. Ajouter les symboles hot
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
                    # Mettre a jour le score
                    self._focus_topics[heat.symbol].heat_score = heat.heat_score

            # 4. Ajouter les symboles warming (si place disponible)
            current_count = len(self._focus_topics)
            if current_count < self.config.max_hot_focus + self.config.max_warming_focus:
                warming = self.heat_detector.get_warming_symbols(
                    self.config.max_warming_focus
                )
                for heat in warming:
                    if heat.symbol not in self._focus_topics:
                        self._focus_topics[heat.symbol] = FocusTopic(
                            symbol=heat.symbol,
                            heat_score=heat.heat_score,
                            priority=self.config.warming_priority,
                            focus_reason="warming",
                            focused_since=now
                        )

            # 5. Ajouter les symboles watchlist (si place disponible)
            current_count = len(self._focus_topics)
            max_total = (
                self.config.max_hot_focus +
                self.config.max_warming_focus +
                self.config.max_watchlist_focus
            )
            if current_count < max_total:
                for symbol in list(self._watchlist)[:self.config.max_watchlist_focus]:
                    if symbol not in self._focus_topics:
                        # Verifier si le symbole a du heat
                        heat = self.heat_detector.get_symbol_heat(symbol)
                        heat_score = heat.heat_score if heat else 0.1

                        self._focus_topics[symbol] = FocusTopic(
                            symbol=symbol,
                            heat_score=heat_score,
                            priority=self.config.watchlist_priority,
                            focus_reason="watchlist",
                            focused_since=now
                        )

            # 6. Trier par priorite et heat
            sorted_topics = self._get_sorted_focus()

            # 7. Mettre a jour le budget
            self._budget.current_focus_count = len(sorted_topics)

            return sorted_topics

    async def _cleanup_stale_focus(self):
        """Retire les topics qui ne sont plus pertinents"""
        now = datetime.now()
        timeout = timedelta(minutes=self.config.focus_timeout_minutes)

        to_remove = []
        for symbol, topic in self._focus_topics.items():
            # Ne pas retirer les manuels
            if topic.focus_reason == "manual":
                continue

            # Verifier le timeout
            if topic.last_checked and now - topic.last_checked > timeout:
                to_remove.append(symbol)
                continue

            # Verifier si refroidi
            heat = self.heat_detector.get_symbol_heat(symbol)
            if heat is None or heat.heat_score < self.config.cooling_threshold:
                # Refroidi, mais garder si watchlist
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
    # QUERIES
    # -------------------------------------------------------------------------

    def get_focus_topics(self, limit: int = 10) -> List[FocusTopic]:
        """Retourne les topics actuellement en focus"""
        topics = self._get_sorted_focus()
        return topics[:limit]

    def get_top_focus(self) -> Optional[FocusTopic]:
        """Retourne le topic avec la plus haute priorite"""
        topics = self._get_sorted_focus()
        return topics[0] if topics else None

    def get_symbols_for_screening(self, limit: int = 5) -> List[str]:
        """
        Retourne les symboles a screener dans ce cycle.
        Priorise ceux qui n'ont pas ete screenes recemment.
        """
        topics = self._get_sorted_focus()

        # Prioriser les non-screenes
        not_screened = [t for t in topics if not t.screened]
        screened = [t for t in topics if t.screened]

        result = not_screened + screened
        return [t.symbol for t in result[:limit]]

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
        """Ajoute un symbole a la watchlist"""
        self._watchlist.add(symbol.upper())
        logger.info(f"Added {symbol} to watchlist")

    def remove_from_watchlist(self, symbol: str):
        """Retire un symbole de la watchlist"""
        self._watchlist.discard(symbol.upper())
        logger.info(f"Removed {symbol} from watchlist")

    def get_watchlist(self) -> List[str]:
        """Retourne la watchlist"""
        return list(self._watchlist)

    def add_manual_focus(self, symbol: str):
        """Ajoute un focus manuel (priorite max)"""
        self._manual_focus.add(symbol.upper())
        logger.info(f"Added manual focus: {symbol}")

    def remove_manual_focus(self, symbol: str):
        """Retire un focus manuel"""
        self._manual_focus.discard(symbol.upper())
        if symbol.upper() in self._focus_topics:
            del self._focus_topics[symbol.upper()]
        logger.info(f"Removed manual focus: {symbol}")

    # -------------------------------------------------------------------------
    # TRACKING
    # -------------------------------------------------------------------------

    async def mark_screened(self, symbol: str):
        """Marque un symbole comme screene"""
        if symbol.upper() in self._focus_topics:
            topic = self._focus_topics[symbol.upper()]
            topic.screened = True
            topic.last_checked = datetime.now()
            topic.checks_count += 1
            self._budget.screens_this_cycle += 1

    async def mark_signal_found(self, symbol: str):
        """Enregistre qu'un signal a ete trouve"""
        if symbol.upper() in self._focus_topics:
            self._focus_topics[symbol.upper()].signals_found += 1

    async def mark_traded(self, symbol: str):
        """Marque un symbole comme trade"""
        if symbol.upper() in self._focus_topics:
            self._focus_topics[symbol.upper()].traded = True
            self._budget.trades_today += 1

    async def reset_cycle(self):
        """Reset les compteurs de cycle (appeler en debut de cycle)"""
        self._budget.screens_this_cycle = 0

        # Reset le flag screened pour permettre re-screening
        for topic in self._focus_topics.values():
            topic.screened = False

    async def reset_daily(self):
        """Reset les compteurs quotidiens"""
        self._budget.trades_today = 0

        # Sauvegarder l'historique
        self._focus_history.append({
            'date': datetime.now().isoformat(),
            'focus_count': len(self._focus_topics),
            'topics': [t.symbol for t in self._focus_topics.values()]
        })

        # Garder seulement 30 jours d'historique
        self._focus_history = self._focus_history[-30:]

    # -------------------------------------------------------------------------
    # BUDGET
    # -------------------------------------------------------------------------

    def can_screen_more(self) -> bool:
        """Verifie si on peut encore screener ce cycle"""
        return self._budget.screens_this_cycle < self._budget.max_screening_per_cycle

    def can_trade_more(self) -> bool:
        """Verifie si on peut encore trader aujourd'hui"""
        return self._budget.trades_today < self._budget.max_trades_per_day

    def get_budget(self) -> AttentionBudget:
        """Retourne le budget actuel"""
        return self._budget

    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------

    async def _load_watchlist(self):
        """Charge la watchlist depuis le disque (merge avec defauts)"""
        path = self._data_dir / "watchlist.json"
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    loaded = set(data.get('symbols', []))
                    if loaded:  # Only update if file has symbols
                        self._watchlist = self._watchlist.union(loaded)
            except Exception as e:
                logger.warning(f"Could not load watchlist: {e}")
        logger.info(f"Loaded watchlist: {len(self._watchlist)} symbols")

    async def save_watchlist(self):
        """Sauvegarde la watchlist"""
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
