"""
Agent State - État Persistant du Robot de Trading
Gère l'état global du système agentique.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Régime de marché détecté"""
    BULL_STRONG = "bull_strong"      # Tendance haussière forte
    BULL_WEAK = "bull_weak"          # Tendance haussière faible
    BEAR_STRONG = "bear_strong"      # Tendance baissière forte
    BEAR_WEAK = "bear_weak"          # Tendance baissière faible
    RANGING = "ranging"              # Marché sans direction
    VOLATILE = "volatile"            # Forte volatilité
    UNKNOWN = "unknown"


class AgentPhase(Enum):
    """Phase actuelle de l'agent"""
    IDLE = "idle"                    # En attente
    DISCOVERY = "discovery"          # Phase de découverte (06:00)
    ANALYSIS = "analysis"            # Analyse des tendances (07:00)
    TRADING = "trading"              # Phase de trading (09:30-16:00)
    AUDIT = "audit"                  # Audit nocturne (20:00)
    PAUSED = "paused"                # En pause (erreur ou kill switch)


@dataclass
class Position:
    """Position ouverte"""
    symbol: str
    quantity: int
    entry_price: float
    entry_date: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    thesis: Optional[str] = None     # Raison de l'entrée
    current_price: Optional[float] = None
    unrealized_pnl: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TradeRecord:
    """Enregistrement d'un trade (historique)"""
    id: str
    symbol: str
    action: str                      # 'BUY' ou 'SELL'
    quantity: int
    price: float
    timestamp: str
    thesis: Optional[str] = None
    outcome: Optional[str] = None    # 'WIN', 'LOSS', 'BREAKEVEN'
    pnl: float = 0.0
    exit_reason: Optional[str] = None

    # Champs pour l'audit et l'apprentissage
    entry_date: Optional[str] = None
    exit_date: Optional[str] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    pnl_pct: float = 0.0
    hold_days: int = 0
    exit_type: Optional[str] = None  # 'stop_loss', 'take_profit', 'trailing_stop', 'manual'
    metadata: Dict[str, Any] = field(default_factory=dict)  # market_regime, vix, rsi, etc.

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeRecord':
        """Crée une instance depuis un dictionnaire"""
        # Filtrer les clés inconnues pour éviter les erreurs
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


@dataclass
class DailyStats:
    """Statistiques quotidiennes"""
    date: str
    trades_count: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    max_drawdown: float = 0.0
    focus_symbols: List[str] = field(default_factory=list)
    discovered_trends: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AgentState:
    """
    État global de l'agent de trading.
    Persiste entre les sessions.
    """

    # Identité
    version: str = "4.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    # Phase actuelle
    current_phase: str = AgentPhase.IDLE.value
    phase_started_at: Optional[str] = None

    # Régime de marché
    market_regime: str = MarketRegime.UNKNOWN.value
    regime_confidence: float = 0.0
    regime_updated_at: Optional[str] = None

    # Capital
    initial_capital: float = 1500.0
    current_capital: float = 1500.0
    peak_capital: float = 1500.0     # Pour calcul drawdown

    # Positions ouvertes
    positions: Dict[str, Dict] = field(default_factory=dict)

    # Watchlist dynamique (stocks à surveiller)
    watchlist: List[str] = field(default_factory=list)
    focus_symbols: List[str] = field(default_factory=list)  # Priorité haute

    # Tendances actives
    active_trends: List[Dict] = field(default_factory=list)
    active_narratives: List[str] = field(default_factory=list)

    # Statistiques
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0

    # Règles apprises (référence au fichier)
    learned_rules_count: int = 0
    last_rule_learned: Optional[str] = None

    # Dernières actions
    last_scan_at: Optional[str] = None
    last_trade_at: Optional[str] = None
    last_audit_at: Optional[str] = None

    # Erreurs et warnings
    recent_errors: List[Dict] = field(default_factory=list)
    active_warnings: List[str] = field(default_factory=list)

    # Historique quotidien (derniers 30 jours)
    daily_history: List[Dict] = field(default_factory=list)

    # Historique complet des trades (pour l'audit et l'apprentissage)
    trade_history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour sérialisation"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentState':
        """Crée une instance depuis un dictionnaire"""
        return cls(**data)


class StateManager:
    """
    Gestionnaire d'état de l'agent.
    Gère la persistance et les mises à jour de l'état.
    """

    STATE_FILE = "agent_state.json"
    HISTORY_FILE = "agent_history.json"

    def __init__(self, data_dir: str = "data"):
        """
        Initialise le gestionnaire d'état.

        Args:
            data_dir: Répertoire de stockage
        """
        self.data_dir = data_dir
        self.state_path = os.path.join(data_dir, self.STATE_FILE)
        self.history_path = os.path.join(data_dir, self.HISTORY_FILE)

        # Créer le répertoire si nécessaire
        os.makedirs(data_dir, exist_ok=True)

        # Charger ou créer l'état
        self.state = self._load_state()

        logger.info(f"StateManager initialized: capital={self.state.current_capital}€, "
                   f"phase={self.state.current_phase}")

    # =========================================================================
    # PERSISTANCE
    # =========================================================================

    def _load_state(self) -> AgentState:
        """Charge l'état depuis le fichier"""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                state = AgentState.from_dict(data)
                logger.info(f"Agent state loaded from {self.state_path}")
                return state
            except Exception as e:
                logger.error(f"Error loading state: {e}")

        # Créer un nouvel état
        return AgentState()

    def save(self):
        """Sauvegarde l'état actuel"""
        self.state.last_updated = datetime.now().isoformat()

        try:
            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump(self.state.to_dict(), f, indent=2, ensure_ascii=False)
            logger.debug(f"State saved to {self.state_path}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    # =========================================================================
    # PHASE MANAGEMENT
    # =========================================================================

    def set_phase(self, phase: AgentPhase):
        """Change la phase actuelle de l'agent"""
        old_phase = self.state.current_phase
        self.state.current_phase = phase.value
        self.state.phase_started_at = datetime.now().isoformat()
        self.save()

        logger.info(f"Agent phase changed: {old_phase} -> {phase.value}")

    def get_phase(self) -> AgentPhase:
        """Retourne la phase actuelle"""
        return AgentPhase(self.state.current_phase)

    # =========================================================================
    # MARKET REGIME
    # =========================================================================

    def update_market_regime(self, regime: MarketRegime, confidence: float):
        """Met à jour le régime de marché détecté"""
        self.state.market_regime = regime.value
        self.state.regime_confidence = confidence
        self.state.regime_updated_at = datetime.now().isoformat()
        self.save()

        logger.info(f"Market regime updated: {regime.value} (confidence: {confidence:.1%})")

    def get_market_regime(self) -> tuple[MarketRegime, float]:
        """Retourne le régime de marché actuel et sa confiance"""
        return MarketRegime(self.state.market_regime), self.state.regime_confidence

    # =========================================================================
    # CAPITAL MANAGEMENT
    # =========================================================================

    def update_capital(self, new_capital: float):
        """Met à jour le capital actuel"""
        old_capital = self.state.current_capital
        self.state.current_capital = new_capital

        # Mettre à jour le peak capital (pour drawdown)
        if new_capital > self.state.peak_capital:
            self.state.peak_capital = new_capital

        self.save()

        pnl = new_capital - old_capital
        logger.info(f"Capital updated: {old_capital:.2f}€ -> {new_capital:.2f}€ (P&L: {pnl:+.2f}€)")

    def get_drawdown(self) -> float:
        """Calcule le drawdown actuel en %"""
        if self.state.peak_capital > 0:
            return (self.state.peak_capital - self.state.current_capital) / self.state.peak_capital
        return 0.0

    # =========================================================================
    # POSITIONS
    # =========================================================================

    def add_position(self, position: Position):
        """Ajoute une nouvelle position"""
        self.state.positions[position.symbol] = position.to_dict()
        self.save()

        logger.info(f"Position added: {position.symbol} x{position.quantity} @ {position.entry_price}")

    def update_position(self, symbol: str, current_price: float):
        """Met à jour le prix actuel d'une position"""
        if symbol in self.state.positions:
            pos = self.state.positions[symbol]
            pos['current_price'] = current_price
            pos['unrealized_pnl'] = (current_price - pos['entry_price']) * pos['quantity']
            self.save()

    def remove_position(self, symbol: str) -> Optional[Dict]:
        """Supprime une position (après vente)"""
        if symbol in self.state.positions:
            position = self.state.positions.pop(symbol)
            self.save()
            return position
        return None

    def get_positions(self) -> Dict[str, Dict]:
        """Retourne toutes les positions ouvertes"""
        return self.state.positions.copy()

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Retourne une position spécifique"""
        return self.state.positions.get(symbol)

    # =========================================================================
    # WATCHLIST & FOCUS
    # =========================================================================

    def update_watchlist(self, symbols: List[str]):
        """Met à jour la watchlist"""
        self.state.watchlist = list(set(symbols))
        self.save()

        logger.info(f"Watchlist updated: {len(self.state.watchlist)} symbols")

    def update_focus_symbols(self, symbols: List[str], append: bool = False):
        """Met à jour les symboles prioritaires avec rotation FIFO

        Args:
            symbols: Liste des symboles à ajouter/remplacer
            append: Si True, ajoute aux existants. Si False, remplace.
        """
        max_symbols = 100

        if append:
            # Mode FIFO: ajouter les nouveaux à la fin, retirer les anciens du début
            existing = self.state.focus_symbols or []
            # Éviter les doublons en gardant l'ordre
            new_symbols = [s for s in symbols if s not in existing]
            combined = existing + new_symbols

            # Si on dépasse la limite, retirer les plus anciens (début de liste)
            if len(combined) > max_symbols:
                removed_count = len(combined) - max_symbols
                logger.info(f"FIFO rotation: removing {removed_count} oldest symbols")
                combined = combined[removed_count:]

            self.state.focus_symbols = combined
        else:
            # Mode remplacement: prendre les 100 premiers
            self.state.focus_symbols = symbols[:max_symbols]

        self.save()
        logger.info(f"Focus symbols updated: {len(self.state.focus_symbols)} symbols (first 5: {self.state.focus_symbols[:5]})")

    # =========================================================================
    # TRENDS & NARRATIVES
    # =========================================================================

    def update_trends(self, trends: List[Dict]):
        """Met à jour les tendances actives"""
        self.state.active_trends = trends
        self.save()

    def update_narratives(self, narratives: List[str]):
        """Met à jour les narratifs actifs"""
        self.state.active_narratives = narratives
        self.save()

    # =========================================================================
    # TRADE RECORDING
    # =========================================================================

    def record_trade(self, trade: TradeRecord):
        """Enregistre un trade dans l'historique"""
        self.state.total_trades += 1

        if trade.outcome == 'WIN':
            self.state.total_wins += 1
        elif trade.outcome == 'LOSS':
            self.state.total_losses += 1

        self.state.total_pnl += trade.pnl

        # Calculer win rate
        if self.state.total_trades > 0:
            self.state.win_rate = self.state.total_wins / self.state.total_trades

        self.state.last_trade_at = trade.timestamp

        # Ajouter à l'historique quotidien
        self._update_daily_stats(trade)

        # Ajouter à l'historique complet des trades (pour audit/apprentissage)
        self.state.trade_history.append(trade.to_dict())

        # Garder les 5000 derniers trades (suffisant pour apprentissage ML)
        if len(self.state.trade_history) > 5000:
            self.state.trade_history = self.state.trade_history[-5000:]

        self.save()

        logger.info(f"Trade recorded: {trade.symbol} {trade.action} - {trade.outcome} ({trade.pnl:+.2f}€)")

    def _update_daily_stats(self, trade: TradeRecord):
        """Met à jour les stats quotidiennes"""
        today = date.today().isoformat()

        # Trouver ou créer les stats du jour
        daily_stats = None
        for stats in self.state.daily_history:
            if stats['date'] == today:
                daily_stats = stats
                break

        if daily_stats is None:
            daily_stats = DailyStats(date=today).to_dict()
            self.state.daily_history.append(daily_stats)

        # Mettre à jour
        daily_stats['trades_count'] += 1
        daily_stats['pnl'] += trade.pnl

        if trade.outcome == 'WIN':
            daily_stats['wins'] += 1
        elif trade.outcome == 'LOSS':
            daily_stats['losses'] += 1

        # Garder seulement les 30 derniers jours
        if len(self.state.daily_history) > 30:
            self.state.daily_history = self.state.daily_history[-30:]

    def get_closed_trades_today(self) -> List[TradeRecord]:
        """
        Retourne les trades fermés aujourd'hui (pour l'audit nocturne).

        Returns:
            Liste des TradeRecord fermés aujourd'hui
        """
        today = date.today().isoformat()
        closed_trades = []

        for trade_dict in self.state.trade_history:
            # Un trade est considéré "fermé aujourd'hui" si:
            # 1. Il a une exit_date qui correspond à aujourd'hui, OU
            # 2. Il a un outcome (WIN/LOSS) et son timestamp est aujourd'hui
            exit_date = trade_dict.get('exit_date')
            timestamp = trade_dict.get('timestamp', '')
            outcome = trade_dict.get('outcome')

            is_closed_today = False

            if exit_date and exit_date.startswith(today):
                is_closed_today = True
            elif outcome and timestamp.startswith(today):
                is_closed_today = True

            if is_closed_today:
                try:
                    closed_trades.append(TradeRecord.from_dict(trade_dict))
                except Exception as e:
                    logger.warning(f"Error parsing trade: {e}")

        logger.info(f"Found {len(closed_trades)} closed trades for {today}")
        return closed_trades

    def get_trades_by_date_range(
        self,
        start_date: str,
        end_date: Optional[str] = None
    ) -> List[TradeRecord]:
        """
        Retourne les trades dans une plage de dates.

        Args:
            start_date: Date de début (YYYY-MM-DD)
            end_date: Date de fin (YYYY-MM-DD), défaut = aujourd'hui

        Returns:
            Liste des TradeRecord dans la plage
        """
        if end_date is None:
            end_date = date.today().isoformat()

        trades = []
        for trade_dict in self.state.trade_history:
            exit_date = trade_dict.get('exit_date', trade_dict.get('timestamp', ''))
            if exit_date:
                trade_date = exit_date[:10]  # YYYY-MM-DD
                if start_date <= trade_date <= end_date:
                    try:
                        trades.append(TradeRecord.from_dict(trade_dict))
                    except Exception as e:
                        logger.warning(f"Error parsing trade: {e}")

        return trades

    def get_trade_history(self, limit: int = 100) -> List[TradeRecord]:
        """
        Retourne les derniers trades.

        Args:
            limit: Nombre maximum de trades à retourner

        Returns:
            Liste des TradeRecord (les plus récents en premier)
        """
        trades = []
        for trade_dict in reversed(self.state.trade_history[-limit:]):
            try:
                trades.append(TradeRecord.from_dict(trade_dict))
            except Exception as e:
                logger.warning(f"Error parsing trade: {e}")
        return trades

    # =========================================================================
    # ERRORS & WARNINGS
    # =========================================================================

    def log_error(self, error: str, context: Optional[Dict] = None):
        """Enregistre une erreur"""
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "context": context or {}
        }

        self.state.recent_errors.append(error_record)

        # Garder seulement les 50 dernières erreurs
        if len(self.state.recent_errors) > 50:
            self.state.recent_errors = self.state.recent_errors[-50:]

        self.save()
        logger.error(f"Error recorded: {error}")

    def add_warning(self, warning: str):
        """Ajoute un warning actif"""
        if warning not in self.state.active_warnings:
            self.state.active_warnings.append(warning)
            self.save()

    def clear_warning(self, warning: str):
        """Supprime un warning"""
        if warning in self.state.active_warnings:
            self.state.active_warnings.remove(warning)
            self.save()

    # =========================================================================
    # LEARNED RULES
    # =========================================================================

    def update_learned_rules_count(self, count: int, last_rule: Optional[str] = None):
        """Met à jour le compteur de règles apprises"""
        self.state.learned_rules_count = count
        if last_rule:
            self.state.last_rule_learned = last_rule
        self.save()

    # =========================================================================
    # TIMESTAMPS
    # =========================================================================

    def update_last_scan(self):
        """Enregistre le dernier scan"""
        self.state.last_scan_at = datetime.now().isoformat()
        self.save()

    def update_last_audit(self):
        """Enregistre le dernier audit"""
        self.state.last_audit_at = datetime.now().isoformat()
        self.save()

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_summary(self) -> Dict:
        """Retourne un résumé de l'état"""
        return {
            "phase": self.state.current_phase,
            "market_regime": self.state.market_regime,
            "capital": {
                "initial": self.state.initial_capital,
                "current": self.state.current_capital,
                "peak": self.state.peak_capital,
                "drawdown": self.get_drawdown()
            },
            "positions": {
                "count": len(self.state.positions),
                "symbols": list(self.state.positions.keys())
            },
            "performance": {
                "total_trades": self.state.total_trades,
                "wins": self.state.total_wins,
                "losses": self.state.total_losses,
                "win_rate": self.state.win_rate,
                "total_pnl": self.state.total_pnl
            },
            "focus": {
                "watchlist_size": len(self.state.watchlist),
                "focus_symbols": self.state.focus_symbols[:5]
            },
            "rules": {
                "learned_count": self.state.learned_rules_count,
                "last_learned": self.state.last_rule_learned
            },
            "last_actions": {
                "scan": self.state.last_scan_at,
                "trade": self.state.last_trade_at,
                "audit": self.state.last_audit_at
            },
            "health": {
                "errors_24h": len([e for e in self.state.recent_errors
                                  if datetime.fromisoformat(e['timestamp']) > datetime.now() - timedelta(hours=24)]),
                "active_warnings": len(self.state.active_warnings)
            }
        }


# =========================================================================
# FACTORY
# =========================================================================

_state_manager: Optional[StateManager] = None


def get_state_manager(data_dir: str = "data") -> StateManager:
    """Retourne l'instance singleton du StateManager"""
    global _state_manager

    if _state_manager is None:
        _state_manager = StateManager(data_dir=data_dir)

    return _state_manager


# =========================================================================
# TESTS
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test du StateManager
    manager = StateManager(data_dir="data/test_state")

    # Test phase
    manager.set_phase(AgentPhase.DISCOVERY)
    print(f"Phase: {manager.get_phase()}")

    # Test capital
    manager.update_capital(1600.0)
    print(f"Drawdown: {manager.get_drawdown()*100:.1f}%")

    # Test position
    position = Position(
        symbol="AAPL",
        quantity=10,
        entry_price=150.0,
        entry_date=datetime.now().isoformat(),
        stop_loss=145.0,
        thesis="RSI breakout"
    )
    manager.add_position(position)

    # Test focus
    manager.update_focus_symbols(["NVDA", "AMD", "TSLA"])

    # Afficher le résumé
    import json
    print("\nAgent Summary:")
    print(json.dumps(manager.get_summary(), indent=2))
