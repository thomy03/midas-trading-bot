"""
Strategy Evolver - Auto-apprentissage et évolution de la stratégie
Analyse les échecs, ajuste les paramètres, propose des modifications.
"""

import asyncio
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class AdaptationLevel(Enum):
    """Niveau d'adaptation requis"""
    TWEAK = "tweak"         # Ajustement automatique
    PAUSE = "pause"         # Désactivation temporaire + notification
    PROPOSE = "propose"     # Proposition nécessitant approbation
    ALERT = "alert"         # Arrêt complet obligatoire


class FailureType(Enum):
    """Types d'échecs identifiés"""
    FALSE_BREAKOUT = "false_breakout"       # RSI breakout non suivi
    VOLUME_TRAP = "volume_trap"             # Volume spike sans mouvement
    SENTIMENT_MISLEAD = "sentiment_mislead"  # Sentiment positif mais baisse
    TIMING_OFF = "timing_off"               # Bon signal, mauvais timing
    SECTOR_WEAKNESS = "sector_weakness"     # Secteur en faiblesse générale
    MARKET_REVERSAL = "market_reversal"     # Retournement marché global
    STOP_TOO_TIGHT = "stop_too_tight"       # Stop touché avant rebond
    UNKNOWN = "unknown"


# Paramètres ajustables automatiquement (dans des bornes sûres)
AUTO_ADJUSTABLE_PARAMS = {
    'rsi_breakout_threshold': {'min': 45, 'max': 65, 'default': 50, 'step': 2},
    'heat_score_minimum': {'min': 0.4, 'max': 0.8, 'default': 0.5, 'step': 0.05},
    'volume_multiplier_min': {'min': 1.0, 'max': 2.5, 'default': 1.5, 'step': 0.25},
    'confidence_score_min': {'min': 50, 'max': 80, 'default': 55, 'step': 5},
    'max_position_count': {'min': 1, 'max': 5, 'default': 3, 'step': 1},
    'stop_loss_buffer_pct': {'min': 0.01, 'max': 0.05, 'default': 0.02, 'step': 0.005},
}

# Modifications nécessitant une approbation
REQUIRE_APPROVAL = [
    'disable_strategy',
    'exclude_market_cap_range',
    'exclude_sector',
    'change_timeframe',
    'modify_guardrails',
    'change_capital_allocation',
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TradeResult:
    """Résultat d'un trade pour analyse"""
    trade_id: str
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    pnl_pct: float
    pnl_amount: float

    # Contexte au moment de l'entrée
    signal_type: str  # "rsi_breakout", "heat_spike", etc.
    confidence_score: float
    heat_score: float
    volume_ratio: float
    rsi_value: float
    ema_alignment: bool
    sector: Optional[str] = None

    # Contexte de sortie
    exit_reason: str = ""  # "stop_loss", "take_profit", "manual", "trailing"
    max_drawdown_pct: float = 0.0
    max_gain_pct: float = 0.0
    hold_duration_hours: float = 0.0

    # Métadonnées
    was_profitable: bool = False
    failure_type: Optional[FailureType] = None


@dataclass
class FailurePattern:
    """Pattern d'échec identifié"""
    failure_type: FailureType
    occurrence_count: int
    total_loss: float
    avg_loss_per_trade: float

    # Conditions communes
    common_conditions: Dict[str, Any] = field(default_factory=dict)

    # Exemples
    example_trades: List[str] = field(default_factory=list)

    # Analyse
    root_cause_hypothesis: str = ""
    suggested_fix: str = ""
    confidence: float = 0.0


@dataclass
class Proposal:
    """Proposition de modification"""
    proposal_id: str
    created_at: datetime
    adaptation_level: AdaptationLevel

    # Ce qui est proposé
    parameter: str
    current_value: Any
    proposed_value: Any

    # Justification
    reason: str
    failure_patterns: List[FailureType]
    expected_improvement: str

    # Risques
    risk_assessment: str
    rollback_plan: str

    # Statut
    status: str = "pending"  # pending, approved, rejected, applied
    applied_at: Optional[datetime] = None
    result: Optional[str] = None


@dataclass
class StrategyHealth:
    """Santé globale de la stratégie"""
    timestamp: datetime

    # Métriques de performance
    win_rate: float  # 0-1
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_pnl: float

    # Tendance récente (7 derniers jours)
    recent_win_rate: float
    recent_trades: int
    streak: int  # Positif = wins, négatif = losses

    # Santé
    health_score: float  # 0-100
    status: str  # "healthy", "warning", "critical", "stopped"

    # Actions recommandées
    recommended_action: AdaptationLevel
    issues: List[str] = field(default_factory=list)


@dataclass
class EvolverConfig:
    """Configuration du Strategy Evolver"""
    # Seuils de santé
    healthy_win_rate: float = 0.50
    warning_win_rate: float = 0.40
    critical_win_rate: float = 0.30

    # Seuils de streak
    max_consecutive_losses: int = 3

    # Fenêtres d'analyse
    recent_window_days: int = 7
    analysis_min_trades: int = 5

    # Auto-adjustment
    auto_adjust_enabled: bool = True
    adjustment_cooldown_hours: int = 24

    # Notifications
    notify_on_warning: bool = True
    notify_on_adjustment: bool = True


# =============================================================================
# STRATEGY EVOLVER
# =============================================================================

class StrategyEvolver:
    """
    Évolue et adapte la stratégie de trading basée sur les résultats.

    4 niveaux d'adaptation:
    1. TWEAK: Ajustement automatique de paramètres (dans les bornes)
    2. PAUSE: Désactivation temporaire + notification
    3. PROPOSE: Suggestion nécessitant approbation humaine
    4. ALERT: Arrêt complet obligatoire (guardrails)
    """

    def __init__(self, config: Optional[EvolverConfig] = None):
        self.config = config or EvolverConfig()

        # Historique des trades
        self._trades: List[TradeResult] = []

        # Paramètres actuels (copie modifiable)
        self._current_params: Dict[str, Any] = {
            param: info['default']
            for param, info in AUTO_ADJUSTABLE_PARAMS.items()
        }

        # Historique des ajustements
        self._adjustments: List[Dict] = []

        # Propositions en attente
        self._pending_proposals: List[Proposal] = []

        # Patterns identifiés
        self._failure_patterns: Dict[FailureType, FailurePattern] = {}

        # Dernière analyse
        self._last_health_check: Optional[StrategyHealth] = None
        self._last_adjustment_time: Optional[datetime] = None

        # Callbacks
        self._notification_callback = None
        self._approval_callback = None

        # Persistence
        self._data_dir = Path("data/evolver")
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Lock
        self._lock = asyncio.Lock()

        logger.info("StrategyEvolver initialized")

    async def initialize(self):
        """Initialise le evolver"""
        await self._load_state()
        logger.info(f"StrategyEvolver ready - {len(self._trades)} trades in history")

    def set_notification_callback(self, callback):
        """Définit le callback pour les notifications"""
        self._notification_callback = callback

    def set_approval_callback(self, callback):
        """Définit le callback pour les demandes d'approbation"""
        self._approval_callback = callback

    # -------------------------------------------------------------------------
    # TRADE RECORDING
    # -------------------------------------------------------------------------

    async def record_trade(self, trade: TradeResult):
        """Enregistre un nouveau trade pour analyse"""
        async with self._lock:
            # Classifier l'échec si applicable
            if not trade.was_profitable:
                trade.failure_type = self._classify_failure(trade)

            self._trades.append(trade)

            # Sauvegarder
            await self._save_trades()

            # Analyse immédiate si streak négatif
            await self._check_immediate_alerts(trade)

            logger.info(
                f"Trade recorded: {trade.symbol} "
                f"{'WIN' if trade.was_profitable else 'LOSS'} "
                f"{trade.pnl_pct:.1f}%"
            )

    def _classify_failure(self, trade: TradeResult) -> FailureType:
        """Classifie le type d'échec"""

        # Stop touché rapidement alors que le trade aurait été gagnant après
        if trade.exit_reason == "stop_loss" and trade.max_gain_pct > abs(trade.pnl_pct):
            return FailureType.STOP_TOO_TIGHT

        # Volume élevé mais pas de mouvement
        if trade.volume_ratio > 2.0 and abs(trade.pnl_pct) < 1.0:
            return FailureType.VOLUME_TRAP

        # RSI breakout non confirmé
        if trade.signal_type == "rsi_breakout" and trade.pnl_pct < -2.0:
            return FailureType.FALSE_BREAKOUT

        # Bon timing mais mauvaise direction (drawdown puis gain)
        if trade.max_gain_pct > 3.0 and trade.pnl_pct < 0:
            return FailureType.TIMING_OFF

        # Sentiment positif mais perte
        if trade.heat_score > 0.7 and trade.pnl_pct < -3.0:
            return FailureType.SENTIMENT_MISLEAD

        return FailureType.UNKNOWN

    async def _check_immediate_alerts(self, trade: TradeResult):
        """Vérifie si des alertes immédiates sont nécessaires"""
        # Compter les pertes consécutives récentes
        recent_trades = self._trades[-self.config.max_consecutive_losses:]
        consecutive_losses = sum(
            1 for t in reversed(recent_trades) if not t.was_profitable
        )

        if consecutive_losses >= self.config.max_consecutive_losses:
            await self._trigger_alert(
                AdaptationLevel.PAUSE,
                f"{consecutive_losses} pertes consécutives détectées",
                {"consecutive_losses": consecutive_losses}
            )

    # -------------------------------------------------------------------------
    # HEALTH CHECK
    # -------------------------------------------------------------------------

    async def check_health(self) -> StrategyHealth:
        """Analyse la santé globale de la stratégie"""
        async with self._lock:
            now = datetime.now()

            if len(self._trades) < self.config.analysis_min_trades:
                return StrategyHealth(
                    timestamp=now,
                    win_rate=0.5,
                    profit_factor=1.0,
                    avg_win=0,
                    avg_loss=0,
                    total_pnl=0,
                    recent_win_rate=0.5,
                    recent_trades=len(self._trades),
                    streak=0,
                    health_score=50,
                    status="insufficient_data",
                    recommended_action=AdaptationLevel.TWEAK,
                    issues=["Pas assez de trades pour analyse"]
                )

            # Métriques globales
            wins = [t for t in self._trades if t.was_profitable]
            losses = [t for t in self._trades if not t.was_profitable]

            win_rate = len(wins) / len(self._trades) if self._trades else 0

            total_wins = sum(t.pnl_amount for t in wins)
            total_losses = abs(sum(t.pnl_amount for t in losses))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

            avg_win = total_wins / len(wins) if wins else 0
            avg_loss = total_losses / len(losses) if losses else 0

            total_pnl = sum(t.pnl_amount for t in self._trades)

            # Métriques récentes
            recent_cutoff = now - timedelta(days=self.config.recent_window_days)
            recent_trades = [t for t in self._trades if t.entry_date > recent_cutoff]
            recent_wins = [t for t in recent_trades if t.was_profitable]
            recent_win_rate = len(recent_wins) / len(recent_trades) if recent_trades else 0.5

            # Streak
            streak = 0
            for trade in reversed(self._trades):
                if trade.was_profitable:
                    if streak >= 0:
                        streak += 1
                    else:
                        break
                else:
                    if streak <= 0:
                        streak -= 1
                    else:
                        break

            # Score de santé (0-100)
            health_score = self._calculate_health_score(
                win_rate, profit_factor, recent_win_rate, streak
            )

            # Déterminer le statut et l'action
            status, action, issues = self._determine_status(
                win_rate, recent_win_rate, streak, health_score
            )

            health = StrategyHealth(
                timestamp=now,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                total_pnl=total_pnl,
                recent_win_rate=recent_win_rate,
                recent_trades=len(recent_trades),
                streak=streak,
                health_score=health_score,
                status=status,
                recommended_action=action,
                issues=issues
            )

            self._last_health_check = health
            return health

    def _calculate_health_score(
        self,
        win_rate: float,
        profit_factor: float,
        recent_win_rate: float,
        streak: int
    ) -> float:
        """Calcule un score de santé 0-100"""
        score = 0

        # Win rate contribution (0-40 points)
        if win_rate >= 0.60:
            score += 40
        elif win_rate >= 0.50:
            score += 30
        elif win_rate >= 0.40:
            score += 20
        elif win_rate >= 0.30:
            score += 10

        # Profit factor (0-25 points)
        if profit_factor >= 2.0:
            score += 25
        elif profit_factor >= 1.5:
            score += 20
        elif profit_factor >= 1.2:
            score += 15
        elif profit_factor >= 1.0:
            score += 10

        # Recent performance (0-25 points)
        if recent_win_rate >= 0.60:
            score += 25
        elif recent_win_rate >= 0.50:
            score += 20
        elif recent_win_rate >= 0.40:
            score += 10

        # Streak bonus/malus (±10 points)
        if streak >= 3:
            score += 10
        elif streak >= 1:
            score += 5
        elif streak <= -3:
            score -= 10
        elif streak <= -1:
            score -= 5

        return max(0, min(100, score))

    def _determine_status(
        self,
        win_rate: float,
        recent_win_rate: float,
        streak: int,
        health_score: float
    ) -> Tuple[str, AdaptationLevel, List[str]]:
        """Détermine le statut et l'action recommandée"""
        issues = []

        # Critical
        if (win_rate < self.config.critical_win_rate or
            streak <= -self.config.max_consecutive_losses or
            health_score < 20):

            if win_rate < self.config.critical_win_rate:
                issues.append(f"Win rate critique: {win_rate:.0%}")
            if streak <= -self.config.max_consecutive_losses:
                issues.append(f"Streak négatif: {streak}")

            return "critical", AdaptationLevel.PAUSE, issues

        # Warning
        if (win_rate < self.config.warning_win_rate or
            recent_win_rate < self.config.warning_win_rate or
            health_score < 40):

            if win_rate < self.config.warning_win_rate:
                issues.append(f"Win rate faible: {win_rate:.0%}")
            if recent_win_rate < self.config.warning_win_rate:
                issues.append(f"Performance récente en baisse: {recent_win_rate:.0%}")

            return "warning", AdaptationLevel.PROPOSE, issues

        # Healthy but could improve
        if health_score < 70:
            if win_rate < self.config.healthy_win_rate:
                issues.append(f"Win rate améliorable: {win_rate:.0%}")
            return "healthy", AdaptationLevel.TWEAK, issues

        return "healthy", AdaptationLevel.TWEAK, []

    # -------------------------------------------------------------------------
    # FAILURE ANALYSIS
    # -------------------------------------------------------------------------

    async def analyze_failures(self) -> List[FailurePattern]:
        """Analyse les patterns d'échec"""
        async with self._lock:
            # Grouper les trades perdants par type d'échec
            failures_by_type: Dict[FailureType, List[TradeResult]] = defaultdict(list)

            for trade in self._trades:
                if not trade.was_profitable and trade.failure_type:
                    failures_by_type[trade.failure_type].append(trade)

            patterns = []

            for failure_type, trades in failures_by_type.items():
                if len(trades) < 2:
                    continue

                total_loss = sum(t.pnl_amount for t in trades)
                avg_loss = total_loss / len(trades)

                # Trouver les conditions communes
                common_conditions = self._find_common_conditions(trades)

                # Générer hypothèse et suggestion
                hypothesis, fix = self._generate_hypothesis(failure_type, common_conditions)

                pattern = FailurePattern(
                    failure_type=failure_type,
                    occurrence_count=len(trades),
                    total_loss=total_loss,
                    avg_loss_per_trade=avg_loss,
                    common_conditions=common_conditions,
                    example_trades=[t.trade_id for t in trades[:3]],
                    root_cause_hypothesis=hypothesis,
                    suggested_fix=fix,
                    confidence=min(0.9, len(trades) * 0.15)
                )

                patterns.append(pattern)
                self._failure_patterns[failure_type] = pattern

            # Trier par impact (pertes totales)
            patterns.sort(key=lambda p: p.total_loss)

            return patterns

    def _find_common_conditions(self, trades: List[TradeResult]) -> Dict[str, Any]:
        """Trouve les conditions communes entre les trades"""
        conditions = {}

        # RSI values
        rsi_values = [t.rsi_value for t in trades]
        if rsi_values:
            conditions['avg_rsi'] = sum(rsi_values) / len(rsi_values)
            conditions['rsi_range'] = (min(rsi_values), max(rsi_values))

        # Heat scores
        heat_scores = [t.heat_score for t in trades]
        if heat_scores:
            conditions['avg_heat'] = sum(heat_scores) / len(heat_scores)

        # Volume ratios
        volume_ratios = [t.volume_ratio for t in trades]
        if volume_ratios:
            conditions['avg_volume_ratio'] = sum(volume_ratios) / len(volume_ratios)

        # Confidence scores
        conf_scores = [t.confidence_score for t in trades]
        if conf_scores:
            conditions['avg_confidence'] = sum(conf_scores) / len(conf_scores)

        # EMA alignment
        ema_aligned = [t.ema_alignment for t in trades]
        conditions['ema_alignment_rate'] = sum(ema_aligned) / len(ema_aligned)

        # Sectors
        sectors = [t.sector for t in trades if t.sector]
        if sectors:
            from collections import Counter
            sector_counts = Counter(sectors)
            conditions['common_sectors'] = sector_counts.most_common(3)

        return conditions

    def _generate_hypothesis(
        self,
        failure_type: FailureType,
        conditions: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Génère une hypothèse et une suggestion de fix"""

        hypotheses = {
            FailureType.FALSE_BREAKOUT: (
                "Les breakouts RSI sont déclenchés trop tôt, avant confirmation du momentum",
                "Augmenter le seuil RSI breakout ou attendre confirmation volume"
            ),
            FailureType.VOLUME_TRAP: (
                "Le volume élevé attire mais n'est pas suivi par un mouvement directionnel",
                "Augmenter le seuil minimum de volume ratio"
            ),
            FailureType.SENTIMENT_MISLEAD: (
                "Le sentiment social ne se traduit pas en mouvement de prix",
                "Réduire le poids du heat score dans la décision"
            ),
            FailureType.TIMING_OFF: (
                "Les entrées sont faites trop tôt ou trop tard dans le mouvement",
                "Ajuster le timing d'entrée ou utiliser des ordres limites"
            ),
            FailureType.STOP_TOO_TIGHT: (
                "Les stop-loss sont trop serrés et se déclenchent avant le rebond",
                "Augmenter le buffer de stop-loss"
            ),
            FailureType.SECTOR_WEAKNESS: (
                "Les trades sont dans des secteurs globalement faibles",
                "Ajouter un filtre de force sectorielle"
            ),
            FailureType.MARKET_REVERSAL: (
                "Les entrées sont contre la tendance générale du marché",
                "Ajouter un filtre de tendance marché (SPY trend)"
            ),
            FailureType.UNKNOWN: (
                "Pattern d'échec non identifié, analyse manuelle requise",
                "Revoir manuellement les trades concernés"
            ),
        }

        return hypotheses.get(failure_type, hypotheses[FailureType.UNKNOWN])

    # -------------------------------------------------------------------------
    # AUTO ADJUSTMENT
    # -------------------------------------------------------------------------

    async def propose_adjustments(self) -> List[Proposal]:
        """Génère des propositions d'ajustement basées sur l'analyse"""
        proposals = []

        # Analyser les échecs
        patterns = await self.analyze_failures()

        for pattern in patterns:
            if pattern.confidence < 0.3 or pattern.occurrence_count < 3:
                continue

            proposal = self._create_proposal_for_pattern(pattern)
            if proposal:
                proposals.append(proposal)
                self._pending_proposals.append(proposal)

        return proposals

    def _create_proposal_for_pattern(self, pattern: FailurePattern) -> Optional[Proposal]:
        """Crée une proposition pour un pattern d'échec"""
        import uuid
        now = datetime.now()

        # Mapper les patterns aux paramètres
        param_mapping = {
            FailureType.FALSE_BREAKOUT: ('rsi_breakout_threshold', 'increase'),
            FailureType.VOLUME_TRAP: ('volume_multiplier_min', 'increase'),
            FailureType.SENTIMENT_MISLEAD: ('heat_score_minimum', 'increase'),
            FailureType.STOP_TOO_TIGHT: ('stop_loss_buffer_pct', 'increase'),
        }

        if pattern.failure_type not in param_mapping:
            return None

        param, direction = param_mapping[pattern.failure_type]
        param_info = AUTO_ADJUSTABLE_PARAMS.get(param)

        if not param_info:
            return None

        current_value = self._current_params.get(param, param_info['default'])

        # Calculer nouvelle valeur
        step = param_info['step']
        if direction == 'increase':
            new_value = min(current_value + step, param_info['max'])
        else:
            new_value = max(current_value - step, param_info['min'])

        if new_value == current_value:
            return None

        # Déterminer le niveau d'adaptation
        level = AdaptationLevel.TWEAK if self.config.auto_adjust_enabled else AdaptationLevel.PROPOSE

        return Proposal(
            proposal_id=str(uuid.uuid4())[:8],
            created_at=now,
            adaptation_level=level,
            parameter=param,
            current_value=current_value,
            proposed_value=new_value,
            reason=pattern.root_cause_hypothesis,
            failure_patterns=[pattern.failure_type],
            expected_improvement=f"Réduction des {pattern.failure_type.value} de ~{pattern.occurrence_count} trades",
            risk_assessment="Risque faible - ajustement dans les bornes sûres",
            rollback_plan=f"Revenir à {param}={current_value}",
            status="pending"
        )

    async def auto_adjust(self, proposal: Proposal) -> bool:
        """Applique un ajustement automatique"""
        async with self._lock:
            # Vérifier le cooldown
            if self._last_adjustment_time:
                cooldown = timedelta(hours=self.config.adjustment_cooldown_hours)
                if datetime.now() - self._last_adjustment_time < cooldown:
                    logger.warning("Adjustment cooldown active, skipping")
                    return False

            # Vérifier que c'est un TWEAK
            if proposal.adaptation_level != AdaptationLevel.TWEAK:
                logger.warning(f"Cannot auto-adjust {proposal.adaptation_level}, requires approval")
                return False

            # Vérifier les bornes
            param_info = AUTO_ADJUSTABLE_PARAMS.get(proposal.parameter)
            if not param_info:
                return False

            if not (param_info['min'] <= proposal.proposed_value <= param_info['max']):
                logger.error(f"Proposed value {proposal.proposed_value} out of bounds")
                return False

            # Appliquer
            old_value = self._current_params.get(proposal.parameter)
            self._current_params[proposal.parameter] = proposal.proposed_value

            # Enregistrer
            self._adjustments.append({
                'timestamp': datetime.now().isoformat(),
                'proposal_id': proposal.proposal_id,
                'parameter': proposal.parameter,
                'old_value': old_value,
                'new_value': proposal.proposed_value,
                'reason': proposal.reason
            })

            proposal.status = "applied"
            proposal.applied_at = datetime.now()

            self._last_adjustment_time = datetime.now()

            await self._save_state()

            # Notification
            if self.config.notify_on_adjustment and self._notification_callback:
                await self._notification_callback(
                    f"🔧 Ajustement automatique: {proposal.parameter} "
                    f"{old_value} → {proposal.proposed_value}\n"
                    f"Raison: {proposal.reason}"
                )

            logger.info(
                f"Auto-adjusted {proposal.parameter}: {old_value} → {proposal.proposed_value}"
            )

            return True

    async def request_approval(self, proposal: Proposal):
        """Demande une approbation humaine pour une proposition"""
        if self._approval_callback:
            await self._approval_callback(proposal)
        else:
            # Fallback: notification
            if self._notification_callback:
                await self._notification_callback(
                    f"📋 Proposition nécessitant approbation:\n"
                    f"Paramètre: {proposal.parameter}\n"
                    f"Changement: {proposal.current_value} → {proposal.proposed_value}\n"
                    f"Raison: {proposal.reason}\n"
                    f"ID: {proposal.proposal_id}"
                )

    async def approve_proposal(self, proposal_id: str) -> bool:
        """Approuve une proposition en attente"""
        for proposal in self._pending_proposals:
            if proposal.proposal_id == proposal_id:
                # Forcer le niveau à TWEAK pour permettre l'application
                original_level = proposal.adaptation_level
                proposal.adaptation_level = AdaptationLevel.TWEAK

                success = await self.auto_adjust(proposal)

                if success:
                    proposal.status = "approved"
                else:
                    proposal.adaptation_level = original_level

                return success

        return False

    async def reject_proposal(self, proposal_id: str, reason: str = ""):
        """Rejette une proposition"""
        for proposal in self._pending_proposals:
            if proposal.proposal_id == proposal_id:
                proposal.status = "rejected"
                proposal.result = reason or "Rejected by user"
                break

    # -------------------------------------------------------------------------
    # ALERT SYSTEM
    # -------------------------------------------------------------------------

    async def _trigger_alert(
        self,
        level: AdaptationLevel,
        message: str,
        context: Dict[str, Any]
    ):
        """Déclenche une alerte"""
        alert_icons = {
            AdaptationLevel.TWEAK: "ℹ️",
            AdaptationLevel.PAUSE: "⚠️",
            AdaptationLevel.PROPOSE: "📋",
            AdaptationLevel.ALERT: "🚨"
        }

        icon = alert_icons.get(level, "❓")

        if self._notification_callback:
            await self._notification_callback(
                f"{icon} [{level.value.upper()}] {message}\n"
                f"Context: {json.dumps(context, default=str)}"
            )

        logger.warning(f"Alert triggered: [{level.value}] {message}")

        # Pour PAUSE et ALERT, recommander une action
        if level in (AdaptationLevel.PAUSE, AdaptationLevel.ALERT):
            await self._recommend_pause()

    async def _recommend_pause(self):
        """Recommande une pause du trading"""
        if self._notification_callback:
            await self._notification_callback(
                "⏸️ RECOMMANDATION: Pause du trading recommandée\n"
                "Le système a détecté une série d'échecs significative.\n"
                "Action suggérée: Analyser les trades récents avant de continuer."
            )

    # -------------------------------------------------------------------------
    # GETTERS
    # -------------------------------------------------------------------------

    def get_current_params(self) -> Dict[str, Any]:
        """Retourne les paramètres actuels"""
        return self._current_params.copy()

    def get_param(self, param: str) -> Any:
        """Retourne la valeur d'un paramètre"""
        return self._current_params.get(param)

    def get_pending_proposals(self) -> List[Proposal]:
        """Retourne les propositions en attente"""
        return [p for p in self._pending_proposals if p.status == "pending"]

    def get_adjustments_history(self) -> List[Dict]:
        """Retourne l'historique des ajustements"""
        return self._adjustments.copy()

    def get_failure_patterns(self) -> Dict[FailureType, FailurePattern]:
        """Retourne les patterns d'échec identifiés"""
        return self._failure_patterns.copy()

    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------

    async def _save_state(self):
        """Sauvegarde l'état"""
        state = {
            'current_params': self._current_params,
            'adjustments': self._adjustments,
            'last_adjustment': self._last_adjustment_time.isoformat() if self._last_adjustment_time else None,
        }

        path = self._data_dir / "evolver_state.json"
        try:
            with open(path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save evolver state: {e}")

    async def _load_state(self):
        """Charge l'état"""
        path = self._data_dir / "evolver_state.json"
        if path.exists():
            try:
                with open(path, 'r') as f:
                    state = json.load(f)

                self._current_params = state.get('current_params', self._current_params)
                self._adjustments = state.get('adjustments', [])

                if state.get('last_adjustment'):
                    self._last_adjustment_time = datetime.fromisoformat(state['last_adjustment'])

                logger.info(f"Loaded evolver state - {len(self._adjustments)} adjustments")
            except Exception as e:
                logger.warning(f"Could not load evolver state: {e}")

        # Charger les trades
        await self._load_trades()

    async def _save_trades(self):
        """Sauvegarde les trades"""
        path = self._data_dir / "trades_history.json"
        try:
            trades_data = []
            for t in self._trades[-1000:]:  # Garder les 1000 derniers
                trades_data.append({
                    'trade_id': t.trade_id,
                    'symbol': t.symbol,
                    'entry_date': t.entry_date.isoformat(),
                    'exit_date': t.exit_date.isoformat() if t.exit_date else None,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'stop_loss': t.stop_loss,
                    'pnl_pct': t.pnl_pct,
                    'pnl_amount': t.pnl_amount,
                    'signal_type': t.signal_type,
                    'confidence_score': t.confidence_score,
                    'heat_score': t.heat_score,
                    'volume_ratio': t.volume_ratio,
                    'rsi_value': t.rsi_value,
                    'ema_alignment': t.ema_alignment,
                    'sector': t.sector,
                    'exit_reason': t.exit_reason,
                    'was_profitable': t.was_profitable,
                    'failure_type': t.failure_type.value if t.failure_type else None,
                })

            with open(path, 'w') as f:
                json.dump(trades_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save trades: {e}")

    async def _load_trades(self):
        """Charge les trades"""
        path = self._data_dir / "trades_history.json"
        if path.exists():
            try:
                with open(path, 'r') as f:
                    trades_data = json.load(f)

                for td in trades_data:
                    trade = TradeResult(
                        trade_id=td['trade_id'],
                        symbol=td['symbol'],
                        entry_date=datetime.fromisoformat(td['entry_date']),
                        exit_date=datetime.fromisoformat(td['exit_date']) if td.get('exit_date') else None,
                        entry_price=td['entry_price'],
                        exit_price=td.get('exit_price'),
                        stop_loss=td['stop_loss'],
                        pnl_pct=td['pnl_pct'],
                        pnl_amount=td['pnl_amount'],
                        signal_type=td['signal_type'],
                        confidence_score=td['confidence_score'],
                        heat_score=td['heat_score'],
                        volume_ratio=td['volume_ratio'],
                        rsi_value=td['rsi_value'],
                        ema_alignment=td['ema_alignment'],
                        sector=td.get('sector'),
                        exit_reason=td.get('exit_reason', ''),
                        was_profitable=td['was_profitable'],
                        failure_type=FailureType(td['failure_type']) if td.get('failure_type') else None,
                    )
                    self._trades.append(trade)

                logger.info(f"Loaded {len(self._trades)} trades from history")
            except Exception as e:
                logger.warning(f"Could not load trades: {e}")

    async def close(self):
        """Ferme proprement"""
        await self._save_state()
        await self._save_trades()
        logger.info("StrategyEvolver closed")


# =============================================================================
# FACTORY
# =============================================================================

_strategy_evolver: Optional[StrategyEvolver] = None


async def get_strategy_evolver(config: Optional[EvolverConfig] = None) -> StrategyEvolver:
    """Retourne l'instance singleton du StrategyEvolver"""
    global _strategy_evolver
    if _strategy_evolver is None:
        _strategy_evolver = StrategyEvolver(config)
        await _strategy_evolver.initialize()
    return _strategy_evolver
