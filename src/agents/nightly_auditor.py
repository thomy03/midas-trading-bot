"""
Nightly Auditor Module
Boucle d'auto-apprentissage nocturne pour améliorer les décisions de trading.

Fonctionnalités:
1. Analyse des trades de la journée
2. Identification des patterns gagnants/perdants
3. Génération de règles dans learned_guidelines.json
4. Ajustement des paramètres de confiance

Le Nightly Auditor s'exécute chaque soir après la clôture du marché (20:00 ET)
et analyse les trades pour en tirer des leçons.

Usage:
    from src.agents.nightly_auditor import NightlyAuditor

    auditor = NightlyAuditor(llm_client=grok_scanner)
    await auditor.run_audit()

    # Obtenir les guidelines apprises
    guidelines = auditor.get_learned_guidelines()
"""

import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class TradeOutcome(Enum):
    """Résultat d'un trade"""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    OPEN = "open"


class LessonType(Enum):
    """Types de leçons apprises"""
    AVOID_PATTERN = "avoid_pattern"      # Pattern à éviter
    SEEK_PATTERN = "seek_pattern"        # Pattern à rechercher
    TIMING_RULE = "timing_rule"          # Règle de timing
    SIZE_RULE = "size_rule"              # Règle de sizing
    SECTOR_RULE = "sector_rule"          # Règle sectorielle
    VOLATILITY_RULE = "volatility_rule"  # Règle de volatilité
    CONFIRMATION_RULE = "confirmation_rule"  # Règle de confirmation


@dataclass
class TradeAnalysis:
    """Analyse d'un trade individuel"""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    outcome: TradeOutcome
    pnl: float
    pnl_pct: float

    # Contexte au moment de l'entrée
    signal_source: str
    confidence_score: float
    rsi_at_entry: float
    volume_ratio: float
    market_regime: str
    sector: str

    # Facteurs identifiés
    contributing_factors: List[str] = field(default_factory=list)
    mistakes_identified: List[str] = field(default_factory=list)


@dataclass
class LearnedGuideline:
    """Règle apprise par l'auditor"""
    id: str
    type: LessonType
    rule: str
    description: str
    confidence: float  # 0-1, augmente avec les confirmations
    created_at: datetime
    last_confirmed: datetime
    confirmation_count: int
    invalidation_count: int

    # Contexte
    based_on_trades: List[str]  # IDs des trades source
    conditions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.type.value,
            'rule': self.rule,
            'description': self.description,
            'confidence': self.confidence,
            'created_at': self.created_at.isoformat(),
            'last_confirmed': self.last_confirmed.isoformat(),
            'confirmation_count': self.confirmation_count,
            'invalidation_count': self.invalidation_count,
            'based_on_trades': self.based_on_trades,
            'conditions': self.conditions
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'LearnedGuideline':
        return cls(
            id=data['id'],
            type=LessonType(data['type']),
            rule=data['rule'],
            description=data['description'],
            confidence=data['confidence'],
            created_at=datetime.fromisoformat(data['created_at']),
            last_confirmed=datetime.fromisoformat(data['last_confirmed']),
            confirmation_count=data['confirmation_count'],
            invalidation_count=data['invalidation_count'],
            based_on_trades=data.get('based_on_trades', []),
            conditions=data.get('conditions', {})
        )


@dataclass
class AuditReport:
    """Rapport d'audit quotidien"""
    date: datetime
    trades_analyzed: int
    wins: int
    losses: int
    total_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float

    # Leçons
    new_guidelines: List[LearnedGuideline]
    confirmed_guidelines: List[str]  # IDs
    invalidated_guidelines: List[str]  # IDs

    # Recommandations
    recommendations: List[str]
    risk_adjustments: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            'date': self.date.isoformat(),
            'trades_analyzed': self.trades_analyzed,
            'wins': self.wins,
            'losses': self.losses,
            'total_pnl': self.total_pnl,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'new_guidelines': [g.to_dict() for g in self.new_guidelines],
            'confirmed_guidelines': self.confirmed_guidelines,
            'invalidated_guidelines': self.invalidated_guidelines,
            'recommendations': self.recommendations,
            'risk_adjustments': self.risk_adjustments
        }


# =============================================================================
# NIGHTLY AUDITOR
# =============================================================================

class NightlyAuditor:
    """
    Auditeur nocturne pour l'auto-apprentissage.

    Analyse les trades de la journée et génère des règles
    dans learned_guidelines.json pour améliorer les futures décisions.

    Processus:
    1. Charger les trades du jour depuis l'état de l'agent
    2. Analyser chaque trade (contexte, résultat, facteurs)
    3. Identifier les patterns gagnants/perdants
    4. Générer ou confirmer des guidelines
    5. Sauvegarder le rapport et les guidelines
    """

    # Seuils de confiance
    MIN_CONFIDENCE_NEW_RULE = 0.3  # Confiance initiale d'une nouvelle règle
    CONFIDENCE_INCREMENT = 0.1     # Incrément par confirmation
    CONFIDENCE_DECREMENT = 0.15    # Décrément par invalidation
    MIN_CONFIDENCE_ACTIVE = 0.2    # En dessous = règle désactivée
    MAX_CONFIDENCE = 0.95          # Maximum

    # Patterns connus à rechercher
    KNOWN_PATTERNS = {
        'volume_spike': 'Volume > 2x moyenne au breakout',
        'rsi_overbought': 'RSI > 70 à l\'entrée',
        'rsi_oversold': 'RSI < 30 à l\'entrée',
        'gap_up': 'Gap haussier > 2% à l\'ouverture',
        'gap_down': 'Gap baissier > 2% à l\'ouverture',
        'high_confidence': 'Score de confiance > 80',
        'low_confidence': 'Score de confiance < 50',
        'weak_volume': 'Volume < moyenne au breakout',
        'counter_trend': 'Trade contre la tendance de marché',
        'sector_momentum': 'Secteur en forte tendance',
        'earnings_nearby': 'Earnings dans les 7 jours',
        'friday_trade': 'Trade passé le vendredi',
        'monday_trade': 'Trade passé le lundi',
        'first_hour': 'Trade dans la première heure',
        'last_hour': 'Trade dans la dernière heure',
    }

    def __init__(
        self,
        llm_client = None,
        data_dir: str = "data/auditor"
    ):
        """
        Initialise l'auditeur.

        Args:
            llm_client: Client LLM pour l'analyse (GrokScanner ou autre)
            data_dir: Répertoire pour les données
        """
        self.llm_client = llm_client
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Fichier des guidelines apprises
        self.guidelines_file = self.data_dir / "learned_guidelines.json"
        self.reports_dir = self.data_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        # Guidelines en mémoire
        self._guidelines: Dict[str, LearnedGuideline] = {}

        # Charger les guidelines existantes
        self._load_guidelines()

    def _load_guidelines(self):
        """Charge les guidelines depuis le fichier JSON"""
        if self.guidelines_file.exists():
            try:
                with open(self.guidelines_file, 'r') as f:
                    data = json.load(f)

                for g_data in data.get('guidelines', []):
                    guideline = LearnedGuideline.from_dict(g_data)
                    self._guidelines[guideline.id] = guideline

                logger.info(f"Loaded {len(self._guidelines)} learned guidelines")

            except Exception as e:
                logger.error(f"Error loading guidelines: {e}")

    def _save_guidelines(self):
        """Sauvegarde les guidelines dans le fichier JSON"""
        try:
            # Filtrer les guidelines actives
            active_guidelines = [
                g.to_dict() for g in self._guidelines.values()
                if g.confidence >= self.MIN_CONFIDENCE_ACTIVE
            ]

            data = {
                'updated_at': datetime.now().isoformat(),
                'total_guidelines': len(self._guidelines),
                'active_guidelines': len(active_guidelines),
                'guidelines': active_guidelines
            }

            with open(self.guidelines_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(active_guidelines)} active guidelines")

        except Exception as e:
            logger.error(f"Error saving guidelines: {e}")

    async def run_audit(
        self,
        trades: List[Dict],
        market_data: Optional[Dict] = None
    ) -> AuditReport:
        """
        Exécute l'audit nocturne.

        Args:
            trades: Liste des trades de la journée
            market_data: Données de marché optionnelles

        Returns:
            Rapport d'audit
        """
        logger.info(f"Starting nightly audit for {len(trades)} trades...")

        # 1. Analyser chaque trade
        analyses = []
        for trade in trades:
            analysis = await self._analyze_trade(trade, market_data)
            analyses.append(analysis)

        # 2. Calculer les stats
        wins = [a for a in analyses if a.outcome == TradeOutcome.WIN]
        losses = [a for a in analyses if a.outcome == TradeOutcome.LOSS]

        total_pnl = sum(a.pnl for a in analyses)
        win_rate = len(wins) / len(analyses) if analyses else 0
        avg_win = sum(a.pnl for a in wins) / len(wins) if wins else 0
        avg_loss = sum(a.pnl for a in losses) / len(losses) if losses else 0

        # 3. Identifier les patterns
        new_guidelines = []
        confirmed_guidelines = []
        invalidated_guidelines = []

        # Analyser les trades gagnants
        if wins:
            win_patterns = await self._identify_patterns(wins, is_winning=True)
            for pattern, confidence in win_patterns:
                guideline = self._create_or_update_guideline(
                    pattern=pattern,
                    is_positive=True,
                    trades=[a.symbol for a in wins],
                    confidence=confidence
                )
                if guideline:
                    if guideline.confirmation_count == 1:
                        new_guidelines.append(guideline)
                    else:
                        confirmed_guidelines.append(guideline.id)

        # Analyser les trades perdants
        if losses:
            loss_patterns = await self._identify_patterns(losses, is_winning=False)
            for pattern, confidence in loss_patterns:
                guideline = self._create_or_update_guideline(
                    pattern=pattern,
                    is_positive=False,
                    trades=[a.symbol for a in losses],
                    confidence=confidence
                )
                if guideline:
                    if guideline.confirmation_count == 1:
                        new_guidelines.append(guideline)
                    else:
                        confirmed_guidelines.append(guideline.id)

        # 4. Vérifier les guidelines existantes
        for gid, guideline in list(self._guidelines.items()):
            # Vérifier si la guideline aurait prédit correctement
            was_correct = await self._verify_guideline(guideline, analyses)

            if was_correct is True:
                if gid not in confirmed_guidelines:
                    confirmed_guidelines.append(gid)
            elif was_correct is False:
                guideline.invalidation_count += 1
                guideline.confidence = max(
                    guideline.confidence - self.CONFIDENCE_DECREMENT,
                    0
                )
                if guideline.confidence < self.MIN_CONFIDENCE_ACTIVE:
                    invalidated_guidelines.append(gid)

        # 5. Générer les recommandations
        recommendations = await self._generate_recommendations(
            analyses, new_guidelines, market_data
        )

        # 6. Ajustements de risque
        risk_adjustments = self._calculate_risk_adjustments(analyses)

        # 7. Créer le rapport
        report = AuditReport(
            date=datetime.now(),
            trades_analyzed=len(analyses),
            wins=len(wins),
            losses=len(losses),
            total_pnl=total_pnl,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            new_guidelines=new_guidelines,
            confirmed_guidelines=confirmed_guidelines,
            invalidated_guidelines=invalidated_guidelines,
            recommendations=recommendations,
            risk_adjustments=risk_adjustments
        )

        # 8. Sauvegarder
        self._save_guidelines()
        self._save_report(report)

        logger.info(
            f"Audit complete: {len(wins)}W/{len(losses)}L, "
            f"PnL: ${total_pnl:.2f}, "
            f"{len(new_guidelines)} new rules, "
            f"{len(confirmed_guidelines)} confirmed"
        )

        return report

    async def _analyze_trade(
        self,
        trade: Dict,
        market_data: Optional[Dict]
    ) -> TradeAnalysis:
        """Analyse un trade individuel"""
        # Déterminer le résultat
        pnl = trade.get('pnl', 0)
        if pnl > 0:
            outcome = TradeOutcome.WIN
        elif pnl < 0:
            outcome = TradeOutcome.LOSS
        elif trade.get('exit_price') is None:
            outcome = TradeOutcome.OPEN
        else:
            outcome = TradeOutcome.BREAKEVEN

        # Calculer le PnL %
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', entry_price)
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price else 0

        # Identifier les facteurs
        factors = []
        mistakes = []

        # Vérifier le volume
        if trade.get('volume_ratio', 1) > 2:
            factors.append('volume_spike')
        elif trade.get('volume_ratio', 1) < 0.7:
            mistakes.append('weak_volume')

        # Vérifier le RSI
        rsi = trade.get('rsi_at_entry', 50)
        if rsi > 70 and outcome == TradeOutcome.LOSS:
            mistakes.append('rsi_overbought')
        elif rsi < 30 and outcome == TradeOutcome.WIN:
            factors.append('rsi_oversold_bounce')

        # Vérifier la confiance
        confidence = trade.get('confidence_score', 50)
        if confidence > 80 and outcome == TradeOutcome.WIN:
            factors.append('high_confidence')
        elif confidence < 50 and outcome == TradeOutcome.LOSS:
            mistakes.append('low_confidence')

        return TradeAnalysis(
            symbol=trade.get('symbol', 'UNKNOWN'),
            entry_date=datetime.fromisoformat(trade.get('entry_date', datetime.now().isoformat())),
            exit_date=datetime.fromisoformat(trade['exit_date']) if trade.get('exit_date') else None,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=trade.get('quantity', 0),
            outcome=outcome,
            pnl=pnl,
            pnl_pct=pnl_pct,
            signal_source=trade.get('signal_source', 'unknown'),
            confidence_score=confidence,
            rsi_at_entry=rsi,
            volume_ratio=trade.get('volume_ratio', 1),
            market_regime=trade.get('market_regime', 'unknown'),
            sector=trade.get('sector', 'unknown'),
            contributing_factors=factors,
            mistakes_identified=mistakes
        )

    async def _identify_patterns(
        self,
        analyses: List[TradeAnalysis],
        is_winning: bool
    ) -> List[Tuple[str, float]]:
        """
        Identifie les patterns communs dans une liste de trades.

        Returns:
            Liste de (pattern_name, confidence)
        """
        patterns = []

        if len(analyses) < 2:
            return patterns

        # Compter les facteurs communs
        if is_winning:
            all_factors = []
            for a in analyses:
                all_factors.extend(a.contributing_factors)
        else:
            all_factors = []
            for a in analyses:
                all_factors.extend(a.mistakes_identified)

        # Trouver les patterns fréquents
        from collections import Counter
        factor_counts = Counter(all_factors)

        for factor, count in factor_counts.items():
            frequency = count / len(analyses)
            if frequency >= 0.5:  # Au moins 50% des trades
                confidence = min(frequency * 0.7, 0.7)  # Cap à 70%
                patterns.append((factor, confidence))

        # Analyser les secteurs
        sectors = [a.sector for a in analyses]
        sector_counts = Counter(sectors)
        most_common_sector = sector_counts.most_common(1)
        if most_common_sector:
            sector, count = most_common_sector[0]
            if count / len(analyses) >= 0.6:
                patterns.append((f"sector_{sector}", 0.5))

        # Analyser les niveaux de confiance
        avg_confidence = sum(a.confidence_score for a in analyses) / len(analyses)
        if is_winning and avg_confidence > 75:
            patterns.append(('high_confidence_wins', 0.6))
        elif not is_winning and avg_confidence < 55:
            patterns.append(('low_confidence_loses', 0.6))

        return patterns

    def _create_or_update_guideline(
        self,
        pattern: str,
        is_positive: bool,
        trades: List[str],
        confidence: float
    ) -> Optional[LearnedGuideline]:
        """Crée ou met à jour une guideline"""
        # Générer l'ID
        guideline_id = f"{'seek' if is_positive else 'avoid'}_{pattern}"

        if guideline_id in self._guidelines:
            # Mettre à jour
            guideline = self._guidelines[guideline_id]
            guideline.confirmation_count += 1
            guideline.last_confirmed = datetime.now()
            guideline.confidence = min(
                guideline.confidence + self.CONFIDENCE_INCREMENT,
                self.MAX_CONFIDENCE
            )
            guideline.based_on_trades.extend(trades)
            return guideline
        else:
            # Créer nouvelle
            lesson_type = LessonType.SEEK_PATTERN if is_positive else LessonType.AVOID_PATTERN

            # Générer la description
            if is_positive:
                rule = f"Rechercher les trades avec le pattern '{pattern}'"
                description = f"Ce pattern a été associé à des trades gagnants"
            else:
                rule = f"Éviter les trades avec le pattern '{pattern}'"
                description = f"Ce pattern a été associé à des trades perdants"

            guideline = LearnedGuideline(
                id=guideline_id,
                type=lesson_type,
                rule=rule,
                description=description,
                confidence=max(confidence, self.MIN_CONFIDENCE_NEW_RULE),
                created_at=datetime.now(),
                last_confirmed=datetime.now(),
                confirmation_count=1,
                invalidation_count=0,
                based_on_trades=trades
            )

            self._guidelines[guideline_id] = guideline
            return guideline

    async def _verify_guideline(
        self,
        guideline: LearnedGuideline,
        analyses: List[TradeAnalysis]
    ) -> Optional[bool]:
        """
        Vérifie si une guideline a été correcte pour les trades analysés.

        Returns:
            True si correcte, False si incorrecte, None si non applicable
        """
        applicable_trades = []

        # Trouver les trades où la guideline s'applique
        for analysis in analyses:
            # Vérifier si le pattern de la guideline est présent
            pattern = guideline.id.split('_', 1)[1] if '_' in guideline.id else guideline.id

            if pattern in analysis.contributing_factors or pattern in analysis.mistakes_identified:
                applicable_trades.append(analysis)

        if not applicable_trades:
            return None  # Non applicable

        # Vérifier si la prédiction était correcte
        is_seek_pattern = guideline.type == LessonType.SEEK_PATTERN

        wins = [t for t in applicable_trades if t.outcome == TradeOutcome.WIN]
        losses = [t for t in applicable_trades if t.outcome == TradeOutcome.LOSS]

        if is_seek_pattern:
            # Une règle "seek" est correcte si les trades sont gagnants
            return len(wins) > len(losses)
        else:
            # Une règle "avoid" est correcte si les trades sont perdants
            return len(losses) > len(wins)

    async def _generate_recommendations(
        self,
        analyses: List[TradeAnalysis],
        new_guidelines: List[LearnedGuideline],
        market_data: Optional[Dict]
    ) -> List[str]:
        """Génère des recommandations basées sur l'analyse"""
        recommendations = []

        if not analyses:
            return ["Pas de trades à analyser aujourd'hui"]

        # Analyser le win rate
        wins = [a for a in analyses if a.outcome == TradeOutcome.WIN]
        win_rate = len(wins) / len(analyses)

        if win_rate < 0.4:
            recommendations.append(
                "Win rate faible (<40%). Considérer augmenter le seuil de confiance minimum."
            )
        elif win_rate > 0.7:
            recommendations.append(
                "Excellent win rate (>70%). Les critères actuels fonctionnent bien."
            )

        # Analyser les pertes
        losses = [a for a in analyses if a.outcome == TradeOutcome.LOSS]
        if losses:
            avg_loss = sum(a.pnl for a in losses) / len(losses)
            avg_win = sum(a.pnl for a in wins) / len(wins) if wins else 0

            if abs(avg_loss) > avg_win * 1.5:
                recommendations.append(
                    f"Les pertes moyennes (${abs(avg_loss):.0f}) sont trop élevées vs gains "
                    f"(${avg_win:.0f}). Resserrer les stop-loss."
                )

        # Analyser les nouvelles guidelines
        for guideline in new_guidelines:
            if guideline.type == LessonType.AVOID_PATTERN:
                recommendations.append(
                    f"Nouvelle règle d'évitement: {guideline.rule}"
                )

        # Vérifier les guidelines à haute confiance
        high_confidence_rules = [
            g for g in self._guidelines.values()
            if g.confidence > 0.7
        ]
        if high_confidence_rules:
            recommendations.append(
                f"{len(high_confidence_rules)} règles à haute confiance. "
                "Ces patterns sont fiables."
            )

        return recommendations

    def _calculate_risk_adjustments(
        self,
        analyses: List[TradeAnalysis]
    ) -> Dict[str, float]:
        """Calcule les ajustements de risque recommandés"""
        adjustments = {}

        if not analyses:
            return adjustments

        # Calculer la volatilité des résultats
        pnls = [a.pnl for a in analyses if a.pnl != 0]
        if pnls:
            import statistics
            try:
                volatility = statistics.stdev(pnls)
                avg_pnl = statistics.mean(pnls)

                # Si volatilité élevée, réduire la taille des positions
                if volatility > abs(avg_pnl) * 2:
                    adjustments['position_size_multiplier'] = 0.8
                elif volatility < abs(avg_pnl) * 0.5:
                    adjustments['position_size_multiplier'] = 1.1
                else:
                    adjustments['position_size_multiplier'] = 1.0

            except statistics.StatisticsError:
                pass

        # Ajuster selon le win rate récent
        wins = [a for a in analyses if a.outcome == TradeOutcome.WIN]
        win_rate = len(wins) / len(analyses)

        if win_rate < 0.3:
            adjustments['confidence_threshold'] = 0.65  # Plus strict
        elif win_rate > 0.6:
            adjustments['confidence_threshold'] = 0.55  # Moins strict
        else:
            adjustments['confidence_threshold'] = 0.60

        return adjustments

    def _save_report(self, report: AuditReport):
        """Sauvegarde le rapport d'audit"""
        filename = f"audit_{report.date.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.reports_dir / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info(f"Saved audit report: {filename}")
        except Exception as e:
            logger.error(f"Error saving audit report: {e}")

    def get_learned_guidelines(self, min_confidence: float = 0.3) -> List[LearnedGuideline]:
        """Retourne les guidelines apprises actives"""
        return [
            g for g in self._guidelines.values()
            if g.confidence >= min_confidence
        ]

    def get_avoid_patterns(self) -> List[str]:
        """Retourne les patterns à éviter"""
        return [
            g.rule for g in self._guidelines.values()
            if g.type == LessonType.AVOID_PATTERN and g.confidence >= 0.5
        ]

    def get_seek_patterns(self) -> List[str]:
        """Retourne les patterns à rechercher"""
        return [
            g.rule for g in self._guidelines.values()
            if g.type == LessonType.SEEK_PATTERN and g.confidence >= 0.5
        ]

    def should_avoid_trade(self, trade_context: Dict) -> Tuple[bool, Optional[str]]:
        """
        Vérifie si un trade devrait être évité selon les guidelines.

        Args:
            trade_context: Contexte du trade (symbol, rsi, volume, etc.)

        Returns:
            (should_avoid, reason)
        """
        for guideline in self._guidelines.values():
            if guideline.type != LessonType.AVOID_PATTERN:
                continue
            if guideline.confidence < 0.5:
                continue

            # Vérifier les conditions
            pattern = guideline.id.split('_', 1)[1] if '_' in guideline.id else guideline.id

            # Volume faible
            if pattern == 'weak_volume' and trade_context.get('volume_ratio', 1) < 0.7:
                return True, "Volume trop faible (règle apprise)"

            # RSI suracheté
            if pattern == 'rsi_overbought' and trade_context.get('rsi', 50) > 70:
                return True, "RSI suracheté (règle apprise)"

            # Confiance faible
            if pattern == 'low_confidence' and trade_context.get('confidence', 50) < 50:
                return True, "Confiance trop faible (règle apprise)"

        return False, None

    async def analyze_with_llm(
        self,
        trades: List[TradeAnalysis],
        prompt_type: str = "pattern_analysis"
    ) -> Optional[str]:
        """
        Utilise le LLM pour une analyse plus profonde.

        Nécessite un llm_client (GrokScanner ou autre).
        """
        if not self.llm_client:
            return None

        # Préparer le contexte
        trade_summaries = []
        for t in trades[:10]:  # Limiter pour le contexte
            trade_summaries.append(
                f"- {t.symbol}: {t.outcome.value}, PnL={t.pnl_pct:.1f}%, "
                f"RSI={t.rsi_at_entry:.0f}, Volume={t.volume_ratio:.1f}x, "
                f"Confidence={t.confidence_score:.0f}"
            )

        context = "\n".join(trade_summaries)

        prompts = {
            "pattern_analysis": f"""Analyse ces trades et identifie les patterns:

{context}

Identifie:
1. Les facteurs communs aux trades gagnants
2. Les erreurs communes aux trades perdants
3. Des règles à ajouter pour améliorer

Réponds en JSON avec: patterns_gagnants, erreurs, recommandations""",

            "risk_assessment": f"""Évalue le risque de ces trades:

{context}

Analyse:
1. La cohérence des résultats
2. Les secteurs/styles qui fonctionnent
3. Les ajustements de taille de position recommandés

Réponds en JSON avec: evaluation, ajustements, warnings"""
        }

        prompt = prompts.get(prompt_type, prompts["pattern_analysis"])

        try:
            response = await self.llm_client.chat(prompt, temperature=0.3)
            return response
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return None


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_auditor_instance: Optional[NightlyAuditor] = None


def get_nightly_auditor(llm_client=None) -> NightlyAuditor:
    """Factory pour obtenir une instance de l'auditeur"""
    global _auditor_instance

    if _auditor_instance is None:
        _auditor_instance = NightlyAuditor(llm_client=llm_client)

    return _auditor_instance


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        print("=== Nightly Auditor Test ===\n")

        auditor = NightlyAuditor()

        # Simuler des trades
        test_trades = [
            {
                'symbol': 'NVDA',
                'entry_date': '2024-12-27T10:30:00',
                'exit_date': '2024-12-27T15:00:00',
                'entry_price': 140.0,
                'exit_price': 145.0,
                'quantity': 10,
                'pnl': 50.0,
                'confidence_score': 82,
                'rsi_at_entry': 55,
                'volume_ratio': 2.3,
                'market_regime': 'bull_strong',
                'sector': 'Technology'
            },
            {
                'symbol': 'TSLA',
                'entry_date': '2024-12-27T11:00:00',
                'exit_date': '2024-12-27T14:30:00',
                'entry_price': 250.0,
                'exit_price': 245.0,
                'quantity': 5,
                'pnl': -25.0,
                'confidence_score': 45,
                'rsi_at_entry': 72,
                'volume_ratio': 0.6,
                'market_regime': 'bull_strong',
                'sector': 'Consumer'
            },
            {
                'symbol': 'AAPL',
                'entry_date': '2024-12-27T09:45:00',
                'exit_date': '2024-12-27T16:00:00',
                'entry_price': 195.0,
                'exit_price': 198.0,
                'quantity': 8,
                'pnl': 24.0,
                'confidence_score': 78,
                'rsi_at_entry': 48,
                'volume_ratio': 1.8,
                'market_regime': 'bull_strong',
                'sector': 'Technology'
            }
        ]

        # Exécuter l'audit
        report = await auditor.run_audit(test_trades)

        print(f"Trades analyzed: {report.trades_analyzed}")
        print(f"Wins: {report.wins}, Losses: {report.losses}")
        print(f"Win Rate: {report.win_rate:.0%}")
        print(f"Total PnL: ${report.total_pnl:.2f}")
        print()

        print("New Guidelines:")
        for g in report.new_guidelines:
            print(f"  - {g.rule} (confidence: {g.confidence:.0%})")

        print()
        print("Recommendations:")
        for r in report.recommendations:
            print(f"  - {r}")

        print()
        print("Risk Adjustments:")
        for k, v in report.risk_adjustments.items():
            print(f"  - {k}: {v}")

        # Vérifier les guidelines sauvegardées
        print()
        print(f"Total guidelines learned: {len(auditor.get_learned_guidelines())}")

    asyncio.run(main())
