"""
Signal Correlator - Apprentissage des correlations signal -> prix
Apprend quelles sources/patterns predisent les mouvements de prix.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SignalRecord:
    """Enregistrement d'un signal detecte"""
    id: str
    source: str  # "social", "grok", "rsi_breakout", "volume_spike", etc.
    symbol: str
    timestamp: datetime
    signal_type: str  # "bullish", "bearish", "neutral"
    strength: float  # 0 to 1
    heat_score: float  # Heat au moment du signal

    # Contexte
    metadata: Dict = field(default_factory=dict)

    # Outcome (rempli plus tard)
    price_at_signal: Optional[float] = None
    price_after_5min: Optional[float] = None
    price_after_30min: Optional[float] = None
    price_after_1h: Optional[float] = None
    price_after_1d: Optional[float] = None

    outcome_5min: Optional[float] = None  # % change
    outcome_30min: Optional[float] = None
    outcome_1h: Optional[float] = None
    outcome_1d: Optional[float] = None

    was_correct: Optional[bool] = None


@dataclass
class SourceWeight:
    """Poids d'une source pour les predictions"""
    source: str
    weight: float  # 0 to 1

    # Stats
    signals_count: int = 0
    correct_count: int = 0
    accuracy: float = 0.0
    avg_return: float = 0.0

    # Historique recent
    recent_accuracy: float = 0.0  # Sur les 20 derniers
    trend: str = "stable"  # "improving", "declining", "stable"

    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class CorrelationInsight:
    """Insight sur une correlation detectee"""
    pattern: str  # "social_spike -> +2% in 30min"
    source: str
    correlation_strength: float
    sample_size: int
    confidence: float
    description: str


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CorrelatorConfig:
    """Configuration du Signal Correlator"""
    # Poids initiaux
    default_weight: float = 0.5

    # Learning
    learning_rate: float = 0.1
    min_samples_for_weight: int = 10
    weight_decay: float = 0.99  # Decay progressif vers 0.5

    # Timeframes pour evaluation
    evaluation_windows: List[int] = field(
        default_factory=lambda: [5, 30, 60, 1440]  # minutes
    )
    primary_window: int = 30  # Window principal pour le scoring

    # Seuils
    correct_threshold: float = 0.005  # 0.5% = correct pour bullish
    significant_move: float = 0.02  # 2% = move significatif

    # Retention
    max_records: int = 10000
    retention_days: int = 90


# =============================================================================
# SIGNAL CORRELATOR
# =============================================================================

class SignalCorrelator:
    """
    Apprend les correlations entre signaux et mouvements de prix.

    Enregistre les signaux, suit les outcomes, et ajuste les poids
    des sources en fonction de leur capacite predictive.
    """

    def __init__(self, config: Optional[CorrelatorConfig] = None):
        self.config = config or CorrelatorConfig()

        # Poids des sources
        self._weights: Dict[str, SourceWeight] = {}

        # Signaux en attente d'evaluation
        self._pending_signals: List[SignalRecord] = []

        # Historique des signaux evalues
        self._evaluated_signals: List[SignalRecord] = []

        # Correlations detectees
        self._correlations: Dict[str, CorrelationInsight] = {}

        # Lock pour thread safety
        self._lock = asyncio.Lock()

        # Persistence
        self._data_dir = Path("data/correlator")
        self._data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("SignalCorrelator initialized")

    async def initialize(self):
        """Initialise le correlator et charge les donnees"""
        await self._load_weights()
        await self._load_signals()
        logger.info(f"SignalCorrelator ready - {len(self._weights)} sources tracked")

    # -------------------------------------------------------------------------
    # SIGNAL RECORDING
    # -------------------------------------------------------------------------

    async def record_signal(
        self,
        source: str,
        symbol: str,
        signal_type: str,
        strength: float,
        heat_score: float,
        price: float,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Enregistre un nouveau signal.

        Returns:
            ID du signal pour reference
        """
        async with self._lock:
            signal_id = f"{source}_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            signal = SignalRecord(
                id=signal_id,
                source=source,
                symbol=symbol.upper(),
                timestamp=datetime.now(),
                signal_type=signal_type,
                strength=strength,
                heat_score=heat_score,
                price_at_signal=price,
                metadata=metadata or {}
            )

            self._pending_signals.append(signal)

            # Initialiser le poids de la source si nouveau
            if source not in self._weights:
                self._weights[source] = SourceWeight(
                    source=source,
                    weight=self.config.default_weight
                )

            logger.debug(f"Recorded signal: {signal_id}")
            return signal_id

    async def update_price(self, symbol: str, current_price: float):
        """
        Met a jour les prix pour evaluer les signaux en attente.
        Appeler regulierement avec les prix actuels.
        """
        now = datetime.now()

        async with self._lock:
            still_pending = []

            for signal in self._pending_signals:
                if signal.symbol != symbol.upper():
                    still_pending.append(signal)
                    continue

                age_minutes = (now - signal.timestamp).total_seconds() / 60

                # Mettre a jour les prix selon l'age
                if age_minutes >= 5 and signal.price_after_5min is None:
                    signal.price_after_5min = current_price
                    signal.outcome_5min = self._calculate_return(
                        signal.price_at_signal, current_price
                    )

                if age_minutes >= 30 and signal.price_after_30min is None:
                    signal.price_after_30min = current_price
                    signal.outcome_30min = self._calculate_return(
                        signal.price_at_signal, current_price
                    )

                if age_minutes >= 60 and signal.price_after_1h is None:
                    signal.price_after_1h = current_price
                    signal.outcome_1h = self._calculate_return(
                        signal.price_at_signal, current_price
                    )

                if age_minutes >= 1440 and signal.price_after_1d is None:
                    signal.price_after_1d = current_price
                    signal.outcome_1d = self._calculate_return(
                        signal.price_at_signal, current_price
                    )

                # Evaluer si on a assez de donnees
                if signal.outcome_30min is not None:
                    await self._evaluate_signal(signal)
                    self._evaluated_signals.append(signal)
                else:
                    still_pending.append(signal)

            self._pending_signals = still_pending

    def _calculate_return(self, price_before: float, price_after: float) -> float:
        """Calcule le retour en pourcentage"""
        if price_before <= 0:
            return 0
        return (price_after - price_before) / price_before

    async def _evaluate_signal(self, signal: SignalRecord):
        """Evalue si un signal etait correct"""
        # Utiliser le window principal (30min par defaut)
        outcome = signal.outcome_30min

        if outcome is None:
            return

        # Determiner si correct
        if signal.signal_type == "bullish":
            signal.was_correct = outcome >= self.config.correct_threshold
        elif signal.signal_type == "bearish":
            signal.was_correct = outcome <= -self.config.correct_threshold
        else:
            signal.was_correct = abs(outcome) < self.config.correct_threshold

        # Mettre a jour les poids de la source
        await self._update_weight(signal)

    async def _update_weight(self, signal: SignalRecord):
        """Met a jour le poids d'une source basee sur le signal"""
        source = signal.source
        weight = self._weights.get(source)

        if weight is None:
            return

        weight.signals_count += 1

        if signal.was_correct:
            weight.correct_count += 1
            # Augmenter le poids
            adjustment = self.config.learning_rate * signal.strength
        else:
            # Diminuer le poids
            adjustment = -self.config.learning_rate * signal.strength

        # Appliquer l'ajustement
        weight.weight = max(0.1, min(0.9, weight.weight + adjustment))

        # Decay vers la moyenne
        weight.weight = (
            weight.weight * self.config.weight_decay +
            self.config.default_weight * (1 - self.config.weight_decay)
        )

        # Recalculer les stats
        weight.accuracy = weight.correct_count / weight.signals_count

        # Calculer la tendance recente
        recent_signals = [
            s for s in self._evaluated_signals[-20:]
            if s.source == source
        ]
        if len(recent_signals) >= 5:
            recent_correct = sum(1 for s in recent_signals if s.was_correct)
            weight.recent_accuracy = recent_correct / len(recent_signals)

            # Determiner la tendance
            if weight.recent_accuracy > weight.accuracy + 0.1:
                weight.trend = "improving"
            elif weight.recent_accuracy < weight.accuracy - 0.1:
                weight.trend = "declining"
            else:
                weight.trend = "stable"

        weight.last_updated = datetime.now()

        logger.debug(
            f"Updated weight for {source}: {weight.weight:.2f} "
            f"(accuracy: {weight.accuracy:.1%})"
        )

    # -------------------------------------------------------------------------
    # QUERIES
    # -------------------------------------------------------------------------

    def get_source_weight(self, source: str) -> float:
        """Retourne le poids d'une source"""
        weight = self._weights.get(source)
        return weight.weight if weight else self.config.default_weight

    def get_source_stats(self, source: str) -> Optional[SourceWeight]:
        """Retourne les stats completes d'une source"""
        return self._weights.get(source)

    def get_all_weights(self) -> Dict[str, float]:
        """Retourne tous les poids"""
        return {s: w.weight for s, w in self._weights.items()}

    def get_best_sources(self, limit: int = 5) -> List[SourceWeight]:
        """Retourne les sources les plus fiables"""
        sources = [
            w for w in self._weights.values()
            if w.signals_count >= self.config.min_samples_for_weight
        ]
        sources.sort(key=lambda w: w.accuracy, reverse=True)
        return sources[:limit]

    def get_worst_sources(self, limit: int = 5) -> List[SourceWeight]:
        """Retourne les sources les moins fiables"""
        sources = [
            w for w in self._weights.values()
            if w.signals_count >= self.config.min_samples_for_weight
        ]
        sources.sort(key=lambda w: w.accuracy)
        return sources[:limit]

    def get_correlations(self) -> List[CorrelationInsight]:
        """Retourne les correlations detectees"""
        return list(self._correlations.values())

    # -------------------------------------------------------------------------
    # CORRELATION ANALYSIS
    # -------------------------------------------------------------------------

    async def analyze_correlations(self):
        """
        Analyse les signaux evalues pour detecter des correlations.
        Appeler periodiquement (ex: chaque heure).
        """
        if len(self._evaluated_signals) < 50:
            return

        async with self._lock:
            self._correlations.clear()

            # Analyser par source
            for source in self._weights.keys():
                signals = [s for s in self._evaluated_signals if s.source == source]

                if len(signals) < 10:
                    continue

                # Correlation heat -> outcome
                heat_outcomes = [(s.heat_score, s.outcome_30min or 0) for s in signals]
                if heat_outcomes:
                    correlation = self._calculate_correlation(heat_outcomes)
                    if abs(correlation) > 0.3:
                        self._correlations[f"{source}_heat"] = CorrelationInsight(
                            pattern=f"High heat + {source} signal -> move",
                            source=source,
                            correlation_strength=correlation,
                            sample_size=len(signals),
                            confidence=min(1.0, len(signals) / 100),
                            description=f"Heat score correle a {correlation:.1%} avec outcome"
                        )

                # Correlation strength -> outcome
                strength_outcomes = [(s.strength, s.outcome_30min or 0) for s in signals]
                if strength_outcomes:
                    correlation = self._calculate_correlation(strength_outcomes)
                    if abs(correlation) > 0.3:
                        self._correlations[f"{source}_strength"] = CorrelationInsight(
                            pattern=f"Strong {source} signal -> bigger move",
                            source=source,
                            correlation_strength=correlation,
                            sample_size=len(signals),
                            confidence=min(1.0, len(signals) / 100),
                            description=f"Signal strength correle a {correlation:.1%} avec outcome"
                        )

            logger.info(f"Found {len(self._correlations)} correlations")

    def _calculate_correlation(self, pairs: List[Tuple[float, float]]) -> float:
        """Calcule la correlation de Pearson"""
        if len(pairs) < 3:
            return 0

        x_vals = [p[0] for p in pairs]
        y_vals = [p[1] for p in pairs]

        try:
            x_mean = statistics.mean(x_vals)
            y_mean = statistics.mean(y_vals)

            numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
            x_std = statistics.stdev(x_vals)
            y_std = statistics.stdev(y_vals)

            if x_std == 0 or y_std == 0:
                return 0

            return numerator / (len(pairs) * x_std * y_std)
        except Exception:
            return 0

    # -------------------------------------------------------------------------
    # WEIGHTED SCORING
    # -------------------------------------------------------------------------

    def calculate_weighted_score(self, signals: Dict[str, float]) -> float:
        """
        Calcule un score pondere base sur plusieurs signaux.

        Args:
            signals: Dict {source: signal_value} ou signal_value est entre -1 et 1

        Returns:
            Score pondere entre -1 et 1
        """
        if not signals:
            return 0

        weighted_sum = 0
        total_weight = 0

        for source, value in signals.items():
            weight = self.get_source_weight(source)
            weighted_sum += value * weight
            total_weight += weight

        if total_weight == 0:
            return 0

        return weighted_sum / total_weight

    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------

    async def _load_weights(self):
        """Charge les poids depuis le disque"""
        path = self._data_dir / "weights.json"
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    for source, weight_data in data.items():
                        self._weights[source] = SourceWeight(
                            source=source,
                            weight=weight_data.get('weight', 0.5),
                            signals_count=weight_data.get('signals_count', 0),
                            correct_count=weight_data.get('correct_count', 0),
                            accuracy=weight_data.get('accuracy', 0),
                            avg_return=weight_data.get('avg_return', 0),
                            recent_accuracy=weight_data.get('recent_accuracy', 0),
                            trend=weight_data.get('trend', 'stable')
                        )
                logger.info(f"Loaded {len(self._weights)} source weights")
            except Exception as e:
                logger.warning(f"Could not load weights: {e}")

    async def _save_weights(self):
        """Sauvegarde les poids sur le disque"""
        path = self._data_dir / "weights.json"
        try:
            data = {}
            for source, weight in self._weights.items():
                data[source] = {
                    'weight': weight.weight,
                    'signals_count': weight.signals_count,
                    'correct_count': weight.correct_count,
                    'accuracy': weight.accuracy,
                    'avg_return': weight.avg_return,
                    'recent_accuracy': weight.recent_accuracy,
                    'trend': weight.trend,
                    'last_updated': weight.last_updated.isoformat()
                }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save weights: {e}")

    async def _load_signals(self):
        """Charge les signaux evalues depuis le disque"""
        path = self._data_dir / "evaluated_signals.json"
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    for signal_data in data[-self.config.max_records:]:
                        signal = SignalRecord(
                            id=signal_data['id'],
                            source=signal_data['source'],
                            symbol=signal_data['symbol'],
                            timestamp=datetime.fromisoformat(signal_data['timestamp']),
                            signal_type=signal_data['signal_type'],
                            strength=signal_data['strength'],
                            heat_score=signal_data['heat_score'],
                            price_at_signal=signal_data.get('price_at_signal'),
                            outcome_30min=signal_data.get('outcome_30min'),
                            was_correct=signal_data.get('was_correct')
                        )
                        self._evaluated_signals.append(signal)
                logger.info(f"Loaded {len(self._evaluated_signals)} evaluated signals")
            except Exception as e:
                logger.warning(f"Could not load signals: {e}")

    async def _save_signals(self):
        """Sauvegarde les signaux evalues"""
        path = self._data_dir / "evaluated_signals.json"
        try:
            # Garder seulement les N derniers
            signals_to_save = self._evaluated_signals[-self.config.max_records:]
            data = []
            for signal in signals_to_save:
                data.append({
                    'id': signal.id,
                    'source': signal.source,
                    'symbol': signal.symbol,
                    'timestamp': signal.timestamp.isoformat(),
                    'signal_type': signal.signal_type,
                    'strength': signal.strength,
                    'heat_score': signal.heat_score,
                    'price_at_signal': signal.price_at_signal,
                    'outcome_30min': signal.outcome_30min,
                    'was_correct': signal.was_correct
                })
            with open(path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Could not save signals: {e}")

    async def save(self):
        """Sauvegarde toutes les donnees"""
        await self._save_weights()
        await self._save_signals()

    async def close(self):
        """Ferme proprement le correlator"""
        await self.save()
        logger.info("SignalCorrelator closed")


# =============================================================================
# FACTORY
# =============================================================================

_signal_correlator: Optional[SignalCorrelator] = None


async def get_signal_correlator(
    config: Optional[CorrelatorConfig] = None
) -> SignalCorrelator:
    """Retourne l'instance singleton du SignalCorrelator"""
    global _signal_correlator
    if _signal_correlator is None:
        _signal_correlator = SignalCorrelator(config)
        await _signal_correlator.initialize()
    return _signal_correlator
