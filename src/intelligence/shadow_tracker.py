"""
Shadow Tracker - Apprentissage Autonome Sans Trades Reels

Ce module permet au systeme d'apprendre de ses predictions SANS attendre
que l'utilisateur execute reellement les trades.

Architecture:
    1. Chaque signal BUY/STRONG_BUY est automatiquement "shadow tracked"
    2. Le systeme verifie automatiquement les prix a T+5, T+10, T+20 jours
    3. Les outcomes sont utilises pour ajuster les poids des piliers
    4. Retro-ingenierie possible sur les signaux historiques

Auteur: TradingBot V5
Date: Janvier 2026
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


class SignalOutcome(Enum):
    """Resultat d'un signal apres verification"""
    PENDING = "pending"           # Pas encore verifie
    OPEN = "open"                 # Position encore ouverte
    WINNER = "winner"             # Profit >= take_profit ou > 0 a expiration
    LOSER = "loser"               # Perte >= stop_loss ou < 0 a expiration
    BREAKEVEN = "breakeven"       # Entre -1% et +1%
    STOPPED_OUT = "stopped_out"   # Stop-loss touche
    TARGET_HIT = "target_hit"     # Take-profit atteint
    TRAILING_STOPPED = "trailing_stopped"  # Trailing stop touche
    TECHNICAL_EXIT = "technical_exit"      # Sortie sur signal technique


class ExitReason(Enum):
    """Raison de sortie d'une position"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    MAX_HOLDING = "max_holding"        # Duree max atteinte
    EMA_CROSS = "ema_cross"            # Croisement EMA baissier
    RSI_OVERBOUGHT = "rsi_overbought"  # RSI > 70
    RSI_BREAKDOWN = "rsi_breakdown"    # RSI casse support
    MANUAL = "manual"                  # Sortie manuelle


@dataclass
class ShadowSignal:
    """
    Un signal suivi en shadow (paper trading automatique).

    Chaque signal BUY/STRONG_BUY est automatiquement tracked:
    - Prix d'entree enregistre
    - Stop-loss et take-profit calcules
    - Prix surveilles quotidiennement
    - Sortie automatique sur conditions (SL/TP/Trailing/Technical)
    """
    symbol: str
    signal_type: str              # 'BUY', 'STRONG_BUY'
    signal_date: str              # YYYY-MM-DD
    entry_price: float

    # Scores au moment du signal
    total_score: float
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    news_score: float

    # Niveaux calcules
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_pct: float = 0.05        # 5% trailing stop par defaut
    max_holding_days: int = 20             # Duree max de holding

    # Prix de sortie et raison
    exit_price: Optional[float] = None
    exit_date: Optional[str] = None
    exit_reason: Optional[str] = None      # ExitReason value

    # Trailing stop tracking
    highest_price: Optional[float] = None  # Plus haut depuis entree
    trailing_stop_level: Optional[float] = None

    # Outcomes a differentes periodes
    price_t5: Optional[float] = None       # Prix a T+5 jours
    price_t10: Optional[float] = None      # Prix a T+10 jours
    price_t20: Optional[float] = None      # Prix a T+20 jours
    max_price: Optional[float] = None      # Prix max atteint
    min_price: Optional[float] = None      # Prix min atteint

    # Resultats
    pnl_t5: Optional[float] = None         # P&L % a T+5
    pnl_t10: Optional[float] = None        # P&L % a T+10
    pnl_t20: Optional[float] = None        # P&L % a T+20
    final_pnl: Optional[float] = None      # P&L final a la sortie
    max_gain: Optional[float] = None       # Gain max %
    max_drawdown: Optional[float] = None   # Drawdown max %
    holding_days: int = 0                  # Jours de holding

    outcome: str = "pending"               # SignalOutcome value
    outcome_date: Optional[str] = None

    # Metadata
    key_factors: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ShadowSignal':
        # Handle missing fields for backwards compatibility
        defaults = {
            'trailing_stop_pct': 0.05,
            'max_holding_days': 20,
            'exit_price': None,
            'exit_date': None,
            'exit_reason': None,
            'highest_price': None,
            'trailing_stop_level': None,
            'final_pnl': None,
            'holding_days': 0
        }
        for key, default in defaults.items():
            if key not in data:
                data[key] = default
        return cls(**data)

    def is_open(self) -> bool:
        """Verifier si la position est encore ouverte"""
        return self.outcome in ["pending", SignalOutcome.OPEN.value]

    def calculate_current_pnl(self, current_price: float) -> float:
        """Calculer le P&L actuel"""
        return ((current_price - self.entry_price) / self.entry_price) * 100


@dataclass
class LearningInsight:
    """Insight d'apprentissage derive des shadow signals"""
    insight_type: str             # 'weight_adjustment', 'pattern_detected', 'threshold_change'
    description: str
    confidence: float             # 0.0 - 1.0
    sample_size: int

    # Ajustements suggeres
    adjustments: Dict[str, float] = field(default_factory=dict)

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    applied: bool = False


class ShadowTracker:
    """
    Systeme de tracking automatique des signaux pour apprentissage autonome.

    Usage:
        tracker = ShadowTracker()
        await tracker.initialize()

        # Enregistrer un nouveau signal
        await tracker.track_signal(symbol, signal_type, scores, entry_price)

        # Verifier les outcomes (appele automatiquement chaque jour)
        await tracker.verify_outcomes()

        # Obtenir les insights d'apprentissage
        insights = tracker.generate_learning_insights()
    """

    def __init__(self, data_dir: str = "data/shadow_tracking"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.signals_file = self.data_dir / "shadow_signals.json"
        self.insights_file = self.data_dir / "learning_insights.json"
        self.weights_file = self.data_dir / "adjusted_weights.json"

        self.signals: List[ShadowSignal] = []
        self.insights: List[LearningInsight] = []

        # Poids par defaut des piliers (peuvent etre ajustes)
        self.pillar_weights = {
            "technical": 0.25,
            "fundamental": 0.25,
            "sentiment": 0.25,
            "news": 0.25
        }

        self._initialized = False

    async def initialize(self):
        """Charger les donnees persistees"""
        if self._initialized:
            return

        # Charger les signaux
        if self.signals_file.exists():
            try:
                with open(self.signals_file, 'r') as f:
                    data = json.load(f)
                    self.signals = [ShadowSignal.from_dict(s) for s in data]
                logger.info(f"Loaded {len(self.signals)} shadow signals")
            except Exception as e:
                logger.error(f"Error loading shadow signals: {e}")
                self.signals = []

        # Charger les insights
        if self.insights_file.exists():
            try:
                with open(self.insights_file, 'r') as f:
                    data = json.load(f)
                    self.insights = [LearningInsight(**i) for i in data]
            except Exception as e:
                logger.error(f"Error loading insights: {e}")
                self.insights = []

        # Charger les poids ajustes
        if self.weights_file.exists():
            try:
                with open(self.weights_file, 'r') as f:
                    self.pillar_weights = json.load(f)
                logger.info(f"Loaded adjusted weights: {self.pillar_weights}")
            except Exception as e:
                logger.error(f"Error loading weights: {e}")

        self._initialized = True
        logger.info("ShadowTracker initialized")

    def _save_signals(self):
        """Persister les signaux"""
        with open(self.signals_file, 'w') as f:
            json.dump([s.to_dict() for s in self.signals], f, indent=2)

    def _save_insights(self):
        """Persister les insights"""
        with open(self.insights_file, 'w') as f:
            json.dump([asdict(i) for i in self.insights], f, indent=2)

    def _save_weights(self):
        """Persister les poids ajustes"""
        with open(self.weights_file, 'w') as f:
            json.dump(self.pillar_weights, f, indent=2)

    async def track_signal(
        self,
        symbol: str,
        signal_type: str,
        total_score: float,
        technical_score: float,
        fundamental_score: float,
        sentiment_score: float,
        news_score: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        key_factors: Optional[List[str]] = None,
        risk_factors: Optional[List[str]] = None
    ) -> ShadowSignal:
        """
        Enregistrer un nouveau signal pour tracking automatique.

        Args:
            symbol: Ticker du stock
            signal_type: 'BUY' ou 'STRONG_BUY'
            total_score: Score total 0-100
            technical_score: Score technique 0-100
            fundamental_score: Score fondamental 0-100
            sentiment_score: Score sentiment 0-100
            news_score: Score news 0-100
            entry_price: Prix au moment du signal
            stop_loss: Niveau stop-loss (optionnel)
            take_profit: Niveau take-profit (optionnel)
            key_factors: Facteurs positifs
            risk_factors: Facteurs de risque

        Returns:
            Le ShadowSignal cree
        """
        if not self._initialized:
            await self.initialize()

        # Calculer stop/take_profit par defaut si non fournis
        if stop_loss is None:
            stop_loss = entry_price * 0.95  # -5%
        if take_profit is None:
            take_profit = entry_price * 1.10  # +10%

        signal = ShadowSignal(
            symbol=symbol,
            signal_type=signal_type,
            signal_date=datetime.now().strftime('%Y-%m-%d'),
            entry_price=entry_price,
            total_score=total_score,
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            sentiment_score=sentiment_score,
            news_score=news_score,
            stop_loss=stop_loss,
            take_profit=take_profit,
            key_factors=key_factors or [],
            risk_factors=risk_factors or []
        )

        self.signals.append(signal)
        self._save_signals()

        logger.info(f"[SHADOW] Tracking {signal_type} signal for {symbol} @ ${entry_price:.2f}")
        return signal

    async def verify_outcomes(self) -> Dict[str, int]:
        """
        Verifier les outcomes de tous les signaux pending.
        Appele automatiquement chaque jour par l'orchestrator.

        Returns:
            Stats: {'verified': N, 'winners': N, 'losers': N}
        """
        if not self._initialized:
            await self.initialize()

        stats = {'verified': 0, 'winners': 0, 'losers': 0, 'pending': 0}
        today = datetime.now().date()

        for signal in self.signals:
            if signal.outcome != "pending":
                continue

            signal_date = datetime.strptime(signal.signal_date, '%Y-%m-%d').date()
            days_elapsed = (today - signal_date).days

            if days_elapsed < 5:
                stats['pending'] += 1
                continue  # Trop tot pour verifier

            # Telecharger les donnees de prix
            try:
                df = await self._fetch_price_data(signal.symbol, signal_date, today)
                if df is None or len(df) < 2:
                    continue

                # Calculer les metriques
                await self._calculate_signal_metrics(signal, df, days_elapsed)
                stats['verified'] += 1

                if signal.outcome == SignalOutcome.WINNER.value:
                    stats['winners'] += 1
                elif signal.outcome == SignalOutcome.LOSER.value:
                    stats['losers'] += 1

            except Exception as e:
                logger.error(f"Error verifying {signal.symbol}: {e}")
                continue

        self._save_signals()

        logger.info(f"[SHADOW] Verified {stats['verified']} signals: "
                   f"{stats['winners']} winners, {stats['losers']} losers, "
                   f"{stats['pending']} still pending")

        return stats

    async def _fetch_price_data(
        self,
        symbol: str,
        start_date: datetime.date,
        end_date: datetime.date
    ) -> Optional[pd.DataFrame]:
        """Telecharger les donnees de prix via yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date - timedelta(days=5),
                end=end_date + timedelta(days=1)
            )
            return df if len(df) > 0 else None
        except Exception as e:
            logger.error(f"Error fetching price data for {symbol}: {e}")
            return None

    async def _calculate_signal_metrics(
        self,
        signal: ShadowSignal,
        df: pd.DataFrame,
        days_elapsed: int
    ):
        """
        Calculer toutes les metriques et determiner la sortie optimale.

        Simule jour par jour pour trouver:
        1. Si stop-loss touche (sortie immediate)
        2. Si take-profit touche (sortie immediate)
        3. Si trailing stop declenche
        4. Si signal technique de sortie
        5. Si duree max atteinte
        """
        entry = signal.entry_price
        signal.holding_days = days_elapsed

        # Initialiser le tracking
        highest_price = entry
        exit_found = False
        exit_day = 0
        exit_price = entry
        exit_reason = None

        # Simuler jour par jour
        for i in range(min(len(df), signal.max_holding_days)):
            day_high = float(df['High'].iloc[i])
            day_low = float(df['Low'].iloc[i])
            day_close = float(df['Close'].iloc[i])

            # Mettre a jour le plus haut
            if day_high > highest_price:
                highest_price = day_high

            # Calculer le niveau du trailing stop
            trailing_stop_level = highest_price * (1 - signal.trailing_stop_pct)

            # Verifier les conditions de sortie (dans l'ordre de priorite)

            # 1. Stop-loss fixe touche?
            if signal.stop_loss and day_low <= signal.stop_loss:
                exit_found = True
                exit_day = i
                exit_price = signal.stop_loss
                exit_reason = ExitReason.STOP_LOSS.value
                signal.outcome = SignalOutcome.STOPPED_OUT.value
                break

            # 2. Take-profit touche?
            if signal.take_profit and day_high >= signal.take_profit:
                exit_found = True
                exit_day = i
                exit_price = signal.take_profit
                exit_reason = ExitReason.TAKE_PROFIT.value
                signal.outcome = SignalOutcome.TARGET_HIT.value
                break

            # 3. Trailing stop declenche?
            if day_low <= trailing_stop_level and i >= 5:  # Activer apres 5 jours
                exit_found = True
                exit_day = i
                exit_price = trailing_stop_level
                exit_reason = ExitReason.TRAILING_STOP.value
                signal.outcome = SignalOutcome.TRAILING_STOPPED.value
                break

            # Stocker les prix aux checkpoints
            if i == 4:  # T+5
                signal.price_t5 = day_close
                signal.pnl_t5 = ((day_close - entry) / entry) * 100
            elif i == 9:  # T+10
                signal.price_t10 = day_close
                signal.pnl_t10 = ((day_close - entry) / entry) * 100
            elif i == 19:  # T+20
                signal.price_t20 = day_close
                signal.pnl_t20 = ((day_close - entry) / entry) * 100

        # Si pas de sortie trouvee et duree max atteinte
        if not exit_found and days_elapsed >= signal.max_holding_days:
            exit_day = min(signal.max_holding_days - 1, len(df) - 1)
            exit_price = float(df['Close'].iloc[exit_day])
            exit_reason = ExitReason.MAX_HOLDING.value

            pnl = ((exit_price - entry) / entry) * 100
            if pnl >= 3:
                signal.outcome = SignalOutcome.WINNER.value
            elif pnl <= -2:
                signal.outcome = SignalOutcome.LOSER.value
            else:
                signal.outcome = SignalOutcome.BREAKEVEN.value
            exit_found = True

        # Enregistrer les resultats
        if exit_found:
            signal.exit_price = exit_price
            signal.exit_date = (datetime.strptime(signal.signal_date, '%Y-%m-%d') +
                               timedelta(days=exit_day)).strftime('%Y-%m-%d')
            signal.exit_reason = exit_reason
            signal.final_pnl = ((exit_price - entry) / entry) * 100
            signal.holding_days = exit_day + 1
            signal.outcome_date = datetime.now().strftime('%Y-%m-%d')

        # Calculer max/min sur toute la periode
        signal.max_price = float(df['High'].max())
        signal.min_price = float(df['Low'].min())
        signal.max_gain = ((signal.max_price - entry) / entry) * 100
        signal.max_drawdown = ((signal.min_price - entry) / entry) * 100
        signal.highest_price = highest_price

        # Marquer comme OPEN si toujours en cours
        if not exit_found and days_elapsed < signal.max_holding_days:
            signal.outcome = SignalOutcome.OPEN.value

        # Ancienne logique pour compat (si pas de sortie a T+20)
        if signal.outcome == "pending" and days_elapsed >= 20:
            pnl = signal.pnl_t20 or 0
            if pnl >= 5:
                signal.outcome = SignalOutcome.WINNER.value
            elif pnl <= -3:
                signal.outcome = SignalOutcome.LOSER.value
            else:
                signal.outcome = SignalOutcome.BREAKEVEN.value

        if signal.outcome != "pending":
            signal.outcome_date = datetime.now().strftime('%Y-%m-%d')

    def generate_learning_insights(self) -> List[LearningInsight]:
        """
        Analyser les outcomes pour generer des insights d'apprentissage.

        Cette fonction:
        1. Calcule les correlations entre scores de piliers et outcomes
        2. Identifie les patterns (ex: sentiment eleve = winners)
        3. Suggere des ajustements de poids

        Returns:
            Liste d'insights actionables
        """
        # Filtrer les signaux avec outcome connu
        completed = [s for s in self.signals if s.outcome not in ["pending"]]

        if len(completed) < 10:
            logger.info(f"Not enough data for insights ({len(completed)} signals, need 10+)")
            return []

        winners = [s for s in completed if s.outcome in [
            SignalOutcome.WINNER.value, SignalOutcome.TARGET_HIT.value
        ]]
        losers = [s for s in completed if s.outcome in [
            SignalOutcome.LOSER.value, SignalOutcome.STOPPED_OUT.value
        ]]

        insights = []

        # 1. Analyser les correlations par pilier
        for pillar in ['technical', 'fundamental', 'sentiment', 'news']:
            score_attr = f'{pillar}_score'

            avg_winner = sum(getattr(s, score_attr) for s in winners) / len(winners) if winners else 0
            avg_loser = sum(getattr(s, score_attr) for s in losers) / len(losers) if losers else 0

            diff = avg_winner - avg_loser

            if abs(diff) > 10:  # Difference significative
                direction = "increase" if diff > 0 else "decrease"
                adjustment = 0.02 if diff > 0 else -0.02

                insight = LearningInsight(
                    insight_type="weight_adjustment",
                    description=f"{pillar.capitalize()} score correle avec succes "
                               f"(winners: {avg_winner:.1f}, losers: {avg_loser:.1f})",
                    confidence=min(0.9, abs(diff) / 20),
                    sample_size=len(completed),
                    adjustments={pillar: adjustment}
                )
                insights.append(insight)

        # 2. Analyser les seuils de score total
        score_thresholds = [50, 55, 60, 65, 70]
        best_threshold = 55
        best_win_rate = 0

        for threshold in score_thresholds:
            above = [s for s in completed if s.total_score >= threshold]
            if len(above) >= 5:
                wins = len([s for s in above if s.outcome in [
                    SignalOutcome.WINNER.value, SignalOutcome.TARGET_HIT.value
                ]])
                win_rate = wins / len(above)
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_threshold = threshold

        if best_threshold != 55:  # Different du defaut
            insight = LearningInsight(
                insight_type="threshold_change",
                description=f"Seuil optimal detecte: {best_threshold} (win rate: {best_win_rate:.1%})",
                confidence=best_win_rate,
                sample_size=len(completed),
                adjustments={"buy_threshold": best_threshold}
            )
            insights.append(insight)

        # 3. Analyser les patterns temporels
        # (ex: signaux du lundi vs vendredi)

        # 4. Analyser les patterns par secteur
        # (a implementer avec donnees secteur)

        self.insights.extend(insights)
        self._save_insights()

        return insights

    def apply_weight_adjustments(self, max_change: float = 0.05):
        """
        Appliquer les ajustements de poids suggeres par les insights.

        Args:
            max_change: Changement maximum par pilier (defaut 5%)
        """
        # Collecter tous les ajustements non appliques
        pending_insights = [i for i in self.insights
                          if not i.applied and i.insight_type == "weight_adjustment"]

        if not pending_insights:
            return

        # Calculer les ajustements agreges
        adjustments = {}
        for insight in pending_insights:
            for pillar, change in insight.adjustments.items():
                if pillar in self.pillar_weights:
                    adjustments[pillar] = adjustments.get(pillar, 0) + change * insight.confidence

        # Appliquer avec limite
        for pillar, change in adjustments.items():
            change = max(-max_change, min(max_change, change))
            self.pillar_weights[pillar] += change

        # Normaliser pour que la somme = 1.0
        total = sum(self.pillar_weights.values())
        self.pillar_weights = {k: v/total for k, v in self.pillar_weights.items()}

        # Marquer comme appliques
        for insight in pending_insights:
            insight.applied = True

        self._save_weights()
        self._save_insights()

        logger.info(f"[LEARNING] Updated weights: {self.pillar_weights}")

    def get_statistics(self) -> Dict:
        """Obtenir les statistiques globales du shadow tracking"""
        total = len(self.signals)
        pending = len([s for s in self.signals if s.outcome == "pending"])
        winners = len([s for s in self.signals if s.outcome in [
            SignalOutcome.WINNER.value, SignalOutcome.TARGET_HIT.value
        ]])
        losers = len([s for s in self.signals if s.outcome in [
            SignalOutcome.LOSER.value, SignalOutcome.STOPPED_OUT.value
        ]])

        completed = winners + losers
        win_rate = (winners / completed * 100) if completed > 0 else 0

        # P&L moyen
        avg_winner_pnl = 0
        avg_loser_pnl = 0

        winner_signals = [s for s in self.signals if s.outcome in [
            SignalOutcome.WINNER.value, SignalOutcome.TARGET_HIT.value
        ]]
        loser_signals = [s for s in self.signals if s.outcome in [
            SignalOutcome.LOSER.value, SignalOutcome.STOPPED_OUT.value
        ]]

        if winner_signals:
            avg_winner_pnl = sum(s.pnl_t20 or s.pnl_t10 or s.pnl_t5 or 0
                                for s in winner_signals) / len(winner_signals)
        if loser_signals:
            avg_loser_pnl = sum(s.pnl_t20 or s.pnl_t10 or s.pnl_t5 or 0
                               for s in loser_signals) / len(loser_signals)

        # Profit factor
        total_gains = sum(max(0, s.pnl_t20 or s.pnl_t10 or s.pnl_t5 or 0)
                        for s in self.signals if s.outcome != "pending")
        total_losses = abs(sum(min(0, s.pnl_t20 or s.pnl_t10 or s.pnl_t5 or 0)
                              for s in self.signals if s.outcome != "pending"))
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')

        return {
            "total_signals": total,
            "pending": pending,
            "verified": completed,
            "winners": winners,
            "losers": losers,
            "win_rate": win_rate,
            "avg_winner_pnl": avg_winner_pnl,
            "avg_loser_pnl": avg_loser_pnl,
            "profit_factor": profit_factor,
            "current_weights": self.pillar_weights,
            "insights_generated": len(self.insights),
            "insights_applied": len([i for i in self.insights if i.applied])
        }

    async def backtest_historical(
        self,
        days_back: int = 180
    ) -> Dict:
        """
        Retro-ingenierie: Rejouer les signaux historiques pour valider le modele.

        Cette fonction simule ce qui se serait passe si on avait suivi
        tous les signaux generes dans le passe.

        Args:
            days_back: Nombre de jours a analyser (defaut 180 = 6 mois)

        Returns:
            Rapport de backtest avec metriques
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)

        historical_signals = [
            s for s in self.signals
            if datetime.strptime(s.signal_date, '%Y-%m-%d') >= cutoff_date
        ]

        if not historical_signals:
            return {"error": "No historical signals to backtest"}

        # Verifier les outcomes de tous les signaux
        await self.verify_outcomes()

        # Generer le rapport
        stats = self.get_statistics()

        # Ajouter l'analyse par mois
        monthly_performance = {}
        for signal in historical_signals:
            month = signal.signal_date[:7]  # YYYY-MM
            if month not in monthly_performance:
                monthly_performance[month] = {'signals': 0, 'winners': 0, 'losers': 0}

            monthly_performance[month]['signals'] += 1
            if signal.outcome in [SignalOutcome.WINNER.value, SignalOutcome.TARGET_HIT.value]:
                monthly_performance[month]['winners'] += 1
            elif signal.outcome in [SignalOutcome.LOSER.value, SignalOutcome.STOPPED_OUT.value]:
                monthly_performance[month]['losers'] += 1

        return {
            "period": f"Last {days_back} days",
            "total_signals": len(historical_signals),
            **stats,
            "monthly_performance": monthly_performance
        }


# Singleton
_shadow_tracker: Optional[ShadowTracker] = None


async def get_shadow_tracker() -> ShadowTracker:
    """Obtenir l'instance singleton du ShadowTracker"""
    global _shadow_tracker
    if _shadow_tracker is None:
        _shadow_tracker = ShadowTracker()
        await _shadow_tracker.initialize()
    return _shadow_tracker
