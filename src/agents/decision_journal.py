"""
Decision Journal - Journal des decisions de trading avec raisonnement complet
Documente chaque decision avec tous les facteurs et le raisonnement.
"""

import asyncio
import logging
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class DecisionAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    SKIP = "SKIP"  # Signal detecte mais ignore


class DecisionOutcome(str, Enum):
    PENDING = "pending"
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    STOPPED_OUT = "stopped_out"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TechnicalFactors:
    """Facteurs techniques de la decision"""
    rsi_value: Optional[float] = None
    rsi_breakout: Optional[str] = None  # "STRONG", "MODERATE", "WEAK", None
    rsi_breakout_age: Optional[int] = None

    ema_alignment: Optional[str] = None  # "bullish", "bearish", "neutral"
    ema_24: Optional[float] = None
    ema_38: Optional[float] = None
    ema_62: Optional[float] = None

    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    distance_to_support: Optional[float] = None

    volume_ratio: Optional[float] = None
    volume_confirmed: bool = False

    confidence_score: float = 0


@dataclass
class SentimentFactors:
    """Facteurs de sentiment/news"""
    heat_score: float = 0
    reddit_mentions: int = 0
    reddit_sentiment: float = 0
    grok_sentiment: float = 0
    grok_summary: str = ""

    news_headlines: List[str] = field(default_factory=list)
    sentiment_sources: List[str] = field(default_factory=list)


@dataclass
class MarketFactors:
    """Facteurs de marche"""
    market_regime: str = "neutral"  # "bull", "bear", "neutral", "volatile"
    sector: str = ""
    sector_momentum: float = 0
    spy_change: float = 0
    vix_level: float = 0


@dataclass
class IntuitionFactors:
    """Facteurs d'intuition (pattern matching)"""
    similar_trades: List[str] = field(default_factory=list)  # IDs de trades similaires
    pattern_description: str = ""
    pattern_win_rate: float = 0
    pattern_avg_return: float = 0


@dataclass
class ExecutionDetails:
    """Details d'execution"""
    quantity: int = 0
    entry_price: float = 0
    position_value: float = 0
    stop_loss: float = 0
    stop_source: str = ""  # "rsi_low", "ema_support", "atr", etc.
    take_profit: Optional[float] = None
    risk_amount: float = 0
    risk_percent: float = 0


@dataclass
class TradeDecision:
    """Decision de trade complete avec raisonnement"""
    # Identification
    id: str
    symbol: str
    action: DecisionAction
    timestamp: datetime

    # Raisonnement genere par LLM
    reasoning: str

    # Facteurs
    technical: TechnicalFactors
    sentiment: SentimentFactors
    market: MarketFactors
    intuition: IntuitionFactors

    # Execution
    execution: ExecutionDetails

    # Outcome (rempli plus tard)
    outcome: DecisionOutcome = DecisionOutcome.PENDING
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    pnl_amount: Optional[float] = None
    pnl_percent: Optional[float] = None
    hold_duration_minutes: Optional[int] = None

    # Meta
    strategy_used: str = ""
    version: str = "1.0"


# =============================================================================
# DECISION JOURNAL
# =============================================================================

class DecisionJournal:
    """
    Journal des decisions de trading.

    Enregistre chaque decision avec:
    - Le raisonnement complet genere par LLM
    - Tous les facteurs (technique, sentiment, marche, intuition)
    - Les details d'execution
    - Le resultat final
    """

    def __init__(self, grok_client=None):
        self.grok_client = grok_client

        # Decisions en memoire
        self._decisions: Dict[str, TradeDecision] = {}
        self._pending_decisions: List[str] = []  # IDs des decisions en attente d'outcome

        # Statistiques
        self._stats = {
            'total_decisions': 0,
            'wins': 0,
            'losses': 0,
            'pending': 0
        }

        # Persistence
        self._data_dir = Path("data/decisions")
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Lock pour thread safety
        self._lock = asyncio.Lock()

        logger.info("DecisionJournal initialized")

    async def initialize(self):
        """Initialise le journal"""
        await self._load_recent_decisions()
        logger.info(f"DecisionJournal ready - {len(self._decisions)} decisions loaded")

    # -------------------------------------------------------------------------
    # CREATION DE DECISION
    # -------------------------------------------------------------------------

    async def create_decision(
        self,
        symbol: str,
        action: DecisionAction,
        technical: TechnicalFactors,
        sentiment: SentimentFactors,
        market: MarketFactors,
        intuition: IntuitionFactors,
        execution: ExecutionDetails,
        strategy: str = "live_heat"
    ) -> TradeDecision:
        """
        Cree une nouvelle decision avec generation du raisonnement.
        """
        async with self._lock:
            # Generer l'ID
            decision_id = f"{symbol}_{action.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Generer le raisonnement
            reasoning = await self._generate_reasoning(
                symbol, action, technical, sentiment, market, intuition
            )

            # Creer la decision
            decision = TradeDecision(
                id=decision_id,
                symbol=symbol,
                action=action,
                timestamp=datetime.now(),
                reasoning=reasoning,
                technical=technical,
                sentiment=sentiment,
                market=market,
                intuition=intuition,
                execution=execution,
                strategy_used=strategy
            )

            # Sauvegarder
            self._decisions[decision_id] = decision

            if action in (DecisionAction.BUY, DecisionAction.SELL):
                self._pending_decisions.append(decision_id)
                self._stats['pending'] += 1

            self._stats['total_decisions'] += 1

            # Persister
            await self._save_decision(decision)

            # Logger
            self._log_decision(decision)

            return decision

    async def _generate_reasoning(
        self,
        symbol: str,
        action: DecisionAction,
        technical: TechnicalFactors,
        sentiment: SentimentFactors,
        market: MarketFactors,
        intuition: IntuitionFactors
    ) -> str:
        """Genere le raisonnement via LLM ou template"""

        # Si Grok disponible, utiliser LLM
        if self.grok_client and hasattr(self.grok_client, 'generate_reasoning'):
            try:
                return await self.grok_client.generate_reasoning(
                    symbol, action.value, technical, sentiment, market, intuition
                )
            except Exception as e:
                logger.warning(f"LLM reasoning failed: {e}")

        # Fallback: template
        return self._generate_template_reasoning(
            symbol, action, technical, sentiment, market, intuition
        )

    def _generate_template_reasoning(
        self,
        symbol: str,
        action: DecisionAction,
        technical: TechnicalFactors,
        sentiment: SentimentFactors,
        market: MarketFactors,
        intuition: IntuitionFactors
    ) -> str:
        """Genere un raisonnement base sur template"""
        parts = []

        # Action
        if action == DecisionAction.BUY:
            parts.append(f"{symbol} presente une opportunite d'achat.")
        elif action == DecisionAction.SELL:
            parts.append(f"Signal de vente detecte sur {symbol}.")
        elif action == DecisionAction.SKIP:
            parts.append(f"Signal ignore pour {symbol}.")
        else:
            parts.append(f"Decision HOLD pour {symbol}.")

        # Technique
        if technical.rsi_breakout:
            parts.append(
                f"Breakout RSI {technical.rsi_breakout} detecte "
                f"(age: {technical.rsi_breakout_age} periodes)."
            )

        if technical.ema_alignment == "bullish":
            parts.append("Les EMAs sont alignees en mode haussier.")
        elif technical.ema_alignment == "bearish":
            parts.append("Les EMAs montrent une configuration baissiere.")

        if technical.volume_confirmed:
            parts.append(f"Volume confirme ({technical.volume_ratio:.1f}x la moyenne).")

        # Sentiment
        if sentiment.heat_score > 0.5:
            parts.append(
                f"Le symbole est 'chaud' (heat: {sentiment.heat_score:.2f}) "
                f"avec {sentiment.reddit_mentions} mentions Reddit."
            )

        if sentiment.grok_sentiment > 0.5:
            parts.append(f"Sentiment Grok bullish ({sentiment.grok_sentiment:.2f}).")
        elif sentiment.grok_sentiment < -0.5:
            parts.append(f"Sentiment Grok bearish ({sentiment.grok_sentiment:.2f}).")

        # Marche
        if market.sector_momentum > 0.02:
            parts.append(f"Le secteur {market.sector} est en momentum (+{market.sector_momentum:.1%}).")

        # Intuition
        if intuition.similar_trades:
            parts.append(
                f"Pattern similaire a {len(intuition.similar_trades)} trades precedents "
                f"(win rate: {intuition.pattern_win_rate:.0%})."
            )

        # Confiance
        parts.append(f"Score de confiance: {technical.confidence_score:.0f}/100.")

        return " ".join(parts)

    def _log_decision(self, decision: TradeDecision):
        """Log la decision dans un format lisible"""
        separator = "=" * 60

        log_text = f"""
{separator}
TRADE DECISION: {decision.action.value} {decision.execution.quantity} {decision.symbol} @ ${decision.execution.entry_price:.2f}
{separator}

RAISONNEMENT:
"{decision.reasoning}"

TECHNIQUE:
- RSI Breakout: {decision.technical.rsi_breakout or 'None'} (age={decision.technical.rsi_breakout_age})
- EMA Alignment: {decision.technical.ema_alignment}
- Support: ${decision.technical.support_level or 0:.2f}
- Confidence Score: {decision.technical.confidence_score:.0f}/100

HEAT:
- Heat Score: {decision.sentiment.heat_score:.2f}
- Reddit: {decision.sentiment.reddit_mentions} mentions ({decision.sentiment.reddit_sentiment:+.2f})
- Grok Sentiment: {decision.sentiment.grok_sentiment:+.2f}

MARCHE:
- Regime: {decision.market.market_regime}
- Secteur: {decision.market.sector} ({decision.market.sector_momentum:+.1%})

EXECUTION:
- Position: {decision.execution.quantity} shares (${decision.execution.position_value:.2f})
- Stop Loss: ${decision.execution.stop_loss:.2f} ({decision.execution.stop_source})
- Risk: ${decision.execution.risk_amount:.2f} ({decision.execution.risk_percent:.1%})
{separator}
"""
        logger.info(log_text)

    # -------------------------------------------------------------------------
    # MISE A JOUR OUTCOME
    # -------------------------------------------------------------------------

    async def update_outcome(
        self,
        decision_id: str,
        exit_price: float,
        outcome: DecisionOutcome,
        exit_reason: str = ""
    ):
        """Met a jour l'outcome d'une decision"""
        async with self._lock:
            if decision_id not in self._decisions:
                logger.warning(f"Decision not found: {decision_id}")
                return

            decision = self._decisions[decision_id]
            now = datetime.now()

            # Calculer P&L
            entry = decision.execution.entry_price
            quantity = decision.execution.quantity

            if decision.action == DecisionAction.BUY:
                pnl_amount = (exit_price - entry) * quantity
                pnl_percent = (exit_price - entry) / entry
            else:  # SELL (short)
                pnl_amount = (entry - exit_price) * quantity
                pnl_percent = (entry - exit_price) / entry

            # Mettre a jour
            decision.exit_price = exit_price
            decision.exit_timestamp = now
            decision.outcome = outcome
            decision.pnl_amount = pnl_amount
            decision.pnl_percent = pnl_percent
            decision.hold_duration_minutes = int(
                (now - decision.timestamp).total_seconds() / 60
            )

            # Stats
            if decision_id in self._pending_decisions:
                self._pending_decisions.remove(decision_id)
                self._stats['pending'] -= 1

            if outcome == DecisionOutcome.WIN:
                self._stats['wins'] += 1
            elif outcome in (DecisionOutcome.LOSS, DecisionOutcome.STOPPED_OUT):
                self._stats['losses'] += 1

            # Sauvegarder
            await self._save_decision(decision)

            # Logger
            logger.info(
                f"Decision {decision_id} closed: {outcome.value} "
                f"P&L: {pnl_percent:+.2%} (${pnl_amount:+.2f})"
            )

    # -------------------------------------------------------------------------
    # QUERIES
    # -------------------------------------------------------------------------

    def get_decision(self, decision_id: str) -> Optional[TradeDecision]:
        """Retourne une decision par ID"""
        return self._decisions.get(decision_id)

    def get_recent_decisions(self, limit: int = 20) -> List[TradeDecision]:
        """Retourne les decisions recentes"""
        decisions = list(self._decisions.values())
        decisions.sort(key=lambda d: d.timestamp, reverse=True)
        return decisions[:limit]

    def get_pending_decisions(self) -> List[TradeDecision]:
        """Retourne les decisions en attente d'outcome"""
        return [
            self._decisions[did]
            for did in self._pending_decisions
            if did in self._decisions
        ]

    def get_decisions_for_symbol(self, symbol: str, limit: int = 10) -> List[TradeDecision]:
        """Retourne les decisions pour un symbole"""
        decisions = [
            d for d in self._decisions.values()
            if d.symbol == symbol.upper()
        ]
        decisions.sort(key=lambda d: d.timestamp, reverse=True)
        return decisions[:limit]

    def get_winning_decisions(self, limit: int = 20) -> List[TradeDecision]:
        """Retourne les decisions gagnantes"""
        decisions = [
            d for d in self._decisions.values()
            if d.outcome == DecisionOutcome.WIN
        ]
        decisions.sort(key=lambda d: d.pnl_percent or 0, reverse=True)
        return decisions[:limit]

    def get_losing_decisions(self, limit: int = 20) -> List[TradeDecision]:
        """Retourne les decisions perdantes"""
        decisions = [
            d for d in self._decisions.values()
            if d.outcome in (DecisionOutcome.LOSS, DecisionOutcome.STOPPED_OUT)
        ]
        decisions.sort(key=lambda d: d.pnl_percent or 0)
        return decisions[:limit]

    def get_stats(self) -> Dict:
        """Retourne les statistiques"""
        total = self._stats['wins'] + self._stats['losses']
        win_rate = self._stats['wins'] / total if total > 0 else 0

        return {
            'total_decisions': self._stats['total_decisions'],
            'wins': self._stats['wins'],
            'losses': self._stats['losses'],
            'pending': self._stats['pending'],
            'win_rate': win_rate,
            'total_trades': total
        }

    def find_similar_trades(
        self,
        technical: TechnicalFactors,
        sentiment: SentimentFactors,
        limit: int = 5
    ) -> List[TradeDecision]:
        """
        Trouve des trades similaires dans l'historique.
        Utilise pour l'intuition/pattern matching.
        """
        scored_decisions = []

        for decision in self._decisions.values():
            if decision.outcome == DecisionOutcome.PENDING:
                continue

            # Calculer similarite
            score = 0

            # Similarite RSI
            if decision.technical.rsi_breakout == technical.rsi_breakout:
                score += 2

            # Similarite EMA
            if decision.technical.ema_alignment == technical.ema_alignment:
                score += 1

            # Similarite heat
            heat_diff = abs(decision.sentiment.heat_score - sentiment.heat_score)
            if heat_diff < 0.2:
                score += 1

            # Similarite sentiment
            sent_diff = abs(decision.sentiment.grok_sentiment - sentiment.grok_sentiment)
            if sent_diff < 0.3:
                score += 1

            if score >= 2:
                scored_decisions.append((score, decision))

        # Trier par score
        scored_decisions.sort(key=lambda x: x[0], reverse=True)

        return [d for _, d in scored_decisions[:limit]]

    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------

    async def _save_decision(self, decision: TradeDecision):
        """Sauvegarde une decision sur le disque"""
        # Fichier par jour
        date_str = decision.timestamp.strftime("%Y%m%d")
        path = self._data_dir / f"decisions_{date_str}.json"

        try:
            # Charger le fichier existant
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = []

            # Trouver et mettre a jour ou ajouter
            decision_dict = self._decision_to_dict(decision)

            found = False
            for i, d in enumerate(data):
                if d.get('id') == decision.id:
                    data[i] = decision_dict
                    found = True
                    break

            if not found:
                data.append(decision_dict)

            # Sauvegarder
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        except Exception as e:
            logger.error(f"Could not save decision: {e}")

    def _decision_to_dict(self, decision: TradeDecision) -> Dict:
        """Convertit une decision en dict serializable"""
        return {
            'id': decision.id,
            'symbol': decision.symbol,
            'action': decision.action.value,
            'timestamp': decision.timestamp.isoformat(),
            'reasoning': decision.reasoning,
            'technical': asdict(decision.technical),
            'sentiment': asdict(decision.sentiment),
            'market': asdict(decision.market),
            'intuition': asdict(decision.intuition),
            'execution': asdict(decision.execution),
            'outcome': decision.outcome.value,
            'exit_price': decision.exit_price,
            'exit_timestamp': decision.exit_timestamp.isoformat() if decision.exit_timestamp else None,
            'pnl_amount': decision.pnl_amount,
            'pnl_percent': decision.pnl_percent,
            'hold_duration_minutes': decision.hold_duration_minutes,
            'strategy_used': decision.strategy_used,
            'version': decision.version
        }

    async def _load_recent_decisions(self, days: int = 30):
        """Charge les decisions recentes"""
        now = datetime.now()

        for i in range(days):
            date = now - timedelta(days=i)
            date_str = date.strftime("%Y%m%d")
            path = self._data_dir / f"decisions_{date_str}.json"

            if not path.exists():
                continue

            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for d in data:
                    decision = self._dict_to_decision(d)
                    self._decisions[decision.id] = decision

                    if decision.outcome == DecisionOutcome.PENDING:
                        self._pending_decisions.append(decision.id)

            except Exception as e:
                logger.warning(f"Could not load decisions from {path}: {e}")

        # Recalculer les stats
        self._recalculate_stats()

    def _dict_to_decision(self, d: Dict) -> TradeDecision:
        """Convertit un dict en TradeDecision"""
        return TradeDecision(
            id=d['id'],
            symbol=d['symbol'],
            action=DecisionAction(d['action']),
            timestamp=datetime.fromisoformat(d['timestamp']),
            reasoning=d['reasoning'],
            technical=TechnicalFactors(**d['technical']),
            sentiment=SentimentFactors(**d['sentiment']),
            market=MarketFactors(**d['market']),
            intuition=IntuitionFactors(**d['intuition']),
            execution=ExecutionDetails(**d['execution']),
            outcome=DecisionOutcome(d['outcome']),
            exit_price=d.get('exit_price'),
            exit_timestamp=datetime.fromisoformat(d['exit_timestamp']) if d.get('exit_timestamp') else None,
            pnl_amount=d.get('pnl_amount'),
            pnl_percent=d.get('pnl_percent'),
            hold_duration_minutes=d.get('hold_duration_minutes'),
            strategy_used=d.get('strategy_used', ''),
            version=d.get('version', '1.0')
        )

    def _recalculate_stats(self):
        """Recalcule les statistiques"""
        self._stats = {
            'total_decisions': len(self._decisions),
            'wins': sum(1 for d in self._decisions.values() if d.outcome == DecisionOutcome.WIN),
            'losses': sum(1 for d in self._decisions.values() if d.outcome in (DecisionOutcome.LOSS, DecisionOutcome.STOPPED_OUT)),
            'pending': len(self._pending_decisions)
        }

    async def close(self):
        """Ferme proprement le journal"""
        logger.info("DecisionJournal closed")


# =============================================================================
# FACTORY
# =============================================================================

_decision_journal: Optional[DecisionJournal] = None


async def get_decision_journal(grok_client=None) -> DecisionJournal:
    """Retourne l'instance singleton du DecisionJournal"""
    global _decision_journal
    if _decision_journal is None:
        _decision_journal = DecisionJournal(grok_client)
        await _decision_journal.initialize()
    return _decision_journal
