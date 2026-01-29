"""
Trade Memory - RAG-based learning system for trading decisions.

This module implements an adaptive memory system that:
1. Vectorizes trade contexts and outcomes
2. Finds similar historical trades before decisions
3. Applies time decay to prioritize recent data
4. Separates memories by market regime
5. Auto-cleans stale patterns

Uses ChromaDB for vector storage and sentence-transformers for embeddings.
"""

import os
import math
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not installed. Run: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Run: pip install sentence-transformers")


class MarketRegime(str, Enum):
    """Market regime for memory segmentation"""
    BULL = "bull"
    BEAR = "bear"
    RANGE = "range"
    HIGH_VOL = "high_volatility"
    UNKNOWN = "unknown"


class RuleStatus(str, Enum):
    """Status of a learned rule"""
    ACTIVE = "active"
    QUESTIONABLE = "questionable"
    DISABLED = "disabled"


@dataclass
class TradeContext:
    """Context captured at trade entry"""
    symbol: str
    timestamp: str

    # Market context
    market_regime: str
    vix_level: float
    spy_trend: str  # "up", "down", "sideways"
    sector_momentum: float

    # Technical signals
    rsi: float
    ema_alignment: str  # "bullish", "bearish", "neutral"
    volume_ratio: float
    breakout_strength: Optional[float] = None

    # Sentiment
    reddit_sentiment: float = 0.0
    grok_sentiment: float = 0.0
    stocktwits_sentiment: float = 0.0
    heat_score: float = 0.0

    # News
    news_sentiment: float = 0.0
    catalyst_present: bool = False

    # Decision
    decision_score: float = 0.0
    confidence: float = 0.0

    def to_text(self) -> str:
        """Convert to text for embedding"""
        parts = [
            f"Symbol: {self.symbol}",
            f"Market: {self.market_regime}, VIX: {self.vix_level:.1f}, SPY: {self.spy_trend}",
            f"Technical: RSI={self.rsi:.1f}, EMA={self.ema_alignment}, Volume={self.volume_ratio:.1f}x",
            f"Sentiment: Reddit={self.reddit_sentiment:.2f}, Grok={self.grok_sentiment:.2f}",
            f"Heat: {self.heat_score:.2f}, News: {self.news_sentiment:.2f}",
            f"Decision: score={self.decision_score:.1f}, confidence={self.confidence:.2f}"
        ]
        return " | ".join(parts)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeContext':
        return cls(**data)


@dataclass
class TradeOutcome:
    """Outcome of a trade"""
    pnl_pct: float
    hold_days: int
    exit_type: str  # "stop_loss", "take_profit", "trailing_stop", "manual"
    max_drawdown: float = 0.0
    max_gain: float = 0.0

    @property
    def is_win(self) -> bool:
        return self.pnl_pct > 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StoredTrade:
    """Complete trade stored in memory"""
    id: str
    context: TradeContext
    outcome: TradeOutcome
    stored_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def age_days(self) -> int:
        stored = datetime.fromisoformat(self.stored_at)
        return (datetime.now() - stored).days

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'context': self.context.to_dict(),
            'outcome': self.outcome.to_dict(),
            'stored_at': self.stored_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StoredTrade':
        return cls(
            id=data['id'],
            context=TradeContext.from_dict(data['context']),
            outcome=TradeOutcome(**data['outcome']),
            stored_at=data.get('stored_at', datetime.now().isoformat())
        )


@dataclass
class SimilarTrade:
    """A trade found by similarity search"""
    trade: StoredTrade
    similarity: float  # 0-1, higher is more similar
    relevance: float   # Adjusted by time decay and regime match

    @property
    def is_win(self) -> bool:
        return self.trade.outcome.is_win


@dataclass
class LearnedRule:
    """A rule extracted from trade patterns"""
    id: str
    pattern: str  # Human-readable pattern description
    conditions: Dict[str, Any]  # Machine-readable conditions

    # Performance
    historical_win_rate: float
    recent_win_rate: float  # Last 30 days
    occurrences: int
    avg_pnl: float

    # Validation
    status: RuleStatus = RuleStatus.ACTIVE
    last_validated: str = field(default_factory=lambda: datetime.now().isoformat())
    consecutive_failures: int = 0

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'LearnedRule':
        data['status'] = RuleStatus(data.get('status', 'active'))
        return cls(**data)


class AdaptiveTradeMemory:
    """
    Adaptive memory system for trading with time decay and regime awareness.

    Features:
    - Regime-specific collections (bull, bear, range, high_vol)
    - Time decay (half-life of 30 days)
    - Automatic cleanup of old trades (>90 days)
    - Continuous rule validation
    """

    # Time decay parameters
    HALF_LIFE_DAYS = 30
    MAX_AGE_DAYS = 90

    # Rule validation
    MIN_OCCURRENCES = 3
    DIVERGENCE_WARNING = 0.15  # 15%
    DIVERGENCE_DISABLE = 0.20  # 20%
    MAX_CONSECUTIVE_FAILURES = 3

    def __init__(self, db_path: str = "data/vector_store"):
        """
        Initialize the adaptive memory.

        Args:
            db_path: Path to ChromaDB storage
        """
        self.db_path = db_path
        self._client = None
        self._collections: Dict[str, Any] = {}
        self._embedder = None
        self._initialized = False

        # Rules storage
        self._rules_path = os.path.join(db_path, "learned_rules.json")
        self._rules: Dict[str, LearnedRule] = {}

        # Create directory
        os.makedirs(db_path, exist_ok=True)

    def _ensure_initialized(self):
        """Lazy initialization of ChromaDB and embedder"""
        if self._initialized:
            return True

        if not CHROMADB_AVAILABLE:
            logger.error("ChromaDB not available. Install with: pip install chromadb")
            return False

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers not available. Install with: pip install sentence-transformers")
            return False

        try:
            # Initialize ChromaDB
            self._client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False)
            )

            # Create collections for each regime
            for regime in MarketRegime:
                if regime != MarketRegime.UNKNOWN:
                    collection_name = f"trades_{regime.value}"
                    self._collections[regime.value] = self._client.get_or_create_collection(
                        name=collection_name,
                        metadata={"regime": regime.value}
                    )

            # Initialize embedder
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

            # Load rules
            self._load_rules()

            self._initialized = True
            logger.info(f"AdaptiveTradeMemory initialized with {len(self._collections)} regime collections")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize AdaptiveTradeMemory: {e}")
            return False

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text"""
        if not self._embedder:
            return []
        return self._embedder.encode(text).tolist()

    def _calculate_relevance(
        self,
        trade_age_days: int,
        trade_regime: str,
        current_regime: str
    ) -> float:
        """
        Calculate relevance weight based on time decay and regime match.

        Args:
            trade_age_days: Age of the trade in days
            trade_regime: Regime when trade was made
            current_regime: Current market regime

        Returns:
            Relevance weight (0-1.5)
        """
        # Time decay: exponential with half-life
        time_weight = math.exp(-trade_age_days * math.log(2) / self.HALF_LIFE_DAYS)

        # Regime match bonus
        if trade_regime == current_regime:
            regime_multiplier = 1.5  # 50% bonus for matching regime
        else:
            regime_multiplier = 0.7  # 30% penalty for different regime

        return time_weight * regime_multiplier

    def store_trade(
        self,
        trade_id: str,
        context: TradeContext,
        outcome: TradeOutcome
    ) -> bool:
        """
        Store a completed trade in memory.

        Args:
            trade_id: Unique identifier for the trade
            context: Trade context at entry
            outcome: Trade outcome (P&L, exit type, etc.)

        Returns:
            True if stored successfully
        """
        if not self._ensure_initialized():
            return False

        try:
            # Create stored trade
            stored = StoredTrade(
                id=trade_id,
                context=context,
                outcome=outcome
            )

            # Get collection for this regime
            regime = context.market_regime
            if regime not in self._collections:
                regime = MarketRegime.UNKNOWN.value

            collection = self._collections.get(regime)
            if not collection:
                logger.warning(f"No collection for regime: {regime}")
                return False

            # Generate embedding
            text = context.to_text()
            embedding = self._get_embedding(text)

            if not embedding:
                logger.error("Failed to generate embedding")
                return False

            # Store in ChromaDB
            collection.add(
                ids=[trade_id],
                embeddings=[embedding],
                metadatas=[{
                    "symbol": context.symbol,
                    "pnl": outcome.pnl_pct,
                    "win": outcome.is_win,
                    "hold_days": outcome.hold_days,
                    "exit_type": outcome.exit_type,
                    "regime": regime,
                    "stored_at": stored.stored_at,
                    "vix": context.vix_level,
                    "rsi": context.rsi,
                    "volume_ratio": context.volume_ratio,
                    "heat_score": context.heat_score,
                    "decision_score": context.decision_score,
                }],
                documents=[json.dumps(stored.to_dict())]
            )

            logger.info(f"Stored trade {trade_id} in {regime} collection (PnL: {outcome.pnl_pct:+.1f}%)")
            return True

        except Exception as e:
            logger.error(f"Failed to store trade: {e}")
            return False

    def find_similar(
        self,
        context: TradeContext,
        top_k: int = 5,
        current_regime: Optional[str] = None
    ) -> List[SimilarTrade]:
        """
        Find similar historical trades.

        Args:
            context: Current trade context
            top_k: Number of similar trades to return
            current_regime: Current market regime (for relevance weighting)

        Returns:
            List of similar trades, sorted by relevance
        """
        if not self._ensure_initialized():
            return []

        try:
            current_regime = current_regime or context.market_regime

            # Generate embedding for current context
            text = context.to_text()
            embedding = self._get_embedding(text)

            if not embedding:
                return []

            # Search primarily in current regime collection
            primary_collection = self._collections.get(current_regime)
            results = []

            if primary_collection and primary_collection.count() > 0:
                query_result = primary_collection.query(
                    query_embeddings=[embedding],
                    n_results=min(top_k * 2, primary_collection.count()),  # Get more, filter later
                    include=["metadatas", "documents", "distances"]
                )

                results.extend(self._process_query_results(query_result, current_regime))

            # Optionally search other regimes with reduced weight
            for regime, collection in self._collections.items():
                if regime != current_regime and collection.count() > 0:
                    query_result = collection.query(
                        query_embeddings=[embedding],
                        n_results=min(3, collection.count()),  # Fewer from other regimes
                        include=["metadatas", "documents", "distances"]
                    )
                    results.extend(self._process_query_results(query_result, current_regime))

            # Sort by relevance and return top_k
            results.sort(key=lambda x: x.relevance, reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error finding similar trades: {e}")
            return []

    def _process_query_results(
        self,
        query_result: Dict,
        current_regime: str
    ) -> List[SimilarTrade]:
        """Process ChromaDB query results into SimilarTrade objects"""
        results = []

        if not query_result or not query_result.get('ids') or not query_result['ids'][0]:
            return results

        ids = query_result['ids'][0]
        metadatas = query_result.get('metadatas', [[]])[0]
        documents = query_result.get('documents', [[]])[0]
        distances = query_result.get('distances', [[]])[0]

        for i, trade_id in enumerate(ids):
            try:
                # Parse stored trade
                trade_data = json.loads(documents[i]) if documents else {}
                stored_trade = StoredTrade.from_dict(trade_data) if trade_data else None

                if not stored_trade:
                    continue

                # Calculate similarity (ChromaDB returns L2 distance, convert to similarity)
                distance = distances[i] if distances else 0
                similarity = 1 / (1 + distance)  # Convert distance to similarity

                # Calculate relevance with time decay
                trade_regime = metadatas[i].get('regime', 'unknown') if metadatas else 'unknown'
                relevance = self._calculate_relevance(
                    stored_trade.age_days,
                    trade_regime,
                    current_regime
                )

                # Combine similarity and relevance
                final_relevance = similarity * relevance

                results.append(SimilarTrade(
                    trade=stored_trade,
                    similarity=similarity,
                    relevance=final_relevance
                ))

            except Exception as e:
                logger.warning(f"Error processing trade {trade_id}: {e}")
                continue

        return results

    def get_historical_win_rate(
        self,
        similar_trades: List[SimilarTrade],
        weighted: bool = True
    ) -> Tuple[float, int]:
        """
        Calculate win rate from similar trades.

        Args:
            similar_trades: List of similar trades
            weighted: Whether to weight by relevance

        Returns:
            (win_rate, sample_size)
        """
        if not similar_trades:
            return 0.5, 0  # Neutral if no data

        if weighted:
            total_weight = sum(t.relevance for t in similar_trades)
            if total_weight == 0:
                return 0.5, 0

            weighted_wins = sum(t.relevance for t in similar_trades if t.is_win)
            win_rate = weighted_wins / total_weight
        else:
            wins = sum(1 for t in similar_trades if t.is_win)
            win_rate = wins / len(similar_trades)

        return win_rate, len(similar_trades)

    def calculate_score_adjustment(
        self,
        similar_trades: List[SimilarTrade],
        max_adjustment: float = 10.0
    ) -> float:
        """
        Calculate score adjustment based on historical performance.

        Args:
            similar_trades: List of similar trades
            max_adjustment: Maximum score adjustment (+/-)

        Returns:
            Score adjustment to add to decision score
        """
        win_rate, sample_size = self.get_historical_win_rate(similar_trades)

        if sample_size < 2:
            return 0.0  # Not enough data

        # Adjust based on deviation from 50%
        # Win rate 70% -> +10 points
        # Win rate 30% -> -10 points
        adjustment = (win_rate - 0.5) * 2 * max_adjustment

        # Scale by sample size confidence (more samples = more confident)
        confidence = min(1.0, sample_size / 10)

        return adjustment * confidence

    async def cleanup_stale_memory(self) -> Dict[str, int]:
        """
        Clean up old trades and rare patterns.
        Should be called nightly in audit phase.

        Returns:
            Dict with cleanup statistics
        """
        if not self._ensure_initialized():
            return {"error": "Not initialized"}

        stats = {"removed_old": 0, "removed_rare": 0}
        cutoff_date = (datetime.now() - timedelta(days=self.MAX_AGE_DAYS)).isoformat()

        try:
            for regime, collection in self._collections.items():
                if collection.count() == 0:
                    continue

                # Get all trades
                all_data = collection.get(include=["metadatas"])

                if not all_data or not all_data.get('ids'):
                    continue

                ids_to_remove = []

                for i, trade_id in enumerate(all_data['ids']):
                    metadata = all_data['metadatas'][i] if all_data.get('metadatas') else {}
                    stored_at = metadata.get('stored_at', '')

                    if stored_at and stored_at < cutoff_date:
                        ids_to_remove.append(trade_id)

                if ids_to_remove:
                    collection.delete(ids=ids_to_remove)
                    stats["removed_old"] += len(ids_to_remove)
                    logger.info(f"Removed {len(ids_to_remove)} old trades from {regime}")

            # Cleanup rules with few occurrences
            rules_to_remove = [
                rule_id for rule_id, rule in self._rules.items()
                if rule.occurrences < self.MIN_OCCURRENCES and rule.status == RuleStatus.DISABLED
            ]

            for rule_id in rules_to_remove:
                del self._rules[rule_id]
                stats["removed_rare"] += 1

            if rules_to_remove:
                self._save_rules()

            logger.info(f"Memory cleanup: removed {stats['removed_old']} old trades, {stats['removed_rare']} rare rules")
            return stats

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return {"error": str(e)}

    # =========================================================================
    # RULE MANAGEMENT
    # =========================================================================

    def _load_rules(self):
        """Load learned rules from disk"""
        if os.path.exists(self._rules_path):
            try:
                with open(self._rules_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._rules = {
                        k: LearnedRule.from_dict(v) for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self._rules)} learned rules")
            except Exception as e:
                logger.error(f"Failed to load rules: {e}")
                self._rules = {}
        else:
            self._rules = {}

    def _save_rules(self):
        """Save learned rules to disk"""
        try:
            with open(self._rules_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {k: v.to_dict() for k, v in self._rules.items()},
                    f, indent=2, ensure_ascii=False
                )
        except Exception as e:
            logger.error(f"Failed to save rules: {e}")

    def add_rule(self, rule: LearnedRule):
        """Add or update a learned rule"""
        self._rules[rule.id] = rule
        self._save_rules()
        logger.info(f"Added/updated rule: {rule.pattern}")

    def get_active_rules(self) -> List[LearnedRule]:
        """Get all active rules"""
        return [r for r in self._rules.values() if r.status == RuleStatus.ACTIVE]

    def validate_rules(self, recent_trades: List[StoredTrade]) -> Dict[str, str]:
        """
        Validate rules against recent trades.

        Args:
            recent_trades: Trades from last 30 days

        Returns:
            Dict of rule_id -> new status
        """
        changes = {}

        for rule_id, rule in self._rules.items():
            if rule.status == RuleStatus.DISABLED:
                continue

            # Find trades matching this rule's conditions
            matching = self._find_matching_trades(rule, recent_trades)

            if len(matching) < self.MIN_OCCURRENCES:
                continue  # Not enough data to validate

            # Calculate recent win rate
            wins = sum(1 for t in matching if t.outcome.is_win)
            recent_win_rate = wins / len(matching)

            # Check divergence
            divergence = abs(recent_win_rate - rule.historical_win_rate)

            old_status = rule.status

            if divergence > self.DIVERGENCE_DISABLE:
                rule.status = RuleStatus.DISABLED
                logger.warning(f"Rule disabled: {rule.pattern} (divergence={divergence:.0%})")
            elif divergence > self.DIVERGENCE_WARNING:
                rule.status = RuleStatus.QUESTIONABLE
            else:
                rule.status = RuleStatus.ACTIVE
                rule.consecutive_failures = 0

            rule.recent_win_rate = recent_win_rate
            rule.last_validated = datetime.now().isoformat()

            if old_status != rule.status:
                changes[rule_id] = rule.status.value

        if changes:
            self._save_rules()

        return changes

    def _find_matching_trades(
        self,
        rule: LearnedRule,
        trades: List[StoredTrade]
    ) -> List[StoredTrade]:
        """Find trades that match a rule's conditions"""
        matching = []

        for trade in trades:
            if self._trade_matches_conditions(trade, rule.conditions):
                matching.append(trade)

        return matching

    def _trade_matches_conditions(
        self,
        trade: StoredTrade,
        conditions: Dict[str, Any]
    ) -> bool:
        """Check if a trade matches rule conditions"""
        ctx = trade.context

        for key, condition in conditions.items():
            value = getattr(ctx, key, None)
            if value is None:
                continue

            op = condition.get('op', '==')
            threshold = condition.get('value')

            if op == '>=' and value < threshold:
                return False
            elif op == '<=' and value > threshold:
                return False
            elif op == '>' and value <= threshold:
                return False
            elif op == '<' and value >= threshold:
                return False
            elif op == '==' and value != threshold:
                return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        if not self._ensure_initialized():
            return {"initialized": False}

        stats = {
            "initialized": True,
            "collections": {},
            "total_trades": 0,
            "rules": {
                "total": len(self._rules),
                "active": len([r for r in self._rules.values() if r.status == RuleStatus.ACTIVE]),
                "questionable": len([r for r in self._rules.values() if r.status == RuleStatus.QUESTIONABLE]),
                "disabled": len([r for r in self._rules.values() if r.status == RuleStatus.DISABLED]),
            }
        }

        for regime, collection in self._collections.items():
            count = collection.count()
            stats["collections"][regime] = count
            stats["total_trades"] += count

        return stats


# =========================================================================
# SINGLETON
# =========================================================================

_memory_instance: Optional[AdaptiveTradeMemory] = None


def get_trade_memory(db_path: str = "data/vector_store") -> AdaptiveTradeMemory:
    """Get or create the global TradeMemory instance"""
    global _memory_instance

    if _memory_instance is None:
        _memory_instance = AdaptiveTradeMemory(db_path)

    return _memory_instance


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def create_context_from_reasoning(
    symbol: str,
    reasoning_result: Any,
    market_context: Any
) -> TradeContext:
    """
    Create a TradeContext from ReasoningEngine output.

    Args:
        symbol: Stock symbol
        reasoning_result: Result from ReasoningEngine.analyze()
        market_context: Current market context

    Returns:
        TradeContext ready for memory storage
    """
    return TradeContext(
        symbol=symbol,
        timestamp=datetime.now().isoformat(),
        market_regime=market_context.regime.value if hasattr(market_context, 'regime') else 'unknown',
        vix_level=getattr(market_context, 'vix_level', 15.0),
        spy_trend=getattr(market_context, 'spy_trend', 'sideways'),
        sector_momentum=getattr(market_context, 'sector_momentum', 0.0),
        rsi=getattr(reasoning_result.technical_score, 'rsi', 50.0) if hasattr(reasoning_result, 'technical_score') else 50.0,
        ema_alignment=getattr(reasoning_result.technical_score, 'ema_alignment', 'neutral') if hasattr(reasoning_result, 'technical_score') else 'neutral',
        volume_ratio=getattr(reasoning_result.technical_score, 'volume_ratio', 1.0) if hasattr(reasoning_result, 'technical_score') else 1.0,
        reddit_sentiment=getattr(reasoning_result.sentiment_score, 'reddit', 0.0) if hasattr(reasoning_result, 'sentiment_score') else 0.0,
        grok_sentiment=getattr(reasoning_result.sentiment_score, 'grok', 0.0) if hasattr(reasoning_result, 'sentiment_score') else 0.0,
        news_sentiment=getattr(reasoning_result.news_score, 'sentiment', 0.0) if hasattr(reasoning_result, 'news_score') else 0.0,
        heat_score=getattr(reasoning_result, 'heat_score', 0.0),
        decision_score=reasoning_result.total_score if hasattr(reasoning_result, 'total_score') else 0.0,
        confidence=reasoning_result.confidence if hasattr(reasoning_result, 'confidence') else 0.5,
    )
