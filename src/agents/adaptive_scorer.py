"""
Adaptive Scoring System - V5 Learning Architecture

This module implements a scoring system that learns from trade outcomes
to dynamically adjust pillar weights, similar to how LLMs learn patterns.

Key concepts:
1. Feature Embeddings: Convert pillar scores + context into dense vectors
2. Outcome Recording: Track trade results for supervised learning
3. Weight Adaptation: Adjust pillar weights based on what works
4. Pattern Memory: Remember successful/failed patterns for future decisions
"""

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import logging
import sqlite3
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class TradeOutcomeType(Enum):
    """Classification of trade outcomes"""
    BIG_WIN = "big_win"          # > +15%
    WIN = "win"                   # +5% to +15%
    SMALL_WIN = "small_win"       # 0% to +5%
    SMALL_LOSS = "small_loss"     # -5% to 0%
    LOSS = "loss"                 # -15% to -5%
    BIG_LOSS = "big_loss"         # < -15%
    STOP_LOSS = "stop_loss"       # Hit stop loss


@dataclass
class FeatureVector:
    """
    Dense representation of an analysis for ML learning.
    Similar to word embeddings in LLMs, but for market analysis.
    """
    # Pillar scores (normalized -1 to +1)
    technical: float = 0.0
    fundamental: float = 0.0
    sentiment: float = 0.0
    news: float = 0.0

    # Technical sub-features
    trend_strength: float = 0.0      # -1 to +1
    momentum: float = 0.0            # -1 to +1
    volume_ratio: float = 0.0        # 0 to 2+ (relative to avg)
    volatility: float = 0.0          # 0 to 1 (normalized ATR)
    rsi: float = 0.0                 # 0 to 1 (RSI/100)

    # Fundamental sub-features
    pe_relative: float = 0.0         # -1 to +1 (vs sector)
    growth_score: float = 0.0        # -1 to +1
    profitability: float = 0.0       # -1 to +1

    # Sentiment sub-features
    social_sentiment: float = 0.0    # -1 to +1
    social_volume: float = 0.0       # 0 to 1 (relative activity)
    news_sentiment: float = 0.0      # -1 to +1

    # Market context
    market_regime: float = 0.0       # -1 (crash) to +1 (bull)
    sector_momentum: float = 0.0     # -1 to +1
    vix_level: float = 0.0           # 0 to 1 (normalized)

    # Meta features
    data_quality: float = 1.0        # 0 to 1
    confidence: float = 0.5          # 0 to 1

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML"""
        return np.array([
            self.technical, self.fundamental, self.sentiment, self.news,
            self.trend_strength, self.momentum, self.volume_ratio, self.volatility, self.rsi,
            self.pe_relative, self.growth_score, self.profitability,
            self.social_sentiment, self.social_volume, self.news_sentiment,
            self.market_regime, self.sector_momentum, self.vix_level,
            self.data_quality, self.confidence
        ], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'FeatureVector':
        """Create from numpy array"""
        return cls(
            technical=float(arr[0]), fundamental=float(arr[1]),
            sentiment=float(arr[2]), news=float(arr[3]),
            trend_strength=float(arr[4]), momentum=float(arr[5]),
            volume_ratio=float(arr[6]), volatility=float(arr[7]), rsi=float(arr[8]),
            pe_relative=float(arr[9]), growth_score=float(arr[10]), profitability=float(arr[11]),
            social_sentiment=float(arr[12]), social_volume=float(arr[13]), news_sentiment=float(arr[14]),
            market_regime=float(arr[15]), sector_momentum=float(arr[16]), vix_level=float(arr[17]),
            data_quality=float(arr[18]), confidence=float(arr[19])
        )

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TradeRecord:
    """Complete record of a trade for learning"""
    # Identification
    trade_id: str
    symbol: str

    # Entry
    entry_date: str
    entry_price: float
    entry_features: FeatureVector
    entry_score: float              # Original score 0-100
    entry_decision: str             # BUY, STRONG_BUY, etc.

    # Pillar weights at entry (for attribution)
    pillar_weights: Dict[str, float] = field(default_factory=lambda: {
        'technical': 0.25, 'fundamental': 0.25, 'sentiment': 0.25, 'news': 0.25
    })

    # Exit (filled when trade closes)
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # take_profit, stop_loss, manual, etc.

    # Outcome
    pnl_pct: Optional[float] = None
    outcome_type: Optional[str] = None
    hold_days: Optional[int] = None

    # Attribution (which pillars contributed most to success/failure)
    pillar_attribution: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['entry_features'] = self.entry_features.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'TradeRecord':
        features = FeatureVector(**d.pop('entry_features', {}))
        return cls(entry_features=features, **d)


@dataclass
class PillarWeights:
    """Learnable weights for each pillar"""
    technical: float = 0.25
    fundamental: float = 0.25
    sentiment: float = 0.25
    news: float = 0.25

    # Learning metadata
    last_updated: str = ""
    update_count: int = 0
    based_on_trades: int = 0

    def normalize(self):
        """Ensure weights sum to 1"""
        total = self.technical + self.fundamental + self.sentiment + self.news
        if total > 0:
            self.technical /= total
            self.fundamental /= total
            self.sentiment /= total
            self.news /= total

    def to_dict(self) -> Dict:
        return {
            'technical': self.technical,
            'fundamental': self.fundamental,
            'sentiment': self.sentiment,
            'news': self.news,
            'last_updated': self.last_updated,
            'update_count': self.update_count,
            'based_on_trades': self.based_on_trades
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'PillarWeights':
        return cls(**d)


# =============================================================================
# ADAPTIVE SCORER
# =============================================================================

class AdaptiveScorer:
    """
    Scoring system that learns from trade outcomes.

    Key features:
    1. Records feature vectors for each analysis
    2. Tracks trade outcomes
    3. Adjusts pillar weights based on what works
    4. Provides attribution (which pillars predicted correctly)
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # SQLite for trade records
        self.db_path = self.data_dir / "adaptive_learning.db"
        self._init_db()

        # Current weights (start with equal)
        self.weights = self._load_weights()

        # Learning parameters
        self.learning_rate = 0.05      # How fast to adjust weights
        self.min_trades_for_update = 20  # Minimum trades before adjusting
        self.max_weight = 0.40          # No pillar can exceed 40%
        self.min_weight = 0.10          # No pillar below 10%

        logger.info(f"AdaptiveScorer initialized. Current weights: {self.weights.to_dict()}")

    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Trade records table
        c.execute('''
            CREATE TABLE IF NOT EXISTS trade_records (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT,
                entry_date TEXT,
                entry_price REAL,
                entry_features TEXT,
                entry_score REAL,
                entry_decision TEXT,
                pillar_weights TEXT,
                exit_date TEXT,
                exit_price REAL,
                exit_reason TEXT,
                pnl_pct REAL,
                outcome_type TEXT,
                hold_days INTEGER,
                pillar_attribution TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Weights history table
        c.execute('''
            CREATE TABLE IF NOT EXISTS weights_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                technical REAL,
                fundamental REAL,
                sentiment REAL,
                news REAL,
                based_on_trades INTEGER,
                reason TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Feature vectors for clustering/similarity
        c.execute('''
            CREATE TABLE IF NOT EXISTS feature_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                features BLOB,
                score REAL,
                outcome_pct REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def _load_weights(self) -> PillarWeights:
        """Load current weights from file"""
        weights_file = self.data_dir / "pillar_weights.json"
        if weights_file.exists():
            try:
                with open(weights_file) as f:
                    data = json.load(f)
                return PillarWeights.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load weights: {e}")
        return PillarWeights()

    def _save_weights(self):
        """Save current weights to file"""
        weights_file = self.data_dir / "pillar_weights.json"
        with open(weights_file, 'w') as f:
            json.dump(self.weights.to_dict(), f, indent=2)

        # Also save to history
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO weights_history
            (technical, fundamental, sentiment, news, based_on_trades, reason)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            self.weights.technical, self.weights.fundamental,
            self.weights.sentiment, self.weights.news,
            self.weights.based_on_trades, "periodic_update"
        ))
        conn.commit()
        conn.close()

    # =========================================================================
    # SCORING
    # =========================================================================

    def calculate_score(
        self,
        pillar_scores: Dict[str, float],
        features: Optional[FeatureVector] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate weighted score using learned weights.

        Args:
            pillar_scores: Raw pillar scores (-100 to +100)
            features: Optional feature vector for logging

        Returns:
            (total_score, weight_contributions)
        """
        # Apply learned weights
        contributions = {}
        for pillar, score in pillar_scores.items():
            weight = getattr(self.weights, pillar, 0.25)
            contributions[pillar] = score * weight

        total = sum(contributions.values())

        # Log feature vector if provided
        if features:
            self._log_features(features, total)

        return total, contributions

    def _log_features(self, features: FeatureVector, score: float):
        """Store feature vector for future learning"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO feature_vectors (symbol, features, score)
            VALUES (?, ?, ?)
        ''', ("unknown", features.to_array().tobytes(), score))
        conn.commit()
        conn.close()

    # =========================================================================
    # TRADE RECORDING
    # =========================================================================

    def record_entry(
        self,
        trade_id: str,
        symbol: str,
        entry_price: float,
        pillar_scores: Dict[str, float],
        entry_score: float,
        decision: str,
        features: Optional[FeatureVector] = None
    ) -> TradeRecord:
        """
        Record trade entry for later learning.

        Args:
            trade_id: Unique identifier
            symbol: Stock symbol
            entry_price: Entry price
            pillar_scores: Individual pillar scores
            entry_score: Combined score
            decision: BUY, STRONG_BUY, etc.
            features: Optional feature vector
        """
        if features is None:
            # Create basic feature vector from pillar scores
            features = FeatureVector(
                technical=pillar_scores.get('technical', 0) / 100,
                fundamental=pillar_scores.get('fundamental', 0) / 100,
                sentiment=pillar_scores.get('sentiment', 0) / 100,
                news=pillar_scores.get('news', 0) / 100
            )

        record = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            entry_date=datetime.now().isoformat(),
            entry_price=entry_price,
            entry_features=features,
            entry_score=entry_score,
            entry_decision=decision,
            pillar_weights=self.weights.to_dict()
        )

        # Save to database
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO trade_records
            (trade_id, symbol, entry_date, entry_price, entry_features,
             entry_score, entry_decision, pillar_weights)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.trade_id, record.symbol, record.entry_date,
            record.entry_price, json.dumps(features.to_dict()),
            record.entry_score, record.entry_decision,
            json.dumps(record.pillar_weights)
        ))
        conn.commit()
        conn.close()

        logger.info(f"[ADAPTIVE] Recorded entry for {symbol} @ {entry_price:.2f}")
        return record

    def record_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str
    ) -> Optional[TradeRecord]:
        """
        Record trade exit and calculate outcome.
        Triggers learning if enough data.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Get entry record
        c.execute('SELECT * FROM trade_records WHERE trade_id = ?', (trade_id,))
        row = c.fetchone()

        if not row:
            logger.warning(f"Trade {trade_id} not found")
            conn.close()
            return None

        # Calculate PnL
        entry_price = row[3]  # entry_price column
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100

        # Determine outcome type
        if pnl_pct > 15:
            outcome = TradeOutcomeType.BIG_WIN.value
        elif pnl_pct > 5:
            outcome = TradeOutcomeType.WIN.value
        elif pnl_pct > 0:
            outcome = TradeOutcomeType.SMALL_WIN.value
        elif pnl_pct > -5:
            outcome = TradeOutcomeType.SMALL_LOSS.value
        elif pnl_pct > -15:
            outcome = TradeOutcomeType.LOSS.value
        else:
            outcome = TradeOutcomeType.BIG_LOSS.value

        if exit_reason == "stop_loss":
            outcome = TradeOutcomeType.STOP_LOSS.value

        # Calculate hold days
        entry_date = datetime.fromisoformat(row[2])
        hold_days = (datetime.now() - entry_date).days

        # Calculate pillar attribution
        entry_features = json.loads(row[4])
        pillar_weights = json.loads(row[7])
        attribution = self._calculate_attribution(entry_features, pnl_pct, pillar_weights)

        # Update record
        c.execute('''
            UPDATE trade_records SET
                exit_date = ?,
                exit_price = ?,
                exit_reason = ?,
                pnl_pct = ?,
                outcome_type = ?,
                hold_days = ?,
                pillar_attribution = ?
            WHERE trade_id = ?
        ''', (
            datetime.now().isoformat(), exit_price, exit_reason,
            pnl_pct, outcome, hold_days, json.dumps(attribution),
            trade_id
        ))
        conn.commit()
        conn.close()

        logger.info(f"[ADAPTIVE] Recorded exit for trade {trade_id}: {pnl_pct:+.2f}% ({outcome})")

        # Trigger learning check
        self._maybe_update_weights()

        return None  # Could return updated record if needed

    def _calculate_attribution(
        self,
        features: Dict,
        pnl_pct: float,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate how much each pillar contributed to the outcome.

        Positive attribution = pillar correctly predicted direction
        Negative attribution = pillar incorrectly predicted direction
        """
        attribution = {}
        outcome_positive = pnl_pct > 0

        for pillar in ['technical', 'fundamental', 'sentiment', 'news']:
            pillar_score = features.get(pillar, 0)
            pillar_predicted_positive = pillar_score > 0

            # If pillar agreed with outcome, positive attribution
            if pillar_predicted_positive == outcome_positive:
                attribution[pillar] = abs(pillar_score) * weights.get(pillar, 0.25)
            else:
                attribution[pillar] = -abs(pillar_score) * weights.get(pillar, 0.25)

        return attribution

    # =========================================================================
    # LEARNING
    # =========================================================================

    def _maybe_update_weights(self):
        """Check if we should update weights based on recent trades"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Count completed trades since last update
        c.execute('''
            SELECT COUNT(*) FROM trade_records
            WHERE pnl_pct IS NOT NULL
        ''')
        total_trades = c.fetchone()[0]

        # Check if enough new trades since last update
        if total_trades - self.weights.based_on_trades < self.min_trades_for_update:
            conn.close()
            return

        logger.info(f"[ADAPTIVE] Updating weights based on {total_trades} trades...")

        # Get recent attribution data
        c.execute('''
            SELECT pillar_attribution, pnl_pct FROM trade_records
            WHERE pillar_attribution IS NOT NULL
            ORDER BY exit_date DESC
            LIMIT 100
        ''')
        rows = c.fetchall()
        conn.close()

        if len(rows) < self.min_trades_for_update:
            return

        # Calculate average attribution per pillar
        pillar_performance = {p: [] for p in ['technical', 'fundamental', 'sentiment', 'news']}

        for attr_json, pnl in rows:
            attribution = json.loads(attr_json)
            for pillar, value in attribution.items():
                if pillar in pillar_performance:
                    pillar_performance[pillar].append(value)

        # Calculate new weights based on performance
        avg_performance = {}
        for pillar, values in pillar_performance.items():
            if values:
                avg_performance[pillar] = np.mean(values)
            else:
                avg_performance[pillar] = 0

        # Normalize and apply learning rate
        total_perf = sum(max(0, p) for p in avg_performance.values())
        if total_perf > 0:
            for pillar in ['technical', 'fundamental', 'sentiment', 'news']:
                current_weight = getattr(self.weights, pillar)
                target_weight = max(0, avg_performance[pillar]) / total_perf

                # Gradual adjustment
                new_weight = current_weight + self.learning_rate * (target_weight - current_weight)

                # Clamp to bounds
                new_weight = max(self.min_weight, min(self.max_weight, new_weight))
                setattr(self.weights, pillar, new_weight)

        # Normalize weights
        self.weights.normalize()
        self.weights.last_updated = datetime.now().isoformat()
        self.weights.update_count += 1
        self.weights.based_on_trades = total_trades

        # Save
        self._save_weights()

        logger.info(f"[ADAPTIVE] Updated weights: {self.weights.to_dict()}")

    # =========================================================================
    # ANALYSIS & REPORTING
    # =========================================================================

    def get_performance_by_pillar(self) -> Dict[str, Dict]:
        """Analyze which pillars predict best"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            SELECT entry_features, pnl_pct FROM trade_records
            WHERE pnl_pct IS NOT NULL
        ''')
        rows = c.fetchall()
        conn.close()

        if not rows:
            return {}

        pillar_stats = {p: {'wins': 0, 'losses': 0, 'total_pnl': 0, 'correct': 0}
                       for p in ['technical', 'fundamental', 'sentiment', 'news']}

        for features_json, pnl in rows:
            features = json.loads(features_json)
            outcome_positive = pnl > 0

            for pillar in pillar_stats:
                pillar_score = features.get(pillar, 0)
                pillar_predicted_positive = pillar_score > 0

                if pillar_predicted_positive == outcome_positive:
                    pillar_stats[pillar]['correct'] += 1

                if pnl > 0:
                    pillar_stats[pillar]['wins'] += 1
                else:
                    pillar_stats[pillar]['losses'] += 1

                pillar_stats[pillar]['total_pnl'] += pnl

        # Calculate accuracy
        total_trades = len(rows)
        for pillar in pillar_stats:
            pillar_stats[pillar]['accuracy'] = pillar_stats[pillar]['correct'] / total_trades if total_trades > 0 else 0
            pillar_stats[pillar]['avg_pnl'] = pillar_stats[pillar]['total_pnl'] / total_trades if total_trades > 0 else 0

        return pillar_stats

    def get_weights_history(self, limit: int = 10) -> List[Dict]:
        """Get history of weight changes"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT technical, fundamental, sentiment, news, based_on_trades, created_at
            FROM weights_history
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        rows = c.fetchall()
        conn.close()

        return [
            {
                'technical': r[0], 'fundamental': r[1],
                'sentiment': r[2], 'news': r[3],
                'trades': r[4], 'date': r[5]
            }
            for r in rows
        ]

    def get_similar_trades(self, features: FeatureVector, limit: int = 5) -> List[Dict]:
        """Find similar historical trades for insight"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            SELECT symbol, entry_features, entry_score, pnl_pct, outcome_type
            FROM trade_records
            WHERE pnl_pct IS NOT NULL
        ''')
        rows = c.fetchall()
        conn.close()

        if not rows:
            return []

        # Calculate similarity (cosine distance)
        target = features.to_array()
        similarities = []

        for symbol, feat_json, score, pnl, outcome in rows:
            hist_features = FeatureVector(**json.loads(feat_json))
            hist_array = hist_features.to_array()

            # Cosine similarity
            dot = np.dot(target, hist_array)
            norm = np.linalg.norm(target) * np.linalg.norm(hist_array)
            similarity = dot / norm if norm > 0 else 0

            similarities.append({
                'symbol': symbol,
                'similarity': similarity,
                'score': score,
                'pnl': pnl,
                'outcome': outcome
            })

        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:limit]


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_adaptive_scorer: Optional[AdaptiveScorer] = None

def get_adaptive_scorer() -> AdaptiveScorer:
    """Get singleton instance"""
    global _adaptive_scorer
    if _adaptive_scorer is None:
        _adaptive_scorer = AdaptiveScorer()
    return _adaptive_scorer
