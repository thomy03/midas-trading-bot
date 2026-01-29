"""
Analysis Store - Persistent storage for analysis results and symbol journeys.

Provides:
1. Symbol Journey persistence (discovery → analysis → decision → outcome)
2. Reasoning results with full pillar breakdown
3. Historical analysis lookup for trend detection
4. Links between related analyses (same symbol across time)
"""

import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Storage paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
JOURNEYS_FILE = DATA_DIR / "symbol_journeys.json"
ANALYSIS_DB = DATA_DIR / "analysis_history.db"


@dataclass
class JourneyStep:
    """A single step in a symbol's journey"""
    step: str  # discovery, analysis, signal, buy, sell, rejected
    timestamp: str
    title: str
    reasoning: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'JourneyStep':
        return cls(**d)


@dataclass
class SymbolJourney:
    """Complete journey for a symbol"""
    symbol: str
    created_at: str
    updated_at: str
    current_status: str  # watching, analyzed, signal, bought, sold, rejected
    current_score: float
    journey: List[JourneyStep] = field(default_factory=list)

    # Links to related analyses
    previous_analysis_id: Optional[str] = None
    next_analysis_id: Optional[str] = None

    def add_step(self, step: JourneyStep):
        """Add a step and update timestamps"""
        self.journey.append(step)
        self.updated_at = datetime.now().isoformat()
        self.current_status = step.step
        if 'total' in step.data:
            self.current_score = step.data['total']
        elif 'score' in step.data:
            self.current_score = step.data['score']

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'current_status': self.current_status,
            'current_score': self.current_score,
            'journey': [s.to_dict() for s in self.journey],
            'previous_analysis_id': self.previous_analysis_id,
            'next_analysis_id': self.next_analysis_id
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'SymbolJourney':
        journey_steps = [JourneyStep.from_dict(s) for s in d.get('journey', [])]
        return cls(
            symbol=d['symbol'],
            created_at=d['created_at'],
            updated_at=d['updated_at'],
            current_status=d['current_status'],
            current_score=d['current_score'],
            journey=journey_steps,
            previous_analysis_id=d.get('previous_analysis_id'),
            next_analysis_id=d.get('next_analysis_id')
        )


class AnalysisStore:
    """
    Persistent store for analysis results and symbol journeys.

    Uses JSON for journeys (frequent access, small data) and
    SQLite for detailed analysis history (queryable, historical).
    """

    def __init__(self):
        self._journeys: Dict[str, SymbolJourney] = {}
        self._ensure_data_dir()
        self._init_database()
        self._load_journeys()

    def _ensure_data_dir(self):
        """Ensure data directory exists"""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize SQLite database for analysis history"""
        with self._get_db() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    total_score REAL,
                    decision TEXT,
                    technical_score REAL,
                    fundamental_score REAL,
                    sentiment_score REAL,
                    news_score REAL,
                    technical_reasoning TEXT,
                    fundamental_reasoning TEXT,
                    sentiment_reasoning TEXT,
                    news_reasoning TEXT,
                    key_factors TEXT,
                    risk_factors TEXT,
                    market_regime TEXT,
                    full_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_symbol_timestamp
                ON analysis_history(symbol, timestamp)
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS journey_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    @contextmanager
    def _get_db(self):
        """Get database connection"""
        conn = sqlite3.connect(str(ANALYSIS_DB))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _load_journeys(self):
        """Load journeys from JSON file"""
        if JOURNEYS_FILE.exists():
            try:
                with open(JOURNEYS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for symbol, journey_data in data.items():
                        self._journeys[symbol] = SymbolJourney.from_dict(journey_data)
                logger.info(f"Loaded {len(self._journeys)} symbol journeys")
            except Exception as e:
                logger.error(f"Failed to load journeys: {e}")
                self._journeys = {}
        else:
            logger.info("No existing journeys file, starting fresh")

    def _save_journeys(self):
        """Save journeys to JSON file"""
        try:
            data = {symbol: journey.to_dict() for symbol, journey in self._journeys.items()}
            with open(JOURNEYS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            logger.debug(f"Saved {len(self._journeys)} journeys to disk")
        except Exception as e:
            logger.error(f"Failed to save journeys: {e}")

    # ========== JOURNEY MANAGEMENT ==========

    def add_journey_step(
        self,
        symbol: str,
        step: str,
        title: str,
        reasoning: str,
        data: Optional[Dict] = None
    ) -> SymbolJourney:
        """
        Add a step to a symbol's journey.
        Creates journey if doesn't exist.

        Args:
            symbol: Stock symbol
            step: Step type (discovery, analysis, signal, buy, sell, rejected)
            title: Short title for the step
            reasoning: Detailed reasoning text
            data: Additional data (scores, factors, etc.)

        Returns:
            Updated SymbolJourney
        """
        now = datetime.now().isoformat()

        if symbol not in self._journeys:
            # Create new journey
            self._journeys[symbol] = SymbolJourney(
                symbol=symbol,
                created_at=now,
                updated_at=now,
                current_status=step,
                current_score=data.get('total', data.get('score', 50)) if data else 50
            )
            # Link to previous analysis if exists
            prev = self.get_previous_analysis(symbol)
            if prev:
                self._journeys[symbol].previous_analysis_id = str(prev['id'])

        journey_step = JourneyStep(
            step=step,
            timestamp=now,
            title=title,
            reasoning=reasoning,
            data=data or {}
        )

        self._journeys[symbol].add_step(journey_step)

        # Persist to disk
        self._save_journeys()

        # Also log to database for historical queries
        self._log_event(symbol, step, {
            'title': title,
            'reasoning': reasoning[:500],  # Truncate for DB
            'score': data.get('total', data.get('score')) if data else None
        })

        return self._journeys[symbol]

    def get_journey(self, symbol: str) -> Optional[SymbolJourney]:
        """Get journey for a symbol"""
        return self._journeys.get(symbol)

    def get_all_journeys(self) -> Dict[str, SymbolJourney]:
        """Get all journeys"""
        return self._journeys.copy()

    def get_journeys_sorted(self, max_count: int = 50) -> List[SymbolJourney]:
        """Get journeys sorted by last update (most recent first)"""
        journeys = list(self._journeys.values())
        journeys.sort(key=lambda j: j.updated_at, reverse=True)
        return journeys[:max_count]

    def clear_old_journeys(self, days: int = 30):
        """Remove journeys older than N days"""
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()

        to_remove = [
            symbol for symbol, journey in self._journeys.items()
            if journey.updated_at < cutoff_str
        ]

        for symbol in to_remove:
            del self._journeys[symbol]

        if to_remove:
            self._save_journeys()
            logger.info(f"Removed {len(to_remove)} old journeys")

    # ========== ANALYSIS HISTORY ==========

    def save_analysis(
        self,
        symbol: str,
        reasoning_result: Any,  # ReasoningResult from reasoning_engine
        market_regime: Optional[str] = None
    ):
        """
        Save detailed analysis result to database.

        Links to the current journey and allows historical lookups.
        """
        try:
            data = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'total_score': getattr(reasoning_result, 'total_score', 0),
                'decision': getattr(reasoning_result, 'decision', None),
                'technical_score': 0,
                'fundamental_score': 0,
                'sentiment_score': 0,
                'news_score': 0,
                'technical_reasoning': '',
                'fundamental_reasoning': '',
                'sentiment_reasoning': '',
                'news_reasoning': '',
                'key_factors': [],
                'risk_factors': [],
                'market_regime': market_regime,
                'full_data': {}
            }

            # Extract pillar data
            if hasattr(reasoning_result, 'technical_score') and reasoning_result.technical_score:
                data['technical_score'] = reasoning_result.technical_score.score
                data['technical_reasoning'] = reasoning_result.technical_score.reasoning
            if hasattr(reasoning_result, 'fundamental_score') and reasoning_result.fundamental_score:
                data['fundamental_score'] = reasoning_result.fundamental_score.score
                data['fundamental_reasoning'] = reasoning_result.fundamental_score.reasoning
            if hasattr(reasoning_result, 'sentiment_score') and reasoning_result.sentiment_score:
                data['sentiment_score'] = reasoning_result.sentiment_score.score
                data['sentiment_reasoning'] = reasoning_result.sentiment_score.reasoning
            if hasattr(reasoning_result, 'news_score') and reasoning_result.news_score:
                data['news_score'] = reasoning_result.news_score.score
                data['news_reasoning'] = reasoning_result.news_score.reasoning

            if hasattr(reasoning_result, 'key_factors'):
                data['key_factors'] = reasoning_result.key_factors[:10]
            if hasattr(reasoning_result, 'risk_factors'):
                data['risk_factors'] = reasoning_result.risk_factors[:10]

            # Store full result for reference
            if hasattr(reasoning_result, 'to_dict'):
                data['full_data'] = reasoning_result.to_dict()

            # Handle decision enum
            if data['decision'] and hasattr(data['decision'], 'value'):
                data['decision'] = data['decision'].value

            with self._get_db() as conn:
                conn.execute('''
                    INSERT INTO analysis_history
                    (symbol, timestamp, total_score, decision,
                     technical_score, fundamental_score, sentiment_score, news_score,
                     technical_reasoning, fundamental_reasoning, sentiment_reasoning, news_reasoning,
                     key_factors, risk_factors, market_regime, full_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['symbol'],
                    data['timestamp'],
                    data['total_score'],
                    data['decision'],
                    data['technical_score'],
                    data['fundamental_score'],
                    data['sentiment_score'],
                    data['news_score'],
                    data['technical_reasoning'],
                    data['fundamental_reasoning'],
                    data['sentiment_reasoning'],
                    data['news_reasoning'],
                    json.dumps(data['key_factors'], default=str),
                    json.dumps(data['risk_factors'], default=str),
                    data['market_regime'],
                    json.dumps(data['full_data'], default=str)
                ))
                conn.commit()
                logger.info(f"Saved analysis for {symbol} (score: {data['total_score']:.1f})")

        except Exception as e:
            logger.error(f"Failed to save analysis for {symbol}: {e}")

    def get_previous_analysis(self, symbol: str, days: int = 30) -> Optional[Dict]:
        """Get the most recent previous analysis for a symbol"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with self._get_db() as conn:
            row = conn.execute('''
                SELECT * FROM analysis_history
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (symbol, cutoff)).fetchone()

            if row:
                return dict(row)
        return None

    def was_recently_analyzed(self, symbol: str, hours: int = 1) -> bool:
        """
        Check if a symbol was analyzed within the last N hours.
        Used to avoid redundant re-analysis.

        Args:
            symbol: Stock symbol
            hours: Cooldown period in hours

        Returns:
            True if symbol was analyzed recently and should be skipped
        """
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        with self._get_db() as conn:
            row = conn.execute('''
                SELECT COUNT(*) as count FROM analysis_history
                WHERE symbol = ? AND timestamp > ?
            ''', (symbol, cutoff)).fetchone()

            return row['count'] > 0 if row else False

    def get_unanalyzed_symbols(self, symbols: List[str], hours: int = 1) -> List[str]:
        """
        Filter a list of symbols to only those not recently analyzed.

        Args:
            symbols: List of symbols to filter
            hours: Cooldown period in hours

        Returns:
            Symbols that haven't been analyzed within the cooldown period
        """
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        with self._get_db() as conn:
            # Get recently analyzed symbols
            placeholders = ','.join('?' for _ in symbols)
            rows = conn.execute(f'''
                SELECT DISTINCT symbol FROM analysis_history
                WHERE symbol IN ({placeholders}) AND timestamp > ?
            ''', (*symbols, cutoff)).fetchall()

            recently_analyzed = {row['symbol'] for row in rows}

        return [s for s in symbols if s not in recently_analyzed]

    def get_symbols_needing_reanalysis(self, hours: int = 24) -> List[str]:
        """
        Get symbols that were analyzed more than N hours ago and need refresh.

        Args:
            hours: How old the analysis should be before needing refresh

        Returns:
            List of symbols that need fresh analysis
        """
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        with self._get_db() as conn:
            rows = conn.execute('''
                SELECT symbol, MAX(timestamp) as last_analyzed
                FROM analysis_history
                GROUP BY symbol
                HAVING MAX(timestamp) < ?
                ORDER BY MAX(timestamp) ASC
                LIMIT 30
            ''', (cutoff,)).fetchall()

            return [row['symbol'] for row in rows]

    def get_analysis_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get analysis history for a symbol"""
        with self._get_db() as conn:
            rows = conn.execute('''
                SELECT * FROM analysis_history
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (symbol, limit)).fetchall()

            return [dict(row) for row in rows]

    def get_score_trend(self, symbol: str, days: int = 30) -> List[Dict]:
        """Get score trend for a symbol over time"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with self._get_db() as conn:
            rows = conn.execute('''
                SELECT timestamp, total_score, decision,
                       technical_score, fundamental_score, sentiment_score, news_score
                FROM analysis_history
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp ASC
            ''', (symbol, cutoff)).fetchall()

            return [dict(row) for row in rows]

    def compare_analyses(self, symbol: str) -> Optional[Dict]:
        """
        Compare current analysis with previous one.
        Returns delta information for each pillar.
        """
        history = self.get_analysis_history(symbol, limit=2)

        if len(history) < 2:
            return None

        current = history[0]
        previous = history[1]

        return {
            'symbol': symbol,
            'current_timestamp': current['timestamp'],
            'previous_timestamp': previous['timestamp'],
            'total_delta': current['total_score'] - previous['total_score'],
            'technical_delta': current['technical_score'] - previous['technical_score'],
            'fundamental_delta': current['fundamental_score'] - previous['fundamental_score'],
            'sentiment_delta': current['sentiment_score'] - previous['sentiment_score'],
            'news_delta': current['news_score'] - previous['news_score'],
            'decision_changed': current['decision'] != previous['decision'],
            'current_decision': current['decision'],
            'previous_decision': previous['decision']
        }

    def _log_event(self, symbol: str, event_type: str, data: Dict):
        """Log a journey event to database"""
        try:
            with self._get_db() as conn:
                conn.execute('''
                    INSERT INTO journey_events (symbol, event_type, event_data)
                    VALUES (?, ?, ?)
                ''', (symbol, event_type, json.dumps(data, default=str)))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to log event: {e}")

    def get_symbol_stats(self, symbol: str) -> Dict:
        """Get statistics for a symbol from historical analyses"""
        history = self.get_analysis_history(symbol, limit=100)

        if not history:
            return {'symbol': symbol, 'analysis_count': 0}

        scores = [h['total_score'] for h in history if h['total_score']]
        decisions = [h['decision'] for h in history if h['decision']]

        return {
            'symbol': symbol,
            'analysis_count': len(history),
            'avg_score': sum(scores) / len(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'buy_signals': sum(1 for d in decisions if d in ['strong_buy', 'buy']),
            'hold_signals': sum(1 for d in decisions if d == 'hold'),
            'sell_signals': sum(1 for d in decisions if d in ['strong_sell', 'sell']),
            'first_analysis': history[-1]['timestamp'] if history else None,
            'last_analysis': history[0]['timestamp'] if history else None
        }


# Singleton instance
_store_instance: Optional[AnalysisStore] = None


def get_analysis_store() -> AnalysisStore:
    """Get or create the AnalysisStore singleton"""
    global _store_instance
    if _store_instance is None:
        _store_instance = AnalysisStore()
    return _store_instance
