"""
Smart Signal Learner V6 - Auto-learning weak signals detection.

Learns to detect:
1. Key person mentions and their market impact
2. Economic events and correlations
3. Social sentiment patterns that precede moves
4. Cross-asset correlations

Author: Jarvis for Thomas
Created: 2026-02-05
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import sqlite3
import hashlib

logger = logging.getLogger(__name__)

# Data paths
DATA_DIR = Path("/app/data")
SIGNALS_DB = DATA_DIR / "smart_signals.db"
LEARNED_SIGNALS_FILE = DATA_DIR / "learned_signals.json"
CORRELATION_MEMORY_FILE = DATA_DIR / "correlation_memory.json"


# =============================================================================
# INFLUENCE DETECTION - Dynamically learned, NOT hardcoded
# =============================================================================

# NO hardcoded influencers! The system discovers who has impact.
# See dynamic_influence_learner.py for the auto-discovery system.

try:
    from src.learning.dynamic_influence_learner import get_influence_learner, DynamicInfluenceLearner
    DYNAMIC_INFLUENCE_AVAILABLE = True
except ImportError:
    DYNAMIC_INFLUENCE_AVAILABLE = False
    DynamicInfluenceLearner = None


# =============================================================================
# ECONOMIC EVENTS - Macro events that move markets
# =============================================================================

ECONOMIC_EVENTS = {
    "cpi": {
        "name": "Consumer Price Index",
        "frequency": "monthly",
        "typical_impact": {
            "hot": {"direction": "bearish", "sectors": ["growth", "tech"], "magnitude": 2.0},
            "cool": {"direction": "bullish", "sectors": ["growth", "tech"], "magnitude": 1.5},
            "inline": {"direction": "neutral", "magnitude": 0.5}
        },
        "affected_symbols": ["SPY", "QQQ", "TLT", "ARKK"]
    },
    "nfp": {
        "name": "Non-Farm Payrolls",
        "frequency": "monthly",
        "typical_impact": {
            "strong": {"direction": "mixed", "sectors": ["banks", "consumer"], "magnitude": 1.5},
            "weak": {"direction": "bullish", "sectors": ["growth"], "magnitude": 1.0}  # Fed pivot hope
        },
        "affected_symbols": ["SPY", "XLF", "XLY"]
    },
    "fomc": {
        "name": "Federal Reserve Meeting",
        "frequency": "6-8 weeks",
        "typical_impact": {
            "hawkish": {"direction": "bearish", "sectors": ["growth", "real_estate"], "magnitude": 2.5},
            "dovish": {"direction": "bullish", "sectors": ["growth", "tech"], "magnitude": 3.0},
            "neutral": {"direction": "volatile", "magnitude": 1.0}
        },
        "affected_symbols": ["SPY", "QQQ", "TLT", "XLF", "XLRE"]
    },
    "earnings": {
        "name": "Earnings Reports",
        "frequency": "quarterly",
        "typical_impact": {
            "beat": {"direction": "bullish", "magnitude": 3.0},
            "miss": {"direction": "bearish", "magnitude": 4.0},  # Miss hurts more
            "guidance_up": {"direction": "bullish", "magnitude": 5.0},
            "guidance_down": {"direction": "bearish", "magnitude": 6.0}
        }
    },
    "pmi": {
        "name": "Purchasing Managers Index",
        "frequency": "monthly",
        "typical_impact": {
            "expansion": {"direction": "bullish", "sectors": ["industrials"], "magnitude": 1.0},
            "contraction": {"direction": "bearish", "sectors": ["industrials"], "magnitude": 1.5}
        },
        "affected_symbols": ["XLI", "CAT", "DE"]
    }
}


# =============================================================================
# SIGNAL PATTERNS - Learned correlations
# =============================================================================

@dataclass
class LearnedSignal:
    """A signal pattern learned from historical data"""
    signal_id: str
    signal_type: str  # "mention", "event", "pattern", "correlation"
    trigger: str      # What triggers this signal
    
    # Impact statistics
    hit_count: int = 0
    success_count: int = 0
    avg_impact_pct: float = 0.0
    avg_reaction_hours: float = 24.0
    
    # Affected assets
    primary_symbols: List[str] = field(default_factory=list)
    sector_impact: Dict[str, float] = field(default_factory=dict)
    
    # Confidence
    confidence: float = 0.5
    last_triggered: str = ""
    last_updated: str = ""
    
    # Direction learned
    typical_direction: str = "unknown"  # bullish, bearish, volatile
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'LearnedSignal':
        return cls(**d)


@dataclass
class SignalEvent:
    """A detected signal event"""
    event_id: str
    signal_type: str
    trigger: str
    detected_at: str
    source: str  # twitter, news, economic_calendar
    
    # Context
    raw_text: Optional[str] = None
    mentioned_symbols: List[str] = field(default_factory=list)
    sentiment: float = 0.0  # -1 to 1
    
    # Outcome tracking
    outcome_tracked: bool = False
    actual_impact_pct: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SmartSignalLearner:
    """
    Learns to detect and predict impact of weak signals.
    
    Key features:
    1. Tracks mentions of key influencers
    2. Monitors economic events
    3. Learns correlations between signals and market moves
    4. Automatically adjusts confidence based on outcomes
    """
    
    def __init__(self):
        self._init_db()
        self.learned_signals = self._load_learned_signals()
        self.correlation_memory = self._load_correlation_memory()
        
        # Runtime tracking
        self.pending_signals: List[SignalEvent] = []
        
        logger.info(f"[SIGNAL LEARNER] Initialized with {len(self.learned_signals)} learned signals")
    
    def _init_db(self):
        """Initialize SQLite database for signal history"""
        SIGNALS_DB.parent.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(SIGNALS_DB)
        c = conn.cursor()
        
        # Signal events table
        c.execute('''
            CREATE TABLE IF NOT EXISTS signal_events (
                event_id TEXT PRIMARY KEY,
                signal_type TEXT,
                trigger TEXT,
                detected_at TEXT,
                source TEXT,
                raw_text TEXT,
                mentioned_symbols TEXT,
                sentiment REAL,
                outcome_tracked INTEGER DEFAULT 0,
                actual_impact_pct REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Correlation history
        c.execute('''
            CREATE TABLE IF NOT EXISTS correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_a TEXT,
                signal_b TEXT,
                correlation_value REAL,
                sample_size INTEGER,
                last_updated TEXT
            )
        ''')
        
        # Market context at signal time
        c.execute('''
            CREATE TABLE IF NOT EXISTS signal_context (
                event_id TEXT,
                vix_level REAL,
                spy_trend TEXT,
                market_regime TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (event_id) REFERENCES signal_events(event_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_learned_signals(self) -> Dict[str, LearnedSignal]:
        """Load learned signal patterns"""
        if LEARNED_SIGNALS_FILE.exists():
            try:
                with open(LEARNED_SIGNALS_FILE) as f:
                    data = json.load(f)
                return {k: LearnedSignal.from_dict(v) for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Failed to load learned signals: {e}")
        
        # Initialize with known influencers and events
        signals = {}
        
        # Add influencer signals
        # V6.1: Dynamic influencer discovery - no hardcoded influencers
        # Influencers are discovered automatically by dynamic_influence_learner.py
        # based on real correlation data from market moves
        
        # Add economic event signals
        # V6.1: Economic events also moved to dynamic learning
        # Events are tracked by the daily learner based on actual market impact
        
        self._save_learned_signals(signals)
        return signals
        return signals
    
    def _save_learned_signals(self, signals: Optional[Dict] = None):
        """Save learned signals to file"""
        if signals is None:
            signals = self.learned_signals
        
        with open(LEARNED_SIGNALS_FILE, 'w') as f:
            json.dump({k: v.to_dict() for k, v in signals.items()}, f, indent=2)
    
    def _load_correlation_memory(self) -> Dict:
        """Load correlation memory"""
        if CORRELATION_MEMORY_FILE.exists():
            try:
                with open(CORRELATION_MEMORY_FILE) as f:
                    return json.load(f)
            except:
                pass
        return {"signal_pairs": {}, "event_sequences": []}
    
    def _save_correlation_memory(self):
        """Save correlation memory"""
        with open(CORRELATION_MEMORY_FILE, 'w') as f:
            json.dump(self.correlation_memory, f, indent=2)
    
    # =========================================================================
    # SIGNAL DETECTION
    # =========================================================================
    
    def detect_influencer_mention(self, text: str, source: str = "twitter") -> List[SignalEvent]:
        """
        Scan text for mentions using DYNAMIC influence detection.
        
        NO hardcoded list - discovers influencers automatically!
        
        Args:
            text: Text to scan (tweet, news headline, etc.)
            source: Where the text came from
            
        Returns:
            List of detected signal events
        """
        detected = []
        
        # Use dynamic influence learner
        if DYNAMIC_INFLUENCE_AVAILABLE:
            try:
                influence_learner = get_influence_learner()
                
                # Extract symbols from text
                mentioned = self._extract_symbols(text)
                
                # Estimate sentiment
                sentiment = self._estimate_sentiment(text)
                
                # Record the mention (for learning)
                influence_learner.record_mention(text, mentioned, source, sentiment)
                
                # Extract entities (names, handles, etc.)
                entities = influence_learner.extract_entities(text, source)
                
                for entity_info in entities:
                    entity = entity_info['entity']
                    
                    # Check if this is a KNOWN influential entity
                    if entity in influence_learner.influencers:
                        inf_data = influence_learner.influencers[entity]
                        
                        # Only signal if they have proven influence
                        if inf_data.get('influence_score', 0) > 0.5:
                            event_id = hashlib.md5(f"{entity}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
                            
                            event = SignalEvent(
                                event_id=event_id,
                                signal_type="discovered_influencer",
                                trigger=entity,
                                detected_at=datetime.now().isoformat(),
                                source=source,
                                raw_text=text[:500],
                                mentioned_symbols=mentioned or inf_data.get('symbols', []),
                                sentiment=sentiment
                            )
                            
                            detected.append(event)
                            self._record_signal_event(event)
                            
                            logger.info(f"[SIGNAL] Discovered influencer {entity}: "
                                       f"score={inf_data.get('influence_score', 0):.2f}, "
                                       f"sentiment={sentiment:.2f}")
                    else:
                        # Unknown entity - still record for future learning
                        # The system will discover if they have influence later
                        pass
                        
            except Exception as e:
                logger.warning(f"[SIGNAL] Dynamic influence error: {e}")
        
        return detected
    
    def detect_economic_event(self, event_type: str, outcome: str, 
                             value: Optional[float] = None) -> Optional[SignalEvent]:
        """
        Record an economic event detection.
        
        Args:
            event_type: Type of event (cpi, nfp, fomc, etc.)
            outcome: Outcome classification (hot, cool, beat, miss, etc.)
            value: Optional numeric value
            
        Returns:
            Signal event if recognized
        """
        if event_type not in ECONOMIC_EVENTS:
            logger.warning(f"Unknown economic event type: {event_type}")
            return None
        
        event_info = ECONOMIC_EVENTS[event_type]
        impact_info = event_info.get("typical_impact", {}).get(outcome, {})
        
        event_id = hashlib.md5(f"{event_type}_{outcome}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        event = SignalEvent(
            event_id=event_id,
            signal_type="event",
            trigger=f"{event_type}_{outcome}",
            detected_at=datetime.now().isoformat(),
            source="economic_calendar",
            raw_text=f"{event_info['name']}: {outcome}" + (f" ({value})" if value else ""),
            mentioned_symbols=event_info.get("affected_symbols", []),
            sentiment=1.0 if impact_info.get("direction") == "bullish" else -1.0 if impact_info.get("direction") == "bearish" else 0.0
        )
        
        self._record_signal_event(event)
        logger.info(f"[SIGNAL] Economic event: {event_type} = {outcome}")
        
        return event
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text (simple regex-based)"""
        import re
        # Match $SYMBOL or standalone caps that look like tickers
        patterns = [
            r'\$([A-Z]{1,5})',           # $AAPL format
            r'\b([A-Z]{2,5})\b'          # AAPL format (2-5 caps)
        ]
        
        symbols = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            symbols.update(matches)
        
        # Filter out common words that aren't tickers
        noise = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 
                'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'HAS', 'HIS', 'HOW',
                'CEO', 'CFO', 'IPO', 'ETF', 'NYSE', 'SEC', 'FED', 'GDP', 'CPI'}
        
        return [s for s in symbols if s not in noise]
    
    def _estimate_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment estimation"""
        text_lower = text.lower()
        
        bullish_words = ['bullish', 'buy', 'long', 'moon', 'rocket', 'surge', 'soar', 
                        'breakout', 'rally', 'beat', 'upgrade', 'growth', 'strong']
        bearish_words = ['bearish', 'sell', 'short', 'crash', 'dump', 'plunge', 'tank',
                        'breakdown', 'miss', 'downgrade', 'weak', 'decline', 'warning']
        
        bull_count = sum(1 for w in bullish_words if w in text_lower)
        bear_count = sum(1 for w in bearish_words if w in text_lower)
        
        if bull_count + bear_count == 0:
            return 0.0
        
        return (bull_count - bear_count) / (bull_count + bear_count)
    
    def _record_signal_event(self, event: SignalEvent):
        """Record signal event to database"""
        conn = sqlite3.connect(SIGNALS_DB)
        c = conn.cursor()
        
        c.execute('''
            INSERT OR REPLACE INTO signal_events 
            (event_id, signal_type, trigger, detected_at, source, raw_text, 
             mentioned_symbols, sentiment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id, event.signal_type, event.trigger, event.detected_at,
            event.source, event.raw_text, json.dumps(event.mentioned_symbols),
            event.sentiment
        ))
        
        conn.commit()
        conn.close()
        
        self.pending_signals.append(event)
    
    # =========================================================================
    # LEARNING FROM OUTCOMES
    # =========================================================================
    
    async def check_signal_outcomes(self, hours_ago: int = 24):
        """
        Check the outcomes of signals detected in the past N hours.
        Update learned weights based on accuracy.
        """
        cutoff = (datetime.now() - timedelta(hours=hours_ago)).isoformat()
        
        conn = sqlite3.connect(SIGNALS_DB)
        c = conn.cursor()
        
        # Get untracked signals
        c.execute('''
            SELECT event_id, signal_type, trigger, detected_at, mentioned_symbols, sentiment
            FROM signal_events
            WHERE detected_at > ? AND outcome_tracked = 0
        ''', (cutoff,))
        
        rows = c.fetchall()
        conn.close()
        
        logger.info(f"[SIGNAL LEARNER] Checking outcomes for {len(rows)} signals...")
        
        for row in rows:
            event_id, signal_type, trigger, detected_at, symbols_json, sentiment = row
            symbols = json.loads(symbols_json) if symbols_json else []
            
            # Get actual market impact
            impact = await self._measure_impact(symbols, detected_at)
            
            if impact is not None:
                # Update the learned signal
                self._update_signal_from_outcome(
                    signal_type=signal_type,
                    trigger=trigger,
                    predicted_direction="bullish" if sentiment > 0 else "bearish" if sentiment < 0 else "neutral",
                    actual_impact=impact,
                    symbols=symbols
                )
                
                # Mark as tracked
                self._mark_outcome_tracked(event_id, impact)
        
        # Save updated learnings
        self._save_learned_signals()
        logger.info("[SIGNAL LEARNER] Outcome check complete, weights updated")
    
    async def _measure_impact(self, symbols: List[str], signal_time: str) -> Optional[float]:
        """
        Measure actual market impact after a signal.
        Returns average % change of mentioned symbols.
        """
        if not symbols:
            return None
        
        try:
            import yfinance as yf
            
            signal_dt = datetime.fromisoformat(signal_time)
            end_dt = min(datetime.now(), signal_dt + timedelta(hours=48))
            
            impacts = []
            for symbol in symbols[:5]:  # Limit to 5 symbols
                try:
                    data = yf.download(symbol, 
                                      start=signal_dt.strftime("%Y-%m-%d"),
                                      end=end_dt.strftime("%Y-%m-%d"),
                                      progress=False)
                    
                    if len(data) >= 2:
                        pct_change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) 
                                     / data['Close'].iloc[0]) * 100
                        impacts.append(pct_change)
                except:
                    continue
            
            return sum(impacts) / len(impacts) if impacts else None
            
        except Exception as e:
            logger.error(f"Error measuring impact: {e}")
            return None
    
    def _update_signal_from_outcome(self, signal_type: str, trigger: str,
                                   predicted_direction: str, actual_impact: float,
                                   symbols: List[str]):
        """Update learned signal based on actual outcome"""
        
        # Find or create the signal
        signal_id = f"{signal_type}_{trigger.replace(' ', '_')}"
        
        if signal_id not in self.learned_signals:
            self.learned_signals[signal_id] = LearnedSignal(
                signal_id=signal_id,
                signal_type=signal_type,
                trigger=trigger,
                primary_symbols=symbols
            )
        
        signal = self.learned_signals[signal_id]
        
        # Update statistics
        signal.hit_count += 1
        
        # Was prediction correct?
        actual_direction = "bullish" if actual_impact > 1 else "bearish" if actual_impact < -1 else "neutral"
        correct = (predicted_direction == actual_direction) or (predicted_direction == "neutral")
        
        if correct:
            signal.success_count += 1
        
        # Update running average impact
        old_avg = signal.avg_impact_pct
        signal.avg_impact_pct = ((old_avg * (signal.hit_count - 1)) + abs(actual_impact)) / signal.hit_count
        
        # Update confidence (exponential moving average)
        accuracy = signal.success_count / signal.hit_count if signal.hit_count > 0 else 0.5
        signal.confidence = 0.8 * signal.confidence + 0.2 * accuracy
        
        # Update typical direction
        if signal.hit_count >= 5:
            if signal.success_count / signal.hit_count > 0.6:
                signal.typical_direction = predicted_direction
        
        signal.last_updated = datetime.now().isoformat()
        
        logger.info(f"[SIGNAL LEARNER] Updated {signal_id}: accuracy={accuracy:.2%}, confidence={signal.confidence:.2f}")
    
    def _mark_outcome_tracked(self, event_id: str, impact: float):
        """Mark signal event as tracked with outcome"""
        conn = sqlite3.connect(SIGNALS_DB)
        c = conn.cursor()
        c.execute('''
            UPDATE signal_events 
            SET outcome_tracked = 1, actual_impact_pct = ?
            WHERE event_id = ?
        ''', (impact, event_id))
        conn.commit()
        conn.close()
    
    # =========================================================================
    # SIGNAL SCORING
    # =========================================================================
    
    def get_signal_score(self, symbol: str) -> Dict[str, Any]:
        """
        Get aggregated signal score for a symbol using DYNAMIC influence data.
        
        Returns:
            Dict with score, active_signals, confidence
        """
        result = {
            "score": 0,
            "active_signals": [],
            "signal_count": 0,
            "confidence": 0.0
        }
        
        # Use dynamic influence learner for signal scoring
        if DYNAMIC_INFLUENCE_AVAILABLE:
            try:
                influence_learner = get_influence_learner()
                influence_result = influence_learner.check_recent_mentions(symbol)
                
                result["score"] = influence_result.get("signal", 0)
                result["confidence"] = influence_result.get("confidence", 0)
                result["signal_count"] = influence_result.get("mention_count", 0)
                result["active_signals"] = [
                    {
                        "trigger": m.get("entity", "unknown"),
                        "type": "discovered_influencer",
                        "sentiment": m.get("sentiment", 0),
                        "confidence": m.get("influence", 0) / 10,  # Normalize
                        "detected": m.get("time", "")
                    }
                    for m in influence_result.get("mentions", [])
                ]
                
            except Exception as e:
                logger.debug(f"[SIGNAL] Dynamic score error for {symbol}: {e}")
        
        # Also check economic events (these are still tracked)
        conn = sqlite3.connect(SIGNALS_DB)
        c = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(hours=48)).isoformat()
        c.execute('''
            SELECT signal_type, trigger, sentiment, detected_at
            FROM signal_events
            WHERE detected_at > ? AND signal_type = 'event'
            ORDER BY detected_at DESC
        ''', (cutoff,))
        
        for signal_type, trigger, sentiment, detected_at in c.fetchall():
            # Economic events affect broad market
            result["active_signals"].append({
                "trigger": trigger,
                "type": "economic_event",
                "sentiment": sentiment,
                "detected": detected_at
            })
            result["score"] += sentiment * 10  # Add event impact
            result["signal_count"] += 1
        
        conn.close()
        
        # Clamp score
        result["score"] = max(-100, min(100, result["score"]))
        
        return result
    
    def get_market_signals_summary(self) -> Dict[str, Any]:
        """Get summary of all active market signals"""
        conn = sqlite3.connect(SIGNALS_DB)
        c = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        c.execute('''
            SELECT signal_type, trigger, sentiment, mentioned_symbols, detected_at
            FROM signal_events
            WHERE detected_at > ?
            ORDER BY detected_at DESC
            LIMIT 20
        ''', (cutoff,))
        
        rows = c.fetchall()
        conn.close()
        
        return {
            "last_24h_signals": len(rows),
            "signals": [
                {
                    "type": r[0],
                    "trigger": r[1],
                    "sentiment": r[2],
                    "symbols": json.loads(r[3]) if r[3] else [],
                    "time": r[4]
                }
                for r in rows
            ],
            "learned_patterns": len(self.learned_signals),
            "top_confidence_signals": sorted(
                [(k, v.confidence, v.hit_count) for k, v in self.learned_signals.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
    
    # =========================================================================
    # PATTERN DISCOVERY
    # =========================================================================
    
    async def discover_patterns(self, min_samples: int = 10):
        """
        Analyze historical signals to discover new patterns.
        Looks for:
        - Signal sequences that predict moves
        - Combinations of signals that amplify impact
        - Temporal patterns (time of day, day of week effects)
        """
        conn = sqlite3.connect(SIGNALS_DB)
        c = conn.cursor()
        
        # Get signals with outcomes
        c.execute('''
            SELECT signal_type, trigger, sentiment, actual_impact_pct, detected_at
            FROM signal_events
            WHERE outcome_tracked = 1 AND actual_impact_pct IS NOT NULL
            ORDER BY detected_at
        ''')
        
        rows = c.fetchall()
        conn.close()
        
        if len(rows) < min_samples:
            logger.info(f"[PATTERN DISCOVERY] Not enough data ({len(rows)}/{min_samples})")
            return
        
        # Analyze signal pairs (A followed by B within 24h)
        pair_outcomes = defaultdict(list)
        
        for i, (type_a, trigger_a, sent_a, impact_a, time_a) in enumerate(rows[:-1]):
            time_a_dt = datetime.fromisoformat(time_a)
            
            for type_b, trigger_b, sent_b, impact_b, time_b in rows[i+1:]:
                time_b_dt = datetime.fromisoformat(time_b)
                
                # Check if within 24h
                if (time_b_dt - time_a_dt).total_seconds() > 86400:
                    break
                
                pair_key = f"{trigger_a} → {trigger_b}"
                pair_outcomes[pair_key].append(impact_b)
        
        # Find significant pairs
        significant_pairs = []
        for pair, impacts in pair_outcomes.items():
            if len(impacts) >= 3:
                avg_impact = sum(impacts) / len(impacts)
                if abs(avg_impact) > 2:  # More than 2% average move
                    significant_pairs.append({
                        "pattern": pair,
                        "avg_impact": avg_impact,
                        "sample_size": len(impacts),
                        "direction": "bullish" if avg_impact > 0 else "bearish"
                    })
        
        if significant_pairs:
            logger.info(f"[PATTERN DISCOVERY] Found {len(significant_pairs)} significant patterns")
            self.correlation_memory["discovered_patterns"] = significant_pairs
            self._save_correlation_memory()
        
        return significant_pairs


# =============================================================================
# SINGLETON
# =============================================================================

_signal_learner: Optional[SmartSignalLearner] = None

def get_signal_learner() -> SmartSignalLearner:
    """Get singleton instance"""
    global _signal_learner
    if _signal_learner is None:
        _signal_learner = SmartSignalLearner()
    return _signal_learner


# =============================================================================
# CLI USAGE
# =============================================================================

if __name__ == "__main__":
    import sys
    
    learner = get_signal_learner()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "detect":
            # Test detection
            text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Elon Musk just tweeted about TSLA going to the moon!"
            signals = learner.detect_influencer_mention(text)
            print(f"Detected {len(signals)} signals:")
            for s in signals:
                print(f"  - {s.trigger}: sentiment={s.sentiment:.2f}, symbols={s.mentioned_symbols}")
        
        elif command == "score":
            symbol = sys.argv[2] if len(sys.argv) > 2 else "TSLA"
            score = learner.get_signal_score(symbol)
            print(f"Signal score for {symbol}:")
            print(f"  Score: {score['score']:.1f}")
            print(f"  Active signals: {score['signal_count']}")
            print(f"  Confidence: {score['confidence']:.2f}")
        
        elif command == "summary":
            summary = learner.get_market_signals_summary()
            print(f"Market Signals Summary:")
            print(f"  Last 24h signals: {summary['last_24h_signals']}")
            print(f"  Learned patterns: {summary['learned_patterns']}")
            print(f"  Top signals by confidence:")
            for name, conf, hits in summary['top_confidence_signals']:
                print(f"    - {name}: {conf:.2f} ({hits} hits)")
        
        elif command == "check":
            # Check outcomes
            asyncio.run(learner.check_signal_outcomes())
        
        else:
            print(f"Unknown command: {command}")
            print("Usage: python smart_signal_learner.py [detect|score|summary|check] [args]")
    else:
        print("Smart Signal Learner V6")
        print("Usage:")
        print("  detect <text>  - Detect signals in text")
        print("  score <symbol> - Get signal score for symbol")
        print("  summary        - Get market signals summary")
        print("  check          - Check signal outcomes and learn")
