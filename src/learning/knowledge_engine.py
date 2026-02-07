"""
Knowledge Engine V6 - Auto-learning from successes AND failures.

This engine:
1. Learns from ERRORS - analyzes why trades failed
2. Detects EXPLODING small caps and reverse-engineers why
3. Builds a KNOWLEDGE BASE of patterns and lessons
4. AUTO-ADJUSTS weights based on real market outcomes
5. Identifies MISLEADING signals (high volatility/activity traps)

Author: Jarvis for Thomas
Created: 2026-02-05
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

# Data paths
DATA_DIR = Path("/app/data")
KNOWLEDGE_DB = DATA_DIR / "knowledge_engine.db"
LESSONS_FILE = DATA_DIR / "learned_lessons.json"
INDICATOR_RELIABILITY_FILE = DATA_DIR / "indicator_reliability.json"
EXPLOSION_PATTERNS_FILE = DATA_DIR / "explosion_patterns.json"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TradeMistake:
    """A trading mistake to learn from"""
    mistake_id: str
    symbol: str
    trade_date: str
    
    # What happened
    entry_score: float
    exit_pnl_pct: float
    hold_days: int
    
    # Why it was wrong
    misleading_signals: List[str]  # Indicators that gave wrong signal
    missed_warnings: List[str]     # Red flags we should have seen
    market_context: str            # What was happening in the market
    
    # The lesson
    lesson: str
    severity: str  # "minor" (<-5%), "major" (<-15%), "catastrophic" (<-30%)
    
    # Pattern recognition
    similar_past_mistakes: int = 0
    pattern_identified: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExplosionEvent:
    """A small cap that exploded - to learn from"""
    event_id: str
    symbol: str
    explosion_date: str
    
    # The explosion
    gain_pct: float
    market_cap_before: float
    volume_spike_ratio: float
    
    # Pre-explosion signals (what we should have seen)
    pre_signals: Dict[str, Any]  # All indicators J-1
    
    # Root cause analysis
    catalyst: str  # "earnings", "news", "influencer", "sector_momentum", "unknown"
    catalyst_details: str
    
    # Was it predictable?
    had_technical_signals: bool
    had_sentiment_signals: bool
    had_volume_signals: bool
    predictability_score: float  # 0-1, how predictable was this?
    
    # Lessons
    key_indicators_present: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class LearnedLesson:
    """A lesson learned from market experience"""
    lesson_id: str
    lesson_type: str  # "mistake", "success", "pattern", "trap"
    
    # The lesson
    title: str
    description: str
    
    # When to apply
    conditions: List[str]  # Market conditions when this applies
    indicators_involved: List[str]
    
    # Statistics
    times_observed: int = 1
    times_helped: int = 0  # Times applying this lesson helped
    confidence: float = 0.5
    
    # Meta
    created_at: str = ""
    last_updated: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class IndicatorReliability:
    """Tracked reliability of each indicator"""
    indicator_name: str
    
    # Performance tracking
    total_signals: int = 0
    correct_signals: int = 0
    false_positives: int = 0  # Said bullish, went down
    false_negatives: int = 0  # Said bearish, went up
    
    # Reliability metrics
    accuracy: float = 0.5
    precision: float = 0.5  # Of bullish calls, how many were right
    recall: float = 0.5     # Of actual up moves, how many did we catch
    
    # Context-dependent reliability
    reliability_by_regime: Dict[str, float] = field(default_factory=dict)
    reliability_by_sector: Dict[str, float] = field(default_factory=dict)
    
    # Trap detection
    is_trap_indicator: bool = False  # Often misleading
    trap_conditions: List[str] = field(default_factory=list)
    
    # Weight recommendation
    recommended_weight: float = 1.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# KNOWLEDGE ENGINE
# =============================================================================

class KnowledgeEngine:
    """
    Auto-learning knowledge engine.
    
    Learns from:
    - Trading mistakes (why did we lose?)
    - Small cap explosions (what signals preceded them?)
    - Indicator reliability (which indicators are trustworthy?)
    - Market patterns (what combinations work?)
    """
    
    def __init__(self):
        self._init_db()
        self.lessons = self._load_lessons()
        self.indicator_reliability = self._load_indicator_reliability()
        self.explosion_patterns = self._load_explosion_patterns()
        
        # Known trap patterns (high activity/volatility that misleads)
        self.known_traps = self._init_known_traps()
        
        logger.info(f"[KNOWLEDGE] Initialized with {len(self.lessons)} lessons, "
                   f"{len(self.indicator_reliability)} tracked indicators")
    
    def _init_db(self):
        """Initialize SQLite database"""
        KNOWLEDGE_DB.parent.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(KNOWLEDGE_DB)
        c = conn.cursor()
        
        # Mistakes table
        c.execute('''
            CREATE TABLE IF NOT EXISTS mistakes (
                mistake_id TEXT PRIMARY KEY,
                symbol TEXT,
                trade_date TEXT,
                entry_score REAL,
                exit_pnl_pct REAL,
                hold_days INTEGER,
                misleading_signals TEXT,
                missed_warnings TEXT,
                market_context TEXT,
                lesson TEXT,
                severity TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Explosions table
        c.execute('''
            CREATE TABLE IF NOT EXISTS explosions (
                event_id TEXT PRIMARY KEY,
                symbol TEXT,
                explosion_date TEXT,
                gain_pct REAL,
                market_cap_before REAL,
                volume_spike_ratio REAL,
                pre_signals TEXT,
                catalyst TEXT,
                catalyst_details TEXT,
                predictability_score REAL,
                key_indicators TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Indicator outcomes (for reliability tracking)
        c.execute('''
            CREATE TABLE IF NOT EXISTS indicator_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                indicator_name TEXT,
                symbol TEXT,
                signal_date TEXT,
                signal_type TEXT,
                signal_value REAL,
                actual_outcome_pct REAL,
                was_correct INTEGER,
                market_regime TEXT,
                sector TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _init_known_traps(self) -> Dict[str, Dict]:
        """Initialize known trap patterns"""
        return {
            "volume_spike_no_catalyst": {
                "description": "High volume spike without clear catalyst - often pump & dump",
                "indicators": ["volume_ratio > 5", "no_news", "small_cap"],
                "action": "reduce_confidence",
                "reduction": 0.3
            },
            "rsi_oversold_downtrend": {
                "description": "RSI oversold but strong downtrend - catching falling knife",
                "indicators": ["rsi < 30", "price_below_sma50", "adx_strong_down"],
                "action": "reduce_confidence",
                "reduction": 0.4
            },
            "high_short_interest_squeeze": {
                "description": "High short interest can cause violent squeezes - unpredictable",
                "indicators": ["short_interest > 20%", "volume_spike"],
                "action": "flag_high_risk",
                "note": "Meme stock territory"
            },
            "earnings_runup": {
                "description": "Stock running up into earnings - often sells the news",
                "indicators": ["earnings_within_5d", "gain_5d > 10%"],
                "action": "reduce_confidence",
                "reduction": 0.25
            },
            "low_float_manipulation": {
                "description": "Low float stocks easily manipulated",
                "indicators": ["float < 10M", "volume_spike"],
                "action": "flag_high_risk",
                "note": "Easy to pump and dump"
            }
        }
    
    def _load_lessons(self) -> Dict[str, LearnedLesson]:
        """Load learned lessons"""
        if LESSONS_FILE.exists():
            try:
                with open(LESSONS_FILE) as f:
                    data = json.load(f)
                return {k: LearnedLesson(**v) for k, v in data.items()}
            except:
                pass
        return self._init_default_lessons()
    
    def _init_default_lessons(self) -> Dict[str, LearnedLesson]:
        """Initialize with some default lessons"""
        lessons = {
            "dont_catch_falling_knife": LearnedLesson(
                lesson_id="dont_catch_falling_knife",
                lesson_type="mistake",
                title="Don't catch falling knives",
                description="RSI oversold in a downtrend often goes MORE oversold. Wait for reversal confirmation.",
                conditions=["rsi < 30", "below_sma50", "strong_downtrend"],
                indicators_involved=["rsi", "sma50", "trend"],
                times_observed=100,  # Well-known lesson
                confidence=0.9
            ),
            "volume_confirms_breakout": LearnedLesson(
                lesson_id="volume_confirms_breakout",
                lesson_type="success",
                title="Volume confirms breakouts",
                description="Breakouts with 2x+ volume are much more reliable than low-volume breakouts.",
                conditions=["price_breakout", "volume_ratio > 2"],
                indicators_involved=["volume", "resistance_break"],
                times_observed=100,
                confidence=0.85
            ),
            "beware_pump_and_dump": LearnedLesson(
                lesson_id="beware_pump_and_dump",
                lesson_type="trap",
                title="Beware pump and dump patterns",
                description="Small cap + huge volume spike + no news = likely pump and dump. Don't chase.",
                conditions=["market_cap < 500M", "volume_spike > 5x", "no_catalyst"],
                indicators_involved=["volume", "market_cap"],
                times_observed=50,
                confidence=0.8
            ),
            "trend_is_friend": LearnedLesson(
                lesson_id="trend_is_friend",
                lesson_type="success", 
                title="Trend is your friend",
                description="Trading WITH the trend (above SMA50, SMA200) has much higher success rate.",
                conditions=["above_sma50", "above_sma200"],
                indicators_involved=["sma50", "sma200", "trend"],
                times_observed=200,
                confidence=0.9
            )
        }
        self._save_lessons(lessons)
        return lessons
    
    def _save_lessons(self, lessons: Optional[Dict] = None):
        """Save lessons to file"""
        if lessons is None:
            lessons = self.lessons
        with open(LESSONS_FILE, 'w') as f:
            json.dump({k: v.to_dict() for k, v in lessons.items()}, f, indent=2)
    
    def _load_indicator_reliability(self) -> Dict[str, IndicatorReliability]:
        """Load indicator reliability data"""
        if INDICATOR_RELIABILITY_FILE.exists():
            try:
                with open(INDICATOR_RELIABILITY_FILE) as f:
                    data = json.load(f)
                return {k: IndicatorReliability(**v) for k, v in data.items()}
            except:
                pass
        return self._init_indicator_reliability()
    
    def _init_indicator_reliability(self) -> Dict[str, IndicatorReliability]:
        """Initialize indicator tracking"""
        indicators = [
            # Trend
            "ema_cross_20_50", "ema_cross_50_200", "price_above_sma20",
            "price_above_sma50", "price_above_sma200", "adx_strong",
            # Momentum
            "rsi_oversold", "rsi_overbought", "macd_bullish", "macd_bearish",
            "stoch_oversold", "stoch_overbought",
            # Volume
            "volume_spike", "obv_rising", "volume_breakout",
            # Volatility
            "bollinger_squeeze", "atr_expansion", "price_near_lower_band",
            # Sentiment
            "sentiment_bullish", "sentiment_bearish", "influencer_mention"
        ]
        
        reliability = {}
        for ind in indicators:
            reliability[ind] = IndicatorReliability(
                indicator_name=ind,
                accuracy=0.5,  # Start neutral
                recommended_weight=1.0
            )
        
        self._save_indicator_reliability(reliability)
        return reliability
    
    def _save_indicator_reliability(self, data: Optional[Dict] = None):
        """Save indicator reliability"""
        if data is None:
            data = self.indicator_reliability
        with open(INDICATOR_RELIABILITY_FILE, 'w') as f:
            json.dump({k: v.to_dict() for k, v in data.items()}, f, indent=2)
    
    def _load_explosion_patterns(self) -> List[Dict]:
        """Load learned explosion patterns"""
        if EXPLOSION_PATTERNS_FILE.exists():
            try:
                with open(EXPLOSION_PATTERNS_FILE) as f:
                    return json.load(f)
            except:
                pass
        return []
    
    def _save_explosion_patterns(self):
        """Save explosion patterns"""
        with open(EXPLOSION_PATTERNS_FILE, 'w') as f:
            json.dump(self.explosion_patterns, f, indent=2)
    
    # =========================================================================
    # LEARN FROM MISTAKES
    # =========================================================================
    
    def record_mistake(self, symbol: str, entry_score: float, exit_pnl_pct: float,
                      hold_days: int, indicators_at_entry: Dict[str, Any],
                      market_context: str = "") -> TradeMistake:
        """
        Record a trading mistake and extract lessons.
        
        Args:
            symbol: Stock symbol
            entry_score: Score when we entered
            exit_pnl_pct: Final P&L percentage (negative for losses)
            hold_days: How long we held
            indicators_at_entry: All indicator values at entry
            market_context: Description of market conditions
        """
        
        # Determine severity
        if exit_pnl_pct < -30:
            severity = "catastrophic"
        elif exit_pnl_pct < -15:
            severity = "major"
        else:
            severity = "minor"
        
        # Analyze what went wrong
        misleading_signals = self._find_misleading_signals(indicators_at_entry, exit_pnl_pct)
        missed_warnings = self._find_missed_warnings(indicators_at_entry, exit_pnl_pct)
        
        # Generate lesson
        lesson = self._generate_mistake_lesson(symbol, misleading_signals, missed_warnings, severity)
        
        mistake = TradeMistake(
            mistake_id=hashlib.md5(f"{symbol}_{datetime.now().isoformat()}".encode()).hexdigest()[:16],
            symbol=symbol,
            trade_date=datetime.now().isoformat(),
            entry_score=entry_score,
            exit_pnl_pct=exit_pnl_pct,
            hold_days=hold_days,
            misleading_signals=misleading_signals,
            missed_warnings=missed_warnings,
            market_context=market_context,
            lesson=lesson,
            severity=severity
        )
        
        # Save to database
        self._save_mistake(mistake)
        
        # Update indicator reliability (penalize misleading indicators)
        for indicator in misleading_signals:
            self._update_indicator_reliability(indicator, was_correct=False, 
                                              outcome_pct=exit_pnl_pct)
        
        # Check if this is a recurring pattern
        similar = self._find_similar_mistakes(misleading_signals)
        if similar > 2:
            self._create_pattern_lesson(misleading_signals, similar)
        
        logger.warning(f"[KNOWLEDGE] Recorded {severity} mistake on {symbol}: {lesson}")
        
        return mistake
    
    def _find_misleading_signals(self, indicators: Dict, outcome_pct: float) -> List[str]:
        """Find which indicators gave wrong signals"""
        misleading = []
        
        # If we lost money, bullish signals were wrong
        if outcome_pct < -5:
            bullish_indicators = [
                ("rsi_oversold", lambda x: x.get('rsi', 50) < 30),
                ("macd_bullish", lambda x: x.get('macd_histogram', 0) > 0),
                ("price_above_sma20", lambda x: x.get('price_vs_sma20', 0) > 0),
                ("volume_spike", lambda x: x.get('volume_ratio', 1) > 2),
                ("sentiment_bullish", lambda x: x.get('sentiment_score', 0) > 30),
                ("bollinger_lower", lambda x: x.get('bb_position', 0.5) < 0.2),
            ]
            
            for name, check_fn in bullish_indicators:
                try:
                    if check_fn(indicators):
                        misleading.append(name)
                except:
                    pass
        
        return misleading
    
    def _find_missed_warnings(self, indicators: Dict, outcome_pct: float) -> List[str]:
        """Find warning signs we should have noticed"""
        warnings = []
        
        if outcome_pct < -10:
            # Check for red flags that were present
            warning_checks = [
                ("below_sma200", indicators.get('price_vs_sma200', 0) < 0, 
                 "Stock was below SMA200 - major downtrend"),
                ("high_volatility", indicators.get('atr_percent', 0) > 5,
                 "ATR > 5% - very volatile, should have used smaller position"),
                ("weak_volume", indicators.get('volume_ratio', 1) < 0.5,
                 "Volume was weak - conviction was low"),
                ("overbought", indicators.get('rsi', 50) > 70,
                 "RSI was overbought - bad time to enter long"),
                ("adx_weak", indicators.get('adx', 25) < 20,
                 "ADX weak - no clear trend, choppy market"),
            ]
            
            for name, condition, description in warning_checks:
                if condition:
                    warnings.append(f"{name}: {description}")
        
        return warnings
    
    def _generate_mistake_lesson(self, symbol: str, misleading: List[str], 
                                 warnings: List[str], severity: str) -> str:
        """Generate a human-readable lesson from the mistake"""
        
        if not misleading and not warnings:
            return f"Loss on {symbol} - market moved against us without clear warning signs."
        
        parts = []
        
        if misleading:
            parts.append(f"Misleading signals: {', '.join(misleading)}")
        
        if warnings:
            parts.append(f"Missed warnings: {warnings[0].split(':')[0]}")
        
        if severity == "catastrophic":
            parts.append("CRITICAL: Need stop-loss or position sizing review!")
        
        return " | ".join(parts)
    
    def _save_mistake(self, mistake: TradeMistake):
        """Save mistake to database"""
        conn = sqlite3.connect(KNOWLEDGE_DB)
        c = conn.cursor()
        c.execute('''
            INSERT INTO mistakes
            (mistake_id, symbol, trade_date, entry_score, exit_pnl_pct, hold_days,
             misleading_signals, missed_warnings, market_context, lesson, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            mistake.mistake_id, mistake.symbol, mistake.trade_date,
            mistake.entry_score, mistake.exit_pnl_pct, mistake.hold_days,
            json.dumps(mistake.misleading_signals), json.dumps(mistake.missed_warnings),
            mistake.market_context, mistake.lesson, mistake.severity
        ))
        conn.commit()
        conn.close()
    
    def _find_similar_mistakes(self, misleading_signals: List[str]) -> int:
        """Find how many similar mistakes we've made"""
        if not misleading_signals:
            return 0
        
        conn = sqlite3.connect(KNOWLEDGE_DB)
        c = conn.cursor()
        
        # Look for mistakes with same misleading signals
        c.execute('SELECT misleading_signals FROM mistakes')
        rows = c.fetchall()
        conn.close()
        
        similar = 0
        for (signals_json,) in rows:
            past_signals = set(json.loads(signals_json))
            current_signals = set(misleading_signals)
            if len(past_signals & current_signals) >= 2:  # At least 2 common signals
                similar += 1
        
        return similar
    
    def _create_pattern_lesson(self, indicators: List[str], occurrences: int):
        """Create a lesson from a recurring mistake pattern"""
        pattern_key = "_".join(sorted(indicators))
        lesson_id = f"pattern_{hashlib.md5(pattern_key.encode()).hexdigest()[:8]}"
        
        if lesson_id not in self.lessons:
            self.lessons[lesson_id] = LearnedLesson(
                lesson_id=lesson_id,
                lesson_type="pattern",
                title=f"Recurring false positive pattern",
                description=f"Combination of {', '.join(indicators)} has led to losses {occurrences} times. Reduce confidence when these appear together.",
                conditions=indicators,
                indicators_involved=indicators,
                times_observed=occurrences,
                confidence=min(0.9, 0.5 + occurrences * 0.1),
                created_at=datetime.now().isoformat()
            )
            self._save_lessons()
            logger.info(f"[KNOWLEDGE] New pattern lesson created: {indicators}")
    
    # =========================================================================
    # LEARN FROM EXPLOSIONS (Small caps that moon)
    # =========================================================================
    
    async def analyze_explosion(self, symbol: str, gain_pct: float,
                               pre_explosion_data: Dict) -> ExplosionEvent:
        """
        Analyze a small cap that exploded to learn what signals preceded it.
        
        Args:
            symbol: Stock that exploded
            gain_pct: How much it gained
            pre_explosion_data: All data from the day before the explosion
        """
        
        # Determine catalyst
        catalyst, catalyst_details = await self._identify_catalyst(symbol, pre_explosion_data)
        
        # Check what signals were present
        had_technical = self._check_technical_signals(pre_explosion_data)
        had_sentiment = self._check_sentiment_signals(pre_explosion_data)
        had_volume = self._check_volume_signals(pre_explosion_data)
        
        # Calculate predictability
        predictability = self._calculate_predictability(
            had_technical, had_sentiment, had_volume, catalyst
        )
        
        # Identify key indicators that were present
        key_indicators = self._extract_key_indicators(pre_explosion_data, gain_pct)
        
        event = ExplosionEvent(
            event_id=hashlib.md5(f"{symbol}_{datetime.now().isoformat()}".encode()).hexdigest()[:16],
            symbol=symbol,
            explosion_date=datetime.now().isoformat(),
            gain_pct=gain_pct,
            market_cap_before=pre_explosion_data.get('market_cap', 0),
            volume_spike_ratio=pre_explosion_data.get('volume_ratio', 1),
            pre_signals=pre_explosion_data,
            catalyst=catalyst,
            catalyst_details=catalyst_details,
            had_technical_signals=had_technical,
            had_sentiment_signals=had_sentiment,
            had_volume_signals=had_volume,
            predictability_score=predictability,
            key_indicators_present=key_indicators
        )
        
        # Save to database
        self._save_explosion(event)
        
        # Update indicator reliability (reward indicators that caught it)
        for indicator in key_indicators:
            self._update_indicator_reliability(indicator, was_correct=True,
                                              outcome_pct=gain_pct)
        
        # Look for patterns
        self._update_explosion_patterns(event)
        
        logger.info(f"[KNOWLEDGE] Analyzed explosion: {symbol} +{gain_pct:.1f}% | "
                   f"Catalyst: {catalyst} | Predictability: {predictability:.0%}")
        
        return event
    
    async def _identify_catalyst(self, symbol: str, data: Dict) -> Tuple[str, str]:
        """Identify what caused the explosion"""
        
        # Check for common catalysts
        if data.get('earnings_just_released'):
            return "earnings", f"Earnings beat/guidance"
        
        if data.get('news_count', 0) > 3:
            return "news", f"High news activity: {data.get('top_headline', 'N/A')}"
        
        if data.get('influencer_mention'):
            return "influencer", f"Mentioned by: {data.get('influencer_name', 'unknown')}"
        
        if data.get('sector_momentum', 0) > 3:
            return "sector_momentum", f"Sector running hot"
        
        if data.get('short_squeeze_potential'):
            return "short_squeeze", f"Short interest: {data.get('short_interest', 0):.1f}%"
        
        return "unknown", "No clear catalyst identified"
    
    def _check_technical_signals(self, data: Dict) -> bool:
        """Check if technical signals were present pre-explosion"""
        signals = [
            data.get('rsi', 50) < 40,  # Not overbought
            data.get('macd_histogram', 0) > 0,  # MACD bullish
            data.get('price_vs_sma20', 0) > 0,  # Above short-term MA
            data.get('volume_ratio', 1) > 1.5,  # Volume building
        ]
        return sum(signals) >= 2
    
    def _check_sentiment_signals(self, data: Dict) -> bool:
        """Check if sentiment signals were present"""
        return (data.get('sentiment_score', 0) > 20 or 
                data.get('social_volume', 0) > 2)
    
    def _check_volume_signals(self, data: Dict) -> bool:
        """Check if volume signals were present"""
        return (data.get('volume_ratio', 1) > 2 or
                data.get('obv_rising', False))
    
    def _calculate_predictability(self, tech: bool, sent: bool, vol: bool, 
                                 catalyst: str) -> float:
        """Calculate how predictable this explosion was"""
        score = 0.0
        
        if tech:
            score += 0.3
        if sent:
            score += 0.25
        if vol:
            score += 0.25
        if catalyst != "unknown":
            score += 0.2
        
        return min(1.0, score)
    
    def _extract_key_indicators(self, data: Dict, gain_pct: float) -> List[str]:
        """Extract which indicators would have caught this move"""
        key = []
        
        if data.get('rsi', 50) < 35:
            key.append("rsi_oversold")
        if data.get('macd_histogram', 0) > 0:
            key.append("macd_bullish")
        if data.get('volume_ratio', 1) > 2:
            key.append("volume_spike")
        if data.get('bollinger_position', 0.5) < 0.3:
            key.append("near_lower_band")
        if data.get('sentiment_score', 0) > 30:
            key.append("sentiment_bullish")
        if data.get('obv_rising', False):
            key.append("obv_rising")
        
        return key
    
    def _save_explosion(self, event: ExplosionEvent):
        """Save explosion event to database"""
        conn = sqlite3.connect(KNOWLEDGE_DB)
        c = conn.cursor()
        c.execute('''
            INSERT INTO explosions
            (event_id, symbol, explosion_date, gain_pct, market_cap_before,
             volume_spike_ratio, pre_signals, catalyst, catalyst_details,
             predictability_score, key_indicators)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id, event.symbol, event.explosion_date, event.gain_pct,
            event.market_cap_before, event.volume_spike_ratio,
            json.dumps(event.pre_signals), event.catalyst, event.catalyst_details,
            event.predictability_score, json.dumps(event.key_indicators_present)
        ))
        conn.commit()
        conn.close()
    
    def _update_explosion_patterns(self, event: ExplosionEvent):
        """Update learned explosion patterns"""
        
        if event.predictability_score > 0.5 and event.key_indicators_present:
            # This was somewhat predictable - learn the pattern
            pattern = {
                "indicators": event.key_indicators_present,
                "catalyst": event.catalyst,
                "avg_gain": event.gain_pct,
                "occurrences": 1,
                "last_seen": event.explosion_date
            }
            
            # Check if similar pattern exists
            for existing in self.explosion_patterns:
                if set(existing["indicators"]) == set(event.key_indicators_present):
                    existing["occurrences"] += 1
                    existing["avg_gain"] = (existing["avg_gain"] + event.gain_pct) / 2
                    existing["last_seen"] = event.explosion_date
                    self._save_explosion_patterns()
                    return
            
            # New pattern
            self.explosion_patterns.append(pattern)
            self._save_explosion_patterns()
    
    # =========================================================================
    # INDICATOR RELIABILITY TRACKING
    # =========================================================================
    
    def _update_indicator_reliability(self, indicator: str, was_correct: bool,
                                     outcome_pct: float, regime: str = "",
                                     sector: str = ""):
        """Update reliability stats for an indicator"""
        
        if indicator not in self.indicator_reliability:
            self.indicator_reliability[indicator] = IndicatorReliability(
                indicator_name=indicator
            )
        
        rel = self.indicator_reliability[indicator]
        
        # Update counts
        rel.total_signals += 1
        if was_correct:
            rel.correct_signals += 1
        else:
            if outcome_pct < 0:
                rel.false_positives += 1
            else:
                rel.false_negatives += 1
        
        # Update accuracy (exponential moving average)
        current_accuracy = rel.correct_signals / rel.total_signals
        rel.accuracy = 0.9 * rel.accuracy + 0.1 * current_accuracy
        
        # Update regime-specific reliability
        if regime:
            if regime not in rel.reliability_by_regime:
                rel.reliability_by_regime[regime] = 0.5
            old = rel.reliability_by_regime[regime]
            rel.reliability_by_regime[regime] = 0.9 * old + 0.1 * (1.0 if was_correct else 0.0)
        
        # Check if this is becoming a trap indicator
        if rel.total_signals > 20 and rel.accuracy < 0.4:
            rel.is_trap_indicator = True
            logger.warning(f"[KNOWLEDGE] Indicator {indicator} marked as TRAP (accuracy: {rel.accuracy:.0%})")
        
        # Update recommended weight
        rel.recommended_weight = 0.5 + (rel.accuracy - 0.5) * 2  # 0.5 at 50%, 1.5 at 100%
        rel.recommended_weight = max(0.1, min(2.0, rel.recommended_weight))
        
        # Save
        self._save_indicator_reliability()
        
        # Record to database
        conn = sqlite3.connect(KNOWLEDGE_DB)
        c = conn.cursor()
        c.execute('''
            INSERT INTO indicator_outcomes
            (indicator_name, symbol, signal_date, signal_type, actual_outcome_pct,
             was_correct, market_regime, sector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            indicator, "", datetime.now().isoformat(), "signal", outcome_pct,
            1 if was_correct else 0, regime, sector
        ))
        conn.commit()
        conn.close()
    
    # =========================================================================
    # QUERY KNOWLEDGE
    # =========================================================================
    
    def get_indicator_weight_adjustments(self) -> Dict[str, float]:
        """Get weight adjustments for all indicators based on learned reliability"""
        adjustments = {}
        for name, rel in self.indicator_reliability.items():
            if rel.total_signals >= 10:  # Only adjust if enough data
                adjustments[name] = rel.recommended_weight
        return adjustments
    
    def check_for_traps(self, indicators: Dict) -> List[Dict]:
        """Check if current indicators match any known trap patterns"""
        traps_found = []
        
        for trap_name, trap_info in self.known_traps.items():
            # Simple pattern matching (could be more sophisticated)
            matches = 0
            for condition in trap_info["indicators"]:
                if "volume_ratio" in condition and indicators.get('volume_ratio', 0) > 5:
                    matches += 1
                if "rsi < 30" in condition and indicators.get('rsi', 50) < 30:
                    matches += 1
                if "small_cap" in condition and indicators.get('market_cap', 1e12) < 500e6:
                    matches += 1
            
            if matches >= 2:
                traps_found.append({
                    "trap": trap_name,
                    "description": trap_info["description"],
                    "action": trap_info["action"],
                    "confidence_reduction": trap_info.get("reduction", 0.2)
                })
        
        return traps_found
    
    def get_relevant_lessons(self, indicators: Dict) -> List[LearnedLesson]:
        """Get lessons relevant to current market conditions"""
        relevant = []
        
        for lesson in self.lessons.values():
            if lesson.confidence < 0.5:
                continue
            
            # Check if conditions match
            matches = 0
            for condition in lesson.conditions:
                if "rsi" in condition.lower() and "oversold" in condition.lower():
                    if indicators.get('rsi', 50) < 30:
                        matches += 1
                if "above_sma" in condition.lower():
                    if indicators.get('price_vs_sma50', 0) > 0:
                        matches += 1
                if "volume" in condition.lower():
                    if indicators.get('volume_ratio', 1) > 2:
                        matches += 1
            
            if matches > 0:
                relevant.append(lesson)
        
        return sorted(relevant, key=lambda x: x.confidence, reverse=True)[:3]
    
    def get_statistics(self) -> Dict:
        """Get knowledge engine statistics"""
        conn = sqlite3.connect(KNOWLEDGE_DB)
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM mistakes')
        mistakes_count = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM explosions')
        explosions_count = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM indicator_outcomes')
        outcomes_count = c.fetchone()[0]
        
        conn.close()
        
        # Find trap indicators
        trap_indicators = [name for name, rel in self.indicator_reliability.items() 
                         if rel.is_trap_indicator]
        
        # Find most reliable indicators
        reliable = sorted(
            [(name, rel.accuracy, rel.total_signals) 
             for name, rel in self.indicator_reliability.items()
             if rel.total_signals >= 10],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total_mistakes_recorded": mistakes_count,
            "total_explosions_analyzed": explosions_count,
            "total_indicator_outcomes": outcomes_count,
            "lessons_learned": len(self.lessons),
            "explosion_patterns": len(self.explosion_patterns),
            "trap_indicators": trap_indicators,
            "most_reliable_indicators": reliable
        }


# =============================================================================
# SINGLETON
# =============================================================================

_knowledge_engine: Optional[KnowledgeEngine] = None

def get_knowledge_engine() -> KnowledgeEngine:
    """Get singleton instance"""
    global _knowledge_engine
    if _knowledge_engine is None:
        _knowledge_engine = KnowledgeEngine()
    return _knowledge_engine


# =============================================================================
# DAILY LEARNING ROUTINE
# =============================================================================

async def run_daily_learning():
    """
    Run daily learning routine:
    1. Analyze yesterday's top movers
    2. Record any mistakes from closed trades
    3. Update indicator reliability
    4. Discover new patterns
    """
    engine = get_knowledge_engine()
    
    logger.info("[KNOWLEDGE] Starting daily learning routine...")
    
    # Get market movers
    try:
        import yfinance as yf
        
        # Get S&P 500 tickers (simplified - in production use full list)
        spy = yf.Ticker("SPY")
        # ... fetch top gainers/losers
        
        # For now, just log stats
        stats = engine.get_statistics()
        logger.info(f"[KNOWLEDGE] Stats: {stats}")
        
    except Exception as e:
        logger.error(f"[KNOWLEDGE] Daily learning error: {e}")
    
    logger.info("[KNOWLEDGE] Daily learning complete")


if __name__ == "__main__":
    import sys
    
    engine = get_knowledge_engine()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "stats":
            stats = engine.get_statistics()
            print(json.dumps(stats, indent=2))
        
        elif cmd == "weights":
            weights = engine.get_indicator_weight_adjustments()
            print("Indicator weight adjustments:")
            for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                print(f"  {name}: {weight:.2f}x")
        
        elif cmd == "lessons":
            print("Learned lessons:")
            for lesson in engine.lessons.values():
                print(f"  [{lesson.lesson_type}] {lesson.title}")
                print(f"    {lesson.description}")
                print(f"    Confidence: {lesson.confidence:.0%}\n")
        
        elif cmd == "traps":
            print("Known trap patterns:")
            for name, info in engine.known_traps.items():
                print(f"  {name}:")
                print(f"    {info['description']}")
                print(f"    Indicators: {info['indicators']}\n")
        
        else:
            print(f"Unknown command: {cmd}")
    else:
        print("Knowledge Engine V6")
        print("Commands: stats, weights, lessons, traps")
