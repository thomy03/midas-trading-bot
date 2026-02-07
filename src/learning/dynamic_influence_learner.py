"""
Dynamic Influence Learner - Discovers WHO has market impact automatically.

NO HARDCODED INFLUENCERS.
The system discovers and tracks anyone whose mentions correlate with price moves.

How it works:
1. When a stock moves significantly, analyze social/news mentions from before
2. Extract WHO was talking about it (names, handles, sources)
3. Track correlation between mentions and subsequent moves
4. Build dynamic influence scores that evolve over time
5. Forget people who lose influence, discover new ones

Author: Jarvis for Thomas
Created: 2026-02-05
"""

import asyncio
import json
import logging
import sqlite3
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

DATA_DIR = Path("/app/data")
INFLUENCE_DB = DATA_DIR / "dynamic_influence.db"
DISCOVERED_INFLUENCERS_FILE = DATA_DIR / "discovered_influencers.json"


class DynamicInfluenceLearner:
    """
    Automatically discovers who has market influence.
    
    NO predefined list - learns everything from data.
    """
    
    def __init__(self):
        self._init_db()
        
        # Discovered influencers (learned, not hardcoded)
        self.influencers: Dict[str, Dict] = self._load_discovered()
        
        # Decay rate - influence fades if not reinforced
        self.influence_decay_rate = 0.95  # 5% decay per day without signal
        self.min_observations = 3  # Need at least 3 correlations to trust
        self.correlation_threshold = 0.3  # Min correlation to consider influential
        
        logger.info(f"[INFLUENCE] Initialized with {len(self.influencers)} discovered influencers")
    
    def _init_db(self):
        """Initialize database for tracking mentions and outcomes"""
        INFLUENCE_DB.parent.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(INFLUENCE_DB)
        c = conn.cursor()
        
        # Mentions table - every mention we detect
        c.execute('''
            CREATE TABLE IF NOT EXISTS mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity TEXT,
                entity_type TEXT,
                symbol TEXT,
                mention_time TEXT,
                source TEXT,
                context TEXT,
                sentiment REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Outcomes table - what happened after mentions
        c.execute('''
            CREATE TABLE IF NOT EXISTS mention_outcomes (
                mention_id INTEGER,
                outcome_1h REAL,
                outcome_4h REAL,
                outcome_24h REAL,
                outcome_48h REAL,
                tracked_at TEXT,
                FOREIGN KEY (mention_id) REFERENCES mentions(id)
            )
        ''')
        
        # Influence scores table - learned influence per entity
        c.execute('''
            CREATE TABLE IF NOT EXISTS influence_scores (
                entity TEXT PRIMARY KEY,
                entity_type TEXT,
                influence_score REAL,
                avg_impact_pct REAL,
                correlation REAL,
                total_mentions INTEGER,
                successful_predictions INTEGER,
                affected_symbols TEXT,
                first_seen TEXT,
                last_seen TEXT,
                last_updated TEXT
            )
        ''')
        
        # Index for fast lookups
        c.execute('CREATE INDEX IF NOT EXISTS idx_mentions_entity ON mentions(entity)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_mentions_symbol ON mentions(symbol)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_mentions_time ON mentions(mention_time)')
        
        conn.commit()
        conn.close()
    
    def _load_discovered(self) -> Dict[str, Dict]:
        """Load discovered influencers"""
        if DISCOVERED_INFLUENCERS_FILE.exists():
            try:
                with open(DISCOVERED_INFLUENCERS_FILE) as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_discovered(self):
        """Save discovered influencers"""
        with open(DISCOVERED_INFLUENCERS_FILE, 'w') as f:
            json.dump(self.influencers, f, indent=2)
    
    # =========================================================================
    # MENTION DETECTION (extract entities from text)
    # =========================================================================
    
    def extract_entities(self, text: str, source: str = "unknown") -> List[Dict]:
        """
        Extract potential influential entities from text.
        
        Looks for:
        - Twitter handles (@username)
        - Names (capitalized words that might be people)
        - Organizations/companies mentioned
        - News sources
        
        NO hardcoded list - extracts anything that looks relevant.
        """
        entities = []
        
        # 1. Twitter/X handles
        handles = re.findall(r'@([A-Za-z0-9_]{1,15})', text)
        for handle in handles:
            entities.append({
                "entity": f"@{handle.lower()}",
                "entity_type": "twitter_handle",
                "raw": handle
            })
        
        # 2. Potential names (2-3 capitalized words together)
        # e.g., "Elon Musk", "Warren Buffett", "Jerome Powell"
        name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b'
        names = re.findall(name_pattern, text)
        
        # Filter out common non-names
        noise_words = {'The Wall', 'New York', 'San Francisco', 'Los Angeles', 
                      'United States', 'Federal Reserve', 'Wall Street',
                      'Last Week', 'Next Week', 'This Morning', 'Breaking News'}
        
        for name in names:
            if name not in noise_words and len(name) > 4:
                entities.append({
                    "entity": name.lower(),
                    "entity_type": "person_name",
                    "raw": name
                })
        
        # 3. News sources / organizations (if mentioned)
        org_patterns = [
            r'\b(CNBC|Bloomberg|Reuters|WSJ|Financial Times|Yahoo Finance)\b',
            r'\b(Goldman Sachs|Morgan Stanley|JP Morgan|Citadel|Blackrock)\b',
            r'\b(SEC|FDA|FTC|DOJ|Fed|FOMC)\b'
        ]
        for pattern in org_patterns:
            orgs = re.findall(pattern, text, re.IGNORECASE)
            for org in orgs:
                entities.append({
                    "entity": org.lower(),
                    "entity_type": "organization",
                    "raw": org
                })
        
        # 4. YouTube/media personalities (channel mentions)
        yt_pattern = r'(?:youtu\.be/|youtube\.com/(?:watch\?v=|channel/|@))([A-Za-z0-9_-]+)'
        yt_matches = re.findall(yt_pattern, text)
        for yt in yt_matches:
            entities.append({
                "entity": f"yt:{yt.lower()}",
                "entity_type": "youtube",
                "raw": yt
            })
        
        return entities
    
    def extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        # $SYMBOL format
        cashtags = re.findall(r'\$([A-Z]{1,5})\b', text)
        
        # Plain SYMBOL format (2-5 caps)
        plain = re.findall(r'\b([A-Z]{2,5})\b', text)
        
        # Filter noise
        noise = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 
                'CEO', 'CFO', 'IPO', 'ETF', 'NYSE', 'SEC', 'USA', 'GDP'}
        
        symbols = set(cashtags)
        symbols.update(s for s in plain if s not in noise)
        
        return list(symbols)
    
    # =========================================================================
    # RECORD MENTIONS
    # =========================================================================
    
    def record_mention(self, text: str, symbols: List[str], source: str,
                      sentiment: float = 0.0) -> List[int]:
        """
        Record a mention for later correlation analysis.
        
        Args:
            text: The text containing the mention
            symbols: Stock symbols mentioned
            source: Where this came from (twitter, news, etc.)
            sentiment: Sentiment score (-1 to 1)
            
        Returns:
            List of mention IDs created
        """
        entities = self.extract_entities(text, source)
        
        if not entities or not symbols:
            return []
        
        mention_ids = []
        conn = sqlite3.connect(INFLUENCE_DB)
        c = conn.cursor()
        
        for entity in entities:
            for symbol in symbols:
                c.execute('''
                    INSERT INTO mentions (entity, entity_type, symbol, mention_time, 
                                         source, context, sentiment)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entity['entity'],
                    entity['entity_type'],
                    symbol,
                    datetime.now().isoformat(),
                    source,
                    text[:500],  # Truncate context
                    sentiment
                ))
                mention_ids.append(c.lastrowid)
        
        conn.commit()
        conn.close()
        
        logger.debug(f"[INFLUENCE] Recorded {len(mention_ids)} mentions from {source}")
        return mention_ids
    
    # =========================================================================
    # TRACK OUTCOMES
    # =========================================================================
    
    async def track_mention_outcomes(self, hours_back: int = 48):
        """
        Track what happened after mentions.
        This is how we learn who actually has influence.
        """
        import yfinance as yf
        
        conn = sqlite3.connect(INFLUENCE_DB)
        c = conn.cursor()
        
        # Get mentions without outcomes
        cutoff = (datetime.now() - timedelta(hours=hours_back)).isoformat()
        c.execute('''
            SELECT m.id, m.entity, m.symbol, m.mention_time, m.sentiment
            FROM mentions m
            LEFT JOIN mention_outcomes o ON m.id = o.mention_id
            WHERE m.mention_time > ? AND o.mention_id IS NULL
        ''', (cutoff,))
        
        mentions = c.fetchall()
        logger.info(f"[INFLUENCE] Tracking outcomes for {len(mentions)} mentions...")
        
        # Group by symbol for efficient data fetching
        symbol_mentions = defaultdict(list)
        for m in mentions:
            symbol_mentions[m[2]].append(m)
        
        # Fetch price data and calculate outcomes
        for symbol, symbol_mentions_list in symbol_mentions.items():
            try:
                # Get price data
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d", interval="1h")
                
                if hist.empty:
                    continue
                
                for mention_id, entity, sym, mention_time, sentiment in symbol_mentions_list:
                    try:
                        mention_dt = datetime.fromisoformat(mention_time)
                        
                        # Find price at mention time
                        hist.index = hist.index.tz_localize(None)
                        before_mention = hist[hist.index <= mention_dt]
                        
                        if before_mention.empty:
                            continue
                        
                        base_price = before_mention['Close'].iloc[-1]
                        
                        # Calculate outcomes at different intervals
                        outcomes = {}
                        for hours, label in [(1, '1h'), (4, '4h'), (24, '24h'), (48, '48h')]:
                            target_time = mention_dt + timedelta(hours=hours)
                            after = hist[hist.index >= target_time]
                            
                            if not after.empty:
                                end_price = after['Close'].iloc[0]
                                outcomes[label] = ((end_price - base_price) / base_price) * 100
                            else:
                                outcomes[label] = None
                        
                        # Record outcome
                        c.execute('''
                            INSERT INTO mention_outcomes 
                            (mention_id, outcome_1h, outcome_4h, outcome_24h, outcome_48h, tracked_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            mention_id,
                            outcomes.get('1h'),
                            outcomes.get('4h'),
                            outcomes.get('24h'),
                            outcomes.get('48h'),
                            datetime.now().isoformat()
                        ))
                        
                    except Exception as e:
                        continue
                
            except Exception as e:
                logger.debug(f"[INFLUENCE] Error tracking {symbol}: {e}")
                continue
            
            await asyncio.sleep(0.2)  # Rate limit
        
        conn.commit()
        conn.close()
        
        # Now update influence scores based on new data
        await self._update_influence_scores()
    
    async def _update_influence_scores(self):
        """
        Calculate influence scores based on mention-outcome correlations.
        This is the core learning algorithm.
        """
        conn = sqlite3.connect(INFLUENCE_DB)
        c = conn.cursor()
        
        # Get all entities with enough data
        c.execute('''
            SELECT 
                m.entity,
                m.entity_type,
                COUNT(*) as mention_count,
                AVG(m.sentiment) as avg_sentiment,
                AVG(o.outcome_24h) as avg_outcome,
                GROUP_CONCAT(DISTINCT m.symbol) as symbols,
                MIN(m.mention_time) as first_seen,
                MAX(m.mention_time) as last_seen
            FROM mentions m
            JOIN mention_outcomes o ON m.id = o.mention_id
            WHERE o.outcome_24h IS NOT NULL
            GROUP BY m.entity
            HAVING mention_count >= ?
        ''', (self.min_observations,))
        
        rows = c.fetchall()
        
        for entity, entity_type, count, avg_sent, avg_outcome, symbols, first, last in rows:
            # Calculate correlation between sentiment and outcome
            c.execute('''
                SELECT m.sentiment, o.outcome_24h
                FROM mentions m
                JOIN mention_outcomes o ON m.id = o.mention_id
                WHERE m.entity = ? AND o.outcome_24h IS NOT NULL
            ''', (entity,))
            
            pairs = c.fetchall()
            
            if len(pairs) < self.min_observations:
                continue
            
            # Simple correlation calculation
            sentiments = [p[0] for p in pairs]
            outcomes = [p[1] for p in pairs]
            
            # Calculate if mentions predict direction
            correct_direction = sum(
                1 for s, o in pairs 
                if (s > 0 and o > 0) or (s < 0 and o < 0) or (abs(s) < 0.1)
            )
            accuracy = correct_direction / len(pairs)
            
            # Calculate average impact (absolute outcome when mentioned)
            avg_impact = sum(abs(o) for o in outcomes) / len(outcomes)
            
            # Influence score: combines accuracy and impact
            influence_score = accuracy * avg_impact
            
            # Only track if meaningful influence
            if influence_score > 0.5 or count >= 10:
                # Update or insert
                c.execute('''
                    INSERT OR REPLACE INTO influence_scores
                    (entity, entity_type, influence_score, avg_impact_pct, correlation,
                     total_mentions, successful_predictions, affected_symbols,
                     first_seen, last_seen, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entity, entity_type, influence_score, avg_impact,
                    accuracy, count, correct_direction, symbols,
                    first, last, datetime.now().isoformat()
                ))
                
                # Update in-memory cache
                self.influencers[entity] = {
                    "entity_type": entity_type,
                    "influence_score": influence_score,
                    "avg_impact_pct": avg_impact,
                    "accuracy": accuracy,
                    "mention_count": count,
                    "symbols": symbols.split(',') if symbols else [],
                    "last_seen": last
                }
                
                logger.info(f"[INFLUENCE] Discovered: {entity} (score={influence_score:.2f}, "
                           f"accuracy={accuracy:.0%}, mentions={count})")
        
        conn.commit()
        conn.close()
        
        # Apply decay to old influencers
        self._apply_decay()
        
        # Save updated discoveries
        self._save_discovered()
    
    def _apply_decay(self):
        """Apply decay to influencers we haven't seen recently"""
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        
        to_remove = []
        for entity, data in self.influencers.items():
            if data.get('last_seen', '') < cutoff:
                # Decay the score
                data['influence_score'] *= self.influence_decay_rate
                
                # Remove if too weak
                if data['influence_score'] < 0.1:
                    to_remove.append(entity)
                    logger.info(f"[INFLUENCE] Removed (decayed): {entity}")
        
        for entity in to_remove:
            del self.influencers[entity]
    
    # =========================================================================
    # QUERY INFLUENCE
    # =========================================================================
    
    def get_influential_entities(self, min_score: float = 1.0) -> List[Dict]:
        """Get list of currently influential entities"""
        return [
            {"entity": k, **v}
            for k, v in self.influencers.items()
            if v.get('influence_score', 0) >= min_score
        ]
    
    def get_influence_for_symbol(self, symbol: str) -> List[Dict]:
        """Get entities that have shown influence over a specific symbol"""
        relevant = []
        for entity, data in self.influencers.items():
            if symbol in data.get('symbols', []):
                relevant.append({"entity": entity, **data})
        
        return sorted(relevant, key=lambda x: x.get('influence_score', 0), reverse=True)
    
    def check_recent_mentions(self, symbol: str, hours: int = 24) -> Dict:
        """
        Check if any known influential entities mentioned this symbol recently.
        
        Returns aggregated signal based on discovered influencers.
        """
        conn = sqlite3.connect(INFLUENCE_DB)
        c = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        c.execute('''
            SELECT m.entity, m.sentiment, m.mention_time, i.influence_score, i.accuracy
            FROM mentions m
            JOIN influence_scores i ON m.entity = i.entity
            WHERE m.symbol = ? AND m.mention_time > ?
            ORDER BY i.influence_score DESC
        ''', (symbol, cutoff))
        
        mentions = c.fetchall()
        conn.close()
        
        if not mentions:
            return {"signal": 0, "confidence": 0, "mentions": []}
        
        # Calculate weighted signal
        total_weight = 0
        weighted_signal = 0
        mention_details = []
        
        for entity, sentiment, time, influence, accuracy in mentions:
            weight = influence * accuracy
            weighted_signal += sentiment * weight
            total_weight += weight
            
            mention_details.append({
                "entity": entity,
                "sentiment": sentiment,
                "influence": influence,
                "time": time
            })
        
        signal = weighted_signal / total_weight if total_weight > 0 else 0
        confidence = min(1.0, total_weight / 5)  # Normalize confidence
        
        return {
            "signal": signal * 50,  # Scale to -50 to +50
            "confidence": confidence,
            "mention_count": len(mentions),
            "mentions": mention_details[:5]  # Top 5
        }
    
    def get_statistics(self) -> Dict:
        """Get learner statistics"""
        conn = sqlite3.connect(INFLUENCE_DB)
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM mentions')
        total_mentions = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM influence_scores')
        discovered_count = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM influence_scores WHERE influence_score > 2')
        high_influence = c.fetchone()[0]
        
        # Top influencers
        c.execute('''
            SELECT entity, influence_score, accuracy, total_mentions
            FROM influence_scores
            ORDER BY influence_score DESC
            LIMIT 10
        ''')
        top = c.fetchall()
        
        conn.close()
        
        return {
            "total_mentions_tracked": total_mentions,
            "entities_discovered": discovered_count,
            "high_influence_entities": high_influence,
            "top_influencers": [
                {"entity": e, "score": s, "accuracy": f"{a:.0%}", "mentions": m}
                for e, s, a, m in top
            ]
        }


# =============================================================================
# INTEGRATION WITH GROK SCANNER
# =============================================================================

class InfluenceAwareScanner:
    """
    Wrapper that adds influence detection to any text scanning.
    Use this to wrap Grok scanner results.
    """
    
    def __init__(self):
        self.learner = DynamicInfluenceLearner()
    
    def process_scan_result(self, text: str, symbols: List[str], 
                           source: str, sentiment: float) -> Dict:
        """
        Process a scan result and extract influence data.
        
        Call this after Grok analyzes tweets/news.
        """
        # Record the mention
        mention_ids = self.learner.record_mention(text, symbols, source, sentiment)
        
        # Check if any known influencers are involved
        entities = self.learner.extract_entities(text, source)
        
        influential_entities = []
        for entity in entities:
            if entity['entity'] in self.learner.influencers:
                inf_data = self.learner.influencers[entity['entity']]
                influential_entities.append({
                    "entity": entity['entity'],
                    "influence_score": inf_data.get('influence_score', 0),
                    "accuracy": inf_data.get('accuracy', 0)
                })
        
        return {
            "mentions_recorded": len(mention_ids),
            "influential_entities": influential_entities,
            "has_known_influencer": len(influential_entities) > 0
        }
    
    def get_symbol_signal(self, symbol: str) -> Dict:
        """Get aggregated influence signal for a symbol"""
        return self.learner.check_recent_mentions(symbol)


# =============================================================================
# SINGLETON
# =============================================================================

_influence_learner: Optional[DynamicInfluenceLearner] = None

def get_influence_learner() -> DynamicInfluenceLearner:
    global _influence_learner
    if _influence_learner is None:
        _influence_learner = DynamicInfluenceLearner()
    return _influence_learner


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    learner = get_influence_learner()
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == "stats":
            stats = learner.get_statistics()
            print(json.dumps(stats, indent=2))
        
        elif cmd == "top":
            top = learner.get_influential_entities(min_score=0.5)
            print(f"Top {len(top)} discovered influencers:")
            for inf in sorted(top, key=lambda x: x['influence_score'], reverse=True)[:20]:
                print(f"  {inf['entity']}: score={inf['influence_score']:.2f}, "
                      f"accuracy={inf.get('accuracy', 0):.0%}, "
                      f"mentions={inf.get('mention_count', 0)}")
        
        elif cmd == "check" and len(sys.argv) > 2:
            symbol = sys.argv[2].upper()
            result = learner.check_recent_mentions(symbol)
            print(f"Influence signal for {symbol}:")
            print(json.dumps(result, indent=2))
        
        elif cmd == "track":
            asyncio.run(learner.track_mention_outcomes())
            print("Outcome tracking complete")
        
        else:
            print(f"Unknown command: {cmd}")
    else:
        print("Dynamic Influence Learner")
        print("Commands:")
        print("  stats       - Show statistics")
        print("  top         - Show top discovered influencers")
        print("  check SYMBOL- Check influence signals for symbol")
        print("  track       - Track mention outcomes (run daily)")
