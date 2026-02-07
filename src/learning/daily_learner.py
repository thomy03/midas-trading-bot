"""
Daily Learner - Automatic learning from market outcomes.

Runs daily to:
1. Record mistakes from closed trades
2. Detect and analyze small cap explosions
3. Update indicator reliability
4. Discover new patterns

Can be called from nightly_auditor or as standalone cron job.

Author: Jarvis for Thomas
Created: 2026-02-05
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR = Path("/app/data")


class DailyLearner:
    """
    Automatic daily learning from market outcomes.
    """
    
    def __init__(self):
        # Import knowledge engine
        try:
            from src.learning.knowledge_engine import get_knowledge_engine
            self.knowledge_engine = get_knowledge_engine()
        except ImportError:
            logger.error("[DAILY LEARNER] Knowledge Engine not available")
            self.knowledge_engine = None
        
        # Import signal learner
        try:
            from src.learning.smart_signal_learner import get_signal_learner
            self.signal_learner = get_signal_learner()
        except ImportError:
            logger.warning("[DAILY LEARNER] Signal Learner not available")
            self.signal_learner = None
        
        # Load trade history
        self.trade_history_file = DATA_DIR / "trade_history.json"
    
    async def run_daily_learning(self) -> Dict:
        """
        Run full daily learning routine.
        
        Returns summary of what was learned.
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "mistakes_recorded": 0,
            "explosions_analyzed": 0,
            "signals_checked": 0,
            "patterns_discovered": 0,
            "lessons": []
        }
        
        logger.info("[DAILY LEARNER] Starting daily learning routine...")
        
        # 1. Analyze closed trades for mistakes
        mistakes = await self._analyze_closed_trades()
        results["mistakes_recorded"] = len(mistakes)
        for m in mistakes:
            results["lessons"].append(f"Mistake on {m['symbol']}: {m['lesson']}")
        
        # 2. Find and analyze small cap explosions
        explosions = await self._find_explosions()
        results["explosions_analyzed"] = len(explosions)
        for e in explosions:
            results["lessons"].append(f"Explosion: {e['symbol']} +{e['gain']:.0f}% - {e['catalyst']}")
        
        # 3. Check signal outcomes (for signal learner)
        if self.signal_learner:
            await self.signal_learner.check_signal_outcomes(hours_ago=48)
            results["signals_checked"] = len(self.signal_learner.pending_signals)
        
        # 4. Discover patterns
        patterns = await self._discover_patterns()
        results["patterns_discovered"] = len(patterns)
        
        # Save learning report
        self._save_learning_report(results)
        
        logger.info(f"[DAILY LEARNER] Complete. Mistakes: {results['mistakes_recorded']}, "
                   f"Explosions: {results['explosions_analyzed']}")
        
        return results
    
    async def _analyze_closed_trades(self) -> List[Dict]:
        """Analyze recently closed trades for mistakes"""
        mistakes = []
        
        if not self.knowledge_engine:
            return mistakes
        
        # Load trade history
        if not self.trade_history_file.exists():
            return mistakes
        
        try:
            with open(self.trade_history_file) as f:
                trades = json.load(f)
        except:
            return mistakes
        
        # Find trades closed in last 24h with losses
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        
        for trade in trades:
            if not trade.get('exit_date'):
                continue
            if trade['exit_date'] < cutoff:
                continue
            
            pnl_pct = trade.get('pnl_pct', 0)
            
            # Record mistakes (losses > 5%)
            if pnl_pct < -5:
                indicators = trade.get('entry_indicators', {})
                
                mistake = self.knowledge_engine.record_mistake(
                    symbol=trade['symbol'],
                    entry_score=trade.get('entry_score', 50),
                    exit_pnl_pct=pnl_pct,
                    hold_days=trade.get('hold_days', 1),
                    indicators_at_entry=indicators,
                    market_context=trade.get('market_context', '')
                )
                
                mistakes.append({
                    "symbol": trade['symbol'],
                    "pnl": pnl_pct,
                    "lesson": mistake.lesson
                })
        
        return mistakes
    
    async def _find_explosions(self, min_gain: float = 20.0) -> List[Dict]:
        """
        Find small caps that exploded today.
        Analyze them to learn what signals preceded the move.
        """
        explosions = []
        
        if not self.knowledge_engine:
            return explosions
        
        try:
            # Get watchlist of small/mid caps
            watchlist = await self._get_smallcap_watchlist()
            
            logger.info(f"[DAILY LEARNER] Scanning {len(watchlist)} small caps for explosions...")
            
            # Download data in batches
            for i in range(0, len(watchlist), 50):
                batch = watchlist[i:i+50]
                batch_str = " ".join(batch)
                
                try:
                    data = yf.download(
                        batch_str,
                        period="5d",
                        progress=False,
                        threads=True
                    )
                    
                    if data.empty:
                        continue
                    
                    # Check each symbol for explosion
                    for symbol in batch:
                        try:
                            if symbol not in data['Close'].columns:
                                continue
                            
                            closes = data['Close'][symbol].dropna()
                            if len(closes) < 2:
                                continue
                            
                            # Calculate 1-day gain
                            gain_1d = ((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]) * 100
                            
                            if gain_1d >= min_gain:
                                # Found an explosion!
                                logger.info(f"[DAILY LEARNER] Explosion detected: {symbol} +{gain_1d:.1f}%")
                                
                                # Get pre-explosion data
                                pre_data = await self._get_pre_explosion_data(symbol, data)
                                
                                # Analyze with knowledge engine
                                event = await self.knowledge_engine.analyze_explosion(
                                    symbol=symbol,
                                    gain_pct=gain_1d,
                                    pre_explosion_data=pre_data
                                )
                                
                                explosions.append({
                                    "symbol": symbol,
                                    "gain": gain_1d,
                                    "catalyst": event.catalyst,
                                    "predictability": event.predictability_score
                                })
                        except Exception as e:
                            continue
                            
                except Exception as e:
                    logger.warning(f"[DAILY LEARNER] Batch download error: {e}")
                    continue
                
                # Rate limit
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"[DAILY LEARNER] Error finding explosions: {e}")
        
        return explosions
    
    async def _get_smallcap_watchlist(self) -> List[str]:
        """Get list of small/mid cap stocks to monitor"""
        
        # Try to load from watchlist file
        watchlist_file = DATA_DIR / "attention" / "watchlist.json"
        if watchlist_file.exists():
            try:
                with open(watchlist_file) as f:
                    data = json.load(f)
                    # Filter for small caps if market cap data available
                    return data.get('symbols', [])[:500]  # Limit to 500 for performance
            except:
                pass
        
        # Fallback: use Russell 2000 proxy (small caps)
        # In production, use a proper small cap list
        return [
            "GME", "AMC", "BBBY", "PLTR", "SOFI", "RIVN", "LCID", 
            "NIO", "PLUG", "FCEL", "SPCE", "OPEN", "WISH", "CLOV",
            "SKLZ", "RIDE", "WKHS", "GOEV", "QS", "BLNK", "CHPT"
        ]
    
    async def _get_pre_explosion_data(self, symbol: str, price_data) -> Dict:
        """Get indicator data from day before explosion"""
        
        pre_data = {
            "symbol": symbol,
            "date": datetime.now().isoformat()
        }
        
        try:
            # Get ticker info
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            pre_data["market_cap"] = info.get("marketCap", 0)
            pre_data["float"] = info.get("floatShares", 0)
            pre_data["short_interest"] = info.get("shortPercentOfFloat", 0)
            
            # Calculate basic indicators from price data
            if symbol in price_data['Close'].columns:
                closes = price_data['Close'][symbol].dropna()
                volumes = price_data['Volume'][symbol].dropna() if 'Volume' in price_data else None
                
                if len(closes) >= 3:
                    # Price change leading up to explosion
                    pre_data["price_change_2d"] = ((closes.iloc[-2] - closes.iloc[-3]) / closes.iloc[-3]) * 100
                    
                if volumes is not None and len(volumes) >= 2:
                    avg_vol = volumes.iloc[:-1].mean()
                    pre_data["volume_ratio"] = volumes.iloc[-2] / avg_vol if avg_vol > 0 else 1
                
                # RSI approximation (simplified)
                if len(closes) >= 14:
                    changes = closes.diff().dropna()
                    gains = changes.where(changes > 0, 0)
                    losses = -changes.where(changes < 0, 0)
                    avg_gain = gains.rolling(14).mean().iloc[-2]
                    avg_loss = losses.rolling(14).mean().iloc[-2]
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        pre_data["rsi"] = 100 - (100 / (1 + rs))
                    else:
                        pre_data["rsi"] = 100
        
        except Exception as e:
            logger.debug(f"Error getting pre-explosion data for {symbol}: {e}")
        
        return pre_data
    
    async def _discover_patterns(self) -> List[Dict]:
        """Discover new patterns from accumulated data"""
        patterns = []
        
        if self.signal_learner:
            try:
                new_patterns = await self.signal_learner.discover_patterns(min_samples=10)
                if new_patterns:
                    patterns.extend(new_patterns)
            except:
                pass
        
        return patterns
    
    def _save_learning_report(self, results: Dict):
        """Save learning report to file"""
        report_file = DATA_DIR / "learning_reports" / f"report_{datetime.now().strftime('%Y%m%d')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)


# Singleton
_daily_learner = None

def get_daily_learner() -> DailyLearner:
    global _daily_learner
    if _daily_learner is None:
        _daily_learner = DailyLearner()
    return _daily_learner


async def run_daily_learning():
    """Entry point for daily learning"""
    learner = get_daily_learner()
    return await learner.run_daily_learning()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(run_daily_learning())
    print(json.dumps(results, indent=2))
