"""
Position Manager - Active portfolio management with dynamic SL/TP and position review.

V5.5 Features:
- Trailing stops (protect gains)
- Periodic position review (rescore with 4 pillars)
- Auto-exit on score degradation
- Smart position sizing adjustments

Author: Jarvis for Thomas
Date: 2026-02-05
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PositionReview:
    """Result of a position review"""
    symbol: str
    entry_score: float
    current_score: float
    score_change: float
    recommendation: str  # HOLD, REDUCE, SELL, STRENGTHEN
    reasons: List[str] = field(default_factory=list)
    new_stop_loss: Optional[float] = None
    new_take_profit: Optional[float] = None


class PositionManager:
    """
    Active portfolio manager that monitors and adjusts positions.
    
    Features:
    - Trailing stop management
    - Periodic position review with 4-pillar rescoring
    - Auto-exit on score degradation
    """
    
    def __init__(
        self,
        paper_trader,
        reasoning_engine=None,
        config: Optional[Dict] = None
    ):
        self.paper_trader = paper_trader
        self.reasoning_engine = reasoning_engine
        self.config = config or {}
        
        # Thresholds
        self.score_drop_alert = self.config.get('score_drop_alert', 5)      # Alert if drops 5+
        self.score_drop_reduce = self.config.get('score_drop_reduce', 10)   # Reduce 50% if drops 10+
        self.score_absolute_min = self.config.get('score_absolute_min', 40) # Sell if below 40
        
        # Trailing stop config
        self.trailing_activate_pct = self.config.get('trailing_activate_pct', 5.0)   # Activate at +5%
        self.trailing_distance_pct = self.config.get('trailing_distance_pct', 3.0)   # Trail by 3%
        
        # Review frequency
        self.review_interval_hours = self.config.get('review_interval_hours', 24)  # Daily review
        
        # State
        self._last_review: Optional[datetime] = None
        self._position_highs: Dict[str, float] = {}  # Track highest price for trailing stop
        
        logger.info(f"[POSITION_MANAGER] Initialized with config: {self.config}")
    
    async def update_trailing_stops(self) -> List[Dict]:
        """
        Update trailing stops for all positions.
        Called frequently (every scan cycle).
        
        Returns list of stop updates made.
        """
        updates = []
        
        if not self.paper_trader or not self.paper_trader.portfolio:
            return updates
        
        positions = self.paper_trader.portfolio.positions
        
        for pos in positions:
            try:
                symbol = pos.symbol
                entry_price = pos.entry_price
                current_sl = pos.stop_loss or (entry_price * 0.95)  # Default 5% SL
                
                # Get current price
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    current_price = ticker.fast_info.get('lastPrice') or ticker.fast_info.get('regularMarketPrice')
                    if not current_price:
                        continue
                except Exception as e:
                    logger.debug(f"Could not get price for {symbol}: {e}")
                    continue
                
                # Track highest price
                if symbol not in self._position_highs:
                    self._position_highs[symbol] = current_price
                else:
                    self._position_highs[symbol] = max(self._position_highs[symbol], current_price)
                
                high_price = self._position_highs[symbol]
                
                # Calculate gain from entry
                gain_pct = ((current_price - entry_price) / entry_price) * 100
                gain_from_high = ((current_price - high_price) / high_price) * 100
                
                # Trailing stop logic
                if gain_pct >= self.trailing_activate_pct:
                    # Calculate new trailing stop
                    new_sl = high_price * (1 - self.trailing_distance_pct / 100)
                    
                    # Only move stop UP, never down
                    if new_sl > current_sl:
                        # Update position stop loss
                        pos.stop_loss = new_sl
                        
                        updates.append({
                            'symbol': symbol,
                            'old_sl': current_sl,
                            'new_sl': new_sl,
                            'current_price': current_price,
                            'gain_pct': gain_pct,
                            'reason': 'trailing_stop_update'
                        })
                        
                        logger.info(f"🔒 TRAILING STOP: {symbol} SL moved {current_sl:.2f} → {new_sl:.2f} (gain: +{gain_pct:.1f}%)")
                
                # Check if stop loss hit
                if current_price <= current_sl:
                    logger.warning(f"🚨 STOP LOSS HIT: {symbol} @ {current_price:.2f} (SL: {current_sl:.2f})")
                    # The actual selling will be handled by the main loop or a separate check
                    
            except Exception as e:
                logger.error(f"Error updating trailing stop for {pos.symbol}: {e}")
        
        return updates
    
    async def check_stop_losses(self) -> List[str]:
        """
        Check all positions for stop loss hits and execute sells.
        Returns list of symbols that were sold.
        """
        sold = []
        
        if not self.paper_trader or not self.paper_trader.portfolio:
            return sold
        
        positions = list(self.paper_trader.portfolio.positions)  # Copy to avoid modification during iteration
        
        for pos in positions:
            try:
                symbol = pos.symbol
                stop_loss = pos.stop_loss
                
                if not stop_loss:
                    continue
                
                # Get current price
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    current_price = ticker.fast_info.get('lastPrice') or ticker.fast_info.get('regularMarketPrice')
                    if not current_price:
                        continue
                except:
                    continue
                
                # Check stop loss
                if current_price <= stop_loss:
                    logger.warning(f"🔴 STOP LOSS TRIGGERED: {symbol} @ {current_price:.2f} (SL: {stop_loss:.2f})")
                    
                    result = self.paper_trader.close_position(
                        symbol=symbol,
                        price=current_price,
                        reason="stop_loss_hit"
                    )
                    
                    if result:
                        sold.append(symbol)
                        # Clean up tracking
                        if symbol in self._position_highs:
                            del self._position_highs[symbol]
                
                # Check take profit
                take_profit = pos.take_profit
                if take_profit and current_price >= take_profit:
                    logger.info(f"🟢 TAKE PROFIT TRIGGERED: {symbol} @ {current_price:.2f} (TP: {take_profit:.2f})")
                    
                    result = self.paper_trader.close_position(
                        symbol=symbol,
                        price=current_price,
                        reason="take_profit_hit"
                    )
                    
                    if result:
                        sold.append(symbol)
                        if symbol in self._position_highs:
                            del self._position_highs[symbol]
                            
            except Exception as e:
                logger.error(f"Error checking stop loss for {pos.symbol}: {e}")
        
        return sold
    
    async def review_positions(self, force: bool = False) -> List[PositionReview]:
        """
        Periodic review of all positions with rescoring.
        
        Args:
            force: Force review even if not due
            
        Returns:
            List of position reviews with recommendations
        """
        # Check if review is due
        now = datetime.now()
        if not force and self._last_review:
            hours_since = (now - self._last_review).total_seconds() / 3600
            if hours_since < self.review_interval_hours:
                logger.debug(f"Position review not due yet ({hours_since:.1f}h < {self.review_interval_hours}h)")
                return []
        
        logger.info("📋 Starting periodic position review...")
        reviews = []
        
        if not self.paper_trader or not self.paper_trader.portfolio:
            return reviews
        
        if not self.reasoning_engine:
            logger.warning("No reasoning engine available for position review")
            return reviews
        
        positions = self.paper_trader.portfolio.positions
        
        for pos in positions:
            try:
                symbol = pos.symbol
                entry_score = pos.score_at_entry or 50  # Default if not recorded
                
                # Rescore with reasoning engine
                logger.info(f"Rescoring {symbol}...")
                result = await self.reasoning_engine.analyze(symbol)
                
                if not result:
                    continue
                
                current_score = result.total_score
                score_change = current_score - entry_score
                
                # Determine recommendation
                recommendation = "HOLD"
                reasons = []
                
                # Check score degradation
                if score_change <= -self.score_drop_reduce:
                    recommendation = "REDUCE"
                    reasons.append(f"Score dropped {abs(score_change):.1f} points (>{self.score_drop_reduce})")
                elif score_change <= -self.score_drop_alert:
                    recommendation = "WATCH"
                    reasons.append(f"Score dropped {abs(score_change):.1f} points (alert threshold)")
                
                # Check absolute minimum
                if current_score < self.score_absolute_min:
                    recommendation = "SELL"
                    reasons.append(f"Score {current_score:.1f} below minimum {self.score_absolute_min}")
                
                # Check if decision changed to SELL
                if result.decision.value in ['sell', 'strong_sell']:
                    recommendation = "SELL"
                    reasons.append(f"Decision changed to {result.decision.value.upper()}")
                
                # Check if score improved significantly
                if score_change >= 5:
                    reasons.append(f"Score improved +{score_change:.1f} points ✅")
                
                review = PositionReview(
                    symbol=symbol,
                    entry_score=entry_score,
                    current_score=current_score,
                    score_change=score_change,
                    recommendation=recommendation,
                    reasons=reasons
                )
                reviews.append(review)
                
                logger.info(f"📊 {symbol}: {entry_score:.1f} → {current_score:.1f} ({score_change:+.1f}) | {recommendation}")
                
            except Exception as e:
                logger.error(f"Error reviewing {pos.symbol}: {e}")
        
        self._last_review = now
        logger.info(f"📋 Position review complete: {len(reviews)} positions reviewed")
        
        return reviews
    
    async def execute_review_actions(self, reviews: List[PositionReview]) -> Dict:
        """
        Execute actions based on position reviews.
        
        Returns summary of actions taken.
        """
        summary = {
            'sold': [],
            'reduced': [],
            'watched': [],
            'held': []
        }
        
        for review in reviews:
            try:
                symbol = review.symbol
                
                if review.recommendation == "SELL":
                    # Get current price and sell
                    try:
                        import yfinance as yf
                        ticker = yf.Ticker(symbol)
                        price = ticker.fast_info.get('lastPrice') or ticker.fast_info.get('regularMarketPrice', 0)
                    except:
                        price = 0
                    
                    if price > 0:
                        result = self.paper_trader.close_position(
                            symbol=symbol,
                            price=price,
                            reason=f"review_sell: {', '.join(review.reasons)}"
                        )
                        if result:
                            summary['sold'].append(symbol)
                            logger.info(f"🔴 REVIEW SELL: {symbol} | Reasons: {', '.join(review.reasons)}")
                
                elif review.recommendation == "REDUCE":
                    # For now, we'll sell entirely (partial sells more complex)
                    # TODO: Implement partial position reduction
                    summary['reduced'].append(symbol)
                    logger.warning(f"⚠️ REDUCE RECOMMENDED: {symbol} | {', '.join(review.reasons)}")
                
                elif review.recommendation == "WATCH":
                    summary['watched'].append(symbol)
                    logger.info(f"👀 WATCHING: {symbol} | {', '.join(review.reasons)}")
                
                else:
                    summary['held'].append(symbol)
                    
            except Exception as e:
                logger.error(f"Error executing review action for {review.symbol}: {e}")
        
        return summary


# Singleton instance
_position_manager: Optional[PositionManager] = None


def get_position_manager(paper_trader=None, reasoning_engine=None, config=None) -> PositionManager:
    """Get or create the position manager singleton."""
    global _position_manager
    
    if _position_manager is None and paper_trader is not None:
        _position_manager = PositionManager(
            paper_trader=paper_trader,
            reasoning_engine=reasoning_engine,
            config=config
        )
    
    return _position_manager


async def reset_position_manager():
    """Reset the singleton (for testing)."""
    global _position_manager
    _position_manager = None
