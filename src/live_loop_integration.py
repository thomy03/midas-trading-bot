"""
Live Loop Integration - Midas Trading Bot
==========================================
Integration file for connecting the broker modules with the live trading loop.

This shows how to modify live_loop.py to use the IB broker or paper trader.
"""

import os
import asyncio
import logging
from typing import Optional
from datetime import datetime

# Import brokers
from brokers import (
    get_broker,
    OrderType,
    OrderSide,
    IB_AVAILABLE
)

# Import BULL optimizations
from scoring.bull_optimizer import (
    MarketRegime,
    ATRTrailingStop,
    TrailingStopConfig,
    PullbackFilter,
    calculate_trend_strength,
    calculate_atr
)
from agents.adaptive_scorer import AdaptiveScorerBullPatch

logger = logging.getLogger(__name__)


class LiveTradingLoop:
    """
    Main live trading loop with broker integration
    
    Supports both paper trading and live IB trading.
    """
    
    def __init__(
        self,
        trading_mode: str = "paper",  # "paper" or "live"
        symbols: list = None,
        check_interval: int = 60,  # seconds
    ):
        self.trading_mode = trading_mode
        self.symbols = symbols or []
        self.check_interval = check_interval
        
        # Initialize broker
        self.broker = get_broker(mode=trading_mode)
        
        # Initialize BULL optimizations
        self.bull_scorer = AdaptiveScorerBullPatch()
        self.position_trackers = {}  # symbol -> ATRTrailingStop
        
        # State
        self.running = False
        self.current_regime = "SIDEWAYS"
        
    async def start(self):
        """Start the live trading loop"""
        logger.info(f"Starting live trading loop in {self.trading_mode} mode")
        
        # Connect to broker
        if not await self.broker.connect():
            logger.error("Failed to connect to broker")
            return
        
        self.running = True
        
        try:
            while self.running:
                await self._trading_cycle()
                await asyncio.sleep(self.check_interval)
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            await self.broker.disconnect()
    
    async def stop(self):
        """Stop the trading loop"""
        self.running = False
        logger.info("Stopping trading loop")
    
    async def _trading_cycle(self):
        """Execute one trading cycle"""
        logger.debug("Running trading cycle")
        
        # 1. Get current positions
        positions = await self.broker.get_positions()
        current_symbols = {p.symbol for p in positions}
        
        # 2. Check existing positions for exit signals
        for position in positions:
            await self._check_position_exit(position)
        
        # 3. Check for new entry signals
        for symbol in self.symbols:
            if symbol not in current_symbols:
                await self._check_entry_signal(symbol)
        
        # 4. Update regime detection
        await self._update_regime()
    
    async def _check_position_exit(self, position):
        """
        Check if we should exit a position
        
        Uses BULL-optimized exit logic with trailing stops and pullback filtering
        """
        symbol = position.symbol
        
        # Get current market data (implement your data fetching)
        current_price = await self._get_current_price(symbol)
        current_atr = await self._get_current_atr(symbol)
        trend_strength = await self._get_trend_strength(symbol)
        momentum_score = await self._get_momentum_score(symbol)
        
        # Initialize tracker if needed
        if symbol not in self.position_trackers:
            tracker = ATRTrailingStop(TrailingStopConfig())
            tracker.initialize(position.avg_cost, current_atr)
            self.position_trackers[symbol] = tracker
        
        tracker = self.position_trackers[symbol]
        
        # Update trailing stop
        stop_price, stop_hit = tracker.update(current_price, current_atr)
        
        # Get regime parameters
        params = self.bull_scorer.get_regime_parameters(self.current_regime)
        
        # Check pullback filter
        pullback_filter = PullbackFilter(
            tolerance=params.get('pullback_tolerance', 0.02)
        )
        
        # Determine if should exit
        should_exit = False
        exit_reason = ""
        
        if stop_hit:
            should_exit = True
            exit_reason = f"Trailing stop hit at {stop_price:.2f}"
        
        # Check if in BULL with good momentum - be more patient
        if self.current_regime == "BULL" and not should_exit:
            current_pnl = (current_price - position.avg_cost) / position.avg_cost
            
            if current_pnl > 0 and momentum_score > 0.6 and trend_strength > 0.6:
                logger.info(f"{symbol}: Holding in BULL - good momentum/trend")
                return  # Don't exit
        
        if should_exit:
            await self._execute_exit(symbol, position.quantity, exit_reason)
    
    async def _check_entry_signal(self, symbol: str):
        """
        Check if we should enter a new position
        
        Uses BULL-optimized scoring and entry logic
        """
        # Get market data and scores (implement your data fetching)
        pillar_scores = await self._get_pillar_scores(symbol)
        trend_strength = await self._get_trend_strength(symbol)
        momentum_score = pillar_scores.get('momentum', 0.5)
        
        # Get current positions count
        positions = await self.broker.get_positions()
        position_count = len(positions)
        
        # Calculate enhanced score
        result = self.bull_scorer.calculate_enhanced_score(
            pillar_scores=pillar_scores,
            regime=self.current_regime,
            trend_strength=trend_strength,
            momentum_score=momentum_score
        )
        
        logger.info(f"{symbol}: Score={result.total_score:.2f}, Regime={self.current_regime}, "
                    f"Recommendation={result.recommendation}")
        
        # Check if should enter
        should_enter, reason = self.bull_scorer.should_enter_trade(
            score=result.total_score,
            regime=self.current_regime,
            trend_strength=trend_strength,
            current_positions=position_count
        )
        
        if should_enter and result.recommendation in ["BUY", "STRONG_BUY"]:
            await self._execute_entry(symbol, result, reason)
    
    async def _execute_entry(self, symbol: str, score_result, reason: str):
        """Execute a buy order"""
        # Get buying power and calculate position size
        buying_power = await self.broker.get_buying_power()
        current_price = await self._get_current_price(symbol)
        
        # Position sizing (example: 5% of capital per position)
        max_position_value = buying_power * 0.05
        quantity = int(max_position_value / current_price)
        
        if quantity < 1:
            logger.warning(f"Insufficient buying power for {symbol}")
            return
        
        # Get adjusted target
        params = self.bull_scorer.get_regime_parameters(self.current_regime)
        target_multiplier = params.get('target_multiplier', 1.0)
        
        logger.info(f"Entering {symbol}: {quantity} shares @ ~{current_price:.2f} "
                    f"(Reason: {reason}, Target mult: {target_multiplier:.2f}x)")
        
        # Place order
        result = await self.broker.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
            wait_for_fill=True
        )
        
        if result.success:
            logger.info(f"Bought {symbol}: {result.filled_quantity} @ {result.filled_price:.2f}")
            
            # Initialize position tracker
            current_atr = await self._get_current_atr(symbol)
            tracker = ATRTrailingStop(TrailingStopConfig(
                atr_multiplier=params.get('trailing_stop_atr', 2.0)
            ))
            tracker.initialize(result.filled_price, current_atr)
            self.position_trackers[symbol] = tracker
        else:
            logger.error(f"Failed to buy {symbol}: {result.message}")
    
    async def _execute_exit(self, symbol: str, quantity: float, reason: str):
        """Execute a sell order"""
        logger.info(f"Exiting {symbol}: {quantity} shares (Reason: {reason})")
        
        result = await self.broker.place_order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=OrderType.MARKET,
            wait_for_fill=True
        )
        
        if result.success:
            logger.info(f"Sold {symbol}: {result.filled_quantity} @ {result.filled_price:.2f}")
            
            # Clean up tracker
            if symbol in self.position_trackers:
                del self.position_trackers[symbol]
        else:
            logger.error(f"Failed to sell {symbol}: {result.message}")
    
    # =========================================================================
    # Data fetching methods - IMPLEMENT THESE based on your data source
    # =========================================================================
    
    async def _get_current_price(self, symbol: str) -> float:
        """Get current price for symbol - IMPLEMENT"""
        # TODO: Implement with your data source
        # Example: return self.data_feed.get_price(symbol)
        raise NotImplementedError("Implement _get_current_price")
    
    async def _get_current_atr(self, symbol: str) -> float:
        """Get current ATR for symbol - IMPLEMENT"""
        # TODO: Implement with your data source
        raise NotImplementedError("Implement _get_current_atr")
    
    async def _get_trend_strength(self, symbol: str) -> float:
        """Get trend strength for symbol - IMPLEMENT"""
        # TODO: Implement with your data source
        raise NotImplementedError("Implement _get_trend_strength")
    
    async def _get_momentum_score(self, symbol: str) -> float:
        """Get momentum score for symbol - IMPLEMENT"""
        # TODO: Implement with your data source
        raise NotImplementedError("Implement _get_momentum_score")
    
    async def _get_pillar_scores(self, symbol: str) -> dict:
        """Get all pillar scores for symbol - IMPLEMENT"""
        # TODO: Implement with your scoring system
        raise NotImplementedError("Implement _get_pillar_scores")
    
    async def _update_regime(self):
        """Update market regime detection - IMPLEMENT"""
        # TODO: Implement your regime detection
        # self.current_regime = detect_regime(...)
        pass


# =========================================================================
# Environment Configuration
# =========================================================================
"""
Add to your .env file:

# Trading Mode
TRADING_MODE=paper  # or "live"

# Interactive Brokers (for live trading)
IB_HOST=127.0.0.1
IB_PORT=7497  # 7497=TWS paper, 7496=TWS live, 4001/4002=Gateway
IB_CLIENT_ID=1
IB_ACCOUNT=  # Leave empty for default
IB_READONLY=false

# Paper Trading
PAPER_INITIAL_CAPITAL=100000
PAPER_COMMISSION=1.0
PAPER_SLIPPAGE=0.001
"""


# =========================================================================
# Usage Example
# =========================================================================
async def main():
    """Example usage of the live trading loop"""
    
    # Configuration from environment
    trading_mode = os.getenv("TRADING_MODE", "paper")
    
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    loop = LiveTradingLoop(
        trading_mode=trading_mode,
        symbols=symbols,
        check_interval=300  # 5 minutes
    )
    
    # Run the loop
    await loop.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
