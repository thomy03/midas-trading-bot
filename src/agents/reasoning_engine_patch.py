"""
Reasoning Engine Patch for BULL Optimization
=============================================
This file contains patches to apply to reasoning_engine.py

Apply by adding these methods/modifications to the existing ReasoningEngine class.
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# Import from the optimizer module
import sys
sys.path.append('..')
from scoring.bull_optimizer import (
    MarketRegime,
    ATRTrailingStop,
    TrailingStopConfig,
    PullbackFilter,
    BullOptimizedScorer,
    BullRegimeConfig,
    calculate_trend_strength,
    calculate_atr
)


@dataclass
class PositionState:
    """Track state for active positions"""
    symbol: str
    entry_price: float
    entry_date: str
    trailing_stop: ATRTrailingStop
    pullback_filter: PullbackFilter
    peak_price: float
    days_held: int = 0


class ReasoningEngineBullPatch:
    """
    Patches for ReasoningEngine to optimize BULL regime performance
    
    To apply: Add these methods to your existing ReasoningEngine class
    or inherit from it.
    """
    
    def __init__(self):
        # Initialize BULL optimization components
        self.bull_scorer = BullOptimizedScorer()
        self.position_states: Dict[str, PositionState] = {}
        self.trailing_stop_config = TrailingStopConfig()
        
    def initialize_position_tracking(
        self, 
        symbol: str, 
        entry_price: float,
        entry_date: str,
        current_atr: float
    ):
        """
        Initialize tracking for a new position
        
        Call this when opening a new position.
        """
        trailing_stop = ATRTrailingStop(self.trailing_stop_config)
        trailing_stop.initialize(entry_price, current_atr)
        
        self.position_states[symbol] = PositionState(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=entry_date,
            trailing_stop=trailing_stop,
            pullback_filter=PullbackFilter(tolerance=0.03, confirmation_bars=2),
            peak_price=entry_price
        )
    
    def update_position_tracking(
        self,
        symbol: str,
        current_price: float,
        current_atr: float
    ) -> Tuple[float, bool]:
        """
        Update position tracking and check if should exit
        
        Returns:
            Tuple[stop_price, should_exit]
        """
        if symbol not in self.position_states:
            return 0.0, False
        
        state = self.position_states[symbol]
        state.days_held += 1
        
        # Update peak price
        if current_price > state.peak_price:
            state.peak_price = current_price
        
        # Update trailing stop
        stop_price, stop_hit = state.trailing_stop.update(current_price, current_atr)
        
        return stop_price, stop_hit
    
    def should_exit_position_enhanced(
        self,
        symbol: str,
        current_price: float,
        current_atr: float,
        regime: str,
        trend_strength: float,
        momentum_score: float,
        original_exit_signal: bool
    ) -> Tuple[bool, str]:
        """
        Enhanced exit logic with BULL optimizations
        
        This method should be called instead of the original exit logic.
        It applies:
        1. ATR trailing stops
        2. Pullback filtering
        3. Extended hold in BULL with strong momentum
        
        Returns:
            Tuple[should_exit, reason]
        """
        if symbol not in self.position_states:
            return original_exit_signal, "No position tracking"
        
        state = self.position_states[symbol]
        market_regime = MarketRegime(regime) if regime in [r.value for r in MarketRegime] else MarketRegime.SIDEWAYS
        
        # Update tracking
        stop_price, stop_hit = self.update_position_tracking(symbol, current_price, current_atr)
        
        # Check trailing stop
        if stop_hit:
            self._cleanup_position(symbol)
            return True, f"Trailing stop hit at {stop_price:.2f}"
        
        # Check pullback filter
        should_exit_pullback = state.pullback_filter.should_exit(
            current_price=current_price,
            peak_price=state.peak_price,
            trend_strength=trend_strength,
            regime=market_regime
        )
        
        if should_exit_pullback and original_exit_signal:
            self._cleanup_position(symbol)
            return True, "Pullback confirmed, exiting"
        
        # Check if we should hold longer in BULL
        current_pnl_pct = (current_price - state.entry_price) / state.entry_price
        should_hold, hold_reason = self.bull_scorer.should_hold(
            current_pnl_pct=current_pnl_pct,
            days_held=state.days_held,
            regime=market_regime,
            trend_strength=trend_strength,
            momentum_score=momentum_score
        )
        
        if should_hold and not stop_hit:
            return False, hold_reason
        
        # If original signal says exit and no override, exit
        if original_exit_signal:
            self._cleanup_position(symbol)
            return True, "Original exit signal"
        
        return False, "Holding position"
    
    def _cleanup_position(self, symbol: str):
        """Clean up position state after exit"""
        if symbol in self.position_states:
            del self.position_states[symbol]
    
    def get_adjusted_target(
        self,
        base_target: float,
        regime: str,
        trend_strength: float
    ) -> float:
        """
        Get regime-adjusted price target
        
        In BULL regime with strong trends, targets are extended by 20%+
        """
        market_regime = MarketRegime(regime) if regime in [r.value for r in MarketRegime] else MarketRegime.SIDEWAYS
        return self.bull_scorer.calculate_target(base_target, market_regime, trend_strength)
    
    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """
        Get pillar weights adjusted for current regime
        
        In BULL, momentum and trend get more weight.
        """
        market_regime = MarketRegime(regime) if regime in [r.value for r in MarketRegime] else MarketRegime.SIDEWAYS
        return self.bull_scorer.get_regime_weights(market_regime)


# =========================================================================
# INTEGRATION EXAMPLE
# =========================================================================
"""
To integrate with existing reasoning_engine.py:

1. Import the patch class:
   from .reasoning_engine_patch import ReasoningEngineBullPatch

2. Add to ReasoningEngine.__init__:
   self.bull_patch = ReasoningEngineBullPatch()

3. In your entry logic, after opening a position:
   self.bull_patch.initialize_position_tracking(
       symbol=symbol,
       entry_price=entry_price,
       entry_date=date_str,
       current_atr=self._calculate_atr(symbol)
   )

4. Replace exit logic:
   # Instead of:
   # if should_exit_signal:
   #     execute_exit()
   
   # Use:
   should_exit, reason = self.bull_patch.should_exit_position_enhanced(
       symbol=symbol,
       current_price=current_price,
       current_atr=self._calculate_atr(symbol),
       regime=self.current_regime,
       trend_strength=self._calculate_trend_strength(symbol),
       momentum_score=self.momentum_scores[symbol],
       original_exit_signal=should_exit_signal
   )
   if should_exit:
       logger.info(f"Exiting {symbol}: {reason}")
       execute_exit()

5. For targets, use:
   adjusted_target = self.bull_patch.get_adjusted_target(
       base_target=original_target,
       regime=self.current_regime,
       trend_strength=trend_strength
   )

6. For scoring weights, use:
   weights = self.bull_patch.get_regime_weights(self.current_regime)
"""
