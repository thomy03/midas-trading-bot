"""
BULL Regime Optimizer - Midas Trading Bot
==========================================
Optimizations for BULL market regime to improve win rate from 53.5% to 60%+

Key strategies:
1. ATR-based trailing stops instead of fixed stops
2. Extended hold times with +20% targets in BULL
3. Pullback filtering to avoid premature exits
4. Momentum/trend weighted scoring
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    VOLATILE = "VOLATILE"
    SIDEWAYS = "SIDEWAYS"


@dataclass
class TrailingStopConfig:
    """Configuration for ATR-based trailing stops"""
    atr_multiplier: float = 2.0  # Stop distance = ATR * multiplier
    min_profit_to_activate: float = 0.02  # Activate trailing after 2% profit
    tighten_at_profit: Dict[float, float] = None  # {profit_level: new_multiplier}
    
    def __post_init__(self):
        if self.tighten_at_profit is None:
            # Progressively tighten stops as profit increases
            self.tighten_at_profit = {
                0.05: 1.8,   # At 5% profit, tighten to 1.8x ATR
                0.10: 1.5,   # At 10% profit, tighten to 1.5x ATR
                0.15: 1.2,   # At 15% profit, tighten to 1.2x ATR
            }


@dataclass
class BullRegimeConfig:
    """BULL-specific trading parameters"""
    target_multiplier: float = 1.20  # +20% targets in BULL
    max_hold_days: int = 15  # Extended hold period (vs 10 in normal)
    pullback_tolerance: float = 0.03  # Tolerate 3% pullback before exit signal
    momentum_weight_boost: float = 1.3  # 30% more weight on momentum
    trend_weight_boost: float = 1.25  # 25% more weight on trend
    min_trend_strength: float = 0.6  # Require stronger trend confirmation


class ATRTrailingStop:
    """
    Adaptive trailing stop using ATR (Average True Range)
    
    In BULL regime, we want to let winners run longer while protecting profits.
    Fixed stops often exit too early in strong trends.
    """
    
    def __init__(self, config: TrailingStopConfig = None):
        self.config = config or TrailingStopConfig()
        self.highest_price = None
        self.trailing_stop_price = None
        self.is_active = False
        
    def initialize(self, entry_price: float, current_atr: float):
        """Initialize stop at entry"""
        self.entry_price = entry_price
        self.highest_price = entry_price
        self.current_atr = current_atr
        self.trailing_stop_price = entry_price - (current_atr * self.config.atr_multiplier)
        self.is_active = False
        
    def update(self, current_price: float, current_atr: float) -> Tuple[float, bool]:
        """
        Update trailing stop based on current price and ATR
        
        Returns:
            Tuple[float, bool]: (stop_price, should_exit)
        """
        self.current_atr = current_atr
        profit_pct = (current_price - self.entry_price) / self.entry_price
        
        # Activate trailing stop after minimum profit reached
        if profit_pct >= self.config.min_profit_to_activate:
            self.is_active = True
        
        # Update highest price
        if current_price > self.highest_price:
            self.highest_price = current_price
            
            # Determine ATR multiplier based on profit level
            multiplier = self.config.atr_multiplier
            for profit_threshold, new_mult in sorted(
                self.config.tighten_at_profit.items(), reverse=True
            ):
                if profit_pct >= profit_threshold:
                    multiplier = new_mult
                    break
            
            # Update trailing stop
            if self.is_active:
                new_stop = self.highest_price - (current_atr * multiplier)
                self.trailing_stop_price = max(self.trailing_stop_price, new_stop)
        
        # Check if stop is hit
        should_exit = current_price <= self.trailing_stop_price
        
        return self.trailing_stop_price, should_exit


class PullbackFilter:
    """
    Filter for avoiding premature exits during temporary pullbacks
    
    In strong BULL trends, small pullbacks are normal and often followed
    by continuation. This filter prevents selling on noise.
    """
    
    def __init__(self, tolerance: float = 0.03, confirmation_bars: int = 2):
        self.tolerance = tolerance  # Max pullback before concern
        self.confirmation_bars = confirmation_bars  # Bars to confirm exit
        self.pullback_start_price = None
        self.bars_below_tolerance = 0
        
    def should_exit(
        self, 
        current_price: float, 
        peak_price: float, 
        trend_strength: float,
        regime: MarketRegime
    ) -> bool:
        """
        Determine if we should exit or hold through pullback
        
        Args:
            current_price: Current asset price
            peak_price: Highest price since entry
            trend_strength: 0-1 indicator of trend strength
            regime: Current market regime
            
        Returns:
            bool: True if should exit, False if should hold
        """
        drawdown = (peak_price - current_price) / peak_price
        
        # In BULL with strong trend, be more tolerant
        adjusted_tolerance = self.tolerance
        if regime == MarketRegime.BULL and trend_strength > 0.7:
            adjusted_tolerance *= 1.5  # 50% more tolerance
        
        if drawdown <= adjusted_tolerance:
            # Within tolerance, reset counter
            self.bars_below_tolerance = 0
            return False
        
        # Beyond tolerance
        self.bars_below_tolerance += 1
        
        # Require confirmation before exit
        if self.bars_below_tolerance >= self.confirmation_bars:
            return True
            
        return False


class BullOptimizedScorer:
    """
    Adaptive scorer with BULL regime optimizations
    
    Adjusts pillar weights and scoring logic based on market regime
    to improve win rate in trending markets.
    """
    
    # Base pillar weights
    BASE_WEIGHTS = {
        'technical': 0.25,
        'fundamental': 0.20,
        'sentiment': 0.15,
        'momentum': 0.20,
        'trend': 0.20,
    }
    
    def __init__(self, regime_config: BullRegimeConfig = None):
        self.regime_config = regime_config or BullRegimeConfig()
        self.weights = self.BASE_WEIGHTS.copy()
        
    def get_regime_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Get adjusted weights based on market regime"""
        weights = self.BASE_WEIGHTS.copy()
        
        if regime == MarketRegime.BULL:
            # Boost momentum and trend weights
            weights['momentum'] *= self.regime_config.momentum_weight_boost
            weights['trend'] *= self.regime_config.trend_weight_boost
            
            # Slightly reduce sentiment (often lagging in trends)
            weights['sentiment'] *= 0.8
            
        elif regime == MarketRegime.VOLATILE:
            # In volatile, technical signals are more important
            weights['technical'] *= 1.2
            weights['momentum'] *= 0.9
            
        elif regime == MarketRegime.BEAR:
            # In bear, be more cautious
            weights['fundamental'] *= 1.2
            weights['sentiment'] *= 1.1
            weights['momentum'] *= 0.8
            
        # Normalize weights to sum to 1
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
    
    def calculate_target(
        self, 
        base_target: float, 
        regime: MarketRegime,
        trend_strength: float
    ) -> float:
        """
        Calculate adjusted price target based on regime
        
        In BULL regime with strong trends, extend targets by 20%+
        """
        if regime == MarketRegime.BULL:
            # Base multiplier from config
            multiplier = self.regime_config.target_multiplier
            
            # Extra boost for very strong trends
            if trend_strength > 0.8:
                multiplier *= 1.1  # Additional 10%
                
            return base_target * multiplier
            
        elif regime == MarketRegime.VOLATILE:
            # Slightly reduce targets in volatile markets
            return base_target * 0.95
            
        return base_target
    
    def should_hold(
        self,
        current_pnl_pct: float,
        days_held: int,
        regime: MarketRegime,
        trend_strength: float,
        momentum_score: float
    ) -> Tuple[bool, str]:
        """
        Determine if position should be held longer
        
        In BULL regime, be more patient with profitable positions
        
        Returns:
            Tuple[bool, str]: (should_hold, reason)
        """
        max_days = self.regime_config.max_hold_days if regime == MarketRegime.BULL else 10
        
        # Always exit if max hold reached
        if days_held >= max_days:
            return False, f"Max hold days ({max_days}) reached"
        
        # In BULL with profit and strong momentum, hold
        if regime == MarketRegime.BULL:
            if current_pnl_pct > 0 and momentum_score > 0.6:
                return True, "BULL regime: positive momentum, holding"
            
            if trend_strength > self.regime_config.min_trend_strength:
                if current_pnl_pct > 0.02:  # At least 2% profit
                    return True, "BULL regime: strong trend, holding winner"
        
        # Default: don't override
        return False, ""


def calculate_trend_strength(prices: np.ndarray, period: int = 20) -> float:
    """
    Calculate trend strength indicator (0-1)
    
    Uses combination of:
    - Price above/below moving average
    - ADX-like directional movement
    - Higher highs/higher lows pattern
    """
    if len(prices) < period:
        return 0.5
    
    recent = prices[-period:]
    ma = np.mean(recent)
    current = prices[-1]
    
    # Price vs MA component
    price_ma_score = 0.5 + min(0.5, max(-0.5, (current - ma) / ma * 5))
    
    # Trend direction component
    start_price = recent[0]
    end_price = recent[-1]
    direction_score = 0.5 + min(0.5, max(-0.5, (end_price - start_price) / start_price * 5))
    
    # Higher highs pattern
    mid = len(recent) // 2
    first_half_high = np.max(recent[:mid])
    second_half_high = np.max(recent[mid:])
    hh_score = 1.0 if second_half_high > first_half_high else 0.3
    
    # Combine
    trend_strength = (price_ma_score * 0.35 + direction_score * 0.35 + hh_score * 0.30)
    
    return np.clip(trend_strength, 0, 1)


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """Calculate Average True Range"""
    if len(high) < period + 1:
        return (high[-1] - low[-1])  # Simple range if not enough data
    
    tr = np.zeros(len(high))
    
    for i in range(1, len(high)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    return np.mean(tr[-period:])


# Export main classes for integration
__all__ = [
    'MarketRegime',
    'TrailingStopConfig', 
    'BullRegimeConfig',
    'ATRTrailingStop',
    'PullbackFilter',
    'BullOptimizedScorer',
    'calculate_trend_strength',
    'calculate_atr'
]
