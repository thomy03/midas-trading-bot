"""
Position Sizer - Dynamic position sizing for Midas V7.

Three sizing methods:
1. Kelly Criterion (half-Kelly for safety)
2. Volatility-adjusted (ATR-based)
3. Confidence-scaled (based on pillar agreement)

Replaces the fixed 10% position sizing.
"""

import numpy as np
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    """Result from position sizing calculation."""
    base_size_pct: float        # Base position size as % of capital
    adjusted_size_pct: float    # After all adjustments
    method: str                 # Which method was primary
    kelly_fraction: float       # Kelly-suggested fraction
    vol_adjustment: float       # Volatility multiplier
    confidence_multiplier: float  # Confidence scaling
    defensive_multiplier: float   # Defensive regime scaling
    reason: str                 # Human-readable explanation


class PositionSizer:
    """Dynamic position sizing combining multiple methods."""

    def __init__(
        self,
        base_size_pct: float = 0.10,        # Default 10%
        max_size_pct: float = 0.20,          # Never more than 20%
        min_size_pct: float = 0.03,          # Never less than 3%
        target_risk_pct: float = 0.015,      # Target 1.5% portfolio risk per trade
        use_kelly: bool = True,
        use_vol_adjust: bool = True,
        use_confidence_scale: bool = True,
        kelly_fraction: float = 0.5,         # Half-Kelly
        lookback_trades: int = 100,          # Trades for Kelly calculation
    ):
        self.base_size_pct = base_size_pct
        self.max_size_pct = max_size_pct
        self.min_size_pct = min_size_pct
        self.target_risk_pct = target_risk_pct
        self.use_kelly = use_kelly
        self.use_vol_adjust = use_vol_adjust
        self.use_confidence_scale = use_confidence_scale
        self.kelly_fraction = kelly_fraction
        self.lookback_trades = lookback_trades

        # Trade history for Kelly calculation
        self._trade_results: List[float] = []

    def calculate_size(
        self,
        capital: float,
        price: float,
        atr_pct: float = 0.02,
        confidence_score: float = 65.0,
        regime: str = 'RANGE',
        defensive_max_invested: float = 1.0,
        current_invested_pct: float = 0.0
    ) -> SizingResult:
        """Calculate optimal position size.

        Args:
            capital: Current available capital
            price: Entry price of the stock
            atr_pct: ATR as percentage of price (e.g., 0.02 = 2%)
            confidence_score: Combined pillar score (0-100)
            regime: Market regime
            defensive_max_invested: Max invested % from DefensiveManager
            current_invested_pct: Currently invested fraction

        Returns:
            SizingResult with calculated position size
        """
        reasons = []

        # 1. Kelly Criterion
        kelly = self._kelly_size()
        if self.use_kelly and kelly > 0:
            base = kelly * self.kelly_fraction
            reasons.append(f"Kelly={kelly:.1%}, half-Kelly={base:.1%}")
        else:
            base = self.base_size_pct
            reasons.append(f"Fixed base={base:.1%}")

        # 2. Volatility adjustment
        if self.use_vol_adjust and atr_pct > 0:
            # Target risk / actual risk
            vol_adj = self.target_risk_pct / atr_pct
            vol_adj = np.clip(vol_adj, 0.5, 2.0)  # Limit adjustment range
            reasons.append(f"Vol adj={vol_adj:.2f} (ATR={atr_pct:.1%})")
        else:
            vol_adj = 1.0

        # 3. Confidence scaling
        if self.use_confidence_scale:
            if confidence_score >= 90:
                conf_mult = 1.5
            elif confidence_score >= 80:
                conf_mult = 1.2
            elif confidence_score >= 70:
                conf_mult = 1.0
            elif confidence_score >= 65:
                conf_mult = 0.8
            else:
                conf_mult = 0.6
            reasons.append(f"Confidence mult={conf_mult:.1f} (score={confidence_score:.0f})")
        else:
            conf_mult = 1.0

        # 4. Defensive regime scaling
        if regime in ('BEAR', 'VOLATILE'):
            def_mult = 0.5 if regime == 'VOLATILE' else 0.6
        elif regime == 'RANGE':
            def_mult = 0.8
        else:
            def_mult = 1.0
        reasons.append(f"Regime mult={def_mult:.1f} ({regime})")

        # 5. Available capacity check
        available_capacity = max(0, defensive_max_invested - current_invested_pct)

        # Combine
        adjusted = base * vol_adj * conf_mult * def_mult
        adjusted = np.clip(adjusted, self.min_size_pct, self.max_size_pct)

        # Don't exceed available capacity
        adjusted = min(adjusted, available_capacity)
        adjusted = max(adjusted, 0)

        return SizingResult(
            base_size_pct=base,
            adjusted_size_pct=adjusted,
            method='kelly' if self.use_kelly and kelly > 0 else 'fixed',
            kelly_fraction=kelly,
            vol_adjustment=vol_adj,
            confidence_multiplier=conf_mult,
            defensive_multiplier=def_mult,
            reason=' | '.join(reasons)
        )

    def _kelly_size(self) -> float:
        """Calculate Kelly criterion from recent trade history.

        f* = (p * b - q) / b
        where:
            p = win rate
            b = avg_win / avg_loss
            q = 1 - p
        """
        if len(self._trade_results) < 20:
            return 0.0

        recent = self._trade_results[-self.lookback_trades:]

        wins = [r for r in recent if r > 0]
        losses = [r for r in recent if r < 0]

        if not wins or not losses:
            return 0.0

        win_rate = len(wins) / len(recent)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss == 0:
            return 0.0

        b = avg_win / avg_loss
        kelly = (win_rate * b - (1 - win_rate)) / b

        return max(0, kelly)  # Never negative

    def add_trade_result(self, pnl_pct: float):
        """Record a trade result for Kelly calculation."""
        self._trade_results.append(pnl_pct)
        # Keep only recent history
        if len(self._trade_results) > self.lookback_trades * 2:
            self._trade_results = self._trade_results[-self.lookback_trades:]

    def calculate_shares(
        self,
        capital: float,
        price: float,
        sizing_result: SizingResult
    ) -> int:
        """Convert sizing result to number of shares."""
        position_value = capital * sizing_result.adjusted_size_pct
        shares = int(position_value / price)
        return max(0, shares)

    def get_stats(self) -> Dict:
        """Get sizing statistics."""
        kelly = self._kelly_size()
        return {
            'total_recorded_trades': len(self._trade_results),
            'current_kelly_fraction': round(kelly, 4),
            'half_kelly': round(kelly * self.kelly_fraction, 4),
            'base_size_pct': self.base_size_pct,
            'max_size_pct': self.max_size_pct,
            'min_size_pct': self.min_size_pct,
        }


# Singleton
_position_sizer = None


def get_position_sizer(**kwargs) -> PositionSizer:
    """Get singleton position sizer."""
    global _position_sizer
    if _position_sizer is None:
        _position_sizer = PositionSizer(**kwargs)
    return _position_sizer
