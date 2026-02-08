"""
Defensive Manager - Protects portfolio in BEAR/VOLATILE regimes.

Instead of short selling (complex for retail subscribers), uses:
1. Cash allocation: Reduce to 30% max invested in BEAR/VOLATILE
2. Tighter stops: 3-4% instead of 8%
3. Higher entry threshold: Score >= 85 instead of 75
4. Quality filter: Only mega/large caps in defensive regimes
5. Optional inverse ETF hedge: SH (Short S&P 500) for advanced users

Designed for a subscriber model where users copy long-only trades.
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DefensiveLevel(Enum):
    """How defensive should the portfolio be"""
    NONE = "none"           # Normal operation (BULL/RANGE)
    CAUTIOUS = "cautious"   # Slightly reduced risk
    DEFENSIVE = "defensive" # Significantly reduced risk (BEAR)
    MAXIMUM = "maximum"     # Maximum protection (VOLATILE)


@dataclass
class DefensiveConfig:
    """Configuration for each defensive level"""
    max_invested_pct: float         # Max % of capital invested
    min_score_threshold: float      # Minimum score to enter
    stop_loss_pct: float            # Tighter stops
    max_positions: int              # Fewer positions
    require_large_cap: bool         # Only large/mega caps
    min_market_cap: int             # Minimum market cap
    hedge_with_inverse_etf: bool    # SH/SDS allocation
    inverse_etf_allocation: float   # % to allocate to inverse ETF


# Defensive configurations by level
DEFENSIVE_CONFIGS = {
    DefensiveLevel.NONE: DefensiveConfig(
        max_invested_pct=1.0,
        min_score_threshold=75.0,
        stop_loss_pct=0.08,
        max_positions=20,
        require_large_cap=False,
        min_market_cap=100_000_000,
        hedge_with_inverse_etf=False,
        inverse_etf_allocation=0.0
    ),
    DefensiveLevel.CAUTIOUS: DefensiveConfig(
        max_invested_pct=0.70,
        min_score_threshold=78.0,
        stop_loss_pct=0.06,
        max_positions=15,
        require_large_cap=False,
        min_market_cap=500_000_000,
        hedge_with_inverse_etf=False,
        inverse_etf_allocation=0.0
    ),
    DefensiveLevel.DEFENSIVE: DefensiveConfig(
        max_invested_pct=0.30,
        min_score_threshold=85.0,
        stop_loss_pct=0.04,
        max_positions=8,
        require_large_cap=True,
        min_market_cap=2_000_000_000,
        hedge_with_inverse_etf=False,
        inverse_etf_allocation=0.0
    ),
    DefensiveLevel.MAXIMUM: DefensiveConfig(
        max_invested_pct=0.15,
        min_score_threshold=90.0,
        stop_loss_pct=0.03,
        max_positions=5,
        require_large_cap=True,
        min_market_cap=10_000_000_000,
        hedge_with_inverse_etf=False,
        inverse_etf_allocation=0.0
    )
}


class DefensiveManager:
    """
    Manages portfolio defensiveness based on market regime.

    Usage:
        dm = DefensiveManager()
        level = dm.get_defensive_level(regime='BEAR', vix=25)
        config = dm.get_config(level)

        # Use config to filter trades
        if score >= config.min_score_threshold:
            # Take trade
        if total_invested > capital * config.max_invested_pct:
            # Don't add more positions
    """

    def __init__(self, enable_inverse_etf: bool = False):
        """
        Args:
            enable_inverse_etf: If True, allow inverse ETF hedging
                               (advanced feature, disabled by default for subscribers)
        """
        self.enable_inverse_etf = enable_inverse_etf
        self.current_level = DefensiveLevel.NONE
        self._level_history = []

    def get_defensive_level(
        self,
        regime: str,
        vix_level: float = 20.0,
        drawdown_pct: float = 0.0,
        consecutive_losses: int = 0
    ) -> DefensiveLevel:
        """
        Determine defensive level based on market conditions.

        Args:
            regime: Market regime ('BULL', 'BEAR', 'RANGE', 'VOLATILE')
            vix_level: Current VIX level
            drawdown_pct: Current portfolio drawdown (0-1)
            consecutive_losses: Number of consecutive losing trades

        Returns:
            DefensiveLevel
        """
        regime = regime.upper()

        # Base level from regime
        if regime == 'VOLATILE' or vix_level > 35:
            level = DefensiveLevel.MAXIMUM
        elif regime == 'BEAR' or vix_level > 25:
            level = DefensiveLevel.DEFENSIVE
        elif regime == 'RANGE' and vix_level > 20:
            level = DefensiveLevel.CAUTIOUS
        else:
            level = DefensiveLevel.NONE

        # Escalate based on portfolio state
        if drawdown_pct > 0.15:  # 15% drawdown -> emergency
            level = DefensiveLevel.MAXIMUM
        elif drawdown_pct > 0.10:  # 10% drawdown -> defensive
            if level.value < DefensiveLevel.DEFENSIVE.value:
                level = DefensiveLevel.DEFENSIVE
        elif consecutive_losses >= 5:
            if level.value < DefensiveLevel.CAUTIOUS.value:
                level = DefensiveLevel.CAUTIOUS

        if level != self.current_level:
            logger.info(
                f"[DEFENSIVE] Level changed: {self.current_level.value} -> {level.value} "
                f"(regime={regime}, VIX={vix_level:.1f}, DD={drawdown_pct:.1%})"
            )
            self._level_history.append({
                'from': self.current_level.value,
                'to': level.value,
                'regime': regime,
                'vix': vix_level,
                'drawdown': drawdown_pct
            })

        self.current_level = level
        return level

    def get_config(self, level: DefensiveLevel = None) -> DefensiveConfig:
        """Get configuration for a defensive level."""
        level = level or self.current_level
        config = DEFENSIVE_CONFIGS[level]

        # Override inverse ETF setting based on initialization
        if not self.enable_inverse_etf:
            config = DefensiveConfig(
                max_invested_pct=config.max_invested_pct,
                min_score_threshold=config.min_score_threshold,
                stop_loss_pct=config.stop_loss_pct,
                max_positions=config.max_positions,
                require_large_cap=config.require_large_cap,
                min_market_cap=config.min_market_cap,
                hedge_with_inverse_etf=False,
                inverse_etf_allocation=0.0
            )

        return config

    def should_enter_trade(
        self,
        score: float,
        market_cap: Optional[int] = None,
        current_positions: int = 0,
        total_invested_pct: float = 0.0
    ) -> Tuple[bool, str]:
        """
        Check if a new trade should be entered given current defensive level.

        Returns:
            (allowed, reason)
        """
        config = self.get_config()

        # Check score threshold
        if score < config.min_score_threshold:
            return False, f"Score {score:.0f} < threshold {config.min_score_threshold:.0f} (defensive level: {self.current_level.value})"

        # Check max positions
        if current_positions >= config.max_positions:
            return False, f"Max positions reached ({config.max_positions}) in {self.current_level.value} mode"

        # Check invested percentage
        if total_invested_pct >= config.max_invested_pct:
            return False, f"Max invested {config.max_invested_pct:.0%} reached in {self.current_level.value} mode"

        # Check market cap
        if config.require_large_cap and market_cap is not None:
            if market_cap < config.min_market_cap:
                return False, f"Market cap ${market_cap/1e9:.1f}B too small for {self.current_level.value} mode (min ${config.min_market_cap/1e9:.0f}B)"

        return True, "OK"

    def get_adjusted_stop_loss(self, entry_price: float, base_stop_pct: float = 0.08) -> float:
        """Get stop loss adjusted for current defensive level."""
        config = self.get_config()
        # Use the tighter of base stop and defensive stop
        stop_pct = min(base_stop_pct, config.stop_loss_pct)
        return entry_price * (1 - stop_pct)

    def get_subscriber_message(self) -> str:
        """Generate a message for subscribers about current market stance."""
        config = self.get_config()
        level = self.current_level

        if level == DefensiveLevel.NONE:
            return "Market conditions: NORMAL. Full allocation active."
        elif level == DefensiveLevel.CAUTIOUS:
            return (f"Market conditions: CAUTIOUS. Reduced to {config.max_invested_pct:.0%} allocation, "
                    f"tighter stops at {config.stop_loss_pct:.0%}.")
        elif level == DefensiveLevel.DEFENSIVE:
            return (f"Market conditions: DEFENSIVE. Only {config.max_invested_pct:.0%} invested, "
                    f"large caps only, stops at {config.stop_loss_pct:.0%}. "
                    f"Consider holding more cash.")
        else:
            return (f"Market conditions: MAXIMUM PROTECTION. Only {config.max_invested_pct:.0%} invested, "
                    f"mega caps only, very tight stops at {config.stop_loss_pct:.0%}. "
                    f"Strongly consider staying in cash.")

    def get_stats(self) -> Dict:
        """Get defensive manager statistics."""
        return {
            'current_level': self.current_level.value,
            'level_changes': len(self._level_history),
            'history': self._level_history[-10:]  # Last 10 changes
        }


# Singleton
_defensive_manager = None


def get_defensive_manager(enable_inverse_etf: bool = False) -> DefensiveManager:
    """Get singleton defensive manager."""
    global _defensive_manager
    if _defensive_manager is None:
        _defensive_manager = DefensiveManager(enable_inverse_etf=enable_inverse_etf)
    return _defensive_manager
