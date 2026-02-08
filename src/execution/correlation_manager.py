"""
Correlation Manager - Prevents portfolio concentration risk.

Monitors:
1. Sector exposure: Max 25% in any single sector
2. Pairwise correlation: Reject if avg correlation > 0.70 with existing positions
3. Diversification score: Track overall portfolio diversification

Designed to reduce drawdowns by ensuring positions aren't all moving together.
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CorrelationConfig:
    """Configuration for correlation management"""
    max_sector_pct: float = 0.25          # Max 25% in one sector
    max_avg_correlation: float = 0.70     # Max avg correlation with existing positions
    correlation_lookback_days: int = 60   # Days for correlation calculation
    min_positions_for_check: int = 2      # Only check correlation with 2+ positions
    max_single_stock_pct: float = 0.15    # Max 15% in a single stock


@dataclass
class PortfolioPosition:
    """Minimal position info for correlation tracking"""
    symbol: str
    sector: str
    value: float  # Current market value


class CorrelationManager:
    """
    Manages portfolio diversification and correlation risk.

    Usage:
        cm = CorrelationManager()
        allowed, reason = cm.check_new_position('AAPL', existing_positions, total_value)
        if allowed:
            # Take the trade
    """

    def __init__(self, config: CorrelationConfig = None):
        self.config = config or CorrelationConfig()
        self._sector_cache: Dict[str, str] = {}
        self._price_cache: Dict[str, pd.Series] = {}

    def check_new_position(
        self,
        new_symbol: str,
        existing_positions: List[PortfolioPosition],
        total_portfolio_value: float,
        new_position_value: float = 0
    ) -> Tuple[bool, str]:
        """
        Check if adding a new position would violate diversification rules.

        Args:
            new_symbol: Symbol to add
            existing_positions: Current portfolio positions
            total_portfolio_value: Total portfolio value
            new_position_value: Planned value of new position

        Returns:
            (allowed, reason)
        """
        if not existing_positions:
            return True, "First position - no diversification check needed"

        # 1. Check single stock concentration
        if new_position_value > 0 and total_portfolio_value > 0:
            position_pct = new_position_value / total_portfolio_value
            if position_pct > self.config.max_single_stock_pct:
                return False, (
                    f"Position size {position_pct:.1%} exceeds max "
                    f"{self.config.max_single_stock_pct:.0%} per stock"
                )

        # 2. Check sector exposure
        sector_ok, sector_msg = self._check_sector_exposure(
            new_symbol, existing_positions, total_portfolio_value, new_position_value
        )
        if not sector_ok:
            return False, sector_msg

        # 3. Check correlation (only if enough positions)
        if len(existing_positions) >= self.config.min_positions_for_check:
            corr_ok, corr_msg = self._check_correlation(
                new_symbol, [p.symbol for p in existing_positions]
            )
            if not corr_ok:
                return False, corr_msg

        return True, "OK"

    def _check_sector_exposure(
        self,
        new_symbol: str,
        positions: List[PortfolioPosition],
        total_value: float,
        new_value: float
    ) -> Tuple[bool, str]:
        """Check if adding would exceed sector limits."""
        new_sector = self._get_sector(new_symbol)

        # Calculate current sector exposure
        sector_value = sum(
            p.value for p in positions
            if p.sector == new_sector
        ) + new_value

        if total_value > 0:
            sector_pct = sector_value / total_value
            if sector_pct > self.config.max_sector_pct:
                return False, (
                    f"Sector '{new_sector}' exposure would be {sector_pct:.1%} "
                    f"(max {self.config.max_sector_pct:.0%})"
                )

        return True, "OK"

    def _check_correlation(
        self,
        new_symbol: str,
        existing_symbols: List[str]
    ) -> Tuple[bool, str]:
        """Check price correlation with existing positions."""
        try:
            new_returns = self._get_returns(new_symbol)
            if new_returns is None:
                return True, "Could not fetch data - allowing"

            correlations = []
            for sym in existing_symbols:
                existing_returns = self._get_returns(sym)
                if existing_returns is None:
                    continue

                # Align dates
                aligned = pd.concat([new_returns, existing_returns], axis=1).dropna()
                if len(aligned) < 20:
                    continue

                corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                if not np.isnan(corr):
                    correlations.append(corr)

            if not correlations:
                return True, "No correlation data available"

            avg_corr = np.mean(correlations)
            max_corr = max(correlations)

            if avg_corr > self.config.max_avg_correlation:
                return False, (
                    f"Avg correlation {avg_corr:.2f} exceeds max "
                    f"{self.config.max_avg_correlation:.2f} "
                    f"(max pair: {max_corr:.2f})"
                )

            return True, f"Avg correlation: {avg_corr:.2f}"

        except Exception as e:
            logger.warning(f"Correlation check error: {e}")
            return True, f"Error in correlation check: {e}"

    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol (cached)."""
        if symbol in self._sector_cache:
            return self._sector_cache[symbol]

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            sector = info.get('sector', 'Unknown')
            self._sector_cache[symbol] = sector
            return sector
        except Exception:
            self._sector_cache[symbol] = 'Unknown'
            return 'Unknown'

    def _get_returns(self, symbol: str) -> Optional[pd.Series]:
        """Get daily returns for correlation calculation (cached)."""
        if symbol in self._price_cache:
            return self._price_cache[symbol]

        try:
            end = datetime.now()
            start = end - timedelta(days=self.config.correlation_lookback_days + 10)

            df = yf.download(
                symbol,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                progress=False
            )

            if df is None or df.empty:
                return None

            # Handle multi-level columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            returns = df['Close'].pct_change().dropna()
            self._price_cache[symbol] = returns
            return returns

        except Exception as e:
            logger.debug(f"Failed to get returns for {symbol}: {e}")
            return None

    def get_portfolio_diversification_score(
        self,
        positions: List[PortfolioPosition],
        total_value: float
    ) -> Dict:
        """
        Calculate portfolio diversification metrics.

        Returns dict with:
        - sector_concentration: HHI of sector weights (lower = more diversified)
        - position_concentration: HHI of position weights
        - avg_correlation: Average pairwise correlation
        - diversification_score: 0-100 overall score
        """
        if not positions or total_value <= 0:
            return {
                'sector_concentration': 0,
                'position_concentration': 0,
                'avg_correlation': 0,
                'diversification_score': 50,
                'sectors': {}
            }

        # Sector weights
        sector_values: Dict[str, float] = {}
        for p in positions:
            sector_values[p.sector] = sector_values.get(p.sector, 0) + p.value

        sector_weights = {s: v / total_value for s, v in sector_values.items()}
        sector_hhi = sum(w ** 2 for w in sector_weights.values())

        # Position weights
        position_weights = [p.value / total_value for p in positions]
        position_hhi = sum(w ** 2 for w in position_weights)

        # Diversification score (0-100, higher = more diversified)
        # Perfect diversification across 20 positions = HHI of 0.05
        div_score = max(0, min(100, (1 - sector_hhi) * 100))

        return {
            'sector_concentration': round(sector_hhi, 3),
            'position_concentration': round(position_hhi, 3),
            'num_sectors': len(sector_values),
            'num_positions': len(positions),
            'diversification_score': round(div_score, 1),
            'sectors': {s: round(w, 3) for s, w in sector_weights.items()},
            'largest_sector': max(sector_weights, key=sector_weights.get) if sector_weights else 'N/A',
            'largest_sector_pct': round(max(sector_weights.values()) if sector_weights else 0, 3)
        }

    def clear_cache(self):
        """Clear all caches."""
        self._sector_cache.clear()
        self._price_cache.clear()


# Singleton
_correlation_manager = None


def get_correlation_manager(config: CorrelationConfig = None) -> CorrelationManager:
    """Get singleton correlation manager."""
    global _correlation_manager
    if _correlation_manager is None:
        _correlation_manager = CorrelationManager(config)
    return _correlation_manager
