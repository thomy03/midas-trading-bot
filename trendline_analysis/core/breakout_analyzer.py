"""
Trendline Breakout Analyzer

Detects and validates breakouts of RSI above resistance trendlines.
A breakout occurs when RSI crosses above the descending trendline,
indicating a potential bullish momentum shift.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass

from .trendline_detector import Trendline, RSITrendlineDetector
from ..config.settings import (
    BREAKOUT_THRESHOLD,
    CONFIRMATION_PERIODS,
    MAX_BREAKOUT_AGE
)


@dataclass
class Breakout:
    """Represents a detected breakout"""
    date: pd.Timestamp
    index: int
    rsi_value: float
    trendline_value: float
    distance_above: float  # How far above trendline (in RSI points)
    strength: str  # 'WEAK', 'MODERATE', 'STRONG'
    is_confirmed: bool
    age_in_periods: int  # How many periods ago
    trendline_quality: float


class TrendlineBreakoutAnalyzer:
    """Analyzes RSI trendline breakouts"""

    def __init__(
        self,
        threshold: float = BREAKOUT_THRESHOLD,
        confirmation_periods: int = CONFIRMATION_PERIODS,
        max_age: int = MAX_BREAKOUT_AGE
    ):
        """
        Initialize breakout analyzer

        Args:
            threshold: Minimum distance above trendline to consider breakout
            confirmation_periods: Periods to wait for confirmation
            max_age: Maximum age (periods) for "recent" breakout
        """
        self.threshold = threshold
        self.confirmation_periods = confirmation_periods
        self.max_age = max_age
        self.detector = RSITrendlineDetector()

    def detect_breakout(
        self,
        rsi: pd.Series,
        trendline: Trendline
    ) -> Optional[Breakout]:
        """
        Detect if and when RSI broke above the trendline

        Args:
            rsi: RSI values
            trendline: Trendline to check for breakout

        Returns:
            Breakout object or None if no breakout detected
        """
        # Start checking from the last peak in the trendline
        start_idx = max(trendline.peak_indices[-1] + 1, trendline.start_idx)
        end_idx = len(rsi) - 1

        breakout_candidates = []

        for i in range(start_idx, end_idx + 1):
            rsi_val = rsi.iloc[i]
            trendline_val = self.detector.get_trendline_value(trendline, i)

            # Check if RSI is above trendline
            distance_above = rsi_val - trendline_val

            if distance_above >= self.threshold:
                # Potential breakout - check if it's a crossing (not already above)
                if i > start_idx:
                    prev_rsi = rsi.iloc[i - 1]
                    prev_trendline = self.detector.get_trendline_value(trendline, i - 1)

                    # True breakout = was below or at trendline, now above
                    if prev_rsi <= prev_trendline + self.threshold:
                        breakout_candidates.append({
                            'index': i,
                            'date': rsi.index[i],
                            'rsi_value': rsi_val,
                            'trendline_value': trendline_val,
                            'distance_above': distance_above
                        })

        if not breakout_candidates:
            return None

        # Get the FIRST breakout (earliest crossing)
        # This is more relevant for trading signals
        first_breakout = breakout_candidates[0]
        breakout_idx = first_breakout['index']
        age = end_idx - breakout_idx

        # Check confirmation (RSI stayed above in following periods)
        is_confirmed = self._check_confirmation(
            rsi,
            trendline,
            breakout_idx,
            self.confirmation_periods
        )

        # Determine strength
        strength = self._calculate_strength(
            first_breakout['distance_above'],
            rsi.iloc[breakout_idx]
        )

        return Breakout(
            date=first_breakout['date'],
            index=breakout_idx,
            rsi_value=first_breakout['rsi_value'],
            trendline_value=first_breakout['trendline_value'],
            distance_above=first_breakout['distance_above'],
            strength=strength,
            is_confirmed=is_confirmed,
            age_in_periods=age,
            trendline_quality=trendline.quality_score
        )

    def _check_confirmation(
        self,
        rsi: pd.Series,
        trendline: Trendline,
        breakout_idx: int,
        periods: int
    ) -> bool:
        """
        Check if breakout is confirmed by staying above trendline

        Args:
            rsi: RSI values
            trendline: Trendline
            breakout_idx: Index where breakout occurred
            periods: Number of periods to check

        Returns:
            True if confirmed, False otherwise
        """
        end_idx = min(breakout_idx + periods, len(rsi) - 1)

        for i in range(breakout_idx + 1, end_idx + 1):
            rsi_val = rsi.iloc[i]
            trendline_val = self.detector.get_trendline_value(trendline, i)

            # If RSI falls back below trendline, not confirmed
            if rsi_val < trendline_val:
                return False

        return True

    def _calculate_strength(self, distance_above: float, rsi_value: float) -> str:
        """
        Calculate breakout strength based on distance and RSI level

        Args:
            distance_above: How far above trendline (RSI points)
            rsi_value: Current RSI value

        Returns:
            'WEAK', 'MODERATE', or 'STRONG'
        """
        # Strong: RSI > 60 and well above trendline (>3 points)
        if rsi_value > 60 and distance_above > 3:
            return 'STRONG'

        # Moderate: RSI 50-60 or decent distance
        elif rsi_value > 50 or distance_above > 2:
            return 'MODERATE'

        # Weak: marginal breakout
        else:
            return 'WEAK'

    def is_recent_breakout(self, breakout: Breakout) -> bool:
        """
        Check if breakout is recent enough to act on

        Args:
            breakout: Breakout object

        Returns:
            True if recent, False otherwise
        """
        return breakout.age_in_periods <= self.max_age

    def analyze(
        self,
        df: pd.DataFrame,
        lookback_periods: int = 52
    ) -> Optional[Dict]:
        """
        Complete analysis: detect trendline + check for breakout

        Args:
            df: DataFrame with OHLCV data
            lookback_periods: How far back to look

        Returns:
            Dictionary with trendline and breakout info, or None
        """
        # Calculate RSI
        rsi = self.detector.calculate_rsi(df)

        # Detect trendline
        trendline = self.detector.detect(df, lookback_periods)

        if trendline is None:
            return None

        # Check for breakout
        breakout = self.detect_breakout(rsi, trendline)

        result = {
            'trendline': trendline,
            'breakout': breakout,
            'has_breakout': breakout is not None,
            'is_recent': breakout is not None and self.is_recent_breakout(breakout),
            'rsi': rsi
        }

        return result

    def get_signal(self, analysis: Dict) -> str:
        """
        Get trading signal based on analysis

        Args:
            analysis: Result from analyze()

        Returns:
            'STRONG_BUY', 'BUY', 'WATCH', or 'NO_SIGNAL'
        """
        if not analysis or not analysis['has_breakout']:
            return 'NO_SIGNAL'

        breakout = analysis['breakout']

        # Not recent enough
        if not analysis['is_recent']:
            return 'NO_SIGNAL'

        # Strong recent breakout
        if breakout.is_confirmed and breakout.strength == 'STRONG':
            return 'STRONG_BUY'

        # Moderate confirmed breakout
        elif breakout.is_confirmed and breakout.strength in ['MODERATE', 'STRONG']:
            return 'BUY'

        # Weak or unconfirmed breakout
        elif breakout.strength != 'WEAK':
            return 'WATCH'

        else:
            return 'NO_SIGNAL'
