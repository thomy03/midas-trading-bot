"""
RSI Trendline Detector

Detects resistance trendlines on RSI by connecting peaks (local maxima).
Uses scipy for peak detection and numpy for linear regression.

IMPORTANT: A trendline is only valid if the RSI respects it as a RESISTANCE
between the peaks (doesn't cross above significantly).
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from ..config.settings import (
    RSI_PERIOD,
    PEAK_PROMINENCE,
    PEAK_DISTANCE,
    MIN_PEAKS_FOR_TRENDLINE,
    MIN_R_SQUARED,
    MIN_SLOPE,
    MAX_SLOPE,
    MAX_RESIDUAL_DISTANCE
)


@dataclass
class Trendline:
    """Represents a detected trendline"""
    slope: float
    intercept: float
    r_squared: float
    peak_indices: List[int]
    peak_dates: List[pd.Timestamp]
    peak_values: List[float]
    start_idx: int
    end_idx: int
    quality_score: float


class RSITrendlineDetector:
    """Detects descending resistance trendlines on RSI"""

    def __init__(
        self,
        prominence: float = PEAK_PROMINENCE,
        distance: int = PEAK_DISTANCE,
        min_peaks: int = MIN_PEAKS_FOR_TRENDLINE,
        min_r_squared: float = MIN_R_SQUARED
    ):
        """
        Initialize the trendline detector

        Args:
            prominence: Minimum prominence for peak detection
            distance: Minimum distance between peaks (in periods)
            min_peaks: Minimum number of peaks to form a trendline
            min_r_squared: Minimum R² for trendline quality
        """
        self.prominence = prominence
        self.distance = distance
        self.min_peaks = min_peaks
        self.min_r_squared = min_r_squared

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
        """
        Calculate RSI indicator

        Args:
            df: DataFrame with 'Close' column
            period: RSI period (default 14)

        Returns:
            RSI values as Series
        """
        # Handle multi-index columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close'].iloc[:, 0] if df['Close'].shape[1] > 0 else df['Close']
        else:
            close = df['Close']

        # Ensure it's a Series
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Avoid division by zero: add small epsilon to loss
        rs = gain / (loss + np.finfo(float).eps)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def detect_peaks(self, rsi: pd.Series) -> Tuple[np.ndarray, Dict]:
        """
        Detect peaks (local maxima) in RSI

        Args:
            rsi: RSI values as Series

        Returns:
            Tuple of (peak_indices, peak_properties)
        """
        peaks, properties = find_peaks(
            rsi.to_numpy().ravel(),  # Ensure 1D array
            prominence=self.prominence,
            distance=self.distance
        )

        return peaks, properties

    def fit_trendline(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Fit a linear trendline to points using least squares regression

        Args:
            x: X values (indices or timestamps)
            y: Y values (RSI values)

        Returns:
            Tuple of (slope, intercept, r_squared)
        """
        # Linear regression: y = mx + b
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs

        # Calculate R²
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return slope, intercept, r_squared

    def validate_resistance(
        self,
        rsi: pd.Series,
        slope: float,
        intercept: float,
        peak_indices: np.ndarray,
        tolerance: float = 2.0
    ) -> bool:
        """
        Validate that the trendline acts as a TRUE RESISTANCE.

        Between each pair of consecutive peaks, the RSI should NOT
        significantly cross above the trendline.

        Args:
            rsi: RSI values
            slope: Trendline slope
            intercept: Trendline intercept
            peak_indices: Indices of peaks forming the trendline
            tolerance: Maximum allowed overshoot above trendline (RSI points)

        Returns:
            True if the trendline is a valid resistance, False otherwise
        """
        # Check between each pair of consecutive peaks
        for i in range(len(peak_indices) - 1):
            start_idx = peak_indices[i]
            end_idx = peak_indices[i + 1]

            # Check all bars between these two peaks
            for idx in range(start_idx + 1, end_idx):
                rsi_value = rsi.iloc[idx]
                trendline_value = slope * idx + intercept

                # If RSI crosses significantly above, this is NOT a valid resistance
                if rsi_value > trendline_value + tolerance:
                    return False

        return True

    def find_best_trendline(
        self,
        rsi: pd.Series,
        peaks: np.ndarray,
        lookback_periods: int = 52
    ) -> Optional[Trendline]:
        """
        Find the best descending resistance trendline

        CRITICAL REQUIREMENTS:
        1. Trendline drawn on 3+ HISTORICAL peaks (not the most recent)
        2. Must have data AFTER the last peak for breakout detection
        3. RSI must RESPECT the trendline as resistance (no significant crosses between peaks)
        4. Good statistical fit (R²) and descending slope

        Args:
            rsi: RSI values
            peaks: Peak indices from detect_peaks
            lookback_periods: How far back to look for peaks

        Returns:
            Best Trendline or None if no valid trendline found
        """
        all_trendlines = self.find_all_trendlines(rsi, peaks, lookback_periods)
        if not all_trendlines:
            return None
        # Return the best one (highest score)
        return all_trendlines[0]

    def find_all_trendlines(
        self,
        rsi: pd.Series,
        peaks: np.ndarray,
        lookback_periods: int = 52
    ) -> List[Trendline]:
        """
        Find ALL valid descending resistance trendlines (not just the best one)

        Uses combinations of peaks (not just consecutive) to find trendlines
        that might skip intermediate peaks with very different RSI values.

        Args:
            rsi: RSI values
            peaks: Peak indices from detect_peaks
            lookback_periods: How far back to look for peaks

        Returns:
            List of all valid Trendlines, sorted by quality score (best first)
        """
        from itertools import combinations

        if len(peaks) < self.min_peaks:
            return []

        # Filter peaks to recent data only
        recent_threshold = len(rsi) - lookback_periods
        recent_peaks = peaks[peaks >= max(0, recent_threshold)]

        if len(recent_peaks) < self.min_peaks:
            return []

        all_trendlines = []
        seen_peak_sets = set()  # Avoid duplicates

        # Minimum bars after last peak to look for breakout
        min_bars_after_last_peak = 2

        # Maximum number of peaks to consider (to limit combinations)
        # Reduced from 20 to 15 for better performance (C(20,6)=38760 vs C(15,6)=5005)
        # Phase 6 optimization: balance between coverage and speed
        max_peaks_to_consider = min(len(recent_peaks), 15)
        peaks_to_use = recent_peaks[-max_peaks_to_consider:]  # Most recent peaks

        # Try ALL combinations of 3 to 6 peaks (not just consecutive ones)
        for num_peaks in range(self.min_peaks, min(len(peaks_to_use) + 1, 7)):
            for combo in combinations(range(len(peaks_to_use)), num_peaks):
                selected_indices = [peaks_to_use[i] for i in combo]
                selected_peaks = np.array(selected_indices)

                # Create a hashable key for this peak set
                peak_key = tuple(selected_peaks.tolist())
                if peak_key in seen_peak_sets:
                    continue
                seen_peak_sets.add(peak_key)

                # Check if we have enough data AFTER the last peak
                last_peak_idx = selected_peaks[-1]
                bars_after = len(rsi) - 1 - last_peak_idx

                if bars_after < min_bars_after_last_peak:
                    continue  # Skip, not enough data after for breakout

                x = selected_peaks.astype(float)
                y = rsi.iloc[selected_peaks].to_numpy()

                # Validate: last peak must be lower than first (overall descending)
                is_descending = y[-1] < y[0]
                if not is_descending:
                    continue  # Skip this combination - not descending overall

                slope, intercept, r_squared = self.fit_trendline(x, y)

                # Validate: must be descending and have good fit
                if not (slope >= MIN_SLOPE and slope <= MAX_SLOPE and r_squared >= self.min_r_squared):
                    continue

                # CRITICAL: Validate that it's a TRUE RESISTANCE
                # RSI must NOT cross significantly above between peaks
                if not self.validate_resistance(rsi, slope, intercept, selected_peaks, tolerance=2.0):
                    continue  # Not a valid resistance line

                # VISUAL QUALITY: Validate peaks are close enough to trendline
                max_residual = max(abs(y[i] - (slope * x[i] + intercept)) for i in range(len(y)))
                if max_residual > MAX_RESIDUAL_DISTANCE:
                    continue  # Peaks too far from trendline, poor visual fit

                # Calculate quality score
                quality_score = self._calculate_quality_score(
                    r_squared, num_peaks, slope, selected_peaks, len(rsi)
                )

                # Bonus for having more bars after (more opportunity for breakout)
                recency_bonus = min(bars_after / 20.0, 1.0) * 10
                quality_score += recency_bonus

                trendline = Trendline(
                    slope=slope,
                    intercept=intercept,
                    r_squared=r_squared,
                    peak_indices=selected_peaks.tolist(),
                    peak_dates=rsi.index[selected_peaks].tolist(),
                    peak_values=y.tolist(),
                    start_idx=selected_peaks[0],
                    end_idx=len(rsi) - 1,
                    quality_score=quality_score
                )
                all_trendlines.append(trendline)

        # Sort by quality score (best first) then filter overlapping trendlines
        all_trendlines.sort(key=lambda t: t.quality_score, reverse=True)

        # Filter to keep only non-overlapping trendlines
        filtered_trendlines = self._filter_overlapping_trendlines(all_trendlines)

        return filtered_trendlines

    def _filter_overlapping_trendlines(
        self,
        trendlines: List[Trendline],
        max_trendlines: int = 3
    ) -> List[Trendline]:
        """
        Filter trendlines to keep diverse ones from different time periods.

        Strategy:
        1. Keep the best trendline
        2. For additional trendlines, prefer those from DIFFERENT time periods
           (check if start dates are far apart)
        3. If same period, require low overlap (< 50% shared peaks)

        Args:
            trendlines: List of trendlines sorted by quality
            max_trendlines: Maximum number of trendlines to return (default 3)

        Returns:
            Filtered list of diverse trendlines (max 3)
        """
        if not trendlines:
            return []

        filtered = [trendlines[0]]  # Keep the best one

        for tl in trendlines[1:]:
            if len(filtered) >= max_trendlines:
                break  # Already have enough trendlines

            should_add = True
            tl_peaks = set(tl.peak_indices)
            tl_start = tl.peak_indices[0]
            tl_end = tl.peak_indices[-1]

            for kept_tl in filtered:
                kept_peaks = set(kept_tl.peak_indices)
                kept_start = kept_tl.peak_indices[0]
                kept_end = kept_tl.peak_indices[-1]

                # Check if trendlines are from different time periods
                # (no temporal overlap at all)
                time_gap = min(abs(tl_start - kept_end), abs(tl_end - kept_start))
                different_periods = (tl_end < kept_start) or (tl_start > kept_end)

                if different_periods:
                    # Different time periods = keep it (no overlap check needed)
                    continue

                # Same time period - check peak overlap
                overlap = len(tl_peaks & kept_peaks)
                overlap_ratio = overlap / min(len(tl_peaks), len(kept_peaks))

                # Reject if too much overlap (> 50%)
                if overlap_ratio > 0.5:
                    should_add = False
                    break

            if should_add:
                filtered.append(tl)

        return filtered

    def _calculate_quality_score(
        self,
        r_squared: float,
        num_points: int,
        slope: float,
        peak_indices: np.ndarray,
        total_length: int
    ) -> float:
        """
        Calculate overall quality score for a trendline

        Args:
            r_squared: Coefficient of determination
            num_points: Number of peaks in trendline
            slope: Trendline slope
            peak_indices: Indices of peaks
            total_length: Total length of data

        Returns:
            Quality score (0-100)
        """
        from ..config.settings import TRENDLINE_QUALITY_WEIGHTS as weights

        # R² score (0-1)
        r_squared_score = r_squared

        # Number of points score (normalized)
        # 3 points = 0.5, 5 points = 0.75, 7+ points = 1.0
        num_points_score = min(1.0, (num_points - 3) / 4 + 0.5)

        # Slope consistency score
        # Optimal slope around -0.2 to -0.1 (descending but not too steep)
        optimal_slope = -0.15
        slope_deviation = abs(slope - optimal_slope)
        slope_score = max(0, 1 - slope_deviation)

        # Recency score (how recent are the peaks)
        avg_peak_position = np.mean(peak_indices) / total_length
        recency_score = avg_peak_position

        # Weighted sum
        total_score = (
            weights['r_squared'] * r_squared_score +
            weights['num_points'] * num_points_score +
            weights['slope_consistency'] * slope_score +
            weights['recency'] * recency_score
        ) * 100

        return total_score

    def get_trendline_value(self, trendline: Trendline, index: int) -> float:
        """
        Get trendline value at a specific index

        Args:
            trendline: Trendline object
            index: Data index

        Returns:
            Trendline value at index
        """
        return trendline.slope * index + trendline.intercept

    def detect(
        self,
        df: pd.DataFrame,
        lookback_periods: int = 52
    ) -> Optional[Trendline]:
        """
        Main method to detect the best RSI resistance trendline

        Args:
            df: DataFrame with OHLCV data
            lookback_periods: How far back to look for peaks

        Returns:
            Best Trendline or None
        """
        # Calculate RSI
        rsi = self.calculate_rsi(df)

        # Detect peaks
        peaks, _ = self.detect_peaks(rsi)

        if len(peaks) == 0:
            return None

        # Find best trendline
        trendline = self.find_best_trendline(rsi, peaks, lookback_periods)

        return trendline
