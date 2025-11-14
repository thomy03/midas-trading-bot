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

        rs = gain / loss
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
        if len(peaks) < self.min_peaks:
            return None

        # Filter peaks to recent data only
        recent_threshold = len(rsi) - lookback_periods
        recent_peaks = peaks[peaks >= max(0, recent_threshold)]

        if len(recent_peaks) < self.min_peaks:
            return None

        best_trendline = None
        best_score = -np.inf

        # Minimum bars after last peak to look for breakout
        # Reduced to 2 to allow more recent trendlines
        min_bars_after_last_peak = 2

        # Try different STARTING positions for the trendline
        for start_idx in range(len(recent_peaks) - self.min_peaks + 1):
            # Try different number of peaks starting from start_idx
            for num_peaks in range(self.min_peaks, min(len(recent_peaks) - start_idx + 1, 10)):
                selected_peaks = recent_peaks[start_idx:start_idx + num_peaks]

                # Check if we have enough data AFTER the last peak
                last_peak_idx = selected_peaks[-1]
                bars_after = len(rsi) - 1 - last_peak_idx

                if bars_after < min_bars_after_last_peak:
                    continue  # Skip, not enough data after for breakout

                x = selected_peaks.astype(float)
                y = rsi.iloc[selected_peaks].to_numpy()

                # ⚠️ ASSOUPLI: Valider que le DERNIER sommet est plus bas que le PREMIER
                # (au lieu d'exiger que CHAQUE sommet soit plus bas que le précédent)
                # Cela permet d'accepter des oscillations intermédiaires tout en gardant une tendance descendante
                is_descending = y[-1] < y[0]  # Dernier < Premier
                if not is_descending:
                    continue  # Skip this combination - not descending overall

                slope, intercept, r_squared = self.fit_trendline(x, y)

                # Validate: must be descending and have good fit
                if not (slope >= MIN_SLOPE and slope <= MAX_SLOPE and r_squared >= self.min_r_squared):
                    continue

                # ⚠️ CRITICAL: Validate that it's a TRUE RESISTANCE
                # RSI must NOT cross significantly above between peaks
                if not self.validate_resistance(rsi, slope, intercept, selected_peaks, tolerance=2.0):
                    continue  # Not a valid resistance line

                # ⚠️ VISUAL QUALITY: Validate peaks are close enough to trendline
                # Calculate maximum distance of any peak from the trendline
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

                if quality_score > best_score:
                    best_score = quality_score
                    best_trendline = Trendline(
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

        return best_trendline

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
