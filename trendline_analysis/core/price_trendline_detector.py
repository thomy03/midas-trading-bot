"""
Price Trendline Detector

Detects support and resistance trendlines on price (High/Low) by connecting peaks and valleys.
Similar to RSI trendline detection but applied to price action.
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import Optional, Tuple
from dataclasses import dataclass

from ..config.settings import (
    PEAK_PROMINENCE,
    PEAK_DISTANCE,
    MIN_PEAKS_FOR_TRENDLINE,
    MIN_R_SQUARED,
    MIN_SLOPE,
    MAX_SLOPE,
    MAX_RESIDUAL_DISTANCE
)


@dataclass
class PriceTrendline:
    """Represents a detected price trendline"""
    peak_indices: np.ndarray
    peak_values: np.ndarray
    peak_dates: pd.DatetimeIndex
    slope: float
    intercept: float
    r_squared: float
    start_idx: int
    end_idx: int
    trendline_type: str  # 'support' or 'resistance'
    quality_score: float


class PriceTrendlineDetector:
    """Detects support and resistance trendlines on price"""

    def __init__(
        self,
        prominence: float = 2.0,  # Réduit pour détecter plus de sommets (était 5.0)
        distance: int = 5,  # Distance minimum entre sommets
        min_peaks: int = 3,  # Minimum 3 sommets comme tu demandes
        min_r_squared: float = 0.40,  # Plus tolérant (était 0.50)
        max_residual_pct: float = 7.0  # Plus tolérant : 7% (était 5%)
    ):
        """
        Initialize price trendline detector

        Args:
            prominence: Minimum prominence for peak detection (% of price)
            distance: Minimum distance between peaks (in periods)
            min_peaks: Minimum number of peaks to form a trendline
            min_r_squared: Minimum R² for trendline quality (0.50 for price volatility)
            max_residual_pct: Maximum % distance of peaks from trendline (5% for price)
        """
        self.prominence = prominence
        self.distance = distance
        self.min_peaks = min_peaks
        self.min_r_squared = min_r_squared
        self.max_residual_pct = max_residual_pct

    def detect_resistance(
        self,
        df: pd.DataFrame,
        lookback_periods: int = 52
    ) -> Optional[PriceTrendline]:
        """
        Detect descending resistance trendline on price

        Mix strategy: High peaks (wicks) + Close of green candles
        This provides more flexible detection with more potential contact points

        Args:
            df: DataFrame with OHLCV data
            lookback_periods: How far back to look for peaks

        Returns:
            PriceTrendline or None if no valid trendline found
        """
        # Extract prices
        high = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        open_price = df['Open'].iloc[:, 0] if isinstance(df['Open'], pd.DataFrame) else df['Open']

        # Calculate relative prominence (as % of price)
        price_range = high.max() - high.min()
        prominence_value = (self.prominence / 100) * price_range

        # 1. Detect peaks on High (wicks)
        high_peaks, _ = find_peaks(
            high.values,
            prominence=prominence_value,
            distance=self.distance
        )

        # 2. Detect peaks on Close for GREEN candles only
        green_candles = close > open_price
        close_for_peaks = np.where(green_candles, close.values, -np.inf)  # Only green candles

        close_peaks, _ = find_peaks(
            close_for_peaks,
            prominence=prominence_value,
            distance=self.distance
        )

        # 3. Combine both sets of peaks
        all_peaks = np.union1d(high_peaks, close_peaks)

        # 4. For each peak, use the higher value (High or Close)
        peak_values = np.array([max(high.iloc[i], close.iloc[i]) for i in all_peaks])

        # 5. Create combined series for trendline fitting
        combined_series = pd.Series(peak_values, index=high.index[all_peaks])

        if len(all_peaks) < self.min_peaks:
            return None

        # Find best descending trendline on combined points
        return self._find_best_trendline(
            combined_series,
            all_peaks,
            lookback_periods,
            trendline_type='resistance',
            should_descend=True
        )

    def detect_support(
        self,
        df: pd.DataFrame,
        lookback_periods: int = 52
    ) -> Optional[PriceTrendline]:
        """
        Detect ascending support trendline on price

        Mix strategy: Low valleys (wicks) + Close of red candles
        This provides more flexible detection with more potential contact points

        Args:
            df: DataFrame with OHLCV data
            lookback_periods: How far back to look for valleys

        Returns:
            PriceTrendline or None if no valid trendline found
        """
        # Extract prices
        low = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        open_price = df['Open'].iloc[:, 0] if isinstance(df['Open'], pd.DataFrame) else df['Open']

        # Calculate relative prominence
        price_range = low.max() - low.min()
        prominence_value = (self.prominence / 100) * price_range

        # 1. Detect valleys on Low (wicks) - invert signal
        low_valleys, _ = find_peaks(
            -low.values,  # Invert to find valleys
            prominence=prominence_value,
            distance=self.distance
        )

        # 2. Detect valleys on Close for RED candles only
        red_candles = close < open_price
        close_for_valleys = np.where(red_candles, close.values, np.inf)  # Only red candles

        close_valleys, _ = find_peaks(
            -close_for_valleys,  # Invert to find valleys
            prominence=prominence_value,
            distance=self.distance
        )

        # 3. Combine both sets of valleys
        all_valleys = np.union1d(low_valleys, close_valleys)

        # 4. For each valley, use the lower value (Low or Close)
        valley_values = np.array([min(low.iloc[i], close.iloc[i]) for i in all_valleys])

        # 5. Create combined series for trendline fitting
        combined_series = pd.Series(valley_values, index=low.index[all_valleys])

        if len(all_valleys) < self.min_peaks:
            return None

        # Find best ascending trendline on combined points
        return self._find_best_trendline(
            combined_series,
            all_valleys,
            lookback_periods,
            trendline_type='support',
            should_descend=False
        )

    def _find_best_trendline(
        self,
        price_series: pd.Series,
        peaks: np.ndarray,
        lookback_periods: int,
        trendline_type: str,
        should_descend: bool
    ) -> Optional[PriceTrendline]:
        """
        Find the best trendline from detected peaks/valleys

        Args:
            price_series: Price series (High or Low)
            peaks: Indices of peaks/valleys
            lookback_periods: How far back to look
            trendline_type: 'support' or 'resistance'
            should_descend: True for resistance, False for support

        Returns:
            Best PriceTrendline or None
        """
        # Filter to recent peaks only
        recent_threshold = len(price_series) - lookback_periods
        recent_peaks = peaks[peaks >= max(0, recent_threshold)]

        if len(recent_peaks) < self.min_peaks:
            return None

        best_trendline = None
        best_score = -np.inf

        min_bars_after_last_peak = 2

        # Try different combinations
        for start_idx in range(len(recent_peaks) - self.min_peaks + 1):
            for num_peaks in range(self.min_peaks, min(len(recent_peaks) - start_idx + 1, 10)):
                selected_peaks = recent_peaks[start_idx:start_idx + num_peaks]

                # Check enough data after last peak
                last_peak_idx = selected_peaks[-1]
                bars_after = len(price_series) - 1 - last_peak_idx

                if bars_after < min_bars_after_last_peak:
                    continue

                x = selected_peaks.astype(float)
                y = price_series.iloc[selected_peaks].to_numpy()

                # Validate peaks progression - MORE FLEXIBLE for price
                # Instead of ALL peaks strictly descending/ascending,
                # we accept if FIRST and LAST show the trend (allows oscillation)
                if should_descend:
                    # For resistance: last peak should be lower than first
                    is_valid = y[-1] < y[0]
                else:
                    # For support: last peak should be higher than first
                    is_valid = y[-1] > y[0]

                if not is_valid:
                    continue

                # Fit trendline
                slope, intercept, r_squared = self._fit_trendline(x, y)

                # For PRICE trendlines, we don't validate absolute slope values
                # because price magnitudes vary greatly ($1 vs $1000 stocks)
                # Direction is already validated via first/last comparison above
                # Just ensure slope has correct sign
                if should_descend and slope > 0:
                    continue  # Resistance must have negative slope
                if not should_descend and slope < 0:
                    continue  # Support must have positive slope

                # Validate R²
                if r_squared < self.min_r_squared:
                    continue

                # Validate residual distance (peaks close to line)
                max_residual = max(abs(y[i] - (slope * x[i] + intercept)) for i in range(len(y)))

                # For price, scale the residual threshold by price level
                avg_price = np.mean(y)
                residual_threshold = (self.max_residual_pct / 100) * avg_price  # 5% of price

                if max_residual > residual_threshold:
                    continue

                # Calculate quality score
                quality_score = self._calculate_quality_score(
                    r_squared, num_peaks, slope, selected_peaks, len(price_series)
                )

                if quality_score > best_score:
                    best_score = quality_score
                    best_trendline = PriceTrendline(
                        peak_indices=selected_peaks,
                        peak_values=y,
                        peak_dates=price_series.index[selected_peaks],
                        slope=slope,
                        intercept=intercept,
                        r_squared=r_squared,
                        start_idx=selected_peaks[0],
                        end_idx=len(price_series) - 1,
                        trendline_type=trendline_type,
                        quality_score=quality_score
                    )

        return best_trendline

    def _fit_trendline(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Fit linear regression and calculate R²

        Args:
            x: X coordinates (indices)
            y: Y coordinates (prices)

        Returns:
            Tuple of (slope, intercept, r_squared)
        """
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs

        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return slope, intercept, r_squared

    def _calculate_quality_score(
        self,
        r_squared: float,
        num_peaks: int,
        slope: float,
        peak_indices: np.ndarray,
        total_length: int
    ) -> float:
        """
        Calculate overall quality score for trendline

        Args:
            r_squared: R² value
            num_peaks: Number of peaks in trendline
            slope: Slope of trendline
            peak_indices: Indices of peaks
            total_length: Total length of data

        Returns:
            Quality score (0-100)
        """
        # R² component (40% weight)
        r2_score = r_squared * 40

        # Number of peaks component (30% weight)
        # More peaks = better, but diminishing returns
        peaks_score = min(30, (num_peaks / 6) * 30)

        # Recency component (20% weight)
        last_peak_idx = peak_indices[-1]
        recency = (total_length - last_peak_idx) / total_length
        recency_score = (1 - recency) * 20

        # Slope consistency (10% weight)
        # Prefer moderate slopes (not too steep, not too flat)
        abs_slope = abs(slope)
        if 0.3 <= abs_slope <= 0.7:
            slope_score = 10
        elif 0.1 <= abs_slope <= 0.9:
            slope_score = 5
        else:
            slope_score = 0

        return r2_score + peaks_score + recency_score + slope_score

    def get_trendline_value(self, trendline: PriceTrendline, index: int) -> float:
        """
        Get trendline value at specific index

        Args:
            trendline: PriceTrendline object
            index: Index to calculate value at

        Returns:
            Trendline value at index
        """
        return trendline.slope * index + trendline.intercept
