"""
Synchronized Price Trendline Detector

Detects price trendlines that are SYNCHRONIZED with RSI trendlines.
This ensures price peaks align temporally with RSI peaks for dual confirmation.
"""

import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass

from .trendline_detector import Trendline
from .price_trendline_detector import PriceTrendline, PriceTrendlineDetector


class SynchronizedPriceDetector(PriceTrendlineDetector):
    """Detects price trendlines synchronized with RSI trendline peaks"""

    def detect_synchronized_resistance(
        self,
        df: pd.DataFrame,
        rsi_trendline: Trendline,
        search_window: int = 5
    ) -> Optional[PriceTrendline]:
        """
        Detect price resistance trendline SYNCHRONIZED with RSI trendline

        For each RSI peak, find the corresponding price peak within ±search_window periods.
        This ensures the price trendline follows the same temporal pattern as RSI.

        Args:
            df: DataFrame with OHLCV data
            rsi_trendline: The RSI trendline to synchronize with
            search_window: How many periods before/after each RSI peak to search for price peak

        Returns:
            PriceTrendline synchronized with RSI, or None
        """
        # Extract prices
        high = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        open_price = df['Open'].iloc[:, 0] if isinstance(df['Open'], pd.DataFrame) else df['Open']

        # For each RSI peak, find corresponding price peak
        price_peak_indices = []
        price_peak_values = []

        for rsi_peak_idx in rsi_trendline.peak_indices:
            # Define search window around RSI peak
            start_search = max(0, rsi_peak_idx - search_window)
            end_search = min(len(high) - 1, rsi_peak_idx + search_window)

            # Find the HIGHEST price in this window (for resistance)
            # Consider both High (wicks) and Close (for green candles)
            max_price = -np.inf
            best_idx = None

            for i in range(start_search, end_search + 1):
                # For resistance: use High of candle, or Close if green candle
                is_green = close.iloc[i] > open_price.iloc[i]
                price_to_check = max(high.iloc[i], close.iloc[i]) if is_green else high.iloc[i]

                if price_to_check > max_price:
                    max_price = price_to_check
                    best_idx = i

            if best_idx is not None:
                price_peak_indices.append(best_idx)
                price_peak_values.append(max_price)

        # Need at least 3 price peaks
        if len(price_peak_indices) < self.min_peaks:
            return None

        # Convert to numpy arrays
        price_peak_indices = np.array(price_peak_indices)
        price_peak_values = np.array(price_peak_values)

        # IMPORTANT: Ne PAS valider la direction !
        # On accepte ascendant, descendant, ou flat
        # C'est normal d'avoir une divergence (RSI baisse, prix monte)

        # Fit trendline
        x = price_peak_indices.astype(float)
        y = price_peak_values

        slope, intercept, r_squared = self._fit_trendline(x, y)

        # Validation R² très permissive (juste pour éviter les outliers extrêmes)
        if r_squared < 0.30:  # Très tolérant pour capturer les divergences
            return None

        # Validation résiduelle très permissive (10% au lieu de 5%)
        max_residual = max(abs(y[i] - (slope * x[i] + intercept)) for i in range(len(y)))
        avg_price = np.mean(y)
        residual_threshold = (10.0 / 100) * avg_price  # 10% tolérance

        if max_residual > residual_threshold:
            return None

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            r_squared,
            len(price_peak_indices),
            slope,
            price_peak_indices,
            len(high)
        )

        # Create price series for peak dates
        price_series = pd.Series(price_peak_values, index=high.index[price_peak_indices])

        # Déterminer le type basé sur la pente réelle (peut être différent du RSI)
        trendline_type = 'resistance' if slope < 0 else 'support' if slope > 0 else 'horizontal'

        return PriceTrendline(
            peak_indices=price_peak_indices,
            peak_values=price_peak_values,
            peak_dates=price_series.index,
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            start_idx=price_peak_indices[0],
            end_idx=len(high) - 1,
            trendline_type=trendline_type,
            quality_score=quality_score
        )

    def detect_synchronized_support(
        self,
        df: pd.DataFrame,
        rsi_trendline: Trendline,
        search_window: int = 5
    ) -> Optional[PriceTrendline]:
        """
        Detect price support trendline SYNCHRONIZED with RSI trendline

        For each RSI valley, find the corresponding price valley within ±search_window periods.

        Args:
            df: DataFrame with OHLCV data
            rsi_trendline: The RSI trendline to synchronize with
            search_window: How many periods before/after each RSI valley to search for price valley

        Returns:
            PriceTrendline synchronized with RSI, or None
        """
        # Extract prices
        low = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        open_price = df['Open'].iloc[:, 0] if isinstance(df['Open'], pd.DataFrame) else df['Open']

        # For each RSI valley, find corresponding price valley
        price_valley_indices = []
        price_valley_values = []

        for rsi_valley_idx in rsi_trendline.peak_indices:
            # Define search window
            start_search = max(0, rsi_valley_idx - search_window)
            end_search = min(len(low) - 1, rsi_valley_idx + search_window)

            # Find the LOWEST price in this window (for support)
            min_price = np.inf
            best_idx = None

            for i in range(start_search, end_search + 1):
                # For support: use Low of candle, or Close if red candle
                is_red = close.iloc[i] < open_price.iloc[i]
                price_to_check = min(low.iloc[i], close.iloc[i]) if is_red else low.iloc[i]

                if price_to_check < min_price:
                    min_price = price_to_check
                    best_idx = i

            if best_idx is not None:
                price_valley_indices.append(best_idx)
                price_valley_values.append(min_price)

        if len(price_valley_indices) < self.min_peaks:
            return None

        # Convert to numpy
        price_valley_indices = np.array(price_valley_indices)
        price_valley_values = np.array(price_valley_values)

        # IMPORTANT: Ne PAS valider la direction !
        # Accepter toutes les directions (divergences possibles)

        # Fit trendline
        x = price_valley_indices.astype(float)
        y = price_valley_values

        slope, intercept, r_squared = self._fit_trendline(x, y)

        # Validation R² très permissive
        if r_squared < 0.30:
            return None

        # Validation résiduelle permissive (10%)
        max_residual = max(abs(y[i] - (slope * x[i] + intercept)) for i in range(len(y)))
        avg_price = np.mean(y)
        residual_threshold = (10.0 / 100) * avg_price

        if max_residual > residual_threshold:
            return None

        # Calculate quality
        quality_score = self._calculate_quality_score(
            r_squared,
            len(price_valley_indices),
            slope,
            price_valley_indices,
            len(low)
        )

        price_series = pd.Series(price_valley_values, index=low.index[price_valley_indices])

        # Déterminer le type basé sur la pente réelle
        trendline_type = 'resistance' if slope < 0 else 'support' if slope > 0 else 'horizontal'

        return PriceTrendline(
            peak_indices=price_valley_indices,
            peak_values=price_valley_values,
            peak_dates=price_series.index,
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            start_idx=price_valley_indices[0],
            end_idx=len(low) - 1,
            trendline_type=trendline_type,
            quality_score=quality_score
        )
