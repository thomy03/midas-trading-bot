"""
EMA analysis module for detecting crossovers and support/resistance zones
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from config.settings import (
    EMA_PERIODS, ZONE_TOLERANCE, MAX_CROSSOVER_AGE_WEEKLY, MAX_CROSSOVER_AGE_DAILY
)
from src.utils.logger import logger


class EMAAnalyzer:
    """Analyzes EMAs for crossovers and support/resistance zones"""

    def __init__(self):
        """Initialize the EMA analyzer"""
        self.ema_periods = EMA_PERIODS

    def calculate_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMAs for the given dataframe

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added EMA columns
        """
        if df is None or df.empty:
            return df

        df = df.copy()

        for period in self.ema_periods:
            ema_col = f'EMA_{period}'
            # Calculate EMA using pandas native method
            df[ema_col] = df['Close'].ewm(span=period, adjust=False).mean()

        logger.debug(f"Calculated EMAs: {self.ema_periods}")
        return df

    def detect_crossovers(
        self,
        df: pd.DataFrame,
        timeframe: str = 'daily'
    ) -> List[Dict]:
        """
        Detect EMA crossovers which can act as support/resistance zones

        Args:
            df: DataFrame with EMA columns
            timeframe: 'weekly' or 'daily'

        Returns:
            List of crossover information dictionaries
        """
        if df is None or df.empty:
            return []

        crossovers = []

        # Check all combinations of EMAs
        ema_pairs = [
            (24, 38),
            (24, 62),
            (38, 62)
        ]

        for fast_period, slow_period in ema_pairs:
            fast_ema = f'EMA_{fast_period}'
            slow_ema = f'EMA_{slow_period}'

            if fast_ema not in df.columns or slow_ema not in df.columns:
                continue

            # Calculate the difference
            df['diff'] = df[fast_ema] - df[slow_ema]

            # Find where the sign changes (crossover points)
            df['cross'] = np.sign(df['diff']).diff()

            # Get crossover indices
            cross_indices = df[df['cross'] != 0].index

            for idx in cross_indices:
                if idx == df.index[0]:  # Skip first row
                    continue

                try:
                    cross_date = df.loc[idx].name
                    cross_price = (df.loc[idx, fast_ema] + df.loc[idx, slow_ema]) / 2

                    # Determine crossover type
                    if df.loc[idx, 'cross'] > 0:
                        cross_type = 'bullish'  # Fast crosses above slow (potential support)
                    else:
                        cross_type = 'bearish'  # Fast crosses below slow (potential resistance)

                    # Calculate age of crossover
                    if isinstance(cross_date, pd.Timestamp):
                        days_ago = (datetime.now() - cross_date.to_pydatetime()).days
                    else:
                        days_ago = 0

                    # Filter based on max age
                    max_age = MAX_CROSSOVER_AGE_WEEKLY if timeframe == 'weekly' else MAX_CROSSOVER_AGE_DAILY

                    # For weekly, convert to weeks
                    if timeframe == 'weekly':
                        age_in_periods = days_ago / 7
                    else:
                        age_in_periods = days_ago

                    if age_in_periods > max_age:
                        continue

                    crossovers.append({
                        'date': cross_date,
                        'price': float(cross_price),
                        'fast_ema': fast_period,
                        'slow_ema': slow_period,
                        'type': cross_type,
                        'days_ago': days_ago,
                        'age_in_periods': age_in_periods
                    })

                except Exception as e:
                    logger.warning(f"Error processing crossover at index {idx}: {e}")
                    continue

        # Remove temporary columns
        df.drop(['diff', 'cross'], axis=1, inplace=True, errors='ignore')

        # Sort by recency (most recent first)
        crossovers.sort(key=lambda x: x['days_ago'])

        logger.debug(f"Detected {len(crossovers)} crossovers in {timeframe} timeframe")
        return crossovers

    def check_ema_alignment(self, df: pd.DataFrame, for_buy: bool = True) -> Tuple[bool, str]:
        """
        Check if EMAs are aligned for a buy signal

        Args:
            df: DataFrame with EMA columns
            for_buy: If True, check for buy alignment (fast > slow), else for sell

        Returns:
            Tuple of (is_aligned, alignment_description)
        """
        if df is None or df.empty:
            return False, "No data"

        # Get the latest values
        latest = df.iloc[-1]

        ema_24 = latest.get('EMA_24', np.nan)
        ema_38 = latest.get('EMA_38', np.nan)
        ema_62 = latest.get('EMA_62', np.nan)

        if pd.isna(ema_24) or pd.isna(ema_38) or pd.isna(ema_62):
            return False, "Missing EMA data"

        # Check alignment for buy signal
        if for_buy:
            # At least 2 EMAs should be in correct order
            # Best case: 24 > 38 > 62 (full bull trend)
            # Acceptable: 24 > 38 OR 24 > 62 OR 38 > 62

            conditions = []
            alignment_parts = []

            if ema_24 > ema_38:
                conditions.append(True)
                alignment_parts.append("24>38")
            else:
                alignment_parts.append("24<38")

            if ema_24 > ema_62:
                conditions.append(True)
                alignment_parts.append("24>62")
            else:
                alignment_parts.append("24<62")

            if ema_38 > ema_62:
                conditions.append(True)
                alignment_parts.append("38>62")
            else:
                alignment_parts.append("38<62")

            alignment_desc = ", ".join(alignment_parts)

            # Need at least 2 favorable conditions
            is_aligned = sum(conditions) >= 2

            return is_aligned, alignment_desc

        else:
            # For sell (not used in this strategy, but included for completeness)
            return False, "Sell signals not implemented"

    def find_support_zones(
        self,
        df: pd.DataFrame,
        crossovers: List[Dict],
        current_price: float
    ) -> List[Dict]:
        """
        Find support zones based on crossovers

        Args:
            df: DataFrame with price and EMA data
            crossovers: List of detected crossovers
            current_price: Current stock price

        Returns:
            List of support zones near current price
        """
        support_zones = []

        for crossover in crossovers:
            cross_price = crossover['price']

            # Calculate distance from current price
            distance_pct = abs((current_price - cross_price) / cross_price * 100)

            # Check if within tolerance zone
            if distance_pct <= ZONE_TOLERANCE:
                # Determine if it's support or resistance
                if current_price >= cross_price:
                    zone_type = 'support'
                else:
                    zone_type = 'resistance'

                # For buy signals, we only care about support zones
                if zone_type == 'support':
                    support_zones.append({
                        'level': cross_price,
                        'distance_pct': distance_pct,
                        'crossover_info': crossover,
                        'zone_type': zone_type,
                        'strength': self._calculate_zone_strength(crossover)
                    })

        # Sort by strength (more recent crossovers are stronger)
        support_zones.sort(key=lambda x: x['strength'], reverse=True)

        logger.debug(f"Found {len(support_zones)} support zones near current price ${current_price:.2f}")
        return support_zones

    def _calculate_zone_strength(self, crossover: Dict) -> float:
        """
        Calculate the strength of a support/resistance zone

        Args:
            crossover: Crossover information dictionary

        Returns:
            Strength score (0-100)
        """
        # More recent = stronger
        # Involve more significant EMAs (62 > 38 > 24) = stronger

        days_ago = crossover.get('days_ago', 999)
        fast_ema = crossover.get('fast_ema', 24)
        slow_ema = crossover.get('slow_ema', 24)

        # Recency score (0-50): newer is better
        recency_score = max(0, 50 - (days_ago / 10))

        # EMA significance score (0-50): higher periods are more significant
        ema_score = (fast_ema + slow_ema) / (62 + 62) * 50

        total_score = recency_score + ema_score

        return min(100, total_score)

    def analyze_stock(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str = 'daily'
    ) -> Optional[Dict]:
        """
        Perform complete EMA analysis on a stock

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            timeframe: 'weekly' or 'daily'

        Returns:
            Analysis results dictionary or None if no signal
        """
        if df is None or df.empty:
            return None

        try:
            # Calculate EMAs
            df = self.calculate_emas(df)

            # Get current price
            current_price = float(df['Close'].iloc[-1])

            # Check EMA alignment
            is_aligned, alignment_desc = self.check_ema_alignment(df, for_buy=True)

            if not is_aligned:
                logger.debug(f"{symbol} ({timeframe}): EMAs not aligned for buy - {alignment_desc}")
                return None

            # Detect crossovers
            crossovers = self.detect_crossovers(df, timeframe)

            if not crossovers:
                logger.debug(f"{symbol} ({timeframe}): No relevant crossovers found")
                return None

            # Find support zones
            support_zones = self.find_support_zones(df, crossovers, current_price)

            if not support_zones:
                logger.debug(f"{symbol} ({timeframe}): No support zones near current price")
                return None

            # Get latest EMA values
            latest = df.iloc[-1]
            ema_values = {
                'ema_24': float(latest['EMA_24']),
                'ema_38': float(latest['EMA_38']),
                'ema_62': float(latest['EMA_62'])
            }

            # Best support zone (highest strength)
            best_support = support_zones[0]

            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': current_price,
                'ema_alignment': alignment_desc,
                'is_aligned': is_aligned,
                'support_zones': support_zones,
                'best_support_level': best_support['level'],
                'distance_to_support_pct': best_support['distance_pct'],
                'zone_strength': best_support['strength'],
                'crossover_info': best_support['crossover_info'],
                'ema_values': ema_values,
                'total_crossovers': len(crossovers)
            }

            logger.info(
                f"{symbol} ({timeframe}): BUY SIGNAL - "
                f"Price: ${current_price:.2f}, "
                f"Support: ${best_support['level']:.2f} "
                f"({best_support['distance_pct']:.1f}% away), "
                f"Alignment: {alignment_desc}"
            )

            return result

        except Exception as e:
            logger.error(f"Error analyzing {symbol} ({timeframe}): {e}")
            return None


# Singleton instance
ema_analyzer = EMAAnalyzer()
