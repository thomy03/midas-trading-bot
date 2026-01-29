"""
Pattern Indicators - Price pattern and level-based indicators.

Includes:
- SupportResistance (Key price levels)
- FibonacciRetracement
- PivotPoints (Standard, Camarilla, Woodie)
- SwingHighLow
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from scipy.signal import argrelextrema

from .library import (
    BaseIndicator,
    IndicatorMetadata,
    IndicatorCategory,
    IndicatorSignal,
    SignalType,
)


class SupportResistance(BaseIndicator):
    """
    Support and Resistance Levels

    Identifies key price levels where price has historically
    reversed or consolidated. Uses local highs/lows.
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="SupportResistance",
            category=IndicatorCategory.PATTERN,
            description="Support and Resistance levels from price action",
            default_params={"lookback": 50, "tolerance": 0.02, "min_touches": 2},
            param_ranges={
                "lookback": (20, 200),
                "tolerance": (0.005, 0.05),
                "min_touches": (2, 5),
            },
            outputs=["SUPPORT", "RESISTANCE", "NEAREST_SUPPORT", "NEAREST_RESISTANCE"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        lookback = self.params["lookback"]
        tolerance = self.params["tolerance"]
        min_touches = self.params["min_touches"]

        # Find local highs and lows
        order = 5  # How many bars on each side to consider

        # Use argrelextrema to find local extrema
        high_indices = argrelextrema(df["High"].values, np.greater, order=order)[0]
        low_indices = argrelextrema(df["Low"].values, np.less, order=order)[0]

        # Get the price levels
        resistance_levels = df["High"].iloc[high_indices].values if len(high_indices) > 0 else []
        support_levels = df["Low"].iloc[low_indices].values if len(low_indices) > 0 else []

        # Cluster nearby levels
        def cluster_levels(levels: np.ndarray, tol: float) -> List[float]:
            if len(levels) == 0:
                return []

            sorted_levels = np.sort(levels)
            clusters = []
            current_cluster = [sorted_levels[0]]

            for level in sorted_levels[1:]:
                if (level - current_cluster[0]) / current_cluster[0] <= tol:
                    current_cluster.append(level)
                else:
                    if len(current_cluster) >= min_touches:
                        clusters.append(np.mean(current_cluster))
                    current_cluster = [level]

            if len(current_cluster) >= min_touches:
                clusters.append(np.mean(current_cluster))

            return clusters

        support_clusters = cluster_levels(np.array(support_levels), tolerance)
        resistance_clusters = cluster_levels(np.array(resistance_levels), tolerance)

        # Store as comma-separated strings (for simplicity)
        df["SUPPORT"] = ",".join([f"{s:.2f}" for s in support_clusters]) if support_clusters else ""
        df["RESISTANCE"] = ",".join([f"{r:.2f}" for r in resistance_clusters]) if resistance_clusters else ""

        # Find nearest levels to current price
        current_price = df["Close"].iloc[-1]

        supports_below = [s for s in support_clusters if s < current_price]
        resistances_above = [r for r in resistance_clusters if r > current_price]

        df["NEAREST_SUPPORT"] = max(supports_below) if supports_below else np.nan
        df["NEAREST_RESISTANCE"] = min(resistances_above) if resistances_above else np.nan

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "NEAREST_SUPPORT" not in df.columns:
            df = self.compute(df)

        price = df["Close"].iloc[-1]
        support = df["NEAREST_SUPPORT"].iloc[-1]
        resistance = df["NEAREST_RESISTANCE"].iloc[-1]

        tolerance = self.params["tolerance"]

        # Price near support
        if not pd.isna(support) and abs(price - support) / price <= tolerance:
            return IndicatorSignal(
                indicator_name="SupportResistance",
                signal_type=SignalType.BULLISH,
                strength=0.7,
                value=price,
                threshold=support,
                message=f"Price near support level (${support:.2f})",
            )
        # Price near resistance
        elif not pd.isna(resistance) and abs(price - resistance) / price <= tolerance:
            return IndicatorSignal(
                indicator_name="SupportResistance",
                signal_type=SignalType.BEARISH,
                strength=0.7,
                value=price,
                threshold=resistance,
                message=f"Price near resistance level (${resistance:.2f})",
            )

        return None


class FibonacciRetracement(BaseIndicator):
    """
    Fibonacci Retracement Levels

    Calculates Fibonacci retracement levels from recent swing high/low.
    Key levels: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="FibonacciRetracement",
            category=IndicatorCategory.PATTERN,
            description="Fibonacci Retracement levels",
            default_params={"lookback": 50},
            param_ranges={"lookback": (20, 200)},
            outputs=["FIB_0", "FIB_236", "FIB_382", "FIB_500", "FIB_618", "FIB_786", "FIB_100"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        lookback = self.params["lookback"]

        # Get recent high and low
        recent_df = df.tail(lookback)
        swing_high = recent_df["High"].max()
        swing_low = recent_df["Low"].min()

        high_idx = recent_df["High"].idxmax()
        low_idx = recent_df["Low"].idxmin()

        # Determine trend direction
        if high_idx > low_idx:
            # Uptrend: low came first, retracement from high
            diff = swing_high - swing_low
            df["FIB_0"] = swing_high
            df["FIB_236"] = swing_high - 0.236 * diff
            df["FIB_382"] = swing_high - 0.382 * diff
            df["FIB_500"] = swing_high - 0.500 * diff
            df["FIB_618"] = swing_high - 0.618 * diff
            df["FIB_786"] = swing_high - 0.786 * diff
            df["FIB_100"] = swing_low
            df["FIB_TREND"] = 1  # Uptrend
        else:
            # Downtrend: high came first, retracement from low
            diff = swing_high - swing_low
            df["FIB_0"] = swing_low
            df["FIB_236"] = swing_low + 0.236 * diff
            df["FIB_382"] = swing_low + 0.382 * diff
            df["FIB_500"] = swing_low + 0.500 * diff
            df["FIB_618"] = swing_low + 0.618 * diff
            df["FIB_786"] = swing_low + 0.786 * diff
            df["FIB_100"] = swing_high
            df["FIB_TREND"] = -1  # Downtrend

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "FIB_618" not in df.columns:
            df = self.compute(df)

        price = df["Close"].iloc[-1]
        trend = df["FIB_TREND"].iloc[-1]
        fib_levels = {
            "0%": df["FIB_0"].iloc[-1],
            "23.6%": df["FIB_236"].iloc[-1],
            "38.2%": df["FIB_382"].iloc[-1],
            "50%": df["FIB_500"].iloc[-1],
            "61.8%": df["FIB_618"].iloc[-1],
            "78.6%": df["FIB_786"].iloc[-1],
            "100%": df["FIB_100"].iloc[-1],
        }

        # Check proximity to key levels
        tolerance = 0.01  # 1%

        for level_name, level_price in fib_levels.items():
            if abs(price - level_price) / price <= tolerance:
                # Key Fibonacci levels for signals
                if level_name in ["38.2%", "50%", "61.8%"]:
                    if trend == 1:
                        return IndicatorSignal(
                            indicator_name="FibonacciRetracement",
                            signal_type=SignalType.BULLISH,
                            strength=0.7 if level_name == "61.8%" else 0.6,
                            value=price,
                            threshold=level_price,
                            message=f"Price at Fibonacci {level_name} retracement (${level_price:.2f})",
                        )
                    else:
                        return IndicatorSignal(
                            indicator_name="FibonacciRetracement",
                            signal_type=SignalType.BEARISH,
                            strength=0.7 if level_name == "61.8%" else 0.6,
                            value=price,
                            threshold=level_price,
                            message=f"Price at Fibonacci {level_name} retracement (${level_price:.2f})",
                        )

        return None


class PivotPoints(BaseIndicator):
    """
    Pivot Points

    Calculates classic pivot points and support/resistance levels.
    Used for intraday support/resistance.
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="PivotPoints",
            category=IndicatorCategory.PATTERN,
            description="Pivot Points - S/R levels from previous period",
            default_params={"type": "classic"},  # classic, woodie, camarilla
            param_ranges={},
            outputs=["PIVOT", "R1", "R2", "R3", "S1", "S2", "S3"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        pivot_type = self.params.get("type", "classic")

        # Use previous period's OHLC
        prev_high = df["High"].shift(1)
        prev_low = df["Low"].shift(1)
        prev_close = df["Close"].shift(1)

        if pivot_type == "classic":
            # Classic Pivot Points
            df["PIVOT"] = (prev_high + prev_low + prev_close) / 3
            df["R1"] = 2 * df["PIVOT"] - prev_low
            df["S1"] = 2 * df["PIVOT"] - prev_high
            df["R2"] = df["PIVOT"] + (prev_high - prev_low)
            df["S2"] = df["PIVOT"] - (prev_high - prev_low)
            df["R3"] = prev_high + 2 * (df["PIVOT"] - prev_low)
            df["S3"] = prev_low - 2 * (prev_high - df["PIVOT"])

        elif pivot_type == "woodie":
            # Woodie Pivot Points
            df["PIVOT"] = (prev_high + prev_low + 2 * df["Close"]) / 4
            df["R1"] = 2 * df["PIVOT"] - prev_low
            df["S1"] = 2 * df["PIVOT"] - prev_high
            df["R2"] = df["PIVOT"] + (prev_high - prev_low)
            df["S2"] = df["PIVOT"] - (prev_high - prev_low)
            df["R3"] = df["R1"] + (prev_high - prev_low)
            df["S3"] = df["S1"] - (prev_high - prev_low)

        elif pivot_type == "camarilla":
            # Camarilla Pivot Points
            df["PIVOT"] = (prev_high + prev_low + prev_close) / 3
            hl_range = prev_high - prev_low
            df["R1"] = prev_close + hl_range * 1.1 / 12
            df["R2"] = prev_close + hl_range * 1.1 / 6
            df["R3"] = prev_close + hl_range * 1.1 / 4
            df["S1"] = prev_close - hl_range * 1.1 / 12
            df["S2"] = prev_close - hl_range * 1.1 / 6
            df["S3"] = prev_close - hl_range * 1.1 / 4

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "PIVOT" not in df.columns:
            df = self.compute(df)

        price = df["Close"].iloc[-1]
        prev_price = df["Close"].iloc[-2] if len(df) > 1 else price
        pivot = df["PIVOT"].iloc[-1]
        r1 = df["R1"].iloc[-1]
        s1 = df["S1"].iloc[-1]

        prev_pivot = df["PIVOT"].iloc[-2] if len(df) > 1 else pivot

        # Price crossing pivot from below
        if prev_price <= prev_pivot and price > pivot:
            return IndicatorSignal(
                indicator_name="PivotPoints",
                signal_type=SignalType.BULLISH,
                strength=0.6,
                value=price,
                threshold=pivot,
                message=f"Price crossed above pivot (${pivot:.2f})",
            )
        # Price crossing pivot from above
        elif prev_price >= prev_pivot and price < pivot:
            return IndicatorSignal(
                indicator_name="PivotPoints",
                signal_type=SignalType.BEARISH,
                strength=0.6,
                value=price,
                threshold=pivot,
                message=f"Price crossed below pivot (${pivot:.2f})",
            )
        # Price near S1 support
        elif abs(price - s1) / price <= 0.005:
            return IndicatorSignal(
                indicator_name="PivotPoints",
                signal_type=SignalType.BULLISH,
                strength=0.7,
                value=price,
                threshold=s1,
                message=f"Price at S1 support (${s1:.2f})",
            )
        # Price near R1 resistance
        elif abs(price - r1) / price <= 0.005:
            return IndicatorSignal(
                indicator_name="PivotPoints",
                signal_type=SignalType.BEARISH,
                strength=0.7,
                value=price,
                threshold=r1,
                message=f"Price at R1 resistance (${r1:.2f})",
            )

        return None


class SwingHighLow(BaseIndicator):
    """
    Swing High/Low Detection

    Identifies swing highs and lows for trend analysis.
    - Higher highs + Higher lows = Uptrend
    - Lower highs + Lower lows = Downtrend
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="SwingHighLow",
            category=IndicatorCategory.PATTERN,
            description="Swing High/Low detection for trend analysis",
            default_params={"order": 5},  # Bars on each side
            param_ranges={"order": (3, 20)},
            outputs=["SWING_HIGH", "SWING_LOW", "TREND_STRUCTURE"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        order = self.params["order"]

        # Find swing highs and lows using argrelextrema
        high_indices = argrelextrema(df["High"].values, np.greater, order=order)[0]
        low_indices = argrelextrema(df["Low"].values, np.less, order=order)[0]

        # Create swing columns
        df["SWING_HIGH"] = np.nan
        df["SWING_LOW"] = np.nan

        if len(high_indices) > 0:
            df.iloc[high_indices, df.columns.get_loc("SWING_HIGH")] = df["High"].iloc[high_indices].values

        if len(low_indices) > 0:
            df.iloc[low_indices, df.columns.get_loc("SWING_LOW")] = df["Low"].iloc[low_indices].values

        # Determine trend structure
        # Get last 4 swing points
        recent_highs = df["SWING_HIGH"].dropna().tail(2).values
        recent_lows = df["SWING_LOW"].dropna().tail(2).values

        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            hh = recent_highs[-1] > recent_highs[-2]  # Higher high
            hl = recent_lows[-1] > recent_lows[-2]    # Higher low
            lh = recent_highs[-1] < recent_highs[-2]  # Lower high
            ll = recent_lows[-1] < recent_lows[-2]    # Lower low

            if hh and hl:
                df["TREND_STRUCTURE"] = 1  # Uptrend
            elif lh and ll:
                df["TREND_STRUCTURE"] = -1  # Downtrend
            else:
                df["TREND_STRUCTURE"] = 0  # Ranging/Indeterminate
        else:
            df["TREND_STRUCTURE"] = 0

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "TREND_STRUCTURE" not in df.columns:
            df = self.compute(df)

        trend = df["TREND_STRUCTURE"].iloc[-1]
        prev_trend = df["TREND_STRUCTURE"].iloc[-2] if len(df) > 1 else trend

        # Trend structure change
        if prev_trend != 1 and trend == 1:
            return IndicatorSignal(
                indicator_name="SwingHighLow",
                signal_type=SignalType.BULLISH,
                strength=0.8,
                value=trend,
                message="Trend structure turned bullish (Higher Highs, Higher Lows)",
            )
        elif prev_trend != -1 and trend == -1:
            return IndicatorSignal(
                indicator_name="SwingHighLow",
                signal_type=SignalType.BEARISH,
                strength=0.8,
                value=trend,
                message="Trend structure turned bearish (Lower Highs, Lower Lows)",
            )

        return None


# List of all indicator classes in this module (for auto-registration)
INDICATORS = [SupportResistance, FibonacciRetracement, PivotPoints, SwingHighLow]
