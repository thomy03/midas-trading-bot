"""
Volatility Indicators - Volatility-based technical indicators.

Includes:
- ATR (Average True Range)
- BollingerBands
- KeltnerChannel
- DonchianChannel
- StandardDeviation
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from .library import (
    BaseIndicator,
    IndicatorMetadata,
    IndicatorCategory,
    IndicatorSignal,
    SignalType,
)


class ATR(BaseIndicator):
    """
    Average True Range (ATR)

    Measures market volatility.
    - High ATR: High volatility
    - Low ATR: Low volatility
    Used for stop-loss calculation and position sizing.
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="ATR",
            category=IndicatorCategory.VOLATILITY,
            description="Average True Range - volatility measure",
            default_params={"period": 14},
            param_ranges={"period": (5, 50)},
            outputs=["ATR", "ATR_PERCENT"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["period"]

        # True Range
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR (using Wilder's smoothing)
        df["ATR"] = tr.ewm(alpha=1/period, min_periods=period).mean()

        # ATR as percentage of price
        df["ATR_PERCENT"] = (df["ATR"] / df["Close"]) * 100

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "ATR" not in df.columns:
            df = self.compute(df)

        atr = df["ATR"].iloc[-1]
        atr_pct = df["ATR_PERCENT"].iloc[-1]

        # Calculate ATR relative to its own average
        atr_sma = df["ATR"].rolling(window=20).mean().iloc[-1]
        atr_ratio = atr / atr_sma if atr_sma > 0 else 1.0

        # High volatility warning
        if atr_ratio > 1.5:
            return IndicatorSignal(
                indicator_name="ATR",
                signal_type=SignalType.NEUTRAL,
                strength=min(1.0, atr_ratio / 2),
                value=atr,
                threshold=atr_sma * 1.5,
                message=f"High volatility ({atr_ratio:.1f}x normal, ATR={atr_pct:.2f}%)",
            )
        # Volatility contraction (potential breakout setup)
        elif atr_ratio < 0.7:
            return IndicatorSignal(
                indicator_name="ATR",
                signal_type=SignalType.NEUTRAL,
                strength=0.5,
                value=atr,
                threshold=atr_sma * 0.7,
                message=f"Low volatility (potential breakout, ATR={atr_pct:.2f}%)",
            )

        return None


class BollingerBands(BaseIndicator):
    """
    Bollinger Bands

    Volatility bands around a moving average.
    - Price near upper band: Overbought
    - Price near lower band: Oversold
    - Band squeeze: Low volatility, potential breakout
    - Band expansion: High volatility
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="BollingerBands",
            category=IndicatorCategory.VOLATILITY,
            description="Bollinger Bands - volatility bands around SMA",
            default_params={"period": 20, "std_dev": 2.0},
            param_ranges={
                "period": (5, 50),
                "std_dev": (1.0, 3.0),
            },
            outputs=["BB_UPPER", "BB_MIDDLE", "BB_LOWER", "BB_WIDTH", "BB_PERCENT"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["period"]
        std_dev = self.params["std_dev"]

        # Middle band (SMA)
        df["BB_MIDDLE"] = df["Close"].rolling(window=period).mean()

        # Standard deviation
        std = df["Close"].rolling(window=period).std()

        # Upper and Lower bands
        df["BB_UPPER"] = df["BB_MIDDLE"] + (std * std_dev)
        df["BB_LOWER"] = df["BB_MIDDLE"] - (std * std_dev)

        # Bandwidth (volatility measure)
        df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / df["BB_MIDDLE"]

        # %B (position within bands)
        df["BB_PERCENT"] = (df["Close"] - df["BB_LOWER"]) / (df["BB_UPPER"] - df["BB_LOWER"])

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "BB_UPPER" not in df.columns:
            df = self.compute(df)

        price = df["Close"].iloc[-1]
        upper = df["BB_UPPER"].iloc[-1]
        lower = df["BB_LOWER"].iloc[-1]
        middle = df["BB_MIDDLE"].iloc[-1]
        bb_pct = df["BB_PERCENT"].iloc[-1]
        bb_width = df["BB_WIDTH"].iloc[-1]

        prev_price = df["Close"].iloc[-2] if len(df) > 1 else price
        prev_lower = df["BB_LOWER"].iloc[-2] if len(df) > 1 else lower
        prev_upper = df["BB_UPPER"].iloc[-2] if len(df) > 1 else upper

        # Price crossing lower band from below (oversold bounce)
        if prev_price <= prev_lower and price > lower:
            return IndicatorSignal(
                indicator_name="BollingerBands",
                signal_type=SignalType.BULLISH,
                strength=0.7,
                value=bb_pct,
                threshold=0,
                message="Price bounced off lower Bollinger Band",
            )
        # Price crossing upper band from above (overbought pullback)
        elif prev_price >= prev_upper and price < upper:
            return IndicatorSignal(
                indicator_name="BollingerBands",
                signal_type=SignalType.BEARISH,
                strength=0.7,
                value=bb_pct,
                threshold=1,
                message="Price pulled back from upper Bollinger Band",
            )
        # Oversold (below lower band)
        elif price < lower:
            return IndicatorSignal(
                indicator_name="BollingerBands",
                signal_type=SignalType.BULLISH,
                strength=0.5,
                value=bb_pct,
                threshold=0,
                message=f"Price below lower Bollinger Band (%B={bb_pct:.2f})",
            )
        # Overbought (above upper band)
        elif price > upper:
            return IndicatorSignal(
                indicator_name="BollingerBands",
                signal_type=SignalType.BEARISH,
                strength=0.5,
                value=bb_pct,
                threshold=1,
                message=f"Price above upper Bollinger Band (%B={bb_pct:.2f})",
            )

        return None


class KeltnerChannel(BaseIndicator):
    """
    Keltner Channel

    Volatility bands based on ATR around an EMA.
    Similar to Bollinger Bands but uses ATR instead of standard deviation.
    - Price above upper: Strong uptrend
    - Price below lower: Strong downtrend
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="KeltnerChannel",
            category=IndicatorCategory.VOLATILITY,
            description="Keltner Channel - ATR-based volatility bands",
            default_params={"ema_period": 20, "atr_period": 10, "multiplier": 2.0},
            param_ranges={
                "ema_period": (10, 50),
                "atr_period": (5, 30),
                "multiplier": (1.0, 3.0),
            },
            outputs=["KC_UPPER", "KC_MIDDLE", "KC_LOWER"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ema_period = self.params["ema_period"]
        atr_period = self.params["atr_period"]
        multiplier = self.params["multiplier"]

        # Middle line (EMA)
        df["KC_MIDDLE"] = df["Close"].ewm(span=ema_period, adjust=False).mean()

        # ATR
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/atr_period, min_periods=atr_period).mean()

        # Upper and Lower channels
        df["KC_UPPER"] = df["KC_MIDDLE"] + (atr * multiplier)
        df["KC_LOWER"] = df["KC_MIDDLE"] - (atr * multiplier)

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "KC_UPPER" not in df.columns:
            df = self.compute(df)

        price = df["Close"].iloc[-1]
        upper = df["KC_UPPER"].iloc[-1]
        lower = df["KC_LOWER"].iloc[-1]
        middle = df["KC_MIDDLE"].iloc[-1]

        prev_price = df["Close"].iloc[-2] if len(df) > 1 else price
        prev_upper = df["KC_UPPER"].iloc[-2] if len(df) > 1 else upper
        prev_lower = df["KC_LOWER"].iloc[-2] if len(df) > 1 else lower

        # Breakout above upper channel
        if prev_price <= prev_upper and price > upper:
            return IndicatorSignal(
                indicator_name="KeltnerChannel",
                signal_type=SignalType.BULLISH,
                strength=0.8,
                value=price,
                threshold=upper,
                message="Keltner Channel breakout (bullish)",
            )
        # Breakdown below lower channel
        elif prev_price >= prev_lower and price < lower:
            return IndicatorSignal(
                indicator_name="KeltnerChannel",
                signal_type=SignalType.BEARISH,
                strength=0.8,
                value=price,
                threshold=lower,
                message="Keltner Channel breakdown (bearish)",
            )
        # Bounce from lower channel
        elif price < lower * 1.02 and price > lower:
            return IndicatorSignal(
                indicator_name="KeltnerChannel",
                signal_type=SignalType.BULLISH,
                strength=0.6,
                value=price,
                threshold=lower,
                message="Price near lower Keltner Channel",
            )

        return None


class DonchianChannel(BaseIndicator):
    """
    Donchian Channel

    High/Low channel over a period.
    - Upper: Highest high
    - Lower: Lowest low
    Used for breakout trading.
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="DonchianChannel",
            category=IndicatorCategory.VOLATILITY,
            description="Donchian Channel - high/low breakout bands",
            default_params={"period": 20},
            param_ranges={"period": (5, 100)},
            outputs=["DC_UPPER", "DC_MIDDLE", "DC_LOWER"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["period"]

        df["DC_UPPER"] = df["High"].rolling(window=period).max()
        df["DC_LOWER"] = df["Low"].rolling(window=period).min()
        df["DC_MIDDLE"] = (df["DC_UPPER"] + df["DC_LOWER"]) / 2

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "DC_UPPER" not in df.columns:
            df = self.compute(df)

        price = df["Close"].iloc[-1]
        high = df["High"].iloc[-1]
        low = df["Low"].iloc[-1]
        upper = df["DC_UPPER"].iloc[-1]
        lower = df["DC_LOWER"].iloc[-1]

        prev_high = df["High"].iloc[-2] if len(df) > 1 else high
        prev_upper = df["DC_UPPER"].iloc[-2] if len(df) > 1 else upper
        prev_low = df["Low"].iloc[-2] if len(df) > 1 else low
        prev_lower = df["DC_LOWER"].iloc[-2] if len(df) > 1 else lower

        # Breakout above upper channel
        if prev_high <= prev_upper and high > upper:
            return IndicatorSignal(
                indicator_name="DonchianChannel",
                signal_type=SignalType.BULLISH,
                strength=0.8,
                value=high,
                threshold=upper,
                message=f"Donchian breakout ({self.params['period']}-period high)",
            )
        # Breakdown below lower channel
        elif prev_low >= prev_lower and low < lower:
            return IndicatorSignal(
                indicator_name="DonchianChannel",
                signal_type=SignalType.BEARISH,
                strength=0.8,
                value=low,
                threshold=lower,
                message=f"Donchian breakdown ({self.params['period']}-period low)",
            )

        return None


class StandardDeviation(BaseIndicator):
    """
    Standard Deviation

    Measures price dispersion around the mean.
    High StdDev = High volatility
    Low StdDev = Low volatility (potential breakout)
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="StandardDeviation",
            category=IndicatorCategory.VOLATILITY,
            description="Standard Deviation - price dispersion measure",
            default_params={"period": 20},
            param_ranges={"period": (5, 100)},
            outputs=["STD_DEV", "STD_DEV_PERCENT"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["period"]

        df["STD_DEV"] = df["Close"].rolling(window=period).std()
        df["STD_DEV_PERCENT"] = (df["STD_DEV"] / df["Close"]) * 100

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "STD_DEV" not in df.columns:
            df = self.compute(df)

        std = df["STD_DEV"].iloc[-1]
        std_pct = df["STD_DEV_PERCENT"].iloc[-1]

        # Compare to average
        std_avg = df["STD_DEV"].rolling(window=50).mean().iloc[-1]
        std_ratio = std / std_avg if std_avg > 0 else 1.0

        if std_ratio > 1.5:
            return IndicatorSignal(
                indicator_name="StandardDeviation",
                signal_type=SignalType.NEUTRAL,
                strength=min(1.0, std_ratio / 2),
                value=std_pct,
                message=f"High volatility ({std_ratio:.1f}x normal)",
            )
        elif std_ratio < 0.6:
            return IndicatorSignal(
                indicator_name="StandardDeviation",
                signal_type=SignalType.NEUTRAL,
                strength=0.5,
                value=std_pct,
                message=f"Low volatility - potential breakout setup",
            )

        return None


# List of all indicator classes in this module (for auto-registration)
INDICATORS = [ATR, BollingerBands, KeltnerChannel, DonchianChannel, StandardDeviation]
