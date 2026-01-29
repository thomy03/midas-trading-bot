"""
Trend Indicators - Trend-following indicators.

Includes:
- EMA (Exponential Moving Average)
- SMA (Simple Moving Average)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
- Supertrend
- Parabolic SAR
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


class EMA(BaseIndicator):
    """
    Exponential Moving Average (EMA)

    Gives more weight to recent prices.
    - Price above EMA: Bullish trend
    - Price below EMA: Bearish trend
    - EMA crossovers signal trend changes
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="EMA",
            category=IndicatorCategory.TREND,
            description="Exponential Moving Average - weighted moving average",
            default_params={"period": 20},
            param_ranges={"period": (2, 200)},
            outputs=["EMA"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["period"]
        col_name = f"EMA_{period}" if period != 20 else "EMA"

        df[col_name] = df["Close"].ewm(span=period, adjust=False).mean()

        # Also store as generic EMA for signal detection
        if col_name != "EMA":
            df["EMA"] = df[col_name]

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "EMA" not in df.columns:
            df = self.compute(df)

        price = df["Close"].iloc[-1]
        ema = df["EMA"].iloc[-1]
        prev_price = df["Close"].iloc[-2] if len(df) > 1 else price
        prev_ema = df["EMA"].iloc[-2] if len(df) > 1 else ema

        # Price crossing above EMA
        if prev_price <= prev_ema and price > ema:
            return IndicatorSignal(
                indicator_name="EMA",
                signal_type=SignalType.BULLISH,
                strength=0.7,
                value=ema,
                threshold=ema,
                message=f"Price crossed above EMA({self.params['period']})",
            )
        # Price crossing below EMA
        elif prev_price >= prev_ema and price < ema:
            return IndicatorSignal(
                indicator_name="EMA",
                signal_type=SignalType.BEARISH,
                strength=0.7,
                value=ema,
                threshold=ema,
                message=f"Price crossed below EMA({self.params['period']})",
            )

        return None


class SMA(BaseIndicator):
    """
    Simple Moving Average (SMA)

    Equal weight to all prices in the period.
    - Price above SMA: Bullish
    - Price below SMA: Bearish
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="SMA",
            category=IndicatorCategory.TREND,
            description="Simple Moving Average - unweighted average",
            default_params={"period": 20},
            param_ranges={"period": (2, 200)},
            outputs=["SMA"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["period"]
        col_name = f"SMA_{period}" if period != 20 else "SMA"

        df[col_name] = df["Close"].rolling(window=period).mean()

        if col_name != "SMA":
            df["SMA"] = df[col_name]

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "SMA" not in df.columns:
            df = self.compute(df)

        price = df["Close"].iloc[-1]
        sma = df["SMA"].iloc[-1]
        prev_price = df["Close"].iloc[-2] if len(df) > 1 else price
        prev_sma = df["SMA"].iloc[-2] if len(df) > 1 else sma

        if prev_price <= prev_sma and price > sma:
            return IndicatorSignal(
                indicator_name="SMA",
                signal_type=SignalType.BULLISH,
                strength=0.6,
                value=sma,
                threshold=sma,
                message=f"Price crossed above SMA({self.params['period']})",
            )
        elif prev_price >= prev_sma and price < sma:
            return IndicatorSignal(
                indicator_name="SMA",
                signal_type=SignalType.BEARISH,
                strength=0.6,
                value=sma,
                threshold=sma,
                message=f"Price crossed below SMA({self.params['period']})",
            )

        return None


class MACD(BaseIndicator):
    """
    Moving Average Convergence Divergence (MACD)

    Trend-following momentum indicator.
    - MACD > Signal: Bullish momentum
    - MACD < Signal: Bearish momentum
    - Histogram growing: Strengthening trend
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="MACD",
            category=IndicatorCategory.TREND,
            description="MACD - trend momentum indicator",
            default_params={"fast": 12, "slow": 26, "signal": 9},
            param_ranges={
                "fast": (5, 20),
                "slow": (20, 50),
                "signal": (5, 15),
            },
            outputs=["MACD", "MACD_SIGNAL", "MACD_HIST"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        fast = self.params["fast"]
        slow = self.params["slow"]
        signal_period = self.params["signal"]

        ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

        df["MACD"] = ema_fast - ema_slow
        df["MACD_SIGNAL"] = df["MACD"].ewm(span=signal_period, adjust=False).mean()
        df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "MACD" not in df.columns:
            df = self.compute(df)

        macd = df["MACD"].iloc[-1]
        signal = df["MACD_SIGNAL"].iloc[-1]
        hist = df["MACD_HIST"].iloc[-1]
        prev_macd = df["MACD"].iloc[-2] if len(df) > 1 else macd
        prev_signal = df["MACD_SIGNAL"].iloc[-2] if len(df) > 1 else signal

        # Bullish crossover
        if prev_macd <= prev_signal and macd > signal:
            strength = 0.8 if macd > 0 else 0.6
            return IndicatorSignal(
                indicator_name="MACD",
                signal_type=SignalType.BULLISH,
                strength=strength,
                value=macd,
                threshold=signal,
                message="MACD bullish crossover",
            )
        # Bearish crossover
        elif prev_macd >= prev_signal and macd < signal:
            strength = 0.8 if macd < 0 else 0.6
            return IndicatorSignal(
                indicator_name="MACD",
                signal_type=SignalType.BEARISH,
                strength=strength,
                value=macd,
                threshold=signal,
                message="MACD bearish crossover",
            )
        # Zero-line crossover
        elif prev_macd <= 0 < macd:
            return IndicatorSignal(
                indicator_name="MACD",
                signal_type=SignalType.BULLISH,
                strength=0.7,
                value=macd,
                threshold=0,
                message="MACD crossed above zero",
            )
        elif prev_macd >= 0 > macd:
            return IndicatorSignal(
                indicator_name="MACD",
                signal_type=SignalType.BEARISH,
                strength=0.7,
                value=macd,
                threshold=0,
                message="MACD crossed below zero",
            )

        return None


class ADX(BaseIndicator):
    """
    Average Directional Index (ADX)

    Measures trend strength (not direction).
    - ADX < 20: Weak/no trend
    - ADX 20-40: Developing trend
    - ADX > 40: Strong trend
    - +DI > -DI: Bullish
    - -DI > +DI: Bearish
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="ADX",
            category=IndicatorCategory.TREND,
            description="Average Directional Index - measures trend strength",
            default_params={"period": 14},
            param_ranges={"period": (7, 50)},
            outputs=["ADX", "PLUS_DI", "MINUS_DI"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["period"]

        # True Range
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Directional Movement
        up_move = df["High"] - df["High"].shift()
        down_move = df["Low"].shift() - df["Low"]

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        # Smoothed averages
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)

        # DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df["ADX"] = dx.ewm(alpha=1/period, min_periods=period).mean()
        df["PLUS_DI"] = plus_di
        df["MINUS_DI"] = minus_di

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "ADX" not in df.columns:
            df = self.compute(df)

        adx = df["ADX"].iloc[-1]
        plus_di = df["PLUS_DI"].iloc[-1]
        minus_di = df["MINUS_DI"].iloc[-1]
        prev_plus = df["PLUS_DI"].iloc[-2] if len(df) > 1 else plus_di
        prev_minus = df["MINUS_DI"].iloc[-2] if len(df) > 1 else minus_di

        trend_threshold = self.params.get("trend_threshold", 25)

        # Strong trend with DI crossover
        if adx > trend_threshold:
            if prev_plus <= prev_minus and plus_di > minus_di:
                return IndicatorSignal(
                    indicator_name="ADX",
                    signal_type=SignalType.BULLISH,
                    strength=min(1.0, adx / 50),
                    value=adx,
                    threshold=trend_threshold,
                    message=f"ADX bullish crossover (+DI>{-1}DI, ADX={adx:.1f})",
                )
            elif prev_plus >= prev_minus and plus_di < minus_di:
                return IndicatorSignal(
                    indicator_name="ADX",
                    signal_type=SignalType.BEARISH,
                    strength=min(1.0, adx / 50),
                    value=adx,
                    threshold=trend_threshold,
                    message=f"ADX bearish crossover (-DI>+DI, ADX={adx:.1f})",
                )

        return None


class Supertrend(BaseIndicator):
    """
    Supertrend Indicator

    Trend-following indicator based on ATR.
    - Price above Supertrend: Uptrend (buy)
    - Price below Supertrend: Downtrend (sell)
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="Supertrend",
            category=IndicatorCategory.TREND,
            description="Supertrend - ATR-based trend indicator",
            default_params={"period": 10, "multiplier": 3.0},
            param_ranges={
                "period": (5, 30),
                "multiplier": (1.0, 5.0),
            },
            outputs=["SUPERTREND", "SUPERTREND_DIRECTION"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["period"]
        multiplier = self.params["multiplier"]

        # ATR
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Basic bands
        hl2 = (df["High"] + df["Low"]) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        # Initialize Supertrend
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)

        # First valid value
        first_valid = atr.first_valid_index()
        if first_valid is not None:
            idx = df.index.get_loc(first_valid)
            supertrend.iloc[idx] = upper_band.iloc[idx]
            direction.iloc[idx] = -1  # Start bearish

        # Calculate Supertrend
        for i in range(1, len(df)):
            if pd.isna(atr.iloc[i]):
                continue

            prev_st = supertrend.iloc[i-1]
            prev_dir = direction.iloc[i-1]

            if pd.isna(prev_st):
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
                continue

            curr_close = df["Close"].iloc[i]
            curr_upper = upper_band.iloc[i]
            curr_lower = lower_band.iloc[i]

            if prev_dir == 1:  # Previous was uptrend
                if curr_close > prev_st:
                    supertrend.iloc[i] = max(curr_lower, prev_st)
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = curr_upper
                    direction.iloc[i] = -1
            else:  # Previous was downtrend
                if curr_close < prev_st:
                    supertrend.iloc[i] = min(curr_upper, prev_st)
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = curr_lower
                    direction.iloc[i] = 1

        df["SUPERTREND"] = supertrend
        df["SUPERTREND_DIRECTION"] = direction

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "SUPERTREND" not in df.columns:
            df = self.compute(df)

        direction = df["SUPERTREND_DIRECTION"].iloc[-1]
        prev_direction = df["SUPERTREND_DIRECTION"].iloc[-2] if len(df) > 1 else direction
        supertrend = df["SUPERTREND"].iloc[-1]

        if prev_direction == -1 and direction == 1:
            return IndicatorSignal(
                indicator_name="Supertrend",
                signal_type=SignalType.BULLISH,
                strength=0.8,
                value=supertrend,
                threshold=supertrend,
                message="Supertrend turned bullish",
            )
        elif prev_direction == 1 and direction == -1:
            return IndicatorSignal(
                indicator_name="Supertrend",
                signal_type=SignalType.BEARISH,
                strength=0.8,
                value=supertrend,
                threshold=supertrend,
                message="Supertrend turned bearish",
            )

        return None


class ParabolicSAR(BaseIndicator):
    """
    Parabolic SAR (Stop and Reverse)

    Trend-following indicator that provides potential entry/exit points.
    - SAR below price: Uptrend
    - SAR above price: Downtrend
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="ParabolicSAR",
            category=IndicatorCategory.TREND,
            description="Parabolic SAR - trailing stop indicator",
            default_params={"af_start": 0.02, "af_increment": 0.02, "af_max": 0.2},
            param_ranges={
                "af_start": (0.01, 0.1),
                "af_increment": (0.01, 0.1),
                "af_max": (0.1, 0.5),
            },
            outputs=["PSAR", "PSAR_DIRECTION"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        af_start = self.params["af_start"]
        af_increment = self.params["af_increment"]
        af_max = self.params["af_max"]

        psar = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        af = af_start
        ep = df["Low"].iloc[0]

        # Initialize
        psar.iloc[0] = df["High"].iloc[0]
        direction.iloc[0] = -1  # Start bearish

        for i in range(1, len(df)):
            prev_psar = psar.iloc[i-1]
            prev_dir = direction.iloc[i-1]

            if prev_dir == 1:  # Uptrend
                psar.iloc[i] = prev_psar + af * (ep - prev_psar)
                psar.iloc[i] = min(psar.iloc[i], df["Low"].iloc[i-1], df["Low"].iloc[i-2] if i > 1 else df["Low"].iloc[i-1])

                if df["Low"].iloc[i] < psar.iloc[i]:
                    direction.iloc[i] = -1
                    psar.iloc[i] = ep
                    af = af_start
                    ep = df["Low"].iloc[i]
                else:
                    direction.iloc[i] = 1
                    if df["High"].iloc[i] > ep:
                        ep = df["High"].iloc[i]
                        af = min(af + af_increment, af_max)
            else:  # Downtrend
                psar.iloc[i] = prev_psar - af * (prev_psar - ep)
                psar.iloc[i] = max(psar.iloc[i], df["High"].iloc[i-1], df["High"].iloc[i-2] if i > 1 else df["High"].iloc[i-1])

                if df["High"].iloc[i] > psar.iloc[i]:
                    direction.iloc[i] = 1
                    psar.iloc[i] = ep
                    af = af_start
                    ep = df["High"].iloc[i]
                else:
                    direction.iloc[i] = -1
                    if df["Low"].iloc[i] < ep:
                        ep = df["Low"].iloc[i]
                        af = min(af + af_increment, af_max)

        df["PSAR"] = psar
        df["PSAR_DIRECTION"] = direction

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "PSAR" not in df.columns:
            df = self.compute(df)

        direction = df["PSAR_DIRECTION"].iloc[-1]
        prev_direction = df["PSAR_DIRECTION"].iloc[-2] if len(df) > 1 else direction
        psar = df["PSAR"].iloc[-1]

        if prev_direction == -1 and direction == 1:
            return IndicatorSignal(
                indicator_name="ParabolicSAR",
                signal_type=SignalType.BULLISH,
                strength=0.7,
                value=psar,
                threshold=psar,
                message="Parabolic SAR turned bullish",
            )
        elif prev_direction == 1 and direction == -1:
            return IndicatorSignal(
                indicator_name="ParabolicSAR",
                signal_type=SignalType.BEARISH,
                strength=0.7,
                value=psar,
                threshold=psar,
                message="Parabolic SAR turned bearish",
            )

        return None


# List of all indicator classes in this module (for auto-registration)
INDICATORS = [EMA, SMA, MACD, ADX, Supertrend, ParabolicSAR]
