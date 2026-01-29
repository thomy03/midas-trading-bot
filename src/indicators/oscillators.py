"""
Oscillator Indicators - Momentum-based indicators.

Includes:
- RSI (Relative Strength Index)
- Stochastic Oscillator
- Williams %R
- ROC (Rate of Change)
- MFI (Money Flow Index)
- CCI (Commodity Channel Index)
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


class RSI(BaseIndicator):
    """
    Relative Strength Index (RSI)

    Measures the speed and magnitude of price changes.
    - RSI < 30: Oversold (potential buy signal)
    - RSI > 70: Overbought (potential sell signal)
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="RSI",
            category=IndicatorCategory.MOMENTUM,
            description="Relative Strength Index - measures momentum on a 0-100 scale",
            default_params={"period": 14},
            param_ranges={"period": (2, 50)},
            outputs=["RSI"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["period"]

        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "RSI" not in df.columns:
            df = self.compute(df)

        rsi = df["RSI"].iloc[-1]
        prev_rsi = df["RSI"].iloc[-2] if len(df) > 1 else rsi

        oversold = self.params.get("oversold", 30)
        overbought = self.params.get("overbought", 70)

        # Check for crossovers
        if prev_rsi <= oversold < rsi:
            return IndicatorSignal(
                indicator_name="RSI",
                signal_type=SignalType.BULLISH,
                strength=min(1.0, (oversold - prev_rsi + 10) / 20),
                value=rsi,
                threshold=oversold,
                message=f"RSI crossed above oversold level ({oversold})",
            )
        elif prev_rsi >= overbought > rsi:
            return IndicatorSignal(
                indicator_name="RSI",
                signal_type=SignalType.BEARISH,
                strength=min(1.0, (prev_rsi - overbought + 10) / 20),
                value=rsi,
                threshold=overbought,
                message=f"RSI crossed below overbought level ({overbought})",
            )
        elif rsi < oversold:
            return IndicatorSignal(
                indicator_name="RSI",
                signal_type=SignalType.BULLISH,
                strength=0.5 + (oversold - rsi) / 60,
                value=rsi,
                threshold=oversold,
                message=f"RSI in oversold territory ({rsi:.1f})",
            )
        elif rsi > overbought:
            return IndicatorSignal(
                indicator_name="RSI",
                signal_type=SignalType.BEARISH,
                strength=0.5 + (rsi - overbought) / 60,
                value=rsi,
                threshold=overbought,
                message=f"RSI in overbought territory ({rsi:.1f})",
            )

        return None


class Stochastic(BaseIndicator):
    """
    Stochastic Oscillator

    Compares closing price to the price range over a period.
    - %K < 20: Oversold
    - %K > 80: Overbought
    - %K crossing %D from below: Buy signal
    - %K crossing %D from above: Sell signal
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="Stochastic",
            category=IndicatorCategory.MOMENTUM,
            description="Stochastic Oscillator - compares close to price range",
            default_params={"k_period": 14, "d_period": 3, "smooth_k": 3},
            param_ranges={
                "k_period": (5, 50),
                "d_period": (2, 10),
                "smooth_k": (1, 10),
            },
            outputs=["STOCH_K", "STOCH_D"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        k_period = self.params["k_period"]
        d_period = self.params["d_period"]
        smooth_k = self.params["smooth_k"]

        low_min = df["Low"].rolling(window=k_period).min()
        high_max = df["High"].rolling(window=k_period).max()

        stoch_k = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
        df["STOCH_K"] = stoch_k.rolling(window=smooth_k).mean()
        df["STOCH_D"] = df["STOCH_K"].rolling(window=d_period).mean()

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "STOCH_K" not in df.columns:
            df = self.compute(df)

        k = df["STOCH_K"].iloc[-1]
        d = df["STOCH_D"].iloc[-1]
        prev_k = df["STOCH_K"].iloc[-2] if len(df) > 1 else k
        prev_d = df["STOCH_D"].iloc[-2] if len(df) > 1 else d

        oversold = self.params.get("oversold", 20)
        overbought = self.params.get("overbought", 80)

        # Bullish crossover in oversold zone
        if prev_k <= prev_d and k > d and k < oversold + 10:
            return IndicatorSignal(
                indicator_name="Stochastic",
                signal_type=SignalType.BULLISH,
                strength=0.8 if k < oversold else 0.6,
                value=k,
                threshold=oversold,
                message=f"Stochastic bullish crossover (%K={k:.1f} > %D={d:.1f})",
            )
        # Bearish crossover in overbought zone
        elif prev_k >= prev_d and k < d and k > overbought - 10:
            return IndicatorSignal(
                indicator_name="Stochastic",
                signal_type=SignalType.BEARISH,
                strength=0.8 if k > overbought else 0.6,
                value=k,
                threshold=overbought,
                message=f"Stochastic bearish crossover (%K={k:.1f} < %D={d:.1f})",
            )

        return None


class WilliamsR(BaseIndicator):
    """
    Williams %R

    Similar to Stochastic but inverted scale.
    - %R > -20: Overbought
    - %R < -80: Oversold
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="WilliamsR",
            category=IndicatorCategory.MOMENTUM,
            description="Williams %R - momentum indicator ranging from -100 to 0",
            default_params={"period": 14},
            param_ranges={"period": (5, 50)},
            outputs=["WILLIAMS_R"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["period"]

        high_max = df["High"].rolling(window=period).max()
        low_min = df["Low"].rolling(window=period).min()

        df["WILLIAMS_R"] = -100 * (high_max - df["Close"]) / (high_max - low_min).replace(0, np.nan)

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "WILLIAMS_R" not in df.columns:
            df = self.compute(df)

        wr = df["WILLIAMS_R"].iloc[-1]
        prev_wr = df["WILLIAMS_R"].iloc[-2] if len(df) > 1 else wr

        oversold = self.params.get("oversold", -80)
        overbought = self.params.get("overbought", -20)

        if prev_wr <= oversold < wr:
            return IndicatorSignal(
                indicator_name="WilliamsR",
                signal_type=SignalType.BULLISH,
                strength=0.7,
                value=wr,
                threshold=oversold,
                message=f"Williams %R crossed above oversold ({wr:.1f})",
            )
        elif prev_wr >= overbought > wr:
            return IndicatorSignal(
                indicator_name="WilliamsR",
                signal_type=SignalType.BEARISH,
                strength=0.7,
                value=wr,
                threshold=overbought,
                message=f"Williams %R crossed below overbought ({wr:.1f})",
            )

        return None


class ROC(BaseIndicator):
    """
    Rate of Change (ROC)

    Measures the percentage change in price over a period.
    Positive values indicate upward momentum, negative values indicate downward.
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="ROC",
            category=IndicatorCategory.MOMENTUM,
            description="Rate of Change - percentage price change over period",
            default_params={"period": 12},
            param_ranges={"period": (1, 100)},
            outputs=["ROC"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["period"]

        df["ROC"] = ((df["Close"] - df["Close"].shift(period)) / df["Close"].shift(period)) * 100

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "ROC" not in df.columns:
            df = self.compute(df)

        roc = df["ROC"].iloc[-1]
        prev_roc = df["ROC"].iloc[-2] if len(df) > 1 else roc

        # Zero-line crossover signals
        if prev_roc <= 0 < roc:
            return IndicatorSignal(
                indicator_name="ROC",
                signal_type=SignalType.BULLISH,
                strength=min(1.0, abs(roc) / 5),
                value=roc,
                threshold=0,
                message=f"ROC crossed above zero ({roc:.2f}%)",
            )
        elif prev_roc >= 0 > roc:
            return IndicatorSignal(
                indicator_name="ROC",
                signal_type=SignalType.BEARISH,
                strength=min(1.0, abs(roc) / 5),
                value=roc,
                threshold=0,
                message=f"ROC crossed below zero ({roc:.2f}%)",
            )

        return None


class MFI(BaseIndicator):
    """
    Money Flow Index (MFI)

    Volume-weighted RSI. Uses both price and volume to measure buying/selling pressure.
    - MFI < 20: Oversold
    - MFI > 80: Overbought
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="MFI",
            category=IndicatorCategory.MOMENTUM,
            description="Money Flow Index - volume-weighted RSI",
            default_params={"period": 14},
            param_ranges={"period": (2, 50)},
            outputs=["MFI"],
            dependencies=["Volume"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["period"]

        # Typical Price
        tp = (df["High"] + df["Low"] + df["Close"]) / 3

        # Raw Money Flow
        raw_mf = tp * df["Volume"]

        # Money Flow Direction
        tp_change = tp.diff()
        positive_mf = raw_mf.where(tp_change > 0, 0)
        negative_mf = raw_mf.where(tp_change < 0, 0)

        # Sum over period
        positive_sum = positive_mf.rolling(window=period).sum()
        negative_sum = negative_mf.rolling(window=period).sum()

        # Money Flow Ratio and MFI
        mf_ratio = positive_sum / negative_sum.replace(0, np.nan)
        df["MFI"] = 100 - (100 / (1 + mf_ratio))

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "MFI" not in df.columns:
            df = self.compute(df)

        mfi = df["MFI"].iloc[-1]
        prev_mfi = df["MFI"].iloc[-2] if len(df) > 1 else mfi

        oversold = self.params.get("oversold", 20)
        overbought = self.params.get("overbought", 80)

        if prev_mfi <= oversold < mfi:
            return IndicatorSignal(
                indicator_name="MFI",
                signal_type=SignalType.BULLISH,
                strength=0.8,
                value=mfi,
                threshold=oversold,
                message=f"MFI crossed above oversold ({mfi:.1f})",
            )
        elif prev_mfi >= overbought > mfi:
            return IndicatorSignal(
                indicator_name="MFI",
                signal_type=SignalType.BEARISH,
                strength=0.8,
                value=mfi,
                threshold=overbought,
                message=f"MFI crossed below overbought ({mfi:.1f})",
            )
        elif mfi < oversold:
            return IndicatorSignal(
                indicator_name="MFI",
                signal_type=SignalType.BULLISH,
                strength=0.5,
                value=mfi,
                threshold=oversold,
                message=f"MFI in oversold territory ({mfi:.1f})",
            )
        elif mfi > overbought:
            return IndicatorSignal(
                indicator_name="MFI",
                signal_type=SignalType.BEARISH,
                strength=0.5,
                value=mfi,
                threshold=overbought,
                message=f"MFI in overbought territory ({mfi:.1f})",
            )

        return None


class CCI(BaseIndicator):
    """
    Commodity Channel Index (CCI)

    Measures price deviation from statistical mean.
    - CCI > 100: Overbought
    - CCI < -100: Oversold
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="CCI",
            category=IndicatorCategory.MOMENTUM,
            description="Commodity Channel Index - price deviation from mean",
            default_params={"period": 20},
            param_ranges={"period": (5, 50)},
            outputs=["CCI"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["period"]

        # Typical Price
        tp = (df["High"] + df["Low"] + df["Close"]) / 3

        # Moving Average of TP
        tp_sma = tp.rolling(window=period).mean()

        # Mean Absolute Deviation
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())

        # CCI
        df["CCI"] = (tp - tp_sma) / (0.015 * mad)

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "CCI" not in df.columns:
            df = self.compute(df)

        cci = df["CCI"].iloc[-1]
        prev_cci = df["CCI"].iloc[-2] if len(df) > 1 else cci

        oversold = self.params.get("oversold", -100)
        overbought = self.params.get("overbought", 100)

        if prev_cci <= oversold < cci:
            return IndicatorSignal(
                indicator_name="CCI",
                signal_type=SignalType.BULLISH,
                strength=0.7,
                value=cci,
                threshold=oversold,
                message=f"CCI crossed above oversold ({cci:.1f})",
            )
        elif prev_cci >= overbought > cci:
            return IndicatorSignal(
                indicator_name="CCI",
                signal_type=SignalType.BEARISH,
                strength=0.7,
                value=cci,
                threshold=overbought,
                message=f"CCI crossed below overbought ({cci:.1f})",
            )

        return None


# List of all indicator classes in this module (for auto-registration)
INDICATORS = [RSI, Stochastic, WilliamsR, ROC, MFI, CCI]
