"""
Volume Indicators - Volume-based technical indicators.

Includes:
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)
- VolumeRatio (Volume relative to average)
- CMF (Chaikin Money Flow)
- ADLine (Accumulation/Distribution Line)
- VWMA (Volume Weighted Moving Average)
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


class OBV(BaseIndicator):
    """
    On-Balance Volume (OBV)

    Cumulative volume indicator that adds volume on up days and
    subtracts on down days. Measures buying/selling pressure.
    - Rising OBV with rising price: Strong uptrend
    - Falling OBV with falling price: Strong downtrend
    - Divergence: Potential trend reversal
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="OBV",
            category=IndicatorCategory.VOLUME,
            description="On-Balance Volume - cumulative volume measure",
            default_params={},
            param_ranges={},
            outputs=["OBV"],
            dependencies=["Volume"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Calculate price direction
        price_change = df["Close"].diff()

        # OBV: add volume on up days, subtract on down days
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df["Volume"].iloc[0]

        for i in range(1, len(df)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + df["Volume"].iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - df["Volume"].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        df["OBV"] = obv

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "OBV" not in df.columns:
            df = self.compute(df)

        # Calculate OBV trend (5-period SMA)
        obv_sma = df["OBV"].rolling(window=5).mean()

        obv = df["OBV"].iloc[-1]
        obv_trend = obv_sma.iloc[-1]
        prev_obv = df["OBV"].iloc[-2] if len(df) > 1 else obv
        prev_trend = obv_sma.iloc[-2] if len(df) > 1 else obv_trend

        price = df["Close"].iloc[-1]
        prev_price = df["Close"].iloc[-2] if len(df) > 1 else price

        # OBV trending up with price
        if obv > prev_obv and price > prev_price:
            return IndicatorSignal(
                indicator_name="OBV",
                signal_type=SignalType.BULLISH,
                strength=0.6,
                value=obv,
                message="OBV confirms uptrend",
            )
        # OBV trending down with price
        elif obv < prev_obv and price < prev_price:
            return IndicatorSignal(
                indicator_name="OBV",
                signal_type=SignalType.BEARISH,
                strength=0.6,
                value=obv,
                message="OBV confirms downtrend",
            )
        # Bullish divergence: price down but OBV up
        elif obv > prev_obv and price < prev_price:
            return IndicatorSignal(
                indicator_name="OBV",
                signal_type=SignalType.BULLISH,
                strength=0.7,
                value=obv,
                message="Bullish OBV divergence",
            )
        # Bearish divergence: price up but OBV down
        elif obv < prev_obv and price > prev_price:
            return IndicatorSignal(
                indicator_name="OBV",
                signal_type=SignalType.BEARISH,
                strength=0.7,
                value=obv,
                message="Bearish OBV divergence",
            )

        return None


class VWAP(BaseIndicator):
    """
    Volume Weighted Average Price (VWAP)

    Average price weighted by volume, typically reset daily.
    - Price above VWAP: Bullish intraday
    - Price below VWAP: Bearish intraday
    Used as dynamic support/resistance.
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="VWAP",
            category=IndicatorCategory.VOLUME,
            description="Volume Weighted Average Price",
            default_params={"anchor": "D"},  # D=Daily, W=Weekly
            param_ranges={},
            outputs=["VWAP"],
            dependencies=["Volume"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Typical price
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3

        # Cumulative TPV and Volume
        tpv = (typical_price * df["Volume"]).cumsum()
        cumulative_volume = df["Volume"].cumsum()

        df["VWAP"] = tpv / cumulative_volume.replace(0, np.nan)

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "VWAP" not in df.columns:
            df = self.compute(df)

        price = df["Close"].iloc[-1]
        vwap = df["VWAP"].iloc[-1]
        prev_price = df["Close"].iloc[-2] if len(df) > 1 else price
        prev_vwap = df["VWAP"].iloc[-2] if len(df) > 1 else vwap

        # Price crossing VWAP
        if prev_price <= prev_vwap and price > vwap:
            return IndicatorSignal(
                indicator_name="VWAP",
                signal_type=SignalType.BULLISH,
                strength=0.6,
                value=vwap,
                threshold=vwap,
                message="Price crossed above VWAP",
            )
        elif prev_price >= prev_vwap and price < vwap:
            return IndicatorSignal(
                indicator_name="VWAP",
                signal_type=SignalType.BEARISH,
                strength=0.6,
                value=vwap,
                threshold=vwap,
                message="Price crossed below VWAP",
            )

        return None


class VolumeRatio(BaseIndicator):
    """
    Volume Ratio

    Current volume relative to average volume.
    - Ratio > 2: High volume (conviction)
    - Ratio > 1.5: Above average volume
    - Ratio < 0.5: Low volume (weak move)
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="VolumeRatio",
            category=IndicatorCategory.VOLUME,
            description="Volume relative to moving average",
            default_params={"period": 20},
            param_ranges={"period": (5, 50)},
            outputs=["VOLUME_RATIO", "VOLUME_SMA"],
            dependencies=["Volume"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["period"]

        df["VOLUME_SMA"] = df["Volume"].rolling(window=period).mean()
        df["VOLUME_RATIO"] = df["Volume"] / df["VOLUME_SMA"].replace(0, np.nan)

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "VOLUME_RATIO" not in df.columns:
            df = self.compute(df)

        ratio = df["VOLUME_RATIO"].iloc[-1]
        price_change = df["Close"].iloc[-1] - df["Close"].iloc[-2] if len(df) > 1 else 0

        high_volume = self.params.get("high_volume", 2.0)
        above_avg = self.params.get("above_average", 1.5)

        if ratio >= high_volume:
            signal_type = SignalType.BULLISH if price_change > 0 else SignalType.BEARISH
            return IndicatorSignal(
                indicator_name="VolumeRatio",
                signal_type=signal_type,
                strength=min(1.0, ratio / 3),
                value=ratio,
                threshold=high_volume,
                message=f"High volume confirmation ({ratio:.1f}x average)",
            )
        elif ratio >= above_avg:
            signal_type = SignalType.BULLISH if price_change > 0 else SignalType.BEARISH
            return IndicatorSignal(
                indicator_name="VolumeRatio",
                signal_type=signal_type,
                strength=0.5,
                value=ratio,
                threshold=above_avg,
                message=f"Above average volume ({ratio:.1f}x)",
            )

        return None


class CMF(BaseIndicator):
    """
    Chaikin Money Flow (CMF)

    Measures buying/selling pressure over a period.
    - CMF > 0: Buying pressure (bullish)
    - CMF < 0: Selling pressure (bearish)
    - CMF > 0.25: Strong buying
    - CMF < -0.25: Strong selling
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="CMF",
            category=IndicatorCategory.VOLUME,
            description="Chaikin Money Flow - buying/selling pressure",
            default_params={"period": 20},
            param_ranges={"period": (5, 50)},
            outputs=["CMF"],
            dependencies=["Volume"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["period"]

        # Money Flow Multiplier
        high_low = df["High"] - df["Low"]
        mf_mult = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / high_low.replace(0, np.nan)

        # Money Flow Volume
        mf_volume = mf_mult * df["Volume"]

        # CMF
        df["CMF"] = mf_volume.rolling(window=period).sum() / df["Volume"].rolling(window=period).sum()

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "CMF" not in df.columns:
            df = self.compute(df)

        cmf = df["CMF"].iloc[-1]
        prev_cmf = df["CMF"].iloc[-2] if len(df) > 1 else cmf

        strong_buy = self.params.get("strong_buy", 0.25)
        strong_sell = self.params.get("strong_sell", -0.25)

        # Zero-line crossover
        if prev_cmf <= 0 < cmf:
            return IndicatorSignal(
                indicator_name="CMF",
                signal_type=SignalType.BULLISH,
                strength=0.6 + min(0.4, cmf),
                value=cmf,
                threshold=0,
                message="CMF crossed above zero (buying pressure)",
            )
        elif prev_cmf >= 0 > cmf:
            return IndicatorSignal(
                indicator_name="CMF",
                signal_type=SignalType.BEARISH,
                strength=0.6 + min(0.4, abs(cmf)),
                value=cmf,
                threshold=0,
                message="CMF crossed below zero (selling pressure)",
            )
        # Strong readings
        elif cmf > strong_buy:
            return IndicatorSignal(
                indicator_name="CMF",
                signal_type=SignalType.BULLISH,
                strength=0.8,
                value=cmf,
                threshold=strong_buy,
                message=f"Strong CMF buying pressure ({cmf:.2f})",
            )
        elif cmf < strong_sell:
            return IndicatorSignal(
                indicator_name="CMF",
                signal_type=SignalType.BEARISH,
                strength=0.8,
                value=cmf,
                threshold=strong_sell,
                message=f"Strong CMF selling pressure ({cmf:.2f})",
            )

        return None


class ADLine(BaseIndicator):
    """
    Accumulation/Distribution Line (A/D Line)

    Similar to OBV but uses the close location value instead of
    price direction. Measures money flow into/out of a security.
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="ADLine",
            category=IndicatorCategory.VOLUME,
            description="Accumulation/Distribution Line",
            default_params={},
            param_ranges={},
            outputs=["AD_LINE"],
            dependencies=["Volume"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Money Flow Multiplier
        high_low = df["High"] - df["Low"]
        mf_mult = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / high_low.replace(0, np.nan)

        # Money Flow Volume
        mf_volume = mf_mult * df["Volume"]

        # A/D Line (cumulative)
        df["AD_LINE"] = mf_volume.cumsum()

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "AD_LINE" not in df.columns:
            df = self.compute(df)

        ad = df["AD_LINE"].iloc[-1]
        prev_ad = df["AD_LINE"].iloc[-2] if len(df) > 1 else ad
        ad_change = ad - prev_ad

        price = df["Close"].iloc[-1]
        prev_price = df["Close"].iloc[-2] if len(df) > 1 else price
        price_change = price - prev_price

        # Confirmation
        if ad_change > 0 and price_change > 0:
            return IndicatorSignal(
                indicator_name="ADLine",
                signal_type=SignalType.BULLISH,
                strength=0.6,
                value=ad,
                message="A/D Line confirms uptrend",
            )
        elif ad_change < 0 and price_change < 0:
            return IndicatorSignal(
                indicator_name="ADLine",
                signal_type=SignalType.BEARISH,
                strength=0.6,
                value=ad,
                message="A/D Line confirms downtrend",
            )
        # Divergence
        elif ad_change > 0 and price_change < 0:
            return IndicatorSignal(
                indicator_name="ADLine",
                signal_type=SignalType.BULLISH,
                strength=0.7,
                value=ad,
                message="Bullish A/D divergence (accumulation)",
            )
        elif ad_change < 0 and price_change > 0:
            return IndicatorSignal(
                indicator_name="ADLine",
                signal_type=SignalType.BEARISH,
                strength=0.7,
                value=ad,
                message="Bearish A/D divergence (distribution)",
            )

        return None


class VWMA(BaseIndicator):
    """
    Volume Weighted Moving Average (VWMA)

    Moving average weighted by volume.
    Similar to EMA but gives more weight to high-volume periods.
    """

    @classmethod
    def get_metadata(cls) -> IndicatorMetadata:
        return IndicatorMetadata(
            name="VWMA",
            category=IndicatorCategory.VOLUME,
            description="Volume Weighted Moving Average",
            default_params={"period": 20},
            param_ranges={"period": (5, 200)},
            outputs=["VWMA"],
            dependencies=["Volume"],
        )

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        period = self.params["period"]

        # VWMA = sum(price * volume) / sum(volume)
        pv = df["Close"] * df["Volume"]
        df["VWMA"] = pv.rolling(window=period).sum() / df["Volume"].rolling(window=period).sum()

        return df

    def get_signal(self, df: pd.DataFrame) -> Optional[IndicatorSignal]:
        if "VWMA" not in df.columns:
            df = self.compute(df)

        price = df["Close"].iloc[-1]
        vwma = df["VWMA"].iloc[-1]
        prev_price = df["Close"].iloc[-2] if len(df) > 1 else price
        prev_vwma = df["VWMA"].iloc[-2] if len(df) > 1 else vwma

        # Price crossing VWMA
        if prev_price <= prev_vwma and price > vwma:
            return IndicatorSignal(
                indicator_name="VWMA",
                signal_type=SignalType.BULLISH,
                strength=0.6,
                value=vwma,
                threshold=vwma,
                message=f"Price crossed above VWMA({self.params['period']})",
            )
        elif prev_price >= prev_vwma and price < vwma:
            return IndicatorSignal(
                indicator_name="VWMA",
                signal_type=SignalType.BEARISH,
                strength=0.6,
                value=vwma,
                threshold=vwma,
                message=f"Price crossed below VWMA({self.params['period']})",
            )

        return None


# List of all indicator classes in this module (for auto-registration)
INDICATORS = [OBV, VWAP, VolumeRatio, CMF, ADLine, VWMA]
