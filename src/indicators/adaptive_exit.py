"""
Adaptive Exit Indicator - Intelligent exit system combining:
- Chandelier Exit (ATR-based trailing stop)
- Bollinger Squeeze (volatility contraction detection)
- Volume exhaustion (A/D + volume ratio)

Replaces fixed take_profit_pct and trailing_stop_pct with adaptive logic.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ExitSignal:
    """Exit signal with reason and metadata"""
    should_exit: bool
    reason: str
    price: float = 0.0
    chandelier_level: float = 0.0
    volume_ratio: float = 0.0
    squeeze_active: bool = False
    ad_trend: str = ""  # 'rising', 'falling', 'neutral'


class AdaptiveExitIndicator:
    """
    Intelligent exit system combining:
    - Chandelier Exit (ATR trailing)
    - Bollinger Squeeze (contraction volatility)
    - Volume exhaustion (A/D + volume ratio)
    """

    def __init__(
        self,
        atr_period: int = 22,
        atr_multiplier: float = 3.0,
        bb_period: int = 20,
        bb_std_mult: float = 2.0,
        volume_lookback: int = 20,
        volume_exhaustion_threshold: float = 0.7,
        volume_confirmation_threshold: float = 1.2
    ):
        """
        Initialize adaptive exit indicator.

        Args:
            atr_period: Period for ATR calculation (Chandelier Exit)
            atr_multiplier: ATR multiplier for stop distance
            bb_period: Period for Bollinger Bands
            bb_std_mult: Standard deviation multiplier for BB
            volume_lookback: Lookback period for volume average
            volume_exhaustion_threshold: Volume ratio below which = exhaustion
            volume_confirmation_threshold: Volume ratio above which = confirmation
        """
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.bb_period = bb_period
        self.bb_std_mult = bb_std_mult
        self.volume_lookback = volume_lookback
        self.volume_exhaustion_threshold = volume_exhaustion_threshold
        self.volume_confirmation_threshold = volume_confirmation_threshold

    def _calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        ATR = SMA of True Range over period
        True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        """
        if period is None:
            period = self.atr_period

        high = df['High']
        low = df['Low']
        close = df['Close']
        prev_close = close.shift(1)

        # True Range components
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        # True Range = max of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = Simple Moving Average of True Range
        atr = true_range.rolling(window=period).mean()

        return atr

    def calculate_chandelier_exit(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Chandelier Exit (long positions).

        Chandelier Exit = Highest High(22) - ATR(22) * 3

        This trailing stop adapts to volatility:
        - High volatility = wider stop
        - Low volatility = tighter stop
        """
        atr = self._calculate_atr(df, self.atr_period)
        highest_high = df['High'].rolling(window=self.atr_period).max()

        chandelier_long = highest_high - (atr * self.atr_multiplier)

        return chandelier_long

    def detect_bollinger_squeeze(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect Bollinger Bands squeeze (volatility contraction).

        Squeeze = bandwidth is in bottom 25% of last 50 periods
        This indicates consolidation before potential breakout.
        """
        sma = df['Close'].rolling(window=self.bb_period).mean()
        std = df['Close'].rolling(window=self.bb_period).std()

        # Bandwidth as percentage of SMA
        bandwidth = (std * self.bb_std_mult * 2) / sma

        # Squeeze = bandwidth below 25th percentile of last 50 periods
        bandwidth_percentile = bandwidth.rolling(window=50).quantile(0.25)
        squeeze = bandwidth < bandwidth_percentile

        return squeeze

    def calculate_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line (Chaikin).

        AD = cumsum(CLV * Volume)
        CLV = ((Close - Low) - (High - Close)) / (High - Low)

        Rising AD = accumulation (bullish)
        Falling AD = distribution (bearish)
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']

        # Handle division by zero (High == Low)
        hl_range = high - low
        hl_range = hl_range.replace(0, np.nan)

        # Close Location Value
        clv = ((close - low) - (high - close)) / hl_range
        clv = clv.fillna(0)

        # A/D Line = cumulative sum of (CLV * Volume)
        ad_line = (clv * volume).cumsum()

        return ad_line

    def detect_selling_exhaustion(self, df: pd.DataFrame, idx: int) -> Tuple[bool, float, str]:
        """
        Detect selling exhaustion using volume and A/D analysis.

        Exhaustion = declining volume + rising A/D (accumulation during low volume)

        Args:
            df: OHLCV DataFrame
            idx: Current index

        Returns:
            (is_exhausted, volume_ratio, ad_trend)
        """
        if idx < self.volume_lookback + 5:
            return False, 1.0, 'neutral'

        # Volume ratio (current vs average)
        current_volume = df['Volume'].iloc[idx]
        avg_volume = df['Volume'].iloc[idx - self.volume_lookback:idx].mean()

        if avg_volume > 0:
            vol_ratio = current_volume / avg_volume
        else:
            vol_ratio = 1.0

        vol_declining = vol_ratio < self.volume_exhaustion_threshold

        # A/D Line trend
        ad_line = self.calculate_ad_line(df)
        ad_current = ad_line.iloc[idx]
        ad_previous = ad_line.iloc[idx - 5]

        if ad_current > ad_previous * 1.02:  # Rising >2%
            ad_trend = 'rising'
        elif ad_current < ad_previous * 0.98:  # Falling >2%
            ad_trend = 'falling'
        else:
            ad_trend = 'neutral'

        # Exhaustion = low volume + accumulation (rising A/D)
        is_exhausted = vol_declining and ad_trend == 'rising'

        return is_exhausted, vol_ratio, ad_trend

    def should_exit(
        self,
        df: pd.DataFrame,
        idx: int,
        position_stop_loss: float,
        entry_price: float = None
    ) -> ExitSignal:
        """
        Main exit logic. Checks in priority order:
        1. Stop-loss touched (RSI peak based) -> EXIT
        2. Chandelier Exit breached + volume confirmation -> EXIT
        3. Squeeze ended + price below EMA -> EXIT

        Args:
            df: OHLCV DataFrame with price data
            idx: Current bar index
            position_stop_loss: Stop-loss price (from RSI peak)
            entry_price: Entry price (optional, for P&L calc)

        Returns:
            ExitSignal with exit decision and metadata
        """
        if idx < max(self.atr_period, self.bb_period, self.volume_lookback) + 5:
            return ExitSignal(
                should_exit=False,
                reason='insufficient_data',
                chandelier_level=0.0
            )

        current_price = df['Close'].iloc[idx]
        current_low = df['Low'].iloc[idx]

        # Pre-calculate indicators
        chandelier = self.calculate_chandelier_exit(df)
        chandelier_level = chandelier.iloc[idx]

        squeeze = self.detect_bollinger_squeeze(df)
        squeeze_active = squeeze.iloc[idx]
        squeeze_previous = squeeze.iloc[idx - 1] if idx > 0 else False

        exhaustion, vol_ratio, ad_trend = self.detect_selling_exhaustion(df, idx)

        # ======== EXIT CHECKS ========

        # 1. STOP-LOSS (Priority 1 - Hardest exit)
        if current_low <= position_stop_loss:
            return ExitSignal(
                should_exit=True,
                reason='stop_loss',
                price=position_stop_loss,
                chandelier_level=chandelier_level,
                volume_ratio=vol_ratio,
                squeeze_active=squeeze_active,
                ad_trend=ad_trend
            )

        # 2. CHANDELIER EXIT + Volume Confirmation
        if current_price < chandelier_level:
            # Need volume confirmation (high volume = real breakdown)
            if vol_ratio > self.volume_confirmation_threshold:
                return ExitSignal(
                    should_exit=True,
                    reason='chandelier_exit',
                    price=current_price,
                    chandelier_level=chandelier_level,
                    volume_ratio=vol_ratio,
                    squeeze_active=squeeze_active,
                    ad_trend=ad_trend
                )

        # 3. SQUEEZE BREAKDOWN
        # Exit when squeeze ends AND price is below short-term EMA
        if not squeeze_active and squeeze_previous:  # Squeeze just ended
            ema_20 = df['Close'].rolling(window=20).mean()
            if current_price < ema_20.iloc[idx]:
                return ExitSignal(
                    should_exit=True,
                    reason='squeeze_breakdown',
                    price=current_price,
                    chandelier_level=chandelier_level,
                    volume_ratio=vol_ratio,
                    squeeze_active=squeeze_active,
                    ad_trend=ad_trend
                )

        # NO EXIT - Continue holding
        return ExitSignal(
            should_exit=False,
            reason='hold',
            price=current_price,
            chandelier_level=chandelier_level,
            volume_ratio=vol_ratio,
            squeeze_active=squeeze_active,
            ad_trend=ad_trend
        )

    def get_exit_levels(self, df: pd.DataFrame, idx: int) -> dict:
        """
        Get current exit levels for display/charting.

        Returns:
            dict with chandelier_level, squeeze_status, volume_ratio
        """
        if idx < max(self.atr_period, self.bb_period, self.volume_lookback):
            return {
                'chandelier_level': None,
                'squeeze_active': False,
                'volume_ratio': 1.0,
                'ad_trend': 'neutral'
            }

        chandelier = self.calculate_chandelier_exit(df)
        squeeze = self.detect_bollinger_squeeze(df)
        _, vol_ratio, ad_trend = self.detect_selling_exhaustion(df, idx)

        return {
            'chandelier_level': chandelier.iloc[idx],
            'squeeze_active': squeeze.iloc[idx],
            'volume_ratio': vol_ratio,
            'ad_trend': ad_trend
        }


# Convenience function for backtester integration
def check_adaptive_exit(
    df: pd.DataFrame,
    idx: int,
    stop_loss: float,
    atr_period: int = 22,
    atr_multiplier: float = 3.0
) -> Tuple[bool, str]:
    """
    Convenience wrapper for backtester integration.

    Args:
        df: OHLCV DataFrame
        idx: Current index
        stop_loss: Position stop-loss price
        atr_period: ATR period for Chandelier Exit
        atr_multiplier: ATR multiplier

    Returns:
        (should_exit, exit_reason)
    """
    indicator = AdaptiveExitIndicator(
        atr_period=atr_period,
        atr_multiplier=atr_multiplier
    )

    signal = indicator.should_exit(df, idx, stop_loss)

    return signal.should_exit, signal.reason
