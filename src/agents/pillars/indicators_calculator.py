"""
Indicators Calculator - Computes 18 raw technical indicators from OHLCV data.

All indicators return RAW numeric values (no scoring).
Categories: Trend(5), Momentum(5), Volume(4), Structure(2), Divergences(2)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class RawIndicators:
    """Container for all 18 raw indicator values"""
    # Trend
    ema_alignment: Optional[float] = None       # % ratio of price alignment with EMAs
    macd_histogram: Optional[float] = None      # MACD histogram normalized by price
    adx_value: Optional[float] = None           # ADX value (0-100)
    adx_direction: Optional[float] = None       # +DI - -DI (positive = bullish)
    vwap_distance: Optional[float] = None       # % distance from VWAP
    ichimoku_cloud: Optional[float] = None      # % distance from cloud

    # Momentum
    rsi: Optional[float] = None                 # RSI 14 (0-100)
    stochastic_k: Optional[float] = None        # Stochastic %K (0-100)
    stochastic_d: Optional[float] = None        # Stochastic %D (0-100)
    cci: Optional[float] = None                 # CCI 20 (unbounded, typically -200 to +200)
    williams_r: Optional[float] = None          # Williams %R (-100 to 0)
    roc: Optional[float] = None                 # Rate of Change 10 (%)

    # Volume
    volume_ratio: Optional[float] = None        # Volume / SMA20(Volume)
    obv_slope: Optional[float] = None           # OBV slope normalized (10-day)
    mfi: Optional[float] = None                 # Money Flow Index (0-100)
    cmf: Optional[float] = None                 # Chaikin Money Flow (-1 to +1)

    # Structure
    support_distance: Optional[float] = None    # % distance to nearest support (negative = below)
    resistance_distance: Optional[float] = None # % distance to nearest resistance (positive = above)
    bollinger_pct_b: Optional[float] = None     # Bollinger %B (0-1, can exceed)

    # Divergences
    rsi_divergence: Optional[float] = None      # +1 bullish, -1 bearish, 0 none
    macd_divergence: Optional[float] = None     # +1 bullish, -1 bearish, 0 none

    # Metadata
    atr_value: Optional[float] = None           # For context/normalization
    atr_percentile: Optional[float] = None      # ATR percentile over available data

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def available_count(self) -> int:
        return sum(1 for k, v in self.__dict__.items()
                   if v is not None and not k.startswith('atr_'))


class IndicatorsCalculator:
    """Calculates all 18 raw indicators from OHLCV DataFrame."""

    def calculate_all(self, df: pd.DataFrame) -> RawIndicators:
        """Calculate all indicators. Returns whatever is possible given data length."""
        result = RawIndicators()

        if df is None or len(df) < 20:
            logger.warning("Insufficient data for indicators calculation")
            return result

        # Ensure columns
        close = df['Close'].values.flatten() if hasattr(df['Close'].values, 'flatten') else df['Close'].values
        high = df['High'].values.flatten() if hasattr(df['High'].values, 'flatten') else df['High'].values
        low = df['Low'].values.flatten() if hasattr(df['Low'].values, 'flatten') else df['Low'].values
        volume = df['Volume'].values.flatten() if hasattr(df['Volume'].values, 'flatten') else df['Volume'].values

        close = close.astype(float)
        high = high.astype(float)
        low = low.astype(float)
        volume = volume.astype(float)

        # ATR for context
        self._calc_atr(result, high, low, close)

        # Trend
        self._calc_ema_alignment(result, close)
        self._calc_macd(result, close)
        self._calc_adx(result, high, low, close)
        self._calc_vwap(result, high, low, close, volume)
        self._calc_ichimoku(result, high, low, close)

        # Momentum
        self._calc_rsi(result, close)
        self._calc_stochastic(result, high, low, close)
        self._calc_cci(result, high, low, close)
        self._calc_williams_r(result, high, low, close)
        self._calc_roc(result, close)

        # Volume
        self._calc_volume_ratio(result, volume)
        self._calc_obv_slope(result, close, volume)
        self._calc_mfi(result, high, low, close, volume)
        self._calc_cmf(result, high, low, close, volume)

        # Structure
        self._calc_support_resistance(result, high, low, close)
        self._calc_bollinger(result, close)

        # Divergences
        self._calc_rsi_divergence(result, close)
        self._calc_macd_divergence(result, close)

        logger.debug(f"Calculated {result.available_count()}/18 indicators")
        return result

    # ─── Helper: EMA ─────────────────────────────────────────────
    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA as numpy array."""
        alpha = 2 / (period + 1)
        out = np.empty_like(data, dtype=float)
        out[0] = data[0]
        for i in range(1, len(data)):
            out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
        return out

    @staticmethod
    def _sma(data: np.ndarray, period: int) -> np.ndarray:
        """Simple moving average (NaN-padded)."""
        out = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            out[i] = np.mean(data[i - period + 1: i + 1])
        return out

    # ─── ATR (context) ───────────────────────────────────────────
    def _calc_atr(self, r: RawIndicators, high, low, close, period=14):
        try:
            if len(close) < period + 1:
                return
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(np.abs(high[1:] - close[:-1]),
                                       np.abs(low[1:] - close[:-1])))
            atr_arr = self._ema(tr, period)
            r.atr_value = float(atr_arr[-1])
            # Percentile over available data
            if len(atr_arr) > 20:
                r.atr_percentile = float(np.searchsorted(np.sort(atr_arr), atr_arr[-1]) / len(atr_arr))
            else:
                r.atr_percentile = 0.5
        except Exception as e:
            logger.debug(f"ATR calc failed: {e}")

    # ─── TREND ───────────────────────────────────────────────────
    def _calc_ema_alignment(self, r: RawIndicators, close):
        try:
            if len(close) < 200:
                # Use what we have
                ema20 = self._ema(close, 20)[-1]
                ema50 = self._ema(close, 50)[-1] if len(close) >= 50 else ema20
                price = close[-1]
                # Alignment = avg distance above/below EMAs
                d20 = (price - ema20) / ema20 * 100
                d50 = (price - ema50) / ema50 * 100
                r.ema_alignment = (d20 + d50) / 2
            else:
                ema20 = self._ema(close, 20)[-1]
                ema50 = self._ema(close, 50)[-1]
                ema200 = self._ema(close, 200)[-1]
                price = close[-1]
                d20 = (price - ema20) / ema20 * 100
                d50 = (price - ema50) / ema50 * 100
                d200 = (price - ema200) / ema200 * 100
                r.ema_alignment = (d20 + d50 + d200) / 3
        except Exception as e:
            logger.debug(f"EMA alignment failed: {e}")

    def _calc_macd(self, r: RawIndicators, close):
        try:
            if len(close) < 35:
                return
            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            macd_line = ema12 - ema26
            signal = self._ema(macd_line, 9)
            histogram = macd_line - signal
            # Normalize by price
            r.macd_histogram = float(histogram[-1] / close[-1] * 100)
        except Exception as e:
            logger.debug(f"MACD calc failed: {e}")

    def _calc_adx(self, r: RawIndicators, high, low, close, period=14):
        try:
            if len(close) < period * 2:
                return
            n = len(close)
            plus_dm = np.zeros(n)
            minus_dm = np.zeros(n)
            tr = np.zeros(n)

            for i in range(1, n):
                up = high[i] - high[i - 1]
                down = low[i - 1] - low[i]
                plus_dm[i] = up if (up > down and up > 0) else 0
                minus_dm[i] = down if (down > up and down > 0) else 0
                tr[i] = max(high[i] - low[i],
                            abs(high[i] - close[i - 1]),
                            abs(low[i] - close[i - 1]))

            atr = self._ema(tr[1:], period)
            plus_di = 100 * self._ema(plus_dm[1:], period) / np.where(atr > 0, atr, 1)
            minus_di = 100 * self._ema(minus_dm[1:], period) / np.where(atr > 0, atr, 1)

            dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, plus_di + minus_di, 1)
            adx = self._ema(dx, period)

            r.adx_value = float(adx[-1])
            r.adx_direction = float(plus_di[-1] - minus_di[-1])
        except Exception as e:
            logger.debug(f"ADX calc failed: {e}")

    def _calc_vwap(self, r: RawIndicators, high, low, close, volume):
        try:
            # Approximate VWAP using cumulative typical price * volume / cumulative volume
            # Use last 20 days for rolling VWAP
            n = min(20, len(close))
            tp = (high[-n:] + low[-n:] + close[-n:]) / 3
            vol = volume[-n:]
            cum_tp_vol = np.cumsum(tp * vol)
            cum_vol = np.cumsum(vol)
            vwap = cum_tp_vol[-1] / cum_vol[-1] if cum_vol[-1] > 0 else close[-1]
            r.vwap_distance = float((close[-1] - vwap) / vwap * 100)
        except Exception as e:
            logger.debug(f"VWAP calc failed: {e}")

    def _calc_ichimoku(self, r: RawIndicators, high, low, close):
        try:
            if len(close) < 52:
                return
            # Tenkan-sen (9), Kijun-sen (26), Senkou Span A, Senkou Span B (52)
            tenkan = (np.max(high[-9:]) + np.min(low[-9:])) / 2
            kijun = (np.max(high[-26:]) + np.min(low[-26:])) / 2
            senkou_a = (tenkan + kijun) / 2
            senkou_b = (np.max(high[-52:]) + np.min(low[-52:])) / 2
            cloud_top = max(senkou_a, senkou_b)
            cloud_bottom = min(senkou_a, senkou_b)
            price = close[-1]

            if price > cloud_top:
                r.ichimoku_cloud = float((price - cloud_top) / cloud_top * 100)
            elif price < cloud_bottom:
                r.ichimoku_cloud = float((price - cloud_bottom) / cloud_bottom * 100)
            else:
                # Inside cloud - return relative position (-1 to +1 range mapped to small %)
                cloud_width = cloud_top - cloud_bottom
                if cloud_width > 0:
                    r.ichimoku_cloud = float(((price - cloud_bottom) / cloud_width - 0.5) * 0.5)
                else:
                    r.ichimoku_cloud = 0.0
        except Exception as e:
            logger.debug(f"Ichimoku calc failed: {e}")

    # ─── MOMENTUM ────────────────────────────────────────────────
    def _calc_rsi(self, r: RawIndicators, close, period=14):
        try:
            if len(close) < period + 1:
                return
            deltas = np.diff(close)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = self._ema(gains, period)[-1]
            avg_loss = self._ema(losses, period)[-1]
            if avg_loss == 0:
                r.rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                r.rsi = float(100 - (100 / (1 + rs)))
        except Exception as e:
            logger.debug(f"RSI calc failed: {e}")

    def _calc_stochastic(self, r: RawIndicators, high, low, close, k_period=14, d_period=3):
        try:
            if len(close) < k_period + d_period:
                return
            # %K
            k_values = np.zeros(len(close))
            for i in range(k_period - 1, len(close)):
                h = np.max(high[i - k_period + 1:i + 1])
                l = np.min(low[i - k_period + 1:i + 1])
                if h - l > 0:
                    k_values[i] = (close[i] - l) / (h - l) * 100
                else:
                    k_values[i] = 50
            r.stochastic_k = float(k_values[-1])
            # %D = SMA of %K
            r.stochastic_d = float(np.mean(k_values[-d_period:]))
        except Exception as e:
            logger.debug(f"Stochastic calc failed: {e}")

    def _calc_cci(self, r: RawIndicators, high, low, close, period=20):
        try:
            if len(close) < period:
                return
            tp = (high + low + close) / 3
            tp_sma = self._sma(tp, period)
            # Mean deviation
            md = np.zeros_like(tp)
            for i in range(period - 1, len(tp)):
                md[i] = np.mean(np.abs(tp[i - period + 1:i + 1] - tp_sma[i]))
            if md[-1] > 0:
                r.cci = float((tp[-1] - tp_sma[-1]) / (0.015 * md[-1]))
            else:
                r.cci = 0.0
        except Exception as e:
            logger.debug(f"CCI calc failed: {e}")

    def _calc_williams_r(self, r: RawIndicators, high, low, close, period=14):
        try:
            if len(close) < period:
                return
            hh = np.max(high[-period:])
            ll = np.min(low[-period:])
            if hh - ll > 0:
                r.williams_r = float((hh - close[-1]) / (hh - ll) * -100)
            else:
                r.williams_r = -50.0
        except Exception as e:
            logger.debug(f"Williams %R calc failed: {e}")

    def _calc_roc(self, r: RawIndicators, close, period=10):
        try:
            if len(close) < period + 1:
                return
            r.roc = float((close[-1] - close[-period - 1]) / close[-period - 1] * 100)
        except Exception as e:
            logger.debug(f"ROC calc failed: {e}")

    # ─── VOLUME ──────────────────────────────────────────────────
    def _calc_volume_ratio(self, r: RawIndicators, volume, period=20):
        try:
            if len(volume) < period:
                return
            sma_vol = np.mean(volume[-period:])
            r.volume_ratio = float(volume[-1] / sma_vol) if sma_vol > 0 else 1.0
        except Exception as e:
            logger.debug(f"Volume ratio calc failed: {e}")

    def _calc_obv_slope(self, r: RawIndicators, close, volume, period=10):
        try:
            if len(close) < period + 1:
                return
            obv = np.zeros(len(close))
            for i in range(1, len(close)):
                if close[i] > close[i - 1]:
                    obv[i] = obv[i - 1] + volume[i]
                elif close[i] < close[i - 1]:
                    obv[i] = obv[i - 1] - volume[i]
                else:
                    obv[i] = obv[i - 1]
            # Slope over last `period` days, normalized
            obv_recent = obv[-period:]
            x = np.arange(period)
            if np.std(obv_recent) > 0:
                slope = np.polyfit(x, obv_recent, 1)[0]
                # Normalize by mean absolute OBV
                mean_abs = np.mean(np.abs(obv_recent)) if np.mean(np.abs(obv_recent)) > 0 else 1
                r.obv_slope = float(slope / mean_abs * 100)
            else:
                r.obv_slope = 0.0
        except Exception as e:
            logger.debug(f"OBV slope calc failed: {e}")

    def _calc_mfi(self, r: RawIndicators, high, low, close, volume, period=14):
        try:
            if len(close) < period + 1:
                return
            tp = (high + low + close) / 3
            raw_mf = tp * volume
            pos_mf = np.zeros(len(close))
            neg_mf = np.zeros(len(close))
            for i in range(1, len(close)):
                if tp[i] > tp[i - 1]:
                    pos_mf[i] = raw_mf[i]
                elif tp[i] < tp[i - 1]:
                    neg_mf[i] = raw_mf[i]
            sum_pos = np.sum(pos_mf[-period:])
            sum_neg = np.sum(neg_mf[-period:])
            if sum_neg > 0:
                mfr = sum_pos / sum_neg
                r.mfi = float(100 - (100 / (1 + mfr)))
            else:
                r.mfi = 100.0
        except Exception as e:
            logger.debug(f"MFI calc failed: {e}")

    def _calc_cmf(self, r: RawIndicators, high, low, close, volume, period=20):
        try:
            if len(close) < period:
                return
            # CLV = ((close - low) - (high - close)) / (high - low)
            hl = high - low
            clv = np.where(hl > 0, ((close - low) - (high - close)) / hl, 0)
            mfv = clv * volume
            sum_mfv = np.sum(mfv[-period:])
            sum_vol = np.sum(volume[-period:])
            r.cmf = float(sum_mfv / sum_vol) if sum_vol > 0 else 0.0
        except Exception as e:
            logger.debug(f"CMF calc failed: {e}")

    # ─── STRUCTURE ───────────────────────────────────────────────
    def _calc_support_resistance(self, r: RawIndicators, high, low, close, lookback=60):
        try:
            n = min(lookback, len(close))
            if n < 20:
                return
            h = high[-n:]
            l = low[-n:]
            price = close[-1]

            # Find swing highs and lows (local extremes over 5-bar window)
            swing_highs = []
            swing_lows = []
            for i in range(2, n - 2):
                if h[i] >= h[i - 1] and h[i] >= h[i - 2] and h[i] >= h[i + 1] and h[i] >= h[i + 2]:
                    swing_highs.append(h[i])
                if l[i] <= l[i - 1] and l[i] <= l[i - 2] and l[i] <= l[i + 1] and l[i] <= l[i + 2]:
                    swing_lows.append(l[i])

            # Nearest support (below price)
            supports = [s for s in swing_lows if s < price]
            if supports:
                nearest_support = max(supports)
                r.support_distance = float((price - nearest_support) / price * 100)
            else:
                r.support_distance = 10.0  # Far from support

            # Nearest resistance (above price)
            resistances = [s for s in swing_highs if s > price]
            if resistances:
                nearest_resistance = min(resistances)
                r.resistance_distance = float((nearest_resistance - price) / price * 100)
            else:
                r.resistance_distance = 10.0  # Far from resistance
        except Exception as e:
            logger.debug(f"S/R calc failed: {e}")

    def _calc_bollinger(self, r: RawIndicators, close, period=20, std_dev=2):
        try:
            if len(close) < period:
                return
            sma = np.mean(close[-period:])
            std = np.std(close[-period:])
            upper = sma + std_dev * std
            lower = sma - std_dev * std
            if upper - lower > 0:
                r.bollinger_pct_b = float((close[-1] - lower) / (upper - lower))
            else:
                r.bollinger_pct_b = 0.5
        except Exception as e:
            logger.debug(f"Bollinger calc failed: {e}")

    # ─── DIVERGENCES ─────────────────────────────────────────────
    def _calc_rsi_divergence(self, r: RawIndicators, close, period=14, lookback=30):
        try:
            if len(close) < lookback + period:
                r.rsi_divergence = 0.0
                return

            # Calculate RSI series
            deltas = np.diff(close)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = self._ema(gains, period)
            avg_loss = self._ema(losses, period)
            rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
            rsi_series = 100 - (100 / (1 + rs))

            # Find last 2 price lows and corresponding RSI
            price_window = close[-lookback:]
            rsi_window = rsi_series[-lookback:]

            # Simple: find two lowest points in price
            mid = lookback // 2
            p1_idx = np.argmin(price_window[:mid])
            p2_idx = mid + np.argmin(price_window[mid:])

            price_low1 = price_window[p1_idx]
            price_low2 = price_window[p2_idx]
            rsi_low1 = rsi_window[p1_idx]
            rsi_low2 = rsi_window[p2_idx]

            # Bullish divergence: price lower low, RSI higher low
            if price_low2 < price_low1 and rsi_low2 > rsi_low1:
                r.rsi_divergence = 1.0
            # Bearish divergence: price higher high, RSI lower high (check highs)
            elif price_low2 > price_low1 and rsi_low2 < rsi_low1:
                r.rsi_divergence = -1.0
            else:
                r.rsi_divergence = 0.0
        except Exception as e:
            logger.debug(f"RSI divergence calc failed: {e}")
            r.rsi_divergence = 0.0

    def _calc_macd_divergence(self, r: RawIndicators, close, lookback=30):
        try:
            if len(close) < 35 + lookback:
                r.macd_divergence = 0.0
                return

            ema12 = self._ema(close, 12)
            ema26 = self._ema(close, 26)
            macd_line = ema12 - ema26
            signal = self._ema(macd_line, 9)
            histogram = macd_line - signal

            price_window = close[-lookback:]
            hist_window = histogram[-lookback:]

            mid = lookback // 2
            p1_idx = np.argmin(price_window[:mid])
            p2_idx = mid + np.argmin(price_window[mid:])

            price_low1 = price_window[p1_idx]
            price_low2 = price_window[p2_idx]
            hist_low1 = hist_window[p1_idx]
            hist_low2 = hist_window[p2_idx]

            if price_low2 < price_low1 and hist_low2 > hist_low1:
                r.macd_divergence = 1.0
            elif price_low2 > price_low1 and hist_low2 < hist_low1:
                r.macd_divergence = -1.0
            else:
                r.macd_divergence = 0.0
        except Exception as e:
            logger.debug(f"MACD divergence calc failed: {e}")
            r.macd_divergence = 0.0
