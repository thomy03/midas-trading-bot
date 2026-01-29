"""
Indicator Discovery - Auto-découverte des indicateurs qui fonctionnent

Ce module ne fait AUCUNE hypothèse sur ce qui marche.
Il teste 30+ indicateurs techniques et laisse les DONNÉES décider
lesquels sont prédictifs du succès.

Architecture:
    1. Calcule TOUS les indicateurs pour chaque signal
    2. Stocke lesquels étaient "ON" au moment du signal
    3. Après N trades, analyse les corrélations
    4. Ajuste automatiquement les poids des indicateurs

Auteur: TradingBot V5
Date: Janvier 2026
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IndicatorState:
    """État d'un indicateur au moment d'un signal"""
    name: str
    value: float  # Valeur numérique (ex: RSI = 45.2)
    is_bullish: bool  # Signal haussier?
    is_bearish: bool  # Signal baissier?
    strength: float  # Force du signal 0-1


@dataclass
class SignalIndicators:
    """Tous les indicateurs au moment d'un signal"""
    symbol: str
    signal_date: str
    entry_price: float
    indicators: Dict[str, IndicatorState] = field(default_factory=dict)

    # Outcome (rempli plus tard)
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    is_winner: Optional[bool] = None

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'signal_date': self.signal_date,
            'entry_price': self.entry_price,
            'indicators': {k: asdict(v) for k, v in self.indicators.items()},
            'exit_price': self.exit_price,
            'pnl_pct': self.pnl_pct,
            'is_winner': self.is_winner
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SignalIndicators':
        indicators = {}
        for k, v in data.get('indicators', {}).items():
            indicators[k] = IndicatorState(**v)
        return cls(
            symbol=data['symbol'],
            signal_date=data['signal_date'],
            entry_price=data['entry_price'],
            indicators=indicators,
            exit_price=data.get('exit_price'),
            pnl_pct=data.get('pnl_pct'),
            is_winner=data.get('is_winner')
        )


class IndicatorCalculator:
    """
    Calcule tous les indicateurs techniques.

    Chaque indicateur retourne un IndicatorState avec:
    - value: valeur numérique
    - is_bullish: True si signal d'achat
    - is_bearish: True si signal de vente
    - strength: force du signal (0-1)
    """

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> Dict[str, IndicatorState]:
        """Calcule TOUS les indicateurs disponibles"""
        if df is None or len(df) < 50:
            return {}

        results = {}
        calc = IndicatorCalculator

        # S'assurer que les colonnes existent
        df = calc._prepare_dataframe(df)

        # === MOMENTUM ===
        results['rsi'] = calc._rsi(df)
        results['rsi_oversold'] = calc._rsi_oversold(df)
        results['rsi_overbought'] = calc._rsi_overbought(df)
        results['rsi_rising'] = calc._rsi_rising(df)
        results['stochastic'] = calc._stochastic(df)
        results['stoch_oversold'] = calc._stoch_oversold(df)
        results['williams_r'] = calc._williams_r(df)
        results['cci'] = calc._cci(df)
        results['mfi'] = calc._mfi(df)
        results['roc'] = calc._rate_of_change(df)

        # === TREND ===
        results['ema_cross_20_50'] = calc._ema_cross(df, 20, 50)
        results['ema_cross_50_200'] = calc._ema_cross(df, 50, 200)
        results['ema_alignment'] = calc._ema_alignment(df)
        results['macd_cross'] = calc._macd_cross(df)
        results['macd_histogram_rising'] = calc._macd_histogram_rising(df)
        results['adx_trend'] = calc._adx_trend(df)
        results['aroon_up'] = calc._aroon(df)
        results['price_above_ema20'] = calc._price_above_ema(df, 20)
        results['price_above_ema50'] = calc._price_above_ema(df, 50)
        results['price_above_ema200'] = calc._price_above_ema(df, 200)

        # === VOLUME ===
        results['volume_spike'] = calc._volume_spike(df)
        results['volume_trend'] = calc._volume_trend(df)
        results['obv_rising'] = calc._obv_rising(df)
        results['vwap_cross'] = calc._vwap_cross(df)
        results['accumulation'] = calc._accumulation_distribution(df)

        # === VOLATILITY ===
        results['bollinger_squeeze'] = calc._bollinger_squeeze(df)
        results['bollinger_breakout'] = calc._bollinger_breakout(df)
        results['atr_low'] = calc._atr_low(df)
        results['keltner_breakout'] = calc._keltner_breakout(df)

        # === PRICE ACTION ===
        results['higher_high'] = calc._higher_high(df)
        results['higher_low'] = calc._higher_low(df)
        results['breakout_resistance'] = calc._breakout_resistance(df)
        results['support_bounce'] = calc._support_bounce(df)
        results['gap_up'] = calc._gap_up(df)

        # === CANDLESTICK PATTERNS ===
        results['bullish_engulfing'] = calc._bullish_engulfing(df)
        results['hammer'] = calc._hammer(df)
        results['morning_star'] = calc._morning_star(df)
        results['doji_reversal'] = calc._doji_reversal(df)

        return results

    @staticmethod
    def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Préparer le dataframe avec les colonnes nécessaires"""
        df = df.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume'] if 'Volume' in df.columns else pd.Series([1]*len(df))

        # EMAs
        df['EMA_20'] = close.ewm(span=20).mean()
        df['EMA_50'] = close.ewm(span=50).mean()
        df['EMA_200'] = close.ewm(span=200).mean() if len(df) >= 200 else close.ewm(span=50).mean()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.001)
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # Stochastic
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        df['Stoch_K'] = 100 * (close - low_14) / (high_14 - low_14 + 0.001)
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()

        # Bollinger Bands
        df['BB_middle'] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * bb_std
        df['BB_lower'] = df['BB_middle'] - 2 * bb_std
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

        # OBV
        obv = [0]
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv

        # ADX
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
        df['ADX'] = dx.rolling(14).mean()
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di

        # Williams %R
        df['Williams_R'] = -100 * (high_14 - close) / (high_14 - low_14 + 0.001)

        # CCI
        tp = (high + low + close) / 3
        tp_sma = tp.rolling(20).mean()
        tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['CCI'] = (tp - tp_sma) / (0.015 * tp_mad + 0.001)

        # MFI
        tp = (high + low + close) / 3
        raw_mf = tp * volume
        pos_mf = raw_mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg_mf = raw_mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        df['MFI'] = 100 - (100 / (1 + pos_mf / (neg_mf + 0.001)))

        # VWAP (approximation sur 20 jours)
        df['VWAP'] = (volume * (high + low + close) / 3).rolling(20).sum() / volume.rolling(20).sum()

        # ROC
        df['ROC'] = ((close - close.shift(10)) / close.shift(10)) * 100

        return df

    # === MOMENTUM INDICATORS ===

    @staticmethod
    def _rsi(df: pd.DataFrame) -> IndicatorState:
        rsi = df['RSI'].iloc[-1]
        return IndicatorState(
            name='rsi',
            value=rsi,
            is_bullish=40 <= rsi <= 60,  # Zone neutre mais stable
            is_bearish=rsi > 70 or rsi < 30,
            strength=abs(50 - rsi) / 50
        )

    @staticmethod
    def _rsi_oversold(df: pd.DataFrame) -> IndicatorState:
        rsi = df['RSI'].iloc[-1]
        rsi_prev = df['RSI'].iloc[-2]
        is_bullish = rsi < 35 and rsi > rsi_prev  # RSI bas mais remonte
        return IndicatorState(
            name='rsi_oversold',
            value=rsi,
            is_bullish=is_bullish,
            is_bearish=False,
            strength=(35 - rsi) / 35 if rsi < 35 else 0
        )

    @staticmethod
    def _rsi_overbought(df: pd.DataFrame) -> IndicatorState:
        rsi = df['RSI'].iloc[-1]
        return IndicatorState(
            name='rsi_overbought',
            value=rsi,
            is_bullish=False,
            is_bearish=rsi > 70,
            strength=(rsi - 70) / 30 if rsi > 70 else 0
        )

    @staticmethod
    def _rsi_rising(df: pd.DataFrame) -> IndicatorState:
        rsi_now = df['RSI'].iloc[-1]
        rsi_5d = df['RSI'].iloc[-5]
        rising = rsi_now > rsi_5d
        return IndicatorState(
            name='rsi_rising',
            value=rsi_now - rsi_5d,
            is_bullish=rising and rsi_now < 65,
            is_bearish=not rising and rsi_now > 35,
            strength=abs(rsi_now - rsi_5d) / 20
        )

    @staticmethod
    def _stochastic(df: pd.DataFrame) -> IndicatorState:
        k = df['Stoch_K'].iloc[-1]
        d = df['Stoch_D'].iloc[-1]
        cross_up = k > d and df['Stoch_K'].iloc[-2] <= df['Stoch_D'].iloc[-2]
        return IndicatorState(
            name='stochastic',
            value=k,
            is_bullish=cross_up and k < 50,
            is_bearish=k > d and df['Stoch_K'].iloc[-2] >= df['Stoch_D'].iloc[-2] and k > 50,
            strength=abs(k - d) / 20
        )

    @staticmethod
    def _stoch_oversold(df: pd.DataFrame) -> IndicatorState:
        k = df['Stoch_K'].iloc[-1]
        d = df['Stoch_D'].iloc[-1]
        return IndicatorState(
            name='stoch_oversold',
            value=k,
            is_bullish=k < 25 and k > d,
            is_bearish=False,
            strength=(25 - k) / 25 if k < 25 else 0
        )

    @staticmethod
    def _williams_r(df: pd.DataFrame) -> IndicatorState:
        wr = df['Williams_R'].iloc[-1]
        return IndicatorState(
            name='williams_r',
            value=wr,
            is_bullish=wr < -80,  # Oversold
            is_bearish=wr > -20,  # Overbought
            strength=abs(wr + 50) / 50
        )

    @staticmethod
    def _cci(df: pd.DataFrame) -> IndicatorState:
        cci = df['CCI'].iloc[-1]
        return IndicatorState(
            name='cci',
            value=cci,
            is_bullish=cci > 100 or (cci > -100 and cci < 0 and cci > df['CCI'].iloc[-2]),
            is_bearish=cci < -100 or cci > 200,
            strength=min(abs(cci) / 200, 1)
        )

    @staticmethod
    def _mfi(df: pd.DataFrame) -> IndicatorState:
        mfi = df['MFI'].iloc[-1]
        return IndicatorState(
            name='mfi',
            value=mfi,
            is_bullish=mfi < 30,  # Oversold
            is_bearish=mfi > 80,  # Overbought
            strength=abs(50 - mfi) / 50
        )

    @staticmethod
    def _rate_of_change(df: pd.DataFrame) -> IndicatorState:
        roc = df['ROC'].iloc[-1]
        return IndicatorState(
            name='roc',
            value=roc,
            is_bullish=roc > 0 and roc < 15,  # Momentum positif mais pas excessif
            is_bearish=roc < -10,
            strength=min(abs(roc) / 20, 1)
        )

    # === TREND INDICATORS ===

    @staticmethod
    def _ema_cross(df: pd.DataFrame, fast: int, slow: int) -> IndicatorState:
        fast_ema = df[f'EMA_{fast}'].iloc[-1]
        slow_ema = df[f'EMA_{slow}'].iloc[-1]
        fast_prev = df[f'EMA_{fast}'].iloc[-2]
        slow_prev = df[f'EMA_{slow}'].iloc[-2]

        cross_up = fast_ema > slow_ema and fast_prev <= slow_prev
        cross_down = fast_ema < slow_ema and fast_prev >= slow_prev
        above = fast_ema > slow_ema

        return IndicatorState(
            name=f'ema_cross_{fast}_{slow}',
            value=(fast_ema - slow_ema) / slow_ema * 100,
            is_bullish=cross_up or (above and fast_ema > fast_prev),
            is_bearish=cross_down,
            strength=abs(fast_ema - slow_ema) / slow_ema * 10
        )

    @staticmethod
    def _ema_alignment(df: pd.DataFrame) -> IndicatorState:
        ema_20 = df['EMA_20'].iloc[-1]
        ema_50 = df['EMA_50'].iloc[-1]
        ema_200 = df['EMA_200'].iloc[-1]

        bullish = ema_20 > ema_50 > ema_200
        bearish = ema_20 < ema_50 < ema_200

        return IndicatorState(
            name='ema_alignment',
            value=1 if bullish else (-1 if bearish else 0),
            is_bullish=bullish,
            is_bearish=bearish,
            strength=1.0 if bullish or bearish else 0.3
        )

    @staticmethod
    def _macd_cross(df: pd.DataFrame) -> IndicatorState:
        macd = df['MACD'].iloc[-1]
        signal = df['MACD_signal'].iloc[-1]
        macd_prev = df['MACD'].iloc[-2]
        signal_prev = df['MACD_signal'].iloc[-2]

        cross_up = macd > signal and macd_prev <= signal_prev
        cross_down = macd < signal and macd_prev >= signal_prev

        return IndicatorState(
            name='macd_cross',
            value=macd - signal,
            is_bullish=cross_up or (macd > signal and macd > 0),
            is_bearish=cross_down,
            strength=min(abs(macd - signal) * 10, 1)
        )

    @staticmethod
    def _macd_histogram_rising(df: pd.DataFrame) -> IndicatorState:
        hist = df['MACD_hist'].iloc[-1]
        hist_prev = df['MACD_hist'].iloc[-2]
        rising = hist > hist_prev

        return IndicatorState(
            name='macd_histogram_rising',
            value=hist,
            is_bullish=rising and hist > -0.5,
            is_bearish=not rising and hist < 0.5,
            strength=abs(hist - hist_prev) * 5
        )

    @staticmethod
    def _adx_trend(df: pd.DataFrame) -> IndicatorState:
        adx = df['ADX'].iloc[-1]
        plus_di = df['Plus_DI'].iloc[-1]
        minus_di = df['Minus_DI'].iloc[-1]

        strong_trend = adx > 25
        bullish = plus_di > minus_di and strong_trend

        return IndicatorState(
            name='adx_trend',
            value=adx,
            is_bullish=bullish,
            is_bearish=minus_di > plus_di and strong_trend,
            strength=min(adx / 50, 1)
        )

    @staticmethod
    def _aroon(df: pd.DataFrame) -> IndicatorState:
        period = 25
        high = df['High']
        low = df['Low']

        aroon_up = ((period - (period - high.rolling(period).apply(lambda x: x.argmax()))) / period) * 100
        aroon_down = ((period - (period - low.rolling(period).apply(lambda x: x.argmin()))) / period) * 100

        up_val = aroon_up.iloc[-1]
        down_val = aroon_down.iloc[-1]

        return IndicatorState(
            name='aroon_up',
            value=up_val - down_val,
            is_bullish=up_val > 70 and down_val < 30,
            is_bearish=down_val > 70 and up_val < 30,
            strength=abs(up_val - down_val) / 100
        )

    @staticmethod
    def _price_above_ema(df: pd.DataFrame, period: int) -> IndicatorState:
        close = df['Close'].iloc[-1]
        ema = df[f'EMA_{period}'].iloc[-1]
        above = close > ema
        distance = (close - ema) / ema * 100

        return IndicatorState(
            name=f'price_above_ema{period}',
            value=distance,
            is_bullish=above and distance < 10,  # Au-dessus mais pas trop étiré
            is_bearish=not above and distance < -5,
            strength=min(abs(distance) / 10, 1)
        )

    # === VOLUME INDICATORS ===

    @staticmethod
    def _volume_spike(df: pd.DataFrame) -> IndicatorState:
        vol = df['Volume'].iloc[-1]
        vol_avg = df['Volume'].rolling(20).mean().iloc[-1]
        ratio = vol / vol_avg if vol_avg > 0 else 1
        price_up = df['Close'].iloc[-1] > df['Close'].iloc[-2]

        return IndicatorState(
            name='volume_spike',
            value=ratio,
            is_bullish=ratio > 1.5 and price_up,
            is_bearish=ratio > 1.5 and not price_up,
            strength=min((ratio - 1) / 2, 1) if ratio > 1 else 0
        )

    @staticmethod
    def _volume_trend(df: pd.DataFrame) -> IndicatorState:
        vol_5 = df['Volume'].iloc[-5:].mean()
        vol_20 = df['Volume'].iloc[-20:].mean()
        increasing = vol_5 > vol_20

        return IndicatorState(
            name='volume_trend',
            value=vol_5 / vol_20 if vol_20 > 0 else 1,
            is_bullish=increasing and df['Close'].iloc[-1] > df['Close'].iloc[-5],
            is_bearish=increasing and df['Close'].iloc[-1] < df['Close'].iloc[-5],
            strength=abs(vol_5 - vol_20) / vol_20 if vol_20 > 0 else 0
        )

    @staticmethod
    def _obv_rising(df: pd.DataFrame) -> IndicatorState:
        obv_now = df['OBV'].iloc[-1]
        obv_5d = df['OBV'].iloc[-5]
        rising = obv_now > obv_5d

        return IndicatorState(
            name='obv_rising',
            value=obv_now - obv_5d,
            is_bullish=rising,
            is_bearish=not rising,
            strength=0.7 if rising else 0.3
        )

    @staticmethod
    def _vwap_cross(df: pd.DataFrame) -> IndicatorState:
        close = df['Close'].iloc[-1]
        vwap = df['VWAP'].iloc[-1]
        close_prev = df['Close'].iloc[-2]
        vwap_prev = df['VWAP'].iloc[-2]

        cross_up = close > vwap and close_prev <= vwap_prev
        above = close > vwap

        return IndicatorState(
            name='vwap_cross',
            value=(close - vwap) / vwap * 100,
            is_bullish=cross_up or above,
            is_bearish=close < vwap and close_prev >= vwap_prev,
            strength=abs(close - vwap) / vwap * 10
        )

    @staticmethod
    def _accumulation_distribution(df: pd.DataFrame) -> IndicatorState:
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']

        mfm = ((close - low) - (high - close)) / (high - low + 0.001)
        adl = (mfm * volume).cumsum()

        adl_rising = adl.iloc[-1] > adl.iloc[-5]

        return IndicatorState(
            name='accumulation',
            value=adl.iloc[-1],
            is_bullish=adl_rising and close.iloc[-1] > close.iloc[-5],
            is_bearish=not adl_rising and close.iloc[-1] < close.iloc[-5],
            strength=0.6 if adl_rising else 0.4
        )

    # === VOLATILITY INDICATORS ===

    @staticmethod
    def _bollinger_squeeze(df: pd.DataFrame) -> IndicatorState:
        width = df['BB_width'].iloc[-1]
        width_avg = df['BB_width'].rolling(50).mean().iloc[-1]
        squeeze = width < width_avg * 0.7

        return IndicatorState(
            name='bollinger_squeeze',
            value=width / width_avg if width_avg > 0 else 1,
            is_bullish=squeeze,  # Squeeze = breakout imminent
            is_bearish=False,
            strength=1 - (width / width_avg) if squeeze else 0
        )

    @staticmethod
    def _bollinger_breakout(df: pd.DataFrame) -> IndicatorState:
        close = df['Close'].iloc[-1]
        upper = df['BB_upper'].iloc[-1]
        lower = df['BB_lower'].iloc[-1]

        breakout_up = close > upper
        breakout_down = close < lower

        return IndicatorState(
            name='bollinger_breakout',
            value=(close - df['BB_middle'].iloc[-1]) / (upper - lower) * 2,
            is_bullish=breakout_up,
            is_bearish=breakout_down,
            strength=1 if breakout_up or breakout_down else 0
        )

    @staticmethod
    def _atr_low(df: pd.DataFrame) -> IndicatorState:
        atr = df['ATR'].iloc[-1]
        atr_avg = df['ATR'].rolling(50).mean().iloc[-1]
        low_vol = atr < atr_avg * 0.8

        return IndicatorState(
            name='atr_low',
            value=atr / atr_avg if atr_avg > 0 else 1,
            is_bullish=low_vol,  # Low volatility = potential breakout
            is_bearish=False,
            strength=1 - (atr / atr_avg) if low_vol else 0
        )

    @staticmethod
    def _keltner_breakout(df: pd.DataFrame) -> IndicatorState:
        close = df['Close'].iloc[-1]
        ema_20 = df['EMA_20'].iloc[-1]
        atr = df['ATR'].iloc[-1]

        upper = ema_20 + 2 * atr
        lower = ema_20 - 2 * atr

        return IndicatorState(
            name='keltner_breakout',
            value=(close - ema_20) / atr if atr > 0 else 0,
            is_bullish=close > upper,
            is_bearish=close < lower,
            strength=abs(close - ema_20) / atr if atr > 0 else 0
        )

    # === PRICE ACTION ===

    @staticmethod
    def _higher_high(df: pd.DataFrame) -> IndicatorState:
        highs = df['High'].iloc[-20:]
        recent_high = highs.iloc[-5:].max()
        prev_high = highs.iloc[-15:-5].max()
        hh = recent_high > prev_high

        return IndicatorState(
            name='higher_high',
            value=(recent_high - prev_high) / prev_high * 100 if prev_high > 0 else 0,
            is_bullish=hh,
            is_bearish=False,
            strength=0.8 if hh else 0.2
        )

    @staticmethod
    def _higher_low(df: pd.DataFrame) -> IndicatorState:
        lows = df['Low'].iloc[-20:]
        recent_low = lows.iloc[-5:].min()
        prev_low = lows.iloc[-15:-5].min()
        hl = recent_low > prev_low

        return IndicatorState(
            name='higher_low',
            value=(recent_low - prev_low) / prev_low * 100 if prev_low > 0 else 0,
            is_bullish=hl,
            is_bearish=False,
            strength=0.8 if hl else 0.2
        )

    @staticmethod
    def _breakout_resistance(df: pd.DataFrame) -> IndicatorState:
        close = df['Close'].iloc[-1]
        high_20 = df['High'].iloc[-20:-1].max()
        breakout = close > high_20

        return IndicatorState(
            name='breakout_resistance',
            value=(close - high_20) / high_20 * 100 if high_20 > 0 else 0,
            is_bullish=breakout,
            is_bearish=False,
            strength=1 if breakout else 0
        )

    @staticmethod
    def _support_bounce(df: pd.DataFrame) -> IndicatorState:
        close = df['Close'].iloc[-1]
        low_20 = df['Low'].iloc[-20:-1].min()
        close_prev = df['Close'].iloc[-3:-1].min()

        # Prix a touché le support et remonte
        bounce = close_prev <= low_20 * 1.02 and close > close_prev

        return IndicatorState(
            name='support_bounce',
            value=(close - low_20) / low_20 * 100 if low_20 > 0 else 0,
            is_bullish=bounce,
            is_bearish=False,
            strength=0.9 if bounce else 0
        )

    @staticmethod
    def _gap_up(df: pd.DataFrame) -> IndicatorState:
        open_today = df['Open'].iloc[-1] if 'Open' in df.columns else df['Close'].iloc[-1]
        close_prev = df['Close'].iloc[-2]
        gap = (open_today - close_prev) / close_prev * 100

        return IndicatorState(
            name='gap_up',
            value=gap,
            is_bullish=gap > 1,  # Gap > 1%
            is_bearish=gap < -1,
            strength=min(abs(gap) / 5, 1)
        )

    # === CANDLESTICK PATTERNS ===

    @staticmethod
    def _bullish_engulfing(df: pd.DataFrame) -> IndicatorState:
        if 'Open' not in df.columns:
            return IndicatorState('bullish_engulfing', 0, False, False, 0)

        o1, c1 = df['Open'].iloc[-2], df['Close'].iloc[-2]
        o2, c2 = df['Open'].iloc[-1], df['Close'].iloc[-1]

        bearish_candle = c1 < o1
        bullish_candle = c2 > o2
        engulfing = bullish_candle and bearish_candle and o2 < c1 and c2 > o1

        return IndicatorState(
            name='bullish_engulfing',
            value=1 if engulfing else 0,
            is_bullish=engulfing,
            is_bearish=False,
            strength=1 if engulfing else 0
        )

    @staticmethod
    def _hammer(df: pd.DataFrame) -> IndicatorState:
        if 'Open' not in df.columns:
            return IndicatorState('hammer', 0, False, False, 0)

        o, h, l, c = df['Open'].iloc[-1], df['High'].iloc[-1], df['Low'].iloc[-1], df['Close'].iloc[-1]
        body = abs(c - o)
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)

        hammer = lower_shadow > 2 * body and upper_shadow < body * 0.5
        downtrend = df['Close'].iloc[-5] > df['Close'].iloc[-2]

        return IndicatorState(
            name='hammer',
            value=lower_shadow / body if body > 0 else 0,
            is_bullish=hammer and downtrend,
            is_bearish=False,
            strength=1 if hammer and downtrend else 0
        )

    @staticmethod
    def _morning_star(df: pd.DataFrame) -> IndicatorState:
        if 'Open' not in df.columns or len(df) < 3:
            return IndicatorState('morning_star', 0, False, False, 0)

        # Jour 1: Grande bougie rouge
        o1, c1 = df['Open'].iloc[-3], df['Close'].iloc[-3]
        bearish1 = c1 < o1 and abs(c1 - o1) > df['ATR'].iloc[-3] * 0.5

        # Jour 2: Petit corps (indécision)
        o2, c2 = df['Open'].iloc[-2], df['Close'].iloc[-2]
        small_body = abs(c2 - o2) < df['ATR'].iloc[-2] * 0.3

        # Jour 3: Grande bougie verte qui clôture au-dessus du milieu de jour 1
        o3, c3 = df['Open'].iloc[-1], df['Close'].iloc[-1]
        bullish3 = c3 > o3 and c3 > (o1 + c1) / 2

        pattern = bearish1 and small_body and bullish3

        return IndicatorState(
            name='morning_star',
            value=1 if pattern else 0,
            is_bullish=pattern,
            is_bearish=False,
            strength=1 if pattern else 0
        )

    @staticmethod
    def _doji_reversal(df: pd.DataFrame) -> IndicatorState:
        if 'Open' not in df.columns:
            return IndicatorState('doji_reversal', 0, False, False, 0)

        o, c = df['Open'].iloc[-1], df['Close'].iloc[-1]
        h, l = df['High'].iloc[-1], df['Low'].iloc[-1]

        body = abs(c - o)
        range_hl = h - l
        doji = body < range_hl * 0.1 if range_hl > 0 else False

        # Doji après une tendance baissière = potentiel retournement
        downtrend = df['Close'].iloc[-5] > df['Close'].iloc[-2]

        return IndicatorState(
            name='doji_reversal',
            value=body / range_hl if range_hl > 0 else 0,
            is_bullish=doji and downtrend,
            is_bearish=doji and not downtrend,
            strength=0.8 if doji else 0
        )


class IndicatorDiscovery:
    """
    Système d'auto-découverte des indicateurs qui fonctionnent.

    Ne fait AUCUNE hypothèse sur ce qui marche.
    Laisse les DONNÉES décider quels indicateurs sont prédictifs.

    Usage:
        discovery = IndicatorDiscovery()
        await discovery.initialize()

        # Lors d'un nouveau signal
        indicators = discovery.analyze_signal(df, symbol, entry_price)

        # Après outcome connu
        discovery.record_outcome(signal_id, pnl, is_winner)

        # Obtenir les poids optimaux
        weights = discovery.get_indicator_weights()
    """

    def __init__(self, data_dir: str = "data/indicator_discovery"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.signals_file = self.data_dir / "indicator_signals.json"
        self.weights_file = self.data_dir / "indicator_weights.json"
        self.stats_file = self.data_dir / "indicator_stats.json"

        self.signals: List[SignalIndicators] = []

        # Poids initiaux égaux pour tous les indicateurs
        self.indicator_weights: Dict[str, float] = {}

        # Statistiques par indicateur
        self.indicator_stats: Dict[str, Dict] = {}

        self._initialized = False

    async def initialize(self):
        """Charger les données persistées"""
        if self._initialized:
            return

        # Charger les signaux
        if self.signals_file.exists():
            try:
                with open(self.signals_file, 'r') as f:
                    data = json.load(f)
                    self.signals = [SignalIndicators.from_dict(s) for s in data]
                logger.info(f"Loaded {len(self.signals)} indicator signals")
            except Exception as e:
                logger.error(f"Error loading indicator signals: {e}")

        # Charger les poids
        if self.weights_file.exists():
            try:
                with open(self.weights_file, 'r') as f:
                    self.indicator_weights = json.load(f)
                logger.info(f"Loaded indicator weights: {len(self.indicator_weights)} indicators")
            except Exception as e:
                logger.error(f"Error loading weights: {e}")

        # Charger les stats
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    self.indicator_stats = json.load(f)
            except Exception as e:
                logger.error(f"Error loading stats: {e}")

        self._initialized = True
        logger.info("IndicatorDiscovery initialized")

    def _save(self):
        """Persister toutes les données"""
        with open(self.signals_file, 'w') as f:
            json.dump([s.to_dict() for s in self.signals], f, indent=2)

        with open(self.weights_file, 'w') as f:
            json.dump(self.indicator_weights, f, indent=2)

        with open(self.stats_file, 'w') as f:
            json.dump(self.indicator_stats, f, indent=2)

    def analyze_signal(
        self,
        df: pd.DataFrame,
        symbol: str,
        entry_price: float
    ) -> SignalIndicators:
        """
        Analyser un signal et enregistrer tous les indicateurs actifs.

        Args:
            df: DataFrame avec OHLCV
            symbol: Ticker
            entry_price: Prix d'entrée

        Returns:
            SignalIndicators avec tous les indicateurs calculés
        """
        # Calculer tous les indicateurs
        indicators = IndicatorCalculator.calculate_all(df)

        # Créer le signal
        signal = SignalIndicators(
            symbol=symbol,
            signal_date=datetime.now().strftime('%Y-%m-%d'),
            entry_price=entry_price,
            indicators=indicators
        )

        # Ajouter à la liste
        self.signals.append(signal)
        self._save()

        # Initialiser les poids si nouveaux indicateurs
        for name in indicators:
            if name not in self.indicator_weights:
                self.indicator_weights[name] = 1.0

        logger.info(f"[INDICATOR] Analyzed {symbol}: {sum(1 for i in indicators.values() if i.is_bullish)} bullish indicators")

        return signal

    def record_outcome(
        self,
        symbol: str,
        signal_date: str,
        exit_price: float,
        pnl_pct: float,
        is_winner: bool
    ):
        """
        Enregistrer l'outcome d'un signal.

        Args:
            symbol: Ticker
            signal_date: Date du signal (YYYY-MM-DD)
            exit_price: Prix de sortie
            pnl_pct: P&L en pourcentage
            is_winner: True si trade gagnant
        """
        # Trouver le signal correspondant
        for signal in self.signals:
            if signal.symbol == symbol and signal.signal_date == signal_date:
                signal.exit_price = exit_price
                signal.pnl_pct = pnl_pct
                signal.is_winner = is_winner
                break

        self._save()

        # Mettre à jour les statistiques des indicateurs
        self._update_indicator_stats()

    def _update_indicator_stats(self):
        """Mettre à jour les statistiques de chaque indicateur"""
        # Réinitialiser les stats
        self.indicator_stats = {}

        # Signaux avec outcome connu
        completed = [s for s in self.signals if s.is_winner is not None]

        if len(completed) < 10:
            logger.info(f"Not enough data for indicator stats ({len(completed)} signals)")
            return

        # Pour chaque indicateur
        all_indicators = set()
        for signal in completed:
            all_indicators.update(signal.indicators.keys())

        for indicator_name in all_indicators:
            # Signaux où cet indicateur était bullish
            bullish_signals = [
                s for s in completed
                if indicator_name in s.indicators and s.indicators[indicator_name].is_bullish
            ]

            if len(bullish_signals) < 5:
                continue

            winners = len([s for s in bullish_signals if s.is_winner])
            win_rate = winners / len(bullish_signals)
            avg_pnl = sum(s.pnl_pct or 0 for s in bullish_signals) / len(bullish_signals)

            self.indicator_stats[indicator_name] = {
                'total_signals': len(bullish_signals),
                'winners': winners,
                'losers': len(bullish_signals) - winners,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'score': win_rate * np.sqrt(len(bullish_signals))  # Score = WR * sqrt(N)
            }

        # Mettre à jour les poids basés sur les stats
        self._update_weights()

    def _update_weights(self):
        """Mettre à jour les poids des indicateurs basés sur leur performance"""
        if not self.indicator_stats:
            return

        # Score de chaque indicateur
        scores = {}
        for name, stats in self.indicator_stats.items():
            if stats['total_signals'] >= 10:  # Minimum 10 signaux
                # Poids = win_rate * 2 (pour que 50% WR = poids 1.0)
                scores[name] = stats['win_rate'] * 2
            else:
                scores[name] = 1.0  # Poids neutre si pas assez de données

        # Normaliser pour que la moyenne soit 1.0
        if scores:
            avg_score = sum(scores.values()) / len(scores)
            self.indicator_weights = {k: v / avg_score for k, v in scores.items()}

        self._save()

        logger.info(f"[INDICATOR] Updated weights based on {len(self.indicator_stats)} indicators")

    def get_indicator_weights(self) -> Dict[str, float]:
        """Obtenir les poids actuels des indicateurs"""
        return self.indicator_weights.copy()

    def get_top_indicators(self, n: int = 10) -> List[Dict]:
        """Obtenir les N meilleurs indicateurs par win rate"""
        sorted_stats = sorted(
            self.indicator_stats.items(),
            key=lambda x: x[1].get('win_rate', 0),
            reverse=True
        )
        return [
            {'name': name, **stats}
            for name, stats in sorted_stats[:n]
            if stats.get('total_signals', 0) >= 10
        ]

    def get_worst_indicators(self, n: int = 5) -> List[Dict]:
        """Obtenir les N pires indicateurs"""
        sorted_stats = sorted(
            self.indicator_stats.items(),
            key=lambda x: x[1].get('win_rate', 1),
            reverse=False
        )
        return [
            {'name': name, **stats}
            for name, stats in sorted_stats[:n]
            if stats.get('total_signals', 0) >= 10
        ]

    def calculate_composite_score(self, indicators: Dict[str, IndicatorState]) -> float:
        """
        Calculer un score composite basé sur les poids appris.

        Args:
            indicators: Dictionnaire des indicateurs calculés

        Returns:
            Score 0-100
        """
        if not indicators:
            return 50.0

        total_weight = 0
        weighted_sum = 0

        for name, state in indicators.items():
            weight = self.indicator_weights.get(name, 1.0)

            if state.is_bullish:
                score = 50 + state.strength * 50  # 50-100
            elif state.is_bearish:
                score = 50 - state.strength * 50  # 0-50
            else:
                score = 50  # Neutre

            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return 50.0

        return weighted_sum / total_weight

    def get_statistics(self) -> Dict:
        """Obtenir les statistiques globales"""
        completed = [s for s in self.signals if s.is_winner is not None]
        pending = len(self.signals) - len(completed)

        winners = len([s for s in completed if s.is_winner])

        return {
            'total_signals': len(self.signals),
            'completed': len(completed),
            'pending': pending,
            'winners': winners,
            'losers': len(completed) - winners,
            'win_rate': winners / len(completed) * 100 if completed else 0,
            'indicators_tracked': len(self.indicator_weights),
            'top_indicators': self.get_top_indicators(5),
            'worst_indicators': self.get_worst_indicators(3)
        }


# Singleton
_indicator_discovery: Optional[IndicatorDiscovery] = None


async def get_indicator_discovery() -> IndicatorDiscovery:
    """Obtenir l'instance singleton"""
    global _indicator_discovery
    if _indicator_discovery is None:
        _indicator_discovery = IndicatorDiscovery()
        await _indicator_discovery.initialize()
    return _indicator_discovery
