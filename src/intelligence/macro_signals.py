"""
Macro Signal Fetcher - Quantitative macro indicators from yfinance.

All signals are backtestable with 10+ years of history.

Signals:
1. Yield curve spread (^TNX - ^IRX) - negative = inverted = recession risk
2. Credit spread (HYG/LQD ratio) - widening = stress
3. Dollar strength (UUP momentum) - strong dollar + inverted curve = high conviction BEAR
4. Defensive rotation (XLU+XLP vs XLK+XLY) - rotation to defensives = BEAR

Combined into a macro regime score from -1 (BEAR) to +1 (BULL).
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class MacroSignalResult:
    """Result from a single macro signal."""
    name: str
    value: float       # Raw value of the indicator
    signal: float      # Normalized signal: -1 (bearish) to +1 (bullish)
    description: str


@dataclass
class MacroRegimeResult:
    """Combined macro regime assessment."""
    score: float              # -1 (BEAR) to +1 (BULL)
    confidence: float         # 0-1
    yield_curve: Optional[MacroSignalResult] = None
    credit_spread: Optional[MacroSignalResult] = None
    dollar_strength: Optional[MacroSignalResult] = None
    defensive_rotation: Optional[MacroSignalResult] = None

    @property
    def bias(self):
        # type: () -> str
        if self.score > 0.3:
            return 'bullish'
        elif self.score < -0.3:
            return 'bearish'
        return 'neutral'


class MacroSignalFetcher:
    """Quantitative macro indicators from yfinance - all backtestable."""

    def __init__(self):
        self._cache = {}  # type: Dict[str, pd.DataFrame]

    def _get_data(self, ticker, start=None, end=None, period='2y'):
        # type: (str, Optional[str], Optional[str], str) -> Optional[pd.DataFrame]
        """Fetch data with caching."""
        cache_key = "%s_%s_%s" % (ticker, start or '', end or period)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            if start and end:
                df = yf.download(ticker, start=start, end=end, progress=False)
            else:
                df = yf.download(ticker, period=period, progress=False)
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            self._cache[cache_key] = df
            return df
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", ticker, e)
            return None

    def get_yield_curve_spread(self, date=None, tnx_df=None, irx_df=None):
        # type: (Optional[pd.Timestamp], Optional[pd.DataFrame], Optional[pd.DataFrame]) -> Optional[MacroSignalResult]
        """Yield curve spread: ^TNX (10Y) - ^IRX (3-month).

        Negative = inverted = recession 6-18 months ahead.
        """
        if tnx_df is None:
            tnx_df = self._get_data('^TNX')
        if irx_df is None:
            irx_df = self._get_data('^IRX')

        if tnx_df is None or irx_df is None:
            return None

        if date is not None:
            tnx_df = tnx_df.loc[:date]
            irx_df = irx_df.loc[:date]

        if tnx_df.empty or irx_df.empty:
            return None

        # Align dates
        common = tnx_df.index.intersection(irx_df.index)
        if len(common) == 0:
            return None

        tnx_val = float(tnx_df.loc[common[-1], 'Close'])
        irx_val = float(irx_df.loc[common[-1], 'Close'])
        spread = tnx_val - irx_val  # In percentage points

        # Signal: spread > 1.5 = bullish, < 0 = bearish
        if spread > 1.5:
            signal = 1.0
            desc = "Yield curve steep (%.2f%%) - expansion" % spread
        elif spread > 0.5:
            signal = 0.5
            desc = "Yield curve normal (%.2f%%)" % spread
        elif spread > 0:
            signal = 0.0
            desc = "Yield curve flat (%.2f%%) - caution" % spread
        elif spread > -0.5:
            signal = -0.5
            desc = "Yield curve slightly inverted (%.2f%%) - recession risk" % spread
        else:
            signal = -1.0
            desc = "Yield curve deeply inverted (%.2f%%) - recession likely" % spread

        return MacroSignalResult(
            name='yield_curve_spread',
            value=spread,
            signal=signal,
            description=desc
        )

    def get_credit_spread(self, date=None, hyg_df=None, lqd_df=None):
        # type: (Optional[pd.Timestamp], Optional[pd.DataFrame], Optional[pd.DataFrame]) -> Optional[MacroSignalResult]
        """Credit spread proxy: HYG/LQD ratio.

        HYG = high yield bonds, LQD = investment grade.
        Falling ratio = widening spreads = stress.
        """
        if hyg_df is None:
            hyg_df = self._get_data('HYG')
        if lqd_df is None:
            lqd_df = self._get_data('LQD')

        if hyg_df is None or lqd_df is None:
            return None

        if date is not None:
            hyg_df = hyg_df.loc[:date]
            lqd_df = lqd_df.loc[:date]

        if hyg_df.empty or lqd_df.empty:
            return None

        common = hyg_df.index.intersection(lqd_df.index)
        if len(common) < 60:
            return None

        # Compute ratio over last 60 days
        hyg_recent = hyg_df.loc[common[-60:], 'Close']
        lqd_recent = lqd_df.loc[common[-60:], 'Close']
        ratio = hyg_recent / lqd_recent
        current_ratio = float(ratio.iloc[-1])

        # 60-day momentum of ratio
        ratio_start = float(ratio.iloc[0])
        momentum = (current_ratio - ratio_start) / ratio_start if ratio_start > 0 else 0

        # Signal: rising ratio = tightening = bullish
        if momentum > 0.02:
            signal = 1.0
            desc = "Credit spreads tightening (HYG/LQD +%.1f%%) - risk-on" % (momentum * 100)
        elif momentum > 0:
            signal = 0.3
            desc = "Credit spreads stable (HYG/LQD +%.1f%%)" % (momentum * 100)
        elif momentum > -0.02:
            signal = -0.3
            desc = "Credit spreads mildly widening (HYG/LQD %.1f%%)" % (momentum * 100)
        elif momentum > -0.05:
            signal = -0.7
            desc = "Credit spreads widening (HYG/LQD %.1f%%) - stress" % (momentum * 100)
        else:
            signal = -1.0
            desc = "Credit spreads blowout (HYG/LQD %.1f%%) - crisis risk" % (momentum * 100)

        return MacroSignalResult(
            name='credit_spread',
            value=current_ratio,
            signal=signal,
            description=desc
        )

    def get_dollar_strength(self, date=None, uup_df=None):
        # type: (Optional[pd.Timestamp], Optional[pd.DataFrame]) -> Optional[MacroSignalResult]
        """Dollar strength: UUP ETF 20-day momentum.

        Strong dollar = headwind for equities (especially multinationals).
        """
        if uup_df is None:
            uup_df = self._get_data('UUP')

        if uup_df is None:
            return None

        if date is not None:
            uup_df = uup_df.loc[:date]

        if len(uup_df) < 20:
            return None

        close = uup_df['Close']
        current = float(close.iloc[-1])
        past_20d = float(close.iloc[-20])
        momentum = (current - past_20d) / past_20d if past_20d > 0 else 0

        # Strong dollar = bearish for stocks (inverse signal)
        if momentum > 0.03:
            signal = -0.8
            desc = "Dollar surging (+%.1f%% 20d) - equity headwind" % (momentum * 100)
        elif momentum > 0.01:
            signal = -0.3
            desc = "Dollar strengthening (+%.1f%% 20d)" % (momentum * 100)
        elif momentum > -0.01:
            signal = 0.0
            desc = "Dollar stable (%.1f%% 20d)" % (momentum * 100)
        elif momentum > -0.03:
            signal = 0.3
            desc = "Dollar weakening (%.1f%% 20d) - equity tailwind" % (momentum * 100)
        else:
            signal = 0.8
            desc = "Dollar falling sharply (%.1f%% 20d) - bullish equities" % (momentum * 100)

        return MacroSignalResult(
            name='dollar_strength',
            value=momentum,
            signal=signal,
            description=desc
        )

    def get_defensive_rotation(self, date=None, sector_data=None):
        # type: (Optional[pd.Timestamp], Optional[Dict[str, pd.DataFrame]]) -> Optional[MacroSignalResult]
        """Defensive rotation: (XLU+XLP) vs (XLK+XLY) relative strength.

        Rotation into defensives = BEAR signal.
        """
        tickers = ['XLU', 'XLP', 'XLK', 'XLY']
        data = {}

        for t in tickers:
            if sector_data and t in sector_data:
                df = sector_data[t]
            else:
                df = self._get_data(t)
            if df is None or len(df) < 60:
                return None
            if date is not None:
                df = df.loc[:date]
            if len(df) < 60:
                return None
            data[t] = df

        # 60-day relative performance
        def _perf_60d(df):
            c = df['Close']
            if len(c) < 60:
                return 0.0
            return (float(c.iloc[-1]) / float(c.iloc[-60]) - 1)

        def_perf = (_perf_60d(data['XLU']) + _perf_60d(data['XLP'])) / 2
        off_perf = (_perf_60d(data['XLK']) + _perf_60d(data['XLY'])) / 2
        spread = off_perf - def_perf  # Positive = offense leading = bullish

        if spread > 0.05:
            signal = 1.0
            desc = "Offense leading defensives by %.1f%% - risk-on" % (spread * 100)
        elif spread > 0.02:
            signal = 0.5
            desc = "Offense slightly leading (%.1f%%)" % (spread * 100)
        elif spread > -0.02:
            signal = 0.0
            desc = "Offense/defense balanced (%.1f%%)" % (spread * 100)
        elif spread > -0.05:
            signal = -0.5
            desc = "Defensive rotation starting (%.1f%%)" % (spread * 100)
        else:
            signal = -1.0
            desc = "Strong defensive rotation (%.1f%%) - risk-off" % (spread * 100)

        return MacroSignalResult(
            name='defensive_rotation',
            value=spread,
            signal=signal,
            description=desc
        )

    def get_macro_regime_score(self, date=None, **kwargs):
        # type: (Optional[pd.Timestamp], ...) -> MacroRegimeResult
        """Combine all 4 signals into a regime score -1 (BEAR) to +1 (BULL).

        Weights:
        - Yield curve: 30% (strongest recession predictor)
        - Credit spread: 30% (real-time stress indicator)
        - Dollar strength: 15% (secondary)
        - Defensive rotation: 25% (confirmation signal)
        """
        yc = self.get_yield_curve_spread(date, **{k: v for k, v in kwargs.items()
                                                    if k in ('tnx_df', 'irx_df')})
        cs = self.get_credit_spread(date, **{k: v for k, v in kwargs.items()
                                              if k in ('hyg_df', 'lqd_df')})
        ds = self.get_dollar_strength(date, **{k: v for k, v in kwargs.items()
                                                if k in ('uup_df',)})
        dr = self.get_defensive_rotation(date, **{k: v for k, v in kwargs.items()
                                                    if k in ('sector_data',)})

        signals = []
        weights = []

        if yc is not None:
            signals.append(yc.signal)
            weights.append(0.30)
        if cs is not None:
            signals.append(cs.signal)
            weights.append(0.30)
        if ds is not None:
            signals.append(ds.signal)
            weights.append(0.15)
        if dr is not None:
            signals.append(dr.signal)
            weights.append(0.25)

        if not signals:
            return MacroRegimeResult(score=0.0, confidence=0.0)

        # Weighted average
        total_w = sum(weights)
        score = sum(s * w for s, w in zip(signals, weights)) / total_w

        # Confidence = % of signals available * agreement level
        agreement = 1.0 - np.std(signals) if len(signals) > 1 else 0.5
        coverage = total_w / 1.0  # 1.0 = all 4 signals available
        confidence = min(1.0, coverage * agreement)

        return MacroRegimeResult(
            score=max(-1.0, min(1.0, score)),
            confidence=confidence,
            yield_curve=yc,
            credit_spread=cs,
            dollar_strength=ds,
            defensive_rotation=dr
        )
