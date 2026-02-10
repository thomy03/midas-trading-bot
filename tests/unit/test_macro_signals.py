"""Tests for MacroSignalFetcher - V8 macro quantitative indicators."""
import sys
import importlib.util
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Direct import to bypass src/intelligence/__init__.py (needs zoneinfo/Python 3.9+)
_spec = importlib.util.spec_from_file_location(
    "macro_signals", "src/intelligence/macro_signals.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["macro_signals"] = _mod
_spec.loader.exec_module(_mod)

MacroSignalFetcher = _mod.MacroSignalFetcher
MacroSignalResult = _mod.MacroSignalResult
MacroRegimeResult = _mod.MacroRegimeResult


def _make_df(values, start='2024-01-01', freq='B'):
    """Helper: create a DataFrame with Close column."""
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.DataFrame({'Close': values}, index=dates)


class TestYieldCurveSpread:
    """Test yield curve spread signal."""

    def test_steep_curve_bullish(self):
        fetcher = MacroSignalFetcher()
        # 10Y at 5%, 3-month at 2% => spread = 3% => bullish
        tnx = _make_df([5.0] * 10)
        irx = _make_df([2.0] * 10)
        result = fetcher.get_yield_curve_spread(tnx_df=tnx, irx_df=irx)
        assert result is not None
        assert result.signal == 1.0
        assert result.value == pytest.approx(3.0)

    def test_inverted_curve_bearish(self):
        fetcher = MacroSignalFetcher()
        # 10Y at 3%, 3-month at 5% => spread = -2% => bearish
        tnx = _make_df([3.0] * 10)
        irx = _make_df([5.0] * 10)
        result = fetcher.get_yield_curve_spread(tnx_df=tnx, irx_df=irx)
        assert result is not None
        assert result.signal == -1.0
        assert result.value == pytest.approx(-2.0)

    def test_flat_curve_neutral(self):
        fetcher = MacroSignalFetcher()
        tnx = _make_df([4.0] * 10)
        irx = _make_df([3.8] * 10)
        result = fetcher.get_yield_curve_spread(tnx_df=tnx, irx_df=irx)
        assert result is not None
        assert result.signal == 0.0  # flat

    def test_with_date_filter(self):
        fetcher = MacroSignalFetcher()
        vals = list(range(10))
        tnx = _make_df([4.0 + v * 0.1 for v in vals])
        irx = _make_df([2.0] * 10)
        # Filter to 5th date
        date = tnx.index[4]
        result = fetcher.get_yield_curve_spread(date=date, tnx_df=tnx, irx_df=irx)
        assert result is not None

    def test_empty_data_returns_none(self):
        fetcher = MacroSignalFetcher()
        result = fetcher.get_yield_curve_spread(tnx_df=pd.DataFrame(), irx_df=pd.DataFrame())
        assert result is None


class TestCreditSpread:
    """Test credit spread proxy signal."""

    def test_tightening_spreads_bullish(self):
        fetcher = MacroSignalFetcher()
        # HYG rising relative to LQD = tightening = bullish
        hyg_vals = [80 + i * 0.3 for i in range(70)]
        lqd_vals = [100.0] * 70
        hyg = _make_df(hyg_vals)
        lqd = _make_df(lqd_vals)
        result = fetcher.get_credit_spread(hyg_df=hyg, lqd_df=lqd)
        assert result is not None
        assert result.signal > 0  # bullish

    def test_widening_spreads_bearish(self):
        fetcher = MacroSignalFetcher()
        # HYG falling relative to LQD = widening = bearish
        hyg_vals = [85 - i * 0.3 for i in range(70)]
        lqd_vals = [100.0] * 70
        hyg = _make_df(hyg_vals)
        lqd = _make_df(lqd_vals)
        result = fetcher.get_credit_spread(hyg_df=hyg, lqd_df=lqd)
        assert result is not None
        assert result.signal < 0  # bearish

    def test_insufficient_data_returns_none(self):
        fetcher = MacroSignalFetcher()
        hyg = _make_df([80.0] * 10)
        lqd = _make_df([100.0] * 10)
        result = fetcher.get_credit_spread(hyg_df=hyg, lqd_df=lqd)
        assert result is None  # < 60 common dates


class TestDollarStrength:
    """Test dollar strength signal."""

    def test_strong_dollar_bearish_equities(self):
        fetcher = MacroSignalFetcher()
        # UUP rising = strong dollar = bearish for equities
        vals = [25 + i * 0.1 for i in range(30)]
        uup = _make_df(vals)
        result = fetcher.get_dollar_strength(uup_df=uup)
        assert result is not None
        assert result.signal < 0  # bearish (inverse)

    def test_weak_dollar_bullish_equities(self):
        fetcher = MacroSignalFetcher()
        # UUP falling = weak dollar = bullish for equities
        vals = [28 - i * 0.1 for i in range(30)]
        uup = _make_df(vals)
        result = fetcher.get_dollar_strength(uup_df=uup)
        assert result is not None
        assert result.signal > 0  # bullish

    def test_insufficient_data_returns_none(self):
        fetcher = MacroSignalFetcher()
        uup = _make_df([25.0] * 5)
        result = fetcher.get_dollar_strength(uup_df=uup)
        assert result is None


class TestDefensiveRotation:
    """Test defensive rotation signal."""

    def test_offense_leading_bullish(self):
        fetcher = MacroSignalFetcher()
        # XLK + XLY rising faster than XLU + XLP
        sector_data = {
            'XLK': _make_df([100 + i * 0.5 for i in range(70)]),
            'XLY': _make_df([100 + i * 0.5 for i in range(70)]),
            'XLU': _make_df([100 + i * 0.05 for i in range(70)]),
            'XLP': _make_df([100 + i * 0.05 for i in range(70)]),
        }
        result = fetcher.get_defensive_rotation(sector_data=sector_data)
        assert result is not None
        assert result.signal > 0  # offense leading = bullish

    def test_defense_leading_bearish(self):
        fetcher = MacroSignalFetcher()
        sector_data = {
            'XLK': _make_df([100 - i * 0.3 for i in range(70)]),
            'XLY': _make_df([100 - i * 0.3 for i in range(70)]),
            'XLU': _make_df([100 + i * 0.3 for i in range(70)]),
            'XLP': _make_df([100 + i * 0.3 for i in range(70)]),
        }
        result = fetcher.get_defensive_rotation(sector_data=sector_data)
        assert result is not None
        assert result.signal < 0  # defense leading = bearish


class TestMacroRegimeScore:
    """Test combined macro regime score."""

    def test_all_bullish_signals(self):
        fetcher = MacroSignalFetcher()
        # Steep yield curve, tightening credit, weak dollar, offense leading
        tnx = _make_df([5.0] * 70)
        irx = _make_df([2.0] * 70)
        hyg_vals = [80 + i * 0.3 for i in range(70)]
        hyg = _make_df(hyg_vals)
        lqd = _make_df([100.0] * 70)
        uup_vals = [28 - i * 0.15 for i in range(30)]
        uup = _make_df(uup_vals)
        sector_data = {
            'XLK': _make_df([100 + i * 0.5 for i in range(70)]),
            'XLY': _make_df([100 + i * 0.5 for i in range(70)]),
            'XLU': _make_df([100.0] * 70),
            'XLP': _make_df([100.0] * 70),
        }

        result = fetcher.get_macro_regime_score(
            tnx_df=tnx, irx_df=irx,
            hyg_df=hyg, lqd_df=lqd,
            uup_df=uup, sector_data=sector_data
        )
        assert result.score > 0.3
        assert result.bias == 'bullish'
        assert result.confidence > 0

    def test_no_data_returns_neutral(self):
        fetcher = MacroSignalFetcher()
        # Pass empty DataFrames for ALL signals to prevent live yfinance fetches
        empty = pd.DataFrame()
        result = fetcher.get_macro_regime_score(
            tnx_df=empty, irx_df=empty,
            hyg_df=empty, lqd_df=empty,
            uup_df=empty,
            sector_data={'XLU': empty, 'XLP': empty, 'XLK': empty, 'XLY': empty}
        )
        assert result.score == 0.0
        assert result.confidence == 0.0

    def test_score_capped(self):
        fetcher = MacroSignalFetcher()
        result = fetcher.get_macro_regime_score(
            tnx_df=_make_df([5.0] * 70),
            irx_df=_make_df([2.0] * 70)
        )
        assert -1.0 <= result.score <= 1.0

    def test_bias_property(self):
        r = MacroRegimeResult(score=0.5, confidence=0.8)
        assert r.bias == 'bullish'
        r2 = MacroRegimeResult(score=-0.5, confidence=0.8)
        assert r2.bias == 'bearish'
        r3 = MacroRegimeResult(score=0.0, confidence=0.8)
        assert r3.bias == 'neutral'
