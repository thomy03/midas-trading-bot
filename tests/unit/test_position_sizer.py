"""Tests for PositionSizer - V7 dynamic sizing."""
import pytest
from src.execution.position_sizer import PositionSizer, SizingResult


class TestBasicSizing:
    """Test basic position sizing."""

    def test_default_size(self):
        ps = PositionSizer(use_kelly=False, use_vol_adjust=False, use_confidence_scale=False)
        result = ps.calculate_size(capital=100000, price=100, regime='BULL')
        assert result.adjusted_size_pct == 0.10  # Default 10%, BULL regime mult=1.0

    def test_max_size_capped(self):
        ps = PositionSizer(base_size_pct=0.30, max_size_pct=0.20)
        result = ps.calculate_size(capital=100000, price=100)
        assert result.adjusted_size_pct <= 0.20

    def test_min_size_enforced(self):
        ps = PositionSizer(base_size_pct=0.01, min_size_pct=0.03)
        result = ps.calculate_size(capital=100000, price=100)
        assert result.adjusted_size_pct >= 0.03


class TestVolatilityAdjustment:
    """Test volatility-based sizing."""

    def test_high_volatility_reduces_size(self):
        ps = PositionSizer(use_kelly=False, use_confidence_scale=False)
        # High ATR = smaller position (target_risk 1.5% / ATR 5% = 0.3 multiplier)
        result = ps.calculate_size(capital=100000, price=100, atr_pct=0.05)
        assert result.adjusted_size_pct < 0.10

    def test_low_volatility_increases_size(self):
        ps = PositionSizer(use_kelly=False, use_confidence_scale=False)
        # Low ATR = larger position (target_risk 1.5% / ATR 0.5% = 3.0, clamped to 2.0)
        result = ps.calculate_size(capital=100000, price=100, atr_pct=0.005)
        assert result.adjusted_size_pct > 0.10

    def test_vol_adjustment_clamped(self):
        """Volatility adjustment is clamped between 0.5 and 2.0."""
        ps = PositionSizer(use_kelly=False, use_confidence_scale=False)
        # Extremely low ATR -> adjustment clamped to 2.0
        result = ps.calculate_size(capital=100000, price=100, atr_pct=0.001)
        assert result.vol_adjustment == 2.0


class TestConfidenceScaling:
    """Test confidence-based scaling."""

    def test_high_confidence_larger(self):
        ps = PositionSizer(use_kelly=False, use_vol_adjust=False)
        result_high = ps.calculate_size(capital=100000, price=100, confidence_score=90)
        result_low = ps.calculate_size(capital=100000, price=100, confidence_score=65)
        assert result_high.adjusted_size_pct > result_low.adjusted_size_pct

    def test_confidence_multipliers(self):
        ps = PositionSizer(use_kelly=False, use_vol_adjust=False)
        result = ps.calculate_size(capital=100000, price=100, confidence_score=95)
        assert result.confidence_multiplier == 1.5

    def test_low_confidence_reduces(self):
        ps = PositionSizer(use_kelly=False, use_vol_adjust=False)
        result = ps.calculate_size(capital=100000, price=100, confidence_score=60)
        assert result.confidence_multiplier == 0.6

    def test_medium_confidence_no_change(self):
        ps = PositionSizer(use_kelly=False, use_vol_adjust=False)
        result = ps.calculate_size(capital=100000, price=100, confidence_score=72)
        assert result.confidence_multiplier == 1.0


class TestKellyCriterion:
    """Test Kelly-based sizing."""

    def test_kelly_with_no_history(self):
        ps = PositionSizer()
        assert ps._kelly_size() == 0.0

    def test_kelly_with_insufficient_history(self):
        ps = PositionSizer()
        for _ in range(15):
            ps.add_trade_result(5.0)
        assert ps._kelly_size() == 0.0  # Need at least 20 trades

    def test_kelly_with_winning_history(self):
        ps = PositionSizer()
        # Add winning trades
        for _ in range(30):
            ps.add_trade_result(5.0)   # 5% win
        for _ in range(10):
            ps.add_trade_result(-3.0)  # 3% loss
        kelly = ps._kelly_size()
        assert kelly > 0

    def test_kelly_with_losing_history(self):
        ps = PositionSizer()
        for _ in range(30):
            ps.add_trade_result(-5.0)
        for _ in range(10):
            ps.add_trade_result(2.0)
        kelly = ps._kelly_size()
        assert kelly == 0.0  # Kelly should be 0 or negative -> clamped to 0

    def test_kelly_all_wins(self):
        ps = PositionSizer()
        for _ in range(25):
            ps.add_trade_result(5.0)
        # All wins, no losses -> returns 0.0 (no losses to compute ratio)
        assert ps._kelly_size() == 0.0


class TestRegimeAdjustment:
    """Test regime-based adjustments."""

    def test_bear_regime_reduces_size(self):
        ps = PositionSizer(use_kelly=False, use_vol_adjust=False, use_confidence_scale=False)
        result = ps.calculate_size(capital=100000, price=100, regime='BEAR')
        assert result.defensive_multiplier < 1.0
        assert result.defensive_multiplier == 0.6

    def test_bull_regime_full_size(self):
        ps = PositionSizer(use_kelly=False, use_vol_adjust=False, use_confidence_scale=False)
        result = ps.calculate_size(capital=100000, price=100, regime='BULL')
        assert result.defensive_multiplier == 1.0

    def test_volatile_regime_smallest(self):
        ps = PositionSizer(use_kelly=False, use_vol_adjust=False, use_confidence_scale=False)
        result = ps.calculate_size(capital=100000, price=100, regime='VOLATILE')
        assert result.defensive_multiplier == 0.5

    def test_range_regime_moderate(self):
        ps = PositionSizer(use_kelly=False, use_vol_adjust=False, use_confidence_scale=False)
        result = ps.calculate_size(capital=100000, price=100, regime='RANGE')
        assert result.defensive_multiplier == 0.8


class TestShareCalculation:
    """Test share count calculation."""

    def test_calculate_shares(self):
        ps = PositionSizer()
        result = SizingResult(
            base_size_pct=0.10, adjusted_size_pct=0.10,
            method='fixed', kelly_fraction=0, vol_adjustment=1,
            confidence_multiplier=1, defensive_multiplier=1, reason=''
        )
        shares = ps.calculate_shares(100000, 50.0, result)
        assert shares == 200  # 10000 / 50 = 200

    def test_zero_shares_on_expensive_stock(self):
        ps = PositionSizer()
        result = SizingResult(
            base_size_pct=0.03, adjusted_size_pct=0.03,
            method='fixed', kelly_fraction=0, vol_adjustment=1,
            confidence_multiplier=1, defensive_multiplier=1, reason=''
        )
        shares = ps.calculate_shares(1000, 5000.0, result)
        assert shares == 0  # 30 / 5000 = 0

    def test_shares_always_non_negative(self):
        ps = PositionSizer()
        result = SizingResult(
            base_size_pct=0.10, adjusted_size_pct=0.0,
            method='fixed', kelly_fraction=0, vol_adjustment=1,
            confidence_multiplier=1, defensive_multiplier=1, reason=''
        )
        shares = ps.calculate_shares(100000, 100.0, result)
        assert shares >= 0


class TestStats:
    """Test statistics reporting."""

    def test_get_stats(self):
        ps = PositionSizer()
        stats = ps.get_stats()
        assert 'total_recorded_trades' in stats
        assert 'current_kelly_fraction' in stats
        assert 'half_kelly' in stats
        assert stats['base_size_pct'] == 0.10
