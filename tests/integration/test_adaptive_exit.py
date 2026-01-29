"""
Tests for Adaptive Exit Indicator

Tests the Chandelier Exit, Bollinger Squeeze, and Volume Exhaustion
components of the adaptive exit system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.indicators.adaptive_exit import AdaptiveExitIndicator, ExitSignal, check_adaptive_exit


def generate_mock_ohlcv(
    start_date: str = '2023-01-01',
    periods: int = 100,
    base_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.001
) -> pd.DataFrame:
    """Generate mock OHLCV data for testing"""
    np.random.seed(42)

    dates = pd.date_range(start=start_date, periods=periods, freq='D')

    # Generate price path with trend and noise
    returns = np.random.normal(trend, volatility, periods)
    prices = base_price * np.cumprod(1 + returns)

    # Generate OHLC from prices
    high = prices * (1 + np.random.uniform(0, 0.02, periods))
    low = prices * (1 - np.random.uniform(0, 0.02, periods))
    open_price = low + np.random.uniform(0.3, 0.7, periods) * (high - low)
    close = low + np.random.uniform(0.3, 0.7, periods) * (high - low)

    # Generate volume
    volume = np.random.uniform(1000000, 5000000, periods)

    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)

    return df


class TestAdaptiveExitIndicator:
    """Test AdaptiveExitIndicator class"""

    @pytest.fixture
    def indicator(self):
        """Create default indicator instance"""
        return AdaptiveExitIndicator()

    @pytest.fixture
    def mock_data(self):
        """Create mock OHLCV data"""
        return generate_mock_ohlcv(periods=150)

    def test_initialization(self):
        """Test indicator initialization with default params"""
        indicator = AdaptiveExitIndicator()

        assert indicator.atr_period == 22
        assert indicator.atr_multiplier == 3.0
        assert indicator.bb_period == 20
        assert indicator.volume_lookback == 20

    def test_initialization_custom_params(self):
        """Test indicator initialization with custom params"""
        indicator = AdaptiveExitIndicator(
            atr_period=14,
            atr_multiplier=2.5,
            bb_period=30
        )

        assert indicator.atr_period == 14
        assert indicator.atr_multiplier == 2.5
        assert indicator.bb_period == 30

    def test_calculate_atr(self, indicator, mock_data):
        """Test ATR calculation"""
        atr = indicator._calculate_atr(mock_data)

        assert isinstance(atr, pd.Series)
        assert len(atr) == len(mock_data)
        # ATR should be positive where valid
        assert (atr.dropna() > 0).all()

    def test_calculate_chandelier_exit(self, indicator, mock_data):
        """Test Chandelier Exit calculation"""
        chandelier = indicator.calculate_chandelier_exit(mock_data)

        assert isinstance(chandelier, pd.Series)
        assert len(chandelier) == len(mock_data)

        # Chandelier should be below highest high
        highest_high = mock_data['High'].rolling(22).max()
        valid_idx = ~(chandelier.isna() | highest_high.isna())
        assert (chandelier[valid_idx] <= highest_high[valid_idx]).all()

    def test_detect_bollinger_squeeze(self, indicator, mock_data):
        """Test Bollinger Squeeze detection"""
        squeeze = indicator.detect_bollinger_squeeze(mock_data)

        assert isinstance(squeeze, pd.Series)
        assert squeeze.dtype == bool or squeeze.dtype == np.bool_

    def test_calculate_ad_line(self, indicator, mock_data):
        """Test Accumulation/Distribution line calculation"""
        ad = indicator.calculate_ad_line(mock_data)

        assert isinstance(ad, pd.Series)
        assert len(ad) == len(mock_data)

    def test_detect_selling_exhaustion(self, indicator, mock_data):
        """Test selling exhaustion detection"""
        idx = 80  # Needs sufficient lookback

        is_exhausted, vol_ratio, ad_trend = indicator.detect_selling_exhaustion(
            mock_data, idx
        )

        assert isinstance(is_exhausted, (bool, np.bool_))
        assert isinstance(vol_ratio, (float, np.floating))
        assert ad_trend in ('rising', 'falling', 'neutral')

    def test_should_exit_insufficient_data(self, indicator, mock_data):
        """Test exit check with insufficient data"""
        signal = indicator.should_exit(
            df=mock_data,
            idx=5,  # Too early
            position_stop_loss=80.0
        )

        assert isinstance(signal, ExitSignal)
        assert signal.should_exit is False
        assert signal.reason == 'insufficient_data'

    def test_should_exit_stop_loss(self, indicator, mock_data):
        """Test stop-loss exit"""
        idx = 80
        # Set stop-loss above current price to trigger exit
        current_low = mock_data['Low'].iloc[idx]
        stop_loss = current_low + 10  # Above current price

        signal = indicator.should_exit(
            df=mock_data,
            idx=idx,
            position_stop_loss=stop_loss
        )

        assert signal.should_exit is True
        assert signal.reason == 'stop_loss'

    def test_should_exit_hold(self, indicator, mock_data):
        """Test hold signal (no exit)"""
        idx = 80
        # Set stop-loss well below current price
        current_price = mock_data['Close'].iloc[idx]
        stop_loss = current_price * 0.5  # 50% below

        signal = indicator.should_exit(
            df=mock_data,
            idx=idx,
            position_stop_loss=stop_loss
        )

        # Should hold unless other exit conditions triggered
        assert isinstance(signal, ExitSignal)
        # Check that we get valid metadata even on hold
        assert signal.chandelier_level > 0

    def test_get_exit_levels(self, indicator, mock_data):
        """Test getting current exit levels"""
        idx = 80
        levels = indicator.get_exit_levels(mock_data, idx)

        assert 'chandelier_level' in levels
        assert 'squeeze_active' in levels
        assert 'volume_ratio' in levels
        assert 'ad_trend' in levels

        assert isinstance(levels['chandelier_level'], float)
        assert isinstance(levels['squeeze_active'], (bool, np.bool_))


class TestCheckAdaptiveExitWrapper:
    """Test the convenience wrapper function"""

    def test_check_adaptive_exit(self):
        """Test convenience wrapper function"""
        df = generate_mock_ohlcv(periods=100)
        idx = 80
        stop_loss = 50.0  # Well below price

        should_exit, reason = check_adaptive_exit(
            df=df,
            idx=idx,
            stop_loss=stop_loss
        )

        assert isinstance(should_exit, bool)
        assert isinstance(reason, str)


class TestChandelierExitBehavior:
    """Test specific Chandelier Exit behaviors"""

    def test_chandelier_adapts_to_volatility(self):
        """Test that Chandelier Exit adapts to volatility via ATR"""
        indicator = AdaptiveExitIndicator(atr_period=22, atr_multiplier=3.0)

        # Generate controlled data with different volatility
        np.random.seed(42)
        periods = 100
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='D')

        # Low volatility data (small high-low range)
        base_price = 100.0
        low_vol_df = pd.DataFrame({
            'Open': [base_price] * periods,
            'High': [base_price * 1.005] * periods,  # 0.5% high
            'Low': [base_price * 0.995] * periods,   # 0.5% low
            'Close': [base_price] * periods,
            'Volume': [1000000] * periods
        }, index=dates)

        # High volatility data (large high-low range)
        high_vol_df = pd.DataFrame({
            'Open': [base_price] * periods,
            'High': [base_price * 1.05] * periods,   # 5% high
            'Low': [base_price * 0.95] * periods,    # 5% low
            'Close': [base_price] * periods,
            'Volume': [1000000] * periods
        }, index=dates)

        # Calculate ATR for both
        atr_low = indicator._calculate_atr(low_vol_df)
        atr_high = indicator._calculate_atr(high_vol_df)

        # High volatility should have larger ATR
        idx = 50
        assert atr_high.iloc[idx] > atr_low.iloc[idx], \
            f"High vol ATR ({atr_high.iloc[idx]}) should be > low vol ATR ({atr_low.iloc[idx]})"


class TestIntegrationWithBacktester:
    """Test integration with backtester system"""

    def test_exit_signal_dataclass(self):
        """Test ExitSignal dataclass"""
        signal = ExitSignal(
            should_exit=True,
            reason='chandelier_exit',
            price=100.0,
            chandelier_level=98.0,
            volume_ratio=1.5,
            squeeze_active=False,
            ad_trend='rising'
        )

        assert signal.should_exit is True
        assert signal.reason == 'chandelier_exit'
        assert signal.price == 100.0
        assert signal.chandelier_level == 98.0

    def test_multiple_exit_priorities(self):
        """Test that stop-loss takes priority over other exits"""
        indicator = AdaptiveExitIndicator()

        # Generate data where price is below stop-loss
        df = generate_mock_ohlcv(periods=100)
        idx = 80

        # Set very high stop-loss
        stop_loss = df['Close'].iloc[idx] * 2

        signal = indicator.should_exit(df, idx, stop_loss)

        # Stop-loss should be the reason (highest priority)
        assert signal.should_exit is True
        assert signal.reason == 'stop_loss'


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_dataframe(self):
        """Test with empty dataframe"""
        indicator = AdaptiveExitIndicator()
        df = pd.DataFrame()

        # Should handle gracefully
        try:
            signal = indicator.should_exit(df, 0, 100.0)
            # Should return insufficient_data or handle error
        except (IndexError, KeyError):
            pass  # Expected for empty data

    def test_zero_volume(self):
        """Test handling of zero volume"""
        indicator = AdaptiveExitIndicator()
        df = generate_mock_ohlcv(periods=100)
        df['Volume'] = 0  # Set all volume to zero

        idx = 80
        # Should not crash with zero volume
        is_exhausted, vol_ratio, ad_trend = indicator.detect_selling_exhaustion(df, idx)

        # Volume ratio should handle division by zero
        assert not np.isnan(vol_ratio) or vol_ratio == 1.0

    def test_identical_high_low(self):
        """Test handling of identical high/low (no range)"""
        indicator = AdaptiveExitIndicator()
        df = generate_mock_ohlcv(periods=100)
        # Set High = Low for some rows
        df.loc[df.index[50:60], 'High'] = df.loc[df.index[50:60], 'Low']

        # Should not crash
        ad = indicator.calculate_ad_line(df)
        assert len(ad) == len(df)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
