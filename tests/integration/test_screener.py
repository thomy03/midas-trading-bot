"""
Integration tests for MarketScreener

Tests the complete screening pipeline:
- EMA detection + historical levels
- RSI breakout integration
- Confidence scoring
- Position sizing
- Alert generation
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.screening.screener import MarketScreener, validate_ema_at_breakout


class MockDataProvider:
    """Mock data provider for testing without API calls"""

    def __init__(self):
        self.call_count = 0

    def get_historical_data(self, symbol: str, period: str = '2y', interval: str = '1wk'):
        """Generate synthetic historical data"""
        self.call_count += 1

        # Determine number of periods based on interval
        if interval == '1wk':
            n = 104 if period == '2y' else 52
        else:  # daily
            n = 504 if period == '2y' else 252

        np.random.seed(hash(symbol) % 1000)  # Reproducible per symbol

        dates = pd.date_range(end=datetime.now(), periods=n, freq='W' if interval == '1wk' else 'D')

        # Create uptrend with oscillation
        base = np.linspace(100, 140, n)
        oscillation = 10 * np.sin(np.linspace(0, 8*np.pi, n))
        noise = np.random.normal(0, 2, n)
        close = base + oscillation + noise

        df = pd.DataFrame({
            'Open': close * (1 - np.random.uniform(0, 0.01, n)),
            'High': close * (1 + np.random.uniform(0, 0.02, n)),
            'Low': close * (1 - np.random.uniform(0, 0.02, n)),
            'Close': close,
            'Volume': np.random.uniform(1000000, 3000000, n)
        }, index=dates)

        return df


class TestValidateEmaAtBreakout:
    """Tests for validate_ema_at_breakout function"""

    @pytest.fixture
    def bullish_ema_df(self):
        """DataFrame with bullish EMA alignment"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Close': np.linspace(100, 150, 100),
            'EMA_24': np.linspace(95, 145, 100),  # Highest (shortest)
            'EMA_38': np.linspace(90, 140, 100),  # Middle
            'EMA_62': np.linspace(85, 135, 100),  # Lowest (longest)
        }, index=dates)
        return df

    @pytest.fixture
    def bearish_ema_df(self):
        """DataFrame with bearish EMA alignment"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'Close': np.linspace(150, 100, 100),
            'EMA_24': np.linspace(135, 85, 100),  # Lowest
            'EMA_38': np.linspace(140, 90, 100),  # Middle
            'EMA_62': np.linspace(145, 95, 100),  # Highest
        }, index=dates)
        return df

    @pytest.mark.integration
    def test_bullish_ema_validation(self, bullish_ema_df):
        """Test validation passes with bullish EMAs"""
        is_valid, conditions, reason = validate_ema_at_breakout(bullish_ema_df, 90)

        assert is_valid is True
        assert conditions >= 2
        assert 'bullish' in reason.lower()

    @pytest.mark.integration
    def test_bearish_ema_validation(self, bearish_ema_df):
        """Test validation fails with bearish EMAs"""
        is_valid, conditions, reason = validate_ema_at_breakout(bearish_ema_df, 90)

        assert is_valid is False
        assert conditions < 2
        assert 'not bullish' in reason.lower()

    @pytest.mark.integration
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()
        is_valid, conditions, reason = validate_ema_at_breakout(empty_df, 0)

        assert is_valid is False
        assert 'no data' in reason.lower()

    @pytest.mark.integration
    def test_invalid_index_handling(self, bullish_ema_df):
        """Test handling of invalid breakout index"""
        # Index out of bounds
        is_valid, conditions, reason = validate_ema_at_breakout(bullish_ema_df, 999)

        assert is_valid is False
        assert 'invalid' in reason.lower()


class TestMarketScreenerIntegration:
    """Integration tests for MarketScreener class"""

    @pytest.fixture
    def screener(self):
        """Create screener instance with mocked dependencies"""
        with patch('src.screening.screener.market_data_fetcher') as mock_data, \
             patch('src.screening.screener.db_manager') as mock_db:

            mock_data_provider = MockDataProvider()
            mock_data.get_historical_data = mock_data_provider.get_historical_data

            screener = MarketScreener(
                use_enhanced_detector=True,
                precision_mode='medium',
                total_capital=10000
            )
            screener.market_data = mock_data_provider

            # Mock yfinance for market cap
            with patch('yfinance.Ticker') as mock_yf:
                mock_ticker = MagicMock()
                mock_ticker.info = {'marketCap': 100e9}  # $100B
                mock_yf.return_value = mock_ticker
                yield screener

    @pytest.mark.integration
    def test_screener_initialization(self):
        """Test screener initializes with correct settings"""
        with patch('src.screening.screener.market_data_fetcher'), \
             patch('src.screening.screener.db_manager'):

            screener = MarketScreener(
                use_enhanced_detector=True,
                precision_mode='high',
                total_capital=15000
            )

            assert screener.use_enhanced_detector is True
            assert screener.precision_mode == 'high'
            assert screener.position_sizer.get_total_capital() == 15000

    @pytest.mark.integration
    def test_screen_single_stock_returns_dict_or_none(self, screener):
        """Test single stock screening returns proper type"""
        result = screener.screen_single_stock('TEST', 'Test Company')

        # Result should be dict (alert) or None (no signal)
        assert result is None or isinstance(result, dict)

    @pytest.mark.integration
    def test_alert_structure_when_generated(self, screener):
        """Test alert has required fields when generated"""
        # Screen multiple symbols to increase chance of getting an alert
        result = None
        for symbol in ['AAPL', 'MSFT', 'GOOGL', 'NVDA']:
            result = screener.screen_single_stock(symbol, f'{symbol} Inc')
            if result:
                break

        if result:
            # Check core fields
            assert 'symbol' in result
            assert 'current_price' in result
            assert 'recommendation' in result

            # Check confidence scoring fields
            assert 'confidence_score' in result
            assert 'confidence_signal' in result

            # Check position sizing fields
            assert 'position_shares' in result or 'stop_loss' in result

    @pytest.mark.integration
    def test_recommendation_values_are_valid(self, screener):
        """Test recommendations are valid values"""
        valid_recommendations = ['STRONG_BUY', 'BUY', 'WATCH', 'OBSERVE', 'NO_SIGNAL']

        # Generate some alerts
        for symbol in ['TEST1', 'TEST2', 'TEST3']:
            result = screener.screen_single_stock(symbol, f'{symbol} Inc')
            if result and 'recommendation' in result:
                assert result['recommendation'] in valid_recommendations

    @pytest.mark.integration
    def test_confidence_score_range(self, screener):
        """Test confidence score is in valid range"""
        for symbol in ['AAPL', 'MSFT']:
            result = screener.screen_single_stock(symbol, f'{symbol} Inc')
            if result and 'confidence_score' in result:
                assert 0 <= result['confidence_score'] <= 100

    @pytest.mark.integration
    def test_ema_validation_at_breakout_included(self, screener):
        """Test EMA validation at breakout is performed"""
        result = screener.screen_single_stock('TEST', 'Test Inc')

        if result and result.get('has_rsi_breakout'):
            # Should have EMA validation fields
            assert 'ema_valid_at_breakout' in result
            assert 'ema_conditions_at_breakout' in result

    @pytest.mark.integration
    def test_volume_confirmation_included(self, screener):
        """Test volume confirmation is included for breakouts"""
        result = screener.screen_single_stock('AAPL', 'Apple Inc')

        if result and result.get('has_rsi_breakout'):
            assert 'volume_ratio' in result
            assert 'volume_confirmed' in result


class TestScreenerAlertCreation:
    """Tests for alert creation methods"""

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing"""
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start='2023-01-01', periods=n, freq='D')

        close = np.linspace(100, 130, n) + np.random.normal(0, 2, n)

        df = pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.02,
            'Low': close * 0.98,
            'Close': close,
            'Volume': np.random.uniform(1000000, 2000000, n),
            'EMA_24': close * 0.97,
            'EMA_38': close * 0.95,
            'EMA_62': close * 0.93
        }, index=dates)

        return df

    @pytest.fixture
    def sample_level(self):
        """Sample historical level for testing"""
        return {
            'level': 120.0,
            'distance_pct': 3.5,
            'is_near': True,
            'crossover_info': {
                'date': pd.Timestamp('2023-06-15'),
                'type': 'bullish',
                'fast_ema': 24,
                'slow_ema': 38,
                'price': 118.5,
                'age_in_periods': 10
            }
        }

    @pytest.mark.integration
    def test_create_alert_historical_level(self, sample_df, sample_level):
        """Test _create_alert_historical_level method"""
        with patch('src.screening.screener.market_data_fetcher'), \
             patch('src.screening.screener.db_manager'), \
             patch('yfinance.Ticker') as mock_yf:

            mock_ticker = MagicMock()
            mock_ticker.info = {'marketCap': 50e9}
            mock_yf.return_value = mock_ticker

            screener = MarketScreener(total_capital=10000)

            # Create a mock RSI result
            mock_rsi_result = MagicMock()
            mock_rsi_result.has_rsi_breakout = True
            mock_rsi_result.has_rsi_trendline = True
            mock_rsi_result.signal = 'STRONG_BUY'
            mock_rsi_result.rsi_breakout = MagicMock(
                strength='STRONG',
                age_in_periods=2,
                index=95,
                date=pd.Timestamp('2023-04-01'),
                rsi_value=62.5,
                volume_ratio=1.5,
                volume_confirmed=True
            )
            mock_rsi_result.rsi_trendline = MagicMock(
                r_squared=0.85,
                peak_indices=[20, 40, 60]
            )
            mock_rsi_result.ema_valid_at_breakout = True
            mock_rsi_result.ema_conditions_at_breakout = 3
            mock_rsi_result.ema_validation_reason = "EMAs bullish"

            alert = screener._create_alert_historical_level(
                symbol='TEST',
                company_name='Test Company',
                historical_level=sample_level,
                rsi_timeframe='weekly',
                rsi_result=mock_rsi_result,
                df_weekly=sample_df
            )

            # Verify alert structure
            assert alert['symbol'] == 'TEST'
            assert alert['company_name'] == 'Test Company'
            assert 'confidence_score' in alert
            assert 'position_shares' in alert
            assert alert['has_rsi_breakout'] is True
            assert alert['volume_ratio'] == 1.5


class TestScreenerPortfolioManagement:
    """Tests for portfolio management integration"""

    @pytest.mark.integration
    def test_get_available_capital(self, tmp_path):
        """Test available capital calculation"""
        import tempfile
        import os

        # Use temp file for portfolio to avoid state pollution
        temp_portfolio = str(tmp_path / "test_portfolio.json")

        with patch('src.screening.screener.market_data_fetcher'), \
             patch('src.screening.screener.db_manager'):

            # Create fresh screener with temp portfolio file
            screener = MarketScreener(
                total_capital=10000,
                portfolio_file=temp_portfolio
            )
            capital = screener.get_available_capital()
            assert capital == 10000  # Initial capital, no positions

    @pytest.mark.integration
    def test_portfolio_summary(self):
        """Test portfolio summary"""
        with patch('src.screening.screener.market_data_fetcher'), \
             patch('src.screening.screener.db_manager'):

            screener = MarketScreener(total_capital=10000)
            summary = screener.get_portfolio_summary()

            assert 'total_capital' in summary
            assert 'available_capital' in summary or 'available' in str(summary).lower()

    @pytest.mark.integration
    def test_update_capital(self):
        """Test capital update"""
        with patch('src.screening.screener.market_data_fetcher'), \
             patch('src.screening.screener.db_manager'):

            screener = MarketScreener(total_capital=10000)
            screener.update_capital(15000)

            capital = screener.get_available_capital()
            assert capital == 15000


class TestScreenerMultipleStocks:
    """Tests for multi-stock screening"""

    @pytest.fixture
    def screener(self):
        """Screener with mocked dependencies"""
        with patch('src.screening.screener.market_data_fetcher') as mock_data, \
             patch('src.screening.screener.db_manager'):

            mock_provider = MockDataProvider()
            mock_data.get_historical_data = mock_provider.get_historical_data

            screener = MarketScreener(total_capital=10000)
            screener.market_data = mock_provider

            return screener

    @pytest.mark.integration
    def test_screen_multiple_stocks_returns_list(self, screener):
        """Test multiple stock screening returns list"""
        stocks = [
            {'symbol': 'AAPL', 'name': 'Apple'},
            {'symbol': 'MSFT', 'name': 'Microsoft'},
            {'symbol': 'GOOGL', 'name': 'Alphabet'}
        ]

        # Mock yfinance for market cap
        with patch('yfinance.Ticker') as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = {'marketCap': 100e9}
            mock_yf.return_value = mock_ticker

            # Patch fundamental scorer to avoid API calls
            with patch('src.screening.screener.fundamental_scorer') as mock_scorer:
                mock_scorer.score_strong_buys = lambda x: x  # Pass through

                alerts = screener.screen_multiple_stocks(stocks)

        assert isinstance(alerts, list)

    @pytest.mark.integration
    def test_empty_stock_list_handling(self, screener):
        """Test handling of empty stock list"""
        with patch('src.screening.screener.fundamental_scorer') as mock_scorer:
            mock_scorer.score_strong_buys = lambda x: x

            alerts = screener.screen_multiple_stocks([])

        assert alerts == []


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
