"""
Integration Tests for Backtesting Module

Tests the complete backtesting pipeline including:
- BacktestConfig validation
- Trade simulation
- Metrics calculation
- Equity curve building
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.backtesting.backtester import Backtester, BacktestConfig, BacktestResult
from src.backtesting.metrics import (
    Trade, PerformanceMetrics, calculate_metrics, format_metrics_report,
    _calculate_drawdown, _calculate_sharpe, _calculate_sortino
)


# ====================
# Test Data Generators
# ====================

def generate_mock_ohlcv(days=252, start_price=100, volatility=0.02, trend=0.0001):
    """Generate realistic OHLCV data"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Generate price with random walk + trend
    returns = np.random.normal(trend, volatility, days)
    price = start_price * np.cumprod(1 + returns)

    # Generate OHLC from price
    high = price * (1 + np.abs(np.random.normal(0, 0.01, days)))
    low = price * (1 - np.abs(np.random.normal(0, 0.01, days)))
    open_price = price * (1 + np.random.normal(0, 0.005, days))

    # Generate volume
    volume = np.random.uniform(1000000, 5000000, days)

    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': price,
        'Volume': volume
    }, index=dates)

    return df


def create_mock_trades(count=10, win_rate=0.6, avg_profit=5.0, avg_loss=-3.0):
    """Create mock trades for testing"""
    np.random.seed(42)
    trades = []
    base_date = datetime.now() - timedelta(days=365)

    for i in range(count):
        is_winner = np.random.random() < win_rate
        pnl_pct = np.random.normal(avg_profit if is_winner else avg_loss, 1.0)

        entry_date = base_date + timedelta(days=i * 10)
        exit_date = entry_date + timedelta(days=np.random.randint(5, 30))
        entry_price = 100 + np.random.uniform(-10, 10)
        exit_price = entry_price * (1 + pnl_pct / 100)
        shares = 10

        trades.append(Trade(
            symbol=f'TEST{i}',
            entry_date=pd.Timestamp(entry_date),
            entry_price=entry_price,
            exit_date=pd.Timestamp(exit_date),
            exit_price=exit_price,
            shares=shares,
            stop_loss=entry_price * 0.95,
            profit_loss=(exit_price - entry_price) * shares,
            profit_loss_pct=pnl_pct,
            exit_reason='take_profit' if is_winner else 'stop_loss',
            signal_strength='STRONG_BUY' if is_winner else 'BUY',
            confidence_score=70 if is_winner else 55,
            volume_ratio=1.2
        ))

    return trades


# ====================
# BacktestConfig Tests
# ====================

class TestBacktestConfig:
    """Test BacktestConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = BacktestConfig()

        assert config.initial_capital == 10000
        assert config.max_positions == 20  # Updated from 5
        assert config.position_size_pct == 0.10  # Updated from 0.20
        assert config.min_confidence_score == 55
        assert config.require_volume_confirmation == True
        assert config.min_volume_ratio == 1.0
        assert config.use_trailing_stop == False
        assert config.trailing_stop_pct == 0.05
        assert config.take_profit_pct == 0.0  # Updated: disabled with adaptive exit
        assert config.max_hold_days == 120  # Updated from 60
        assert config.use_enhanced_detector == True
        assert config.precision_mode == 'medium'
        assert config.timeframe == 'weekly'

    def test_custom_config(self):
        """Test custom configuration"""
        config = BacktestConfig(
            initial_capital=50000,
            max_positions=10,
            position_size_pct=0.10,
            min_confidence_score=65,
            use_trailing_stop=True,
            trailing_stop_pct=0.08,
            timeframe='daily'
        )

        assert config.initial_capital == 50000
        assert config.max_positions == 10
        assert config.position_size_pct == 0.10
        assert config.min_confidence_score == 65
        assert config.use_trailing_stop == True
        assert config.trailing_stop_pct == 0.08
        assert config.timeframe == 'daily'


# ====================
# Trade Dataclass Tests
# ====================

class TestTrade:
    """Test Trade dataclass"""

    def test_trade_creation(self):
        """Test creating a trade"""
        trade = Trade(
            symbol='AAPL',
            entry_date=pd.Timestamp('2024-01-01'),
            entry_price=100.0,
            exit_date=pd.Timestamp('2024-01-15'),
            exit_price=110.0,
            shares=10,
            stop_loss=95.0,
            profit_loss=100.0,
            profit_loss_pct=10.0,
            exit_reason='take_profit',
            signal_strength='STRONG_BUY',
            confidence_score=75.0,
            volume_ratio=1.5
        )

        assert trade.symbol == 'AAPL'
        assert trade.entry_price == 100.0
        assert trade.exit_price == 110.0
        assert trade.shares == 10
        assert trade.profit_loss == 100.0
        assert trade.profit_loss_pct == 10.0
        assert trade.exit_reason == 'take_profit'

    def test_losing_trade(self):
        """Test creating a losing trade"""
        trade = Trade(
            symbol='TSLA',
            entry_date=pd.Timestamp('2024-01-01'),
            entry_price=200.0,
            exit_date=pd.Timestamp('2024-01-10'),
            exit_price=180.0,
            shares=5,
            stop_loss=185.0,
            profit_loss=-100.0,
            profit_loss_pct=-10.0,
            exit_reason='stop_loss',
            signal_strength='BUY',
            confidence_score=58.0,
            volume_ratio=1.1
        )

        assert trade.profit_loss == -100.0
        assert trade.profit_loss_pct == -10.0
        assert trade.exit_reason == 'stop_loss'


# ====================
# Metrics Calculation Tests
# ====================

class TestMetricsCalculation:
    """Test performance metrics calculation"""

    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation"""
        trades = create_mock_trades(count=20, win_rate=0.6)
        metrics = calculate_metrics(trades, initial_capital=10000)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_trades == 20
        assert metrics.win_rate >= 0 and metrics.win_rate <= 100
        assert metrics.initial_capital == 10000

    def test_calculate_metrics_empty(self):
        """Test metrics with no trades"""
        metrics = calculate_metrics([], initial_capital=10000)

        assert metrics.total_trades == 0
        assert metrics.win_rate == 0
        assert metrics.total_return == 0
        assert metrics.profit_factor == 0
        assert metrics.initial_capital == 10000
        assert metrics.final_capital == 10000

    def test_calculate_metrics_all_winners(self):
        """Test metrics with all winning trades"""
        trades = create_mock_trades(count=10, win_rate=1.0, avg_profit=10.0)
        metrics = calculate_metrics(trades, initial_capital=10000)

        assert metrics.winning_trades == metrics.total_trades
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 100.0
        assert metrics.total_profit > 0
        assert metrics.profit_factor == float('inf')

    def test_calculate_metrics_all_losers(self):
        """Test metrics with all losing trades"""
        trades = create_mock_trades(count=10, win_rate=0.0, avg_loss=-5.0)
        metrics = calculate_metrics(trades, initial_capital=10000)

        assert metrics.winning_trades == 0
        assert metrics.losing_trades == metrics.total_trades
        assert metrics.win_rate == 0.0
        assert metrics.total_profit < 0

    def test_avg_win_loss_calculation(self):
        """Test average win/loss calculation"""
        trades = create_mock_trades(count=20, win_rate=0.5, avg_profit=8.0, avg_loss=-4.0)
        metrics = calculate_metrics(trades, initial_capital=10000)

        assert metrics.avg_win > 0
        assert metrics.avg_loss < 0

    def test_best_worst_trade(self):
        """Test best/worst trade identification"""
        trades = create_mock_trades(count=10)
        metrics = calculate_metrics(trades, initial_capital=10000)

        # Best should be positive (or at least >= worst)
        assert metrics.best_trade >= metrics.worst_trade


class TestDrawdownCalculation:
    """Test drawdown calculation"""

    def test_no_drawdown(self):
        """Test with continuously winning trades (no drawdown)"""
        trades = [
            Trade(
                symbol='TEST',
                entry_date=pd.Timestamp('2024-01-01'),
                entry_price=100.0,
                exit_date=pd.Timestamp(f'2024-01-{10+i}'),
                exit_price=100.0 + (i+1)*5,
                shares=10,
                stop_loss=95.0,
                profit_loss=50.0 * (i+1),
                profit_loss_pct=5.0 * (i+1),
                exit_reason='take_profit',
                signal_strength='STRONG_BUY',
                confidence_score=75.0,
                volume_ratio=1.2
            )
            for i in range(5)
        ]

        max_dd, dd_duration = _calculate_drawdown(trades, 10000)

        assert max_dd == 0  # No drawdown with all winners

    def test_drawdown_with_losses(self):
        """Test drawdown with losses"""
        trades = create_mock_trades(count=20, win_rate=0.4)
        max_dd, dd_duration = _calculate_drawdown(trades, 10000)

        assert max_dd >= 0  # Drawdown should be non-negative
        assert dd_duration >= 0


class TestSharpeRatio:
    """Test Sharpe ratio calculation"""

    def test_sharpe_positive_returns(self):
        """Test Sharpe with positive returns"""
        daily_returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        sharpe = _calculate_sharpe(daily_returns)

        assert isinstance(sharpe, float)

    def test_sharpe_with_none(self):
        """Test Sharpe with None returns"""
        sharpe = _calculate_sharpe(None)
        assert sharpe == 0

    def test_sharpe_empty_returns(self):
        """Test Sharpe with empty returns"""
        sharpe = _calculate_sharpe(pd.Series([]))
        assert sharpe == 0


class TestSortinoRatio:
    """Test Sortino ratio calculation"""

    def test_sortino_basic(self):
        """Test Sortino with mixed returns"""
        daily_returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        sortino = _calculate_sortino(daily_returns)

        assert isinstance(sortino, float)

    def test_sortino_with_none(self):
        """Test Sortino with None returns"""
        sortino = _calculate_sortino(None)
        assert sortino == 0


# ====================
# Backtester Class Tests
# ====================

class TestBacktester:
    """Test Backtester class"""

    def test_backtester_initialization_default(self):
        """Test default initialization"""
        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester()

            assert backtester.config is not None
            assert backtester.config.initial_capital == 10000

    def test_backtester_initialization_custom(self):
        """Test custom initialization"""
        config = BacktestConfig(
            initial_capital=25000,
            max_positions=3
        )

        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester(config)

            assert backtester.config.initial_capital == 25000
            assert backtester.config.max_positions == 3

    def test_validate_signal_min_confidence(self):
        """Test signal validation with confidence score"""
        config = BacktestConfig(min_confidence_score=60)

        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester(config)

            # Signal below threshold
            signal_low = {
                'confidence_score': 55,
                'volume_ratio': 1.5
            }
            assert backtester._validate_signal(signal_low) == False

            # Signal above threshold
            signal_high = {
                'confidence_score': 65,
                'volume_ratio': 1.5
            }
            assert backtester._validate_signal(signal_high) == True

    def test_validate_signal_volume_confirmation(self):
        """Test signal validation with volume confirmation"""
        config = BacktestConfig(
            min_confidence_score=50,
            require_volume_confirmation=True,
            min_volume_ratio=1.2
        )

        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester(config)

            # Low volume
            signal_low_vol = {
                'confidence_score': 60,
                'volume_ratio': 0.8
            }
            assert backtester._validate_signal(signal_low_vol) == False

            # High volume
            signal_high_vol = {
                'confidence_score': 60,
                'volume_ratio': 1.5
            }
            assert backtester._validate_signal(signal_high_vol) == True


class TestBacktesterEquityCurve:
    """Test equity curve building"""

    def test_build_equity_curve_empty_trades(self):
        """Test equity curve with no trades"""
        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester()

            start = pd.Timestamp('2024-01-01')
            end = pd.Timestamp('2024-12-31')

            equity_curve = backtester._build_equity_curve([], start, end)

            assert isinstance(equity_curve, pd.Series)
            assert len(equity_curve) > 0
            # All values should be initial capital
            assert all(v == 10000 for v in equity_curve.values)

    def test_build_equity_curve_with_trades(self):
        """Test equity curve with trades"""
        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester()

            trades = create_mock_trades(count=5)
            start = trades[0].entry_date
            end = trades[-1].exit_date

            equity_curve = backtester._build_equity_curve(trades, start, end)

            assert isinstance(equity_curve, pd.Series)
            assert len(equity_curve) > 0


# ====================
# BacktestResult Tests
# ====================

class TestBacktestResult:
    """Test BacktestResult dataclass"""

    def test_result_str_format(self):
        """Test string representation of result"""
        trades = create_mock_trades(count=5)
        metrics = calculate_metrics(trades, initial_capital=10000)
        config = BacktestConfig()

        result = BacktestResult(
            config=config,
            metrics=metrics,
            trades=trades,
            equity_curve=pd.Series([10000, 10500, 11000]),
            signals_detected=10,
            signals_taken=5
        )

        # __str__ should return formatted report
        report = str(result)
        assert 'BACKTEST PERFORMANCE REPORT' in report
        assert 'Total Trades' in report


# ====================
# Report Formatting Tests
# ====================

class TestReportFormatting:
    """Test report formatting"""

    def test_format_metrics_report_basic(self):
        """Test basic report formatting"""
        trades = create_mock_trades(count=10)
        metrics = calculate_metrics(trades, initial_capital=10000)

        report = format_metrics_report(metrics)

        assert isinstance(report, str)
        assert 'BACKTEST PERFORMANCE REPORT' in report
        assert 'Initial Capital' in report
        assert 'Final Capital' in report
        assert 'Winning Trades' in report  # Win rate is shown in Winning Trades line
        assert 'Profit Factor' in report
        assert 'Max Drawdown' in report

    def test_format_metrics_report_empty(self):
        """Test report with empty metrics"""
        metrics = calculate_metrics([], initial_capital=10000)

        report = format_metrics_report(metrics)

        assert isinstance(report, str)
        assert 'Total Trades:     0' in report


# ====================
# Integration Tests
# ====================

class TestBacktestIntegration:
    """Integration tests for full backtest workflow"""

    def test_full_backtest_workflow_mock(self):
        """Test full backtest with mocked data provider"""
        config = BacktestConfig(
            initial_capital=10000,
            min_confidence_score=50,
            require_volume_confirmation=False
        )

        # Create mock data provider
        mock_data_provider = Mock()
        mock_data_provider.get_historical_data.return_value = generate_mock_ohlcv(days=500)

        with patch.dict('sys.modules', {
            'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()
        }):
            backtester = Backtester(config)

            # Mock RSI analyzer to not find signals
            backtester.rsi_analyzer.analyze = Mock(return_value=None)

            result = backtester.run(
                symbols=['TEST1', 'TEST2'],
                start_date='2024-01-01',
                end_date='2024-12-01',
                data_provider=mock_data_provider
            )

            assert isinstance(result, BacktestResult)
            assert result.metrics is not None
            assert result.config == config

    def test_backtest_with_multiple_symbols(self):
        """Test backtest handles multiple symbols"""
        # Disable daily fallback to have predictable call count
        config = BacktestConfig(initial_capital=10000, use_daily_fallback=False)

        mock_data_provider = Mock()
        mock_data_provider.get_historical_data.return_value = generate_mock_ohlcv(days=500)

        with patch.dict('sys.modules', {
            'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()
        }):
            backtester = Backtester(config)
            backtester.rsi_analyzer.analyze = Mock(return_value=None)

            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

            result = backtester.run(
                symbols=symbols,
                start_date='2024-01-01',
                end_date='2024-12-01',
                data_provider=mock_data_provider
            )

            assert result.metrics is not None
            # Data provider should be called for each symbol
            assert mock_data_provider.get_historical_data.call_count == len(symbols)


class TestExitConditions:
    """Test exit condition logic"""

    def test_check_exit_stop_loss(self):
        """Test stop-loss exit"""
        config = BacktestConfig()

        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester(config)

            position = {
                'entry_price': 100.0,
                'stop_loss': 95.0,
                'highest_price': 100.0
            }

            # Create df where current low is below stop-loss
            df = pd.DataFrame({
                'Close': [100.0, 99.0, 94.0],
                'High': [101.0, 100.0, 95.0],
                'Low': [99.0, 98.0, 93.0]  # Third bar low triggers stop
            }, index=pd.date_range('2024-01-01', periods=3))

            should_exit, reason = backtester._check_exit(position, df, 2)

            assert should_exit == True
            assert reason == 'stop_loss'

    def test_check_exit_take_profit(self):
        """Test take-profit exit"""
        config = BacktestConfig(take_profit_pct=0.10)  # 10% take profit

        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester(config)

            position = {
                'entry_price': 100.0,
                'stop_loss': 90.0,
                'highest_price': 110.0
            }

            # Create df where current price is above take profit
            df = pd.DataFrame({
                'Close': [100.0, 105.0, 112.0],  # 12% profit
                'High': [101.0, 106.0, 113.0],
                'Low': [99.0, 104.0, 111.0]
            }, index=pd.date_range('2024-01-01', periods=3))

            should_exit, reason = backtester._check_exit(position, df, 2)

            assert should_exit == True
            assert reason == 'take_profit'

    def test_check_exit_trailing_stop(self):
        """Test trailing stop exit"""
        config = BacktestConfig(
            use_trailing_stop=True,
            trailing_stop_pct=0.05  # 5% trailing stop
        )

        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester(config)

            position = {
                'entry_price': 100.0,
                'stop_loss': 90.0,
                'highest_price': 120.0  # Highest was 120
            }

            # Create df where price dropped 6% from high (below 5% trailing)
            # Trailing stop = 120 * 0.95 = 114
            df = pd.DataFrame({
                'Close': [120.0, 118.0, 112.0],
                'High': [121.0, 119.0, 113.0],
                'Low': [119.0, 117.0, 111.0]  # Low is below trailing stop
            }, index=pd.date_range('2024-01-01', periods=3))

            should_exit, reason = backtester._check_exit(position, df, 2)

            assert should_exit == True
            assert reason == 'trailing_stop'

    def test_check_exit_max_hold_period(self):
        """Test max hold period exit"""
        # Disable adaptive exit and set take_profit high so it doesn't trigger
        config = BacktestConfig(max_hold_days=30, use_adaptive_exit=False, take_profit_pct=0.50)

        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester(config)

            position = {
                'entry_price': 100.0,
                'stop_loss': 90.0,
                'highest_price': 105.0,
                'entry_date': pd.Timestamp('2024-01-01')
            }

            # Create df with date 35 days after entry
            df = pd.DataFrame({
                'Close': [102.0],
                'High': [103.0],
                'Low': [101.0]
            }, index=[pd.Timestamp('2024-02-05')])  # 35 days after entry

            should_exit, reason = backtester._check_exit(position, df, 0)

            assert should_exit == True
            assert reason == 'max_hold_period'

    def test_check_exit_no_exit(self):
        """Test when no exit condition is met"""
        config = BacktestConfig(take_profit_pct=0.20, max_hold_days=60)

        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester(config)

            position = {
                'entry_price': 100.0,
                'stop_loss': 90.0,
                'highest_price': 105.0,
                'entry_date': pd.Timestamp('2024-01-01')
            }

            # Create df with normal price movement
            df = pd.DataFrame({
                'Close': [103.0],  # 3% profit
                'High': [104.0],
                'Low': [102.0]  # Above stop loss
            }, index=[pd.Timestamp('2024-01-15')])  # 14 days later

            should_exit, reason = backtester._check_exit(position, df, 0)

            assert should_exit == False
            assert reason == ''


class TestOpenPosition:
    """Test position opening logic"""

    def test_open_position_basic(self):
        """Test basic position opening"""
        config = BacktestConfig(position_size_pct=0.20)

        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester(config)

            # Mock signal with trendline
            mock_trendline = Mock()
            mock_trendline.peak_indices = [10, 20, 30]

            signal = {
                'signal': 'STRONG_BUY',
                'confidence_score': 75,
                'volume_ratio': 1.5,
                'rsi_trendline': mock_trendline
            }

            df = generate_mock_ohlcv(days=100)

            position = backtester._open_position(
                symbol='AAPL',
                entry_date=pd.Timestamp('2024-01-15'),
                entry_price=150.0,
                signal=signal,
                df=df,
                idx=50,
                capital=10000
            )

            assert position['symbol'] == 'AAPL'
            assert position['entry_price'] == 150.0
            assert position['shares'] > 0
            assert position['stop_loss'] > 0
            assert position['confidence_score'] == 75

    def test_open_position_calculates_shares(self):
        """Test that position size is calculated correctly"""
        config = BacktestConfig(
            initial_capital=10000,
            position_size_pct=0.20  # 20% = $2000
        )

        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester(config)

            signal = {
                'signal': 'BUY',
                'confidence_score': 60,
                'volume_ratio': 1.0,
                'rsi_trendline': None
            }

            df = generate_mock_ohlcv(days=100)

            # At $100/share, $2000 = 20 shares
            position = backtester._open_position(
                symbol='TEST',
                entry_date=pd.Timestamp('2024-01-15'),
                entry_price=100.0,
                signal=signal,
                df=df,
                idx=50,
                capital=10000
            )

            # Expected: 10000 * 0.20 / 100 = 20 shares
            assert position['shares'] == 20


class TestClosePosition:
    """Test position closing logic"""

    def test_close_position_profit(self):
        """Test closing a profitable position"""
        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester()

            position = {
                'symbol': 'AAPL',
                'entry_date': pd.Timestamp('2024-01-01'),
                'entry_price': 100.0,
                'shares': 10,
                'stop_loss': 95.0,
                'signal_strength': 'STRONG_BUY',
                'confidence_score': 75,
                'volume_ratio': 1.5,
                'highest_price': 120.0
            }

            trade = backtester._close_position(
                position=position,
                exit_price=115.0,
                exit_date=pd.Timestamp('2024-02-01'),
                exit_reason='take_profit'
            )

            assert isinstance(trade, Trade)
            assert trade.symbol == 'AAPL'
            assert trade.profit_loss == 150.0  # (115 - 100) * 10
            assert trade.profit_loss_pct == 15.0  # 15% profit
            assert trade.exit_reason == 'take_profit'

    def test_close_position_loss(self):
        """Test closing a losing position"""
        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester()

            position = {
                'symbol': 'TSLA',
                'entry_date': pd.Timestamp('2024-01-01'),
                'entry_price': 200.0,
                'shares': 5,
                'stop_loss': 180.0,
                'signal_strength': 'BUY',
                'confidence_score': 58,
                'volume_ratio': 1.1,
                'highest_price': 205.0
            }

            trade = backtester._close_position(
                position=position,
                exit_price=180.0,
                exit_date=pd.Timestamp('2024-01-15'),
                exit_reason='stop_loss'
            )

            assert trade.profit_loss == -100.0  # (180 - 200) * 5
            assert trade.profit_loss_pct == -10.0
            assert trade.exit_reason == 'stop_loss'


# ====================
# Edge Cases
# ====================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_symbol_list(self):
        """Test backtest with empty symbol list"""
        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester()

            result = backtester.run(
                symbols=[],
                start_date='2024-01-01',
                end_date='2024-12-01'
            )

            assert result.metrics.total_trades == 0

    def test_insufficient_data(self):
        """Test with insufficient historical data"""
        mock_data_provider = Mock()
        # Return only 50 rows (less than required lookback)
        mock_data_provider.get_historical_data.return_value = generate_mock_ohlcv(days=50)

        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester()

            result = backtester.run(
                symbols=['TEST'],
                start_date='2024-01-01',
                end_date='2024-12-01',
                data_provider=mock_data_provider
            )

            # Should handle gracefully with no trades
            assert result.metrics.total_trades == 0

    def test_null_data_from_provider(self):
        """Test handling of null data from provider"""
        mock_data_provider = Mock()
        mock_data_provider.get_historical_data.return_value = None

        with patch.dict('sys.modules', {'trendline_analysis.core.enhanced_rsi_breakout_analyzer': MagicMock()}):
            backtester = Backtester()

            result = backtester.run(
                symbols=['INVALID'],
                start_date='2024-01-01',
                end_date='2024-12-01',
                data_provider=mock_data_provider
            )

            assert result.metrics.total_trades == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
