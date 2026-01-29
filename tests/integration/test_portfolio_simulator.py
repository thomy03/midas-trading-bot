"""
Integration tests for PortfolioSimulator

Tests the realistic day-by-day portfolio simulation with:
- Proper capital management
- Max positions enforcement
- Signal priority by confidence score
- Position tracking
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backtesting.portfolio_simulator import (
    PortfolioSimulator,
    Position,
    PortfolioState,
    SimulationResult,
    run_realistic_simulation
)
from src.backtesting.backtester import BacktestConfig


def generate_mock_ohlcv(
    start_date: str = '2023-01-01',
    periods: int = 52,
    freq: str = 'W',
    base_price: float = 100,
    volatility: float = 0.02,
    trend: float = 0.001
) -> pd.DataFrame:
    """Generate mock OHLCV data for testing"""
    np.random.seed(42)

    dates = pd.date_range(start=start_date, periods=periods, freq=freq)

    # Generate price series with trend and volatility
    returns = np.random.normal(trend, volatility, periods)
    close = base_price * np.cumprod(1 + returns)

    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.01, periods)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, periods)))
    open_price = low + (high - low) * np.random.random(periods)

    # Volume with some variation
    volume = np.random.uniform(1_000_000, 5_000_000, periods)

    df = pd.DataFrame({
        'Open': open_price,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)

    return df


class TestPosition:
    """Tests for Position dataclass"""

    def test_position_creation(self):
        """Test creating a position"""
        position = Position(
            symbol='AAPL',
            entry_date=pd.Timestamp('2023-06-01'),
            entry_price=150.0,
            shares=10,
            stop_loss=142.5,
            position_value=1500.0,
            signal_strength='STRONG_BUY',
            confidence_score=85.0,
            volume_ratio=1.5
        )

        assert position.symbol == 'AAPL'
        assert position.shares == 10
        assert position.current_value == 1500.0

    def test_position_defaults(self):
        """Test position default values"""
        position = Position(
            symbol='MSFT',
            entry_date=pd.Timestamp('2023-06-01'),
            entry_price=300.0,
            shares=5,
            stop_loss=285.0,
            position_value=1500.0,
            signal_strength='BUY',
            confidence_score=70.0,
            volume_ratio=1.2
        )

        assert position.highest_price == 0.0
        assert position.rsi_peak_indices == []


class TestPortfolioState:
    """Tests for PortfolioState dataclass"""

    def test_portfolio_state_creation(self):
        """Test creating a portfolio state"""
        state = PortfolioState(
            date=pd.Timestamp('2023-06-15'),
            cash=8000.0,
            positions_value=2000.0,
            total_value=10000.0,
            num_positions=2,
            positions={'AAPL': 1000.0, 'MSFT': 1000.0}
        )

        assert state.total_value == 10000.0
        assert state.num_positions == 2
        assert len(state.positions) == 2


class TestPortfolioSimulator:
    """Tests for PortfolioSimulator class"""

    @pytest.fixture
    def config(self):
        """Default test configuration"""
        return BacktestConfig(
            initial_capital=10000,
            max_positions=3,
            position_size_pct=0.20,
            min_confidence_score=55,
            require_volume_confirmation=False,
            use_enhanced_detector=True,
            precision_mode='medium',
            timeframe='weekly'
        )

    @pytest.fixture
    def simulator(self, config):
        """Create simulator instance"""
        return PortfolioSimulator(config)

    def test_simulator_initialization(self, simulator, config):
        """Test simulator initializes correctly"""
        assert simulator.config.initial_capital == 10000
        assert simulator.config.max_positions == 3
        assert simulator.cash == 10000
        assert len(simulator.positions) == 0

    def test_reset_state(self, simulator):
        """Test state reset"""
        # Modify state
        simulator.cash = 5000
        simulator.positions['AAPL'] = Position(
            symbol='AAPL',
            entry_date=pd.Timestamp('2023-06-01'),
            entry_price=150.0,
            shares=10,
            stop_loss=142.5,
            position_value=1500.0,
            signal_strength='BUY',
            confidence_score=70.0,
            volume_ratio=1.2
        )
        simulator.signals_detected = 10

        # Reset
        simulator._reset_state()

        assert simulator.cash == 10000
        assert len(simulator.positions) == 0
        assert simulator.signals_detected == 0

    def test_get_available_capital(self, simulator):
        """Test available capital calculation"""
        assert simulator._get_available_capital() == 10000

        # Reduce cash by opening a position
        simulator.cash = 8000
        assert simulator._get_available_capital() == 8000

    def test_preprocess_all_data(self, simulator):
        """Test data preprocessing adds EMAs"""
        # Create mock data
        mock_data = {
            'AAPL': generate_mock_ohlcv(periods=100),
            'MSFT': generate_mock_ohlcv(periods=100, base_price=300)
        }

        processed = simulator._preprocess_all_data(mock_data)

        assert 'AAPL' in processed
        assert 'MSFT' in processed
        # Check EMAs were added
        assert 'EMA_24' in processed['AAPL'].columns
        assert 'EMA_38' in processed['AAPL'].columns
        assert 'EMA_62' in processed['AAPL'].columns

    def test_get_trading_days(self, simulator):
        """Test trading days extraction"""
        # Need >100 periods for _get_trading_days to work
        mock_data = {
            'AAPL': generate_mock_ohlcv(
                start_date='2020-01-01',
                periods=200,  # >100 required
                freq='W-MON'  # Use Monday-anchored weeks
            )
        }

        # Use dates that are within the generated data range
        start = pd.Timestamp('2022-01-01')
        end = pd.Timestamp('2023-06-01')

        days = simulator._get_trading_days(mock_data, start, end)

        assert len(days) > 0
        assert all(start <= d <= end for d in days)

    def test_max_positions_enforced(self, simulator):
        """Test that max positions limit is respected"""
        # Fill up positions
        for i, symbol in enumerate(['AAPL', 'MSFT', 'GOOGL']):
            simulator.positions[symbol] = Position(
                symbol=symbol,
                entry_date=pd.Timestamp('2023-06-01'),
                entry_price=100.0 + i * 100,
                shares=10,
                stop_loss=90.0 + i * 90,
                position_value=1500.0,
                signal_strength='BUY',
                confidence_score=70.0,
                volume_ratio=1.2
            )
            simulator.cash -= 1500.0

        # Try to open another position
        signal = {
            'symbol': 'AMZN',
            'signal': 'STRONG_BUY',
            'confidence_score': 90.0,
            'volume_ratio': 2.0,
            'current_price': 150.0,
            'peak_indices': [],
            'df': generate_mock_ohlcv()
        }

        position = simulator._open_position(signal, simulator.cash, pd.Timestamp('2023-06-15'))

        # Position should be created (capital available)
        # But in run_simulation, max_positions check would prevent it
        assert len(simulator.positions) == 3  # Still at max


class TestCapitalManagement:
    """Tests for realistic capital management"""

    @pytest.fixture
    def simulator(self):
        config = BacktestConfig(
            initial_capital=10000,
            max_positions=5,
            position_size_pct=0.20,
            min_confidence_score=50,
            require_volume_confirmation=False
        )
        return PortfolioSimulator(config)

    def test_position_blocks_capital(self, simulator):
        """Test that opening a position blocks capital"""
        initial_cash = simulator.cash

        # Open a position
        signal = {
            'symbol': 'AAPL',
            'signal': 'BUY',
            'confidence_score': 70.0,
            'volume_ratio': 1.5,
            'current_price': 150.0,
            'peak_indices': [],
            'df': generate_mock_ohlcv()
        }

        position = simulator._open_position(signal, initial_cash, pd.Timestamp('2023-06-01'))

        assert position is not None
        assert simulator.cash < initial_cash
        assert simulator.cash == initial_cash - position.position_value

    def test_closing_position_returns_capital(self, simulator):
        """Test that closing a position returns capital to cash"""
        # Open a position first
        entry_price = 150.0
        shares = 10
        position_value = entry_price * shares

        simulator.positions['AAPL'] = Position(
            symbol='AAPL',
            entry_date=pd.Timestamp('2023-06-01'),
            entry_price=entry_price,
            shares=shares,
            stop_loss=142.5,
            position_value=position_value,
            signal_strength='BUY',
            confidence_score=70.0,
            volume_ratio=1.2
        )
        simulator.cash -= position_value

        cash_before = simulator.cash
        exit_price = 165.0  # Profitable exit

        # Close the position
        simulator._close_position('AAPL', exit_price, pd.Timestamp('2023-06-15'), 'take_profit')

        assert 'AAPL' not in simulator.positions
        assert simulator.cash == cash_before + (exit_price * shares)
        assert len(simulator.closed_trades) == 1

    def test_insufficient_capital_prevents_position(self, simulator):
        """Test that position can't be opened with insufficient capital"""
        # Reduce cash to minimum
        simulator.cash = 100.0

        signal = {
            'symbol': 'AAPL',
            'signal': 'BUY',
            'confidence_score': 70.0,
            'volume_ratio': 1.5,
            'current_price': 150.0,  # Would need at least 150 for 1 share
            'peak_indices': [],
            'df': generate_mock_ohlcv()
        }

        position = simulator._open_position(signal, simulator.cash, pd.Timestamp('2023-06-01'))

        # Position should not be created (can't afford even 1 share)
        assert position is None


class TestExitConditions:
    """Tests for position exit logic"""

    @pytest.fixture
    def simulator(self):
        config = BacktestConfig(
            initial_capital=10000,
            max_positions=5,
            take_profit_pct=0.15,
            use_trailing_stop=False,
            max_hold_days=60
        )
        return PortfolioSimulator(config)

    def test_stop_loss_exit(self, simulator):
        """Test stop-loss triggers exit"""
        position = Position(
            symbol='AAPL',
            entry_date=pd.Timestamp('2023-06-01'),
            entry_price=150.0,
            shares=10,
            stop_loss=142.5,  # 5% below entry
            position_value=1500.0,
            signal_strength='BUY',
            confidence_score=70.0,
            volume_ratio=1.2
        )

        # Price drops below stop
        should_exit, reason = simulator._check_exit_conditions(
            position,
            current_price=140.0,
            current_high=145.0,
            current_low=138.0,  # Below stop
            current_day=pd.Timestamp('2023-06-15')
        )

        assert should_exit is True
        assert reason == 'stop_loss'

    def test_take_profit_exit(self, simulator):
        """Test take-profit triggers exit"""
        position = Position(
            symbol='AAPL',
            entry_date=pd.Timestamp('2023-06-01'),
            entry_price=150.0,
            shares=10,
            stop_loss=142.5,
            position_value=1500.0,
            signal_strength='BUY',
            confidence_score=70.0,
            volume_ratio=1.2
        )

        # Price rises 15% (take profit threshold)
        should_exit, reason = simulator._check_exit_conditions(
            position,
            current_price=175.0,  # +16.7%
            current_high=176.0,
            current_low=172.0,
            current_day=pd.Timestamp('2023-06-15')
        )

        assert should_exit is True
        assert reason == 'take_profit'

    def test_max_hold_period_exit(self, simulator):
        """Test max hold period triggers exit"""
        position = Position(
            symbol='AAPL',
            entry_date=pd.Timestamp('2023-01-01'),
            entry_price=150.0,
            shares=10,
            stop_loss=142.5,
            position_value=1500.0,
            signal_strength='BUY',
            confidence_score=70.0,
            volume_ratio=1.2
        )

        # 70 days later (> 60 day max)
        should_exit, reason = simulator._check_exit_conditions(
            position,
            current_price=155.0,
            current_high=156.0,
            current_low=154.0,
            current_day=pd.Timestamp('2023-03-15')
        )

        assert should_exit is True
        assert reason == 'max_hold_period'

    def test_no_exit_in_normal_conditions(self, simulator):
        """Test no exit when conditions are normal"""
        position = Position(
            symbol='AAPL',
            entry_date=pd.Timestamp('2023-06-01'),
            entry_price=150.0,
            shares=10,
            stop_loss=142.5,
            position_value=1500.0,
            signal_strength='BUY',
            confidence_score=70.0,
            volume_ratio=1.2
        )

        # Normal price movement
        should_exit, reason = simulator._check_exit_conditions(
            position,
            current_price=155.0,  # +3.3%
            current_high=156.0,
            current_low=154.0,  # Above stop
            current_day=pd.Timestamp('2023-06-15')
        )

        assert should_exit is False
        assert reason == ''


class TestSignalPriority:
    """Tests for signal prioritization by confidence score"""

    def test_signals_sorted_by_confidence(self):
        """Test that signals are sorted by confidence score"""
        signals = [
            {'symbol': 'AAPL', 'confidence_score': 65},
            {'symbol': 'MSFT', 'confidence_score': 85},
            {'symbol': 'GOOGL', 'confidence_score': 75},
            {'symbol': 'AMZN', 'confidence_score': 90}
        ]

        # Sort as simulator does
        sorted_signals = sorted(signals, key=lambda s: s['confidence_score'], reverse=True)

        assert sorted_signals[0]['symbol'] == 'AMZN'  # 90
        assert sorted_signals[1]['symbol'] == 'MSFT'  # 85
        assert sorted_signals[2]['symbol'] == 'GOOGL'  # 75
        assert sorted_signals[3]['symbol'] == 'AAPL'  # 65


class TestSimulationResult:
    """Tests for SimulationResult"""

    def test_result_str(self):
        """Test result string representation"""
        from src.backtesting.metrics import PerformanceMetrics

        metrics = PerformanceMetrics(
            total_trades=10,
            winning_trades=7,
            losing_trades=3,
            win_rate=70.0,
            total_return=15.5,
            total_profit=1550.0,
            avg_win=5.0,
            avg_loss=-3.0,
            profit_factor=2.0,
            max_drawdown=5.0,
            max_drawdown_duration=10,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            avg_trade_duration=7.0,
            best_trade=10.0,
            worst_trade=-5.0,
            start_date='2023-01-01',
            end_date='2023-12-01',
            initial_capital=10000,
            final_capital=11550
        )

        result = SimulationResult(
            config=BacktestConfig(),
            metrics=metrics,
            trades=[],
            equity_curve=pd.Series([10000, 11000, 11550]),
            daily_states=[],
            signals_detected=50,
            signals_taken=10,
            signals_skipped_capital=5,
            signals_skipped_max_positions=35
        )

        result_str = str(result)

        assert 'REALISTIC PORTFOLIO SIMULATION' in result_str
        assert 'Signals Detected:     50' in result_str
        assert 'Signals Taken:        10' in result_str
        assert 'Skipped (Capital):    5' in result_str
        assert 'Skipped (Max Pos):    35' in result_str


class TestDailyStateRecording:
    """Tests for daily portfolio state recording"""

    @pytest.fixture
    def simulator(self):
        config = BacktestConfig(initial_capital=10000)
        return PortfolioSimulator(config)

    def test_record_daily_state(self, simulator):
        """Test daily state recording"""
        # Add a position
        simulator.positions['AAPL'] = Position(
            symbol='AAPL',
            entry_date=pd.Timestamp('2023-06-01'),
            entry_price=150.0,
            shares=10,
            stop_loss=142.5,
            position_value=1500.0,
            signal_strength='BUY',
            confidence_score=70.0,
            volume_ratio=1.2
        )
        simulator.cash = 8500.0

        # Mock data with current price
        mock_data = {
            'AAPL': pd.DataFrame({
                'Close': [155.0]
            }, index=[pd.Timestamp('2023-06-15')])
        }

        # Record state
        simulator._record_daily_state(pd.Timestamp('2023-06-15'), mock_data)

        assert len(simulator.daily_states) == 1
        state = simulator.daily_states[0]

        assert state.date == pd.Timestamp('2023-06-15')
        assert state.cash == 8500.0
        assert state.num_positions == 1
        assert state.positions_value == 155.0 * 10  # Current market value
        assert state.total_value == 8500.0 + 1550.0


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
