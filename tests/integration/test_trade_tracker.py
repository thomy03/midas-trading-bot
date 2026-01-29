"""
Integration tests for Trade Tracker

Tests the trade tracking and performance metrics:
- Trade opening/closing
- P&L calculations
- Win rate
- Sharpe ratio
- Max drawdown
- Equity curve
"""
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
import tempfile
import os

from src.utils.trade_tracker import TradeTracker, Trade


class TestTradeBasics:
    """Tests for basic trade operations"""

    @pytest.fixture
    def tracker(self, tmp_path):
        """Fresh tracker instance with temp file"""
        temp_file = str(tmp_path / "test_trades.json")
        return TradeTracker(data_file=temp_file, initial_capital=10000)

    @pytest.mark.integration
    def test_open_trade(self, tracker):
        """Test opening a new trade"""
        trade = tracker.open_trade(
            symbol='AAPL',
            entry_price=150.0,
            shares=10,
            stop_loss=145.0
        )

        assert trade is not None
        assert trade.symbol == 'AAPL'
        assert trade.entry_price == 150.0
        assert trade.shares == 10
        assert trade.stop_loss == 145.0
        assert trade.is_open is True
        assert trade.status == 'open'

    @pytest.mark.integration
    def test_close_trade(self, tracker):
        """Test closing a trade"""
        trade = tracker.open_trade(
            symbol='MSFT',
            entry_price=300.0,
            shares=5,
            stop_loss=290.0
        )

        closed = tracker.close_trade(
            trade_id=trade.trade_id,
            exit_price=320.0,
            exit_reason='target'
        )

        assert closed is not None
        assert closed.status == 'closed'
        assert closed.exit_price == 320.0
        assert closed.exit_reason == 'target'
        assert closed.pnl == 100.0  # (320 - 300) * 5

    @pytest.mark.integration
    def test_close_trade_by_symbol(self, tracker):
        """Test closing a trade by symbol"""
        tracker.open_trade('GOOGL', 2800.0, 2, 2700.0)

        closed = tracker.close_trade_by_symbol('GOOGL', 2900.0, 'manual')

        assert closed is not None
        assert closed.symbol == 'GOOGL'
        assert closed.status == 'closed'

    @pytest.mark.integration
    def test_trade_pnl_calculation(self, tracker):
        """Test P&L calculations"""
        trade = tracker.open_trade('TEST', 100.0, 10, 95.0)
        tracker.close_trade(trade.trade_id, 110.0)

        # P&L = (110 - 100) * 10 = 100
        assert trade.pnl == 100.0
        # P&L % = ((110 - 100) / 100) * 100 = 10%
        assert trade.pnl_pct == 10.0

    @pytest.mark.integration
    def test_losing_trade(self, tracker):
        """Test P&L for losing trade"""
        trade = tracker.open_trade('LOSS', 100.0, 10, 95.0)
        tracker.close_trade(trade.trade_id, 90.0)

        assert trade.pnl == -100.0
        assert trade.pnl_pct == -10.0
        assert trade.is_winner is False

    @pytest.mark.integration
    def test_get_open_trades(self, tracker):
        """Test getting open trades"""
        tracker.open_trade('OPEN1', 100.0, 10, 95.0)
        tracker.open_trade('OPEN2', 200.0, 5, 190.0)
        trade3 = tracker.open_trade('CLOSE1', 150.0, 8, 145.0)
        tracker.close_trade(trade3.trade_id, 160.0)

        open_trades = tracker.get_open_trades()
        assert len(open_trades) == 2
        assert all(t.is_open for t in open_trades)

    @pytest.mark.integration
    def test_get_closed_trades(self, tracker):
        """Test getting closed trades"""
        t1 = tracker.open_trade('C1', 100.0, 10, 95.0)
        t2 = tracker.open_trade('C2', 200.0, 5, 190.0)
        tracker.open_trade('OPEN', 150.0, 8, 145.0)

        tracker.close_trade(t1.trade_id, 110.0)
        tracker.close_trade(t2.trade_id, 210.0)

        closed_trades = tracker.get_closed_trades()
        assert len(closed_trades) == 2
        assert all(not t.is_open for t in closed_trades)

    @pytest.mark.integration
    def test_delete_trade(self, tracker):
        """Test deleting a trade"""
        trade = tracker.open_trade('DEL', 100.0, 10, 95.0)
        trade_id = trade.trade_id

        result = tracker.delete_trade(trade_id)
        assert result is True
        assert tracker.get_trade(trade_id) is None


class TestPerformanceMetrics:
    """Tests for performance metrics calculations"""

    @pytest.fixture
    def tracker_with_trades(self, tmp_path):
        """Tracker with sample trades for metrics testing"""
        temp_file = str(tmp_path / "test_metrics.json")
        tracker = TradeTracker(data_file=temp_file, initial_capital=10000)

        # Add some winning trades
        for i, (symbol, entry, exit_) in enumerate([
            ('WIN1', 100.0, 115.0),  # +15%
            ('WIN2', 200.0, 220.0),  # +10%
            ('WIN3', 150.0, 165.0),  # +10%
        ]):
            t = tracker.open_trade(symbol, entry, 10, entry * 0.95,
                                  entry_date=f'2023-{i+1:02d}-01')
            tracker.close_trade(t.trade_id, exit_, exit_date=f'2023-{i+1:02d}-15')

        # Add some losing trades
        for i, (symbol, entry, exit_) in enumerate([
            ('LOSS1', 100.0, 92.0),  # -8%
            ('LOSS2', 200.0, 185.0),  # -7.5%
        ], start=4):
            t = tracker.open_trade(symbol, entry, 10, entry * 0.95,
                                  entry_date=f'2023-{i:02d}-01')
            tracker.close_trade(t.trade_id, exit_, exit_date=f'2023-{i:02d}-15')

        return tracker

    @pytest.mark.integration
    def test_total_pnl(self, tracker_with_trades):
        """Test total P&L calculation"""
        total_pnl = tracker_with_trades.get_total_pnl()

        # WIN1: (115-100)*10 = 150
        # WIN2: (220-200)*10 = 200
        # WIN3: (165-150)*10 = 150
        # LOSS1: (92-100)*10 = -80
        # LOSS2: (185-200)*10 = -150
        expected = 150 + 200 + 150 - 80 - 150
        assert total_pnl == expected

    @pytest.mark.integration
    def test_win_rate(self, tracker_with_trades):
        """Test win rate calculation"""
        win_rate, wins, losses = tracker_with_trades.get_win_rate()

        assert wins == 3
        assert losses == 2
        assert win_rate == 60.0  # 3/5 = 60%

    @pytest.mark.integration
    def test_average_pnl(self, tracker_with_trades):
        """Test average P&L calculations"""
        avg_pnl, avg_win, avg_loss = tracker_with_trades.get_average_pnl()

        # Average win = (150 + 200 + 150) / 3 = 166.67
        assert abs(avg_win - 166.67) < 0.1

        # Average loss = (-80 + -150) / 2 = -115
        assert avg_loss == -115.0

    @pytest.mark.integration
    def test_profit_factor(self, tracker_with_trades):
        """Test profit factor calculation"""
        profit_factor = tracker_with_trades.get_profit_factor()

        # Gross profit = 150 + 200 + 150 = 500
        # Gross loss = 80 + 150 = 230
        # Profit factor = 500 / 230 = 2.17
        assert abs(profit_factor - 2.17) < 0.1

    @pytest.mark.integration
    def test_sharpe_ratio_calculated(self, tracker_with_trades):
        """Test Sharpe ratio is calculated"""
        sharpe = tracker_with_trades.get_sharpe_ratio()

        # Just verify it returns a number
        assert isinstance(sharpe, float)

    @pytest.mark.integration
    def test_sortino_ratio_calculated(self, tracker_with_trades):
        """Test Sortino ratio is calculated"""
        sortino = tracker_with_trades.get_sortino_ratio()

        assert isinstance(sortino, float)

    @pytest.mark.integration
    def test_max_drawdown(self, tracker_with_trades):
        """Test max drawdown calculation"""
        max_dd_amount, max_dd_pct = tracker_with_trades.get_max_drawdown()

        # Drawdown should be positive
        assert max_dd_amount >= 0
        assert max_dd_pct >= 0

    @pytest.mark.integration
    def test_consecutive_stats(self, tracker_with_trades):
        """Test consecutive wins/losses stats"""
        stats = tracker_with_trades.get_consecutive_stats()

        assert 'max_consecutive_wins' in stats
        assert 'max_consecutive_losses' in stats
        assert 'current_streak' in stats
        assert stats['max_consecutive_wins'] >= 0
        assert stats['max_consecutive_losses'] >= 0

    @pytest.mark.integration
    def test_performance_summary(self, tracker_with_trades):
        """Test complete performance summary"""
        summary = tracker_with_trades.get_performance_summary()

        required_keys = [
            'initial_capital', 'current_equity', 'total_pnl', 'roi_pct',
            'total_trades', 'open_trades', 'closed_trades',
            'wins', 'losses', 'win_rate',
            'avg_pnl', 'avg_win', 'avg_loss',
            'profit_factor', 'sharpe_ratio', 'sortino_ratio',
            'max_drawdown', 'max_drawdown_pct'
        ]

        for key in required_keys:
            assert key in summary, f"Missing key: {key}"


class TestEquityCurve:
    """Tests for equity curve generation"""

    @pytest.fixture
    def tracker_with_trades(self, tmp_path):
        """Tracker with trades for equity curve"""
        temp_file = str(tmp_path / "test_equity.json")
        tracker = TradeTracker(data_file=temp_file, initial_capital=10000)

        trades_data = [
            ('T1', 100.0, 110.0, '2023-01-01', '2023-01-10'),  # +100
            ('T2', 100.0, 95.0, '2023-01-15', '2023-01-20'),   # -50
            ('T3', 100.0, 120.0, '2023-02-01', '2023-02-10'),  # +200
        ]

        for symbol, entry, exit_, entry_dt, exit_dt in trades_data:
            t = tracker.open_trade(symbol, entry, 10, entry * 0.9, entry_date=entry_dt)
            tracker.close_trade(t.trade_id, exit_, exit_date=exit_dt)

        return tracker

    @pytest.mark.integration
    def test_equity_curve_dataframe(self, tracker_with_trades):
        """Test equity curve returns DataFrame"""
        equity_df = tracker_with_trades.get_equity_curve()

        assert isinstance(equity_df, pd.DataFrame)
        assert not equity_df.empty

    @pytest.mark.integration
    def test_equity_curve_columns(self, tracker_with_trades):
        """Test equity curve has required columns"""
        equity_df = tracker_with_trades.get_equity_curve()

        required_cols = ['date', 'equity', 'pnl', 'cumulative_pnl', 'drawdown_pct']
        for col in required_cols:
            assert col in equity_df.columns, f"Missing column: {col}"

    @pytest.mark.integration
    def test_equity_curve_values(self, tracker_with_trades):
        """Test equity curve values are correct"""
        equity_df = tracker_with_trades.get_equity_curve()

        # Initial capital = 10000
        # After T1: 10000 + 100 = 10100
        # After T2: 10100 - 50 = 10050
        # After T3: 10050 + 200 = 10250
        final_equity = equity_df['equity'].iloc[-1]
        assert final_equity == 10250

    @pytest.mark.integration
    def test_equity_curve_empty_tracker(self, tmp_path):
        """Test equity curve with no trades"""
        temp_file = str(tmp_path / "empty_equity.json")
        tracker = TradeTracker(data_file=temp_file)

        equity_df = tracker.get_equity_curve()
        assert equity_df.empty


class TestMonthlyReturns:
    """Tests for monthly returns"""

    @pytest.fixture
    def tracker_with_monthly_trades(self, tmp_path):
        """Tracker with trades across months"""
        temp_file = str(tmp_path / "test_monthly.json")
        tracker = TradeTracker(data_file=temp_file, initial_capital=10000)

        # January trades
        t1 = tracker.open_trade('JAN1', 100, 10, 95, entry_date='2023-01-05')
        tracker.close_trade(t1.trade_id, 110, exit_date='2023-01-15')

        t2 = tracker.open_trade('JAN2', 100, 10, 95, entry_date='2023-01-20')
        tracker.close_trade(t2.trade_id, 105, exit_date='2023-01-25')

        # February trades
        t3 = tracker.open_trade('FEB1', 100, 10, 95, entry_date='2023-02-01')
        tracker.close_trade(t3.trade_id, 90, exit_date='2023-02-10')

        return tracker

    @pytest.mark.integration
    def test_monthly_returns_dataframe(self, tracker_with_monthly_trades):
        """Test monthly returns returns DataFrame"""
        monthly_df = tracker_with_monthly_trades.get_monthly_returns()

        assert isinstance(monthly_df, pd.DataFrame)
        assert not monthly_df.empty

    @pytest.mark.integration
    def test_monthly_returns_columns(self, tracker_with_monthly_trades):
        """Test monthly returns has required columns"""
        monthly_df = tracker_with_monthly_trades.get_monthly_returns()

        required_cols = ['year', 'month', 'pnl', 'trades', 'win_rate']
        for col in required_cols:
            assert col in monthly_df.columns

    @pytest.mark.integration
    def test_monthly_returns_values(self, tracker_with_monthly_trades):
        """Test monthly returns values"""
        monthly_df = tracker_with_monthly_trades.get_monthly_returns()

        # January: +100 + +50 = +150
        jan = monthly_df[(monthly_df['year'] == 2023) & (monthly_df['month'] == 1)]
        assert len(jan) == 1
        assert jan['pnl'].iloc[0] == 150
        assert jan['trades'].iloc[0] == 2

        # February: -100
        feb = monthly_df[(monthly_df['year'] == 2023) & (monthly_df['month'] == 2)]
        assert len(feb) == 1
        assert feb['pnl'].iloc[0] == -100
        assert feb['trades'].iloc[0] == 1


class TestSymbolPerformance:
    """Tests for symbol performance"""

    @pytest.fixture
    def tracker_multi_symbol(self, tmp_path):
        """Tracker with multiple symbols"""
        temp_file = str(tmp_path / "test_symbols.json")
        tracker = TradeTracker(data_file=temp_file, initial_capital=10000)

        # AAPL trades
        t1 = tracker.open_trade('AAPL', 150, 10, 145, entry_date='2023-01-01')
        tracker.close_trade(t1.trade_id, 160, exit_date='2023-01-10')

        t2 = tracker.open_trade('AAPL', 155, 10, 150, entry_date='2023-01-15')
        tracker.close_trade(t2.trade_id, 165, exit_date='2023-01-20')

        # MSFT trades
        t3 = tracker.open_trade('MSFT', 300, 5, 290, entry_date='2023-02-01')
        tracker.close_trade(t3.trade_id, 280, exit_date='2023-02-10')

        return tracker

    @pytest.mark.integration
    def test_symbol_performance_dataframe(self, tracker_multi_symbol):
        """Test symbol performance returns DataFrame"""
        symbol_df = tracker_multi_symbol.get_symbol_performance()

        assert isinstance(symbol_df, pd.DataFrame)
        assert not symbol_df.empty

    @pytest.mark.integration
    def test_symbol_performance_columns(self, tracker_multi_symbol):
        """Test symbol performance columns"""
        symbol_df = tracker_multi_symbol.get_symbol_performance()

        required_cols = ['symbol', 'trades', 'pnl', 'win_rate', 'avg_pnl']
        for col in required_cols:
            assert col in symbol_df.columns

    @pytest.mark.integration
    def test_symbol_performance_values(self, tracker_multi_symbol):
        """Test symbol performance values"""
        symbol_df = tracker_multi_symbol.get_symbol_performance()

        # AAPL: +100 + +100 = +200, 2 trades, 100% win rate
        aapl = symbol_df[symbol_df['symbol'] == 'AAPL']
        assert aapl['pnl'].iloc[0] == 200
        assert aapl['trades'].iloc[0] == 2
        assert aapl['win_rate'].iloc[0] == 100.0

        # MSFT: -100, 1 trade, 0% win rate
        msft = symbol_df[symbol_df['symbol'] == 'MSFT']
        assert msft['pnl'].iloc[0] == -100
        assert msft['trades'].iloc[0] == 1
        assert msft['win_rate'].iloc[0] == 0.0


class TestPersistence:
    """Tests for data persistence"""

    @pytest.mark.integration
    def test_trades_persist(self, tmp_path):
        """Test trades are persisted to file"""
        temp_file = str(tmp_path / "persist_test.json")

        # Create tracker and add trade
        tracker1 = TradeTracker(data_file=temp_file, initial_capital=10000)
        tracker1.open_trade('PERSIST', 100, 10, 95)

        # Create new tracker with same file
        tracker2 = TradeTracker(data_file=temp_file)

        assert len(tracker2.trades) == 1
        assert tracker2.trades[0].symbol == 'PERSIST'

    @pytest.mark.integration
    def test_initial_capital_persists(self, tmp_path):
        """Test initial capital persists"""
        temp_file = str(tmp_path / "capital_test.json")

        tracker1 = TradeTracker(data_file=temp_file, initial_capital=25000)
        tracker1.set_initial_capital(30000)

        tracker2 = TradeTracker(data_file=temp_file)
        assert tracker2.initial_capital == 30000


class TestEdgeCases:
    """Tests for edge cases"""

    @pytest.fixture
    def tracker(self, tmp_path):
        temp_file = str(tmp_path / "edge_test.json")
        return TradeTracker(data_file=temp_file, initial_capital=10000)

    @pytest.mark.integration
    def test_close_nonexistent_trade(self, tracker):
        """Test closing non-existent trade"""
        result = tracker.close_trade('NONEXISTENT', 100.0)
        assert result is None

    @pytest.mark.integration
    def test_metrics_with_no_trades(self, tracker):
        """Test metrics with no trades"""
        assert tracker.get_total_pnl() == 0
        assert tracker.get_win_rate() == (0.0, 0, 0)
        assert tracker.get_average_pnl() == (0.0, 0.0, 0.0)
        assert tracker.get_profit_factor() == 0.0
        assert tracker.get_sharpe_ratio() == 0.0

    @pytest.mark.integration
    def test_metrics_with_one_trade(self, tracker):
        """Test metrics with single trade"""
        t = tracker.open_trade('SINGLE', 100, 10, 95)
        tracker.close_trade(t.trade_id, 110)

        # Should not raise errors
        tracker.get_performance_summary()
        tracker.get_equity_curve()

    @pytest.mark.integration
    def test_all_winning_trades(self, tracker):
        """Test profit factor with all wins"""
        for i in range(3):
            t = tracker.open_trade(f'WIN{i}', 100, 10, 95, entry_date=f'2023-01-{i+1:02d}')
            tracker.close_trade(t.trade_id, 110, exit_date=f'2023-01-{i+10:02d}')

        # Profit factor should be infinity
        pf = tracker.get_profit_factor()
        assert pf == float('inf')

    @pytest.mark.integration
    def test_clear_all_trades(self, tracker):
        """Test clearing all trades"""
        tracker.open_trade('T1', 100, 10, 95)
        tracker.open_trade('T2', 200, 5, 190)

        tracker.clear_all_trades()

        assert len(tracker.trades) == 0
        assert tracker.get_total_pnl() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
