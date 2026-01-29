"""
Performance Metrics for Backtesting

Calculates trading performance metrics:
- Win rate
- Profit factor
- Max drawdown
- Sharpe ratio
- Total return
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class PerformanceMetrics:
    """Trading performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float  # 0-100%

    total_return: float  # Total % return
    total_profit: float  # Absolute profit in currency

    avg_win: float  # Average winning trade %
    avg_loss: float  # Average losing trade %
    profit_factor: float  # Gross profit / Gross loss

    max_drawdown: float  # Maximum drawdown %
    max_drawdown_duration: int  # Days in max drawdown

    sharpe_ratio: float  # Risk-adjusted return
    sortino_ratio: float  # Downside risk-adjusted return

    avg_trade_duration: float  # Average days in trade
    best_trade: float  # Best single trade %
    worst_trade: float  # Worst single trade %

    # Additional context
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float


@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: int
    stop_loss: float

    # Results
    profit_loss: float  # Absolute P&L
    profit_loss_pct: float  # % P&L
    exit_reason: str  # 'stop_loss', 'take_profit', 'signal_exit', 'end_of_period'

    # Context
    signal_strength: str  # 'STRONG_BUY', 'BUY', 'WATCH'
    confidence_score: float
    volume_ratio: float


def calculate_metrics(
    trades: List[Trade],
    initial_capital: float,
    daily_returns: Optional[pd.Series] = None
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics from trade history

    Args:
        trades: List of completed trades
        initial_capital: Starting capital
        daily_returns: Optional daily portfolio returns for Sharpe calculation

    Returns:
        PerformanceMetrics with all calculated values
    """
    if not trades:
        return _empty_metrics(initial_capital)

    # Basic trade stats
    total_trades = len(trades)
    profits = [t.profit_loss_pct for t in trades]
    abs_profits = [t.profit_loss for t in trades]

    winning_trades = sum(1 for p in profits if p > 0)
    losing_trades = sum(1 for p in profits if p < 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    # Profit/Loss analysis
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]

    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    gross_profit = sum(p for p in abs_profits if p > 0)
    gross_loss = abs(sum(p for p in abs_profits if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Total return
    total_profit = sum(abs_profits)
    total_return = (total_profit / initial_capital * 100) if initial_capital > 0 else 0
    final_capital = initial_capital + total_profit

    # Best/Worst trades
    best_trade = max(profits) if profits else 0
    worst_trade = min(profits) if profits else 0

    # Trade duration
    durations = [(t.exit_date - t.entry_date).days for t in trades]
    avg_trade_duration = np.mean(durations) if durations else 0

    # Drawdown calculation
    max_dd, max_dd_duration = _calculate_drawdown(trades, initial_capital)

    # Risk-adjusted returns
    sharpe = _calculate_sharpe(daily_returns) if daily_returns is not None else 0
    sortino = _calculate_sortino(daily_returns) if daily_returns is not None else 0

    # Date range
    start_date = min(t.entry_date for t in trades).strftime('%Y-%m-%d')
    end_date = max(t.exit_date for t in trades).strftime('%Y-%m-%d')

    return PerformanceMetrics(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=round(win_rate, 1),
        total_return=round(total_return, 2),
        total_profit=round(total_profit, 2),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        profit_factor=round(profit_factor, 2),
        max_drawdown=round(max_dd, 2),
        max_drawdown_duration=max_dd_duration,
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        avg_trade_duration=round(avg_trade_duration, 1),
        best_trade=round(best_trade, 2),
        worst_trade=round(worst_trade, 2),
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        final_capital=round(final_capital, 2)
    )


def _calculate_drawdown(trades: List[Trade], initial_capital: float) -> tuple:
    """
    Calculate maximum drawdown and duration

    Returns:
        (max_drawdown_pct, max_drawdown_days)
    """
    if not trades:
        return (0, 0)

    # Build equity curve
    equity = initial_capital
    peak = equity
    max_dd = 0
    max_dd_duration = 0

    current_dd_start = None

    for trade in sorted(trades, key=lambda t: t.exit_date):
        equity += trade.profit_loss

        if equity > peak:
            peak = equity
            current_dd_start = None
        else:
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
                if current_dd_start is None:
                    current_dd_start = trade.entry_date
                max_dd_duration = (trade.exit_date - current_dd_start).days

    return (max_dd, max_dd_duration)


def _calculate_sharpe(daily_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio

    Args:
        daily_returns: Daily portfolio returns
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        Annualized Sharpe ratio
    """
    if daily_returns is None or len(daily_returns) < 2:
        return 0

    excess_returns = daily_returns - risk_free_rate / 252

    if excess_returns.std() == 0:
        return 0

    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    return sharpe


def _calculate_sortino(daily_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino ratio (uses downside deviation)

    Args:
        daily_returns: Daily portfolio returns
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        Annualized Sortino ratio
    """
    if daily_returns is None or len(daily_returns) < 2:
        return 0

    excess_returns = daily_returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0

    sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    return sortino


def _empty_metrics(initial_capital: float) -> PerformanceMetrics:
    """Return empty metrics when no trades"""
    return PerformanceMetrics(
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        win_rate=0,
        total_return=0,
        total_profit=0,
        avg_win=0,
        avg_loss=0,
        profit_factor=0,
        max_drawdown=0,
        max_drawdown_duration=0,
        sharpe_ratio=0,
        sortino_ratio=0,
        avg_trade_duration=0,
        best_trade=0,
        worst_trade=0,
        start_date='N/A',
        end_date='N/A',
        initial_capital=initial_capital,
        final_capital=initial_capital
    )


def format_metrics_report(metrics: PerformanceMetrics) -> str:
    """Format metrics as a readable report"""

    report = f"""
================================================================================
                        BACKTEST PERFORMANCE REPORT
================================================================================

Period: {metrics.start_date} to {metrics.end_date}

CAPITAL
-------
Initial Capital:  ${metrics.initial_capital:,.2f}
Final Capital:    ${metrics.final_capital:,.2f}
Total Return:     {metrics.total_return:+.2f}%
Total Profit:     ${metrics.total_profit:+,.2f}

TRADE STATISTICS
----------------
Total Trades:     {metrics.total_trades}
Winning Trades:   {metrics.winning_trades} ({metrics.win_rate:.1f}%)
Losing Trades:    {metrics.losing_trades}

Average Win:      {metrics.avg_win:+.2f}%
Average Loss:     {metrics.avg_loss:.2f}%
Profit Factor:    {metrics.profit_factor:.2f}

Best Trade:       {metrics.best_trade:+.2f}%
Worst Trade:      {metrics.worst_trade:.2f}%
Avg Duration:     {metrics.avg_trade_duration:.1f} days

RISK METRICS
------------
Max Drawdown:     {metrics.max_drawdown:.2f}%
DD Duration:      {metrics.max_drawdown_duration} days
Sharpe Ratio:     {metrics.sharpe_ratio:.2f}
Sortino Ratio:    {metrics.sortino_ratio:.2f}

================================================================================
"""
    return report
