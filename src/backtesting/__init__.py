"""
Backtesting Module for TradingBot V3

Provides historical simulation of trading strategies:
- Trade simulation based on RSI breakout + EMA signals
- Performance metrics calculation
- Report generation
- Realistic portfolio simulation with capital management
"""

from .backtester import Backtester, BacktestResult, BacktestConfig
from .metrics import calculate_metrics, PerformanceMetrics, Trade
from .portfolio_simulator import (
    PortfolioSimulator,
    SimulationResult,
    Position,
    PortfolioState,
    run_realistic_simulation
)

__all__ = [
    # Basic backtester
    'Backtester',
    'BacktestResult',
    'BacktestConfig',
    # Metrics
    'calculate_metrics',
    'PerformanceMetrics',
    'Trade',
    # Realistic simulation
    'PortfolioSimulator',
    'SimulationResult',
    'Position',
    'PortfolioState',
    'run_realistic_simulation'
]
