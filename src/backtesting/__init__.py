"""
Backtesting Module for MIDAS

Provides historical simulation of trading strategies:
- V7 backtester: Trained ML model + dynamic sizing + risk management (Mode E)
- V6.2 backtester: Tests the real 5-pillar system with ML Gate & regime detection
- Legacy backtester: RSI breakout + EMA signals (V3)
- Performance metrics with proper Sharpe calculation
- Transaction costs modeling
- Out-of-sample / walk-forward testing
- Monte Carlo simulation for statistical validation
"""

from .metrics import calculate_metrics, PerformanceMetrics, Trade

# Legacy backtester (requires trendline_analysis package)
try:
    from .backtester import Backtester, BacktestResult, BacktestConfig
    from .portfolio_simulator import (
        PortfolioSimulator,
        SimulationResult,
        Position,
        PortfolioState,
        run_realistic_simulation
    )
except ImportError:
    Backtester = None
    BacktestResult = None
    BacktestConfig = None
    PortfolioSimulator = None
    SimulationResult = None
    Position = None
    PortfolioState = None
    run_realistic_simulation = None

# V6.2 backtester (standalone, no extra dependencies)
from .v6_backtester import (
    V6Backtester,
    V6BacktestResult,
    V6BacktestConfig,
    TransactionCosts,
    ScoringMode,
    MarketRegime,
    run_v6_backtest,
    run_oos_backtest,
    run_abc_comparison,
)

# V7: Monte Carlo (optional)
try:
    from .monte_carlo import MonteCarloSimulator, MonteCarloResult, run_monte_carlo
except ImportError:
    MonteCarloSimulator = None
    MonteCarloResult = None
    run_monte_carlo = None

__all__ = [
    # V6.2 backtester (recommended)
    'V6Backtester',
    'V6BacktestResult',
    'V6BacktestConfig',
    'TransactionCosts',
    'ScoringMode',
    'run_v6_backtest',
    'run_oos_backtest',
    'run_abc_comparison',
    'MarketRegime',
    # Legacy backtester
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
