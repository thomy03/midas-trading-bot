"""
Strategy Sandbox - Validation environment for new strategies.

Before a strategy goes into production, it must pass:
1. Backtest validation (6 months historical data)
2. Paper trading validation (min 3 trades, 1 week)

This module handles the validation lifecycle.
"""

import os
import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging
import asyncio

from .strategy_composer import (
    StrategyDefinition,
    StrategyStatus,
    IndicatorCondition,
    ConditionLogic,
)
from src.indicators import get_indicator_registry, compute_indicator

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from backtesting a strategy"""
    strategy_name: str
    start_date: str
    end_date: str

    # Performance metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    total_pnl_percent: float = 0.0
    avg_win_percent: float = 0.0
    avg_loss_percent: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_percent: float = 0.0

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    avg_hold_days: float = 0.0

    # Validation
    passed: bool = False
    failure_reason: str = ""

    # Trade details
    trades: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PaperTradeResult:
    """Results from paper trading a strategy"""
    strategy_name: str
    start_date: str
    end_date: str

    total_trades: int = 0
    winning_trades: int = 0
    win_rate: float = 0.0
    total_pnl_percent: float = 0.0

    # Validation
    passed: bool = False
    failure_reason: str = ""

    trades: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SandboxConfig:
    """Configuration for the sandbox"""
    backtest_months: int = 6
    min_backtest_win_rate: float = 0.45
    min_backtest_sharpe: float = 0.5
    min_backtest_trades: int = 10
    max_backtest_drawdown: float = 0.20

    paper_trading_days: int = 7
    min_paper_trades: int = 3
    min_paper_win_rate: float = 0.40

    initial_capital: float = 10000.0
    position_size_percent: float = 0.10  # 10% per trade


class StrategySandbox:
    """
    Validation sandbox for new trading strategies.

    Lifecycle:
    1. Strategy proposed by LLM
    2. Backtest on historical data
    3. If backtest passes -> Paper trading
    4. If paper trading passes -> Production ready
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.registry = get_indicator_registry()

        # Storage paths
        self.data_dir = "data/sandbox"
        os.makedirs(self.data_dir, exist_ok=True)

        # Paper trading state
        self._paper_trades: Dict[str, List[Dict]] = {}
        self._paper_start_dates: Dict[str, datetime] = {}

    async def validate_strategy(
        self,
        strategy: StrategyDefinition,
        symbols: List[str] = None
    ) -> Tuple[bool, str, Dict]:
        """
        Full validation pipeline for a strategy.

        Returns:
            Tuple of (passed, reason, results_dict)
        """
        symbols = symbols or ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
                              'META', 'TSLA', 'AMD', 'CRM', 'NFLX']

        logger.info(f"Starting validation for strategy: {strategy.name}")

        # Phase 1: Backtest
        strategy.status = StrategyStatus.BACKTESTING
        backtest_result = await self.run_backtest(strategy, symbols)

        if not backtest_result.passed:
            strategy.status = StrategyStatus.ARCHIVED
            return False, f"Backtest failed: {backtest_result.failure_reason}", backtest_result.to_dict()

        logger.info(f"Backtest passed: win_rate={backtest_result.win_rate:.1%}, "
                    f"sharpe={backtest_result.sharpe_ratio:.2f}")

        # Phase 2: Paper trading
        strategy.status = StrategyStatus.PAPER_TRADING
        # Note: Paper trading is ongoing, not immediate
        # We just initialize it here
        self._start_paper_trading(strategy)

        return True, "Backtest passed, paper trading started", {
            "backtest": backtest_result.to_dict(),
            "paper_status": "started"
        }

    async def run_backtest(
        self,
        strategy: StrategyDefinition,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> BacktestResult:
        """
        Run a backtest on the strategy.

        Args:
            strategy: The strategy to test
            symbols: List of symbols to test on
            start_date: Start date (default: 6 months ago)
            end_date: End date (default: today)

        Returns:
            BacktestResult with performance metrics
        """
        # Default dates
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_dt = datetime.now() - timedelta(days=self.config.backtest_months * 30)
            start_date = start_dt.strftime('%Y-%m-%d')

        result = BacktestResult(
            strategy_name=strategy.name,
            start_date=start_date,
            end_date=end_date
        )

        all_trades = []
        equity_curve = [self.config.initial_capital]

        for symbol in symbols:
            try:
                # Fetch historical data
                df = await self._fetch_data(symbol, start_date, end_date)
                if df is None or len(df) < 50:
                    continue

                # Compute all required indicators
                df = self._compute_strategy_indicators(df, strategy)

                # Generate signals
                trades = self._backtest_symbol(df, strategy, symbol)
                all_trades.extend(trades)

            except Exception as e:
                logger.warning(f"Error backtesting {symbol}: {e}")
                continue

        if not all_trades:
            result.failure_reason = "No trades generated"
            return result

        # Calculate metrics
        result.trades = all_trades
        result.total_trades = len(all_trades)
        result.winning_trades = sum(1 for t in all_trades if t['pnl_percent'] > 0)
        result.losing_trades = sum(1 for t in all_trades if t['pnl_percent'] <= 0)
        result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0

        # P&L metrics
        pnls = [t['pnl_percent'] for t in all_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        result.total_pnl_percent = sum(pnls)
        result.avg_win_percent = np.mean(wins) if wins else 0
        result.avg_loss_percent = np.mean(losses) if losses else 0

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0.001
        result.profit_factor = total_wins / total_losses

        # Sharpe ratio (simplified)
        if len(pnls) > 1:
            result.sharpe_ratio = np.mean(pnls) / (np.std(pnls) + 0.001) * np.sqrt(252)
        else:
            result.sharpe_ratio = 0

        # Sortino ratio
        downside = [p for p in pnls if p < 0]
        if downside:
            downside_std = np.std(downside)
            result.sortino_ratio = np.mean(pnls) / (downside_std + 0.001) * np.sqrt(252)
        else:
            result.sortino_ratio = result.sharpe_ratio

        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        result.max_drawdown_percent = abs(min(drawdowns)) if len(drawdowns) > 0 else 0

        # Average hold time
        hold_days = [t.get('hold_days', 5) for t in all_trades]
        result.avg_hold_days = np.mean(hold_days) if hold_days else 0

        # Validation
        result.passed = self._validate_backtest(result)
        if not result.passed and not result.failure_reason:
            result.failure_reason = self._get_failure_reason(result)

        return result

    def _compute_strategy_indicators(
        self,
        df: pd.DataFrame,
        strategy: StrategyDefinition
    ) -> pd.DataFrame:
        """Compute all indicators required by the strategy"""
        df = df.copy()

        # Get all unique indicators from conditions
        indicators = set()
        for cond in strategy.entry_conditions + strategy.exit_conditions:
            indicators.add((cond.indicator, json.dumps(cond.params, sort_keys=True)))

        # Compute each indicator
        for indicator_name, params_json in indicators:
            params = json.loads(params_json)
            try:
                df = compute_indicator(indicator_name, df, **params)
            except Exception as e:
                logger.warning(f"Error computing {indicator_name}: {e}")

        return df

    def _evaluate_conditions(
        self,
        row: pd.Series,
        conditions: List[IndicatorCondition],
        logic: ConditionLogic
    ) -> bool:
        """Evaluate if conditions are met for a given row"""
        if not conditions:
            return False

        results = []
        for cond in conditions:
            try:
                # Get indicator value
                col = cond.output_column or cond.indicator
                if col not in row.index:
                    # Try alternative column names
                    possible_cols = [c for c in row.index if cond.indicator in c]
                    if possible_cols:
                        col = possible_cols[0]
                    else:
                        results.append(False)
                        continue

                value = row[col]
                if pd.isna(value):
                    results.append(False)
                    continue

                # Compare
                compare_value = cond.value
                if isinstance(compare_value, str) and compare_value in row.index:
                    compare_value = row[compare_value]

                if cond.operator == '>':
                    results.append(value > compare_value)
                elif cond.operator == '<':
                    results.append(value < compare_value)
                elif cond.operator == '>=':
                    results.append(value >= compare_value)
                elif cond.operator == '<=':
                    results.append(value <= compare_value)
                elif cond.operator == '==':
                    results.append(value == compare_value)
                elif cond.operator in ['crosses_above', 'crosses_below']:
                    # For crossover, we need previous value - handle in _backtest_symbol
                    results.append(True)  # Placeholder
                else:
                    results.append(False)

            except Exception as e:
                logger.debug(f"Condition evaluation error: {e}")
                results.append(False)

        if logic == ConditionLogic.AND:
            return all(results)
        else:  # OR
            return any(results)

    def _backtest_symbol(
        self,
        df: pd.DataFrame,
        strategy: StrategyDefinition,
        symbol: str
    ) -> List[Dict]:
        """Backtest a strategy on a single symbol"""
        trades = []
        position = None

        for i in range(20, len(df)):  # Start after indicator warmup
            row = df.iloc[i]
            prev_row = df.iloc[i-1]

            if position is None:
                # Check entry conditions
                if self._evaluate_conditions(row, strategy.entry_conditions, strategy.entry_logic):
                    # Open position
                    entry_price = row['Close']
                    stop_loss = self._calculate_stop_loss(df.iloc[:i+1], strategy, entry_price)

                    position = {
                        'symbol': symbol,
                        'entry_date': str(row.name),
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'entry_idx': i
                    }

            else:
                # Check exit conditions
                exit_signal = self._evaluate_conditions(row, strategy.exit_conditions, strategy.exit_logic)
                stop_hit = row['Low'] <= position['stop_loss']
                take_profit = (row['High'] - position['entry_price']) / position['entry_price'] >= strategy.take_profit_value

                if exit_signal or stop_hit or take_profit:
                    # Close position
                    if stop_hit:
                        exit_price = position['stop_loss']
                        exit_reason = 'stop_loss'
                    elif take_profit:
                        exit_price = position['entry_price'] * (1 + strategy.take_profit_value)
                        exit_reason = 'take_profit'
                    else:
                        exit_price = row['Close']
                        exit_reason = 'signal'

                    pnl_percent = (exit_price - position['entry_price']) / position['entry_price'] * 100
                    hold_days = i - position['entry_idx']

                    trades.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'entry_price': position['entry_price'],
                        'exit_date': str(row.name),
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl_percent': pnl_percent,
                        'hold_days': hold_days
                    })

                    position = None

        return trades

    def _calculate_stop_loss(
        self,
        df: pd.DataFrame,
        strategy: StrategyDefinition,
        entry_price: float
    ) -> float:
        """Calculate stop loss price based on strategy settings"""
        if strategy.stop_loss_type == 'atr':
            # ATR-based stop loss
            if 'ATR' not in df.columns:
                df = compute_indicator('ATR', df, period=14)
            atr = df['ATR'].iloc[-1]
            return entry_price - (atr * strategy.stop_loss_value)

        elif strategy.stop_loss_type == 'percent':
            return entry_price * (1 - strategy.stop_loss_value)

        elif strategy.stop_loss_type == 'support':
            # Use recent low as support
            lookback = min(20, len(df))
            support = df['Low'].iloc[-lookback:].min()
            return support * 0.99  # 1% below support

        else:
            # Default: 5% stop loss
            return entry_price * 0.95

    def _validate_backtest(self, result: BacktestResult) -> bool:
        """Check if backtest results meet minimum requirements"""
        if result.total_trades < self.config.min_backtest_trades:
            result.failure_reason = f"Too few trades: {result.total_trades} < {self.config.min_backtest_trades}"
            return False

        if result.win_rate < self.config.min_backtest_win_rate:
            result.failure_reason = f"Win rate too low: {result.win_rate:.1%} < {self.config.min_backtest_win_rate:.1%}"
            return False

        if result.sharpe_ratio < self.config.min_backtest_sharpe:
            result.failure_reason = f"Sharpe too low: {result.sharpe_ratio:.2f} < {self.config.min_backtest_sharpe:.2f}"
            return False

        if result.max_drawdown_percent > self.config.max_backtest_drawdown * 100:
            result.failure_reason = f"Drawdown too high: {result.max_drawdown_percent:.1f}% > {self.config.max_backtest_drawdown * 100:.1f}%"
            return False

        return True

    def _get_failure_reason(self, result: BacktestResult) -> str:
        """Generate failure reason message"""
        issues = []
        if result.total_trades < self.config.min_backtest_trades:
            issues.append(f"trades={result.total_trades}")
        if result.win_rate < self.config.min_backtest_win_rate:
            issues.append(f"win_rate={result.win_rate:.1%}")
        if result.sharpe_ratio < self.config.min_backtest_sharpe:
            issues.append(f"sharpe={result.sharpe_ratio:.2f}")
        return f"Failed metrics: {', '.join(issues)}"

    async def _fetch_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch historical data for a symbol"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                return None

            # Standardize column names
            df.columns = [c.title() for c in df.columns]
            return df

        except Exception as e:
            logger.warning(f"Error fetching data for {symbol}: {e}")
            return None

    def _start_paper_trading(self, strategy: StrategyDefinition):
        """Initialize paper trading for a strategy"""
        self._paper_trades[strategy.name] = []
        self._paper_start_dates[strategy.name] = datetime.now()
        logger.info(f"Paper trading started for {strategy.name}")

    def record_paper_trade(
        self,
        strategy_name: str,
        trade: Dict[str, Any]
    ):
        """Record a paper trade for a strategy"""
        if strategy_name not in self._paper_trades:
            self._paper_trades[strategy_name] = []

        trade['timestamp'] = datetime.now().isoformat()
        self._paper_trades[strategy_name].append(trade)

        # Save to file
        self._save_paper_trades(strategy_name)

    def _save_paper_trades(self, strategy_name: str):
        """Save paper trades to file"""
        filepath = os.path.join(self.data_dir, f"paper_{strategy_name}.json")
        with open(filepath, 'w') as f:
            json.dump(self._paper_trades.get(strategy_name, []), f, indent=2)

    def check_paper_trading_status(
        self,
        strategy_name: str
    ) -> Tuple[bool, Optional[PaperTradeResult]]:
        """
        Check if paper trading period is complete and if it passed.

        Returns:
            Tuple of (is_complete, result_if_complete)
        """
        if strategy_name not in self._paper_start_dates:
            return False, None

        start_date = self._paper_start_dates[strategy_name]
        elapsed_days = (datetime.now() - start_date).days

        if elapsed_days < self.config.paper_trading_days:
            return False, None

        # Evaluate results
        trades = self._paper_trades.get(strategy_name, [])
        result = PaperTradeResult(
            strategy_name=strategy_name,
            start_date=start_date.isoformat(),
            end_date=datetime.now().isoformat(),
            total_trades=len(trades),
            trades=trades
        )

        if trades:
            result.winning_trades = sum(1 for t in trades if t.get('pnl_percent', 0) > 0)
            result.win_rate = result.winning_trades / len(trades)
            result.total_pnl_percent = sum(t.get('pnl_percent', 0) for t in trades)

        # Validate
        if result.total_trades < self.config.min_paper_trades:
            result.failure_reason = f"Too few trades: {result.total_trades} < {self.config.min_paper_trades}"
            result.passed = False
        elif result.win_rate < self.config.min_paper_win_rate:
            result.failure_reason = f"Win rate too low: {result.win_rate:.1%}"
            result.passed = False
        else:
            result.passed = True

        return True, result


# Singleton instance
_sandbox_instance: Optional[StrategySandbox] = None


def get_strategy_sandbox(config: Optional[SandboxConfig] = None) -> StrategySandbox:
    """Get or create the global StrategySandbox instance"""
    global _sandbox_instance
    if _sandbox_instance is None:
        _sandbox_instance = StrategySandbox(config)
    return _sandbox_instance
