"""
Backtester for TradingBot V3

Simulates trading strategy on historical data:
1. Scans for signals (RSI breakout + EMA conditions)
2. Simulates entries at signal price
3. Tracks positions and exits (stop-loss, take-profit, signal exit)
4. Calculates performance metrics
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from .metrics import Trade, PerformanceMetrics, calculate_metrics, format_metrics_report

# Import trading components (trendline_analysis removed in V6+)
try:
    from trendline_analysis.core.rsi_breakout_analyzer import RSIBreakoutAnalyzer
    from trendline_analysis.core.enhanced_rsi_breakout_analyzer import EnhancedRSIBreakoutAnalyzer
except ImportError:
    RSIBreakoutAnalyzer = None
    EnhancedRSIBreakoutAnalyzer = None
from src.indicators.ema_analyzer import ema_analyzer
from src.utils.confidence_scorer import confidence_scorer
from src.utils.position_sizing import calculate_position_size


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 10000
    max_positions: int = 20  # Maximum concurrent positions (increased from 5)
    position_size_pct: float = 0.10  # Max 10% per position

    # Signal thresholds
    min_confidence_score: float = 55  # Minimum score to enter
    require_volume_confirmation: bool = True
    min_volume_ratio: float = 1.0

    # Exit conditions - ADAPTIVE SYSTEM (replaces fixed %)
    use_adaptive_exit: bool = True  # Use Chandelier + Bollinger + Volume
    chandelier_atr_period: int = 22  # ATR period for Chandelier Exit
    chandelier_multiplier: float = 3.0  # ATR multiplier (standard)
    max_hold_days: int = 120  # Maximum holding period (increased from 60)

    # Legacy exit options (disabled by default with adaptive exit)
    use_trailing_stop: bool = False  # Deprecated - use adaptive_exit instead
    trailing_stop_pct: float = 0.05  # Deprecated
    take_profit_pct: float = 0.0  # Disabled - exit via adaptive indicators

    # Strategy settings
    use_enhanced_detector: bool = True
    precision_mode: str = 'medium'
    timeframe: str = 'weekly'

    # Dual-timeframe (like real screener)
    use_daily_fallback: bool = True  # If weekly has no signal but EMAs bullish, check daily
    min_ema_conditions_for_fallback: int = 2  # Need at least 2/3 EMA conditions to trigger fallback

    # Sector Momentum Filter
    use_sector_filter: bool = False  # Only take signals in bullish sectors (vs SPY)
    min_sector_momentum: float = 0.0  # Minimum sector performance vs SPY (%)

    # Market Narrative Filter
    use_narrative_filter: bool = False  # Boost/filter signals aligned with market narratives
    narrative_boost_scores: bool = True  # Add narrative boost to confidence scores
    require_narrative_alignment: bool = False  # Only take signals aligned with current narrative

    # V8.2: Transaction costs for realistic backtesting
    transaction_fee_pct: float = 0.001   # 0.1% per trade (entry + exit)
    slippage_pct: float = 0.0005         # 0.05% slippage per trade
    use_realistic_costs: bool = True     # Enable fees + slippage


@dataclass
class BacktestResult:
    """Complete backtest results"""
    config: BacktestConfig
    metrics: PerformanceMetrics
    trades: List[Trade]
    equity_curve: pd.Series
    signals_detected: int
    signals_taken: int

    def __str__(self):
        return format_metrics_report(self.metrics)


class Backtester:
    """
    Historical strategy backtester

    Usage:
        backtester = Backtester(config)
        result = backtester.run(symbols, start_date, end_date)
        print(result)
    """

    def __init__(self, config: BacktestConfig = None):
        """
        Initialize backtester

        Args:
            config: Backtesting configuration (uses defaults if None)
        """
        self.config = config or BacktestConfig()

        # Initialize analyzer
        if self.config.use_enhanced_detector:
            self.rsi_analyzer = EnhancedRSIBreakoutAnalyzer(
                precision_mode=self.config.precision_mode
            )
        else:
            self.rsi_analyzer = RSIBreakoutAnalyzer()

        self.ema_analyzer = ema_analyzer

    def run(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        data_provider=None
    ) -> BacktestResult:
        """
        Run backtest on multiple symbols

        Args:
            symbols: List of stock symbols to test
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data_provider: Optional custom data provider (uses yfinance if None)

        Returns:
            BacktestResult with complete analysis
        """
        from src.data.market_data import market_data_fetcher

        data_fetcher = data_provider or market_data_fetcher

        all_trades = []
        signals_detected = 0
        signals_taken = 0

        # Track portfolio state
        capital = self.config.initial_capital
        positions = {}  # symbol -> position info
        equity_history = []

        # Convert dates
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        print(f"Running backtest from {start_date} to {end_date}")
        print(f"Symbols: {len(symbols)}")
        print(f"Config: {self.config.precision_mode} precision, "
              f"min score={self.config.min_confidence_score}")
        if self.config.use_realistic_costs:
            print(f"Costs: fee={self.config.transaction_fee_pct*100:.2f}%/trade, "
                  f"slippage={self.config.slippage_pct*100:.3f}%/trade")

        # Process each symbol
        for symbol in symbols:
            try:
                symbol_trades, symbol_signals = self._backtest_symbol(
                    symbol=symbol,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    data_fetcher=data_fetcher,
                    capital=capital
                )

                signals_detected += symbol_signals
                all_trades.extend(symbol_trades)

            except Exception as e:
                print(f"Error backtesting {symbol}: {e}")
                continue

        signals_taken = len(all_trades)

        # Calculate metrics
        metrics = calculate_metrics(
            trades=all_trades,
            initial_capital=self.config.initial_capital
        )

        # Build equity curve
        equity_curve = self._build_equity_curve(all_trades, start_dt, end_dt)

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=all_trades,
            equity_curve=equity_curve,
            signals_detected=signals_detected,
            signals_taken=signals_taken
        )

    def _backtest_symbol(
        self,
        symbol: str,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
        data_fetcher,
        capital: float
    ) -> Tuple[List[Trade], int]:
        """
        Backtest a single symbol using walk-forward analysis

        Implements dual-timeframe logic like the real screener:
        1. Check weekly first
        2. If no weekly signal but EMAs bullish (≥2/3), check daily fallback

        Returns:
            (trades, signals_count)
        """
        trades = []
        signals_count = 0

        # Fetch weekly data (always needed for primary check)
        df_weekly = data_fetcher.get_historical_data(
            symbol,
            period='5y',
            interval='1wk'
        )

        if df_weekly is None or len(df_weekly) < 100:
            return ([], 0)

        # Normalize index
        if df_weekly.index.tz is not None:
            df_weekly.index = df_weekly.index.tz_localize(None)

        # Add EMAs to weekly
        df_weekly = self.ema_analyzer.calculate_emas(df_weekly)

        # Fetch daily data if daily fallback is enabled
        df_daily = None
        if self.config.use_daily_fallback:
            df_daily = data_fetcher.get_historical_data(
                symbol,
                period='5y',
                interval='1d'
            )
            if df_daily is not None and len(df_daily) >= 100:
                if df_daily.index.tz is not None:
                    df_daily.index = df_daily.index.tz_localize(None)
                df_daily = self.ema_analyzer.calculate_emas(df_daily)
            else:
                df_daily = None

        # Walk forward through weekly data
        lookback_weekly = 104
        lookback_daily = 252

        current_position = None

        for i in range(lookback_weekly, len(df_weekly)):
            current_date = df_weekly.index[i]

            # Skip if outside backtest period
            if current_date < start_dt or current_date > end_dt:
                continue

            current_price = df_weekly['Close'].iloc[i]

            # Check for exit if we have a position
            if current_position is not None:
                # Use the appropriate dataframe for exit check
                exit_df = df_daily if current_position.get('timeframe') == 'daily' and df_daily is not None else df_weekly
                exit_idx = self._find_nearest_index(exit_df, current_date)

                if exit_idx is not None:
                    exit_signal, exit_reason = self._check_exit(
                        current_position, exit_df, exit_idx
                    )

                    if exit_signal:
                        exit_price = exit_df['Close'].iloc[exit_idx]
                        trade = self._close_position(
                            current_position, exit_price, current_date, exit_reason
                        )
                        trades.append(trade)
                        current_position = None

            # Check for entry if no position
            if current_position is None:
                entry_signal = self._check_entry_dual_timeframe(
                    df_weekly, df_daily, i, current_date, lookback_weekly, lookback_daily
                )

                if entry_signal is not None:
                    signals_count += 1

                    # Check if signal meets our criteria
                    if self._validate_signal(entry_signal):
                        # Use price from the signal's timeframe
                        entry_price = entry_signal.get('entry_price', current_price)
                        entry_df = df_daily if entry_signal.get('timeframe') == 'daily' and df_daily is not None else df_weekly
                        entry_idx = self._find_nearest_index(entry_df, current_date) or i

                        current_position = self._open_position(
                            symbol=symbol,
                            entry_date=current_date,
                            entry_price=entry_price,
                            signal=entry_signal,
                            df=entry_df,
                            idx=entry_idx,
                            capital=capital
                        )
                        if current_position:
                            current_position['timeframe'] = entry_signal.get('timeframe', 'weekly')

        # Close any remaining position at end
        if current_position is not None:
            trade = self._close_position(
                current_position,
                df_weekly['Close'].iloc[-1],
                df_weekly.index[-1],
                'end_of_period'
            )
            trades.append(trade)

        return (trades, signals_count)

    def _find_nearest_index(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Optional[int]:
        """Find the nearest index in df for a given date"""
        if df is None or df.empty:
            return None

        # Find dates <= target_date
        mask = df.index <= target_date
        if not mask.any():
            return None

        nearest_date = df.index[mask][-1]
        return df.index.get_loc(nearest_date)

    def _check_entry_dual_timeframe(
        self,
        df_weekly: pd.DataFrame,
        df_daily: Optional[pd.DataFrame],
        weekly_idx: int,
        current_date: pd.Timestamp,
        lookback_weekly: int,
        lookback_daily: int
    ) -> Optional[Dict]:
        """
        Check for entry signal using dual-timeframe logic:
        1. Check weekly first
        2. If no weekly signal but EMAs bullish, check daily fallback

        Returns:
            Signal dict if entry detected, None otherwise
        """
        # 1. Check weekly signal
        weekly_signal = self._check_entry(df_weekly, weekly_idx, lookback_weekly)

        if weekly_signal is not None:
            weekly_signal['timeframe'] = 'weekly'
            weekly_signal['entry_price'] = df_weekly['Close'].iloc[weekly_idx]
            return weekly_signal

        # 2. Check if daily fallback should be triggered
        if not self.config.use_daily_fallback or df_daily is None:
            return None

        # Check EMA alignment on weekly
        row = df_weekly.iloc[weekly_idx]
        ema_24 = row.get('EMA_24', 0)
        ema_38 = row.get('EMA_38', 0)
        ema_62 = row.get('EMA_62', 0)

        ema_conditions = sum([
            ema_24 > ema_38,
            ema_24 > ema_62,
            ema_38 > ema_62
        ])

        # Only fallback to daily if EMAs are bullish
        if ema_conditions < self.config.min_ema_conditions_for_fallback:
            return None

        # 3. Find corresponding daily index
        daily_idx = self._find_nearest_index(df_daily, current_date)
        if daily_idx is None or daily_idx < lookback_daily:
            return None

        # 4. Check daily signal
        daily_signal = self._check_entry(df_daily, daily_idx, lookback_daily)

        if daily_signal is not None:
            daily_signal['timeframe'] = 'daily'
            daily_signal['entry_price'] = df_daily['Close'].iloc[daily_idx]
            daily_signal['weekly_ema_conditions'] = ema_conditions
            return daily_signal

        return None

    def _check_entry(self, df: pd.DataFrame, idx: int, lookback: int) -> Optional[Dict]:
        """
        Check for entry signal at given index

        Returns:
            Signal dict if entry detected, None otherwise
        """
        # Use data up to current index for analysis (no lookahead)
        analysis_df = df.iloc[:idx + 1].copy()

        # Check RSI breakout
        result = self.rsi_analyzer.analyze(analysis_df, lookback_periods=lookback)

        if result is None or not result.has_rsi_breakout:
            return None

        # Check EMA alignment
        row = analysis_df.iloc[-1]
        ema_24 = row.get('EMA_24', 0)
        ema_38 = row.get('EMA_38', 0)
        ema_62 = row.get('EMA_62', 0)

        ema_conditions = sum([
            ema_24 > ema_38,
            ema_24 > ema_62,
            ema_38 > ema_62
        ])

        if ema_conditions < 2:
            return None  # EMAs not bullish

        # Calculate confidence score
        current_price = row['Close']

        # Prepare breakout data
        rsi_breakout_data = {
            'strength': result.rsi_breakout.strength,
            'age_in_periods': result.rsi_breakout.age_in_periods,
            'index': result.rsi_breakout.index,
            'volume_ratio': getattr(result.rsi_breakout, 'volume_ratio', 1.0)
        }

        rsi_trendline_data = {
            'r_squared': result.rsi_trendline.r_squared
        } if result.rsi_trendline else None

        # Find nearest support level (simplified)
        support_level = min(ema_24, ema_38, ema_62)

        confidence = confidence_scorer.calculate_score(
            df=analysis_df,
            ema_24=ema_24,
            ema_38=ema_38,
            ema_62=ema_62,
            current_price=current_price,
            support_level=support_level,
            rsi_breakout=rsi_breakout_data,
            rsi_trendline=rsi_trendline_data
        )

        return {
            'signal': result.signal,
            'confidence_score': confidence.total_score,
            'confidence_signal': confidence.signal,
            'rsi_breakout': result.rsi_breakout,
            'rsi_trendline': result.rsi_trendline,
            'volume_ratio': rsi_breakout_data['volume_ratio'],
            'ema_conditions': ema_conditions
        }

    def _validate_signal(self, signal: Dict) -> bool:
        """Check if signal meets entry criteria"""
        # Check confidence score
        if signal['confidence_score'] < self.config.min_confidence_score:
            return False

        # Check volume confirmation
        if self.config.require_volume_confirmation:
            if signal['volume_ratio'] < self.config.min_volume_ratio:
                return False

        return True

    def _open_position(
        self,
        symbol: str,
        entry_date: pd.Timestamp,
        entry_price: float,
        signal: Dict,
        df: pd.DataFrame,
        idx: int,
        capital: float
    ) -> Dict:
        """Open a new position"""
        # V8.2: Apply slippage to entry price (buy higher than quoted)
        if self.config.use_realistic_costs:
            entry_price = entry_price * (1 + self.config.slippage_pct)

        # Calculate position size
        position_value = capital * self.config.position_size_pct
        shares = int(position_value / entry_price)

        if shares <= 0:
            shares = 1

        # Calculate stop-loss from RSI peaks
        stop_loss = entry_price * 0.95  # Default 5%

        if signal['rsi_trendline'] is not None:
            peak_indices = signal['rsi_trendline'].peak_indices

            # Find lowest price at RSI peaks
            valid_indices = [i for i in peak_indices if i < len(df)]
            if valid_indices:
                peak_lows = [df['Low'].iloc[i] for i in valid_indices]
                stop_loss = min(peak_lows) * 0.99  # 1% below lowest peak

                # Validate stop-loss range (2-20%)
                stop_pct = (entry_price - stop_loss) / entry_price * 100
                if stop_pct < 2:
                    stop_loss = entry_price * 0.97
                elif stop_pct > 20:
                    stop_loss = entry_price * 0.85

        return {
            'symbol': symbol,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'shares': shares,
            'stop_loss': stop_loss,
            'signal_strength': signal['signal'],
            'confidence_score': signal['confidence_score'],
            'volume_ratio': signal['volume_ratio'],
            'highest_price': entry_price  # For trailing stop
        }

    def _check_exit(self, position: Dict, df: pd.DataFrame, idx: int) -> Tuple[bool, str]:
        """
        Check exit conditions

        Returns:
            (should_exit, reason)
        """
        current_price = df['Close'].iloc[idx]
        current_high = df['High'].iloc[idx]
        current_low = df['Low'].iloc[idx]

        entry_price = position['entry_price']
        stop_loss = position['stop_loss']

        # Update highest price for trailing stop
        if current_high > position['highest_price']:
            position['highest_price'] = current_high

        # Check stop-loss
        if current_low <= stop_loss:
            return (True, 'stop_loss')

        # Check trailing stop
        if self.config.use_trailing_stop:
            trailing_stop = position['highest_price'] * (1 - self.config.trailing_stop_pct)
            if current_low <= trailing_stop:
                return (True, 'trailing_stop')

        # Check take-profit
        profit_pct = (current_price - entry_price) / entry_price
        if profit_pct >= self.config.take_profit_pct:
            return (True, 'take_profit')

        # Check max hold period
        hold_days = (df.index[idx] - position['entry_date']).days
        if hold_days >= self.config.max_hold_days:
            return (True, 'max_hold_period')

        return (False, '')

    def _close_position(
        self,
        position: Dict,
        exit_price: float,
        exit_date: pd.Timestamp,
        exit_reason: str
    ) -> Trade:
        """Close position and create Trade record"""
        entry_price = position['entry_price']
        shares = position['shares']

        # V8.2: Apply slippage on exit (sell lower than quoted)
        if self.config.use_realistic_costs:
            exit_price = exit_price * (1 - self.config.slippage_pct)

        # V8.2: Calculate transaction fees (entry + exit)
        total_fees = 0.0
        if self.config.use_realistic_costs:
            entry_fee = entry_price * shares * self.config.transaction_fee_pct
            exit_fee = exit_price * shares * self.config.transaction_fee_pct
            total_fees = entry_fee + exit_fee

        profit_loss = (exit_price - entry_price) * shares - total_fees
        profit_loss_pct = ((exit_price - entry_price) / entry_price * 100) - (self.config.transaction_fee_pct * 200 if self.config.use_realistic_costs else 0)

        return Trade(
            symbol=position['symbol'],
            entry_date=position['entry_date'],
            entry_price=entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            shares=shares,
            stop_loss=position['stop_loss'],
            profit_loss=profit_loss,
            profit_loss_pct=profit_loss_pct,
            exit_reason=exit_reason,
            signal_strength=position['signal_strength'],
            confidence_score=position['confidence_score'],
            volume_ratio=position['volume_ratio']
        )

    def _build_equity_curve(
        self,
        trades: List[Trade],
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp
    ) -> pd.Series:
        """Build equity curve from trades"""
        if not trades:
            date_range = pd.date_range(start_dt, end_dt, freq='D')
            return pd.Series(
                [self.config.initial_capital] * len(date_range),
                index=date_range
            )

        # Sort trades by exit date
        sorted_trades = sorted(trades, key=lambda t: t.exit_date)

        # Build cumulative equity (aggregate by date to avoid duplicates)
        equity = self.config.initial_capital
        equity_by_date = {start_dt: equity}

        for trade in sorted_trades:
            equity += trade.profit_loss
            # Use the latest equity value for each date
            equity_by_date[trade.exit_date] = equity

        equity_by_date[end_dt] = equity

        # Create series from unique dates
        dates = sorted(equity_by_date.keys())
        values = [equity_by_date[d] for d in dates]
        equity_series = pd.Series(values, index=pd.DatetimeIndex(dates))

        # Resample to daily, forward-filling gaps
        try:
            return equity_series.resample('D').ffill()
        except Exception:
            # Fallback: return as-is if resampling fails
            return equity_series


def run_quick_backtest(
    symbols: List[str],
    start_date: str = '2023-01-01',
    end_date: str = '2024-12-01',
    initial_capital: float = 10000
) -> BacktestResult:
    """
    Quick backtest helper function

    Args:
        symbols: List of symbols to test
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital

    Returns:
        BacktestResult
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        use_enhanced_detector=True,
        precision_mode='medium',
        min_confidence_score=55,
        require_volume_confirmation=True
    )

    backtester = Backtester(config)
    return backtester.run(symbols, start_date, end_date)
