"""
Portfolio Simulator for TradingBot V3

Realistic day-by-day simulation with proper capital management:
- Chronological simulation respecting available capital
- Max positions enforced at all times
- Signal priority by confidence score
- Proper position tracking with blocked capital
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from .metrics import Trade, PerformanceMetrics, calculate_metrics, format_metrics_report
from .backtester import BacktestConfig

# Import trading components (trendline_analysis removed in V6+)
try:
    from trendline_analysis.core.rsi_breakout_analyzer import RSIBreakoutAnalyzer
    from trendline_analysis.core.enhanced_rsi_breakout_analyzer import EnhancedRSIBreakoutAnalyzer
except ImportError:
    RSIBreakoutAnalyzer = None
    EnhancedRSIBreakoutAnalyzer = None
from src.indicators.ema_analyzer import ema_analyzer
from src.utils.confidence_scorer import confidence_scorer
from src.indicators.adaptive_exit import AdaptiveExitIndicator
from src.utils.sector_analyzer import SectorAnalyzer
from src.utils.narrative_tracker import NarrativeTracker


@dataclass
class Position:
    """Represents an open trading position"""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop_loss: float
    position_value: float  # Entry cost (blocked capital)

    # Signal info
    signal_strength: str
    confidence_score: float
    volume_ratio: float

    # Tracking
    highest_price: float = 0.0
    rsi_peak_indices: List[int] = field(default_factory=list)
    timeframe: str = 'weekly'  # 'weekly' or 'daily' (for dual-timeframe support)

    @property
    def current_value(self) -> float:
        """Current position value based on entry (for capital tracking)"""
        return self.position_value


@dataclass
class PortfolioState:
    """Daily portfolio snapshot"""
    date: pd.Timestamp
    cash: float
    positions_value: float  # Mark-to-market value of positions
    total_value: float
    num_positions: int
    positions: Dict[str, float]  # symbol -> current value


@dataclass
class SimulationResult:
    """Complete simulation results"""
    config: BacktestConfig
    metrics: PerformanceMetrics
    trades: List[Trade]
    equity_curve: pd.Series
    daily_states: List[PortfolioState]
    signals_detected: int
    signals_taken: int
    signals_skipped_capital: int
    signals_skipped_max_positions: int
    signals_skipped_sector: int = 0
    signals_boosted_narrative: int = 0
    signals_skipped_narrative: int = 0
    open_positions: List = field(default_factory=list)

    def __str__(self):
        sector_line = f"Skipped (Sector):     {self.signals_skipped_sector}\n" if self.signals_skipped_sector > 0 else ""
        narrative_line = ""
        if self.signals_boosted_narrative > 0 or self.signals_skipped_narrative > 0:
            narrative_line = f"Boosted (Narrative):  {self.signals_boosted_narrative}\n"
            if self.signals_skipped_narrative > 0:
                narrative_line += f"Skipped (Narrative):  {self.signals_skipped_narrative}\n"
        header = f"""
================================================================================
                      REALISTIC PORTFOLIO SIMULATION
================================================================================
Signals Detected:     {self.signals_detected}
Signals Taken:        {self.signals_taken}
Skipped (Capital):    {self.signals_skipped_capital}
Skipped (Max Pos):    {self.signals_skipped_max_positions}
{sector_line}{narrative_line}"""
        return header + format_metrics_report(self.metrics)


class PortfolioSimulator:
    """
    Realistic portfolio simulator with day-by-day capital management

    Unlike the basic Backtester which processes symbols independently,
    this simulator processes chronologically, respecting:
    - Available capital at each moment
    - Maximum positions limit
    - Signal priority by confidence score

    Usage:
        simulator = PortfolioSimulator(config)
        result = simulator.run_simulation(all_data, '2023-01-01', '2024-12-01')
        print(result)
    """

    def __init__(self, config: BacktestConfig = None):
        """
        Initialize simulator

        Args:
            config: Backtesting configuration
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

        # Sector analyzer (if sector filter enabled)
        self.sector_analyzer: Optional[SectorAnalyzer] = None
        if self.config.use_sector_filter:
            self.sector_analyzer = SectorAnalyzer(
                min_momentum_vs_spy=self.config.min_sector_momentum
            )
        self.signals_skipped_sector = 0

        # Narrative tracker (if narrative filter enabled)
        self.narrative_tracker: Optional[NarrativeTracker] = None
        if self.config.use_narrative_filter:
            self.narrative_tracker = NarrativeTracker()
        self.signals_boosted_narrative = 0
        self.signals_skipped_narrative = 0

        # Portfolio state
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        self.daily_states: List[PortfolioState] = []

        # Statistics
        self.signals_detected = 0
        self.signals_taken = 0
        self.signals_skipped_capital = 0
        self.signals_skipped_max_positions = 0

        # Progress callback
        self.progress_callback: Optional[Callable[[int, int, str], None]] = None

    def run_simulation(
        self,
        all_data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        all_data_daily: Optional[Dict[str, pd.DataFrame]] = None
    ) -> SimulationResult:
        """
        Run chronological simulation

        Implements dual-timeframe logic like the real screener:
        1. Check weekly first
        2. If no weekly signal but EMAs bullish (≥2/3), check daily fallback

        Args:
            all_data: Pre-fetched WEEKLY data for all symbols {symbol: DataFrame}
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            progress_callback: Optional callback(current_day, total_days, status)
            all_data_daily: Pre-fetched DAILY data (for fallback, optional)

        Returns:
            SimulationResult with complete analysis
        """
        self.progress_callback = progress_callback

        # Reset state
        self._reset_state()

        # Convert dates
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        # Pre-process all data (add EMAs)
        print(f"Pre-processing {len(all_data)} weekly symbols...")
        processed_weekly = self._preprocess_all_data(all_data)

        # Pre-process daily data if provided
        processed_daily = None
        if all_data_daily and self.config.use_daily_fallback:
            print(f"Pre-processing {len(all_data_daily)} daily symbols...")
            processed_daily = self._preprocess_all_data(all_data_daily)

        # Load sector ETF data if sector filter is enabled
        if self.sector_analyzer is not None:
            from src.data.market_data import market_data_fetcher
            print("Loading sector ETF data for sector filter...")
            self.sector_analyzer.load_sector_data(market_data_fetcher, period='2y')

        # Get trading days (using first symbol as reference)
        trading_days = self._get_trading_days(processed_weekly, start_dt, end_dt)

        print(f"\nRunning simulation from {start_date} to {end_date}")
        print(f"Symbols: {len(processed_weekly)}, Trading days: {len(trading_days)}")
        print(f"Dual-timeframe: {'Yes (weekly + daily fallback)' if processed_daily else 'No (weekly only)'}")
        print(f"Sector filter: {'Enabled (min vs SPY: ' + str(self.config.min_sector_momentum) + '%)' if self.sector_analyzer else 'Disabled'}")
        print(f"Initial capital: ${self.config.initial_capital:,.2f}")
        print(f"Max positions: {self.config.max_positions}")
        print("-" * 60)

        # Main simulation loop - day by day
        for day_idx, current_day in enumerate(trading_days):
            # Progress update
            if self.progress_callback and day_idx % 5 == 0:
                self.progress_callback(
                    day_idx,
                    len(trading_days),
                    f"Processing {current_day.strftime('%Y-%m-%d')}"
                )

            # 1. Process exits for open positions (use appropriate timeframe data)
            self._process_exits(current_day, processed_weekly, processed_daily)

            # 2. Calculate available capital
            available_capital = self._get_available_capital()

            # 3. Check if we can open new positions
            if len(self.positions) >= self.config.max_positions:
                # Record state and continue
                self._record_daily_state(current_day, processed_weekly)
                continue

            min_position_value = self.config.initial_capital * 0.05  # Min 5% of capital
            if available_capital < min_position_value:
                # Record state and continue
                self._record_daily_state(current_day, processed_weekly)
                continue

            # 4. Scan ALL symbols for signals on this day (dual-timeframe)
            signals = self._scan_all_symbols_dual_timeframe(
                current_day, processed_weekly, processed_daily
            )

            if signals:
                self.signals_detected += len(signals)

                # 5. Sort by confidence score (highest first)
                signals.sort(key=lambda s: s['confidence_score'], reverse=True)

                # 6. Open positions respecting limits
                for signal in signals:
                    # Check max positions
                    if len(self.positions) >= self.config.max_positions:
                        self.signals_skipped_max_positions += 1
                        continue

                    # Check available capital
                    available_capital = self._get_available_capital()
                    if available_capital < min_position_value:
                        self.signals_skipped_capital += 1
                        continue

                    # Skip if already have position in this symbol
                    if signal['symbol'] in self.positions:
                        continue

                    # Try to open position
                    position = self._open_position(signal, available_capital, current_day)
                    if position:
                        self.positions[signal['symbol']] = position
                        self.signals_taken += 1

            # 7. Record daily state
            self._record_daily_state(current_day, processed_weekly)

        # Close any remaining positions at end
        self._close_all_positions(trading_days[-1] if trading_days else end_dt, processed_weekly)

        # Build results
        return self._build_result(start_dt, end_dt)

    def _reset_state(self):
        """Reset portfolio state for new simulation"""
        self.cash = self.config.initial_capital
        self.positions = {}
        self.closed_trades = []
        self.daily_states = []
        self.signals_detected = 0
        self.signals_taken = 0
        self.signals_skipped_capital = 0
        self.signals_skipped_max_positions = 0
        self.signals_skipped_sector = 0
        self.signals_boosted_narrative = 0
        self.signals_skipped_narrative = 0

    def _preprocess_all_data(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Add EMAs and normalize all dataframes"""
        processed = {}

        for symbol, df in all_data.items():
            if df is None or len(df) < 50:
                continue

            try:
                # Normalize index
                if df.index.tz is not None:
                    df = df.copy()
                    df.index = df.index.tz_localize(None)

                # Add EMAs
                df = self.ema_analyzer.calculate_emas(df)
                processed[symbol] = df

            except Exception as e:
                print(f"Error preprocessing {symbol}: {e}")
                continue

        return processed

    def _get_trading_days(
        self,
        data: Dict[str, pd.DataFrame],
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp
    ) -> List[pd.Timestamp]:
        """Get list of trading days from the data"""
        # Use the first symbol with sufficient data as reference
        for symbol, df in data.items():
            if df is not None and len(df) > 100:
                days = df.index[(df.index >= start_dt) & (df.index <= end_dt)]
                return sorted(days.tolist())

        return []

    def _get_available_capital(self) -> float:
        """Calculate available capital (cash not blocked in positions)"""
        return self.cash

    def _get_total_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value (cash + positions at market)"""
        positions_value = 0.0

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                positions_value += position.shares * current_prices[symbol]
            else:
                # Use position entry value if price not available
                positions_value += position.position_value

        return self.cash + positions_value

    def _process_exits(
        self,
        current_day: pd.Timestamp,
        all_data_weekly: Dict[str, pd.DataFrame],
        all_data_daily: Optional[Dict[str, pd.DataFrame]] = None
    ):
        """Check and process exits for all open positions"""
        symbols_to_close = []

        for symbol, position in list(self.positions.items()):
            # Use daily data for daily-timeframe positions, weekly otherwise
            if hasattr(position, 'timeframe') and position.timeframe == 'daily' and all_data_daily:
                df = all_data_daily.get(symbol)
            else:
                df = all_data_weekly.get(symbol)

            if df is None:
                continue

            # Find current day's data (or nearest prior date)
            if current_day in df.index:
                idx = df.index.get_loc(current_day)
            else:
                # Find nearest date <= current_day
                mask = df.index <= current_day
                if not mask.any():
                    continue
                idx = len(df.index[mask]) - 1

            # Get current prices
            current_price = df['Close'].iloc[idx]
            current_high = df['High'].iloc[idx]
            current_low = df['Low'].iloc[idx]

            # Update highest price for trailing stop
            if current_high > position.highest_price:
                position.highest_price = current_high

            # Check exit conditions (pass df and idx for adaptive exit)
            should_exit, exit_reason = self._check_exit_conditions(
                position, current_price, current_high, current_low, current_day,
                df=df, idx=idx
            )

            if should_exit:
                # Determine exit price based on reason
                if exit_reason == 'stop_loss':
                    exit_price = min(position.stop_loss, current_price)
                elif exit_reason == 'trailing_stop':
                    trailing_stop = position.highest_price * (1 - self.config.trailing_stop_pct)
                    exit_price = min(trailing_stop, current_price)
                elif exit_reason in ('chandelier_exit', 'squeeze_breakdown'):
                    # Adaptive exits use current price
                    exit_price = current_price
                else:
                    exit_price = current_price

                symbols_to_close.append((symbol, exit_price, exit_reason))

        # Close positions
        for symbol, exit_price, exit_reason in symbols_to_close:
            self._close_position(symbol, exit_price, current_day, exit_reason)

    def _check_exit_conditions(
        self,
        position: Position,
        current_price: float,
        current_high: float,
        current_low: float,
        current_day: pd.Timestamp,
        df: pd.DataFrame = None,
        idx: int = None
    ) -> Tuple[bool, str]:
        """
        Check if position should be exited.

        Uses adaptive exit system (Chandelier + Bollinger Squeeze + Volume)
        when use_adaptive_exit is enabled, otherwise falls back to legacy logic.
        """
        entry_price = position.entry_price

        # ===== ADAPTIVE EXIT SYSTEM =====
        if self.config.use_adaptive_exit and df is not None and idx is not None:
            # Use the new adaptive exit indicator
            exit_indicator = AdaptiveExitIndicator(
                atr_period=self.config.chandelier_atr_period,
                atr_multiplier=self.config.chandelier_multiplier
            )

            signal = exit_indicator.should_exit(
                df=df,
                idx=idx,
                position_stop_loss=position.stop_loss,
                entry_price=entry_price
            )

            if signal.should_exit:
                return (True, signal.reason)

            # Also check max hold period (safety net)
            hold_days = (current_day - position.entry_date).days
            if hold_days >= self.config.max_hold_days:
                return (True, 'max_hold_period')

            return (False, '')

        # ===== LEGACY EXIT SYSTEM (fallback) =====
        # Check stop-loss
        if current_low <= position.stop_loss:
            return (True, 'stop_loss')

        # Check trailing stop (legacy)
        if self.config.use_trailing_stop:
            trailing_stop = position.highest_price * (1 - self.config.trailing_stop_pct)
            if current_low <= trailing_stop:
                return (True, 'trailing_stop')

        # Check take-profit (legacy - disabled when take_profit_pct = 0)
        if self.config.take_profit_pct > 0:
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= self.config.take_profit_pct:
                return (True, 'take_profit')

        # Check max hold period
        hold_days = (current_day - position.entry_date).days
        if hold_days >= self.config.max_hold_days:
            return (True, 'max_hold_period')

        return (False, '')

    def _scan_all_symbols(
        self,
        current_day: pd.Timestamp,
        all_data: Dict[str, pd.DataFrame]
    ) -> List[Dict]:
        """Scan all symbols for entry signals on given day"""
        signals = []
        lookback = 104 if self.config.timeframe == 'weekly' else 252

        for symbol, df in all_data.items():
            # Skip if already have position
            if symbol in self.positions:
                continue

            # Check if we have data for this day
            if current_day not in df.index:
                continue

            idx = df.index.get_loc(current_day)

            # Need enough lookback data
            if idx < lookback:
                continue

            # Check for entry signal
            signal = self._check_entry_signal(symbol, df, idx, lookback)
            if signal is not None:
                signals.append(signal)

        return signals

    def _scan_all_symbols_dual_timeframe(
        self,
        current_day: pd.Timestamp,
        all_data_weekly: Dict[str, pd.DataFrame],
        all_data_daily: Optional[Dict[str, pd.DataFrame]]
    ) -> List[Dict]:
        """
        Scan all symbols using dual-timeframe logic like the real screener:
        1. Check weekly first
        2. If no weekly signal but EMAs bullish (≥2/3), check daily fallback

        Args:
            current_day: Current simulation day
            all_data_weekly: Weekly data for all symbols
            all_data_daily: Daily data for all symbols (optional)

        Returns:
            List of signals found
        """
        signals = []
        lookback_weekly = 104
        lookback_daily = 252

        for symbol, df_weekly in all_data_weekly.items():
            # Skip if already have position
            if symbol in self.positions:
                continue

            # Check if we have weekly data for this day
            if current_day not in df_weekly.index:
                continue

            weekly_idx = df_weekly.index.get_loc(current_day)

            # Need enough lookback data
            if weekly_idx < lookback_weekly:
                continue

            # 1. Check weekly signal first
            weekly_signal = self._check_entry_signal(symbol, df_weekly, weekly_idx, lookback_weekly)

            if weekly_signal is not None:
                # Check sector filter if enabled
                if self.sector_analyzer is not None:
                    if not self.sector_analyzer.is_sector_bullish(symbol, current_day):
                        self.signals_skipped_sector += 1
                        continue  # Skip signal in bearish sector

                # Apply narrative boost/filter if enabled
                if self.narrative_tracker is not None:
                    boost_info = self.narrative_tracker.get_narrative_boost(symbol, current_day)

                    # Filter if require_narrative_alignment is True
                    if self.config.require_narrative_alignment and not boost_info.is_aligned:
                        self.signals_skipped_narrative += 1
                        continue

                    # Apply boost to confidence score
                    if self.config.narrative_boost_scores and boost_info.is_aligned:
                        original_score = weekly_signal['confidence_score']
                        weekly_signal['confidence_score'] = min(100, original_score + boost_info.boost_pct)
                        weekly_signal['confidence_score_raw'] = original_score
                        weekly_signal['narrative_aligned'] = True
                        weekly_signal['narrative'] = boost_info.narrative.value
                        weekly_signal['narrative_boost'] = boost_info.boost_pct
                        self.signals_boosted_narrative += 1

                weekly_signal['timeframe'] = 'weekly'
                signals.append(weekly_signal)
                continue  # Got weekly signal, no need for daily fallback

            # 2. Check if daily fallback should be triggered
            if not self.config.use_daily_fallback or all_data_daily is None:
                continue

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

            # Only fallback to daily if weekly EMAs are bullish
            if ema_conditions < self.config.min_ema_conditions_for_fallback:
                continue

            # 3. Check daily signal
            df_daily = all_data_daily.get(symbol)
            if df_daily is None:
                continue

            # Find nearest daily date <= current_day
            daily_mask = df_daily.index <= current_day
            if not daily_mask.any():
                continue

            daily_idx = len(df_daily.index[daily_mask]) - 1
            if daily_idx < lookback_daily:
                continue

            daily_signal = self._check_entry_signal(symbol, df_daily, daily_idx, lookback_daily)

            if daily_signal is not None:
                # Check sector filter if enabled
                if self.sector_analyzer is not None:
                    if not self.sector_analyzer.is_sector_bullish(symbol, current_day):
                        self.signals_skipped_sector += 1
                        continue  # Skip signal in bearish sector

                # Apply narrative boost/filter if enabled
                if self.narrative_tracker is not None:
                    boost_info = self.narrative_tracker.get_narrative_boost(symbol, current_day)

                    # Filter if require_narrative_alignment is True
                    if self.config.require_narrative_alignment and not boost_info.is_aligned:
                        self.signals_skipped_narrative += 1
                        continue

                    # Apply boost to confidence score
                    if self.config.narrative_boost_scores and boost_info.is_aligned:
                        original_score = daily_signal['confidence_score']
                        daily_signal['confidence_score'] = min(100, original_score + boost_info.boost_pct)
                        daily_signal['confidence_score_raw'] = original_score
                        daily_signal['narrative_aligned'] = True
                        daily_signal['narrative'] = boost_info.narrative.value
                        daily_signal['narrative_boost'] = boost_info.boost_pct
                        self.signals_boosted_narrative += 1

                daily_signal['timeframe'] = 'daily'
                daily_signal['weekly_ema_conditions'] = ema_conditions
                signals.append(daily_signal)

        return signals

    def _check_entry_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        idx: int,
        lookback: int
    ) -> Optional[Dict]:
        """Check for entry signal at given index"""
        # Use data up to current index (no lookahead)
        analysis_df = df.iloc[:idx + 1].copy()

        # Check RSI breakout
        try:
            result = self.rsi_analyzer.analyze(analysis_df, lookback_periods=lookback)
        except Exception:
            return None

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

        current_price = row['Close']

        # Get volume ratio
        volume_ratio = getattr(result.rsi_breakout, 'volume_ratio', 1.0)

        # Check volume confirmation if required
        if self.config.require_volume_confirmation:
            if volume_ratio < self.config.min_volume_ratio:
                return None

        # Calculate confidence score
        rsi_breakout_data = {
            'strength': result.rsi_breakout.strength,
            'age_in_periods': result.rsi_breakout.age_in_periods,
            'index': result.rsi_breakout.index,
            'volume_ratio': volume_ratio
        }

        rsi_trendline_data = {
            'r_squared': result.rsi_trendline.r_squared
        } if result.rsi_trendline else None

        support_level = min(ema_24, ema_38, ema_62)

        try:
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
        except Exception:
            return None

        # Check minimum confidence
        if confidence.total_score < self.config.min_confidence_score:
            return None

        # Get RSI peak indices for stop-loss calculation
        peak_indices = []
        if result.rsi_trendline is not None:
            peak_indices = result.rsi_trendline.peak_indices

        return {
            'symbol': symbol,
            'signal': result.signal,
            'confidence_score': confidence.total_score,
            'confidence_signal': confidence.signal,
            'rsi_breakout': result.rsi_breakout,
            'rsi_trendline': result.rsi_trendline,
            'volume_ratio': volume_ratio,
            'ema_conditions': ema_conditions,
            'current_price': current_price,
            'peak_indices': peak_indices,
            'df': analysis_df  # Keep reference for stop calculation
        }

    def _open_position(
        self,
        signal: Dict,
        available_capital: float,
        current_day: pd.Timestamp
    ) -> Optional[Position]:
        """Open a new position"""
        entry_price = signal['current_price']

        # Calculate position size (max 20% of total capital per position)
        max_position_value = self.config.initial_capital * self.config.position_size_pct
        position_value = min(max_position_value, available_capital)

        # Calculate shares
        shares = int(position_value / entry_price)
        if shares <= 0:
            return None

        # Actual position value
        actual_value = shares * entry_price

        # Calculate stop-loss from RSI peaks
        stop_loss = entry_price * 0.95  # Default 5%

        if signal['peak_indices'] and 'df' in signal:
            df = signal['df']
            valid_indices = [i for i in signal['peak_indices'] if i < len(df)]
            if valid_indices:
                peak_lows = [df['Low'].iloc[i] for i in valid_indices]
                stop_loss = min(peak_lows) * 0.99  # 1% below lowest peak

                # Validate stop-loss range (2-20%)
                stop_pct = (entry_price - stop_loss) / entry_price * 100
                if stop_pct < 2:
                    stop_loss = entry_price * 0.97
                elif stop_pct > 20:
                    stop_loss = entry_price * 0.85

        # Deduct from cash
        self.cash -= actual_value

        return Position(
            symbol=signal['symbol'],
            entry_date=current_day,
            entry_price=entry_price,
            shares=shares,
            stop_loss=stop_loss,
            position_value=actual_value,
            signal_strength=signal['signal'],
            confidence_score=signal['confidence_score'],
            volume_ratio=signal['volume_ratio'],
            highest_price=entry_price,
            rsi_peak_indices=signal['peak_indices'],
            timeframe=signal.get('timeframe', 'weekly')  # Track which timeframe triggered entry
        )

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_date: pd.Timestamp,
        exit_reason: str
    ):
        """Close a position and record the trade"""
        position = self.positions.pop(symbol, None)
        if position is None:
            return

        # Calculate P&L
        profit_loss = (exit_price - position.entry_price) * position.shares
        profit_loss_pct = (exit_price - position.entry_price) / position.entry_price * 100

        # Return capital to cash
        exit_value = exit_price * position.shares
        self.cash += exit_value

        # Record trade
        trade = Trade(
            symbol=symbol,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            shares=position.shares,
            stop_loss=position.stop_loss,
            profit_loss=profit_loss,
            profit_loss_pct=profit_loss_pct,
            exit_reason=exit_reason,
            signal_strength=position.signal_strength,
            confidence_score=position.confidence_score,
            volume_ratio=position.volume_ratio
        )

        self.closed_trades.append(trade)

    def _close_all_positions(self, end_date: pd.Timestamp, all_data: Dict[str, pd.DataFrame]):
        """Close all remaining positions at end of simulation"""
        for symbol in list(self.positions.keys()):
            df = all_data.get(symbol)
            if df is not None and len(df) > 0:
                exit_price = df['Close'].iloc[-1]
            else:
                exit_price = self.positions[symbol].entry_price  # Fallback

            self._close_position(symbol, exit_price, end_date, 'end_of_period')

    def _record_daily_state(self, current_day: pd.Timestamp, all_data: Dict[str, pd.DataFrame]):
        """Record portfolio state for this day"""
        # Calculate positions value at market
        positions_value = 0.0
        position_details = {}

        for symbol, position in self.positions.items():
            df = all_data.get(symbol)
            if df is not None and current_day in df.index:
                current_price = df.loc[current_day, 'Close']
                current_value = position.shares * current_price
            else:
                current_value = position.position_value

            positions_value += current_value
            position_details[symbol] = current_value

        state = PortfolioState(
            date=current_day,
            cash=self.cash,
            positions_value=positions_value,
            total_value=self.cash + positions_value,
            num_positions=len(self.positions),
            positions=position_details
        )

        self.daily_states.append(state)

    def _build_result(self, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> SimulationResult:
        """Build final simulation result"""
        # Calculate metrics
        metrics = calculate_metrics(
            trades=self.closed_trades,
            initial_capital=self.config.initial_capital
        )

        # Build equity curve from daily states
        if self.daily_states:
            dates = [s.date for s in self.daily_states]
            values = [s.total_value for s in self.daily_states]
            equity_curve = pd.Series(values, index=pd.DatetimeIndex(dates))
        else:
            equity_curve = pd.Series([self.config.initial_capital], index=[start_dt])

        return SimulationResult(
            config=self.config,
            metrics=metrics,
            trades=self.closed_trades,
            equity_curve=equity_curve,
            daily_states=self.daily_states,
            signals_detected=self.signals_detected,
            signals_taken=self.signals_taken,
            signals_skipped_capital=self.signals_skipped_capital,
            signals_skipped_max_positions=self.signals_skipped_max_positions,
            signals_skipped_sector=self.signals_skipped_sector,
            signals_boosted_narrative=self.signals_boosted_narrative,
            signals_skipped_narrative=self.signals_skipped_narrative,
            open_positions=list(self.positions.values())
        )


def run_realistic_simulation(
    symbols: List[str],
    start_date: str = '2023-01-01',
    end_date: str = '2024-12-01',
    initial_capital: float = 10000,
    max_positions: int = 5,
    data_provider=None,
    progress_callback=None
) -> SimulationResult:
    """
    Convenience function to run realistic portfolio simulation

    Args:
        symbols: List of symbols to simulate
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
        max_positions: Maximum concurrent positions
        data_provider: Optional data fetcher
        progress_callback: Optional progress callback

    Returns:
        SimulationResult
    """
    from src.data.market_data import market_data_fetcher

    data_fetcher = data_provider or market_data_fetcher

    # Configuration
    config = BacktestConfig(
        initial_capital=initial_capital,
        max_positions=max_positions,
        use_enhanced_detector=True,
        precision_mode='medium',
        min_confidence_score=55,
        require_volume_confirmation=True
    )

    # Pre-fetch all data
    print(f"Fetching data for {len(symbols)} symbols...")

    interval = '1wk' if config.timeframe == 'weekly' else '1d'
    all_data = {}

    # Batch fetch for efficiency
    batch_size = 100
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        print(f"  Batch {i // batch_size + 1}: {len(batch)} symbols")

        batch_data = data_fetcher.get_batch_historical_data(
            batch,
            period='5y',
            interval=interval
        )
        all_data.update(batch_data)

    print(f"Loaded data for {len(all_data)} symbols")

    # Run simulation
    simulator = PortfolioSimulator(config)
    return simulator.run_simulation(
        all_data=all_data,
        start_date=start_date,
        end_date=end_date,
        progress_callback=progress_callback
    )
