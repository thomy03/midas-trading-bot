"""
Trendline Breakout Screener

Filters EMA screener results by checking for RSI trendline breakouts.
Workflow:
1. Takes symbols that passed EMA screening
2. Checks for trendline breakout on weekly first
3. Falls back to daily if no weekly trendline found
4. Returns only symbols with confirmed breakouts
"""

import pandas as pd
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.breakout_analyzer import TrendlineBreakoutAnalyzer
from ..config.settings import (
    WEEKLY_LOOKBACK_PERIODS,
    DAILY_LOOKBACK_PERIODS,
    TIMEFRAME_PRIORITY
)


class TrendlineBreakoutScreener:
    """Screens for RSI trendline breakouts on EMA-filtered symbols"""

    def __init__(self, max_workers: int = 4):
        """
        Initialize the trendline screener

        Args:
            max_workers: Maximum number of parallel workers
        """
        self.analyzer = TrendlineBreakoutAnalyzer()
        self.max_workers = max_workers

    def screen_single_symbol(
        self,
        symbol: str,
        df_weekly: Optional[pd.DataFrame] = None,
        df_daily: Optional[pd.DataFrame] = None
    ) -> Optional[Dict]:
        """
        Screen a single symbol for trendline breakout

        Args:
            symbol: Stock symbol
            df_weekly: Weekly OHLCV data (optional, will fetch if not provided)
            df_daily: Daily OHLCV data (optional)

        Returns:
            Dictionary with breakout info or None
        """
        result = None

        # Try weekly first
        if df_weekly is not None and len(df_weekly) >= WEEKLY_LOOKBACK_PERIODS:
            weekly_analysis = self.analyzer.analyze(
                df_weekly,
                lookback_periods=WEEKLY_LOOKBACK_PERIODS
            )

            if weekly_analysis and weekly_analysis['has_breakout']:
                signal = self.analyzer.get_signal(weekly_analysis)

                if signal != 'NO_SIGNAL':
                    result = {
                        'symbol': symbol,
                        'timeframe': 'weekly',
                        'signal': signal,
                        'analysis': weekly_analysis,
                        'breakout_date': weekly_analysis['breakout'].date,
                        'breakout_strength': weekly_analysis['breakout'].strength,
                        'is_confirmed': weekly_analysis['breakout'].is_confirmed,
                        'trendline_quality': weekly_analysis['breakout'].trendline_quality,
                        'rsi_value': weekly_analysis['breakout'].rsi_value
                    }
                    return result

        # If no weekly trendline, try daily
        if df_daily is not None and len(df_daily) >= DAILY_LOOKBACK_PERIODS:
            daily_analysis = self.analyzer.analyze(
                df_daily,
                lookback_periods=DAILY_LOOKBACK_PERIODS
            )

            if daily_analysis and daily_analysis['has_breakout']:
                signal = self.analyzer.get_signal(daily_analysis)

                if signal != 'NO_SIGNAL':
                    result = {
                        'symbol': symbol,
                        'timeframe': 'daily',
                        'signal': signal,
                        'analysis': daily_analysis,
                        'breakout_date': daily_analysis['breakout'].date,
                        'breakout_strength': daily_analysis['breakout'].strength,
                        'is_confirmed': daily_analysis['breakout'].is_confirmed,
                        'trendline_quality': daily_analysis['breakout'].trendline_quality,
                        'rsi_value': daily_analysis['breakout'].rsi_value
                    }

        return result

    def screen_multiple_symbols(
        self,
        symbols_data: List[Dict],
        use_parallel: bool = True
    ) -> List[Dict]:
        """
        Screen multiple symbols for trendline breakouts

        Args:
            symbols_data: List of dicts with 'symbol', 'df_weekly', 'df_daily'
            use_parallel: Whether to use parallel processing

        Returns:
            List of symbols with breakouts, sorted by quality
        """
        results = []

        if use_parallel and len(symbols_data) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_symbol = {
                    executor.submit(
                        self.screen_single_symbol,
                        data['symbol'],
                        data.get('df_weekly'),
                        data.get('df_daily')
                    ): data['symbol']
                    for data in symbols_data
                }

                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        print(f"Error screening {symbol}: {e}")
        else:
            # Sequential processing
            for data in symbols_data:
                result = self.screen_single_symbol(
                    data['symbol'],
                    data.get('df_weekly'),
                    data.get('df_daily')
                )
                if result:
                    results.append(result)

        # Sort by quality and strength
        results.sort(
            key=lambda x: (
                self._signal_priority(x['signal']),
                x['trendline_quality'],
                1 if x['is_confirmed'] else 0
            ),
            reverse=True
        )

        return results

    def _signal_priority(self, signal: str) -> int:
        """Get priority for signal type (higher = better)"""
        priority_map = {
            'STRONG_BUY': 3,
            'BUY': 2,
            'WATCH': 1,
            'NO_SIGNAL': 0
        }
        return priority_map.get(signal, 0)

    def filter_ema_results(
        self,
        ema_alerts: List[Dict],
        data_provider_func
    ) -> List[Dict]:
        """
        Filter EMA screener results by trendline breakouts

        This is the main integration point with the existing screener.

        Args:
            ema_alerts: List of alerts from EMA screener
            data_provider_func: Function to fetch data for a symbol
                                Should accept (symbol, timeframe) and return DataFrame

        Returns:
            List of symbols that have both EMA signals AND trendline breakouts
        """
        symbols_data = []

        for alert in ema_alerts:
            symbol = alert['symbol']
            timeframe = alert.get('timeframe', 'weekly')

            # Fetch data for trendline analysis
            try:
                df_weekly = data_provider_func(symbol, 'weekly')
                df_daily = data_provider_func(symbol, 'daily')

                symbols_data.append({
                    'symbol': symbol,
                    'df_weekly': df_weekly,
                    'df_daily': df_daily,
                    'ema_alert': alert  # Keep original EMA alert info
                })
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue

        # Screen for trendline breakouts
        breakout_results = self.screen_multiple_symbols(symbols_data)

        # Merge with EMA alert info
        enhanced_results = []
        for breakout in breakout_results:
            # Find corresponding EMA alert
            ema_alert = next(
                (data['ema_alert'] for data in symbols_data if data['symbol'] == breakout['symbol']),
                None
            )

            if ema_alert:
                # Merge both signals
                enhanced_results.append({
                    **breakout,
                    'ema_signal': ema_alert.get('recommendation', 'UNKNOWN'),
                    'ema_timeframe': ema_alert.get('timeframe', 'unknown'),
                    'combined_score': self._calculate_combined_score(breakout, ema_alert)
                })

        # Sort by combined score
        enhanced_results.sort(key=lambda x: x['combined_score'], reverse=True)

        return enhanced_results

    def _calculate_combined_score(self, breakout_result: Dict, ema_alert: Dict) -> float:
        """
        Calculate combined score from EMA and trendline signals

        Args:
            breakout_result: Trendline breakout result
            ema_alert: EMA alert info

        Returns:
            Combined score (0-100)
        """
        # Trendline component (0-60 points)
        trendline_score = breakout_result['trendline_quality'] * 0.6

        # Signal strength component (0-20 points)
        signal_strength = {
            'STRONG_BUY': 20,
            'BUY': 15,
            'WATCH': 10
        }.get(breakout_result['signal'], 0)

        # Confirmation bonus (0-10 points)
        confirmation_bonus = 10 if breakout_result['is_confirmed'] else 5

        # Timeframe bonus (0-10 points)
        timeframe_bonus = 10 if breakout_result['timeframe'] == 'weekly' else 5

        total = trendline_score + signal_strength + confirmation_bonus + timeframe_bonus

        return min(100, total)
