"""
Main screening logic for market analysis
"""
import json
from typing import List, Dict, Optional
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from config.settings import (
    TIMEFRAMES, MAX_WORKERS, LOOKBACK_DAYS_WEEKLY, LOOKBACK_DAYS_DAILY
)
from src.data.market_data import market_data_fetcher
from src.indicators.ema_analyzer import ema_analyzer
from src.database.db_manager import db_manager
from src.utils.logger import logger


class MarketScreener:
    """Main market screener for detecting buy signals"""

    def __init__(self):
        """Initialize the market screener"""
        self.market_data = market_data_fetcher
        self.ema_analyzer = ema_analyzer
        self.db = db_manager

    def screen_single_stock(
        self,
        symbol: str,
        company_name: str = ""
    ) -> Optional[Dict]:
        """
        Screen a single stock on both weekly and daily timeframes

        Args:
            symbol: Stock symbol
            company_name: Company name (optional)

        Returns:
            Alert dictionary if signal found, None otherwise
        """
        try:
            logger.info(f"Screening {symbol}...")

            # Step 1: Screen on weekly timeframe
            weekly_result = self._screen_timeframe(symbol, 'weekly')

            if weekly_result:
                # Found signal on weekly - this is the best signal
                logger.info(f"{symbol}: WEEKLY BUY SIGNAL DETECTED")
                return self._create_alert(symbol, company_name, weekly_result, 'weekly')

            # Step 2: If no weekly signal, check if EMAs are aligned on weekly
            # If aligned, proceed to daily screening
            weekly_data = self.market_data.get_historical_data(
                symbol,
                period='2y',
                interval=TIMEFRAMES['weekly']
            )

            if weekly_data is not None and not weekly_data.empty:
                weekly_data_with_emas = self.ema_analyzer.calculate_emas(weekly_data)
                is_aligned, _ = self.ema_analyzer.check_ema_alignment(weekly_data_with_emas, for_buy=True)

                if is_aligned:
                    logger.debug(f"{symbol}: Weekly EMAs aligned, checking daily...")

                    # Screen on daily timeframe
                    daily_result = self._screen_timeframe(symbol, 'daily')

                    if daily_result:
                        logger.info(f"{symbol}: DAILY BUY SIGNAL DETECTED (Weekly EMAs aligned)")
                        return self._create_alert(symbol, company_name, daily_result, 'daily')

            logger.debug(f"{symbol}: No signals detected")
            return None

        except Exception as e:
            logger.error(f"Error screening {symbol}: {e}")
            return None

    def _screen_timeframe(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """
        Screen a stock on a specific timeframe

        Args:
            symbol: Stock symbol
            timeframe: 'weekly' or 'daily'

        Returns:
            Analysis result or None
        """
        try:
            # Get appropriate period and interval
            if timeframe == 'weekly':
                period = '2y'
                interval = TIMEFRAMES['weekly']
            else:  # daily
                period = '2y'
                interval = TIMEFRAMES['daily']

            # Fetch historical data
            df = self.market_data.get_historical_data(symbol, period=period, interval=interval)

            if df is None or df.empty:
                logger.warning(f"No data available for {symbol} on {timeframe}")
                return None

            # Analyze with EMAs
            result = self.ema_analyzer.analyze_stock(df, symbol, timeframe)

            return result

        except Exception as e:
            logger.error(f"Error screening {symbol} on {timeframe}: {e}")
            return None

    def _create_alert(
        self,
        symbol: str,
        company_name: str,
        analysis_result: Dict,
        timeframe: str
    ) -> Dict:
        """
        Create an alert from analysis result

        Args:
            symbol: Stock symbol
            company_name: Company name
            analysis_result: Analysis result dictionary
            timeframe: Timeframe of the signal

        Returns:
            Alert dictionary
        """
        crossover_info = analysis_result.get('crossover_info', {})
        ema_values = analysis_result.get('ema_values', {})

        alert = {
            'symbol': symbol,
            'company_name': company_name or symbol,
            'timeframe': timeframe,
            'current_price': analysis_result['current_price'],
            'support_level': analysis_result['best_support_level'],
            'distance_to_support_pct': analysis_result['distance_to_support_pct'],
            'ema_24': ema_values.get('ema_24'),
            'ema_38': ema_values.get('ema_38'),
            'ema_62': ema_values.get('ema_62'),
            'ema_alignment': analysis_result['ema_alignment'],
            'crossover_info': json.dumps(crossover_info),
            'recommendation': self._get_recommendation(analysis_result),
            'is_notified': False
        }

        return alert

    def _get_recommendation(self, analysis_result: Dict) -> str:
        """
        Get recommendation based on analysis

        Args:
            analysis_result: Analysis result dictionary

        Returns:
            Recommendation string
        """
        distance = analysis_result['distance_to_support_pct']
        strength = analysis_result['zone_strength']

        if distance <= 1.0 and strength >= 70:
            return 'STRONG_BUY'
        elif distance <= 2.0 and strength >= 50:
            return 'BUY'
        elif distance <= 3.5:
            return 'WATCH'
        else:
            return 'OBSERVE'

    def screen_multiple_stocks(self, stocks: List[Dict]) -> List[Dict]:
        """
        Screen multiple stocks in parallel

        Args:
            stocks: List of stock dictionaries with 'symbol' and 'name' keys

        Returns:
            List of alert dictionaries
        """
        alerts = []
        total_stocks = len(stocks)

        logger.info(f"Starting screening of {total_stocks} stocks...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_stock = {
                executor.submit(
                    self.screen_single_stock,
                    stock['symbol'],
                    stock.get('name', '')
                ): stock
                for stock in stocks
            }

            completed = 0
            for future in as_completed(future_to_stock):
                stock = future_to_stock[future]
                completed += 1

                try:
                    alert = future.result()
                    if alert:
                        alerts.append(alert)
                        logger.info(
                            f"[{completed}/{total_stocks}] {stock['symbol']}: "
                            f"ALERT GENERATED ({alert['recommendation']})"
                        )
                    else:
                        logger.debug(f"[{completed}/{total_stocks}] {stock['symbol']}: No signal")

                except Exception as e:
                    logger.error(f"Error processing {stock['symbol']}: {e}")

                # Progress update every 10%
                if completed % max(1, total_stocks // 10) == 0:
                    logger.info(f"Progress: {completed}/{total_stocks} stocks screened "
                              f"({len(alerts)} alerts so far)")

        logger.info(f"Screening complete: {len(alerts)} alerts generated from {total_stocks} stocks")
        return alerts

    def run_daily_screening(self) -> Dict:
        """
        Run the complete daily screening process

        Returns:
            Dictionary with screening results and statistics
        """
        start_time = time.time()
        logger.info("="*80)
        logger.info("STARTING DAILY MARKET SCREENING")
        logger.info("="*80)

        try:
            # Step 1: Get and filter stocks
            logger.info("Step 1: Fetching and filtering stocks...")
            ticker_to_market = self.market_data.get_all_tickers()

            if not ticker_to_market:
                logger.error("No tickers found to screen")
                return {
                    'status': 'FAILED',
                    'error': 'No tickers found',
                    'alerts': []
                }

            filtered_stocks = self.market_data.filter_stocks_parallel(ticker_to_market)

            if not filtered_stocks:
                logger.warning("No stocks passed the filtering criteria")
                return {
                    'status': 'SUCCESS',
                    'total_stocks_analyzed': len(ticker_to_market),
                    'total_alerts_generated': 0,
                    'alerts': []
                }

            logger.info(f"Filtered to {len(filtered_stocks)} stocks meeting criteria")

            # Step 2: Screen all filtered stocks
            logger.info("Step 2: Screening stocks for buy signals...")
            alerts = self.screen_multiple_stocks(filtered_stocks)

            # Step 3: Save alerts to database
            logger.info("Step 3: Saving alerts to database...")
            for alert in alerts:
                try:
                    self.db.save_alert(alert)
                except Exception as e:
                    logger.error(f"Error saving alert for {alert['symbol']}: {e}")

            # Step 4: Save screening history
            execution_time = time.time() - start_time
            history_data = {
                'total_stocks_analyzed': len(filtered_stocks),
                'total_alerts_generated': len(alerts),
                'timeframe': 'weekly+daily',
                'execution_time_seconds': execution_time,
                'status': 'SUCCESS'
            }
            self.db.save_screening_history(history_data)

            logger.info("="*80)
            logger.info(f"SCREENING COMPLETE in {execution_time:.1f} seconds")
            logger.info(f"Stocks analyzed: {len(filtered_stocks)}")
            logger.info(f"Alerts generated: {len(alerts)}")
            logger.info("="*80)

            return {
                'status': 'SUCCESS',
                'total_stocks_analyzed': len(filtered_stocks),
                'total_alerts_generated': len(alerts),
                'execution_time_seconds': execution_time,
                'alerts': alerts
            }

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Screening failed: {e}")

            # Save failed history
            history_data = {
                'total_stocks_analyzed': 0,
                'total_alerts_generated': 0,
                'timeframe': 'weekly+daily',
                'execution_time_seconds': execution_time,
                'status': 'FAILED',
                'error_message': str(e)
            }
            self.db.save_screening_history(history_data)

            return {
                'status': 'FAILED',
                'error': str(e),
                'execution_time_seconds': execution_time,
                'alerts': []
            }


# Singleton instance
market_screener = MarketScreener()
