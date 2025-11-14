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

# Import RSI Breakout Analyzer
from trendline_analysis.core.rsi_breakout_analyzer import RSIBreakoutAnalyzer


class MarketScreener:
    """Main market screener for detecting buy signals"""

    def __init__(self):
        """Initialize the market screener"""
        self.market_data = market_data_fetcher
        self.ema_analyzer = ema_analyzer
        self.db = db_manager
        self.rsi_analyzer = RSIBreakoutAnalyzer()  # RSI breakout detector

    def screen_single_stock(
        self,
        symbol: str,
        company_name: str = ""
    ) -> Optional[Dict]:
        """
        Screen a single stock using HISTORICAL SUPPORT LEVELS + RSI BREAKOUT

        NOUVELLE LOGIQUE (User Request):
        1. Détecter TOUS les niveaux historiques (crossovers EMA)
        2. Vérifier que niveaux valides (toutes EMAs au-dessus)
        3. Identifier niveaux PROCHES (< 8%)
        4. Pour chaque niveau proche → Rechercher oblique RSI + breakout
        5. Si niveau proche + RSI breakout → SIGNAL!

        Args:
            symbol: Stock symbol
            company_name: Company name (optional)

        Returns:
            Alert dictionary if signal found, None otherwise
        """
        try:
            logger.info(f"Screening {symbol}...")

            # Step 1: Get weekly data with EMAs
            df_weekly = self.market_data.get_historical_data(
                symbol,
                period='2y',
                interval=TIMEFRAMES['weekly']
            )

            if df_weekly is None or df_weekly.empty:
                logger.warning(f"No weekly data for {symbol}")
                return None

            df_weekly = self.ema_analyzer.calculate_emas(df_weekly)
            current_price = float(df_weekly['Close'].iloc[-1])

            # Step 2: Detect ALL historical crossovers (no distance filter)
            crossovers = self.ema_analyzer.detect_crossovers(df_weekly, 'weekly')

            if not crossovers:
                logger.debug(f"{symbol}: No historical crossovers found")
                return None

            # Step 3: Get ALL historical support levels
            historical_levels = self.ema_analyzer.find_historical_support_levels(
                df_weekly, crossovers, current_price
            )

            logger.debug(f"{symbol}: Found {len(historical_levels)} historical support levels")

            # Step 4: Filter for NEAR levels (< 8% distance)
            near_levels = [level for level in historical_levels if level['is_near']]

            if not near_levels:
                logger.debug(f"{symbol}: {len(historical_levels)} historical levels but none near (all > 8%)")
                return None

            logger.info(f"{symbol}: {len(near_levels)} NEAR historical level(s) detected! Checking RSI...")

            # Step 5: For each near level, check RSI breakout
            for level in near_levels:
                level_price = level['level']
                distance_pct = level['distance_pct']
                crossover_info = level['crossover_info']

                logger.info(f"{symbol}: Price ${current_price:.2f} approaching level ${level_price:.2f} ({distance_pct:.1f}%)")

                # Check RSI breakout on weekly first
                rsi_weekly = self._check_rsi_breakout(symbol, 'weekly')

                if rsi_weekly and rsi_weekly.has_rsi_breakout:
                    logger.info(f"{symbol}: HISTORICAL LEVEL + RSI BREAKOUT WEEKLY → STRONG SIGNAL!")
                    return self._create_alert_historical_level(
                        symbol, company_name, level, 'weekly', rsi_weekly, df_weekly
                    )

                # Check RSI breakout on daily
                rsi_daily = self._check_rsi_breakout(symbol, 'daily')

                if rsi_daily and rsi_daily.has_rsi_breakout:
                    logger.info(f"{symbol}: HISTORICAL LEVEL + RSI BREAKOUT DAILY → STRONG SIGNAL!")
                    return self._create_alert_historical_level(
                        symbol, company_name, level, 'daily', rsi_daily, df_weekly
                    )

                # Check if at least RSI trendline (even without breakout)
                if (rsi_weekly and rsi_weekly.has_rsi_trendline) or (rsi_daily and rsi_daily.has_rsi_trendline):
                    logger.info(f"{symbol}: HISTORICAL LEVEL + RSI trendline → BUY signal")
                    rsi_result = rsi_weekly if rsi_weekly and rsi_weekly.has_rsi_trendline else rsi_daily
                    rsi_tf = 'weekly' if rsi_weekly and rsi_weekly.has_rsi_trendline else 'daily'
                    return self._create_alert_historical_level(
                        symbol, company_name, level, rsi_tf, rsi_result, df_weekly
                    )

            # Near level but no RSI trendline → WATCH signal
            logger.info(f"{symbol}: Near historical level but no RSI trendline → WATCH")
            return self._create_alert_historical_level(
                symbol, company_name, near_levels[0], 'weekly', None, df_weekly
            )

        except Exception as e:
            logger.error(f"Error screening {symbol}: {e}")
            return None

    def _check_rsi_breakout(self, symbol: str, timeframe: str):
        """
        Vérifie si un RSI breakout existe sur le timeframe donné

        Args:
            symbol: Stock symbol
            timeframe: 'weekly' or 'daily'

        Returns:
            RSIBreakoutResult ou None
        """
        try:
            # Get appropriate period, interval, and lookback
            if timeframe == 'weekly':
                period = '2y'
                interval = TIMEFRAMES['weekly']
                lookback = 104  # ~2 ans de weekly
            else:  # daily
                period = '1y'
                interval = TIMEFRAMES['daily']
                lookback = 252  # ~1 an de daily

            # Fetch historical data
            df = self.market_data.get_historical_data(symbol, period=period, interval=interval)

            if df is None or df.empty:
                logger.warning(f"No data available for {symbol} RSI analysis on {timeframe}")
                return None

            # Analyze RSI breakout
            result = self.rsi_analyzer.analyze(df, lookback_periods=lookback)

            if result:
                logger.debug(f"{symbol} RSI {timeframe}: trendline={result.has_rsi_trendline}, "
                           f"breakout={result.has_rsi_breakout}, signal={result.signal}")

            return result

        except Exception as e:
            logger.error(f"Error checking RSI breakout for {symbol} on {timeframe}: {e}")
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

    def _create_alert_historical_level(
        self,
        symbol: str,
        company_name: str,
        historical_level: Dict,
        rsi_timeframe: str,
        rsi_result,
        df_weekly
    ) -> Dict:
        """
        Create alert for HISTORICAL SUPPORT LEVEL + RSI BREAKOUT

        Args:
            symbol: Stock symbol
            company_name: Company name
            historical_level: Historical level dict from find_historical_support_levels()
            rsi_timeframe: 'weekly' or 'daily'
            rsi_result: RSIBreakoutResult (optional)
            df_weekly: Weekly dataframe with EMAs

        Returns:
            Alert dictionary
        """
        crossover_info = historical_level['crossover_info']
        current_price = float(df_weekly['Close'].iloc[-1])
        latest = df_weekly.iloc[-1]

        alert = {
            'symbol': symbol,
            'company_name': company_name or symbol,
            'timeframe': 'weekly',  # Historical levels based on weekly crossovers
            'current_price': current_price,
            'support_level': historical_level['level'],
            'distance_to_support_pct': historical_level['distance_pct'],
            'support_type': 'historical_crossover',  # NEW: Indicate this is a historical level
            'ema_24': float(latest.get('EMA_24', 0)),
            'ema_38': float(latest.get('EMA_38', 0)),
            'ema_62': float(latest.get('EMA_62', 0)),
            'ema_alignment': True,  # Assume aligned if historical level is valid
            'crossover_date': crossover_info['date'].strftime('%Y-%m-%d'),
            'crossover_age_weeks': crossover_info.get('age_in_periods', 0),
            'crossover_type': crossover_info['type'],
            'crossover_emas': f"EMA{crossover_info['fast_ema']}xEMA{crossover_info['slow_ema']}",
            'crossover_info': json.dumps(crossover_info),
            'recommendation': self._get_recommendation_historical(historical_level, rsi_result),
            'is_notified': False
        }

        # Add RSI information if available
        if rsi_result:
            alert['has_rsi_breakout'] = rsi_result.has_rsi_breakout
            alert['rsi_signal'] = rsi_result.signal
            alert['rsi_timeframe'] = rsi_timeframe

            if rsi_result.rsi_breakout:
                alert['rsi_breakout_date'] = rsi_result.rsi_breakout.date.strftime('%Y-%m-%d')
                alert['rsi_breakout_value'] = rsi_result.rsi_breakout.rsi_value
                alert['rsi_breakout_strength'] = rsi_result.rsi_breakout.strength
                alert['rsi_breakout_age'] = rsi_result.rsi_breakout.age_in_periods

            if rsi_result.rsi_trendline:
                alert['rsi_trendline_peaks'] = len(rsi_result.rsi_trendline.peak_indices)
                alert['rsi_trendline_r2'] = rsi_result.rsi_trendline.r_squared
        else:
            alert['has_rsi_breakout'] = False
            alert['rsi_signal'] = 'NO_SIGNAL'

        return alert

    def _create_alert(
        self,
        symbol: str,
        company_name: str,
        analysis_result: Dict,
        timeframe: str,
        rsi_result=None,
        rsi_timeframe: str = None
    ) -> Dict:
        """
        Create an alert from analysis result AVEC RSI BREAKOUT

        Args:
            symbol: Stock symbol
            company_name: Company name
            analysis_result: Analysis result dictionary (EMA)
            timeframe: Timeframe of the EMA signal
            rsi_result: RSIBreakoutResult (optional)
            rsi_timeframe: Timeframe du RSI breakout (si différent du timeframe EMA)

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
            'recommendation': self._get_recommendation(analysis_result, rsi_result),
            'is_notified': False
        }

        # Ajouter les informations RSI si disponibles
        if rsi_result:
            alert['has_rsi_breakout'] = rsi_result.has_rsi_breakout
            alert['rsi_signal'] = rsi_result.signal
            alert['rsi_timeframe'] = rsi_timeframe or timeframe

            if rsi_result.rsi_breakout:
                alert['rsi_breakout_date'] = rsi_result.rsi_breakout.date.strftime('%Y-%m-%d')
                alert['rsi_breakout_value'] = rsi_result.rsi_breakout.rsi_value
                alert['rsi_breakout_strength'] = rsi_result.rsi_breakout.strength
                alert['rsi_breakout_age'] = rsi_result.rsi_breakout.age_in_periods

            if rsi_result.rsi_trendline:
                alert['rsi_trendline_peaks'] = len(rsi_result.rsi_trendline.peak_indices)
                alert['rsi_trendline_r2'] = rsi_result.rsi_trendline.r_squared
        else:
            alert['has_rsi_breakout'] = False
            alert['rsi_signal'] = 'NO_SIGNAL'

        return alert

    def _get_recommendation_historical(self, historical_level: Dict, rsi_result=None) -> str:
        """
        Get recommendation for HISTORICAL SUPPORT LEVEL

        LOGIQUE:
        - Prix proche niveau historique (< 8%) + RSI breakout → STRONG_BUY
        - Prix proche niveau historique + RSI trendline → BUY
        - Prix proche niveau historique seul → WATCH

        Args:
            historical_level: Historical level dict
            rsi_result: RSIBreakoutResult (optional)

        Returns:
            Recommendation string
        """
        distance = historical_level['distance_pct']

        # NIVEAU 1: RSI Breakout → STRONG_BUY
        if rsi_result and rsi_result.has_rsi_breakout:
            if distance <= 3.0:
                return 'STRONG_BUY'
            elif distance <= 6.0:
                return 'BUY'
            else:
                return 'WATCH'

        # NIVEAU 2: RSI Trendline (sans breakout) → BUY ou WATCH
        elif rsi_result and rsi_result.has_rsi_trendline:
            if distance <= 3.0:
                return 'BUY'
            elif distance <= 6.0:
                return 'WATCH'
            else:
                return 'OBSERVE'

        # NIVEAU 3: Niveau historique seul (pas de RSI) → WATCH ou OBSERVE
        else:
            if distance <= 4.0:
                return 'WATCH'
            else:
                return 'OBSERVE'

    def _get_recommendation(self, analysis_result: Dict, rsi_result=None) -> str:
        """
        Get recommendation based on EMA analysis AND RSI presence

        LOGIQUE FLEXIBLE:
        - STRONG_BUY: EMA signal + RSI breakout récent
        - BUY: EMA signal + RSI trendline (sans breakout)
        - WATCH: EMA signal seul (sans RSI)

        Args:
            analysis_result: Analysis result dictionary (EMA)
            rsi_result: RSIBreakoutResult (optional)

        Returns:
            Recommendation string
        """
        distance = analysis_result['distance_to_support_pct']

        # NIVEAU 1: RSI Breakout → STRONG_BUY
        if rsi_result and rsi_result.has_rsi_breakout:
            if distance <= 2.0:
                return 'STRONG_BUY'
            elif distance <= 4.0:
                return 'BUY'
            else:
                return 'WATCH'

        # NIVEAU 2: RSI Trendline (sans breakout) → BUY ou WATCH
        elif rsi_result and rsi_result.has_rsi_trendline:
            if distance <= 2.0:
                return 'BUY'
            elif distance <= 5.0:
                return 'WATCH'
            else:
                return 'OBSERVE'

        # NIVEAU 3: EMA seul (pas de RSI) → WATCH ou OBSERVE
        else:
            if distance <= 3.0:
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
