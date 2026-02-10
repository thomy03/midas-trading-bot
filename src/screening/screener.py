"""
Main screening logic for market analysis

Integrates:
- EMA historical support levels
- RSI trendline breakout detection
- Confidence scoring (0-100)
- Position sizing with portfolio tracking
"""
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd


def _json_serializable(obj: Any) -> Any:
    """Convert non-JSON-serializable objects to serializable format"""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, pd.Timedelta):
        return str(obj)
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    return str(obj)


def _serialize_crossover_info(crossover_info: Dict) -> str:
    """Serialize crossover_info dict to JSON, handling Timestamps"""
    serializable = {}
    for key, value in crossover_info.items():
        serializable[key] = _json_serializable(value) if not isinstance(value, (str, int, float, bool, type(None))) else value
    return json.dumps(serializable)


from config.settings import (
    TIMEFRAMES, MAX_WORKERS, LOOKBACK_DAYS_WEEKLY, LOOKBACK_DAYS_DAILY, CAPITAL
)
from src.data.market_data import market_data_fetcher
from src.indicators.ema_analyzer import ema_analyzer
from src.database.db_manager import db_manager
from src.utils.logger import logger

# Import RSI Breakout Analyzer (optional - removed in V6+)
try:
    from trendline_analysis.core.rsi_breakout_analyzer import RSIBreakoutAnalyzer
    from trendline_analysis.core.enhanced_rsi_breakout_analyzer import EnhancedRSIBreakoutAnalyzer
    TRENDLINE_AVAILABLE = True
except ImportError:
    RSIBreakoutAnalyzer = None
    EnhancedRSIBreakoutAnalyzer = None
    TRENDLINE_AVAILABLE = False

# Import Confidence Scorer and Position Sizer
from src.utils.confidence_scorer import confidence_scorer, ConfidenceScore
from src.utils.position_sizing import PositionSizer, PositionSize, format_position_recommendation

# Import Layer 2 Fundamental Scorer
from src.utils.fundamental_scorer import fundamental_scorer, FundamentalScorer

# Import Economic Calendar for earnings warnings
from src.utils.economic_calendar import economic_calendar


def validate_ema_at_breakout(df: pd.DataFrame, breakout_index: int) -> tuple:
    """
    Validate that EMAs were aligned (bullish) at the moment of breakout.

    This is CRITICAL: we must check EMA conditions at the breakout moment,
    not at the latest bar. A breakout with bearish EMAs is less reliable.

    Args:
        df: DataFrame with EMA columns (EMA_24, EMA_38, EMA_62)
        breakout_index: Index where the breakout occurred

    Returns:
        (is_valid, conditions_met, reason)
        - is_valid: True if at least 2/3 EMA conditions met at breakout
        - conditions_met: Number of conditions met (0-3)
        - reason: Description of the validation result
    """
    if df is None or df.empty:
        return (False, 0, "No data available")

    # Ensure breakout_index is within bounds
    if breakout_index < 0 or breakout_index >= len(df):
        return (False, 0, f"Invalid breakout index: {breakout_index}")

    # Get EMA values at breakout moment
    try:
        row = df.iloc[breakout_index]
        ema_24 = float(row.get('EMA_24', 0))
        ema_38 = float(row.get('EMA_38', 0))
        ema_62 = float(row.get('EMA_62', 0))
    except Exception as e:
        return (False, 0, f"Error reading EMAs: {e}")

    # Check if EMAs exist
    if ema_24 == 0 or ema_38 == 0 or ema_62 == 0:
        return (False, 0, "EMAs not calculated at breakout moment")

    # Count bullish conditions (same logic as elsewhere)
    conditions = [
        ema_24 > ema_38,  # Short > Medium
        ema_24 > ema_62,  # Short > Long
        ema_38 > ema_62   # Medium > Long
    ]
    conditions_met = sum(conditions)

    # At least 2/3 conditions for bullish alignment
    is_valid = conditions_met >= 2

    if is_valid:
        reason = f"EMAs bullish at breakout ({conditions_met}/3 conditions)"
    else:
        reason = f"EMAs NOT bullish at breakout ({conditions_met}/3 conditions)"

    return (is_valid, conditions_met, reason)


class MarketScreener:
    """Main market screener for detecting buy signals"""

    def __init__(
        self,
        use_enhanced_detector: bool = True,
        precision_mode: str = 'medium',
        total_capital: float = None,
        portfolio_file: str = None
    ):
        """
        Initialize the market screener

        Args:
            use_enhanced_detector: Si True, utilise le détecteur haute précision (recommandé)
            precision_mode: 'high', 'medium', ou 'low' (uniquement si use_enhanced_detector=True)
                - 'high': Stricte (R²>0.65) - Moins d'obliques mais excellente qualité
                - 'medium': Équilibré (R²>0.50) - Recommandé pour screening quotidien
                - 'low': Permissif (R²>0.35) - Plus d'obliques, qualité variable
            total_capital: Capital total de trading (default: settings.CAPITAL)
            portfolio_file: Fichier JSON pour persister les positions ouvertes
        """
        self.market_data = market_data_fetcher
        self.ema_analyzer = ema_analyzer
        self.db = db_manager

        # Choisir le détecteur RSI
        if use_enhanced_detector:
            self.rsi_analyzer = EnhancedRSIBreakoutAnalyzer(precision_mode=precision_mode)
            logger.info(f"Using ENHANCED RSI detector (precision={precision_mode})")
        else:
            self.rsi_analyzer = RSIBreakoutAnalyzer()
            logger.info("Using STANDARD RSI detector")

        self.use_enhanced_detector = use_enhanced_detector
        self.precision_mode = precision_mode if use_enhanced_detector else 'standard'

        # Position sizing avec portfolio tracking
        capital = total_capital if total_capital is not None else CAPITAL
        self.position_sizer = PositionSizer(
            total_capital=capital,
            portfolio_file=portfolio_file or 'data/portfolio.json'
        )
        logger.info(f"Position sizer initialized: capital={capital}EUR, "
                   f"available={self.position_sizer.get_available_capital():.2f}EUR")

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

            # Get EMA values for alignment check (needed for both weekly and daily fallback)
            latest = df_weekly.iloc[-1]
            ema_24 = float(latest.get('EMA_24', 0))
            ema_38 = float(latest.get('EMA_38', 0))
            ema_62 = float(latest.get('EMA_62', 0))

            # Check EMA alignment (at least 2 out of 3 conditions must be met)
            ema_conditions_met = sum([
                ema_24 > ema_38,  # Short > Medium
                ema_24 > ema_62,  # Short > Long
                ema_38 > ema_62   # Medium > Long
            ])
            has_bullish_ema_weekly = ema_conditions_met >= 2

            # ============================================================
            # LOGIC B: WEEKLY-ONLY for weekly support, DAILY FALLBACK separate
            # ============================================================

            # WEEKLY PATH: Support weekly proche -> RSI WEEKLY uniquement
            if near_levels and has_bullish_ema_weekly:
                logger.info(f"{symbol}: {len(near_levels)} NEAR weekly support(s) + EMA bullish ({ema_conditions_met}/3) -> Checking RSI WEEKLY...")

                for level in near_levels:
                    level_price = level['level']
                    distance_pct = level['distance_pct']

                    logger.info(f"{symbol}: Price ${current_price:.2f} approaching weekly level ${level_price:.2f} ({distance_pct:.1f}%)")

                    # Check RSI on WEEKLY ONLY (pass df_weekly for EMA validation at breakout)
                    rsi_weekly = self._check_rsi_breakout(symbol, 'weekly', df_with_emas=df_weekly)

                    if rsi_weekly and rsi_weekly.has_rsi_breakout:
                        logger.info(f"{symbol}: WEEKLY SUPPORT + RSI BREAKOUT WEEKLY -> STRONG SIGNAL!")
                        return self._create_alert_historical_level(
                            symbol, company_name, level, 'weekly', rsi_weekly, df_weekly
                        )

                    if rsi_weekly and rsi_weekly.has_rsi_trendline:
                        logger.info(f"{symbol}: WEEKLY SUPPORT + RSI trendline WEEKLY -> BUY signal")
                        return self._create_alert_historical_level(
                            symbol, company_name, level, 'weekly', rsi_weekly, df_weekly
                        )

                # Near weekly level but no RSI weekly signal -> WATCH
                logger.info(f"{symbol}: Near weekly support but no RSI weekly signal -> WATCH")
                return self._create_alert_historical_level(
                    symbol, company_name, near_levels[0], 'weekly', None, df_weekly
                )

            # DAILY FALLBACK: No weekly signal -> check DAILY (support daily + RSI daily)
            if has_bullish_ema_weekly:
                logger.debug(f"{symbol}: No weekly signal, EMA bullish ({ema_conditions_met}/3) -> checking DAILY FALLBACK...")
                return self._check_daily_fallback(symbol, company_name, df_weekly, ema_conditions_met)
            else:
                logger.debug(f"{symbol}: No near weekly support and EMA not bullish ({ema_conditions_met}/3) -> no signal")
                return None

        except Exception as e:
            logger.error(f"Error screening {symbol}: {e}")
            return None

    def _check_rsi_breakout(self, symbol: str, timeframe: str, df_with_emas: pd.DataFrame = None):
        """
        Vérifie si un RSI breakout existe sur le timeframe donné

        IMPORTANT: Validates EMA alignment at the moment of breakout (not latest bar)

        Args:
            symbol: Stock symbol
            timeframe: 'weekly' or 'daily'
            df_with_emas: Optional DataFrame with EMAs already calculated (for validation)

        Returns:
            RSIBreakoutResult ou None (with ema_valid_at_breakout attribute added)
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

                # CRITICAL: Validate EMA alignment at breakout moment
                if result.has_rsi_breakout and result.rsi_breakout:
                    breakout_idx = result.rsi_breakout.index

                    # Use provided df_with_emas or calculate EMAs on current df
                    if df_with_emas is not None:
                        validation_df = df_with_emas
                    else:
                        validation_df = self.ema_analyzer.calculate_emas(df)

                    is_valid, conditions, reason = validate_ema_at_breakout(
                        validation_df, breakout_idx
                    )

                    # Add validation info to result (as attributes)
                    result.ema_valid_at_breakout = is_valid
                    result.ema_conditions_at_breakout = conditions
                    result.ema_validation_reason = reason

                    if not is_valid:
                        logger.info(f"{symbol}: RSI breakout detected but {reason} - weakening signal")
                        # Downgrade signal if EMAs were not bullish at breakout
                        if result.signal == 'STRONG_BUY':
                            result.signal = 'BUY'
                        elif result.signal == 'BUY':
                            result.signal = 'WATCH'
                else:
                    # No breakout, set defaults
                    result.ema_valid_at_breakout = True
                    result.ema_conditions_at_breakout = 0
                    result.ema_validation_reason = "No breakout to validate"

            return result

        except Exception as e:
            logger.error(f"Error checking RSI breakout for {symbol} on {timeframe}: {e}")
            return None

    def _check_daily_fallback(
        self,
        symbol: str,
        company_name: str,
        df_weekly: pd.DataFrame,
        weekly_ema_conditions: int
    ) -> Optional[Dict]:
        """
        DAILY FALLBACK: Check daily timeframe when weekly has no near support
        but weekly EMAs are bullish (2/3 conditions met)

        Conditions for daily signal:
        1. Weekly EMAs bullish (already verified by caller)
        2. Daily EMAs bullish (at least 2/3 conditions)
        3. Daily price near support level (< 8%)
        4. Daily RSI breakout

        Args:
            symbol: Stock symbol
            company_name: Company name
            df_weekly: Weekly dataframe (for reference)
            weekly_ema_conditions: Number of weekly EMA conditions met (for logging)

        Returns:
            Alert dictionary if daily signal found, None otherwise
        """
        try:
            # Step 1: Get daily data with EMAs
            df_daily = self.market_data.get_historical_data(
                symbol,
                period='1y',
                interval=TIMEFRAMES['daily']
            )

            if df_daily is None or df_daily.empty:
                logger.debug(f"{symbol}: No daily data for fallback check")
                return None

            df_daily = self.ema_analyzer.calculate_emas(df_daily)
            current_price = float(df_daily['Close'].iloc[-1])

            # Step 2: Check daily EMA alignment (at least 2/3 conditions)
            latest_daily = df_daily.iloc[-1]
            ema_24_d = float(latest_daily.get('EMA_24', 0))
            ema_38_d = float(latest_daily.get('EMA_38', 0))
            ema_62_d = float(latest_daily.get('EMA_62', 0))

            daily_ema_conditions = sum([
                ema_24_d > ema_38_d,
                ema_24_d > ema_62_d,
                ema_38_d > ema_62_d
            ])
            has_bullish_ema_daily = daily_ema_conditions >= 2

            if not has_bullish_ema_daily:
                logger.debug(f"{symbol}: Daily EMA not bullish ({daily_ema_conditions}/3) - no daily fallback")
                return None

            # Step 3: Detect daily crossovers and find near support levels
            crossovers_daily = self.ema_analyzer.detect_crossovers(df_daily, 'daily')

            if not crossovers_daily:
                logger.debug(f"{symbol}: No daily crossovers found for fallback")
                return None

            historical_levels_daily = self.ema_analyzer.find_historical_support_levels(
                df_daily, crossovers_daily, current_price
            )

            near_levels_daily = [level for level in historical_levels_daily if level['is_near']]

            if not near_levels_daily:
                logger.debug(f"{symbol}: No near daily support levels for fallback")
                return None

            logger.info(f"{symbol}: DAILY FALLBACK - {len(near_levels_daily)} near support(s), EMA bullish W:{weekly_ema_conditions}/3 D:{daily_ema_conditions}/3")

            # Step 4: Check RSI breakout on daily (pass df_daily for EMA validation at breakout)
            for level in near_levels_daily:
                rsi_daily = self._check_rsi_breakout(symbol, 'daily', df_with_emas=df_daily)

                if rsi_daily and rsi_daily.has_rsi_breakout:
                    logger.info(f"{symbol}: DAILY FALLBACK + RSI BREAKOUT -> SIGNAL!")
                    return self._create_alert_daily_fallback(
                        symbol, company_name, level, rsi_daily, df_daily, df_weekly,
                        weekly_ema_conditions, daily_ema_conditions
                    )

                # Also accept RSI trendline (weaker signal)
                if rsi_daily and rsi_daily.has_rsi_trendline:
                    logger.info(f"{symbol}: DAILY FALLBACK + RSI trendline -> BUY signal")
                    return self._create_alert_daily_fallback(
                        symbol, company_name, level, rsi_daily, df_daily, df_weekly,
                        weekly_ema_conditions, daily_ema_conditions
                    )

            logger.debug(f"{symbol}: Daily fallback - near support but no RSI signal")
            return None

        except Exception as e:
            logger.error(f"Error in daily fallback for {symbol}: {e}")
            return None

    def _create_alert_daily_fallback(
        self,
        symbol: str,
        company_name: str,
        daily_level: Dict,
        rsi_result,
        df_daily: pd.DataFrame,
        df_weekly: pd.DataFrame,
        weekly_ema_conditions: int,
        daily_ema_conditions: int
    ) -> Dict:
        """
        Create alert for DAILY FALLBACK signal

        Args:
            symbol: Stock symbol
            company_name: Company name
            daily_level: Daily support level dict
            rsi_result: RSI breakout result
            df_daily: Daily dataframe
            df_weekly: Weekly dataframe (for reference)
            weekly_ema_conditions: Weekly EMA conditions met
            daily_ema_conditions: Daily EMA conditions met

        Returns:
            Alert dictionary
        """
        crossover_info = daily_level['crossover_info']
        current_price = float(df_daily['Close'].iloc[-1])
        latest_daily = df_daily.iloc[-1]
        latest_weekly = df_weekly.iloc[-1]

        ema_24_d = float(latest_daily.get('EMA_24', 0))
        ema_38_d = float(latest_daily.get('EMA_38', 0))
        ema_62_d = float(latest_daily.get('EMA_62', 0))

        # Prepare RSI data for scoring
        rsi_breakout_data = None
        rsi_trendline_data = None
        rsi_peak_indices = []

        if rsi_result:
            if rsi_result.rsi_breakout:
                rsi_breakout_data = {
                    'strength': rsi_result.rsi_breakout.strength,
                    'age_in_periods': rsi_result.rsi_breakout.age_in_periods,
                    'index': rsi_result.rsi_breakout.index,
                    'volume_ratio': getattr(rsi_result.rsi_breakout, 'volume_ratio', 1.0)
                }
            if rsi_result.rsi_trendline:
                rsi_trendline_data = {
                    'r_squared': rsi_result.rsi_trendline.r_squared
                }
                rsi_peak_indices = list(rsi_result.rsi_trendline.peak_indices)

        # Calculate confidence score
        confidence = confidence_scorer.calculate_score(
            df=df_daily,
            ema_24=ema_24_d,
            ema_38=ema_38_d,
            ema_62=ema_62_d,
            current_price=current_price,
            support_level=daily_level['level'],
            rsi_breakout=rsi_breakout_data,
            rsi_trendline=rsi_trendline_data
        )

        # Calculate position sizing
        market_cap = self._get_market_cap(symbol)
        position = self.position_sizer.calculate(
            df=df_daily,
            entry_price=current_price,
            symbol=symbol,
            market_cap=market_cap,
            rsi_peaks_indices=rsi_peak_indices
        )

        alert = {
            'symbol': symbol,
            'company_name': company_name or symbol,
            'timeframe': 'daily',  # DAILY FALLBACK
            'current_price': current_price,
            'support_level': daily_level['level'],
            'distance_to_support_pct': daily_level['distance_pct'],
            'support_type': 'daily_fallback',  # Mark as daily fallback
            'ema_24': ema_24_d,
            'ema_38': ema_38_d,
            'ema_62': ema_62_d,
            'ema_alignment': True,
            'ema_weekly_conditions': weekly_ema_conditions,
            'ema_daily_conditions': daily_ema_conditions,
            'crossover_date': crossover_info['date'].strftime('%Y-%m-%d'),
            'crossover_age_weeks': crossover_info.get('age_in_periods', 0),
            'crossover_type': crossover_info['type'],
            'crossover_emas': f"EMA{crossover_info['fast_ema']}xEMA{crossover_info['slow_ema']}",
            'crossover_info': _serialize_crossover_info(crossover_info),
            'recommendation': confidence.signal,
            'is_notified': False,

            # Confidence scoring
            'confidence_score': confidence.total_score,
            'confidence_signal': f"{confidence.signal} ({confidence.total_score:.0f}/100)",
            'score_ema': confidence.ema_score,
            'score_support': confidence.support_score,
            'score_rsi': confidence.rsi_score,
            'score_freshness': confidence.freshness_score,
            'score_volume': confidence.volume_score,

            # Position sizing
            'position_shares': position.shares,
            'position_value': position.position_value,
            'stop_loss': position.stop_loss,
            'stop_source': position.stop_source,
            'risk_amount': position.risk_amount,
            'risk_pct': position.risk_pct,
            'market_cap_b': market_cap
        }

        # Add RSI information
        if rsi_result:
            alert['has_rsi_breakout'] = rsi_result.has_rsi_breakout
            alert['rsi_signal'] = rsi_result.signal
            alert['rsi_timeframe'] = 'daily'

            if rsi_result.rsi_breakout:
                alert['rsi_breakout_date'] = rsi_result.rsi_breakout.date.strftime('%Y-%m-%d')
                alert['rsi_breakout_value'] = rsi_result.rsi_breakout.rsi_value
                alert['rsi_breakout_strength'] = rsi_result.rsi_breakout.strength
                alert['rsi_breakout_age'] = rsi_result.rsi_breakout.age_in_periods
                # Volume confirmation at breakout
                alert['volume_ratio'] = getattr(rsi_result.rsi_breakout, 'volume_ratio', 1.0)
                alert['volume_confirmed'] = getattr(rsi_result.rsi_breakout, 'volume_confirmed', True)

            if rsi_result.rsi_trendline:
                alert['rsi_trendline_peaks'] = len(rsi_result.rsi_trendline.peak_indices)
                alert['rsi_trendline_r2'] = rsi_result.rsi_trendline.r_squared
                # Store actual peak indices for Pro Chart to display the exact same trendline
                alert['rsi_peak_indices'] = list(rsi_result.rsi_trendline.peak_indices)

            # Add EMA validation at breakout info
            alert['ema_valid_at_breakout'] = getattr(rsi_result, 'ema_valid_at_breakout', True)
            alert['ema_conditions_at_breakout'] = getattr(rsi_result, 'ema_conditions_at_breakout', 0)
            alert['ema_validation_reason'] = getattr(rsi_result, 'ema_validation_reason', '')
        else:
            alert['has_rsi_breakout'] = False
            alert['rsi_signal'] = 'NO_SIGNAL'
            alert['ema_valid_at_breakout'] = True
            alert['ema_conditions_at_breakout'] = 0
            alert['ema_validation_reason'] = 'No RSI result'
            alert['volume_ratio'] = 1.0
            alert['volume_confirmed'] = True

        # Add earnings warning if applicable
        try:
            has_earnings_soon, days_until = economic_calendar.has_earnings_soon(symbol, days_threshold=7)
            alert['has_earnings_soon'] = has_earnings_soon
            alert['days_until_earnings'] = days_until
            alert['earnings_warning'] = economic_calendar.get_earnings_warning(symbol)
        except Exception as e:
            logger.debug(f"Could not fetch earnings for {symbol}: {e}")
            alert['has_earnings_soon'] = False
            alert['days_until_earnings'] = None
            alert['earnings_warning'] = None

        return alert

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
        WITH confidence scoring and position sizing

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

        ema_24 = float(latest.get('EMA_24', 0))
        ema_38 = float(latest.get('EMA_38', 0))
        ema_62 = float(latest.get('EMA_62', 0))

        # Prepare RSI data for scoring
        rsi_breakout_data = None
        rsi_trendline_data = None
        rsi_peak_indices = []

        if rsi_result:
            if rsi_result.rsi_breakout:
                rsi_breakout_data = {
                    'strength': rsi_result.rsi_breakout.strength,
                    'age_in_periods': rsi_result.rsi_breakout.age_in_periods,
                    'index': rsi_result.rsi_breakout.index,
                    'volume_ratio': getattr(rsi_result.rsi_breakout, 'volume_ratio', 1.0)
                }
            if rsi_result.rsi_trendline:
                rsi_trendline_data = {
                    'r_squared': rsi_result.rsi_trendline.r_squared
                }
                rsi_peak_indices = list(rsi_result.rsi_trendline.peak_indices)

        # Calculate confidence score
        confidence = confidence_scorer.calculate_score(
            df=df_weekly,
            ema_24=ema_24,
            ema_38=ema_38,
            ema_62=ema_62,
            current_price=current_price,
            support_level=historical_level['level'],
            rsi_breakout=rsi_breakout_data,
            rsi_trendline=rsi_trendline_data
        )

        # Calculate position sizing
        market_cap = self._get_market_cap(symbol)
        position = self.position_sizer.calculate(
            df=df_weekly,
            entry_price=current_price,
            symbol=symbol,
            market_cap=market_cap,
            rsi_peaks_indices=rsi_peak_indices
        )

        alert = {
            'symbol': symbol,
            'company_name': company_name or symbol,
            'timeframe': 'weekly',  # Historical levels based on weekly crossovers
            'current_price': current_price,
            'support_level': historical_level['level'],
            'distance_to_support_pct': historical_level['distance_pct'],
            'support_type': 'historical_crossover',  # NEW: Indicate this is a historical level
            'ema_24': ema_24,
            'ema_38': ema_38,
            'ema_62': ema_62,
            'ema_alignment': True,  # Assume aligned if historical level is valid
            'crossover_date': crossover_info['date'].strftime('%Y-%m-%d'),
            'crossover_age_weeks': crossover_info.get('age_in_periods', 0),
            'crossover_type': crossover_info['type'],
            'crossover_emas': f"EMA{crossover_info['fast_ema']}xEMA{crossover_info['slow_ema']}",
            'crossover_info': _serialize_crossover_info(crossover_info),
            'recommendation': confidence.signal,  # Use score-based signal
            'is_notified': False,

            # Confidence scoring
            'confidence_score': confidence.total_score,
            'confidence_signal': f"{confidence.signal} ({confidence.total_score:.0f}/100)",
            'score_ema': confidence.ema_score,
            'score_support': confidence.support_score,
            'score_rsi': confidence.rsi_score,
            'score_freshness': confidence.freshness_score,
            'score_volume': confidence.volume_score,

            # Position sizing
            'position_shares': position.shares,
            'position_value': position.position_value,
            'stop_loss': position.stop_loss,
            'stop_source': position.stop_source,
            'risk_amount': position.risk_amount,
            'risk_pct': position.risk_pct,
            'market_cap_b': market_cap
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
                # Volume confirmation at breakout
                alert['volume_ratio'] = getattr(rsi_result.rsi_breakout, 'volume_ratio', 1.0)
                alert['volume_confirmed'] = getattr(rsi_result.rsi_breakout, 'volume_confirmed', True)

            if rsi_result.rsi_trendline:
                alert['rsi_trendline_peaks'] = len(rsi_result.rsi_trendline.peak_indices)
                alert['rsi_trendline_r2'] = rsi_result.rsi_trendline.r_squared
                # Store actual peak indices for Pro Chart to display the exact same trendline
                alert['rsi_peak_indices'] = list(rsi_result.rsi_trendline.peak_indices)

            # Add EMA validation at breakout info
            alert['ema_valid_at_breakout'] = getattr(rsi_result, 'ema_valid_at_breakout', True)
            alert['ema_conditions_at_breakout'] = getattr(rsi_result, 'ema_conditions_at_breakout', 0)
            alert['ema_validation_reason'] = getattr(rsi_result, 'ema_validation_reason', '')
        else:
            alert['has_rsi_breakout'] = False
            alert['rsi_signal'] = 'NO_SIGNAL'
            alert['ema_valid_at_breakout'] = True
            alert['ema_conditions_at_breakout'] = 0
            alert['ema_validation_reason'] = 'No RSI result'
            alert['volume_ratio'] = 1.0
            alert['volume_confirmed'] = True

        # Add earnings warning if applicable
        try:
            has_earnings_soon, days_until = economic_calendar.has_earnings_soon(symbol, days_threshold=7)
            alert['has_earnings_soon'] = has_earnings_soon
            alert['days_until_earnings'] = days_until
            alert['earnings_warning'] = economic_calendar.get_earnings_warning(symbol)
        except Exception as e:
            logger.debug(f"Could not fetch earnings for {symbol}: {e}")
            alert['has_earnings_soon'] = False
            alert['days_until_earnings'] = None
            alert['earnings_warning'] = None

        return alert

    def _get_market_cap(self, symbol: str) -> float:
        """
        Get market cap in billions for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Market cap in billions USD, or None if unavailable
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            market_cap = info.get('marketCap', 0)
            if market_cap and market_cap > 0:
                return market_cap / 1e9  # Convert to billions
            return None
        except Exception as e:
            logger.debug(f"Could not get market cap for {symbol}: {e}")
            return None

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
            'crossover_info': _serialize_crossover_info(crossover_info),
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
                # Store actual peak indices for Pro Chart to display the exact same trendline
                alert['rsi_peak_indices'] = list(rsi_result.rsi_trendline.peak_indices)
        else:
            alert['has_rsi_breakout'] = False
            alert['rsi_signal'] = 'NO_SIGNAL'

        # Add earnings warning if applicable
        try:
            has_earnings_soon, days_until = economic_calendar.has_earnings_soon(symbol, days_threshold=7)
            alert['has_earnings_soon'] = has_earnings_soon
            alert['days_until_earnings'] = days_until
            alert['earnings_warning'] = economic_calendar.get_earnings_warning(symbol)
        except Exception as e:
            logger.debug(f"Could not fetch earnings for {symbol}: {e}")
            alert['has_earnings_soon'] = False
            alert['days_until_earnings'] = None
            alert['earnings_warning'] = None

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

        # Apply Layer 2 scoring to STRONG_BUY signals
        alerts = self.apply_layer2_scoring(alerts)

        return alerts

    def apply_layer2_scoring(self, alerts: List[Dict]) -> List[Dict]:
        """
        Apply Layer 2 (Fundamental & Sentiment) scoring to STRONG_BUY alerts

        This method:
        1. Filters alerts to find STRONG_BUY signals
        2. Applies fundamental scoring (earnings, margins, sector context)
        3. Returns all alerts with L2 scores added to STRONG_BUY ones

        Args:
            alerts: List of alerts from Layer 1 screening

        Returns:
            List of alerts with L2 data added to STRONG_BUY signals
        """
        if not alerts:
            return alerts

        # Separate STRONG_BUY from others
        strong_buys = [a for a in alerts if a.get('confidence_signal', '').startswith('STRONG_BUY')]
        others = [a for a in alerts if not a.get('confidence_signal', '').startswith('STRONG_BUY')]

        if not strong_buys:
            logger.info("[L2] No STRONG_BUY signals to analyze")
            return alerts

        logger.info(f"[L2] Found {len(strong_buys)} STRONG_BUY signals for Layer 2 analysis")

        # Apply Layer 2 scoring
        scored_strong_buys = fundamental_scorer.score_strong_buys(strong_buys)

        # Count results
        elite_count = sum(1 for a in scored_strong_buys if a.get('l2_is_elite', False))
        excluded_count = sum(1 for a in scored_strong_buys if a.get('l2_excluded', False))

        logger.info(f"[L2] Results: {elite_count} ELITE, {excluded_count} excluded, "
                   f"{len(strong_buys) - elite_count - excluded_count} standard")

        # Combine back: scored STRONG_BUY first (sorted by L2 score), then others
        return scored_strong_buys + others

    def get_elite_signals(self, alerts: List[Dict]) -> List[Dict]:
        """
        Get only ELITE signals from scored alerts

        Args:
            alerts: List of alerts with L2 scores

        Returns:
            List of ELITE signals (L2 score > threshold, not excluded)
        """
        return [
            alert for alert in alerts
            if not alert.get('l2_excluded', True)
            and alert.get('l2_is_elite', False)
        ]

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


    # ==========================================
    # PORTFOLIO MANAGEMENT METHODS
    # ==========================================

    def get_portfolio_summary(self) -> Dict:
        """
        Get current portfolio summary

        Returns:
            Dict with total_capital, invested, available, positions
        """
        return self.position_sizer.get_portfolio_summary()

    def get_available_capital(self) -> float:
        """Get capital available for new positions"""
        return self.position_sizer.get_available_capital()

    def update_capital(self, new_capital: float):
        """
        Update total trading capital

        Args:
            new_capital: New total capital in EUR
        """
        self.position_sizer.update_capital(new_capital)
        logger.info(f"Capital updated to {new_capital:.2f}EUR")

    def add_position(self, symbol: str, shares: int, entry_price: float, stop_loss: float):
        """
        Manually add an open position

        Args:
            symbol: Stock symbol
            shares: Number of shares
            entry_price: Entry price per share
            stop_loss: Stop-loss price
        """
        from src.utils.position_sizing import OpenPosition
        self.position_sizer.portfolio.add_position(
            symbol=symbol,
            shares=shares,
            entry_price=entry_price,
            stop_loss=stop_loss
        )
        logger.info(f"Position added: {symbol} {shares} shares @ {entry_price:.2f}EUR")

    def close_position(self, symbol: str) -> bool:
        """
        Close a position by symbol

        Args:
            symbol: Stock symbol to close

        Returns:
            True if position was closed, False if not found
        """
        closed = self.position_sizer.close_position(symbol)
        if closed:
            logger.info(f"Position closed: {symbol}")
            return True
        else:
            logger.warning(f"Position not found: {symbol}")
            return False

    def list_positions(self) -> List[Dict]:
        """
        List all open positions

        Returns:
            List of position dictionaries
        """
        summary = self.get_portfolio_summary()
        return summary.get('positions', [])


# Singleton instance with ENHANCED detector (medium precision - balanced quality/quantity)
# Pour utiliser l'ancien détecteur: MarketScreener(use_enhanced_detector=False)
# Pour mode high precision: MarketScreener(precision_mode='high')
# Pour mode low precision: MarketScreener(precision_mode='low')
market_screener = MarketScreener(use_enhanced_detector=True, precision_mode='medium')
