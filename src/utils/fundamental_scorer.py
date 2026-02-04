"""
Fundamental Scorer - Layer 2 scoring for STRONG_BUY signals

Applies fundamental and sentiment analysis to filter STRONG_BUY signals
and identify ELITE opportunities (score > 45/60).

Components:
- Kill Switches: Earnings proximity, low volume
- Company Health: Revenue growth, margins, earnings surprise (20 pts)
- Market Context: SPY trend, sector ETF trend (10 pts)
- Sentiment (Phase 2): Grok API analysis (30 pts)
"""

import yfinance as yf
import pandas as pd
import requests
import json
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from src.utils.logger import logger
from config import settings


@dataclass
class FundamentalScore:
    """Result of Layer 2 fundamental scoring"""
    symbol: str
    total_score: float  # 0-60 (30 without Grok)
    health_score: float  # 0-20
    context_score: float  # 0-10 (20 for CRYPTO)
    sentiment_score: float  # 0-30 (40 for CRYPTO)
    is_elite: bool  # score > 45
    excluded: bool  # True if killed by filter
    exclusion_reason: Optional[str]
    asset_type: str  # EQUITY, CRYPTO_PROXY, CRYPTO
    details: Dict


class FundamentalScorer:
    """
    Layer 2 Scoring for STRONG_BUY signals

    Scoring composition:
    - Company Health: 20 points (revenue growth, margins, earnings surprise)
    - Market Context: 10 points (SPY trend, sector ETF trend)
    - Sentiment (Phase 2): 30 points (Grok API)

    Total: 60 points max (30 without Grok in Phase 1)
    ELITE threshold: > 45/60 (> 22/30 in Phase 1)
    """

    # Mapping sector yfinance -> ETF US
    SECTOR_ETF_MAP = {
        'Technology': 'XLK',
        'Financial Services': 'XLF',
        'Financials': 'XLF',
        'Energy': 'XLE',
        'Healthcare': 'XLV',
        'Health Care': 'XLV',
        'Consumer Cyclical': 'XLY',
        'Consumer Discretionary': 'XLY',
        'Industrials': 'XLI',
        'Basic Materials': 'XLB',
        'Materials': 'XLB',
        'Utilities': 'XLU',
        'Real Estate': 'XLRE',
        'Communication Services': 'XLC',
        'Consumer Defensive': 'XLP',
        'Consumer Staples': 'XLP',
    }

    # Crypto-proxy stocks (Bitcoin miners, crypto exchanges, etc.)
    CRYPTO_PROXY_SYMBOLS = [
        'COIN',   # Coinbase
        'MSTR',   # MicroStrategy
        'MARA',   # Marathon Digital
        'RIOT',   # Riot Platforms
        'CLSK',   # CleanSpark
        'HUT',    # Hut 8 Mining
        'BITF',   # Bitfarms
        'CORZ',   # Core Scientific
        'WULF',   # TeraWulf
        'CIFR',   # Cipher Mining
        'IREN',   # Iris Energy
    ]

    # Kill switch thresholds
    MIN_VOLUME = 200_000  # Minimum average volume for stocks
    MIN_CRYPTO_VOLUME_USD = 5_000_000  # $5M minimum for crypto
    EARNINGS_SAFETY_DAYS = 5  # Days before earnings to exclude
    HYPER_GROWTH_THRESHOLD = 0.40  # 40% revenue growth "pardons" negative margins

    # Scoring thresholds
    ELITE_THRESHOLD = 45  # Full scoring with Grok
    ELITE_THRESHOLD_PHASE1 = 22  # Phase 1 without Grok (30 pts max)
    
    # EU thresholds (50% of US due to lower valuations)
    EU_ELITE_THRESHOLD = 23  # 50% of 45
    EU_ELITE_THRESHOLD_PHASE1 = 11  # 50% of 22
    
    # EU suffixes
    EU_SUFFIXES = ['.PA', '.DE', '.AS', '.BR', '.LS', '.MI', '.MC', '.L']

    def __init__(self, use_grok: bool = False, grok_api_key: str = None):
        """
        Initialize FundamentalScorer

        Args:
            use_grok: Enable Grok API sentiment analysis (Phase 2)
            grok_api_key: API key for xAI Grok
        """
        self.use_grok = use_grok
        self.grok_api_key = grok_api_key
        self._spy_trend = None  # Cache SPY trend
        self._btc_trend = None  # Cache BTC trend for crypto
        self._etf_cache = {}  # Cache ETF trends

    def _detect_asset_type(self, symbol: str) -> str:
        """
        Detect asset type for appropriate scoring mode

        Returns:
            'CRYPTO' - Pure crypto (BTC-USD, ETH-USD, etc.)
            'CRYPTO_PROXY' - Crypto-related stocks (COIN, MSTR, miners)
            'EQUITY' - Standard stocks
        """
        # Pure crypto (yfinance format: BTC-USD, ETH-EUR, etc.)
        if '-' in symbol and symbol.split('-')[1] in ['USD', 'EUR', 'GBP', 'JPY', 'USDT']:
            return 'CRYPTO'

        # Crypto-proxy stocks
        if symbol.upper() in self.CRYPTO_PROXY_SYMBOLS:
            return 'CRYPTO_PROXY'
    
    def is_eu_symbol(self, symbol: str) -> bool:
        """Check if symbol is European based on suffix"""
        return any(symbol.upper().endswith(suffix) for suffix in self.EU_SUFFIXES)

        return 'EQUITY'

    def score_strong_buys(self, alerts: List[Dict]) -> List[Dict]:
        """
        Apply Layer 2 scoring to all STRONG_BUY alerts

        Args:
            alerts: List of STRONG_BUY alerts from Layer 1

        Returns:
            List of alerts with L2 scores added, sorted by score
        """
        if not alerts:
            return []

        logger.info(f"[L2] Starting Layer 2 scoring for {len(alerts)} STRONG_BUY signals...")

        # Pre-fetch SPY trend (used for all)
        self._spy_trend = self._get_spy_trend()

        scored_alerts = []
        elite_count = 0
        excluded_count = 0

        for alert in alerts:
            symbol = alert.get('symbol', 'UNKNOWN')

            try:
                score = self.score_single(symbol, alert)

                # Add L2 data to alert
                alert['l2_score'] = score.total_score
                alert['l2_health'] = score.health_score
                alert['l2_context'] = score.context_score
                alert['l2_sentiment'] = score.sentiment_score
                alert['l2_is_elite'] = score.is_elite
                alert['l2_excluded'] = score.excluded
                alert['l2_exclusion_reason'] = score.exclusion_reason
                alert['l2_asset_type'] = score.asset_type
                alert['l2_details'] = score.details

                if score.excluded:
                    excluded_count += 1
                    logger.info(f"[L2] {symbol}: EXCLUDED - {score.exclusion_reason}")
                elif score.is_elite:
                    elite_count += 1
                    logger.info(f"[L2] {symbol}: ELITE ({score.total_score:.0f}/60)")
                else:
                    logger.debug(f"[L2] {symbol}: {score.total_score:.0f}/60")

                scored_alerts.append(alert)

            except Exception as e:
                logger.error(f"[L2] Error scoring {symbol}: {e}")
                alert['l2_score'] = 0
                alert['l2_excluded'] = True
                alert['l2_exclusion_reason'] = f"Error: {str(e)[:50]}"
                scored_alerts.append(alert)

        # Sort by L2 score (highest first), non-excluded first
        scored_alerts.sort(
            key=lambda x: (not x.get('l2_excluded', True), x.get('l2_score', 0)),
            reverse=True
        )

        logger.info(f"[L2] Completed: {elite_count} ELITE, {excluded_count} excluded, "
                   f"{len(alerts) - excluded_count - elite_count} standard")

        return scored_alerts

    def score_single(self, symbol: str, alert: Dict) -> FundamentalScore:
        """
        Score a single STRONG_BUY signal

        Args:
            symbol: Stock symbol
            alert: Alert dictionary from Layer 1

        Returns:
            FundamentalScore with detailed breakdown
        """
        details = {}

        # Detect asset type
        asset_type = self._detect_asset_type(symbol)
        details['asset_type'] = asset_type

        # Get ticker info
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
        except Exception as e:
            return FundamentalScore(
                symbol=symbol,
                total_score=0,
                health_score=0,
                context_score=0,
                sentiment_score=0,
                is_elite=False,
                excluded=True,
                exclusion_reason=f"Cannot fetch data: {str(e)[:30]}",
                asset_type=asset_type,
                details={}
            )

        # === KILL SWITCHES ===

        # 1. Check earnings proximity (skip for pure crypto)
        if asset_type != 'CRYPTO':
            is_near_earnings, earnings_info = self._check_earnings_proximity(ticker)
            details['earnings'] = earnings_info

            if is_near_earnings:
                return FundamentalScore(
                    symbol=symbol,
                    total_score=0,
                    health_score=0,
                    context_score=0,
                    sentiment_score=0,
                    is_elite=False,
                    excluded=True,
                    exclusion_reason=f"Earnings in {earnings_info.get('days_to_earnings', '?')} days",
                    asset_type=asset_type,
                    details=details
                )

        # 2. Check volume (different thresholds by asset type)
        if asset_type == 'CRYPTO':
            # For crypto, check 24h volume in USD
            volume_24h = info.get('volume24Hr', 0) or info.get('regularMarketVolume', 0)
            price = info.get('regularMarketPrice', 1)
            volume_usd = volume_24h * price if volume_24h else 0
            details['volume'] = {
                'volume_24h_usd': volume_usd,
                'min_required_usd': self.MIN_CRYPTO_VOLUME_USD
            }

            if volume_usd < self.MIN_CRYPTO_VOLUME_USD:
                return FundamentalScore(
                    symbol=symbol,
                    total_score=0,
                    health_score=0,
                    context_score=0,
                    sentiment_score=0,
                    is_elite=False,
                    excluded=True,
                    exclusion_reason=f"Low crypto volume (${volume_usd/1e6:.1f}M < ${self.MIN_CRYPTO_VOLUME_USD/1e6:.0f}M)",
                    asset_type=asset_type,
                    details=details
                )
        else:
            # Standard stock volume check
            avg_volume = info.get('averageVolume', 0)
            details['volume'] = {'average_volume': avg_volume, 'min_required': self.MIN_VOLUME}

            if avg_volume < self.MIN_VOLUME:
                return FundamentalScore(
                    symbol=symbol,
                    total_score=0,
                    health_score=0,
                    context_score=0,
                    sentiment_score=0,
                    is_elite=False,
                    excluded=True,
                    exclusion_reason=f"Low volume ({avg_volume:,.0f} < {self.MIN_VOLUME:,.0f})",
                    asset_type=asset_type,
                    details=details
                )

        # === SCORING BY ASSET TYPE ===

        if asset_type == 'CRYPTO':
            # CRYPTO MODE: Health=0, Context=20, Sentiment=40
            health_score = 0
            details['health'] = {'status': 'N/A for pure crypto', 'total': 0, 'max': 0}

            context_score, context_details = self._score_crypto_context(symbol, info)
            details['context'] = context_details

            # Sentiment (Phase 2) - doubled weight for crypto
            sentiment_score = 0
            if self.use_grok and self.grok_api_key:
                base_sentiment, sentiment_details = self._score_sentiment_grok(symbol)
                sentiment_score = base_sentiment * (40 / 30)  # Scale from 30 to 40 max
                details['sentiment'] = sentiment_details
            else:
                details['sentiment'] = {'status': 'Phase 2 - Grok API not configured', 'max': 40}

        elif asset_type == 'CRYPTO_PROXY':
            # CRYPTO_PROXY MODE: Health=20 (with hyper-growth), Context=10 (BTC instead of sector), Sentiment=30
            health_score, health_details = self._score_company_health(info, is_crypto_proxy=True)
            details['health'] = health_details

            context_score, context_details = self._score_crypto_proxy_context(info)
            details['context'] = context_details

            sentiment_score = 0
            if self.use_grok and self.grok_api_key:
                sentiment_score, sentiment_details = self._score_sentiment_grok(symbol)
                details['sentiment'] = sentiment_details
            else:
                details['sentiment'] = {'status': 'Phase 2 - Grok API not configured'}

        else:
            # EQUITY MODE: Standard scoring
            health_score, health_details = self._score_company_health(info, is_crypto_proxy=False)
            details['health'] = health_details

            sector = info.get('sector', 'Unknown')
            context_score, context_details = self._score_market_context(sector)
            details['context'] = context_details

            sentiment_score = 0
            if self.use_grok and self.grok_api_key:
                sentiment_score, sentiment_details = self._score_sentiment_grok(symbol)
                details['sentiment'] = sentiment_details
            else:
                details['sentiment'] = {'status': 'Phase 2 - Grok API not configured'}

        # Total score
        total_score = health_score + context_score + sentiment_score

        # Determine ELITE status (threshold varies by asset type and phase)
        # Phase 1 (no Grok): ~73% of max score
        # Phase 2 (with Grok): ~75% of max score (45/60)
        if self.use_grok:
            # Use EU thresholds if European symbol (50% of US)
            if self.is_eu_symbol(symbol):
                threshold = self.EU_ELITE_THRESHOLD  # 23/60 for EU
            else:
                threshold = self.ELITE_THRESHOLD  # 45/60 for US
        else:
            # Phase 1 thresholds proportional to max possible score
            if asset_type == 'CRYPTO':
                # CRYPTO without Grok: max 20 pts, threshold ~15 (75%)
                threshold = 15
            else:
                # EQUITY/CRYPTO_PROXY without Grok: max 30 pts, threshold 22 (73%)
                threshold = self.EU_ELITE_THRESHOLD_PHASE1 if self.is_eu_symbol(symbol) else self.ELITE_THRESHOLD_PHASE1

        is_elite = total_score > threshold

        return FundamentalScore(
            symbol=symbol,
            total_score=total_score,
            health_score=health_score,
            context_score=context_score,
            sentiment_score=sentiment_score,
            is_elite=is_elite,
            excluded=False,
            exclusion_reason=None,
            asset_type=asset_type,
            details=details
        )

    def _check_earnings_proximity(self, ticker) -> Tuple[bool, Dict]:
        """
        Check if earnings are within safety window

        Returns:
            (is_near_earnings, info_dict)
        """
        try:
            calendar = ticker.calendar

            if calendar is None or calendar.empty:
                return False, {'status': 'No earnings date found', 'days_to_earnings': None}

            # calendar can be DataFrame or dict
            if isinstance(calendar, pd.DataFrame):
                if 'Earnings Date' in calendar.index:
                    earnings_dates = calendar.loc['Earnings Date']
                    if isinstance(earnings_dates, pd.Series) and len(earnings_dates) > 0:
                        next_earnings = pd.to_datetime(earnings_dates.iloc[0])
                    else:
                        next_earnings = pd.to_datetime(earnings_dates)
                else:
                    return False, {'status': 'No earnings date in calendar'}
            elif isinstance(calendar, dict):
                earnings_date = calendar.get('Earnings Date')
                if earnings_date:
                    if isinstance(earnings_date, list) and len(earnings_date) > 0:
                        next_earnings = pd.to_datetime(earnings_date[0])
                    else:
                        next_earnings = pd.to_datetime(earnings_date)
                else:
                    return False, {'status': 'No earnings date found'}
            else:
                return False, {'status': 'Unknown calendar format'}

            # Calculate days to earnings
            today = datetime.now()
            if pd.isna(next_earnings):
                return False, {'status': 'Invalid earnings date'}

            days_to_earnings = (next_earnings - today).days

            info = {
                'next_earnings': next_earnings.strftime('%Y-%m-%d'),
                'days_to_earnings': days_to_earnings
            }

            # Check if within safety window
            if 0 <= days_to_earnings <= self.EARNINGS_SAFETY_DAYS:
                return True, info

            return False, info

        except Exception as e:
            return False, {'status': f'Error checking earnings: {str(e)[:30]}'}

    def _score_company_health(self, info: Dict, is_crypto_proxy: bool = False) -> Tuple[float, Dict]:
        """
        Score company health (0-20 points)

        Components:
        - Revenue Growth > 15%: +10 pts
        - Profit Margins > 0: +5 pts (or Revenue > 40% for hyper-growth)
        - Earnings Surprise > 0 (earningsQuarterlyGrowth): +5 pts

        Args:
            info: yfinance ticker info dict
            is_crypto_proxy: If True, apply hyper-growth clause (40% revenue pardons negative margins)
        """
        score = 0
        details = {}

        # Revenue Growth (0-10 pts)
        revenue_growth = info.get('revenueGrowth')
        is_hyper_growth = False

        if revenue_growth is not None:
            details['revenue_growth'] = round(revenue_growth * 100, 1)
            is_hyper_growth = revenue_growth > self.HYPER_GROWTH_THRESHOLD  # > 40%

            if revenue_growth > 0.15:  # > 15%
                score += 10
                details['revenue_growth_score'] = 10
            elif revenue_growth > 0.05:  # > 5%
                score += 5
                details['revenue_growth_score'] = 5
            else:
                details['revenue_growth_score'] = 0

            if is_hyper_growth:
                details['hyper_growth'] = True
        else:
            details['revenue_growth'] = None
            details['revenue_growth_score'] = 0

        # Profit Margins (0-5 pts)
        # Hyper-Growth clause: Revenue > 40% "pardons" negative margins
        profit_margins = info.get('profitMargins')
        if profit_margins is not None:
            details['profit_margins'] = round(profit_margins * 100, 1)

            if profit_margins > 0:
                score += 5
                details['profit_margins_score'] = 5
            elif is_crypto_proxy and is_hyper_growth:
                # Hyper-growth clause: 40%+ revenue growth pardons negative margins
                score += 5
                details['profit_margins_score'] = 5
                details['profit_margins_pardoned'] = True
            else:
                details['profit_margins_score'] = 0
        else:
            details['profit_margins'] = None
            # If no margin data but hyper-growth, give benefit of doubt
            if is_crypto_proxy and is_hyper_growth:
                score += 5
                details['profit_margins_score'] = 5
                details['profit_margins_pardoned'] = True
            else:
                details['profit_margins_score'] = 0

        # Earnings Growth / Surprise (0-5 pts)
        earnings_growth = info.get('earningsQuarterlyGrowth')
        if earnings_growth is not None:
            details['earnings_growth'] = round(earnings_growth * 100, 1)
            if earnings_growth > 0:
                score += 5
                details['earnings_growth_score'] = 5
            else:
                details['earnings_growth_score'] = 0
        else:
            details['earnings_growth'] = None
            details['earnings_growth_score'] = 0

        details['total'] = score
        details['max'] = 20
        details['is_crypto_proxy'] = is_crypto_proxy

        return score, details

    def _score_market_context(self, sector: str) -> Tuple[float, Dict]:
        """
        Score market context (0-10 points)

        Components:
        - SPY > EMA 20 Weekly: +5 pts
        - Sector ETF > SMA 50: +5 pts
        """
        score = 0
        details = {'sector': sector}

        # SPY trend (0-5 pts)
        if self._spy_trend is None:
            self._spy_trend = self._get_spy_trend()

        spy_bullish = self._spy_trend.get('bullish', False)
        details['spy_above_ema20w'] = spy_bullish

        if spy_bullish:
            score += 5
            details['spy_score'] = 5
        else:
            details['spy_score'] = 0

        # Sector ETF trend (0-5 pts)
        etf_symbol = self.SECTOR_ETF_MAP.get(sector)
        if etf_symbol:
            details['sector_etf'] = etf_symbol

            if etf_symbol not in self._etf_cache:
                self._etf_cache[etf_symbol] = self._get_etf_trend(etf_symbol)

            etf_trend = self._etf_cache[etf_symbol]
            etf_bullish = etf_trend.get('bullish', False)
            details['etf_above_sma50'] = etf_bullish

            if etf_bullish:
                score += 5
                details['etf_score'] = 5
            else:
                details['etf_score'] = 0
        else:
            details['sector_etf'] = None
            details['etf_above_sma50'] = None
            details['etf_score'] = 0

        details['total'] = score
        details['max'] = 10

        return score, details

    def _get_spy_trend(self) -> Dict:
        """Get SPY trend: above EMA 20 weekly"""
        try:
            spy = yf.Ticker('SPY')
            hist = spy.history(period='6mo', interval='1wk')

            if hist.empty or len(hist) < 20:
                return {'bullish': False, 'error': 'Insufficient data'}

            hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()
            latest = hist.iloc[-1]

            bullish = latest['Close'] > latest['EMA_20']

            return {
                'bullish': bullish,
                'close': round(latest['Close'], 2),
                'ema20': round(latest['EMA_20'], 2)
            }
        except Exception as e:
            logger.error(f"Error getting SPY trend: {e}")
            return {'bullish': False, 'error': str(e)}

    def _get_etf_trend(self, etf_symbol: str) -> Dict:
        """Get sector ETF trend: above SMA 50"""
        try:
            etf = yf.Ticker(etf_symbol)
            hist = etf.history(period='6mo', interval='1d')

            if hist.empty or len(hist) < 50:
                return {'bullish': False, 'error': 'Insufficient data'}

            hist['SMA_50'] = hist['Close'].rolling(50).mean()
            latest = hist.iloc[-1]

            bullish = latest['Close'] > latest['SMA_50']

            return {
                'bullish': bullish,
                'close': round(latest['Close'], 2),
                'sma50': round(latest['SMA_50'], 2)
            }
        except Exception as e:
            logger.error(f"Error getting {etf_symbol} trend: {e}")
            return {'bullish': False, 'error': str(e)}

    def _get_btc_trend(self) -> Dict:
        """Get BTC trend: above EMA 20 daily"""
        try:
            btc = yf.Ticker('BTC-USD')
            hist = btc.history(period='3mo', interval='1d')

            if hist.empty or len(hist) < 20:
                return {'bullish': False, 'error': 'Insufficient data'}

            hist['EMA_20'] = hist['Close'].ewm(span=20, adjust=False).mean()
            latest = hist.iloc[-1]

            bullish = latest['Close'] > latest['EMA_20']

            # Calculate ATH distance
            all_time_high = hist['High'].max()
            ath_distance_pct = ((all_time_high - latest['Close']) / all_time_high) * 100

            return {
                'bullish': bullish,
                'close': round(latest['Close'], 2),
                'ema20': round(latest['EMA_20'], 2),
                'ath': round(all_time_high, 2),
                'ath_distance_pct': round(ath_distance_pct, 1)
            }
        except Exception as e:
            logger.error(f"Error getting BTC trend: {e}")
            return {'bullish': False, 'error': str(e)}

    def _score_crypto_context(self, symbol: str, info: Dict) -> Tuple[float, Dict]:
        """
        Score crypto context (0-20 points)

        Components:
        - BTC > EMA20 Daily: +10 pts (auto +10 if symbol IS BTC)
        - Distance ATH < 50%: +10 pts
        """
        score = 0
        details = {'mode': 'CRYPTO'}

        # Fetch BTC trend if not cached
        if self._btc_trend is None:
            self._btc_trend = self._get_btc_trend()

        # Check if this IS BTC
        is_btc = symbol.upper().startswith('BTC')

        # BTC trend score (0-10 pts)
        if is_btc:
            # BTC itself gets auto +10 (no redundant check)
            score += 10
            details['btc_trend_score'] = 10
            details['btc_is_self'] = True
        else:
            btc_bullish = self._btc_trend.get('bullish', False)
            details['btc_above_ema20'] = btc_bullish
            details['btc_close'] = self._btc_trend.get('close')
            details['btc_ema20'] = self._btc_trend.get('ema20')

            if btc_bullish:
                score += 10
                details['btc_trend_score'] = 10
            else:
                details['btc_trend_score'] = 0

        # ATH distance score (0-10 pts)
        # For the crypto itself, calculate its ATH distance
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1y', interval='1d')

            if not hist.empty:
                current_price = info.get('regularMarketPrice') or hist['Close'].iloc[-1]
                ath = hist['High'].max()
                ath_distance_pct = ((ath - current_price) / ath) * 100

                details['ath'] = round(ath, 2)
                details['ath_distance_pct'] = round(ath_distance_pct, 1)

                if ath_distance_pct < 50:  # Within 50% of ATH
                    score += 10
                    details['ath_score'] = 10
                elif ath_distance_pct < 70:  # Within 70% of ATH
                    score += 5
                    details['ath_score'] = 5
                else:
                    details['ath_score'] = 0
            else:
                details['ath_score'] = 0
                details['ath_error'] = 'No historical data'

        except Exception as e:
            details['ath_score'] = 0
            details['ath_error'] = str(e)[:30]

        details['total'] = score
        details['max'] = 20

        return score, details

    def _score_crypto_proxy_context(self, info: Dict) -> Tuple[float, Dict]:
        """
        Score crypto-proxy context (0-10 points)

        Components:
        - SPY > EMA 20 Weekly: +5 pts
        - BTC > EMA 20 Daily: +5 pts (instead of sector ETF)
        """
        score = 0
        details = {'mode': 'CRYPTO_PROXY'}

        # SPY trend (0-5 pts)
        if self._spy_trend is None:
            self._spy_trend = self._get_spy_trend()

        spy_bullish = self._spy_trend.get('bullish', False)
        details['spy_above_ema20w'] = spy_bullish

        if spy_bullish:
            score += 5
            details['spy_score'] = 5
        else:
            details['spy_score'] = 0

        # BTC trend instead of sector ETF (0-5 pts)
        if self._btc_trend is None:
            self._btc_trend = self._get_btc_trend()

        btc_bullish = self._btc_trend.get('bullish', False)
        details['btc_above_ema20'] = btc_bullish
        details['btc_close'] = self._btc_trend.get('close')
        details['btc_ema20'] = self._btc_trend.get('ema20')

        if btc_bullish:
            score += 5
            details['btc_score'] = 5
        else:
            details['btc_score'] = 0

        details['total'] = score
        details['max'] = 10

        return score, details

    def _score_sentiment_grok(self, symbol: str) -> Tuple[float, Dict]:
        """
        Score sentiment via Grok API (0-30 points)

        Components:
        - Sentiment bullish: +10 pts
        - Mentions rising: +10 pts
        - Catalyst identified: +10 pts
        """
        details = {'symbol': symbol}

        try:
            # Build prompt for Grok
            prompt = f"""Analyze the current market sentiment for ${symbol} based on recent social media discussions, news, and market trends from the last 48 hours.

Respond ONLY with valid JSON in this exact format:
{{
  "sentiment": "bullish" or "bearish" or "neutral",
  "sentiment_confidence": 0-100,
  "mention_trend": "rising" or "stable" or "falling",
  "catalysts": ["catalyst1", "catalyst2"],
  "summary": "Brief 1-sentence summary"
}}

Focus on Twitter/X discussions, Reddit (wallstreetbets, stocks), and recent news. Be objective and data-driven."""

            # Call Grok API
            headers = {
                'Authorization': f'Bearer {self.grok_api_key}',
                'Content-Type': 'application/json'
            }

            payload = {
                'model': getattr(settings, 'GROK_MODEL', 'grok-beta'),
                'messages': [
                    {'role': 'system', 'content': 'You are a financial sentiment analyst. Respond only with valid JSON.'},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.3,
                'max_tokens': 500
            }

            response = requests.post(
                getattr(settings, 'GROK_API_URL', 'https://api.x.ai/v1/chat/completions'),
                headers=headers,
                json=payload,
                timeout=getattr(settings, 'GROK_TIMEOUT', 30)
            )

            if response.status_code != 200:
                logger.error(f"[Grok] API error {response.status_code}: {response.text[:100]}")
                return 0, {'error': f'API error {response.status_code}', 'status': 'failed'}

            # Parse response
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')

            # Parse JSON from response
            try:
                # Handle potential markdown code blocks
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()

                sentiment_data = json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"[Grok] Failed to parse JSON for {symbol}: {content[:100]}")
                return 0, {'error': 'Failed to parse response', 'raw': content[:100]}

            # Extract data
            sentiment = sentiment_data.get('sentiment', 'neutral').lower()
            sentiment_confidence = sentiment_data.get('sentiment_confidence', 50)
            mention_trend = sentiment_data.get('mention_trend', 'stable').lower()
            catalysts = sentiment_data.get('catalysts', [])
            summary = sentiment_data.get('summary', '')

            # Calculate score
            score = 0

            # Sentiment score (0-10 pts)
            if sentiment == 'bullish':
                if sentiment_confidence >= 70:
                    score += 10
                elif sentiment_confidence >= 50:
                    score += 7
                else:
                    score += 4
                details['sentiment_score'] = score
            elif sentiment == 'neutral':
                score += 3
                details['sentiment_score'] = 3
            else:  # bearish
                details['sentiment_score'] = 0

            # Mention trend score (0-10 pts)
            mention_score = 0
            if mention_trend == 'rising':
                mention_score = 10
            elif mention_trend == 'stable':
                mention_score = 5
            else:  # falling
                mention_score = 0
            score += mention_score
            details['mention_trend_score'] = mention_score

            # Catalyst score (0-10 pts)
            catalyst_score = 0
            if catalysts and len(catalysts) > 0:
                if len(catalysts) >= 2:
                    catalyst_score = 10
                else:
                    catalyst_score = 5
            score += catalyst_score
            details['catalyst_score'] = catalyst_score

            # Store raw data
            details['sentiment'] = sentiment
            details['sentiment_confidence'] = sentiment_confidence
            details['mention_trend'] = mention_trend
            details['catalysts'] = catalysts[:3]  # Limit to 3
            details['summary'] = summary
            details['total'] = score
            details['max'] = 30
            details['status'] = 'success'

            logger.info(f"[Grok] {symbol}: {sentiment} ({sentiment_confidence}%), mentions {mention_trend}, {len(catalysts)} catalysts -> {score}/30")

            return score, details

        except requests.exceptions.Timeout:
            logger.warning(f"[Grok] Timeout for {symbol}")
            return 0, {'error': 'Request timeout', 'status': 'timeout'}
        except requests.exceptions.RequestException as e:
            logger.error(f"[Grok] Request error for {symbol}: {e}")
            return 0, {'error': str(e)[:50], 'status': 'request_error'}
        except Exception as e:
            logger.error(f"[Grok] Unexpected error for {symbol}: {e}")
            return 0, {'error': str(e)[:50], 'status': 'error'}

    def get_elite_signals(self, scored_alerts: List[Dict]) -> List[Dict]:
        """
        Filter scored alerts to return only ELITE signals

        Args:
            scored_alerts: Alerts with L2 scores

        Returns:
            List of ELITE alerts (not excluded, score > threshold)
        """
        return [
            alert for alert in scored_alerts
            if not alert.get('l2_excluded', True) and alert.get('l2_is_elite', False)
        ]


# Singleton instance - uses settings from config
fundamental_scorer = FundamentalScorer(
    use_grok=getattr(settings, 'GROK_ENABLED', False),
    grok_api_key=getattr(settings, 'GROK_API_KEY', None)
)
