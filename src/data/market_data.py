"""
Market data fetching and stock filtering module
"""
import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time
from config.settings import (
    MIN_MARKET_CAP, MIN_DAILY_VOLUME, MAX_STOCKS, MARKETS,
    MIN_MARKET_CAP_NASDAQ, MIN_MARKET_CAP_SP500, MIN_MARKET_CAP_EUROPE, MIN_MARKET_CAP_ASIA_ADR,
    US_INDICES, EUROPEAN_INDICES, LOOKBACK_DAYS_WEEKLY, LOOKBACK_DAYS_DAILY,
    MAX_WORKERS, MAX_RETRIES, RETRY_DELAY, CUSTOM_SYMBOLS
)
from src.utils.logger import logger


class MarketDataFetcher:
    """Fetches and filters market data"""

    def __init__(self):
        """Initialize the market data fetcher"""
        self.cache = {}
        self.cache_timestamp = {}

    def get_index_components(self, index_symbol: str) -> List[str]:
        """
        Get list of stocks in an index
        Note: yfinance doesn't provide direct access to index components,
        so we'll use a workaround or external source

        Args:
            index_symbol: Index symbol (e.g., '^GSPC')

        Returns:
            List of stock symbols
        """
        # For now, we'll return an empty list and rely on user's custom symbols
        # In production, you would integrate with an API that provides index components
        # or use a pre-built list
        logger.warning(f"Index component fetching not fully implemented for {index_symbol}")
        return []

    def get_nasdaq_tickers(self) -> List[str]:
        """
        Get NASDAQ tickers from Wikipedia

        NOTE: For production use, consider:
        1. Using a paid data provider (e.g., Alpha Vantage, Polygon.io, IEX Cloud)
        2. Storing the complete list in a local database and updating periodically
        3. Using an FTP download from NASDAQ directly

        Example database storage:
        - Create a 'stock_universe' table with columns: symbol, exchange, last_updated
        - Update weekly via scheduled task
        - Query from local DB instead of fetching each time

        Returns:
            List of NASDAQ ticker symbols
        """
        try:
            # Method 1: Try NASDAQ-100 from Wikipedia
            url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
            tables = pd.read_html(url)
            df = tables[4]  # Usually the 5th table contains the list
            tickers = df['Ticker'].tolist()
            logger.info(f"Retrieved {len(tickers)} NASDAQ-100 tickers from Wikipedia")
            return tickers
        except Exception as e:
            logger.error(f"Error fetching NASDAQ tickers from Wikipedia: {e}")
            logger.error("SOLUTION: Consider implementing a database-backed ticker list")
            logger.error("See function docstring for implementation suggestions")
            # Return empty list instead of incomplete fallback
            return []

    def get_sp500_tickers(self) -> List[str]:
        """
        Get S&P 500 tickers from Wikipedia

        Returns:
            List of S&P 500 ticker symbols
        """
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            df = tables[0]
            tickers = df['Symbol'].tolist()
            logger.info(f"Retrieved {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception as e:
            logger.error(f"Error fetching S&P 500 tickers: {e}")
            return []

    def get_european_tickers(self) -> List[str]:
        """
        Get European stock tickers

        NOTE: European ticker sources:
        1. Euro Stoxx 50: https://en.wikipedia.org/wiki/EURO_STOXX_50
        2. FTSE 100: https://en.wikipedia.org/wiki/FTSE_100_Index
        3. DAX: https://en.wikipedia.org/wiki/DAX
        4. CAC 40: https://en.wikipedia.org/wiki/CAC_40

        For production:
        - Store complete lists in database (recommended)
        - Use financial data API (Bloomberg, Refinitiv, etc.)
        - Download CSV from exchange websites

        Returns:
            List of European ticker symbols
        """
        try:
            # Try to fetch Euro Stoxx 50 from Wikipedia
            url = 'https://en.wikipedia.org/wiki/EURO_STOXX_50'
            tables = pd.read_html(url)
            # Find the table with ticker information
            for table in tables:
                if 'Ticker' in table.columns or 'Symbol' in table.columns:
                    ticker_col = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
                    tickers = table[ticker_col].tolist()
                    logger.info(f"Retrieved {len(tickers)} Euro Stoxx 50 tickers")
                    return tickers

            logger.warning("Could not find ticker table in Euro Stoxx 50 page")
            return []

        except Exception as e:
            logger.error(f"Error fetching European tickers: {e}")
            logger.error("SOLUTION: Implement database-backed ticker storage for European markets")
            return []

    def get_asian_adr_tickers(self) -> List[str]:
        """
        Get Asian ADR (American Depositary Receipt) tickers

        Returns:
            List of Asian ADR ticker symbols
        """
        # Common Asian ADRs traded in US markets
        asian_adrs = [
            'BABA', 'JD', 'PDD', 'BIDU',  # Chinese
            'TSM', 'UMC',  # Taiwanese
            'SONY', 'NMR', 'SMFG', 'MFG',  # Japanese
            'KB', 'SKM',  # Korean
            'INFY', 'WIT', 'HDB', 'IBN',  # Indian
        ]
        logger.info(f"Retrieved {len(asian_adrs)} Asian ADR tickers")
        return asian_adrs

    def get_all_tickers(self) -> Dict[str, str]:
        """
        Get all tickers based on market configuration

        Returns:
            Dictionary mapping ticker symbols to their market source
            (e.g., {'AAPL': 'NASDAQ', 'JPM': 'SP500', ...})
        """
        ticker_to_market = {}

        # Get NASDAQ tickers
        if MARKETS.get('NASDAQ', False):
            nasdaq_tickers = self.get_nasdaq_tickers()
            for ticker in nasdaq_tickers:
                ticker_to_market[ticker.strip().upper()] = 'NASDAQ'

        # Get S&P 500 tickers (excluding already added NASDAQ tickers)
        if MARKETS.get('SP500', False):
            sp500_tickers = self.get_sp500_tickers()
            for ticker in sp500_tickers:
                clean_ticker = ticker.strip().upper()
                # Only add if not already in NASDAQ
                if clean_ticker not in ticker_to_market:
                    ticker_to_market[clean_ticker] = 'SP500'

        # Get European tickers
        if MARKETS.get('EUROPE', False):
            european_tickers = self.get_european_tickers()
            for ticker in european_tickers:
                ticker_to_market[ticker.strip().upper()] = 'EUROPE'

        # Get Asian ADR tickers
        if MARKETS.get('ASIA_ADR', False):
            adr_tickers = self.get_asian_adr_tickers()
            for ticker in adr_tickers:
                ticker_to_market[ticker.strip().upper()] = 'ASIA_ADR'

        # Add custom symbols (default to SP500 category)
        for ticker in CUSTOM_SYMBOLS:
            clean_ticker = ticker.strip().upper()
            if clean_ticker and clean_ticker not in ticker_to_market:
                ticker_to_market[clean_ticker] = 'CUSTOM'

        logger.info(f"Total tickers collected: {len(ticker_to_market)}")
        logger.info(f"  NASDAQ: {sum(1 for m in ticker_to_market.values() if m == 'NASDAQ')}")
        logger.info(f"  SP500: {sum(1 for m in ticker_to_market.values() if m == 'SP500')}")
        logger.info(f"  EUROPE: {sum(1 for m in ticker_to_market.values() if m == 'EUROPE')}")
        logger.info(f"  ASIA_ADR: {sum(1 for m in ticker_to_market.values() if m == 'ASIA_ADR')}")
        logger.info(f"  CUSTOM: {sum(1 for m in ticker_to_market.values() if m == 'CUSTOM')}")

        return ticker_to_market

    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """
        Get stock information with retry logic

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with stock info or None if failed
        """
        for attempt in range(MAX_RETRIES):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                # Check if we got valid data
                if not info or 'symbol' not in info:
                    logger.warning(f"No valid info for {symbol}")
                    return None

                return info

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}. Retrying...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Failed to get info for {symbol} after {MAX_RETRIES} attempts: {e}")
                    return None

        return None

    def filter_by_criteria(self, symbol: str, market: str = 'SP500') -> Tuple[bool, Optional[Dict]]:
        """
        Check if stock meets filtering criteria based on its market

        Args:
            symbol: Stock symbol
            market: Market source ('NASDAQ', 'SP500', 'EUROPE', 'ASIA_ADR', 'CUSTOM')

        Returns:
            Tuple of (passes_filter, stock_info)
        """
        info = self.get_stock_info(symbol)

        if not info:
            return False, None

        try:
            # Determine the minimum market cap based on the market
            if market == 'NASDAQ':
                min_cap = MIN_MARKET_CAP_NASDAQ
            elif market == 'SP500' or market == 'CUSTOM':
                min_cap = MIN_MARKET_CAP_SP500
            elif market == 'EUROPE':
                min_cap = MIN_MARKET_CAP_EUROPE
            elif market == 'ASIA_ADR':
                min_cap = MIN_MARKET_CAP_ASIA_ADR
            else:
                min_cap = MIN_MARKET_CAP  # Fallback

            # Get market cap
            market_cap = info.get('marketCap', 0)
            if market_cap < min_cap * 1_000_000:  # Convert to actual value
                logger.debug(f"{symbol} ({market}) filtered out: Market cap ${market_cap/1e6:.1f}M < ${min_cap}M")
                return False, info

            # Get average volume and current price to estimate daily volume in USD
            avg_volume = info.get('averageVolume', 0)
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))

            if avg_volume == 0 or current_price == 0:
                logger.debug(f"{symbol} ({market}) filtered out: Missing volume or price data")
                return False, info

            daily_volume_usd = avg_volume * current_price

            if daily_volume_usd < MIN_DAILY_VOLUME:
                logger.debug(f"{symbol} ({market}) filtered out: Daily volume ${daily_volume_usd/1e3:.1f}k < ${MIN_DAILY_VOLUME/1e3:.1f}k")
                return False, info

            logger.debug(f"{symbol} ({market}) passes filters: Market cap ${market_cap/1e6:.1f}M, Daily volume ${daily_volume_usd/1e3:.1f}k")
            return True, info

        except Exception as e:
            logger.error(f"Error filtering {symbol}: {e}")
            return False, info

    def filter_stocks_parallel(self, ticker_to_market: Dict[str, str]) -> List[Dict]:
        """
        Filter stocks in parallel using multiple threads

        Args:
            ticker_to_market: Dictionary mapping ticker symbols to their market source

        Returns:
            List of dictionaries with filtered stock information
        """
        filtered_stocks = []

        logger.info(f"Filtering {len(ticker_to_market)} stocks with {MAX_WORKERS} workers...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_ticker = {
                executor.submit(self.filter_by_criteria, ticker, market): (ticker, market)
                for ticker, market in ticker_to_market.items()
            }

            for future in as_completed(future_to_ticker):
                ticker, market = future_to_ticker[future]
                try:
                    passes, info = future.result()
                    if passes and info:
                        filtered_stocks.append({
                            'symbol': ticker,
                            'name': info.get('longName', ticker),
                            'market': market,
                            'market_cap': info.get('marketCap', 0),
                            'sector': info.get('sector', 'Unknown'),
                            'industry': info.get('industry', 'Unknown')
                        })

                        # Stop if we reached max stocks
                        if len(filtered_stocks) >= MAX_STOCKS:
                            logger.info(f"Reached maximum of {MAX_STOCKS} stocks")
                            break

                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")

        logger.info(f"Filtered down to {len(filtered_stocks)} stocks meeting criteria")
        return filtered_stocks

    def get_historical_data(
        self,
        symbol: str,
        period: str = '1y',
        interval: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """
        Get historical price data for a stock

        Args:
            symbol: Stock symbol
            period: Data period (e.g., '1y', '2y', '5y')
            interval: Data interval (e.g., '1d', '1wk')

        Returns:
            DataFrame with historical data or None if failed
        """
        cache_key = f"{symbol}_{period}_{interval}"

        # Check cache
        if cache_key in self.cache:
            cache_time = self.cache_timestamp.get(cache_key)
            if cache_time and (datetime.now() - cache_time).seconds < 3600:  # 1 hour cache
                logger.debug(f"Using cached data for {symbol}")
                return self.cache[cache_key]

        for attempt in range(MAX_RETRIES):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)

                if df.empty:
                    logger.warning(f"No historical data for {symbol}")
                    return None

                # Cache the data
                self.cache[cache_key] = df
                self.cache_timestamp[cache_key] = datetime.now()

                logger.debug(f"Retrieved {len(df)} data points for {symbol} ({interval})")
                return df

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}. Retrying...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error(f"Failed to get historical data for {symbol} after {MAX_RETRIES} attempts: {e}")
                    return None

        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a stock

        Args:
            symbol: Stock symbol

        Returns:
            Current price or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d')
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return None


# Singleton instance
market_data_fetcher = MarketDataFetcher()
