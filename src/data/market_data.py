"""
Market data fetching and stock filtering module

Performance optimizations (Phase 6):
- Disk cache for historical data (TTL 4h)
- Batch download for multiple tickers
- Market cap caching in SQLite
"""
import yfinance as yf
import pandas as pd
import json
import random
import requests
import io
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import time
from config.settings import (
    MIN_MARKET_CAP, MIN_DAILY_VOLUME, MAX_STOCKS, MARKETS,
    MIN_MARKET_CAP_NASDAQ, MIN_MARKET_CAP_SP500, MIN_MARKET_CAP_EUROPE, MIN_MARKET_CAP_ASIA_ADR,
    US_INDICES, EUROPEAN_INDICES, LOOKBACK_DAYS_WEEKLY, LOOKBACK_DAYS_DAILY,
    MAX_WORKERS, MAX_RETRIES, RETRY_DELAY, CUSTOM_SYMBOLS, TICKERS_JSON_DIR
)
from src.utils.logger import logger

# Disk cache configuration
CACHE_DIR = Path("data/cache")
CACHE_TTL_HOURS = 24  # Historical data valid for 24 hours (optimized for nightly prefetch)


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 30.0) -> float:
    """
    Calculate exponential backoff delay with jitter

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap

    Returns:
        Delay in seconds
    """
    delay = base_delay * (2 ** attempt)
    jitter = random.uniform(0, 0.1 * delay)  # ±10% jitter
    return min(delay + jitter, max_delay)


class MarketDataFetcher:
    """Fetches and filters market data"""

    def __init__(self):
        """Initialize the market data fetcher"""
        self.cache = {}
        self.cache_timestamp = {}
        self.tickers_dir = Path(TICKERS_JSON_DIR)
        # Ensure disk cache directory exists
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ==================== Disk Cache Methods ====================

    def _get_cache_path(self, symbol: str, interval: str, period: str) -> Path:
        """Get path for cached data file"""
        # Sanitize symbol for filename (replace / with _)
        safe_symbol = symbol.replace('/', '_').replace('\\', '_')
        return CACHE_DIR / f"{safe_symbol}_{period}_{interval}.parquet"

    def _is_cache_valid(self, cache_path: Path, ttl_hours: int = CACHE_TTL_HOURS) -> bool:
        """Check if disk cache is still valid"""
        if not cache_path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return age.total_seconds() < ttl_hours * 3600

    def _load_from_disk_cache(self, symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
        """Load data from disk cache if valid"""
        cache_path = self._get_cache_path(symbol, interval, period)
        if self._is_cache_valid(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                logger.debug(f"Loaded {symbol} ({interval}) from disk cache")
                return df
            except Exception as e:
                logger.warning(f"Failed to read cache for {symbol}: {e}")
        return None

    def _save_to_disk_cache(self, symbol: str, interval: str, period: str, df: pd.DataFrame):
        """Save data to disk cache"""
        try:
            cache_path = self._get_cache_path(symbol, interval, period)
            df.to_parquet(cache_path)
            logger.debug(f"Saved {symbol} ({interval}) to disk cache")
        except Exception as e:
            logger.warning(f"Failed to cache {symbol}: {e}")

    def clear_disk_cache(self, older_than_hours: int = None):
        """Clear disk cache, optionally only files older than specified hours"""
        try:
            count = 0
            for cache_file in CACHE_DIR.glob("*.parquet"):
                if older_than_hours:
                    age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if age.total_seconds() < older_than_hours * 3600:
                        continue
                cache_file.unlink()
                count += 1
            logger.info(f"Cleared {count} cache files")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    # ==================== Batch Download Methods ====================

    def get_batch_historical_data(
        self,
        symbols: List[str],
        period: str = '1y',
        interval: str = '1d',
        batch_size: int = 50
    ) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for multiple tickers in batches

        Much faster than individual downloads (1 API call per batch vs per symbol)

        Args:
            symbols: List of stock symbols
            period: Data period ('1y', '2y', etc.)
            interval: Data interval ('1d', '1wk')
            batch_size: Number of symbols per batch (default 50)

        Returns:
            Dict mapping symbols to their DataFrames
        """
        results = {}
        symbols_to_fetch = []

        # First check disk cache
        for symbol in symbols:
            cached = self._load_from_disk_cache(symbol, interval, period)
            if cached is not None:
                results[symbol] = cached
            else:
                symbols_to_fetch.append(symbol)

        if not symbols_to_fetch:
            logger.info(f"All {len(results)} symbols loaded from cache")
            return results

        logger.info(f"Batch downloading {len(symbols_to_fetch)} symbols ({len(results)} from cache)")

        # Download in batches
        for i in range(0, len(symbols_to_fetch), batch_size):
            batch = symbols_to_fetch[i:i + batch_size]
            batch_str = ' '.join(batch)

            try:
                # yfinance supports downloading multiple tickers at once
                data = yf.download(
                    batch_str,
                    period=period,
                    interval=interval,
                    group_by='ticker',
                    progress=False,
                    threads=True
                )

                if data.empty:
                    logger.warning(f"Empty response for batch {i // batch_size + 1}")
                    continue

                # Handle single vs multiple ticker response
                if len(batch) == 1:
                    # Single ticker - data is not grouped
                    symbol = batch[0]
                    if not data.empty:
                        results[symbol] = data.copy()
                        self._save_to_disk_cache(symbol, interval, period, data)
                else:
                    # Multiple tickers - data is grouped by ticker
                    for symbol in batch:
                        try:
                            if symbol in data.columns.get_level_values(0):
                                ticker_data = data[symbol].dropna(how='all')
                                if not ticker_data.empty:
                                    results[symbol] = ticker_data.copy()
                                    self._save_to_disk_cache(symbol, interval, period, ticker_data)
                        except Exception as e:
                            logger.debug(f"Could not extract data for {symbol}: {e}")

                logger.info(f"Batch {i // batch_size + 1}: downloaded {len([s for s in batch if s in results])}/{len(batch)} symbols")

            except Exception as e:
                logger.error(f"Batch download error: {e}")
                # Fallback to individual downloads for this batch
                for symbol in batch:
                    if symbol not in results:
                        df = self.get_historical_data(symbol, period, interval)
                        if df is not None:
                            results[symbol] = df

            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(symbols_to_fetch):
                time.sleep(0.5)

        logger.info(f"Batch download complete: {len(results)}/{len(symbols)} symbols")
        return results

    def get_market_cap_cached(self, symbol: str, max_age_days: int = 7) -> Optional[float]:
        """
        Get market cap from DB cache, fetch from API if not cached

        Args:
            symbol: Stock symbol
            max_age_days: Maximum age of cached data

        Returns:
            Market cap in actual value (not millions)
        """
        try:
            from src.database.db_manager import db_manager

            # Try cache first
            cached = db_manager.get_cached_market_cap(symbol, max_age_days)
            if cached:
                return cached.get('market_cap')

            # Fetch from API
            info = self.get_stock_info(symbol)
            if info:
                market_cap = info.get('marketCap', 0)
                if market_cap:
                    db_manager.save_market_cap(
                        symbol=symbol,
                        market_cap=market_cap,
                        sector=info.get('sector'),
                        industry=info.get('industry'),
                        company_name=info.get('longName')
                    )
                return market_cap
            return None
        except Exception as e:
            logger.error(f"Error getting market cap for {symbol}: {e}")
            return None

    def _load_tickers_from_json(self, market: str) -> List[str]:
        """
        Load tickers from local JSON file

        Args:
            market: Market name (nasdaq, sp500, europe, asia_adr)

        Returns:
            List of ticker symbols, empty list if file not found or error
        """
        json_path = self.tickers_dir / f"{market}.json"

        if not json_path.exists():
            logger.debug(f"JSON file not found for {market}: {json_path}")
            return []

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check freshness (warn if older than 7 days)
            updated_str = data.get('updated', '')
            if updated_str:
                try:
                    updated = datetime.fromisoformat(updated_str.replace('Z', '+00:00'))
                    age_days = (datetime.now(updated.tzinfo) - updated).days
                    if age_days > 7:
                        logger.warning(f"Tickers JSON for {market} is {age_days} days old. Consider updating.")
                except (ValueError, TypeError, AttributeError):
                    pass  # Date parsing failed, ignore age check

            tickers = [t.get('symbol', '') for t in data.get('tickers', [])]
            tickers = [t for t in tickers if t]  # Filter empty

            logger.info(f"Loaded {len(tickers)} tickers from JSON for {market}")
            return tickers

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {json_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading tickers from {json_path}: {e}")
            return []

    def _get_tickers_with_fallback(self, market: str, wikipedia_fetcher) -> List[str]:
        """
        Get tickers from JSON first, fallback to Wikipedia if not available

        Args:
            market: Market name
            wikipedia_fetcher: Callable to fetch from Wikipedia as fallback

        Returns:
            List of ticker symbols
        """
        # Try JSON first
        tickers = self._load_tickers_from_json(market)

        if tickers:
            return tickers

        # Fallback to Wikipedia
        logger.info(f"No JSON data for {market}, falling back to Wikipedia...")
        return wikipedia_fetcher()

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

    def get_nasdaq_tickers(self, full_exchange: bool = True) -> List[str]:
        """
        Get NASDAQ tickers - full exchange listing or just NASDAQ-100

        Args:
            full_exchange: If True, fetch ALL NASDAQ-listed stocks (~3000+)
                          If False, fetch only NASDAQ-100 index (~100)

        Returns:
            List of NASDAQ ticker symbols
        """
        if full_exchange:
            # Try to fetch full NASDAQ listing
            tickers = self._fetch_full_nasdaq_listing()
            if tickers:
                return tickers
            logger.warning("Full NASDAQ fetch failed, falling back to JSON/Wikipedia")

        return self._get_tickers_with_fallback('nasdaq', self._fetch_nasdaq_from_wikipedia)

    def _fetch_full_nasdaq_listing(self) -> List[str]:
        """
        Fetch complete NASDAQ exchange listing (~3000+ stocks)
        Uses NASDAQ's official traded symbols file

        Returns:
            List of ALL NASDAQ ticker symbols
        """
        try:
            # NASDAQ provides official listing files
            url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange=nasdaq"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()
            rows = data.get('data', {}).get('table', {}).get('rows', [])

            if not rows:
                logger.warning("No rows returned from NASDAQ API")
                return []

            tickers = []
            for row in rows:
                symbol = row.get('symbol', '').strip()
                if symbol and not any(c in symbol for c in ['^', '/', '$']):
                    # Filter out warrants, preferred shares, etc.
                    if not symbol.endswith('W') and not symbol.endswith('R'):
                        tickers.append(symbol)

            logger.info(f"Retrieved {len(tickers)} tickers from NASDAQ API (full exchange)")

            # Cache to JSON for future use
            self._save_full_listing_to_json('nasdaq_full', tickers)

            return tickers

        except Exception as e:
            logger.warning(f"NASDAQ API failed: {e}, trying alternative source...")
            return self._fetch_nasdaq_from_ftp()

    def _fetch_nasdaq_from_ftp(self) -> List[str]:
        """
        Alternative: Fetch from NASDAQ FTP (nasdaqtraded.txt)

        Returns:
            List of NASDAQ ticker symbols
        """
        try:
            # Alternative source: NASDAQ trader FTP
            url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            # Parse the pipe-delimited file
            lines = response.text.strip().split('\n')
            tickers = []

            for line in lines[1:]:  # Skip header
                parts = line.split('|')
                if len(parts) >= 2:
                    # Column 1 is NASDAQ Global Select (Y/N), Column 2 is Symbol
                    nasdaq_listed = parts[0].strip()
                    symbol = parts[1].strip()

                    # Only include actual NASDAQ stocks
                    if nasdaq_listed == 'Y' and symbol:
                        # Skip test symbols and special characters
                        if not any(c in symbol for c in ['^', '/', '$', ' ']):
                            if symbol != 'File Creation Time':
                                tickers.append(symbol)

            logger.info(f"Retrieved {len(tickers)} tickers from NASDAQ FTP (full exchange)")

            # Cache to JSON
            self._save_full_listing_to_json('nasdaq_full', tickers)

            return tickers

        except Exception as e:
            logger.error(f"NASDAQ FTP fetch failed: {e}")
            return []

    def _save_full_listing_to_json(self, market: str, tickers: List[str]):
        """
        Cache full exchange listing to JSON for future use

        Args:
            market: Market identifier (e.g., 'nasdaq_full')
            tickers: List of ticker symbols
        """
        try:
            json_path = self.tickers_dir / f"{market}.json"

            data = {
                'updated': datetime.now().isoformat(),
                'source': 'exchange_api',
                'market': market,
                'count': len(tickers),
                'tickers': [{'symbol': t} for t in tickers]
            }

            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Cached {len(tickers)} tickers to {json_path}")

        except Exception as e:
            logger.warning(f"Could not cache tickers: {e}")

    def _fetch_nasdaq_from_wikipedia(self) -> List[str]:
        """
        Fetch NASDAQ-100 tickers from Wikipedia (fallback method)
        NOTE: This only returns ~100 stocks (index components, not full exchange)

        Returns:
            List of NASDAQ-100 ticker symbols
        """
        try:
            url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
            tables = pd.read_html(url)
            df = tables[4]  # Usually the 5th table contains the list
            tickers = df['Ticker'].tolist()
            logger.info(f"Retrieved {len(tickers)} NASDAQ-100 tickers from Wikipedia (index only)")
            return tickers
        except Exception as e:
            logger.error(f"Error fetching NASDAQ tickers from Wikipedia: {e}")
            return []

    def get_sp500_tickers(self, include_nyse_full: bool = False) -> List[str]:
        """
        Get S&P 500 tickers, optionally including full NYSE listing

        Args:
            include_nyse_full: If True, fetch ALL NYSE stocks (~2800+)
                              If False, fetch only S&P 500 index (~500)

        Returns:
            List of ticker symbols
        """
        if include_nyse_full:
            # Fetch full NYSE listing
            tickers = self._fetch_full_nyse_listing()
            if tickers:
                return tickers
            logger.warning("Full NYSE fetch failed, falling back to S&P 500")

        # Try to fetch ~500 largest NYSE stocks via API
        tickers = self._fetch_sp500_via_api()
        if tickers and len(tickers) >= 400:
            return tickers

        # Try Wikipedia as fallback
        tickers = self._fetch_sp500_from_wikipedia()
        if tickers and len(tickers) >= 400:
            return tickers

        # Final fallback to JSON
        return self._load_tickers_from_json('sp500') or []

    def _fetch_sp500_via_api(self) -> List[str]:
        """
        Fetch ~500 largest NYSE stocks via NASDAQ API (approximation of S&P 500)

        Returns:
            List of ~500 ticker symbols
        """
        try:
            # Fetch NYSE stocks sorted by market cap, limit to 503 (S&P 500 size)
            url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=503&exchange=nyse"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()
            rows = data.get('data', {}).get('table', {}).get('rows', [])

            if not rows:
                logger.warning("No rows returned from NYSE API for S&P 500")
                return []

            tickers = []
            for row in rows:
                symbol = row.get('symbol', '').strip()
                if symbol and not any(c in symbol for c in ['^', '/', '$']):
                    if not symbol.endswith('W') and not symbol.endswith('R'):
                        tickers.append(symbol)

            logger.info(f"Retrieved {len(tickers)} S&P 500 tickers from NYSE API")
            return tickers

        except Exception as e:
            logger.warning(f"NYSE API failed for S&P 500: {e}")
            return []

    def _fetch_full_nyse_listing(self) -> List[str]:
        """
        Fetch complete NYSE exchange listing (~2800+ stocks)

        Returns:
            List of ALL NYSE ticker symbols
        """
        try:
            # Use NASDAQ's API which also covers NYSE
            url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange=nyse"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()
            rows = data.get('data', {}).get('table', {}).get('rows', [])

            if not rows:
                logger.warning("No rows returned from NYSE API")
                return []

            tickers = []
            for row in rows:
                symbol = row.get('symbol', '').strip()
                if symbol and not any(c in symbol for c in ['^', '/', '$']):
                    # Filter out warrants, preferred shares, etc.
                    if not symbol.endswith('W') and not symbol.endswith('R'):
                        tickers.append(symbol)

            logger.info(f"Retrieved {len(tickers)} tickers from NYSE API (full exchange)")

            # Cache to JSON
            self._save_full_listing_to_json('nyse_full', tickers)

            return tickers

        except Exception as e:
            logger.warning(f"NYSE API failed: {e}")
            return []

    def _fetch_sp500_from_wikipedia(self) -> List[str]:
        """
        Fetch S&P 500 tickers from Wikipedia (fallback method)

        Returns:
            List of S&P 500 ticker symbols
        """
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            df = tables[0]
            tickers = df['Symbol'].tolist()
            logger.info(f"Retrieved {len(tickers)} S&P 500 tickers from Wikipedia")
            return tickers
        except Exception as e:
            logger.error(f"Error fetching S&P 500 tickers from Wikipedia: {e}")
            return []

    def get_amex_tickers(self) -> List[str]:
        """
        Get AMEX (NYSE American) tickers

        Returns:
            List of AMEX ticker symbols
        """
        try:
            url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=5000&exchange=amex"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()
            rows = data.get('data', {}).get('table', {}).get('rows', [])

            tickers = []
            for row in rows:
                symbol = row.get('symbol', '').strip()
                if symbol and not any(c in symbol for c in ['^', '/', '$']):
                    if not symbol.endswith('W') and not symbol.endswith('R'):
                        tickers.append(symbol)

            logger.info(f"Retrieved {len(tickers)} tickers from AMEX (full exchange)")
            return tickers

        except Exception as e:
            logger.warning(f"AMEX API failed: {e}")
            return []

    def get_all_us_tickers(self) -> List[str]:
        """
        Get ALL US stocks from NASDAQ + NYSE + AMEX

        Returns:
            List of all US ticker symbols (~6000+)
        """
        all_tickers = set()

        # NASDAQ (full)
        nasdaq = self.get_nasdaq_tickers(full_exchange=True)
        all_tickers.update(nasdaq)
        logger.info(f"Added {len(nasdaq)} NASDAQ tickers")

        # NYSE (full)
        nyse = self._fetch_full_nyse_listing()
        all_tickers.update(nyse)
        logger.info(f"Added {len(nyse)} NYSE tickers")

        # AMEX
        amex = self.get_amex_tickers()
        all_tickers.update(amex)
        logger.info(f"Added {len(amex)} AMEX tickers")

        logger.info(f"Total US tickers: {len(all_tickers)}")
        return list(all_tickers)

    def get_european_tickers(self) -> List[str]:
        """
        Get European stock tickers from local JSON file or Wikipedia (fallback)

        Returns:
            List of European ticker symbols
        """
        return self._get_tickers_with_fallback('europe', self._fetch_europe_from_wikipedia)

    def _fetch_europe_from_wikipedia(self) -> List[str]:
        """
        Fetch European tickers from Wikipedia (fallback method)

        Returns:
            List of European ticker symbols
        """
        try:
            url = 'https://en.wikipedia.org/wiki/EURO_STOXX_50'
            tables = pd.read_html(url)
            for table in tables:
                if 'Ticker' in table.columns or 'Symbol' in table.columns:
                    ticker_col = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
                    tickers = table[ticker_col].tolist()
                    logger.info(f"Retrieved {len(tickers)} Euro Stoxx 50 tickers from Wikipedia")
                    return tickers

            logger.warning("Could not find ticker table in Euro Stoxx 50 page")
            return []

        except Exception as e:
            logger.error(f"Error fetching European tickers from Wikipedia: {e}")
            return []

    def get_asian_adr_tickers(self) -> List[str]:
        """
        Get Asian ADR (American Depositary Receipt) tickers from local JSON file

        Returns:
            List of Asian ADR ticker symbols
        """
        # Try JSON first
        tickers = self._load_tickers_from_json('asia_adr')

        if tickers:
            return tickers

        # Fallback to hardcoded list
        asian_adrs = [
            'BABA', 'JD', 'PDD', 'BIDU',  # Chinese
            'TSM', 'UMC',  # Taiwanese
            'SONY', 'NMR', 'SMFG', 'MFG',  # Japanese
            'KB', 'SKM',  # Korean
            'INFY', 'WIT', 'HDB', 'IBN',  # Indian
        ]
        logger.info(f"Using fallback: {len(asian_adrs)} Asian ADR tickers")
        return asian_adrs

    def get_crypto_tickers(self) -> List[str]:
        """
        Get cryptocurrency tickers from JSON file

        Returns:
            List of crypto ticker symbols (e.g., BTC-USD, ETH-USD)
        """
        tickers = self._load_tickers_from_json('crypto')
        if tickers:
            return tickers

        # Fallback to hardcoded list
        crypto_fallback = [
            'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD',
            'ADA-USD', 'AVAX-USD', 'DOGE-USD', 'DOT-USD', 'MATIC-USD'
        ]
        logger.info(f"Using fallback: {len(crypto_fallback)} crypto tickers")
        return crypto_fallback

    def get_cac40_tickers(self) -> List[str]:
        """
        Get CAC40 (French) tickers from JSON file

        Returns:
            List of CAC40 ticker symbols (e.g., MC.PA, TTE.PA)
        """
        tickers = self._load_tickers_from_json('cac40')
        if tickers:
            return tickers

        # Fallback to top CAC40 stocks
        cac40_fallback = [
            'MC.PA', 'OR.PA', 'RMS.PA', 'TTE.PA', 'SAN.PA',
            'AI.PA', 'AIR.PA', 'BNP.PA', 'SU.PA', 'CAP.PA'
        ]
        logger.info(f"Using fallback: {len(cac40_fallback)} CAC40 tickers")
        return cac40_fallback

    def get_dax_tickers(self) -> List[str]:
        """
        Get DAX (German) tickers from JSON file

        Returns:
            List of DAX ticker symbols (e.g., SAP.DE, SIE.DE)
        """
        tickers = self._load_tickers_from_json('dax')
        if tickers:
            return tickers

        # Fallback to top DAX stocks
        dax_fallback = [
            'SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'BAS.DE',
            'BAYN.DE', 'BMW.DE', 'MBG.DE', 'MUV2.DE', 'ADS.DE'
        ]
        logger.info(f"Using fallback: {len(dax_fallback)} DAX tickers")
        return dax_fallback

    def get_all_tickers(self, full_exchange: bool = True) -> Dict[str, str]:
        """
        Get all tickers based on market configuration

        Args:
            full_exchange: If True, fetch ALL stocks from each exchange (thousands)
                          If False, fetch only index components (~100-500 per index)

        Returns:
            Dictionary mapping ticker symbols to their market source
            (e.g., {'AAPL': 'NASDAQ', 'JPM': 'SP500', 'BTC-USD': 'CRYPTO', ...})
        """
        ticker_to_market = {}

        # Get NASDAQ tickers
        if MARKETS.get('NASDAQ', False):
            nasdaq_tickers = self.get_nasdaq_tickers(full_exchange=full_exchange)
            for ticker in nasdaq_tickers:
                ticker_to_market[ticker.strip().upper()] = 'NASDAQ'

        # Get S&P 500 / NYSE tickers (excluding already added NASDAQ tickers)
        if MARKETS.get('SP500', False):
            sp500_tickers = self.get_sp500_tickers(include_nyse_full=full_exchange)
            for ticker in sp500_tickers:
                clean_ticker = ticker.strip().upper()
                # Only add if not already in NASDAQ
                if clean_ticker not in ticker_to_market:
                    ticker_to_market[clean_ticker] = 'SP500'

        # Get Crypto tickers
        if MARKETS.get('CRYPTO', False):
            crypto_tickers = self.get_crypto_tickers()
            for ticker in crypto_tickers:
                ticker_to_market[ticker.strip().upper()] = 'CRYPTO'

        # Get CAC40 tickers
        if MARKETS.get('CAC40', False):
            cac40_tickers = self.get_cac40_tickers()
            for ticker in cac40_tickers:
                ticker_to_market[ticker.strip().upper()] = 'CAC40'

        # Get DAX tickers
        if MARKETS.get('DAX', False):
            dax_tickers = self.get_dax_tickers()
            for ticker in dax_tickers:
                ticker_to_market[ticker.strip().upper()] = 'DAX'

        # Get European tickers (Euro Stoxx 50)
        if MARKETS.get('EUROPE', False):
            european_tickers = self.get_european_tickers()
            for ticker in european_tickers:
                clean_ticker = ticker.strip().upper()
                if clean_ticker not in ticker_to_market:
                    ticker_to_market[clean_ticker] = 'EUROPE'

        # Get Asian ADR tickers
        if MARKETS.get('ASIA_ADR', False):
            adr_tickers = self.get_asian_adr_tickers()
            for ticker in adr_tickers:
                ticker_to_market[ticker.strip().upper()] = 'ASIA_ADR'

        # Add custom symbols (default to CUSTOM category)
        for ticker in CUSTOM_SYMBOLS:
            clean_ticker = ticker.strip().upper()
            if clean_ticker and clean_ticker not in ticker_to_market:
                ticker_to_market[clean_ticker] = 'CUSTOM'

        logger.info(f"Total tickers collected: {len(ticker_to_market)}")
        logger.info(f"  NASDAQ: {sum(1 for m in ticker_to_market.values() if m == 'NASDAQ')}")
        logger.info(f"  SP500/NYSE: {sum(1 for m in ticker_to_market.values() if m == 'SP500')}")
        logger.info(f"  CRYPTO: {sum(1 for m in ticker_to_market.values() if m == 'CRYPTO')}")
        logger.info(f"  CAC40: {sum(1 for m in ticker_to_market.values() if m == 'CAC40')}")
        logger.info(f"  DAX: {sum(1 for m in ticker_to_market.values() if m == 'DAX')}")
        logger.info(f"  EUROPE: {sum(1 for m in ticker_to_market.values() if m == 'EUROPE')}")
        logger.info(f"  ASIA_ADR: {sum(1 for m in ticker_to_market.values() if m == 'ASIA_ADR')}")
        logger.info(f"  CUSTOM: {sum(1 for m in ticker_to_market.values() if m == 'CUSTOM')}")

        return ticker_to_market

    def get_tickers_for_markets(
        self,
        markets: List[str],
        full_exchange: bool = True
    ) -> Dict[str, str]:
        """
        Get tickers for specific markets only

        Args:
            markets: List of market names to fetch (e.g., ['NASDAQ', 'SP500'])
            full_exchange: If True, fetch ALL stocks from each exchange

        Returns:
            Dictionary mapping ticker symbols to their market source
        """
        ticker_to_market = {}

        if 'NASDAQ' in markets:
            nasdaq_tickers = self.get_nasdaq_tickers(full_exchange=full_exchange)
            for ticker in nasdaq_tickers:
                ticker_to_market[ticker.strip().upper()] = 'NASDAQ'

        if 'SP500' in markets or 'NYSE' in markets:
            sp500_tickers = self.get_sp500_tickers(include_nyse_full=full_exchange)
            for ticker in sp500_tickers:
                clean_ticker = ticker.strip().upper()
                if clean_ticker not in ticker_to_market:
                    ticker_to_market[clean_ticker] = 'SP500'

        if 'CRYPTO' in markets:
            crypto_tickers = self.get_crypto_tickers()
            for ticker in crypto_tickers:
                ticker_to_market[ticker.strip().upper()] = 'CRYPTO'

        if 'EUROPE' in markets:
            european_tickers = self.get_european_tickers()
            for ticker in european_tickers:
                ticker_to_market[ticker.strip().upper()] = 'EUROPE'

        if 'ASIA_ADR' in markets:
            adr_tickers = self.get_asian_adr_tickers()
            for ticker in adr_tickers:
                ticker_to_market[ticker.strip().upper()] = 'ASIA_ADR'

        if 'CAC40' in markets:
            cac40_tickers = self.get_cac40_tickers()
            for ticker in cac40_tickers:
                ticker_to_market[ticker.strip().upper()] = 'CAC40'

        if 'DAX' in markets:
            dax_tickers = self.get_dax_tickers()
            for ticker in dax_tickers:
                ticker_to_market[ticker.strip().upper()] = 'DAX'

        logger.info(f"Fetched {len(ticker_to_market)} tickers for markets: {markets}")
        return ticker_to_market

    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """
        Get stock information with exponential backoff retry logic

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
                    delay = exponential_backoff(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
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
        Get historical price data for a stock with validation

        Args:
            symbol: Stock symbol
            period: Data period (e.g., '1y', '2y', '5y')
            interval: Data interval (e.g., '1d', '1wk')

        Returns:
            DataFrame with historical data or None if failed
        """
        cache_key = f"{symbol}_{period}_{interval}"

        # Check memory cache first (fastest)
        if cache_key in self.cache:
            cache_time = self.cache_timestamp.get(cache_key)
            if cache_time and (datetime.now() - cache_time).seconds < 3600:  # 1 hour cache
                logger.debug(f"Using memory cached data for {symbol}")
                return self.cache[cache_key]

        # Check disk cache (Phase 6 optimization)
        disk_cached = self._load_from_disk_cache(symbol, interval, period)
        if disk_cached is not None:
            # Also put in memory cache
            self.cache[cache_key] = disk_cached
            self.cache_timestamp[cache_key] = datetime.now()
            return disk_cached

        for attempt in range(MAX_RETRIES):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)

                if df.empty:
                    logger.warning(f"No historical data for {symbol}")
                    return None

                # Validate data quality
                is_valid, reason = self._validate_historical_data(df, symbol, period, interval)
                if not is_valid:
                    logger.warning(f"Data quality issue for {symbol}: {reason}")
                    # Still return data but log the warning
                    # Could return None for strict validation

                # Cache the data (memory)
                self.cache[cache_key] = df
                self.cache_timestamp[cache_key] = datetime.now()

                # Also cache to disk (Phase 6 optimization)
                self._save_to_disk_cache(symbol, interval, period, df)

                logger.debug(f"Retrieved {len(df)} data points for {symbol} ({interval})")
                return df

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = exponential_backoff(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to get historical data for {symbol} after {MAX_RETRIES} attempts: {e}")
                    return None

        return None

    def _validate_historical_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        period: str,
        interval: str
    ) -> Tuple[bool, str]:
        """
        Validate historical data quality

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol (for logging)
            period: Requested period
            interval: Data interval

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        # Calculate expected minimum periods based on interval
        expected_periods = {
            ('1y', '1d'): 200,   # ~252 trading days, allow 80%
            ('2y', '1d'): 400,
            ('1y', '1wk'): 40,   # ~52 weeks, allow 80%
            ('2y', '1wk'): 80,
        }

        min_expected = expected_periods.get((period, interval), 50)

        # Check minimum data points
        if len(df) < min_expected * 0.5:  # Allow 50% minimum
            return False, f"Insufficient data: {len(df)} rows, expected ~{min_expected}"

        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"

        # Check for excessive NaN values (>10% in Close)
        nan_pct = df['Close'].isna().sum() / len(df) * 100
        if nan_pct > 10:
            return False, f"Too many NaN values: {nan_pct:.1f}% in Close"

        # Check for zero/negative prices
        if (df['Close'] <= 0).any():
            return False, "Contains zero or negative prices"

        # Check for stale data (last date too old)
        if not df.index.empty:
            last_date = df.index[-1]
            if hasattr(last_date, 'date'):
                last_date = last_date.date()
            else:
                last_date = pd.Timestamp(last_date).date()

            days_old = (datetime.now().date() - last_date).days
            max_age = 7 if interval == '1d' else 14  # Allow weekends/holidays
            if days_old > max_age:
                return False, f"Stale data: last date is {days_old} days old"

        return True, "OK"

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

    def prefetch_all_markets(
        self,
        markets: List[str] = None,
        period: str = '5y',
        interval: str = '1wk',
        batch_size: int = 100,
        progress_callback=None,
        exclude_crypto: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Pre-download historical data for all configured markets

        Designed for nightly batch downloads to populate the cache
        before running simulations or backtests.

        Args:
            markets: List of markets to fetch (None = all configured)
            period: Data period ('5y' recommended for backtesting)
            interval: Data interval ('1wk' or '1d')
            batch_size: Symbols per batch (default 100)
            progress_callback: Optional callback(current, total, market, status)
            exclude_crypto: If True, skip crypto tickers

        Returns:
            Dict mapping symbols to their DataFrames

        Example:
            # Prefetch all markets
            data = fetcher.prefetch_all_markets()

            # Prefetch specific markets only
            data = fetcher.prefetch_all_markets(markets=['NASDAQ', 'SP500'])
        """
        from datetime import datetime

        start_time = datetime.now()

        # Get all tickers
        if markets:
            ticker_to_market = self.get_tickers_for_markets(markets, full_exchange=True)
        else:
            ticker_to_market = self.get_all_tickers(full_exchange=True)

        # Exclude crypto if requested
        if exclude_crypto:
            ticker_to_market = {
                k: v for k, v in ticker_to_market.items()
                if v != 'CRYPTO'
            }

        all_symbols = list(ticker_to_market.keys())
        total_symbols = len(all_symbols)

        logger.info(f"Prefetching {total_symbols} symbols for {interval} data...")
        logger.info(f"Markets: {set(ticker_to_market.values())}")

        # Track results
        all_data = {}
        cached_count = 0
        downloaded_count = 0
        failed_count = 0

        # Process in batches
        total_batches = (total_symbols + batch_size - 1) // batch_size

        for batch_idx, i in enumerate(range(0, total_symbols, batch_size)):
            batch = all_symbols[i:i + batch_size]
            batch_markets = set(ticker_to_market[s] for s in batch)

            # Progress callback
            if progress_callback:
                progress_callback(
                    batch_idx + 1,
                    total_batches,
                    ', '.join(batch_markets),
                    f"Processing batch {batch_idx + 1}/{total_batches}"
                )

            logger.info(f"Batch {batch_idx + 1}/{total_batches}: {len(batch)} symbols ({batch_markets})")

            # First check cache
            symbols_to_fetch = []
            for symbol in batch:
                cached = self._load_from_disk_cache(symbol, interval, period)
                if cached is not None:
                    all_data[symbol] = cached
                    cached_count += 1
                else:
                    symbols_to_fetch.append(symbol)

            if not symbols_to_fetch:
                logger.info(f"  All {len(batch)} symbols from cache")
                continue

            # Batch download remaining symbols
            try:
                batch_str = ' '.join(symbols_to_fetch)
                data = yf.download(
                    batch_str,
                    period=period,
                    interval=interval,
                    group_by='ticker',
                    progress=False,
                    threads=True
                )

                if data.empty:
                    logger.warning(f"  Empty response for batch {batch_idx + 1}")
                    failed_count += len(symbols_to_fetch)
                    continue

                # Handle single vs multiple ticker response
                if len(symbols_to_fetch) == 1:
                    symbol = symbols_to_fetch[0]
                    if not data.empty:
                        all_data[symbol] = data.copy()
                        self._save_to_disk_cache(symbol, interval, period, data)
                        downloaded_count += 1
                else:
                    for symbol in symbols_to_fetch:
                        try:
                            if symbol in data.columns.get_level_values(0):
                                ticker_data = data[symbol].dropna(how='all')
                                if not ticker_data.empty and len(ticker_data) > 10:
                                    all_data[symbol] = ticker_data.copy()
                                    self._save_to_disk_cache(symbol, interval, period, ticker_data)
                                    downloaded_count += 1
                                else:
                                    failed_count += 1
                            else:
                                failed_count += 1
                        except Exception as e:
                            logger.debug(f"Could not extract {symbol}: {e}")
                            failed_count += 1

                logger.info(f"  Downloaded {downloaded_count}/{len(symbols_to_fetch)} from API")

            except Exception as e:
                logger.error(f"  Batch download error: {e}")
                failed_count += len(symbols_to_fetch)

            # Rate limiting delay between batches
            if batch_idx < total_batches - 1:
                time.sleep(1.0)

        # Final summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 60)
        logger.info(f"PREFETCH COMPLETE")
        logger.info(f"  Total symbols: {total_symbols}")
        logger.info(f"  From cache: {cached_count}")
        logger.info(f"  Downloaded: {downloaded_count}")
        logger.info(f"  Failed: {failed_count}")
        logger.info(f"  Success rate: {len(all_data) / total_symbols * 100:.1f}%")
        logger.info(f"  Time elapsed: {elapsed / 60:.1f} minutes")
        logger.info("=" * 60)

        return all_data

    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the disk cache

        Returns:
            Dict with cache statistics
        """
        try:
            cache_files = list(CACHE_DIR.glob("*.parquet"))
            total_size = sum(f.stat().st_size for f in cache_files)

            # Count by interval
            weekly_files = [f for f in cache_files if '_1wk.parquet' in f.name]
            daily_files = [f for f in cache_files if '_1d.parquet' in f.name]

            # Check freshness
            now = datetime.now()
            valid_count = sum(1 for f in cache_files if self._is_cache_valid(f))

            # Get oldest and newest
            if cache_files:
                mtimes = [f.stat().st_mtime for f in cache_files]
                oldest = datetime.fromtimestamp(min(mtimes))
                newest = datetime.fromtimestamp(max(mtimes))
            else:
                oldest = newest = None

            return {
                'total_files': len(cache_files),
                'total_size_mb': total_size / (1024 * 1024),
                'weekly_files': len(weekly_files),
                'daily_files': len(daily_files),
                'valid_files': valid_count,
                'expired_files': len(cache_files) - valid_count,
                'oldest_file': oldest.isoformat() if oldest else None,
                'newest_file': newest.isoformat() if newest else None,
                'ttl_hours': CACHE_TTL_HOURS
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}


# Singleton instance
market_data_fetcher = MarketDataFetcher()
