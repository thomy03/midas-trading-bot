"""
Financial Modeling Prep (FMP) API Client

Async client for pre-screening stocks before IBKR validation.
This allows scanning 3000+ tickers without hitting IBKR pacing violations.

Usage:
    config = FMPConfig(api_key="your_key")
    client = FMPClient(config)
    await client.initialize()

    # Scan stocks
    candidates = await client.get_stock_screener(
        market_cap_min=500_000_000,
        volume_min=500_000
    )

    # Get historical data
    historical = await client.get_bulk_historical(symbols, days=200)

    await client.close()
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

try:
    import httpx
except ImportError:
    httpx = None  # Will be checked at runtime

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FMPConfig:
    """Configuration for FMP API client"""
    api_key: str
    base_url: str = "https://financialmodelingprep.com/stable"
    rate_limit: int = 300  # requests per minute
    timeout: int = 30  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0


class FMPClient:
    """
    Async client for Financial Modeling Prep API.

    Used for pre-screening stocks before IBKR validation to avoid
    pacing violations (Error 162).
    """

    def __init__(self, config: Optional[FMPConfig] = None):
        """
        Initialize FMP client.

        Args:
            config: FMP configuration. If None, loads from environment.
        """
        if config is None:
            api_key = os.getenv('FMP_API_KEY', '')
            if not api_key:
                logger.warning("FMP_API_KEY not set - FMP client will not work")
            config = FMPConfig(api_key=api_key)

        self.config = config
        self.client: Optional[httpx.AsyncClient] = None
        self._request_times: List[float] = []
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Initialize the async HTTP client."""
        if httpx is None:
            raise ImportError("httpx is required for FMPClient. Install with: pip install httpx")

        if self._initialized:
            return

        self.client = httpx.AsyncClient(
            timeout=self.config.timeout,
            headers={
                "Accept": "application/json",
                "User-Agent": "TradingBot/1.0"
            }
        )
        self._initialized = True
        logger.info("FMP client initialized")

    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
            self._initialized = False
            logger.info("FMP client closed")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _rate_limit(self):
        """Respect the rate limit of 300 requests/minute."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            # Clean up old request times (> 60s)
            self._request_times = [t for t in self._request_times if now - t < 60]

            if len(self._request_times) >= self.config.rate_limit:
                # Wait until oldest request is > 60s old
                wait = 60 - (now - self._request_times[0]) + 0.1
                logger.debug(f"Rate limit reached, waiting {wait:.1f}s")
                await asyncio.sleep(wait)

            self._request_times.append(now)

    async def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Make a GET request to FMP API.

        Args:
            endpoint: API endpoint (e.g., "stock-screener")
            params: Query parameters

        Returns:
            JSON response data
        """
        if not self._initialized:
            await self.initialize()

        await self._rate_limit()

        params = params or {}
        params["apikey"] = self.config.api_key

        url = f"{self.config.base_url}/{endpoint}"

        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.get(url, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limited - wait and retry
                    wait = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"FMP rate limited, waiting {wait}s (attempt {attempt + 1})")
                    await asyncio.sleep(wait)
                elif e.response.status_code == 401:
                    logger.error("FMP API key invalid or expired")
                    raise
                else:
                    logger.error(f"FMP HTTP error {e.response.status_code}: {e}")
                    raise
            except httpx.RequestError as e:
                wait = self.config.retry_delay * (2 ** attempt)
                logger.warning(f"FMP request error: {e}, retrying in {wait}s")
                await asyncio.sleep(wait)

        raise RuntimeError(f"FMP request failed after {self.config.max_retries} attempts")

    # === SCREENING ENDPOINTS ===

    async def get_stock_screener(
        self,
        market_cap_min: int = 300_000_000,
        market_cap_max: Optional[int] = None,
        volume_min: int = 500_000,
        price_min: float = 5.0,
        price_max: float = 500.0,
        exchange: str = "NASDAQ,NYSE",
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        is_etf: bool = False,
        is_actively_trading: bool = True,
        limit: int = 1000
    ) -> List[Dict]:
        """
        Stock screener - filter stocks by criteria.

        https://financialmodelingprep.com/stable/stock-screener

        Args:
            market_cap_min: Minimum market cap (default $300M)
            market_cap_max: Maximum market cap (optional)
            volume_min: Minimum daily volume (default 500K)
            price_min: Minimum price (default $5)
            price_max: Maximum price (default $500)
            exchange: Exchange filter (default "NASDAQ,NYSE")
            sector: Sector filter (optional)
            industry: Industry filter (optional)
            is_etf: Include ETFs (default False)
            is_actively_trading: Only actively trading (default True)
            limit: Maximum results (default 1000)

        Returns:
            List of stocks matching criteria
        """
        params = {
            "marketCapMoreThan": market_cap_min,
            "volumeMoreThan": volume_min,
            "priceMoreThan": price_min,
            "priceLowerThan": price_max,
            "exchange": exchange,
            "isEtf": str(is_etf).lower(),
            "isActivelyTrading": str(is_actively_trading).lower(),
            "limit": limit
        }

        if market_cap_max:
            params["marketCapLowerThan"] = market_cap_max
        if sector:
            params["sector"] = sector
        if industry:
            params["industry"] = industry

        result = await self._get("stock-screener", params)
        logger.info(f"FMP screener returned {len(result)} stocks")
        return result

    async def get_gainers_losers(self) -> Dict[str, List[Dict]]:
        """
        Get top gainers and losers of the day.

        Returns:
            Dict with 'gainers' and 'losers' lists
        """
        gainers = await self._get("stock_market/gainers")
        losers = await self._get("stock_market/losers")

        logger.info(f"FMP: {len(gainers)} gainers, {len(losers)} losers")
        return {"gainers": gainers, "losers": losers}

    async def get_most_active(self) -> List[Dict]:
        """Get most actively traded stocks."""
        return await self._get("stock_market/actives")

    # === HISTORICAL DATA ENDPOINTS ===

    async def get_historical_prices(
        self,
        symbol: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Get historical OHLCV data for a symbol.

        https://financialmodelingprep.com/stable/historical-price-full/AAPL

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)

        Returns:
            List of daily OHLCV bars (newest first)
        """
        params = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        data = await self._get(f"historical-price-full/{symbol}", params)

        # FMP returns {"symbol": "AAPL", "historical": [...]}
        if isinstance(data, dict):
            return data.get("historical", [])
        return []

    async def get_bulk_historical(
        self,
        symbols: List[str],
        days: int = 200,
        max_concurrent: int = 50
    ) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for multiple symbols in parallel.

        Args:
            symbols: List of stock symbols
            days: Number of days of history (default 200)
            max_concurrent: Maximum concurrent requests (default 50)

        Returns:
            Dict mapping symbol to DataFrame with OHLCV data
        """
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_one(symbol: str) -> tuple:
            async with semaphore:
                try:
                    data = await self.get_historical_prices(symbol, from_date=from_date)
                    if data:
                        df = self._to_dataframe(data)
                        return (symbol, df)
                    return (symbol, None)
                except Exception as e:
                    logger.debug(f"FMP error for {symbol}: {e}")
                    return (symbol, None)

        logger.info(f"Fetching historical data for {len(symbols)} symbols...")

        results = await asyncio.gather(*[fetch_one(s) for s in symbols])

        # Filter out None values
        result_dict = {symbol: df for symbol, df in results if df is not None}

        logger.info(f"Successfully fetched {len(result_dict)}/{len(symbols)} symbols")
        return result_dict

    def _to_dataframe(self, data: List[Dict]) -> pd.DataFrame:
        """
        Convert FMP historical data to DataFrame.

        Args:
            data: List of OHLCV dictionaries from FMP

        Returns:
            DataFrame with standard column names
        """
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Convert date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df.set_index('date', inplace=True)

        # Rename columns to standard format
        column_map = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adjClose': 'Adj Close'
        }

        df = df.rename(columns=column_map)

        # Ensure required columns exist
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                logger.warning(f"Missing column {col} in FMP data")

        return df

    # === QUOTE ENDPOINTS ===

    async def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time (or 15min delayed) quote for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Quote data or None
        """
        data = await self._get("profile", params={"symbol": symbol})
        return data[0] if data else None

    async def get_bulk_quotes(self, symbols: List[str]) -> List[Dict]:
        """
        Get quotes for multiple symbols (max 500 per request).

        Args:
            symbols: List of stock symbols (max 500)

        Returns:
            List of quote data
        """
        # FMP allows comma-separated symbols (max 500)
        symbols_str = ",".join(symbols[:500])
        return await self._get(f"quote/{symbols_str}")

    # === FUNDAMENTAL DATA ===

    async def get_profile(self, symbol: str) -> Optional[Dict]:
        """
        Get company profile (sector, industry, market cap, etc.)

        Args:
            symbol: Stock symbol

        Returns:
            Company profile or None
        """
        data = await self._get(f"profile/{symbol}")
        return data[0] if data else None

    async def get_key_metrics(self, symbol: str, limit: int = 1) -> List[Dict]:
        """
        Get key financial metrics (P/E, ROE, etc.)

        Args:
            symbol: Stock symbol
            limit: Number of periods (default 1 = latest)

        Returns:
            List of metrics by period
        """
        return await self._get(f"key-metrics/{symbol}", {"limit": limit})

    async def get_financial_ratios(self, symbol: str, limit: int = 1) -> List[Dict]:
        """
        Get financial ratios.

        Args:
            symbol: Stock symbol
            limit: Number of periods

        Returns:
            List of ratios by period
        """
        return await self._get(f"ratios/{symbol}", {"limit": limit})


# Singleton instance factory
_fmp_client: Optional[FMPClient] = None


async def get_fmp_client() -> FMPClient:
    """
    Get or create singleton FMP client.

    Returns:
        Initialized FMP client
    """
    global _fmp_client

    if _fmp_client is None:
        _fmp_client = FMPClient()
        await _fmp_client.initialize()

    return _fmp_client
