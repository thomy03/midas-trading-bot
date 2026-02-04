"""
Async Market Data Fetcher

Performance optimization: async wrappers for yfinance calls
Uses aiohttp for non-blocking HTTP requests and asyncio for concurrent fetching.

Usage:
    from src.data.async_market_data import AsyncMarketDataFetcher
    
    async def main():
        fetcher = AsyncMarketDataFetcher()
        
        # Fetch multiple symbols concurrently
        data = await fetcher.get_batch_historical_async(['AAPL', 'GOOGL', 'MSFT'])
        
        # Get current prices for multiple symbols
        prices = await fetcher.get_current_prices_async(['AAPL', 'GOOGL'])
"""

import asyncio
import aiohttp
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor

from src.data.market_data import MarketDataFetcher, CACHE_DIR, CACHE_TTL_HOURS
from src.utils.logger import logger


class AsyncMarketDataFetcher:
    """
    Async wrapper for market data fetching operations.
    
    Provides non-blocking alternatives to the synchronous MarketDataFetcher
    for use in async contexts (FastAPI, async schedulers, etc.)
    """
    
    def __init__(self, max_concurrent: int = 10):
        """
        Initialize async fetcher
        
        Args:
            max_concurrent: Maximum concurrent requests (default 10)
        """
        self.sync_fetcher = MarketDataFetcher()
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._executor.shutdown(wait=False)
    
    # ==================== Async Historical Data ====================
    
    async def get_historical_async(
        self,
        symbol: str,
        period: str = '1y',
        interval: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """
        Async wrapper for getting historical data for a single symbol.
        
        First checks disk cache, then fetches from API if needed.
        
        Args:
            symbol: Stock symbol
            period: Data period ('1y', '2y', etc.)
            interval: Data interval ('1d', '1wk')
            
        Returns:
            DataFrame with OHLCV data or None
        """
        async with self._semaphore:
            # Check disk cache first (sync but fast I/O)
            cached = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self.sync_fetcher._load_from_disk_cache,
                symbol, interval, period
            )
            
            if cached is not None:
                logger.debug(f"[ASYNC] Cache hit for {symbol}")
                return cached
            
            # Fetch from API (run sync yfinance in executor)
            logger.debug(f"[ASYNC] Fetching {symbol} from API")
            df = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self.sync_fetcher.get_historical_data,
                symbol, period, interval
            )
            
            return df
    
    async def get_batch_historical_async(
        self,
        symbols: List[str],
        period: str = '1y',
        interval: str = '1d',
        progress_callback: Optional[callable] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols concurrently.
        
        Much faster than sequential fetching for large symbol lists.
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            progress_callback: Optional async callback(completed, total)
            
        Returns:
            Dict mapping symbols to their DataFrames
        """
        results = {}
        total = len(symbols)
        completed = 0
        
        async def fetch_one(symbol: str) -> tuple:
            nonlocal completed
            df = await self.get_historical_async(symbol, period, interval)
            completed += 1
            if progress_callback:
                await progress_callback(completed, total)
            return symbol, df
        
        # Create tasks for all symbols
        tasks = [fetch_one(s) for s in symbols]
        
        # Gather results (respects semaphore limit)
        fetched = await asyncio.gather(*tasks, return_exceptions=True)
        
        for item in fetched:
            if isinstance(item, Exception):
                logger.warning(f"[ASYNC] Fetch error: {item}")
                continue
            symbol, df = item
            if df is not None:
                results[symbol] = df
        
        logger.info(f"[ASYNC] Fetched {len(results)}/{total} symbols")
        return results
    
    # ==================== Async Price Fetching ====================
    
    async def get_current_price_async(self, symbol: str) -> Optional[float]:
        """
        Async wrapper for getting current price.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price or None
        """
        async with self._semaphore:
            price = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self.sync_fetcher.get_current_price,
                symbol
            )
            return price
    
    async def get_current_prices_async(
        self,
        symbols: List[str]
    ) -> Dict[str, float]:
        """
        Get current prices for multiple symbols concurrently.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dict mapping symbols to their current prices
        """
        async def fetch_price(symbol: str) -> tuple:
            price = await self.get_current_price_async(symbol)
            return symbol, price
        
        tasks = [fetch_price(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        prices = {}
        for item in results:
            if isinstance(item, Exception):
                continue
            symbol, price = item
            if price is not None:
                prices[symbol] = price
        
        return prices
    
    # ==================== Async Stock Info ====================
    
    async def get_stock_info_async(self, symbol: str) -> Optional[Dict]:
        """
        Async wrapper for getting stock info.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dict with stock info or None
        """
        async with self._semaphore:
            info = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self.sync_fetcher.get_stock_info,
                symbol
            )
            return info
    
    async def get_batch_stock_info_async(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict]:
        """
        Get stock info for multiple symbols concurrently.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dict mapping symbols to their info
        """
        async def fetch_info(symbol: str) -> tuple:
            info = await self.get_stock_info_async(symbol)
            return symbol, info
        
        tasks = [fetch_info(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        infos = {}
        for item in results:
            if isinstance(item, Exception):
                continue
            symbol, info = item
            if info is not None:
                infos[symbol] = info
        
        return infos
    
    # ==================== Async Ticker Fetching ====================
    
    async def get_all_tickers_async(self, full_exchange: bool = True) -> Dict[str, str]:
        """
        Async wrapper for getting all tickers.
        
        Args:
            full_exchange: If True, fetch all stocks from exchanges
            
        Returns:
            Dict mapping symbols to their market source
        """
        # This involves HTTP requests to NASDAQ API, run in executor
        tickers = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self.sync_fetcher.get_all_tickers,
            full_exchange
        )
        return tickers
    
    # ==================== Async Prefetch ====================
    
    async def prefetch_async(
        self,
        symbols: List[str],
        period: str = '2y',
        interval: str = '1wk',
        progress_callback: Optional[callable] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Prefetch data for multiple symbols with progress tracking.
        
        Optimized for warming up the cache before analysis runs.
        
        Args:
            symbols: List of stock symbols
            period: Data period
            interval: Data interval
            progress_callback: Optional async callback(completed, total, status)
            
        Returns:
            Dict mapping symbols to their DataFrames
        """
        start_time = datetime.now()
        total = len(symbols)
        
        logger.info(f"[ASYNC] Starting prefetch for {total} symbols...")
        
        # Check what's already cached
        cached_symbols = []
        to_fetch = []
        
        for symbol in symbols:
            cache_path = self.sync_fetcher._get_cache_path(symbol, interval, period)
            if self.sync_fetcher._is_cache_valid(cache_path):
                cached_symbols.append(symbol)
            else:
                to_fetch.append(symbol)
        
        logger.info(f"[ASYNC] {len(cached_symbols)} cached, {len(to_fetch)} to fetch")
        
        # Load cached data
        results = {}
        for symbol in cached_symbols:
            df = self.sync_fetcher._load_from_disk_cache(symbol, interval, period)
            if df is not None:
                results[symbol] = df
        
        if progress_callback:
            await progress_callback(len(results), total, "Loading cached...")
        
        # Fetch remaining
        if to_fetch:
            fetched = await self.get_batch_historical_async(
                to_fetch, period, interval,
                progress_callback=lambda c, t: progress_callback(
                    len(results) + c, total, f"Fetching {c}/{t}..."
                ) if progress_callback else None
            )
            results.update(fetched)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"[ASYNC] Prefetch complete: {len(results)}/{total} in {elapsed:.1f}s")
        
        return results


# Convenience functions for one-off async operations

async def fetch_prices(symbols: List[str]) -> Dict[str, float]:
    """
    Quick async function to fetch current prices.
    
    Usage:
        prices = await fetch_prices(['AAPL', 'GOOGL', 'MSFT'])
    """
    async with AsyncMarketDataFetcher() as fetcher:
        return await fetcher.get_current_prices_async(symbols)


async def fetch_historical(
    symbols: List[str],
    period: str = '1y',
    interval: str = '1d'
) -> Dict[str, pd.DataFrame]:
    """
    Quick async function to fetch historical data.
    
    Usage:
        data = await fetch_historical(['AAPL', 'GOOGL'], period='2y')
    """
    async with AsyncMarketDataFetcher() as fetcher:
        return await fetcher.get_batch_historical_async(symbols, period, interval)


# Example usage and testing
if __name__ == '__main__':
    async def test_async_fetcher():
        """Test the async market data fetcher"""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
        
        print("Testing AsyncMarketDataFetcher...")
        
        async with AsyncMarketDataFetcher(max_concurrent=5) as fetcher:
            # Test batch historical
            print("\n1. Fetching historical data...")
            start = datetime.now()
            data = await fetcher.get_batch_historical_async(symbols, period='1y')
            elapsed = (datetime.now() - start).total_seconds()
            print(f"   Got {len(data)} symbols in {elapsed:.2f}s")
            for sym, df in data.items():
                print(f"   - {sym}: {len(df)} rows")
            
            # Test batch prices
            print("\n2. Fetching current prices...")
            start = datetime.now()
            prices = await fetcher.get_current_prices_async(symbols)
            elapsed = (datetime.now() - start).total_seconds()
            print(f"   Got {len(prices)} prices in {elapsed:.2f}s")
            for sym, price in prices.items():
                print(f"   - {sym}: ${price:.2f}")
            
            # Test batch info
            print("\n3. Fetching stock info...")
            start = datetime.now()
            infos = await fetcher.get_batch_stock_info_async(symbols[:3])
            elapsed = (datetime.now() - start).total_seconds()
            print(f"   Got {len(infos)} infos in {elapsed:.2f}s")
            for sym, info in infos.items():
                print(f"   - {sym}: {info.get('longName', 'N/A')}")
        
        print("\nAsync tests complete!")
    
    # Run the test
    asyncio.run(test_async_fetcher())
