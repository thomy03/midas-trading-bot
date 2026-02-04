"""
Polygon.io API Client
Provides fundamental data, news, and ticker details
Complements yfinance for better data quality
"""
import os
import requests
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from src.utils.logger import logger

class PolygonClient:
    """Client for Polygon.io API (free tier: 5 calls/min)"""
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY")
        self._last_call = 0
        self._min_interval = 12.5  # 5 calls/min = 1 call per 12 seconds (with margin)
        
    def _rate_limit(self):
        """Enforce rate limiting for free tier"""
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()
    
    def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with rate limiting"""
        if not self.api_key:
            logger.warning("Polygon API key not configured")
            return None
            
        self._rate_limit()
        
        params = params or {}
        params["apiKey"] = self.api_key
        
        try:
            url = f"{self.BASE_URL}{endpoint}"
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("Polygon rate limit hit, waiting...")
                time.sleep(60)
                return None
            else:
                logger.warning(f"Polygon API error {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"Polygon request failed: {e}")
            return None
    
    def get_ticker_details(self, symbol: str) -> Optional[Dict]:
        """Get company details (sector, industry, market cap, etc.)"""
        data = self._request(f"/v3/reference/tickers/{symbol}")
        if data and data.get("status") == "OK":
            return data.get("results", {})
        return None
    
    def get_financials(self, symbol: str, limit: int = 4) -> Optional[List[Dict]]:
        """Get company financials (quarterly reports)"""
        data = self._request(f"/vX/reference/financials", {
            "ticker": symbol,
            "limit": limit,
            "sort": "filing_date",
            "order": "desc"
        })
        if data and data.get("status") == "OK":
            return data.get("results", [])
        return None
    
    def get_news(self, symbol: str, limit: int = 10) -> Optional[List[Dict]]:
        """Get recent news for a ticker"""
        data = self._request(f"/v2/reference/news", {
            "ticker": symbol,
            "limit": limit,
            "order": "desc"
        })
        if data:
            return data.get("results", [])
        return None
    
    def get_market_status(self) -> Optional[Dict]:
        """Get current market status (open/closed)"""
        return self._request("/v1/marketstatus/now")
    
    def get_daily_bars(self, symbol: str, days: int = 30) -> Optional[List[Dict]]:
        """Get daily OHLCV bars (limited use - prefer yfinance for bulk)"""
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        data = self._request(f"/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}")
        if data and data.get("status") == "OK":
            return data.get("results", [])
        return None


# Singleton instance
_client = None

def get_polygon_client() -> PolygonClient:
    """Get or create Polygon client singleton"""
    global _client
    if _client is None:
        _client = PolygonClient()
    return _client
