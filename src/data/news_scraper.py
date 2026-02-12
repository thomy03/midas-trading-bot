"""
News Scraper - Free RSS-based news scraping for financial headlines.

Sources (no API key required):
- Yahoo Finance RSS
- Google News RSS
- MarketWatch RSS

Uses feedparser and aiohttp.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)

# RSS Feed URLs
YAHOO_FINANCE_RSS = "https://finance.yahoo.com/rss/headline?s={symbol}"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={symbol}+stock"
MARKETWATCH_RSS = "https://www.marketwatch.com/rss/topstories"


class NewsScraperError(Exception):
    pass


class FreeNewsScraper:
    """Scrapes free RSS sources for financial news headlines."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers=self._headers
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _fetch_feed(self, url: str) -> str:
        """Fetch RSS feed content."""
        if not AIOHTTP_AVAILABLE:
            raise NewsScraperError("aiohttp not installed")
        session = await self._get_session()
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.text()
                else:
                    logger.warning(f"RSS fetch failed ({resp.status}): {url}")
                    return ""
        except Exception as e:
            logger.warning(f"RSS fetch error for {url}: {e}")
            return ""

    def _parse_feed(self, content: str, source: str) -> List[Dict[str, Any]]:
        """Parse RSS feed content into headline dicts."""
        if not FEEDPARSER_AVAILABLE or not content:
            return []
        
        feed = feedparser.parse(content)
        headlines = []
        
        for entry in feed.entries[:15]:
            published = None
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                try:
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
                except Exception:
                    pass
            
            headlines.append({
                "title": entry.get("title", ""),
                "source": source,
                "link": entry.get("link", ""),
                "published": published or datetime.now(timezone.utc).isoformat(),
                "summary": entry.get("summary", "")[:200] if entry.get("summary") else ""
            })
        
        return headlines

    async def get_yahoo_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Get news from Yahoo Finance RSS."""
        url = YAHOO_FINANCE_RSS.format(symbol=symbol)
        content = await self._fetch_feed(url)
        return self._parse_feed(content, "Yahoo Finance")

    async def get_google_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Get news from Google News RSS."""
        url = GOOGLE_NEWS_RSS.format(symbol=symbol)
        content = await self._fetch_feed(url)
        return self._parse_feed(content, "Google News")

    async def get_marketwatch_news(self) -> List[Dict[str, Any]]:
        """Get general market news from MarketWatch RSS."""
        content = await self._fetch_feed(MARKETWATCH_RSS)
        return self._parse_feed(content, "MarketWatch")

    async def get_all_news(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Fetch news from all free sources for a symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            List of headline dicts with keys: title, source, link, published, summary
        """
        if not FEEDPARSER_AVAILABLE:
            logger.error("feedparser not installed. Run: pip install feedparser")
            return []

        # Fetch all sources concurrently
        results = await asyncio.gather(
            self.get_yahoo_news(symbol),
            self.get_google_news(symbol),
            self.get_marketwatch_news(),
            return_exceptions=True
        )

        all_headlines = []
        source_names = ["Yahoo Finance", "Google News", "MarketWatch"]
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Error fetching from {source_names[i]}: {result}")
            elif isinstance(result, list):
                all_headlines.extend(result)

        # Deduplicate by title (case-insensitive)
        seen_titles = set()
        unique_headlines = []
        for h in all_headlines:
            title_lower = h["title"].lower().strip()
            if title_lower and title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_headlines.append(h)

        # Sort by published date (newest first)
        unique_headlines.sort(key=lambda x: x.get("published", ""), reverse=True)

        logger.info(f"[NEWS_SCRAPER] {symbol}: Found {len(unique_headlines)} unique headlines from {len(source_names)} sources")
        return unique_headlines


# Singleton
_scraper: Optional[FreeNewsScraper] = None


def get_news_scraper() -> FreeNewsScraper:
    """Get or create the FreeNewsScraper singleton."""
    global _scraper
    if _scraper is None:
        _scraper = FreeNewsScraper()
    return _scraper
