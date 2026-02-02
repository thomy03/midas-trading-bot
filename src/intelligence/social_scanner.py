"""
Social Scanner Module
Scanner des réseaux sociaux pour détecter les tendances et le sentiment.

Sources:
- StockTwits: Sentiment et trending tickers
- Reddit: Subreddits financiers (wallstreetbets, stocks, investing, etc.)

Usage:
    from src.intelligence.social_scanner import SocialScanner

    scanner = SocialScanner()
    await scanner.initialize()

    # Scan complet
    result = await scanner.full_scan()

    # Scan StockTwits seulement
    stocktwits_data = await scanner.scan_stocktwits()

    # Scan Reddit seulement
    reddit_data = await scanner.scan_reddit()
"""

import os
import re
import json
import asyncio
import aiohttp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from collections import Counter
import logging
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SocialMention:
    """Mention d'un symbole sur les réseaux sociaux"""
    symbol: str
    source: str  # 'stocktwits', 'reddit'
    text: str
    sentiment: str  # 'bullish', 'bearish', 'neutral'
    sentiment_score: float  # -1 to +1
    author: str
    timestamp: datetime
    engagement: int  # likes, upvotes
    url: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'source': self.source,
            'text': self.text[:500],  # Truncate long texts
            'sentiment': self.sentiment,
            'sentiment_score': self.sentiment_score,
            'author': self.author,
            'timestamp': self.timestamp.isoformat(),
            'engagement': self.engagement,
            'url': self.url
        }


@dataclass
class TrendingSymbol:
    """Symbole en tendance sur les réseaux sociaux"""
    symbol: str
    mention_count: int
    avg_sentiment: float  # -1 to +1
    sentiment_label: str  # 'very_bullish', 'bullish', 'neutral', 'bearish', 'very_bearish'
    sources: List[str]
    momentum_24h: float  # Change vs last 24h
    top_mentions: List[SocialMention] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'mention_count': self.mention_count,
            'avg_sentiment': self.avg_sentiment,
            'sentiment_label': self.sentiment_label,
            'sources': self.sources,
            'momentum_24h': self.momentum_24h,
            'top_mentions': [m.to_dict() for m in self.top_mentions[:3]]
        }


@dataclass
class SocialScanResult:
    """Résultat d'un scan social complet"""
    timestamp: datetime
    trending_symbols: List[TrendingSymbol]
    total_mentions: int
    overall_sentiment: float
    hot_topics: List[str]
    sources_stats: Dict[str, int]

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'trending_symbols': [t.to_dict() for t in self.trending_symbols],
            'total_mentions': self.total_mentions,
            'overall_sentiment': self.overall_sentiment,
            'hot_topics': self.hot_topics,
            'sources_stats': self.sources_stats
        }

    def get_top_symbols(self, n: int = 10, min_mentions: int = 3) -> List[str]:
        """Retourne les N symboles les plus mentionnés"""
        return [
            t.symbol for t in self.trending_symbols[:n]
            if t.mention_count >= min_mentions
        ]

    def get_bullish_symbols(self, min_sentiment: float = 0.3) -> List[str]:
        """Retourne les symboles avec sentiment bullish"""
        return [
            t.symbol for t in self.trending_symbols
            if t.avg_sentiment >= min_sentiment
        ]


# =============================================================================
# STOCKTWITS SCANNER
# =============================================================================

class StockTwitsScanner:
    """
    Scanner StockTwits API

    StockTwits est une plateforme sociale centrée sur les marchés financiers.
    Leurs API publiques permettent de récupérer:
    - Trending symbols
    - Messages par symbole
    - Sentiment global

    Rate limits: 200 requests/hour pour les API publiques
    """

    BASE_URL = "https://api.stocktwits.com/api/2"

    # Endpoints publics (pas de token requis)
    ENDPOINTS = {
        'trending': '/trending/symbols.json',
        'symbol_stream': '/streams/symbol/{symbol}.json',
        'user_stream': '/streams/user/{user_id}.json',
        'home_stream': '/streams/home.json',
        'suggested': '/streams/suggested.json'
    }

    def __init__(self, access_token: Optional[str] = None):
        """
        Initialise le scanner StockTwits.

        Args:
            access_token: Token d'accès optionnel pour les API privées
                         (plus de requêtes, accès à plus de données)
        """
        self.access_token = access_token or os.getenv('STOCKTWITS_ACCESS_TOKEN')
        self.session: Optional[aiohttp.ClientSession] = None
        self._request_count = 0
        self._last_reset = datetime.now()

        # Cache pour éviter les requêtes répétées
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    async def initialize(self):
        """Initialise la session HTTP"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        logger.info("StockTwits scanner initialized")

    async def close(self):
        """Ferme la session HTTP"""
        if self.session:
            await self.session.close()
            self.session = None

    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Effectue une requête à l'API StockTwits avec gestion du rate limiting.
        """
        # Check rate limit (200/hour)
        now = datetime.now()
        if (now - self._last_reset).total_seconds() > 3600:
            self._request_count = 0
            self._last_reset = now

        if self._request_count >= 190:  # Marge de sécurité
            wait_time = 3600 - (now - self._last_reset).total_seconds()
            logger.warning(f"StockTwits rate limit proche, attente {wait_time:.0f}s")
            return None

        # Check cache
        cache_key = f"{endpoint}:{json.dumps(params or {})}"
        if cache_key in self._cache:
            data, cached_at = self._cache[cache_key]
            if now - cached_at < self._cache_ttl:
                return data

        # Make request
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}

        if self.access_token:
            params['access_token'] = self.access_token

        try:
            async with self.session.get(url, params=params) as response:
                self._request_count += 1

                if response.status == 200:
                    data = await response.json()
                    self._cache[cache_key] = (data, now)
                    return data
                elif response.status == 429:
                    logger.warning("StockTwits rate limit atteint")
                    return None
                else:
                    logger.error(f"StockTwits error {response.status}: {await response.text()}")
                    return None

        except Exception as e:
            logger.error(f"StockTwits request error: {e}")
            return None

    async def get_trending(self) -> List[Dict]:
        """
        Récupère les symboles en tendance sur StockTwits.

        Returns:
            Liste de symboles trending avec leurs stats
        """
        data = await self._make_request(self.ENDPOINTS['trending'])
        if not data or 'symbols' not in data:
            return []

        trending = []
        for symbol_data in data['symbols']:
            trending.append({
                'symbol': symbol_data.get('symbol', ''),
                'title': symbol_data.get('title', ''),
                'watchlist_count': symbol_data.get('watchlist_count', 0),
                'is_following': symbol_data.get('is_following', False)
            })

        return trending

    async def get_symbol_messages(
        self,
        symbol: str,
        limit: int = 30,
        filter_type: str = 'all'
    ) -> List[SocialMention]:
        """
        Récupère les messages récents pour un symbole.

        Args:
            symbol: Ticker du symbole (ex: 'AAPL')
            limit: Nombre max de messages
            filter_type: 'all', 'top', 'charts', 'videos'

        Returns:
            Liste de mentions sociales
        """
        endpoint = self.ENDPOINTS['symbol_stream'].format(symbol=symbol.upper())
        params = {'limit': min(limit, 30), 'filter': filter_type}

        data = await self._make_request(endpoint, params)
        if not data or 'messages' not in data:
            return []

        mentions = []
        for msg in data['messages']:
            # Extraire le sentiment
            entities = msg.get('entities', {})
            sentiment_data = entities.get('sentiment', {})

            if sentiment_data:
                sentiment = sentiment_data.get('basic', 'neutral').lower()
                if sentiment == 'bullish':
                    sentiment_score = 0.7
                elif sentiment == 'bearish':
                    sentiment_score = -0.7
                else:
                    sentiment_score = 0.0
            else:
                # Analyse basique du texte si pas de sentiment taggé
                sentiment, sentiment_score = self._analyze_text_sentiment(msg.get('body', ''))

            mention = SocialMention(
                symbol=symbol.upper(),
                source='stocktwits',
                text=msg.get('body', ''),
                sentiment=sentiment,
                sentiment_score=sentiment_score,
                author=msg.get('user', {}).get('username', 'unknown'),
                timestamp=self._parse_timestamp(msg.get('created_at', '')),
                engagement=msg.get('likes', {}).get('total', 0),
                url=f"https://stocktwits.com/message/{msg.get('id', '')}"
            )
            mentions.append(mention)

        return mentions

    async def get_suggested_symbols(self) -> List[str]:
        """Récupère les symboles suggérés par StockTwits"""
        data = await self._make_request(self.ENDPOINTS['suggested'])
        if not data or 'messages' not in data:
            return []

        # Extraire les symboles des messages suggérés
        symbols = set()
        for msg in data['messages']:
            for sym in msg.get('symbols', []):
                symbols.add(sym.get('symbol', ''))

        return list(symbols)

    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse un timestamp StockTwits"""
        try:
            # Format: "2024-12-27T15:30:00Z"
            return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        except (ValueError, TypeError, AttributeError):
            return datetime.now()

    def _analyze_text_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Analyse basique du sentiment d'un texte.
        Utilisé quand StockTwits n'a pas taggé le sentiment.
        """
        text_lower = text.lower()

        bullish_words = [
            'moon', 'rocket', 'buy', 'long', 'calls', 'bull', 'breakout',
            'squeeze', 'yolo', 'diamond hands', 'to the moon', 'undervalued',
            'bullish', 'pump', 'rally', 'green', 'ath', 'all time high',
            '🚀', '💎', '🌙', '📈', '💰', '🐂'
        ]

        bearish_words = [
            'sell', 'short', 'puts', 'bear', 'crash', 'dump', 'tank',
            'overvalued', 'bearish', 'red', 'falling knife', 'dead cat',
            'bag holder', 'bagholder', 'rip', 'rug pull', 'scam',
            '📉', '🐻', '💀', '⚠️'
        ]

        bullish_count = sum(1 for word in bullish_words if word in text_lower)
        bearish_count = sum(1 for word in bearish_words if word in text_lower)

        if bullish_count > bearish_count:
            score = min(0.3 + (bullish_count - bearish_count) * 0.1, 1.0)
            return 'bullish', score
        elif bearish_count > bullish_count:
            score = max(-0.3 - (bearish_count - bullish_count) * 0.1, -1.0)
            return 'bearish', score
        else:
            return 'neutral', 0.0


# =============================================================================
# REDDIT SCANNER
# =============================================================================

class RedditScanner:
    """
    Scanner Reddit pour les subreddits financiers.

    Utilise l'API Reddit publique (pas d'auth requise pour le read-only).
    Rate limit: ~60 requests/minute pour les API non-authentifiées

    Subreddits surveillés:
    - r/wallstreetbets: Options, YOLO trades, meme stocks
    - r/stocks: Discussions générales
    - r/investing: Long-term investing
    - r/options: Options trading
    - r/pennystocks: Small caps
    - r/stockmarket: General market
    - r/ValueInvesting: Value plays
    - r/Daytrading: Day trading
    """

    BASE_URL = "https://www.reddit.com"

    # Subreddits à scanner avec leur poids (importance)
    SUBREDDITS = {
        'wallstreetbets': 1.5,      # Très actif, meme stocks
        'stocks': 1.2,              # Discussions générales
        'investing': 1.0,           # Long-term
        'options': 1.0,             # Options
        'pennystocks': 0.8,         # Small caps (plus risqué)
        'stockmarket': 0.8,         # General
        'ValueInvesting': 0.7,      # Value plays
        'Daytrading': 0.7,          # Day trading
        'SecurityAnalysis': 0.5,    # Deep analysis
        'SPACs': 0.3,               # SPACs (moins populaire maintenant)
    }

    # Regex pour extraire les tickers
    TICKER_PATTERN = re.compile(r'\$([A-Z]{1,5})\b|(?<!\w)([A-Z]{2,5})(?!\w)')

    # Mots à ignorer (pas des tickers)
    # V4.8: Extended list to avoid false positives from common words & financial terms
    IGNORE_WORDS = {
        # Single letters and common short words
        'I', 'A', 'AM', 'PM', 'AN', 'AS', 'AT', 'BE', 'BY', 'DO', 'GO',
        'IF', 'IN', 'IS', 'IT', 'ME', 'MY', 'NO', 'OF', 'ON', 'OR', 'SO',
        'TO', 'UP', 'US', 'WE',

        # Common 3-letter words (often mistaken for tickers)
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL',
        'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET',
        'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'LET', 'MAY', 'NEW',
        'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID',
        'SAY', 'SHE', 'TOO', 'USE', 'ANY', 'BIG', 'END', 'FAR',
        'FEW', 'GOT', 'HAD', 'HAS', 'OWN', 'PUT', 'RUN', 'SET',
        'TOP', 'TRY', 'WHY', 'YET', 'AGO', 'BAD', 'BUY', 'LOW',

        # Common 4-5 letter words
        'THAT', 'WITH', 'HAVE', 'THIS', 'WILL', 'YOUR', 'FROM',
        'THEY', 'BEEN', 'CALL', 'COME', 'EACH', 'FIND', 'GIVE',
        'GOOD', 'JUST', 'KNOW', 'LAST', 'LONG', 'LOOK', 'MADE',
        'MAKE', 'MORE', 'MOST', 'MUCH', 'MUST', 'NEED', 'NEXT',
        'ONLY', 'OVER', 'SAME', 'TAKE', 'THAN', 'THEM', 'THEN',
        'VERY', 'WANT', 'WELL', 'WHAT', 'WHEN', 'WORK', 'YEAR',
        'ALSO', 'BACK', 'BEEN', 'BEST', 'BOTH', 'DOWN', 'EVEN',
        'EVER', 'FACT', 'FEEL', 'GOES', 'GREAT', 'HIGH', 'INTO',
        'KEEP', 'LEFT', 'LESS', 'LIFE', 'LIKE', 'LINE', 'LIVE',
        'MOVE', 'NAME', 'NEWS', 'PART', 'PLAY', 'REAL', 'SAID',
        'SELL', 'SHOW', 'SOME', 'STOP', 'SURE', 'TELL', 'TERM',
        'TIME', 'TURN', 'USED', 'WEEK', 'ZERO', 'HOLD', 'RISK',

        # Reddit/Social slang
        'DD', 'YOLO', 'FOMO', 'HODL', 'FUD', 'WSB', 'IMO', 'IMHO', 'TBH',
        'OP', 'TL', 'DR', 'TLDR', 'EDIT', 'RIP', 'LOL', 'WTF', 'BTW',
        'AFAIK', 'IIRC', 'LMAO', 'ROFL', 'SMH', 'TIL', 'YMMV', 'AMA',

        # Business/Finance titles & roles
        'CEO', 'CFO', 'COO', 'CTO', 'CIO', 'CMO', 'VP', 'SVP', 'EVP',
        'MD', 'DIR', 'MGR', 'HR', 'PR', 'IR',

        # Market events & terms
        'IPO', 'ATH', 'ATL', 'DIP', 'RUN', 'GAP', 'TOP', 'BOT',
        'SEC', 'FED', 'NYSE', 'NASDAQ', 'AMEX', 'OTC', 'PINK',

        # Economic indicators & metrics (often mistaken for tickers!)
        'GDP', 'CPI', 'PPI', 'PMI', 'NFP', 'FOMC', 'QE', 'QT',
        'ETF', 'ETN', 'REIT', 'BDC', 'MLP', 'SPAC', 'ADR',

        # Financial ratios & metrics - CRITICAL to filter!
        'EPS', 'PE', 'PB', 'PS', 'PCF', 'PEG', 'NAV', 'AUM',
        'DCF', 'NPV', 'IRR', 'ROI', 'ROE', 'ROA', 'ROIC', 'ROCE',
        'EBITDA', 'EBIT', 'GAAP', 'FCF', 'OCF', 'CAPEX', 'OPEX',
        'EV', 'TAM', 'SAM', 'SOM', 'CAGR', 'YOY', 'QOQ', 'MOM',
        'TTM', 'FY', 'FWD', 'LTM', 'NTM', 'YTD', 'MTD', 'WTD',

        # Technical analysis terms
        'RSI', 'EMA', 'SMA', 'MACD', 'VWAP', 'IV', 'HV', 'VIX',
        'ITM', 'OTM', 'ATM', 'DTE', 'OI', 'VOL', 'ADX', 'CCI',
        'BB', 'KC', 'ATR', 'OBV', 'MFI', 'CMF', 'PPO', 'ROC',

        # Currencies
        'USA', 'UK', 'EU', 'US', 'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD',
        'CHF', 'CNY', 'HKD', 'SGD', 'NZD', 'SEK', 'NOK', 'DKK', 'MXN',
        'BTC', 'ETH', 'USDT', 'USDC',

        # Options terms
        'CALL', 'PUT', 'LEG', 'LEAP', 'LONG', 'SHORT', 'STRADDLE',
        'STRANGLE', 'SPREAD', 'IRON', 'FLY',

        # Common false positive words
        'JUST', 'LIKE', 'WHAT', 'WHEN', 'WHERE', 'WHICH', 'WHILE',
        'ABOUT', 'AFTER', 'AGAIN', 'BEING', 'BELOW', 'COULD',
        'DOING', 'DURING', 'EVERY', 'FIRST', 'FOUND', 'GOING',
        'HAVING', 'THEIR', 'THERE', 'THESE', 'THING', 'THINK',
        'THOSE', 'THREE', 'TODAY', 'UNDER', 'UNTIL', 'WHERE',
        'WOULD', 'STILL', 'SINCE', 'MIGHT', 'NEVER', 'OFTEN',
        'OTHER', 'POINT', 'PRICE', 'RIGHT', 'SHARE', 'SMALL',
        'START', 'STATE', 'STOCK', 'TRADE', 'VALUE', 'WORLD',

        # Government agencies & regulations (often appear in financial news)
        'FDA', 'SEC', 'FTC', 'DOJ', 'EPA', 'FCC', 'IRS', 'DOE',
        'CDC', 'NIH', 'WHO', 'FBI', 'CIA', 'NSA', 'DOD', 'DHS',

        # Retirement & tax terms
        'IRA', 'ROTH', 'HSA', 'FSA',

        # Tech terms (AI, chips, cloud)
        'TPU', 'GPU', 'CPU', 'NPU', 'API', 'SDK', 'SSD', 'HDD',
        'RAM', 'ROM', 'LLM', 'GPT', 'AGI', 'ASI', 'NLP', 'CNN',
        'AWS', 'GCP', 'SQL', 'DNS', 'VPN', 'IOT', 'OTA', 'USB',

        # Index & market references
        'SP', 'DJI', 'NDX', 'RUT', 'SOX', 'XLF', 'XLE', 'XLK',

        # Misc commonly confused
        'MDA', 'LLAP', 'RVNC', 'CEO', 'CTO', 'CFO', 'COO',
        'USA', 'NYC', 'GDP', 'PCE', 'ISM', 'ADP',
    }

    def __init__(self):
        """Initialise le scanner Reddit"""
        self.session: Optional[aiohttp.ClientSession] = None
        self._request_count = 0
        self._last_reset = datetime.now()

        # Cache
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(minutes=10)

    async def initialize(self):
        """Initialise la session HTTP"""
        if self.session is None:
            headers = {
                'User-Agent': 'TradingBot/1.0 (by /u/tradingbot_research)'
            }
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        logger.info("Reddit scanner initialized")

    async def close(self):
        """Ferme la session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def _make_request(self, url: str) -> Optional[Dict]:
        """Effectue une requête à Reddit avec rate limiting"""
        now = datetime.now()

        # Rate limit: ~60/min
        if (now - self._last_reset).total_seconds() > 60:
            self._request_count = 0
            self._last_reset = now

        if self._request_count >= 55:  # Marge
            wait_time = 60 - (now - self._last_reset).total_seconds()
            logger.warning(f"Reddit rate limit proche, attente {wait_time:.0f}s")
            await asyncio.sleep(wait_time)
            self._request_count = 0
            self._last_reset = datetime.now()

        # Cache check
        if url in self._cache:
            data, cached_at = self._cache[url]
            if now - cached_at < self._cache_ttl:
                return data

        try:
            async with self.session.get(url) as response:
                self._request_count += 1

                if response.status == 200:
                    data = await response.json()
                    self._cache[url] = (data, now)
                    return data
                elif response.status == 429:
                    logger.warning("Reddit rate limit atteint")
                    await asyncio.sleep(60)
                    return None
                else:
                    logger.error(f"Reddit error {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Reddit request error: {e}")
            return None

    async def get_hot_posts(
        self,
        subreddit: str,
        limit: int = 25
    ) -> List[Dict]:
        """
        Récupère les posts "hot" d'un subreddit.

        Args:
            subreddit: Nom du subreddit (sans r/)
            limit: Nombre de posts (max 100)

        Returns:
            Liste de posts
        """
        url = f"{self.BASE_URL}/r/{subreddit}/hot.json?limit={min(limit, 100)}"
        data = await self._make_request(url)

        if not data or 'data' not in data:
            return []

        posts = []
        for child in data['data'].get('children', []):
            post_data = child.get('data', {})
            posts.append({
                'title': post_data.get('title', ''),
                'selftext': post_data.get('selftext', ''),
                'author': post_data.get('author', ''),
                'score': post_data.get('score', 0),
                'upvote_ratio': post_data.get('upvote_ratio', 0.5),
                'num_comments': post_data.get('num_comments', 0),
                'created_utc': post_data.get('created_utc', 0),
                'permalink': post_data.get('permalink', ''),
                'subreddit': subreddit
            })

        return posts

    async def get_new_posts(
        self,
        subreddit: str,
        limit: int = 25
    ) -> List[Dict]:
        """Récupère les nouveaux posts"""
        url = f"{self.BASE_URL}/r/{subreddit}/new.json?limit={min(limit, 100)}"
        data = await self._make_request(url)

        if not data or 'data' not in data:
            return []

        posts = []
        for child in data['data'].get('children', []):
            post_data = child.get('data', {})
            posts.append({
                'title': post_data.get('title', ''),
                'selftext': post_data.get('selftext', ''),
                'author': post_data.get('author', ''),
                'score': post_data.get('score', 0),
                'upvote_ratio': post_data.get('upvote_ratio', 0.5),
                'num_comments': post_data.get('num_comments', 0),
                'created_utc': post_data.get('created_utc', 0),
                'permalink': post_data.get('permalink', ''),
                'subreddit': subreddit
            })

        return posts

    async def scan_subreddit(self, subreddit: str) -> List[SocialMention]:
        """
        Scanne un subreddit et extrait les mentions de symboles.

        Args:
            subreddit: Nom du subreddit

        Returns:
            Liste de mentions sociales
        """
        # Récupérer hot et new posts
        hot_posts = await self.get_hot_posts(subreddit, limit=25)
        new_posts = await self.get_new_posts(subreddit, limit=15)

        # Combiner sans duplicats
        seen_titles = set()
        all_posts = []
        for post in hot_posts + new_posts:
            if post['title'] not in seen_titles:
                seen_titles.add(post['title'])
                all_posts.append(post)

        mentions = []
        weight = self.SUBREDDITS.get(subreddit, 1.0)

        for post in all_posts:
            # Extraire les tickers du titre et du texte
            full_text = f"{post['title']} {post['selftext']}"
            symbols = self._extract_tickers(full_text)

            # Analyser le sentiment
            sentiment, sentiment_score = self._analyze_post_sentiment(post)

            # Créer une mention pour chaque symbole trouvé
            for symbol in symbols:
                mention = SocialMention(
                    symbol=symbol,
                    source='reddit',
                    text=post['title'],
                    sentiment=sentiment,
                    sentiment_score=sentiment_score * weight,
                    author=post['author'],
                    timestamp=datetime.fromtimestamp(post['created_utc']),
                    engagement=post['score'],
                    url=f"https://reddit.com{post['permalink']}"
                )
                mentions.append(mention)

        return mentions

    async def scan_all_subreddits(self) -> List[SocialMention]:
        """Scanne tous les subreddits configurés"""
        all_mentions = []

        tasks = [
            self.scan_subreddit(sub)
            for sub in self.SUBREDDITS.keys()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_mentions.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Reddit scan error: {result}")

        return all_mentions

    def _extract_tickers(self, text: str) -> Set[str]:
        """
        Extrait les tickers potentiels d'un texte.

        Args:
            text: Texte à analyser

        Returns:
            Set de symboles uniques
        """
        symbols = set()

        # Pattern 1: $SYMBOL
        for match in re.finditer(r'\$([A-Z]{1,5})\b', text):
            symbol = match.group(1)
            if symbol not in self.IGNORE_WORDS and len(symbol) >= 1:
                symbols.add(symbol)

        # Pattern 2: Mots en majuscules (2-5 lettres)
        words = re.findall(r'\b[A-Z]{2,5}\b', text)
        for word in words:
            if word not in self.IGNORE_WORDS:
                # Vérifier le contexte - doit ressembler à un ticker
                symbols.add(word)

        return symbols

    def _analyze_post_sentiment(self, post: Dict) -> Tuple[str, float]:
        """
        Analyse le sentiment d'un post Reddit.

        Utilise plusieurs signaux:
        - Upvote ratio
        - Mots clés bullish/bearish
        - Score du post
        """
        text = f"{post['title']} {post['selftext']}".lower()

        # Mots clés
        bullish_words = [
            'buy', 'calls', 'long', 'moon', 'rocket', 'bullish', 'yolo',
            'undervalued', 'breakout', 'squeeze', 'diamond hands', 'going up',
            'to the moon', 'all in', 'buying more', 'green', 'rally',
            '🚀', '💎', '📈', '🐂', '💰'
        ]

        bearish_words = [
            'sell', 'puts', 'short', 'crash', 'dump', 'bearish', 'tank',
            'overvalued', 'bubble', 'rug pull', 'scam', 'falling knife',
            'bag holder', 'red', 'rip', 'dead', 'avoid',
            '📉', '🐻', '💀', '⚠️'
        ]

        bullish_count = sum(1 for w in bullish_words if w in text)
        bearish_count = sum(1 for w in bearish_words if w in text)

        # Upvote ratio influence
        upvote_bonus = (post.get('upvote_ratio', 0.5) - 0.5) * 0.5

        # Calculer le score
        if bullish_count > bearish_count:
            base_score = 0.3 + (bullish_count - bearish_count) * 0.1
            score = min(base_score + upvote_bonus, 1.0)
            return 'bullish', score
        elif bearish_count > bullish_count:
            base_score = -0.3 - (bearish_count - bullish_count) * 0.1
            score = max(base_score + upvote_bonus, -1.0)
            return 'bearish', score
        else:
            return 'neutral', upvote_bonus


# =============================================================================
# SOCIAL SCANNER (COMBINED)
# =============================================================================

class SocialScanner:
    """
    Scanner social unifié combinant StockTwits et Reddit.

    Fonctionnalités:
    - Agrégation des mentions multi-sources
    - Détection des symboles trending
    - Analyse de sentiment globale
    - Cache pour les scans fréquents

    Usage:
        scanner = SocialScanner()
        await scanner.initialize()
        result = await scanner.full_scan()
        print(result.get_top_symbols(10))
    """

    def __init__(
        self,
        stocktwits_token: Optional[str] = None,
        enable_stocktwits: bool = False,
        enable_reddit: bool = False
    ):
        """
        Initialise le scanner social.

        Args:
            stocktwits_token: Token d'accès StockTwits optionnel
            enable_stocktwits: Activer le scan StockTwits
            enable_reddit: Activer le scan Reddit
        """
        self.stocktwits = StockTwitsScanner(access_token=stocktwits_token) if enable_stocktwits else None
        self.reddit = RedditScanner() if enable_reddit else None

        self._last_scan: Optional[SocialScanResult] = None
        self._scan_cache_ttl = timedelta(minutes=15)

        # Historique des mentions pour détecter le momentum
        self._mention_history: Dict[str, List[Tuple[datetime, int]]] = {}

        # Persistance
        self._data_dir = Path("data/social")
        self._data_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """Initialise les scanners"""
        if self.stocktwits:
            await self.stocktwits.initialize()
        if self.reddit:
            await self.reddit.initialize()

        # Charger l'historique
        self._load_history()

        logger.info("Social scanner initialized")

    async def close(self):
        """Ferme les scanners"""
        # Sauvegarder l'historique
        self._save_history()

        if self.stocktwits:
            await self.stocktwits.close()
        if self.reddit:
            await self.reddit.close()

    def _load_history(self):
        """Charge l'historique des mentions"""
        history_file = self._data_dir / "mention_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    for symbol, entries in data.items():
                        self._mention_history[symbol] = [
                            (datetime.fromisoformat(e[0]), e[1])
                            for e in entries
                        ]
            except Exception as e:
                logger.error(f"Error loading mention history: {e}")

    def _save_history(self):
        """Sauvegarde l'historique des mentions"""
        history_file = self._data_dir / "mention_history.json"
        try:
            # Nettoyer les entrées de plus de 7 jours
            cutoff = datetime.now() - timedelta(days=7)
            clean_history = {}
            for symbol, entries in self._mention_history.items():
                recent = [(ts.isoformat(), count) for ts, count in entries if ts > cutoff]
                if recent:
                    clean_history[symbol] = recent

            with open(history_file, 'w') as f:
                json.dump(clean_history, f)
        except Exception as e:
            logger.error(f"Error saving mention history: {e}")

    async def full_scan(self, use_cache: bool = True) -> SocialScanResult:
        """
        Effectue un scan complet de tous les réseaux sociaux.

        Args:
            use_cache: Utiliser le cache si disponible

        Returns:
            Résultat du scan avec trending symbols et sentiment
        """
        # Check cache
        if use_cache and self._last_scan:
            age = datetime.now() - self._last_scan.timestamp
            if age < self._scan_cache_ttl:
                logger.info(f"Using cached scan ({age.seconds}s old)")
                return self._last_scan

        logger.info("Starting full social scan...")

        all_mentions: List[SocialMention] = []
        sources_stats = {}

        # Scan StockTwits
        if self.stocktwits:
            try:
                # Get trending first
                trending = await self.stocktwits.get_trending()

                # Get messages for top trending symbols
                for sym_data in trending[:20]:
                    symbol = sym_data['symbol']
                    mentions = await self.stocktwits.get_symbol_messages(symbol, limit=20)
                    all_mentions.extend(mentions)

                sources_stats['stocktwits'] = len([m for m in all_mentions if m.source == 'stocktwits'])
                logger.info(f"StockTwits: {sources_stats['stocktwits']} mentions")

            except Exception as e:
                logger.error(f"StockTwits scan error: {e}")
                sources_stats['stocktwits'] = 0

        # Scan Reddit
        if self.reddit:
            try:
                reddit_mentions = await self.reddit.scan_all_subreddits()
                all_mentions.extend(reddit_mentions)
                sources_stats['reddit'] = len(reddit_mentions)
                logger.info(f"Reddit: {sources_stats['reddit']} mentions")

            except Exception as e:
                logger.error(f"Reddit scan error: {e}")
                sources_stats['reddit'] = 0

        # Agréger par symbole
        trending_symbols = self._aggregate_mentions(all_mentions)

        # Calculer le sentiment global
        if all_mentions:
            overall_sentiment = sum(m.sentiment_score for m in all_mentions) / len(all_mentions)
        else:
            overall_sentiment = 0.0

        # Extraire les hot topics
        hot_topics = self._extract_hot_topics(all_mentions)

        # Mettre à jour l'historique
        now = datetime.now()
        for ts in trending_symbols:
            if ts.symbol not in self._mention_history:
                self._mention_history[ts.symbol] = []
            self._mention_history[ts.symbol].append((now, ts.mention_count))

        # Créer le résultat
        result = SocialScanResult(
            timestamp=now,
            trending_symbols=trending_symbols,
            total_mentions=len(all_mentions),
            overall_sentiment=overall_sentiment,
            hot_topics=hot_topics,
            sources_stats=sources_stats
        )

        self._last_scan = result

        logger.info(
            f"Social scan complete: {len(trending_symbols)} trending symbols, "
            f"{len(all_mentions)} total mentions, sentiment={overall_sentiment:.2f}"
        )

        return result

    async def scan_stocktwits(self) -> List[SocialMention]:
        """Scan StockTwits seulement"""
        if not self.stocktwits:
            return []

        all_mentions = []
        trending = await self.stocktwits.get_trending()

        for sym_data in trending[:15]:
            mentions = await self.stocktwits.get_symbol_messages(sym_data['symbol'], limit=20)
            all_mentions.extend(mentions)

        return all_mentions

    async def scan_reddit(self) -> List[SocialMention]:
        """Scan Reddit seulement"""
        if not self.reddit:
            return []

        return await self.reddit.scan_all_subreddits()

    async def get_symbol_sentiment(self, symbol: str) -> Dict:
        """
        Récupère le sentiment détaillé pour un symbole spécifique.

        Args:
            symbol: Ticker du symbole

        Returns:
            Dict avec sentiment, mentions récentes, etc.
        """
        mentions = []

        # StockTwits
        if self.stocktwits:
            try:
                st_mentions = await self.stocktwits.get_symbol_messages(symbol, limit=30)
                mentions.extend(st_mentions)
            except Exception as e:
                logger.error(f"StockTwits error for {symbol}: {e}")

        if not mentions:
            return {
                'symbol': symbol,
                'found': False,
                'mention_count': 0,
                'avg_sentiment': 0.0
            }

        avg_sentiment = sum(m.sentiment_score for m in mentions) / len(mentions)

        # Catégoriser
        if avg_sentiment > 0.5:
            label = 'very_bullish'
        elif avg_sentiment > 0.2:
            label = 'bullish'
        elif avg_sentiment < -0.5:
            label = 'very_bearish'
        elif avg_sentiment < -0.2:
            label = 'bearish'
        else:
            label = 'neutral'

        return {
            'symbol': symbol,
            'found': True,
            'mention_count': len(mentions),
            'avg_sentiment': avg_sentiment,
            'sentiment_label': label,
            'recent_mentions': [m.to_dict() for m in mentions[:5]]
        }

    async def get_symbol_details_for_narrative(self, symbol: str) -> Dict:
        """
        Récupère les détails enrichis d'un symbole pour le NarrativeGenerator.

        Inclut les thèmes extraits, les posts clés, et le contexte.

        Args:
            symbol: Ticker du symbole

        Returns:
            Dict avec données enrichies pour rapport narratif
        """
        result = {
            'symbol': symbol,
            'reddit': {
                'mentions': 0,
                'baseline': 10,  # Valeur moyenne historique
                'sentiment': 0.5,
                'themes': [],
                'top_posts': [],
                'catalysts': []
            },
            'stocktwits': {
                'mentions': 0,
                'sentiment': 0.5,
                'top_posts': []
            }
        }

        # Chercher dans le dernier scan
        if self._last_scan:
            for trending in self._last_scan.trending_symbols:
                if trending.symbol == symbol:
                    result['reddit']['mentions'] = trending.mention_count
                    result['reddit']['sentiment'] = trending.avg_sentiment

                    # Extraire les thèmes des posts
                    themes = []
                    top_posts = []
                    catalysts = []

                    for mention in trending.top_mentions:
                        # Ajouter le post
                        top_posts.append(mention.text[:200])

                        # Extraire les thèmes du texte
                        text_lower = mention.text.lower()

                        # Thèmes techniques
                        if any(w in text_lower for w in ['breakout', 'technical', 'support', 'resistance']):
                            themes.append('Technical Analysis')
                        if any(w in text_lower for w in ['earnings', 'revenue', 'guidance']):
                            themes.append('Earnings')
                            catalysts.append('Earnings related')
                        if any(w in text_lower for w in ['ai', 'artificial intelligence', 'chip', 'semiconductor']):
                            themes.append('AI/Tech')
                        if any(w in text_lower for w in ['partnership', 'deal', 'contract']):
                            themes.append('Business Deal')
                            catalysts.append('Partnership/Deal')
                        if any(w in text_lower for w in ['fda', 'approval', 'trial']):
                            themes.append('FDA/Biotech')
                            catalysts.append('FDA Catalyst')
                        if any(w in text_lower for w in ['ces', 'conference', 'keynote', 'announcement']):
                            themes.append('Event')
                            catalysts.append('Upcoming Event')
                        if any(w in text_lower for w in ['squeeze', 'short interest', 'gamma']):
                            themes.append('Short Squeeze')
                        if any(w in text_lower for w in ['undervalued', 'cheap', 'value']):
                            themes.append('Value Play')
                        if any(w in text_lower for w in ['datacenter', 'cloud', 'demand']):
                            themes.append('Demand Growth')

                    result['reddit']['themes'] = list(set(themes))[:5]
                    result['reddit']['top_posts'] = top_posts[:3]
                    result['reddit']['catalysts'] = list(set(catalysts))[:3]

                    break

        # StockTwits data
        if self.stocktwits:
            try:
                st_mentions = await self.stocktwits.get_symbol_messages(symbol, limit=20)
                if st_mentions:
                    result['stocktwits']['mentions'] = len(st_mentions)
                    result['stocktwits']['sentiment'] = sum(m.sentiment_score for m in st_mentions) / len(st_mentions)
                    result['stocktwits']['top_posts'] = [m.text[:200] for m in st_mentions[:3]]
            except Exception as e:
                logger.debug(f"StockTwits error for {symbol}: {e}")

        # Calculate momentum if we have history
        if symbol in self._mention_history and self._mention_history[symbol]:
            current = result['reddit']['mentions']
            baseline = self._calculate_baseline(symbol)
            result['reddit']['baseline'] = baseline
            result['reddit']['momentum_pct'] = ((current - baseline) / baseline * 100) if baseline > 0 else 0

        return result

    def _calculate_baseline(self, symbol: str) -> int:
        """Calculate baseline mentions for a symbol from history"""
        if symbol not in self._mention_history:
            return 10

        history = self._mention_history[symbol]
        if len(history) < 2:
            return 10

        # Average of historical mentions
        total = sum(count for _, count in history)
        return max(1, total // len(history))

    def _aggregate_mentions(self, mentions: List[SocialMention]) -> List[TrendingSymbol]:
        """
        Agrège les mentions par symbole pour créer la liste trending.

        Args:
            mentions: Liste de toutes les mentions

        Returns:
            Liste de symboles trending triés par mentions
        """
        # Grouper par symbole
        symbol_data: Dict[str, List[SocialMention]] = {}
        for m in mentions:
            if m.symbol not in symbol_data:
                symbol_data[m.symbol] = []
            symbol_data[m.symbol].append(m)

        trending = []
        now = datetime.now()

        for symbol, symbol_mentions in symbol_data.items():
            if len(symbol_mentions) < 2:  # Minimum 2 mentions
                continue

            # Calculer les stats
            avg_sentiment = sum(m.sentiment_score for m in symbol_mentions) / len(symbol_mentions)
            sources = list(set(m.source for m in symbol_mentions))

            # Catégoriser le sentiment
            if avg_sentiment > 0.5:
                label = 'very_bullish'
            elif avg_sentiment > 0.2:
                label = 'bullish'
            elif avg_sentiment < -0.5:
                label = 'very_bearish'
            elif avg_sentiment < -0.2:
                label = 'bearish'
            else:
                label = 'neutral'

            # Calculer le momentum (vs 24h précédentes)
            momentum = self._calculate_momentum(symbol, len(symbol_mentions))

            # Top mentions par engagement
            top_mentions = sorted(symbol_mentions, key=lambda m: m.engagement, reverse=True)[:3]

            trending.append(TrendingSymbol(
                symbol=symbol,
                mention_count=len(symbol_mentions),
                avg_sentiment=avg_sentiment,
                sentiment_label=label,
                sources=sources,
                momentum_24h=momentum,
                top_mentions=top_mentions
            ))

        # Trier par mentions (pondérées par sentiment positif)
        trending.sort(
            key=lambda t: t.mention_count * (1 + max(0, t.avg_sentiment)),
            reverse=True
        )

        return trending

    def _calculate_momentum(self, symbol: str, current_mentions: int) -> float:
        """
        Calcule le momentum des mentions vs les 24h précédentes.

        Returns:
            Ratio de changement (-1 à +∞)
        """
        if symbol not in self._mention_history:
            return 0.0

        # Mentions des dernières 24h
        cutoff = datetime.now() - timedelta(hours=24)
        history = self._mention_history[symbol]

        old_mentions = sum(count for ts, count in history if ts < cutoff)

        if old_mentions == 0:
            return 1.0 if current_mentions > 0 else 0.0

        return (current_mentions - old_mentions) / old_mentions

    def _extract_hot_topics(self, mentions: List[SocialMention]) -> List[str]:
        """
        Extrait les sujets chauds des mentions.

        Recherche des patterns communs dans les textes.
        """
        topics_counter = Counter()

        topic_patterns = [
            (r'\b(earnings|revenue|beat|miss)\b', 'earnings'),
            (r'\b(FDA|approval|trial|drug)\b', 'biotech_catalyst'),
            (r'\b(squeeze|short|gamma)\b', 'short_squeeze'),
            (r'\b(AI|artificial intelligence|GPT|LLM)\b', 'AI'),
            (r'\b(EV|electric vehicle|battery)\b', 'EV'),
            (r'\b(rate|fed|powell|fomc)\b', 'fed_policy'),
            (r'\b(crypto|bitcoin|btc|eth)\b', 'crypto'),
            (r'\b(merger|acquisition|buyout)\b', 'M&A'),
            (r'\b(dividend|yield)\b', 'dividend'),
            (r'\b(recession|crash|bear market)\b', 'macro_fear'),
        ]

        for mention in mentions:
            text = mention.text.lower()
            for pattern, topic in topic_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    topics_counter[topic] += 1

        # Retourner les top 5 topics
        return [topic for topic, _ in topics_counter.most_common(5)]

    def get_cached_result(self) -> Optional[SocialScanResult]:
        """Retourne le dernier scan en cache"""
        return self._last_scan


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_social_scanner_instance: Optional[SocialScanner] = None


async def get_social_scanner(
    stocktwits_token: Optional[str] = None,
    enable_stocktwits: bool = False,
    enable_reddit: bool = False
) -> SocialScanner:
    """
    Factory pour obtenir une instance du scanner social.

    Singleton pattern pour éviter les multiples instances.
    """
    global _social_scanner_instance

    if _social_scanner_instance is None:
        _social_scanner_instance = SocialScanner(
            stocktwits_token=stocktwits_token,
            enable_stocktwits=enable_stocktwits,
            enable_reddit=enable_reddit
        )
        await _social_scanner_instance.initialize()

    return _social_scanner_instance


async def close_social_scanner():
    """Ferme le scanner global"""
    global _social_scanner_instance

    if _social_scanner_instance:
        await _social_scanner_instance.close()
        _social_scanner_instance = None


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        print("=== Social Scanner Test ===\n")

        scanner = SocialScanner()
        await scanner.initialize()

        try:
            # Full scan
            print("Running full social scan...")
            result = await scanner.full_scan()

            print(f"\nTotal mentions: {result.total_mentions}")
            print(f"Overall sentiment: {result.overall_sentiment:.2f}")
            print(f"Sources: {result.sources_stats}")
            print(f"\nHot topics: {result.hot_topics}")

            print("\nTop 10 Trending Symbols:")
            for ts in result.trending_symbols[:10]:
                print(f"  ${ts.symbol}: {ts.mention_count} mentions, "
                      f"sentiment={ts.avg_sentiment:.2f} ({ts.sentiment_label}), "
                      f"momentum={ts.momentum_24h:+.0%}")

            # Bullish symbols
            bullish = result.get_bullish_symbols(min_sentiment=0.3)
            print(f"\nBullish symbols: {bullish[:10]}")

        finally:
            await scanner.close()

    asyncio.run(main())
