"""
News Fetcher Module
Récupération de news financières depuis différentes sources.

Sources supportées:
- NewsAPI (news générales)
- Alpha Vantage News (news financières)
- Reddit (r/wallstreetbets, r/stocks, r/investing)
- Arxiv (publications IA/tech)
"""

import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import time
import logging

logger = logging.getLogger(__name__)


class NewsSource(Enum):
    """Sources de news disponibles"""
    NEWS_API = "newsapi"
    ALPHA_VANTAGE = "alpha_vantage"
    REDDIT = "social"
    ARXIV = "arxiv"
    RSS_FEEDS = "rss_feeds"


@dataclass
class NewsArticle:
    """Article de news"""
    title: str
    source: str
    url: str
    published_at: datetime
    content: str
    summary: Optional[str] = None
    sentiment: Optional[float] = None  # -1 à +1
    symbols: List[str] = field(default_factory=list)
    sectors: List[str] = field(default_factory=list)
    relevance_score: float = 0.5


@dataclass
class ResearchPaper:
    """Publication académique (Arxiv, etc.)"""
    title: str
    authors: List[str]
    abstract: str
    url: str
    published_at: datetime
    categories: List[str]
    relevance_score: float = 0.5


class NewsFetcher:
    """
    Récupérateur de news multi-sources.
    """

    # RSS Feeds pour les news financières
    RSS_FEEDS = {
        # Original
        'reuters_markets': 'https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best',
        'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
        'cnbc': 'https://www.cnbc.com/id/10000664/device/rss/rss.html',
        'yahoo_finance': 'https://finance.yahoo.com/rss/topstories',
        'seeking_alpha': 'https://seekingalpha.com/market_currents.xml',
        # Tech
        'techcrunch': 'https://techcrunch.com/feed/',
        'the_verge': 'https://www.theverge.com/rss/index.xml',
        'ars_technica': 'https://feeds.arstechnica.com/arstechnica/index',
        'wired': 'https://www.wired.com/feed/rss',
        'hacker_news': 'https://hnrss.org/frontpage',
        # Science
        'science_daily': 'https://www.sciencedaily.com/rss/all.xml',
        'nature_news': 'https://www.nature.com/nature.rss',
        # Business/Finance
        'marketwatch': 'https://feeds.content.dowjones.io/public/rss/mw_topstories',
        'les_echos': 'https://syndication.lesechos.fr/rss/rss_une.xml',
        'bfm_bourse': 'https://www.tradingsat.com/rss/actualites-bourse.xml',
        # General
        'ap_news': 'https://rsshub.app/apnews/topics/business',
        'bbc_business': 'https://feeds.bbci.co.uk/news/business/rss.xml',
        # EU Specific
        'boursorama': 'https://www.boursorama.com/rss/actualites/dernieres.xml',
        'investing_com': 'https://www.investing.com/rss/news.rss',
        'zonebourse': 'https://www.zonebourse.com/rss/',
    }

    # Subsocials pertinents
    REDDIT_SUBS = [
        'wallstreetbets',
        'stocks',
        'investing',
        'stockmarket',
        'options',
    ]

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        alphavantage_key: Optional[str] = None
    ):
        """
        Initialise le fetcher.

        Args:
            newsapi_key: Clé API NewsAPI (newsapi.org)
            alphavantage_key: Clé API Alpha Vantage
        """
        self.newsapi_key = newsapi_key or os.getenv('NEWSAPI_KEY')
        self.alphavantage_key = alphavantage_key or os.getenv('ALPHAVANTAGE_KEY')
        self.session: Optional[aiohttp.ClientSession] = None
        self._rss_cache: List[NewsArticle] = []
        self._rss_cache_time: float = 0.0
        self._rss_cache_ttl: float = 900.0  # 15 minutes

    async def _ensure_session(self):
        """Crée la session HTTP si nécessaire"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Ferme la session HTTP"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def fetch_all_rss(self) -> List[NewsArticle]:
        """Fetch ALL RSS feeds in parallel. Cache for 15 min."""
        now = time.time()
        if self._rss_cache and (now - self._rss_cache_time) < self._rss_cache_ttl:
            return self._rss_cache

        await self._ensure_session()
        import xml.etree.ElementTree as ET

        async def _fetch_one(name: str, url: str) -> List[NewsArticle]:
            articles = []
            try:
                async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return []
                    text = await resp.text()
                    root = ET.fromstring(text)
                    # Try RSS 2.0
                    items = root.findall('.//item')
                    if not items:
                        # Try Atom
                        ns = {'atom': 'http://www.w3.org/2005/Atom'}
                        items = root.findall('atom:entry', ns)
                    for item in items[:20]:
                        title = link = pub_date = desc = None
                        # RSS 2.0
                        t = item.find('title')
                        if t is not None and t.text:
                            title = t.text.strip()
                        l = item.find('link')
                        if l is not None:
                            link = l.text.strip() if l.text else (l.get('href') or '')
                        d = item.find('description')
                        if d is not None and d.text:
                            desc = d.text[:500]
                        p = item.find('pubDate')
                        if p is not None and p.text:
                            try:
                                from email.utils import parsedate_to_datetime
                                pub_date = parsedate_to_datetime(p.text)
                            except Exception:
                                pub_date = datetime.now()
                        # Atom fallback
                        if title is None:
                            ns2 = {'atom': 'http://www.w3.org/2005/Atom'}
                            t2 = item.find('atom:title', ns2)
                            if t2 is not None and t2.text:
                                title = t2.text.strip()
                        if not link:
                            ns2 = {'atom': 'http://www.w3.org/2005/Atom'}
                            l2 = item.find('atom:link', ns2)
                            if l2 is not None:
                                link = l2.get('href', '')
                        if pub_date is None:
                            ns2 = {'atom': 'http://www.w3.org/2005/Atom'}
                            u = item.find('atom:updated', ns2) or item.find('atom:published', ns2)
                            if u is not None and u.text:
                                try:
                                    pub_date = datetime.fromisoformat(u.text.replace('Z', '+00:00'))
                                except Exception:
                                    pub_date = datetime.now()
                        if not title:
                            continue
                        articles.append(NewsArticle(
                            title=title,
                            source=name,
                            url=link or '',
                            published_at=pub_date or datetime.now(),
                            content=desc or '',
                            summary=desc or title,
                        ))
            except Exception as e:
                logger.debug(f'[NEWS] RSS {name} failed: {e}')
            return articles

        tasks = [_fetch_one(name, url) for name, url in self.RSS_FEEDS.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_articles = []
        for r in results:
            if isinstance(r, list):
                all_articles.extend(r)
        def _sort_key(a):
            try:
                dt = a.published_at
                if dt.tzinfo is not None:
                    return dt.replace(tzinfo=None)
                return dt
            except Exception:
                return datetime.min
        all_articles.sort(key=_sort_key, reverse=True)
        self._rss_cache = all_articles
        self._rss_cache_time = now
        logger.info(f'[NEWS] RSS cache refreshed: {len(all_articles)} articles from {len(self.RSS_FEEDS)} feeds')
        return all_articles

    async def fetch_symbol_news(self, symbol: str, company_name: str = None) -> List[NewsArticle]:
        """Search cached RSS + NewsAPI for symbol-specific news."""
        await self.fetch_all_rss()

        search_terms = [symbol.upper()]
        # Also search without exchange suffix (.PA, .DE, .AS, etc.)
        bare = symbol.split('.')[0].upper()
        if bare != symbol.upper():
            search_terms.append(bare)
        if company_name:
            search_terms.append(company_name.upper())
        TICKER_NAMES = {
            'AAPL': 'APPLE', 'MSFT': 'MICROSOFT', 'GOOGL': 'GOOGLE', 'GOOG': 'GOOGLE',
            'AMZN': 'AMAZON', 'NVDA': 'NVIDIA', 'META': 'META', 'TSLA': 'TESLA',
            'AMD': 'AMD', 'NFLX': 'NETFLIX', 'CRM': 'SALESFORCE', 'ADBE': 'ADOBE',
            'INTC': 'INTEL', 'AVGO': 'BROADCOM', 'QCOM': 'QUALCOMM',
            'JPM': 'JPMORGAN', 'GS': 'GOLDMAN', 'BAC': 'BANK OF AMERICA',
            'V': 'VISA', 'MA': 'MASTERCARD', 'DIS': 'DISNEY', 'COST': 'COSTCO',
            'PFE': 'PFIZER', 'JNJ': 'JOHNSON', 'UNH': 'UNITEDHEALTH',
            'XOM': 'EXXON', 'CVX': 'CHEVRON', 'WMT': 'WALMART',
        }
        mapped = TICKER_NAMES.get(symbol.upper())
        if mapped and mapped not in [t.upper() for t in search_terms]:
            search_terms.append(mapped)

        relevant = []
        for article in self._rss_cache:
            text = (article.title + ' ' + (article.content or '')).upper()
            for term in search_terms:
                if term in text:
                    relevant.append(article)
                    break

        if self.newsapi_key and len(relevant) < 3:
            try:
                api_results = await self.fetch_newsapi(
                    query=company_name or symbol,
                    from_date=datetime.now() - timedelta(days=3),
                    page_size=5
                )
                relevant.extend(api_results)
            except Exception as e:
                logger.debug(f'[NEWS] NewsAPI symbol search failed for {symbol}: {e}')

        seen = set()
        unique = []
        for a in relevant:
            key = a.title.lower()[:80]
            if key not in seen:
                seen.add(key)
                unique.append(a)
        unique.sort(key=lambda a: a.published_at, reverse=True)
        return unique[:20]

    async def fetch_latest(self) -> dict:
        """
        Fetch latest news from all available sources.
        Called by IntelligenceOrchestrator._collect_intelligence().
        Returns dict with news articles and social posts.
        """
        results = {"articles": [], "social": [], "sectors": {}, "rss": []}
        
        # RSS feeds (cached 15 min)
        try:
            rss_articles = await self.fetch_all_rss()
            results["rss"] = rss_articles[:30]  # Top 30 most recent
            results["articles"].extend(rss_articles[:15])
        except Exception as e:
            logger.warning(f"[NEWS] RSS fetch failed: {e}")
        
        # Market-wide news
        try:
            articles = await self.fetch_newsapi(
                query="stock market trading earnings",
                page_size=10
            )
            results["articles"].extend(articles)
        except Exception as e:
            logger.warning(f"[NEWS] NewsAPI fetch failed: {e}")
        
        # Social/Reddit posts
        try:
            social = await self.fetch_social_posts(subsocials=["wallstreetbets", "stocks", "investing"], limit=10)
            results["social"].extend(social)
        except Exception as e:
            logger.warning(f"[NEWS] Social fetch failed: {e}")
        
        # Sector news
        try:
            sector_keywords = {"technology": ["tech", "AI", "semiconductor", "software"], "healthcare": ["pharma", "biotech", "FDA", "drug"], "finance": ["banking", "rates", "Fed", "earnings"]}
            for sector, kw in sector_keywords.items():
                sector_news = await self.fetch_sector_news(sector=sector, keywords=kw, days_back=3)
                results["sectors"][sector] = sector_news
        except Exception as e:
            logger.warning(f"[NEWS] Sector news failed: {e}")
        
        logger.info(f"[NEWS] Fetched {len(results['articles'])} articles, {len(results['social'])} social posts")
        return results

    async def fetch_newsapi(
        self,
        query: str,
        from_date: Optional[datetime] = None,
        language: str = 'en',
        page_size: int = 20
    ) -> List[NewsArticle]:
        """
        Récupère des news depuis NewsAPI.

        Args:
            query: Mots-clés de recherche
            from_date: Date de début (défaut: 7 jours)
            language: Langue des articles
            page_size: Nombre d'articles max

        Returns:
            Liste d'articles
        """
        if not self.newsapi_key:
            return []

        await self._ensure_session()

        from_date = from_date or (datetime.now() - timedelta(days=7))

        params = {
            'q': query,
            'from': from_date.strftime('%Y-%m-%d'),
            'language': language,
            'sortBy': 'relevancy',
            'pageSize': page_size,
            'apiKey': self.newsapi_key
        }

        try:
            async with self.session.get(
                'https://newsapi.org/v2/everything',
                params=params
            ) as response:
                if response.status != 200:
                    return []

                data = await response.json()
                articles = []

                for item in data.get('articles', []):
                    try:
                        published = datetime.fromisoformat(
                            item['publishedAt'].replace('Z', '+00:00')
                        )
                    except (ValueError, TypeError, KeyError, AttributeError):
                        published = datetime.now()

                    articles.append(NewsArticle(
                        title=item.get('title', ''),
                        source=item.get('source', {}).get('name', 'Unknown'),
                        url=item.get('url', ''),
                        published_at=published,
                        content=item.get('content', '') or item.get('description', ''),
                        summary=item.get('description', '')
                    ))

                return articles

        except Exception as e:
            print(f"NewsAPI error: {e}")
            return []

    async def fetch_alphavantage_news(
        self,
        tickers: List[str],
        topics: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[NewsArticle]:
        """
        Récupère des news depuis Alpha Vantage News API.

        Args:
            tickers: Liste de symboles boursiers
            topics: Topics (technology, finance, etc.)
            limit: Nombre max d'articles

        Returns:
            Liste d'articles
        """
        if not self.alphavantage_key:
            return []

        await self._ensure_session()

        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': ','.join(tickers),
            'limit': limit,
            'apikey': self.alphavantage_key
        }

        if topics:
            params['topics'] = ','.join(topics)

        try:
            async with self.session.get(
                'https://www.alphavantage.co/query',
                params=params
            ) as response:
                if response.status != 200:
                    return []

                data = await response.json()
                articles = []

                for item in data.get('feed', []):
                    try:
                        published = datetime.strptime(
                            item['time_published'],
                            '%Y%m%dT%H%M%S'
                        )
                    except (ValueError, TypeError, KeyError):
                        published = datetime.now()

                    # Extraire les symboles mentionnés
                    symbols = [
                        t['ticker'] for t in item.get('ticker_sentiment', [])
                    ]

                    # Sentiment moyen
                    sentiments = [
                        float(t.get('ticker_sentiment_score', 0))
                        for t in item.get('ticker_sentiment', [])
                    ]
                    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

                    articles.append(NewsArticle(
                        title=item.get('title', ''),
                        source=item.get('source', 'Unknown'),
                        url=item.get('url', ''),
                        published_at=published,
                        content=item.get('summary', ''),
                        summary=item.get('summary', ''),
                        sentiment=avg_sentiment,
                        symbols=symbols
                    ))

                return articles

        except Exception as e:
            print(f"Alpha Vantage error: {e}")
            return []

    async def fetch_social_posts(
        self,
        subsocials: Optional[List[str]] = None,
        query: Optional[str] = None,
        limit: int = 25,
        time_filter: str = 'day'
    ) -> List[NewsArticle]:
        """
        Récupère des posts depuis Reddit (via l'API publique).

        Args:
            subsocials: Liste de subsocials (défaut: REDDIT_SUBS)
            query: Recherche optionnelle
            limit: Nombre max de posts par subsocial
            time_filter: Filtre temporel (hour, day, week, month, year, all)

        Returns:
            Liste de posts formatés comme NewsArticle
        """
        await self._ensure_session()

        subsocials = subsocials or self.REDDIT_SUBS
        all_posts = []

        headers = {
            'User-Agent': 'TradingBot/1.0 (Market Research)'
        }

        for subsocial in subsocials:
            try:
                url = f'https://www.reddit.com/r/{subsocial}/hot.json'
                params = {'limit': limit, 't': time_filter}

                async with self.session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        continue

                    data = await response.json()
                    posts = data.get('data', {}).get('children', [])

                    for post in posts:
                        post_data = post.get('data', {})

                        # Ignorer les posts non pertinents
                        if post_data.get('stickied') or post_data.get('is_video'):
                            continue

                        try:
                            published = datetime.fromtimestamp(
                                post_data.get('created_utc', 0)
                            )
                        except (ValueError, TypeError, OSError):
                            published = datetime.now()

                        # Calculer un score de relevance basé sur upvotes/comments
                        upvotes = post_data.get('ups', 0)
                        comments = post_data.get('num_comments', 0)
                        relevance = min(1.0, (upvotes + comments * 2) / 1000)

                        all_posts.append(NewsArticle(
                            title=post_data.get('title', ''),
                            source=f'Reddit r/{subsocial}',
                            url=f"https://www.reddit.com{post_data.get('permalink', '')}",
                            published_at=published,
                            content=post_data.get('selftext', '')[:2000],
                            summary=post_data.get('title', ''),
                            relevance_score=relevance
                        ))

            except Exception as e:
                print(f"Reddit error for r/{subsocial}: {e}")
                continue

        # Trier par relevance
        all_posts.sort(key=lambda x: x.relevance_score, reverse=True)
        return all_posts

    async def fetch_arxiv_papers(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        max_results: int = 20
    ) -> List[ResearchPaper]:
        """
        Recherche des publications sur Arxiv.

        Args:
            query: Mots-clés de recherche
            categories: Catégories Arxiv (cs.AI, cs.LG, q-fin, etc.)
            max_results: Nombre max de résultats

        Returns:
            Liste de publications
        """
        await self._ensure_session()

        # Construire la requête Arxiv
        search_query = query
        if categories:
            cat_query = ' OR '.join([f'cat:{cat}' for cat in categories])
            search_query = f'({query}) AND ({cat_query})'

        params = {
            'search_query': f'all:{search_query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }

        try:
            async with self.session.get(
                'http://export.arxiv.org/api/query',
                params=params
            ) as response:
                if response.status != 200:
                    return []

                # Parser le XML Atom
                text = await response.text()
                papers = self._parse_arxiv_response(text)
                return papers

        except Exception as e:
            print(f"Arxiv error: {e}")
            return []

    def _parse_arxiv_response(self, xml_text: str) -> List[ResearchPaper]:
        """Parse la réponse XML d'Arxiv"""
        import xml.etree.ElementTree as ET

        papers = []

        try:
            root = ET.fromstring(xml_text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns)
                summary = entry.find('atom:summary', ns)
                published = entry.find('atom:published', ns)

                # Auteurs
                authors = [
                    author.find('atom:name', ns).text
                    for author in entry.findall('atom:author', ns)
                    if author.find('atom:name', ns) is not None
                ]

                # Catégories
                categories = [
                    cat.get('term', '')
                    for cat in entry.findall('atom:category', ns)
                ]

                # URL
                link = entry.find("atom:link[@title='pdf']", ns)
                url = link.get('href', '') if link is not None else ''

                try:
                    pub_date = datetime.fromisoformat(
                        published.text.replace('Z', '+00:00')
                    ) if published is not None else datetime.now()
                except (ValueError, TypeError, AttributeError):
                    pub_date = datetime.now()

                papers.append(ResearchPaper(
                    title=title.text.strip() if title is not None else '',
                    authors=authors,
                    abstract=summary.text.strip() if summary is not None else '',
                    url=url,
                    published_at=pub_date,
                    categories=categories
                ))

        except Exception as e:
            print(f"Arxiv parsing error: {e}")

        return papers

    async def fetch_sector_news(
        self,
        sector: str,
        keywords: List[str],
        days_back: int = 7
    ) -> List[NewsArticle]:
        """
        Récupère les news pour un secteur spécifique.

        Args:
            sector: Nom du secteur
            keywords: Mots-clés associés au secteur
            days_back: Nombre de jours d'historique

        Returns:
            Liste d'articles combinés de plusieurs sources
        """
        query = ' OR '.join(keywords)
        from_date = datetime.now() - timedelta(days=days_back)

        # Récupérer depuis plusieurs sources en parallèle
        tasks = [
            self.fetch_newsapi(query, from_date),
            self.fetch_social_posts(query=query.replace(' OR ', ' '))
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles = []
        for result in results:
            if isinstance(result, list):
                for article in result:
                    article.sectors = [sector]
                all_articles.extend(result)

        # Dédupliquer par titre
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title_lower = article.title.lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_articles.append(article)

        # Trier par date
        unique_articles.sort(key=lambda x: x.published_at, reverse=True)

        return unique_articles


# Singleton
news_fetcher: Optional[NewsFetcher] = None


def get_news_fetcher() -> NewsFetcher:
    """Retourne le singleton NewsFetcher"""
    global news_fetcher
    if news_fetcher is None:
        news_fetcher = NewsFetcher()
    return news_fetcher
