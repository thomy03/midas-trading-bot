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
        'reuters_markets': 'https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best',
        'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss',
        'cnbc': 'https://www.cnbc.com/id/10000664/device/rss/rss.html',
        'yahoo_finance': 'https://finance.yahoo.com/rss/topstories',
        'seeking_alpha': 'https://seekingalpha.com/market_currents.xml',
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

    async def _ensure_session(self):
        """Crée la session HTTP si nécessaire"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Ferme la session HTTP"""
        if self.session and not self.session.closed:
            await self.session.close()

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
                url = f'https://www.social.com/r/{subsocial}/hot.json'
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
                            url=f"https://social.com{post_data.get('permalink', '')}",
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
