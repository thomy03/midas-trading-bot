"""
Social Scanner Module
Scanner des réseaux sociaux pour détecter les tendances et le sentiment.

Sources:
- StockTwits: Sentiment et trending tickers

Usage:
    from src.intelligence.social_scanner import SocialScanner

    scanner = SocialScanner()
    await scanner.initialize()

    # Scan complet
    result = await scanner.full_scan()

    # Scan StockTwits seulement
    stocktwits_data = await scanner.scan_stocktwits()
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
    source: str  # 'stocktwits'
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
# SOCIAL SCANNER (COMBINED)
# =============================================================================

class SocialScanner:
    """
    Scanner social utilisant StockTwits.

    Fonctionnalités:
    - Agrégation des mentions
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
        enable_reddit: bool = False  # Deprecated, kept for compatibility
    ):
        """
        Initialise le scanner social.

        Args:
            stocktwits_token: Token d'accès StockTwits optionnel
            enable_stocktwits: Activer le scan StockTwits
            enable_reddit: Deprecated, ignored
        """
        self.stocktwits = StockTwitsScanner(access_token=stocktwits_token) if enable_stocktwits else None

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

        # Charger l'historique
        self._load_history()

        logger.info("Social scanner initialized")

    async def close(self):
        """Ferme les scanners"""
        # Sauvegarder l'historique
        self._save_history()

        if self.stocktwits:
            await self.stocktwits.close()

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
        Effectue un scan complet de StockTwits.

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

        Args:
            symbol: Ticker du symbole

        Returns:
            Dict avec données enrichies pour rapport narratif
        """
        result = {
            'symbol': symbol,
            'stocktwits': {
                'mentions': 0,
                'sentiment': 0.5,
                'top_posts': []
            }
        }

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

        return result

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
    enable_reddit: bool = False  # Deprecated, ignored
) -> SocialScanner:
    """
    Factory pour obtenir une instance du scanner social.

    Singleton pattern pour éviter les multiples instances.
    """
    global _social_scanner_instance

    if _social_scanner_instance is None:
        _social_scanner_instance = SocialScanner(
            stocktwits_token=stocktwits_token,
            enable_stocktwits=enable_stocktwits
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

        scanner = SocialScanner(enable_stocktwits=True)
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
