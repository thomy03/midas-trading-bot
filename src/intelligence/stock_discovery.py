"""
Stock Discovery Module
Découverte dynamique de stocks au-delà du SP500/NASDAQ100.

Ce module permet de:
1. Obtenir l'univers complet NASDAQ/NYSE/AMEX
2. Filtrer par critères (volume, market cap, secteur)
3. Identifier les stocks "buzz" des réseaux sociaux
4. Croiser les signaux techniques + sociaux

Usage:
    from src.intelligence.stock_discovery import StockDiscovery

    discovery = StockDiscovery()
    await discovery.initialize()

    # Découvrir les stocks trending
    trending = await discovery.discover_trending()

    # Filtrer par critères
    filtered = await discovery.filter_universe(min_volume=1_000_000, min_mcap=500_000_000)

    # Obtenir les candidats pour le screening
    candidates = await discovery.get_screening_candidates()
"""

import os
import json
import asyncio
import aiohttp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from collections import Counter
import logging
from pathlib import Path
import csv

# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class StockInfo:
    """Informations sur un stock"""
    symbol: str
    name: str
    exchange: str  # NASDAQ, NYSE, AMEX
    sector: str
    industry: str
    market_cap: float  # En USD
    avg_volume: float  # Volume moyen 20j
    price: float
    change_pct: float  # Variation du jour

    # Scores calculés
    social_score: float = 0.0  # Score des réseaux sociaux
    technical_score: float = 0.0  # Score technique
    momentum_score: float = 0.0  # Momentum price

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'name': self.name,
            'exchange': self.exchange,
            'sector': self.sector,
            'industry': self.industry,
            'market_cap': self.market_cap,
            'avg_volume': self.avg_volume,
            'price': self.price,
            'change_pct': self.change_pct,
            'social_score': self.social_score,
            'technical_score': self.technical_score,
            'momentum_score': self.momentum_score
        }

    @property
    def market_cap_category(self) -> str:
        """Catégorie de market cap"""
        if self.market_cap >= 200_000_000_000:
            return 'mega'
        elif self.market_cap >= 10_000_000_000:
            return 'large'
        elif self.market_cap >= 2_000_000_000:
            return 'mid'
        elif self.market_cap >= 300_000_000:
            return 'small'
        else:
            return 'micro'


@dataclass
class DiscoveryResult:
    """Résultat de la découverte de stocks"""
    timestamp: datetime
    total_universe: int
    filtered_count: int
    trending_symbols: List[str]
    candidates: List[StockInfo]
    filters_applied: Dict[str, Any]
    sources_used: List[str]

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_universe': self.total_universe,
            'filtered_count': self.filtered_count,
            'trending_symbols': self.trending_symbols,
            'candidates': [c.to_dict() for c in self.candidates],
            'filters_applied': self.filters_applied,
            'sources_used': self.sources_used
        }


# =============================================================================
# STOCK DISCOVERY
# =============================================================================

class StockDiscovery:
    """
    Découverte dynamique de stocks.

    Sources pour l'univers:
    1. NASDAQ Screener API (gratuit)
    2. Finviz screener (gratuit)
    3. Yahoo Finance screening
    4. Listes locales (backup)

    Critères de filtrage:
    - Volume minimum (liquidité)
    - Market cap minimum (éviter penny stocks)
    - Prix minimum (>$1 pour éviter les délisted)
    - Exchange (NASDAQ, NYSE, AMEX)
    """

    # URLs pour récupérer les listes
    SOURCES = {
        'nasdaq': 'https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange=NASDAQ',
        'nyse': 'https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange=NYSE',
        'amex': 'https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange=AMEX',
    }

    # Headers pour simuler un navigateur (certaines APIs bloquent les bots)
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }

    # Secteurs d'intérêt prioritaires
    PRIORITY_SECTORS = [
        'Technology', 'Health Care', 'Consumer Cyclical',
        'Financial Services', 'Communication Services',
        'Industrials', 'Energy'
    ]

    # Symboles à toujours inclure (mega caps référence)
    ALWAYS_INCLUDE = [
        'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META',
        'TSLA', 'BRK.B', 'UNH', 'JNJ', 'JPM', 'V', 'XOM', 'PG',
        'MA', 'HD', 'CVX', 'LLY', 'MRK', 'ABBV', 'AVGO', 'PEP',
        'KO', 'COST', 'TMO', 'MCD', 'WMT', 'CSCO', 'ACN', 'ABT'
    ]

    def __init__(self):
        """Initialise le module de découverte"""
        self.session: Optional[aiohttp.ClientSession] = None

        # Cache de l'univers complet
        self._universe: Dict[str, StockInfo] = {}
        self._universe_updated: Optional[datetime] = None
        self._universe_ttl = timedelta(hours=24)  # Refresh quotidien

        # Données sociales (injectées depuis social_scanner)
        self._social_data: Dict[str, Dict] = {}

        # Persistance
        self._data_dir = Path("data/discovery")
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Cache des candidats
        self._last_candidates: List[StockInfo] = []

    async def initialize(self):
        """Initialise la session HTTP"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(
                headers=self.HEADERS,
                timeout=timeout
            )

        # Charger le cache local si disponible
        await self._load_cached_universe()

        logger.info(f"Stock discovery initialized ({len(self._universe)} stocks in cache)")

    async def close(self):
        """Ferme la session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def _load_cached_universe(self):
        """Charge l'univers depuis le cache local"""
        cache_file = self._data_dir / "universe_cache.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)

                updated = datetime.fromisoformat(data.get('updated', '2000-01-01'))

                # Vérifier si le cache est encore valide
                if datetime.now() - updated < self._universe_ttl:
                    for stock_data in data.get('stocks', []):
                        symbol = stock_data['symbol']
                        self._universe[symbol] = StockInfo(**stock_data)
                    self._universe_updated = updated
                    logger.info(f"Loaded {len(self._universe)} stocks from cache")
                    return

            except Exception as e:
                logger.warning(f"Error loading universe cache: {e}")

        # Si pas de cache valide, charger les listes locales de base
        await self._load_local_tickers()

    async def _load_local_tickers(self):
        """Charge les tickers depuis les fichiers locaux (backup)"""
        tickers_dir = Path("data/tickers")

        # Charger les différentes listes
        sources = ['sp500.json', 'nasdaq100.json', 'europe.json', 'crypto.json']

        for source in sources:
            filepath = tickers_dir / source
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    # Format peut varier
                    if isinstance(data, list):
                        tickers_data = data
                    elif isinstance(data, dict):
                        tickers_data = data.get('tickers', [])
                    else:
                        continue

                    for ticker_item in tickers_data:
                        # Gérer différents formats
                        if isinstance(ticker_item, str):
                            symbol = ticker_item
                            name = ticker_item
                            sector = 'Unknown'
                        elif isinstance(ticker_item, dict):
                            symbol = ticker_item.get('symbol', '')
                            name = ticker_item.get('name', symbol)
                            sector = ticker_item.get('sector', 'Unknown')
                        else:
                            continue

                        if symbol and symbol not in self._universe:
                            self._universe[symbol] = StockInfo(
                                symbol=symbol,
                                name=name,
                                exchange='UNKNOWN',
                                sector=sector,
                                industry='Unknown',
                                market_cap=0,
                                avg_volume=0,
                                price=0,
                                change_pct=0
                            )
                except Exception as e:
                    logger.warning(f"Error loading {source}: {e}")

        logger.info(f"Loaded {len(self._universe)} stocks from local files")

    async def refresh_universe(self, force: bool = False) -> int:
        """
        Rafraîchit l'univers complet depuis les APIs.

        Args:
            force: Forcer le refresh même si le cache est valide

        Returns:
            Nombre de stocks dans l'univers
        """
        if not force and self._universe_updated:
            age = datetime.now() - self._universe_updated
            if age < self._universe_ttl:
                logger.info(f"Universe cache still valid ({age.seconds//3600}h old)")
                return len(self._universe)

        logger.info("Refreshing stock universe from NASDAQ API...")

        new_universe: Dict[str, StockInfo] = {}

        # Récupérer les 3 exchanges
        for exchange, url in self.SOURCES.items():
            try:
                stocks = await self._fetch_exchange(url, exchange.upper())
                for stock in stocks:
                    new_universe[stock.symbol] = stock

                logger.info(f"Fetched {len(stocks)} stocks from {exchange.upper()}")
                await asyncio.sleep(1)  # Rate limiting

            except Exception as e:
                logger.error(f"Error fetching {exchange}: {e}")

        # Ajouter les toujours inclus
        for symbol in self.ALWAYS_INCLUDE:
            if symbol not in new_universe:
                new_universe[symbol] = StockInfo(
                    symbol=symbol,
                    name=symbol,
                    exchange='NASDAQ',
                    sector='Unknown',
                    industry='Unknown',
                    market_cap=100_000_000_000,  # Assume large
                    avg_volume=10_000_000,
                    price=100,
                    change_pct=0
                )

        if new_universe:
            self._universe = new_universe
            self._universe_updated = datetime.now()

            # Sauvegarder en cache
            await self._save_universe_cache()

        logger.info(f"Universe refreshed: {len(self._universe)} total stocks")
        return len(self._universe)

    async def _fetch_exchange(self, url: str, exchange: str) -> List[StockInfo]:
        """Récupère les stocks d'un exchange depuis l'API NASDAQ"""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"NASDAQ API returned {response.status}")
                    return []

                data = await response.json()

        except Exception as e:
            logger.error(f"Error fetching from {url}: {e}")
            return []

        stocks = []
        rows = data.get('data', {}).get('table', {}).get('rows', [])

        for row in rows:
            try:
                symbol = row.get('symbol', '').strip()

                # Filtrer les symboles invalides
                if not symbol or len(symbol) > 5 or ' ' in symbol:
                    continue

                # Filtrer les warrants, units, etc.
                if any(x in symbol for x in ['.', '^', '$']):
                    continue

                # Parser les données
                market_cap = self._parse_market_cap(row.get('marketCap', '0'))
                price = self._parse_number(row.get('lastsale', '$0'))
                volume = self._parse_number(row.get('volume', '0'))
                change_pct = self._parse_number(row.get('pctchange', '0%'))

                # Filtrer les penny stocks et faible volume
                if price < 1 or market_cap < 50_000_000:
                    continue

                stock = StockInfo(
                    symbol=symbol,
                    name=row.get('name', symbol)[:100],
                    exchange=exchange,
                    sector=row.get('sector', 'Unknown') or 'Unknown',
                    industry=row.get('industry', 'Unknown') or 'Unknown',
                    market_cap=market_cap,
                    avg_volume=volume,
                    price=price,
                    change_pct=change_pct
                )

                stocks.append(stock)

            except Exception as e:
                logger.debug(f"Error parsing row: {e}")
                continue

        return stocks

    def _parse_market_cap(self, value: str) -> float:
        """Parse une valeur de market cap (ex: '1.5B', '500M')"""
        if not value or value == 'N/A':
            return 0

        value = str(value).upper().replace(',', '').replace('$', '').strip()

        multipliers = {'T': 1e12, 'B': 1e9, 'M': 1e6, 'K': 1e3}

        for suffix, mult in multipliers.items():
            if suffix in value:
                try:
                    return float(value.replace(suffix, '')) * mult
                except ValueError:
                    return 0

        try:
            return float(value)
        except ValueError:
            return 0

    def _parse_number(self, value: str) -> float:
        """Parse un nombre avec symboles ($, %, etc.)"""
        if not value or value == 'N/A':
            return 0

        # Nettoyer
        value = str(value).replace('$', '').replace('%', '').replace(',', '').strip()

        try:
            return float(value)
        except ValueError:
            return 0

    async def _save_universe_cache(self):
        """Sauvegarde l'univers en cache"""
        cache_file = self._data_dir / "universe_cache.json"

        try:
            data = {
                'updated': self._universe_updated.isoformat(),
                'stocks': [s.to_dict() for s in self._universe.values()]
            }

            with open(cache_file, 'w') as f:
                json.dump(data, f)

            logger.info(f"Saved {len(self._universe)} stocks to cache")

        except Exception as e:
            logger.error(f"Error saving universe cache: {e}")

    def update_social_scores(self, social_data: Dict[str, Dict]):
        """
        Met à jour les scores sociaux depuis le social scanner.

        Args:
            social_data: Dict {symbol: {mentions, sentiment, ...}}
        """
        self._social_data = social_data

        for symbol, data in social_data.items():
            if symbol in self._universe:
                # Calculer le score social
                mentions = data.get('mention_count', 0)
                sentiment = data.get('avg_sentiment', 0)

                # Score = mentions normalisées + boost sentiment
                mention_score = min(mentions / 100, 1.0)  # Cap à 100 mentions
                sentiment_boost = 0.3 if sentiment > 0.3 else (0 if sentiment > -0.3 else -0.3)

                self._universe[symbol].social_score = mention_score + sentiment_boost

    async def filter_universe(
        self,
        min_volume: float = 500_000,
        min_mcap: float = 300_000_000,
        max_mcap: Optional[float] = None,
        min_price: float = 1.0,
        max_price: Optional[float] = None,
        sectors: Optional[List[str]] = None,
        exchanges: Optional[List[str]] = None,
        min_social_score: float = 0.0
    ) -> List[StockInfo]:
        """
        Filtre l'univers selon les critères.

        Args:
            min_volume: Volume moyen minimum
            min_mcap: Market cap minimum (USD)
            max_mcap: Market cap maximum (optionnel)
            min_price: Prix minimum
            max_price: Prix maximum (optionnel)
            sectors: Secteurs à inclure (ou None pour tous)
            exchanges: Exchanges à inclure
            min_social_score: Score social minimum

        Returns:
            Liste de stocks filtrés
        """
        filtered = []

        for stock in self._universe.values():
            # Filtres de base
            if stock.avg_volume < min_volume:
                continue
            if stock.market_cap < min_mcap:
                continue
            if max_mcap and stock.market_cap > max_mcap:
                continue
            if stock.price < min_price:
                continue
            if max_price and stock.price > max_price:
                continue

            # Filtre secteur
            if sectors and stock.sector not in sectors:
                continue

            # Filtre exchange
            if exchanges and stock.exchange not in exchanges:
                continue

            # Filtre social
            if stock.social_score < min_social_score:
                continue

            filtered.append(stock)

        return filtered

    async def discover_trending(
        self,
        social_trending: Optional[List[str]] = None,
        grok_trending: Optional[List[str]] = None,
        min_mentions: int = 3,
        limit: int = 50
    ) -> List[StockInfo]:
        """
        Découvre les stocks trending basés sur les signaux sociaux.

        Args:
            social_trending: Symboles trending du social scanner
            grok_trending: Symboles trending de Grok
            min_mentions: Mentions minimum pour être considéré trending
            limit: Limite de résultats

        Returns:
            Liste de stocks trending
        """
        # Combiner les sources
        trending_symbols = set()

        if social_trending:
            trending_symbols.update(social_trending)

        if grok_trending:
            trending_symbols.update(grok_trending)

        # Ajouter ceux avec haut score social
        for symbol, stock in self._universe.items():
            if stock.social_score >= 0.5:
                trending_symbols.add(symbol)

        # Filtrer et enrichir
        trending = []
        for symbol in trending_symbols:
            if symbol in self._universe:
                stock = self._universe[symbol]
                # Vérifier les critères de base
                if stock.market_cap >= 100_000_000 and stock.price >= 1:
                    trending.append(stock)

        # Trier par score social puis market cap
        trending.sort(key=lambda s: (s.social_score, s.market_cap), reverse=True)

        return trending[:limit]

    async def get_screening_candidates(
        self,
        max_candidates: int = 200,
        include_always: bool = True,
        include_trending: bool = True,
        include_filtered: bool = True,
        social_data: Optional[Dict] = None,
        grok_data: Optional[Dict] = None
    ) -> DiscoveryResult:
        """
        Obtient la liste des candidats pour le screening.

        Combine plusieurs sources:
        1. Toujours inclus (mega caps référence)
        2. Trending des réseaux sociaux
        3. Filtrés par critères techniques

        Args:
            max_candidates: Nombre max de candidats
            include_always: Inclure les mega caps de référence
            include_trending: Inclure les trending sociaux
            include_filtered: Inclure les filtrés par volume/mcap
            social_data: Données du social scanner
            grok_data: Données du Grok scanner

        Returns:
            DiscoveryResult avec la liste des candidats
        """
        candidates: Dict[str, StockInfo] = {}
        sources_used = []
        trending_symbols = []

        # Mettre à jour les scores sociaux si fournis
        if social_data:
            self.update_social_scores(social_data)

        # 1. Toujours inclus (mega caps)
        if include_always:
            for symbol in self.ALWAYS_INCLUDE:
                if symbol in self._universe:
                    candidates[symbol] = self._universe[symbol]
            sources_used.append('always_include')

        # 2. Trending sociaux
        if include_trending:
            social_trending = []
            grok_trending = []

            if social_data:
                social_trending = [
                    s for s, d in social_data.items()
                    if d.get('mention_count', 0) >= 3
                ]

            if grok_data and 'trending_symbols' in grok_data:
                grok_trending = list(grok_data['trending_symbols'].keys())

            trending = await self.discover_trending(
                social_trending=social_trending,
                grok_trending=grok_trending,
                limit=50
            )

            for stock in trending:
                candidates[stock.symbol] = stock
                trending_symbols.append(stock.symbol)

            sources_used.append('social_trending')

        # 3. Filtrés par critères techniques
        if include_filtered:
            # Priorité aux secteurs d'intérêt
            priority_filtered = await self.filter_universe(
                min_volume=1_000_000,
                min_mcap=1_000_000_000,
                sectors=self.PRIORITY_SECTORS
            )

            # Ajouter les meilleurs
            for stock in priority_filtered[:100]:
                if stock.symbol not in candidates:
                    candidates[stock.symbol] = stock

            sources_used.append('filtered_universe')

        # Convertir en liste et trier
        candidate_list = list(candidates.values())

        # Trier par: social_score DESC, market_cap DESC
        candidate_list.sort(
            key=lambda s: (s.social_score * 1e12 + s.market_cap),
            reverse=True
        )

        # Limiter
        candidate_list = candidate_list[:max_candidates]

        # Créer le résultat
        result = DiscoveryResult(
            timestamp=datetime.now(),
            total_universe=len(self._universe),
            filtered_count=len(candidate_list),
            trending_symbols=trending_symbols,
            candidates=candidate_list,
            filters_applied={
                'max_candidates': max_candidates,
                'include_always': include_always,
                'include_trending': include_trending,
                'include_filtered': include_filtered
            },
            sources_used=sources_used
        )

        self._last_candidates = candidate_list
        return result

    async def get_sector_stocks(
        self,
        sector: str,
        limit: int = 50
    ) -> List[StockInfo]:
        """
        Obtient les stocks d'un secteur spécifique.

        Args:
            sector: Nom du secteur
            limit: Limite de résultats

        Returns:
            Liste de stocks du secteur
        """
        stocks = [
            s for s in self._universe.values()
            if s.sector == sector and s.market_cap >= 500_000_000
        ]

        # Trier par market cap
        stocks.sort(key=lambda s: s.market_cap, reverse=True)

        return stocks[:limit]

    async def search_by_industry(
        self,
        industry_keywords: List[str],
        limit: int = 30
    ) -> List[StockInfo]:
        """
        Recherche des stocks par mots-clés dans l'industrie.

        Args:
            industry_keywords: Mots-clés (ex: ['AI', 'semiconductor'])
            limit: Limite de résultats

        Returns:
            Liste de stocks correspondants
        """
        matches = []

        for stock in self._universe.values():
            industry_lower = stock.industry.lower()
            name_lower = stock.name.lower()

            for keyword in industry_keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in industry_lower or keyword_lower in name_lower:
                    matches.append(stock)
                    break

        # Trier par market cap
        matches.sort(key=lambda s: s.market_cap, reverse=True)

        return matches[:limit]

    def get_symbol_info(self, symbol: str) -> Optional[StockInfo]:
        """Obtient les infos d'un symbole"""
        return self._universe.get(symbol)

    def get_all_symbols(self) -> List[str]:
        """Retourne tous les symboles de l'univers"""
        return list(self._universe.keys())

    def get_universe_stats(self) -> Dict:
        """Retourne des statistiques sur l'univers"""
        if not self._universe:
            return {'total': 0}

        # Compter par exchange
        by_exchange = Counter(s.exchange for s in self._universe.values())

        # Compter par secteur
        by_sector = Counter(s.sector for s in self._universe.values())

        # Compter par market cap
        by_mcap = Counter(s.market_cap_category for s in self._universe.values())

        return {
            'total': len(self._universe),
            'updated': self._universe_updated.isoformat() if self._universe_updated else None,
            'by_exchange': dict(by_exchange),
            'by_sector': dict(by_sector.most_common(10)),
            'by_market_cap': dict(by_mcap)
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_stock_discovery_instance: Optional[StockDiscovery] = None


async def get_stock_discovery() -> StockDiscovery:
    """
    Factory pour obtenir une instance de StockDiscovery.

    Singleton pattern.
    """
    global _stock_discovery_instance

    if _stock_discovery_instance is None:
        _stock_discovery_instance = StockDiscovery()
        await _stock_discovery_instance.initialize()

    return _stock_discovery_instance


async def close_stock_discovery():
    """Ferme l'instance globale"""
    global _stock_discovery_instance

    if _stock_discovery_instance:
        await _stock_discovery_instance.close()
        _stock_discovery_instance = None


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        print("=== Stock Discovery Test ===\n")

        discovery = StockDiscovery()
        await discovery.initialize()

        try:
            # Stats de l'univers
            stats = discovery.get_universe_stats()
            print(f"Universe stats:")
            print(f"  Total stocks: {stats['total']}")
            print(f"  Updated: {stats.get('updated', 'Never')}")
            print(f"  By exchange: {stats.get('by_exchange', {})}")
            print(f"  By market cap: {stats.get('by_market_cap', {})}")

            # Refresh si nécessaire
            if stats['total'] < 100:
                print("\nRefreshing universe from NASDAQ API...")
                count = await discovery.refresh_universe(force=True)
                print(f"Refreshed: {count} stocks")

            # Obtenir des candidats
            print("\nGetting screening candidates...")
            result = await discovery.get_screening_candidates(max_candidates=50)

            print(f"\nDiscovery Result:")
            print(f"  Total universe: {result.total_universe}")
            print(f"  Filtered candidates: {result.filtered_count}")
            print(f"  Sources used: {result.sources_used}")

            print("\nTop 10 candidates:")
            for stock in result.candidates[:10]:
                print(f"  {stock.symbol}: {stock.name[:30]} | "
                      f"MCap: ${stock.market_cap/1e9:.1f}B | "
                      f"Sector: {stock.sector}")

            # Recherche par industrie
            print("\nSearching for AI/semiconductor stocks...")
            ai_stocks = await discovery.search_by_industry(['AI', 'semiconductor', 'artificial'])
            print(f"Found {len(ai_stocks)} AI-related stocks:")
            for stock in ai_stocks[:5]:
                print(f"  {stock.symbol}: {stock.industry}")

        finally:
            await discovery.close()

    asyncio.run(main())
