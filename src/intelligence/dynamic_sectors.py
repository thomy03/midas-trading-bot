"""
Dynamic Sectors Module
Génération dynamique des secteurs et de leurs composants.

Au lieu de secteurs hardcodés, ce module:
1. Récupère les secteurs depuis l'API NASDAQ
2. Utilise le LLM pour identifier les thèmes émergents
3. Met à jour automatiquement la composition des secteurs
4. Garde un cache avec TTL pour la performance
"""

import os
import json
import asyncio
import aiohttp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SectorInfo:
    """Information sur un secteur"""
    name: str
    display_name: str
    symbols: List[str]
    etf: Optional[str] = None  # ETF représentatif (XLK, XLV, etc.)
    momentum_5d: float = 0.0
    momentum_20d: float = 0.0
    avg_volume_ratio: float = 1.0
    sentiment: float = 0.0
    last_updated: Optional[datetime] = None
    source: str = "api"  # "api", "llm", "hardcoded"

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "symbols": self.symbols,
            "etf": self.etf,
            "momentum_5d": self.momentum_5d,
            "momentum_20d": self.momentum_20d,
            "avg_volume_ratio": self.avg_volume_ratio,
            "sentiment": self.sentiment,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "source": self.source
        }


class DynamicSectors:
    """
    Gestionnaire de secteurs dynamiques.

    Sources:
    1. API NASDAQ - Liste des secteurs et leurs composants
    2. Yahoo Finance - Données sectorielles
    3. LLM (Gemini) - Thèmes émergents et nouveaux secteurs
    4. Fallback - Secteurs hardcodés si APIs indisponibles
    """

    # ETFs sectoriels pour référence
    SECTOR_ETFS = {
        "Technology": "XLK",
        "Health Care": "XLV",
        "Financials": "XLF",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Energy": "XLE",
        "Industrials": "XLI",
        "Materials": "XLB",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Communication Services": "XLC",
    }

    # Mapping des secteurs NASDAQ vers nos noms internes
    SECTOR_MAPPING = {
        "Technology": ["Technology", "Information Technology"],
        "Healthcare": ["Health Care", "Healthcare"],
        "Finance": ["Financial Services", "Financials"],
        "Consumer": ["Consumer Discretionary", "Consumer Cyclical"],
        "Energy": ["Energy"],
        "Industrials": ["Industrials"],
        "Communications": ["Communication Services"],
        "RealEstate": ["Real Estate"],
        "Utilities": ["Utilities"],
        "Materials": ["Basic Materials", "Materials"],
    }

    # Thèmes spéciaux (cross-sectoriels) - initialisés dynamiquement
    SPECIAL_THEMES = [
        "AI_Semiconductors",
        "Clean_Energy",
        "Biotech",
        "Crypto_Adjacent",
        "Defense",
        "Space",
        "Obesity_Drugs",
        "Quantum_Computing",
    ]

    # Fallback hardcodé (utilisé uniquement si tout échoue)
    FALLBACK_SECTORS = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "CRM", "ADBE"],
        "AI_Semiconductors": ["NVDA", "AMD", "AVGO", "QCOM", "INTC", "ARM", "MRVL", "MU"],
        "Healthcare": ["UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "DHR"],
        "Finance": ["JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD", "MPC", "VLO"],
    }

    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        llm_model: str = "google/gemini-3-flash-preview",
        cache_ttl_hours: int = 24,
        data_dir: str = "data/sectors"
    ):
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.llm_model = llm_model
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.session: Optional[aiohttp.ClientSession] = None
        self._sectors: Dict[str, SectorInfo] = {}
        self._last_refresh: Optional[datetime] = None

    async def initialize(self):
        """Initialise le module"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(timeout=timeout)

        # Charger le cache
        await self._load_cache()

        # Rafraîchir si nécessaire
        if self._needs_refresh():
            await self.refresh_sectors()

        logger.info(f"DynamicSectors initialized: {len(self._sectors)} sectors")

    async def close(self):
        """Ferme les connexions"""
        if self.session:
            await self.session.close()
            self.session = None

    def _needs_refresh(self) -> bool:
        """Vérifie si les secteurs doivent être rafraîchis"""
        if not self._last_refresh:
            return True
        return datetime.now() - self._last_refresh > self.cache_ttl

    async def _load_cache(self):
        """Charge les secteurs depuis le cache"""
        cache_file = self.data_dir / "sectors_cache.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self._last_refresh = datetime.fromisoformat(data.get("updated", "2000-01-01"))

                for name, sector_data in data.get("sectors", {}).items():
                    self._sectors[name] = SectorInfo(
                        name=name,
                        display_name=sector_data.get("display_name", name),
                        symbols=sector_data.get("symbols", []),
                        etf=sector_data.get("etf"),
                        source=sector_data.get("source", "cache")
                    )

                logger.info(f"Loaded {len(self._sectors)} sectors from cache")

            except Exception as e:
                logger.warning(f"Error loading sector cache: {e}")

    async def _save_cache(self):
        """Sauvegarde les secteurs dans le cache"""
        cache_file = self.data_dir / "sectors_cache.json"

        try:
            data = {
                "updated": datetime.now().isoformat(),
                "sectors": {name: s.to_dict() for name, s in self._sectors.items()}
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self._sectors)} sectors to cache")

        except Exception as e:
            logger.error(f"Error saving sector cache: {e}")

    async def refresh_sectors(self, force: bool = False) -> int:
        """
        Rafraîchit les secteurs depuis les APIs.

        Args:
            force: Forcer le refresh même si le cache est valide

        Returns:
            Nombre de secteurs
        """
        if not force and not self._needs_refresh():
            return len(self._sectors)

        logger.info("Refreshing sectors from APIs...")
        new_sectors: Dict[str, SectorInfo] = {}

        # 1. Récupérer depuis l'API NASDAQ
        try:
            nasdaq_sectors = await self._fetch_nasdaq_sectors()
            for name, symbols in nasdaq_sectors.items():
                if symbols:
                    new_sectors[name] = SectorInfo(
                        name=name,
                        display_name=name.replace("_", " "),
                        symbols=symbols[:50],  # Top 50 par market cap
                        etf=self.SECTOR_ETFS.get(name),
                        source="nasdaq_api"
                    )
            logger.info(f"Fetched {len(nasdaq_sectors)} sectors from NASDAQ API")
        except Exception as e:
            logger.warning(f"NASDAQ API failed: {e}")

        # 2. Identifier les thèmes émergents avec LLM
        if self.openrouter_api_key:
            try:
                llm_themes = await self._identify_emerging_themes()
                for theme_name, symbols in llm_themes.items():
                    if symbols and theme_name not in new_sectors:
                        new_sectors[theme_name] = SectorInfo(
                            name=theme_name,
                            display_name=theme_name.replace("_", " "),
                            symbols=symbols,
                            source="llm"
                        )
                logger.info(f"LLM identified {len(llm_themes)} emerging themes")
            except Exception as e:
                logger.warning(f"LLM theme identification failed: {e}")

        # 3. Fallback si peu de secteurs
        if len(new_sectors) < 5:
            logger.warning("Using fallback hardcoded sectors")
            for name, symbols in self.FALLBACK_SECTORS.items():
                if name not in new_sectors:
                    new_sectors[name] = SectorInfo(
                        name=name,
                        display_name=name.replace("_", " "),
                        symbols=symbols,
                        source="hardcoded"
                    )

        # Mettre à jour
        self._sectors = new_sectors
        self._last_refresh = datetime.now()

        # Sauvegarder le cache
        await self._save_cache()

        return len(self._sectors)

    async def _fetch_nasdaq_sectors(self) -> Dict[str, List[str]]:
        """
        Récupère les secteurs depuis l'API NASDAQ.

        Returns:
            Dict {sector_name: [symbols]}
        """
        sectors: Dict[str, List[str]] = {}

        url = "https://api.nasdaq.com/api/screener/stocks"
        params = {
            "tableonly": "true",
            "limit": 5000,
            "exchange": "NASDAQ,NYSE"
        }

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status != 200:
                    logger.warning(f"NASDAQ API returned {response.status}")
                    return sectors

                data = await response.json()
                rows = data.get("data", {}).get("table", {}).get("rows", [])

                for row in rows:
                    sector = row.get("sector", "").strip()
                    symbol = row.get("symbol", "").strip()
                    market_cap = row.get("marketCap", "")

                    if not sector or not symbol or len(symbol) > 5:
                        continue

                    # Normaliser le nom du secteur
                    normalized = self._normalize_sector_name(sector)
                    if normalized:
                        if normalized not in sectors:
                            sectors[normalized] = []
                        sectors[normalized].append(symbol)

        except Exception as e:
            logger.error(f"Error fetching NASDAQ sectors: {e}")

        return sectors

    def _normalize_sector_name(self, raw_sector: str) -> Optional[str]:
        """Normalise le nom d'un secteur"""
        raw_lower = raw_sector.lower()

        for normalized_name, aliases in self.SECTOR_MAPPING.items():
            for alias in aliases:
                if alias.lower() == raw_lower:
                    return normalized_name

        # Retourner tel quel si pas de mapping
        if raw_sector and len(raw_sector) > 2:
            return raw_sector.replace(" ", "_")

        return None

    async def _identify_emerging_themes(self) -> Dict[str, List[str]]:
        """
        Utilise le LLM pour identifier les thèmes émergents.

        Returns:
            Dict {theme_name: [symbols]}
        """
        themes: Dict[str, List[str]] = {}

        if not self.openrouter_api_key:
            return themes

        prompt = """You are a financial analyst. Identify the current HOT investment themes
and the top 10 stocks for each theme. Focus on themes that are trending NOW.

Return ONLY valid JSON in this exact format:
{
    "AI_Semiconductors": ["NVDA", "AMD", "AVGO", "ARM", "MRVL", "QCOM", "INTC", "MU", "AMAT", "LRCX"],
    "Obesity_Drugs": ["LLY", "NVO", "AMGN", "VKTX", "GPCR"],
    "Space_Economy": ["RKLB", "LUNR", "RDW", "ASTS", "SPCE"],
    "Nuclear_Renaissance": ["CEG", "VST", "CCJ", "SMR", "NNE"],
    "Quantum_Computing": ["IONQ", "RGTI", "QBTS", "QUBT"],
    "Crypto_Adjacent": ["COIN", "MSTR", "RIOT", "MARA", "HUT", "CLSK"],
    "Defense_Tech": ["PLTR", "LMT", "RTX", "NOC", "GD", "LHX"],
    "Clean_Energy": ["ENPH", "FSLR", "SEDG", "RUN", "NEE", "AES"]
}

Only include themes with at least 5 investable stocks. Use ONLY US-listed tickers."""

        try:
            async with self.session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1000
                }
            ) as response:
                if response.status != 200:
                    logger.warning(f"OpenRouter returned {response.status}")
                    return themes

                data = await response.json()
                content = data["choices"][0]["message"]["content"]

                # Parser le JSON
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    themes = json.loads(json_match.group())

        except Exception as e:
            logger.error(f"Error identifying themes with LLM: {e}")

        return themes

    def get_sector(self, name: str) -> Optional[SectorInfo]:
        """Retourne un secteur par son nom"""
        return self._sectors.get(name)

    def get_all_sectors(self) -> Dict[str, SectorInfo]:
        """Retourne tous les secteurs"""
        return self._sectors.copy()

    def get_sector_symbols(self, name: str) -> List[str]:
        """Retourne les symboles d'un secteur"""
        sector = self._sectors.get(name)
        return sector.symbols if sector else []

    def get_all_symbols(self) -> Set[str]:
        """Retourne tous les symboles uniques de tous les secteurs"""
        all_symbols = set()
        for sector in self._sectors.values():
            all_symbols.update(sector.symbols)
        return all_symbols

    def get_sectors_for_symbol(self, symbol: str) -> List[str]:
        """Retourne les secteurs contenant un symbole"""
        return [
            name for name, sector in self._sectors.items()
            if symbol in sector.symbols
        ]

    async def get_hot_sectors(self, limit: int = 5) -> List[SectorInfo]:
        """
        Retourne les secteurs les plus "chauds" (momentum + sentiment).

        Args:
            limit: Nombre de secteurs à retourner

        Returns:
            Liste triée par score
        """
        # Calculer un score composite
        scored = []
        for sector in self._sectors.values():
            score = (
                sector.momentum_5d * 0.4 +
                sector.sentiment * 0.3 +
                (sector.avg_volume_ratio - 1) * 0.3
            )
            scored.append((score, sector))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:limit]]

    def get_stats(self) -> Dict:
        """Retourne des statistiques sur les secteurs"""
        sources = {}
        for sector in self._sectors.values():
            sources[sector.source] = sources.get(sector.source, 0) + 1

        return {
            "total_sectors": len(self._sectors),
            "total_symbols": len(self.get_all_symbols()),
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "by_source": sources,
            "sectors": list(self._sectors.keys())
        }


# Singleton
_dynamic_sectors: Optional[DynamicSectors] = None


async def get_dynamic_sectors() -> DynamicSectors:
    """Retourne l'instance singleton"""
    global _dynamic_sectors
    if _dynamic_sectors is None:
        _dynamic_sectors = DynamicSectors()
        await _dynamic_sectors.initialize()
    return _dynamic_sectors


# CLI Test
if __name__ == "__main__":
    async def main():
        print("=== Dynamic Sectors Test ===\n")

        ds = DynamicSectors()
        await ds.initialize()

        try:
            # Stats
            stats = ds.get_stats()
            print(f"Stats: {json.dumps(stats, indent=2)}")

            # Force refresh
            print("\nRefreshing sectors...")
            count = await ds.refresh_sectors(force=True)
            print(f"Refreshed: {count} sectors")

            # Afficher les secteurs
            print("\nSectors:")
            for name, sector in ds.get_all_sectors().items():
                print(f"  {name}: {len(sector.symbols)} stocks (source: {sector.source})")
                print(f"    Top 5: {', '.join(sector.symbols[:5])}")

        finally:
            await ds.close()

    asyncio.run(main())
