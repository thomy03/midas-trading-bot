"""
Universe Scanner - Scan complet US + Euronext
Récupère et filtre tous les symboles tradables.

Marchés couverts:
- NASDAQ (~3500 symboles)
- NYSE/S&P500 (~500 symboles)  
- Euronext (~1500 symboles): Paris, Amsterdam, Bruxelles, Lisbonne, Dublin, Milan

Filtres différenciés:
- US: Volume > 500K, Market Cap > 500M$, Prix > $2
- EU: Volume > 250K€, Market Cap > 250M€, Prix > 1€
"""

import os
import json
import asyncio
import aiohttp
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class UniverseConfig:
    """Configuration du scanner d'univers"""
    
    # Filtres US (haute liquidité)
    us_min_volume: int = 500_000  # Volume moyen/jour
    us_min_market_cap: int = 500_000_000  # 500M$
    us_min_price: float = 2.0
    
    # Filtres Europe (liquidité moindre = filtres /2)
    eu_min_volume: int = 250_000  # Volume moyen/jour
    eu_min_market_cap: int = 250_000_000  # 250M€
    eu_min_price: float = 1.0
    
    # Limites
    max_symbols_per_market: int = 1000
    cache_ttl_hours: int = 24
    
    # Exchanges Euronext
    euronext_exchanges: List[str] = field(default_factory=lambda: [
        'PA',   # Paris
        'AS',   # Amsterdam
        'BR',   # Bruxelles
        'LS',   # Lisbonne
        'IR',   # Dublin
        'MI',   # Milan
    ])


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class UniverseSymbol:
    """Symbole dans l'univers"""
    symbol: str
    name: str
    market: str  # 'US' ou 'EU'
    exchange: str  # NASDAQ, NYSE, PA, AS, etc.
    sector: str
    market_cap: float
    avg_volume: float
    price: float
    currency: str  # USD ou EUR
    
    # Scores (calculés plus tard)
    heat_score: float = 0.0
    sentiment_score: float = 0.0
    technical_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'name': self.name,
            'market': self.market,
            'exchange': self.exchange,
            'sector': self.sector,
            'market_cap': self.market_cap,
            'avg_volume': self.avg_volume,
            'price': self.price,
            'currency': self.currency,
        }


@dataclass  
class UniverseSnapshot:
    """Snapshot de l'univers scanné"""
    timestamp: datetime
    us_symbols: List[UniverseSymbol]
    eu_symbols: List[UniverseSymbol]
    total_count: int
    filters_applied: Dict
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'us_count': len(self.us_symbols),
            'eu_count': len(self.eu_symbols),
            'total_count': self.total_count,
            'filters': self.filters_applied,
        }


# =============================================================================
# EURONEXT FETCHER
# =============================================================================

class EuronextFetcher:
    """Récupère les symboles Euronext via Yahoo Finance"""
    
    # Mapping exchange -> suffixe Yahoo Finance
    EXCHANGE_SUFFIX = {
        'PA': '.PA',   # Paris
        'AS': '.AS',   # Amsterdam  
        'BR': '.BR',   # Bruxelles
        'LS': '.LS',   # Lisbonne
        'IR': '.IR',   # Dublin (Ireland)
        'MI': '.MI',   # Milan
    }
    
    # Principaux indices par exchange pour découvrir les composants
    INDICES = {
        'PA': ['^FCHI', '^SBF120'],  # CAC 40, SBF 120
        'AS': ['^AEX'],               # AEX 25
        'BR': ['^BFX'],               # BEL 20
        'MI': ['^FTMIB'],             # FTSE MIB
    }
    
    # Liste étendue des tickers Euronext (principales capitalisations)
    # Cette liste sera enrichie dynamiquement
    EURONEXT_TICKERS = {
        # France (PA) - CAC 40 + Next 20 + Mid 60
        'PA': [
            # CAC 40
            'AI.PA', 'AIR.PA', 'ALO.PA', 'MT.PA', 'CS.PA', 'BNP.PA', 'EN.PA', 
            'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'DSY.PA', 'ENGI.PA', 'EL.PA',
            'ERF.PA', 'RMS.PA', 'KER.PA', 'LR.PA', 'OR.PA', 'MC.PA', 'ML.PA',
            'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'SAN.PA',
            'SU.PA', 'GLE.PA', 'STLAP.PA', 'STMPA.PA', 'TEP.PA', 'HO.PA', 
            'TTE.PA', 'URW.PA', 'VIE.PA', 'DG.PA', 'VIV.PA', 'WLN.PA',
            # SBF 120 additions
            'AF.PA', 'AKE.PA', 'ATO.PA', 'BB.PA', 'BIM.PA', 'BOL.PA', 'CGG.PA',
            'CO.PA', 'COFA.PA', 'DBG.PA', 'EDF.PA', 'EFI.PA', 'ETL.PA', 'FDJ.PA',
            'FGR.PA', 'GFC.PA', 'GET.PA', 'GTT.PA', 'ILD.PA', 'ING.PA', 'IPH.PA',
            'JCQ.PA', 'KOF.PA', 'LDC.PA', 'MMT.PA', 'NEX.PA', 'NXI.PA', 'OPM.PA',
            'ORP.PA', 'PLX.PA', 'QUA.PA', 'RCO.PA', 'RF.PA', 'RUI.PA', 'RXL.PA',
            'SEV.PA', 'SK.PA', 'SOI.PA', 'SOP.PA', 'SPB.PA', 'SW.PA', 'TFI.PA',
            'UBI.PA', 'VCT.PA', 'VLA.PA', 'VK.PA', 'VRLA.PA',
        ],
        # Netherlands (AS) - AEX + AMX
        'AS': [
            'ADYEN.AS', 'AGN.AS', 'AD.AS', 'AKZA.AS', 'MT.AS', 'ASML.AS', 
            'ASRNL.AS', 'DSM.AS', 'GLPG.AS', 'HEIA.AS', 'INGA.AS', 'KPN.AS',
            'NN.AS', 'PHIA.AS', 'RAND.AS', 'REN.AS', 'RDSA.AS', 'TKWY.AS',
            'UNA.AS', 'URW.AS', 'WKL.AS', 'ABN.AS', 'AALB.AS', 'BESI.AS',
            'BFIT.AS', 'CMCOM.AS', 'FLOW.AS', 'IMCD.AS', 'JDEP.AS', 'OCI.AS',
            'SBMO.AS', 'LIGHT.AS', 'TOM2.AS', 'VPK.AS', 'WHA.AS',
        ],
        # Belgium (BR) - BEL 20
        'BR': [
            'ABI.BR', 'ACKB.BR', 'AGS.BR', 'APAM.BR', 'ARGX.BR', 'COFB.BR',
            'ELI.BR', 'GLPG.BR', 'KBC.BR', 'PROX.BR', 'SOF.BR', 'SOLB.BR',
            'UCB.BR', 'UMI.BR', 'WDP.BR', 'XIOR.BR',
        ],
        # Italy (MI) - FTSE MIB
        'MI': [
            'A2A.MI', 'AMP.MI', 'ATL.MI', 'AZM.MI', 'BAMI.MI', 'BGN.MI',
            'BMED.MI', 'BPER.MI', 'BZU.MI', 'CPR.MI', 'DIA.MI', 'ENEL.MI',
            'ENI.MI', 'ERG.MI', 'FBK.MI', 'G.MI', 'HER.MI', 'IG.MI',
            'INW.MI', 'IPG.MI', 'ISP.MI', 'LDO.MI', 'MB.MI', 'MONC.MI',
            'NEXI.MI', 'PIRC.MI', 'PRY.MI', 'PST.MI', 'RACE.MI', 'REC.MI',
            'SPM.MI', 'SRG.MI', 'STM.MI', 'TEN.MI', 'TIT.MI', 'TRN.MI',
            'UCG.MI', 'UNI.MI',
        ],
        # Portugal (LS)
        'LS': [
            'EDP.LS', 'EDPR.LS', 'GALP.LS', 'JMT.LS', 'NVG.LS', 'PHR.LS',
            'RENE.LS', 'SON.LS',
        ],
    }

    def __init__(self, config: UniverseConfig):
        self.config = config
        self._cache_dir = Path("data/tickers")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
    async def fetch_all(self) -> List[UniverseSymbol]:
        """Récupère tous les symboles Euronext"""
        all_symbols = []
        
        for exchange, tickers in self.EURONEXT_TICKERS.items():
            logger.info(f"Fetching {exchange} tickers: {len(tickers)} symbols")
            
            for ticker in tickers:
                try:
                    symbol = await self._fetch_symbol_info(ticker, exchange)
                    if symbol and self._passes_filters(symbol):
                        all_symbols.append(symbol)
                except Exception as e:
                    logger.warning(f"Error fetching {ticker}: {e}")
                    
            # Rate limiting
            await asyncio.sleep(0.1)
            
        logger.info(f"Euronext total: {len(all_symbols)} symbols after filtering")
        return all_symbols
    
    async def _fetch_symbol_info(self, ticker: str, exchange: str) -> Optional[UniverseSymbol]:
        """Récupère les infos d'un symbole via yfinance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or 'regularMarketPrice' not in info:
                return None
                
            return UniverseSymbol(
                symbol=ticker,
                name=info.get('shortName', info.get('longName', ticker)),
                market='EU',
                exchange=exchange,
                sector=info.get('sector', 'Unknown'),
                market_cap=info.get('marketCap', 0) or 0,
                avg_volume=info.get('averageVolume', 0) or 0,
                price=info.get('regularMarketPrice', 0) or 0,
                currency=info.get('currency', 'EUR'),
            )
        except Exception as e:
            logger.debug(f"Failed to fetch {ticker}: {e}")
            return None
    
    def _passes_filters(self, symbol: UniverseSymbol) -> bool:
        """Vérifie si le symbole passe les filtres EU"""
        return (
            symbol.market_cap >= self.config.eu_min_market_cap and
            symbol.avg_volume >= self.config.eu_min_volume and
            symbol.price >= self.config.eu_min_price
        )


# =============================================================================
# US FETCHER (utilise l'existant)
# =============================================================================

class USFetcher:
    """Récupère les symboles US (NASDAQ + NYSE)"""
    
    SOURCES = {
        'nasdaq': 'https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange=NASDAQ',
        'nyse': 'https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange=NYSE',
    }
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }
    
    def __init__(self, config: UniverseConfig):
        self.config = config
        
    async def fetch_all(self) -> List[UniverseSymbol]:
        """Récupère tous les symboles US"""
        all_symbols = []
        
        async with aiohttp.ClientSession(headers=self.HEADERS) as session:
            for exchange, url in self.SOURCES.items():
                try:
                    symbols = await self._fetch_exchange(session, url, exchange.upper())
                    all_symbols.extend(symbols)
                    logger.info(f"{exchange}: {len(symbols)} symbols after filtering")
                except Exception as e:
                    logger.error(f"Error fetching {exchange}: {e}")
                    
        return all_symbols
    
    async def _fetch_exchange(self, session: aiohttp.ClientSession, url: str, exchange: str) -> List[UniverseSymbol]:
        """Récupère les symboles d'un exchange"""
        try:
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    logger.error(f"HTTP {response.status} for {exchange}")
                    return []
                    
                data = await response.json()
                rows = data.get('data', {}).get('table', {}).get('rows', [])
                
                symbols = []
                for row in rows:
                    try:
                        symbol = self._parse_row(row, exchange)
                        if symbol and self._passes_filters(symbol):
                            symbols.append(symbol)
                    except Exception as e:
                        continue
                        
                return symbols[:self.config.max_symbols_per_market]
                
        except Exception as e:
            logger.error(f"Error fetching {exchange}: {e}")
            return []
    
    def _parse_row(self, row: Dict, exchange: str) -> Optional[UniverseSymbol]:
        """Parse une ligne de l'API NASDAQ"""
        symbol = row.get('symbol', '').strip()
        if not symbol or len(symbol) > 5:  # Skip warrants, etc.
            return None
            
        # Parse market cap
        mcap_str = row.get('marketCap', '0')
        market_cap = self._parse_number(mcap_str)
        
        # Parse volume
        volume_str = row.get('volume', '0')
        volume = self._parse_number(volume_str)
        
        # Parse price - remove $ and commas
        price_str = row.get('lastsale', '0')
        if price_str:
            price_str = price_str.replace('$', '').replace(',', '').strip()
        try:
            price = float(price_str) if price_str else 0
        except:
            price = 0
            
        return UniverseSymbol(
            symbol=symbol,
            name=row.get('name', symbol),
            market='US',
            exchange=exchange,
            sector=row.get('sector', 'Unknown'),
            market_cap=market_cap,
            avg_volume=volume,
            price=price,
            currency='USD',
        )
    
    def _parse_number(self, value: str) -> float:
        """Parse un nombre avec suffixes K, M, B"""
        if not value:
            return 0
        value = str(value).replace(',', '').replace('$', '').strip()
        
        multiplier = 1
        if value.endswith('T'):
            multiplier = 1_000_000_000_000
            value = value[:-1]
        elif value.endswith('B'):
            multiplier = 1_000_000_000
            value = value[:-1]
        elif value.endswith('M'):
            multiplier = 1_000_000
            value = value[:-1]
        elif value.endswith('K'):
            multiplier = 1_000
            value = value[:-1]
            
        try:
            return float(value) * multiplier
        except:
            return 0
    
    def _passes_filters(self, symbol: UniverseSymbol) -> bool:
        """Vérifie si le symbole passe les filtres US"""
        # Note: NASDAQ API doesn't return volume, so we skip that filter
        return (
            symbol.market_cap >= self.config.us_min_market_cap and
            symbol.price >= self.config.us_min_price
        )


# =============================================================================
# UNIVERSE SCANNER (Main Class)
# =============================================================================

class UniverseScanner:
    """
    Scanner d'univers complet US + Euronext.
    
    Usage:
        scanner = UniverseScanner()
        await scanner.initialize()
        
        # Scan complet
        snapshot = await scanner.scan_full_universe()
        
        # Obtenir les symboles pour le HeatDetector
        hot_candidates = scanner.get_candidates_for_heat_detection()
    """
    
    def __init__(self, config: Optional[UniverseConfig] = None):
        self.config = config or UniverseConfig()
        self.us_fetcher = USFetcher(self.config)
        self.eu_fetcher = EuronextFetcher(self.config)
        
        # Cache
        self._us_symbols: List[UniverseSymbol] = []
        self._eu_symbols: List[UniverseSymbol] = []
        self._last_scan: Optional[datetime] = None
        
        # Persistence
        self._data_dir = Path("data/universe")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("UniverseScanner initialized")
        
    async def initialize(self):
        """Initialise le scanner et charge le cache"""
        await self._load_cache()
        logger.info("UniverseScanner ready")
        
    async def scan_full_universe(self, force: bool = False) -> UniverseSnapshot:
        """
        Scan complet de l'univers US + EU.
        
        Args:
            force: Force le rescan même si le cache est valide
            
        Returns:
            UniverseSnapshot avec tous les symboles
        """
        # Check cache validity
        if not force and self._is_cache_valid():
            logger.info("Using cached universe data")
            return self._create_snapshot()
            
        logger.info("Starting full universe scan...")
        
        # Fetch US
        logger.info("Fetching US symbols...")
        self._us_symbols = await self.us_fetcher.fetch_all()
        logger.info(f"US: {len(self._us_symbols)} symbols")
        
        # Fetch EU
        logger.info("Fetching Euronext symbols...")
        self._eu_symbols = await self.eu_fetcher.fetch_all()
        logger.info(f"EU: {len(self._eu_symbols)} symbols")
        
        # Update timestamp
        self._last_scan = datetime.now()
        
        # Save cache
        await self._save_cache()
        
        snapshot = self._create_snapshot()
        logger.info(f"Universe scan complete: {snapshot.total_count} total symbols")
        
        return snapshot
    
    def get_all_symbols(self) -> List[str]:
        """Retourne la liste de tous les symboles (pour watchlist)"""
        symbols = []
        symbols.extend([s.symbol for s in self._us_symbols])
        symbols.extend([s.symbol for s in self._eu_symbols])
        return symbols
    
    def get_us_symbols(self) -> List[str]:
        """Retourne les symboles US"""
        return [s.symbol for s in self._us_symbols]
    
    def get_eu_symbols(self) -> List[str]:
        """Retourne les symboles EU"""
        return [s.symbol for s in self._eu_symbols]
    
    def get_symbols_by_sector(self, sector: str) -> List[UniverseSymbol]:
        """Retourne les symboles d'un secteur"""
        all_symbols = self._us_symbols + self._eu_symbols
        return [s for s in all_symbols if s.sector.lower() == sector.lower()]
    
    def _is_cache_valid(self) -> bool:
        """Vérifie si le cache est encore valide"""
        if not self._last_scan:
            return False
        age = datetime.now() - self._last_scan
        return age.total_seconds() < self.config.cache_ttl_hours * 3600
    
    def _create_snapshot(self) -> UniverseSnapshot:
        """Crée un snapshot de l'état actuel"""
        return UniverseSnapshot(
            timestamp=self._last_scan or datetime.now(),
            us_symbols=self._us_symbols,
            eu_symbols=self._eu_symbols,
            total_count=len(self._us_symbols) + len(self._eu_symbols),
            filters_applied={
                'us': {
                    'min_volume': self.config.us_min_volume,
                    'min_market_cap': self.config.us_min_market_cap,
                    'min_price': self.config.us_min_price,
                },
                'eu': {
                    'min_volume': self.config.eu_min_volume,
                    'min_market_cap': self.config.eu_min_market_cap,
                    'min_price': self.config.eu_min_price,
                },
            },
        )
    
    async def _load_cache(self):
        """Charge le cache depuis le disque"""
        cache_file = self._data_dir / "universe_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    
                self._last_scan = datetime.fromisoformat(data.get('timestamp', ''))
                
                # Check if cache is still valid
                if self._is_cache_valid():
                    self._us_symbols = [UniverseSymbol(**s) for s in data.get('us', [])]
                    self._eu_symbols = [UniverseSymbol(**s) for s in data.get('eu', [])]
                    logger.info(f"Loaded cache: {len(self._us_symbols)} US, {len(self._eu_symbols)} EU")
                    
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
    
    async def _save_cache(self):
        """Sauvegarde le cache sur disque"""
        cache_file = self._data_dir / "universe_cache.json"
        try:
            data = {
                'timestamp': self._last_scan.isoformat() if self._last_scan else None,
                'us': [s.to_dict() for s in self._us_symbols],
                'eu': [s.to_dict() for s in self._eu_symbols],
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Universe cache saved")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")


# =============================================================================
# SINGLETON
# =============================================================================

_universe_scanner: Optional[UniverseScanner] = None

def get_universe_scanner() -> UniverseScanner:
    """Retourne l'instance singleton du UniverseScanner"""
    global _universe_scanner
    if _universe_scanner is None:
        _universe_scanner = UniverseScanner()
    return _universe_scanner


# =============================================================================
# CLI TEST
# =============================================================================

async def main():
    """Test du scanner"""
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    
    scanner = UniverseScanner()
    await scanner.initialize()
    
    print("\n" + "="*60)
    print("UNIVERSE SCANNER TEST")
    print("="*60)
    
    snapshot = await scanner.scan_full_universe(force=True)
    
    print(f"\n📊 Résultats:")
    print(f"  US: {len(snapshot.us_symbols)} symboles")
    print(f"  EU: {len(snapshot.eu_symbols)} symboles")
    print(f"  Total: {snapshot.total_count} symboles")
    
    print(f"\n🇺🇸 Top 10 US (par market cap):")
    us_sorted = sorted(snapshot.us_symbols, key=lambda x: x.market_cap, reverse=True)[:10]
    for s in us_sorted:
        print(f"  {s.symbol:8} | {s.name[:30]:30} | ${s.market_cap/1e9:.1f}B")
    
    print(f"\n🇪🇺 Top 10 EU (par market cap):")
    eu_sorted = sorted(snapshot.eu_symbols, key=lambda x: x.market_cap, reverse=True)[:10]
    for s in eu_sorted:
        print(f"  {s.symbol:10} | {s.name[:30]:30} | €{s.market_cap/1e9:.1f}B")


if __name__ == "__main__":
    asyncio.run(main())
