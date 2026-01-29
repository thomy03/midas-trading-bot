"""
Sector Momentum Analyzer
Analyse le momentum sectoriel via ETF proxies pour filtrer les signaux.

Un secteur est considéré "bullish" si:
1. Performance 20j > 0
2. Performance vs SPY >= seuil minimum
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@dataclass
class SectorMomentum:
    """Données de momentum pour un secteur."""
    sector: str
    etf_symbol: str
    perf_10d: float  # Performance sur 10 jours (%)
    perf_20d: float  # Performance sur 20 jours (%)
    vs_spy_10d: float  # Différence de performance vs SPY sur 10j (%)
    is_bullish: bool # Secteur en tendance haussière
    rank: int        # Rang parmi les secteurs (1 = meilleur)


class SectorAnalyzer:
    """
    Analyseur de momentum sectoriel utilisant des ETF comme proxies.

    Utilise les performances relatives vs SPY pour identifier les secteurs
    en tendance haussière et filtrer les signaux de trading.
    """

    # Mapping Secteur -> ETF proxy
    SECTOR_ETF_MAP = {
        'Technology': 'XLK',           # Tech (AAPL, MSFT, etc.)
        'AI_Semiconductors': 'SMH',    # Semiconducteurs/IA (NVDA, AMD)
        'Healthcare': 'XLV',           # Santé (JNJ, PFE)
        'Finance': 'XLF',              # Finance (JPM, BAC)
        'Energy': 'XLE',               # Énergie (XOM, CVX)
        'Consumer_Discretionary': 'XLY',  # Conso discrétionnaire (AMZN, TSLA)
        'Industrials': 'XLI',          # Industriels (CAT, HON)
        'Materials': 'XLB',            # Matériaux (LIN, APD)
        'Utilities': 'XLU',            # Services publics
        'Real_Estate': 'XLRE',         # Immobilier
        'Communications': 'XLC',       # Communications (META, GOOGL)
        'Consumer_Staples': 'XLP',     # Conso de base (PG, KO)
    }

    # Mapping Symbole -> Secteur
    SYMBOL_SECTOR_MAP = {
        # US Tech
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'GOOGL': 'Technology',
        'CRM': 'Technology',
        'SAP.DE': 'Technology',

        # AI / Semiconducteurs
        'NVDA': 'AI_Semiconductors',
        'AMD': 'AI_Semiconductors',

        # Communications
        'META': 'Communications',
        'NFLX': 'Communications',
        'DTE.DE': 'Communications',

        # Consumer Discretionary
        'TSLA': 'Consumer_Discretionary',
        'AMZN': 'Consumer_Discretionary',
        'MC.PA': 'Consumer_Discretionary',  # LVMH - Luxe
        'BMW.DE': 'Consumer_Discretionary',
        'MBG.DE': 'Consumer_Discretionary',  # Mercedes

        # Finance
        'JPM': 'Finance',
        'BNP.PA': 'Finance',
        'ALV.DE': 'Finance',  # Allianz

        # Healthcare
        'SAN.PA': 'Healthcare',  # Sanofi

        # Energy
        'XOM': 'Energy',
        'TTE.PA': 'Energy',  # TotalEnergies

        # Healthcare
        'JNJ': 'Healthcare',

        # Consumer Staples
        'PG': 'Consumer_Staples',
        'KO': 'Consumer_Staples',
        'OR.PA': 'Consumer_Staples',  # L'Oréal

        # Industrials
        'AI.PA': 'Industrials',   # Air Liquide
        'AIR.PA': 'Industrials',  # Airbus
        'SIE.DE': 'Industrials',  # Siemens

        # Materials
        'SU.PA': 'Materials',     # Schneider Electric
        'BAS.DE': 'Materials',    # BASF
    }

    def __init__(self, min_momentum_vs_spy: float = 0.0):
        """
        Initialise l'analyseur.

        Args:
            min_momentum_vs_spy: Seuil minimum de surperformance vs SPY (en %)
                                 0.0 = neutre ou mieux que SPY
                                 -2.0 = tolère jusqu'à -2% vs SPY
        """
        self.min_momentum_vs_spy = min_momentum_vs_spy
        self._sector_data: Dict[str, pd.DataFrame] = {}
        self._spy_data: Optional[pd.DataFrame] = None
        self._momentum_cache: Dict[str, Dict[str, SectorMomentum]] = {}
        self._last_load_date: Optional[datetime] = None

    def load_sector_data(self, market_data_fetcher, period: str = '1y') -> bool:
        """
        Charge les données des ETF sectoriels et SPY.

        Args:
            market_data_fetcher: Instance de MarketDataFetcher
            period: Période de données à charger

        Returns:
            True si chargement réussi
        """
        print("  [SectorAnalyzer] Loading sector ETF data...")

        # Liste de tous les ETF à charger
        etf_symbols = list(set(self.SECTOR_ETF_MAP.values())) + ['SPY']

        # Charger les données
        etf_data = market_data_fetcher.get_batch_historical_data(
            etf_symbols,
            period=period,
            interval='1d',
            batch_size=20
        )

        # Stocker SPY séparément
        if 'SPY' in etf_data:
            self._spy_data = etf_data['SPY']
            del etf_data['SPY']
        else:
            print("  [SectorAnalyzer] WARNING: Could not load SPY data")
            return False

        # Stocker les données sectorielles par secteur
        for sector, etf in self.SECTOR_ETF_MAP.items():
            if etf in etf_data:
                self._sector_data[sector] = etf_data[etf]

        loaded_count = len(self._sector_data)
        print(f"  [SectorAnalyzer] Loaded {loaded_count}/{len(self.SECTOR_ETF_MAP)} sector ETFs")

        self._last_load_date = datetime.now()
        self._momentum_cache.clear()

        return loaded_count > 0

    def _calculate_performance(self, df: pd.DataFrame, lookback: int, as_of_date: datetime = None) -> float:
        """
        Calcule la performance sur N jours.

        Args:
            df: DataFrame avec colonne 'Close'
            lookback: Nombre de jours
            as_of_date: Date de référence (None = dernière date disponible)

        Returns:
            Performance en pourcentage
        """
        if df is None or df.empty or len(df) < lookback:
            return 0.0

        try:
            # Filtrer jusqu'à la date si spécifiée
            if as_of_date is not None:
                as_of_date = pd.to_datetime(as_of_date)
                df = df[df.index <= as_of_date]

            if len(df) < lookback:
                return 0.0

            current_price = df['Close'].iloc[-1]
            past_price = df['Close'].iloc[-lookback]

            if past_price > 0:
                return ((current_price / past_price) - 1) * 100
            return 0.0

        except Exception as e:
            return 0.0

    def calculate_sector_momentum(self, as_of_date: datetime = None) -> Dict[str, SectorMomentum]:
        """
        Calcule le momentum de tous les secteurs.

        Args:
            as_of_date: Date de référence pour le calcul

        Returns:
            Dict[sector_name, SectorMomentum]
        """
        # Vérifier le cache
        cache_key = str(as_of_date) if as_of_date else 'latest'
        if cache_key in self._momentum_cache:
            return self._momentum_cache[cache_key]

        if self._spy_data is None or len(self._sector_data) == 0:
            return {}

        # Calculer performance SPY sur 10 et 20 jours
        spy_perf_10d = self._calculate_performance(self._spy_data, 10, as_of_date)
        spy_perf_20d = self._calculate_performance(self._spy_data, 20, as_of_date)

        momentum_data = {}

        for sector, df in self._sector_data.items():
            perf_10d = self._calculate_performance(df, 10, as_of_date)
            perf_20d = self._calculate_performance(df, 20, as_of_date)
            vs_spy_10d = perf_10d - spy_perf_10d

            # Un secteur est bullish si:
            # Sa performance 10j vs SPY est >= au seuil (critère assoupli)
            # On ne demande plus de performance positive absolue
            is_bullish = vs_spy_10d >= self.min_momentum_vs_spy

            momentum_data[sector] = SectorMomentum(
                sector=sector,
                etf_symbol=self.SECTOR_ETF_MAP.get(sector, 'N/A'),
                perf_10d=perf_10d,
                perf_20d=perf_20d,
                vs_spy_10d=vs_spy_10d,
                is_bullish=is_bullish,
                rank=0  # Sera calculé après
            )

        # Calculer les rangs (1 = meilleur vs_spy)
        sorted_sectors = sorted(momentum_data.values(), key=lambda x: x.vs_spy_10d, reverse=True)
        for i, sm in enumerate(sorted_sectors, 1):
            momentum_data[sm.sector].rank = i

        # Mettre en cache
        self._momentum_cache[cache_key] = momentum_data

        return momentum_data

    def get_symbol_sector(self, symbol: str) -> Optional[str]:
        """
        Retourne le secteur d'un symbole.

        Args:
            symbol: Symbole boursier

        Returns:
            Nom du secteur ou None si inconnu
        """
        return self.SYMBOL_SECTOR_MAP.get(symbol, None)

    def is_sector_bullish(self, symbol: str, as_of_date: datetime = None) -> bool:
        """
        Vérifie si le secteur d'un symbole est en tendance haussière.

        Args:
            symbol: Symbole boursier
            as_of_date: Date de référence

        Returns:
            True si le secteur est bullish, False sinon
        """
        sector = self.get_symbol_sector(symbol)

        if sector is None:
            # Symbole non mappé -> on laisse passer par défaut
            return True

        momentum = self.calculate_sector_momentum(as_of_date)

        if sector not in momentum:
            # Secteur sans données -> on laisse passer
            return True

        return momentum[sector].is_bullish

    def get_sector_momentum(self, symbol: str, as_of_date: datetime = None) -> Optional[SectorMomentum]:
        """
        Retourne les données de momentum du secteur d'un symbole.

        Args:
            symbol: Symbole boursier
            as_of_date: Date de référence

        Returns:
            SectorMomentum ou None
        """
        sector = self.get_symbol_sector(symbol)

        if sector is None:
            return None

        momentum = self.calculate_sector_momentum(as_of_date)
        return momentum.get(sector, None)

    def filter_signals_by_sector(self, signals: List[dict], as_of_date: datetime = None) -> List[dict]:
        """
        Filtre une liste de signaux pour ne garder que ceux dans des secteurs bullish.

        Args:
            signals: Liste de signaux (dicts avec au moins 'symbol')
            as_of_date: Date de référence

        Returns:
            Liste filtrée des signaux
        """
        if not signals:
            return []

        filtered = []

        for signal in signals:
            symbol = signal.get('symbol', '')

            if self.is_sector_bullish(symbol, as_of_date):
                filtered.append(signal)

        return filtered

    def get_bullish_sectors(self, as_of_date: datetime = None) -> List[SectorMomentum]:
        """
        Retourne la liste des secteurs en tendance haussière, triés par performance.

        Args:
            as_of_date: Date de référence

        Returns:
            Liste de SectorMomentum triée par vs_spy décroissant
        """
        momentum = self.calculate_sector_momentum(as_of_date)
        bullish = [sm for sm in momentum.values() if sm.is_bullish]
        return sorted(bullish, key=lambda x: x.vs_spy, reverse=True)

    def get_bearish_sectors(self, as_of_date: datetime = None) -> List[SectorMomentum]:
        """
        Retourne la liste des secteurs en tendance baissière, triés par performance.

        Args:
            as_of_date: Date de référence

        Returns:
            Liste de SectorMomentum triée par vs_spy croissant
        """
        momentum = self.calculate_sector_momentum(as_of_date)
        bearish = [sm for sm in momentum.values() if not sm.is_bullish]
        return sorted(bearish, key=lambda x: x.vs_spy)

    def print_sector_summary(self, as_of_date: datetime = None):
        """
        Affiche un résumé du momentum sectoriel.

        Args:
            as_of_date: Date de référence
        """
        momentum = self.calculate_sector_momentum(as_of_date)

        if not momentum:
            print("  [SectorAnalyzer] No sector data available")
            return

        print("\n  === SECTOR MOMENTUM ANALYSIS ===")
        print(f"  {'Rank':<5} {'Sector':<25} {'ETF':<6} {'10d%':<8} {'vs SPY 10d':<12} {'Status':<10}")
        print("  " + "-" * 75)

        sorted_sectors = sorted(momentum.values(), key=lambda x: x.rank)

        for sm in sorted_sectors:
            status = "BULLISH" if sm.is_bullish else "BEARISH"
            print(f"  {sm.rank:<5} {sm.sector:<25} {sm.etf_symbol:<6} {sm.perf_10d:>+6.1f}% {sm.vs_spy_10d:>+10.1f}%  {status:<10}")

        bullish_count = sum(1 for sm in momentum.values() if sm.is_bullish)
        print(f"\n  Bullish sectors: {bullish_count}/{len(momentum)}")


# Singleton instance
sector_analyzer = SectorAnalyzer(min_momentum_vs_spy=0.0)
