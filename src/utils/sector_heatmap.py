"""
Sector Heatmap for TradingBot V3

Provides sector-level market analysis:
- Performance by sector (1D, 1W, 1M)
- Signal distribution across sectors
- Top gainers/losers per sector
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# Sector definitions with representative stocks
SECTORS = {
    'Technology': {
        'etf': 'XLK',
        'color': '#2962FF',
        'stocks': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AVGO', 'ORCL', 'CRM', 'AMD', 'ADBE']
    },
    'Healthcare': {
        'etf': 'XLV',
        'color': '#00C853',
        'stocks': ['UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY']
    },
    'Financials': {
        'etf': 'XLF',
        'color': '#FF6D00',
        'stocks': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'USB']
    },
    'Consumer Discretionary': {
        'etf': 'XLY',
        'color': '#AB47BC',
        'stocks': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG']
    },
    'Consumer Staples': {
        'etf': 'XLP',
        'color': '#26A69A',
        'stocks': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'CL', 'MDLZ', 'EL']
    },
    'Energy': {
        'etf': 'XLE',
        'color': '#F44336',
        'stocks': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL']
    },
    'Industrials': {
        'etf': 'XLI',
        'color': '#795548',
        'stocks': ['CAT', 'UNP', 'HON', 'UPS', 'BA', 'RTX', 'DE', 'LMT', 'GE', 'MMM']
    },
    'Materials': {
        'etf': 'XLB',
        'color': '#607D8B',
        'stocks': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'DOW', 'DD', 'PPG']
    },
    'Real Estate': {
        'etf': 'XLRE',
        'color': '#9C27B0',
        'stocks': ['PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'O', 'SPG', 'WELL', 'DLR', 'AVB']
    },
    'Utilities': {
        'etf': 'XLU',
        'color': '#FFEB3B',
        'stocks': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'ED', 'WEC']
    },
    'Communication Services': {
        'etf': 'XLC',
        'color': '#00BCD4',
        'stocks': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'EA']
    }
}


@dataclass
class SectorPerformance:
    """Sector performance data"""
    name: str
    etf: str
    color: str
    perf_1d: float
    perf_1w: float
    perf_1m: float
    perf_ytd: float
    top_gainers: List[Dict]  # [{symbol, name, perf}]
    top_losers: List[Dict]
    signal_count: int  # Number of signals in this sector
    avg_volume_ratio: float


class SectorHeatmapBuilder:
    """Builds sector heatmap data"""

    def __init__(self):
        self.sectors = SECTORS
        self._cache = {}
        self._cache_time = None
        self._cache_duration = timedelta(minutes=15)

    def get_sector_performance(
        self,
        use_cache: bool = True
    ) -> List[SectorPerformance]:
        """
        Get performance data for all sectors

        Args:
            use_cache: Whether to use cached data (15min cache)

        Returns:
            List of SectorPerformance objects
        """
        # Check cache
        if use_cache and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_duration:
                return self._cache.get('sector_perf', [])

        results = []

        # Fetch ETF data in parallel
        etf_data = self._fetch_etf_data()

        for sector_name, sector_info in self.sectors.items():
            try:
                etf = sector_info['etf']
                data = etf_data.get(etf)

                if data is None or data.empty:
                    continue

                # Calculate performance
                perf_1d = self._calc_performance(data, 1)
                perf_1w = self._calc_performance(data, 5)
                perf_1m = self._calc_performance(data, 21)
                perf_ytd = self._calc_ytd_performance(data)

                # Get top movers (simplified - just from ETF components)
                top_gainers, top_losers = self._get_top_movers(sector_info['stocks'][:5])

                results.append(SectorPerformance(
                    name=sector_name,
                    etf=etf,
                    color=sector_info['color'],
                    perf_1d=perf_1d,
                    perf_1w=perf_1w,
                    perf_1m=perf_1m,
                    perf_ytd=perf_ytd,
                    top_gainers=top_gainers,
                    top_losers=top_losers,
                    signal_count=0,  # Will be updated by screener
                    avg_volume_ratio=1.0
                ))

            except Exception as e:
                logger.warning(f"Error getting data for {sector_name}: {e}")
                continue

        # Sort by 1-day performance
        results.sort(key=lambda x: x.perf_1d, reverse=True)

        # Update cache
        self._cache['sector_perf'] = results
        self._cache_time = datetime.now()

        return results

    def _fetch_etf_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch ETF data in parallel"""
        etfs = [info['etf'] for info in self.sectors.values()]

        results = {}

        def fetch_single(etf: str) -> Tuple[str, Optional[pd.DataFrame]]:
            try:
                ticker = yf.Ticker(etf)
                data = ticker.history(period='3mo')
                return etf, data
            except Exception as e:
                logger.warning(f"Error fetching {etf}: {e}")
                return etf, None

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_single, etf): etf for etf in etfs}
            for future in as_completed(futures):
                etf, data = future.result()
                results[etf] = data

        return results

    def _calc_performance(self, data: pd.DataFrame, days: int) -> float:
        """Calculate percentage performance over N days"""
        if data is None or len(data) < days + 1:
            return 0.0

        try:
            current = data['Close'].iloc[-1]
            past = data['Close'].iloc[-days - 1]
            return ((current - past) / past) * 100
        except Exception:
            return 0.0

    def _calc_ytd_performance(self, data: pd.DataFrame) -> float:
        """Calculate year-to-date performance"""
        if data is None or data.empty:
            return 0.0

        try:
            current = data['Close'].iloc[-1]
            # Find first trading day of year
            year_start = datetime(datetime.now().year, 1, 1)
            ytd_data = data[data.index >= pd.Timestamp(year_start)]

            if ytd_data.empty:
                return 0.0

            start_price = ytd_data['Close'].iloc[0]
            return ((current - start_price) / start_price) * 100
        except Exception:
            return 0.0

    def _get_top_movers(self, symbols: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """Get top gainers and losers from a list of symbols"""
        movers = []

        def fetch_stock(symbol: str) -> Optional[Dict]:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d')
                if hist.empty or len(hist) < 2:
                    return None

                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                perf = ((current - prev) / prev) * 100

                info = ticker.info
                name = info.get('shortName', symbol)[:20]

                return {
                    'symbol': symbol,
                    'name': name,
                    'price': current,
                    'perf': perf
                }
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_stock, s): s for s in symbols}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    movers.append(result)

        # Sort by performance
        movers.sort(key=lambda x: x['perf'], reverse=True)

        gainers = [m for m in movers if m['perf'] > 0][:3]
        losers = [m for m in movers if m['perf'] < 0][-3:]

        return gainers, losers

    def get_sector_for_symbol(self, symbol: str) -> Optional[str]:
        """Get sector name for a given symbol"""
        symbol = symbol.upper()
        for sector_name, sector_info in self.sectors.items():
            if symbol in sector_info['stocks']:
                return sector_name
        return None

    def update_signal_counts(
        self,
        alerts: List[Dict],
        sector_perfs: List[SectorPerformance]
    ) -> List[SectorPerformance]:
        """
        Update signal counts in sector performance data

        Args:
            alerts: List of alerts with 'symbol' key
            sector_perfs: List of SectorPerformance to update

        Returns:
            Updated list of SectorPerformance
        """
        # Count signals per sector
        sector_signals = {}
        for alert in alerts:
            symbol = alert.get('symbol', '')
            sector = self.get_sector_for_symbol(symbol)
            if sector:
                sector_signals[sector] = sector_signals.get(sector, 0) + 1

        # Update performance objects
        for perf in sector_perfs:
            perf.signal_count = sector_signals.get(perf.name, 0)

        return sector_perfs

    def get_heatmap_data(self) -> pd.DataFrame:
        """
        Get data formatted for heatmap visualization

        Returns:
            DataFrame with sectors and performance metrics
        """
        perfs = self.get_sector_performance()

        data = []
        for p in perfs:
            data.append({
                'Sector': p.name,
                'ETF': p.etf,
                '1D %': p.perf_1d,
                '1W %': p.perf_1w,
                '1M %': p.perf_1m,
                'YTD %': p.perf_ytd,
                'Signals': p.signal_count,
                'Color': p.color
            })

        return pd.DataFrame(data)

    def get_market_breadth(self) -> Dict:
        """
        Calculate market breadth indicators

        Returns:
            Dict with advancing/declining sectors
        """
        perfs = self.get_sector_performance()

        advancing = sum(1 for p in perfs if p.perf_1d > 0)
        declining = sum(1 for p in perfs if p.perf_1d < 0)
        unchanged = len(perfs) - advancing - declining

        avg_perf = np.mean([p.perf_1d for p in perfs]) if perfs else 0

        return {
            'advancing': advancing,
            'declining': declining,
            'unchanged': unchanged,
            'total': len(perfs),
            'avg_performance': avg_perf,
            'breadth_ratio': advancing / max(declining, 1)
        }

    def get_sector_rotation_signal(self) -> Dict:
        """
        Analyze sector rotation patterns

        Returns:
            Dict with rotation analysis
        """
        perfs = self.get_sector_performance()

        if not perfs:
            return {'signal': 'NEUTRAL', 'reason': 'No data'}

        # Defensive sectors
        defensive = ['Utilities', 'Consumer Staples', 'Healthcare']
        # Cyclical sectors
        cyclical = ['Technology', 'Consumer Discretionary', 'Financials', 'Industrials']

        def_perf = np.mean([p.perf_1w for p in perfs if p.name in defensive])
        cyc_perf = np.mean([p.perf_1w for p in perfs if p.name in cyclical])

        if cyc_perf > def_perf + 1:
            signal = 'RISK_ON'
            reason = f"Cyclical sectors ({cyc_perf:.1f}%) outperforming defensive ({def_perf:.1f}%)"
        elif def_perf > cyc_perf + 1:
            signal = 'RISK_OFF'
            reason = f"Defensive sectors ({def_perf:.1f}%) outperforming cyclical ({cyc_perf:.1f}%)"
        else:
            signal = 'NEUTRAL'
            reason = f"Balanced rotation (Cyc: {cyc_perf:.1f}%, Def: {def_perf:.1f}%)"

        return {
            'signal': signal,
            'reason': reason,
            'cyclical_avg': cyc_perf,
            'defensive_avg': def_perf
        }


# Singleton instance
sector_heatmap_builder = SectorHeatmapBuilder()
