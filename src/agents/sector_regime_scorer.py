"""
Sector-Regime Scoring Module

Applies sector-based bonus/malus to composite scores depending on market regime.

Philosophy:
- BULL: Favor growth/tech, penalize slow defensives
- RANGE: Mostly neutral, slight penalty on speculative small caps
- BEAR: Favor defensives/dividends, penalize growth without profits
- VOLATILE: Strongly favor mega-cap defensives, penalize cyclicals

Bonuses are kept moderate (3-7 points on 0-100 scale) to avoid overfitting.
The main alpha comes from hard filtering (DefensiveManager) not from scoring tweaks.
"""

import logging
from typing import Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ─── Sector Categories ───────────────────────────────────────────────────────

class SectorCategory(Enum):
    GROWTH = "growth"
    DEFENSIVE = "defensive"
    CYCLICAL = "cyclical"
    NEUTRAL = "neutral"


# Map yfinance sector names to categories
SECTOR_CATEGORY_MAP = {
    # Growth - high beta, tech-driven, outperform in BULL
    'Technology': SectorCategory.GROWTH,
    'Communication Services': SectorCategory.GROWTH,

    # Defensive - low beta, stable earnings, outperform in BEAR
    'Consumer Defensive': SectorCategory.DEFENSIVE,
    'Utilities': SectorCategory.DEFENSIVE,
    'Healthcare': SectorCategory.DEFENSIVE,
    'Real Estate': SectorCategory.DEFENSIVE,

    # Cyclical - follow economic cycle, underperform in BEAR
    'Consumer Cyclical': SectorCategory.CYCLICAL,
    'Industrials': SectorCategory.CYCLICAL,
    'Basic Materials': SectorCategory.CYCLICAL,
    'Energy': SectorCategory.CYCLICAL,

    # Neutral - mixed characteristics
    'Financial Services': SectorCategory.NEUTRAL,
}


# ─── Regime Bonus/Malus Matrix ───────────────────────────────────────────────
# Values are score adjustments on a 0-100 scale
# Kept moderate to avoid overfitting

REGIME_SECTOR_ADJUSTMENTS = {
    'BULL': {
        SectorCategory.GROWTH: +5,       # Tech/growth outperform in BULL
        SectorCategory.DEFENSIVE: -3,     # Utilities/staples underperform
        SectorCategory.CYCLICAL: +3,      # Cyclicals benefit from expansion
        SectorCategory.NEUTRAL: 0,
    },
    'RANGE': {
        SectorCategory.GROWTH: 0,         # Neutral - no strong trend
        SectorCategory.DEFENSIVE: 0,
        SectorCategory.CYCLICAL: 0,
        SectorCategory.NEUTRAL: 0,
    },
    'BEAR': {
        SectorCategory.GROWTH: -5,        # Growth stocks punished in BEAR
        SectorCategory.DEFENSIVE: +5,     # Defensives are safe havens
        SectorCategory.CYCLICAL: -3,      # Cyclicals suffer
        SectorCategory.NEUTRAL: 0,
    },
    'VOLATILE': {
        SectorCategory.GROWTH: -7,        # High beta = high pain
        SectorCategory.DEFENSIVE: +7,     # Low beta = safety
        SectorCategory.CYCLICAL: -5,      # Cyclicals get crushed
        SectorCategory.NEUTRAL: -2,
    },
}


# ─── Market Cap Bonus/Malus by Regime ────────────────────────────────────────
# Small caps outperform in BULL, underperform in BEAR/VOLATILE

def _market_cap_adjustment(market_cap_b: float, regime: str) -> float:
    """
    Bonus/malus based on market cap and regime.

    Args:
        market_cap_b: Market cap in billions (e.g., 2.5 for $2.5B)
        regime: Market regime string (BULL/BEAR/RANGE/VOLATILE)

    Returns:
        Score adjustment (-5 to +5)
    """
    if market_cap_b <= 0:
        return 0.0

    is_small = market_cap_b < 2        # < $2B
    is_mid = 2 <= market_cap_b < 10    # $2B - $10B
    is_large = 10 <= market_cap_b < 100  # $10B - $100B
    is_mega = market_cap_b >= 100      # > $100B

    if regime == 'BULL':
        # Small/mid caps get a bonus in BULL (small cap effect)
        if is_small:
            return +3
        elif is_mid:
            return +2
        return 0

    elif regime == 'BEAR':
        # Small caps penalized, mega caps favored
        if is_small:
            return -5
        elif is_mid:
            return -2
        elif is_mega:
            return +3
        return 0

    elif regime == 'VOLATILE':
        # Only mega caps are safe
        if is_small:
            return -5
        elif is_mid:
            return -3
        elif is_mega:
            return +3
        return 0

    # RANGE: slight penalty on micro/small speculative
    if is_small:
        return -2
    return 0


# ─── Static Symbol-to-Sector Mapping ─────────────────────────────────────────
# Built from config/symbol_universe.py for fast backtesting without API calls

def _build_symbol_sector_map():
    """Build a static symbol -> sector mapping from symbol_universe lists."""
    mapping = {}
    try:
        from config.symbol_universe import (
            SP500_TECH, SP500_HEALTHCARE, SP500_FINANCIALS,
            SP500_CONSUMER_DISC, SP500_CONSUMER_STAPLES, SP500_ENERGY,
            SP500_INDUSTRIALS, SP500_MATERIALS, SP500_COMMUNICATION,
            SP500_UTILITIES, SP500_REALESTATE,
            MIDCAP_TECH, MIDCAP_HEALTHCARE, MIDCAP_FINANCIALS,
            MIDCAP_CONSUMER, MIDCAP_INDUSTRIAL, MIDCAP_ENERGY_MATERIALS,
            MIDCAP_COMM_REIT, POPULAR_ADDITIONS,
        )

        sector_lists = {
            'Technology': SP500_TECH + MIDCAP_TECH,
            'Healthcare': SP500_HEALTHCARE + MIDCAP_HEALTHCARE,
            'Financial Services': SP500_FINANCIALS + MIDCAP_FINANCIALS,
            'Consumer Cyclical': SP500_CONSUMER_DISC + MIDCAP_CONSUMER,
            'Consumer Defensive': SP500_CONSUMER_STAPLES,
            'Energy': SP500_ENERGY,
            'Industrials': SP500_INDUSTRIALS + MIDCAP_INDUSTRIAL,
            'Basic Materials': SP500_MATERIALS,
            'Communication Services': SP500_COMMUNICATION,
            'Utilities': SP500_UTILITIES,
            'Real Estate': SP500_REALESTATE,
        }

        # Midcap energy/materials split
        for sym in MIDCAP_ENERGY_MATERIALS:
            if sym not in mapping:
                # Energy companies
                if sym in ('AR', 'RRC', 'EQT', 'CNX', 'MTDR', 'CHRD', 'SM',
                           'NOV', 'PTEN', 'HP', 'LBRT', 'RES', 'WHD', 'PUMP'):
                    mapping[sym] = 'Energy'
                else:
                    mapping[sym] = 'Basic Materials'

        # MIDCAP_COMM_REIT split
        comm_symbols = ('SPOT', 'RDDT', 'GRAB', 'SE', 'BIDU', 'JD', 'PDD',
                        'BABA', 'TME', 'BILI', 'IQ', 'ZTO', 'VNET', 'WB')
        for sym in MIDCAP_COMM_REIT:
            if sym not in mapping:
                if sym in comm_symbols:
                    mapping[sym] = 'Communication Services'
                else:
                    mapping[sym] = 'Real Estate'

        # Popular additions - categorize by subgroup
        ev_clean = ('RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'QS', 'CHPT', 'BLNK',
                    'PLUG', 'FCEL', 'BE', 'ENPH', 'SEDG', 'RUN', 'ARRY', 'MAXN',
                    'FSLR', 'CSIQ', 'JKS', 'DQ')
        ai_semi = ('TSM', 'ASML', 'OLED', 'LSCC', 'RMBS', 'POWI',
                   'SLAB', 'DIOD', 'SITM', 'AMBA', 'CEVA', 'ALGM')
        biotech_extra = ('BIIB', 'RPRX', 'UTHR', 'MEDP', 'ENSG', 'OMCL',
                         'ANGO', 'ATEC', 'SIBN', 'LIVN', 'NVST')
        fintech = ('FIS', 'FISV', 'GPN', 'WEX', 'GDOT', 'CPAY')
        defense = ('HII', 'LHX', 'TDG', 'HEI', 'KTOS', 'MRCY', 'CACI', 'BWXT')

        for sym in POPULAR_ADDITIONS:
            if sym in mapping:
                continue
            if sym in ev_clean or sym in ai_semi:
                mapping[sym] = 'Technology'
            elif sym in biotech_extra:
                mapping[sym] = 'Healthcare'
            elif sym in fintech:
                mapping[sym] = 'Financial Services'
            elif sym in defense:
                mapping[sym] = 'Industrials'
            else:
                mapping[sym] = 'Consumer Cyclical'  # Default for misc

        # Main sector lists
        for sector, symbols in sector_lists.items():
            for sym in symbols:
                if sym not in mapping:  # Don't override specific mappings
                    mapping[sym] = sector

    except ImportError:
        logger.warning("config.symbol_universe not found, sector mapping empty")

    return mapping


# Module-level cache
_SYMBOL_SECTOR_MAP = None


def get_symbol_sector_map():
    """Get cached symbol -> sector mapping."""
    global _SYMBOL_SECTOR_MAP
    if _SYMBOL_SECTOR_MAP is None:
        _SYMBOL_SECTOR_MAP = _build_symbol_sector_map()
    return _SYMBOL_SECTOR_MAP


def get_sector(symbol, info=None):
    """
    Get sector for a symbol.
    First checks static map (fast, for backtesting),
    then falls back to yfinance info dict (live trading).
    """
    # Fast path: static map
    sector_map = get_symbol_sector_map()
    if symbol in sector_map:
        return sector_map[symbol]

    # Slow path: from yfinance info dict
    if info and isinstance(info, dict):
        return info.get('sector', 'Unknown')

    return 'Unknown'


# ─── Main Scorer Class ───────────────────────────────────────────────────────

class SectorRegimeScorer:
    """
    Applies sector and market-cap based score adjustments per regime.

    Usage:
        scorer = SectorRegimeScorer()
        adjustment = scorer.get_adjustment('AAPL', regime='BULL')
        final_score = base_score + adjustment

        # With market cap info:
        adjustment = scorer.get_adjustment('AAPL', regime='BULL', market_cap=2.8e12)
    """

    def __init__(self, enable_market_cap_adj: bool = True):
        self.enable_market_cap_adj = enable_market_cap_adj
        self._sector_cache = {}  # symbol -> sector (for yfinance lookups)

    def get_adjustment(
        self,
        symbol: str,
        regime: str,
        sector: Optional[str] = None,
        market_cap: Optional[float] = None,
        info: Optional[Dict] = None,
    ) -> float:
        """
        Calculate total score adjustment for a symbol in a given regime.

        Args:
            symbol: Stock ticker
            regime: Market regime (BULL/BEAR/RANGE/VOLATILE)
            sector: Sector name (if known). If None, looked up from mapping.
            market_cap: Market cap in USD (e.g., 2.8e12 for $2.8T). Optional.
            info: yfinance info dict (fallback for sector lookup)

        Returns:
            Score adjustment (-12 to +12 on 0-100 scale)
        """
        # Resolve sector
        if sector is None:
            sector = get_sector(symbol, info)

        if sector == 'Unknown':
            return 0.0

        # Get sector category
        category = SECTOR_CATEGORY_MAP.get(sector, SectorCategory.NEUTRAL)

        # Sector adjustment
        regime_adjustments = REGIME_SECTOR_ADJUSTMENTS.get(regime, {})
        sector_adj = regime_adjustments.get(category, 0)

        # Market cap adjustment
        cap_adj = 0.0
        if self.enable_market_cap_adj and market_cap is not None and market_cap > 0:
            cap_adj = _market_cap_adjustment(market_cap / 1e9, regime)

        total = sector_adj + cap_adj

        if abs(total) > 0:
            logger.debug(
                f"[SECTOR] {symbol} ({sector}/{category.value}) in {regime}: "
                f"sector={sector_adj:+d}, cap={cap_adj:+.0f}, total={total:+.1f}"
            )

        return total

    def get_adjustment_batch(
        self,
        symbols,
        regime,
        sector_map=None,
        market_caps=None,
    ):
        """
        Get adjustments for multiple symbols at once (backtesting optimization).

        Args:
            symbols: List of tickers
            regime: Current regime string
            sector_map: Dict of symbol -> sector (optional, uses static map)
            market_caps: Dict of symbol -> market_cap in USD (optional)

        Returns:
            Dict of symbol -> adjustment
        """
        results = {}
        for sym in symbols:
            sector = sector_map.get(sym) if sector_map else None
            cap = market_caps.get(sym) if market_caps else None
            results[sym] = self.get_adjustment(sym, regime, sector=sector, market_cap=cap)
        return results


# ─── Singleton ────────────────────────────────────────────────────────────────

_scorer = None


def get_sector_regime_scorer(**kwargs):
    """Get singleton SectorRegimeScorer."""
    global _scorer
    if _scorer is None:
        _scorer = SectorRegimeScorer(**kwargs)
    return _scorer
