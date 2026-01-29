"""
Hybrid Screener: FMP Pre-screening + IBKR Validation

Architecture:
    Phase 1: FMP scans 3000 tickers (no IBKR limits)
    Phase 2: IBKR validates 100 candidates (within pacing limits)
    Phase 3: IBKR executes orders (via placeOrder)

This approach avoids IBKR Pacing Violations (Error 162) which occur
when requesting historical data for too many symbols.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

try:
    from ib_insync import IB, Stock, util
except ImportError:
    IB = None
    Stock = None
    util = None

from src.data.fmp_client import FMPClient, get_fmp_client
from src.indicators.ema_analyzer import EMAAnalyzer
from trendline_analysis.core.rsi_breakout_analyzer import RSIBreakoutAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ScreeningCandidate:
    """A stock candidate from pre-screening"""
    symbol: str
    signal: str  # 'STRONG_BUY', 'BUY', 'WATCH'
    rsi_at_breakout: float
    strength: str  # 'WEAK', 'MODERATE', 'STRONG'
    source: str = 'FMP'  # 'FMP' or 'IBKR'

    # Optional validation data
    ibkr_signal: Optional[str] = None
    ibkr_rsi: Optional[float] = None
    validated: bool = False
    tolerance_applied: bool = False

    # Additional metadata
    market_cap: Optional[float] = None
    volume: Optional[int] = None
    price: Optional[float] = None


@dataclass
class HybridScreenerConfig:
    """Configuration for hybrid screener"""
    # FMP pre-screening parameters
    market_cap_min: int = 500_000_000  # $500M minimum
    volume_min: int = 500_000  # 500K daily volume
    price_min: float = 5.0
    price_max: float = 300.0
    max_fmp_candidates: int = 3000

    # IBKR validation parameters
    max_ibkr_candidates: int = 100  # Stay within pacing limits
    ibkr_rate_limit_delay: float = 0.3  # 300ms between requests

    # Signal validation
    require_volume_confirmation: bool = True
    min_rsi_breakout_distance: float = 0.0

    # Tolerance for FMP/IBKR data mismatch
    allow_signal_downgrade: bool = True  # Accept IBKR=BUY if FMP=STRONG_BUY


class HybridScreener:
    """
    Hybrid screener combining FMP pre-screening with IBKR validation.

    Usage:
        screener = HybridScreener()
        await screener.initialize()

        # Phase 1: FMP pre-screening
        candidates = await screener.run_prescreening()

        # Phase 2: IBKR validation (if connected)
        validated = await screener.run_ibkr_validation(candidates)
    """

    def __init__(
        self,
        config: Optional[HybridScreenerConfig] = None,
        fmp_client: Optional[FMPClient] = None,
        ibkr_client: Optional[Any] = None
    ):
        """
        Initialize hybrid screener.

        Args:
            config: Screener configuration
            fmp_client: FMP API client (created if None)
            ibkr_client: IB-insync client (optional)
        """
        self.config = config or HybridScreenerConfig()
        self.fmp = fmp_client
        self.ibkr = ibkr_client

        self.ema_analyzer = EMAAnalyzer()
        self.rsi_analyzer = RSIBreakoutAnalyzer()

        self._initialized = False

    async def initialize(self):
        """Initialize the screener and its clients."""
        if self._initialized:
            return

        # Initialize FMP client if not provided
        if self.fmp is None:
            self.fmp = await get_fmp_client()

        self._initialized = True
        logger.info("HybridScreener initialized")

    async def run_prescreening(self) -> List[ScreeningCandidate]:
        """
        Phase 1: FMP Pre-screening

        Scans thousands of stocks using FMP API (no IBKR limits).
        Filters by market cap, volume, price, and RSI breakout signals.

        Returns:
            List of candidates with RSI breakout signals
        """
        if not self._initialized:
            await self.initialize()

        logger.info("=" * 60)
        logger.info("PHASE 1: FMP Pre-screening")
        logger.info("=" * 60)

        # Step 1: Get base candidates from FMP screener
        logger.info("Step 1: Running FMP stock screener...")
        base_candidates = await self.fmp.get_stock_screener(
            market_cap_min=self.config.market_cap_min,
            volume_min=self.config.volume_min,
            price_min=self.config.price_min,
            price_max=self.config.price_max,
            limit=self.config.max_fmp_candidates
        )

        symbols = [c.get("symbol") for c in base_candidates if c.get("symbol")]
        logger.info(f"FMP screener returned {len(symbols)} candidates")

        # Step 2: Add momentum candidates (top gainers)
        logger.info("Step 2: Adding momentum candidates...")
        try:
            gainers_losers = await self.fmp.get_gainers_losers()
            gainer_symbols = [
                g.get("symbol") for g in gainers_losers.get("gainers", [])[:50]
                if g.get("symbol")
            ]
            symbols = list(set(symbols + gainer_symbols))
            logger.info(f"Added {len(gainer_symbols)} gainers, total: {len(symbols)}")
        except Exception as e:
            logger.warning(f"Could not fetch gainers: {e}")

        # Step 3: Download historical data (batch)
        logger.info(f"Step 3: Downloading historical data for {len(symbols)} symbols...")
        historical_data = await self.fmp.get_bulk_historical(
            symbols,
            days=200,
            max_concurrent=50
        )
        logger.info(f"Successfully downloaded {len(historical_data)} symbols")

        # Step 4: Analyze RSI breakouts
        logger.info("Step 4: Analyzing RSI breakouts...")
        candidates_with_signal = []

        for symbol, df in historical_data.items():
            if df is None or len(df) < 50:
                continue

            try:
                # Run RSI breakout analysis
                result = self.rsi_analyzer.analyze(df, lookback_periods=52)

                if result is None or not result.get('has_breakout'):
                    continue

                breakout = result.get('breakout')
                signal = self.rsi_analyzer.get_signal(result)

                if signal in ['STRONG_BUY', 'BUY']:
                    # Get metadata from FMP candidate list
                    fmp_info = next(
                        (c for c in base_candidates if c.get("symbol") == symbol),
                        {}
                    )

                    candidate = ScreeningCandidate(
                        symbol=symbol,
                        signal=signal,
                        rsi_at_breakout=breakout.rsi_value if breakout else 0,
                        strength=breakout.strength if breakout else 'UNKNOWN',
                        source='FMP',
                        market_cap=fmp_info.get('marketCap'),
                        volume=fmp_info.get('volume'),
                        price=fmp_info.get('price')
                    )
                    candidates_with_signal.append(candidate)

            except Exception as e:
                logger.debug(f"Error analyzing {symbol}: {e}")
                continue

        logger.info(f"Phase 1 complete: {len(candidates_with_signal)} candidates with RSI signals")
        logger.info(f"  - STRONG_BUY: {sum(1 for c in candidates_with_signal if c.signal == 'STRONG_BUY')}")
        logger.info(f"  - BUY: {sum(1 for c in candidates_with_signal if c.signal == 'BUY')}")

        return candidates_with_signal

    async def run_ibkr_validation(
        self,
        candidates: List[ScreeningCandidate]
    ) -> List[ScreeningCandidate]:
        """
        Phase 2: IBKR Validation

        Downloads official IBKR data for top candidates and re-validates signals.
        Limits to max_ibkr_candidates to avoid pacing violations.

        Args:
            candidates: Candidates from Phase 1

        Returns:
            List of validated candidates
        """
        if not self.ibkr:
            logger.warning("IBKR client not connected - skipping validation")
            return candidates

        if IB is None:
            logger.warning("ib_insync not installed - skipping IBKR validation")
            return candidates

        # Check IBKR connection
        if not self.ibkr.isConnected():
            logger.warning("IBKR not connected - skipping validation")
            return candidates

        logger.info("=" * 60)
        logger.info("PHASE 2: IBKR Validation")
        logger.info("=" * 60)

        # Limit to max candidates
        candidates_to_validate = candidates[:self.config.max_ibkr_candidates]
        logger.info(f"Validating {len(candidates_to_validate)} candidates with IBKR...")

        validated = []

        for i, candidate in enumerate(candidates_to_validate):
            try:
                logger.debug(f"[{i + 1}/{len(candidates_to_validate)}] Validating {candidate.symbol}...")

                # Create IBKR contract
                contract = Stock(candidate.symbol, 'SMART', 'USD')

                # Request historical data
                bars = await self.ibkr.reqHistoricalDataAsync(
                    contract,
                    endDateTime='',
                    durationStr='6 M',
                    barSizeSetting='1 day',
                    whatToShow='TRADES',
                    useRTH=True
                )

                if not bars:
                    logger.debug(f"{candidate.symbol}: No IBKR data")
                    continue

                # Convert to DataFrame
                df = util.df(bars)

                if df is None or len(df) < 50:
                    logger.debug(f"{candidate.symbol}: Insufficient IBKR data")
                    continue

                # Re-analyze with IBKR data
                result = self.rsi_analyzer.analyze(df, lookback_periods=52)

                if result is None:
                    continue

                ibkr_signal = self.rsi_analyzer.get_signal(result)
                breakout = result.get('breakout')

                # Update candidate with IBKR results
                candidate.ibkr_signal = ibkr_signal
                candidate.ibkr_rsi = breakout.rsi_value if breakout else None

                # Validate signal
                if self._validate_signal(candidate):
                    candidate.validated = True
                    validated.append(candidate)
                    logger.info(f"{candidate.symbol}: VALIDATED - FMP={candidate.signal}, IBKR={ibkr_signal}")
                else:
                    logger.debug(f"{candidate.symbol}: REJECTED - FMP={candidate.signal}, IBKR={ibkr_signal}")

                # Rate limiting to avoid pacing violations
                await asyncio.sleep(self.config.ibkr_rate_limit_delay)

            except Exception as e:
                logger.error(f"IBKR validation error for {candidate.symbol}: {e}")
                continue

        logger.info(f"Phase 2 complete: {len(validated)}/{len(candidates_to_validate)} validated")

        return validated

    def _validate_signal(self, candidate: ScreeningCandidate) -> bool:
        """
        Validate if FMP and IBKR signals are consistent.

        Handles data mismatch between sources with configurable tolerance.

        Args:
            candidate: Candidate with both FMP and IBKR signals

        Returns:
            True if valid, False otherwise
        """
        fmp_signal = candidate.signal
        ibkr_signal = candidate.ibkr_signal

        if ibkr_signal is None:
            return False

        # Exact match
        if ibkr_signal == fmp_signal:
            return True

        # Signal matches (both are buy signals)
        if ibkr_signal in ['STRONG_BUY', 'BUY'] and fmp_signal in ['STRONG_BUY', 'BUY']:
            return True

        # Tolerance: Accept IBKR=WATCH if FMP=STRONG_BUY (data mismatch)
        if self.config.allow_signal_downgrade:
            if fmp_signal == 'STRONG_BUY' and ibkr_signal == 'WATCH':
                candidate.tolerance_applied = True
                return True

        return False

    async def run_full_scan(
        self,
        validate_with_ibkr: bool = True
    ) -> List[ScreeningCandidate]:
        """
        Run complete screening process.

        Args:
            validate_with_ibkr: Whether to validate with IBKR (if connected)

        Returns:
            List of final candidates
        """
        # Phase 1: FMP pre-screening
        candidates = await self.run_prescreening()

        if not candidates:
            logger.info("No candidates found in pre-screening")
            return []

        # Phase 2: IBKR validation (optional)
        if validate_with_ibkr and self.ibkr:
            candidates = await self.run_ibkr_validation(candidates)

        # Sort by signal strength
        signal_priority = {'STRONG_BUY': 3, 'BUY': 2, 'WATCH': 1}
        candidates.sort(
            key=lambda c: (signal_priority.get(c.signal, 0), c.rsi_at_breakout),
            reverse=True
        )

        logger.info(f"Full scan complete: {len(candidates)} final candidates")
        return candidates

    async def close(self):
        """Close the FMP client to prevent async session leaks."""
        logger.info("Closing HybridScreener...")

        if self.fmp:
            try:
                await self.fmp.close()
            except Exception as e:
                logger.warning(f"Error closing FMP client: {e}")

        self._initialized = False
        logger.info("HybridScreener closed")


# Singleton instance
_hybrid_screener: Optional[HybridScreener] = None


async def get_hybrid_screener(
    ibkr_client: Optional[Any] = None
) -> HybridScreener:
    """
    Get or create singleton HybridScreener instance.

    Args:
        ibkr_client: Optional IBKR client for validation

    Returns:
        Initialized HybridScreener
    """
    global _hybrid_screener

    if _hybrid_screener is None:
        _hybrid_screener = HybridScreener(ibkr_client=ibkr_client)
        await _hybrid_screener.initialize()
    elif ibkr_client and _hybrid_screener.ibkr is None:
        _hybrid_screener.ibkr = ibkr_client

    return _hybrid_screener
