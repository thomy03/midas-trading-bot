"""
Market Context - Overall market regime detection and awareness.

Analyzes:
- Market regime (bull/bear/range)
- VIX levels (fear/greed)
- Market breadth (advance/decline)
- Sector rotation
- Correlation analysis

Provides context to adjust trading decisions.
"""

import os
import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """Current market regime"""
    STRONG_BULL = "strong_bull"      # Strong uptrend, low volatility
    BULL = "bull"                     # Uptrend
    RANGE = "range"                   # Sideways, consolidation
    BEAR = "bear"                     # Downtrend
    STRONG_BEAR = "strong_bear"       # Strong downtrend, high volatility
    CRASH = "crash"                   # Extreme selling, panic


class VolatilityRegime(str, Enum):
    """Volatility environment"""
    LOW = "low"           # VIX < 15
    NORMAL = "normal"     # VIX 15-20
    ELEVATED = "elevated" # VIX 20-30
    HIGH = "high"         # VIX 30-40
    EXTREME = "extreme"   # VIX > 40


class RiskAppetite(str, Enum):
    """Market risk appetite"""
    RISK_ON = "risk_on"       # Growth, small caps outperform
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"     # Defensive, large caps outperform


@dataclass
class MarketBreadth:
    """Market breadth indicators"""
    advance_decline_ratio: float = 1.0
    new_highs: int = 0
    new_lows: int = 0
    percent_above_200ma: float = 0.5
    percent_above_50ma: float = 0.5
    mcclellan_oscillator: float = 0.0

    @property
    def is_healthy(self) -> bool:
        return (self.advance_decline_ratio > 1.0 and
                self.percent_above_200ma > 0.5)


@dataclass
class SectorPerformance:
    """Sector relative performance"""
    sector: str
    return_1d: float = 0.0
    return_5d: float = 0.0
    return_1m: float = 0.0
    relative_strength: float = 0.0  # vs SPY

    @property
    def is_leading(self) -> bool:
        return self.relative_strength > 0.02


@dataclass
class MarketContext:
    """Complete market context snapshot"""
    timestamp: str

    # Regime
    regime: MarketRegime
    volatility: VolatilityRegime
    risk_appetite: RiskAppetite

    # Key indices
    spy_price: float = 0.0
    spy_change_1d: float = 0.0
    spy_change_5d: float = 0.0
    spy_trend: str = "neutral"  # "up", "down", "neutral"

    # Volatility
    vix_level: float = 20.0
    vix_change: float = 0.0
    vix_percentile: float = 0.5  # 0-1, where current VIX sits historically

    # Breadth
    breadth: MarketBreadth = field(default_factory=MarketBreadth)

    # Sectors (top 3 and bottom 3)
    leading_sectors: List[SectorPerformance] = field(default_factory=list)
    lagging_sectors: List[SectorPerformance] = field(default_factory=list)

    # Risk assessment
    risk_score: float = 0.5  # 0 = extreme risk, 1 = low risk

    # Trading implications
    position_size_multiplier: float = 1.0  # Adjust position sizing
    recommended_actions: List[str] = field(default_factory=list)

    # V8: Macro signals
    yield_curve_spread: Optional[float] = None
    credit_spread_momentum: Optional[float] = None
    dollar_momentum: Optional[float] = None
    defensive_rotation: Optional[float] = None
    macro_regime_score: Optional[float] = None  # -1 (BEAR) to +1 (BULL)
    macro_regime_bias: Optional[str] = None     # bullish/bearish/neutral

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'regime': self.regime.value,
            'volatility': self.volatility.value,
            'risk_appetite': self.risk_appetite.value,
            'spy': {
                'price': self.spy_price,
                'change_1d': self.spy_change_1d,
                'change_5d': self.spy_change_5d,
                'trend': self.spy_trend
            },
            'vix': {
                'level': self.vix_level,
                'change': self.vix_change,
                'percentile': self.vix_percentile
            },
            'risk_score': self.risk_score,
            'position_size_multiplier': self.position_size_multiplier,
            'recommended_actions': self.recommended_actions,
            'macro': {
                'yield_curve_spread': self.yield_curve_spread,
                'credit_spread_momentum': self.credit_spread_momentum,
                'dollar_momentum': self.dollar_momentum,
                'defensive_rotation': self.defensive_rotation,
                'macro_regime_score': self.macro_regime_score,
                'macro_regime_bias': self.macro_regime_bias,
            }
        }

    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = [
            f"Market Context ({self.timestamp[:10]})",
            f"Regime: {self.regime.value.replace('_', ' ').upper()}",
            f"VIX: {self.vix_level:.1f} ({self.volatility.value})",
            f"SPY: ${self.spy_price:.2f} ({self.spy_change_1d:+.2f}%)",
            f"Risk Score: {self.risk_score:.0%}",
            f"Position Sizing: {self.position_size_multiplier:.0%}"
        ]
        if self.recommended_actions:
            lines.append("Actions: " + ", ".join(self.recommended_actions[:2]))
        return "\n".join(lines)


class MarketContextAnalyzer:
    """
    Analyzes overall market conditions to provide context for trading decisions.
    """

    # Sector ETFs for rotation analysis
    SECTOR_ETFS = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLU': 'Utilities',
        'XLB': 'Materials',
        'XLRE': 'Real Estate',
        'XLC': 'Communication Services'
    }

    def __init__(self, enable_macro=True):
        self._cache: Optional[MarketContext] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = 300  # 5 minutes
        self._enable_macro = enable_macro
        self._macro_fetcher = None

    async def get_context(self, force_refresh: bool = False) -> MarketContext:
        """
        Get current market context.

        Args:
            force_refresh: Bypass cache

        Returns:
            MarketContext with current conditions
        """
        # Check cache
        if not force_refresh and self._cache and self._cache_time:
            age = (datetime.now() - self._cache_time).total_seconds()
            if age < self._cache_ttl:
                return self._cache

        logger.info("Analyzing market context...")

        try:
            # V8: Fetch macro signals in parallel with existing data
            fetch_tasks = [
                self._fetch_spy_data(),
                self._fetch_vix_data(),
                self._fetch_sector_data(),
            ]
            if self._enable_macro:
                fetch_tasks.append(self._fetch_macro_data())

            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            spy_data = results[0] if not isinstance(results[0], Exception) else {}
            vix_data = results[1] if not isinstance(results[1], Exception) else {}
            sector_data = results[2] if not isinstance(results[2], Exception) else []
            macro_data = None
            if self._enable_macro and len(results) > 3:
                macro_data = results[3] if not isinstance(results[3], Exception) else None

            if isinstance(spy_data, Exception):
                logger.warning(f"SPY data fetch failed: {spy_data}")
                spy_data = {}
            if isinstance(vix_data, Exception):
                logger.warning(f"VIX data fetch failed: {vix_data}")
                vix_data = {}
            if isinstance(sector_data, Exception):
                logger.warning(f"Sector data fetch failed: {sector_data}")
                sector_data = []

            # Determine regime (V8: with optional macro input)
            regime = self._determine_regime(spy_data, vix_data, macro_data=macro_data)

            # Determine volatility regime
            volatility = self._determine_volatility_regime(vix_data)

            # Determine risk appetite
            risk_appetite = self._determine_risk_appetite(sector_data, spy_data)

            # Calculate breadth (simplified)
            breadth = self._calculate_breadth(spy_data)

            # Process sectors
            leading, lagging = self._process_sectors(sector_data)

            # Calculate risk score
            risk_score = self._calculate_risk_score(regime, volatility, vix_data)

            # Determine position size multiplier
            position_mult = self._calculate_position_multiplier(risk_score, volatility)

            # Generate recommendations
            recommendations = self._generate_recommendations(regime, volatility, risk_appetite)

            # V8: Extract macro fields
            macro_kwargs = {}
            if macro_data is not None:
                if macro_data.yield_curve is not None:
                    macro_kwargs['yield_curve_spread'] = macro_data.yield_curve.value
                if macro_data.credit_spread is not None:
                    macro_kwargs['credit_spread_momentum'] = macro_data.credit_spread.value
                if macro_data.dollar_strength is not None:
                    macro_kwargs['dollar_momentum'] = macro_data.dollar_strength.value
                if macro_data.defensive_rotation is not None:
                    macro_kwargs['defensive_rotation'] = macro_data.defensive_rotation.value
                macro_kwargs['macro_regime_score'] = macro_data.score
                macro_kwargs['macro_regime_bias'] = macro_data.bias

            context = MarketContext(
                timestamp=datetime.now().isoformat(),
                regime=regime,
                volatility=volatility,
                risk_appetite=risk_appetite,
                spy_price=spy_data.get('price', 0),
                spy_change_1d=spy_data.get('change_1d', 0),
                spy_change_5d=spy_data.get('change_5d', 0),
                spy_trend=spy_data.get('trend', 'neutral'),
                vix_level=vix_data.get('level', 20),
                vix_change=vix_data.get('change', 0),
                vix_percentile=vix_data.get('percentile', 0.5),
                breadth=breadth,
                leading_sectors=leading,
                lagging_sectors=lagging,
                risk_score=risk_score,
                position_size_multiplier=position_mult,
                recommended_actions=recommendations,
                **macro_kwargs
            )

            # Cache result
            self._cache = context
            self._cache_time = datetime.now()

            return context

        except Exception as e:
            logger.error(f"Market context analysis failed: {e}")
            # Return default context
            return MarketContext(
                timestamp=datetime.now().isoformat(),
                regime=MarketRegime.RANGE,
                volatility=VolatilityRegime.NORMAL,
                risk_appetite=RiskAppetite.NEUTRAL
            )

    async def _fetch_spy_data(self) -> Dict[str, Any]:
        """Fetch SPY index data"""
        loop = asyncio.get_event_loop()

        def fetch():
            ticker = yf.Ticker('SPY')
            hist = ticker.history(period='3mo')

            if hist.empty:
                return {}

            price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else price
            prev_5d = hist['Close'].iloc[-6] if len(hist) > 5 else price

            change_1d = ((price - prev_close) / prev_close) * 100
            change_5d = ((price - prev_5d) / prev_5d) * 100

            # Calculate trend using EMAs
            ema20 = hist['Close'].ewm(span=20).mean().iloc[-1]
            ema50 = hist['Close'].ewm(span=50).mean().iloc[-1]

            if price > ema20 > ema50:
                trend = 'up'
            elif price < ema20 < ema50:
                trend = 'down'
            else:
                trend = 'neutral'

            # 200-day for regime
            ema200 = hist['Close'].ewm(span=200, min_periods=50).mean().iloc[-1] if len(hist) > 50 else None

            return {
                'price': price,
                'change_1d': change_1d,
                'change_5d': change_5d,
                'trend': trend,
                'ema20': ema20,
                'ema50': ema50,
                'ema200': ema200,
                'hist': hist
            }

        return await loop.run_in_executor(None, fetch)

    async def _fetch_vix_data(self) -> Dict[str, Any]:
        """Fetch VIX volatility index data"""
        loop = asyncio.get_event_loop()

        def fetch():
            ticker = yf.Ticker('^VIX')
            hist = ticker.history(period='1y')

            if hist.empty:
                return {'level': 20, 'change': 0, 'percentile': 0.5}

            level = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2] if len(hist) > 1 else level
            change = ((level - prev) / prev) * 100

            # Calculate percentile (where current VIX sits in 1-year range)
            all_values = hist['Close'].values
            percentile = (all_values < level).sum() / len(all_values)

            return {
                'level': level,
                'change': change,
                'percentile': percentile,
                'avg': hist['Close'].mean(),
                'high': hist['Close'].max(),
                'low': hist['Close'].min()
            }

        return await loop.run_in_executor(None, fetch)

    async def _fetch_sector_data(self) -> List[Dict[str, Any]]:
        """Fetch sector ETF performance data"""
        loop = asyncio.get_event_loop()

        def fetch():
            sectors = []

            # Get SPY for relative strength
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(period='1mo')
            spy_return_1m = 0
            if not spy_hist.empty and len(spy_hist) > 1:
                spy_return_1m = ((spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[0])
                                 / spy_hist['Close'].iloc[0]) * 100

            for symbol, name in self.SECTOR_ETFS.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='1mo')

                    if hist.empty or len(hist) < 2:
                        continue

                    price = hist['Close'].iloc[-1]
                    prev_1d = hist['Close'].iloc[-2]
                    prev_5d = hist['Close'].iloc[-6] if len(hist) > 5 else hist['Close'].iloc[0]
                    prev_1m = hist['Close'].iloc[0]

                    return_1d = ((price - prev_1d) / prev_1d) * 100
                    return_5d = ((price - prev_5d) / prev_5d) * 100
                    return_1m = ((price - prev_1m) / prev_1m) * 100

                    relative_strength = return_1m - spy_return_1m

                    sectors.append({
                        'symbol': symbol,
                        'name': name,
                        'return_1d': return_1d,
                        'return_5d': return_5d,
                        'return_1m': return_1m,
                        'relative_strength': relative_strength
                    })
                except Exception as e:
                    logger.debug(f"Error fetching {symbol}: {e}")
                    continue

            return sectors

        return await loop.run_in_executor(None, fetch)

    async def _fetch_macro_data(self):
        """V8: Fetch macro signal data."""
        loop = asyncio.get_event_loop()

        def fetch():
            if self._macro_fetcher is None:
                from .macro_signals import MacroSignalFetcher
                self._macro_fetcher = MacroSignalFetcher()
            return self._macro_fetcher.get_macro_regime_score()

        return await loop.run_in_executor(None, fetch)

    def _determine_regime(self, spy_data: Dict, vix_data: Dict, macro_data=None) -> MarketRegime:
        """Determine current market regime.

        V8: Blends technical regime (70%) with macro signals (30%) when available.
        """
        if not spy_data:
            return MarketRegime.RANGE

        trend = spy_data.get('trend', 'neutral')
        change_5d = spy_data.get('change_5d', 0)
        vix = vix_data.get('level', 20)
        price = spy_data.get('price', 0)
        ema200 = spy_data.get('ema200')

        # Crash detection (override - no blending)
        if change_5d < -10 or vix > 40:
            return MarketRegime.CRASH

        # Technical regime score: map to numeric
        tech_score = 0.0  # neutral
        if trend == 'down' and vix > 30:
            tech_score = -1.0  # strong bear
        elif trend == 'down' or (ema200 and price < ema200):
            tech_score = -0.5  # bear
        elif trend == 'up' and vix < 15 and change_5d > 2:
            tech_score = 1.0  # strong bull
        elif trend == 'up' or (ema200 and price > ema200 * 1.05):
            tech_score = 0.5  # bull

        # V8: Blend with macro if available (30% macro, 70% technical)
        if macro_data is not None and hasattr(macro_data, 'score') and macro_data.confidence > 0.3:
            blended = tech_score * 0.70 + macro_data.score * 0.30
        else:
            blended = tech_score

        # Map blended score back to regime
        if blended <= -0.75:
            return MarketRegime.STRONG_BEAR
        elif blended <= -0.25:
            return MarketRegime.BEAR
        elif blended >= 0.75:
            return MarketRegime.STRONG_BULL
        elif blended >= 0.25:
            return MarketRegime.BULL

        return MarketRegime.RANGE

    def _determine_volatility_regime(self, vix_data: Dict) -> VolatilityRegime:
        """Determine volatility environment"""
        vix = vix_data.get('level', 20)

        if vix > 40:
            return VolatilityRegime.EXTREME
        elif vix > 30:
            return VolatilityRegime.HIGH
        elif vix > 20:
            return VolatilityRegime.ELEVATED
        elif vix > 15:
            return VolatilityRegime.NORMAL
        else:
            return VolatilityRegime.LOW

    def _determine_risk_appetite(
        self,
        sector_data: List[Dict],
        spy_data: Dict
    ) -> RiskAppetite:
        """Determine market risk appetite based on sector rotation"""
        if not sector_data:
            return RiskAppetite.NEUTRAL

        # Risk-on sectors: XLK (Tech), XLY (Consumer Disc), XLF (Financials)
        # Risk-off sectors: XLU (Utilities), XLP (Consumer Staples), XLV (Healthcare)

        risk_on_sectors = {'XLK', 'XLY', 'XLF'}
        risk_off_sectors = {'XLU', 'XLP', 'XLV'}

        risk_on_strength = 0
        risk_off_strength = 0

        for sector in sector_data:
            symbol = sector.get('symbol', '')
            rs = sector.get('relative_strength', 0)

            if symbol in risk_on_sectors:
                risk_on_strength += rs
            elif symbol in risk_off_sectors:
                risk_off_strength += rs

        if risk_on_strength > risk_off_strength + 2:
            return RiskAppetite.RISK_ON
        elif risk_off_strength > risk_on_strength + 2:
            return RiskAppetite.RISK_OFF
        else:
            return RiskAppetite.NEUTRAL

    def _calculate_breadth(self, spy_data: Dict) -> MarketBreadth:
        """Calculate market breadth (simplified version)"""
        # This is simplified - real implementation would use
        # advance/decline data from exchanges

        hist = spy_data.get('hist')
        if hist is None or hist.empty:
            return MarketBreadth()

        # Estimate using SPY components' behavior via price action
        closes = hist['Close']
        ema50 = closes.ewm(span=50).mean()
        ema200 = closes.ewm(span=200, min_periods=50).mean()

        # Rough estimate
        recent_above_50 = (closes.iloc[-20:] > ema50.iloc[-20:]).mean()
        recent_above_200 = (closes.iloc[-20:] > ema200.iloc[-20:]).mean() if len(closes) > 50 else 0.5

        return MarketBreadth(
            advance_decline_ratio=1.0 + (recent_above_50 - 0.5),
            percent_above_50ma=recent_above_50,
            percent_above_200ma=recent_above_200
        )

    def _process_sectors(
        self,
        sector_data: List[Dict]
    ) -> Tuple[List[SectorPerformance], List[SectorPerformance]]:
        """Process sectors into leading and lagging"""
        if not sector_data:
            return [], []

        sectors = [
            SectorPerformance(
                sector=s.get('name', ''),
                return_1d=s.get('return_1d', 0),
                return_5d=s.get('return_5d', 0),
                return_1m=s.get('return_1m', 0),
                relative_strength=s.get('relative_strength', 0)
            )
            for s in sector_data
        ]

        # Sort by relative strength
        sectors.sort(key=lambda x: x.relative_strength, reverse=True)

        return sectors[:3], sectors[-3:]

    def _calculate_risk_score(
        self,
        regime: MarketRegime,
        volatility: VolatilityRegime,
        vix_data: Dict
    ) -> float:
        """Calculate overall risk score (0 = high risk, 1 = low risk)"""
        score = 0.5

        # Regime contribution
        regime_scores = {
            MarketRegime.STRONG_BULL: 0.9,
            MarketRegime.BULL: 0.75,
            MarketRegime.RANGE: 0.5,
            MarketRegime.BEAR: 0.3,
            MarketRegime.STRONG_BEAR: 0.15,
            MarketRegime.CRASH: 0.0
        }
        score = regime_scores.get(regime, 0.5)

        # Volatility adjustment
        volatility_adj = {
            VolatilityRegime.LOW: 0.1,
            VolatilityRegime.NORMAL: 0.05,
            VolatilityRegime.ELEVATED: -0.05,
            VolatilityRegime.HIGH: -0.15,
            VolatilityRegime.EXTREME: -0.25
        }
        score += volatility_adj.get(volatility, 0)

        # VIX spike adjustment
        vix_change = vix_data.get('change', 0)
        if vix_change > 20:  # VIX spiked 20%+
            score -= 0.1

        return max(0, min(1, score))

    def _calculate_position_multiplier(
        self,
        risk_score: float,
        volatility: VolatilityRegime
    ) -> float:
        """Calculate position size multiplier based on conditions"""
        # Base multiplier from risk score
        multiplier = 0.5 + (risk_score * 0.5)  # 0.5 to 1.0

        # Volatility adjustment
        vol_adj = {
            VolatilityRegime.LOW: 1.1,
            VolatilityRegime.NORMAL: 1.0,
            VolatilityRegime.ELEVATED: 0.8,
            VolatilityRegime.HIGH: 0.6,
            VolatilityRegime.EXTREME: 0.4
        }
        multiplier *= vol_adj.get(volatility, 1.0)

        return max(0.25, min(1.25, multiplier))

    def _generate_recommendations(
        self,
        regime: MarketRegime,
        volatility: VolatilityRegime,
        risk_appetite: RiskAppetite
    ) -> List[str]:
        """Generate actionable recommendations"""
        recs = []

        # Regime-based
        if regime == MarketRegime.CRASH:
            recs.append("HALT new positions - wait for stabilization")
            recs.append("Review stop-losses on all positions")
        elif regime == MarketRegime.STRONG_BEAR:
            recs.append("Reduce position sizes")
            recs.append("Focus on defensive sectors")
        elif regime == MarketRegime.BEAR:
            recs.append("Be selective - quality over quantity")
        elif regime == MarketRegime.STRONG_BULL:
            recs.append("Full position sizing OK")
            recs.append("Trail stops on winners")
        elif regime == MarketRegime.BULL:
            recs.append("Normal trading conditions")

        # Volatility-based
        if volatility in [VolatilityRegime.HIGH, VolatilityRegime.EXTREME]:
            recs.append("Widen stops to avoid whipsaws")
            recs.append("Consider smaller positions")

        # Risk appetite-based
        if risk_appetite == RiskAppetite.RISK_ON:
            recs.append("Growth and tech may outperform")
        elif risk_appetite == RiskAppetite.RISK_OFF:
            recs.append("Favor defensive and value")

        return recs[:4]  # Max 4 recommendations


# Singleton
_context_analyzer: Optional[MarketContextAnalyzer] = None


async def get_market_context(force_refresh: bool = False) -> MarketContext:
    """Get current market context"""
    global _context_analyzer
    if _context_analyzer is None:
        _context_analyzer = MarketContextAnalyzer()
    return await _context_analyzer.get_context(force_refresh)


def get_market_context_analyzer() -> MarketContextAnalyzer:
    """Get the MarketContextAnalyzer instance"""
    global _context_analyzer
    if _context_analyzer is None:
        _context_analyzer = MarketContextAnalyzer()
    return _context_analyzer
