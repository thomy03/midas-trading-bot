"""
Fundamental Pillar - Fundamental analysis using financial metrics.

Analyzes:
- Valuation (P/E, P/B, P/S, PEG)
- Growth (Revenue, Earnings, EPS growth)
- Profitability (Margins, ROE, ROA)
- Financial health (Debt, Current ratio)
- Dividends (Yield, Payout ratio)

Contributes 25% to final decision score.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import asyncio

from .base import BasePillar, PillarScore

logger = logging.getLogger(__name__)


class FundamentalPillar(BasePillar):
    """
    Fundamental analysis pillar.

    Uses yfinance to fetch company fundamentals and score
    based on valuation, growth, profitability, and health.
    """

    def __init__(self, weight: float = 0.25):
        super().__init__(weight)
        self._cache_ttl = 3600  # 1 hour cache for fundamentals

        # Category weights
        self.category_weights = {
            'valuation': 0.30,
            'growth': 0.30,
            'profitability': 0.25,
            'health': 0.15
        }

        # Sector average P/E ratios (rough estimates)
        self.sector_pe = {
            'Technology': 30,
            'Healthcare': 25,
            'Financial Services': 15,
            'Consumer Cyclical': 20,
            'Consumer Defensive': 22,
            'Energy': 12,
            'Industrials': 18,
            'Basic Materials': 14,
            'Utilities': 18,
            'Real Estate': 35,
            'Communication Services': 22,
            'default': 20
        }

    def get_name(self) -> str:
        return "Fundamental"

    async def analyze(
        self,
        symbol: str,
        data: Dict[str, Any]
    ) -> PillarScore:
        """
        Perform fundamental analysis on a symbol.

        Args:
            symbol: Stock symbol
            data: Optional pre-fetched fundamentals

        Returns:
            PillarScore with fundamental analysis result
        """
        try:
            # Fetch fundamentals if not provided
            fundamentals = data.get('fundamentals')
            if fundamentals is None:
                fundamentals = await self._fetch_fundamentals(symbol)

            if not fundamentals:
                logger.warning(f"[FUNDAMENTAL] {symbol}: No fundamental data available")
                return self._create_score(
                    score=0,
                    reasoning=f"Unable to fetch fundamentals for {symbol}",
                    data_quality=0.0
                )

            logger.info(f"[FUNDAMENTAL] {symbol}: Analyzing fundamentals (sector: {fundamentals.get('sector', 'Unknown')})...")

            factors = []
            category_scores = {}
            data_points = 0
            total_possible = 0

            # 1. Valuation Analysis (30%)
            val_score, val_factors, val_data = self._analyze_valuation(fundamentals)
            category_scores['valuation'] = val_score
            factors.extend(val_factors)
            data_points += val_data[0]
            total_possible += val_data[1]

            # 2. Growth Analysis (30%)
            growth_score, growth_factors, growth_data = self._analyze_growth(fundamentals)
            category_scores['growth'] = growth_score
            factors.extend(growth_factors)
            data_points += growth_data[0]
            total_possible += growth_data[1]

            # 3. Profitability Analysis (25%)
            prof_score, prof_factors, prof_data = self._analyze_profitability(fundamentals)
            category_scores['profitability'] = prof_score
            factors.extend(prof_factors)
            data_points += prof_data[0]
            total_possible += prof_data[1]

            # 4. Financial Health (15%)
            health_score, health_factors, health_data = self._analyze_health(fundamentals)
            category_scores['health'] = health_score
            factors.extend(health_factors)
            data_points += health_data[0]
            total_possible += health_data[1]

            # Calculate weighted total
            total_score = sum(
                category_scores[cat] * self.category_weights[cat]
                for cat in category_scores
            )

            # Data quality
            data_quality = data_points / total_possible if total_possible > 0 else 0.5

            # Generate reasoning
            reasoning = self._generate_reasoning(
                symbol, fundamentals, category_scores, factors
            )

            # V4.4: Verbose logging
            signal = "strong" if total_score > 30 else "good" if total_score > 0 else "weak" if total_score > -30 else "poor"
            logger.info(f"[FUNDAMENTAL] {symbol}: Score={total_score:.1f}/100 ({signal}) | Valuation={category_scores.get('valuation', 0):.0f} Growth={category_scores.get('growth', 0):.0f} Profit={category_scores.get('profitability', 0):.0f} Health={category_scores.get('health', 0):.0f} | Quality={data_quality:.0%}")

            return self._create_score(
                score=total_score,
                reasoning=reasoning,
                factors=factors,
                confidence=0.7,  # Fundamentals are less dynamic
                data_quality=data_quality
            )

        except Exception as e:
            logger.error(f"Fundamental analysis failed for {symbol}: {e}")
            return self._create_score(
                score=0,
                reasoning=f"Analysis error: {str(e)}",
                data_quality=0.0
            )

    async def _fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, lambda: ticker.info)

            return info or {}

        except Exception as e:
            logger.warning(f"Failed to fetch fundamentals for {symbol}: {e}")
            return {}

    def _analyze_valuation(self, info: Dict) -> tuple[float, List[Dict], tuple[int, int]]:
        """Analyze valuation metrics"""
        factors = []
        scores = []
        data_available = 0
        data_total = 4  # PE, PB, PS, PEG

        sector = info.get('sector', 'default')
        sector_pe = self.sector_pe.get(sector, self.sector_pe['default'])

        # P/E Ratio
        pe = info.get('trailingPE') or info.get('forwardPE')
        if pe is not None and pe > 0:
            data_available += 1
            if pe < sector_pe * 0.6:
                pe_score = 70
                pe_msg = f"Undervalued P/E ({pe:.1f} vs sector {sector_pe})"
            elif pe < sector_pe * 0.9:
                pe_score = 40
                pe_msg = f"Fair P/E ({pe:.1f})"
            elif pe < sector_pe * 1.3:
                pe_score = 0
                pe_msg = f"Slightly overvalued ({pe:.1f})"
            elif pe < sector_pe * 2:
                pe_score = -40
                pe_msg = f"Overvalued P/E ({pe:.1f})"
            else:
                pe_score = -70
                pe_msg = f"Very overvalued ({pe:.1f})"

            factors.append({
                'metric': 'P/E',
                'score': pe_score,
                'message': pe_msg,
                'value': pe
            })
            scores.append(pe_score)

        # P/B Ratio
        pb = info.get('priceToBook')
        if pb is not None and pb > 0:
            data_available += 1
            if pb < 1:
                pb_score = 60
                pb_msg = f"Below book value (P/B={pb:.2f})"
            elif pb < 2:
                pb_score = 30
                pb_msg = f"Reasonable P/B ({pb:.2f})"
            elif pb < 4:
                pb_score = 0
                pb_msg = f"Above book value ({pb:.2f})"
            else:
                pb_score = -40
                pb_msg = f"High P/B ratio ({pb:.2f})"

            factors.append({
                'metric': 'P/B',
                'score': pb_score,
                'message': pb_msg,
                'value': pb
            })
            scores.append(pb_score)

        # P/S Ratio
        ps = info.get('priceToSalesTrailing12Months')
        if ps is not None and ps > 0:
            data_available += 1
            if ps < 1:
                ps_score = 50
                ps_msg = f"Low P/S ratio ({ps:.2f})"
            elif ps < 3:
                ps_score = 20
                ps_msg = f"Reasonable P/S ({ps:.2f})"
            elif ps < 10:
                ps_score = -20
                ps_msg = f"High P/S ({ps:.2f})"
            else:
                ps_score = -50
                ps_msg = f"Very high P/S ({ps:.2f})"

            factors.append({
                'metric': 'P/S',
                'score': ps_score,
                'message': ps_msg,
                'value': ps
            })
            scores.append(ps_score)

        # PEG Ratio
        peg = info.get('pegRatio')
        if peg is not None and peg > 0:
            data_available += 1
            if peg < 1:
                peg_score = 60
                peg_msg = f"Undervalued growth (PEG={peg:.2f})"
            elif peg < 1.5:
                peg_score = 30
                peg_msg = f"Fair growth value (PEG={peg:.2f})"
            elif peg < 2:
                peg_score = 0
                peg_msg = f"Slight premium (PEG={peg:.2f})"
            else:
                peg_score = -40
                peg_msg = f"Growth overpriced (PEG={peg:.2f})"

            factors.append({
                'metric': 'PEG',
                'score': peg_score,
                'message': peg_msg,
                'value': peg
            })
            scores.append(peg_score)

        avg_score = np.mean(scores) if scores else 0
        return avg_score, factors, (data_available, data_total)

    def _analyze_growth(self, info: Dict) -> tuple[float, List[Dict], tuple[int, int]]:
        """Analyze growth metrics"""
        factors = []
        scores = []
        data_available = 0
        data_total = 3

        # Revenue Growth
        rev_growth = info.get('revenueGrowth')
        if rev_growth is not None:
            data_available += 1
            rev_pct = rev_growth * 100
            if rev_growth > 0.30:
                rev_score = 70
                rev_msg = f"Strong revenue growth ({rev_pct:.1f}%)"
            elif rev_growth > 0.15:
                rev_score = 50
                rev_msg = f"Good revenue growth ({rev_pct:.1f}%)"
            elif rev_growth > 0.05:
                rev_score = 20
                rev_msg = f"Moderate growth ({rev_pct:.1f}%)"
            elif rev_growth > 0:
                rev_score = 0
                rev_msg = f"Slow growth ({rev_pct:.1f}%)"
            else:
                rev_score = -50
                rev_msg = f"Revenue declining ({rev_pct:.1f}%)"

            factors.append({
                'metric': 'Revenue Growth',
                'score': rev_score,
                'message': rev_msg,
                'value': rev_growth
            })
            scores.append(rev_score)

        # Earnings Growth
        earnings_growth = info.get('earningsGrowth')
        if earnings_growth is not None:
            data_available += 1
            earn_pct = earnings_growth * 100
            if earnings_growth > 0.25:
                earn_score = 60
                earn_msg = f"Strong earnings growth ({earn_pct:.1f}%)"
            elif earnings_growth > 0.10:
                earn_score = 40
                earn_msg = f"Good earnings growth ({earn_pct:.1f}%)"
            elif earnings_growth > 0:
                earn_score = 10
                earn_msg = f"Modest earnings growth ({earn_pct:.1f}%)"
            else:
                earn_score = -50
                earn_msg = f"Earnings declining ({earn_pct:.1f}%)"

            factors.append({
                'metric': 'Earnings Growth',
                'score': earn_score,
                'message': earn_msg,
                'value': earnings_growth
            })
            scores.append(earn_score)

        # EPS
        eps = info.get('trailingEps')
        forward_eps = info.get('forwardEps')
        if eps is not None and forward_eps is not None and eps != 0:
            data_available += 1
            eps_growth = (forward_eps - eps) / abs(eps)
            eps_pct = eps_growth * 100
            if eps_growth > 0.20:
                eps_score = 50
                eps_msg = f"EPS expected to grow {eps_pct:.1f}%"
            elif eps_growth > 0.05:
                eps_score = 25
                eps_msg = f"Moderate EPS growth expected ({eps_pct:.1f}%)"
            elif eps_growth > -0.10:
                eps_score = 0
                eps_msg = f"Flat EPS expected ({eps_pct:.1f}%)"
            else:
                eps_score = -40
                eps_msg = f"EPS expected to decline ({eps_pct:.1f}%)"

            factors.append({
                'metric': 'EPS Forecast',
                'score': eps_score,
                'message': eps_msg,
                'value': eps_growth
            })
            scores.append(eps_score)

        avg_score = np.mean(scores) if scores else 0
        return avg_score, factors, (data_available, data_total)

    def _analyze_profitability(self, info: Dict) -> tuple[float, List[Dict], tuple[int, int]]:
        """Analyze profitability metrics"""
        factors = []
        scores = []
        data_available = 0
        data_total = 3

        # Profit Margin
        margin = info.get('profitMargins')
        if margin is not None:
            data_available += 1
            margin_pct = margin * 100
            if margin > 0.20:
                margin_score = 60
                margin_msg = f"Excellent margin ({margin_pct:.1f}%)"
            elif margin > 0.10:
                margin_score = 40
                margin_msg = f"Good margin ({margin_pct:.1f}%)"
            elif margin > 0.05:
                margin_score = 20
                margin_msg = f"Average margin ({margin_pct:.1f}%)"
            elif margin > 0:
                margin_score = 0
                margin_msg = f"Low margin ({margin_pct:.1f}%)"
            else:
                margin_score = -50
                margin_msg = f"Negative margin ({margin_pct:.1f}%)"

            factors.append({
                'metric': 'Profit Margin',
                'score': margin_score,
                'message': margin_msg,
                'value': margin
            })
            scores.append(margin_score)

        # ROE
        roe = info.get('returnOnEquity')
        if roe is not None:
            data_available += 1
            roe_pct = roe * 100
            if roe > 0.20:
                roe_score = 60
                roe_msg = f"Excellent ROE ({roe_pct:.1f}%)"
            elif roe > 0.15:
                roe_score = 40
                roe_msg = f"Good ROE ({roe_pct:.1f}%)"
            elif roe > 0.08:
                roe_score = 20
                roe_msg = f"Average ROE ({roe_pct:.1f}%)"
            elif roe > 0:
                roe_score = 0
                roe_msg = f"Low ROE ({roe_pct:.1f}%)"
            else:
                roe_score = -40
                roe_msg = f"Negative ROE ({roe_pct:.1f}%)"

            factors.append({
                'metric': 'ROE',
                'score': roe_score,
                'message': roe_msg,
                'value': roe
            })
            scores.append(roe_score)

        # Operating Margin
        op_margin = info.get('operatingMargins')
        if op_margin is not None:
            data_available += 1
            op_pct = op_margin * 100
            if op_margin > 0.25:
                op_score = 50
                op_msg = f"Strong operating margin ({op_pct:.1f}%)"
            elif op_margin > 0.15:
                op_score = 30
                op_msg = f"Good operating margin ({op_pct:.1f}%)"
            elif op_margin > 0.05:
                op_score = 10
                op_msg = f"Average operating margin ({op_pct:.1f}%)"
            else:
                op_score = -30
                op_msg = f"Weak operating margin ({op_pct:.1f}%)"

            factors.append({
                'metric': 'Operating Margin',
                'score': op_score,
                'message': op_msg,
                'value': op_margin
            })
            scores.append(op_score)

        avg_score = np.mean(scores) if scores else 0
        return avg_score, factors, (data_available, data_total)

    def _analyze_health(self, info: Dict) -> tuple[float, List[Dict], tuple[int, int]]:
        """Analyze financial health metrics"""
        factors = []
        scores = []
        data_available = 0
        data_total = 2

        # Debt/Equity
        debt_equity = info.get('debtToEquity')
        if debt_equity is not None:
            data_available += 1
            if debt_equity < 30:
                de_score = 50
                de_msg = f"Low debt (D/E={debt_equity:.1f}%)"
            elif debt_equity < 80:
                de_score = 30
                de_msg = f"Moderate debt (D/E={debt_equity:.1f}%)"
            elif debt_equity < 150:
                de_score = 0
                de_msg = f"High debt (D/E={debt_equity:.1f}%)"
            else:
                de_score = -50
                de_msg = f"Very high debt (D/E={debt_equity:.1f}%)"

            factors.append({
                'metric': 'Debt/Equity',
                'score': de_score,
                'message': de_msg,
                'value': debt_equity
            })
            scores.append(de_score)

        # Current Ratio
        current_ratio = info.get('currentRatio')
        if current_ratio is not None:
            data_available += 1
            if current_ratio > 2:
                cr_score = 40
                cr_msg = f"Strong liquidity (CR={current_ratio:.2f})"
            elif current_ratio > 1.5:
                cr_score = 30
                cr_msg = f"Good liquidity (CR={current_ratio:.2f})"
            elif current_ratio > 1:
                cr_score = 10
                cr_msg = f"Adequate liquidity (CR={current_ratio:.2f})"
            else:
                cr_score = -40
                cr_msg = f"Liquidity concern (CR={current_ratio:.2f})"

            factors.append({
                'metric': 'Current Ratio',
                'score': cr_score,
                'message': cr_msg,
                'value': current_ratio
            })
            scores.append(cr_score)

        avg_score = np.mean(scores) if scores else 0
        return avg_score, factors, (data_available, data_total)

    def _generate_reasoning(
        self,
        symbol: str,
        info: Dict,
        category_scores: Dict[str, float],
        factors: List[Dict]
    ) -> str:
        """Generate human-readable reasoning"""
        parts = []

        # Company info
        name = info.get('shortName', symbol)
        sector = info.get('sector', 'Unknown')
        parts.append(f"Fundamental analysis for {name} ({sector})")

        # Overall assessment
        total = sum(category_scores.get(c, 0) * self.category_weights.get(c, 0.25)
                    for c in category_scores)

        if total > 40:
            parts.append("Fundamentals: STRONG")
        elif total > 15:
            parts.append("Fundamentals: GOOD")
        elif total > -15:
            parts.append("Fundamentals: AVERAGE")
        elif total > -40:
            parts.append("Fundamentals: WEAK")
        else:
            parts.append("Fundamentals: POOR")

        # Category summaries
        for cat, score in category_scores.items():
            quality = "strong" if score > 30 else "good" if score > 0 else "weak" if score > -30 else "poor"
            parts.append(f"- {cat.capitalize()}: {quality} ({score:+.0f})")

        # Key metrics
        key_factors = sorted(factors, key=lambda x: abs(x.get('score', 0)), reverse=True)[:3]
        if key_factors:
            parts.append("\nKey metrics:")
            for f in key_factors:
                parts.append(f"  * {f.get('message', 'N/A')}")

        return "\n".join(parts)


# Singleton
_fundamental_pillar: Optional[FundamentalPillar] = None


def get_fundamental_pillar() -> FundamentalPillar:
    """Get or create the FundamentalPillar singleton"""
    global _fundamental_pillar
    if _fundamental_pillar is None:
        _fundamental_pillar = FundamentalPillar()
    return _fundamental_pillar
