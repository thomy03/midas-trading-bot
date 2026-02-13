"""
Adaptive Fundamental Pillar V8.1 - Sector-relative scoring with 18 metrics.

Replaces the legacy fixed-threshold fundamental pillar with:
- 18 metrics across 6 categories (Valuation, Cash Flow, Growth, Profitability, Health, Quality)
- Percentile-based scoring relative to sector peers
- Trend detection (improving/stable/deteriorating)
- Regime-aware learned weights
- Full BasePillar interface compliance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import asyncio
import json
import os
from pathlib import Path

from .base import BasePillar, PillarScore
from .sector_stats import (
    get_sector_stats_manager,
    SectorStatsManager,
    normalize_sector,
)

logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent / "data"
WEIGHTS_FILE = DATA_DIR / "learned_weights" / "fundamental_weights.json"

# Metrics where LOWER values are BETTER (inverted percentile)
LOWER_IS_BETTER = {
    "pe_ratio", "pb_ratio", "ps_ratio", "peg_ratio", "debt_equity",
}

# All 18 metrics grouped by category
METRIC_CATEGORIES = {
    "valuation": ["pe_ratio", "pb_ratio", "ps_ratio", "peg_ratio"],
    "cash_flow": ["fcf_yield", "ocf_revenue_ratio", "fcf"],
    "growth": ["revenue_growth_yoy", "earnings_growth_yoy", "revenue_growth_qoq_trend"],
    "profitability": ["net_profit_margin", "roe", "operating_margin"],
    "health": ["debt_equity", "current_ratio", "interest_coverage"],
    "quality": ["earnings_surprise_avg", "margin_trend"],
}

# Default category weights by market regime
DEFAULT_REGIME_WEIGHTS = {
    "bull": {
        "valuation": 0.12,
        "cash_flow": 0.18,
        "growth": 0.28,
        "profitability": 0.20,
        "health": 0.10,
        "quality": 0.12,
    },
    "bear": {
        "valuation": 0.25,
        "cash_flow": 0.20,
        "growth": 0.10,
        "profitability": 0.15,
        "health": 0.20,
        "quality": 0.10,
    },
    "volatile": {
        "valuation": 0.15,
        "cash_flow": 0.22,
        "growth": 0.13,
        "profitability": 0.18,
        "health": 0.22,
        "quality": 0.10,
    },
    "neutral": {
        "valuation": 0.18,
        "cash_flow": 0.18,
        "growth": 0.18,
        "profitability": 0.18,
        "health": 0.16,
        "quality": 0.12,
    },
}

# V8.2 Sprint 3: Sector-specific weight ADJUSTMENTS (additive to regime weights)
# Rationale: Tech stocks → growth matters more, Energy → cash flow, Healthcare → quality/news
SECTOR_WEIGHT_ADJUSTMENTS = {
    "technology": {
        "growth": +0.08, "quality": +0.04,
        "valuation": -0.06, "health": -0.06,
    },
    "communication_services": {
        "growth": +0.06, "profitability": +0.04,
        "valuation": -0.05, "health": -0.05,
    },
    "healthcare": {
        "quality": +0.08, "cash_flow": +0.04,
        "growth": -0.04, "valuation": -0.08,
    },
    "energy": {
        "cash_flow": +0.10, "valuation": +0.04,
        "growth": -0.08, "quality": -0.06,
    },
    "financials": {
        "health": +0.08, "valuation": +0.06,
        "growth": -0.06, "cash_flow": -0.08,
    },
    "consumer_discretionary": {
        "growth": +0.06, "profitability": +0.04,
        "health": -0.04, "cash_flow": -0.06,
    },
    "consumer_staples": {
        "profitability": +0.06, "quality": +0.06,
        "growth": -0.08, "valuation": -0.04,
    },
    "industrials": {
        "cash_flow": +0.06, "profitability": +0.04,
        "quality": -0.04, "growth": -0.06,
    },
    "materials": {
        "cash_flow": +0.08, "valuation": +0.04,
        "growth": -0.06, "quality": -0.06,
    },
    "utilities": {
        "health": +0.08, "cash_flow": +0.06,
        "growth": -0.10, "quality": -0.04,
    },
    "real_estate": {
        "cash_flow": +0.10, "health": +0.06,
        "growth": -0.08, "quality": -0.08,
    },
}

# Map yfinance sector names to our keys
SECTOR_NAME_MAP = {
    "technology": "technology",
    "communication services": "communication_services",
    "healthcare": "healthcare",
    "energy": "energy",
    "financials": "financials",
    "financial services": "financials",
    "consumer cyclical": "consumer_discretionary",
    "consumer discretionary": "consumer_discretionary",
    "consumer defensive": "consumer_staples",
    "consumer staples": "consumer_staples",
    "industrials": "industrials",
    "basic materials": "materials",
    "materials": "materials",
    "utilities": "utilities",
    "real estate": "real_estate",
}


class AdaptiveFundamentalPillar(BasePillar):
    """
    Adaptive Fundamental Pillar with sector-relative percentile scoring.
    
    18 metrics scored against sector peers, with trend bonuses
    and regime-aware category weighting.
    """

    def __init__(self, weight: float = 0.25):
        super().__init__(weight)
        self._cache_ttl = 3600  # 1 hour
        self._sector_stats = get_sector_stats_manager()
        self._weights = self._load_weights()

    def get_name(self) -> str:
        return "Fundamental"

    def _load_weights(self) -> Dict:
        """Load learned weights or use defaults."""
        try:
            if WEIGHTS_FILE.exists():
                with open(WEIGHTS_FILE) as f:
                    w = json.load(f)
                logger.info(f"[FUNDAMENTAL] Loaded learned weights from {WEIGHTS_FILE}")
                return w
        except Exception as e:
            logger.warning(f"[FUNDAMENTAL] Failed to load weights: {e}")
        return DEFAULT_REGIME_WEIGHTS

    def _get_regime_weights(self, regime: str = "neutral", sector: str = "") -> Dict[str, float]:
        """Get category weights for current market regime + sector.
        V8.2 Sprint 3: Applies sector-specific adjustments on top of regime weights."""
        regime_lower = regime.lower() if regime else "neutral"
        # Map MarketRegime enum values
        regime_map = {
            "bullish": "bull", "bull": "bull",
            "bearish": "bear", "bear": "bear",
            "volatile": "volatile", "high_volatility": "volatile",
            "neutral": "neutral", "low_volatility": "neutral",
            "ranging": "neutral", "recovery": "bull",
        }
        key = regime_map.get(regime_lower, "neutral")
        base_weights = dict(self._weights.get(key, self._weights.get("neutral", DEFAULT_REGIME_WEIGHTS["neutral"])))

        # V8.2: Apply sector-specific adjustments
        sector_key = SECTOR_NAME_MAP.get(sector.lower(), "") if sector else ""
        if sector_key and sector_key in SECTOR_WEIGHT_ADJUSTMENTS:
            adjustments = SECTOR_WEIGHT_ADJUSTMENTS[sector_key]
            for cat, adj in adjustments.items():
                if cat in base_weights:
                    base_weights[cat] = max(0.05, base_weights[cat] + adj)
            # Re-normalize to sum to 1.0
            total = sum(base_weights.values())
            if total > 0:
                base_weights = {k: v / total for k, v in base_weights.items()}
            logger.debug(f"[FUNDAMENTAL] Sector-adjusted weights ({sector_key}): {base_weights}")

        return base_weights

    async def analyze(self, symbol: str, data: Dict[str, Any]) -> PillarScore:
        """Perform adaptive fundamental analysis."""
        try:
            # Get market regime from data context
            regime = "neutral"
            market_ctx = data.get("market_context")
            if market_ctx:
                if hasattr(market_ctx, 'regime'):
                    regime = str(market_ctx.regime.value) if hasattr(market_ctx.regime, 'value') else str(market_ctx.regime)
                elif isinstance(market_ctx, dict):
                    regime = market_ctx.get("regime", "neutral")

            # Fetch raw fundamentals
            fundamentals = data.get('fundamentals')
            if fundamentals is None:
                fundamentals = await self._fetch_fundamentals(symbol)

            if not fundamentals:
                logger.warning(f"[FUNDAMENTAL] {symbol}: No data available")
                return self._create_score(score=0, reasoning=f"No fundamental data for {symbol}", data_quality=0.0)

            sector_raw = fundamentals.get('sector', 'Unknown')
            sector = normalize_sector(sector_raw)
            name = fundamentals.get('shortName', symbol)

            logger.info(f"[FUNDAMENTAL] {symbol}: Analyzing (sector={sector}, regime={regime})...")

            # Extract all 18 metrics
            metrics = self._extract_metrics(fundamentals)

            # Score each metric relative to sector
            category_scores: Dict[str, List[float]] = {cat: [] for cat in METRIC_CATEGORIES}
            factors = []
            data_points = 0
            total_possible = 18

            for category, metric_names in METRIC_CATEGORIES.items():
                for metric_name in metric_names:
                    value = metrics.get(metric_name)
                    if value is None:
                        continue

                    data_points += 1

                    # Get sector stats and calculate percentile
                    stats = self._sector_stats.get_sector_stats(sector, metric_name)
                    percentile = self._sector_stats.calculate_percentile(value, stats)

                    # Invert for "lower is better" metrics
                    if metric_name in LOWER_IS_BETTER:
                        percentile = 1.0 - percentile

                    # Convert percentile to score (-100 to +100)
                    score = self._percentile_to_score(percentile)

                    # Calculate trend bonus/malus
                    trend_values = metrics.get(f"{metric_name}_history", [])
                    trend_label, trend_bonus = self._calculate_trend(value, trend_values, metric_name)
                    score += trend_bonus

                    score = max(-100, min(100, score))
                    category_scores[category].append(score)

                    # Build factor
                    factors.append({
                        'metric': metric_name,
                        'category': category,
                        'value': value,
                        'percentile': round(percentile, 3),
                        'score': round(score, 1),
                        'trend': trend_label,
                        'message': self._format_metric_message(metric_name, value, percentile, trend_label, sector),
                    })

            # Weighted category combination (V8.2: sector-aware)
            cat_weights = self._get_regime_weights(regime, sector)
            total_score = 0.0
            cat_summaries = {}

            for cat, scores_list in category_scores.items():
                if scores_list:
                    cat_avg = np.mean(scores_list)
                else:
                    cat_avg = 0.0
                cat_summaries[cat] = cat_avg
                total_score += cat_avg * cat_weights.get(cat, 0.15)

            total_score = max(-100, min(100, total_score))
            data_quality = data_points / total_possible if total_possible > 0 else 0.5

            # Reasoning
            reasoning = self._generate_reasoning(symbol, name, sector, regime, cat_summaries, factors, total_score)

            signal = "strong" if total_score > 30 else "good" if total_score > 0 else "weak" if total_score > -30 else "poor"
            cat_str = " | ".join(f"{c.title()}={s:.0f}" for c, s in cat_summaries.items())
            logger.info(f"[FUNDAMENTAL] {symbol}: Score={total_score:.1f}/100 ({signal}) | {cat_str} | Quality={data_quality:.0%}")

            return self._create_score(
                score=total_score,
                reasoning=reasoning,
                factors=factors,
                confidence=min(0.9, 0.5 + data_quality * 0.4),
                data_quality=data_quality,
            )

        except Exception as e:
            logger.error(f"[FUNDAMENTAL] {symbol}: Analysis failed: {e}", exc_info=True)
            return self._create_score(score=0, reasoning=f"Analysis error: {str(e)}", data_quality=0.0)

    async def _fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data via yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, lambda: ticker.info)

            # Also try to get quarterly financials for trends
            try:
                quarterly_income = await loop.run_in_executor(None, lambda: ticker.quarterly_income_stmt)
                quarterly_balance = await loop.run_in_executor(None, lambda: ticker.quarterly_balance_sheet)
                quarterly_cashflow = await loop.run_in_executor(None, lambda: ticker.quarterly_cashflow)
                
                if info is None:
                    info = {}
                info['_quarterly_income'] = quarterly_income
                info['_quarterly_balance'] = quarterly_balance
                info['_quarterly_cashflow'] = quarterly_cashflow
            except Exception:
                pass

            return info or {}
        except Exception as e:
            logger.warning(f"[FUNDAMENTAL] {symbol}: yfinance fetch failed: {e}")
            return {}

    def _extract_metrics(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all 18 metrics from yfinance info dict."""
        metrics = {}

        # === VALUATION (4) ===
        metrics["pe_ratio"] = info.get("trailingPE") or info.get("forwardPE")
        metrics["pb_ratio"] = info.get("priceToBook")
        metrics["ps_ratio"] = info.get("priceToSalesTrailing12Months")
        metrics["peg_ratio"] = info.get("pegRatio")

        # === CASH FLOW (3) ===
        fcf = info.get("freeCashflow")
        mcap = info.get("marketCap")
        metrics["fcf"] = fcf
        if fcf and mcap and mcap > 0:
            metrics["fcf_yield"] = fcf / mcap
        else:
            metrics["fcf_yield"] = None

        ocf = info.get("operatingCashflow")
        rev = info.get("totalRevenue")
        if ocf and rev and rev > 0:
            metrics["ocf_revenue_ratio"] = ocf / rev
        else:
            metrics["ocf_revenue_ratio"] = None

        # === GROWTH (3) ===
        metrics["revenue_growth_yoy"] = info.get("revenueGrowth")
        metrics["earnings_growth_yoy"] = info.get("earningsGrowth")

        # QoQ trend from quarterly data
        metrics["revenue_growth_qoq_trend"] = self._calc_revenue_qoq_trend(info)

        # === PROFITABILITY (3) ===
        metrics["net_profit_margin"] = info.get("profitMargins")
        metrics["roe"] = info.get("returnOnEquity")
        metrics["operating_margin"] = info.get("operatingMargins")

        # === HEALTH (3) ===
        metrics["debt_equity"] = info.get("debtToEquity")
        metrics["current_ratio"] = info.get("currentRatio")
        metrics["interest_coverage"] = self._calc_interest_coverage(info)

        # === QUALITY (2) ===
        metrics["earnings_surprise_avg"] = self._calc_earnings_surprise(info)
        metrics["margin_trend"] = self._calc_margin_trend(info)

        # History for trends (operating margin over time)
        metrics["operating_margin_history"] = self._get_operating_margin_history(info)
        metrics["roe_history"] = self._get_roe_history(info)

        # Filter out invalid values
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and (np.isnan(v) or np.isinf(v)):
                metrics[k] = None

        return metrics

    def _calc_revenue_qoq_trend(self, info: Dict) -> Optional[float]:
        """Calculate QoQ revenue acceleration/deceleration."""
        qi = info.get('_quarterly_income')
        if qi is None or not hasattr(qi, 'columns') or len(qi.columns) < 4:
            return None
        try:
            # Get Total Revenue row
            rev_row = None
            for label in ['Total Revenue', 'TotalRevenue']:
                if label in qi.index:
                    rev_row = qi.loc[label]
                    break
            if rev_row is None:
                return None
            
            vals = [v for v in rev_row.values[:4] if pd.notna(v) and v > 0]
            if len(vals) < 3:
                return None
            
            # QoQ growth rates (most recent first in yfinance)
            growths = []
            for i in range(len(vals) - 1):
                if vals[i + 1] > 0:
                    growths.append((vals[i] - vals[i + 1]) / vals[i + 1])
            
            if len(growths) < 2:
                return None
            
            # Trend: are growth rates accelerating or decelerating?
            # Positive = accelerating, negative = decelerating
            return growths[0] - growths[-1]
        except Exception:
            return None

    def _calc_interest_coverage(self, info: Dict) -> Optional[float]:
        """Calculate interest coverage ratio."""
        # Try from quarterly income statement
        qi = info.get('_quarterly_income')
        if qi is not None and hasattr(qi, 'index'):
            try:
                ebit = None
                interest = None
                for label in ['EBIT', 'Operating Income']:
                    if label in qi.index and pd.notna(qi.iloc[qi.index.get_loc(label), 0]):
                        ebit = float(qi.iloc[qi.index.get_loc(label), 0])
                        break
                for label in ['Interest Expense', 'InterestExpense']:
                    if label in qi.index and pd.notna(qi.iloc[qi.index.get_loc(label), 0]):
                        interest = abs(float(qi.iloc[qi.index.get_loc(label), 0]))
                        break
                if ebit is not None and interest and interest > 0:
                    return ebit / interest
            except Exception:
                pass
        
        # Fallback: rough estimate from info
        ebitda = info.get('ebitda')
        # yfinance doesn't always have interest expense directly
        # Use totalDebt and approximate interest
        total_debt = info.get('totalDebt', 0)
        if ebitda and total_debt and total_debt > 0:
            # Rough: assume ~4% interest rate
            approx_interest = total_debt * 0.04
            if approx_interest > 0:
                return ebitda / approx_interest
        return None

    def _calc_earnings_surprise(self, info: Dict) -> Optional[float]:
        """Calculate average earnings surprise from recent quarters."""
        # yfinance doesn't directly provide earnings surprise history
        # We approximate from trailing vs forward EPS
        trailing_eps = info.get('trailingEps')
        forward_eps = info.get('forwardEps')
        
        # If we have earnings estimates, compute implied beat/miss
        # This is approximate - a real implementation would use earnings calendar
        if trailing_eps and forward_eps and forward_eps != 0:
            # If trailing > forward estimates (for previous quarter), it's a beat
            # This is a rough proxy
            return (trailing_eps - forward_eps) / abs(forward_eps) * 0.5
        return None

    def _calc_margin_trend(self, info: Dict) -> Optional[float]:
        """Calculate operating margin trend (current vs 1 year ago)."""
        current_margin = info.get('operatingMargins')
        if current_margin is None:
            return None

        qi = info.get('_quarterly_income')
        if qi is None or not hasattr(qi, 'columns') or len(qi.columns) < 4:
            return None
        
        try:
            # Get operating income and revenue for oldest available quarter
            op_income_row = None
            rev_row = None
            for label in ['Operating Income', 'EBIT']:
                if label in qi.index:
                    op_income_row = qi.loc[label]
                    break
            for label in ['Total Revenue', 'TotalRevenue']:
                if label in qi.index:
                    rev_row = qi.loc[label]
                    break
            
            if op_income_row is None or rev_row is None:
                return None
            
            # Oldest quarter (last column)
            old_oi = op_income_row.values[-1]
            old_rev = rev_row.values[-1]
            
            if pd.notna(old_oi) and pd.notna(old_rev) and old_rev > 0:
                old_margin = old_oi / old_rev
                return current_margin - old_margin
        except Exception:
            pass
        return None

    def _get_operating_margin_history(self, info: Dict) -> List[float]:
        """Get operating margin values over last quarters."""
        qi = info.get('_quarterly_income')
        if qi is None or not hasattr(qi, 'columns'):
            return []
        try:
            op_row = None
            rev_row = None
            for label in ['Operating Income', 'EBIT']:
                if label in qi.index:
                    op_row = qi.loc[label]
                    break
            for label in ['Total Revenue', 'TotalRevenue']:
                if label in qi.index:
                    rev_row = qi.loc[label]
                    break
            if op_row is None or rev_row is None:
                return []
            margins = []
            for i in range(min(4, len(op_row))):
                oi = op_row.values[i]
                rv = rev_row.values[i]
                if pd.notna(oi) and pd.notna(rv) and rv > 0:
                    margins.append(oi / rv)
            return margins
        except Exception:
            return []

    def _get_roe_history(self, info: Dict) -> List[float]:
        """Get ROE values over last quarters (approximate)."""
        qi = info.get('_quarterly_income')
        qb = info.get('_quarterly_balance')
        if qi is None or qb is None:
            return []
        try:
            ni_row = None
            eq_row = None
            for label in ['Net Income', 'NetIncome']:
                if label in qi.index:
                    ni_row = qi.loc[label]
                    break
            for label in ['Stockholders Equity', 'StockholdersEquity', 'Total Equity Gross Minority Interest']:
                if label in qb.index:
                    eq_row = qb.loc[label]
                    break
            if ni_row is None or eq_row is None:
                return []
            roes = []
            n = min(4, len(ni_row), len(eq_row))
            for i in range(n):
                ni = ni_row.values[i]
                eq = eq_row.values[i]
                if pd.notna(ni) and pd.notna(eq) and eq > 0:
                    roes.append((ni * 4) / eq)  # Annualize quarterly NI
            return roes
        except Exception:
            return []

    def _percentile_to_score(self, percentile: float) -> float:
        """
        Convert percentile (0-1) to score (-100 to +100).
        Uses smooth interpolation with anchors.
        """
        # Anchors: percentile → score
        anchors = [
            (0.0, -100),
            (0.10, -60),
            (0.25, -25),
            (0.40, -5),
            (0.50, 0),
            (0.60, 5),
            (0.75, 25),
            (0.90, 60),
            (1.0, 100),
        ]

        p = max(0, min(1, percentile))

        for i in range(len(anchors) - 1):
            p_lo, s_lo = anchors[i]
            p_hi, s_hi = anchors[i + 1]
            if p_lo <= p <= p_hi:
                if p_hi == p_lo:
                    return s_lo
                frac = (p - p_lo) / (p_hi - p_lo)
                return s_lo + frac * (s_hi - s_lo)

        return 0

    def _calculate_trend(self, current: float, history: List[float], metric_name: str) -> Tuple[str, float]:
        """
        Calculate trend from historical values.
        Returns (label, bonus_score).
        """
        if not history or len(history) < 2:
            return 'stable', 0

        try:
            all_vals = history + [current] if current not in history else history
            if len(all_vals) < 2:
                return 'stable', 0

            # Simple linear regression slope
            x = np.arange(len(all_vals))
            y = np.array(all_vals, dtype=float)
            
            # Remove NaN
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return 'stable', 0
            
            x, y = x[mask], y[mask]
            
            # Normalize slope by mean to make it scale-independent
            mean_y = np.mean(np.abs(y))
            if mean_y == 0:
                return 'stable', 0
            
            slope = np.polyfit(x, y, 1)[0]
            normalized_slope = slope / mean_y

            # Thresholds for trend classification
            if metric_name in LOWER_IS_BETTER:
                normalized_slope = -normalized_slope  # Declining D/E is improving

            if normalized_slope > 0.05:
                return 'improving', 8
            elif normalized_slope < -0.05:
                return 'deteriorating', -8
            return 'stable', 0
        except Exception:
            return 'stable', 0

    def _format_metric_message(self, metric: str, value: float, percentile: float, trend: str, sector: str) -> str:
        """Format human-readable metric message."""
        pct_label = f"p{int(percentile * 100)}"
        trend_arrow = "↑" if trend == "improving" else "↓" if trend == "deteriorating" else "→"
        
        LABELS = {
            "pe_ratio": f"P/E {value:.1f}",
            "pb_ratio": f"P/B {value:.2f}",
            "ps_ratio": f"P/S {value:.2f}",
            "peg_ratio": f"PEG {value:.2f}",
            "fcf": f"FCF ${value/1e9:.1f}B" if abs(value) > 1e6 else f"FCF {value:.0f}",
            "fcf_yield": f"FCF Yield {value*100:.1f}%",
            "ocf_revenue_ratio": f"OCF/Rev {value*100:.1f}%",
            "revenue_growth_yoy": f"Rev Growth {value*100:.1f}%",
            "earnings_growth_yoy": f"Earnings Growth {value*100:.1f}%",
            "revenue_growth_qoq_trend": f"Rev QoQ Trend {value*100:+.1f}pp",
            "net_profit_margin": f"Net Margin {value*100:.1f}%",
            "roe": f"ROE {value*100:.1f}%",
            "operating_margin": f"Op Margin {value*100:.1f}%",
            "debt_equity": f"D/E {value:.0f}%",
            "current_ratio": f"Current Ratio {value:.2f}",
            "interest_coverage": f"Interest Coverage {value:.1f}x",
            "earnings_surprise_avg": f"Earnings Surprise {value*100:+.1f}%",
            "margin_trend": f"Margin Trend {value*100:+.1f}pp",
        }
        label = LABELS.get(metric, f"{metric}={value:.2f}")
        return f"{label} ({pct_label} in {sector}) {trend_arrow}"

    def _generate_reasoning(self, symbol: str, name: str, sector: str, regime: str,
                            cat_scores: Dict[str, float], factors: List[Dict], total: float) -> str:
        """Generate human-readable reasoning."""
        parts = []
        parts.append(f"Adaptive fundamental analysis for {name} ({sector})")
        parts.append(f"Market regime: {regime} | Score: {total:+.1f}/100")

        if total > 40:
            parts.append("Overall: STRONG fundamentals vs sector peers")
        elif total > 15:
            parts.append("Overall: GOOD fundamentals vs sector peers")
        elif total > -15:
            parts.append("Overall: AVERAGE fundamentals vs sector peers")
        elif total > -40:
            parts.append("Overall: WEAK fundamentals vs sector peers")
        else:
            parts.append("Overall: POOR fundamentals vs sector peers")

        for cat, score in cat_scores.items():
            q = "strong" if score > 25 else "good" if score > 0 else "weak" if score > -25 else "poor"
            parts.append(f"  {cat.replace('_', ' ').title()}: {q} ({score:+.0f})")

        # Top 3 factors
        sorted_factors = sorted(factors, key=lambda x: abs(x.get('score', 0)), reverse=True)[:4]
        if sorted_factors:
            parts.append("\nKey factors:")
            for f in sorted_factors:
                parts.append(f"  • {f['message']} (score: {f['score']:+.0f})")

        return "\n".join(parts)


# Singleton
_adaptive_fundamental: Optional[AdaptiveFundamentalPillar] = None


def get_adaptive_fundamental_pillar() -> AdaptiveFundamentalPillar:
    """Get or create the AdaptiveFundamentalPillar singleton."""
    global _adaptive_fundamental
    if _adaptive_fundamental is None:
        _adaptive_fundamental = AdaptiveFundamentalPillar()
    return _adaptive_fundamental


# Backward-compatible aliases
FundamentalPillar = AdaptiveFundamentalPillar
get_fundamental_pillar = get_adaptive_fundamental_pillar
