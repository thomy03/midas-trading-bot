"""
Sector Statistics Cache - Pre-computed percentile data for sector-relative scoring.

Loads from data/sector_fundamentals_cache.json (updated weekly).
Falls back to hardcoded medians if cache doesn't exist.
"""

import json
import os
import logging
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Base path for data files
DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent / "data"
CACHE_FILE = DATA_DIR / "sector_fundamentals_cache.json"
TICKERS_DIR = DATA_DIR / "tickers"

# Sector name mapping (yfinance sector → our canonical names)
SECTOR_ALIASES = {
    "Information Technology": "Technology",
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples": "Consumer Defensive",
    "Financials": "Financial Services",
    "Health Care": "Healthcare",
    "Materials": "Basic Materials",
    "Communication Services": "Communication Services",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    # Already canonical
    "Technology": "Technology",
    "Consumer Cyclical": "Consumer Cyclical",
    "Consumer Defensive": "Consumer Defensive",
    "Financial Services": "Financial Services",
    "Healthcare": "Healthcare",
    "Basic Materials": "Basic Materials",
}

# Metrics we track
METRIC_KEYS = [
    "pe_ratio", "pb_ratio", "ps_ratio", "peg_ratio",
    "fcf", "fcf_yield", "ocf_revenue_ratio",
    "revenue_growth_yoy", "earnings_growth_yoy", "revenue_growth_qoq_trend",
    "net_profit_margin", "roe", "operating_margin",
    "debt_equity", "current_ratio", "interest_coverage",
    "earnings_surprise_avg", "margin_trend",
]

# Hardcoded fallback medians (used if cache doesn't exist yet)
FALLBACK_SECTOR_STATS = {
    "Technology": {
        "pe_ratio": {"p10": 15, "p25": 22, "median": 32, "p75": 50, "p90": 85},
        "pb_ratio": {"p10": 2.0, "p25": 4.0, "median": 7.0, "p75": 14, "p90": 25},
        "ps_ratio": {"p10": 1.5, "p25": 3.0, "median": 6.0, "p75": 12, "p90": 25},
        "peg_ratio": {"p10": 0.5, "p25": 1.0, "median": 1.5, "p75": 2.5, "p90": 4.0},
        "fcf_yield": {"p10": 0.005, "p25": 0.015, "median": 0.03, "p75": 0.05, "p90": 0.08},
        "ocf_revenue_ratio": {"p10": 0.05, "p25": 0.15, "median": 0.25, "p75": 0.35, "p90": 0.45},
        "revenue_growth_yoy": {"p10": -0.05, "p25": 0.03, "median": 0.10, "p75": 0.20, "p90": 0.40},
        "earnings_growth_yoy": {"p10": -0.20, "p25": -0.02, "median": 0.10, "p75": 0.25, "p90": 0.50},
        "net_profit_margin": {"p10": 0.02, "p25": 0.08, "median": 0.15, "p75": 0.25, "p90": 0.35},
        "roe": {"p10": 0.05, "p25": 0.12, "median": 0.20, "p75": 0.30, "p90": 0.45},
        "operating_margin": {"p10": 0.03, "p25": 0.10, "median": 0.20, "p75": 0.30, "p90": 0.40},
        "debt_equity": {"p10": 0, "p25": 15, "median": 50, "p75": 100, "p90": 200},
        "current_ratio": {"p10": 0.8, "p25": 1.2, "median": 1.8, "p75": 2.8, "p90": 4.5},
        "interest_coverage": {"p10": 2, "p25": 5, "median": 15, "p75": 40, "p90": 100},
        "earnings_surprise_avg": {"p10": -0.05, "p25": 0.0, "median": 0.03, "p75": 0.08, "p90": 0.15},
        "margin_trend": {"p10": -0.05, "p25": -0.01, "median": 0.0, "p75": 0.02, "p90": 0.05},
    },
    "Healthcare": {
        "pe_ratio": {"p10": 12, "p25": 18, "median": 28, "p75": 45, "p90": 80},
        "pb_ratio": {"p10": 1.5, "p25": 3.0, "median": 5.0, "p75": 10, "p90": 20},
        "ps_ratio": {"p10": 1.0, "p25": 2.5, "median": 5.0, "p75": 10, "p90": 20},
        "peg_ratio": {"p10": 0.5, "p25": 1.0, "median": 1.8, "p75": 2.8, "p90": 4.5},
        "fcf_yield": {"p10": 0.005, "p25": 0.02, "median": 0.04, "p75": 0.06, "p90": 0.09},
        "ocf_revenue_ratio": {"p10": 0.05, "p25": 0.12, "median": 0.20, "p75": 0.30, "p90": 0.40},
        "revenue_growth_yoy": {"p10": -0.05, "p25": 0.02, "median": 0.08, "p75": 0.18, "p90": 0.35},
        "earnings_growth_yoy": {"p10": -0.25, "p25": -0.05, "median": 0.08, "p75": 0.22, "p90": 0.50},
        "net_profit_margin": {"p10": -0.10, "p25": 0.05, "median": 0.12, "p75": 0.22, "p90": 0.35},
        "roe": {"p10": 0.02, "p25": 0.08, "median": 0.15, "p75": 0.25, "p90": 0.40},
        "operating_margin": {"p10": -0.05, "p25": 0.08, "median": 0.18, "p75": 0.28, "p90": 0.38},
        "debt_equity": {"p10": 0, "p25": 20, "median": 60, "p75": 120, "p90": 250},
        "current_ratio": {"p10": 0.8, "p25": 1.3, "median": 2.0, "p75": 3.5, "p90": 6.0},
        "interest_coverage": {"p10": 1, "p25": 4, "median": 10, "p75": 30, "p90": 80},
        "earnings_surprise_avg": {"p10": -0.05, "p25": 0.0, "median": 0.03, "p75": 0.08, "p90": 0.15},
        "margin_trend": {"p10": -0.05, "p25": -0.01, "median": 0.0, "p75": 0.02, "p90": 0.05},
    },
    "Financial Services": {
        "pe_ratio": {"p10": 8, "p25": 11, "median": 15, "p75": 20, "p90": 30},
        "pb_ratio": {"p10": 0.5, "p25": 0.9, "median": 1.3, "p75": 2.0, "p90": 3.5},
        "ps_ratio": {"p10": 1.0, "p25": 2.0, "median": 3.5, "p75": 5.5, "p90": 8.0},
        "peg_ratio": {"p10": 0.5, "p25": 0.8, "median": 1.2, "p75": 2.0, "p90": 3.0},
        "fcf_yield": {"p10": 0.01, "p25": 0.03, "median": 0.06, "p75": 0.09, "p90": 0.12},
        "ocf_revenue_ratio": {"p10": 0.05, "p25": 0.15, "median": 0.25, "p75": 0.40, "p90": 0.55},
        "revenue_growth_yoy": {"p10": -0.08, "p25": 0.0, "median": 0.05, "p75": 0.12, "p90": 0.25},
        "earnings_growth_yoy": {"p10": -0.20, "p25": -0.03, "median": 0.08, "p75": 0.20, "p90": 0.40},
        "net_profit_margin": {"p10": 0.05, "p25": 0.12, "median": 0.20, "p75": 0.30, "p90": 0.40},
        "roe": {"p10": 0.05, "p25": 0.08, "median": 0.12, "p75": 0.18, "p90": 0.25},
        "operating_margin": {"p10": 0.10, "p25": 0.18, "median": 0.28, "p75": 0.38, "p90": 0.50},
        "debt_equity": {"p10": 20, "p25": 50, "median": 120, "p75": 250, "p90": 500},
        "current_ratio": {"p10": 0.5, "p25": 0.8, "median": 1.1, "p75": 1.5, "p90": 2.0},
        "interest_coverage": {"p10": 1, "p25": 2, "median": 5, "p75": 10, "p90": 20},
        "earnings_surprise_avg": {"p10": -0.03, "p25": 0.0, "median": 0.02, "p75": 0.06, "p90": 0.12},
        "margin_trend": {"p10": -0.04, "p25": -0.01, "median": 0.0, "p75": 0.01, "p90": 0.04},
    },
    "Energy": {
        "pe_ratio": {"p10": 5, "p25": 8, "median": 12, "p75": 18, "p90": 30},
        "pb_ratio": {"p10": 0.5, "p25": 0.8, "median": 1.5, "p75": 2.5, "p90": 4.0},
        "ps_ratio": {"p10": 0.3, "p25": 0.6, "median": 1.2, "p75": 2.0, "p90": 3.5},
        "peg_ratio": {"p10": 0.3, "p25": 0.6, "median": 1.0, "p75": 1.8, "p90": 3.0},
        "fcf_yield": {"p10": 0.02, "p25": 0.04, "median": 0.08, "p75": 0.12, "p90": 0.18},
        "ocf_revenue_ratio": {"p10": 0.05, "p25": 0.10, "median": 0.18, "p75": 0.28, "p90": 0.38},
        "revenue_growth_yoy": {"p10": -0.20, "p25": -0.05, "median": 0.05, "p75": 0.15, "p90": 0.35},
        "earnings_growth_yoy": {"p10": -0.40, "p25": -0.10, "median": 0.05, "p75": 0.25, "p90": 0.60},
        "net_profit_margin": {"p10": 0.01, "p25": 0.05, "median": 0.10, "p75": 0.18, "p90": 0.25},
        "roe": {"p10": 0.02, "p25": 0.08, "median": 0.15, "p75": 0.25, "p90": 0.35},
        "operating_margin": {"p10": 0.02, "p25": 0.08, "median": 0.15, "p75": 0.25, "p90": 0.35},
        "debt_equity": {"p10": 10, "p25": 25, "median": 50, "p75": 90, "p90": 180},
        "current_ratio": {"p10": 0.6, "p25": 0.9, "median": 1.3, "p75": 1.8, "p90": 2.5},
        "interest_coverage": {"p10": 1, "p25": 3, "median": 8, "p75": 20, "p90": 50},
        "earnings_surprise_avg": {"p10": -0.08, "p25": -0.02, "median": 0.02, "p75": 0.06, "p90": 0.12},
        "margin_trend": {"p10": -0.06, "p25": -0.02, "median": 0.0, "p75": 0.02, "p90": 0.06},
    },
    "Consumer Cyclical": {
        "pe_ratio": {"p10": 10, "p25": 15, "median": 22, "p75": 35, "p90": 55},
        "pb_ratio": {"p10": 1.0, "p25": 2.0, "median": 4.0, "p75": 8.0, "p90": 15},
        "ps_ratio": {"p10": 0.3, "p25": 0.8, "median": 1.8, "p75": 4.0, "p90": 8.0},
        "peg_ratio": {"p10": 0.5, "p25": 0.9, "median": 1.4, "p75": 2.2, "p90": 3.5},
        "fcf_yield": {"p10": 0.01, "p25": 0.02, "median": 0.04, "p75": 0.07, "p90": 0.10},
        "ocf_revenue_ratio": {"p10": 0.03, "p25": 0.08, "median": 0.13, "p75": 0.20, "p90": 0.30},
        "revenue_growth_yoy": {"p10": -0.08, "p25": 0.0, "median": 0.08, "p75": 0.18, "p90": 0.35},
        "earnings_growth_yoy": {"p10": -0.25, "p25": -0.05, "median": 0.10, "p75": 0.25, "p90": 0.50},
        "net_profit_margin": {"p10": 0.01, "p25": 0.04, "median": 0.08, "p75": 0.14, "p90": 0.22},
        "roe": {"p10": 0.03, "p25": 0.10, "median": 0.18, "p75": 0.30, "p90": 0.50},
        "operating_margin": {"p10": 0.02, "p25": 0.06, "median": 0.12, "p75": 0.18, "p90": 0.28},
        "debt_equity": {"p10": 5, "p25": 20, "median": 60, "p75": 130, "p90": 300},
        "current_ratio": {"p10": 0.7, "p25": 1.0, "median": 1.5, "p75": 2.2, "p90": 3.5},
        "interest_coverage": {"p10": 1, "p25": 3, "median": 8, "p75": 20, "p90": 50},
        "earnings_surprise_avg": {"p10": -0.05, "p25": 0.0, "median": 0.03, "p75": 0.07, "p90": 0.12},
        "margin_trend": {"p10": -0.04, "p25": -0.01, "median": 0.0, "p75": 0.01, "p90": 0.04},
    },
    "Consumer Defensive": {
        "pe_ratio": {"p10": 12, "p25": 16, "median": 22, "p75": 28, "p90": 38},
        "pb_ratio": {"p10": 1.5, "p25": 2.5, "median": 4.5, "p75": 8.0, "p90": 14},
        "ps_ratio": {"p10": 0.5, "p25": 1.0, "median": 2.0, "p75": 3.5, "p90": 6.0},
        "peg_ratio": {"p10": 0.8, "p25": 1.2, "median": 2.0, "p75": 3.0, "p90": 4.5},
        "fcf_yield": {"p10": 0.01, "p25": 0.025, "median": 0.04, "p75": 0.06, "p90": 0.09},
        "ocf_revenue_ratio": {"p10": 0.05, "p25": 0.08, "median": 0.13, "p75": 0.18, "p90": 0.25},
        "revenue_growth_yoy": {"p10": -0.03, "p25": 0.01, "median": 0.04, "p75": 0.08, "p90": 0.15},
        "earnings_growth_yoy": {"p10": -0.12, "p25": -0.02, "median": 0.05, "p75": 0.12, "p90": 0.25},
        "net_profit_margin": {"p10": 0.02, "p25": 0.05, "median": 0.08, "p75": 0.14, "p90": 0.22},
        "roe": {"p10": 0.05, "p25": 0.10, "median": 0.18, "p75": 0.30, "p90": 0.50},
        "operating_margin": {"p10": 0.03, "p25": 0.06, "median": 0.12, "p75": 0.18, "p90": 0.25},
        "debt_equity": {"p10": 10, "p25": 30, "median": 70, "p75": 140, "p90": 280},
        "current_ratio": {"p10": 0.5, "p25": 0.8, "median": 1.1, "p75": 1.6, "p90": 2.2},
        "interest_coverage": {"p10": 2, "p25": 4, "median": 10, "p75": 20, "p90": 40},
        "earnings_surprise_avg": {"p10": -0.03, "p25": 0.0, "median": 0.02, "p75": 0.05, "p90": 0.10},
        "margin_trend": {"p10": -0.03, "p25": -0.01, "median": 0.0, "p75": 0.01, "p90": 0.03},
    },
    "Industrials": {
        "pe_ratio": {"p10": 10, "p25": 15, "median": 22, "p75": 30, "p90": 45},
        "pb_ratio": {"p10": 1.0, "p25": 2.0, "median": 3.5, "p75": 6.0, "p90": 12},
        "ps_ratio": {"p10": 0.5, "p25": 1.0, "median": 2.0, "p75": 3.5, "p90": 6.0},
        "peg_ratio": {"p10": 0.5, "p25": 1.0, "median": 1.5, "p75": 2.5, "p90": 4.0},
        "fcf_yield": {"p10": 0.01, "p25": 0.025, "median": 0.04, "p75": 0.06, "p90": 0.09},
        "ocf_revenue_ratio": {"p10": 0.03, "p25": 0.08, "median": 0.13, "p75": 0.20, "p90": 0.28},
        "revenue_growth_yoy": {"p10": -0.08, "p25": 0.0, "median": 0.06, "p75": 0.14, "p90": 0.25},
        "earnings_growth_yoy": {"p10": -0.20, "p25": -0.03, "median": 0.08, "p75": 0.20, "p90": 0.40},
        "net_profit_margin": {"p10": 0.02, "p25": 0.05, "median": 0.09, "p75": 0.14, "p90": 0.20},
        "roe": {"p10": 0.05, "p25": 0.10, "median": 0.18, "p75": 0.28, "p90": 0.40},
        "operating_margin": {"p10": 0.03, "p25": 0.07, "median": 0.13, "p75": 0.20, "p90": 0.28},
        "debt_equity": {"p10": 5, "p25": 20, "median": 60, "p75": 120, "p90": 250},
        "current_ratio": {"p10": 0.8, "p25": 1.1, "median": 1.5, "p75": 2.2, "p90": 3.2},
        "interest_coverage": {"p10": 2, "p25": 4, "median": 10, "p75": 25, "p90": 60},
        "earnings_surprise_avg": {"p10": -0.04, "p25": 0.0, "median": 0.03, "p75": 0.06, "p90": 0.12},
        "margin_trend": {"p10": -0.04, "p25": -0.01, "median": 0.0, "p75": 0.01, "p90": 0.04},
    },
    "Basic Materials": {
        "pe_ratio": {"p10": 6, "p25": 10, "median": 15, "p75": 22, "p90": 35},
        "pb_ratio": {"p10": 0.6, "p25": 1.0, "median": 1.8, "p75": 3.0, "p90": 5.0},
        "ps_ratio": {"p10": 0.3, "p25": 0.7, "median": 1.3, "p75": 2.5, "p90": 4.0},
        "peg_ratio": {"p10": 0.3, "p25": 0.7, "median": 1.2, "p75": 2.0, "p90": 3.5},
        "fcf_yield": {"p10": 0.01, "p25": 0.03, "median": 0.06, "p75": 0.10, "p90": 0.15},
        "ocf_revenue_ratio": {"p10": 0.03, "p25": 0.08, "median": 0.15, "p75": 0.22, "p90": 0.30},
        "revenue_growth_yoy": {"p10": -0.15, "p25": -0.03, "median": 0.05, "p75": 0.15, "p90": 0.30},
        "earnings_growth_yoy": {"p10": -0.35, "p25": -0.10, "median": 0.05, "p75": 0.25, "p90": 0.55},
        "net_profit_margin": {"p10": 0.01, "p25": 0.04, "median": 0.08, "p75": 0.14, "p90": 0.22},
        "roe": {"p10": 0.03, "p25": 0.07, "median": 0.13, "p75": 0.22, "p90": 0.35},
        "operating_margin": {"p10": 0.02, "p25": 0.06, "median": 0.12, "p75": 0.20, "p90": 0.30},
        "debt_equity": {"p10": 5, "p25": 15, "median": 40, "p75": 80, "p90": 160},
        "current_ratio": {"p10": 0.8, "p25": 1.2, "median": 1.8, "p75": 2.8, "p90": 4.0},
        "interest_coverage": {"p10": 1, "p25": 3, "median": 8, "p75": 18, "p90": 40},
        "earnings_surprise_avg": {"p10": -0.06, "p25": -0.01, "median": 0.02, "p75": 0.06, "p90": 0.12},
        "margin_trend": {"p10": -0.05, "p25": -0.02, "median": 0.0, "p75": 0.02, "p90": 0.05},
    },
    "Utilities": {
        "pe_ratio": {"p10": 10, "p25": 14, "median": 19, "p75": 25, "p90": 35},
        "pb_ratio": {"p10": 0.8, "p25": 1.2, "median": 1.8, "p75": 2.5, "p90": 3.5},
        "ps_ratio": {"p10": 1.0, "p25": 1.8, "median": 2.8, "p75": 4.0, "p90": 6.0},
        "peg_ratio": {"p10": 0.8, "p25": 1.2, "median": 2.0, "p75": 3.0, "p90": 4.5},
        "fcf_yield": {"p10": 0.01, "p25": 0.02, "median": 0.04, "p75": 0.06, "p90": 0.08},
        "ocf_revenue_ratio": {"p10": 0.10, "p25": 0.18, "median": 0.25, "p75": 0.35, "p90": 0.45},
        "revenue_growth_yoy": {"p10": -0.03, "p25": 0.01, "median": 0.04, "p75": 0.08, "p90": 0.15},
        "earnings_growth_yoy": {"p10": -0.10, "p25": -0.02, "median": 0.05, "p75": 0.12, "p90": 0.22},
        "net_profit_margin": {"p10": 0.05, "p25": 0.08, "median": 0.12, "p75": 0.18, "p90": 0.25},
        "roe": {"p10": 0.04, "p25": 0.07, "median": 0.10, "p75": 0.14, "p90": 0.18},
        "operating_margin": {"p10": 0.08, "p25": 0.14, "median": 0.22, "p75": 0.30, "p90": 0.40},
        "debt_equity": {"p10": 40, "p25": 70, "median": 120, "p75": 180, "p90": 300},
        "current_ratio": {"p10": 0.4, "p25": 0.6, "median": 0.9, "p75": 1.2, "p90": 1.6},
        "interest_coverage": {"p10": 1, "p25": 2, "median": 3, "p75": 5, "p90": 8},
        "earnings_surprise_avg": {"p10": -0.03, "p25": 0.0, "median": 0.02, "p75": 0.04, "p90": 0.08},
        "margin_trend": {"p10": -0.03, "p25": -0.01, "median": 0.0, "p75": 0.01, "p90": 0.03},
    },
    "Real Estate": {
        "pe_ratio": {"p10": 12, "p25": 20, "median": 35, "p75": 55, "p90": 90},
        "pb_ratio": {"p10": 0.6, "p25": 1.0, "median": 1.8, "p75": 3.0, "p90": 5.0},
        "ps_ratio": {"p10": 2.0, "p25": 4.0, "median": 7.0, "p75": 12, "p90": 20},
        "peg_ratio": {"p10": 0.8, "p25": 1.5, "median": 2.5, "p75": 4.0, "p90": 6.0},
        "fcf_yield": {"p10": 0.01, "p25": 0.025, "median": 0.04, "p75": 0.06, "p90": 0.09},
        "ocf_revenue_ratio": {"p10": 0.15, "p25": 0.25, "median": 0.35, "p75": 0.50, "p90": 0.65},
        "revenue_growth_yoy": {"p10": -0.05, "p25": 0.0, "median": 0.05, "p75": 0.10, "p90": 0.20},
        "earnings_growth_yoy": {"p10": -0.15, "p25": -0.03, "median": 0.05, "p75": 0.15, "p90": 0.30},
        "net_profit_margin": {"p10": 0.05, "p25": 0.15, "median": 0.25, "p75": 0.40, "p90": 0.55},
        "roe": {"p10": 0.02, "p25": 0.05, "median": 0.08, "p75": 0.12, "p90": 0.18},
        "operating_margin": {"p10": 0.10, "p25": 0.20, "median": 0.35, "p75": 0.50, "p90": 0.65},
        "debt_equity": {"p10": 20, "p25": 50, "median": 90, "p75": 150, "p90": 280},
        "current_ratio": {"p10": 0.3, "p25": 0.5, "median": 0.8, "p75": 1.2, "p90": 2.0},
        "interest_coverage": {"p10": 1, "p25": 2, "median": 3, "p75": 5, "p90": 8},
        "earnings_surprise_avg": {"p10": -0.04, "p25": 0.0, "median": 0.02, "p75": 0.05, "p90": 0.10},
        "margin_trend": {"p10": -0.03, "p25": -0.01, "median": 0.0, "p75": 0.01, "p90": 0.03},
    },
    "Communication Services": {
        "pe_ratio": {"p10": 10, "p25": 15, "median": 22, "p75": 35, "p90": 55},
        "pb_ratio": {"p10": 1.0, "p25": 2.0, "median": 3.5, "p75": 6.0, "p90": 12},
        "ps_ratio": {"p10": 1.0, "p25": 2.0, "median": 4.0, "p75": 7.0, "p90": 12},
        "peg_ratio": {"p10": 0.5, "p25": 0.9, "median": 1.4, "p75": 2.2, "p90": 3.5},
        "fcf_yield": {"p10": 0.01, "p25": 0.02, "median": 0.04, "p75": 0.07, "p90": 0.10},
        "ocf_revenue_ratio": {"p10": 0.05, "p25": 0.12, "median": 0.22, "p75": 0.32, "p90": 0.42},
        "revenue_growth_yoy": {"p10": -0.05, "p25": 0.02, "median": 0.08, "p75": 0.15, "p90": 0.30},
        "earnings_growth_yoy": {"p10": -0.20, "p25": -0.03, "median": 0.08, "p75": 0.22, "p90": 0.45},
        "net_profit_margin": {"p10": 0.02, "p25": 0.08, "median": 0.15, "p75": 0.25, "p90": 0.35},
        "roe": {"p10": 0.03, "p25": 0.08, "median": 0.15, "p75": 0.25, "p90": 0.40},
        "operating_margin": {"p10": 0.03, "p25": 0.10, "median": 0.20, "p75": 0.30, "p90": 0.40},
        "debt_equity": {"p10": 10, "p25": 30, "median": 70, "p75": 130, "p90": 250},
        "current_ratio": {"p10": 0.6, "p25": 0.9, "median": 1.3, "p75": 1.9, "p90": 3.0},
        "interest_coverage": {"p10": 2, "p25": 4, "median": 10, "p75": 25, "p90": 60},
        "earnings_surprise_avg": {"p10": -0.04, "p25": 0.0, "median": 0.03, "p75": 0.07, "p90": 0.12},
        "margin_trend": {"p10": -0.04, "p25": -0.01, "median": 0.0, "p75": 0.02, "p90": 0.04},
    },
}

# Default stats for unknown sectors
DEFAULT_SECTOR_STATS = FALLBACK_SECTOR_STATS["Industrials"]


def normalize_sector(sector: str) -> str:
    """Normalize sector name to canonical form."""
    if not sector:
        return "Unknown"
    return SECTOR_ALIASES.get(sector, sector)


class SectorStatsManager:
    """Manages sector statistics cache with fallback to hardcoded values."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._loaded = False
        self._load_cache()

    def _load_cache(self):
        """Load sector stats from cache file."""
        try:
            if CACHE_FILE.exists():
                with open(CACHE_FILE, 'r') as f:
                    self._cache = json.load(f)
                # Check freshness (7 days)
                for sector_data in self._cache.values():
                    if isinstance(sector_data, dict) and "updated_at" in sector_data:
                        updated = datetime.fromisoformat(sector_data["updated_at"])
                        if (datetime.now() - updated).days > 7:
                            logger.info(f"Sector cache is stale (>7 days), will use fallback for missing")
                self._loaded = True
                logger.info(f"Loaded sector stats cache: {len(self._cache)} sectors")
            else:
                logger.info("No sector cache file found, using hardcoded fallback")
        except Exception as e:
            logger.warning(f"Failed to load sector cache: {e}, using fallback")

    def get_sector_stats(self, sector: str, metric_name: str) -> Dict[str, float]:
        """
        Get percentile distribution for a metric in a sector.
        
        Returns: {'p10': ..., 'p25': ..., 'median': ..., 'p75': ..., 'p90': ...}
        """
        canonical = normalize_sector(sector)

        # Try live cache first
        if canonical in self._cache and metric_name in self._cache[canonical]:
            return self._cache[canonical][metric_name]

        # Fallback to hardcoded
        if canonical in FALLBACK_SECTOR_STATS and metric_name in FALLBACK_SECTOR_STATS[canonical]:
            return FALLBACK_SECTOR_STATS[canonical][metric_name]

        # Ultimate fallback: use default sector stats
        if metric_name in DEFAULT_SECTOR_STATS:
            return DEFAULT_SECTOR_STATS[metric_name]

        # Metric not found at all — return neutral distribution
        return {"p10": 0, "p25": 25, "median": 50, "p75": 75, "p90": 100}

    def calculate_percentile(self, value: float, stats: Dict[str, float]) -> float:
        """
        Calculate where a value falls in the sector distribution.
        Returns 0.0 to 1.0 (percentile as fraction).
        
        Uses linear interpolation between known percentile points.
        """
        if value is None:
            return 0.5  # neutral

        points = [
            (0.10, stats.get("p10", 0)),
            (0.25, stats.get("p25", 25)),
            (0.50, stats.get("median", 50)),
            (0.75, stats.get("p75", 75)),
            (0.90, stats.get("p90", 100)),
        ]

        # Below p10
        if value <= points[0][1]:
            return 0.05

        # Above p90
        if value >= points[-1][1]:
            return 0.95

        # Interpolate between adjacent points
        for i in range(len(points) - 1):
            pct_lo, val_lo = points[i]
            pct_hi, val_hi = points[i + 1]
            if val_lo <= value <= val_hi:
                if val_hi == val_lo:
                    return (pct_lo + pct_hi) / 2
                frac = (value - val_lo) / (val_hi - val_lo)
                return pct_lo + frac * (pct_hi - pct_lo)

        return 0.5

    def rebuild_cache(self):
        """
        Rebuild sector stats cache from ticker files (SP500 + CAC40).
        This should be called weekly (e.g., Sunday evening via cron).
        """
        logger.info("Rebuilding sector fundamentals cache...")

        # Load tickers grouped by sector
        tickers_by_sector: Dict[str, List[str]] = {}
        
        for ticker_file in ["sp500.json", "cac40.json"]:
            fpath = TICKERS_DIR / ticker_file
            if not fpath.exists():
                continue
            try:
                with open(fpath) as f:
                    data = json.load(f)
                for item in data.get("tickers", []):
                    sector = normalize_sector(item.get("sector", "Unknown"))
                    symbol = item["symbol"]
                    if sector not in tickers_by_sector:
                        tickers_by_sector[sector] = []
                    tickers_by_sector[sector].append(symbol)
            except Exception as e:
                logger.error(f"Failed to load {ticker_file}: {e}")

        new_cache = {}

        for sector, symbols in tickers_by_sector.items():
            logger.info(f"Processing sector {sector}: {len(symbols)} tickers")
            metrics_data: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    if not info:
                        continue

                    # Extract metrics
                    _safe_append(metrics_data, "pe_ratio", info.get("trailingPE"))
                    _safe_append(metrics_data, "pb_ratio", info.get("priceToBook"))
                    _safe_append(metrics_data, "ps_ratio", info.get("priceToSalesTrailing12Months"))
                    _safe_append(metrics_data, "peg_ratio", info.get("pegRatio"))
                    
                    # FCF
                    mcap = info.get("marketCap", 0)
                    fcf = info.get("freeCashflow")
                    _safe_append(metrics_data, "fcf", fcf)
                    if fcf and mcap and mcap > 0:
                        _safe_append(metrics_data, "fcf_yield", fcf / mcap)
                    
                    ocf = info.get("operatingCashflow")
                    rev = info.get("totalRevenue")
                    if ocf and rev and rev > 0:
                        _safe_append(metrics_data, "ocf_revenue_ratio", ocf / rev)

                    _safe_append(metrics_data, "revenue_growth_yoy", info.get("revenueGrowth"))
                    _safe_append(metrics_data, "earnings_growth_yoy", info.get("earningsGrowth"))
                    _safe_append(metrics_data, "net_profit_margin", info.get("profitMargins"))
                    _safe_append(metrics_data, "roe", info.get("returnOnEquity"))
                    _safe_append(metrics_data, "operating_margin", info.get("operatingMargins"))
                    _safe_append(metrics_data, "debt_equity", info.get("debtToEquity"))
                    _safe_append(metrics_data, "current_ratio", info.get("currentRatio"))
                    
                    # Interest coverage: approx from EBIT / interest expense
                    ebit = info.get("ebitda")  # Approx
                    interest = info.get("interestExpense")  # Might not exist
                    # yfinance doesn't reliably provide interest coverage directly
                    # We'll skip or use what's available

                except Exception as e:
                    logger.debug(f"Failed to fetch {symbol}: {e}")
                    continue

            # Compute percentiles
            sector_stats = {}
            for metric, values in metrics_data.items():
                if len(values) < 5:
                    continue
                arr = np.array(values)
                sector_stats[metric] = {
                    "p10": float(np.percentile(arr, 10)),
                    "p25": float(np.percentile(arr, 25)),
                    "median": float(np.median(arr)),
                    "p75": float(np.percentile(arr, 75)),
                    "p90": float(np.percentile(arr, 90)),
                }
            
            sector_stats["updated_at"] = datetime.now().isoformat()[:10]
            sector_stats["sample_size"] = len(symbols)
            new_cache[sector] = sector_stats

        # Save
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, 'w') as f:
            json.dump(new_cache, f, indent=2)
        
        self._cache = new_cache
        logger.info(f"Sector cache rebuilt: {len(new_cache)} sectors saved to {CACHE_FILE}")
        return new_cache


def _safe_append(data: Dict[str, List], key: str, value):
    """Append value to list if it's a valid number."""
    if value is not None and isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
        data[key].append(float(value))


# Singleton
_sector_stats: Optional[SectorStatsManager] = None


def get_sector_stats_manager() -> SectorStatsManager:
    """Get or create the SectorStatsManager singleton."""
    global _sector_stats
    if _sector_stats is None:
        _sector_stats = SectorStatsManager()
    return _sector_stats
