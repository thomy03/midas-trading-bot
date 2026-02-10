"""
V6.2 Backtester - Tests the REAL 5-pillar system

Unlike the legacy backtester (RSI+EMA only), this backtester replicates
the actual V6.2 live system:
  1. Technical Pillar (25+ indicators)
  2. Fundamental Pillar (valuation, growth, profitability, health)
  3. Sentiment Pillar (simulated neutral for backtest - no historical Twitter data)
  4. News Pillar (simulated neutral for backtest - no historical LLM data)
  5. ML Pillar (40 features, heuristic fallback in backtest)

Plus:
  - Regime detection (BULL/BEAR/RANGE/VOLATILE)
  - Regime-weighted aggregation
  - Adaptive ML Gate (vol > 3% -> disable ML)
  - Transaction costs (commission, spread, slippage)
  - Mark-to-market equity curve for proper Sharpe calculation
  - Out-of-sample / walk-forward support
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
import json
import os
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .metrics import Trade, PerformanceMetrics, calculate_metrics, format_metrics_report

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────

class MarketRegime(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    RANGE = "RANGE"
    VOLATILE = "VOLATILE"


class ScoringMode(Enum):
    """Scoring mode for A/B/C/D/E comparison"""
    TECH_ONLY = "A"          # Technical pillar only (baseline)
    THREE_PILLARS = "B"      # Tech + Fundamental + ML (redistributed weights)
    THREE_PILLARS_VIX = "C"  # Tech + Fund + ML + VIX sentiment proxy
    FULL_LIVE = "D"          # All 5 pillars (live mode - needs real API data)
    ML_TRAINED = "E"         # V7: Trained ML model replaces heuristic


@dataclass
class TransactionCosts:
    """Model real-world transaction costs"""
    commission_per_trade: float = 1.00    # IBKR fixed per trade (USD)
    spread_pct: float = 0.05             # Bid-ask spread (%)
    slippage_pct: float = 0.10           # Execution slippage (%)
    tax_pct: float = 0.00                # Tax on gains (0 for PEA)

    def entry_cost_factor(self) -> float:
        """Multiplicative cost on entry price (buy higher)"""
        return 1 + (self.spread_pct + self.slippage_pct) / 100

    def exit_cost_factor(self) -> float:
        """Multiplicative cost on exit price (sell lower)"""
        return 1 - (self.spread_pct + self.slippage_pct) / 100


@dataclass
class V6BacktestConfig:
    """Configuration for V6.2 backtester"""
    initial_capital: float = 100_000
    max_positions: int = 20
    position_size_pct: float = 0.10       # 10% of CURRENT capital per position

    # V7.1: Leverage / margin trading
    leverage: float = 1.3                  # 1.3x = 130% buying power (optimized, IBKR Reg T allows 2x)
    margin_cost_annual: float = 0.06       # Annual margin interest rate (6% IBKR)

    # Scoring thresholds
    # Live system uses 75 with all 5 real pillars.
    # In backtest, sentiment/news are neutral (50) or VIX proxy, which drags
    # weighted scores toward the mean. 55 is optimized via parameter sweep.
    buy_threshold: float = 55.0           # Score >= 55 -> BUY (optimized from 65)
    sell_threshold: float = 40.0          # Score < 40 -> SELL signal

    # Exit parameters
    default_stop_loss_pct: float = 0.08   # 8% default stop
    trailing_atr_multiplier: float = 3.5  # Trailing stop = entry - ATR * multiplier (optimized from 2.5)
    max_hold_days: int = 120

    # ML Gate
    ml_gate_enabled: bool = True
    ml_gate_volatility_threshold: float = 0.03  # 3%
    ml_boost_points: float = 5.0
    ml_block_threshold: float = 0.40
    ml_boost_threshold: float = 0.60

    # Regime detection parameters
    regime_enabled: bool = True
    bull_threshold: float = 3.0           # SPY > EMA50 + 3%
    bear_threshold: float = -3.0          # SPY < EMA50 - 3%
    volatile_vix_threshold: float = 30.0
    volatile_vol_threshold: float = 35.0

    # Scoring mode (A/B/C/D/E comparison) - B outperforms C (VIX proxy hurts)
    scoring_mode: ScoringMode = ScoringMode.THREE_PILLARS

    # Transaction costs
    costs: TransactionCosts = field(default_factory=TransactionCosts)

    # Walk-forward / out-of-sample
    train_end_date: Optional[str] = None      # If set, ML trains only on data before this
    validation_end_date: Optional[str] = None  # If set, tune params on this period
    # Test period = everything after validation_end_date

    # V7: Dynamic position sizing
    dynamic_sizing: bool = False           # Use PositionSizer instead of fixed %
    target_risk_pct: float = 0.015         # 1.5% portfolio risk per trade

    # V7: Defensive mode in backtest
    defensive_mode: bool = False           # Apply DefensiveManager logic

    # V7: Correlation checks in backtest
    correlation_checks: bool = False       # Apply CorrelationManager logic
    max_sector_pct: float = 0.25           # Max sector exposure

    # V7: Fundamental bias correction
    fundamental_bias_penalty: float = 0.7  # Multiply fundamental score by this (30% penalty)

    # V7.1: Adaptive profit target exit
    profit_target_enabled: bool = True
    profit_target_pct: float = 0.18         # Exit when trade gains >= 18%

    # V7.1: Score-based exit (technical degradation)
    score_exit_enabled: bool = True
    score_exit_threshold: float = 35.0      # Exit if tech_score drops below this
    score_exit_check_interval: int = 5      # Re-check every N days

    # V7.1: Regime-change trailing stop tightening
    regime_tightening_enabled: bool = True
    regime_tight_atr_multiplier: float = 2.0  # Tighter ATR multiplier in BEAR/VOLATILE

    # V7.1: Cross-sectional momentum scoring
    momentum_scoring_enabled: bool = True
    momentum_lookback_months: int = 6       # Lookback period for relative momentum
    momentum_bonus_max: float = 10.0        # Max bonus points for top momentum
    momentum_penalty_max: float = 5.0       # Max penalty for bottom momentum

    # V7.1: Market breadth filter
    breadth_filter_enabled: bool = True
    breadth_bearish_threshold: float = 0.30  # Below 30% = reduce sizing
    breadth_sizing_reduction: float = 0.50   # Reduce to 50% size when breadth bearish

    # V7.1: Regime-adaptive thresholds
    regime_adaptive_enabled: bool = True
    regime_buy_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'BULL': 53.0,
        'RANGE': 55.0,
        'BEAR': 60.0,
        'VOLATILE': 58.0
    })
    regime_atr_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'BULL': 3.5,
        'RANGE': 3.5,
        'BEAR': 2.5,
        'VOLATILE': 2.5
    })

    # V7.1: Volatility-scaled position sizing
    vol_scaling_enabled: bool = True
    vol_target: float = 0.15               # Target annualized vol (15%)
    vol_scaling_min: float = 0.5            # Min scaling factor (don't go below 50%)
    vol_scaling_max: float = 1.5            # Max scaling factor (don't go above 150%)

    # V8: Macro regime signals (yield curve, credit spreads, etc.)
    macro_regime_enabled: bool = False     # Blend macro signals into regime detection
    # V8: Thematic ETF momentum bonus
    thematic_etf_enabled: bool = False     # Add ETF outperformance bonus to scoring

    # V7: Sector-regime scoring (disabled by default - no measurable impact in backtest)
    sector_regime_scoring: bool = False    # Apply sector bonus/malus by regime

    # V7: ML trained model path
    ml_model_path: str = 'models/ml_model_v7.joblib'
    ml_scaler_path: str = 'models/ml_scaler_v7.joblib'
    use_regime_models: bool = False        # Use per-regime ML models

    # Pillar weights by regime (from pillar_weights.json)
    regime_weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'BULL': {
            'technical': 0.22, 'fundamental': 0.15,
            'sentiment': 0.12, 'news': 0.05, 'ml': 0.26, 'trend': 0.20
        },
        'BEAR': {
            'technical': 0.22, 'fundamental': 0.28,
            'sentiment': 0.18, 'news': 0.05, 'ml': 0.16, 'trend': 0.11
        },
        'RANGE': {
            'technical': 0.28, 'fundamental': 0.22,
            'sentiment': 0.15, 'news': 0.05, 'ml': 0.17, 'trend': 0.13
        },
        'VOLATILE': {
            'technical': 0.30, 'fundamental': 0.18,
            'sentiment': 0.17, 'news': 0.05, 'ml': 0.18, 'trend': 0.12
        }
    })

    # Regime-specific stop loss overrides
    regime_stops: Dict[str, float] = field(default_factory=lambda: {
        'BULL': 0.08,
        'BEAR': 0.05,
        'RANGE': 0.06,
        'VOLATILE': 0.04
    })


# ── Profile Presets ─────────────────────────────────────────────────────

# Diversified universe (30 symbols across sectors)
SYMBOLS_DIVERSIFIED = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',      # Tech
    'JPM', 'BAC', 'GS', 'V', 'MA',                  # Financials
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK',             # Healthcare
    'XOM', 'CVX', 'COP',                              # Energy
    'PG', 'KO', 'PEP', 'WMT', 'COST',               # Consumer
    'CAT', 'HON', 'UPS',                              # Industrials
    'DIS', 'NFLX', 'CMCSA',                           # Media
    'NEE',                                              # Utilities
]

# Tech-heavy universe (30 symbols, Nasdaq 100 top holdings + growth)
SYMBOLS_TECH_HEAVY = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',  # Mega tech
    'AVGO', 'ADBE', 'CRM', 'AMD', 'INTC', 'QCOM', 'TXN',      # Semis + software
    'NFLX', 'PYPL', 'ISRG', 'INTU', 'AMAT', 'LRCX',           # Growth tech
    'COST', 'PEP', 'AMGN', 'GILD', 'MRNA',                      # Nasdaq staples/biotech
    'V', 'MA', 'UNH', 'JPM', 'GS',                               # Financials (some)
]


def make_profile_config(profile):
    # type: (str) -> Tuple[V6BacktestConfig, List[str]]
    """Create a (config, symbols) pair for a named profile.

    Args:
        profile: 'growth' or 'safe'

    Returns:
        (V6BacktestConfig, list of symbols)
    """
    if profile == 'growth':
        # Tech-Heavy 1.0x + momentum - approaches QQQ CAGR with better Sharpe
        # Ablation: momentum is the ONLY V7.1 feature that improves Growth
        #   +2.65% CAGR, +0.05 Sharpe (selects top-performing stocks)
        config = V6BacktestConfig(
            buy_threshold=55.0,
            trailing_atr_multiplier=3.5,
            scoring_mode=ScoringMode.THREE_PILLARS,
            leverage=1.0,
            # Only momentum ON (proven +2.65% CAGR in ablation)
            profit_target_enabled=False,
            score_exit_enabled=False,
            regime_tightening_enabled=False,
            momentum_scoring_enabled=True,
            breadth_filter_enabled=False,
            regime_adaptive_enabled=False,
            vol_scaling_enabled=False,
        )
        return config, list(SYMBOLS_TECH_HEAVY)

    elif profile == 'safe':
        # Diversified 1.3x + regime thresholds - beats SPY, half the MaxDD
        # V8: macro_regime adds +1.22% CAGR, +0.07 Sharpe, -0.8% MaxDD
        config = V6BacktestConfig(
            buy_threshold=55.0,
            trailing_atr_multiplier=3.5,
            scoring_mode=ScoringMode.THREE_PILLARS,
            leverage=1.3,
            margin_cost_annual=0.06,
            # Only regime_adaptive ON (best Sharpe/DD reducer from ablation)
            profit_target_enabled=False,
            score_exit_enabled=False,
            regime_tightening_enabled=False,
            momentum_scoring_enabled=False,
            breadth_filter_enabled=False,
            regime_adaptive_enabled=True,
            vol_scaling_enabled=False,
            # V8: macro regime improves Safe (+1.22% CAGR, +0.07 Sharpe, -0.8% MaxDD)
            macro_regime_enabled=True,
            thematic_etf_enabled=False,
        )
        return config, list(SYMBOLS_DIVERSIFIED)

    else:
        raise ValueError("Unknown profile: %s. Use 'growth' or 'safe'." % profile)


# ── Data Management ──────────────────────────────────────────────────────

class BacktestDataManager:
    """Manages historical data fetching and caching for backtesting"""

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._fundamental_cache: Dict[str, Dict] = {}
        self._failed: set = set()  # Cache failed symbols to avoid retrying

    def get_ohlcv(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV data via yfinance with caching"""
        cache_key = f"{symbol}_{start}_{end}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        if cache_key in self._failed:
            return None

        try:
            df = yf.download(symbol, start=start, end=end, progress=False)
            if df is None or df.empty or len(df) < 50:
                self._failed.add(cache_key)
                return None

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            self._cache[cache_key] = df
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
            self._failed.add(cache_key)
            return None

    def get_spy_data(self, start: str, end: str) -> Optional[pd.DataFrame]:
        """Fetch SPY data for regime detection"""
        return self.get_ohlcv("SPY", start, end)

    def get_vix_data(self, start: str, end: str) -> Optional[pd.DataFrame]:
        """Fetch VIX data for regime detection"""
        return self.get_ohlcv("^VIX", start, end)

    def get_fundamentals(self, symbol: str) -> Dict:
        """Fetch fundamental data (cached). Note: uses CURRENT fundamentals
        which introduces survivorship/lookahead bias. This is acknowledged."""
        if symbol in self._fundamental_cache:
            return self._fundamental_cache[symbol]

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            self._fundamental_cache[symbol] = info
            return info
        except Exception as e:
            logger.warning(f"Failed to fetch fundamentals for {symbol}: {e}")
            return {}


# ── Technical Scoring (replicates technical_pillar.py logic) ─────────────

class PrecomputedIndicators:
    """Pre-computed indicator time series for a single symbol.
    Computed ONCE, then looked up by date index - avoids recomputation."""

    def __init__(self, df: pd.DataFrame):
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        n = len(df)

        # ── EMAs ──
        self.ema20 = close.ewm(span=20, adjust=False).mean()
        self.ema50 = close.ewm(span=50, adjust=False).mean()
        self.ema200 = close.ewm(span=200, adjust=False).mean() if n >= 200 else None
        self.ema12 = close.ewm(span=12, adjust=False).mean()
        self.ema26 = close.ewm(span=26, adjust=False).mean()
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume

        # ── MACD ──
        macd = self.ema12 - self.ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        self.macd = macd
        self.macd_signal = signal
        self.macd_hist = macd - signal

        # ── ADX / DI ──
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.atr = tr.rolling(14).mean()

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        self.plus_di = 100 * (plus_dm.rolling(14).mean() / self.atr)
        self.minus_di = 100 * (minus_dm.rolling(14).mean() / self.atr)
        dx = 100 * abs(self.plus_di - self.minus_di) / (self.plus_di + self.minus_di)
        self.adx = dx.rolling(14).mean()

        # ── RSI ──
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss_s
        self.rsi = 100 - (100 / (1 + rs))

        # ── Stochastic ──
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        range14 = high14 - low14
        self.stoch_k = (close - low14) / range14 * 100
        self.stoch_d = self.stoch_k.rolling(3).mean()

        # ── Volume ──
        self.avg_vol = volume.rolling(20).mean()
        self.price_change = close.pct_change()

        # ── OBV ──
        self.obv = (np.sign(close.diff()) * volume).cumsum()

        # ── Bollinger Bands ──
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        self.bb_upper = sma20 + 2 * std20
        self.bb_lower = sma20 - 2 * std20

        # ATR average for ratio
        self.atr_avg = self.atr.rolling(20).mean()

        # ── Volatility (20-day returns std) ──
        self.vol_20d = close.pct_change().rolling(20).std()

        # ── Pre-compute the final tech + ML scores using VECTORIZED operations ──
        self._precompute_scores_vectorized(n)

    def _precompute_scores_vectorized(self, n: int):
        """Fully vectorized score computation - no Python for-loops."""
        idx = self.close.index
        c = self.close.values
        h = self.high.values
        lo = self.low.values
        v = self.volume.values

        e20 = self.ema20.values
        e50 = self.ema50.values
        e200 = self.ema200.values if self.ema200 is not None else None

        # ── TREND SCORE (3 sub-scores averaged) ──
        # 1. EMA alignment
        ema_score = np.zeros(n)
        if e200 is not None:
            bull_align = (c > e20) & (e20 > e50) & (e50 > e200)
            bear_align = (c < e20) & (e20 < e50) & (e50 < e200)
            above_50 = c > e50
            below_50 = c < e50
            ema_score = np.where(bull_align, 80,
                       np.where(bear_align, -80,
                       np.where(above_50, 30,
                       np.where(below_50, -30, 0))))
            # Before 200 bars, use simplified
            mask_short = np.arange(n) < 200
            bull_short = (c > e20) & (e20 > e50) & mask_short
            bear_short = (c < e20) & (e20 < e50) & mask_short
            ema_score = np.where(bull_short, 60,
                       np.where(bear_short, -60,
                       np.where(mask_short, 0, ema_score)))
        else:
            bull_s = (c > e20) & (e20 > e50)
            bear_s = (c < e20) & (e20 < e50)
            ema_score = np.where(bull_s, 60, np.where(bear_s, -60, 0))

        # 2. MACD
        macd_v = self.macd.values
        sig_v = self.macd_signal.values
        hist_v = self.macd_hist.values
        prev_hist = np.roll(hist_v, 1); prev_hist[0] = hist_v[0]

        macd_bull = (macd_v > sig_v) & (hist_v > 0)
        macd_bear = (macd_v < sig_v) & (hist_v < 0)
        macd_score = np.where(macd_bull & (hist_v > prev_hist), 70,
                    np.where(macd_bull, 40,
                    np.where(macd_bear & (hist_v < prev_hist), -70,
                    np.where(macd_bear, -40, 0))))

        # 3. ADX
        adx_v = np.nan_to_num(self.adx.values, nan=25.0)
        pdi_v = np.nan_to_num(self.plus_di.values, nan=25.0)
        mdi_v = np.nan_to_num(self.minus_di.values, nan=25.0)
        adx_strong = adx_v > 25
        adx_bull = adx_strong & (pdi_v > mdi_v)
        adx_bear = adx_strong & (pdi_v <= mdi_v)
        adx_score = np.where(adx_bull, np.minimum(80, 30 + adx_v),
                   np.where(adx_bear, np.maximum(-80, -30 - adx_v), 0))

        trend_raw = (ema_score + macd_score + adx_score) / 3.0

        # ── MOMENTUM SCORE (2 sub-scores averaged) ──
        rsi_v = np.nan_to_num(self.rsi.values, nan=50.0)
        prev_rsi = np.roll(rsi_v, 1); prev_rsi[0] = rsi_v[0]
        rsi_score = np.where(rsi_v < 30, 60,
                   np.where(rsi_v > 70, -60,
                   np.where((rsi_v > 50) & (rsi_v > prev_rsi), 30,
                   np.where((rsi_v < 50) & (rsi_v < prev_rsi), -30, 0))))

        k_v = np.nan_to_num(self.stoch_k.values, nan=50.0)
        d_v = np.nan_to_num(self.stoch_d.values, nan=50.0)
        stoch_score = np.where((k_v < 20) & (d_v < 20), 50,
                     np.where((k_v > 80) & (d_v > 80), -50,
                     np.where(k_v > d_v, 20,
                     np.where(k_v < d_v, -20, 0))))

        momentum_raw = (rsi_score + stoch_score) / 2.0

        # ── VOLUME SCORE (2 sub-scores averaged) ──
        avg_v = np.nan_to_num(self.avg_vol.values, nan=1.0)
        avg_v = np.where(avg_v == 0, 1.0, avg_v)
        vol_ratio = v / avg_v
        p_chg = np.nan_to_num(self.price_change.values, nan=0.0)
        pos_chg = p_chg > 0

        vol_score1 = np.where(vol_ratio > 2.0, np.where(pos_chg, 70, -70),
                    np.where(vol_ratio > 1.5, np.where(pos_chg, 40, -40),
                    np.where(vol_ratio < 0.5, 0,
                    np.where(pos_chg, 10, -10))))

        # OBV divergence (5-day)
        obv_v = self.obv.values
        obv_5 = np.roll(obv_v, 5)
        close_5 = np.roll(c, 5)
        obv_5_abs = np.abs(obv_5)
        obv_5_abs = np.where(obv_5_abs == 0, 1, obv_5_abs)
        obv_change = (obv_v - obv_5) / obv_5_abs
        price_5d = (c - close_5) / np.where(close_5 == 0, 1, close_5)
        obv_pos = obv_change > 0
        obv_neg = obv_change < 0
        price_pos = price_5d > 0
        price_neg = price_5d < 0

        vol_score2 = np.where(obv_pos & price_pos, 40,
                    np.where(obv_neg & price_neg, -40,
                    np.where(obv_pos & price_neg, 50,
                    np.where(obv_neg & price_pos, -50, 0))))
        # First 5 bars: no OBV divergence
        vol_score2[:5] = 0

        volume_raw = (vol_score1 + vol_score2) / 2.0

        # ── VOLATILITY SCORE (2 sub-scores averaged) ──
        atr_v = np.nan_to_num(self.atr.values, nan=0.0)
        atr_avg_v = np.nan_to_num(self.atr_avg.values, nan=0.0)
        atr_avg_v = np.where(atr_avg_v == 0, 1, atr_avg_v)
        atr_ratio = atr_v / atr_avg_v
        atr_score = np.where(atr_ratio > 1.5, -20,
                   np.where(atr_ratio < 0.7, 20, 0))

        bb_u = np.nan_to_num(self.bb_upper.values, nan=0.0)
        bb_l = np.nan_to_num(self.bb_lower.values, nan=0.0)
        bb_range = bb_u - bb_l
        bb_range_safe = np.where(bb_range == 0, 1, bb_range)
        bb_pct = (c - bb_l) / bb_range_safe

        bb_score = np.where(c < bb_l, 50,
                  np.where(c > bb_u, -50,
                  np.where(bb_pct < 0.2, 30,
                  np.where(bb_pct > 0.8, -30, 0))))

        volatility_raw = (atr_score + bb_score) / 2.0

        # ── Weighted tech score ──
        tech_raw = trend_raw * 0.30 + momentum_raw * 0.25 + volume_raw * 0.25 + volatility_raw * 0.20
        self.tech_scores = pd.Series((tech_raw + 100) / 2, index=idx)
        # Set first 100 bars to neutral
        self.tech_scores.iloc[:100] = 50.0

        # ── ML heuristic score (vectorized) ──
        ml = np.full(n, 50.0)
        ml += np.where(e20 > e50, 5, -5)  # Trend
        ml += np.where(macd_v > sig_v, 5, -5)  # MACD
        ml += np.where(adx_strong & (pdi_v > mdi_v), 5,
              np.where(adx_strong & (pdi_v <= mdi_v), -5, 0))  # ADX
        ml += np.where(rsi_v < 30, 10, np.where(rsi_v > 70, -10,
              np.where(rsi_v > 50, 3, 0)))  # RSI
        ml += np.where(k_v < 20, 5, np.where(k_v > 80, -5, 0))  # Stochastic
        ml += np.where(vol_ratio > 2, np.where(pos_chg, 10, -10),
              np.where(vol_ratio > 1.5, np.where(pos_chg, 5, -5), 0))  # Volume
        ml += np.where(bb_pct < 0.2, 8, np.where(bb_pct > 0.8, -8, 0))  # BB

        self.ml_scores = pd.Series(np.clip(ml, 0, 100), index=idx)
        self.ml_scores.iloc[:100] = 50.0


class BacktestTechnicalScorer:
    """Replicates TechnicalPillar scoring using pre-computed indicators.
    Indicators are computed ONCE per symbol, then looked up by date."""

    def __init__(self):
        self._indicators: Dict[str, PrecomputedIndicators] = {}

    def precompute(self, symbol: str, df: pd.DataFrame):
        """Pre-compute all indicators for a symbol (called once)."""
        if df is not None and len(df) >= 100:
            self._indicators[symbol] = PrecomputedIndicators(df)

    def score_at(self, symbol: str, date_loc: int) -> float:
        """Look up pre-computed technical score at a date index."""
        ind = self._indicators.get(symbol)
        if ind is None or date_loc >= len(ind.tech_scores):
            return 50.0
        return float(ind.tech_scores.iloc[date_loc])

    def ml_score_at(self, symbol: str, date_loc: int) -> Tuple[float, float]:
        """Look up pre-computed ML score and volatility at a date index."""
        ind = self._indicators.get(symbol)
        if ind is None or date_loc >= len(ind.ml_scores):
            return 50.0, 0.02
        ml_score = float(ind.ml_scores.iloc[date_loc])
        vol = float(ind.vol_20d.iloc[date_loc]) if not pd.isna(ind.vol_20d.iloc[date_loc]) else 0.02
        return ml_score, vol

    def get_atr_at(self, symbol: str, date_loc: int) -> float:
        """Get ATR value at date index."""
        ind = self._indicators.get(symbol)
        if ind is None or date_loc >= len(ind.atr):
            return 0.0
        val = ind.atr.iloc[date_loc]
        return float(val) if not pd.isna(val) else 0.0

    def score(self, df: pd.DataFrame) -> float:
        """Fallback: compute from scratch (used for small runs)."""
        if df is None or len(df) < 50:
            return 50.0
        ind = PrecomputedIndicators(df)
        return float(ind.tech_scores.iloc[-1])

    def _calc_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Fallback ATR calculation."""
        high, low, close = df['High'], df['Low'], df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()


# ── Fundamental Scoring (replicates fundamental_pillar.py logic) ─────────

class BacktestFundamentalScorer:
    """Replicates FundamentalPillar scoring from yfinance info dict.
    NOTE: Uses current fundamentals -> introduces lookahead bias.
    This is acknowledged and can't be avoided without expensive historical data.

    V7: Apply uncertainty penalty (30%) to account for using current fundamentals
    on historical data. Score is multiplied by bias_penalty factor.
    """

    def __init__(self, bias_penalty: float = 1.0):
        self.bias_penalty = bias_penalty  # V7: 0.7 = 30% penalty for lookahead
        self.sector_pe = {
            'Technology': 30, 'Healthcare': 25, 'Financial Services': 15,
            'Consumer Cyclical': 20, 'Consumer Defensive': 22, 'Energy': 12,
            'Industrials': 18, 'Basic Materials': 14, 'Utilities': 18,
            'Real Estate': 35, 'Communication Services': 22, 'default': 20
        }

    @staticmethod
    def _safe_num(val):
        """Safely convert yfinance info value to float."""
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def score(self, info: Dict) -> float:
        """Calculate fundamental score. Returns 0-100."""
        if not info:
            return 50.0  # Neutral

        scores = []

        # Valuation
        pe = self._safe_num(info.get('trailingPE')) or self._safe_num(info.get('forwardPE'))
        sector = info.get('sector', 'default')
        sector_pe = self.sector_pe.get(sector, 20)

        if pe and pe > 0:
            if pe < sector_pe * 0.6:
                scores.append(70)
            elif pe < sector_pe * 0.9:
                scores.append(40)
            elif pe < sector_pe * 1.3:
                scores.append(0)
            elif pe < sector_pe * 2:
                scores.append(-40)
            else:
                scores.append(-70)

        # PEG
        peg = self._safe_num(info.get('pegRatio'))
        if peg and peg > 0:
            if peg < 1:
                scores.append(60)
            elif peg < 1.5:
                scores.append(30)
            elif peg < 2:
                scores.append(0)
            else:
                scores.append(-40)

        # Revenue Growth
        rev_growth = self._safe_num(info.get('revenueGrowth'))
        if rev_growth is not None:
            if rev_growth > 0.30:
                scores.append(70)
            elif rev_growth > 0.15:
                scores.append(50)
            elif rev_growth > 0.05:
                scores.append(20)
            elif rev_growth > 0:
                scores.append(0)
            else:
                scores.append(-50)

        # Profit Margin
        margin = self._safe_num(info.get('profitMargins'))
        if margin is not None:
            if margin > 0.20:
                scores.append(60)
            elif margin > 0.10:
                scores.append(40)
            elif margin > 0.05:
                scores.append(20)
            elif margin > 0:
                scores.append(0)
            else:
                scores.append(-50)

        # Debt/Equity
        de = self._safe_num(info.get('debtToEquity'))
        if de is not None:
            if de < 30:
                scores.append(50)
            elif de < 80:
                scores.append(30)
            elif de < 150:
                scores.append(0)
            else:
                scores.append(-50)

        if not scores:
            return 50.0

        raw = np.mean(scores)  # -100 to +100 range
        normalized = (raw + 100) / 2  # Normalize to 0-100

        # V7: Apply bias penalty - pull score toward neutral (50)
        if self.bias_penalty < 1.0:
            normalized = 50 + (normalized - 50) * self.bias_penalty

        return normalized


# ── ML Feature Extraction (replicates ml_pillar.py logic) ────────────────

class BacktestMLScorer:
    """ML scoring now handled by PrecomputedIndicators.
    This class kept for backward compatibility."""

    def score(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Fallback: compute from scratch (used when indicators not pre-computed)."""
        if df is None or len(df) < 50:
            return 50.0, 0.02
        ind = PrecomputedIndicators(df)
        ml_score = float(ind.ml_scores.iloc[-1])
        vol = float(ind.vol_20d.iloc[-1]) if not pd.isna(ind.vol_20d.iloc[-1]) else 0.02
        return ml_score, vol


# ── VIX Sentiment Proxy (for Mode C) ────────────────────────────────────

class BacktestVIXSentimentProxy:
    """Uses VIX as a proxy for market sentiment in backtesting.

    V7 Enhanced: Incorporates VIX level + rate of change + mean reversion.

    Contrarian logic (matching live contrarian_sentiment mode):
    - Extreme fear (VIX > 35) = panic → contrarian BUYING opportunity
    - High fear (VIX 28-35) = elevated fear → mildly bullish contrarian
    - Normal fear (VIX 20-28) = neutral to slightly cautious
    - Low fear (VIX 14-20) = normal market → neutral
    - Very low (VIX < 14) = complacency/euphoria → contrarian SELL signal

    V7 additions:
    - VIX spike detection (rapid increase = panic, not sustained high)
    - VIX mean-reversion signal (extreme VIX tends to revert)
    - VIX trend (rising vs falling VIX)
    """

    def score(self, vix_level: float, vix_5d_ago: float = None, vix_20d_ago: float = None) -> float:
        """Convert VIX level to a sentiment score (0-100).
        50 = neutral, >50 = bullish, <50 = bearish.

        Args:
            vix_level: Current VIX level
            vix_5d_ago: VIX level 5 days ago (for spike detection)
            vix_20d_ago: VIX level 20 days ago (for trend)
        """
        # Base score from VIX level (contrarian)
        if vix_level > 40:
            base = 75.0
        elif vix_level > 35:
            base = 68.0
        elif vix_level > 28:
            base = 60.0
        elif vix_level > 22:
            base = 46.0
        elif vix_level > 18:
            base = 50.0
        elif vix_level > 14:
            base = 52.0
        elif vix_level > 11:
            base = 40.0
        else:
            base = 32.0

        # V7: VIX spike adjustment (rapid increase = buy signal)
        if vix_5d_ago is not None and vix_5d_ago > 0:
            vix_change_5d = (vix_level - vix_5d_ago) / vix_5d_ago
            if vix_change_5d > 0.30:  # VIX spiked >30% in 5 days = panic
                base = min(base + 10, 85.0)  # Strong contrarian buy
            elif vix_change_5d > 0.15:
                base = min(base + 5, 80.0)
            elif vix_change_5d < -0.20:  # VIX collapsed = complacency building
                base = max(base - 5, 25.0)

        # V7: VIX trend adjustment (mean reversion)
        if vix_20d_ago is not None and vix_20d_ago > 0:
            vix_change_20d = (vix_level - vix_20d_ago) / vix_20d_ago
            if vix_level > 30 and vix_change_20d < -0.10:
                # VIX was high but is falling → recovery underway
                base = min(base + 5, 80.0)
            elif vix_level < 15 and vix_change_20d > 0.10:
                # VIX was low but rising → caution building
                base = max(base - 3, 30.0)

        return base


# ── Regime Detection ─────────────────────────────────────────────────────

class BacktestRegimeDetector:
    """Detects market regime from SPY and VIX data at each point in time."""

    def __init__(self, config: V6BacktestConfig):
        self.config = config

    def detect(
        self,
        spy_df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame],
        date: pd.Timestamp
    ) -> MarketRegime:
        """Detect regime at a specific date using only data available at that time."""
        try:
            # Get SPY data up to this date
            spy_slice = spy_df.loc[:date]
            if len(spy_slice) < 50:
                return MarketRegime.RANGE

            close = spy_slice['Close']
            ema50 = close.ewm(span=50, adjust=False).mean()

            current_price = float(close.iloc[-1])
            ema50_val = float(ema50.iloc[-1])
            price_vs_ema50 = (current_price / ema50_val - 1) * 100

            # 20-day trend
            if len(close) >= 20:
                trend_20d = (float(close.iloc[-1]) / float(close.iloc[-20]) - 1) * 100
            else:
                trend_20d = 0

            # Volatility
            returns = close.pct_change()
            volatility = float(returns.rolling(20).std().iloc[-1]) * np.sqrt(252) * 100

            # VIX level
            vix_level = 20.0
            if vix_df is not None:
                vix_slice = vix_df.loc[:date]
                if not vix_slice.empty:
                    vix_level = float(vix_slice['Close'].iloc[-1])

            # Detection logic (matches regime_adapter.py)
            if vix_level > self.config.volatile_vix_threshold or volatility > self.config.volatile_vol_threshold:
                return MarketRegime.VOLATILE
            elif price_vs_ema50 > self.config.bull_threshold and trend_20d > 3 and vix_level < 20:
                return MarketRegime.BULL
            elif price_vs_ema50 < self.config.bear_threshold and trend_20d < -3:
                return MarketRegime.BEAR
            else:
                return MarketRegime.RANGE

        except Exception as e:
            logger.debug(f"Regime detection error at {date}: {e}")
            return MarketRegime.RANGE


# ── ML Gate (replicates adaptive_ml_gate.py) ─────────────────────────────

class BacktestMLGate:
    """Simulates the Adaptive ML Gate in backtesting.
    Without a trained model, uses ML heuristic confidence as proxy."""

    def __init__(self, config: V6BacktestConfig):
        self.config = config

    def apply(
        self,
        base_score: float,
        ml_score: float,
        volatility: float
    ) -> Tuple[float, str]:
        """Apply ML Gate. Returns (gated_score, mode)."""
        if not self.config.ml_gate_enabled:
            return base_score, 'DISABLED'

        # High volatility -> 5 pillars only
        if volatility > self.config.ml_gate_volatility_threshold:
            return base_score, '5P_ONLY'

        # Use ML score as confidence proxy (0-100 -> 0-1)
        ml_confidence = ml_score / 100.0

        if ml_confidence > self.config.ml_boost_threshold:
            return base_score + self.config.ml_boost_points, 'ML_BOOST'
        elif ml_confidence < self.config.ml_block_threshold:
            return 0.0, 'ML_BLOCK'
        else:
            return base_score, 'ML_NEUTRAL'


# ── ML Trained Model (Mode E) ────────────────────────────────────────────

class BacktestMLTrainedScorer:
    """V7: Uses a trained ML model instead of heuristic for scoring.

    Loads model from models/ml_model_v7.joblib and predicts
    trade success probability using extracted features.
    """

    def __init__(self, model_path: str = 'models/ml_model_v7.joblib',
                 scaler_path: str = 'models/ml_scaler_v7.joblib',
                 use_regime_models: bool = False):
        self.model = None
        self.scaler = None
        self.regime_models = {}
        self.regime_scalers = {}
        self.use_regime_models = use_regime_models
        self._feature_extractor = None

        try:
            import joblib
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                logger.info(f"[ML-E] Loaded trained model from {model_path}")

                # Load per-regime models if available
                if use_regime_models:
                    model_dir = os.path.dirname(model_path)
                    for regime in ['BULL', 'BEAR', 'RANGE', 'VOLATILE']:
                        r_model_path = os.path.join(model_dir, f'ml_model_{regime}.joblib')
                        r_scaler_path = os.path.join(model_dir, f'ml_scaler_{regime}.joblib')
                        if os.path.exists(r_model_path):
                            self.regime_models[regime] = joblib.load(r_model_path)
                            if os.path.exists(r_scaler_path):
                                self.regime_scalers[regime] = joblib.load(r_scaler_path)
                            logger.info(f"[ML-E] Loaded regime model: {regime}")
            else:
                logger.warning(f"[ML-E] Model not found at {model_path} - Mode E will fall back to heuristic")
        except Exception as e:
            logger.warning(f"[ML-E] Failed to load model: {e}")

    def _get_feature_extractor(self):
        if self._feature_extractor is None:
            try:
                from src.ml.training_pipeline import FeatureExtractor
                self._feature_extractor = FeatureExtractor()
            except ImportError:
                self._feature_extractor = None
        return self._feature_extractor

    def predict(
        self,
        df: pd.DataFrame,
        date_loc: int,
        regime: str = 'RANGE',
        spy_df: pd.DataFrame = None,
        vix_df: pd.DataFrame = None
    ) -> Tuple[float, float]:
        """Predict ML score using trained model.

        Returns:
            (ml_score 0-100, volatility)
        """
        extractor = self._get_feature_extractor()
        if extractor is None or self.model is None:
            return 50.0, 0.02  # Fallback

        features = extractor.extract_at(df, date_loc, spy_df, vix_df)
        if features is None:
            return 50.0, 0.02

        try:
            from src.ml.training_pipeline import FEATURE_NAMES
            X = np.array([[features.get(f, 0) for f in FEATURE_NAMES]])
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Use regime-specific model if available
            model = self.model
            scaler = self.scaler
            if self.use_regime_models and regime in self.regime_models:
                model = self.regime_models[regime]
                scaler = self.regime_scalers.get(regime, self.scaler)

            if scaler is not None:
                X = scaler.transform(X)

            proba = model.predict_proba(X)[0]
            success_proba = proba[1] if len(proba) > 1 else proba[0]
            ml_score = success_proba * 100

            vol = features.get('volatility_20d', 20.0) / 100.0  # Convert to decimal

            return ml_score, vol

        except Exception as e:
            logger.debug(f"[ML-E] Prediction error: {e}")
            return 50.0, 0.02

    @property
    def is_available(self) -> bool:
        return self.model is not None


# ── Position Tracking ────────────────────────────────────────────────────

@dataclass
class BacktestPosition:
    """Tracks an open position during backtesting"""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float          # Raw market price
    effective_entry: float      # After costs
    shares: int
    stop_loss: float
    highest_price: float
    confidence_score: float
    regime_at_entry: str
    trailing_atr: float = 0.0   # ATR at entry for trailing stop
    sector: str = 'Unknown'     # V7: For correlation tracking
    last_score_check_day: int = 0  # V7.1: Last day we checked tech score


# ── Main V6 Backtester ──────────────────────────────────────────────────

@dataclass
class V6BacktestResult:
    """Complete V6 backtest results"""
    config: V6BacktestConfig
    metrics: PerformanceMetrics
    trades: List[Trade]
    equity_curve: pd.Series
    daily_returns: pd.Series
    signals_evaluated: int
    signals_taken: int
    regime_history: Dict[str, int]
    ml_gate_stats: Dict[str, int]
    score_stats: Dict[str, float] = field(default_factory=dict)  # min/max/mean/median/p75/p90

    def __str__(self):
        return format_metrics_report(self.metrics)


class V6Backtester:
    """
    V6.2 Backtester - Tests the REAL 5-pillar system with all components.

    Usage:
        config = V6BacktestConfig(initial_capital=100_000)
        bt = V6Backtester(config)
        result = bt.run(symbols=['AAPL', 'MSFT', ...], start='2015-01-01', end='2025-01-01')
        print(result)
    """

    def __init__(self, config: V6BacktestConfig = None, data_mgr: BacktestDataManager = None):
        self.config = config or V6BacktestConfig()
        self.data_mgr = data_mgr or BacktestDataManager()
        self.tech_scorer = BacktestTechnicalScorer()
        self.fund_scorer = BacktestFundamentalScorer(
            bias_penalty=self.config.fundamental_bias_penalty
        )
        self.ml_scorer = BacktestMLScorer()
        self.regime_detector = BacktestRegimeDetector(self.config)
        self.ml_gate = BacktestMLGate(self.config)
        self.vix_proxy = BacktestVIXSentimentProxy()

        # V7: Trained ML model (Mode E)
        self.ml_trained = None
        if self.config.scoring_mode == ScoringMode.ML_TRAINED:
            self.ml_trained = BacktestMLTrainedScorer(
                model_path=self.config.ml_model_path,
                scaler_path=self.config.ml_scaler_path,
                use_regime_models=self.config.use_regime_models
            )

        # V8: Macro signal fetcher (direct import to bypass __init__.py zoneinfo issue)
        self.macro_fetcher = None
        if self.config.macro_regime_enabled:
            try:
                import importlib.util as _ilu
                _macro_path = os.path.join(os.path.dirname(__file__), '..', 'intelligence', 'macro_signals.py')
                _spec = _ilu.spec_from_file_location("macro_signals", _macro_path)
                _macro_mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_macro_mod)
                self.macro_fetcher = _macro_mod.MacroSignalFetcher()
            except Exception as e:
                logger.warning("MacroSignalFetcher not available: %s", e)

        # V8: ETF momentum function (loaded via importlib to bypass __init__.py)
        self._etf_momentum_func = None
        self._etf_proxies = {}
        if self.config.thematic_etf_enabled:
            try:
                import importlib.util as _ilu
                _orch_path = os.path.join(os.path.dirname(__file__), '..', 'intelligence', 'intelligence_orchestrator.py')
                _spec = _ilu.spec_from_file_location("intelligence_orchestrator", _orch_path)
                _orch_mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_orch_mod)
                self._etf_momentum_func = _orch_mod.calculate_etf_momentum_bonus
                self._etf_proxies = _orch_mod.THEME_ETF_PROXIES
            except Exception as e:
                logger.warning("ETF momentum not available: %s", e)

        # V7: Sector-regime scorer
        self.sector_scorer = None
        if self.config.sector_regime_scoring:
            try:
                import importlib.util as _ilu
                _scorer_path = os.path.join(os.path.dirname(__file__), '..', 'agents', 'sector_regime_scorer.py')
                _spec = _ilu.spec_from_file_location("sector_regime_scorer", _scorer_path)
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                self.sector_scorer = _mod.SectorRegimeScorer()
            except Exception as e:
                logger.warning(f"SectorRegimeScorer not available: {e}")

        # V7: Position sizer
        self.position_sizer = None
        if self.config.dynamic_sizing:
            try:
                from src.execution.position_sizer import PositionSizer
                self.position_sizer = PositionSizer(
                    base_size_pct=self.config.position_size_pct,
                    target_risk_pct=self.config.target_risk_pct
                )
            except ImportError:
                logger.warning("PositionSizer not available - using fixed sizing")

    def _compute_momentum_scores(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp
    ) -> Dict[str, float]:
        """V7.1: Cross-sectional momentum scoring.

        Computes relative return of each symbol vs the median of the group
        over the lookback period. Returns bonus/penalty in points.
        """
        if not self.config.momentum_scoring_enabled:
            return {}

        lookback_days = self.config.momentum_lookback_months * 21  # ~21 trading days/month
        returns = {}
        for sym, df in symbol_data.items():
            if date not in df.index:
                continue
            loc = df.index.get_loc(date)
            if loc < lookback_days:
                continue
            price_now = float(df['Close'].iloc[loc])
            price_past = float(df['Close'].iloc[loc - lookback_days])
            if price_past > 0:
                returns[sym] = (price_now - price_past) / price_past

        if len(returns) < 5:
            return {}

        median_ret = np.median(list(returns.values()))
        scores = {}
        sorted_rets = sorted(returns.values())
        n = len(sorted_rets)

        for sym, ret in returns.items():
            # Rank-based: percentile position
            rank = sum(1 for r in sorted_rets if r <= ret) / n
            if rank >= 0.75:
                # Top quartile: bonus proportional to rank
                scores[sym] = self.config.momentum_bonus_max * (rank - 0.75) / 0.25
            elif rank <= 0.25:
                # Bottom quartile: penalty proportional to rank
                scores[sym] = -self.config.momentum_penalty_max * (0.25 - rank) / 0.25
            else:
                scores[sym] = 0.0

        return scores

    def _compute_breadth(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp
    ) -> float:
        """V7.1: Market breadth - % of symbols above their EMA50.

        Returns a float 0.0-1.0 representing the fraction of symbols
        whose close is above their 50-day EMA.
        """
        above = 0
        total = 0
        for sym, df in symbol_data.items():
            if date not in df.index:
                continue
            loc = df.index.get_loc(date)
            if loc < 50:
                continue
            total += 1
            ind = self.tech_scorer._indicators.get(sym)
            if ind is not None and loc < len(ind.ema50):
                if float(ind.close.iloc[loc]) > float(ind.ema50.iloc[loc]):
                    above += 1
        return above / total if total > 0 else 0.5

    def _get_vol_scaling_factor(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        date: pd.Timestamp
    ) -> float:
        """V7.1: Volatility-scaled position sizing factor.

        Returns a multiplier for position size based on realized market vol
        vs target vol. If realized vol is 2x target, factor = 0.5.
        """
        if not self.config.vol_scaling_enabled:
            return 1.0

        # Use SPY realized vol as market proxy
        spy_key = None
        for key in self.data_mgr._cache:
            if key.startswith('SPY_'):
                spy_key = key
                break
        if spy_key is None:
            return 1.0

        spy_df = self.data_mgr._cache[spy_key]
        if date not in spy_df.index:
            return 1.0

        loc = spy_df.index.get_loc(date)
        if loc < 60:
            return 1.0

        # 20-day realized annualized vol
        returns = spy_df['Close'].pct_change().iloc[max(0, loc-20):loc+1]
        realized_vol = float(returns.std()) * np.sqrt(252)
        if realized_vol <= 0:
            return 1.0

        factor = self.config.vol_target / realized_vol
        return max(self.config.vol_scaling_min, min(self.config.vol_scaling_max, factor))

    def run(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        progress_callback=None
    ) -> V6BacktestResult:
        """Run the V6.2 backtest."""
        logger.info(f"V6.2 Backtest: {start_date} to {end_date}, {len(symbols)} symbols")
        logger.info(f"Config: capital={self.config.initial_capital}, "
                     f"buy_threshold={self.config.buy_threshold}, "
                     f"costs={self.config.costs.commission_per_trade}+{self.config.costs.spread_pct}%+{self.config.costs.slippage_pct}%")

        # Fetch SPY and VIX for regime detection (with buffer for EMA calculation)
        buffer_start = (pd.Timestamp(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
        spy_df = self.data_mgr.get_spy_data(buffer_start, end_date)
        vix_df = self.data_mgr.get_vix_data(buffer_start, end_date)

        if spy_df is None:
            logger.error("Could not fetch SPY data - aborting backtest")
            return self._empty_result()

        # V8: Pre-fetch macro indicator data
        tnx_df, irx_df, hyg_df, lqd_df, uup_df = None, None, None, None, None
        macro_sector_data = {}
        if self.config.macro_regime_enabled and self.macro_fetcher is not None:
            logger.info("Pre-fetching macro indicator data...")
            tnx_df = self.data_mgr.get_ohlcv('^TNX', buffer_start, end_date)
            irx_df = self.data_mgr.get_ohlcv('^IRX', buffer_start, end_date)
            hyg_df = self.data_mgr.get_ohlcv('HYG', buffer_start, end_date)
            lqd_df = self.data_mgr.get_ohlcv('LQD', buffer_start, end_date)
            uup_df = self.data_mgr.get_ohlcv('UUP', buffer_start, end_date)
            for etf_sym in ['XLU', 'XLP', 'XLK', 'XLY']:
                df = self.data_mgr.get_ohlcv(etf_sym, buffer_start, end_date)
                if df is not None:
                    macro_sector_data[etf_sym] = df

        # V8: Pre-fetch thematic ETF data
        if self.config.thematic_etf_enabled and self._etf_proxies:
            logger.info("Pre-fetching thematic ETF data...")
            for etf_sym in self._etf_proxies:
                self.data_mgr.get_ohlcv(etf_sym, buffer_start, end_date)

        # Pre-fetch fundamentals for all symbols
        logger.info("Pre-fetching fundamental data...")
        fundamentals = {}
        for sym in symbols:
            fundamentals[sym] = self.data_mgr.get_fundamentals(sym)

        # Pre-fetch OHLCV for all symbols
        logger.info("Pre-fetching OHLCV data...")
        symbol_data: Dict[str, pd.DataFrame] = {}
        for i, sym in enumerate(symbols):
            df = self.data_mgr.get_ohlcv(sym, buffer_start, end_date)
            if df is not None and len(df) >= 100:
                symbol_data[sym] = df
            if progress_callback and i % 10 == 0:
                progress_callback(f"Fetching data: {i+1}/{len(symbols)}")

        logger.info(f"Got data for {len(symbol_data)}/{len(symbols)} symbols")

        # Pre-compute all technical indicators ONCE per symbol
        logger.info("Pre-computing indicators for all symbols...")
        for i, (sym, sym_df) in enumerate(symbol_data.items()):
            self.tech_scorer.precompute(sym, sym_df)
            if progress_callback and i % 50 == 0:
                progress_callback(f"Computing indicators: {i+1}/{len(symbol_data)}")
        logger.info("Indicators pre-computed")

        # Build trading day index from SPY
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        trading_days = spy_df.loc[start_ts:end_ts].index

        # Portfolio state
        capital = self.config.initial_capital
        positions: Dict[str, BacktestPosition] = {}
        all_trades: List[Trade] = []
        equity_history: Dict[pd.Timestamp, float] = {}
        signals_evaluated = 0
        signals_taken = 0
        regime_counts: Dict[str, int] = {'BULL': 0, 'BEAR': 0, 'RANGE': 0, 'VOLATILE': 0}
        gate_stats: Dict[str, int] = {'5P_ONLY': 0, 'ML_BOOST': 0, 'ML_BLOCK': 0, 'ML_NEUTRAL': 0, 'DISABLED': 0}
        score_distribution: List[float] = []  # Track all generated scores
        total_margin_cost = 0.0  # V7.1: Accumulated margin interest

        # Walk forward through trading days
        mode = self.config.scoring_mode
        logger.info(f"Walking forward through {len(trading_days)} trading days (mode={mode.value})...")
        for day_idx, date in enumerate(trading_days):
            # Detect regime (V8: with optional macro blend)
            regime = self.regime_detector.detect(spy_df, vix_df, date)

            # V8: Blend macro signals into regime if enabled
            if self.macro_fetcher is not None and self.config.macro_regime_enabled:
                try:
                    macro_result = self.macro_fetcher.get_macro_regime_score(
                        date=date, tnx_df=tnx_df, irx_df=irx_df,
                        hyg_df=hyg_df, lqd_df=lqd_df, uup_df=uup_df,
                        sector_data=macro_sector_data
                    )
                    if macro_result.confidence > 0.3:
                        # Map current regime to numeric, blend, re-map
                        regime_num = {'BULL': 0.5, 'BEAR': -0.5, 'RANGE': 0.0, 'VOLATILE': -0.3}
                        tech_score = regime_num.get(regime.value, 0.0)
                        blended = tech_score * 0.70 + macro_result.score * 0.30
                        if blended > 0.25:
                            regime = MarketRegime.BULL
                        elif blended < -0.25:
                            regime = MarketRegime.BEAR
                        elif blended < -0.15:
                            regime = MarketRegime.VOLATILE
                        else:
                            regime = MarketRegime.RANGE
                except Exception as e:
                    logger.debug("Macro regime blend failed at %s: %s", date, e)

            regime_counts[regime.value] += 1

            # Get VIX level for mode C/E
            vix_level = 20.0
            vix_5d_ago = None
            vix_20d_ago = None
            if vix_df is not None:
                vix_slice = vix_df.loc[:date]
                if not vix_slice.empty:
                    vix_level = float(vix_slice['Close'].iloc[-1])
                    # V7: Historical VIX for enhanced proxy
                    if len(vix_slice) >= 6:
                        vix_5d_ago = float(vix_slice['Close'].iloc[-6])
                    if len(vix_slice) >= 21:
                        vix_20d_ago = float(vix_slice['Close'].iloc[-21])

            # Mark-to-market: calculate portfolio value
            invested_value = 0.0
            portfolio_value = capital
            for pos in positions.values():
                sym_df = symbol_data.get(pos.symbol)
                if sym_df is not None and date in sym_df.index:
                    current_price = float(sym_df.loc[date, 'Close'])
                else:
                    current_price = pos.entry_price
                pos_val = pos.shares * current_price
                invested_value += pos_val
                portfolio_value += pos_val

            # V7.1: Margin interest cost (daily)
            # If capital < 0, we're borrowing from broker
            if capital < 0:
                daily_rate = self.config.margin_cost_annual / 252
                margin_interest = abs(capital) * daily_rate
                capital -= margin_interest
                portfolio_value -= margin_interest
                total_margin_cost += margin_interest

            equity_history[date] = portfolio_value

            # Check exits for all positions
            symbols_to_close = []
            for sym, pos in positions.items():
                sym_df = symbol_data.get(sym)
                if sym_df is None or date not in sym_df.index:
                    continue

                current_price = float(sym_df.loc[date, 'Close'])
                current_low = float(sym_df.loc[date, 'Low'])
                current_high = float(sym_df.loc[date, 'High'])

                # Update highest price
                if current_high > pos.highest_price:
                    pos.highest_price = current_high

                should_exit, exit_reason = self._check_exit(
                    pos, current_price, current_low, date, regime,
                    symbol_data=symbol_data
                )

                if should_exit:
                    trade = self._close_position(pos, current_price, date, exit_reason)
                    all_trades.append(trade)
                    # Return full exit proceeds (invested capital + net P&L)
                    effective_exit_price = current_price * self.config.costs.exit_cost_factor()
                    capital += effective_exit_price * pos.shares - self.config.costs.commission_per_trade
                    symbols_to_close.append(sym)

            for sym in symbols_to_close:
                del positions[sym]

            # Check entries (weekly frequency for efficiency, matching live system)
            if day_idx % 5 != 0:  # Scan weekly
                continue

            # Get regime-specific weights
            weights = self.config.regime_weights.get(regime.value, self.config.regime_weights['RANGE'])
            regime_stop = self.config.regime_stops.get(regime.value, self.config.default_stop_loss_pct)

            # V7.1: Compute cross-sectional momentum scores for this date
            momentum_scores = self._compute_momentum_scores(symbol_data, date)

            # V7.1: Compute market breadth
            breadth = self._compute_breadth(symbol_data, date)

            # V7.1: Compute vol scaling factor
            vol_scale = self._get_vol_scaling_factor(symbol_data, date)

            # V7.1: Regime-adaptive buy threshold
            if self.config.regime_adaptive_enabled:
                effective_buy_threshold = self.config.regime_buy_thresholds.get(
                    regime.value, self.config.buy_threshold
                )
            else:
                effective_buy_threshold = self.config.buy_threshold

            for sym, sym_df in symbol_data.items():
                if sym in positions:
                    continue  # Already holding
                if len(positions) >= self.config.max_positions:
                    break  # Max positions reached

                if date not in sym_df.index:
                    continue

                # Get date index location (no lookahead)
                date_loc = sym_df.index.get_loc(date)
                if date_loc < 100:
                    continue

                signals_evaluated += 1

                # ── Score each pillar using pre-computed lookups ──

                # 1. Technical score (0-100) - pre-computed
                tech_score = self.tech_scorer.score_at(sym, date_loc)

                if mode == ScoringMode.TECH_ONLY:
                    base_score = tech_score
                    ml_score = 50.0
                    volatility = 0.02
                    gate_mode = 'DISABLED'

                elif mode == ScoringMode.ML_TRAINED:
                    # V7 Mode E: Use trained ML model as confidence gate
                    fund_score = self.fund_scorer.score(fundamentals.get(sym, {}))
                    sentiment_score = self.vix_proxy.score(vix_level, vix_5d_ago, vix_20d_ago)

                    # Base score from non-ML pillars (tech + fund + sentiment)
                    w_tech = weights.get('technical', 0.25) + weights.get('trend', 0.15)
                    w_fund = weights.get('fundamental', 0.20)
                    w_sent = weights.get('sentiment', 0.15)
                    w_base = w_tech + w_fund + w_sent
                    base_score = (
                        tech_score * (w_tech / w_base) +
                        fund_score * (w_fund / w_base) +
                        sentiment_score * (w_sent / w_base)
                    )

                    # Only call expensive ML prediction for promising candidates
                    ml_score = 50.0
                    volatility = 0.02
                    if base_score >= 55.0 and self.ml_trained and self.ml_trained.is_available:
                        ml_score, volatility = self.ml_trained.predict(
                            sym_df, date_loc, regime.value, spy_df, vix_df
                        )
                        # ML acts as gate: boost/penalize based on ML confidence
                        ml_confidence = (ml_score - 50.0) / 50.0  # -1 to +1
                        ml_adjustment = ml_confidence * 10.0  # +/- 10 points max
                        base_score = max(0, min(100, base_score + ml_adjustment))

                    gate_mode = 'DISABLED'

                else:
                    # Modes B, C, D: all backtestable pillars
                    fund_score = self.fund_scorer.score(fundamentals.get(sym, {}))
                    ml_score, volatility = self.tech_scorer.ml_score_at(sym, date_loc)

                    if mode == ScoringMode.THREE_PILLARS:
                        w_tech = weights.get('technical', 0.25) + weights.get('trend', 0.15)
                        w_fund = weights.get('fundamental', 0.20)
                        w_ml = weights.get('ml', 0.20)
                        w_total = w_tech + w_fund + w_ml
                        base_score = (
                            tech_score * (w_tech / w_total) +
                            fund_score * (w_fund / w_total) +
                            ml_score * (w_ml / w_total)
                        )

                    elif mode == ScoringMode.THREE_PILLARS_VIX:
                        sentiment_score = self.vix_proxy.score(vix_level, vix_5d_ago, vix_20d_ago)
                        news_score = 50.0
                        base_score = (
                            tech_score * (weights.get('technical', 0.25) + weights.get('trend', 0.15)) +
                            fund_score * weights.get('fundamental', 0.20) +
                            sentiment_score * weights.get('sentiment', 0.15) +
                            news_score * weights.get('news', 0.05) +
                            ml_score * weights.get('ml', 0.20)
                        )

                    else:
                        sentiment_score = 50.0
                        news_score = 50.0
                        base_score = (
                            tech_score * (weights.get('technical', 0.25) + weights.get('trend', 0.15)) +
                            fund_score * weights.get('fundamental', 0.20) +
                            sentiment_score * weights.get('sentiment', 0.15) +
                            news_score * weights.get('news', 0.05) +
                            ml_score * weights.get('ml', 0.20)
                        )

                    gate_mode = 'DISABLED'

                # ── V7: Sector-regime adjustment ──
                if self.sector_scorer is not None:
                    sector_adj = self.sector_scorer.get_adjustment(
                        sym, regime=regime.value,
                        market_cap=fundamentals.get(sym, {}).get('marketCap')
                    )
                    base_score = max(0, min(100, base_score + sector_adj))

                # ── Apply ML Gate ──
                if mode != ScoringMode.TECH_ONLY:
                    gated_score, gate_mode = self.ml_gate.apply(base_score, ml_score, volatility)
                else:
                    gated_score = base_score
                gate_stats[gate_mode] = gate_stats.get(gate_mode, 0) + 1

                # ── V7.1: Add cross-sectional momentum bonus ──
                if momentum_scores and sym in momentum_scores:
                    gated_score = max(0, min(100, gated_score + momentum_scores[sym]))

                # ── V8: Thematic ETF momentum bonus ──
                if self.config.thematic_etf_enabled and self._etf_momentum_func is not None:
                    try:
                        etf_bonus = self._etf_momentum_func(sym, date, self.data_mgr)
                        if etf_bonus > 0:
                            gated_score = min(100, gated_score + etf_bonus)
                    except Exception:
                        pass

                score_distribution.append(gated_score)

                # ── Entry decision ──
                if gated_score >= effective_buy_threshold:
                    current_price = float(sym_df.loc[date, 'Close'])

                    # V7: Defensive mode check
                    if self.config.defensive_mode:
                        invested_value = sum(
                            p.shares * float(symbol_data[p.symbol].loc[date, 'Close'])
                            if p.symbol in symbol_data and date in symbol_data[p.symbol].index
                            else p.shares * p.entry_price
                            for p in positions.values()
                        )
                        invested_pct = invested_value / portfolio_value if portfolio_value > 0 else 0

                        # Defensive limits by regime (thresholds relative to buy_threshold)
                        bt = self.config.buy_threshold
                        if regime == MarketRegime.VOLATILE:
                            max_invested, min_score = 0.30, bt + 15
                        elif regime == MarketRegime.BEAR:
                            max_invested, min_score = 0.50, bt + 10
                        elif regime == MarketRegime.RANGE:
                            max_invested, min_score = 0.80, bt + 3
                        else:
                            max_invested, min_score = 1.0, bt

                        if invested_pct >= max_invested:
                            continue  # Skip - max invested reached
                        if gated_score < min_score:
                            continue  # Skip - score too low for regime

                    # V7: Correlation check (sector limits)
                    if self.config.correlation_checks and positions:
                        sym_sector = fundamentals.get(sym, {}).get('sector', 'Unknown')
                        sector_value = sum(
                            p.shares * float(symbol_data[p.symbol].loc[date, 'Close'])
                            if p.symbol in symbol_data and date in symbol_data[p.symbol].index
                            else p.shares * p.entry_price
                            for p in positions.values() if p.sector == sym_sector
                        )
                        if portfolio_value > 0 and sector_value / portfolio_value > self.config.max_sector_pct:
                            continue  # Skip - sector too concentrated

                    # V7: Dynamic position sizing
                    atr_val = self.tech_scorer.get_atr_at(sym, date_loc)
                    atr_pct = atr_val / current_price if atr_val > 0 and current_price > 0 else 0.02

                    if self.position_sizer is not None:
                        invested_pct_now = 1 - (capital / portfolio_value) if portfolio_value > 0 else 0
                        max_inv = 1.0
                        if self.config.defensive_mode:
                            if regime == MarketRegime.VOLATILE:
                                max_inv = 0.30
                            elif regime == MarketRegime.BEAR:
                                max_inv = 0.50
                            elif regime == MarketRegime.RANGE:
                                max_inv = 0.80

                        sizing = self.position_sizer.calculate_size(
                            capital=capital,
                            price=current_price,
                            atr_pct=atr_pct,
                            confidence_score=gated_score,
                            regime=regime.value,
                            defensive_max_invested=max_inv,
                            current_invested_pct=invested_pct_now
                        )
                        position_value = capital * sizing.adjusted_size_pct
                    else:
                        # V7.1: Leverage multiplies buying power
                        buying_power = portfolio_value * self.config.leverage
                        available = max(0, buying_power - invested_value)
                        position_value = min(
                            available * self.config.position_size_pct,
                            portfolio_value * self.config.position_size_pct * self.config.leverage
                        )

                    # V7.1: Volatility-scaled sizing
                    position_value *= vol_scale

                    # V7.1: Breadth filter - reduce sizing in weak markets
                    if self.config.breadth_filter_enabled and breadth < self.config.breadth_bearish_threshold:
                        position_value *= self.config.breadth_sizing_reduction

                    effective_entry = current_price * self.config.costs.entry_cost_factor()
                    shares = int(position_value / effective_entry)

                    if shares <= 0:
                        continue

                    # ATR-based stop (pre-computed)
                    # V7.1: Use regime-adaptive ATR multiplier
                    if self.config.regime_adaptive_enabled:
                        entry_atr_mult = self.config.regime_atr_multipliers.get(
                            regime.value, self.config.trailing_atr_multiplier
                        )
                    else:
                        entry_atr_mult = self.config.trailing_atr_multiplier

                    if atr_val > 0:
                        trailing_atr = atr_val
                        stop_loss = current_price - (atr_val * entry_atr_mult)
                        # Clamp to regime stop
                        max_stop = current_price * (1 - regime_stop)
                        stop_loss = max(stop_loss, max_stop)
                    else:
                        trailing_atr = 0
                        stop_loss = current_price * (1 - regime_stop)

                    # V7: Tighter stops in defensive mode
                    if self.config.defensive_mode and regime in (MarketRegime.BEAR, MarketRegime.VOLATILE):
                        tight_stop_pct = 0.04 if regime == MarketRegime.VOLATILE else 0.05
                        tight_stop = current_price * (1 - tight_stop_pct)
                        stop_loss = max(stop_loss, tight_stop)

                    # Deduct cost from capital (can go negative with leverage = margin borrowing)
                    total_cost = effective_entry * shares + self.config.costs.commission_per_trade
                    # V7.1: With leverage, max borrowable = (leverage - 1) * portfolio_value
                    max_margin = (self.config.leverage - 1) * portfolio_value
                    if total_cost > capital + max_margin:
                        continue

                    capital -= effective_entry * shares + self.config.costs.commission_per_trade

                    sym_sector = fundamentals.get(sym, {}).get('sector', 'Unknown')
                    positions[sym] = BacktestPosition(
                        symbol=sym,
                        entry_date=date,
                        entry_price=current_price,
                        effective_entry=effective_entry,
                        shares=shares,
                        stop_loss=stop_loss,
                        highest_price=current_price,
                        confidence_score=gated_score,
                        regime_at_entry=regime.value,
                        trailing_atr=trailing_atr,
                        sector=sym_sector
                    )
                    signals_taken += 1

                    # V7: Record trade for Kelly criterion
                    if self.position_sizer is not None and all_trades:
                        last_trade = all_trades[-1]
                        self.position_sizer.add_trade_result(last_trade.profit_loss_pct)

            if progress_callback and day_idx % 50 == 0:
                progress_callback(
                    f"Day {day_idx}/{len(trading_days)} | "
                    f"Trades: {len(all_trades)} | Positions: {len(positions)} | "
                    f"Regime: {regime.value}"
                )

        # Close remaining positions at end
        for sym, pos in positions.items():
            sym_df = symbol_data.get(sym)
            if sym_df is not None and not sym_df.empty:
                last_price = float(sym_df['Close'].iloc[-1])
            else:
                last_price = pos.entry_price

            trade = self._close_position(pos, last_price, trading_days[-1], 'end_of_period')
            all_trades.append(trade)
            effective_exit_price = last_price * self.config.costs.exit_cost_factor()
            capital += effective_exit_price * pos.shares - self.config.costs.commission_per_trade

        # Build equity curve and daily returns
        equity_series = pd.Series(equity_history)
        equity_series = equity_series.sort_index()

        # Fill gaps with forward fill
        try:
            equity_series = equity_series.resample('D').ffill()
        except Exception:
            pass

        daily_returns = equity_series.pct_change().dropna()

        # Compute score distribution stats
        score_stats = {}
        if score_distribution:
            scores_arr = np.array(score_distribution)
            score_stats = {
                'min': float(np.min(scores_arr)),
                'max': float(np.max(scores_arr)),
                'mean': float(np.mean(scores_arr)),
                'median': float(np.median(scores_arr)),
                'p75': float(np.percentile(scores_arr, 75)),
                'p90': float(np.percentile(scores_arr, 90)),
                'p95': float(np.percentile(scores_arr, 95)),
                'above_threshold': int(np.sum(scores_arr >= self.config.buy_threshold)),
                'total_scored': len(scores_arr),
                'pct_above': float(np.sum(scores_arr >= self.config.buy_threshold) / len(scores_arr) * 100),
            }
            logger.info(f"Score stats: mean={score_stats['mean']:.1f}, "
                         f"p75={score_stats['p75']:.1f}, p90={score_stats['p90']:.1f}, "
                         f"above threshold ({self.config.buy_threshold}): "
                         f"{score_stats['above_threshold']}/{score_stats['total_scored']} "
                         f"({score_stats['pct_above']:.1f}%)")

        # Calculate metrics with proper daily returns
        metrics = calculate_metrics(
            trades=all_trades,
            initial_capital=self.config.initial_capital,
            daily_returns=daily_returns
        )

        logger.info(f"V6.2 Backtest complete: {len(all_trades)} trades, "
                     f"Sharpe={metrics.sharpe_ratio:.2f}, "
                     f"Return={metrics.total_return:.1f}%")

        return V6BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=all_trades,
            equity_curve=equity_series,
            daily_returns=daily_returns,
            signals_evaluated=signals_evaluated,
            signals_taken=signals_taken,
            regime_history=regime_counts,
            ml_gate_stats=gate_stats,
            score_stats=score_stats
        )

    def _check_exit(
        self,
        pos: BacktestPosition,
        current_price: float,
        current_low: float,
        date: pd.Timestamp,
        regime: MarketRegime,
        symbol_data: Dict[str, pd.DataFrame] = None
    ) -> Tuple[bool, str]:
        """Check exit conditions for a position."""
        # Stop loss hit
        if current_low <= pos.stop_loss:
            return True, 'stop_loss'

        hold_days = (date - pos.entry_date).days

        # V7.1: Adaptive profit target exit
        if self.config.profit_target_enabled:
            gain_pct = (current_price - pos.entry_price) / pos.entry_price
            if gain_pct >= self.config.profit_target_pct:
                return True, 'profit_target'

        # V7.1: Regime-change trailing stop tightening
        # Use tighter ATR multiplier if regime is BEAR/VOLATILE
        if pos.trailing_atr > 0:
            if self.config.regime_tightening_enabled and regime in (MarketRegime.BEAR, MarketRegime.VOLATILE):
                atr_mult = self.config.regime_tight_atr_multiplier
            elif self.config.regime_adaptive_enabled:
                atr_mult = self.config.regime_atr_multipliers.get(
                    regime.value, self.config.trailing_atr_multiplier
                )
            else:
                atr_mult = self.config.trailing_atr_multiplier
            trailing_stop = pos.highest_price - (pos.trailing_atr * atr_mult)
            if current_low <= trailing_stop:
                return True, 'trailing_stop'

        # V7.1: Score-based exit (technical degradation)
        if (self.config.score_exit_enabled and symbol_data is not None
                and hold_days - pos.last_score_check_day >= self.config.score_exit_check_interval):
            pos.last_score_check_day = hold_days
            sym_df = symbol_data.get(pos.symbol)
            if sym_df is not None and date in sym_df.index:
                date_loc = sym_df.index.get_loc(date)
                tech_score = self.tech_scorer.score_at(pos.symbol, date_loc)
                if tech_score < self.config.score_exit_threshold:
                    return True, 'score_degradation'

        # Max hold period
        if hold_days >= self.config.max_hold_days:
            return True, 'max_hold_period'

        # Regime change: tighten stops in BEAR/VOLATILE
        if regime in (MarketRegime.BEAR, MarketRegime.VOLATILE):
            regime_stop = self.config.regime_stops.get(regime.value, 0.05)
            emergency_stop = pos.entry_price * (1 - regime_stop)
            if current_low <= emergency_stop:
                return True, f'regime_stop_{regime.value}'

        return False, ''

    def _close_position(
        self,
        pos: BacktestPosition,
        exit_price: float,
        exit_date: pd.Timestamp,
        exit_reason: str
    ) -> Trade:
        """Close a position and create a Trade record with costs."""
        effective_exit = exit_price * self.config.costs.exit_cost_factor()
        commission = self.config.costs.commission_per_trade

        gross_pnl = (effective_exit - pos.effective_entry) * pos.shares
        net_pnl = gross_pnl - commission  # Exit commission

        pnl_pct = (effective_exit - pos.effective_entry) / pos.effective_entry * 100

        return Trade(
            symbol=pos.symbol,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            shares=pos.shares,
            stop_loss=pos.stop_loss,
            profit_loss=net_pnl,
            profit_loss_pct=pnl_pct,
            exit_reason=exit_reason,
            signal_strength='V6_SCORE',
            confidence_score=pos.confidence_score,
            volume_ratio=0.0
        )

    def _empty_result(self) -> V6BacktestResult:
        """Return empty result on error."""
        from .metrics import _empty_metrics
        return V6BacktestResult(
            config=self.config,
            metrics=_empty_metrics(self.config.initial_capital),
            trades=[],
            equity_curve=pd.Series(dtype=float),
            daily_returns=pd.Series(dtype=float),
            signals_evaluated=0,
            signals_taken=0,
            regime_history={},
            ml_gate_stats={}
        )


# ── Convenience Functions ────────────────────────────────────────────────

def run_v6_backtest(
    symbols: List[str],
    start_date: str = '2015-01-01',
    end_date: str = '2025-01-01',
    initial_capital: float = 100_000,
    include_costs: bool = True,
    scoring_mode: ScoringMode = ScoringMode.THREE_PILLARS_VIX
) -> V6BacktestResult:
    """Quick V6.2 backtest helper."""
    costs = TransactionCosts() if include_costs else TransactionCosts(
        commission_per_trade=0, spread_pct=0, slippage_pct=0
    )

    config = V6BacktestConfig(
        initial_capital=initial_capital,
        costs=costs,
        scoring_mode=scoring_mode
    )

    bt = V6Backtester(config)
    return bt.run(symbols, start_date, end_date)


def run_oos_backtest(
    symbols: List[str],
    train_start: str = '2015-01-01',
    train_end: str = '2021-12-31',
    validation_end: str = '2023-12-31',
    test_end: str = '2025-12-31',
    initial_capital: float = 100_000
) -> Dict[str, V6BacktestResult]:
    """Run out-of-sample backtest with train/validation/test splits.

    Returns dict with keys 'train', 'validation', 'test' containing results
    for each period. Compare CAGR across periods to detect overfitting.
    """
    config = V6BacktestConfig(
        initial_capital=initial_capital,
        costs=TransactionCosts()
    )

    results = {}
    bt = V6Backtester(config)

    # Train period
    logger.info("=== TRAIN PERIOD ===")
    results['train'] = bt.run(symbols, train_start, train_end)

    # Validation period
    logger.info("=== VALIDATION PERIOD ===")
    results['validation'] = bt.run(symbols, train_end, validation_end)

    # Test period (DO NOT OPTIMIZE ON THIS)
    logger.info("=== TEST PERIOD ===")
    results['test'] = bt.run(symbols, validation_end, test_end)

    # Overfitting check
    train_cagr = results['train'].metrics.total_return
    test_cagr = results['test'].metrics.total_return

    train_years = max(1, (pd.Timestamp(train_end) - pd.Timestamp(train_start)).days / 365)
    test_years = max(1, (pd.Timestamp(test_end) - pd.Timestamp(validation_end)).days / 365)

    train_ann = (1 + train_cagr / 100) ** (1 / train_years) - 1
    test_ann = (1 + test_cagr / 100) ** (1 / test_years) - 1

    if train_ann > 0:
        retention = test_ann / train_ann
        logger.info(f"OOS Retention: {retention:.1%} (test CAGR / train CAGR)")
        if retention < 0.6:
            logger.warning("OVERFITTING DETECTED: Test performance < 60% of train")
        elif retention < 0.8:
            logger.warning("Moderate degradation: Consider simplifying the strategy")
        else:
            logger.info("Good out-of-sample retention")
    else:
        logger.warning("Train CAGR is negative - strategy may not be viable")

    return results


def run_abc_comparison(
    symbols: List[str],
    start_date: str = '2015-01-01',
    end_date: str = '2025-01-01',
    initial_capital: float = 100_000,
    buy_threshold: float = 65.0
) -> Dict[str, V6BacktestResult]:
    """Run A vs B vs C comparison on the same symbols and period.

    Mode A: Technical pillar only (baseline)
    Mode B: Tech + Fundamental + ML (redistributed weights, no sentiment/news)
    Mode C: Tech + Fund + ML + VIX sentiment proxy (contrarian)

    All modes use the same transaction costs, regime detection, and exit logic.
    The data is fetched once and shared across all 3 runs.

    Returns dict with keys 'A', 'B', 'C'.
    """
    modes = {
        'A': ScoringMode.TECH_ONLY,
        'B': ScoringMode.THREE_PILLARS,
        'C': ScoringMode.THREE_PILLARS_VIX,
    }

    # Pre-fetch all data ONCE and share across modes
    print(f"\n  Pre-fetching data for {len(symbols)} symbols...")
    shared_data_mgr = BacktestDataManager()
    buffer_start = (pd.Timestamp(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')

    # Fetch SPY + VIX
    print(f"  Fetching SPY & VIX...")
    shared_data_mgr.get_spy_data(buffer_start, end_date)
    shared_data_mgr.get_vix_data(buffer_start, end_date)

    # Fetch all symbols OHLCV + fundamentals
    loaded = 0
    failed = 0
    for i, sym in enumerate(symbols):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Fetching: {i+1}/{len(symbols)} symbols ({loaded} loaded, {failed} failed)...")
        df = shared_data_mgr.get_ohlcv(sym, buffer_start, end_date)
        if df is not None and len(df) >= 100:
            loaded += 1
        else:
            failed += 1
        shared_data_mgr.get_fundamentals(sym)

    print(f"  Data ready: {loaded}/{len(symbols)} symbols loaded ({failed} failed)")

    # Pre-compute technical indicators ONCE (shared across all 3 modes)
    print(f"  Pre-computing indicators for {loaded} symbols...")
    shared_scorer = BacktestTechnicalScorer()
    buffer_start = (pd.Timestamp(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
    precomputed = 0
    for sym in symbols:
        cache_key = f"{sym}_{buffer_start}_{end_date}"
        if cache_key in shared_data_mgr._cache:
            df = shared_data_mgr._cache[cache_key]
            if len(df) >= 100:
                shared_scorer.precompute(sym, df)
                precomputed += 1
    print(f"  Indicators ready: {precomputed} symbols pre-computed")
    print(f"  Running 3 modes with shared data + indicators (no re-download)...\n")

    results = {}

    for label, mode in modes.items():
        print(f"\n{'='*60}")
        print(f"  MODE {label}: {mode.name}")
        print(f"{'='*60}")

        config = V6BacktestConfig(
            initial_capital=initial_capital,
            scoring_mode=mode,
            buy_threshold=buy_threshold,
            costs=TransactionCosts()
        )

        bt = V6Backtester(config, data_mgr=shared_data_mgr)
        bt.tech_scorer = shared_scorer  # Share pre-computed indicators
        result = bt.run(symbols, start_date, end_date)
        results[label] = result

        m = result.metrics
        years = max(1, (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25)
        cagr = ((1 + m.total_return / 100) ** (1 / years) - 1) * 100

        print(f"\n  Results Mode {label}:")
        print(f"    CAGR:         {cagr:.1f}%")
        print(f"    Sharpe:       {m.sharpe_ratio:.2f}")
        print(f"    Sortino:      {m.sortino_ratio:.2f}")
        print(f"    Max DD:       -{m.max_drawdown:.1f}%")
        print(f"    Win Rate:     {m.win_rate:.1f}%")
        print(f"    Trades:       {m.total_trades}")
        print(f"    Final Cap:    ${m.final_capital:,.0f}")
        print(f"    Profit Factor:{m.profit_factor:.2f}")
        ss = result.score_stats
        if ss:
            print(f"    Score Distribution:")
            print(f"      Mean: {ss['mean']:.1f} | Median: {ss['median']:.1f} | "
                  f"P75: {ss['p75']:.1f} | P90: {ss['p90']:.1f} | P95: {ss['p95']:.1f}")
            print(f"      Min: {ss['min']:.1f} | Max: {ss['max']:.1f}")
            print(f"      Above threshold ({config.buy_threshold}): "
                  f"{ss['above_threshold']}/{ss['total_scored']} ({ss['pct_above']:.1f}%)")
        print(f"    Evaluated: {result.signals_evaluated} | Taken: {result.signals_taken}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"  COMPARISON TABLE: A vs B vs C")
    print(f"{'='*80}")
    print(f"{'Metric':<20} {'A (Tech Only)':<18} {'B (3 Pillars)':<18} {'C (3P + VIX)':<18}")
    print(f"{'-'*74}")

    for label in ['A', 'B', 'C']:
        m = results[label].metrics
        years = max(1, (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25)
        cagr = ((1 + m.total_return / 100) ** (1 / years) - 1) * 100
        results[label]._cagr = cagr  # Store for later access

    metrics_rows = [
        ('CAGR', lambda m, y: f"{((1+m.total_return/100)**(1/y)-1)*100:.1f}%"),
        ('Total Return', lambda m, y: f"{m.total_return:.1f}%"),
        ('Sharpe Ratio', lambda m, y: f"{m.sharpe_ratio:.2f}"),
        ('Sortino Ratio', lambda m, y: f"{m.sortino_ratio:.2f}"),
        ('Max Drawdown', lambda m, y: f"-{m.max_drawdown:.1f}%"),
        ('Win Rate', lambda m, y: f"{m.win_rate:.1f}%"),
        ('Profit Factor', lambda m, y: f"{m.profit_factor:.2f}"),
        ('Total Trades', lambda m, y: f"{m.total_trades}"),
        ('Avg Win', lambda m, y: f"{m.avg_win:.1f}%"),
        ('Avg Loss', lambda m, y: f"{m.avg_loss:.1f}%"),
        ('Final Capital', lambda m, y: f"${m.final_capital:,.0f}"),
    ]

    years = max(1, (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25)
    for name, fmt_fn in metrics_rows:
        vals = []
        for label in ['A', 'B', 'C']:
            vals.append(fmt_fn(results[label].metrics, years))
        print(f"{name:<20} {vals[0]:<18} {vals[1]:<18} {vals[2]:<18}")

    # Score distribution
    print(f"\n{'Score Stats':<20} {'A (Tech Only)':<18} {'B (3 Pillars)':<18} {'C (3P + VIX)':<18}")
    print(f"{'-'*74}")
    for stat_name in ['mean', 'median', 'p75', 'p90', 'max', 'pct_above']:
        vals = []
        for label in ['A', 'B', 'C']:
            ss = results[label].score_stats
            if ss:
                v = ss.get(stat_name, 0)
                if stat_name == 'pct_above':
                    vals.append(f"{v:.1f}%")
                else:
                    vals.append(f"{v:.1f}")
            else:
                vals.append("N/A")
        display_name = stat_name if stat_name != 'pct_above' else f'% >= threshold'
        print(f"{display_name:<20} {vals[0]:<18} {vals[1]:<18} {vals[2]:<18}")

    # Regime breakdown
    print(f"\n{'Regime':<20} {'A signals':<18} {'B signals':<18} {'C signals':<18}")
    print(f"{'-'*74}")
    for regime in ['BULL', 'BEAR', 'RANGE', 'VOLATILE']:
        vals = []
        for label in ['A', 'B', 'C']:
            rh = results[label].regime_history
            vals.append(f"{rh.get(regime, 0)} days")
        print(f"{regime:<20} {vals[0]:<18} {vals[1]:<18} {vals[2]:<18}")

    # ML Gate stats (B and C only)
    print(f"\n{'ML Gate':<20} {'A (disabled)':<18} {'B':<18} {'C':<18}")
    print(f"{'-'*74}")
    for gate in ['ML_BOOST', 'ML_BLOCK', 'ML_NEUTRAL', '5P_ONLY']:
        vals = []
        for label in ['A', 'B', 'C']:
            gs = results[label].ml_gate_stats
            vals.append(f"{gs.get(gate, 0)}")
        print(f"{gate:<20} {vals[0]:<18} {vals[1]:<18} {vals[2]:<18}")

    # Winner announcement
    best = max(results.items(), key=lambda x: x[1].metrics.sharpe_ratio)
    print(f"\n{'='*80}")
    print(f"  BEST RISK-ADJUSTED: Mode {best[0]} (Sharpe {best[1].metrics.sharpe_ratio:.2f})")
    best_return = max(results.items(), key=lambda x: x[1].metrics.total_return)
    print(f"  BEST RETURN:        Mode {best_return[0]} ({best_return[1].metrics.total_return:.1f}%)")
    lowest_dd = min(results.items(), key=lambda x: x[1].metrics.max_drawdown)
    print(f"  LOWEST DRAWDOWN:    Mode {lowest_dd[0]} (-{lowest_dd[1].metrics.max_drawdown:.1f}%)")
    print(f"{'='*80}")

    return results
