"""
Parameter Optimizer - Auto-Learning Engine for TradingBot

This module implements a comprehensive learning system that:
1. Analyzes all completed trades with full indicator snapshots
2. Identifies winning/losing patterns and combinations
3. Proposes parameter adjustments based on statistical analysis
4. Integrates with NightlyAuditor for daily optimization

Usage:
    from src.agents.parameter_optimizer import ParameterOptimizer, get_parameter_optimizer

    optimizer = get_parameter_optimizer()
    report = await optimizer.optimize_from_trades(trades, pillar_configs)

    # Apply optimizations
    optimizer.apply_optimizations(report.recommended_changes)
"""

import os
import json
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from enum import Enum
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class TradeResult(str, Enum):
    """Trade outcome classification"""
    BIG_WIN = "big_win"          # > +10%
    WIN = "win"                   # +2% to +10%
    SMALL_WIN = "small_win"       # 0% to +2%
    BREAKEVEN = "breakeven"       # -1% to 0%
    SMALL_LOSS = "small_loss"     # -5% to -1%
    LOSS = "loss"                 # -10% to -5%
    BIG_LOSS = "big_loss"         # < -10%


class ParameterType(str, Enum):
    """Types of parameters that can be optimized"""
    THRESHOLD = "threshold"       # RSI levels, score thresholds
    PERIOD = "period"            # Lookback periods
    WEIGHT = "weight"            # Pillar weights
    MULTIPLIER = "multiplier"    # ATR multiplier, volume ratio


class OptimizationConfidence(str, Enum):
    """Confidence level of optimization suggestion"""
    HIGH = "high"                # > 20 samples, statistically significant
    MEDIUM = "medium"            # 10-20 samples, likely significant
    LOW = "low"                  # < 10 samples, indicative only


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class IndicatorSnapshot:
    """Complete snapshot of all indicators at trade entry"""
    # Technical - Trend
    rsi_value: float = 50.0
    rsi_period: int = 14
    macd_value: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    ema_20: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    ema_alignment: str = "neutral"  # bullish, bearish, neutral
    adx_value: float = 25.0
    
    # Technical - Momentum
    stochastic_k: float = 50.0
    stochastic_d: float = 50.0
    williams_r: float = -50.0
    cci_value: float = 0.0
    roc_value: float = 0.0
    
    # Technical - Volume
    volume_ratio: float = 1.0
    obv_trend: str = "flat"
    vwap_position: float = 0.0      # % above/below VWAP
    cmf_value: float = 0.0          # Chaikin Money Flow
    mfi_value: float = 50.0         # Money Flow Index
    
    # Technical - Volatility
    atr_value: float = 0.0
    atr_percent: float = 2.0        # ATR as % of price
    bb_position: float = 0.5        # 0=lower band, 0.5=middle, 1=upper
    bb_width: float = 0.0
    
    # Technical - Support/Resistance
    distance_to_support: float = 0.0
    distance_to_resistance: float = 0.0
    
    # Fundamental
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    current_ratio: Optional[float] = None
    
    # Sentiment
    sentiment_score: float = 0.0     # -100 to +100
    grok_score: float = 0.0          # Grok X/Twitter sentiment
    social_volume: float = 0.0       # Relative social activity
    sentiment_sources: List[str] = field(default_factory=list)
    
    # News
    news_score: float = 0.0          # -100 to +100
    news_count: int = 0
    has_earnings_soon: bool = False
    has_analyst_update: bool = False
    
    # Market Context
    market_regime: str = "neutral"
    spy_trend: str = "flat"
    vix_level: float = 20.0
    sector_momentum: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'IndicatorSnapshot':
        """Create from dictionary, ignoring unknown keys"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class TradeAnalysisRecord:
    """Complete record of a trade for learning"""
    # Trade identity
    trade_id: str
    symbol: str
    sector: str
    
    # Timing
    entry_date: str
    exit_date: Optional[str]
    hold_days: int
    
    # Prices
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit: Optional[float]
    
    # Results
    pnl_amount: float
    pnl_percent: float
    result: TradeResult
    exit_type: str  # stop_loss, take_profit, trailing, manual
    
    # Scores at entry
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    news_score: float
    combined_score: float
    confidence: float
    
    # Full indicator snapshot
    indicators: IndicatorSnapshot
    
    # Pillar weights used
    weights_used: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['indicators'] = self.indicators.to_dict()
        d['result'] = self.result.value
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeAnalysisRecord':
        indicators = IndicatorSnapshot.from_dict(data.get('indicators', {}))
        result = TradeResult(data.get('result', 'breakeven'))
        
        return cls(
            trade_id=data.get('trade_id', ''),
            symbol=data.get('symbol', ''),
            sector=data.get('sector', 'Unknown'),
            entry_date=data.get('entry_date', ''),
            exit_date=data.get('exit_date'),
            hold_days=data.get('hold_days', 0),
            entry_price=data.get('entry_price', 0),
            exit_price=data.get('exit_price'),
            stop_loss=data.get('stop_loss', 0),
            take_profit=data.get('take_profit'),
            pnl_amount=data.get('pnl_amount', 0),
            pnl_percent=data.get('pnl_percent', 0),
            result=result,
            exit_type=data.get('exit_type', 'unknown'),
            technical_score=data.get('technical_score', 0),
            fundamental_score=data.get('fundamental_score', 0),
            sentiment_score=data.get('sentiment_score', 0),
            news_score=data.get('news_score', 0),
            combined_score=data.get('combined_score', 0),
            confidence=data.get('confidence', 0),
            indicators=indicators,
            weights_used=data.get('weights_used', {})
        )


@dataclass
class PatternStatistics:
    """Statistics for an identified pattern"""
    pattern_id: str
    description: str
    
    # Sample info
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Performance
    avg_return: float
    median_return: float
    std_return: float
    best_return: float
    worst_return: float
    profit_factor: float  # Total wins / Total losses
    
    # Conditions
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Confidence
    confidence: OptimizationConfidence = OptimizationConfidence.LOW
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['confidence'] = self.confidence.value
        return d


@dataclass
class ParameterChange:
    """A proposed parameter change"""
    parameter_name: str
    parameter_type: ParameterType
    pillar: str  # technical, fundamental, sentiment, news, combined
    
    current_value: Any
    proposed_value: Any
    
    reason: str
    expected_improvement: float  # Estimated % improvement in win rate
    
    # Supporting data
    sample_size: int
    win_rate_current: float
    win_rate_proposed: float
    confidence: OptimizationConfidence
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['parameter_type'] = self.parameter_type.value
        d['confidence'] = self.confidence.value
        return d


@dataclass
class OptimizationReport:
    """Daily optimization report"""
    date: str
    trades_analyzed: int
    
    # Performance summary
    total_pnl: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    
    # Patterns identified
    winning_patterns: List[PatternStatistics]
    losing_patterns: List[PatternStatistics]
    
    # Recommended changes
    recommended_changes: List[ParameterChange]
    
    # Pillar analysis
    pillar_performance: Dict[str, Dict[str, float]]  # pillar -> {win_rate, avg_contribution, etc}
    
    # Weight recommendations
    recommended_weights: Dict[str, float]
    
    # Warnings
    warnings: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'date': self.date,
            'trades_analyzed': self.trades_analyzed,
            'total_pnl': self.total_pnl,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'winning_patterns': [p.to_dict() for p in self.winning_patterns],
            'losing_patterns': [p.to_dict() for p in self.losing_patterns],
            'recommended_changes': [c.to_dict() for c in self.recommended_changes],
            'pillar_performance': self.pillar_performance,
            'recommended_weights': self.recommended_weights,
            'warnings': self.warnings
        }


# =============================================================================
# PARAMETER OPTIMIZER
# =============================================================================

class ParameterOptimizer:
    """
    Comprehensive parameter optimization engine.
    
    Analyzes completed trades to:
    1. Identify winning/losing patterns
    2. Suggest parameter adjustments
    3. Optimize pillar weights
    4. Track optimization history
    """
    
    # Default parameter bounds (safety limits)
    PARAMETER_BOUNDS = {
        # RSI
        'rsi_period': {'min': 7, 'max': 28, 'default': 14, 'step': 1},
        'rsi_oversold': {'min': 20, 'max': 40, 'default': 30, 'step': 5},
        'rsi_overbought': {'min': 60, 'max': 80, 'default': 70, 'step': 5},
        
        # MACD
        'macd_fast': {'min': 8, 'max': 16, 'default': 12, 'step': 1},
        'macd_slow': {'min': 20, 'max': 30, 'default': 26, 'step': 1},
        'macd_signal': {'min': 6, 'max': 12, 'default': 9, 'step': 1},
        
        # EMA
        'ema_short': {'min': 10, 'max': 30, 'default': 20, 'step': 5},
        'ema_medium': {'min': 40, 'max': 60, 'default': 50, 'step': 5},
        'ema_long': {'min': 150, 'max': 250, 'default': 200, 'step': 10},
        
        # Volume
        'volume_ratio_min': {'min': 1.0, 'max': 3.0, 'default': 1.5, 'step': 0.25},
        
        # ADX
        'adx_threshold': {'min': 15, 'max': 35, 'default': 25, 'step': 5},
        
        # ATR
        'atr_multiplier': {'min': 1.0, 'max': 3.0, 'default': 2.0, 'step': 0.25},
        
        # Score thresholds
        'min_confidence': {'min': 40, 'max': 80, 'default': 55, 'step': 5},
        'min_combined_score': {'min': 45, 'max': 75, 'default': 55, 'step': 5},
        
        # Pillar weights (must sum to 1.0)
        'weight_technical': {'min': 0.15, 'max': 0.40, 'default': 0.25, 'step': 0.05},
        'weight_fundamental': {'min': 0.10, 'max': 0.35, 'default': 0.25, 'step': 0.05},
        'weight_sentiment': {'min': 0.10, 'max': 0.35, 'default': 0.25, 'step': 0.05},
        'weight_news': {'min': 0.05, 'max': 0.25, 'default': 0.25, 'step': 0.05},
    }
    
    # Minimum samples for statistical significance
    MIN_SAMPLES_HIGH = 20
    MIN_SAMPLES_MEDIUM = 10
    MIN_SAMPLES_LOW = 5
    
    def __init__(self, data_dir: str = "data/auditor"):
        """
        Initialize the optimizer.
        
        Args:
            data_dir: Directory for storing optimization data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # History file
        self.history_file = self.data_dir / "parameter_history.json"
        self.patterns_file = self.data_dir / "learned_patterns.json"
        self.trades_db_file = self.data_dir / "trade_analysis_db.json"
        
        # Load history
        self._history: List[Dict] = []
        self._patterns: Dict[str, PatternStatistics] = {}
        self._trade_records: List[TradeAnalysisRecord] = []
        
        self._load_history()
        self._load_patterns()
        self._load_trade_records()
        
        logger.info(f"ParameterOptimizer initialized with {len(self._trade_records)} trade records")
    
    def _load_history(self):
        """Load parameter change history"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self._history = json.load(f).get('changes', [])
                logger.info(f"Loaded {len(self._history)} parameter changes from history")
            except Exception as e:
                logger.error(f"Error loading history: {e}")
    
    def _save_history(self):
        """Save parameter change history"""
        try:
            data = {
                'updated_at': datetime.now().isoformat(),
                'total_changes': len(self._history),
                'changes': self._history
            }
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def _load_patterns(self):
        """Load learned patterns"""
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'r') as f:
                    data = json.load(f)
                    for p in data.get('patterns', []):
                        pattern = PatternStatistics(**{
                            **p,
                            'confidence': OptimizationConfidence(p.get('confidence', 'low'))
                        })
                        self._patterns[pattern.pattern_id] = pattern
                logger.info(f"Loaded {len(self._patterns)} patterns")
            except Exception as e:
                logger.error(f"Error loading patterns: {e}")
    
    def _save_patterns(self):
        """Save learned patterns"""
        try:
            data = {
                'updated_at': datetime.now().isoformat(),
                'total_patterns': len(self._patterns),
                'patterns': [p.to_dict() for p in self._patterns.values()]
            }
            with open(self.patterns_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    def _load_trade_records(self):
        """Load trade analysis records"""
        if self.trades_db_file.exists():
            try:
                with open(self.trades_db_file, 'r') as f:
                    data = json.load(f)
                    for t in data.get('trades', []):
                        record = TradeAnalysisRecord.from_dict(t)
                        self._trade_records.append(record)
            except Exception as e:
                logger.error(f"Error loading trade records: {e}")
    
    def _save_trade_records(self):
        """Save trade analysis records"""
        try:
            data = {
                'updated_at': datetime.now().isoformat(),
                'total_trades': len(self._trade_records),
                'trades': [t.to_dict() for t in self._trade_records]
            }
            with open(self.trades_db_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trade records: {e}")
    
    # -------------------------------------------------------------------------
    # TRADE RECORDING
    # -------------------------------------------------------------------------
    
    def record_trade(
        self,
        trade_data: Dict,
        pillar_scores: Dict[str, float],
        indicators: Dict[str, Any],
        weights: Dict[str, float]
    ) -> TradeAnalysisRecord:
        """
        Record a completed trade for analysis.
        
        Args:
            trade_data: Basic trade info (symbol, prices, dates, pnl)
            pillar_scores: Scores from each pillar at entry
            indicators: All indicator values at entry
            weights: Pillar weights used
            
        Returns:
            TradeAnalysisRecord
        """
        # Classify result
        pnl_pct = trade_data.get('pnl_percent', 0)
        if pnl_pct > 10:
            result = TradeResult.BIG_WIN
        elif pnl_pct > 2:
            result = TradeResult.WIN
        elif pnl_pct > 0:
            result = TradeResult.SMALL_WIN
        elif pnl_pct > -1:
            result = TradeResult.BREAKEVEN
        elif pnl_pct > -5:
            result = TradeResult.SMALL_LOSS
        elif pnl_pct > -10:
            result = TradeResult.LOSS
        else:
            result = TradeResult.BIG_LOSS
        
        # Create indicator snapshot
        snapshot = IndicatorSnapshot(
            rsi_value=indicators.get('rsi', 50),
            rsi_period=indicators.get('rsi_period', 14),
            macd_value=indicators.get('macd', 0),
            macd_signal=indicators.get('macd_signal', 0),
            macd_histogram=indicators.get('macd_histogram', 0),
            ema_20=indicators.get('ema_20', 0),
            ema_50=indicators.get('ema_50', 0),
            ema_200=indicators.get('ema_200', 0),
            ema_alignment=indicators.get('ema_alignment', 'neutral'),
            adx_value=indicators.get('adx', 25),
            stochastic_k=indicators.get('stoch_k', 50),
            stochastic_d=indicators.get('stoch_d', 50),
            williams_r=indicators.get('williams_r', -50),
            cci_value=indicators.get('cci', 0),
            roc_value=indicators.get('roc', 0),
            volume_ratio=indicators.get('volume_ratio', 1),
            obv_trend=indicators.get('obv_trend', 'flat'),
            vwap_position=indicators.get('vwap_position', 0),
            cmf_value=indicators.get('cmf', 0),
            mfi_value=indicators.get('mfi', 50),
            atr_value=indicators.get('atr', 0),
            atr_percent=indicators.get('atr_percent', 2),
            bb_position=indicators.get('bb_position', 0.5),
            bb_width=indicators.get('bb_width', 0),
            distance_to_support=indicators.get('distance_to_support', 0),
            distance_to_resistance=indicators.get('distance_to_resistance', 0),
            pe_ratio=indicators.get('pe_ratio'),
            peg_ratio=indicators.get('peg_ratio'),
            revenue_growth=indicators.get('revenue_growth'),
            profit_margin=indicators.get('profit_margin'),
            debt_to_equity=indicators.get('debt_to_equity'),
            roe=indicators.get('roe'),
            current_ratio=indicators.get('current_ratio'),
            sentiment_score=indicators.get('sentiment_score', 0),
            grok_score=indicators.get('grok_score', 0),
            social_volume=indicators.get('social_volume', 0),
            sentiment_sources=indicators.get('sentiment_sources', []),
            news_score=indicators.get('news_score', 0),
            news_count=indicators.get('news_count', 0),
            has_earnings_soon=indicators.get('has_earnings_soon', False),
            has_analyst_update=indicators.get('has_analyst_update', False),
            market_regime=indicators.get('market_regime', 'neutral'),
            spy_trend=indicators.get('spy_trend', 'flat'),
            vix_level=indicators.get('vix_level', 20),
            sector_momentum=indicators.get('sector_momentum', 0)
        )
        
        # Calculate hold days
        entry_date = trade_data.get('entry_date', '')
        exit_date = trade_data.get('exit_date')
        hold_days = 0
        if entry_date and exit_date:
            try:
                entry = datetime.fromisoformat(entry_date.replace('Z', '+00:00'))
                exit = datetime.fromisoformat(exit_date.replace('Z', '+00:00'))
                hold_days = (exit - entry).days
            except:
                pass
        
        # Create record
        record = TradeAnalysisRecord(
            trade_id=trade_data.get('trade_id', f"{trade_data.get('symbol', 'UNK')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"),
            symbol=trade_data.get('symbol', ''),
            sector=trade_data.get('sector', 'Unknown'),
            entry_date=entry_date,
            exit_date=exit_date,
            hold_days=hold_days,
            entry_price=trade_data.get('entry_price', 0),
            exit_price=trade_data.get('exit_price'),
            stop_loss=trade_data.get('stop_loss', 0),
            take_profit=trade_data.get('take_profit'),
            pnl_amount=trade_data.get('pnl_amount', 0),
            pnl_percent=pnl_pct,
            result=result,
            exit_type=trade_data.get('exit_type', 'unknown'),
            technical_score=pillar_scores.get('technical', 0),
            fundamental_score=pillar_scores.get('fundamental', 0),
            sentiment_score=pillar_scores.get('sentiment', 0),
            news_score=pillar_scores.get('news', 0),
            combined_score=pillar_scores.get('combined', 0),
            confidence=pillar_scores.get('confidence', 0),
            indicators=snapshot,
            weights_used=weights
        )
        
        self._trade_records.append(record)
        self._save_trade_records()
        
        logger.info(f"Recorded trade: {record.symbol} - {result.value} ({pnl_pct:+.1f}%)")
        return record
    
    # -------------------------------------------------------------------------
    # PATTERN ANALYSIS
    # -------------------------------------------------------------------------
    
    async def identify_patterns(
        self,
        trades: Optional[List[TradeAnalysisRecord]] = None,
        min_samples: int = 5
    ) -> Tuple[List[PatternStatistics], List[PatternStatistics]]:
        """
        Identify winning and losing patterns from trades.
        
        Returns:
            (winning_patterns, losing_patterns)
        """
        trades = trades or self._trade_records
        
        if len(trades) < min_samples:
            logger.warning(f"Not enough trades for pattern analysis ({len(trades)} < {min_samples})")
            return [], []
        
        winning_patterns = []
        losing_patterns = []
        
        # Separate wins and losses
        wins = [t for t in trades if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
        losses = [t for t in trades if t.result in [TradeResult.BIG_LOSS, TradeResult.LOSS, TradeResult.SMALL_LOSS]]
        
        # Analyze RSI patterns
        rsi_patterns = self._analyze_indicator_ranges(
            trades, 'rsi_value',
            lambda t: t.indicators.rsi_value,
            buckets=[(0, 30, "oversold"), (30, 50, "neutral_low"), (50, 70, "neutral_high"), (70, 100, "overbought")]
        )
        winning_patterns.extend([p for p in rsi_patterns if p.win_rate > 0.55])
        losing_patterns.extend([p for p in rsi_patterns if p.win_rate < 0.40])
        
        # Analyze Volume patterns
        volume_patterns = self._analyze_indicator_ranges(
            trades, 'volume_ratio',
            lambda t: t.indicators.volume_ratio,
            buckets=[(0, 0.7, "low"), (0.7, 1.3, "normal"), (1.3, 2.0, "elevated"), (2.0, 100, "spike")]
        )
        winning_patterns.extend([p for p in volume_patterns if p.win_rate > 0.55])
        losing_patterns.extend([p for p in volume_patterns if p.win_rate < 0.40])
        
        # Analyze ADX patterns (trend strength)
        adx_patterns = self._analyze_indicator_ranges(
            trades, 'adx_value',
            lambda t: t.indicators.adx_value,
            buckets=[(0, 20, "weak_trend"), (20, 40, "moderate_trend"), (40, 60, "strong_trend"), (60, 100, "extreme_trend")]
        )
        winning_patterns.extend([p for p in adx_patterns if p.win_rate > 0.55])
        losing_patterns.extend([p for p in adx_patterns if p.win_rate < 0.40])
        
        # Analyze Bollinger position
        bb_patterns = self._analyze_indicator_ranges(
            trades, 'bb_position',
            lambda t: t.indicators.bb_position,
            buckets=[(0, 0.2, "near_lower"), (0.2, 0.4, "lower_middle"), (0.4, 0.6, "middle"), (0.6, 0.8, "upper_middle"), (0.8, 1.0, "near_upper")]
        )
        winning_patterns.extend([p for p in bb_patterns if p.win_rate > 0.55])
        losing_patterns.extend([p for p in bb_patterns if p.win_rate < 0.40])
        
        # Analyze EMA alignment
        ema_patterns = self._analyze_categorical(
            trades, 'ema_alignment',
            lambda t: t.indicators.ema_alignment
        )
        winning_patterns.extend([p for p in ema_patterns if p.win_rate > 0.55])
        losing_patterns.extend([p for p in ema_patterns if p.win_rate < 0.40])
        
        # Analyze Market Regime
        regime_patterns = self._analyze_categorical(
            trades, 'market_regime',
            lambda t: t.indicators.market_regime
        )
        winning_patterns.extend([p for p in regime_patterns if p.win_rate > 0.55])
        losing_patterns.extend([p for p in regime_patterns if p.win_rate < 0.40])
        
        # Analyze VIX levels
        vix_patterns = self._analyze_indicator_ranges(
            trades, 'vix_level',
            lambda t: t.indicators.vix_level,
            buckets=[(0, 15, "low_fear"), (15, 20, "normal"), (20, 30, "elevated"), (30, 100, "fear")]
        )
        winning_patterns.extend([p for p in vix_patterns if p.win_rate > 0.55])
        losing_patterns.extend([p for p in vix_patterns if p.win_rate < 0.40])
        
        # Analyze Combined Score ranges
        score_patterns = self._analyze_indicator_ranges(
            trades, 'combined_score',
            lambda t: t.combined_score,
            buckets=[(0, 55, "low_conviction"), (55, 65, "moderate"), (65, 75, "high_conviction"), (75, 100, "very_high")]
        )
        winning_patterns.extend([p for p in score_patterns if p.win_rate > 0.55])
        losing_patterns.extend([p for p in score_patterns if p.win_rate < 0.40])
        
        # Analyze combinations
        combo_patterns = self._analyze_combinations(trades)
        winning_patterns.extend([p for p in combo_patterns if p.win_rate > 0.60])
        losing_patterns.extend([p for p in combo_patterns if p.win_rate < 0.35])
        
        # Store patterns
        for p in winning_patterns + losing_patterns:
            self._patterns[p.pattern_id] = p
        self._save_patterns()
        
        logger.info(f"Identified {len(winning_patterns)} winning and {len(losing_patterns)} losing patterns")
        return winning_patterns, losing_patterns
    
    def _analyze_indicator_ranges(
        self,
        trades: List[TradeAnalysisRecord],
        indicator_name: str,
        value_fn,
        buckets: List[Tuple[float, float, str]]
    ) -> List[PatternStatistics]:
        """Analyze patterns based on indicator value ranges"""
        patterns = []
        
        for low, high, bucket_name in buckets:
            bucket_trades = [t for t in trades if low <= value_fn(t) < high]
            
            if len(bucket_trades) < self.MIN_SAMPLES_LOW:
                continue
            
            wins = [t for t in bucket_trades if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
            losses = [t for t in bucket_trades if t.result in [TradeResult.BIG_LOSS, TradeResult.LOSS, TradeResult.SMALL_LOSS]]
            
            returns = [t.pnl_percent for t in bucket_trades]
            win_rate = len(wins) / len(bucket_trades) if bucket_trades else 0
            
            total_wins = sum(t.pnl_percent for t in wins) if wins else 0
            total_losses = abs(sum(t.pnl_percent for t in losses)) if losses else 0.01
            
            # Determine confidence
            if len(bucket_trades) >= self.MIN_SAMPLES_HIGH:
                confidence = OptimizationConfidence.HIGH
            elif len(bucket_trades) >= self.MIN_SAMPLES_MEDIUM:
                confidence = OptimizationConfidence.MEDIUM
            else:
                confidence = OptimizationConfidence.LOW
            
            pattern = PatternStatistics(
                pattern_id=f"{indicator_name}_{bucket_name}",
                description=f"{indicator_name} in range [{low:.1f}, {high:.1f}] ({bucket_name})",
                total_trades=len(bucket_trades),
                winning_trades=len(wins),
                losing_trades=len(losses),
                win_rate=win_rate,
                avg_return=statistics.mean(returns) if returns else 0,
                median_return=statistics.median(returns) if returns else 0,
                std_return=statistics.stdev(returns) if len(returns) > 1 else 0,
                best_return=max(returns) if returns else 0,
                worst_return=min(returns) if returns else 0,
                profit_factor=total_wins / total_losses,
                conditions={indicator_name: {'min': low, 'max': high}},
                confidence=confidence
            )
            patterns.append(pattern)
        
        return patterns
    
    def _analyze_categorical(
        self,
        trades: List[TradeAnalysisRecord],
        category_name: str,
        value_fn
    ) -> List[PatternStatistics]:
        """Analyze patterns based on categorical values"""
        patterns = []
        
        # Group by category
        by_category = defaultdict(list)
        for t in trades:
            cat = value_fn(t)
            by_category[cat].append(t)
        
        for cat, cat_trades in by_category.items():
            if len(cat_trades) < self.MIN_SAMPLES_LOW:
                continue
            
            wins = [t for t in cat_trades if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
            losses = [t for t in cat_trades if t.result in [TradeResult.BIG_LOSS, TradeResult.LOSS, TradeResult.SMALL_LOSS]]
            
            returns = [t.pnl_percent for t in cat_trades]
            win_rate = len(wins) / len(cat_trades) if cat_trades else 0
            
            total_wins = sum(t.pnl_percent for t in wins) if wins else 0
            total_losses = abs(sum(t.pnl_percent for t in losses)) if losses else 0.01
            
            # Determine confidence
            if len(cat_trades) >= self.MIN_SAMPLES_HIGH:
                confidence = OptimizationConfidence.HIGH
            elif len(cat_trades) >= self.MIN_SAMPLES_MEDIUM:
                confidence = OptimizationConfidence.MEDIUM
            else:
                confidence = OptimizationConfidence.LOW
            
            pattern = PatternStatistics(
                pattern_id=f"{category_name}_{cat}",
                description=f"{category_name} = {cat}",
                total_trades=len(cat_trades),
                winning_trades=len(wins),
                losing_trades=len(losses),
                win_rate=win_rate,
                avg_return=statistics.mean(returns) if returns else 0,
                median_return=statistics.median(returns) if returns else 0,
                std_return=statistics.stdev(returns) if len(returns) > 1 else 0,
                best_return=max(returns) if returns else 0,
                worst_return=min(returns) if returns else 0,
                profit_factor=total_wins / total_losses,
                conditions={category_name: cat},
                confidence=confidence
            )
            patterns.append(pattern)
        
        return patterns
    
    def _analyze_combinations(
        self,
        trades: List[TradeAnalysisRecord]
    ) -> List[PatternStatistics]:
        """Analyze multi-condition pattern combinations"""
        patterns = []
        
        # Define interesting combinations to check
        combinations = [
            # RSI oversold + Volume spike
            {
                'id': 'rsi_oversold_volume_spike',
                'desc': 'RSI < 30 + Volume > 2x',
                'filter': lambda t: t.indicators.rsi_value < 30 and t.indicators.volume_ratio > 2.0
            },
            # RSI overbought + Weak volume
            {
                'id': 'rsi_overbought_weak_volume',
                'desc': 'RSI > 70 + Volume < 0.7x',
                'filter': lambda t: t.indicators.rsi_value > 70 and t.indicators.volume_ratio < 0.7
            },
            # Bullish EMA + Strong ADX
            {
                'id': 'bullish_ema_strong_trend',
                'desc': 'Bullish EMA + ADX > 30',
                'filter': lambda t: t.indicators.ema_alignment == 'bullish' and t.indicators.adx_value > 30
            },
            # Bearish EMA + High VIX
            {
                'id': 'bearish_ema_high_vix',
                'desc': 'Bearish EMA + VIX > 25',
                'filter': lambda t: t.indicators.ema_alignment == 'bearish' and t.indicators.vix_level > 25
            },
            # High conviction + Low VIX
            {
                'id': 'high_conviction_low_vix',
                'desc': 'Score > 70 + VIX < 18',
                'filter': lambda t: t.combined_score > 70 and t.indicators.vix_level < 18
            },
            # Volume spike + Bullish news
            {
                'id': 'volume_spike_bullish_news',
                'desc': 'Volume > 2x + News > 50',
                'filter': lambda t: t.indicators.volume_ratio > 2.0 and t.news_score > 50
            },
            # Technical + Sentiment alignment
            {
                'id': 'tech_sentiment_alignment',
                'desc': 'Technical > 60 + Sentiment > 60',
                'filter': lambda t: t.technical_score > 60 and t.sentiment_score > 60
            },
            # Near support + Volume
            {
                'id': 'near_support_volume',
                'desc': 'BB < 0.2 + Volume > 1.5x',
                'filter': lambda t: t.indicators.bb_position < 0.2 and t.indicators.volume_ratio > 1.5
            },
            # Strong grok sentiment
            {
                'id': 'strong_grok_bullish',
                'desc': 'Grok > 70 + Volume > 1.3x',
                'filter': lambda t: t.indicators.grok_score > 70 and t.indicators.volume_ratio > 1.3
            },
            # Bull market regime + High score
            {
                'id': 'bull_regime_high_score',
                'desc': 'Bull regime + Score > 65',
                'filter': lambda t: 'bull' in t.indicators.market_regime.lower() and t.combined_score > 65
            }
        ]
        
        for combo in combinations:
            combo_trades = [t for t in trades if combo['filter'](t)]
            
            if len(combo_trades) < self.MIN_SAMPLES_LOW:
                continue
            
            wins = [t for t in combo_trades if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
            losses = [t for t in combo_trades if t.result in [TradeResult.BIG_LOSS, TradeResult.LOSS, TradeResult.SMALL_LOSS]]
            
            returns = [t.pnl_percent for t in combo_trades]
            win_rate = len(wins) / len(combo_trades) if combo_trades else 0
            
            total_wins = sum(t.pnl_percent for t in wins) if wins else 0
            total_losses = abs(sum(t.pnl_percent for t in losses)) if losses else 0.01
            
            # Determine confidence
            if len(combo_trades) >= self.MIN_SAMPLES_HIGH:
                confidence = OptimizationConfidence.HIGH
            elif len(combo_trades) >= self.MIN_SAMPLES_MEDIUM:
                confidence = OptimizationConfidence.MEDIUM
            else:
                confidence = OptimizationConfidence.LOW
            
            pattern = PatternStatistics(
                pattern_id=combo['id'],
                description=combo['desc'],
                total_trades=len(combo_trades),
                winning_trades=len(wins),
                losing_trades=len(losses),
                win_rate=win_rate,
                avg_return=statistics.mean(returns) if returns else 0,
                median_return=statistics.median(returns) if returns else 0,
                std_return=statistics.stdev(returns) if len(returns) > 1 else 0,
                best_return=max(returns) if returns else 0,
                worst_return=min(returns) if returns else 0,
                profit_factor=total_wins / total_losses,
                conditions={'combination': combo['desc']},
                confidence=confidence
            )
            patterns.append(pattern)
        
        return patterns
    
    # -------------------------------------------------------------------------
    # PARAMETER OPTIMIZATION
    # -------------------------------------------------------------------------
    
    async def suggest_parameter_changes(
        self,
        trades: Optional[List[TradeAnalysisRecord]] = None
    ) -> List[ParameterChange]:
        """
        Analyze trades and suggest parameter changes.
        
        Returns:
            List of recommended ParameterChange objects
        """
        trades = trades or self._trade_records
        changes = []
        
        if len(trades) < self.MIN_SAMPLES_LOW:
            logger.warning(f"Not enough trades for parameter optimization ({len(trades)})")
            return changes
        
        # 1. Analyze RSI threshold optimization
        rsi_change = self._optimize_rsi_threshold(trades)
        if rsi_change:
            changes.append(rsi_change)
        
        # 2. Analyze Volume threshold
        volume_change = self._optimize_volume_threshold(trades)
        if volume_change:
            changes.append(volume_change)
        
        # 3. Analyze minimum confidence score
        conf_change = self._optimize_confidence_threshold(trades)
        if conf_change:
            changes.append(conf_change)
        
        # 4. Analyze minimum combined score
        score_change = self._optimize_score_threshold(trades)
        if score_change:
            changes.append(score_change)
        
        # 5. Analyze ADX threshold
        adx_change = self._optimize_adx_threshold(trades)
        if adx_change:
            changes.append(adx_change)
        
        # 6. Pillar weight optimization
        weight_changes = self._optimize_pillar_weights(trades)
        changes.extend(weight_changes)
        
        logger.info(f"Generated {len(changes)} parameter change suggestions")
        return changes
    
    def _optimize_rsi_threshold(self, trades: List[TradeAnalysisRecord]) -> Optional[ParameterChange]:
        """Find optimal RSI oversold threshold"""
        current_threshold = 30  # Default
        
        # Test different thresholds
        best_threshold = current_threshold
        best_win_rate = 0
        
        for threshold in [20, 25, 30, 35, 40]:
            oversold_trades = [t for t in trades if t.indicators.rsi_value < threshold]
            if len(oversold_trades) < self.MIN_SAMPLES_LOW:
                continue
            
            wins = [t for t in oversold_trades if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
            win_rate = len(wins) / len(oversold_trades)
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_threshold = threshold
        
        if best_threshold != current_threshold and best_win_rate > 0.50:
            # Calculate current win rate at current threshold
            current_trades = [t for t in trades if t.indicators.rsi_value < current_threshold]
            current_wins = [t for t in current_trades if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
            current_win_rate = len(current_wins) / len(current_trades) if current_trades else 0
            
            improvement = (best_win_rate - current_win_rate) * 100
            
            if improvement > 5:  # At least 5% improvement
                sample_size = len([t for t in trades if t.indicators.rsi_value < best_threshold])
                confidence = OptimizationConfidence.HIGH if sample_size >= 20 else OptimizationConfidence.MEDIUM if sample_size >= 10 else OptimizationConfidence.LOW
                
                return ParameterChange(
                    parameter_name='rsi_oversold',
                    parameter_type=ParameterType.THRESHOLD,
                    pillar='technical',
                    current_value=current_threshold,
                    proposed_value=best_threshold,
                    reason=f"RSI < {best_threshold} shows {best_win_rate:.0%} win rate vs {current_win_rate:.0%} at current {current_threshold}",
                    expected_improvement=improvement,
                    sample_size=sample_size,
                    win_rate_current=current_win_rate,
                    win_rate_proposed=best_win_rate,
                    confidence=confidence
                )
        
        return None
    
    def _optimize_volume_threshold(self, trades: List[TradeAnalysisRecord]) -> Optional[ParameterChange]:
        """Find optimal volume ratio threshold"""
        current_threshold = 1.5  # Default
        
        best_threshold = current_threshold
        best_win_rate = 0
        
        for threshold in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
            high_vol_trades = [t for t in trades if t.indicators.volume_ratio > threshold]
            if len(high_vol_trades) < self.MIN_SAMPLES_LOW:
                continue
            
            wins = [t for t in high_vol_trades if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
            win_rate = len(wins) / len(high_vol_trades)
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_threshold = threshold
        
        if best_threshold != current_threshold and best_win_rate > 0.50:
            current_trades = [t for t in trades if t.indicators.volume_ratio > current_threshold]
            current_wins = [t for t in current_trades if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
            current_win_rate = len(current_wins) / len(current_trades) if current_trades else 0
            
            improvement = (best_win_rate - current_win_rate) * 100
            
            if abs(improvement) > 3:
                sample_size = len([t for t in trades if t.indicators.volume_ratio > best_threshold])
                confidence = OptimizationConfidence.HIGH if sample_size >= 20 else OptimizationConfidence.MEDIUM if sample_size >= 10 else OptimizationConfidence.LOW
                
                return ParameterChange(
                    parameter_name='volume_ratio_min',
                    parameter_type=ParameterType.THRESHOLD,
                    pillar='technical',
                    current_value=current_threshold,
                    proposed_value=best_threshold,
                    reason=f"Volume > {best_threshold}x shows {best_win_rate:.0%} win rate",
                    expected_improvement=improvement,
                    sample_size=sample_size,
                    win_rate_current=current_win_rate,
                    win_rate_proposed=best_win_rate,
                    confidence=confidence
                )
        
        return None
    
    def _optimize_confidence_threshold(self, trades: List[TradeAnalysisRecord]) -> Optional[ParameterChange]:
        """Find optimal confidence threshold"""
        current_threshold = 55
        
        best_threshold = current_threshold
        best_performance = 0  # Combined metric: win_rate * avg_return
        
        for threshold in range(40, 85, 5):
            high_conf_trades = [t for t in trades if t.confidence > threshold]
            if len(high_conf_trades) < self.MIN_SAMPLES_LOW:
                continue
            
            wins = [t for t in high_conf_trades if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
            win_rate = len(wins) / len(high_conf_trades)
            avg_return = statistics.mean([t.pnl_percent for t in high_conf_trades])
            
            performance = win_rate * (1 + avg_return / 10)  # Weight win rate more
            
            if performance > best_performance:
                best_performance = performance
                best_threshold = threshold
        
        if best_threshold != current_threshold:
            current_trades = [t for t in trades if t.confidence > current_threshold]
            current_wins = [t for t in current_trades if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
            current_win_rate = len(current_wins) / len(current_trades) if current_trades else 0
            
            proposed_trades = [t for t in trades if t.confidence > best_threshold]
            proposed_wins = [t for t in proposed_trades if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
            proposed_win_rate = len(proposed_wins) / len(proposed_trades) if proposed_trades else 0
            
            improvement = (proposed_win_rate - current_win_rate) * 100
            
            if abs(improvement) > 3:
                sample_size = len(proposed_trades)
                confidence = OptimizationConfidence.HIGH if sample_size >= 20 else OptimizationConfidence.MEDIUM if sample_size >= 10 else OptimizationConfidence.LOW
                
                return ParameterChange(
                    parameter_name='min_confidence',
                    parameter_type=ParameterType.THRESHOLD,
                    pillar='combined',
                    current_value=current_threshold,
                    proposed_value=best_threshold,
                    reason=f"Confidence > {best_threshold} shows better risk-adjusted returns",
                    expected_improvement=improvement,
                    sample_size=sample_size,
                    win_rate_current=current_win_rate,
                    win_rate_proposed=proposed_win_rate,
                    confidence=confidence
                )
        
        return None
    
    def _optimize_score_threshold(self, trades: List[TradeAnalysisRecord]) -> Optional[ParameterChange]:
        """Find optimal combined score threshold"""
        current_threshold = 55
        
        best_threshold = current_threshold
        best_win_rate = 0
        
        for threshold in range(45, 80, 5):
            high_score_trades = [t for t in trades if t.combined_score > threshold]
            if len(high_score_trades) < self.MIN_SAMPLES_LOW:
                continue
            
            wins = [t for t in high_score_trades if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
            win_rate = len(wins) / len(high_score_trades)
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_threshold = threshold
        
        if best_threshold != current_threshold and best_win_rate > 0.50:
            current_trades = [t for t in trades if t.combined_score > current_threshold]
            current_wins = [t for t in current_trades if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
            current_win_rate = len(current_wins) / len(current_trades) if current_trades else 0
            
            improvement = (best_win_rate - current_win_rate) * 100
            
            if abs(improvement) > 3:
                sample_size = len([t for t in trades if t.combined_score > best_threshold])
                confidence = OptimizationConfidence.HIGH if sample_size >= 20 else OptimizationConfidence.MEDIUM if sample_size >= 10 else OptimizationConfidence.LOW
                
                return ParameterChange(
                    parameter_name='min_combined_score',
                    parameter_type=ParameterType.THRESHOLD,
                    pillar='combined',
                    current_value=current_threshold,
                    proposed_value=best_threshold,
                    reason=f"Combined score > {best_threshold} shows {best_win_rate:.0%} win rate",
                    expected_improvement=improvement,
                    sample_size=sample_size,
                    win_rate_current=current_win_rate,
                    win_rate_proposed=best_win_rate,
                    confidence=confidence
                )
        
        return None
    
    def _optimize_adx_threshold(self, trades: List[TradeAnalysisRecord]) -> Optional[ParameterChange]:
        """Find optimal ADX trend strength threshold"""
        current_threshold = 25
        
        best_threshold = current_threshold
        best_win_rate = 0
        
        for threshold in [15, 20, 25, 30, 35, 40]:
            strong_trend_trades = [t for t in trades if t.indicators.adx_value > threshold]
            if len(strong_trend_trades) < self.MIN_SAMPLES_LOW:
                continue
            
            wins = [t for t in strong_trend_trades if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
            win_rate = len(wins) / len(strong_trend_trades)
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_threshold = threshold
        
        if best_threshold != current_threshold and best_win_rate > 0.50:
            current_trades = [t for t in trades if t.indicators.adx_value > current_threshold]
            current_wins = [t for t in current_trades if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
            current_win_rate = len(current_wins) / len(current_trades) if current_trades else 0
            
            improvement = (best_win_rate - current_win_rate) * 100
            
            if abs(improvement) > 3:
                sample_size = len([t for t in trades if t.indicators.adx_value > best_threshold])
                confidence = OptimizationConfidence.HIGH if sample_size >= 20 else OptimizationConfidence.MEDIUM if sample_size >= 10 else OptimizationConfidence.LOW
                
                return ParameterChange(
                    parameter_name='adx_threshold',
                    parameter_type=ParameterType.THRESHOLD,
                    pillar='technical',
                    current_value=current_threshold,
                    proposed_value=best_threshold,
                    reason=f"ADX > {best_threshold} shows {best_win_rate:.0%} win rate in trending markets",
                    expected_improvement=improvement,
                    sample_size=sample_size,
                    win_rate_current=current_win_rate,
                    win_rate_proposed=best_win_rate,
                    confidence=confidence
                )
        
        return None
    
    def _optimize_pillar_weights(self, trades: List[TradeAnalysisRecord]) -> List[ParameterChange]:
        """
        Optimize pillar weights based on which pillars contributed most to winning trades.
        """
        changes = []
        
        if len(trades) < self.MIN_SAMPLES_MEDIUM:
            return changes
        
        # Separate wins and losses
        wins = [t for t in trades if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
        losses = [t for t in trades if t.result in [TradeResult.BIG_LOSS, TradeResult.LOSS, TradeResult.SMALL_LOSS]]
        
        if not wins or not losses:
            return changes
        
        # Analyze pillar contribution
        pillars = ['technical', 'fundamental', 'sentiment', 'news']
        pillar_scores = {p: {'win_avg': 0, 'loss_avg': 0, 'contribution': 0} for p in pillars}
        
        for pillar in pillars:
            win_scores = [getattr(t, f'{pillar}_score', 0) for t in wins]
            loss_scores = [getattr(t, f'{pillar}_score', 0) for t in losses]
            
            pillar_scores[pillar]['win_avg'] = statistics.mean(win_scores) if win_scores else 0
            pillar_scores[pillar]['loss_avg'] = statistics.mean(loss_scores) if loss_scores else 0
            pillar_scores[pillar]['contribution'] = pillar_scores[pillar]['win_avg'] - pillar_scores[pillar]['loss_avg']
        
        # Calculate optimal weights based on contribution
        total_contribution = sum(max(0, p['contribution']) for p in pillar_scores.values())
        
        if total_contribution == 0:
            return changes
        
        default_weight = 0.25
        
        for pillar in pillars:
            contribution = max(0, pillar_scores[pillar]['contribution'])
            optimal_weight = (contribution / total_contribution) * 0.8 + 0.05  # Ensure minimum 5%
            optimal_weight = min(0.40, max(0.10, optimal_weight))  # Clamp to bounds
            optimal_weight = round(optimal_weight * 20) / 20  # Round to 0.05
            
            if abs(optimal_weight - default_weight) >= 0.05:
                changes.append(ParameterChange(
                    parameter_name=f'weight_{pillar}',
                    parameter_type=ParameterType.WEIGHT,
                    pillar=pillar,
                    current_value=default_weight,
                    proposed_value=optimal_weight,
                    reason=f"{pillar.title()} pillar shows {pillar_scores[pillar]['contribution']:.1f} point differential (wins vs losses)",
                    expected_improvement=(optimal_weight - default_weight) * 10,  # Rough estimate
                    sample_size=len(trades),
                    win_rate_current=len(wins) / len(trades),
                    win_rate_proposed=len(wins) / len(trades) * (1 + abs(optimal_weight - default_weight)),
                    confidence=OptimizationConfidence.MEDIUM if len(trades) >= 15 else OptimizationConfidence.LOW
                ))
        
        return changes
    
    # -------------------------------------------------------------------------
    # MAIN OPTIMIZATION
    # -------------------------------------------------------------------------
    
    async def optimize_from_trades(
        self,
        trades: Optional[List[Dict]] = None,
        current_config: Optional[Dict] = None
    ) -> OptimizationReport:
        """
        Run full optimization analysis and generate report.
        
        Args:
            trades: Raw trade data (will convert to TradeAnalysisRecord)
            current_config: Current configuration/weights
            
        Returns:
            OptimizationReport with all findings and recommendations
        """
        # Convert trades if provided as raw dicts
        if trades:
            for trade in trades:
                if not isinstance(trade, TradeAnalysisRecord):
                    # Convert and add to records
                    pillar_scores = {
                        'technical': trade.get('technical_score', 0),
                        'fundamental': trade.get('fundamental_score', 0),
                        'sentiment': trade.get('sentiment_score', 0),
                        'news': trade.get('news_score', 0),
                        'combined': trade.get('combined_score', 0),
                        'confidence': trade.get('confidence', 0)
                    }
                    indicators = trade.get('indicators', {})
                    weights = trade.get('weights', {'technical': 0.25, 'fundamental': 0.25, 'sentiment': 0.25, 'news': 0.25})
                    
                    self.record_trade(trade, pillar_scores, indicators, weights)
        
        trades_to_analyze = self._trade_records
        
        if len(trades_to_analyze) < self.MIN_SAMPLES_LOW:
            return OptimizationReport(
                date=datetime.now().isoformat(),
                trades_analyzed=len(trades_to_analyze),
                total_pnl=0,
                win_rate=0,
                profit_factor=0,
                avg_win=0,
                avg_loss=0,
                winning_patterns=[],
                losing_patterns=[],
                recommended_changes=[],
                pillar_performance={},
                recommended_weights={'technical': 0.25, 'fundamental': 0.25, 'sentiment': 0.25, 'news': 0.25},
                warnings=[f"Insufficient trade data ({len(trades_to_analyze)} trades). Need at least {self.MIN_SAMPLES_LOW}."]
            )
        
        # Calculate basic stats
        wins = [t for t in trades_to_analyze if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
        losses = [t for t in trades_to_analyze if t.result in [TradeResult.BIG_LOSS, TradeResult.LOSS, TradeResult.SMALL_LOSS]]
        
        total_pnl = sum(t.pnl_percent for t in trades_to_analyze)
        win_rate = len(wins) / len(trades_to_analyze) if trades_to_analyze else 0
        
        total_wins = sum(t.pnl_percent for t in wins) if wins else 0
        total_losses = abs(sum(t.pnl_percent for t in losses)) if losses else 0.01
        profit_factor = total_wins / total_losses
        
        avg_win = statistics.mean([t.pnl_percent for t in wins]) if wins else 0
        avg_loss = statistics.mean([t.pnl_percent for t in losses]) if losses else 0
        
        # Identify patterns
        winning_patterns, losing_patterns = await self.identify_patterns(trades_to_analyze)
        
        # Get parameter change suggestions
        recommended_changes = await self.suggest_parameter_changes(trades_to_analyze)
        
        # Analyze pillar performance
        pillar_performance = self._analyze_pillar_performance(trades_to_analyze)
        
        # Calculate recommended weights
        recommended_weights = self._calculate_recommended_weights(pillar_performance)
        
        # Generate warnings
        warnings = []
        if win_rate < 0.40:
            warnings.append(f"Low win rate ({win_rate:.0%}). Consider tightening entry criteria.")
        if profit_factor < 1.0:
            warnings.append(f"Profit factor < 1 ({profit_factor:.2f}). Strategy is losing money overall.")
        if avg_loss > avg_win * 2:
            warnings.append(f"Average loss (${avg_loss:.1f}) is much larger than average win (${avg_win:.1f}). Review stop-loss strategy.")
        if len(losses) > len(wins) * 2:
            warnings.append("Significantly more losing trades than winning. Quality over quantity!")
        
        report = OptimizationReport(
            date=datetime.now().isoformat(),
            trades_analyzed=len(trades_to_analyze),
            total_pnl=total_pnl,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            winning_patterns=winning_patterns[:10],  # Top 10
            losing_patterns=losing_patterns[:10],
            recommended_changes=recommended_changes,
            pillar_performance=pillar_performance,
            recommended_weights=recommended_weights,
            warnings=warnings
        )
        
        # Save report
        self._save_optimization_report(report)
        
        # Record changes to history
        for change in recommended_changes:
            self._history.append({
                'date': datetime.now().isoformat(),
                'change': change.to_dict(),
                'applied': False
            })
        self._save_history()
        
        logger.info(f"Optimization complete: {len(recommended_changes)} changes suggested, {len(winning_patterns)} winning patterns, {len(losing_patterns)} losing patterns")
        
        return report
    
    def _analyze_pillar_performance(self, trades: List[TradeAnalysisRecord]) -> Dict[str, Dict[str, float]]:
        """Analyze how each pillar contributed to trade outcomes"""
        performance = {}
        
        wins = [t for t in trades if t.result in [TradeResult.BIG_WIN, TradeResult.WIN, TradeResult.SMALL_WIN]]
        losses = [t for t in trades if t.result in [TradeResult.BIG_LOSS, TradeResult.LOSS, TradeResult.SMALL_LOSS]]
        
        for pillar in ['technical', 'fundamental', 'sentiment', 'news']:
            all_scores = [getattr(t, f'{pillar}_score', 0) for t in trades]
            win_scores = [getattr(t, f'{pillar}_score', 0) for t in wins]
            loss_scores = [getattr(t, f'{pillar}_score', 0) for t in losses]
            
            performance[pillar] = {
                'avg_score': statistics.mean(all_scores) if all_scores else 0,
                'avg_win_score': statistics.mean(win_scores) if win_scores else 0,
                'avg_loss_score': statistics.mean(loss_scores) if loss_scores else 0,
                'contribution_delta': (statistics.mean(win_scores) if win_scores else 0) - (statistics.mean(loss_scores) if loss_scores else 0),
                'correlation_to_outcome': self._calculate_correlation([getattr(t, f'{pillar}_score', 0) for t in trades], [t.pnl_percent for t in trades]) if len(trades) > 2 else 0
            }
        
        return performance
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        n = len(x)
        if n < 3:
            return 0
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        
        sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
        sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
        
        denominator = (sum_sq_x * sum_sq_y) ** 0.5
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def _calculate_recommended_weights(self, pillar_performance: Dict) -> Dict[str, float]:
        """Calculate recommended pillar weights based on performance"""
        weights = {}
        
        # Use contribution delta as primary metric
        deltas = {p: data['contribution_delta'] for p, data in pillar_performance.items()}
        
        # Shift to positive range
        min_delta = min(deltas.values())
        shifted = {p: d - min_delta + 1 for p, d in deltas.items()}  # +1 to avoid zero
        
        total = sum(shifted.values())
        
        for pillar in ['technical', 'fundamental', 'sentiment', 'news']:
            raw_weight = shifted[pillar] / total if total > 0 else 0.25
            # Apply bounds
            weight = max(0.10, min(0.40, raw_weight))
            # Round to 0.05
            weight = round(weight * 20) / 20
            weights[pillar] = weight
        
        # Normalize to sum to 1.0
        total_weights = sum(weights.values())
        weights = {p: w / total_weights for p, w in weights.items()}
        
        return weights
    
    def _save_optimization_report(self, report: OptimizationReport):
        """Save optimization report to file"""
        reports_dir = self.data_dir / "optimization_reports"
        reports_dir.mkdir(exist_ok=True)
        
        filename = f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = reports_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            logger.info(f"Saved optimization report: {filename}")
        except Exception as e:
            logger.error(f"Error saving optimization report: {e}")
    
    # -------------------------------------------------------------------------
    # APPLY OPTIMIZATIONS
    # -------------------------------------------------------------------------
    
    def apply_optimizations(
        self,
        changes: List[ParameterChange],
        config_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply recommended parameter changes.
        
        Args:
            changes: List of ParameterChange to apply
            config_file: Optional config file to update
            
        Returns:
            Updated configuration dict
        """
        config = {}
        
        # Load existing config if provided
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
            except:
                pass
        
        # Apply changes
        for change in changes:
            # Only apply HIGH or MEDIUM confidence changes automatically
            if change.confidence == OptimizationConfidence.LOW:
                logger.info(f"Skipping low-confidence change: {change.parameter_name}")
                continue
            
            # Validate bounds
            bounds = self.PARAMETER_BOUNDS.get(change.parameter_name)
            if bounds:
                if not (bounds['min'] <= change.proposed_value <= bounds['max']):
                    logger.warning(f"Proposed value {change.proposed_value} for {change.parameter_name} outside bounds [{bounds['min']}, {bounds['max']}]")
                    continue
            
            config[change.parameter_name] = change.proposed_value
            
            # Mark as applied in history
            for h in self._history:
                if h.get('change', {}).get('parameter_name') == change.parameter_name:
                    h['applied'] = True
                    h['applied_at'] = datetime.now().isoformat()
            
            logger.info(f"Applied: {change.parameter_name} = {change.proposed_value} (was {change.current_value})")
        
        self._save_history()
        
        # Save config if file provided
        if config_file:
            try:
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving config: {e}")
        
        return config
    
    def get_parameter_bounds(self, parameter: str) -> Optional[Dict]:
        """Get bounds for a parameter"""
        return self.PARAMETER_BOUNDS.get(parameter)
    
    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get recent parameter change history"""
        return self._history[-limit:]
    
    def get_patterns(self, min_confidence: OptimizationConfidence = OptimizationConfidence.LOW) -> List[PatternStatistics]:
        """Get learned patterns"""
        confidence_order = {
            OptimizationConfidence.LOW: 0,
            OptimizationConfidence.MEDIUM: 1,
            OptimizationConfidence.HIGH: 2
        }
        min_level = confidence_order.get(min_confidence, 0)
        
        return [
            p for p in self._patterns.values()
            if confidence_order.get(p.confidence, 0) >= min_level
        ]


# =============================================================================
# FACTORY
# =============================================================================

_optimizer_instance: Optional[ParameterOptimizer] = None


def get_parameter_optimizer(data_dir: str = "data/auditor") -> ParameterOptimizer:
    """Factory to get ParameterOptimizer singleton"""
    global _optimizer_instance
    
    if _optimizer_instance is None:
        _optimizer_instance = ParameterOptimizer(data_dir=data_dir)
    
    return _optimizer_instance


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("=== Parameter Optimizer Test ===\n")
        
        optimizer = ParameterOptimizer()
        
        # Simulate some trades
        test_trades = [
            {
                'trade_id': 'TEST001',
                'symbol': 'NVDA',
                'sector': 'Technology',
                'entry_date': '2025-01-15T10:30:00',
                'exit_date': '2025-01-16T15:00:00',
                'entry_price': 140.0,
                'exit_price': 147.0,
                'stop_loss': 135.0,
                'pnl_amount': 70.0,
                'pnl_percent': 5.0,
                'exit_type': 'take_profit',
                'technical_score': 72,
                'fundamental_score': 65,
                'sentiment_score': 78,
                'news_score': 55,
                'combined_score': 68,
                'confidence': 75,
                'indicators': {
                    'rsi': 35,
                    'volume_ratio': 2.3,
                    'adx': 32,
                    'ema_alignment': 'bullish',
                    'bb_position': 0.3,
                    'vix_level': 16,
                    'market_regime': 'bull_strong',
                    'grok_score': 72
                },
                'weights': {'technical': 0.25, 'fundamental': 0.25, 'sentiment': 0.25, 'news': 0.25}
            },
            {
                'trade_id': 'TEST002',
                'symbol': 'TSLA',
                'sector': 'Consumer',
                'entry_date': '2025-01-15T11:00:00',
                'exit_date': '2025-01-15T14:30:00',
                'entry_price': 250.0,
                'exit_price': 242.0,
                'stop_loss': 245.0,
                'pnl_amount': -40.0,
                'pnl_percent': -3.2,
                'exit_type': 'stop_loss',
                'technical_score': 55,
                'fundamental_score': 45,
                'sentiment_score': 62,
                'news_score': 48,
                'combined_score': 52,
                'confidence': 48,
                'indicators': {
                    'rsi': 72,
                    'volume_ratio': 0.6,
                    'adx': 18,
                    'ema_alignment': 'bearish',
                    'bb_position': 0.85,
                    'vix_level': 22,
                    'market_regime': 'volatile',
                    'grok_score': 55
                },
                'weights': {'technical': 0.25, 'fundamental': 0.25, 'sentiment': 0.25, 'news': 0.25}
            },
            {
                'trade_id': 'TEST003',
                'symbol': 'AAPL',
                'sector': 'Technology',
                'entry_date': '2025-01-16T09:45:00',
                'exit_date': '2025-01-17T16:00:00',
                'entry_price': 195.0,
                'exit_price': 201.0,
                'stop_loss': 190.0,
                'pnl_amount': 48.0,
                'pnl_percent': 3.1,
                'exit_type': 'trailing_stop',
                'technical_score': 68,
                'fundamental_score': 72,
                'sentiment_score': 65,
                'news_score': 70,
                'combined_score': 69,
                'confidence': 72,
                'indicators': {
                    'rsi': 42,
                    'volume_ratio': 1.8,
                    'adx': 28,
                    'ema_alignment': 'bullish',
                    'bb_position': 0.4,
                    'vix_level': 17,
                    'market_regime': 'bull_weak',
                    'grok_score': 68
                },
                'weights': {'technical': 0.25, 'fundamental': 0.25, 'sentiment': 0.25, 'news': 0.25}
            }
        ]
        
        # Record trades
        for trade in test_trades:
            pillar_scores = {
                'technical': trade['technical_score'],
                'fundamental': trade['fundamental_score'],
                'sentiment': trade['sentiment_score'],
                'news': trade['news_score'],
                'combined': trade['combined_score'],
                'confidence': trade['confidence']
            }
            optimizer.record_trade(trade, pillar_scores, trade['indicators'], trade['weights'])
        
        # Run optimization
        report = await optimizer.optimize_from_trades()
        
        print(f"Trades analyzed: {report.trades_analyzed}")
        print(f"Win Rate: {report.win_rate:.0%}")
        print(f"Profit Factor: {report.profit_factor:.2f}")
        print(f"Total PnL: {report.total_pnl:+.1f}%")
        print()
        
        print("Winning Patterns:")
        for p in report.winning_patterns[:5]:
            print(f"  - {p.description}: {p.win_rate:.0%} win rate ({p.total_trades} trades)")
        print()
        
        print("Losing Patterns:")
        for p in report.losing_patterns[:5]:
            print(f"  - {p.description}: {p.win_rate:.0%} win rate ({p.total_trades} trades)")
        print()
        
        print("Recommended Changes:")
        for c in report.recommended_changes:
            print(f"  - {c.parameter_name}: {c.current_value} → {c.proposed_value}")
            print(f"    Reason: {c.reason}")
            print(f"    Confidence: {c.confidence.value}")
        print()
        
        print("Recommended Weights:")
        for pillar, weight in report.recommended_weights.items():
            print(f"  - {pillar}: {weight:.0%}")
        print()
        
        print("Warnings:")
        for w in report.warnings:
            print(f"  ⚠️ {w}")
    
    asyncio.run(main())
