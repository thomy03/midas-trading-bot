"""
Indicators Package - Technical Analysis Indicator Library

This package provides a comprehensive library of technical indicators
that can be used by the Strategy Composer to build trading strategies.

Usage:
    from src.indicators import get_indicator_registry, compute_indicator

    # Get the registry
    registry = get_indicator_registry()

    # Compute an indicator
    df = compute_indicator('RSI', df, period=14)

    # Get a signal
    signal = get_signal('MACD', df)

    # List all available indicators
    all_indicators = registry.list_all()
"""

# Core library
from .library import (
    # Base classes
    BaseIndicator,
    IndicatorMetadata,
    IndicatorCategory,
    IndicatorSignal,
    SignalType,
    # Registry
    IndicatorRegistry,
    get_indicator_registry,
    # Convenience functions
    compute_indicator,
    get_signal,
    # Condition system
    ConditionOperator,
    IndicatorCondition,
)

# Oscillators
from .oscillators import (
    RSI,
    Stochastic,
    WilliamsR,
    ROC,
    MFI,
    CCI,
)

# Trend
from .trend import (
    EMA,
    SMA,
    MACD,
    ADX,
    Supertrend,
    ParabolicSAR,
)

# Volume
from .volume import (
    OBV,
    VWAP,
    VolumeRatio,
    CMF,
    ADLine,
    VWMA,
)

# Volatility
from .volatility import (
    ATR,
    BollingerBands,
    KeltnerChannel,
    DonchianChannel,
    StandardDeviation,
)

# Patterns
from .patterns import (
    SupportResistance,
    FibonacciRetracement,
    PivotPoints,
    SwingHighLow,
)

# Keep existing EMAAnalyzer for backwards compatibility
from .ema_analyzer import EMAAnalyzer

__all__ = [
    # Core
    'BaseIndicator',
    'IndicatorMetadata',
    'IndicatorCategory',
    'IndicatorSignal',
    'SignalType',
    'IndicatorRegistry',
    'get_indicator_registry',
    'compute_indicator',
    'get_signal',
    'ConditionOperator',
    'IndicatorCondition',

    # Oscillators
    'RSI',
    'Stochastic',
    'WilliamsR',
    'ROC',
    'MFI',
    'CCI',

    # Trend
    'EMA',
    'SMA',
    'MACD',
    'ADX',
    'Supertrend',
    'ParabolicSAR',

    # Volume
    'OBV',
    'VWAP',
    'VolumeRatio',
    'CMF',
    'ADLine',
    'VWMA',

    # Volatility
    'ATR',
    'BollingerBands',
    'KeltnerChannel',
    'DonchianChannel',
    'StandardDeviation',

    # Patterns
    'SupportResistance',
    'FibonacciRetracement',
    'PivotPoints',
    'SwingHighLow',

    # Legacy
    'EMAAnalyzer',
]
