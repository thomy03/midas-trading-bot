"""
Sample data fixtures for testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_sample_price_data(
    periods: int = 100,
    start_price: float = 100.0,
    volatility: float = 0.02
) -> pd.DataFrame:
    """
    Create sample OHLCV price data for testing

    Args:
        periods: Number of periods to generate
        start_price: Starting price
        volatility: Price volatility (standard deviation)

    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=periods),
        periods=periods,
        freq='D'
    )

    # Generate random walk
    returns = np.random.normal(0, volatility, periods)
    prices = start_price * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, periods)),
        'High': prices * (1 + np.random.uniform(0, 0.02, periods)),
        'Low': prices * (1 + np.random.uniform(-0.02, 0, periods)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, periods)
    }, index=dates)

    return data


def create_uptrend_data(periods: int = 100) -> pd.DataFrame:
    """Create data with clear uptrend"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=periods),
        periods=periods,
        freq='D'
    )

    # Linear uptrend with noise
    trend = np.linspace(100, 150, periods)
    noise = np.random.normal(0, 2, periods)
    prices = trend + noise

    data = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, periods)
    }, index=dates)

    return data


def create_downtrend_data(periods: int = 100) -> pd.DataFrame:
    """Create data with clear downtrend"""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=periods),
        periods=periods,
        freq='D'
    )

    # Linear downtrend with noise
    trend = np.linspace(150, 100, periods)
    noise = np.random.normal(0, 2, periods)
    prices = trend + noise

    data = pd.DataFrame({
        'Open': prices * 1.01,
        'High': prices * 1.02,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, periods)
    }, index=dates)

    return data


SAMPLE_SYMBOLS = {
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
    'finance': ['JPM', 'BAC', 'GS', 'MS', 'C'],
    'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
    'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO']
}


SAMPLE_ALERT = {
    'symbol': 'AAPL',
    'company_name': 'Apple Inc.',
    'timeframe': 'weekly',
    'current_price': 175.50,
    'support_level': 170.00,
    'distance_to_support_pct': 3.24,
    'ema_24': 172.30,
    'ema_38': 169.80,
    'ema_62': 167.50,
    'ema_alignment': True,
    'recommendation': 'BUY',
    'has_rsi_breakout': True,
    'rsi_signal': 'BREAKOUT'
}
