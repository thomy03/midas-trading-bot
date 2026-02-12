"""
Liquidity Filter - Sprint 1C Fix 2
Checks if a symbol is liquid enough to trade.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def check_liquidity(symbol: str, min_avg_volume: int = 500_000, 
                    min_avg_dollar_volume: float = 1_000_000,
                    max_spread_pct: float = 0.5) -> Dict[str, Any]:
    """
    Check if a symbol is liquid enough for trading.
    
    Args:
        symbol: Ticker symbol
        min_avg_volume: Min 20-day avg volume in shares (default 500K)
        min_avg_dollar_volume: Min 20-day avg dollar volume (default $1M)
        max_spread_pct: Max estimated spread % (default 0.5%)
    
    Returns:
        Dict with 'liquid' bool and details
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1mo')
        
        if hist.empty or len(hist) < 5:
            return {
                'liquid': False,
                'avg_volume': 0,
                'avg_dollar_volume': 0,
                'estimated_spread_pct': 999,
                'reason': f'Insufficient price history for {symbol}'
            }
        
        tail = hist.tail(20)
        avg_volume = float(tail['Volume'].mean())
        avg_price = float(tail['Close'].mean())
        avg_dollar_volume = avg_volume * avg_price
        
        # Spread proxy: average (High-Low)/Close
        avg_spread = float(((tail['High'] - tail['Low']) / tail['Close']).mean()) * 100
        
        is_liquid = (avg_volume > min_avg_volume and 
                     avg_dollar_volume > min_avg_dollar_volume and
                     avg_spread < max_spread_pct)
        
        reasons = []
        if avg_volume <= min_avg_volume:
            reasons.append(f'Low volume ({avg_volume:,.0f} < {min_avg_volume:,.0f})')
        if avg_dollar_volume <= min_avg_dollar_volume:
            reasons.append(f'Low $ volume (${avg_dollar_volume:,.0f} < ${min_avg_dollar_volume:,.0f})')
        if avg_spread >= max_spread_pct:
            reasons.append(f'Wide spread ({avg_spread:.2f}% >= {max_spread_pct}%)')
        
        return {
            'liquid': is_liquid,
            'avg_volume': avg_volume,
            'avg_dollar_volume': avg_dollar_volume,
            'estimated_spread_pct': round(avg_spread, 3),
            'reason': '; '.join(reasons) if reasons else None
        }
    except Exception as e:
        logger.warning(f"[LIQUIDITY] Error checking {symbol}: {e}")
        return {
            'liquid': True,  # fail-open: don't block on errors
            'avg_volume': 0,
            'avg_dollar_volume': 0,
            'estimated_spread_pct': 0,
            'reason': f'Error: {e}'
        }
