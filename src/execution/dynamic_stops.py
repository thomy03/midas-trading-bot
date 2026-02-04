"""
Dynamic Stop Loss / Take Profit Calculator

Uses ATR (Average True Range) for volatility-adjusted stops:
- Stop Loss: 2x ATR below entry (tighter for high confidence)
- Take Profit: Based on risk/reward ratio and confidence
"""

import logging
from typing import Optional, Tuple
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_atr(symbol: str, period: int = 14) -> Optional[float]:
    """
    Calculate ATR (Average True Range) for a symbol.
    
    Args:
        symbol: Stock symbol
        period: ATR period (default 14)
        
    Returns:
        ATR value or None if calculation fails
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1mo", interval="1d")
        
        if df.empty or len(df) < period:
            logger.warning(f"Not enough data for ATR calculation: {symbol}")
            return None
        
        # Calculate True Range
        df['HL'] = df['High'] - df['Low']
        df['HC'] = abs(df['High'] - df['Close'].shift(1))
        df['LC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
        
        # ATR is the moving average of True Range
        atr = df['TR'].rolling(window=period).mean().iloc[-1]
        
        logger.debug(f"{symbol} ATR({period}): {atr:.2f}")
        return float(atr)
        
    except Exception as e:
        logger.warning(f"Could not calculate ATR for {symbol}: {e}")
        return None


def calculate_atr_percent(symbol: str, price: float, period: int = 14) -> Optional[float]:
    """Calculate ATR as percentage of price"""
    atr = calculate_atr(symbol, period)
    if atr and price > 0:
        return (atr / price) * 100
    return None


def calculate_dynamic_stops(
    symbol: str,
    entry_price: float,
    confidence_score: float = 50.0,
    atr_multiplier_sl: float = 2.0,
    risk_reward_ratio: float = 1.5,
    min_sl_pct: float = 2.0,
    max_sl_pct: float = 10.0,
    min_tp_pct: float = 5.0,
    max_tp_pct: float = 30.0
) -> Tuple[float, float]:
    """
    Calculate dynamic stop loss and take profit based on ATR and confidence.
    
    Args:
        symbol: Stock symbol
        entry_price: Entry price
        confidence_score: Signal confidence (0-100)
        atr_multiplier_sl: ATR multiplier for stop loss (default 2x)
        risk_reward_ratio: Target risk/reward ratio (default 1.5)
        min_sl_pct: Minimum stop loss % (floor)
        max_sl_pct: Maximum stop loss % (ceiling)
        min_tp_pct: Minimum take profit %
        max_tp_pct: Maximum take profit %
        
    Returns:
        Tuple of (stop_loss_price, take_profit_price)
    """
    # Get ATR
    atr = calculate_atr(symbol)
    
    if atr and atr > 0:
        # ATR-based stop loss
        # Higher confidence = tighter stop (1.5x ATR), lower = wider (2.5x ATR)
        confidence_factor = 1.0 - (confidence_score - 50) / 100  # 0.5 for 100%, 1.0 for 50%
        adjusted_multiplier = atr_multiplier_sl * (0.75 + confidence_factor * 0.5)
        
        sl_distance = atr * adjusted_multiplier
        sl_pct = (sl_distance / entry_price) * 100
        
        # Clamp to min/max
        sl_pct = max(min_sl_pct, min(sl_pct, max_sl_pct))
        
        # Take profit based on risk/reward
        # Higher confidence = higher R:R target (up to 2.5x)
        rr_boost = (confidence_score - 50) / 50 * 0.5  # 0 to 0.5 boost
        adjusted_rr = risk_reward_ratio + rr_boost
        
        tp_pct = sl_pct * adjusted_rr
        tp_pct = max(min_tp_pct, min(tp_pct, max_tp_pct))
        
        logger.info(
            f"{symbol}: ATR={atr:.2f}, SL={sl_pct:.1f}% (conf={confidence_score:.0f}), "
            f"TP={tp_pct:.1f}% (R:R={adjusted_rr:.1f})"
        )
    else:
        # Fallback to confidence-based stops
        # Higher confidence = tighter stop, higher target
        base_sl = 5.0
        base_tp = 10.0
        
        # Adjust based on confidence
        conf_factor = (confidence_score - 50) / 50  # -1 to 1
        sl_pct = base_sl * (1 - conf_factor * 0.3)  # 3.5% to 6.5%
        tp_pct = base_tp * (1 + conf_factor * 0.5)  # 5% to 15%
        
        sl_pct = max(min_sl_pct, min(sl_pct, max_sl_pct))
        tp_pct = max(min_tp_pct, min(tp_pct, max_tp_pct))
        
        logger.info(
            f"{symbol}: No ATR, using confidence-based stops: SL={sl_pct:.1f}%, TP={tp_pct:.1f}%"
        )
    
    stop_loss = entry_price * (1 - sl_pct / 100)
    take_profit = entry_price * (1 + tp_pct / 100)
    
    return stop_loss, take_profit


def calculate_position_size_from_risk(
    capital: float,
    entry_price: float,
    stop_loss: float,
    risk_per_trade_pct: float = 1.0,
    max_position_pct: float = 10.0
) -> int:
    """
    Calculate position size based on risk management.
    
    Args:
        capital: Total capital
        entry_price: Entry price
        stop_loss: Stop loss price
        risk_per_trade_pct: Max risk per trade as % of capital (default 1%)
        max_position_pct: Max position size as % of capital (default 10%)
        
    Returns:
        Number of shares to buy
    """
    # Risk per share
    risk_per_share = entry_price - stop_loss
    if risk_per_share <= 0:
        return 0
    
    # Max dollar risk
    max_risk = capital * (risk_per_trade_pct / 100)
    
    # Shares based on risk
    shares_from_risk = int(max_risk / risk_per_share)
    
    # Shares based on max position size
    max_position_value = capital * (max_position_pct / 100)
    shares_from_position = int(max_position_value / entry_price)
    
    # Take the smaller of the two
    shares = min(shares_from_risk, shares_from_position)
    
    logger.debug(
        f"Position sizing: risk={risk_per_share:.2f}/share, "
        f"from_risk={shares_from_risk}, from_position={shares_from_position}, final={shares}"
    )
    
    return max(1, shares)  # At least 1 share


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with a few symbols
    for symbol, price, score in [("AAPL", 185.0, 70), ("NVDA", 800.0, 85), ("TOST", 42.0, 57)]:
        sl, tp = calculate_dynamic_stops(symbol, price, score)
        print(f"{symbol} @ {price}: SL={sl:.2f}, TP={tp:.2f}")
