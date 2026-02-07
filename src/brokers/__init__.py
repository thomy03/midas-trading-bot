"""
Brokers Module - Midas Trading Bot
===================================
Broker integrations for live and paper trading.
"""

from .paper_trader import (
    PaperTrader,
    PaperTraderContext,
    PaperTradingConfig,
    OrderType,
    OrderSide,
    OrderResult,
    PositionInfo,
)

# Only import IB if available
try:
    from .ib_broker import (
        IBBroker,
        IBBrokerContext,
        IBConfig,
    )
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    IBBroker = None
    IBBrokerContext = None
    IBConfig = None


def get_broker(mode: str = "paper", config=None):
    """
    Factory function to get the appropriate broker
    
    Args:
        mode: "paper" or "live"
        config: Broker configuration (optional)
        
    Returns:
        Broker instance
    """
    if mode == "live":
        if not IB_AVAILABLE:
            raise ImportError("ib_insync required for live trading. Run: pip install ib_insync")
        return IBBroker(config)
    else:
        return PaperTrader(config)


__all__ = [
    'PaperTrader',
    'PaperTraderContext',
    'PaperTradingConfig',
    'IBBroker',
    'IBBrokerContext',
    'IBConfig',
    'OrderType',
    'OrderSide',
    'OrderResult',
    'PositionInfo',
    'get_broker',
    'IB_AVAILABLE',
]
