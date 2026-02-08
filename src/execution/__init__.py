"""
Execution Module - Order execution & portfolio risk management.
Connexion aux brokers pour le trading réel + risk controls.
"""

from .ibkr_executor import (
    IBKRExecutor,
    MockIBKRExecutor,
    OrderRequest,
    OrderResult,
    OrderType,
    OrderAction,
    OrderStatus,
    Position,
    AccountInfo,
    get_ibkr_executor,
    close_ibkr_executor
)

from .paper_trader import (
    PaperTrader,
    Position as PaperPosition,
    TradeRecord,
    get_paper_trader
)

from .defensive_manager import (
    DefensiveManager,
    DefensiveLevel,
    DefensiveConfig,
    get_defensive_manager
)

from .correlation_manager import (
    CorrelationManager,
    CorrelationConfig,
    PortfolioPosition,
    get_correlation_manager
)

from .position_sizer import (
    PositionSizer,
    SizingResult,
    get_position_sizer,
)

__all__ = [
    # Broker execution
    'IBKRExecutor',
    'MockIBKRExecutor',
    'OrderRequest',
    'OrderResult',
    'OrderType',
    'OrderAction',
    'OrderStatus',
    'Position',
    'AccountInfo',
    'get_ibkr_executor',
    'close_ibkr_executor',
    # Paper trading
    'PaperTrader',
    'PaperPosition',
    'TradeRecord',
    'get_paper_trader',
    # V6.2 Risk management
    'DefensiveManager',
    'DefensiveLevel',
    'DefensiveConfig',
    'get_defensive_manager',
    'CorrelationManager',
    'CorrelationConfig',
    'PortfolioPosition',
    'get_correlation_manager',
    # V7 Position sizing
    'PositionSizer',
    'SizingResult',
    'get_position_sizer',
]
