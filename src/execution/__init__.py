"""
Execution Module - Exécution des ordres
Connexion aux brokers pour le trading réel.
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

__all__ = [
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
    'PaperTrader',
    'PaperPosition',
    'TradeRecord',
    'get_paper_trader'
]
