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
    'close_ibkr_executor'
]
