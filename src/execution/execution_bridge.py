"""
Execution Bridge - Unified interface for PaperTrader and IBKRExecutor.

Allows seamless switching between paper trading and live IBKR execution
via the EXECUTION_MODE environment variable (paper|ibkr).

Created: 2026-02-11
"""

import os
import asyncio
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _get_currency_for_symbol(symbol: str) -> str:
    """Determine currency based on symbol suffix."""
    suffixes_eur = ('.PA', '.DE', '.AS', '.BR', '.MI', '.MC', '.LS')
    suffixes_gbp = ('.L',)
    suffixes_chf = ('.SW',)
    if any(symbol.upper().endswith(s) for s in suffixes_eur):
        return 'EUR'
    if any(symbol.upper().endswith(s) for s in suffixes_gbp):
        return 'GBP'
    if any(symbol.upper().endswith(s) for s in suffixes_chf):
        return 'CHF'
    return 'USD'


def _clean_symbol_for_ibkr(symbol: str) -> str:
    """Convert yfinance symbol to IBKR symbol (strip exchange suffix)."""
    # yfinance: AIR.PA -> IBKR: AIR
    if '.' in symbol:
        return symbol.split('.')[0]
    return symbol


class ExecutionBridge:
    """
    Unified execution interface that delegates to PaperTrader or IBKRExecutor.
    
    Config via env var EXECUTION_MODE=paper|ibkr (default: paper).
    IBKR config via midas_config.yaml or env vars.
    """

    def __init__(self):
        self.mode = os.environ.get('EXECUTION_MODE', 'paper').lower()
        self._backend = None
        self._ibkr_connected = False
        logger.info(f"ExecutionBridge initialized in '{self.mode}' mode")

    def _get_paper_trader(self):
        """Lazy-load PaperTrader backend."""
        if self._backend is None:
            from .paper_trader import get_paper_trader
            self._backend = get_paper_trader()
        return self._backend

    async def _get_ibkr_executor(self):
        """Lazy-load and connect IBKRExecutor backend."""
        if self._backend is None:
            try:
                from .ibkr_executor import IBKRExecutor
                # Read config
                config = self._load_ibkr_config()
                self._backend = IBKRExecutor(
                    host=config.get('host', '127.0.0.1'),
                    port=config.get('port', 4002),
                    client_id=config.get('client_id', 1),
                    readonly=config.get('readonly', False),
                )
                connected = await self._backend.connect(timeout=15)
                self._ibkr_connected = connected
                if not connected:
                    logger.error("Failed to connect to IBKR Gateway - orders will fail")
                else:
                    logger.info("Connected to IBKR Gateway successfully")
            except Exception as e:
                logger.error(f"IBKR initialization failed: {e}")
                self._ibkr_connected = False
        return self._backend

    def _load_ibkr_config(self) -> dict:
        """Load IBKR config from midas_config.yaml."""
        try:
            import yaml
            from pathlib import Path
            config_path = Path("config/midas_config.yaml")
            if config_path.exists():
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                return cfg.get('execution', {}).get('ibkr', {})
        except Exception as e:
            logger.warning(f"Could not load IBKR config: {e}")
        return {}

    # =========================================================================
    # UNIFIED API
    # =========================================================================

    def open_position(
        self,
        symbol: str,
        price: float,
        score: float = 0.0,
        decision_type: str = "BUY",
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        pillar_technical: Optional[float] = None,
        pillar_fundamental: Optional[float] = None,
        pillar_sentiment: Optional[float] = None,
        pillar_news: Optional[float] = None,
        reasoning: Optional[str] = None,
        company_name: Optional[str] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
    ):
        """Open a position. Returns Position object or None."""
        logger.info(f"[BRIDGE:{self.mode}] open_position {symbol} @ {price:.2f} (score={score:.1f})")

        if self.mode == 'ibkr':
            return self._open_position_ibkr(
                symbol, price, score, decision_type,
                stop_loss_pct, take_profit_pct,
                pillar_technical, pillar_fundamental,
                pillar_sentiment, pillar_news,
                reasoning, company_name, sector, industry,
            )

        # Paper mode (default)
        pt = self._get_paper_trader()
        return pt.open_position(
            symbol=symbol, price=price, score=score,
            decision_type=decision_type,
            stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
            pillar_technical=pillar_technical,
            pillar_fundamental=pillar_fundamental,
            pillar_sentiment=pillar_sentiment,
            pillar_news=pillar_news,
            reasoning=reasoning, company_name=company_name,
            sector=sector, industry=industry,
        )

    def _open_position_ibkr(self, symbol, price, score, decision_type,
                             stop_loss_pct, take_profit_pct,
                             pillar_technical, pillar_fundamental,
                             pillar_sentiment, pillar_news,
                             reasoning, company_name, sector, industry):
        """Open position via IBKR (sync wrapper around async)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context - schedule coroutine
                import concurrent.futures
                future = asyncio.ensure_future(self._open_position_ibkr_async(
                    symbol, price, score, decision_type,
                    stop_loss_pct, take_profit_pct,
                    pillar_technical, pillar_fundamental,
                    pillar_sentiment, pillar_news,
                    reasoning, company_name, sector, industry,
                ))
                # Can't await here in sync context; log and also record in paper trader
                logger.info(f"[BRIDGE:ibkr] Order submitted async for {symbol}")
                # Also record in paper trader for tracking
                pt = self._get_paper_trader_for_tracking()
                if pt:
                    return pt.open_position(
                        symbol=symbol, price=price, score=score,
                        decision_type=decision_type,
                        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
                        pillar_technical=pillar_technical,
                        pillar_fundamental=pillar_fundamental,
                        pillar_sentiment=pillar_sentiment,
                        pillar_news=pillar_news,
                        reasoning=reasoning, company_name=company_name,
                        sector=sector, industry=industry,
                    )
                return None
            else:
                return loop.run_until_complete(self._open_position_ibkr_async(
                    symbol, price, score, decision_type,
                    stop_loss_pct, take_profit_pct,
                    pillar_technical, pillar_fundamental,
                    pillar_sentiment, pillar_news,
                    reasoning, company_name, sector, industry,
                ))
        except Exception as e:
            logger.error(f"[BRIDGE:ibkr] open_position failed for {symbol}: {e}")
            return None

    async def _open_position_ibkr_async(self, symbol, price, score, decision_type,
                                         stop_loss_pct, take_profit_pct,
                                         pillar_technical, pillar_fundamental,
                                         pillar_sentiment, pillar_news,
                                         reasoning, company_name, sector, industry):
        """Async IBKR order placement."""
        try:
            from .ibkr_executor import OrderRequest, OrderAction, OrderType
            executor = await self._get_ibkr_executor()
            if not executor or not self._ibkr_connected:
                logger.error(f"[BRIDGE:ibkr] Not connected, cannot open {symbol}")
                return None

            ibkr_symbol = _clean_symbol_for_ibkr(symbol)
            currency = _get_currency_for_symbol(symbol)

            # Calculate quantity (same logic as PaperTrader for consistency)
            from .paper_trader import get_paper_trader
            pt = get_paper_trader()
            # Use paper trader's sizing logic
            from .dynamic_stops import calculate_dynamic_stops, calculate_position_size_from_risk
            try:
                stop_loss, take_profit = calculate_dynamic_stops(symbol, price, score)
            except Exception:
                stop_loss = price * 0.95
                take_profit = price * 1.15

            quantity = calculate_position_size_from_risk(
                capital=pt.get_total_value(),
                entry_price=price,
                stop_loss=stop_loss,
                risk_per_trade_pct=1.0,
                max_position_pct=8.0,
            )
            if quantity < 1:
                logger.warning(f"[BRIDGE:ibkr] Quantity too small for {symbol}")
                return None

            request = OrderRequest(
                symbol=ibkr_symbol,
                action=OrderAction.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal_source='midas_bridge',
                confidence_score=score,
            )

            result = await executor.place_order(request)
            logger.info(f"[BRIDGE:ibkr] Order result for {symbol}: {result.status.value}")

            # Also track in paper trader for unified portfolio view
            pt.open_position(
                symbol=symbol, price=price, score=score,
                decision_type=decision_type,
                pillar_technical=pillar_technical,
                pillar_fundamental=pillar_fundamental,
                pillar_sentiment=pillar_sentiment,
                pillar_news=pillar_news,
                reasoning=reasoning, company_name=company_name,
                sector=sector, industry=industry,
            )

            return result
        except Exception as e:
            logger.error(f"[BRIDGE:ibkr] Async open failed for {symbol}: {e}")
            return None

    def close_position(self, symbol: str, price: float, reason: str = "manual"):
        """Close a position. Returns TradeRecord or None."""
        logger.info(f"[BRIDGE:{self.mode}] close_position {symbol} @ {price:.2f} ({reason})")

        if self.mode == 'ibkr':
            return self._close_position_ibkr(symbol, price, reason)

        pt = self._get_paper_trader()
        return pt.close_position(symbol=symbol, price=price, reason=reason)

    def _close_position_ibkr(self, symbol, price, reason):
        """Close via IBKR."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self._close_position_ibkr_async(symbol, reason))
                # Also close in paper tracker
                from .paper_trader import get_paper_trader
                pt = get_paper_trader()
                return pt.close_position(symbol=symbol, price=price, reason=reason)
            else:
                return loop.run_until_complete(
                    self._close_position_ibkr_async(symbol, reason)
                )
        except Exception as e:
            logger.error(f"[BRIDGE:ibkr] close_position failed for {symbol}: {e}")
            # Fallback: at least close paper position
            from .paper_trader import get_paper_trader
            return get_paper_trader().close_position(symbol=symbol, price=price, reason=reason)

    async def _close_position_ibkr_async(self, symbol, reason):
        """Async IBKR position close."""
        try:
            executor = await self._get_ibkr_executor()
            if not executor or not self._ibkr_connected:
                logger.error(f"[BRIDGE:ibkr] Not connected, cannot close {symbol}")
                return None
            ibkr_symbol = _clean_symbol_for_ibkr(symbol)
            result = await executor.close_position(ibkr_symbol)
            logger.info(f"[BRIDGE:ibkr] Close result for {symbol}: {result.status.value if result else 'None'}")
            return result
        except Exception as e:
            logger.error(f"[BRIDGE:ibkr] Async close failed for {symbol}: {e}")
            return None

    def get_positions(self) -> dict:
        """Get current positions dict {symbol: Position}."""
        pt = self._get_paper_trader()
        return pt.positions

    def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        pt = self._get_paper_trader()
        return pt.get_total_value()

    def update_prices(self):
        """Update all position prices and check stops."""
        pt = self._get_paper_trader()
        return pt.update_prices()

    def save_state(self):
        """Save portfolio state."""
        pt = self._get_paper_trader()
        return pt.save_state()

    def get_portfolio_summary(self) -> dict:
        """Get portfolio summary."""
        pt = self._get_paper_trader()
        return pt.get_portfolio_summary()

    def _get_paper_trader_for_tracking(self):
        """Get paper trader instance for IBKR tracking (separate from main backend)."""
        try:
            from .paper_trader import get_paper_trader
            return get_paper_trader()
        except Exception:
            return None

    @property
    def positions(self):
        """Compatibility: access positions dict directly."""
        return self.get_positions()

    @property
    def cash(self):
        """Compatibility: access cash."""
        pt = self._get_paper_trader()
        return pt.cash

    @property
    def trade_history(self):
        """Compatibility: access trade history."""
        pt = self._get_paper_trader()
        return pt.trade_history

    @property
    def initial_capital(self):
        """Compatibility: access initial capital."""
        pt = self._get_paper_trader()
        return pt.initial_capital


# Singleton
_bridge: Optional[ExecutionBridge] = None


def get_execution_bridge() -> ExecutionBridge:
    """Get singleton ExecutionBridge instance."""
    global _bridge
    if _bridge is None:
        _bridge = ExecutionBridge()
    return _bridge
