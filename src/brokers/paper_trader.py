"""
Paper Trader - Midas Trading Bot
=================================
Simulated trading for backtesting and paper trading mode.

Provides the same interface as IBBroker but executes trades
in a simulated environment with realistic fills.
"""

import os
import json
import asyncio
import logging
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import random

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "Pending"
    SUBMITTED = "Submitted"
    FILLED = "Filled"
    PARTIALLY_FILLED = "PartiallyFilled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"


@dataclass
class PaperOrder:
    """Simulated order"""
    order_id: int
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0
    filled_price: float = 0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'created_at': self.created_at.isoformat(),
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
        }


@dataclass
class PaperPosition:
    """Simulated position"""
    symbol: str
    quantity: float
    avg_cost: float
    realized_pnl: float = 0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OrderResult:
    """Result of an order operation"""
    success: bool
    order_id: Optional[int] = None
    message: str = ""
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    status: str = ""


@dataclass
class PositionInfo:
    """Current position information"""
    symbol: str
    quantity: float
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class PaperTradingConfig:
    """Paper trading configuration"""
    initial_capital: float = 100000.0
    commission_per_trade: float = 1.0  # Fixed commission per trade
    commission_pct: float = 0.0001  # 0.01% per trade
    slippage_pct: float = 0.001  # 0.1% slippage on market orders
    fill_probability: float = 0.98  # 98% chance limit orders fill at price
    state_file: str = "paper_trading_state.json"
    
    @classmethod
    def from_env(cls) -> 'PaperTradingConfig':
        return cls(
            initial_capital=float(os.getenv("PAPER_INITIAL_CAPITAL", "100000")),
            commission_per_trade=float(os.getenv("PAPER_COMMISSION", "1.0")),
            slippage_pct=float(os.getenv("PAPER_SLIPPAGE", "0.001")),
        )


class PriceFeed:
    """Simple price feed for paper trading"""
    
    def __init__(self):
        self._prices: Dict[str, float] = {}
        self._callbacks: List[Callable] = []
    
    def set_price(self, symbol: str, price: float):
        """Set current price for a symbol"""
        self._prices[symbol] = price
        for callback in self._callbacks:
            callback(symbol, price)
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        return self._prices.get(symbol)
    
    def on_price_update(self, callback: Callable):
        """Register callback for price updates"""
        self._callbacks.append(callback)


class PaperTrader:
    """
    Paper trading simulator
    
    Provides the same interface as IBBroker for seamless switching
    between paper and live trading.
    
    Usage:
        trader = PaperTrader()
        await trader.connect()  # Load state
        
        # Set prices (from your data feed)
        trader.price_feed.set_price("AAPL", 150.0)
        
        # Place orders
        result = await trader.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)
        
        # Check positions
        positions = await trader.get_positions()
    """
    
    def __init__(self, config: PaperTradingConfig = None):
        self.config = config or PaperTradingConfig.from_env()
        self.price_feed = PriceFeed()
        
        # State
        self._connected = False
        self._cash = self.config.initial_capital
        self._positions: Dict[str, PaperPosition] = {}
        self._orders: Dict[int, PaperOrder] = {}
        self._order_history: List[PaperOrder] = []
        self._next_order_id = 1
        self._realized_pnl = 0.0
        
        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'order_status': [],
            'position_update': [],
            'error': [],
        }
        
        # Setup price feed callback for stop/limit orders
        self.price_feed.on_price_update(self._check_pending_orders)
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    @property
    def portfolio(self):
        """Compatibility property for live_loop.py"""
        class _PortfolioWrapper:
            def __init__(wrapper_self, positions_dict):
                wrapper_self._positions = positions_dict
            @property
            def positions(wrapper_self):
                return list(wrapper_self._positions.values())
        return _PortfolioWrapper(self._positions)

    
    async def connect(self) -> bool:
        """Connect (load state from file)"""
        self._load_state()
        self._connected = True
        logger.info(f"Paper trader connected. Cash: ${self._cash:,.2f}")
        return True
    
    async def disconnect(self):
        """Disconnect (save state to file)"""
        self._save_state()
        self._connected = False
        logger.info("Paper trader disconnected")
    
    def _load_state(self):
        """Load state from file"""
        state_path = Path(self.config.state_file)
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                self._cash = state.get('cash', self.config.initial_capital)
                self._next_order_id = state.get('next_order_id', 1)
                self._realized_pnl = state.get('realized_pnl', 0.0)
                
                # Load positions
                for symbol, pos_data in state.get('positions', {}).items():
                    self._positions[symbol] = PaperPosition(
                        symbol=pos_data['symbol'],
                        quantity=pos_data['quantity'],
                        avg_cost=pos_data['avg_cost'],
                        realized_pnl=pos_data.get('realized_pnl', 0)
                    )
                
                logger.info(f"Loaded paper trading state from {state_path}")
                
            except Exception as e:
                logger.error(f"Error loading state: {e}")
    
    def _save_state(self):
        """Save state to file"""
        state = {
            'cash': self._cash,
            'next_order_id': self._next_order_id,
            'realized_pnl': self._realized_pnl,
            'positions': {s: p.to_dict() for s, p in self._positions.items()},
            'saved_at': datetime.now().isoformat(),
        }
        
        try:
            with open(self.config.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Saved paper trading state to {self.config.state_file}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply slippage to a price"""
        slippage = price * self.config.slippage_pct
        if side == OrderSide.BUY:
            return price + slippage
        else:
            return price - slippage
    
    def _calculate_commission(self, value: float) -> float:
        """Calculate commission for a trade"""
        return self.config.commission_per_trade + (value * self.config.commission_pct)
    
    def _check_pending_orders(self, symbol: str, price: float):
        """Check if any pending orders should be filled"""
        for order_id, order in list(self._orders.items()):
            if order.symbol != symbol or order.status != OrderStatus.SUBMITTED:
                continue
            
            should_fill = False
            fill_price = price
            
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and price <= order.limit_price:
                    should_fill = True
                    fill_price = order.limit_price
                elif order.side == OrderSide.SELL and price >= order.limit_price:
                    should_fill = True
                    fill_price = order.limit_price
                    
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and price >= order.stop_price:
                    should_fill = True
                    fill_price = self._apply_slippage(price, order.side)
                elif order.side == OrderSide.SELL and price <= order.stop_price:
                    should_fill = True
                    fill_price = self._apply_slippage(price, order.side)
            
            if should_fill:
                self._fill_order(order, fill_price)
    
    def _fill_order(self, order: PaperOrder, fill_price: float):
        """Fill an order"""
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.filled_at = datetime.now()
        
        # Calculate trade value and commission
        trade_value = fill_price * order.quantity
        commission = self._calculate_commission(trade_value)
        
        # Update position
        if order.side == OrderSide.BUY:
            self._cash -= (trade_value + commission)
            self._update_position_buy(order.symbol, order.quantity, fill_price)
        else:
            self._cash += (trade_value - commission)
            pnl = self._update_position_sell(order.symbol, order.quantity, fill_price)
            self._realized_pnl += pnl
        
        # Move to history
        self._order_history.append(order)
        if order.order_id in self._orders:
            del self._orders[order.order_id]
        
        # Trigger callbacks
        for callback in self._callbacks['order_status']:
            callback(order)
        
        logger.info(f"Filled order {order.order_id}: {order.side.value} {order.quantity} {order.symbol} @ {fill_price:.2f}")
    
    def _update_position_buy(self, symbol: str, quantity: float, price: float):
        """Update position after a buy"""
        if symbol in self._positions:
            pos = self._positions[symbol]
            total_cost = (pos.quantity * pos.avg_cost) + (quantity * price)
            total_qty = pos.quantity + quantity
            pos.avg_cost = total_cost / total_qty
            pos.quantity = total_qty
        else:
            self._positions[symbol] = PaperPosition(
                symbol=symbol,
                quantity=quantity,
                avg_cost=price
            )
    
    def _update_position_sell(self, symbol: str, quantity: float, price: float) -> float:
        """Update position after a sell, returns realized P&L"""
        if symbol not in self._positions:
            logger.warning(f"Selling {symbol} but no position exists")
            return 0.0
        
        pos = self._positions[symbol]
        pnl = (price - pos.avg_cost) * quantity
        pos.quantity -= quantity
        pos.realized_pnl += pnl
        
        # Remove position if fully closed
        if pos.quantity <= 0:
            del self._positions[symbol]
        
        return pnl
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float = None,
        stop_price: float = None,
        time_in_force: str = "GTC",
        wait_for_fill: bool = True,
        timeout: float = 60.0
    ) -> OrderResult:
        """
        Place an order (paper trading)
        
        Same interface as IBBroker.place_order
        """
        if not self.is_connected:
            return OrderResult(success=False, message="Not connected")
        
        # Validate order
        current_price = self.price_feed.get_price(symbol)
        if current_price is None and order_type == OrderType.MARKET:
            return OrderResult(success=False, message=f"No price available for {symbol}")
        
        # Check buying power
        if side == OrderSide.BUY:
            estimated_cost = (limit_price or current_price or 0) * quantity
            if estimated_cost > self._cash:
                return OrderResult(
                    success=False, 
                    message=f"Insufficient funds. Need ${estimated_cost:.2f}, have ${self._cash:.2f}"
                )
        else:
            # Check if we have position to sell
            pos = self._positions.get(symbol)
            if not pos or pos.quantity < quantity:
                return OrderResult(
                    success=False,
                    message=f"Insufficient position. Have {pos.quantity if pos else 0}, need {quantity}"
                )
        
        # Create order
        order = PaperOrder(
            order_id=self._next_order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            status=OrderStatus.SUBMITTED
        )
        self._next_order_id += 1
        self._orders[order.order_id] = order
        
        logger.info(f"Placed paper order: {order.side.value} {order.quantity} {order.symbol} ({order.order_type.value})")
        
        # Market orders fill immediately
        if order_type == OrderType.MARKET and current_price:
            fill_price = self._apply_slippage(current_price, side)
            self._fill_order(order, fill_price)
            
            return OrderResult(
                success=True,
                order_id=order.order_id,
                message="Order filled",
                filled_price=order.filled_price,
                filled_quantity=order.filled_quantity,
                status=order.status.value
            )
        
        # Limit/Stop orders are pending
        return OrderResult(
            success=True,
            order_id=order.order_id,
            message="Order submitted",
            status=order.status.value
        )
    
    async def cancel_order(self, order_id: int) -> bool:
        """Cancel an open order"""
        if order_id in self._orders:
            order = self._orders[order_id]
            order.status = OrderStatus.CANCELLED
            self._order_history.append(order)
            del self._orders[order_id]
            logger.info(f"Cancelled order {order_id}")
            return True
        return False
    
    async def get_positions(self) -> List[PositionInfo]:
        """Get all current positions"""
        positions = []
        
        for symbol, pos in self._positions.items():
            market_price = self.price_feed.get_price(symbol) or pos.avg_cost
            market_value = pos.quantity * market_price
            unrealized_pnl = (market_price - pos.avg_cost) * pos.quantity
            
            positions.append(PositionInfo(
                symbol=symbol,
                quantity=pos.quantity,
                avg_cost=pos.avg_cost,
                market_price=market_price,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=pos.realized_pnl
            ))
        
        return positions
    
    async def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary"""
        positions = await self.get_positions()
        total_value = self._cash + sum(p.market_value for p in positions)
        total_unrealized = sum(p.unrealized_pnl for p in positions)
        
        return {
            'Cash': {'value': self._cash, 'currency': 'USD'},
            'TotalCashValue': {'value': self._cash, 'currency': 'USD'},
            'NetLiquidation': {'value': total_value, 'currency': 'USD'},
            'UnrealizedPnL': {'value': total_unrealized, 'currency': 'USD'},
            'RealizedPnL': {'value': self._realized_pnl, 'currency': 'USD'},
            'BuyingPower': {'value': self._cash, 'currency': 'USD'},
            'InitialCapital': {'value': self.config.initial_capital, 'currency': 'USD'},
            'TotalReturn': {'value': (total_value / self.config.initial_capital - 1) * 100, 'currency': '%'},
        }
    
    async def get_buying_power(self) -> float:
        """Get available buying power"""
        return self._cash
    
    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders"""
        return [order.to_dict() for order in self._orders.values()]
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get all completed trades"""
        return [
            order.to_dict() 
            for order in self._order_history 
            if order.status == OrderStatus.FILLED
        ]
    
    def on_order_status(self, callback: Callable):
        """Register callback for order status updates"""
        self._callbacks['order_status'].append(callback)
    
    def on_position_update(self, callback: Callable):
        """Register callback for position updates"""
        self._callbacks['position_update'].append(callback)
    
    def reset(self):
        """Reset to initial state"""
        self._cash = self.config.initial_capital
        self._positions.clear()
        self._orders.clear()
        self._order_history.clear()
        self._realized_pnl = 0.0
        self._next_order_id = 1
        logger.info("Paper trader reset to initial state")


# Context manager
class PaperTraderContext:
    """Context manager for PaperTrader"""
    
    def __init__(self, config: PaperTradingConfig = None):
        self.trader = PaperTrader(config)
    
    async def __aenter__(self) -> PaperTrader:
        await self.trader.connect()
        return self.trader
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.trader.disconnect()


# Example usage
async def example_usage():
    """Example paper trading session"""
    
    async with PaperTraderContext() as trader:
        # Set some prices
        trader.price_feed.set_price("AAPL", 150.0)
        trader.price_feed.set_price("GOOGL", 2800.0)
        
        # Place market order
        result = await trader.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET
        )
        print(f"Order result: {result}")
        
        # Check positions
        positions = await trader.get_positions()
        for pos in positions:
            print(f"Position: {pos.symbol} {pos.quantity} @ {pos.avg_cost:.2f}")
        
        # Check account
        summary = await trader.get_account_summary()
        print(f"Net Liquidation: ${summary['NetLiquidation']['value']:,.2f}")
        
        # Price update - simulate some profit
        trader.price_feed.set_price("AAPL", 155.0)
        
        # Sell
        result = await trader.place_order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.MARKET
        )
        
        # Check final state
        summary = await trader.get_account_summary()
        print(f"Realized P&L: ${summary['RealizedPnL']['value']:,.2f}")


if __name__ == "__main__":
    asyncio.run(example_usage())

# Compatibility property for live_loop.py
class PortfolioWrapper:
    def __init__(self, trader):
        self._trader = trader
    
    @property
    def positions(self):
        return list(self._trader._positions.values())

# Add to PaperTrader class
PaperTrader.portfolio = property(lambda self: PortfolioWrapper(self))
