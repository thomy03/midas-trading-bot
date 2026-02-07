"""
Interactive Brokers Integration - Midas Trading Bot
====================================================
Client for connecting to IB TWS/Gateway via ib_insync

Features:
- Connection management with auto-reconnect
- Order placement (market, limit, stop)
- Position tracking
- Account info and P&L
- Paper trading mode support
"""

import os
import asyncio
import logging
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

try:
    from ib_insync import IB, Stock, Forex, Contract, Order, Trade, Position
    from ib_insync import MarketOrder, LimitOrder, StopOrder, StopLimitOrder
    from ib_insync.util import isNan
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    logging.warning("ib_insync not installed. Run: pip install ib_insync")

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class IBConfig:
    """Interactive Brokers connection configuration"""
    host: str = "127.0.0.1"
    port: int = 7497  # 7497 for TWS paper, 7496 for TWS live, 4001/4002 for Gateway
    client_id: int = 1
    timeout: int = 30
    readonly: bool = False
    account: str = ""  # Leave empty to use default account
    
    @classmethod
    def from_env(cls) -> 'IBConfig':
        """Load configuration from environment variables"""
        return cls(
            host=os.getenv("IB_HOST", "127.0.0.1"),
            port=int(os.getenv("IB_PORT", "7497")),
            client_id=int(os.getenv("IB_CLIENT_ID", "1")),
            timeout=int(os.getenv("IB_TIMEOUT", "30")),
            readonly=os.getenv("IB_READONLY", "false").lower() == "true",
            account=os.getenv("IB_ACCOUNT", ""),
        )


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


class IBBroker:
    """
    Interactive Brokers broker client
    
    Usage:
        broker = IBBroker()
        await broker.connect()
        
        # Place order
        result = await broker.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)
        
        # Get positions
        positions = await broker.get_positions()
        
        await broker.disconnect()
    """
    
    def __init__(self, config: IBConfig = None):
        if not IB_AVAILABLE:
            raise ImportError("ib_insync is required. Install with: pip install ib_insync")
            
        self.config = config or IBConfig.from_env()
        self.ib = IB()
        self._connected = False
        self._callbacks: Dict[str, List[Callable]] = {
            'order_status': [],
            'position_update': [],
            'error': [],
        }
        
    @property
    def is_connected(self) -> bool:
        return self._connected and self.ib.isConnected()
    
    async def connect(self) -> bool:
        """Connect to IB TWS/Gateway"""
        try:
            logger.info(f"Connecting to IB at {self.config.host}:{self.config.port}")
            
            await self.ib.connectAsync(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout,
                readonly=self.config.readonly,
            )
            
            self._connected = True
            self._setup_event_handlers()
            
            # Get account info
            accounts = self.ib.managedAccounts()
            logger.info(f"Connected to IB. Accounts: {accounts}")
            
            if self.config.account and self.config.account not in accounts:
                logger.warning(f"Configured account {self.config.account} not found in {accounts}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from IB"""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")
    
    def _setup_event_handlers(self):
        """Setup IB event handlers"""
        self.ib.errorEvent += self._on_error
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.positionEvent += self._on_position
        
    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Contract):
        """Handle IB errors"""
        # Filter out non-critical messages
        if errorCode in [2104, 2106, 2158]:  # Market data farm connected messages
            return
            
        logger.warning(f"IB Error {errorCode}: {errorString}")
        for callback in self._callbacks['error']:
            callback(reqId, errorCode, errorString)
    
    def _on_order_status(self, trade: Trade):
        """Handle order status updates"""
        logger.info(f"Order {trade.order.orderId} status: {trade.orderStatus.status}")
        for callback in self._callbacks['order_status']:
            callback(trade)
    
    def _on_position(self, position: Position):
        """Handle position updates"""
        for callback in self._callbacks['position_update']:
            callback(position)
    
    def create_stock_contract(self, symbol: str, exchange: str = "SMART", currency: str = "USD") -> Contract:
        """Create a stock contract"""
        return Stock(symbol, exchange, currency)
    
    def create_forex_contract(self, pair: str) -> Contract:
        """Create a forex contract (e.g., 'EURUSD')"""
        return Forex(pair)
    
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
        Place an order
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            side: BUY or SELL
            quantity: Number of shares/units
            order_type: MARKET, LIMIT, STOP, or STOP_LIMIT
            limit_price: Price for limit orders
            stop_price: Trigger price for stop orders
            time_in_force: GTC (Good Till Cancel), DAY, etc.
            wait_for_fill: Wait for order to fill before returning
            timeout: Max seconds to wait for fill
            
        Returns:
            OrderResult with order details
        """
        if not self.is_connected:
            return OrderResult(success=False, message="Not connected to IB")
        
        try:
            # Create contract
            contract = self.create_stock_contract(symbol)
            await self.ib.qualifyContractsAsync(contract)
            
            # Create order based on type
            if order_type == OrderType.MARKET:
                order = MarketOrder(side.value, quantity)
            elif order_type == OrderType.LIMIT:
                if limit_price is None:
                    return OrderResult(success=False, message="Limit price required for LIMIT orders")
                order = LimitOrder(side.value, quantity, limit_price)
            elif order_type == OrderType.STOP:
                if stop_price is None:
                    return OrderResult(success=False, message="Stop price required for STOP orders")
                order = StopOrder(side.value, quantity, stop_price)
            elif order_type == OrderType.STOP_LIMIT:
                if stop_price is None or limit_price is None:
                    return OrderResult(success=False, message="Both stop and limit prices required")
                order = StopLimitOrder(side.value, quantity, limit_price, stop_price)
            else:
                return OrderResult(success=False, message=f"Unknown order type: {order_type}")
            
            order.tif = time_in_force
            
            # Place the order
            trade = self.ib.placeOrder(contract, order)
            logger.info(f"Placed {side.value} order for {quantity} {symbol}: {order}")
            
            if wait_for_fill:
                # Wait for fill with timeout
                start_time = datetime.now()
                while (datetime.now() - start_time).total_seconds() < timeout:
                    if trade.isDone():
                        break
                    await asyncio.sleep(0.5)
                    self.ib.sleep(0)  # Process IB events
                
                if trade.orderStatus.status == 'Filled':
                    return OrderResult(
                        success=True,
                        order_id=trade.order.orderId,
                        message="Order filled",
                        filled_price=trade.orderStatus.avgFillPrice,
                        filled_quantity=trade.orderStatus.filled,
                        status=trade.orderStatus.status
                    )
                else:
                    return OrderResult(
                        success=False,
                        order_id=trade.order.orderId,
                        message=f"Order not filled: {trade.orderStatus.status}",
                        status=trade.orderStatus.status
                    )
            else:
                return OrderResult(
                    success=True,
                    order_id=trade.order.orderId,
                    message="Order submitted",
                    status="Submitted"
                )
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return OrderResult(success=False, message=str(e))
    
    async def cancel_order(self, order_id: int) -> bool:
        """Cancel an open order"""
        if not self.is_connected:
            return False
            
        try:
            # Find the order
            for trade in self.ib.openTrades():
                if trade.order.orderId == order_id:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Cancelled order {order_id}")
                    return True
            
            logger.warning(f"Order {order_id} not found in open trades")
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def get_positions(self) -> List[PositionInfo]:
        """Get all current positions"""
        if not self.is_connected:
            return []
        
        positions = []
        self.ib.reqPositions()
        await asyncio.sleep(1)  # Wait for position data
        
        for pos in self.ib.positions():
            if pos.position != 0:
                # Get market price
                contract = pos.contract
                await self.ib.qualifyContractsAsync(contract)
                
                ticker = self.ib.reqMktData(contract, '', False, False)
                await asyncio.sleep(0.5)
                
                market_price = ticker.last if ticker.last and not isNan(ticker.last) else pos.avgCost
                market_value = pos.position * market_price
                unrealized_pnl = market_value - (pos.position * pos.avgCost)
                
                positions.append(PositionInfo(
                    symbol=pos.contract.symbol,
                    quantity=pos.position,
                    avg_cost=pos.avgCost,
                    market_price=market_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=0  # Need to track separately
                ))
                
                self.ib.cancelMktData(contract)
        
        return positions
    
    async def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary information"""
        if not self.is_connected:
            return {}
        
        summary = {}
        
        # Request account values
        self.ib.reqAccountSummary()
        await asyncio.sleep(1)
        
        for item in self.ib.accountSummary():
            summary[item.tag] = {
                'value': item.value,
                'currency': item.currency
            }
        
        return summary
    
    async def get_buying_power(self) -> float:
        """Get available buying power"""
        summary = await self.get_account_summary()
        if 'BuyingPower' in summary:
            return float(summary['BuyingPower']['value'])
        return 0.0
    
    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders"""
        if not self.is_connected:
            return []
        
        orders = []
        for trade in self.ib.openTrades():
            orders.append({
                'order_id': trade.order.orderId,
                'symbol': trade.contract.symbol,
                'side': trade.order.action,
                'quantity': trade.order.totalQuantity,
                'order_type': trade.order.orderType,
                'limit_price': trade.order.lmtPrice,
                'stop_price': trade.order.auxPrice,
                'status': trade.orderStatus.status,
                'filled': trade.orderStatus.filled,
            })
        
        return orders
    
    def on_order_status(self, callback: Callable):
        """Register callback for order status updates"""
        self._callbacks['order_status'].append(callback)
    
    def on_position_update(self, callback: Callable):
        """Register callback for position updates"""
        self._callbacks['position_update'].append(callback)
    
    def on_error(self, callback: Callable):
        """Register callback for errors"""
        self._callbacks['error'].append(callback)


# Async context manager support
class IBBrokerContext:
    """Context manager for IBBroker"""
    
    def __init__(self, config: IBConfig = None):
        self.broker = IBBroker(config)
    
    async def __aenter__(self) -> IBBroker:
        await self.broker.connect()
        return self.broker
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.broker.disconnect()


# Example usage
async def example_usage():
    """Example of how to use the IB broker"""
    
    config = IBConfig.from_env()
    
    async with IBBrokerContext(config) as broker:
        # Check connection
        print(f"Connected: {broker.is_connected}")
        
        # Get account info
        summary = await broker.get_account_summary()
        print(f"Buying Power: {summary.get('BuyingPower', {}).get('value', 'N/A')}")
        
        # Get positions
        positions = await broker.get_positions()
        for pos in positions:
            print(f"{pos.symbol}: {pos.quantity} @ {pos.avg_cost:.2f}")
        
        # Place a test order (commented out for safety)
        # result = await broker.place_order(
        #     symbol="AAPL",
        #     side=OrderSide.BUY,
        #     quantity=1,
        #     order_type=OrderType.LIMIT,
        #     limit_price=150.00
        # )
        # print(f"Order result: {result}")


if __name__ == "__main__":
    asyncio.run(example_usage())
