"""
IBKR Watchdog: Automatic reconnection and kill switch

TWS/IB Gateway often crashes or restarts, especially overnight.
This watchdog monitors the connection and automatically reconnects.

Features:
- Automatic reconnection with exponential backoff
- Kill switch on connection failure (5 consecutive failures)
- Daily loss limit monitoring
- Heartbeat logging

Usage:
    from ib_insync import IB

    ib = IB()
    watchdog = IBKRWatchdog(ib)
    await watchdog.start()

    # ... your trading logic ...

    await watchdog.stop()
"""

import asyncio
import logging
from datetime import datetime, time
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass

try:
    from ib_insync import IB
except ImportError:
    IB = None

logger = logging.getLogger(__name__)


@dataclass
class WatchdogConfig:
    """Configuration for IBKR Watchdog"""
    # Connection settings
    host: str = '127.0.0.1'
    port: int = 7497  # 7497 = Paper, 7496 = Live
    client_id: int = 1

    # Reconnection settings
    check_interval: float = 10.0  # Check every 10 seconds
    max_reconnect_attempts: int = 5
    initial_reconnect_delay: float = 2.0  # Exponential backoff start
    max_reconnect_delay: float = 60.0

    # Kill switch settings
    max_daily_loss_pct: float = 0.03  # 3% max daily loss
    max_consecutive_failures: int = 5

    # Trading hours (Eastern Time)
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)


class IBKRWatchdog:
    """
    Monitors IBKR connection and handles automatic reconnection.

    Also implements kill switch for risk management:
    - Disconnection protection
    - Daily loss limit
    - Consecutive failure limit
    """

    def __init__(
        self,
        ib: Optional["IB"] = None,
        config: Optional[WatchdogConfig] = None,
        on_disconnect: Optional[Callable[[], Awaitable[None]]] = None,
        on_reconnect: Optional[Callable[[], Awaitable[None]]] = None,
        on_kill_switch: Optional[Callable[[str], Awaitable[None]]] = None
    ):
        """
        Initialize IBKR Watchdog.

        Args:
            ib: IB-insync client instance
            config: Watchdog configuration
            on_disconnect: Callback when disconnected
            on_reconnect: Callback when reconnected
            on_kill_switch: Callback when kill switch triggered
        """
        if IB is None:
            raise ImportError("ib_insync is required. Install with: pip install ib_insync")

        self.ib = ib or IB()
        self.config = config or WatchdogConfig()

        # Callbacks
        self.on_disconnect = on_disconnect
        self.on_reconnect = on_reconnect
        self.on_kill_switch = on_kill_switch

        # State
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._consecutive_failures = 0
        self._last_connected = datetime.now()
        self._daily_pnl = 0.0
        self._kill_switch_triggered = False

    @property
    def is_running(self) -> bool:
        """Check if watchdog is running."""
        return self._running

    @property
    def is_connected(self) -> bool:
        """Check if IBKR is connected."""
        return self.ib.isConnected() if self.ib else False

    @property
    def kill_switch_active(self) -> bool:
        """Check if kill switch has been triggered."""
        return self._kill_switch_triggered

    async def start(self):
        """Start the watchdog monitoring loop."""
        if self._running:
            logger.warning("Watchdog already running")
            return

        # Initial connection if not connected
        if not self.is_connected:
            await self._connect()

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("IBKR Watchdog started")

    async def stop(self):
        """Stop the watchdog."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("IBKR Watchdog stopped")

    async def _monitor_loop(self):
        """Main monitoring loop - checks connection periodically."""
        while self._running:
            try:
                await self._check_connection()
                await self._check_daily_loss()
                await asyncio.sleep(self.config.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                await asyncio.sleep(self.config.check_interval)

    async def _check_connection(self):
        """Check IBKR connection and reconnect if needed."""
        if self._kill_switch_triggered:
            return

        if not self.is_connected:
            logger.warning("IBKR disconnected! Attempting reconnection...")

            if self.on_disconnect:
                try:
                    await self.on_disconnect()
                except Exception as e:
                    logger.error(f"on_disconnect callback error: {e}")

            await self._reconnect()
        else:
            # Connection OK - reset failure counter
            if self._consecutive_failures > 0:
                logger.info("Connection restored")
                self._consecutive_failures = 0
                self._last_connected = datetime.now()

    async def _connect(self):
        """Initial connection to IBKR."""
        try:
            await self.ib.connectAsync(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id
            )
            logger.info(f"Connected to IBKR at {self.config.host}:{self.config.port}")
            self._last_connected = datetime.now()
            return True

        except Exception as e:
            logger.error(f"Initial connection failed: {e}")
            return False

    async def _reconnect(self):
        """Attempt to reconnect with exponential backoff."""
        for attempt in range(self.config.max_reconnect_attempts):
            try:
                # Calculate delay with exponential backoff
                delay = min(
                    self.config.initial_reconnect_delay * (2 ** attempt),
                    self.config.max_reconnect_delay
                )

                logger.info(f"Reconnection attempt {attempt + 1}/{self.config.max_reconnect_attempts}...")

                # Disconnect first if partially connected
                if self.ib.isConnected():
                    self.ib.disconnect()
                    await asyncio.sleep(1)

                # Try to connect
                await self.ib.connectAsync(
                    host=self.config.host,
                    port=self.config.port,
                    clientId=self.config.client_id
                )

                logger.info("Reconnected to IBKR successfully")
                self._consecutive_failures = 0
                self._last_connected = datetime.now()

                if self.on_reconnect:
                    try:
                        await self.on_reconnect()
                    except Exception as e:
                        logger.error(f"on_reconnect callback error: {e}")

                return True

            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")
                self._consecutive_failures += 1

                if attempt < self.config.max_reconnect_attempts - 1:
                    logger.info(f"Waiting {delay:.1f}s before retry...")
                    await asyncio.sleep(delay)

        # All attempts failed - trigger kill switch
        await self._trigger_kill_switch("connection_failure")
        return False

    async def _check_daily_loss(self):
        """Check if daily loss limit has been exceeded."""
        if self._kill_switch_triggered:
            return

        if not self.is_connected:
            return

        try:
            # Get account summary
            account_values = self.ib.accountSummary()

            # Find daily P&L
            for av in account_values:
                if av.tag == 'RealizedPnL':
                    self._daily_pnl = float(av.value) if av.value else 0.0
                    break

            # Get total equity
            total_equity = 0.0
            for av in account_values:
                if av.tag == 'NetLiquidation':
                    total_equity = float(av.value) if av.value else 0.0
                    break

            if total_equity > 0:
                daily_loss_pct = abs(min(0, self._daily_pnl)) / total_equity

                if daily_loss_pct >= self.config.max_daily_loss_pct:
                    logger.critical(
                        f"DAILY LOSS LIMIT EXCEEDED: {daily_loss_pct * 100:.1f}% "
                        f"(limit: {self.config.max_daily_loss_pct * 100:.1f}%)"
                    )
                    await self._trigger_kill_switch("daily_loss_limit")

        except Exception as e:
            logger.debug(f"Could not check daily loss: {e}")

    async def _trigger_kill_switch(self, reason: str):
        """
        Trigger the kill switch - stops all trading.

        Args:
            reason: Reason for triggering ('connection_failure', 'daily_loss_limit', etc.)
        """
        if self._kill_switch_triggered:
            return

        self._kill_switch_triggered = True
        logger.critical(f"KILL SWITCH TRIGGERED: {reason}")

        # Close all positions
        try:
            await self._emergency_close_all()
        except Exception as e:
            logger.error(f"Error closing positions: {e}")

        # Cancel all orders
        try:
            await self._cancel_all_orders()
        except Exception as e:
            logger.error(f"Error canceling orders: {e}")

        # Notify via callback
        if self.on_kill_switch:
            try:
                await self.on_kill_switch(reason)
            except Exception as e:
                logger.error(f"on_kill_switch callback error: {e}")

    async def _emergency_close_all(self):
        """Emergency close all positions."""
        if not self.is_connected:
            logger.error("Cannot close positions - not connected")
            return

        try:
            positions = self.ib.positions()

            for position in positions:
                if position.position != 0:
                    logger.warning(f"Closing position: {position.contract.symbol} x {position.position}")

                    # Create market order to close
                    from ib_insync import MarketOrder

                    action = 'SELL' if position.position > 0 else 'BUY'
                    quantity = abs(position.position)

                    order = MarketOrder(action, quantity)
                    trade = self.ib.placeOrder(position.contract, order)

                    logger.info(f"Close order placed: {trade}")

        except Exception as e:
            logger.error(f"Error in emergency close: {e}")

    async def _cancel_all_orders(self):
        """Cancel all open orders."""
        if not self.is_connected:
            logger.error("Cannot cancel orders - not connected")
            return

        try:
            self.ib.reqGlobalCancel()
            logger.info("All orders canceled")

        except Exception as e:
            logger.error(f"Error canceling orders: {e}")

    def reset_kill_switch(self):
        """
        Reset the kill switch (use with caution).

        Only call this after the issue has been resolved and
        you're ready to resume trading.
        """
        if self._kill_switch_triggered:
            logger.warning("Resetting kill switch - trading can resume")
            self._kill_switch_triggered = False
            self._consecutive_failures = 0


class IBKRKillSwitch:
    """
    Standalone kill switch for use without watchdog.

    Provides quick access to emergency functions.
    """

    def __init__(self, ib: "IB"):
        self.ib = ib
        self._triggered = False

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    async def check(self, daily_pnl_pct: float = 0.0) -> bool:
        """
        Check kill switch conditions.

        Args:
            daily_pnl_pct: Current daily P&L as percentage (negative = loss)

        Returns:
            True if kill switch triggered
        """
        # Check connection
        if not self.ib.isConnected():
            logger.warning("IBKR not connected - kill switch condition met")
            self._triggered = True

        # Check daily loss (3% limit)
        if daily_pnl_pct <= -0.03:
            logger.warning(f"Daily loss {daily_pnl_pct * 100:.1f}% exceeds limit - kill switch triggered")
            self._triggered = True

        return self._triggered

    async def emergency_close_all(self):
        """Close all positions immediately."""
        if not self.ib.isConnected():
            raise ConnectionError("Cannot close - IBKR not connected")

        positions = self.ib.positions()
        for pos in positions:
            if pos.position != 0:
                from ib_insync import MarketOrder
                action = 'SELL' if pos.position > 0 else 'BUY'
                order = MarketOrder(action, abs(pos.position))
                self.ib.placeOrder(pos.contract, order)

    async def notify_critical(self, message: str):
        """Send critical notification."""
        logger.critical(f"CRITICAL: {message}")
        # TODO: Integrate with notification_manager for Telegram/Email
