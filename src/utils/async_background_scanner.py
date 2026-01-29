"""
Async Background Scanner - 100% asyncio compatible with ib_insync

Migrated from background_scanner.py to work properly with IBKR's asyncio-native API.
The original threading-based scanner causes issues when mixed with ib_insync.

Features:
- Full async/await support
- Concurrent scanning with asyncio.Semaphore (rate limiting)
- Compatible with ib_insync and FMP client
- State persistence across restarts
- Pause/Resume/Cancel support

Usage:
    from src.utils.async_background_scanner import AsyncBackgroundScanner

    scanner = AsyncBackgroundScanner()

    async def screen_stock(symbol: str, name: str) -> Optional[Dict]:
        # Your screening logic here
        return {"symbol": symbol, "signal": "BUY"}

    await scanner.start_scan(stocks, screen_stock)

    # Check progress
    state = await scanner.get_state()
    print(f"Progress: {state.completed_count}/{state.total_stocks}")
"""

import asyncio
import json
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Awaitable, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# State file location
STATE_FILE = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / 'data' / 'async_scan_state.json'


class ScanStatus(Enum):
    """Scan status enumeration"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class ScanState:
    """Scan state data class"""
    status: ScanStatus = ScanStatus.IDLE
    total_stocks: int = 0
    completed_count: int = 0
    alerts: List[Dict] = field(default_factory=list)
    pending_stocks: List[Dict] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    current_symbol: str = ""
    scan_id: int = 0
    pause_requested: bool = False
    cancel_requested: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'status': self.status.value,
            'total_stocks': self.total_stocks,
            'completed_count': self.completed_count,
            'alerts': self.alerts,
            'pending_stocks': self.pending_stocks,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error_message': self.error_message,
            'current_symbol': self.current_symbol,
            'scan_id': self.scan_id,
            'pause_requested': self.pause_requested,
            'cancel_requested': self.cancel_requested
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ScanState':
        """Create from dictionary"""
        status_str = data.get('status', 'idle')
        try:
            status = ScanStatus(status_str)
        except ValueError:
            status = ScanStatus.IDLE

        start_time = None
        if data.get('start_time'):
            try:
                start_time = datetime.fromisoformat(data['start_time'])
            except (ValueError, TypeError):
                pass

        end_time = None
        if data.get('end_time'):
            try:
                end_time = datetime.fromisoformat(data['end_time'])
            except (ValueError, TypeError):
                pass

        return cls(
            status=status,
            total_stocks=data.get('total_stocks', 0),
            completed_count=data.get('completed_count', 0),
            alerts=data.get('alerts', []),
            pending_stocks=data.get('pending_stocks', []),
            start_time=start_time,
            end_time=end_time,
            error_message=data.get('error_message'),
            current_symbol=data.get('current_symbol', ''),
            scan_id=data.get('scan_id', 0),
            pause_requested=data.get('pause_requested', False),
            cancel_requested=data.get('cancel_requested', False)
        )


class AsyncBackgroundScanner:
    """
    Fully async background scanner for market screening.

    Designed for use with ib_insync and other async APIs.

    Usage:
        scanner = AsyncBackgroundScanner(max_concurrent=20)

        # Start scan
        await scanner.start_scan(stocks, screen_function)

        # Monitor progress
        while scanner.is_running:
            state = await scanner.get_state()
            print(f"{state.completed_count}/{state.total_stocks}")
            await asyncio.sleep(1)

        # Get results
        final = await scanner.get_state()
        alerts = final.alerts
    """

    def __init__(
        self,
        max_concurrent: int = 20,
        rate_limit_delay: float = 0.1,
        state_file: Path = STATE_FILE
    ):
        """
        Initialize async scanner.

        Args:
            max_concurrent: Maximum concurrent screening tasks
            rate_limit_delay: Delay between starting new tasks (for IBKR pacing)
            state_file: Path to state persistence file
        """
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay
        self.state_file = state_file

        self._state = ScanState()
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

        # Load persisted state
        self._load_state()

    def _load_state(self):
        """Load state from file"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self._state = ScanState.from_dict(data)

                    # Reset running state if it was interrupted
                    if self._state.status == ScanStatus.RUNNING:
                        self._state.status = ScanStatus.PAUSED
                        logger.warning("Scanner was interrupted, state set to PAUSED")
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            self._state = ScanState()

    async def _save_state(self):
        """Save state to file (async-safe)"""
        async with self._lock:
            try:
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.state_file, 'w') as f:
                    json.dump(self._state.to_dict(), f, indent=2)
            except Exception as e:
                logger.error(f"Error saving state: {e}")

    @property
    def is_running(self) -> bool:
        """Check if scan is currently running"""
        return self._state.status == ScanStatus.RUNNING

    @property
    def is_paused(self) -> bool:
        """Check if scan is paused"""
        return self._state.status == ScanStatus.PAUSED

    async def get_state(self) -> ScanState:
        """Get current scan state"""
        async with self._lock:
            return self._state

    async def start_scan(
        self,
        stocks: List[Dict],
        screen_function: Callable[[str, str], Awaitable[Optional[Dict]]],
        resume: bool = False
    ) -> bool:
        """
        Start a new async scan.

        Args:
            stocks: List of stocks [{"symbol": "AAPL", "name": "Apple Inc."}, ...]
            screen_function: Async function(symbol, name) -> Optional[Dict]
            resume: Resume from paused state

        Returns:
            True if scan started, False otherwise
        """
        async with self._lock:
            if self._state.status == ScanStatus.RUNNING:
                logger.warning("Scan already running")
                return False

            if resume and self._state.status == ScanStatus.PAUSED:
                self._state.status = ScanStatus.RUNNING
                self._state.pause_requested = False
                self._state.cancel_requested = False
                stocks_to_scan = self._state.pending_stocks
                logger.info(f"Resuming scan with {len(stocks_to_scan)} remaining stocks")
            else:
                self._state = ScanState(
                    status=ScanStatus.RUNNING,
                    total_stocks=len(stocks),
                    completed_count=0,
                    alerts=[],
                    pending_stocks=stocks.copy(),
                    start_time=datetime.now(),
                    scan_id=self._state.scan_id + 1
                )
                stocks_to_scan = stocks

        await self._save_state()

        # Create semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        # Start scan task
        self._task = asyncio.create_task(
            self._run_scan(stocks_to_scan, screen_function)
        )

        logger.info(f"Async scan started: {len(stocks_to_scan)} stocks, max concurrent: {self.max_concurrent}")
        return True

    async def pause_scan(self):
        """Request scan to pause"""
        async with self._lock:
            if self._state.status == ScanStatus.RUNNING:
                self._state.pause_requested = True
                logger.info("Scan pause requested")
        await self._save_state()

    async def cancel_scan(self):
        """Request scan to cancel"""
        async with self._lock:
            if self._state.status in [ScanStatus.RUNNING, ScanStatus.PAUSED]:
                self._state.cancel_requested = True
                logger.info("Scan cancel requested")

                # Cancel the task if running
                if self._task and not self._task.done():
                    self._task.cancel()

        await self._save_state()

    async def resume_scan(
        self,
        screen_function: Callable[[str, str], Awaitable[Optional[Dict]]]
    ) -> bool:
        """Resume a paused scan"""
        return await self.start_scan([], screen_function, resume=True)

    async def reset(self):
        """Reset scanner to idle state"""
        async with self._lock:
            scan_id = self._state.scan_id
            self._state = ScanState(scan_id=scan_id)
        await self._save_state()

    async def _run_scan(
        self,
        stocks: List[Dict],
        screen_function: Callable[[str, str], Awaitable[Optional[Dict]]]
    ):
        """
        Internal scan execution (runs as async task).

        Uses asyncio.gather with semaphore for controlled concurrency.
        """
        try:
            logger.info(f"Scan task started with {len(stocks)} stocks")

            # Process stocks with controlled concurrency
            tasks = []
            for stock in stocks:
                # Check for cancel/pause before creating new tasks
                if self._state.cancel_requested:
                    logger.info("Scan cancelled before completion")
                    break

                if self._state.pause_requested:
                    # Mark remaining stocks as pending
                    remaining_idx = stocks.index(stock)
                    async with self._lock:
                        self._state.pending_stocks = stocks[remaining_idx:]
                        self._state.status = ScanStatus.PAUSED
                        self._state.pause_requested = False
                    await self._save_state()
                    logger.info(f"Scan paused with {len(stocks) - remaining_idx} stocks remaining")
                    return

                task = asyncio.create_task(
                    self._screen_with_semaphore(stock, screen_function)
                )
                tasks.append(task)

                # Small delay to prevent overwhelming IBKR
                if self.rate_limit_delay > 0:
                    await asyncio.sleep(self.rate_limit_delay)

            # Wait for all tasks to complete
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.debug(f"Task {i} failed: {result}")
                    # Results are already processed in _screen_with_semaphore

            # Check final state
            if self._state.cancel_requested:
                async with self._lock:
                    self._state.status = ScanStatus.CANCELLED
                    self._state.end_time = datetime.now()
                    self._state.cancel_requested = False
                    self._state.pending_stocks = []
                await self._save_state()
                logger.info("Scan cancelled")
            else:
                async with self._lock:
                    self._state.status = ScanStatus.COMPLETED
                    self._state.end_time = datetime.now()
                    self._state.pending_stocks = []
                await self._save_state()
                logger.info(f"Scan completed: {len(self._state.alerts)} alerts")

        except asyncio.CancelledError:
            async with self._lock:
                self._state.status = ScanStatus.CANCELLED
                self._state.end_time = datetime.now()
            await self._save_state()
            logger.info("Scan task cancelled")
            raise

        except Exception as e:
            logger.error(f"Scan error: {e}")
            async with self._lock:
                self._state.status = ScanStatus.ERROR
                self._state.error_message = str(e)
                self._state.end_time = datetime.now()
            await self._save_state()

    async def _screen_with_semaphore(
        self,
        stock: Dict,
        screen_function: Callable[[str, str], Awaitable[Optional[Dict]]]
    ) -> Optional[Dict]:
        """Screen a stock with semaphore-controlled concurrency"""
        async with self._semaphore:
            try:
                symbol = stock.get('symbol', '')
                name = stock.get('name', symbol)

                # Update current symbol
                async with self._lock:
                    self._state.current_symbol = symbol

                # Call the screening function
                result = await screen_function(symbol, name)

                # Update state
                async with self._lock:
                    self._state.completed_count += 1

                    # Remove from pending
                    self._state.pending_stocks = [
                        s for s in self._state.pending_stocks
                        if s.get('symbol') != symbol
                    ]

                    # Add to alerts if signal found
                    if result:
                        result['market'] = stock.get('market', 'N/A')
                        self._state.alerts.append(result)
                        logger.info(f"Signal: {symbol} - Total: {len(self._state.alerts)}")

                # Periodic state save (every 10 stocks)
                if self._state.completed_count % 10 == 0:
                    await self._save_state()

                return result

            except Exception as e:
                logger.debug(f"Error screening {stock.get('symbol', '?')}: {e}")
                async with self._lock:
                    self._state.completed_count += 1
                return None

    def get_progress_info(self) -> Dict:
        """Get formatted progress information"""
        state = self._state

        elapsed = 0
        if state.start_time:
            end = state.end_time or datetime.now()
            elapsed = (end - state.start_time).total_seconds()

        rate = state.completed_count / elapsed if elapsed > 0 else 0
        remaining = state.total_stocks - state.completed_count
        eta = remaining / rate if rate > 0 else 0

        return {
            'status': state.status.value,
            'completed': state.completed_count,
            'total': state.total_stocks,
            'alerts_count': len(state.alerts),
            'elapsed_seconds': elapsed,
            'rate_per_second': rate,
            'eta_seconds': eta,
            'current_symbol': state.current_symbol,
            'progress_pct': (state.completed_count / state.total_stocks * 100) if state.total_stocks > 0 else 0
        }


# =============================================================================
# ASYNC SCAN SCHEDULER
# =============================================================================

SCHEDULER_CONFIG_FILE = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / 'data' / 'async_scheduler_config.json'


class ScheduleType(Enum):
    """Types of schedules"""
    ONCE = "once"
    DAILY = "daily"
    HOURLY = "hourly"
    WEEKLY = "weekly"
    INTERVAL = "interval"


@dataclass
class ScheduleConfig:
    """Configuration for scheduled scans"""
    enabled: bool = False
    schedule_type: str = "daily"
    time_of_day: str = "09:30"
    days_of_week: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    interval_minutes: int = 60
    markets: List[str] = field(default_factory=lambda: ["NASDAQ", "SP500"])
    notify_on_completion: bool = True
    notify_on_signals: bool = True
    min_signal_priority: str = "medium"
    last_run: Optional[str] = None
    next_run: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ScheduleConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SchedulerState:
    """Scheduler runtime state"""
    running: bool = False
    next_scheduled_run: Optional[datetime] = None
    last_run_time: Optional[datetime] = None
    last_run_alerts: int = 0
    consecutive_errors: int = 0
    total_runs: int = 0


class AsyncScanScheduler:
    """
    Async scheduler for automated market scans.

    Fully compatible with ib_insync and asyncio event loops.

    Usage:
        scheduler = AsyncScanScheduler()

        scheduler.set_screen_function(my_screen_func)
        scheduler.set_stocks_provider(get_stocks_func)

        await scheduler.start()

        # Later...
        await scheduler.stop()
    """

    def __init__(self, config_path: Path = SCHEDULER_CONFIG_FILE):
        self.config_path = config_path
        self.config = self._load_config()
        self.state = SchedulerState()
        self._scanner = AsyncBackgroundScanner()
        self._notification_manager = None
        self._screen_function: Optional[Callable] = None
        self._stocks_provider: Optional[Callable] = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    def _load_config(self) -> ScheduleConfig:
        """Load scheduler configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    return ScheduleConfig.from_dict(data)
        except Exception as e:
            logger.warning(f"Error loading scheduler config: {e}")
        return ScheduleConfig()

    def _save_config(self):
        """Save scheduler configuration"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving scheduler config: {e}")

    def update_config(self, **kwargs):
        """Update scheduler configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        if self.config.enabled:
            self.config.next_run = self._calculate_next_run().isoformat()

        self._save_config()

    def set_notification_manager(self, manager):
        """Set notification manager for alerts"""
        self._notification_manager = manager

    def set_screen_function(self, func: Callable[[str, str], Awaitable[Optional[Dict]]]):
        """Set the async screening function"""
        self._screen_function = func

    def set_stocks_provider(self, provider: Callable[[], Awaitable[List[Dict]]]):
        """Set async function that provides stocks to scan"""
        self._stocks_provider = provider

    def _calculate_next_run(self) -> datetime:
        """Calculate next scheduled run time"""
        now = datetime.now()

        if self.config.schedule_type == "once":
            hour, minute = map(int, self.config.time_of_day.split(':'))
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            return target

        elif self.config.schedule_type == "daily":
            hour, minute = map(int, self.config.time_of_day.split(':'))
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            while target.weekday() not in self.config.days_of_week or target <= now:
                target += timedelta(days=1)
            return target

        elif self.config.schedule_type == "weekly":
            hour, minute = map(int, self.config.time_of_day.split(':'))
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            while target.weekday() not in self.config.days_of_week or target <= now:
                target += timedelta(days=1)
            return target

        elif self.config.schedule_type == "hourly":
            target = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            return target

        elif self.config.schedule_type == "interval":
            return now + timedelta(minutes=self.config.interval_minutes)

        return now + timedelta(hours=1)

    async def start(self) -> bool:
        """Start the scheduler"""
        if self.state.running:
            logger.warning("Scheduler already running")
            return False

        if not self._screen_function:
            logger.error("No screen function set")
            return False

        if not self._stocks_provider:
            logger.error("No stocks provider set")
            return False

        self._stop_event.clear()
        self.state.running = True
        self.state.next_scheduled_run = self._calculate_next_run()

        self._task = asyncio.create_task(self._scheduler_loop())

        logger.info(f"Async scheduler started. Next run: {self.state.next_scheduled_run}")
        return True

    async def stop(self):
        """Stop the scheduler"""
        self._stop_event.set()
        self.state.running = False

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Async scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while not self._stop_event.is_set():
            try:
                if not self.config.enabled:
                    await asyncio.sleep(60)
                    continue

                now = datetime.now()
                next_run = self.state.next_scheduled_run

                if next_run and now >= next_run:
                    logger.info(f"Executing scheduled scan at {now}")
                    await self._execute_scheduled_scan()

                    self.state.next_scheduled_run = self._calculate_next_run()
                    self.config.last_run = now.isoformat()
                    self.config.next_run = self.state.next_scheduled_run.isoformat()
                    self._save_config()

                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                self.state.consecutive_errors += 1
                if self.state.consecutive_errors > 5:
                    logger.error("Too many consecutive errors, stopping scheduler")
                    await self.stop()
                    break
                await asyncio.sleep(60)

    async def _execute_scheduled_scan(self):
        """Execute a scheduled scan"""
        try:
            # Get stocks to scan
            if asyncio.iscoroutinefunction(self._stocks_provider):
                stocks = await self._stocks_provider()
            else:
                stocks = self._stocks_provider()

            if not stocks:
                logger.warning("No stocks to scan")
                return

            logger.info(f"Starting scheduled scan: {len(stocks)} stocks")

            # Start the scan
            await self._scanner.start_scan(
                stocks=stocks,
                screen_function=self._screen_function
            )

            # Wait for completion
            max_wait = 3600
            waited = 0
            while waited < max_wait:
                state = await self._scanner.get_state()
                if state.status in [ScanStatus.COMPLETED, ScanStatus.ERROR, ScanStatus.CANCELLED]:
                    break
                await asyncio.sleep(10)
                waited += 10

            # Process results
            final_state = await self._scanner.get_state()
            self.state.last_run_time = datetime.now()
            self.state.last_run_alerts = len(final_state.alerts)
            self.state.total_runs += 1
            self.state.consecutive_errors = 0

            # Send notifications
            await self._send_notifications(final_state)

        except Exception as e:
            logger.error(f"Scheduled scan error: {e}")
            self.state.consecutive_errors += 1

    async def _send_notifications(self, scan_state: ScanState):
        """Send notifications based on scan results"""
        if not self._notification_manager:
            return

        try:
            alerts = scan_state.alerts

            if self.config.notify_on_completion:
                summary_text = f"""
📊 <b>Scan Completed</b>

⏱️ Duration: {self._format_duration(scan_state.start_time, scan_state.end_time)}
📈 Stocks scanned: {scan_state.total_stocks}
🎯 Alerts found: {len(alerts)}

Status: {scan_state.status.value.upper()}
"""
                if hasattr(self._notification_manager, '_telegram') and self._notification_manager._telegram:
                    await self._notification_manager._telegram.send_message(summary_text)

            if self.config.notify_on_signals and alerts:
                for alert in alerts:
                    await self._notification_manager.send_alert_notification(alert)

        except Exception as e:
            logger.error(f"Error sending notifications: {e}")

    def _format_duration(self, start: Optional[datetime], end: Optional[datetime]) -> str:
        """Format duration between two times"""
        if not start or not end:
            return "N/A"

        duration = (end - start).total_seconds()
        if duration < 60:
            return f"{duration:.0f}s"
        elif duration < 3600:
            return f"{duration/60:.1f}m"
        else:
            return f"{duration/3600:.1f}h"

    def get_status(self) -> Dict:
        """Get scheduler status"""
        return {
            'running': self.state.running,
            'enabled': self.config.enabled,
            'schedule_type': self.config.schedule_type,
            'time_of_day': self.config.time_of_day,
            'days_of_week': self.config.days_of_week,
            'markets': self.config.markets,
            'next_run': self.state.next_scheduled_run.isoformat() if self.state.next_scheduled_run else None,
            'last_run': self.state.last_run_time.isoformat() if self.state.last_run_time else None,
            'last_alerts': self.state.last_run_alerts,
            'total_runs': self.state.total_runs,
            'notify_on_completion': self.config.notify_on_completion,
            'notify_on_signals': self.config.notify_on_signals
        }

    async def trigger_now(self) -> bool:
        """Manually trigger a scan now"""
        if self._scanner.is_running:
            logger.warning("Scan already in progress")
            return False

        if not self._screen_function or not self._stocks_provider:
            logger.error("Screen function or stocks provider not set")
            return False

        asyncio.create_task(self._execute_scheduled_scan())
        return True


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

# Async versions (use these with ib_insync)
async_background_scanner = AsyncBackgroundScanner()
async_scan_scheduler = AsyncScanScheduler()


async def get_async_scanner() -> AsyncBackgroundScanner:
    """Get the async background scanner instance"""
    return async_background_scanner


async def get_async_scheduler() -> AsyncScanScheduler:
    """Get the async scan scheduler instance"""
    return async_scan_scheduler
