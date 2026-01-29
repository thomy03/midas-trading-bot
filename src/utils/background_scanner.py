"""
Background Scanner - Runs market scans in a separate thread

Features:
- Background scanning with progress tracking
- Scheduled scanning (daily, hourly, custom cron)
- Notification integration (alerts on completion/signals)
- State persistence across Streamlit reruns

Uses a temp file to persist state across Streamlit reruns.
"""

import threading
import json
import tempfile
import os
import asyncio
from datetime import datetime, time, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Use a fixed temp file path for state persistence
STATE_FILE = os.path.join(tempfile.gettempdir(), 'tradingbot_scan_state.json')
_LOCK = threading.Lock()
_SCAN_THREAD = None


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


def _save_state(state_dict: dict):
    """Save state to temp file"""
    try:
        # Convert datetime objects to strings
        state_copy = state_dict.copy()
        if state_copy.get('start_time'):
            state_copy['start_time'] = state_copy['start_time'].isoformat()
        if state_copy.get('end_time'):
            state_copy['end_time'] = state_copy['end_time'].isoformat()

        with open(STATE_FILE, 'w') as f:
            json.dump(state_copy, f)
    except Exception as e:
        logger.error(f"Error saving state: {e}")


def _load_state() -> dict:
    """Load state from temp file"""
    default_state = {
        'status': 'idle',
        'total_stocks': 0,
        'completed_count': 0,
        'alerts': [],
        'pending_stocks': [],
        'start_time': None,
        'end_time': None,
        'error_message': None,
        'current_symbol': '',
        'scan_id': 0,
        'pause_requested': False,
        'cancel_requested': False,
    }

    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)

            # Convert datetime strings back to datetime objects
            if state.get('start_time'):
                state['start_time'] = datetime.fromisoformat(state['start_time'])
            if state.get('end_time'):
                state['end_time'] = datetime.fromisoformat(state['end_time'])

            return state
    except Exception as e:
        logger.error(f"Error loading state: {e}")

    return default_state


class BackgroundScanner:
    """Background scanner using file-based state persistence"""

    def get_state(self) -> ScanState:
        """Get current scan state"""
        with _LOCK:
            state_dict = _load_state()

            status_str = state_dict.get('status', 'idle')
            try:
                status = ScanStatus(status_str)
            except ValueError:
                status = ScanStatus.IDLE

            return ScanState(
                status=status,
                total_stocks=state_dict.get('total_stocks', 0),
                completed_count=state_dict.get('completed_count', 0),
                alerts=state_dict.get('alerts', []),
                pending_stocks=state_dict.get('pending_stocks', []),
                start_time=state_dict.get('start_time'),
                end_time=state_dict.get('end_time'),
                error_message=state_dict.get('error_message'),
                current_symbol=state_dict.get('current_symbol', ''),
                scan_id=state_dict.get('scan_id', 0),
                pause_requested=state_dict.get('pause_requested', False),
                cancel_requested=state_dict.get('cancel_requested', False)
            )

    def is_running(self) -> bool:
        """Check if scan is currently running"""
        state = _load_state()
        return state.get('status') == 'running'

    def is_paused(self) -> bool:
        """Check if scan is paused"""
        state = _load_state()
        return state.get('status') == 'paused'

    def start_scan(
        self,
        stocks: List[Dict],
        screen_function: Callable[[str, str], Optional[Dict]],
        num_workers: int = 10,
        resume: bool = False
    ) -> bool:
        """Start a new scan in background thread"""
        global _SCAN_THREAD

        with _LOCK:
            state = _load_state()

            if state.get('status') == 'running':
                logger.warning("Scan already running")
                return False

            if resume and state.get('status') == 'paused':
                state['status'] = 'running'
                state['pause_requested'] = False
                state['cancel_requested'] = False
                stocks_to_scan = state.get('pending_stocks', [])
            else:
                scan_id = state.get('scan_id', 0) + 1
                state = {
                    'status': 'running',
                    'total_stocks': len(stocks),
                    'completed_count': 0,
                    'alerts': [],
                    'pending_stocks': stocks,
                    'start_time': datetime.now(),
                    'end_time': None,
                    'error_message': None,
                    'current_symbol': '',
                    'scan_id': scan_id,
                    'pause_requested': False,
                    'cancel_requested': False,
                }
                stocks_to_scan = stocks

            _save_state(state)

        # Start scan thread
        _SCAN_THREAD = threading.Thread(
            target=self._run_scan,
            args=(stocks_to_scan, screen_function, num_workers),
            daemon=True
        )
        _SCAN_THREAD.start()

        logger.info(f"Background scan started: {len(stocks_to_scan)} stocks with {num_workers} workers")
        return True

    def pause_scan(self):
        """Request scan to pause"""
        with _LOCK:
            state = _load_state()
            if state.get('status') == 'running':
                state['pause_requested'] = True
                _save_state(state)
                logger.info("Scan pause requested")

    def resume_scan(self, screen_function: Callable, num_workers: int = 10) -> bool:
        """Resume a paused scan"""
        return self.start_scan([], screen_function, num_workers, resume=True)

    def cancel_scan(self):
        """Request scan to cancel"""
        with _LOCK:
            state = _load_state()
            if state.get('status') in ['running', 'paused']:
                state['cancel_requested'] = True
                _save_state(state)
                logger.info("Scan cancel requested")

    def reset(self):
        """Reset scanner to idle state"""
        with _LOCK:
            state = _load_state()
            scan_id = state.get('scan_id', 0)
            new_state = {
                'status': 'idle',
                'total_stocks': 0,
                'completed_count': 0,
                'alerts': [],
                'pending_stocks': [],
                'start_time': None,
                'end_time': None,
                'error_message': None,
                'current_symbol': '',
                'scan_id': scan_id,
                'pause_requested': False,
                'cancel_requested': False,
            }
            _save_state(new_state)

    def _run_scan(
        self,
        stocks: List[Dict],
        screen_function: Callable,
        num_workers: int
    ):
        """Internal scan execution (runs in background thread)"""
        try:
            logger.info(f"Scan thread started with {len(stocks)} stocks")

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {}

                for stock in stocks:
                    # Check for cancel/pause
                    state = _load_state()
                    if state.get('cancel_requested'):
                        break
                    if state.get('pause_requested'):
                        remaining_idx = stocks.index(stock)
                        with _LOCK:
                            state = _load_state()
                            state['pending_stocks'] = stocks[remaining_idx:]
                            state['status'] = 'paused'
                            state['pause_requested'] = False
                            _save_state(state)
                        logger.info(f"Scan paused with {len(stocks) - remaining_idx} stocks remaining")
                        return

                    future = executor.submit(self._screen_stock, stock, screen_function)
                    futures[future] = stock

                # Process results
                for future in as_completed(futures):
                    state = _load_state()

                    if state.get('cancel_requested'):
                        for f in futures:
                            f.cancel()
                        with _LOCK:
                            state = _load_state()
                            state['status'] = 'cancelled'
                            state['end_time'] = datetime.now()
                            state['cancel_requested'] = False
                            _save_state(state)
                        logger.info("Scan cancelled")
                        return

                    if state.get('pause_requested'):
                        completed_symbols = {
                            futures[f]['symbol'] for f in futures
                            if f.done() and not f.cancelled()
                        }
                        with _LOCK:
                            state = _load_state()
                            state['pending_stocks'] = [
                                s for s in stocks if s['symbol'] not in completed_symbols
                            ]
                            state['status'] = 'paused'
                            state['pause_requested'] = False
                            _save_state(state)
                        logger.info(f"Scan paused")
                        return

                    try:
                        result = future.result(timeout=60)
                        stock = futures[future]

                        with _LOCK:
                            state = _load_state()
                            state['completed_count'] = state.get('completed_count', 0) + 1
                            state['current_symbol'] = stock['symbol']

                            # Remove from pending
                            state['pending_stocks'] = [
                                s for s in state.get('pending_stocks', [])
                                if s['symbol'] != stock['symbol']
                            ]

                            if result:
                                result['market'] = stock.get('market', 'N/A')
                                state['alerts'] = state.get('alerts', []) + [result]
                                logger.info(f"Signal: {stock['symbol']} - Total: {len(state['alerts'])}")

                            _save_state(state)

                    except Exception as e:
                        logger.debug(f"Error processing {futures[future]['symbol']}: {e}")
                        with _LOCK:
                            state = _load_state()
                            state['completed_count'] = state.get('completed_count', 0) + 1
                            _save_state(state)

            # Scan completed
            with _LOCK:
                state = _load_state()
                state['status'] = 'completed'
                state['end_time'] = datetime.now()
                state['pending_stocks'] = []
                _save_state(state)
            logger.info(f"Scan completed: {len(state.get('alerts', []))} alerts")

        except Exception as e:
            logger.error(f"Scan error: {e}")
            with _LOCK:
                state = _load_state()
                state['status'] = 'error'
                state['error_message'] = str(e)
                state['end_time'] = datetime.now()
                _save_state(state)

    def _screen_stock(self, stock: Dict, screen_function: Callable) -> Optional[Dict]:
        """Screen a single stock"""
        try:
            return screen_function(stock['symbol'], stock.get('name', stock['symbol']))
        except Exception as e:
            logger.debug(f"Error screening {stock['symbol']}: {e}")
            return None

    def get_progress_info(self) -> Dict:
        """Get formatted progress information"""
        state = self.get_state()

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


# ==================== Scheduler Classes ====================

SCHEDULER_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'data', 'scheduler_config.json'
)

_SCHEDULER_THREAD = None
_SCHEDULER_STOP_EVENT = threading.Event()


class ScheduleType(Enum):
    """Types of schedules"""
    ONCE = "once"
    DAILY = "daily"
    HOURLY = "hourly"
    WEEKLY = "weekly"
    INTERVAL = "interval"  # Every N minutes


@dataclass
class ScheduleConfig:
    """Configuration for scheduled scans"""
    enabled: bool = False
    schedule_type: str = "daily"  # once, daily, hourly, weekly, interval
    time_of_day: str = "09:30"  # HH:MM format for daily/weekly
    days_of_week: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri
    interval_minutes: int = 60  # For interval type
    markets: List[str] = field(default_factory=lambda: ["NASDAQ", "SP500"])
    notify_on_completion: bool = True
    notify_on_signals: bool = True
    min_signal_priority: str = "medium"  # low, medium, high, urgent
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


class ScanScheduler:
    """
    Scheduler for automated market scans.

    Supports:
    - Daily scans at specific times
    - Weekly scans on specific days
    - Interval-based scans (every N minutes)
    - Notification integration
    """

    def __init__(self, config_path: str = SCHEDULER_CONFIG_FILE):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.state = SchedulerState()
        self._scanner = BackgroundScanner()
        self._notification_manager = None
        self._screen_function = None
        self._stocks_provider = None

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

        # Recalculate next run
        if self.config.enabled:
            self.config.next_run = self._calculate_next_run().isoformat()

        self._save_config()

    def set_notification_manager(self, manager):
        """Set notification manager for alerts"""
        self._notification_manager = manager

    def set_screen_function(self, func: Callable):
        """Set the screening function to use"""
        self._screen_function = func

    def set_stocks_provider(self, provider: Callable[[], List[Dict]]):
        """Set function that provides stocks to scan"""
        self._stocks_provider = provider

    def _calculate_next_run(self) -> datetime:
        """Calculate next scheduled run time"""
        now = datetime.now()

        if self.config.schedule_type == "once":
            # Parse time and return today or tomorrow
            hour, minute = map(int, self.config.time_of_day.split(':'))
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            return target

        elif self.config.schedule_type == "daily":
            hour, minute = map(int, self.config.time_of_day.split(':'))
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # Check if today is a valid day
            while target.weekday() not in self.config.days_of_week or target <= now:
                target += timedelta(days=1)

            return target

        elif self.config.schedule_type == "weekly":
            hour, minute = map(int, self.config.time_of_day.split(':'))
            target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # Find next valid day
            while target.weekday() not in self.config.days_of_week or target <= now:
                target += timedelta(days=1)

            return target

        elif self.config.schedule_type == "hourly":
            # Next hour at minute 0
            target = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            return target

        elif self.config.schedule_type == "interval":
            # Add interval minutes
            return now + timedelta(minutes=self.config.interval_minutes)

        return now + timedelta(hours=1)

    def start(self) -> bool:
        """Start the scheduler"""
        global _SCHEDULER_THREAD, _SCHEDULER_STOP_EVENT

        if self.state.running:
            logger.warning("Scheduler already running")
            return False

        if not self._screen_function:
            logger.error("No screen function set")
            return False

        if not self._stocks_provider:
            logger.error("No stocks provider set")
            return False

        _SCHEDULER_STOP_EVENT.clear()
        self.state.running = True
        self.state.next_scheduled_run = self._calculate_next_run()

        _SCHEDULER_THREAD = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
        _SCHEDULER_THREAD.start()

        logger.info(f"Scheduler started. Next run: {self.state.next_scheduled_run}")
        return True

    def stop(self):
        """Stop the scheduler"""
        global _SCHEDULER_STOP_EVENT

        _SCHEDULER_STOP_EVENT.set()
        self.state.running = False
        logger.info("Scheduler stopped")

    def _scheduler_loop(self):
        """Main scheduler loop"""
        while not _SCHEDULER_STOP_EVENT.is_set():
            try:
                if not self.config.enabled:
                    _SCHEDULER_STOP_EVENT.wait(60)  # Check every minute
                    continue

                now = datetime.now()
                next_run = self.state.next_scheduled_run

                if next_run and now >= next_run:
                    logger.info(f"Executing scheduled scan at {now}")
                    self._execute_scheduled_scan()

                    # Calculate next run
                    self.state.next_scheduled_run = self._calculate_next_run()
                    self.config.last_run = now.isoformat()
                    self.config.next_run = self.state.next_scheduled_run.isoformat()
                    self._save_config()

                # Wait for next check (every 30 seconds)
                _SCHEDULER_STOP_EVENT.wait(30)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                self.state.consecutive_errors += 1
                if self.state.consecutive_errors > 5:
                    logger.error("Too many consecutive errors, stopping scheduler")
                    self.stop()
                    break
                _SCHEDULER_STOP_EVENT.wait(60)

    def _execute_scheduled_scan(self):
        """Execute a scheduled scan"""
        try:
            # Get stocks to scan
            stocks = self._stocks_provider()
            if not stocks:
                logger.warning("No stocks to scan")
                return

            logger.info(f"Starting scheduled scan: {len(stocks)} stocks")

            # Start the scan
            self._scanner.start_scan(
                stocks=stocks,
                screen_function=self._screen_function,
                num_workers=10
            )

            # Wait for completion (with timeout)
            max_wait = 3600  # 1 hour max
            waited = 0
            while waited < max_wait:
                state = self._scanner.get_state()
                if state.status in [ScanStatus.COMPLETED, ScanStatus.ERROR, ScanStatus.CANCELLED]:
                    break
                _SCHEDULER_STOP_EVENT.wait(10)
                waited += 10

            # Process results
            final_state = self._scanner.get_state()
            self.state.last_run_time = datetime.now()
            self.state.last_run_alerts = len(final_state.alerts)
            self.state.total_runs += 1
            self.state.consecutive_errors = 0

            # Send notifications
            self._send_notifications(final_state)

        except Exception as e:
            logger.error(f"Scheduled scan error: {e}")
            self.state.consecutive_errors += 1

    def _send_notifications(self, scan_state: ScanState):
        """Send notifications based on scan results"""
        if not self._notification_manager:
            return

        try:
            alerts = scan_state.alerts

            # Send completion notification
            if self.config.notify_on_completion:
                summary_text = f"""
📊 <b>Scan Completed</b>

⏱️ Duration: {self._format_duration(scan_state.start_time, scan_state.end_time)}
📈 Stocks scanned: {scan_state.total_stocks}
🎯 Alerts found: {len(alerts)}

Status: {scan_state.status.value.upper()}
"""
                if self._notification_manager._telegram:
                    asyncio.run(
                        self._notification_manager._telegram.send_message(summary_text)
                    )

            # Send individual signal notifications
            if self.config.notify_on_signals and alerts:
                for alert in alerts:
                    asyncio.run(
                        self._notification_manager.send_alert_notification(alert)
                    )

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

    def trigger_now(self) -> bool:
        """Manually trigger a scan now"""
        if self._scanner.is_running():
            logger.warning("Scan already in progress")
            return False

        if not self._screen_function or not self._stocks_provider:
            logger.error("Screen function or stocks provider not set")
            return False

        # Execute in background thread
        threading.Thread(
            target=self._execute_scheduled_scan,
            daemon=True
        ).start()

        return True


# Singleton instances
background_scanner = BackgroundScanner()
scan_scheduler = ScanScheduler()
