"""
Economic Calendar for TradingBot V3

Provides economic event tracking:
- Earnings dates for tracked stocks
- FOMC/Fed meeting dates
- CPI release dates
- Other major economic events
"""
import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of economic events"""
    EARNINGS = "earnings"
    FOMC = "fomc"
    CPI = "cpi"
    JOBS = "jobs"
    GDP = "gdp"
    RETAIL_SALES = "retail_sales"
    CUSTOM = "custom"


class EventImpact(Enum):
    """Impact level of economic events"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class EconomicEvent:
    """Represents an economic event"""
    event_id: str
    event_type: EventType
    title: str
    date: datetime
    impact: EventImpact
    symbol: Optional[str] = None  # For earnings
    description: str = ""
    previous_value: Optional[str] = None
    forecast_value: Optional[str] = None
    actual_value: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'title': self.title,
            'date': self.date.isoformat(),
            'impact': self.impact.value,
            'symbol': self.symbol,
            'description': self.description,
            'previous_value': self.previous_value,
            'forecast_value': self.forecast_value,
            'actual_value': self.actual_value
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EconomicEvent':
        """Create from dictionary"""
        return cls(
            event_id=data['event_id'],
            event_type=EventType(data['event_type']),
            title=data['title'],
            date=datetime.fromisoformat(data['date']),
            impact=EventImpact(data['impact']),
            symbol=data.get('symbol'),
            description=data.get('description', ''),
            previous_value=data.get('previous_value'),
            forecast_value=data.get('forecast_value'),
            actual_value=data.get('actual_value')
        )


# FOMC Meeting dates for 2024-2025
FOMC_DATES_2024 = [
    datetime(2024, 1, 30), datetime(2024, 1, 31),
    datetime(2024, 3, 19), datetime(2024, 3, 20),
    datetime(2024, 4, 30), datetime(2024, 5, 1),
    datetime(2024, 6, 11), datetime(2024, 6, 12),
    datetime(2024, 7, 30), datetime(2024, 7, 31),
    datetime(2024, 9, 17), datetime(2024, 9, 18),
    datetime(2024, 11, 6), datetime(2024, 11, 7),
    datetime(2024, 12, 17), datetime(2024, 12, 18),
]

FOMC_DATES_2025 = [
    datetime(2025, 1, 28), datetime(2025, 1, 29),
    datetime(2025, 3, 18), datetime(2025, 3, 19),
    datetime(2025, 5, 6), datetime(2025, 5, 7),
    datetime(2025, 6, 17), datetime(2025, 6, 18),
    datetime(2025, 7, 29), datetime(2025, 7, 30),
    datetime(2025, 9, 16), datetime(2025, 9, 17),
    datetime(2025, 11, 4), datetime(2025, 11, 5),
    datetime(2025, 12, 16), datetime(2025, 12, 17),
]

# CPI Release dates (typically mid-month)
CPI_DATES_2024 = [
    datetime(2024, 1, 11), datetime(2024, 2, 13),
    datetime(2024, 3, 12), datetime(2024, 4, 10),
    datetime(2024, 5, 15), datetime(2024, 6, 12),
    datetime(2024, 7, 11), datetime(2024, 8, 14),
    datetime(2024, 9, 11), datetime(2024, 10, 10),
    datetime(2024, 11, 13), datetime(2024, 12, 11),
]

CPI_DATES_2025 = [
    datetime(2025, 1, 15), datetime(2025, 2, 12),
    datetime(2025, 3, 12), datetime(2025, 4, 10),
    datetime(2025, 5, 13), datetime(2025, 6, 11),
    datetime(2025, 7, 11), datetime(2025, 8, 12),
    datetime(2025, 9, 10), datetime(2025, 10, 10),
    datetime(2025, 11, 13), datetime(2025, 12, 10),
]

# Jobs Report dates (first Friday of month, typically)
JOBS_DATES_2024 = [
    datetime(2024, 1, 5), datetime(2024, 2, 2),
    datetime(2024, 3, 8), datetime(2024, 4, 5),
    datetime(2024, 5, 3), datetime(2024, 6, 7),
    datetime(2024, 7, 5), datetime(2024, 8, 2),
    datetime(2024, 9, 6), datetime(2024, 10, 4),
    datetime(2024, 11, 1), datetime(2024, 12, 6),
]

JOBS_DATES_2025 = [
    datetime(2025, 1, 10), datetime(2025, 2, 7),
    datetime(2025, 3, 7), datetime(2025, 4, 4),
    datetime(2025, 5, 2), datetime(2025, 6, 6),
    datetime(2025, 7, 3), datetime(2025, 8, 1),
    datetime(2025, 9, 5), datetime(2025, 10, 3),
    datetime(2025, 11, 7), datetime(2025, 12, 5),
]


class EconomicCalendar:
    """Economic calendar tracking major market events"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.events_file = self.data_dir / "economic_events.json"
        self.earnings_cache_file = self.data_dir / "earnings_cache.json"

        self._events: List[EconomicEvent] = []
        self._earnings_cache: Dict[str, Dict] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(hours=4)

        self._load_data()
        self._initialize_fixed_events()

    def _load_data(self):
        """Load events and cache from files"""
        # Load custom events
        if self.events_file.exists():
            try:
                with open(self.events_file, 'r') as f:
                    data = json.load(f)
                    self._events = [EconomicEvent.from_dict(e) for e in data]
            except Exception as e:
                logger.warning(f"Error loading events: {e}")
                self._events = []

        # Load earnings cache
        if self.earnings_cache_file.exists():
            try:
                with open(self.earnings_cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self._earnings_cache = cache_data.get('cache', {})
                    cache_time = cache_data.get('cache_time')
                    if cache_time:
                        self._cache_time = datetime.fromisoformat(cache_time)
            except Exception as e:
                logger.warning(f"Error loading earnings cache: {e}")

    def _save_data(self):
        """Save events to file"""
        try:
            with open(self.events_file, 'w') as f:
                json.dump([e.to_dict() for e in self._events], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving events: {e}")

    def _save_earnings_cache(self):
        """Save earnings cache to file"""
        try:
            with open(self.earnings_cache_file, 'w') as f:
                json.dump({
                    'cache': self._earnings_cache,
                    'cache_time': self._cache_time.isoformat() if self._cache_time else None
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving earnings cache: {e}")

    def _initialize_fixed_events(self):
        """Initialize FOMC, CPI, and Jobs events"""
        # Clear existing fixed events to avoid duplicates
        self._events = [e for e in self._events if e.event_type == EventType.CUSTOM]

        # Add FOMC meetings
        for date in FOMC_DATES_2024 + FOMC_DATES_2025:
            # Only add decision days (second day of meeting)
            if date.day in [d.day for d in FOMC_DATES_2024 + FOMC_DATES_2025
                           if (d + timedelta(days=1)).day == date.day or d.day % 2 == 1]:
                continue  # Skip first days

            event = EconomicEvent(
                event_id=f"fomc_{date.strftime('%Y%m%d')}",
                event_type=EventType.FOMC,
                title="FOMC Interest Rate Decision",
                date=date,
                impact=EventImpact.HIGH,
                description="Federal Reserve interest rate decision and economic projections"
            )
            self._events.append(event)

        # Add CPI releases
        for date in CPI_DATES_2024 + CPI_DATES_2025:
            event = EconomicEvent(
                event_id=f"cpi_{date.strftime('%Y%m%d')}",
                event_type=EventType.CPI,
                title="CPI Inflation Report",
                date=date,
                impact=EventImpact.HIGH,
                description="Consumer Price Index - Monthly inflation data"
            )
            self._events.append(event)

        # Add Jobs reports
        for date in JOBS_DATES_2024 + JOBS_DATES_2025:
            event = EconomicEvent(
                event_id=f"jobs_{date.strftime('%Y%m%d')}",
                event_type=EventType.JOBS,
                title="Non-Farm Payrolls",
                date=date,
                impact=EventImpact.HIGH,
                description="Monthly employment situation report"
            )
            self._events.append(event)

    def get_earnings_date(self, symbol: str, use_cache: bool = True) -> Optional[Dict]:
        """
        Get earnings date for a symbol

        Args:
            symbol: Stock symbol
            use_cache: Whether to use cached data

        Returns:
            Dict with earnings info or None
        """
        symbol = symbol.upper()

        # Check cache
        if use_cache and symbol in self._earnings_cache:
            if self._cache_time and datetime.now() - self._cache_time < self._cache_duration:
                return self._earnings_cache[symbol]

        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar

            if calendar is None or (isinstance(calendar, pd.DataFrame) and calendar.empty):
                return None

            # yfinance returns calendar as DataFrame or dict
            if isinstance(calendar, pd.DataFrame):
                if 'Earnings Date' in calendar.index:
                    earnings_dates = calendar.loc['Earnings Date']
                    if isinstance(earnings_dates, pd.Series):
                        next_date = earnings_dates.iloc[0]
                    else:
                        next_date = earnings_dates
                else:
                    return None
            elif isinstance(calendar, dict):
                earnings_dates = calendar.get('Earnings Date', [])
                if earnings_dates:
                    next_date = earnings_dates[0] if isinstance(earnings_dates, list) else earnings_dates
                else:
                    return None
            else:
                return None

            # Convert to datetime if needed
            if isinstance(next_date, pd.Timestamp):
                next_date = next_date.to_pydatetime()
            elif isinstance(next_date, str):
                next_date = datetime.fromisoformat(next_date)

            result = {
                'symbol': symbol,
                'earnings_date': next_date,
                'days_until': (next_date - datetime.now()).days if next_date else None
            }

            # Update cache
            self._earnings_cache[symbol] = result
            self._cache_time = datetime.now()
            self._save_earnings_cache()

            return result

        except Exception as e:
            logger.warning(f"Error fetching earnings for {symbol}: {e}")
            return None

    def get_earnings_dates_batch(
        self,
        symbols: List[str],
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Get earnings dates for multiple symbols in parallel

        Args:
            symbols: List of stock symbols
            use_cache: Whether to use cached data

        Returns:
            List of earnings info dicts
        """
        results = []
        symbols_to_fetch = []

        # Check cache first
        if use_cache and self._cache_time and datetime.now() - self._cache_time < self._cache_duration:
            for symbol in symbols:
                symbol = symbol.upper()
                if symbol in self._earnings_cache:
                    results.append(self._earnings_cache[symbol])
                else:
                    symbols_to_fetch.append(symbol)
        else:
            symbols_to_fetch = [s.upper() for s in symbols]

        # Fetch remaining in parallel
        if symbols_to_fetch:
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(self.get_earnings_date, s, False): s
                    for s in symbols_to_fetch
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)

        # Sort by date
        results.sort(key=lambda x: x.get('earnings_date') or datetime.max)

        return results

    def get_upcoming_events(
        self,
        days_ahead: int = 30,
        event_types: Optional[List[EventType]] = None,
        include_earnings: bool = True,
        symbols: Optional[List[str]] = None
    ) -> List[EconomicEvent]:
        """
        Get upcoming economic events

        Args:
            days_ahead: Number of days to look ahead
            event_types: Filter by event types
            include_earnings: Include earnings events
            symbols: Symbols to include earnings for

        Returns:
            List of upcoming events sorted by date
        """
        now = datetime.now()
        cutoff = now + timedelta(days=days_ahead)

        events = []

        # Add fixed events (FOMC, CPI, Jobs)
        for event in self._events:
            if event.date >= now and event.date <= cutoff:
                if event_types is None or event.event_type in event_types:
                    events.append(event)

        # Add earnings events
        if include_earnings and symbols:
            earnings_data = self.get_earnings_dates_batch(symbols)
            for data in earnings_data:
                if data and data.get('earnings_date'):
                    earnings_date = data['earnings_date']
                    if isinstance(earnings_date, datetime) and now <= earnings_date <= cutoff:
                        event = EconomicEvent(
                            event_id=f"earnings_{data['symbol']}_{earnings_date.strftime('%Y%m%d')}",
                            event_type=EventType.EARNINGS,
                            title=f"{data['symbol']} Earnings",
                            date=earnings_date,
                            impact=EventImpact.MEDIUM,
                            symbol=data['symbol'],
                            description=f"Quarterly earnings report for {data['symbol']}"
                        )
                        if event_types is None or EventType.EARNINGS in event_types:
                            events.append(event)

        # Sort by date
        events.sort(key=lambda x: x.date)

        return events

    def get_events_for_date(self, date: datetime) -> List[EconomicEvent]:
        """Get all events for a specific date"""
        target_date = date.date()
        return [e for e in self._events if e.date.date() == target_date]

    def get_next_event(self, event_type: EventType) -> Optional[EconomicEvent]:
        """Get the next upcoming event of a specific type"""
        now = datetime.now()
        upcoming = [e for e in self._events if e.event_type == event_type and e.date >= now]
        return min(upcoming, key=lambda x: x.date) if upcoming else None

    def add_custom_event(
        self,
        title: str,
        date: datetime,
        impact: EventImpact = EventImpact.MEDIUM,
        description: str = "",
        symbol: Optional[str] = None
    ) -> EconomicEvent:
        """
        Add a custom event to the calendar

        Args:
            title: Event title
            date: Event date
            impact: Impact level
            description: Event description
            symbol: Related stock symbol

        Returns:
            Created event
        """
        event = EconomicEvent(
            event_id=f"custom_{date.strftime('%Y%m%d')}_{len(self._events)}",
            event_type=EventType.CUSTOM,
            title=title,
            date=date,
            impact=impact,
            symbol=symbol,
            description=description
        )
        self._events.append(event)
        self._save_data()
        return event

    def remove_event(self, event_id: str) -> bool:
        """Remove an event by ID"""
        original_count = len(self._events)
        self._events = [e for e in self._events if e.event_id != event_id]
        if len(self._events) < original_count:
            self._save_data()
            return True
        return False

    def get_calendar_dataframe(
        self,
        days_ahead: int = 30,
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get calendar as DataFrame for visualization

        Args:
            days_ahead: Number of days to look ahead
            symbols: Symbols to include earnings for

        Returns:
            DataFrame with event data
        """
        events = self.get_upcoming_events(
            days_ahead=days_ahead,
            include_earnings=True,
            symbols=symbols
        )

        if not events:
            return pd.DataFrame()

        data = []
        for event in events:
            data.append({
                'Date': event.date.strftime('%Y-%m-%d'),
                'Time': event.date.strftime('%H:%M') if event.date.hour != 0 else 'TBD',
                'Event': event.title,
                'Type': event.event_type.value.upper(),
                'Impact': event.impact.value.upper(),
                'Symbol': event.symbol or '-',
                'Description': event.description,
                'Days Until': (event.date - datetime.now()).days
            })

        return pd.DataFrame(data)

    def has_earnings_soon(
        self,
        symbol: str,
        days_threshold: int = 7
    ) -> Tuple[bool, Optional[int]]:
        """
        Check if a symbol has earnings coming soon

        Args:
            symbol: Stock symbol
            days_threshold: Days to consider "soon"

        Returns:
            Tuple of (has_earnings_soon, days_until)
        """
        earnings = self.get_earnings_date(symbol)
        if not earnings or not earnings.get('days_until'):
            return False, None

        days_until = earnings['days_until']
        return days_until <= days_threshold and days_until >= 0, days_until

    def get_week_events(self, week_offset: int = 0) -> Dict[str, List[EconomicEvent]]:
        """
        Get events organized by day for a specific week

        Args:
            week_offset: 0 for current week, 1 for next week, etc.

        Returns:
            Dict with day names as keys and events as values
        """
        today = datetime.now()
        start_of_week = today - timedelta(days=today.weekday())
        start_of_week = start_of_week + timedelta(weeks=week_offset)
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)

        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        week_events = {day: [] for day in days}

        for i, day in enumerate(days):
            day_date = start_of_week + timedelta(days=i)
            for event in self._events:
                if event.date.date() == day_date.date():
                    week_events[day].append(event)

        return week_events

    def get_high_impact_events(self, days_ahead: int = 7) -> List[EconomicEvent]:
        """Get only high impact events"""
        return [
            e for e in self.get_upcoming_events(days_ahead=days_ahead)
            if e.impact == EventImpact.HIGH
        ]

    def get_earnings_warning(self, symbol: str) -> Optional[str]:
        """
        Get a warning message if earnings are coming soon

        Args:
            symbol: Stock symbol

        Returns:
            Warning message or None
        """
        has_soon, days = self.has_earnings_soon(symbol, days_threshold=7)
        if has_soon:
            if days == 0:
                return f"⚠️ {symbol} reports earnings TODAY"
            elif days == 1:
                return f"⚠️ {symbol} reports earnings TOMORROW"
            else:
                return f"⚠️ {symbol} reports earnings in {days} days"
        return None


# Singleton instance
economic_calendar = EconomicCalendar()
