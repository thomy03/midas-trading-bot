"""
Integration tests for Economic Calendar

Tests economic event tracking and earnings dates:
- FOMC meeting dates
- CPI release dates
- Jobs report dates
- Earnings date fetching
- Event filtering and visualization data
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.utils.economic_calendar import (
    EconomicCalendar,
    EconomicEvent,
    EventType,
    EventImpact,
    economic_calendar,
    FOMC_DATES_2024,
    FOMC_DATES_2025,
    CPI_DATES_2024,
    CPI_DATES_2025
)
from src.utils.interactive_chart import InteractiveChartBuilder


class TestEventType:
    """Tests for EventType enum"""

    @pytest.mark.integration
    def test_event_types_defined(self):
        """Test all event types are defined"""
        expected = ['earnings', 'fomc', 'cpi', 'jobs', 'gdp', 'retail_sales', 'custom']
        for event_type in expected:
            assert EventType(event_type) is not None

    @pytest.mark.integration
    def test_event_impact_levels(self):
        """Test impact levels are defined"""
        assert EventImpact.HIGH.value == 'high'
        assert EventImpact.MEDIUM.value == 'medium'
        assert EventImpact.LOW.value == 'low'


class TestEconomicEvent:
    """Tests for EconomicEvent dataclass"""

    @pytest.fixture
    def sample_event(self):
        """Sample event for testing"""
        return EconomicEvent(
            event_id='test_001',
            event_type=EventType.FOMC,
            title='FOMC Interest Rate Decision',
            date=datetime(2025, 1, 29, 14, 0),
            impact=EventImpact.HIGH,
            description='Federal Reserve rate decision'
        )

    @pytest.mark.integration
    def test_event_attributes(self, sample_event):
        """Test event has correct attributes"""
        assert sample_event.event_id == 'test_001'
        assert sample_event.event_type == EventType.FOMC
        assert sample_event.impact == EventImpact.HIGH
        assert sample_event.symbol is None

    @pytest.mark.integration
    def test_event_to_dict(self, sample_event):
        """Test event serialization"""
        data = sample_event.to_dict()

        assert data['event_id'] == 'test_001'
        assert data['event_type'] == 'fomc'
        assert data['impact'] == 'high'
        assert 'date' in data

    @pytest.mark.integration
    def test_event_from_dict(self, sample_event):
        """Test event deserialization"""
        data = sample_event.to_dict()
        restored = EconomicEvent.from_dict(data)

        assert restored.event_id == sample_event.event_id
        assert restored.event_type == sample_event.event_type
        assert restored.title == sample_event.title

    @pytest.mark.integration
    def test_earnings_event_with_symbol(self):
        """Test earnings event has symbol"""
        event = EconomicEvent(
            event_id='earnings_aapl',
            event_type=EventType.EARNINGS,
            title='AAPL Earnings',
            date=datetime.now() + timedelta(days=5),
            impact=EventImpact.MEDIUM,
            symbol='AAPL'
        )

        assert event.symbol == 'AAPL'
        assert event.event_type == EventType.EARNINGS


class TestFixedDates:
    """Tests for fixed economic dates"""

    @pytest.mark.integration
    def test_fomc_dates_2024_exist(self):
        """Test FOMC dates for 2024 are defined"""
        assert len(FOMC_DATES_2024) > 0
        for date in FOMC_DATES_2024:
            assert date.year == 2024
            assert isinstance(date, datetime)

    @pytest.mark.integration
    def test_fomc_dates_2025_exist(self):
        """Test FOMC dates for 2025 are defined"""
        assert len(FOMC_DATES_2025) > 0
        for date in FOMC_DATES_2025:
            assert date.year == 2025

    @pytest.mark.integration
    def test_cpi_dates_exist(self):
        """Test CPI dates are defined"""
        assert len(CPI_DATES_2024) == 12  # One per month
        assert len(CPI_DATES_2025) == 12


class TestEconomicCalendar:
    """Tests for EconomicCalendar class"""

    @pytest.fixture
    def calendar(self, tmp_path):
        """Fresh calendar instance with temp directory"""
        return EconomicCalendar(data_dir=str(tmp_path))

    @pytest.mark.integration
    def test_calendar_initialization(self, calendar):
        """Test calendar initializes correctly"""
        assert calendar is not None
        assert len(calendar._events) > 0  # Should have fixed events

    @pytest.mark.integration
    def test_fixed_events_loaded(self, calendar):
        """Test fixed events (FOMC, CPI, Jobs) are loaded"""
        event_types = [e.event_type for e in calendar._events]

        assert EventType.FOMC in event_types
        assert EventType.CPI in event_types
        assert EventType.JOBS in event_types

    @pytest.mark.integration
    def test_get_upcoming_events_no_earnings(self, calendar):
        """Test getting upcoming events without earnings"""
        events = calendar.get_upcoming_events(
            days_ahead=90,
            include_earnings=False
        )

        assert isinstance(events, list)
        # All events should be fixed types (no earnings)
        for event in events:
            assert event.event_type != EventType.EARNINGS

    @pytest.mark.integration
    def test_get_upcoming_events_with_filter(self, calendar):
        """Test filtering events by type"""
        events = calendar.get_upcoming_events(
            days_ahead=90,
            event_types=[EventType.FOMC],
            include_earnings=False
        )

        for event in events:
            assert event.event_type == EventType.FOMC

    @pytest.mark.integration
    def test_get_next_event(self, calendar):
        """Test getting next event of specific type"""
        next_fomc = calendar.get_next_event(EventType.FOMC)

        if next_fomc:
            assert next_fomc.event_type == EventType.FOMC
            assert next_fomc.date >= datetime.now()

    @pytest.mark.integration
    def test_add_custom_event(self, calendar):
        """Test adding custom event"""
        event = calendar.add_custom_event(
            title='Custom Test Event',
            date=datetime.now() + timedelta(days=10),
            impact=EventImpact.LOW,
            description='Test description'
        )

        assert event.event_type == EventType.CUSTOM
        assert event.title == 'Custom Test Event'
        assert event in calendar._events

    @pytest.mark.integration
    def test_remove_event(self, calendar):
        """Test removing event"""
        # Add an event
        event = calendar.add_custom_event(
            title='To Remove',
            date=datetime.now() + timedelta(days=5)
        )

        event_id = event.event_id
        count_before = len(calendar._events)

        # Remove it
        result = calendar.remove_event(event_id)

        assert result is True
        assert len(calendar._events) == count_before - 1

    @pytest.mark.integration
    def test_get_events_for_date(self, calendar):
        """Test getting events for specific date"""
        # Use a known FOMC date
        if FOMC_DATES_2025:
            test_date = FOMC_DATES_2025[0]
            events = calendar.get_events_for_date(test_date)

            # May have events on that date
            for event in events:
                assert event.date.date() == test_date.date()

    @pytest.mark.integration
    def test_get_week_events(self, calendar):
        """Test getting weekly event view"""
        week_events = calendar.get_week_events(week_offset=0)

        assert isinstance(week_events, dict)
        expected_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in expected_days:
            assert day in week_events
            assert isinstance(week_events[day], list)

    @pytest.mark.integration
    def test_get_high_impact_events(self, calendar):
        """Test getting high impact events"""
        events = calendar.get_high_impact_events(days_ahead=90)

        for event in events:
            assert event.impact == EventImpact.HIGH

    @pytest.mark.integration
    def test_get_calendar_dataframe(self, calendar):
        """Test getting DataFrame for visualization"""
        df = calendar.get_calendar_dataframe(days_ahead=90)

        if not df.empty:
            assert 'Date' in df.columns
            assert 'Event' in df.columns
            assert 'Type' in df.columns
            assert 'Impact' in df.columns


class TestEarningsIntegration:
    """Tests for earnings date fetching"""

    @pytest.fixture
    def calendar(self, tmp_path):
        """Fresh calendar instance"""
        return EconomicCalendar(data_dir=str(tmp_path))

    @pytest.mark.integration
    def test_has_earnings_soon(self, calendar):
        """Test earnings soon check"""
        # Mock the get_earnings_date method
        with patch.object(calendar, 'get_earnings_date') as mock_earnings:
            mock_earnings.return_value = {
                'symbol': 'AAPL',
                'earnings_date': datetime.now() + timedelta(days=3),
                'days_until': 3
            }

            has_soon, days = calendar.has_earnings_soon('AAPL', days_threshold=7)

            assert has_soon is True
            assert days == 3

    @pytest.mark.integration
    def test_has_earnings_not_soon(self, calendar):
        """Test when earnings are not soon"""
        with patch.object(calendar, 'get_earnings_date') as mock_earnings:
            mock_earnings.return_value = {
                'symbol': 'AAPL',
                'earnings_date': datetime.now() + timedelta(days=30),
                'days_until': 30
            }

            has_soon, days = calendar.has_earnings_soon('AAPL', days_threshold=7)

            assert has_soon is False
            assert days == 30

    @pytest.mark.integration
    def test_get_earnings_warning(self, calendar):
        """Test earnings warning message"""
        with patch.object(calendar, 'has_earnings_soon') as mock_soon:
            mock_soon.return_value = (True, 2)

            warning = calendar.get_earnings_warning('AAPL')

            assert warning is not None
            assert 'AAPL' in warning
            assert '2 days' in warning

    @pytest.mark.integration
    def test_get_earnings_warning_today(self, calendar):
        """Test earnings warning for today"""
        with patch.object(calendar, 'has_earnings_soon') as mock_soon:
            mock_soon.return_value = (True, 0)

            warning = calendar.get_earnings_warning('AAPL')

            assert 'TODAY' in warning

    @pytest.mark.integration
    def test_get_earnings_warning_tomorrow(self, calendar):
        """Test earnings warning for tomorrow"""
        with patch.object(calendar, 'has_earnings_soon') as mock_soon:
            mock_soon.return_value = (True, 1)

            warning = calendar.get_earnings_warning('AAPL')

            assert 'TOMORROW' in warning

    @pytest.mark.integration
    def test_get_earnings_dates_batch(self, calendar):
        """Test batch earnings date fetching"""
        with patch.object(calendar, 'get_earnings_date') as mock_earnings:
            mock_earnings.side_effect = [
                {'symbol': 'AAPL', 'earnings_date': datetime.now() + timedelta(days=5), 'days_until': 5},
                {'symbol': 'MSFT', 'earnings_date': datetime.now() + timedelta(days=10), 'days_until': 10},
                None  # GOOGL has no data
            ]

            results = calendar.get_earnings_dates_batch(['AAPL', 'MSFT', 'GOOGL'], use_cache=False)

            assert len(results) == 2  # Only 2 have data
            symbols = [r['symbol'] for r in results]
            assert 'AAPL' in symbols
            assert 'MSFT' in symbols


class TestCalendarVisualization:
    """Tests for calendar visualization methods"""

    @pytest.fixture
    def chart_builder(self):
        return InteractiveChartBuilder()

    @pytest.fixture
    def mock_events(self):
        """Mock events for visualization testing"""
        now = datetime.now()
        return [
            EconomicEvent(
                event_id='fomc_1',
                event_type=EventType.FOMC,
                title='FOMC Meeting',
                date=now + timedelta(days=5),
                impact=EventImpact.HIGH
            ),
            EconomicEvent(
                event_id='cpi_1',
                event_type=EventType.CPI,
                title='CPI Report',
                date=now + timedelta(days=10),
                impact=EventImpact.HIGH
            ),
            EconomicEvent(
                event_id='earnings_1',
                event_type=EventType.EARNINGS,
                title='AAPL Earnings',
                date=now + timedelta(days=7),
                impact=EventImpact.MEDIUM,
                symbol='AAPL'
            )
        ]

    @pytest.mark.integration
    def test_create_calendar_timeline(self, chart_builder, mock_events):
        """Test timeline chart creation"""
        fig = chart_builder.create_calendar_timeline(mock_events, height=400)

        assert fig is not None
        assert len(fig.data) > 0

    @pytest.mark.integration
    def test_create_calendar_timeline_empty(self, chart_builder):
        """Test timeline with empty events"""
        fig = chart_builder.create_calendar_timeline([], height=400)

        assert fig is not None
        # Should have annotation about no events
        assert len(fig.layout.annotations) > 0

    @pytest.mark.integration
    def test_create_calendar_table(self, chart_builder, mock_events):
        """Test table chart creation"""
        fig = chart_builder.create_calendar_table(mock_events, height=400)

        assert fig is not None
        assert len(fig.data) > 0

    @pytest.mark.integration
    def test_create_earnings_calendar(self, chart_builder):
        """Test earnings calendar creation"""
        earnings_data = [
            {'symbol': 'AAPL', 'earnings_date': datetime.now() + timedelta(days=5), 'days_until': 5},
            {'symbol': 'MSFT', 'earnings_date': datetime.now() + timedelta(days=10), 'days_until': 10}
        ]

        fig = chart_builder.create_earnings_calendar(earnings_data, height=400)

        assert fig is not None
        assert len(fig.data) > 0

    @pytest.mark.integration
    def test_create_week_calendar(self, chart_builder):
        """Test weekly calendar view"""
        week_events = {
            'Monday': [],
            'Tuesday': [],
            'Wednesday': [],
            'Thursday': [],
            'Friday': [],
            'Saturday': [],
            'Sunday': []
        }

        fig = chart_builder.create_week_calendar(week_events, height=300)

        assert fig is not None

    @pytest.mark.integration
    def test_create_event_impact_chart(self, chart_builder, mock_events):
        """Test impact distribution chart"""
        fig = chart_builder.create_event_impact_chart(mock_events, height=300)

        assert fig is not None
        assert len(fig.data) > 0


class TestSingletonInstance:
    """Tests for singleton calendar instance"""

    @pytest.mark.integration
    def test_singleton_exists(self):
        """Test singleton is available"""
        assert economic_calendar is not None
        assert isinstance(economic_calendar, EconomicCalendar)

    @pytest.mark.integration
    def test_singleton_has_events(self):
        """Test singleton has fixed events loaded"""
        assert len(economic_calendar._events) > 0


class TestCaching:
    """Tests for caching behavior"""

    @pytest.fixture
    def calendar(self, tmp_path):
        """Fresh calendar with temp directory"""
        return EconomicCalendar(data_dir=str(tmp_path))

    @pytest.mark.integration
    def test_earnings_cache_used(self, calendar):
        """Test that earnings cache is used"""
        # Add to cache
        test_data = {
            'symbol': 'AAPL',
            'earnings_date': datetime.now() + timedelta(days=10),
            'days_until': 10
        }
        calendar._earnings_cache['AAPL'] = test_data
        calendar._cache_time = datetime.now()

        # Should return cached data
        result = calendar.get_earnings_date('AAPL', use_cache=True)

        assert result == test_data

    @pytest.mark.integration
    def test_cache_bypass(self, calendar):
        """Test cache can be bypassed"""
        # Add to cache
        calendar._earnings_cache['AAPL'] = {'old': 'data'}
        calendar._cache_time = datetime.now()

        # Mock the actual fetch
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.calendar = None

            # Bypass cache should try to fetch
            result = calendar.get_earnings_date('AAPL', use_cache=False)

            # yfinance was called
            mock_ticker.assert_called_once_with('AAPL')


class TestPersistence:
    """Tests for data persistence"""

    @pytest.mark.integration
    def test_custom_events_saved(self, tmp_path):
        """Test custom events are persisted"""
        calendar1 = EconomicCalendar(data_dir=str(tmp_path))

        # Add custom event
        calendar1.add_custom_event(
            title='Persisted Event',
            date=datetime.now() + timedelta(days=20),
            impact=EventImpact.HIGH
        )

        # Create new calendar instance
        calendar2 = EconomicCalendar(data_dir=str(tmp_path))

        # Should have the custom event
        custom_events = [e for e in calendar2._events if e.event_type == EventType.CUSTOM]
        assert len(custom_events) >= 1
        assert any(e.title == 'Persisted Event' for e in custom_events)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
