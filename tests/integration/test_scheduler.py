"""
Integration tests for Background Scanner and Scheduler

Tests cover:
- ScanStatus enum
- ScanState dataclass
- BackgroundScanner basic operations
- ScheduleConfig dataclass
- ScanScheduler configuration and scheduling logic
"""
import pytest
import tempfile
import json
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.utils.background_scanner import (
    ScanStatus,
    ScanState,
    BackgroundScanner,
    ScheduleType,
    ScheduleConfig,
    SchedulerState,
    ScanScheduler,
    background_scanner,
    scan_scheduler
)


# ================== ScanStatus Tests ==================

class TestScanStatus:
    """Tests for ScanStatus enum"""

    def test_status_values(self):
        """Test all status values exist"""
        assert ScanStatus.IDLE.value == "idle"
        assert ScanStatus.RUNNING.value == "running"
        assert ScanStatus.PAUSED.value == "paused"
        assert ScanStatus.COMPLETED.value == "completed"
        assert ScanStatus.CANCELLED.value == "cancelled"
        assert ScanStatus.ERROR.value == "error"

    def test_status_from_string(self):
        """Test creating status from string"""
        assert ScanStatus("idle") == ScanStatus.IDLE
        assert ScanStatus("running") == ScanStatus.RUNNING


# ================== ScanState Tests ==================

class TestScanState:
    """Tests for ScanState dataclass"""

    def test_default_state(self):
        """Test default state values"""
        state = ScanState()

        assert state.status == ScanStatus.IDLE
        assert state.total_stocks == 0
        assert state.completed_count == 0
        assert state.alerts == []
        assert state.pending_stocks == []
        assert state.start_time is None
        assert state.end_time is None
        assert state.error_message is None
        assert state.current_symbol == ""
        assert state.scan_id == 0
        assert state.pause_requested is False
        assert state.cancel_requested is False

    def test_state_with_values(self):
        """Test state with custom values"""
        now = datetime.now()
        state = ScanState(
            status=ScanStatus.RUNNING,
            total_stocks=100,
            completed_count=50,
            alerts=[{'symbol': 'AAPL'}],
            start_time=now,
            current_symbol='MSFT'
        )

        assert state.status == ScanStatus.RUNNING
        assert state.total_stocks == 100
        assert state.completed_count == 50
        assert len(state.alerts) == 1
        assert state.current_symbol == 'MSFT'


# ================== BackgroundScanner Tests ==================

class TestBackgroundScanner:
    """Tests for BackgroundScanner class"""

    def test_singleton_exists(self):
        """Test singleton instance exists"""
        assert background_scanner is not None
        assert isinstance(background_scanner, BackgroundScanner)

    def test_get_state(self):
        """Test getting current state"""
        scanner = BackgroundScanner()
        state = scanner.get_state()

        assert isinstance(state, ScanState)
        assert isinstance(state.status, ScanStatus)

    def test_is_running_false_when_idle(self):
        """Test is_running returns False when idle"""
        scanner = BackgroundScanner()
        scanner.reset()

        assert scanner.is_running() is False

    def test_is_paused_false_when_idle(self):
        """Test is_paused returns False when idle"""
        scanner = BackgroundScanner()
        scanner.reset()

        assert scanner.is_paused() is False

    def test_reset(self):
        """Test reset clears state"""
        scanner = BackgroundScanner()
        scanner.reset()

        state = scanner.get_state()
        assert state.status == ScanStatus.IDLE
        assert state.total_stocks == 0
        assert state.completed_count == 0

    def test_get_progress_info(self):
        """Test getting progress information"""
        scanner = BackgroundScanner()
        scanner.reset()

        progress = scanner.get_progress_info()

        assert 'status' in progress
        assert 'completed' in progress
        assert 'total' in progress
        assert 'alerts_count' in progress
        assert 'elapsed_seconds' in progress
        assert 'rate_per_second' in progress
        assert 'progress_pct' in progress

    def test_progress_pct_zero_when_no_stocks(self):
        """Test progress percentage is 0 when no stocks"""
        scanner = BackgroundScanner()
        scanner.reset()

        progress = scanner.get_progress_info()
        assert progress['progress_pct'] == 0


# ================== ScheduleType Tests ==================

class TestScheduleType:
    """Tests for ScheduleType enum"""

    def test_schedule_types(self):
        """Test all schedule types exist"""
        assert ScheduleType.ONCE.value == "once"
        assert ScheduleType.DAILY.value == "daily"
        assert ScheduleType.HOURLY.value == "hourly"
        assert ScheduleType.WEEKLY.value == "weekly"
        assert ScheduleType.INTERVAL.value == "interval"


# ================== ScheduleConfig Tests ==================

class TestScheduleConfig:
    """Tests for ScheduleConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ScheduleConfig()

        assert config.enabled is False
        assert config.schedule_type == "daily"
        assert config.time_of_day == "09:30"
        assert config.days_of_week == [0, 1, 2, 3, 4]  # Mon-Fri
        assert config.interval_minutes == 60
        assert config.markets == ["NASDAQ", "SP500"]
        assert config.notify_on_completion is True
        assert config.notify_on_signals is True
        assert config.min_signal_priority == "medium"

    def test_config_with_values(self):
        """Test configuration with custom values"""
        config = ScheduleConfig(
            enabled=True,
            schedule_type="weekly",
            time_of_day="14:00",
            days_of_week=[0, 2, 4],  # Mon, Wed, Fri
            markets=["CRYPTO"]
        )

        assert config.enabled is True
        assert config.schedule_type == "weekly"
        assert config.time_of_day == "14:00"
        assert config.days_of_week == [0, 2, 4]
        assert config.markets == ["CRYPTO"]

    def test_config_to_dict(self):
        """Test converting config to dictionary"""
        config = ScheduleConfig(enabled=True, schedule_type="hourly")
        data = config.to_dict()

        assert isinstance(data, dict)
        assert data['enabled'] is True
        assert data['schedule_type'] == "hourly"
        assert 'markets' in data
        assert 'days_of_week' in data

    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        data = {
            'enabled': True,
            'schedule_type': 'interval',
            'interval_minutes': 30,
            'markets': ['EUROPE', 'DAX']
        }

        config = ScheduleConfig.from_dict(data)

        assert config.enabled is True
        assert config.schedule_type == 'interval'
        assert config.interval_minutes == 30
        assert 'EUROPE' in config.markets

    def test_config_from_dict_ignores_unknown_keys(self):
        """Test that unknown keys are ignored"""
        data = {
            'enabled': True,
            'unknown_key': 'value',
            'another_unknown': 123
        }

        config = ScheduleConfig.from_dict(data)
        assert config.enabled is True


# ================== SchedulerState Tests ==================

class TestSchedulerState:
    """Tests for SchedulerState dataclass"""

    def test_default_state(self):
        """Test default state values"""
        state = SchedulerState()

        assert state.running is False
        assert state.next_scheduled_run is None
        assert state.last_run_time is None
        assert state.last_run_alerts == 0
        assert state.consecutive_errors == 0
        assert state.total_runs == 0


# ================== ScanScheduler Tests ==================

class TestScanScheduler:
    """Tests for ScanScheduler class"""

    def test_singleton_exists(self):
        """Test singleton instance exists"""
        assert scan_scheduler is not None
        assert isinstance(scan_scheduler, ScanScheduler)

    def test_scheduler_init(self):
        """Test scheduler initialization"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)
            assert scheduler.config is not None
            assert scheduler.state is not None
            assert scheduler.state.running is False
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_update_config(self):
        """Test updating scheduler configuration"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)
            scheduler.update_config(
                enabled=True,
                schedule_type="hourly",
                markets=["CRYPTO"]
            )

            assert scheduler.config.enabled is True
            assert scheduler.config.schedule_type == "hourly"
            assert "CRYPTO" in scheduler.config.markets
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_get_status(self):
        """Test getting scheduler status"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)
            status = scheduler.get_status()

            assert 'running' in status
            assert 'enabled' in status
            assert 'schedule_type' in status
            assert 'markets' in status
            assert 'next_run' in status
            assert 'last_run' in status
            assert 'total_runs' in status
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_calculate_next_run_daily(self):
        """Test calculating next run for daily schedule"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)
            scheduler.config.schedule_type = "daily"
            scheduler.config.time_of_day = "09:30"
            scheduler.config.days_of_week = [0, 1, 2, 3, 4]  # Weekdays

            next_run = scheduler._calculate_next_run()

            assert next_run is not None
            assert isinstance(next_run, datetime)
            assert next_run > datetime.now()
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_calculate_next_run_hourly(self):
        """Test calculating next run for hourly schedule"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)
            scheduler.config.schedule_type = "hourly"

            next_run = scheduler._calculate_next_run()

            assert next_run is not None
            assert next_run > datetime.now()
            assert next_run.minute == 0
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_calculate_next_run_interval(self):
        """Test calculating next run for interval schedule"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)
            scheduler.config.schedule_type = "interval"
            scheduler.config.interval_minutes = 30

            next_run = scheduler._calculate_next_run()

            assert next_run is not None
            # Should be approximately 30 minutes from now
            diff = (next_run - datetime.now()).total_seconds()
            assert 29 * 60 <= diff <= 31 * 60
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_set_screen_function(self):
        """Test setting screen function"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)
            mock_func = Mock()
            scheduler.set_screen_function(mock_func)

            assert scheduler._screen_function == mock_func
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_set_stocks_provider(self):
        """Test setting stocks provider"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)
            mock_provider = Mock(return_value=[{'symbol': 'AAPL'}])
            scheduler.set_stocks_provider(mock_provider)

            assert scheduler._stocks_provider == mock_provider
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_set_notification_manager(self):
        """Test setting notification manager"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)
            mock_manager = Mock()
            scheduler.set_notification_manager(mock_manager)

            assert scheduler._notification_manager == mock_manager
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_start_fails_without_screen_function(self):
        """Test start fails without screen function"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)
            scheduler._screen_function = None
            scheduler._stocks_provider = Mock()

            result = scheduler.start()
            assert result is False
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_start_fails_without_stocks_provider(self):
        """Test start fails without stocks provider"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)
            scheduler._screen_function = Mock()
            scheduler._stocks_provider = None

            result = scheduler.start()
            assert result is False
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_trigger_now_fails_when_scan_running(self):
        """Test trigger_now fails when scan is already running"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)
            scheduler._screen_function = Mock()
            scheduler._stocks_provider = Mock()

            # Mock scanner as running
            with patch.object(scheduler._scanner, 'is_running', return_value=True):
                result = scheduler.trigger_now()
                assert result is False
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_format_duration_seconds(self):
        """Test duration formatting for seconds"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)

            start = datetime.now()
            end = start + timedelta(seconds=45)

            result = scheduler._format_duration(start, end)
            assert result == "45s"
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_format_duration_minutes(self):
        """Test duration formatting for minutes"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)

            start = datetime.now()
            end = start + timedelta(minutes=5)

            result = scheduler._format_duration(start, end)
            assert "5.0m" in result
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_format_duration_hours(self):
        """Test duration formatting for hours"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)

            start = datetime.now()
            end = start + timedelta(hours=2)

            result = scheduler._format_duration(start, end)
            assert "2.0h" in result
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_format_duration_none(self):
        """Test duration formatting with None values"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)

            result = scheduler._format_duration(None, None)
            assert result == "N/A"
        finally:
            Path(config_path).unlink(missing_ok=True)


# ================== Config Persistence Tests ==================

class TestConfigPersistence:
    """Tests for configuration persistence"""

    def test_config_saved_to_file(self):
        """Test configuration is saved to file"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)
            scheduler.update_config(
                enabled=True,
                schedule_type="weekly",
                markets=["DAX", "CAC40"]
            )

            # Read file and verify
            with open(config_path, 'r') as f:
                saved = json.load(f)

            assert saved['enabled'] is True
            assert saved['schedule_type'] == "weekly"
            assert "DAX" in saved['markets']
        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_config_loaded_from_file(self):
        """Test configuration is loaded from file"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            config_data = {
                'enabled': True,
                'schedule_type': 'interval',
                'interval_minutes': 45,
                'markets': ['CRYPTO']
            }
            json.dump(config_data, f)
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)

            assert scheduler.config.enabled is True
            assert scheduler.config.schedule_type == 'interval'
            assert scheduler.config.interval_minutes == 45
            assert 'CRYPTO' in scheduler.config.markets
        finally:
            Path(config_path).unlink(missing_ok=True)


# ================== Integration Tests ==================

class TestSchedulerIntegration:
    """Integration tests for scheduler system"""

    def test_full_config_update_cycle(self):
        """Test complete configuration update cycle"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            # Create scheduler
            scheduler = ScanScheduler(config_path=config_path)

            # Update config
            scheduler.update_config(
                enabled=True,
                schedule_type="daily",
                time_of_day="10:00",
                markets=["NASDAQ", "SP500"],
                notify_on_completion=True,
                notify_on_signals=False
            )

            # Verify status
            status = scheduler.get_status()
            assert status['enabled'] is True
            assert status['schedule_type'] == "daily"
            assert "NASDAQ" in status['markets']
            assert status['notify_on_completion'] is True
            assert status['notify_on_signals'] is False

            # Create new scheduler instance (simulating restart)
            scheduler2 = ScanScheduler(config_path=config_path)

            # Verify config persisted
            assert scheduler2.config.enabled is True
            assert scheduler2.config.schedule_type == "daily"
            assert scheduler2.config.time_of_day == "10:00"

        finally:
            Path(config_path).unlink(missing_ok=True)

    def test_scheduler_with_mock_scan(self):
        """Test scheduler with mocked scan execution"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            scheduler = ScanScheduler(config_path=config_path)

            # Mock functions
            mock_screen = Mock(return_value={'symbol': 'AAPL', 'recommendation': 'BUY'})
            mock_provider = Mock(return_value=[{'symbol': 'AAPL', 'name': 'Apple'}])

            scheduler.set_screen_function(mock_screen)
            scheduler.set_stocks_provider(mock_provider)

            # Trigger scan
            with patch.object(scheduler._scanner, 'is_running', return_value=False):
                result = scheduler.trigger_now()
                assert result is True

        finally:
            Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
