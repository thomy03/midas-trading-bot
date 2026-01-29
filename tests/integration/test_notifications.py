"""
Integration tests for Notification Manager

Tests cover:
- NotificationConfig dataclass
- NotificationPriority and NotificationChannel enums
- TelegramNotifier formatting
- EmailNotifier formatting
- NotificationManager configuration and filtering
- Priority determination
- Quiet hours logic
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import asyncio

from src.utils.notification_manager import (
    NotificationConfig,
    NotificationChannel,
    NotificationPriority,
    Notification,
    TelegramNotifier,
    EmailNotifier,
    NotificationManager
)


# ================== NotificationConfig Tests ==================

class TestNotificationConfig:
    """Tests for NotificationConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = NotificationConfig()

        assert config.telegram_enabled is False
        assert config.email_enabled is False
        assert config.in_app_enabled is True
        assert config.min_priority == "medium"
        assert config.alert_types == ["STRONG_BUY", "BUY"]
        assert config.quiet_hours_start is None
        assert config.quiet_hours_end is None

    def test_config_with_values(self):
        """Test configuration with custom values"""
        config = NotificationConfig(
            telegram_enabled=True,
            telegram_bot_token="test_token",
            telegram_chat_id="12345",
            email_enabled=True,
            min_priority="high",
            alert_types=["STRONG_BUY"]
        )

        assert config.telegram_enabled is True
        assert config.telegram_bot_token == "test_token"
        assert config.telegram_chat_id == "12345"
        assert config.min_priority == "high"
        assert config.alert_types == ["STRONG_BUY"]

    def test_config_to_dict(self):
        """Test converting config to dictionary"""
        config = NotificationConfig(
            telegram_enabled=True,
            min_priority="high"
        )

        data = config.to_dict()

        assert isinstance(data, dict)
        assert data['telegram_enabled'] is True
        assert data['min_priority'] == "high"
        assert 'smtp_server' in data

    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        data = {
            'telegram_enabled': True,
            'telegram_bot_token': 'test_token',
            'min_priority': 'urgent',
            'alert_types': ['STRONG_BUY', 'BUY', 'WATCH']
        }

        config = NotificationConfig.from_dict(data)

        assert config.telegram_enabled is True
        assert config.telegram_bot_token == 'test_token'
        assert config.min_priority == 'urgent'
        assert 'WATCH' in config.alert_types

    def test_config_from_dict_ignores_unknown_keys(self):
        """Test that unknown keys are ignored"""
        data = {
            'telegram_enabled': True,
            'unknown_key': 'value',
            'another_unknown': 123
        }

        config = NotificationConfig.from_dict(data)
        assert config.telegram_enabled is True


# ================== Enum Tests ==================

class TestEnums:
    """Tests for notification enums"""

    def test_notification_channel_values(self):
        """Test NotificationChannel enum values"""
        assert NotificationChannel.TELEGRAM.value == "telegram"
        assert NotificationChannel.EMAIL.value == "email"
        assert NotificationChannel.IN_APP.value == "in_app"

    def test_notification_priority_values(self):
        """Test NotificationPriority enum values"""
        assert NotificationPriority.LOW.value == "low"
        assert NotificationPriority.MEDIUM.value == "medium"
        assert NotificationPriority.HIGH.value == "high"
        assert NotificationPriority.URGENT.value == "urgent"


# ================== Notification Tests ==================

class TestNotification:
    """Tests for Notification dataclass"""

    def test_notification_creation(self):
        """Test creating a notification"""
        notification = Notification(
            id="test_123",
            title="Test Alert",
            message="This is a test",
            priority=NotificationPriority.HIGH,
            channel=NotificationChannel.TELEGRAM
        )

        assert notification.id == "test_123"
        assert notification.title == "Test Alert"
        assert notification.sent is False
        assert notification.error is None

    def test_notification_to_dict(self):
        """Test converting notification to dictionary"""
        notification = Notification(
            id="test_123",
            title="Test Alert",
            message="This is a test",
            priority=NotificationPriority.HIGH,
            channel=NotificationChannel.TELEGRAM,
            sent=True
        )

        data = notification.to_dict()

        assert data['id'] == "test_123"
        assert data['priority'] == "high"
        assert data['channel'] == "telegram"
        assert data['sent'] is True


# ================== TelegramNotifier Tests ==================

class TestTelegramNotifier:
    """Tests for TelegramNotifier"""

    def test_init(self):
        """Test TelegramNotifier initialization"""
        notifier = TelegramNotifier("test_token", "12345")

        assert notifier.bot_token == "test_token"
        assert notifier.chat_id == "12345"
        assert "test_token" in notifier.api_url

    def test_format_alert_basic(self):
        """Test formatting a basic alert"""
        notifier = TelegramNotifier("token", "chat")
        alert = {
            'symbol': 'AAPL',
            'recommendation': 'STRONG_BUY',
            'current_price': 175.50,
            'support_level': 170.00,
            'confidence_score': 85,
            'distance_to_support_pct': -3.1
        }

        message = notifier.format_alert(alert)

        assert 'AAPL' in message
        assert 'STRONG_BUY' in message
        assert '175.50' in message
        assert '170.00' in message
        assert '85' in message

    def test_format_alert_with_earnings(self):
        """Test formatting alert with earnings warning"""
        notifier = TelegramNotifier("token", "chat")
        alert = {
            'symbol': 'NVDA',
            'recommendation': 'BUY',
            'current_price': 450.00,
            'support_level': 440.00,
            'confidence_score': 72,
            'distance_to_support_pct': -2.2,
            'has_earnings_soon': True,
            'days_until_earnings': 5
        }

        message = notifier.format_alert(alert)

        assert 'NVDA' in message
        assert 'Earnings in 5 days' in message

    def test_format_alert_emoji_by_recommendation(self):
        """Test correct emoji for each recommendation"""
        notifier = TelegramNotifier("token", "chat")

        alerts = [
            {'symbol': 'A', 'recommendation': 'STRONG_BUY', 'current_price': 100, 'support_level': 95, 'confidence_score': 80, 'distance_to_support_pct': -5},
            {'symbol': 'B', 'recommendation': 'BUY', 'current_price': 100, 'support_level': 95, 'confidence_score': 70, 'distance_to_support_pct': -5},
            {'symbol': 'C', 'recommendation': 'WATCH', 'current_price': 100, 'support_level': 95, 'confidence_score': 60, 'distance_to_support_pct': -5}
        ]

        messages = [notifier.format_alert(a) for a in alerts]

        # Each message should contain emoji
        for msg in messages:
            assert any(emoji in msg for emoji in ['🚀', '📈', '👀', '📊'])


# ================== EmailNotifier Tests ==================

class TestEmailNotifier:
    """Tests for EmailNotifier"""

    def test_init(self):
        """Test EmailNotifier initialization"""
        notifier = EmailNotifier(
            smtp_server="smtp.test.com",
            smtp_port=587,
            username="user@test.com",
            password="password",
            from_addr="from@test.com"
        )

        assert notifier.smtp_server == "smtp.test.com"
        assert notifier.smtp_port == 587
        assert notifier.from_addr == "from@test.com"

    def test_format_alert_html(self):
        """Test formatting alert as HTML"""
        notifier = EmailNotifier(
            "smtp.test.com", 587, "user", "pass", "from@test.com"
        )
        alert = {
            'symbol': 'MSFT',
            'recommendation': 'STRONG_BUY',
            'current_price': 380.00,
            'support_level': 365.00,
            'confidence_score': 88,
            'distance_to_support_pct': -3.9,
            'timeframe': 'weekly'
        }

        html = notifier.format_alert_html(alert)

        assert 'MSFT' in html
        assert 'STRONG_BUY' in html
        assert '380.00' in html
        assert '88' in html
        assert 'WEEKLY' in html
        assert '<html>' in html
        assert '</html>' in html

    def test_format_alert_html_with_earnings(self):
        """Test HTML formatting with earnings warning"""
        notifier = EmailNotifier(
            "smtp.test.com", 587, "user", "pass", "from@test.com"
        )
        alert = {
            'symbol': 'GOOGL',
            'recommendation': 'BUY',
            'current_price': 140.00,
            'support_level': 135.00,
            'confidence_score': 75,
            'distance_to_support_pct': -3.5,
            'timeframe': 'daily',
            'has_earnings_soon': True,
            'days_until_earnings': 3
        }

        html = notifier.format_alert_html(alert)

        assert 'Earnings in 3 days' in html
        assert 'Warning' in html

    def test_format_alert_html_colors(self):
        """Test that different recommendations have different colors"""
        notifier = EmailNotifier(
            "smtp.test.com", 587, "user", "pass", "from@test.com"
        )

        strong_buy_alert = {
            'symbol': 'A', 'recommendation': 'STRONG_BUY',
            'current_price': 100, 'support_level': 95,
            'confidence_score': 80, 'distance_to_support_pct': -5, 'timeframe': 'weekly'
        }
        sell_alert = {
            'symbol': 'B', 'recommendation': 'SELL',
            'current_price': 100, 'support_level': 95,
            'confidence_score': 30, 'distance_to_support_pct': -5, 'timeframe': 'weekly'
        }

        html1 = notifier.format_alert_html(strong_buy_alert)
        html2 = notifier.format_alert_html(sell_alert)

        # STRONG_BUY should have green color
        assert '#00C853' in html1
        # SELL should have red color
        assert '#F44336' in html2


# ================== NotificationManager Tests ==================

class TestNotificationManager:
    """Tests for NotificationManager"""

    @patch('src.utils.notification_manager.NotificationManager._load_config')
    def test_init_with_default_config(self, mock_load):
        """Test manager initialization with default config"""
        mock_load.return_value = NotificationConfig()

        manager = NotificationManager()

        assert manager.config is not None
        assert manager._telegram is None  # Not enabled by default
        assert manager._email is None  # Not enabled by default

    @patch('src.utils.notification_manager.NotificationManager._load_config')
    def test_init_with_telegram_enabled(self, mock_load):
        """Test manager initialization with Telegram enabled"""
        mock_load.return_value = NotificationConfig(
            telegram_enabled=True,
            telegram_bot_token="test_token",
            telegram_chat_id="12345"
        )

        manager = NotificationManager()

        assert manager._telegram is not None
        assert manager._telegram.bot_token == "test_token"

    def test_is_quiet_hours_disabled(self):
        """Test quiet hours when disabled"""
        with patch.object(NotificationManager, '_load_config') as mock:
            mock.return_value = NotificationConfig(
                quiet_hours_start=None,
                quiet_hours_end=None
            )
            manager = NotificationManager()

            assert manager.is_quiet_hours() is False

    def test_is_quiet_hours_during_quiet_period(self):
        """Test quiet hours during quiet period"""
        with patch.object(NotificationManager, '_load_config') as mock:
            mock.return_value = NotificationConfig(
                quiet_hours_start=22,
                quiet_hours_end=7
            )
            manager = NotificationManager()

            # Mock current hour to be 23 (within quiet hours)
            with patch('src.utils.notification_manager.datetime') as mock_dt:
                mock_dt.now.return_value.hour = 23
                mock_dt.now.return_value.isoformat = datetime.now().isoformat
                mock_dt.now.return_value.strftime = datetime.now().strftime
                assert manager.is_quiet_hours() is True

    def test_is_quiet_hours_outside_quiet_period(self):
        """Test quiet hours outside quiet period"""
        with patch.object(NotificationManager, '_load_config') as mock:
            mock.return_value = NotificationConfig(
                quiet_hours_start=22,
                quiet_hours_end=7
            )
            manager = NotificationManager()

            # Mock current hour to be 12 (outside quiet hours)
            with patch('src.utils.notification_manager.datetime') as mock_dt:
                mock_dt.now.return_value.hour = 12
                mock_dt.now.return_value.isoformat = datetime.now().isoformat
                mock_dt.now.return_value.strftime = datetime.now().strftime
                assert manager.is_quiet_hours() is False

    @patch('src.utils.notification_manager.NotificationManager._load_config')
    def test_should_notify_priority_filter(self, mock_load):
        """Test notification filtering by priority"""
        mock_load.return_value = NotificationConfig(min_priority="high")
        manager = NotificationManager()

        # Low and medium should be filtered
        assert manager.should_notify(NotificationPriority.LOW) is False
        assert manager.should_notify(NotificationPriority.MEDIUM) is False

        # High and urgent should pass
        assert manager.should_notify(NotificationPriority.HIGH) is True
        assert manager.should_notify(NotificationPriority.URGENT) is True

    @patch('src.utils.notification_manager.NotificationManager._load_config')
    def test_should_notify_alert_type_filter(self, mock_load):
        """Test notification filtering by alert type"""
        mock_load.return_value = NotificationConfig(
            alert_types=["STRONG_BUY"]
        )
        manager = NotificationManager()

        # STRONG_BUY should pass
        assert manager.should_notify(NotificationPriority.HIGH, "STRONG_BUY") is True

        # Other types should be filtered
        assert manager.should_notify(NotificationPriority.HIGH, "BUY") is False
        assert manager.should_notify(NotificationPriority.HIGH, "WATCH") is False

    @patch('src.utils.notification_manager.NotificationManager._load_config')
    def test_get_priority_for_alert_strong_buy_high_confidence(self, mock_load):
        """Test priority determination for STRONG_BUY with high confidence"""
        mock_load.return_value = NotificationConfig()
        manager = NotificationManager()

        alert = {'recommendation': 'STRONG_BUY', 'confidence_score': 90}
        priority = manager._get_priority_for_alert(alert)

        assert priority == NotificationPriority.URGENT

    @patch('src.utils.notification_manager.NotificationManager._load_config')
    def test_get_priority_for_alert_strong_buy_low_confidence(self, mock_load):
        """Test priority determination for STRONG_BUY with low confidence"""
        mock_load.return_value = NotificationConfig()
        manager = NotificationManager()

        alert = {'recommendation': 'STRONG_BUY', 'confidence_score': 65}
        priority = manager._get_priority_for_alert(alert)

        assert priority == NotificationPriority.HIGH

    @patch('src.utils.notification_manager.NotificationManager._load_config')
    def test_get_priority_for_alert_buy(self, mock_load):
        """Test priority determination for BUY"""
        mock_load.return_value = NotificationConfig()
        manager = NotificationManager()

        alert = {'recommendation': 'BUY', 'confidence_score': 70}
        priority = manager._get_priority_for_alert(alert)

        assert priority == NotificationPriority.MEDIUM

    @patch('src.utils.notification_manager.NotificationManager._load_config')
    def test_get_priority_for_alert_watch(self, mock_load):
        """Test priority determination for WATCH"""
        mock_load.return_value = NotificationConfig()
        manager = NotificationManager()

        alert = {'recommendation': 'WATCH', 'confidence_score': 50}
        priority = manager._get_priority_for_alert(alert)

        assert priority == NotificationPriority.LOW

    @patch('src.utils.notification_manager.NotificationManager._load_config')
    def test_update_config(self, mock_load):
        """Test configuration update"""
        mock_load.return_value = NotificationConfig()
        manager = NotificationManager()

        # Update configuration
        with patch.object(manager, '_save_config'):
            manager.update_config(
                min_priority="urgent",
                alert_types=["STRONG_BUY"]
            )

        assert manager.config.min_priority == "urgent"
        assert manager.config.alert_types == ["STRONG_BUY"]

    @patch('src.utils.notification_manager.NotificationManager._load_config')
    def test_get_notification_stats_empty(self, mock_load):
        """Test getting stats with no notifications"""
        mock_load.return_value = NotificationConfig()
        manager = NotificationManager()

        stats = manager.get_notification_stats()

        assert stats['total'] == 0
        assert stats['sent'] == 0
        assert stats['failed'] == 0

    @patch('src.utils.notification_manager.NotificationManager._load_config')
    def test_get_recent_notifications(self, mock_load):
        """Test getting recent notifications"""
        mock_load.return_value = NotificationConfig()
        manager = NotificationManager()

        # Add some notifications
        manager._notifications.append(Notification(
            id="1", title="Test 1", message="msg",
            priority=NotificationPriority.HIGH,
            channel=NotificationChannel.IN_APP,
            sent=True
        ))
        manager._notifications.append(Notification(
            id="2", title="Test 2", message="msg",
            priority=NotificationPriority.MEDIUM,
            channel=NotificationChannel.TELEGRAM,
            sent=True
        ))

        recent = manager.get_recent_notifications(limit=5)

        assert len(recent) == 2
        assert recent[0]['id'] in ['1', '2']


# ================== Async Tests ==================

class TestAsyncNotifications:
    """Tests for async notification methods"""

    @pytest.mark.asyncio
    async def test_send_alert_notification_filtered(self):
        """Test that filtered alerts don't send notifications"""
        with patch.object(NotificationManager, '_load_config') as mock:
            mock.return_value = NotificationConfig(
                min_priority="urgent",
                alert_types=["STRONG_BUY"]
            )
            manager = NotificationManager()

            # BUY alert should be filtered (below urgent priority)
            alert = {
                'symbol': 'TEST',
                'recommendation': 'BUY',
                'current_price': 100,
                'confidence_score': 70
            }

            results = await manager.send_alert_notification(alert)

            assert results == {}  # No notifications sent

    @pytest.mark.asyncio
    async def test_send_alert_notification_in_app(self):
        """Test in-app notification is always created"""
        with patch.object(NotificationManager, '_load_config') as mock:
            mock.return_value = NotificationConfig(
                in_app_enabled=True,
                min_priority="low"
            )
            manager = NotificationManager()

            alert = {
                'symbol': 'AAPL',
                'recommendation': 'STRONG_BUY',
                'current_price': 175,
                'confidence_score': 90
            }

            results = await manager.send_alert_notification(alert)

            assert results.get('in_app') is True
            assert len(manager._notifications) == 1


# ================== Integration Tests ==================

class TestNotificationIntegration:
    """Integration tests for notification system"""

    @patch('src.utils.notification_manager.NotificationManager._load_config')
    def test_full_notification_flow(self, mock_load):
        """Test complete notification flow from alert to stats"""
        mock_load.return_value = NotificationConfig(
            in_app_enabled=True,
            min_priority="medium"
        )
        manager = NotificationManager()

        # Create and send notifications
        alerts = [
            {'symbol': 'AAPL', 'recommendation': 'STRONG_BUY', 'current_price': 175, 'confidence_score': 90},
            {'symbol': 'MSFT', 'recommendation': 'BUY', 'current_price': 380, 'confidence_score': 75},
            {'symbol': 'GOOGL', 'recommendation': 'WATCH', 'current_price': 140, 'confidence_score': 50}  # Filtered (LOW priority)
        ]

        for alert in alerts:
            asyncio.run(manager.send_alert_notification(alert))

        # Check stats
        stats = manager.get_notification_stats()
        assert stats['total'] == 2  # Only 2 passed filter
        assert stats['by_channel']['in_app']['sent'] == 2

        # Check recent notifications
        recent = manager.get_recent_notifications()
        symbols = [n['data'].get('symbol') for n in recent]
        assert 'AAPL' in symbols
        assert 'MSFT' in symbols

    @patch('src.utils.notification_manager.NotificationManager._load_config')
    def test_config_persistence_flow(self, mock_load):
        """Test configuration save and reload"""
        import json
        from pathlib import Path
        import tempfile

        # Use temporary file for config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            mock_load.return_value = NotificationConfig(
                min_priority="high",
                alert_types=["STRONG_BUY", "BUY"]
            )
            manager = NotificationManager(config_path=config_path)

            # Update config
            manager.update_config(min_priority="urgent")

            # Verify file was saved
            with open(config_path, 'r') as f:
                saved = json.load(f)
                assert saved['min_priority'] == "urgent"

        finally:
            Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
