"""
Notification Manager for TradingBot V3

Provides multi-channel notifications:
- Telegram bot integration
- Email notifications (SMTP)
- In-app notifications (stored in DB)

Configuration via environment variables or settings file.
"""
import os
import json
import smtplib
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Available notification channels"""
    TELEGRAM = "telegram"
    EMAIL = "email"
    IN_APP = "in_app"


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class NotificationConfig:
    """Configuration for notifications"""
    # Telegram settings
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Email settings
    email_enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)

    # In-app settings
    in_app_enabled: bool = True

    # Filter settings
    min_priority: str = "medium"  # Only send notifications >= this priority
    alert_types: List[str] = field(default_factory=lambda: ["STRONG_BUY", "BUY"])
    quiet_hours_start: Optional[int] = None  # Hour (0-23) to start quiet period
    quiet_hours_end: Optional[int] = None    # Hour (0-23) to end quiet period

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'NotificationConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Notification:
    """A notification message"""
    id: str
    title: str
    message: str
    priority: NotificationPriority
    channel: NotificationChannel
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict = field(default_factory=dict)
    sent: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'message': self.message,
            'priority': self.priority.value,
            'channel': self.channel.value,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'sent': self.sent,
            'error': self.error
        }


class TelegramNotifier:
    """Telegram bot notification handler"""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send message via Telegram bot"""
        try:
            import aiohttp

            url = f"{self.api_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"Telegram API error: {error}")
                        return False
        except ImportError:
            logger.warning("aiohttp not installed, using sync request")
            return self._send_message_sync(text, parse_mode)
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    def _send_message_sync(self, text: str, parse_mode: str = "HTML") -> bool:
        """Synchronous fallback for sending messages"""
        try:
            import requests

            url = f"{self.api_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode
            }

            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Telegram sync send error: {e}")
            return False

    def format_alert(self, alert: Dict) -> str:
        """Format alert for Telegram"""
        symbol = alert.get('symbol', 'N/A')
        recommendation = alert.get('recommendation', 'N/A')
        price = alert.get('current_price', 0)
        support = alert.get('support_level', 0)
        confidence = alert.get('confidence_score', 0)
        distance = alert.get('distance_to_support_pct', 0)

        # Emoji based on recommendation
        emoji = {
            'STRONG_BUY': '🚀',
            'BUY': '📈',
            'WATCH': '👀',
            'HOLD': '⏸️',
            'SELL': '📉'
        }.get(recommendation, '📊')

        # Earnings warning
        earnings_warning = ""
        if alert.get('has_earnings_soon'):
            days = alert.get('days_until_earnings', '?')
            earnings_warning = f"\n⚠️ <b>Earnings in {days} days!</b>"

        message = f"""
{emoji} <b>{symbol}</b> - {recommendation}

💰 Price: ${price:.2f}
📊 Support: ${support:.2f} ({distance:+.1f}%)
🎯 Confidence: {confidence:.0f}/100
{earnings_warning}

<i>TradingBot V3</i>
"""
        return message.strip()


class EmailNotifier:
    """Email notification handler"""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr

    def send_email(
        self,
        to_addrs: List[str],
        subject: str,
        body: str,
        html: bool = True
    ) -> bool:
        """Send email notification"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_addr
            msg['To'] = ', '.join(to_addrs)

            if html:
                part = MIMEText(body, 'html')
            else:
                part = MIMEText(body, 'plain')

            msg.attach(part)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, to_addrs, msg.as_string())

            return True
        except Exception as e:
            logger.error(f"Email send error: {e}")
            return False

    def format_alert_html(self, alert: Dict) -> str:
        """Format alert as HTML email"""
        symbol = alert.get('symbol', 'N/A')
        recommendation = alert.get('recommendation', 'N/A')
        price = alert.get('current_price', 0)
        support = alert.get('support_level', 0)
        confidence = alert.get('confidence_score', 0)
        distance = alert.get('distance_to_support_pct', 0)
        timeframe = alert.get('timeframe', 'weekly')

        # Color based on recommendation
        color = {
            'STRONG_BUY': '#00C853',
            'BUY': '#2962FF',
            'WATCH': '#FFC107',
            'HOLD': '#9E9E9E',
            'SELL': '#F44336'
        }.get(recommendation, '#9E9E9E')

        earnings_section = ""
        if alert.get('has_earnings_soon'):
            days = alert.get('days_until_earnings', '?')
            earnings_section = f"""
            <tr>
                <td colspan="2" style="background-color: #FFF3CD; padding: 10px; color: #856404;">
                    ⚠️ <strong>Warning:</strong> Earnings in {days} days!
                </td>
            </tr>
            """

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .container {{ max-width: 600px; margin: 0 auto; }}
        .header {{ background-color: {color}; color: white; padding: 20px; text-align: center; }}
        .content {{ padding: 20px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        td {{ padding: 10px; border-bottom: 1px solid #eee; }}
        .label {{ color: #666; }}
        .value {{ font-weight: bold; }}
        .footer {{ text-align: center; padding: 20px; color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{symbol}</h1>
            <h2>{recommendation}</h2>
        </div>
        <div class="content">
            <table>
                <tr>
                    <td class="label">Current Price</td>
                    <td class="value">${price:.2f}</td>
                </tr>
                <tr>
                    <td class="label">Support Level</td>
                    <td class="value">${support:.2f} ({distance:+.1f}%)</td>
                </tr>
                <tr>
                    <td class="label">Confidence Score</td>
                    <td class="value">{confidence:.0f}/100</td>
                </tr>
                <tr>
                    <td class="label">Timeframe</td>
                    <td class="value">{timeframe.upper()}</td>
                </tr>
                {earnings_section}
            </table>
        </div>
        <div class="footer">
            Generated by TradingBot V3 • {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>
</body>
</html>
"""
        return html


class NotificationManager:
    """
    Central notification manager.

    Handles routing notifications to appropriate channels
    based on configuration and priority.
    """

    def __init__(self, config_path: str = "data/notification_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self._telegram: Optional[TelegramNotifier] = None
        self._email: Optional[EmailNotifier] = None

        self._notifications: List[Notification] = []
        self._executor = ThreadPoolExecutor(max_workers=2)

        self._init_channels()

    def _load_config(self) -> NotificationConfig:
        """Load configuration from file or environment"""
        config = NotificationConfig()

        # Try loading from file
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    config = NotificationConfig.from_dict(data)
            except Exception as e:
                logger.warning(f"Error loading config: {e}")

        # Override with environment variables
        if os.getenv('TELEGRAM_BOT_TOKEN'):
            config.telegram_enabled = True
            config.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
            config.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')

        if os.getenv('SMTP_USERNAME'):
            config.email_enabled = True
            config.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            config.smtp_port = int(os.getenv('SMTP_PORT', '587'))
            config.smtp_username = os.getenv('SMTP_USERNAME', '')
            config.smtp_password = os.getenv('SMTP_PASSWORD', '')
            config.email_from = os.getenv('EMAIL_FROM', config.smtp_username)
            config.email_to = os.getenv('EMAIL_TO', '').split(',')

        return config

    def _save_config(self):
        """Save configuration to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def _init_channels(self):
        """Initialize notification channels"""
        if self.config.telegram_enabled and self.config.telegram_bot_token:
            self._telegram = TelegramNotifier(
                self.config.telegram_bot_token,
                self.config.telegram_chat_id
            )
            logger.info("Telegram notifications enabled")

        if self.config.email_enabled and self.config.smtp_username:
            self._email = EmailNotifier(
                self.config.smtp_server,
                self.config.smtp_port,
                self.config.smtp_username,
                self.config.smtp_password,
                self.config.email_from
            )
            logger.info("Email notifications enabled")

    def update_config(self, **kwargs):
        """Update configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        self._save_config()
        self._init_channels()

    def is_quiet_hours(self) -> bool:
        """Check if current time is within quiet hours"""
        if self.config.quiet_hours_start is None or self.config.quiet_hours_end is None:
            return False

        current_hour = datetime.now().hour
        start = self.config.quiet_hours_start
        end = self.config.quiet_hours_end

        if start <= end:
            return start <= current_hour < end
        else:  # Overnight quiet hours (e.g., 22:00 - 07:00)
            return current_hour >= start or current_hour < end

    def should_notify(self, priority: NotificationPriority, alert_type: str = None) -> bool:
        """Check if notification should be sent based on filters"""
        # Check quiet hours (except for urgent)
        if priority != NotificationPriority.URGENT and self.is_quiet_hours():
            return False

        # Check minimum priority
        priority_order = ['low', 'medium', 'high', 'urgent']
        min_priority_idx = priority_order.index(self.config.min_priority)
        current_priority_idx = priority_order.index(priority.value)

        if current_priority_idx < min_priority_idx:
            return False

        # Check alert type filter
        if alert_type and self.config.alert_types:
            if alert_type not in self.config.alert_types:
                return False

        return True

    def _get_priority_for_alert(self, alert: Dict) -> NotificationPriority:
        """Determine notification priority based on alert"""
        recommendation = alert.get('recommendation', '')
        confidence = alert.get('confidence_score', 0)

        if recommendation == 'STRONG_BUY' and confidence >= 80:
            return NotificationPriority.URGENT
        elif recommendation == 'STRONG_BUY':
            return NotificationPriority.HIGH
        elif recommendation == 'BUY':
            return NotificationPriority.MEDIUM
        else:
            return NotificationPriority.LOW

    async def send_alert_notification(self, alert: Dict) -> Dict[str, bool]:
        """
        Send alert notification to all enabled channels.

        Returns dict with channel -> success status
        """
        recommendation = alert.get('recommendation', '')
        priority = self._get_priority_for_alert(alert)

        if not self.should_notify(priority, recommendation):
            logger.debug(f"Notification filtered: {recommendation} with priority {priority.value}")
            return {}

        results = {}
        symbol = alert.get('symbol', 'N/A')

        # Telegram
        if self._telegram and self.config.telegram_enabled:
            try:
                message = self._telegram.format_alert(alert)
                success = await self._telegram.send_message(message)
                results['telegram'] = success

                notification = Notification(
                    id=f"tg_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    title=f"{symbol} - {recommendation}",
                    message=message,
                    priority=priority,
                    channel=NotificationChannel.TELEGRAM,
                    data=alert,
                    sent=success
                )
                self._notifications.append(notification)
            except Exception as e:
                logger.error(f"Telegram notification error: {e}")
                results['telegram'] = False

        # Email
        if self._email and self.config.email_enabled:
            try:
                subject = f"[TradingBot] {symbol} - {recommendation}"
                body = self._email.format_alert_html(alert)
                success = self._email.send_email(
                    self.config.email_to,
                    subject,
                    body
                )
                results['email'] = success

                notification = Notification(
                    id=f"email_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    title=subject,
                    message=body[:200],
                    priority=priority,
                    channel=NotificationChannel.EMAIL,
                    data=alert,
                    sent=success
                )
                self._notifications.append(notification)
            except Exception as e:
                logger.error(f"Email notification error: {e}")
                results['email'] = False

        # In-app (always)
        if self.config.in_app_enabled:
            notification = Notification(
                id=f"app_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                title=f"{symbol} - {recommendation}",
                message=f"Price ${alert.get('current_price', 0):.2f}, Confidence {alert.get('confidence_score', 0):.0f}/100",
                priority=priority,
                channel=NotificationChannel.IN_APP,
                data=alert,
                sent=True
            )
            self._notifications.append(notification)
            results['in_app'] = True

        return results

    def send_alert_notification_sync(self, alert: Dict) -> Dict[str, bool]:
        """Synchronous wrapper for send_alert_notification"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, schedule as task
                future = asyncio.ensure_future(self.send_alert_notification(alert))
                return {}  # Can't wait in sync context
            else:
                return loop.run_until_complete(self.send_alert_notification(alert))
        except RuntimeError:
            # No event loop, create new one
            return asyncio.run(self.send_alert_notification(alert))

    async def send_daily_summary(self, alerts: List[Dict]) -> Dict[str, bool]:
        """Send daily summary of all alerts"""
        if not alerts:
            return {}

        # Format summary
        summary_lines = [f"📊 <b>Daily Summary - {len(alerts)} Alerts</b>\n"]

        for alert in alerts[:10]:  # Top 10
            symbol = alert.get('symbol', 'N/A')
            rec = alert.get('recommendation', 'N/A')
            conf = alert.get('confidence_score', 0)
            emoji = '🚀' if rec == 'STRONG_BUY' else '📈' if rec == 'BUY' else '👀'
            summary_lines.append(f"{emoji} {symbol}: {rec} ({conf:.0f}/100)")

        summary = '\n'.join(summary_lines)

        results = {}

        if self._telegram and self.config.telegram_enabled:
            success = await self._telegram.send_message(summary)
            results['telegram'] = success

        return results

    def get_recent_notifications(self, limit: int = 50) -> List[Dict]:
        """Get recent notifications"""
        sorted_notifs = sorted(
            self._notifications,
            key=lambda n: n.timestamp,
            reverse=True
        )
        return [n.to_dict() for n in sorted_notifs[:limit]]

    def get_notification_stats(self) -> Dict:
        """Get notification statistics"""
        total = len(self._notifications)
        sent = sum(1 for n in self._notifications if n.sent)
        failed = total - sent

        by_channel = {}
        for channel in NotificationChannel:
            channel_notifs = [n for n in self._notifications if n.channel == channel]
            by_channel[channel.value] = {
                'total': len(channel_notifs),
                'sent': sum(1 for n in channel_notifs if n.sent)
            }

        return {
            'total': total,
            'sent': sent,
            'failed': failed,
            'by_channel': by_channel
        }

    def test_telegram(self) -> bool:
        """Test Telegram connection"""
        if not self._telegram:
            return False

        try:
            message = "🔔 Test notification from TradingBot V3\n\nTelegram is configured correctly!"
            return self._telegram._send_message_sync(message)
        except Exception as e:
            logger.error(f"Telegram test failed: {e}")
            return False

    def test_email(self) -> bool:
        """Test email connection"""
        if not self._email:
            return False

        try:
            subject = "[TradingBot] Test Notification"
            body = "<h1>Test Notification</h1><p>Email is configured correctly!</p>"
            return self._email.send_email(self.config.email_to, subject, body)
        except Exception as e:
            logger.error(f"Email test failed: {e}")
            return False


# Singleton instance
notification_manager = NotificationManager()
