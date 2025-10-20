"""
Notification system for sending alerts via Telegram and Email
"""
import os
from typing import List, Dict, Optional
from datetime import datetime
import asyncio
from telegram import Bot
from telegram.error import TelegramError
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from config.settings import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    EMAIL_ENABLED, EMAIL_FROM, EMAIL_TO, SMTP_SERVER, SMTP_PORT,
    SMTP_USERNAME, SMTP_PASSWORD
)
from src.utils.logger import logger


class Notifier:
    """Handles all notifications (Telegram, Email, etc.)"""

    def __init__(self):
        """Initialize the notifier"""
        self.telegram_enabled = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
        self.email_enabled = EMAIL_ENABLED

        if self.telegram_enabled:
            self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
            logger.info("Telegram notifications enabled")
        else:
            logger.warning("Telegram notifications disabled (missing credentials)")

        if self.email_enabled:
            logger.info("Email notifications enabled")
        else:
            logger.debug("Email notifications disabled")

    def format_alert_message(self, alert: Dict) -> str:
        """
        Format a single alert for display

        Args:
            alert: Alert dictionary

        Returns:
            Formatted message string
        """
        symbol = alert.get('symbol', 'N/A')
        company_name = alert.get('company_name', symbol)
        timeframe = alert.get('timeframe', 'N/A').upper()
        current_price = alert.get('current_price', 0)
        support_level = alert.get('support_level', 0)
        distance = alert.get('distance_to_support_pct', 0)
        recommendation = alert.get('recommendation', 'N/A')
        ema_alignment = alert.get('ema_alignment', 'N/A')

        ema_24 = alert.get('ema_24', 0)
        ema_38 = alert.get('ema_38', 0)
        ema_62 = alert.get('ema_62', 0)

        # Emoji based on recommendation
        emoji_map = {
            'STRONG_BUY': '🔥',
            'BUY': '✅',
            'WATCH': '👀',
            'OBSERVE': '📊'
        }
        emoji = emoji_map.get(recommendation, '📈')

        message = f"""
{emoji} *{symbol}* - {company_name}

📊 *Timeframe:* {timeframe}
💰 *Current Price:* ${current_price:.2f}
🎯 *Support Level:* ${support_level:.2f}
📏 *Distance:* {distance:.2f}%

📈 *EMAs:*
  • EMA 24: ${ema_24:.2f}
  • EMA 38: ${ema_38:.2f}
  • EMA 62: ${ema_62:.2f}
  • Alignment: {ema_alignment}

💡 *Recommendation:* {recommendation}
"""
        return message.strip()

    def format_daily_report(self, screening_results: Dict, alerts: List[Dict]) -> str:
        """
        Format the daily screening report

        Args:
            screening_results: Screening results dictionary
            alerts: List of alert dictionaries

        Returns:
            Formatted report string
        """
        total_stocks = screening_results.get('total_stocks_analyzed', 0)
        total_alerts = screening_results.get('total_alerts_generated', 0)
        execution_time = screening_results.get('execution_time_seconds', 0)
        status = screening_results.get('status', 'UNKNOWN')

        # Header
        report = f"""
📊 *DAILY MARKET SCREENING REPORT*
🗓 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*40}

📈 *Summary:*
  • Stocks Analyzed: {total_stocks}
  • Alerts Generated: {total_alerts}
  • Execution Time: {execution_time:.1f}s
  • Status: {status}

{'='*40}
"""

        if not alerts:
            report += "\n⚠️ *No buy signals detected today.*\n"
            return report

        # Group alerts by recommendation
        strong_buys = [a for a in alerts if a.get('recommendation') == 'STRONG_BUY']
        buys = [a for a in alerts if a.get('recommendation') == 'BUY']
        watches = [a for a in alerts if a.get('recommendation') == 'WATCH']
        observes = [a for a in alerts if a.get('recommendation') == 'OBSERVE']

        if strong_buys:
            report += f"\n🔥 *STRONG BUY ({len(strong_buys)})*\n"
            for alert in strong_buys[:5]:  # Top 5
                report += f"  • {alert['symbol']} @ ${alert['current_price']:.2f} ({alert['timeframe']})\n"

        if buys:
            report += f"\n✅ *BUY ({len(buys)})*\n"
            for alert in buys[:5]:  # Top 5
                report += f"  • {alert['symbol']} @ ${alert['current_price']:.2f} ({alert['timeframe']})\n"

        if watches:
            report += f"\n👀 *WATCH ({len(watches)})*\n"
            for alert in watches[:5]:  # Top 5
                report += f"  • {alert['symbol']} @ ${alert['current_price']:.2f} ({alert['timeframe']})\n"

        if observes:
            report += f"\n📊 *OBSERVE ({len(observes)})*\n"
            for alert in observes[:3]:  # Top 3
                report += f"  • {alert['symbol']} @ ${alert['current_price']:.2f} ({alert['timeframe']})\n"

        report += f"\n{'='*40}\n"
        report += "\n💡 *Top 3 Opportunities:*\n\n"

        # Show detailed info for top 3 alerts
        top_alerts = strong_buys[:3] if strong_buys else buys[:3] if buys else alerts[:3]

        for i, alert in enumerate(top_alerts, 1):
            report += f"{i}. {self.format_alert_message(alert)}\n\n"

        return report

    async def send_telegram_message(self, message: str) -> bool:
        """
        Send a message via Telegram

        Args:
            message: Message to send

        Returns:
            True if successful, False otherwise
        """
        if not self.telegram_enabled:
            logger.warning("Telegram not enabled, skipping notification")
            return False

        try:
            # Split message if too long (Telegram limit is 4096 characters)
            max_length = 4000
            if len(message) > max_length:
                # Split into chunks
                chunks = [message[i:i+max_length] for i in range(0, len(message), max_length)]
                for chunk in chunks:
                    await self.bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=chunk,
                        parse_mode='Markdown'
                    )
                    await asyncio.sleep(1)  # Avoid rate limiting
            else:
                await self.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=message,
                    parse_mode='Markdown'
                )

            logger.info("Telegram message sent successfully")
            return True

        except TelegramError as e:
            logger.error(f"Telegram error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False

    def send_email(self, subject: str, body: str, is_html: bool = False) -> bool:
        """
        Send an email notification

        Args:
            subject: Email subject
            body: Email body
            is_html: If True, send as HTML email

        Returns:
            True if successful, False otherwise
        """
        if not self.email_enabled:
            logger.debug("Email not enabled, skipping")
            return False

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = EMAIL_FROM
            msg['To'] = EMAIL_TO

            if is_html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)

            logger.info(f"Email sent to {EMAIL_TO}")
            return True

        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False

    def send_daily_report(self, screening_results: Dict, alerts: List[Dict]) -> bool:
        """
        Send the daily screening report via all enabled channels

        Args:
            screening_results: Screening results dictionary
            alerts: List of alert dictionaries

        Returns:
            True if at least one notification was sent successfully
        """
        report = self.format_daily_report(screening_results, alerts)
        success = False

        # Send via Telegram
        if self.telegram_enabled:
            try:
                result = asyncio.run(self.send_telegram_message(report))
                success = success or result
            except Exception as e:
                logger.error(f"Error sending Telegram report: {e}")

        # Send via Email
        if self.email_enabled:
            subject = f"Daily Market Screening - {datetime.now().strftime('%Y-%m-%d')}"
            # Convert markdown to plain text for email
            plain_report = report.replace('*', '').replace('_', '')
            result = self.send_email(subject, plain_report)
            success = success or result

        return success

    def send_alert(self, alert: Dict) -> bool:
        """
        Send a single alert notification

        Args:
            alert: Alert dictionary

        Returns:
            True if at least one notification was sent successfully
        """
        message = self.format_alert_message(alert)
        success = False

        # Send via Telegram
        if self.telegram_enabled:
            try:
                result = asyncio.run(self.send_telegram_message(message))
                success = success or result
            except Exception as e:
                logger.error(f"Error sending Telegram alert: {e}")

        # Send via Email
        if self.email_enabled:
            subject = f"Buy Alert: {alert.get('symbol')} - {alert.get('recommendation')}"
            plain_message = message.replace('*', '').replace('_', '')
            result = self.send_email(subject, plain_message)
            success = success or result

        return success

    def test_notifications(self) -> Dict[str, bool]:
        """
        Test all notification channels

        Returns:
            Dictionary with test results for each channel
        """
        results = {}

        test_message = """
🧪 *Test Notification*

This is a test message from your Market Screener.
If you're receiving this, notifications are working! ✅
"""

        # Test Telegram
        if self.telegram_enabled:
            try:
                results['telegram'] = asyncio.run(self.send_telegram_message(test_message))
            except Exception as e:
                logger.error(f"Telegram test failed: {e}")
                results['telegram'] = False
        else:
            results['telegram'] = None

        # Test Email
        if self.email_enabled:
            try:
                results['email'] = self.send_email(
                    "Market Screener - Test Notification",
                    test_message.replace('*', '')
                )
            except Exception as e:
                logger.error(f"Email test failed: {e}")
                results['email'] = False
        else:
            results['email'] = None

        return results


# Singleton instance
notifier = Notifier()
