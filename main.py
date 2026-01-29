"""
Market Screener - Main Entry Point

DEPRECATED: This entry point is deprecated in favor of V4 agents.
Please use:
  - python run_agent.py --mode live     # For automated trading
  - python webapp.py                     # For web dashboard

This file is kept for reference and backward compatibility.

This script runs the daily market screening and sends notifications.
"""
import warnings
warnings.warn(
    "main.py is DEPRECATED. Use 'python run_agent.py' or 'python webapp.py' instead.",
    DeprecationWarning,
    stacklevel=2
)
import os
import sys
import argparse
import schedule
import time
from datetime import datetime
from dotenv import load_dotenv
import pytz

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.settings import DAILY_REPORT_TIME, TIMEZONE
from src.screening.screener import market_screener
from src.notifications.notifier import notifier
from src.database.db_manager import db_manager
from src.utils.logger import logger


def run_screening():
    """Run the complete market screening process"""
    logger.info("Starting market screening job...")

    try:
        # Run screening
        results = market_screener.run_daily_screening()

        # Get alerts
        alerts = results.get('alerts', [])

        # Send daily report
        if alerts or results.get('status') == 'SUCCESS':
            logger.info("Sending daily report...")
            notifier.send_daily_report(results, alerts)
        else:
            logger.warning("No alerts to report")

        logger.info("Market screening job completed")

    except Exception as e:
        logger.error(f"Error in screening job: {e}")
        # Send error notification
        error_message = f"⚠️ *Market Screening Failed*\n\nError: {str(e)}"
        try:
            import asyncio
            asyncio.run(notifier.send_telegram_message(error_message))
        except:
            pass


def run_once():
    """Run screening once and exit"""
    logger.info("Running one-time screening...")
    run_screening()
    logger.info("One-time screening complete. Exiting.")


def test_notifications():
    """Test all notification channels"""
    logger.info("Testing notification channels...")
    results = notifier.test_notifications()

    logger.info("Notification test results:")
    for channel, result in results.items():
        if result is None:
            status = "DISABLED"
        elif result:
            status = "✅ SUCCESS"
        else:
            status = "❌ FAILED"
        logger.info(f"  {channel.upper()}: {status}")


def run_scheduler():
    """Run the scheduler for daily automated screening"""
    # Get timezone
    tz = pytz.timezone(TIMEZONE)

    # Schedule daily job
    schedule_time = DAILY_REPORT_TIME.strftime('%H:%M')
    schedule.every().day.at(schedule_time).do(run_screening)

    logger.info(f"Scheduler started. Daily screening scheduled for {schedule_time} {TIMEZONE}")
    logger.info("Press Ctrl+C to stop")

    # Run once immediately on start (optional - comment out if not desired)
    logger.info("Running initial screening...")
    run_screening()

    # Keep running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")


def screen_symbol(symbol: str):
    """
    Screen a specific symbol manually

    Args:
        symbol: Stock symbol to screen
    """
    logger.info(f"Screening {symbol}...")

    from src.data.market_data import market_data_fetcher

    # Get stock info
    info = market_data_fetcher.get_stock_info(symbol)
    if not info:
        logger.error(f"Could not fetch info for {symbol}")
        return

    company_name = info.get('longName', symbol)

    # Screen the stock
    alert = market_screener.screen_single_stock(symbol, company_name)

    if alert:
        logger.info(f"\n{'='*60}")
        logger.info(f"BUY SIGNAL DETECTED for {symbol}")
        logger.info(f"{'='*60}")
        logger.info(f"Company: {company_name}")
        logger.info(f"Timeframe: {alert['timeframe']}")
        logger.info(f"Current Price: ${alert['current_price']:.2f}")
        logger.info(f"Support Level: ${alert['support_level']:.2f}")
        logger.info(f"Distance: {alert['distance_to_support_pct']:.2f}%")
        logger.info(f"EMA Alignment: {alert['ema_alignment']}")
        logger.info(f"Recommendation: {alert['recommendation']}")
        logger.info(f"{'='*60}\n")

        # Ask if user wants to send notification
        response = input("Send notification? (y/n): ")
        if response.lower() == 'y':
            notifier.send_alert(alert)
            db_manager.save_alert(alert)
    else:
        logger.info(f"No buy signal detected for {symbol}")


def show_recent_alerts(days: int = 7):
    """
    Show recent alerts from database

    Args:
        days: Number of days to look back
    """
    logger.info(f"Fetching alerts from last {days} days...")

    alerts = db_manager.get_recent_alerts(days=days)

    if not alerts:
        logger.info("No alerts found")
        return

    logger.info(f"\nFound {len(alerts)} alerts:\n")
    logger.info(f"{'='*80}")

    for alert in alerts:
        logger.info(f"Symbol: {alert.symbol} ({alert.timeframe})")
        logger.info(f"Date: {alert.alert_date.strftime('%Y-%m-%d %H:%M')}")
        logger.info(f"Price: ${alert.current_price:.2f}")
        logger.info(f"Support: ${alert.support_level:.2f} ({alert.distance_to_support_pct:.1f}% away)")
        logger.info(f"Recommendation: {alert.recommendation}")
        logger.info(f"Notified: {'Yes' if alert.is_notified else 'No'}")
        logger.info(f"{'-'*80}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Market Screener - Automated stock screening for buy signals'
    )

    parser.add_argument(
        'command',
        choices=['run', 'schedule', 'test', 'screen', 'alerts'],
        help='Command to execute'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        help='Stock symbol to screen (use with "screen" command)'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to look back for alerts (use with "alerts" command)'
    )

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # Check if .env exists
    if not os.path.exists('.env'):
        logger.warning(".env file not found. Please create one based on .env.example")
        logger.warning("Telegram notifications will not work without proper configuration")

    # Execute command
    if args.command == 'run':
        run_once()

    elif args.command == 'schedule':
        run_scheduler()

    elif args.command == 'test':
        test_notifications()

    elif args.command == 'screen':
        if not args.symbol:
            logger.error("Please provide a symbol with --symbol flag")
            sys.exit(1)
        screen_symbol(args.symbol.upper())

    elif args.command == 'alerts':
        show_recent_alerts(days=args.days)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
