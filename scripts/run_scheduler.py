#!/usr/bin/env python3
"""
TradingBot V3 - Scheduler Runner

Standalone script to run the scan scheduler as a background service.
Used by Docker container and can be run directly.

Usage:
    python scripts/run_scheduler.py

Environment variables:
    SCHEDULER_ENABLED: Enable/disable scheduler (default: true)
    SCHEDULER_TIME: Time of day for daily scans (default: 09:30)
    SCHEDULER_MARKETS: Comma-separated markets (default: NASDAQ,SP500)
"""

import os
import sys
import signal
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('scheduler')


def get_stocks_for_markets(markets: list) -> list:
    """Get stocks list based on configured markets"""
    from src.data.market_data import market_data_fetcher

    stocks = []
    for market in markets:
        market = market.strip().upper()
        try:
            if market == "NASDAQ":
                symbols = market_data_fetcher.get_nasdaq_symbols()[:200]
                stocks.extend(symbols)
                logger.info(f"Added {len(symbols)} NASDAQ stocks")
            elif market == "SP500":
                symbols = market_data_fetcher.get_sp500_symbols()
                stocks.extend(symbols)
                logger.info(f"Added {len(symbols)} SP500 stocks")
            elif market == "EUROPE":
                symbols = market_data_fetcher.get_europe_symbols()
                stocks.extend(symbols)
                logger.info(f"Added {len(symbols)} European stocks")
            elif market == "CRYPTO":
                symbols = market_data_fetcher.get_crypto_symbols()
                stocks.extend(symbols)
                logger.info(f"Added {len(symbols)} crypto symbols")
            elif market == "CAC40":
                symbols = market_data_fetcher.get_cac40_symbols()
                stocks.extend(symbols)
                logger.info(f"Added {len(symbols)} CAC40 stocks")
            elif market == "DAX":
                symbols = market_data_fetcher.get_dax_symbols()
                stocks.extend(symbols)
                logger.info(f"Added {len(symbols)} DAX stocks")
        except Exception as e:
            logger.error(f"Error loading {market} stocks: {e}")

    return stocks


def main():
    """Main scheduler entry point"""
    from src.utils.background_scanner import scan_scheduler
    from src.screening.screener import market_screener
    from src.utils.notification_manager import notification_manager

    logger.info("=" * 60)
    logger.info("TradingBot V3 - Scheduler Starting")
    logger.info("=" * 60)

    # Read configuration from environment
    scheduler_enabled = os.getenv('SCHEDULER_ENABLED', 'true').lower() == 'true'
    scheduler_time = os.getenv('SCHEDULER_TIME', '09:30')
    scheduler_markets = os.getenv('SCHEDULER_MARKETS', 'NASDAQ,SP500').split(',')

    logger.info(f"Scheduler enabled: {scheduler_enabled}")
    logger.info(f"Scan time: {scheduler_time}")
    logger.info(f"Markets: {scheduler_markets}")

    if not scheduler_enabled:
        logger.warning("Scheduler is disabled. Set SCHEDULER_ENABLED=true to enable.")
        logger.info("Running in standby mode...")
        # Keep process alive but do nothing
        signal.pause()
        return

    # Configure scheduler
    scan_scheduler.update_config(
        enabled=True,
        schedule_type="daily",
        time_of_day=scheduler_time,
        markets=scheduler_markets,
        notify_on_completion=True,
        notify_on_signals=True
    )

    # Set up dependencies
    scan_scheduler.set_screen_function(market_screener.screen_single_stock_weekly)
    scan_scheduler.set_stocks_provider(lambda: get_stocks_for_markets(scheduler_markets))
    scan_scheduler.set_notification_manager(notification_manager)

    # Handle shutdown signals
    def shutdown_handler(signum, frame):
        logger.info("Shutdown signal received")
        scan_scheduler.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    # Start scheduler
    logger.info("Starting scheduler...")
    if scan_scheduler.start():
        status = scan_scheduler.get_status()
        logger.info(f"Scheduler started successfully")
        logger.info(f"Next run: {status['next_run']}")

        # Keep process alive
        try:
            signal.pause()
        except AttributeError:
            # Windows doesn't have signal.pause()
            import time
            while True:
                time.sleep(60)
                status = scan_scheduler.get_status()
                if status['next_run']:
                    next_run = datetime.fromisoformat(status['next_run'])
                    time_until = (next_run - datetime.now()).total_seconds()
                    if time_until > 0:
                        hours = int(time_until // 3600)
                        minutes = int((time_until % 3600) // 60)
                        logger.debug(f"Next run in {hours}h {minutes}m")
    else:
        logger.error("Failed to start scheduler")
        sys.exit(1)


if __name__ == "__main__":
    main()
