#!/usr/bin/env python
"""
Market Data Prefetch Script

Pre-downloads historical data for all configured markets.
Designed to run as a scheduled task (Windows Task Scheduler or cron)
before market open to ensure fresh cache for simulations.

Usage:
    python scripts/prefetch_market_data.py              # Weekly data (default)
    python scripts/prefetch_market_data.py --daily      # Daily data
    python scripts/prefetch_market_data.py --both       # Both weekly and daily
    python scripts/prefetch_market_data.py --stats      # Show cache stats only

Recommended Schedule:
    - Weekly data: Once per week (Sunday night)
    - Daily data: Every day at 06:00 (before market open)
"""

import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.market_data import market_data_fetcher
from src.utils.logger import logger


def log_to_file(log_dir: Path, message: str):
    """Append message to daily log file"""
    log_file = log_dir / f"prefetch_{datetime.now().strftime('%Y%m%d')}.log"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")


def show_cache_stats():
    """Display cache statistics"""
    stats = market_data_fetcher.get_cache_stats()

    if 'error' in stats:
        print(f"Error getting cache stats: {stats['error']}")
        return

    print("\n" + "=" * 60)
    print("CACHE STATISTICS")
    print("=" * 60)
    print(f"  Total files:    {stats['total_files']}")
    print(f"  Total size:     {stats['total_size_mb']:.1f} MB")
    print(f"  Weekly files:   {stats['weekly_files']}")
    print(f"  Daily files:    {stats['daily_files']}")
    print(f"  Valid files:    {stats['valid_files']}")
    print(f"  Expired files:  {stats['expired_files']}")
    print(f"  TTL:            {stats['ttl_hours']} hours")
    if stats['oldest_file']:
        print(f"  Oldest file:    {stats['oldest_file']}")
    if stats['newest_file']:
        print(f"  Newest file:    {stats['newest_file']}")
    print("=" * 60)


def run_prefetch(
    interval: str,
    markets: list = None,
    log_dir: Path = None
):
    """
    Run prefetch for specified interval

    Args:
        interval: '1wk' or '1d'
        markets: List of markets (None = all configured)
        log_dir: Directory for log files
    """
    interval_name = 'weekly' if interval == '1wk' else 'daily'

    print(f"\n{'=' * 60}")
    print(f"PREFETCH {interval_name.upper()} DATA")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}\n")

    if log_dir:
        log_to_file(log_dir, f"Starting {interval_name} prefetch")

    def progress_callback(current, total, market, status):
        """Progress callback for console output"""
        pct = current / total * 100 if total > 0 else 0
        print(f"  [{pct:5.1f}%] Batch {current}/{total} - {market}: {status}")

    try:
        # Run prefetch
        all_data = market_data_fetcher.prefetch_all_markets(
            markets=markets,
            period='5y',
            interval=interval,
            batch_size=100,
            progress_callback=progress_callback,
            exclude_crypto=True  # Skip crypto as per user preference
        )

        success_count = len(all_data)
        message = f"Prefetch complete: {success_count} symbols cached"
        print(f"\n{message}")

        if log_dir:
            log_to_file(log_dir, message)

        return True

    except Exception as e:
        error_msg = f"Prefetch failed: {e}"
        print(f"\nERROR: {error_msg}")
        logger.error(error_msg)

        if log_dir:
            log_to_file(log_dir, f"ERROR: {error_msg}")

        return False


def main():
    parser = argparse.ArgumentParser(
        description='Pre-download market data for backtesting/simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download weekly data (fastest, good for backtesting)
    python scripts/prefetch_market_data.py

    # Download daily data (more granular, longer)
    python scripts/prefetch_market_data.py --daily

    # Download both weekly and daily
    python scripts/prefetch_market_data.py --both

    # Download specific markets only
    python scripts/prefetch_market_data.py --markets NASDAQ SP500

    # Show cache statistics
    python scripts/prefetch_market_data.py --stats

    # Clear expired cache files
    python scripts/prefetch_market_data.py --clear-expired
        """
    )

    parser.add_argument(
        '--daily',
        action='store_true',
        help='Download daily data (1d interval)'
    )
    parser.add_argument(
        '--weekly',
        action='store_true',
        help='Download weekly data (1wk interval) - default'
    )
    parser.add_argument(
        '--both',
        action='store_true',
        help='Download both daily and weekly data'
    )
    parser.add_argument(
        '--markets',
        nargs='+',
        choices=['NASDAQ', 'SP500', 'EUROPE', 'CAC40', 'DAX', 'ASIA_ADR'],
        help='Specific markets to download (default: all configured)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show cache statistics only'
    )
    parser.add_argument(
        '--clear-expired',
        action='store_true',
        help='Clear expired cache files before prefetch'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'logs'),
        help='Directory for log files'
    )

    args = parser.parse_args()

    # Setup log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("TRADINGBOT V3 - MARKET DATA PREFETCH")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Stats only
    if args.stats:
        show_cache_stats()
        return

    # Clear expired cache
    if args.clear_expired:
        print("\nClearing expired cache files...")
        market_data_fetcher.clear_disk_cache(older_than_hours=24)
        print("Expired cache cleared.")
        log_to_file(log_dir, "Cleared expired cache files")

    # Determine what to download
    do_weekly = args.weekly or args.both or (not args.daily and not args.both)
    do_daily = args.daily or args.both

    success = True

    # Weekly prefetch
    if do_weekly:
        if not run_prefetch('1wk', args.markets, log_dir):
            success = False

    # Daily prefetch
    if do_daily:
        if not run_prefetch('1d', args.markets, log_dir):
            success = False

    # Show final stats
    show_cache_stats()

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("PREFETCH COMPLETED SUCCESSFULLY")
    else:
        print("PREFETCH COMPLETED WITH ERRORS")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    log_to_file(log_dir, f"Prefetch finished - Success: {success}")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
