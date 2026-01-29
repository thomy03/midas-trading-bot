#!/usr/bin/env python3
"""
Trend Discovery Runner
Script pour exécuter la découverte de tendances manuellement ou via scheduler.

Usage:
    # Scan immédiat
    python scripts/run_trend_discovery.py

    # Scan d'un secteur spécifique
    python scripts/run_trend_discovery.py --sector Technology

    # Démarrer le scheduler
    python scripts/run_trend_discovery.py --schedule

    # Avec un modèle spécifique
    python scripts/run_trend_discovery.py --model anthropic/claude-3-sonnet
"""

import sys
import os
import asyncio
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.intelligence.trend_discovery import (
    TrendDiscovery,
    TrendDiscoveryScheduler,
    get_trend_discovery
)
from config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_immediate_scan(model: str = None, sector: str = None):
    """Exécute un scan immédiat."""
    api_key = settings.OPENROUTER_API_KEY
    model = model or settings.OPENROUTER_MODEL

    if not api_key:
        logger.warning("OPENROUTER_API_KEY not set. Running quantitative analysis only.")

    discovery = TrendDiscovery(
        openrouter_api_key=api_key,
        model=model,
        data_dir=settings.TREND_DATA_DIR
    )

    try:
        await discovery.initialize()

        if sector:
            # Scan d'un secteur spécifique
            logger.info(f"Scanning sector: {sector}")
            result = await discovery.scan_sector(sector)
            print(f"\n{'='*60}")
            print(f"SECTOR ANALYSIS: {sector}")
            print(f"{'='*60}")
            print(f"Analyzed at: {result['analyzed_at']}")
            print(f"Stocks analyzed: {result['stock_count']}")
            print(f"Sector momentum: {result['sector_momentum']:.2%}")
            print(f"High volume stocks: {result['high_volume_count']}")
            print(f"\nTop Performers (5d):")
            for stock in result['top_performers'][:5]:
                print(f"  {stock['symbol']}: {stock['return_5d']:+.2f}% (vol ratio: {stock['volume_ratio']:.1f}x)")
        else:
            # Scan quotidien complet
            logger.info("Running full daily scan...")
            report = await discovery.daily_scan()

            # Afficher le rapport
            print(f"\n{report.summary()}")

            # Afficher les détails supplémentaires
            if report.narrative_updates:
                print(f"\n--- New Narratives Identified ---")
                for narrative in report.narrative_updates[:3]:
                    print(f"  - {narrative.get('name', 'Unknown')}")
                    print(f"    {narrative.get('description', '')[:100]}...")

            # Symboles à surveiller
            if report.watchlist_additions:
                print(f"\n--- Focus Symbols for Screening ---")
                print(f"  {', '.join(report.watchlist_additions[:15])}")

            # Sauvegarder les symboles focus pour le screener
            focus_symbols = discovery.get_focus_symbols(min_confidence=0.4)
            if focus_symbols:
                focus_path = os.path.join(settings.TREND_DATA_DIR, 'focus_symbols.json')
                import json
                with open(focus_path, 'w') as f:
                    json.dump({
                        'date': datetime.now().isoformat(),
                        'symbols': focus_symbols
                    }, f, indent=2)
                logger.info(f"Focus symbols saved to {focus_path}")

    finally:
        await discovery.close()


async def run_scheduler(model: str = None):
    """Démarre le scheduler pour exécution quotidienne."""
    api_key = settings.OPENROUTER_API_KEY
    model = model or settings.OPENROUTER_MODEL
    scan_time = getattr(settings, 'TREND_SCAN_TIME', '06:00')

    if not api_key:
        logger.error("OPENROUTER_API_KEY required for scheduled runs. Set it in .env file.")
        return

    discovery = TrendDiscovery(
        openrouter_api_key=api_key,
        model=model,
        data_dir=settings.TREND_DATA_DIR
    )

    await discovery.initialize()

    scheduler = TrendDiscoveryScheduler(
        discovery=discovery,
        run_time=scan_time,
        timezone=settings.TIMEZONE
    )

    try:
        logger.info(f"Starting TrendDiscovery scheduler (daily at {scan_time})")
        await scheduler.start()

        # Garder le scheduler en vie
        while True:
            await asyncio.sleep(3600)

    except KeyboardInterrupt:
        logger.info("Stopping scheduler...")
    finally:
        await scheduler.stop()
        await discovery.close()


def main():
    parser = argparse.ArgumentParser(
        description='Run Trend Discovery analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_trend_discovery.py                    # Full daily scan
    python scripts/run_trend_discovery.py --sector AI_Semiconductors
    python scripts/run_trend_discovery.py --schedule         # Start scheduler
    python scripts/run_trend_discovery.py --model anthropic/claude-3-sonnet
        """
    )

    parser.add_argument(
        '--sector',
        type=str,
        help='Scan a specific sector (e.g., Technology, AI_Semiconductors, Healthcare)'
    )

    parser.add_argument(
        '--schedule',
        action='store_true',
        help='Start the daily scheduler instead of immediate scan'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='OpenRouter model to use (default: from settings)'
    )

    parser.add_argument(
        '--list-sectors',
        action='store_true',
        help='List available sectors'
    )

    parser.add_argument(
        '--list-themes',
        action='store_true',
        help='List all themes (predefined + learned)'
    )

    parser.add_argument(
        '--learned-themes',
        action='store_true',
        help='Show only themes discovered by LLM'
    )

    args = parser.parse_args()

    if args.list_sectors:
        print("Available sectors:")
        for sector in TrendDiscovery.SECTORS.keys():
            symbols = TrendDiscovery.SECTORS[sector][:5]
            print(f"  {sector}: {', '.join(symbols)}...")
        return

    if args.list_themes:
        discovery = TrendDiscovery(data_dir=settings.TREND_DATA_DIR)
        all_themes = discovery.get_all_themes()
        learned_count = len(all_themes) - len(TrendDiscovery.THEMES_KEYWORDS)

        print(f"\nAll themes ({len(all_themes)} total: {len(TrendDiscovery.THEMES_KEYWORDS)} predefined + {learned_count} learned):\n")

        print("PREDEFINED THEMES:")
        for theme in TrendDiscovery.THEMES_KEYWORDS.keys():
            keywords = TrendDiscovery.THEMES_KEYWORDS[theme][:4]
            print(f"  {theme}: {', '.join(keywords)}...")

        if learned_count > 0:
            print("\nLEARNED THEMES (discovered by LLM):")
            for name, data in discovery._load_learned_themes().items():
                occ = data.get('occurrence_count', 1)
                print(f"  {name} (seen {occ}x): {data.get('description', '')[:60]}...")
        return

    if args.learned_themes:
        discovery = TrendDiscovery(data_dir=settings.TREND_DATA_DIR)
        learned = discovery.get_learned_themes_summary()

        if not learned:
            print("No themes learned yet. Run a daily scan with LLM to discover new themes.")
            return

        print(f"\nLearned Themes ({len(learned)} discovered by LLM):\n")
        for theme in learned:
            print(f"  {theme['name']}")
            print(f"    Seen: {theme['occurrence_count']} times")
            print(f"    Discovered: {theme['discovered_at'][:10]}")
            print(f"    Symbols: {', '.join(theme['symbols']) if theme['symbols'] else 'None'}")
            print(f"    {theme['description']}")
            print()
        return

    if args.schedule:
        asyncio.run(run_scheduler(args.model))
    else:
        asyncio.run(run_immediate_scan(args.model, args.sector))


if __name__ == '__main__':
    main()
