#!/usr/bin/env python3
"""Nightly opportunity analysis - run via cron or daily tasks."""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime, timezone
from src.learning.opportunity_tracker import (
    evaluate_missed_opportunities,
    auto_adjust_thresholds,
    weekly_report,
)
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Opportunity Analysis - Nightly Run ===")

    # 1. Evaluate missed opportunities
    result = evaluate_missed_opportunities(lookback_days=1)
    summary = result.get("summary", {})
    logger.info(f"Evaluations: {summary}")

    # 2. Auto-adjust thresholds
    adj = auto_adjust_thresholds()
    logger.info(f"Threshold adjustment: {adj}")

    # 3. Weekly report on Sundays
    if datetime.now(timezone.utc).weekday() == 6:  # Sunday
        report = weekly_report()
        logger.info(f"Weekly report:\n{report}")
        # Could send via Telegram here
    
    logger.info("=== Done ===")


if __name__ == "__main__":
    main()
