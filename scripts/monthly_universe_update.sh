#!/bin/bash
# Monthly Universe Update Script
# Runs on 1st of each month at 4:00 AM

cd /root/tradingbot-github
source .venv/bin/activate
python3 src/intelligence/universe_scanner.py --monthly

# Also update inside Docker container
docker compose exec -T agent python3 -c '
from src.intelligence.universe_scanner import UniverseScanner
import asyncio
asyncio.run(UniverseScanner().scan_full_universe(force=True))
' 2>/dev/null || true

echo "Universe update completed at $(date)" >> /var/log/tradingbot-universe-update.log
