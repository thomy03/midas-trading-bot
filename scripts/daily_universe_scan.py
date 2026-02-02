#!/usr/bin/env python3
"""
Daily Universe Scan - Scanne les 3000+ actions de l'univers
A lancer 1x par jour avant l'ouverture US (14h30 Paris)
"""

import json
import requests
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

API_BASE = 'http://localhost:8000'
BATCH_SIZE = 50  # Symbols per batch
MAX_WORKERS = 4

def load_universe():
    """Load symbols from universe cache"""
    cache_path = Path('/app/data/universe/universe_cache.json')
    with open(cache_path, 'r') as f:
        data = json.load(f)
    
    symbols = []
    for stock in data.get('us', []):
        symbols.append(stock['symbol'])
    for stock in data.get('eu', []):
        symbols.append(stock['symbol'])
    
    return symbols

def screen_batch(symbols: list) -> list:
    """Screen a batch of symbols"""
    try:
        response = requests.post(
            f'{API_BASE}/api/v1/screen',
            json={'symbols': symbols},
            timeout=120
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f'Batch failed: {response.status_code}')
            return []
    except Exception as e:
        logger.error(f'Batch error: {e}')
        return []

def main():
    logger.info('=' * 60)
    logger.info('🔍 DAILY UNIVERSE SCAN')
    logger.info('=' * 60)
    
    # Load universe
    symbols = load_universe()
    logger.info(f'Loaded {len(symbols)} symbols')
    
    # Split into batches
    batches = [symbols[i:i+BATCH_SIZE] for i in range(0, len(symbols), BATCH_SIZE)]
    logger.info(f'Processing {len(batches)} batches...')
    
    all_alerts = []
    start_time = time.time()
    
    for i, batch in enumerate(batches):
        logger.info(f'Batch {i+1}/{len(batches)} ({len(batch)} symbols)...')
        alerts = screen_batch(batch)
        all_alerts.extend(alerts)
        
        if alerts:
            for a in alerts:
                logger.info(f"  🎯 {a['symbol']}: {a['recommendation']} (conf: {a.get('confidence_score', 0):.0f})")
        
        # Rate limit
        time.sleep(2)
    
    elapsed = time.time() - start_time
    logger.info('')
    logger.info('=' * 60)
    logger.info(f'✅ Scan complete in {elapsed:.0f}s')
    logger.info(f'📊 Found {len(all_alerts)} signals')
    
    # Summary
    if all_alerts:
        by_rec = {}
        for a in all_alerts:
            rec = a['recommendation']
            by_rec[rec] = by_rec.get(rec, 0) + 1
        logger.info(f'   By type: {by_rec}')
        
        # Save results
        results_path = Path('/app/data/signals/daily_scan.json')
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'total_scanned': len(symbols),
                'signals_found': len(all_alerts),
                'alerts': all_alerts
            }, f, indent=2)
        logger.info(f'   Saved to {results_path}')

if __name__ == '__main__':
    main()
