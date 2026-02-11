"""
Signal Tracker - Enregistre chaque signal (pris ou rejeté) avec contexte complet.
Sprint 1C - Feedback Loop
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

SIGNALS_DIR = os.environ.get('MIDAS_SIGNALS_DIR', '/app/data/signals/tracked')


def _ensure_dir():
    Path(SIGNALS_DIR).mkdir(parents=True, exist_ok=True)


def record(signal_data: Dict[str, Any]):
    """
    Record a signal (taken or rejected) with full indicator context.
    
    Expected signal_data keys:
        symbol, score, decision ('BUY'/'SELL'/'HOLD'/'REJECTED'),
        regime, sector,
        pillar_scores: {technical, fundamental, sentiment, news, ml_regime},
        normalized_values: {indicator_name: 0-100 value, ...},
        raw_values: {indicator_name: raw_value, ...},
        rejection_reason: str or None,
        timestamp: ISO string or None
    """
    _ensure_dir()
    
    ts = signal_data.get('timestamp') or datetime.utcnow().isoformat()
    date_str = ts[:10]
    
    entry = {
        'symbol': signal_data.get('symbol', 'UNKNOWN'),
        'score': signal_data.get('score', 0),
        'decision': signal_data.get('decision', 'UNKNOWN'),
        'regime': signal_data.get('regime', 'unknown'),
        'sector': signal_data.get('sector', 'unknown'),
        'pillar_scores': signal_data.get('pillar_scores', {}),
        'normalized_values': signal_data.get('normalized_values', {}),
        'raw_values': signal_data.get('raw_values', {}),
        'rejection_reason': signal_data.get('rejection_reason'),
        'timestamp': ts,
        'key_factors': signal_data.get('key_factors', []),
    }
    
    # Append to daily file
    daily_file = os.path.join(SIGNALS_DIR, f'{date_str}.json')
    
    existing = []
    if os.path.exists(daily_file):
        try:
            with open(daily_file, 'r') as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []
    
    existing.append(entry)
    
    with open(daily_file, 'w') as f:
        json.dump(existing, f, indent=2, default=str)
    
    logger.info(f"[SIGNAL_TRACKER] Recorded {entry['decision']} signal for {entry['symbol']} (score={entry['score']:.1f})")


def get_signals_for_date(date_str: str):
    """Load all signals for a given date (YYYY-MM-DD)."""
    daily_file = os.path.join(SIGNALS_DIR, f'{date_str}.json')
    if not os.path.exists(daily_file):
        return []
    try:
        with open(daily_file, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def get_signals_range(start_date: str, end_date: str):
    """Get all signals between two dates (inclusive)."""
    from datetime import timedelta
    results = []
    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    while current <= end:
        ds = current.strftime('%Y-%m-%d')
        results.extend(get_signals_for_date(ds))
        current += timedelta(days=1)
    return results
