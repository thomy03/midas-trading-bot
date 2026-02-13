"""
Activity Logger - Structured activity logging for the trading bot.
Logs all bot actions to data/activity_log.json for API/dashboard consumption.
"""
import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

_LOG_PATH = Path("data/activity_log.json")
_MAX_ENTRIES = 2000
_lock = threading.Lock()


def _ensure_dir():
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_activity(
    activity_type: str,
    symbol: str = "",
    details: str = "",
    scores: Optional[Dict[str, Any]] = None,
    reasoning: str = "",
    strategy: str = "",
    decision: str = "",
    extra: Optional[Dict[str, Any]] = None,
):
    """Append one activity entry to the log file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "type": activity_type,
        "symbol": symbol,
        "details": details,
        "scores": scores or {},
        "reasoning": reasoning,
        "strategy": strategy,
        "decision": decision,
    }
    if extra:
        entry.update(extra)

    with _lock:
        _ensure_dir()
        entries: List[dict] = []
        if _LOG_PATH.exists():
            try:
                with open(_LOG_PATH, "r") as f:
                    entries = json.load(f)
            except Exception:
                entries = []
        entries.append(entry)
        # Keep only last N entries
        if len(entries) > _MAX_ENTRIES:
            entries = entries[-_MAX_ENTRIES:]
        with open(_LOG_PATH, "w") as f:
            json.dump(entries, f, indent=1, default=str)


def get_activities(
    limit: int = 50,
    activity_type: Optional[str] = None,
    strategy: Optional[str] = None,
    symbol: Optional[str] = None,
) -> List[dict]:
    """Read activities with optional filters."""
    _ensure_dir()
    if not _LOG_PATH.exists():
        return []
    try:
        with open(_LOG_PATH, "r") as f:
            entries = json.load(f)
    except Exception:
        return []

    # Filter
    if activity_type:
        entries = [e for e in entries if e.get("type") == activity_type]
    if strategy:
        entries = [e for e in entries if strategy.lower() in e.get("strategy", "").lower()]
    if symbol:
        entries = [e for e in entries if e.get("symbol", "").upper() == symbol.upper()]

    # Return latest first
    entries = list(reversed(entries))
    return entries[:limit]
