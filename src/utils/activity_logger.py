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
        "agent": os.environ.get("AGENT_ID", "unknown"),
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
    agent: Optional[str] = None,
) -> List[dict]:
    """Read activities with optional filters. Merges LLM + NoLLM logs."""
    entries = []

    # Read from both LLM and NoLLM data dirs
    for log_path in [_LOG_PATH, Path("data-nollm/activity_log.json")]:
        if log_path.exists():
            try:
                with open(log_path, "r") as f:
                    data = json.load(f)
                    # Ensure agent field exists
                    default_agent = "nollm" if "nollm" in str(log_path) else "llm"
                    for e in data:
                        if not e.get("agent"):
                            e["agent"] = default_agent
                    entries.extend(data)
            except Exception:
                pass

    # Filter
    if activity_type:
        entries = [e for e in entries if e.get("type") == activity_type]
    if strategy:
        entries = [e for e in entries if strategy.lower() in e.get("strategy", "").lower()]
    if symbol:
        entries = [e for e in entries if e.get("symbol", "").upper() == symbol.upper()]
    if agent:
        entries = [e for e in entries if e.get("agent", "").lower() == agent.lower()]

    # Sort by timestamp desc, return latest
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return entries[:limit]
