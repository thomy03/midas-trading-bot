"""
Opportunity Tracker - Track rejected signals and evaluate missed opportunities.
Helps determine if scoring thresholds are too strict or too lax.
"""

import json
import logging
import os
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Paths
_BASE = Path(__file__).resolve().parent.parent.parent  # tradingbot-github/
DATA_DIR = _BASE / "data"
REJECTED_SIGNALS_FILE = DATA_DIR / "rejected_signals.json"
OPPORTUNITY_ANALYSIS_FILE = DATA_DIR / "opportunity_analysis.json"
THRESHOLD_HISTORY_FILE = DATA_DIR / "threshold_history.json"

_write_lock = Lock()


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. log_rejection
# ---------------------------------------------------------------------------

def log_rejection(
    symbol: str,
    score: float,
    reason: str,
    price: float,
    pillars_detail: Optional[Dict] = None,
    timestamp: Optional[str] = None,
    regime: str = "",
    extra: Optional[Dict] = None,
):
    """Append a rejection entry (JSONL) to rejected_signals.json."""
    _ensure_data_dir()
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    entry = {
        "symbol": symbol,
        "score": round(float(score), 2),
        "reason": reason,
        "price": round(float(price), 4) if price else 0,
        "pillars": pillars_detail or {},
        "regime": regime,
        "timestamp": ts,
    }
    if extra:
        entry.update(extra)
    with _write_lock:
        with open(REJECTED_SIGNALS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    logger.debug(f"[OPP_TRACKER] Logged rejection: {symbol} score={score} reason={reason}")


# ---------------------------------------------------------------------------
# 2. evaluate_missed_opportunities
# ---------------------------------------------------------------------------

def _load_rejections(lookback_days: int = 1) -> List[Dict]:
    if not REJECTED_SIGNALS_FILE.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    results = []
    with open(REJECTED_SIGNALS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    results.append(entry)
            except Exception:
                continue
    return results


def evaluate_missed_opportunities(lookback_days: int = 1) -> Dict:
    """Evaluate rejected signals by checking current prices via yfinance."""
    import yfinance as yf

    rejections = _load_rejections(lookback_days)
    if not rejections:
        logger.info("[OPP_TRACKER] No rejections to evaluate")
        return {"evaluations": [], "summary": {}}

    # Group by symbol to batch yfinance calls
    symbols = list({r["symbol"] for r in rejections})
    current_prices = {}
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            p = t.fast_info.get("lastPrice") or t.fast_info.get("regularMarketPrice", 0)
            if p and p > 0:
                current_prices[sym] = float(p)
        except Exception as e:
            logger.debug(f"[OPP_TRACKER] Price fetch failed for {sym}: {e}")

    evaluations = []
    stats = {"missed_opportunity": 0, "correct_rejection": 0, "neutral": 0, "total": 0}

    for rej in rejections:
        sym = rej["symbol"]
        rej_price = float(rej.get("price", 0))
        cur_price = current_prices.get(sym)
        if not cur_price or not rej_price or rej_price <= 0:
            continue

        pct_change = ((cur_price - rej_price) / rej_price) * 100
        if pct_change > 2:
            classification = "missed_opportunity"
        elif pct_change < 0:
            classification = "correct_rejection"
        else:
            classification = "neutral"

        stats[classification] += 1
        stats["total"] += 1

        evaluations.append({
            "symbol": sym,
            "rejection_score": rej["score"],
            "reason": rej["reason"],
            "rejection_price": round(rej_price, 4),
            "current_price": round(cur_price, 4),
            "pct_change": round(pct_change, 2),
            "classification": classification,
            "rejection_time": rej["timestamp"],
            "regime": rej.get("regime", ""),
            "pillars": rej.get("pillars", {}),
        })

    # Sort by pct_change desc (biggest missed opportunities first)
    evaluations.sort(key=lambda x: x["pct_change"], reverse=True)

    result = {
        "evaluations": evaluations,
        "summary": stats,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Save
    _ensure_data_dir()
    with _write_lock:
        with open(OPPORTUNITY_ANALYSIS_FILE, "w") as f:
            json.dump(result, f, indent=2)

    miss_rate = (stats["missed_opportunity"] / stats["total"] * 100) if stats["total"] > 0 else 0
    logger.info(f"[OPP_TRACKER] Evaluated {stats['total']} rejections: "
                f"{stats['missed_opportunity']} missed, {stats['correct_rejection']} correct, "
                f"{stats['neutral']} neutral ({miss_rate:.1f}% miss rate)")

    return result


# ---------------------------------------------------------------------------
# 3. get_threshold_recommendation
# ---------------------------------------------------------------------------

def get_threshold_recommendation() -> Dict:
    """Analyze rejection stats and recommend threshold adjustments."""
    if not OPPORTUNITY_ANALYSIS_FILE.exists():
        return {"recommendation": "no_data", "stats": {}, "suggested_adjustment": 0}

    with open(OPPORTUNITY_ANALYSIS_FILE, "r") as f:
        data = json.load(f)

    evals = data.get("evaluations", [])
    summary = data.get("summary", {})
    total = summary.get("total", 0)
    if total == 0:
        return {"recommendation": "no_data", "stats": summary, "suggested_adjustment": 0}

    missed = summary.get("missed_opportunity", 0)
    correct = summary.get("correct_rejection", 0)
    miss_rate = missed / total

    # Per-regime breakdown
    regime_stats = {}
    for e in evals:
        r = e.get("regime", "UNKNOWN") or "UNKNOWN"
        if r not in regime_stats:
            regime_stats[r] = {"missed_opportunity": 0, "correct_rejection": 0, "neutral": 0, "total": 0}
        regime_stats[r][e["classification"]] += 1
        regime_stats[r]["total"] += 1

    if miss_rate > 0.30:
        recommendation = "lower_threshold"
        # Scale adjustment: 30% → -1, 50%+ → -3
        adjustment = -min(3, max(1, int((miss_rate - 0.2) * 10)))
    elif miss_rate < 0.10:
        recommendation = "thresholds_ok"
        # Could even raise if very few misses
        adjustment = 1 if miss_rate < 0.05 and total >= 10 else 0
    else:
        recommendation = "moderate"
        adjustment = 0

    # Average missed opportunity score (how close were they to passing?)
    missed_scores = [e["rejection_score"] for e in evals if e["classification"] == "missed_opportunity"]
    avg_missed_score = sum(missed_scores) / len(missed_scores) if missed_scores else 0

    return {
        "recommendation": recommendation,
        "miss_rate": round(miss_rate * 100, 1),
        "stats": summary,
        "regime_breakdown": regime_stats,
        "avg_missed_score": round(avg_missed_score, 1),
        "suggested_adjustment": adjustment,
    }


# ---------------------------------------------------------------------------
# 4. auto_adjust_thresholds
# ---------------------------------------------------------------------------

def auto_adjust_thresholds(config_path: str = "config/midas_config.yaml") -> Dict:
    """Apply threshold recommendations gradually. Max ±3/day, bounds [55, 90]."""
    import yaml

    rec = get_threshold_recommendation()
    adjustment = rec.get("suggested_adjustment", 0)
    if adjustment == 0:
        logger.info(f"[OPP_TRACKER] No threshold adjustment needed ({rec['recommendation']})")
        return {"adjusted": False, "recommendation": rec}

    cfg_path = _BASE / config_path
    if not cfg_path.exists():
        logger.warning(f"[OPP_TRACKER] Config not found: {cfg_path}")
        return {"adjusted": False, "error": "config_not_found"}

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    scoring = config.get("scoring", {})
    thresholds = scoring.get("thresholds", {})
    old_buy = thresholds.get("buy", 55)
    old_strong = thresholds.get("strong_buy", 70)

    # Clamp adjustment
    adjustment = max(-3, min(3, adjustment))
    new_buy = max(55, min(90, old_buy + adjustment))
    new_strong = max(55, min(90, old_strong + adjustment))

    if new_buy == old_buy and new_strong == old_strong:
        return {"adjusted": False, "recommendation": rec}

    thresholds["buy"] = new_buy
    thresholds["strong_buy"] = new_strong
    scoring["thresholds"] = thresholds
    config["scoring"] = scoring

    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Log to history
    history_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "old_buy": old_buy,
        "new_buy": new_buy,
        "old_strong_buy": old_strong,
        "new_strong_buy": new_strong,
        "adjustment": adjustment,
        "miss_rate": rec.get("miss_rate", 0),
        "recommendation": rec["recommendation"],
    }
    _ensure_data_dir()
    with _write_lock:
        with open(THRESHOLD_HISTORY_FILE, "a") as f:
            f.write(json.dumps(history_entry) + "\n")

    logger.info(f"[OPP_TRACKER] Thresholds adjusted: buy {old_buy}→{new_buy}, strong_buy {old_strong}→{new_strong}")
    return {"adjusted": True, "old": {"buy": old_buy, "strong_buy": old_strong},
            "new": {"buy": new_buy, "strong_buy": new_strong}, "recommendation": rec}


# ---------------------------------------------------------------------------
# 5. weekly_report
# ---------------------------------------------------------------------------

def weekly_report() -> str:
    """Generate a Telegram-formatted weekly report."""
    # Load all rejections from last 7 days
    rejections = _load_rejections(lookback_days=7)

    # Load latest analysis
    evals = []
    if OPPORTUNITY_ANALYSIS_FILE.exists():
        try:
            with open(OPPORTUNITY_ANALYSIS_FILE, "r") as f:
                data = json.load(f)
            evals = data.get("evaluations", [])
        except Exception:
            pass

    total_rejections = len(rejections)
    missed = [e for e in evals if e["classification"] == "missed_opportunity"]
    correct = [e for e in evals if e["classification"] == "correct_rejection"]
    miss_rate = (len(missed) / len(evals) * 100) if evals else 0

    lines = [
        "📊 *Opportunity Tracker - Weekly Report*",
        f"━━━━━━━━━━━━━━━━━━━━━━━━",
        f"📋 Total rejections: *{total_rejections}*",
        f"🔍 Evaluated: *{len(evals)}*",
        f"❌ Missed opportunities: *{len(missed)}* ({miss_rate:.1f}%)",
        f"✅ Correct rejections: *{len(correct)}*",
        "",
    ]

    # Top 5 missed
    if missed:
        missed_sorted = sorted(missed, key=lambda x: x["pct_change"], reverse=True)[:5]
        lines.append("🔥 *Top 5 Missed Opportunities:*")
        for i, m in enumerate(missed_sorted, 1):
            lines.append(
                f"  {i}. *{m['symbol']}* +{m['pct_change']:.1f}% "
                f"(score: {m['rejection_score']}, reason: {m['reason']})"
            )
        lines.append("")

    # Threshold evolution
    if THRESHOLD_HISTORY_FILE.exists():
        history = []
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        try:
            with open(THRESHOLD_HISTORY_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    ts = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts >= cutoff:
                        history.append(entry)
        except Exception:
            pass

        if history:
            lines.append("📈 *Threshold Changes This Week:*")
            for h in history[-5:]:
                lines.append(
                    f"  • buy: {h['old_buy']}→{h['new_buy']}, "
                    f"strong: {h['old_strong_buy']}→{h['new_strong_buy']} "
                    f"(miss rate: {h.get('miss_rate', '?')}%)"
                )

    rec = get_threshold_recommendation()
    if rec.get("recommendation") != "no_data":
        lines.append("")
        lines.append(f"💡 *Recommendation:* {rec['recommendation']} (adj: {rec['suggested_adjustment']:+d})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: singleton / async wrappers
# ---------------------------------------------------------------------------

_instance = None

def get_opportunity_tracker():
    """Return module-level functions (no class needed, but provides namespace)."""
    return {
        "log_rejection": log_rejection,
        "evaluate_missed_opportunities": evaluate_missed_opportunities,
        "get_threshold_recommendation": get_threshold_recommendation,
        "auto_adjust_thresholds": auto_adjust_thresholds,
        "weekly_report": weekly_report,
    }
