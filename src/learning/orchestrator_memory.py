"""
Orchestrator Memory - Tracks decisions and accuracy over time.

Saves each orchestrator decision (symbol, score, action, timestamp)
and tracks whether the prediction was correct based on subsequent price movement.

Storage: /app/data/orchestrator_memory.json
"""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_PATH = "/app/data/orchestrator_memory.json"


@dataclass
class OrchestratorDecision:
    """A single orchestrator decision record."""
    symbol: str
    total_score: float
    action: str  # 'buy', 'sell', 'hold', 'strong_buy', 'strong_sell'
    timestamp: str
    pillar_scores: Dict[str, float]  # technical, fundamental, sentiment, news
    entry_price: Optional[float] = None
    verified: bool = False
    outcome_price: Optional[float] = None
    outcome_pnl_pct: Optional[float] = None
    correct: Optional[bool] = None  # Was the prediction right?
    verification_date: Optional[str] = None


class OrchestratorMemory:
    """
    Persistent memory for orchestrator decisions.
    Tracks accuracy and provides learning summaries.
    """

    def __init__(self, memory_path: str = DEFAULT_MEMORY_PATH):
        self.memory_path = memory_path
        self.decisions: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load decision history from disk."""
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, 'r') as f:
                    data = json.load(f)
                self.decisions = data.get("decisions", [])
                logger.info(f"[MEMORY] Loaded {len(self.decisions)} decisions from {self.memory_path}")
            except Exception as e:
                logger.warning(f"[MEMORY] Failed to load memory: {e}")
                self.decisions = []
        else:
            logger.info(f"[MEMORY] No existing memory at {self.memory_path}, starting fresh")
            self.decisions = []

    def _save(self):
        """Persist decisions to disk."""
        try:
            os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
            data = {
                "decisions": self.decisions,
                "last_updated": datetime.now().isoformat(),
                "total_decisions": len(self.decisions)
            }
            with open(self.memory_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"[MEMORY] Failed to save memory: {e}")

    def record_decision(
        self,
        symbol: str,
        total_score: float,
        action: str,
        pillar_scores: Dict[str, float],
        entry_price: Optional[float] = None
    ):
        """Record a new orchestrator decision."""
        decision = {
            "symbol": symbol,
            "total_score": round(total_score, 2),
            "action": action,
            "timestamp": datetime.now().isoformat(),
            "pillar_scores": {k: round(v, 2) for k, v in pillar_scores.items()},
            "entry_price": entry_price,
            "verified": False,
            "outcome_price": None,
            "outcome_pnl_pct": None,
            "correct": None,
            "verification_date": None
        }
        self.decisions.append(decision)
        self._save()
        logger.info(f"[MEMORY] Recorded decision: {symbol} -> {action} (score={total_score:.1f})")

    def verify_decision(self, index: int, outcome_price: float):
        """
        Verify a past decision against actual price movement.
        
        Args:
            index: Index in decisions list
            outcome_price: Current/outcome price to compare against entry
        """
        if index < 0 or index >= len(self.decisions):
            return
        
        decision = self.decisions[index]
        if decision["verified"] or decision.get("entry_price") is None:
            return

        entry = decision["entry_price"]
        if entry <= 0:
            return

        pnl_pct = ((outcome_price - entry) / entry) * 100
        action = decision["action"]

        # Determine if prediction was correct
        if action in ("buy", "strong_buy"):
            correct = pnl_pct > 0
        elif action in ("sell", "strong_sell"):
            correct = pnl_pct < 0
        else:
            correct = abs(pnl_pct) < 2  # Hold is correct if price didn't move much

        decision["verified"] = True
        decision["outcome_price"] = round(outcome_price, 2)
        decision["outcome_pnl_pct"] = round(pnl_pct, 2)
        decision["correct"] = correct
        decision["verification_date"] = datetime.now().isoformat()
        self._save()

    def get_unverified_decisions(self) -> List[Dict[str, Any]]:
        """Get decisions that haven't been verified yet."""
        return [d for d in self.decisions if not d.get("verified") and d.get("entry_price")]

    def get_accuracy_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Get accuracy statistics for recent decisions.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dict with accuracy metrics
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        recent = [
            d for d in self.decisions
            if d.get("verified") and d.get("timestamp", "") >= cutoff
        ]

        if not recent:
            return {
                "total_verified": 0,
                "accuracy": 0.0,
                "avg_pnl_pct": 0.0,
                "best_trade": None,
                "worst_trade": None,
                "win_rate_buy": 0.0,
                "period_days": days
            }

        correct_count = sum(1 for d in recent if d.get("correct"))
        pnls = [d["outcome_pnl_pct"] for d in recent if d.get("outcome_pnl_pct") is not None]
        
        buy_decisions = [d for d in recent if d["action"] in ("buy", "strong_buy")]
        buy_wins = sum(1 for d in buy_decisions if d.get("correct"))

        best = max(recent, key=lambda d: d.get("outcome_pnl_pct", -999))
        worst = min(recent, key=lambda d: d.get("outcome_pnl_pct", 999))

        return {
            "total_verified": len(recent),
            "accuracy": round(correct_count / len(recent) * 100, 1) if recent else 0,
            "avg_pnl_pct": round(sum(pnls) / len(pnls), 2) if pnls else 0,
            "best_trade": {"symbol": best["symbol"], "pnl": best.get("outcome_pnl_pct")},
            "worst_trade": {"symbol": worst["symbol"], "pnl": worst.get("outcome_pnl_pct")},
            "win_rate_buy": round(buy_wins / len(buy_decisions) * 100, 1) if buy_decisions else 0,
            "period_days": days
        }

    def get_summary_for_prompt(self, days: int = 14) -> str:
        """
        Generate a concise summary suitable for injection into LLM prompts.
        
        Returns:
            Formatted string with recent accuracy and patterns
        """
        stats = self.get_accuracy_stats(days)
        
        if stats["total_verified"] == 0:
            return "No verified trading history yet. Making decisions without historical accuracy data."

        # Recent decisions (last 5)
        recent_decisions = sorted(
            [d for d in self.decisions if d.get("verified")],
            key=lambda d: d.get("timestamp", ""),
            reverse=True
        )[:5]

        lines = [
            f"=== ORCHESTRATOR MEMORY (last {days} days) ===",
            f"Verified decisions: {stats['total_verified']}",
            f"Overall accuracy: {stats['accuracy']}%",
            f"Average P&L: {stats['avg_pnl_pct']:+.2f}%",
            f"Buy win rate: {stats['win_rate_buy']}%",
        ]

        if stats.get("best_trade"):
            lines.append(f"Best: {stats['best_trade']['symbol']} ({stats['best_trade']['pnl']:+.1f}%)")
        if stats.get("worst_trade"):
            lines.append(f"Worst: {stats['worst_trade']['symbol']} ({stats['worst_trade']['pnl']:+.1f}%)")

        if recent_decisions:
            lines.append("\nRecent verified decisions:")
            for d in recent_decisions:
                emoji = "✅" if d.get("correct") else "❌"
                lines.append(f"  {emoji} {d['symbol']}: {d['action']} (score={d['total_score']}) -> {d.get('outcome_pnl_pct', 0):+.1f}%")

        return "\n".join(lines)

    def get_recent_decisions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the most recent decisions."""
        return sorted(
            self.decisions,
            key=lambda d: d.get("timestamp", ""),
            reverse=True
        )[:limit]


# Singleton
_memory: Optional[OrchestratorMemory] = None


def get_orchestrator_memory(path: str = DEFAULT_MEMORY_PATH) -> OrchestratorMemory:
    """Get or create the OrchestratorMemory singleton."""
    global _memory
    if _memory is None:
        _memory = OrchestratorMemory(path)
    return _memory
