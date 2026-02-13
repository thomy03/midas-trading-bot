"""
Source Reliability Tracker V8.2

Tracks prediction accuracy per intelligence source (Grok, RSS feeds, Gemini Research,
Reddit, etc.) and auto-weights them by historical hit rate.

Mechanism:
1. When IntelligenceOrchestrator produces an adjustment for a symbol, log it with source.
2. At J+1/J+5/J+20, check actual price movement.
3. Calculate hit rate per source.
4. Expose weights to the orchestrator for source-level confidence scaling.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SourcePrediction:
    """A prediction logged from an intelligence source."""
    source: str             # e.g. 'grok', 'rss_reuters', 'gemini_research', 'reddit'
    symbol: str
    adjustment: float       # predicted adjustment (-15 to +15)
    direction: str          # 'bullish', 'bearish', 'neutral'
    confidence: float       # 0-1
    timestamp: str          # ISO format
    # Outcome tracking (filled later)
    price_at_prediction: Optional[float] = None
    price_j1: Optional[float] = None
    price_j5: Optional[float] = None
    price_j20: Optional[float] = None
    outcome_j1: Optional[str] = None   # 'correct', 'wrong', 'neutral'
    outcome_j5: Optional[str] = None
    outcome_checked: bool = False


@dataclass
class SourceStats:
    """Aggregated reliability stats for a source."""
    source: str
    total_predictions: int = 0
    checked_predictions: int = 0
    correct_j1: int = 0
    correct_j5: int = 0
    wrong_j1: int = 0
    wrong_j5: int = 0
    avg_confidence: float = 0.5
    hit_rate_j1: float = 0.5
    hit_rate_j5: float = 0.5
    reliability_weight: float = 1.0  # final weight (0.3 to 1.5)


class SourceReliabilityTracker:
    """
    Tracks and scores intelligence source reliability over time.

    Usage:
        tracker = SourceReliabilityTracker()
        # Log a prediction
        tracker.log_prediction('grok', 'NVDA', adjustment=+8.0, direction='bullish',
                               confidence=0.8, price=850.0)
        # Later: check outcomes
        await tracker.check_outcomes(price_fetcher)
        # Get source weights for orchestrator
        weights = tracker.get_source_weights()
        # -> {'grok': 1.2, 'rss_reuters': 0.9, 'reddit': 0.5, ...}
    """

    def __init__(self, data_dir: str = "data/intelligence"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._predictions_file = self._data_dir / "source_predictions.json"
        self._stats_file = self._data_dir / "source_stats.json"
        self._predictions: List[SourcePrediction] = []
        self._stats: Dict[str, SourceStats] = {}
        self._load()

    def _load(self):
        """Load predictions and stats from disk."""
        # Load predictions
        if self._predictions_file.exists():
            try:
                with open(self._predictions_file, 'r') as f:
                    data = json.load(f)
                    self._predictions = [
                        SourcePrediction(**p) for p in data.get('predictions', [])[-2000:]
                    ]
                logger.info(f"[SOURCE_TRACKER] Loaded {len(self._predictions)} predictions")
            except Exception as e:
                logger.warning(f"[SOURCE_TRACKER] Failed to load predictions: {e}")

        # Load stats
        if self._stats_file.exists():
            try:
                with open(self._stats_file, 'r') as f:
                    data = json.load(f)
                    for name, s in data.get('sources', {}).items():
                        self._stats[name] = SourceStats(**s)
                logger.info(f"[SOURCE_TRACKER] Loaded stats for {len(self._stats)} sources")
            except Exception as e:
                logger.warning(f"[SOURCE_TRACKER] Failed to load stats: {e}")

    def _save(self):
        """Persist to disk."""
        try:
            # Keep last 2000 predictions
            recent = self._predictions[-2000:]
            with open(self._predictions_file, 'w') as f:
                json.dump({
                    'updated_at': datetime.now().isoformat(),
                    'total': len(recent),
                    'predictions': [asdict(p) for p in recent]
                }, f, indent=2)

            with open(self._stats_file, 'w') as f:
                json.dump({
                    'updated_at': datetime.now().isoformat(),
                    'sources': {name: asdict(s) for name, s in self._stats.items()}
                }, f, indent=2)
        except Exception as e:
            logger.error(f"[SOURCE_TRACKER] Save failed: {e}")

    # ── Logging Predictions ──────────────────────────────────────────────

    def log_prediction(self, source: str, symbol: str, adjustment: float,
                       direction: str, confidence: float, price: Optional[float] = None):
        """Log a new prediction from an intelligence source."""
        pred = SourcePrediction(
            source=source,
            symbol=symbol.upper(),
            adjustment=adjustment,
            direction=direction,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            price_at_prediction=price,
        )
        self._predictions.append(pred)

        # Update running stats
        if source not in self._stats:
            self._stats[source] = SourceStats(source=source)
        self._stats[source].total_predictions += 1

        # Auto-save every 20 predictions
        if len(self._predictions) % 20 == 0:
            self._save()

    def log_brief_predictions(self, brief, source_map: Optional[Dict] = None):
        """Log all predictions from an IntelligenceBrief.

        Args:
            brief: IntelligenceBrief object
            source_map: Optional override mapping event source to tracker source name
        """
        for event in getattr(brief, 'events', []):
            src = source_map.get(event.source, event.source) if source_map else event.source
            for symbol, adj in event.affected_symbols.items():
                direction = 'bullish' if adj > 0 else ('bearish' if adj < 0 else 'neutral')
                self.log_prediction(
                    source=src, symbol=symbol, adjustment=adj,
                    direction=direction, confidence=event.confidence,
                )

    # ── Outcome Checking ─────────────────────────────────────────────────

    async def check_outcomes(self, price_fetcher=None):
        """Check outcomes for unchecked predictions that are old enough.

        Args:
            price_fetcher: async callable(symbol) -> float (current price)
                          If None, uses yfinance.
        """
        now = datetime.now()
        checked_count = 0

        for pred in self._predictions:
            if pred.outcome_checked:
                continue
            if pred.price_at_prediction is None:
                continue

            pred_time = datetime.fromisoformat(pred.timestamp)
            age_days = (now - pred_time).total_seconds() / 86400

            # Need at least 1 day for J+1 check
            if age_days < 1:
                continue

            # Get current price
            current_price = None
            if price_fetcher:
                try:
                    current_price = await price_fetcher(pred.symbol)
                except Exception:
                    pass

            if current_price is None:
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(pred.symbol)
                    current_price = float(ticker.fast_info.get('lastPrice', 0))
                    if current_price <= 0:
                        continue
                except Exception:
                    continue

            entry_price = pred.price_at_prediction

            # J+1 check
            if age_days >= 1 and pred.outcome_j1 is None:
                pred.price_j1 = current_price
                move_pct = (current_price - entry_price) / entry_price * 100
                pred.outcome_j1 = self._evaluate_outcome(pred.direction, move_pct)
                self._update_stats(pred.source, 'j1', pred.outcome_j1)

            # J+5 check
            if age_days >= 5 and pred.outcome_j5 is None:
                pred.price_j5 = current_price
                move_pct = (current_price - entry_price) / entry_price * 100
                pred.outcome_j5 = self._evaluate_outcome(pred.direction, move_pct)
                self._update_stats(pred.source, 'j5', pred.outcome_j5)
                pred.outcome_checked = True
                checked_count += 1

        if checked_count > 0:
            self._recalculate_weights()
            self._save()
            logger.info(f"[SOURCE_TRACKER] Checked outcomes for {checked_count} predictions")

    @staticmethod
    def _evaluate_outcome(predicted_direction: str, actual_move_pct: float) -> str:
        """Evaluate if a prediction was correct."""
        if predicted_direction == 'neutral':
            return 'neutral'
        if predicted_direction == 'bullish':
            if actual_move_pct > 1.0:
                return 'correct'
            elif actual_move_pct < -1.0:
                return 'wrong'
            return 'neutral'
        if predicted_direction == 'bearish':
            if actual_move_pct < -1.0:
                return 'correct'
            elif actual_move_pct > 1.0:
                return 'wrong'
            return 'neutral'
        return 'neutral'

    def _update_stats(self, source: str, horizon: str, outcome: str):
        """Update source stats with a new outcome."""
        if source not in self._stats:
            self._stats[source] = SourceStats(source=source)
        stats = self._stats[source]
        stats.checked_predictions += 1

        if horizon == 'j1':
            if outcome == 'correct':
                stats.correct_j1 += 1
            elif outcome == 'wrong':
                stats.wrong_j1 += 1
        elif horizon == 'j5':
            if outcome == 'correct':
                stats.correct_j5 += 1
            elif outcome == 'wrong':
                stats.wrong_j5 += 1

    def _recalculate_weights(self):
        """Recalculate reliability weights for all sources."""
        for source, stats in self._stats.items():
            # J+1 hit rate
            total_j1 = stats.correct_j1 + stats.wrong_j1
            if total_j1 >= 5:  # need minimum sample size
                stats.hit_rate_j1 = stats.correct_j1 / total_j1
            else:
                stats.hit_rate_j1 = 0.5  # prior

            # J+5 hit rate
            total_j5 = stats.correct_j5 + stats.wrong_j5
            if total_j5 >= 5:
                stats.hit_rate_j5 = stats.correct_j5 / total_j5
            else:
                stats.hit_rate_j5 = 0.5

            # Combined weight: J+5 matters more (60/40 split)
            combined_rate = stats.hit_rate_j1 * 0.4 + stats.hit_rate_j5 * 0.6

            # Map hit rate to weight: 0.3 (terrible) to 1.5 (excellent)
            # 50% hit rate = 1.0 (neutral)
            # 70% hit rate = 1.5 (max boost)
            # 30% hit rate = 0.3 (max penalty)
            stats.reliability_weight = max(0.3, min(1.5, 0.5 + combined_rate * 2.0))

    # ── Querying ─────────────────────────────────────────────────────────

    def get_source_weights(self) -> Dict[str, float]:
        """Get reliability weights for all tracked sources.
        Returns dict: source_name -> weight (0.3 to 1.5).
        Used by IntelligenceOrchestrator to scale adjustments."""
        return {name: stats.reliability_weight for name, stats in self._stats.items()}

    def get_source_stats(self) -> Dict[str, dict]:
        """Get detailed stats for all sources (for dashboard/debugging)."""
        return {name: asdict(stats) for name, stats in self._stats.items()}

    def get_best_sources(self, limit: int = 5) -> List[str]:
        """Get the most reliable source names."""
        sorted_sources = sorted(
            self._stats.values(),
            key=lambda s: s.reliability_weight,
            reverse=True
        )
        return [s.source for s in sorted_sources[:limit]]

    def get_worst_sources(self, limit: int = 3) -> List[str]:
        """Get the least reliable source names."""
        sorted_sources = sorted(
            self._stats.values(),
            key=lambda s: s.reliability_weight,
        )
        return [s.source for s in sorted_sources[:limit] if s.checked_predictions >= 5]

    def get_summary(self) -> str:
        """Human-readable summary of source reliability."""
        lines = ["Source Reliability Summary:"]
        for name, stats in sorted(self._stats.items(), key=lambda x: x[1].reliability_weight, reverse=True):
            lines.append(
                f"  {name}: weight={stats.reliability_weight:.2f} "
                f"hit_j1={stats.hit_rate_j1:.0%} hit_j5={stats.hit_rate_j5:.0%} "
                f"({stats.checked_predictions} checked/{stats.total_predictions} total)"
            )
        return '\n'.join(lines)


# Singleton
_tracker_instance: Optional[SourceReliabilityTracker] = None


def get_source_tracker() -> SourceReliabilityTracker:
    """Get or create the global source tracker."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = SourceReliabilityTracker()
    return _tracker_instance
