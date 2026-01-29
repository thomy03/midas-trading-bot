"""
Pattern Extractor - Automatic rule extraction from trade history.

This module analyzes historical trades to:
1. Identify common patterns in winning/losing trades
2. Extract actionable rules
3. Validate rules against recent performance
4. Generate human-readable insights

Uses statistical analysis and clustering to find patterns.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import statistics

from .trade_memory import (
    StoredTrade, TradeContext, TradeOutcome,
    LearnedRule, RuleStatus, AdaptiveTradeMemory, get_trade_memory
)

logger = logging.getLogger(__name__)


@dataclass
class PatternCandidate:
    """A potential pattern identified from trades"""
    description: str
    conditions: Dict[str, Any]
    trades: List[StoredTrade]
    win_rate: float
    avg_pnl: float
    sample_size: int
    confidence: float  # Statistical confidence

    @property
    def is_significant(self) -> bool:
        """Check if pattern is statistically significant"""
        return self.sample_size >= 5 and self.confidence >= 0.7


@dataclass
class ExtractionResult:
    """Result of pattern extraction"""
    timestamp: str
    trades_analyzed: int
    patterns_found: int
    rules_created: int
    rules_updated: int
    insights: List[str]
    new_rules: List[LearnedRule] = field(default_factory=list)


class PatternExtractor:
    """
    Extracts trading patterns from historical data.

    Looks for patterns based on:
    - Technical indicators (RSI levels, volume, EMA alignment)
    - Market context (regime, VIX, sector momentum)
    - Sentiment signals (Reddit, Grok, heat score)
    - Time-based patterns (day of week, hour)
    """

    # Thresholds for pattern detection
    MIN_SAMPLE_SIZE = 5
    MIN_WIN_RATE_DIFF = 0.10  # Pattern must show 10%+ difference from baseline
    MIN_CONFIDENCE = 0.70

    # Buckets for continuous variables
    RSI_BUCKETS = [(0, 30, "oversold"), (30, 50, "low"), (50, 70, "neutral"), (70, 100, "overbought")]
    VOLUME_BUCKETS = [(0, 1.0, "low"), (1.0, 1.5, "normal"), (1.5, 2.0, "high"), (2.0, float('inf'), "very_high")]
    VIX_BUCKETS = [(0, 15, "low"), (15, 20, "moderate"), (20, 30, "high"), (30, float('inf'), "extreme")]
    HEAT_BUCKETS = [(0, 0.3, "cold"), (0.3, 0.5, "warm"), (0.5, 0.7, "hot"), (0.7, 1.0, "very_hot")]

    def __init__(self, memory: Optional[AdaptiveTradeMemory] = None):
        """
        Initialize the pattern extractor.

        Args:
            memory: TradeMemory instance (uses singleton if not provided)
        """
        self.memory = memory or get_trade_memory()

    def _bucketize(self, value: float, buckets: List[Tuple[float, float, str]]) -> str:
        """Assign a value to a bucket"""
        for low, high, label in buckets:
            if low <= value < high:
                return label
        return "unknown"

    def _get_bucket_conditions(self, bucket: str, buckets: List[Tuple[float, float, str]]) -> Dict[str, Any]:
        """Get conditions for a bucket"""
        for low, high, label in buckets:
            if label == bucket:
                if high == float('inf'):
                    return {"op": ">=", "value": low}
                elif low == 0:
                    return {"op": "<", "value": high}
                else:
                    return {"op": "range", "low": low, "high": high}
        return {}

    def extract_patterns(
        self,
        trades: List[StoredTrade],
        min_sample: int = None
    ) -> List[PatternCandidate]:
        """
        Extract patterns from a list of trades.

        Args:
            trades: Historical trades to analyze
            min_sample: Minimum sample size for pattern

        Returns:
            List of pattern candidates
        """
        if not trades:
            return []

        min_sample = min_sample or self.MIN_SAMPLE_SIZE
        patterns = []

        # Calculate baseline win rate
        baseline_win_rate = sum(1 for t in trades if t.outcome.is_win) / len(trades)
        logger.info(f"Baseline win rate: {baseline_win_rate:.1%} ({len(trades)} trades)")

        # 1. RSI-based patterns
        patterns.extend(self._extract_rsi_patterns(trades, baseline_win_rate, min_sample))

        # 2. Volume-based patterns
        patterns.extend(self._extract_volume_patterns(trades, baseline_win_rate, min_sample))

        # 3. Market regime patterns
        patterns.extend(self._extract_regime_patterns(trades, baseline_win_rate, min_sample))

        # 4. VIX-based patterns
        patterns.extend(self._extract_vix_patterns(trades, baseline_win_rate, min_sample))

        # 5. Heat score patterns
        patterns.extend(self._extract_heat_patterns(trades, baseline_win_rate, min_sample))

        # 6. Combined patterns (most valuable)
        patterns.extend(self._extract_combined_patterns(trades, baseline_win_rate, min_sample))

        # Filter significant patterns
        significant = [p for p in patterns if p.is_significant]
        logger.info(f"Found {len(significant)} significant patterns out of {len(patterns)} candidates")

        return significant

    def _extract_rsi_patterns(
        self,
        trades: List[StoredTrade],
        baseline: float,
        min_sample: int
    ) -> List[PatternCandidate]:
        """Extract RSI-based patterns"""
        patterns = []
        grouped = defaultdict(list)

        for trade in trades:
            bucket = self._bucketize(trade.context.rsi, self.RSI_BUCKETS)
            grouped[bucket].append(trade)

        for bucket, bucket_trades in grouped.items():
            if len(bucket_trades) < min_sample:
                continue

            win_rate = sum(1 for t in bucket_trades if t.outcome.is_win) / len(bucket_trades)
            diff = win_rate - baseline

            if abs(diff) >= self.MIN_WIN_RATE_DIFF:
                avg_pnl = statistics.mean(t.outcome.pnl_pct for t in bucket_trades)
                confidence = self._calculate_confidence(len(bucket_trades), win_rate, baseline)

                patterns.append(PatternCandidate(
                    description=f"RSI {bucket}: win rate {win_rate:.1%} vs baseline {baseline:.1%}",
                    conditions={"rsi": self._get_bucket_conditions(bucket, self.RSI_BUCKETS)},
                    trades=bucket_trades,
                    win_rate=win_rate,
                    avg_pnl=avg_pnl,
                    sample_size=len(bucket_trades),
                    confidence=confidence
                ))

        return patterns

    def _extract_volume_patterns(
        self,
        trades: List[StoredTrade],
        baseline: float,
        min_sample: int
    ) -> List[PatternCandidate]:
        """Extract volume-based patterns"""
        patterns = []
        grouped = defaultdict(list)

        for trade in trades:
            bucket = self._bucketize(trade.context.volume_ratio, self.VOLUME_BUCKETS)
            grouped[bucket].append(trade)

        for bucket, bucket_trades in grouped.items():
            if len(bucket_trades) < min_sample:
                continue

            win_rate = sum(1 for t in bucket_trades if t.outcome.is_win) / len(bucket_trades)
            diff = win_rate - baseline

            if abs(diff) >= self.MIN_WIN_RATE_DIFF:
                avg_pnl = statistics.mean(t.outcome.pnl_pct for t in bucket_trades)
                confidence = self._calculate_confidence(len(bucket_trades), win_rate, baseline)

                patterns.append(PatternCandidate(
                    description=f"Volume {bucket}: win rate {win_rate:.1%}",
                    conditions={"volume_ratio": self._get_bucket_conditions(bucket, self.VOLUME_BUCKETS)},
                    trades=bucket_trades,
                    win_rate=win_rate,
                    avg_pnl=avg_pnl,
                    sample_size=len(bucket_trades),
                    confidence=confidence
                ))

        return patterns

    def _extract_regime_patterns(
        self,
        trades: List[StoredTrade],
        baseline: float,
        min_sample: int
    ) -> List[PatternCandidate]:
        """Extract market regime patterns"""
        patterns = []
        grouped = defaultdict(list)

        for trade in trades:
            grouped[trade.context.market_regime].append(trade)

        for regime, regime_trades in grouped.items():
            if len(regime_trades) < min_sample:
                continue

            win_rate = sum(1 for t in regime_trades if t.outcome.is_win) / len(regime_trades)
            diff = win_rate - baseline

            if abs(diff) >= self.MIN_WIN_RATE_DIFF:
                avg_pnl = statistics.mean(t.outcome.pnl_pct for t in regime_trades)
                confidence = self._calculate_confidence(len(regime_trades), win_rate, baseline)

                patterns.append(PatternCandidate(
                    description=f"Market regime {regime}: win rate {win_rate:.1%}",
                    conditions={"market_regime": {"op": "==", "value": regime}},
                    trades=regime_trades,
                    win_rate=win_rate,
                    avg_pnl=avg_pnl,
                    sample_size=len(regime_trades),
                    confidence=confidence
                ))

        return patterns

    def _extract_vix_patterns(
        self,
        trades: List[StoredTrade],
        baseline: float,
        min_sample: int
    ) -> List[PatternCandidate]:
        """Extract VIX-based patterns"""
        patterns = []
        grouped = defaultdict(list)

        for trade in trades:
            bucket = self._bucketize(trade.context.vix_level, self.VIX_BUCKETS)
            grouped[bucket].append(trade)

        for bucket, bucket_trades in grouped.items():
            if len(bucket_trades) < min_sample:
                continue

            win_rate = sum(1 for t in bucket_trades if t.outcome.is_win) / len(bucket_trades)
            diff = win_rate - baseline

            if abs(diff) >= self.MIN_WIN_RATE_DIFF:
                avg_pnl = statistics.mean(t.outcome.pnl_pct for t in bucket_trades)
                confidence = self._calculate_confidence(len(bucket_trades), win_rate, baseline)

                patterns.append(PatternCandidate(
                    description=f"VIX {bucket}: win rate {win_rate:.1%}",
                    conditions={"vix_level": self._get_bucket_conditions(bucket, self.VIX_BUCKETS)},
                    trades=bucket_trades,
                    win_rate=win_rate,
                    avg_pnl=avg_pnl,
                    sample_size=len(bucket_trades),
                    confidence=confidence
                ))

        return patterns

    def _extract_heat_patterns(
        self,
        trades: List[StoredTrade],
        baseline: float,
        min_sample: int
    ) -> List[PatternCandidate]:
        """Extract heat score patterns"""
        patterns = []
        grouped = defaultdict(list)

        for trade in trades:
            bucket = self._bucketize(trade.context.heat_score, self.HEAT_BUCKETS)
            grouped[bucket].append(trade)

        for bucket, bucket_trades in grouped.items():
            if len(bucket_trades) < min_sample:
                continue

            win_rate = sum(1 for t in bucket_trades if t.outcome.is_win) / len(bucket_trades)
            diff = win_rate - baseline

            if abs(diff) >= self.MIN_WIN_RATE_DIFF:
                avg_pnl = statistics.mean(t.outcome.pnl_pct for t in bucket_trades)
                confidence = self._calculate_confidence(len(bucket_trades), win_rate, baseline)

                patterns.append(PatternCandidate(
                    description=f"Heat score {bucket}: win rate {win_rate:.1%}",
                    conditions={"heat_score": self._get_bucket_conditions(bucket, self.HEAT_BUCKETS)},
                    trades=bucket_trades,
                    win_rate=win_rate,
                    avg_pnl=avg_pnl,
                    sample_size=len(bucket_trades),
                    confidence=confidence
                ))

        return patterns

    def _extract_combined_patterns(
        self,
        trades: List[StoredTrade],
        baseline: float,
        min_sample: int
    ) -> List[PatternCandidate]:
        """Extract combined patterns (multiple conditions)"""
        patterns = []

        # Combine RSI + Volume
        grouped = defaultdict(list)
        for trade in trades:
            rsi_bucket = self._bucketize(trade.context.rsi, self.RSI_BUCKETS)
            vol_bucket = self._bucketize(trade.context.volume_ratio, self.VOLUME_BUCKETS)
            key = f"{rsi_bucket}_{vol_bucket}"
            grouped[key].append(trade)

        for key, combo_trades in grouped.items():
            if len(combo_trades) < min_sample:
                continue

            rsi_bucket, vol_bucket = key.split("_")
            win_rate = sum(1 for t in combo_trades if t.outcome.is_win) / len(combo_trades)
            diff = win_rate - baseline

            if abs(diff) >= self.MIN_WIN_RATE_DIFF * 1.5:  # Higher threshold for combined
                avg_pnl = statistics.mean(t.outcome.pnl_pct for t in combo_trades)
                confidence = self._calculate_confidence(len(combo_trades), win_rate, baseline)

                patterns.append(PatternCandidate(
                    description=f"RSI {rsi_bucket} + Volume {vol_bucket}: win rate {win_rate:.1%}",
                    conditions={
                        "rsi": self._get_bucket_conditions(rsi_bucket, self.RSI_BUCKETS),
                        "volume_ratio": self._get_bucket_conditions(vol_bucket, self.VOLUME_BUCKETS)
                    },
                    trades=combo_trades,
                    win_rate=win_rate,
                    avg_pnl=avg_pnl,
                    sample_size=len(combo_trades),
                    confidence=confidence
                ))

        # Combine Regime + VIX
        grouped = defaultdict(list)
        for trade in trades:
            vix_bucket = self._bucketize(trade.context.vix_level, self.VIX_BUCKETS)
            key = f"{trade.context.market_regime}_{vix_bucket}"
            grouped[key].append(trade)

        for key, combo_trades in grouped.items():
            if len(combo_trades) < min_sample:
                continue

            parts = key.split("_")
            regime = parts[0]
            vix_bucket = "_".join(parts[1:])

            win_rate = sum(1 for t in combo_trades if t.outcome.is_win) / len(combo_trades)
            diff = win_rate - baseline

            if abs(diff) >= self.MIN_WIN_RATE_DIFF * 1.5:
                avg_pnl = statistics.mean(t.outcome.pnl_pct for t in combo_trades)
                confidence = self._calculate_confidence(len(combo_trades), win_rate, baseline)

                patterns.append(PatternCandidate(
                    description=f"Regime {regime} + VIX {vix_bucket}: win rate {win_rate:.1%}",
                    conditions={
                        "market_regime": {"op": "==", "value": regime},
                        "vix_level": self._get_bucket_conditions(vix_bucket, self.VIX_BUCKETS)
                    },
                    trades=combo_trades,
                    win_rate=win_rate,
                    avg_pnl=avg_pnl,
                    sample_size=len(combo_trades),
                    confidence=confidence
                ))

        return patterns

    def _calculate_confidence(
        self,
        sample_size: int,
        observed_rate: float,
        baseline_rate: float
    ) -> float:
        """
        Calculate statistical confidence in a pattern.
        Uses a simplified approach based on sample size and difference from baseline.
        """
        # More samples = more confidence
        sample_factor = min(1.0, sample_size / 20)

        # Larger difference from baseline = more confidence
        diff = abs(observed_rate - baseline_rate)
        diff_factor = min(1.0, diff / 0.20)  # Max out at 20% difference

        return (sample_factor * 0.6 + diff_factor * 0.4)

    def patterns_to_rules(
        self,
        patterns: List[PatternCandidate]
    ) -> List[LearnedRule]:
        """
        Convert pattern candidates to learned rules.

        Args:
            patterns: Significant patterns

        Returns:
            List of learned rules
        """
        rules = []

        for pattern in patterns:
            rule_id = f"rule_{hash(pattern.description) % 100000}"

            rule = LearnedRule(
                id=rule_id,
                pattern=pattern.description,
                conditions=pattern.conditions,
                historical_win_rate=pattern.win_rate,
                recent_win_rate=pattern.win_rate,  # Same initially
                occurrences=pattern.sample_size,
                avg_pnl=pattern.avg_pnl,
                status=RuleStatus.ACTIVE,
                last_validated=datetime.now().isoformat()
            )
            rules.append(rule)

        return rules

    async def run_extraction(
        self,
        trades: List[StoredTrade]
    ) -> ExtractionResult:
        """
        Run full pattern extraction and save to memory.

        Args:
            trades: Historical trades to analyze

        Returns:
            Extraction result with statistics
        """
        logger.info(f"Running pattern extraction on {len(trades)} trades")

        # Extract patterns
        patterns = self.extract_patterns(trades)

        # Convert to rules
        new_rules = self.patterns_to_rules(patterns)

        # Save to memory
        created = 0
        updated = 0
        existing_rules = {r.id: r for r in self.memory.get_active_rules()}

        for rule in new_rules:
            if rule.id in existing_rules:
                # Update existing rule
                old_rule = existing_rules[rule.id]
                old_rule.occurrences = rule.occurrences
                old_rule.recent_win_rate = rule.historical_win_rate
                old_rule.last_validated = datetime.now().isoformat()
                self.memory.add_rule(old_rule)
                updated += 1
            else:
                # Add new rule
                self.memory.add_rule(rule)
                created += 1

        # Generate insights
        insights = self._generate_insights(patterns, trades)

        result = ExtractionResult(
            timestamp=datetime.now().isoformat(),
            trades_analyzed=len(trades),
            patterns_found=len(patterns),
            rules_created=created,
            rules_updated=updated,
            insights=insights,
            new_rules=new_rules
        )

        logger.info(f"Extraction complete: {created} new rules, {updated} updated")
        return result

    def _generate_insights(
        self,
        patterns: List[PatternCandidate],
        trades: List[StoredTrade]
    ) -> List[str]:
        """Generate human-readable insights from patterns"""
        insights = []

        if not patterns:
            insights.append("No significant patterns found in recent trades.")
            return insights

        # Best pattern
        best = max(patterns, key=lambda p: p.win_rate)
        insights.append(f"Best pattern: {best.description} ({best.sample_size} trades)")

        # Worst pattern
        worst = min(patterns, key=lambda p: p.win_rate)
        if worst.win_rate < 0.4:
            insights.append(f"Avoid: {worst.description} (only {worst.win_rate:.0%} win rate)")

        # Volume insight
        volume_patterns = [p for p in patterns if 'volume_ratio' in p.conditions]
        if volume_patterns:
            high_vol = [p for p in volume_patterns if 'high' in p.description.lower()]
            if high_vol and high_vol[0].win_rate > 0.6:
                insights.append("High volume confirms trades - consider requiring volume > 1.5x")

        # Regime insight
        regime_patterns = [p for p in patterns if 'market_regime' in p.conditions]
        for p in regime_patterns:
            if 'bear' in p.description.lower() and p.win_rate < 0.4:
                insights.append("Bear market performance is poor - consider reducing position sizes")
            elif 'bull' in p.description.lower() and p.win_rate > 0.6:
                insights.append("Bull market shows strong performance - maintain strategy")

        return insights


# =========================================================================
# SINGLETON
# =========================================================================

_extractor_instance: Optional[PatternExtractor] = None


def get_pattern_extractor() -> PatternExtractor:
    """Get or create the global PatternExtractor instance"""
    global _extractor_instance

    if _extractor_instance is None:
        _extractor_instance = PatternExtractor()

    return _extractor_instance
