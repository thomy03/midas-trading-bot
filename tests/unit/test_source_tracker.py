"""Tests for SourceReliabilityTracker - V8.2 source tracking and weighting."""
import sys
import os
import json
import pytest
import tempfile
import importlib.util
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Direct import to bypass potential circular imports
_spec = importlib.util.spec_from_file_location(
    "source_tracker", "src/intelligence/source_tracker.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["source_tracker"] = _mod
_spec.loader.exec_module(_mod)

SourceReliabilityTracker = _mod.SourceReliabilityTracker
SourcePrediction = _mod.SourcePrediction
SourceStats = _mod.SourceStats


class TestSourcePrediction:
    """Test SourcePrediction creation and serialization."""

    def test_create_prediction(self):
        pred = SourcePrediction(
            source="grok",
            symbol="AAPL",
            adjustment=5.0,
            direction="bullish",
            confidence=0.8,
            timestamp=datetime.now().isoformat()
        )
        assert pred.source == "grok"
        assert pred.symbol == "AAPL"
        assert pred.adjustment == 5.0
        assert pred.direction == "bullish"


class TestSourceReliabilityTracker:
    """Test SourceReliabilityTracker core functionality."""

    def _make_tracker(self, tmp_dir):
        return SourceReliabilityTracker(data_dir=tmp_dir)

    def test_log_prediction(self, tmp_path):
        tracker = self._make_tracker(str(tmp_path))
        tracker.log_prediction(
            source="reuters",
            symbol="MSFT",
            adjustment=3.0,
            direction="bullish",
            confidence=0.7
        )
        assert len(tracker._predictions) == 1
        assert tracker._predictions[0].source == "reuters"
        assert tracker._predictions[0].symbol == "MSFT"

    def test_log_brief_predictions(self, tmp_path):
        tracker = self._make_tracker(str(tmp_path))
        # log_brief_predictions expects an IntelligenceBrief object with .events attr
        event = MagicMock()
        event.source = "grok"
        event.affected_symbols = {"AAPL": -5.0, "NVDA": 3.0}
        event.confidence = 0.8

        brief = MagicMock()
        brief.events = [event]

        tracker.log_brief_predictions(brief)
        assert len(tracker._predictions) >= 2

    def test_persistence(self, tmp_path):
        tracker1 = self._make_tracker(str(tmp_path))
        tracker1.log_prediction("grok", "AAPL", 5.0, "bullish", 0.8)
        tracker1._save()

        # Load in new instance
        tracker2 = self._make_tracker(str(tmp_path))
        assert len(tracker2._predictions) == 1
        assert tracker2._predictions[0].source == "grok"

    def test_get_source_weights_default(self, tmp_path):
        tracker = self._make_tracker(str(tmp_path))
        weights = tracker.get_source_weights()
        assert isinstance(weights, dict)
        # Default weight should be 1.0 for unknown sources
        assert weights.get("unknown_source", 1.0) == 1.0

    def test_evaluate_outcome_correct_bullish(self, tmp_path):
        # _evaluate_outcome is static: (predicted_direction, actual_move_pct) -> str
        result = SourceReliabilityTracker._evaluate_outcome("bullish", 3.0)
        assert result == "correct"

    def test_evaluate_outcome_incorrect_bullish(self, tmp_path):
        result = SourceReliabilityTracker._evaluate_outcome("bullish", -3.0)
        assert result == "wrong"

    def test_evaluate_outcome_correct_bearish(self, tmp_path):
        result = SourceReliabilityTracker._evaluate_outcome("bearish", -3.0)
        assert result == "correct"

    def test_weight_calculation(self, tmp_path):
        tracker = self._make_tracker(str(tmp_path))

        # Manually set up stats for reuters (good) and reddit (bad)
        from dataclasses import fields
        tracker._stats["reuters"] = SourceStats(
            source="reuters", total_predictions=10,
            checked_predictions=10, correct_j1=9, wrong_j1=1,
            correct_j5=8, wrong_j5=2
        )
        tracker._stats["reddit"] = SourceStats(
            source="reddit", total_predictions=10,
            checked_predictions=10, correct_j1=2, wrong_j1=8,
            correct_j5=1, wrong_j5=9
        )

        tracker._recalculate_weights()

        stats = tracker.get_source_stats()
        assert stats["reuters"]["hit_rate_j1"] > stats["reddit"]["hit_rate_j1"]
        assert stats["reuters"]["reliability_weight"] > stats["reddit"]["reliability_weight"]

    def test_get_best_worst_sources(self, tmp_path):
        tracker = self._make_tracker(str(tmp_path))

        tracker._stats["reuters"] = SourceStats(
            source="reuters", total_predictions=10,
            checked_predictions=10, correct_j1=9, wrong_j1=1,
            correct_j5=8, wrong_j5=2
        )
        tracker._stats["reddit"] = SourceStats(
            source="reddit", total_predictions=10,
            checked_predictions=10, correct_j1=2, wrong_j1=8,
            correct_j5=1, wrong_j5=9
        )

        tracker._recalculate_weights()

        best = tracker.get_best_sources(limit=1)
        worst = tracker.get_worst_sources(limit=1)
        assert "reuters" in best
        assert "reddit" in worst
