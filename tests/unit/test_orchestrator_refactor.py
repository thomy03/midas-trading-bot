"""Tests for refactored orchestrator - V8.2 Sprint 4 delegate pattern."""
import sys
import os
import json
import pytest
import asyncio
import importlib.util
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime, time
from dataclasses import dataclass


# Direct imports
_spec_orch = importlib.util.spec_from_file_location(
    "orchestrator", "src/agents/orchestrator.py"
)
_mod_orch = importlib.util.module_from_spec(_spec_orch)
sys.modules["orchestrator_test"] = _mod_orch

# We need to mock heavy dependencies before loading
sys.modules.setdefault('src.agents.state', MagicMock())
sys.modules.setdefault('src.agents.guardrails', MagicMock())
sys.modules.setdefault('src.intelligence.analysis_store', MagicMock())
sys.modules.setdefault('src.data.market_data', MagicMock())
sys.modules.setdefault('src.execution.position_manager', MagicMock())
sys.modules.setdefault('src.execution.portfolio_rotation', MagicMock())

# Load the execution coordinator for testing helpers
_spec_exec = importlib.util.spec_from_file_location(
    "execution_coordinator", "src/agents/execution_coordinator.py"
)
_mod_exec = importlib.util.module_from_spec(_spec_exec)
sys.modules["execution_coordinator_test"] = _mod_exec

_spec_learn = importlib.util.spec_from_file_location(
    "learning_feedback", "src/agents/learning_feedback.py"
)
_mod_learn = importlib.util.module_from_spec(_spec_learn)
sys.modules["learning_feedback_test"] = _mod_learn


class TestExecutionCoordinatorHelpers:
    """Test execution coordinator helper methods."""

    def test_reasoning_to_alert_with_stop_loss(self):
        """Test that SL is calculated from ATR when available."""
        agent = MagicMock()
        agent.config = MagicMock()
        agent.state_manager = MagicMock()
        agent.guardrails = MagicMock()

        try:
            _spec_exec.loader.exec_module(_mod_exec)
            ExecutionCoordinator = _mod_exec.ExecutionCoordinator
        except Exception:
            pytest.skip("Cannot load execution_coordinator (missing deps)")

        ec = ExecutionCoordinator(agent)

        # Mock reasoning result with ATR in factors
        reasoning_result = MagicMock()
        reasoning_result.decision.value = "buy"
        reasoning_result.confidence = 0.75
        reasoning_result.reasoning_summary = "Test"
        reasoning_result.total_score = 72.0
        reasoning_result.timestamp = datetime.now().isoformat()
        reasoning_result.key_factors = ["Strong EMA"]
        reasoning_result.risk_factors = ["High VIX"]

        # Technical score with ATR and close price
        tech_score = MagicMock()
        tech_score.score = 65.0
        tech_score.factors = [{"close": 150.0, "atr": 3.5, "rsi": 55}]
        reasoning_result.technical_score = tech_score

        fund_score = MagicMock()
        fund_score.score = 60.0
        reasoning_result.fundamental_score = fund_score

        sent_score = MagicMock()
        sent_score.score = 50.0
        sent_score.factors = []
        reasoning_result.sentiment_score = sent_score

        news_score = MagicMock()
        news_score.score = 55.0
        reasoning_result.news_score = news_score

        alert = ec._reasoning_to_alert("AAPL", reasoning_result)

        assert alert["symbol"] == "AAPL"
        assert alert["signal"] == "BUY"
        assert alert["price"] == 150.0
        # SL should be price - 2*ATR = 150 - 7 = 143
        assert alert["stop_loss"] == 143.0

    def test_reasoning_to_alert_default_stop_loss(self):
        """Test that SL defaults to 5% when no ATR."""
        agent = MagicMock()
        agent.config = MagicMock()
        agent.state_manager = MagicMock()
        agent.guardrails = MagicMock()

        try:
            _spec_exec.loader.exec_module(_mod_exec)
            ExecutionCoordinator = _mod_exec.ExecutionCoordinator
        except Exception:
            pytest.skip("Cannot load execution_coordinator (missing deps)")

        ec = ExecutionCoordinator(agent)

        reasoning_result = MagicMock()
        reasoning_result.decision.value = "strong_buy"
        reasoning_result.confidence = 0.85
        reasoning_result.reasoning_summary = "Test"
        reasoning_result.total_score = 80.0
        reasoning_result.timestamp = datetime.now().isoformat()
        reasoning_result.key_factors = []
        reasoning_result.risk_factors = []

        tech_score = MagicMock()
        tech_score.score = 70.0
        tech_score.factors = [{"close": 200.0, "rsi": 60}]  # No ATR
        reasoning_result.technical_score = tech_score

        fund_score = MagicMock()
        fund_score.score = 65.0
        reasoning_result.fundamental_score = fund_score

        sent_score = MagicMock()
        sent_score.score = 55.0
        sent_score.factors = []
        reasoning_result.sentiment_score = sent_score

        news_score = MagicMock()
        news_score.score = 60.0
        reasoning_result.news_score = news_score

        alert = ec._reasoning_to_alert("MSFT", reasoning_result)

        assert alert["price"] == 200.0
        assert alert["stop_loss"] == 190.0  # 200 * 0.95

    def test_build_scoring_details(self):
        """Test scoring details dict construction."""
        agent = MagicMock()
        agent.config = MagicMock()
        agent.state_manager = MagicMock()
        agent.guardrails = MagicMock()

        try:
            _spec_exec.loader.exec_module(_mod_exec)
            ExecutionCoordinator = _mod_exec.ExecutionCoordinator
        except Exception:
            pytest.skip("Cannot load execution_coordinator (missing deps)")

        ec = ExecutionCoordinator(agent)

        reasoning_result = MagicMock()
        reasoning_result.total_score = 72.5
        reasoning_result.decision.value = "buy"
        reasoning_result.key_factors = ["Factor1", "Factor2"]
        reasoning_result.risk_factors = ["Risk1"]
        reasoning_result.reasoning_summary = "Test summary"

        for pillar in ["technical", "fundamental", "sentiment", "news"]:
            score = MagicMock()
            score.score = 60.0
            score.reasoning = f"{pillar} reasoning"
            score.factors = [{"test": True}]
            score.signal.value = "bullish"
            score.data_quality = 0.9
            setattr(reasoning_result, f"{pillar}_score", score)

        details = ec._build_scoring_details("TEST", reasoning_result)

        assert details["total_score"] == 72.5
        assert details["decision"] == "buy"
        assert "pillar_details" in details
        assert "technical" in details["pillar_details"]
        assert details["pillar_details"]["technical"]["score"] == 60.0


class TestLearningFeedbackHelpers:
    """Test learning feedback helper methods."""

    def test_extract_failure_patterns_stop_loss(self):
        """Test failure pattern extraction for stop-loss exits."""
        try:
            _spec_learn.loader.exec_module(_mod_learn)
            LearningFeedbackCoordinator = _mod_learn.LearningFeedbackCoordinator
        except Exception:
            pytest.skip("Cannot load learning_feedback (missing deps)")

        # Create mock trades that mostly hit stop loss
        trades = []
        for i in range(5):
            t = MagicMock()
            t.exit_type = "stop_loss"
            t.hold_days = 2
            t.metadata = {"vix_level": 30, "rsi": 72}
            trades.append(t)

        patterns = LearningFeedbackCoordinator._extract_failure_patterns(trades)

        assert any("stop loss" in p.lower() for p in patterns)
        assert any("VIX" in p for p in patterns)
        assert any("RSI" in p for p in patterns)

    def test_extract_failure_patterns_long_hold(self):
        """Test failure pattern extraction for long holding periods."""
        try:
            _spec_learn.loader.exec_module(_mod_learn)
            LearningFeedbackCoordinator = _mod_learn.LearningFeedbackCoordinator
        except Exception:
            pytest.skip("Cannot load learning_feedback (missing deps)")

        trades = []
        for i in range(3):
            t = MagicMock()
            t.exit_type = "max_hold_period"
            t.hold_days = 45
            t.metadata = {}
            trades.append(t)

        patterns = LearningFeedbackCoordinator._extract_failure_patterns(trades)

        assert any("hold" in p.lower() for p in patterns)

    def test_extract_failure_patterns_empty(self):
        """Test failure pattern extraction with no trades."""
        try:
            _spec_learn.loader.exec_module(_mod_learn)
            LearningFeedbackCoordinator = _mod_learn.LearningFeedbackCoordinator
        except Exception:
            pytest.skip("Cannot load learning_feedback (missing deps)")

        patterns = LearningFeedbackCoordinator._extract_failure_patterns([])
        assert patterns == []


class TestGuardrailsMonthlyPnL:
    """Test monthly P&L tracking in guardrails."""

    def test_monthly_pnl_file_created(self, tmp_path):
        """Test that monthly P&L file is created on first trade."""
        _spec_guard = importlib.util.spec_from_file_location(
            "guardrails", "src/agents/guardrails.py"
        )
        _mod_guard = importlib.util.module_from_spec(_spec_guard)
        try:
            _spec_guard.loader.exec_module(_mod_guard)
        except Exception:
            pytest.skip("Cannot load guardrails (missing deps)")

        TradingGuardrails = _mod_guard.TradingGuardrails

        guardrails = TradingGuardrails(
            capital=15000.0,
            data_dir=str(tmp_path)
        )

        # Record a trade
        guardrails.record_monthly_trade("AAPL", 150.0, 2.5)

        # Check file exists
        pnl_file = os.path.join(str(tmp_path), "monthly_pnl.json")
        assert os.path.exists(pnl_file)

        with open(pnl_file) as f:
            data = json.load(f)

        assert data["month"] == datetime.now().strftime("%Y-%m")
        assert len(data["trades"]) == 1
        assert data["trades"][0]["symbol"] == "AAPL"
        assert data["trades"][0]["pnl"] == 150.0

    def test_monthly_pnl_accumulates(self, tmp_path):
        """Test that multiple trades accumulate correctly."""
        _spec_guard = importlib.util.spec_from_file_location(
            "guardrails", "src/agents/guardrails.py"
        )
        _mod_guard = importlib.util.module_from_spec(_spec_guard)
        try:
            _spec_guard.loader.exec_module(_mod_guard)
        except Exception:
            pytest.skip("Cannot load guardrails (missing deps)")

        TradingGuardrails = _mod_guard.TradingGuardrails

        guardrails = TradingGuardrails(
            capital=15000.0,
            data_dir=str(tmp_path)
        )

        guardrails.record_monthly_trade("AAPL", 150.0, 2.5)
        guardrails.record_monthly_trade("MSFT", -50.0, -1.0)
        guardrails.record_monthly_trade("NVDA", 200.0, 3.5)

        total = guardrails._get_monthly_pnl()
        assert total == 300.0  # 150 - 50 + 200
