"""Tests for IntelligenceOrchestrator - V8 LLM brain + ETF momentum."""
import sys
import importlib.util
import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Direct import to bypass src/intelligence/__init__.py (needs zoneinfo/Python 3.9+)
_spec = importlib.util.spec_from_file_location(
    "intelligence_orchestrator", "src/intelligence/intelligence_orchestrator.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["intelligence_orchestrator"] = _mod
_spec.loader.exec_module(_mod)

IntelligenceOrchestrator = _mod.IntelligenceOrchestrator
IntelligenceBrief = _mod.IntelligenceBrief
MarketEvent = _mod.MarketEvent
MegatrendSignal = _mod.MegatrendSignal
calculate_etf_momentum_bonus = _mod.calculate_etf_momentum_bonus
THEME_ETF_PROXIES = _mod.THEME_ETF_PROXIES
_SYMBOL_TO_ETF = _mod._SYMBOL_TO_ETF


# ── ETF Momentum Tests (backtestable, no mocks needed) ──────────────────

def _make_df(values, start='2023-01-01'):
    dates = pd.bdate_range(start=start, periods=len(values))
    return pd.DataFrame({'Close': values}, index=dates)


class MockDataMgr:
    """Minimal data manager mock for ETF momentum tests."""
    def __init__(self):
        self._cache = {}


class TestETFMomentumBonus:
    """Test backtestable ETF momentum bonus."""

    def test_symbol_not_in_mapping_returns_zero(self):
        dm = MockDataMgr()
        date = pd.Timestamp('2023-06-01')
        assert calculate_etf_momentum_bonus('UNKNOWN', date, dm) == 0.0

    def test_no_data_returns_zero(self):
        dm = MockDataMgr()
        date = pd.Timestamp('2023-06-01')
        assert calculate_etf_momentum_bonus('NVDA', date, dm) == 0.0

    def test_strong_outperformance_gives_max_bonus(self):
        dm = MockDataMgr()
        # SMH outperforms SPY by >15%
        n = 100
        spy_vals = [100 + i * 0.1 for i in range(n)]
        smh_vals = [100 + i * 0.5 for i in range(n)]
        dm._cache['SPY_2023'] = _make_df(spy_vals)
        dm._cache['SMH_2023'] = _make_df(smh_vals)

        date = dm._cache['SPY_2023'].index[80]
        bonus = calculate_etf_momentum_bonus('NVDA', date, dm)
        assert bonus == 8.0

    def test_moderate_outperformance_gives_medium_bonus(self):
        dm = MockDataMgr()
        n = 100
        spy_vals = [100 + i * 0.2 for i in range(n)]
        smh_vals = [100 + i * 0.35 for i in range(n)]
        dm._cache['SPY_2023'] = _make_df(spy_vals)
        dm._cache['SMH_2023'] = _make_df(smh_vals)

        date = dm._cache['SPY_2023'].index[80]
        bonus = calculate_etf_momentum_bonus('NVDA', date, dm)
        assert bonus > 0

    def test_underperformance_gives_zero(self):
        dm = MockDataMgr()
        n = 100
        spy_vals = [100 + i * 0.5 for i in range(n)]
        smh_vals = [100 + i * 0.1 for i in range(n)]
        dm._cache['SPY_2023'] = _make_df(spy_vals)
        dm._cache['SMH_2023'] = _make_df(smh_vals)

        date = dm._cache['SPY_2023'].index[80]
        bonus = calculate_etf_momentum_bonus('NVDA', date, dm)
        assert bonus == 0.0  # Never negative

    def test_insufficient_history_returns_zero(self):
        dm = MockDataMgr()
        dm._cache['SPY_2023'] = _make_df([100.0] * 30)
        dm._cache['SMH_2023'] = _make_df([100.0] * 30)

        date = dm._cache['SPY_2023'].index[20]
        assert calculate_etf_momentum_bonus('NVDA', date, dm) == 0.0

    def test_symbol_to_etf_mapping(self):
        """Verify key symbols map to correct ETFs."""
        assert _SYMBOL_TO_ETF.get('NVDA') == 'SMH'
        assert _SYMBOL_TO_ETF.get('AAPL') == 'XLK'
        assert _SYMBOL_TO_ETF.get('JPM') == 'XLF'
        assert _SYMBOL_TO_ETF.get('XOM') == 'XLE'
        assert _SYMBOL_TO_ETF.get('PG') == 'XLP'


# ── Orchestrator Tests (mocked LLM) ─────────────────────────────────────

class TestBriefParsing:
    """Test parsing of LLM JSON response into IntelligenceBrief."""

    def _make_orchestrator(self, symbols=None):
        gemini = MagicMock()
        gemini.is_available.return_value = True
        return IntelligenceOrchestrator(
            gemini_client=gemini,
            portfolio_symbols=symbols or ['AAPL', 'MSFT', 'NVDA']
        )

    def test_parse_valid_response(self):
        orch = self._make_orchestrator()
        data = {
            "events": [{
                "event_type": "geopolitical",
                "summary": "Trade war escalation",
                "affected_symbols": {"AAPL": -5.0, "NVDA": -3.0},
                "confidence": 0.8,
                "reasoning": "Tariffs on China",
                "ttl_hours": 48
            }],
            "megatrends": [{
                "theme": "AI infrastructure",
                "strength": "established",
                "symbols": {"NVDA": 8.0, "MSFT": 3.0},
                "reasoning": "Capex surge"
            }],
            "macro_regime_bias": "bullish",
            "portfolio_alerts": ["Watch NVDA earnings"],
            "reasoning_summary": "Mixed signals"
        }
        brief = orch._parse_brief(data)
        assert len(brief.events) == 1
        assert brief.events[0].affected_symbols['AAPL'] == -5.0
        assert len(brief.megatrends) == 1
        assert brief.megatrends[0].symbols['NVDA'] == 8.0
        assert brief.macro_regime_bias == 'bullish'
        assert len(brief.portfolio_alerts) == 1

    def test_parse_empty_response(self):
        orch = self._make_orchestrator()
        brief = orch._parse_brief({})
        assert brief.events == []
        assert brief.megatrends == []
        assert brief.macro_regime_bias == 'neutral'

    def test_adjustments_capped(self):
        orch = self._make_orchestrator()
        data = {
            "events": [{
                "event_type": "crash",
                "summary": "Market crash",
                "affected_symbols": {"AAPL": -50.0},  # Over limit
                "confidence": 1.0,
                "reasoning": "Big crash",
                "ttl_hours": 24
            }],
            "megatrends": [{
                "theme": "Hype",
                "strength": "emerging",
                "symbols": {"NVDA": 99.0},  # Over limit
                "reasoning": "Extreme hype"
            }]
        }
        brief = orch._parse_brief(data)
        assert brief.events[0].affected_symbols['AAPL'] == -15.0
        assert brief.megatrends[0].symbols['NVDA'] == 10.0

    def test_filters_to_portfolio_symbols_only(self):
        orch = self._make_orchestrator(symbols=['AAPL'])
        data = {
            "events": [{
                "event_type": "sector",
                "summary": "Tech rally",
                "affected_symbols": {"AAPL": 5.0, "TSLA": 10.0, "UNKNOWN": 3.0},
                "confidence": 0.7,
                "reasoning": "...",
                "ttl_hours": 24
            }]
        }
        brief = orch._parse_brief(data)
        assert 'AAPL' in brief.events[0].affected_symbols
        assert 'TSLA' not in brief.events[0].affected_symbols
        assert 'UNKNOWN' not in brief.events[0].affected_symbols


class TestSymbolAdjustment:
    """Test get_symbol_adjustment method."""

    def test_combines_events_and_trends(self):
        orch = IntelligenceOrchestrator(
            gemini_client=MagicMock(),
            portfolio_symbols=['NVDA']
        )
        brief = IntelligenceBrief(
            timestamp=datetime.now(),
            events=[
                MarketEvent(
                    event_type='tariff', summary='', confidence=0.8,
                    affected_symbols={'NVDA': -5.0}, reasoning='', source='news', ttl_hours=24
                )
            ],
            megatrends=[
                MegatrendSignal(
                    theme='AI', strength='established',
                    symbols={'NVDA': 8.0}, etf_momentum=0.0, reasoning=''
                )
            ]
        )
        adj = orch.get_symbol_adjustment('NVDA', brief)
        assert adj == 3.0  # -5 + 8 = 3

    def test_capped_at_15(self):
        orch = IntelligenceOrchestrator(
            gemini_client=MagicMock(),
            portfolio_symbols=['NVDA']
        )
        brief = IntelligenceBrief(
            timestamp=datetime.now(),
            events=[
                MarketEvent(
                    event_type='a', summary='', confidence=0.8,
                    affected_symbols={'NVDA': 10.0}, reasoning='', source='news', ttl_hours=24
                ),
                MarketEvent(
                    event_type='b', summary='', confidence=0.8,
                    affected_symbols={'NVDA': 10.0}, reasoning='', source='news', ttl_hours=24
                )
            ]
        )
        adj = orch.get_symbol_adjustment('NVDA', brief)
        assert adj == 15.0  # Capped

    def test_negative_cap(self):
        orch = IntelligenceOrchestrator(
            gemini_client=MagicMock(),
            portfolio_symbols=['AAPL']
        )
        brief = IntelligenceBrief(
            timestamp=datetime.now(),
            events=[
                MarketEvent(
                    event_type='crash', summary='', confidence=1.0,
                    affected_symbols={'AAPL': -10.0}, reasoning='', source='', ttl_hours=24
                ),
                MarketEvent(
                    event_type='crash2', summary='', confidence=1.0,
                    affected_symbols={'AAPL': -10.0}, reasoning='', source='', ttl_hours=24
                ),
            ]
        )
        adj = orch.get_symbol_adjustment('AAPL', brief)
        assert adj == -15.0  # Capped negative

    def test_missing_symbol_returns_zero(self):
        orch = IntelligenceOrchestrator(
            gemini_client=MagicMock(),
            portfolio_symbols=['AAPL']
        )
        brief = IntelligenceBrief(timestamp=datetime.now())
        adj = orch.get_symbol_adjustment('AAPL', brief)
        assert adj == 0.0


class TestCaching:
    """Test 15-minute cache behavior."""

    @pytest.mark.asyncio
    async def test_cache_returns_same_brief(self):
        gemini = AsyncMock()
        gemini.is_available.return_value = True
        gemini.chat_json = AsyncMock(return_value={
            "events": [], "megatrends": [],
            "macro_regime_bias": "neutral",
            "portfolio_alerts": [], "reasoning_summary": "test"
        })

        orch = IntelligenceOrchestrator(
            gemini_client=gemini,
            portfolio_symbols=['AAPL']
        )

        # First call generates
        brief1 = await orch.get_brief()
        # Second call should be cached (no new LLM call)
        brief2 = await orch.get_brief()
        assert brief1 is brief2
        assert gemini.chat_json.call_count == 1

    @pytest.mark.asyncio
    async def test_force_bypasses_cache(self):
        gemini = AsyncMock()
        gemini.is_available.return_value = True
        gemini.chat_json = AsyncMock(return_value={
            "events": [], "megatrends": [],
            "macro_regime_bias": "neutral",
            "portfolio_alerts": [], "reasoning_summary": "test"
        })

        orch = IntelligenceOrchestrator(
            gemini_client=gemini,
            portfolio_symbols=['AAPL']
        )

        await orch.get_brief()
        await orch.get_brief(force=True)
        assert gemini.chat_json.call_count == 2


class TestPromptBuilding:
    """Test prompt construction."""

    def test_empty_data_produces_prompt(self):
        orch = IntelligenceOrchestrator(
            gemini_client=MagicMock(),
            portfolio_symbols=['AAPL', 'MSFT']
        )
        prompt = orch._build_reasoning_prompt({})
        assert 'AAPL' in prompt
        assert 'MSFT' in prompt

    def test_market_context_included(self):
        orch = IntelligenceOrchestrator(
            gemini_client=MagicMock(),
            portfolio_symbols=['AAPL']
        )
        mock_ctx = MagicMock()
        mock_ctx.get_summary.return_value = "Market: BULL, VIX: 14"
        prompt = orch._build_reasoning_prompt({'market_context': mock_ctx})
        assert 'MARKET CONTEXT' in prompt
        assert 'BULL' in prompt
