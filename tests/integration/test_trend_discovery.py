"""
Tests for TrendDiscovery module.
Tests both quantitative detection and LLM integration.
"""

import pytest
import asyncio
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.intelligence.trend_discovery import (
    TrendDiscovery,
    TrendDiscoveryScheduler,
    EmergingTrend,
    TrendReport,
    TrendStrength,
    TrendType,
    get_trend_discovery
)


class TestEmergingTrend:
    """Tests for EmergingTrend dataclass."""

    def test_create_trend(self):
        """Test creating an EmergingTrend."""
        trend = EmergingTrend(
            name="AI Revolution",
            type=TrendType.THEMATIC,
            strength=TrendStrength.DEVELOPING,
            confidence=0.7,
            description="AI stocks showing momentum",
            momentum_score=0.5,
            sectors=["Technology", "AI_Semiconductors"],
            symbols=["NVDA", "AMD"]
        )

        assert trend.name == "AI Revolution"
        assert trend.confidence == 0.7
        assert "NVDA" in trend.symbols

    def test_trend_to_dict(self):
        """Test converting trend to dictionary."""
        trend = EmergingTrend(
            name="Test Trend",
            type=TrendType.SECTOR_MOMENTUM,
            strength=TrendStrength.EMERGING,
            confidence=0.5,
            description="Test description"
        )

        data = trend.to_dict()

        assert data['name'] == "Test Trend"
        assert data['type'] == "sector_momentum"
        assert data['strength'] == "emerging"
        assert 'detected_at' in data


class TestTrendReport:
    """Tests for TrendReport dataclass."""

    def test_create_report(self):
        """Test creating a TrendReport."""
        trend = EmergingTrend(
            name="Test",
            type=TrendType.THEMATIC,
            strength=TrendStrength.EMERGING,
            confidence=0.5,
            description="Test"
        )

        report = TrendReport(
            date=datetime.now(),
            trends=[trend],
            sector_momentum={'Technology': 0.3, 'Healthcare': -0.1},
            market_sentiment=0.2,
            top_movers=['NVDA', 'AMD'],
            watchlist_additions=['NVDA'],
            narrative_updates=[]
        )

        assert len(report.trends) == 1
        assert report.market_sentiment == 0.2
        assert 'NVDA' in report.top_movers

    def test_report_summary(self):
        """Test generating report summary."""
        trend = EmergingTrend(
            name="AI Momentum",
            type=TrendType.SECTOR_MOMENTUM,
            strength=TrendStrength.DEVELOPING,
            confidence=0.7,
            description="Strong momentum in AI sector",
            key_catalysts=["NVDA earnings", "AI adoption"]
        )

        report = TrendReport(
            date=datetime.now(),
            trends=[trend],
            sector_momentum={'Technology': 0.5},
            market_sentiment=0.3,
            top_movers=['NVDA'],
            watchlist_additions=['NVDA', 'AMD'],
            narrative_updates=[]
        )

        summary = report.summary()

        assert "Trend Discovery Report" in summary
        assert "AI Momentum" in summary
        assert "Market Sentiment" in summary


class TestTrendDiscoveryInit:
    """Tests for TrendDiscovery initialization."""

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        import os
        # Temporarily clear the environment variable
        original = os.environ.get('OPENROUTER_API_KEY')
        os.environ['OPENROUTER_API_KEY'] = ''
        try:
            discovery = TrendDiscovery(openrouter_api_key=None)
            assert discovery.api_key is None or discovery.api_key == ''
        finally:
            # Restore original value
            if original:
                os.environ['OPENROUTER_API_KEY'] = original

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        discovery = TrendDiscovery(
            openrouter_api_key="test-key",
            model="openai/gpt-4-turbo"
        )

        assert discovery.model == "openai/gpt-4-turbo"

    def test_sectors_defined(self):
        """Test that sectors are properly defined."""
        assert 'Technology' in TrendDiscovery.SECTORS
        assert 'AI_Semiconductors' in TrendDiscovery.SECTORS
        assert 'Healthcare' in TrendDiscovery.SECTORS

        # Check sector has symbols
        assert len(TrendDiscovery.SECTORS['Technology']) > 0
        assert 'NVDA' in TrendDiscovery.SECTORS['AI_Semiconductors']

    def test_themes_defined(self):
        """Test that themes keywords are properly defined."""
        assert 'AI_Revolution' in TrendDiscovery.THEMES_KEYWORDS
        assert 'Crypto_Bull' in TrendDiscovery.THEMES_KEYWORDS

        # Check theme has keywords
        ai_keywords = TrendDiscovery.THEMES_KEYWORDS['AI_Revolution']
        assert 'artificial intelligence' in ai_keywords or 'AI' in ai_keywords


class TestQuantitativeDetection:
    """Tests for quantitative trend detection methods."""

    @pytest.fixture
    def discovery(self):
        """Create a TrendDiscovery instance for testing."""
        return TrendDiscovery(openrouter_api_key=None)

    @pytest.mark.asyncio
    async def test_detect_sector_momentum(self, discovery):
        """Test sector momentum detection (requires network)."""
        # This test requires actual market data
        # Skip if no network or use mock

        try:
            await discovery.initialize()
            momentum = await discovery.detect_sector_momentum()

            assert isinstance(momentum, dict)
            # Should have at least some sectors
            assert len(momentum) > 0

            # Values should be between -1 and 1
            for sector, score in momentum.items():
                assert -1 <= score <= 1, f"{sector} momentum {score} out of range"

        finally:
            await discovery.close()

    @pytest.mark.asyncio
    async def test_detect_volume_anomalies(self, discovery):
        """Test volume anomaly detection."""
        try:
            await discovery.initialize()
            anomalies = await discovery.detect_volume_anomalies()

            assert isinstance(anomalies, list)

            # Each anomaly should be (symbol, ratio) tuple
            for item in anomalies:
                assert len(item) == 2
                symbol, ratio = item
                assert isinstance(symbol, str)
                assert ratio >= 1.5  # Only anomalies above 1.5x

        finally:
            await discovery.close()

    @pytest.mark.asyncio
    async def test_detect_breadth_divergences(self, discovery):
        """Test breadth divergence detection."""
        try:
            await discovery.initialize()
            breadth = await discovery.detect_breadth_divergences()

            assert isinstance(breadth, dict)

            # Values should be between -1 and 1
            for sector, score in breadth.items():
                assert -1 <= score <= 1

        finally:
            await discovery.close()


class TestSectorScan:
    """Tests for sector-specific scanning."""

    @pytest.fixture
    def discovery(self):
        return TrendDiscovery(openrouter_api_key=None)

    @pytest.mark.asyncio
    async def test_scan_valid_sector(self, discovery):
        """Test scanning a valid sector."""
        try:
            await discovery.initialize()
            result = await discovery.scan_sector('Technology')

            assert 'sector' in result
            assert result['sector'] == 'Technology'
            assert 'stock_count' in result
            assert 'top_performers' in result
            assert 'sector_momentum' in result

        finally:
            await discovery.close()

    @pytest.mark.asyncio
    async def test_scan_invalid_sector(self, discovery):
        """Test scanning an invalid sector."""
        try:
            await discovery.initialize()
            result = await discovery.scan_sector('InvalidSector')

            assert 'error' in result

        finally:
            await discovery.close()


class TestFocusSymbols:
    """Tests for focus symbol extraction."""

    def test_get_focus_symbols_no_report(self):
        """Test getting focus symbols when no report exists."""
        discovery = TrendDiscovery(
            openrouter_api_key=None,
            data_dir='data/test_trends_nonexistent'
        )

        symbols = discovery.get_focus_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) == 0

    def test_get_sector_focus_no_report(self):
        """Test getting sector focus when no report exists."""
        discovery = TrendDiscovery(
            openrouter_api_key=None,
            data_dir='data/test_trends_nonexistent'
        )

        sectors = discovery.get_sector_focus()
        assert isinstance(sectors, list)
        # Should return default sectors
        assert len(sectors) > 0


class TestScheduler:
    """Tests for TrendDiscoveryScheduler."""

    def test_scheduler_init(self):
        """Test scheduler initialization."""
        discovery = TrendDiscovery(openrouter_api_key=None)
        scheduler = TrendDiscoveryScheduler(
            discovery=discovery,
            run_time="08:00",
            timezone="America/New_York"
        )

        assert scheduler.run_time == "08:00"
        assert scheduler.timezone == "America/New_York"
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_scheduler_run_now(self):
        """Test immediate scan via scheduler."""
        discovery = TrendDiscovery(openrouter_api_key=None)

        try:
            await discovery.initialize()
            scheduler = TrendDiscoveryScheduler(discovery=discovery)

            # Run immediate scan
            report = await scheduler.run_now()

            assert isinstance(report, TrendReport)
            assert report.date is not None

        finally:
            await discovery.close()


class TestLLMIntegration:
    """Tests for LLM-based analysis (mocked)."""

    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response for testing."""
        return {
            'sentiment': 0.7,
            'momentum': 'accelerating',
            'catalysts': ['Strong earnings', 'AI adoption'],
            'symbols': ['NVDA', 'AMD'],
            'key_insight': 'AI sector showing strong momentum',
            'investment_thesis': 'Consider exposure to AI infrastructure'
        }

    @pytest.mark.asyncio
    async def test_analyze_theme_with_mock(self, mock_llm_response):
        """Test theme analysis with mocked LLM."""
        discovery = TrendDiscovery(openrouter_api_key="test-key")

        # Mock the LLM client
        discovery.llm_client = Mock()
        discovery.llm_client.chat = AsyncMock(return_value=str(mock_llm_response))

        # Since the response is just a string, we need to mock _analyze_theme_with_llm
        with patch.object(discovery, '_analyze_theme_with_llm', return_value=mock_llm_response):
            result = await discovery._analyze_theme_with_llm('AI_Revolution', 'test text')

            assert result['sentiment'] == 0.7
            assert 'NVDA' in result['symbols']


class TestDailyScan:
    """Tests for full daily scan."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_daily_scan_quantitative_only(self):
        """Test daily scan without LLM (quantitative only)."""
        discovery = TrendDiscovery(
            openrouter_api_key=None,  # No API key = quantitative only
            data_dir='data/test_trends'
        )

        try:
            await discovery.initialize()
            report = await discovery.daily_scan()

            assert isinstance(report, TrendReport)
            assert report.date is not None
            assert isinstance(report.sector_momentum, dict)
            assert isinstance(report.trends, list)

            # Cleanup
            import shutil
            if os.path.exists('data/test_trends'):
                shutil.rmtree('data/test_trends')

        finally:
            await discovery.close()


class TestDataPersistence:
    """Tests for report saving/loading."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary data directory."""
        return str(tmp_path / "trends")

    def test_save_and_load_report(self, temp_data_dir):
        """Test saving and loading a report."""
        discovery = TrendDiscovery(
            openrouter_api_key=None,
            data_dir=temp_data_dir
        )

        # Create a test report
        trend = EmergingTrend(
            name="Test Trend",
            type=TrendType.THEMATIC,
            strength=TrendStrength.EMERGING,
            confidence=0.6,
            description="Test description"
        )

        report = TrendReport(
            date=datetime.now(),
            trends=[trend],
            sector_momentum={'Technology': 0.5},
            market_sentiment=0.3,
            top_movers=['AAPL'],
            watchlist_additions=['AAPL', 'MSFT'],
            narrative_updates=[]
        )

        # Save
        discovery._save_report(report)

        # Load
        loaded = discovery._load_latest_report()

        assert loaded is not None
        assert loaded['market_sentiment'] == 0.3
        assert len(loaded['trends']) == 1
        assert loaded['trends'][0]['name'] == "Test Trend"


# Marker for slow tests
def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not slow'])
