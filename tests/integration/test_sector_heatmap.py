"""
Integration tests for Sector Heatmap

Tests sector performance tracking and visualization:
- Sector performance data
- Market breadth calculation
- Sector rotation analysis
- Heatmap visualizations
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from src.utils.sector_heatmap import (
    SectorHeatmapBuilder,
    SectorPerformance,
    sector_heatmap_builder,
    SECTORS
)
from src.utils.interactive_chart import InteractiveChartBuilder


class TestSectorDefinitions:
    """Tests for sector definitions"""

    @pytest.mark.integration
    def test_sectors_defined(self):
        """Test all sectors are defined"""
        expected_sectors = [
            'Technology', 'Healthcare', 'Financials',
            'Consumer Discretionary', 'Consumer Staples', 'Energy',
            'Industrials', 'Materials', 'Real Estate',
            'Utilities', 'Communication Services'
        ]

        for sector in expected_sectors:
            assert sector in SECTORS, f"Missing sector: {sector}"

    @pytest.mark.integration
    def test_sector_has_required_fields(self):
        """Test each sector has required fields"""
        for name, info in SECTORS.items():
            assert 'etf' in info, f"Sector {name} missing 'etf'"
            assert 'color' in info, f"Sector {name} missing 'color'"
            assert 'stocks' in info, f"Sector {name} missing 'stocks'"
            assert len(info['stocks']) >= 5, f"Sector {name} has too few stocks"


class TestSectorPerformance:
    """Tests for SectorPerformance dataclass"""

    @pytest.fixture
    def sample_performance(self):
        """Sample SectorPerformance object"""
        return SectorPerformance(
            name='Technology',
            etf='XLK',
            color='#2962FF',
            perf_1d=1.5,
            perf_1w=3.2,
            perf_1m=-2.1,
            perf_ytd=15.8,
            top_gainers=[{'symbol': 'NVDA', 'perf': 5.2}],
            top_losers=[{'symbol': 'INTC', 'perf': -3.1}],
            signal_count=3,
            avg_volume_ratio=1.5
        )

    @pytest.mark.integration
    def test_performance_attributes(self, sample_performance):
        """Test SectorPerformance attributes"""
        assert sample_performance.name == 'Technology'
        assert sample_performance.etf == 'XLK'
        assert sample_performance.perf_1d == 1.5
        assert sample_performance.perf_1w == 3.2
        assert sample_performance.perf_1m == -2.1
        assert sample_performance.perf_ytd == 15.8

    @pytest.mark.integration
    def test_performance_top_movers(self, sample_performance):
        """Test top movers data"""
        assert len(sample_performance.top_gainers) == 1
        assert sample_performance.top_gainers[0]['symbol'] == 'NVDA'
        assert len(sample_performance.top_losers) == 1


class TestSectorHeatmapBuilder:
    """Tests for SectorHeatmapBuilder class"""

    @pytest.fixture
    def builder(self):
        """Fresh builder instance"""
        return SectorHeatmapBuilder()

    @pytest.fixture
    def mock_sector_data(self):
        """Mock sector performance data"""
        return [
            SectorPerformance(
                name='Technology',
                etf='XLK',
                color='#2962FF',
                perf_1d=2.0,
                perf_1w=5.0,
                perf_1m=8.0,
                perf_ytd=20.0,
                top_gainers=[],
                top_losers=[],
                signal_count=0,
                avg_volume_ratio=1.0
            ),
            SectorPerformance(
                name='Healthcare',
                etf='XLV',
                color='#00C853',
                perf_1d=-1.0,
                perf_1w=-2.0,
                perf_1m=3.0,
                perf_ytd=10.0,
                top_gainers=[],
                top_losers=[],
                signal_count=0,
                avg_volume_ratio=1.0
            ),
            SectorPerformance(
                name='Financials',
                etf='XLF',
                color='#FF6D00',
                perf_1d=0.5,
                perf_1w=1.5,
                perf_1m=-1.0,
                perf_ytd=5.0,
                top_gainers=[],
                top_losers=[],
                signal_count=0,
                avg_volume_ratio=1.0
            )
        ]

    @pytest.mark.integration
    def test_get_sector_for_symbol(self, builder):
        """Test symbol to sector mapping"""
        assert builder.get_sector_for_symbol('AAPL') == 'Technology'
        assert builder.get_sector_for_symbol('JPM') == 'Financials'
        assert builder.get_sector_for_symbol('XOM') == 'Energy'
        assert builder.get_sector_for_symbol('UNKNOWN') is None

    @pytest.mark.integration
    def test_get_sector_for_symbol_case_insensitive(self, builder):
        """Test symbol lookup is case insensitive"""
        assert builder.get_sector_for_symbol('aapl') == 'Technology'
        assert builder.get_sector_for_symbol('Aapl') == 'Technology'

    @pytest.mark.integration
    def test_update_signal_counts(self, builder, mock_sector_data):
        """Test updating signal counts"""
        alerts = [
            {'symbol': 'AAPL'},
            {'symbol': 'MSFT'},
            {'symbol': 'NVDA'},
            {'symbol': 'JPM'}
        ]

        updated = builder.update_signal_counts(alerts, mock_sector_data)

        tech = next(s for s in updated if s.name == 'Technology')
        fin = next(s for s in updated if s.name == 'Financials')

        assert tech.signal_count == 3  # AAPL, MSFT, NVDA
        assert fin.signal_count == 1   # JPM

    @pytest.mark.integration
    def test_get_heatmap_data(self, builder):
        """Test heatmap data generation"""
        with patch.object(builder, 'get_sector_performance') as mock_perf:
            mock_perf.return_value = [
                SectorPerformance(
                    name='Tech', etf='XLK', color='#2962FF',
                    perf_1d=1.0, perf_1w=2.0, perf_1m=3.0, perf_ytd=10.0,
                    top_gainers=[], top_losers=[], signal_count=0, avg_volume_ratio=1.0
                )
            ]

            df = builder.get_heatmap_data()

            assert isinstance(df, pd.DataFrame)
            assert 'Sector' in df.columns
            assert '1D %' in df.columns
            assert '1W %' in df.columns


class TestMarketBreadth:
    """Tests for market breadth calculations"""

    @pytest.fixture
    def builder_with_data(self):
        """Builder with mocked data"""
        builder = SectorHeatmapBuilder()

        # Mock the performance data
        mock_data = [
            SectorPerformance('Tech', 'XLK', '#fff', 2.0, 3.0, 5.0, 15.0, [], [], 0, 1.0),
            SectorPerformance('Health', 'XLV', '#fff', 1.0, 2.0, 3.0, 10.0, [], [], 0, 1.0),
            SectorPerformance('Fin', 'XLF', '#fff', -1.0, -2.0, -1.0, 5.0, [], [], 0, 1.0),
            SectorPerformance('Energy', 'XLE', '#fff', -2.0, -3.0, -5.0, -10.0, [], [], 0, 1.0),
        ]

        builder._cache['sector_perf'] = mock_data
        builder._cache_time = datetime.now()

        return builder

    @pytest.mark.integration
    def test_market_breadth_advancing(self, builder_with_data):
        """Test advancing/declining count"""
        breadth = builder_with_data.get_market_breadth()

        assert breadth['advancing'] == 2   # Tech, Health > 0
        assert breadth['declining'] == 2   # Fin, Energy < 0
        assert breadth['total'] == 4

    @pytest.mark.integration
    def test_market_breadth_ratio(self, builder_with_data):
        """Test breadth ratio calculation"""
        breadth = builder_with_data.get_market_breadth()

        assert breadth['breadth_ratio'] == 1.0  # 2/2

    @pytest.mark.integration
    def test_market_breadth_avg_performance(self, builder_with_data):
        """Test average performance calculation"""
        breadth = builder_with_data.get_market_breadth()

        # (2 + 1 + -1 + -2) / 4 = 0
        assert breadth['avg_performance'] == 0.0


class TestSectorRotation:
    """Tests for sector rotation analysis"""

    @pytest.fixture
    def builder_with_rotation(self):
        """Builder with data for rotation testing"""
        builder = SectorHeatmapBuilder()

        # Cyclical outperforming
        mock_data = [
            SectorPerformance('Technology', 'XLK', '#fff', 2.0, 4.0, 8.0, 20.0, [], [], 0, 1.0),
            SectorPerformance('Financials', 'XLF', '#fff', 1.5, 3.0, 5.0, 15.0, [], [], 0, 1.0),
            SectorPerformance('Consumer Discretionary', 'XLY', '#fff', 1.0, 3.5, 6.0, 18.0, [], [], 0, 1.0),
            SectorPerformance('Utilities', 'XLU', '#fff', 0.5, 1.0, 2.0, 5.0, [], [], 0, 1.0),
            SectorPerformance('Consumer Staples', 'XLP', '#fff', 0.3, 0.5, 1.5, 4.0, [], [], 0, 1.0),
        ]

        builder._cache['sector_perf'] = mock_data
        builder._cache_time = datetime.now()

        return builder

    @pytest.mark.integration
    def test_sector_rotation_risk_on(self, builder_with_rotation):
        """Test RISK_ON signal detection"""
        rotation = builder_with_rotation.get_sector_rotation_signal()

        assert rotation['signal'] == 'RISK_ON'
        assert 'outperforming' in rotation['reason'].lower()

    @pytest.mark.integration
    def test_sector_rotation_contains_averages(self, builder_with_rotation):
        """Test rotation contains cyclical/defensive averages"""
        rotation = builder_with_rotation.get_sector_rotation_signal()

        assert 'cyclical_avg' in rotation
        assert 'defensive_avg' in rotation


class TestHeatmapVisualization:
    """Tests for heatmap visualization methods"""

    @pytest.fixture
    def chart_builder(self):
        return InteractiveChartBuilder()

    @pytest.fixture
    def mock_sector_data(self):
        return [
            SectorPerformance('Tech', 'XLK', '#2962FF', 2.0, 5.0, 10.0, 25.0, [], [], 2, 1.5),
            SectorPerformance('Health', 'XLV', '#00C853', -1.0, 2.0, 5.0, 15.0, [], [], 1, 1.2),
            SectorPerformance('Fin', 'XLF', '#FF6D00', 0.5, -1.0, 3.0, 10.0, [], [], 0, 1.0),
        ]

    @pytest.mark.integration
    def test_create_sector_heatmap(self, chart_builder, mock_sector_data):
        """Test treemap creation"""
        fig = chart_builder.create_sector_heatmap(mock_sector_data, metric='1D %')

        assert fig is not None
        assert len(fig.data) > 0

    @pytest.mark.integration
    def test_create_sector_heatmap_empty(self, chart_builder):
        """Test treemap with empty data"""
        fig = chart_builder.create_sector_heatmap([])

        assert fig is not None
        # Should have annotation about no data
        assert len(fig.layout.annotations) > 0

    @pytest.mark.integration
    def test_create_sector_bar_chart(self, chart_builder, mock_sector_data):
        """Test bar chart creation"""
        fig = chart_builder.create_sector_bar_chart(mock_sector_data, metric='1W %')

        assert fig is not None
        assert len(fig.data) > 0

    @pytest.mark.integration
    def test_create_market_breadth_gauge(self, chart_builder):
        """Test breadth gauge creation"""
        breadth_data = {
            'advancing': 7,
            'declining': 4,
            'total': 11,
            'breadth_ratio': 1.75
        }

        fig = chart_builder.create_market_breadth_gauge(breadth_data)

        assert fig is not None
        assert len(fig.data) > 0

    @pytest.mark.integration
    def test_create_sector_rotation_chart(self, chart_builder, mock_sector_data):
        """Test rotation chart creation"""
        fig = chart_builder.create_sector_rotation_chart(mock_sector_data)

        assert fig is not None
        assert len(fig.data) >= len(mock_sector_data)

    @pytest.mark.integration
    def test_heatmap_different_metrics(self, chart_builder, mock_sector_data):
        """Test heatmap with different metrics"""
        for metric in ['1D %', '1W %', '1M %', 'YTD %']:
            fig = chart_builder.create_sector_heatmap(mock_sector_data, metric=metric)
            assert fig is not None


class TestSingletonInstance:
    """Tests for singleton instance"""

    @pytest.mark.integration
    def test_singleton_exists(self):
        """Test singleton is available"""
        assert sector_heatmap_builder is not None
        assert isinstance(sector_heatmap_builder, SectorHeatmapBuilder)

    @pytest.mark.integration
    def test_singleton_has_sectors(self):
        """Test singleton has sector definitions"""
        assert len(sector_heatmap_builder.sectors) == 11


class TestCaching:
    """Tests for caching behavior"""

    @pytest.mark.integration
    def test_cache_is_used(self):
        """Test that cache is used on subsequent calls"""
        builder = SectorHeatmapBuilder()

        # Mock data
        mock_data = [
            SectorPerformance('Tech', 'XLK', '#fff', 1.0, 2.0, 3.0, 10.0, [], [], 0, 1.0)
        ]
        builder._cache['sector_perf'] = mock_data
        builder._cache_time = datetime.now()

        # Should return cached data without API call
        result = builder.get_sector_performance(use_cache=True)

        assert result == mock_data

    @pytest.mark.integration
    def test_cache_bypass(self):
        """Test cache can be bypassed"""
        builder = SectorHeatmapBuilder()

        # Set old cache
        mock_data = [
            SectorPerformance('Tech', 'XLK', '#fff', 1.0, 2.0, 3.0, 10.0, [], [], 0, 1.0)
        ]
        builder._cache['sector_perf'] = mock_data
        builder._cache_time = datetime.now()

        # Bypass cache should try to fetch new data
        # (This will make API call, so we mock it)
        with patch.object(builder, '_fetch_etf_data') as mock_fetch:
            mock_fetch.return_value = {}
            result = builder.get_sector_performance(use_cache=False)

            # With empty ETF data, should return empty list
            # The mock was called because cache was bypassed
            mock_fetch.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
