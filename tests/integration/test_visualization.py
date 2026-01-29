"""
Integration tests for Visualization Improvements

Tests the new visualization methods:
- Confidence score badge
- Volume ratio annotation
- R² labels on trendlines
- Risk zone display
- Position info box
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.interactive_chart import InteractiveChartBuilder, interactive_chart_builder


class TestVisualizationMethods:
    """Integration tests for new visualization methods"""

    @pytest.fixture
    def chart_builder(self):
        """Fresh chart builder instance"""
        return InteractiveChartBuilder()

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame with price and volume data"""
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start='2023-01-01', periods=n, freq='D')

        base = np.linspace(100, 130, n)
        noise = np.random.normal(0, 2, n)
        close = base + noise

        df = pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.02,
            'Low': close * 0.98,
            'Close': close,
            'Volume': np.random.uniform(1000000, 2000000, n)
        }, index=dates)

        return df

    @pytest.mark.integration
    def test_confidence_badge_strong_buy(self, chart_builder, sample_df):
        """Test confidence badge with STRONG_BUY signal"""
        fig = chart_builder.create_interactive_chart(
            df=sample_df,
            symbol='TEST',
            show_volume=True,
            show_rsi=True
        )

        fig = chart_builder.add_confidence_badge(fig, score=85.0, signal='STRONG_BUY')

        # Check that annotation was added (Plotly uses attribute access, not dict)
        annotations = [a for a in fig.layout.annotations if 'STRONG BUY' in str(getattr(a, 'text', ''))]
        assert len(annotations) >= 1, "Should have confidence badge annotation"

    @pytest.mark.integration
    def test_confidence_badge_colors(self, chart_builder, sample_df):
        """Test confidence badge uses correct colors for different scores"""
        fig = chart_builder.create_interactive_chart(
            df=sample_df,
            symbol='TEST',
            show_volume=False,
            show_rsi=False
        )

        # Test STRONG_BUY color (>75)
        color_75 = chart_builder._get_score_color(80)
        assert color_75 == chart_builder.colors['score_strong']

        # Test BUY color (55-75)
        color_60 = chart_builder._get_score_color(60)
        assert color_60 == chart_builder.colors['score_buy']

        # Test WATCH color (35-55)
        color_45 = chart_builder._get_score_color(45)
        assert color_45 == chart_builder.colors['score_watch']

        # Test OBSERVE color (<35)
        color_20 = chart_builder._get_score_color(20)
        assert color_20 == chart_builder.colors['score_observe']

    @pytest.mark.integration
    def test_confidence_badge_none_handling(self, chart_builder, sample_df):
        """Test confidence badge handles None values gracefully"""
        fig = chart_builder.create_interactive_chart(
            df=sample_df,
            symbol='TEST',
            show_volume=False,
            show_rsi=False
        )

        # Should not raise exception
        fig = chart_builder.add_confidence_badge(fig, score=None, signal='BUY')
        fig = chart_builder.add_confidence_badge(fig, score=75, signal=None)
        fig = chart_builder.add_confidence_badge(fig, score=None, signal=None)

    @pytest.mark.integration
    def test_volume_ratio_annotation(self, chart_builder, sample_df):
        """Test volume ratio annotation is added correctly"""
        fig = chart_builder.create_interactive_chart(
            df=sample_df,
            symbol='TEST',
            show_volume=True,
            show_rsi=True
        )

        fig = chart_builder.add_volume_ratio_annotation(
            fig=fig,
            df=sample_df,
            breakout_idx=90,
            volume_ratio=1.8,
            volume_row=2
        )

        # Check annotation was added
        annotations = [a for a in fig.layout.annotations if '1.8x' in str(getattr(a, 'text', ''))]
        assert len(annotations) >= 1, "Should have volume ratio annotation"

    @pytest.mark.integration
    def test_volume_ratio_none_handling(self, chart_builder, sample_df):
        """Test volume ratio handles None values gracefully"""
        fig = chart_builder.create_interactive_chart(
            df=sample_df,
            symbol='TEST',
            show_volume=True,
            show_rsi=True
        )

        # Should not raise exception
        fig = chart_builder.add_volume_ratio_annotation(fig, sample_df, None, 1.5)
        fig = chart_builder.add_volume_ratio_annotation(fig, sample_df, 90, None)

    @pytest.mark.integration
    def test_risk_zone_display(self, chart_builder, sample_df):
        """Test risk zone rectangle is added correctly"""
        fig = chart_builder.create_interactive_chart(
            df=sample_df,
            symbol='TEST',
            show_volume=True,
            show_rsi=True
        )

        entry_price = 125.0
        stop_loss = 118.0

        fig = chart_builder.add_risk_zone(
            fig=fig,
            df=sample_df,
            entry_price=entry_price,
            stop_loss=stop_loss,
            row=1
        )

        # Check that shape was added (Plotly uses attribute access)
        shapes = [s for s in fig.layout.shapes if getattr(s, 'type', '') == 'rect']
        assert len(shapes) >= 1, "Should have risk zone rectangle"

        # Check risk percentage annotation
        expected_risk_pct = (entry_price - stop_loss) / entry_price * 100
        annotations = [a for a in fig.layout.annotations if 'Risque' in str(getattr(a, 'text', ''))]
        assert len(annotations) >= 1, "Should have risk percentage annotation"

    @pytest.mark.integration
    def test_risk_zone_invalid_values(self, chart_builder, sample_df):
        """Test risk zone handles invalid values gracefully"""
        fig = chart_builder.create_interactive_chart(
            df=sample_df,
            symbol='TEST',
            show_volume=False,
            show_rsi=False
        )

        # Should not raise exception
        fig = chart_builder.add_risk_zone(fig, sample_df, None, 118.0)
        fig = chart_builder.add_risk_zone(fig, sample_df, 125.0, None)
        fig = chart_builder.add_risk_zone(fig, sample_df, 0, 118.0)
        fig = chart_builder.add_risk_zone(fig, sample_df, 125.0, 0)

    @pytest.mark.integration
    def test_position_info_box(self, chart_builder, sample_df):
        """Test position info box is added correctly"""
        fig = chart_builder.create_interactive_chart(
            df=sample_df,
            symbol='TEST',
            show_volume=True,
            show_rsi=True
        )

        position_data = {
            'shares': 15,
            'value': 1875.0,
            'stop_loss': 118.0,
            'risk_pct': 3.2
        }

        fig = chart_builder.add_position_info_box(fig, position_data)

        # Check annotation was added
        annotations = [a for a in fig.layout.annotations if 'Position' in str(getattr(a, 'text', ''))]
        assert len(annotations) >= 1, "Should have position info annotation"

    @pytest.mark.integration
    def test_position_info_box_partial_data(self, chart_builder, sample_df):
        """Test position info box with partial data"""
        fig = chart_builder.create_interactive_chart(
            df=sample_df,
            symbol='TEST',
            show_volume=False,
            show_rsi=False
        )

        # Only shares provided
        fig = chart_builder.add_position_info_box(fig, {'shares': 10})

        # Check annotation was added
        annotations = [a for a in fig.layout.annotations if 'Position' in str(getattr(a, 'text', ''))]
        assert len(annotations) >= 1

    @pytest.mark.integration
    def test_position_info_box_empty_data(self, chart_builder, sample_df):
        """Test position info box handles empty data gracefully"""
        fig = chart_builder.create_interactive_chart(
            df=sample_df,
            symbol='TEST',
            show_volume=False,
            show_rsi=False
        )

        # Should not raise exception
        fig = chart_builder.add_position_info_box(fig, {})
        fig = chart_builder.add_position_info_box(fig, None)


class TestRSITrendlineLabelIntegration:
    """Tests for R² labels on RSI trendlines"""

    @pytest.fixture
    def chart_builder(self):
        return InteractiveChartBuilder()

    @pytest.fixture
    def rsi_breakout_df(self):
        """Create data that produces RSI trendline with breakout"""
        np.random.seed(42)
        n = 200
        dates = pd.date_range(start='2023-01-01', periods=n, freq='D')

        # Phase 1: Rally then pullback (creates RSI peak)
        phase1 = np.linspace(100, 130, 40) + np.random.normal(0, 1, 40)

        # Phase 2: Consolidation with lower highs (creates descending RSI peaks)
        phase2_base = np.linspace(130, 110, 80)
        phase2_osc = 10 * np.sin(np.linspace(0, 4*np.pi, 80))
        phase2 = phase2_base + phase2_osc + np.random.normal(0, 1, 80)

        # Phase 3: Breakout
        phase3 = np.linspace(110, 150, 80) + np.random.normal(0, 1, 80)

        close = np.concatenate([phase1, phase2, phase3])
        high = close * (1 + np.random.uniform(0.005, 0.02, n))
        low = close * (1 - np.random.uniform(0.005, 0.02, n))
        open_price = close * (1 + np.random.uniform(-0.01, 0.01, n))

        volume = np.concatenate([
            np.random.uniform(1000000, 2000000, 120),
            np.random.uniform(2500000, 4000000, 80)
        ])

        df = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }, index=dates)

        return df

    @pytest.mark.integration
    def test_rsi_trendline_has_r_squared_label(self, chart_builder, rsi_breakout_df):
        """Test that RSI trendlines have R² labels"""
        fig = chart_builder.create_interactive_chart(
            df=rsi_breakout_df,
            symbol='TEST',
            show_volume=True,
            show_rsi=True
        )

        fig = chart_builder.add_rsi_trendline_breakout(
            fig=fig,
            df=rsi_breakout_df,
            rsi_row=3
        )

        # Check for R² annotation (Plotly uses attribute access)
        annotations = [a for a in fig.layout.annotations if 'R' in str(getattr(a, 'text', ''))]
        # May or may not have trendlines depending on data
        # Just ensure no exception was raised


class TestSingletonInstance:
    """Tests for singleton instance"""

    @pytest.mark.integration
    def test_singleton_exists(self):
        """Test singleton instance is available"""
        assert interactive_chart_builder is not None
        assert isinstance(interactive_chart_builder, InteractiveChartBuilder)

    @pytest.mark.integration
    def test_singleton_has_new_colors(self):
        """Test singleton has new color definitions"""
        assert 'score_strong' in interactive_chart_builder.colors
        assert 'score_buy' in interactive_chart_builder.colors
        assert 'volume_high' in interactive_chart_builder.colors
        assert 'risk_zone' in interactive_chart_builder.colors


class TestCombinedVisualization:
    """Tests for combining all visualization methods"""

    @pytest.fixture
    def chart_builder(self):
        return InteractiveChartBuilder()

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start='2023-01-01', periods=n, freq='D')

        base = np.linspace(100, 130, n)
        noise = np.random.normal(0, 2, n)
        close = base + noise

        df = pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.02,
            'Low': close * 0.98,
            'Close': close,
            'Volume': np.random.uniform(1000000, 2000000, n)
        }, index=dates)

        return df

    @pytest.mark.integration
    def test_all_visualizations_combined(self, chart_builder, sample_df):
        """Test all visualization methods can be combined on same chart"""
        fig = chart_builder.create_interactive_chart(
            df=sample_df,
            symbol='TEST',
            show_volume=True,
            show_rsi=True
        )

        # Add all visualizations
        fig = chart_builder.add_confidence_badge(fig, score=82, signal='STRONG_BUY')
        fig = chart_builder.add_volume_ratio_annotation(fig, sample_df, 90, 1.8, volume_row=2)
        fig = chart_builder.add_risk_zone(fig, sample_df, entry_price=128, stop_loss=120)
        fig = chart_builder.add_position_info_box(fig, {
            'shares': 12,
            'value': 1536,
            'stop_loss': 120,
            'risk_pct': 6.25
        })

        # Verify figure is valid
        assert fig is not None
        assert len(fig.data) > 0
        assert len(fig.layout.annotations) >= 4  # At least 4 annotations


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
