"""Tests for CorrelationManager - V7 diversification."""
import pytest
from unittest.mock import patch, MagicMock
from src.execution.correlation_manager import (
    CorrelationManager, CorrelationConfig, PortfolioPosition
)


class TestSectorExposure:
    """Test sector concentration limits."""

    def test_first_position_always_allowed(self):
        cm = CorrelationManager()
        allowed, reason = cm.check_new_position('AAPL', [], 100000)
        assert allowed is True

    def test_sector_limit_respected(self):
        cm = CorrelationManager(CorrelationConfig(max_sector_pct=0.25))
        positions = [
            PortfolioPosition(symbol='AAPL', sector='Technology', value=25000),
        ]
        # Adding another tech stock that would push sector to 31%
        cm._sector_cache['MSFT'] = 'Technology'
        allowed, reason = cm.check_new_position(
            'MSFT', positions, 100000, new_position_value=6000
        )
        # Sector would be (25000+6000)/100000 = 31% > 25%
        assert allowed is False

    def test_different_sector_allowed(self):
        cm = CorrelationManager(CorrelationConfig(max_sector_pct=0.25))
        positions = [
            PortfolioPosition(symbol='AAPL', sector='Technology', value=20000),
        ]
        cm._sector_cache['JPM'] = 'Financial Services'
        allowed, reason = cm.check_new_position(
            'JPM', positions, 100000, new_position_value=10000
        )
        assert allowed is True

    def test_single_stock_limit(self):
        cm = CorrelationManager(CorrelationConfig(max_single_stock_pct=0.15))
        positions = [
            PortfolioPosition(symbol='AAPL', sector='Technology', value=10000),
        ]
        # 20% of portfolio in one stock > 15% limit
        allowed, reason = cm.check_new_position(
            'MSFT', positions, 100000, new_position_value=20000
        )
        assert allowed is False

    def test_single_stock_within_limit(self):
        cm = CorrelationManager(CorrelationConfig(max_single_stock_pct=0.15))
        positions = [
            PortfolioPosition(symbol='AAPL', sector='Technology', value=10000),
        ]
        cm._sector_cache['MSFT'] = 'Technology'
        # 10% of portfolio < 15% limit
        allowed, reason = cm.check_new_position(
            'MSFT', positions, 100000, new_position_value=10000
        )
        # Stock pct is OK (10%), but need to check sector too
        # Sector would be (10000+10000)/100000 = 20% < 25% default
        assert allowed is True

    def test_sector_at_boundary(self):
        """Exactly at 25% should still be allowed."""
        cm = CorrelationManager(CorrelationConfig(max_sector_pct=0.25))
        positions = [
            PortfolioPosition(symbol='AAPL', sector='Technology', value=20000),
        ]
        cm._sector_cache['MSFT'] = 'Technology'
        allowed, reason = cm.check_new_position(
            'MSFT', positions, 100000, new_position_value=5000
        )
        # Sector would be (20000+5000)/100000 = 25% = limit, not > limit
        assert allowed is True

    def test_zero_portfolio_value(self):
        """Zero total value should not cause division error."""
        cm = CorrelationManager()
        positions = [
            PortfolioPosition(symbol='AAPL', sector='Technology', value=0),
        ]
        cm._sector_cache['MSFT'] = 'Technology'
        allowed, reason = cm.check_new_position(
            'MSFT', positions, 0, new_position_value=0
        )
        assert allowed is True


class TestDiversificationScore:
    """Test portfolio diversification metrics."""

    def test_empty_portfolio(self):
        cm = CorrelationManager()
        score = cm.get_portfolio_diversification_score([], 0)
        assert score['diversification_score'] == 50

    def test_single_sector_concentrated(self):
        cm = CorrelationManager()
        positions = [
            PortfolioPosition('AAPL', 'Technology', 50000),
            PortfolioPosition('MSFT', 'Technology', 50000),
        ]
        score = cm.get_portfolio_diversification_score(positions, 100000)
        assert score['num_sectors'] == 1
        assert score['sector_concentration'] == 1.0  # HHI = 1 (all in one sector)

    def test_diversified_portfolio(self):
        cm = CorrelationManager()
        positions = [
            PortfolioPosition('AAPL', 'Technology', 25000),
            PortfolioPosition('JPM', 'Financial Services', 25000),
            PortfolioPosition('JNJ', 'Healthcare', 25000),
            PortfolioPosition('XOM', 'Energy', 25000),
        ]
        score = cm.get_portfolio_diversification_score(positions, 100000)
        assert score['num_sectors'] == 4
        assert score['sector_concentration'] == 0.25  # HHI = 4 * 0.25^2
        assert score['diversification_score'] == 75.0

    def test_two_sectors_score(self):
        cm = CorrelationManager()
        positions = [
            PortfolioPosition('AAPL', 'Technology', 50000),
            PortfolioPosition('JPM', 'Financial Services', 50000),
        ]
        score = cm.get_portfolio_diversification_score(positions, 100000)
        assert score['num_sectors'] == 2
        # HHI = 0.5^2 + 0.5^2 = 0.5
        assert score['sector_concentration'] == 0.5
        assert score['diversification_score'] == 50.0

    def test_largest_sector_identified(self):
        cm = CorrelationManager()
        positions = [
            PortfolioPosition('AAPL', 'Technology', 60000),
            PortfolioPosition('JPM', 'Financial Services', 40000),
        ]
        score = cm.get_portfolio_diversification_score(positions, 100000)
        assert score['largest_sector'] == 'Technology'
        assert score['largest_sector_pct'] == 0.6

    def test_position_count(self):
        cm = CorrelationManager()
        positions = [
            PortfolioPosition('AAPL', 'Technology', 30000),
            PortfolioPosition('MSFT', 'Technology', 20000),
            PortfolioPosition('JPM', 'Financial Services', 50000),
        ]
        score = cm.get_portfolio_diversification_score(positions, 100000)
        assert score['num_positions'] == 3
        assert score['num_sectors'] == 2


class TestCacheManagement:
    """Test cache behavior."""

    def test_clear_cache(self):
        cm = CorrelationManager()
        cm._sector_cache['AAPL'] = 'Technology'
        cm._price_cache['AAPL'] = None
        cm.clear_cache()
        assert len(cm._sector_cache) == 0
        assert len(cm._price_cache) == 0

    def test_sector_cache_populated(self):
        cm = CorrelationManager()
        cm._sector_cache['AAPL'] = 'Technology'
        assert cm._sector_cache['AAPL'] == 'Technology'


class TestDefaultConfig:
    """Test default configuration values."""

    def test_default_max_sector_pct(self):
        cm = CorrelationManager()
        assert cm.config.max_sector_pct == 0.25

    def test_default_max_avg_correlation(self):
        cm = CorrelationManager()
        assert cm.config.max_avg_correlation == 0.70

    def test_default_max_single_stock_pct(self):
        cm = CorrelationManager()
        assert cm.config.max_single_stock_pct == 0.15

    def test_custom_config(self):
        config = CorrelationConfig(
            max_sector_pct=0.30,
            max_avg_correlation=0.80,
            max_single_stock_pct=0.20
        )
        cm = CorrelationManager(config)
        assert cm.config.max_sector_pct == 0.30
        assert cm.config.max_avg_correlation == 0.80
        assert cm.config.max_single_stock_pct == 0.20


class TestCorrelationCheck:
    """Test correlation checking with mocked data."""

    def test_skip_correlation_with_few_positions(self):
        """Correlation check requires min_positions_for_check (default 2)."""
        cm = CorrelationManager()
        positions = [
            PortfolioPosition(symbol='AAPL', sector='Technology', value=10000),
        ]
        cm._sector_cache['MSFT'] = 'Technology'
        # Only 1 position, correlation check skipped
        # Sector check: (10000+10000)/100000 = 20% < 25%
        allowed, reason = cm.check_new_position(
            'MSFT', positions, 100000, new_position_value=10000
        )
        assert allowed is True
