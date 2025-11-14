"""
Unit tests for EMA Analyzer
"""
import pytest
import pandas as pd
import numpy as np
from src.indicators.ema_analyzer import EMAAnalyzer


class TestEMAAnalyzer:
    """Test cases for EMA Analyzer"""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        return data

    @pytest.fixture
    def analyzer(self):
        """Create EMA Analyzer instance"""
        return EMAAnalyzer()

    @pytest.mark.unit
    def test_calculate_emas(self, analyzer, sample_data):
        """Test EMA calculation"""
        result = analyzer.calculate_emas(sample_data)

        # Check that EMA columns are added
        assert 'EMA_24' in result.columns
        assert 'EMA_38' in result.columns
        assert 'EMA_62' in result.columns

        # Check that EMAs are not NaN after warmup period
        assert not result['EMA_24'].iloc[-1:].isna().any()
        assert not result['EMA_38'].iloc[-1:].isna().any()
        assert not result['EMA_62'].iloc[-1:].isna().any()

    @pytest.mark.unit
    def test_ema_values_are_numeric(self, analyzer, sample_data):
        """Test that EMA values are numeric"""
        result = analyzer.calculate_emas(sample_data)

        assert pd.api.types.is_numeric_dtype(result['EMA_24'])
        assert pd.api.types.is_numeric_dtype(result['EMA_38'])
        assert pd.api.types.is_numeric_dtype(result['EMA_62'])

    @pytest.mark.unit
    def test_empty_dataframe(self, analyzer):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame()

        # Should handle empty dataframe gracefully
        with pytest.raises((ValueError, KeyError)):
            analyzer.calculate_emas(empty_df)

    @pytest.mark.unit
    def test_detect_crossovers(self, analyzer, sample_data):
        """Test crossover detection"""
        df_with_emas = analyzer.calculate_emas(sample_data)
        crossovers = analyzer.detect_crossovers(df_with_emas, 'daily')

        # Crossovers should be a list
        assert isinstance(crossovers, list)

        # Each crossover should have required fields
        for crossover in crossovers:
            assert 'date' in crossover
            assert 'type' in crossover
            assert 'price' in crossover
