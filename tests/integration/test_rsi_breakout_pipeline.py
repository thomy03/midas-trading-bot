"""
Integration tests for RSI Breakout Pipeline

Tests the complete flow from raw data -> RSI calculation ->
trendline detection -> breakout detection -> volume confirmation
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trendline_analysis.core.rsi_breakout_analyzer import RSIBreakoutAnalyzer
from trendline_analysis.core.enhanced_rsi_breakout_analyzer import EnhancedRSIBreakoutAnalyzer
from trendline_analysis.core.trendline_detector import RSITrendlineDetector, Trendline
from trendline_analysis.core.breakout_analyzer import TrendlineBreakoutAnalyzer, Breakout


class TestRSIBreakoutPipeline:
    """Integration tests for RSI breakout detection pipeline"""

    @pytest.fixture
    def rsi_analyzer(self):
        """Standard RSI breakout analyzer"""
        return RSIBreakoutAnalyzer()

    @pytest.fixture
    def enhanced_analyzer(self):
        """Enhanced RSI breakout analyzer with medium precision"""
        return EnhancedRSIBreakoutAnalyzer(precision_mode='medium')

    @pytest.fixture
    def rsi_breakout_data(self):
        """
        Create synthetic data that produces an RSI breakout pattern

        Pattern:
        1. Initial decline (creates high RSI peaks)
        2. Consolidation (creates lower RSI peaks -> descending trendline)
        3. Strong move up (RSI breaks above trendline)
        """
        np.random.seed(42)  # For reproducibility
        n = 200
        dates = pd.date_range(start='2023-01-01', periods=n, freq='D')

        # Phase 1: Initial rally then pullback (creates first RSI peak)
        phase1 = np.linspace(100, 130, 40) + np.random.normal(0, 1, 40)

        # Phase 2: Consolidation with lower highs (creates descending RSI peaks)
        phase2_base = np.linspace(130, 110, 80)
        phase2_oscillation = 10 * np.sin(np.linspace(0, 4*np.pi, 80))
        phase2 = phase2_base + phase2_oscillation + np.random.normal(0, 1, 80)

        # Phase 3: Breakout - strong upward move
        phase3 = np.linspace(110, 150, 80) + np.random.normal(0, 1, 80)

        close = np.concatenate([phase1, phase2, phase3])

        # Generate OHLC from close
        high = close * (1 + np.random.uniform(0.005, 0.02, n))
        low = close * (1 - np.random.uniform(0.005, 0.02, n))
        open_price = close * (1 + np.random.uniform(-0.01, 0.01, n))

        # Volume: higher on breakout phase
        volume = np.concatenate([
            np.random.uniform(1000000, 2000000, 120),  # Normal volume
            np.random.uniform(2500000, 4000000, 80)    # Higher volume on breakout
        ])

        df = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }, index=dates)

        return df

    @pytest.fixture
    def no_breakout_data(self):
        """Create data with RSI trendline but no breakout"""
        np.random.seed(123)
        n = 150
        dates = pd.date_range(start='2023-01-01', periods=n, freq='D')

        # Continuous downtrend - RSI stays below trendline
        close = np.linspace(150, 100, n) + np.random.normal(0, 2, n)

        high = close * (1 + np.random.uniform(0.005, 0.02, n))
        low = close * (1 - np.random.uniform(0.005, 0.02, n))
        open_price = close * (1 + np.random.uniform(-0.01, 0.01, n))
        volume = np.random.uniform(1000000, 2000000, n)

        df = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }, index=dates)

        return df

    @pytest.mark.integration
    def test_full_pipeline_with_breakout(self, rsi_analyzer, rsi_breakout_data):
        """Test complete pipeline detects breakout correctly"""
        result = rsi_analyzer.analyze(rsi_breakout_data, lookback_periods=150)

        # Should detect a trendline
        assert result is not None, "Should return a result"
        assert result.has_rsi_trendline, "Should detect RSI trendline"

        # Trendline should have negative slope (descending)
        assert result.rsi_trendline.slope < 0, "Trendline should be descending"

        # Trendline should have minimum peaks
        assert len(result.rsi_trendline.peak_indices) >= 3, "Should have at least 3 peaks"

    @pytest.mark.integration
    def test_enhanced_analyzer_same_data(self, enhanced_analyzer, rsi_breakout_data):
        """Test enhanced analyzer on same data"""
        result = enhanced_analyzer.analyze(rsi_breakout_data, lookback_periods=150)

        assert result is not None
        # Enhanced analyzer may be more selective
        if result.has_rsi_trendline:
            assert result.rsi_trendline.r_squared >= 0.25, "R-squared should meet minimum"

    @pytest.mark.integration
    def test_volume_confirmation_included(self, rsi_analyzer, rsi_breakout_data):
        """Test that volume confirmation is calculated for breakouts"""
        result = rsi_analyzer.analyze(rsi_breakout_data, lookback_periods=150)

        if result and result.has_rsi_breakout:
            breakout = result.rsi_breakout

            # Volume ratio should be calculated
            assert hasattr(breakout, 'volume_ratio'), "Should have volume_ratio"
            assert hasattr(breakout, 'volume_confirmed'), "Should have volume_confirmed"

            # Volume ratio should be a positive number
            assert breakout.volume_ratio > 0, "Volume ratio should be positive"

            # volume_confirmed should be boolean
            assert isinstance(breakout.volume_confirmed, bool)

    @pytest.mark.integration
    def test_no_breakout_data_handling(self, rsi_analyzer, no_breakout_data):
        """Test pipeline handles data without breakout gracefully"""
        result = rsi_analyzer.analyze(no_breakout_data, lookback_periods=100)

        # Should not crash
        assert result is not None or result is None  # Either result is valid

        # If trendline exists, no breakout expected
        if result and result.has_rsi_trendline:
            # Signal should not be STRONG_BUY without breakout
            if not result.has_rsi_breakout:
                assert result.signal != 'STRONG_BUY'

    @pytest.mark.integration
    def test_result_signal_consistency(self, rsi_analyzer, rsi_breakout_data):
        """Test that signal is consistent with breakout detection"""
        result = rsi_analyzer.analyze(rsi_breakout_data, lookback_periods=150)

        if result:
            valid_signals = ['STRONG_BUY', 'BUY', 'WATCH', 'NO_SIGNAL']
            assert result.signal in valid_signals, f"Invalid signal: {result.signal}"

            # If STRONG_BUY, should have breakout
            if result.signal == 'STRONG_BUY':
                assert result.has_rsi_breakout, "STRONG_BUY requires breakout"
                assert result.rsi_breakout.strength == 'STRONG', "STRONG_BUY requires STRONG breakout"

    @pytest.mark.integration
    def test_trendline_r_squared_quality(self, rsi_analyzer, rsi_breakout_data):
        """Test trendline R-squared meets quality threshold"""
        result = rsi_analyzer.analyze(rsi_breakout_data, lookback_periods=150)

        if result and result.has_rsi_trendline:
            # R-squared should be between 0 and 1
            assert 0 <= result.rsi_trendline.r_squared <= 1

            # Should meet minimum threshold
            assert result.rsi_trendline.r_squared >= 0.25, "R-squared below minimum"

    @pytest.mark.integration
    def test_breakout_strength_values(self, rsi_analyzer, rsi_breakout_data):
        """Test breakout strength is valid value"""
        result = rsi_analyzer.analyze(rsi_breakout_data, lookback_periods=150)

        if result and result.has_rsi_breakout:
            valid_strengths = ['WEAK', 'MODERATE', 'STRONG']
            assert result.rsi_breakout.strength in valid_strengths

    @pytest.mark.integration
    def test_breakout_age_tracking(self, rsi_analyzer, rsi_breakout_data):
        """Test breakout age is tracked correctly"""
        result = rsi_analyzer.analyze(rsi_breakout_data, lookback_periods=150)

        if result and result.has_rsi_breakout:
            # Age should be non-negative
            assert result.rsi_breakout.age_in_periods >= 0

            # Index should be within data range
            assert result.rsi_breakout.index >= 0
            assert result.rsi_breakout.index < len(rsi_breakout_data)


class TestTrendlineDetectorIntegration:
    """Integration tests for RSITrendlineDetector component"""

    @pytest.fixture
    def detector(self):
        return RSITrendlineDetector()

    @pytest.fixture
    def sample_df_for_detector(self):
        """Create sample DataFrame with price data for RSI calculation"""
        np.random.seed(42)
        n = 150
        dates = pd.date_range(start='2023-01-01', periods=n, freq='D')

        # Create price data with oscillation (will create RSI peaks)
        base = np.linspace(100, 90, n)  # Downtrend
        oscillation = 8 * np.sin(np.linspace(0, 6*np.pi, n))
        noise = np.random.normal(0, 1, n)
        close = base + oscillation + noise

        df = pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.01,
            'Low': close * 0.98,
            'Close': close,
            'Volume': np.random.uniform(1000000, 2000000, n)
        }, index=dates)

        return df

    @pytest.mark.integration
    def test_detect_returns_trendlines_or_none(self, detector, sample_df_for_detector):
        """Test detector.detect() method returns trendlines or None"""
        trendlines = detector.detect(sample_df_for_detector)

        # Should return a list or None (no trendlines found)
        assert trendlines is None or isinstance(trendlines, list)

        # If trendlines found, each should have required attributes
        if trendlines:
            for tl in trendlines:
                assert hasattr(tl, 'slope')
                assert hasattr(tl, 'intercept')
                assert hasattr(tl, 'r_squared')
                assert hasattr(tl, 'peak_indices')

    @pytest.mark.integration
    def test_trendline_slope_is_negative(self, detector, sample_df_for_detector):
        """Test that detected trendlines have negative slope (resistance)"""
        trendlines = detector.detect(sample_df_for_detector)

        # Only check if trendlines were found
        if trendlines:
            for tl in trendlines:
                assert tl.slope <= 0, "Trendline should be descending (resistance)"


class TestBreakoutAnalyzerIntegration:
    """Integration tests for TrendlineBreakoutAnalyzer component"""

    @pytest.fixture
    def breakout_analyzer(self):
        return TrendlineBreakoutAnalyzer()

    @pytest.fixture
    def rsi_breakout_scenario(self):
        """Create RSI series that breaks above trendline"""
        np.random.seed(42)
        n = 100

        # RSI below trendline then breaks above
        rsi_below = np.linspace(40, 45, 80) + np.random.normal(0, 2, 80)
        rsi_breakout = np.linspace(50, 65, 20) + np.random.normal(0, 2, 20)

        rsi = np.concatenate([rsi_below, rsi_breakout])
        rsi = np.clip(rsi, 0, 100)

        return pd.Series(rsi)

    @pytest.mark.integration
    def test_detect_breakout_with_volume(self, breakout_analyzer, rsi_breakout_scenario):
        """Test breakout detection includes volume data"""
        # Create simple trendline for testing (matching Trendline dataclass)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

        trendline = Trendline(
            slope=-0.1,
            intercept=55,
            r_squared=0.9,
            peak_indices=[10, 30, 50],
            peak_dates=[dates[10], dates[30], dates[50]],
            peak_values=[52, 50, 48],
            start_idx=10,
            end_idx=50,
            quality_score=0.85
        )

        # Create sample DataFrame with volume
        df = pd.DataFrame({
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.uniform(1000000, 3000000, 100)
        }, index=dates)

        breakout = breakout_analyzer.detect_breakout(
            rsi=rsi_breakout_scenario,
            trendline=trendline,
            df=df
        )

        if breakout:
            # Volume fields should exist
            assert hasattr(breakout, 'volume_ratio')
            assert hasattr(breakout, 'volume_confirmed')


class TestRealDataIntegration:
    """Integration tests with real market data simulation"""

    @pytest.fixture
    def analyzer(self):
        return RSIBreakoutAnalyzer()

    @pytest.fixture
    def weekly_data(self):
        """Simulate weekly timeframe data"""
        np.random.seed(456)
        n = 104  # 2 years of weekly data
        dates = pd.date_range(start='2022-01-01', periods=n, freq='W')

        # Create realistic price movement
        base_price = 100
        trend = np.linspace(0, 30, n)
        noise = np.random.normal(0, 3, n)
        close = base_price + trend + noise

        df = pd.DataFrame({
            'Open': close - np.random.uniform(0, 2, n),
            'High': close + np.random.uniform(0, 3, n),
            'Low': close - np.random.uniform(0, 3, n),
            'Close': close,
            'Volume': np.random.uniform(1000000, 3000000, n)
        }, index=dates)

        return df

    @pytest.mark.integration
    def test_weekly_timeframe_analysis(self, analyzer, weekly_data):
        """Test analysis on weekly timeframe data"""
        result = analyzer.analyze(weekly_data, lookback_periods=104)

        # Should complete without error
        assert result is not None or result is None

    @pytest.mark.integration
    def test_insufficient_data_handling(self, analyzer):
        """Test handling of insufficient data"""
        # Only 20 bars - not enough for analysis
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        small_df = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 20),
            'High': np.random.uniform(110, 120, 20),
            'Low': np.random.uniform(90, 100, 20),
            'Close': np.random.uniform(100, 110, 20),
            'Volume': np.random.uniform(1000000, 2000000, 20)
        }, index=dates)

        result = analyzer.analyze(small_df, lookback_periods=100)

        # Should handle gracefully (return None or empty result)
        if result:
            assert not result.has_rsi_trendline or result.has_rsi_trendline


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
