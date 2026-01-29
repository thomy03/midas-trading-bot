"""
Integration tests for Confidence Scoring System

Tests the complete scoring pipeline:
- EMA alignment scoring
- Support proximity scoring
- RSI breakout quality scoring
- Freshness scoring
- Volume confirmation scoring
- Signal classification
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils.confidence_scorer import ConfidenceScorer, ConfidenceScore, confidence_scorer


class TestConfidenceScorerIntegration:
    """Integration tests for ConfidenceScorer class"""

    @pytest.fixture
    def scorer(self):
        """Fresh scorer instance"""
        return ConfidenceScorer()

    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame with price and volume data"""
        np.random.seed(42)
        n = 100
        dates = pd.date_range(start='2023-01-01', periods=n, freq='D')

        close = np.linspace(100, 130, n) + np.random.normal(0, 2, n)

        df = pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.02,
            'Low': close * 0.98,
            'Close': close,
            'Volume': np.random.uniform(1000000, 2000000, n)
        }, index=dates)

        return df

    @pytest.mark.integration
    def test_complete_score_calculation(self, scorer, sample_df):
        """Test complete score calculation with all components"""
        # Bullish EMAs
        ema_24 = 125.0
        ema_38 = 120.0
        ema_62 = 115.0

        current_price = 127.0
        support_level = 123.0

        rsi_breakout = {
            'strength': 'STRONG',
            'age_in_periods': 1,
            'index': 95,
            'volume_ratio': 1.8
        }

        rsi_trendline = {
            'r_squared': 0.85
        }

        result = scorer.calculate_score(
            df=sample_df,
            ema_24=ema_24,
            ema_38=ema_38,
            ema_62=ema_62,
            current_price=current_price,
            support_level=support_level,
            rsi_breakout=rsi_breakout,
            rsi_trendline=rsi_trendline
        )

        # Verify result type
        assert isinstance(result, ConfidenceScore)

        # Verify total score is sum of components
        expected_total = (
            result.ema_score +
            result.support_score +
            result.rsi_score +
            result.freshness_score +
            result.volume_score
        )
        assert abs(result.total_score - expected_total) < 0.5

        # Verify score is in valid range
        assert 0 <= result.total_score <= 100

        # With strong conditions, should be STRONG_BUY
        assert result.signal in ['STRONG_BUY', 'BUY']

    @pytest.mark.integration
    def test_ema_alignment_scoring(self, scorer, sample_df):
        """Test EMA alignment component scoring"""
        # Perfect alignment: 24 > 38 > 62
        result = scorer.calculate_score(
            df=sample_df,
            ema_24=130,
            ema_38=125,
            ema_62=120,
            current_price=135,
            support_level=125,
            rsi_breakout=None,
            rsi_trendline=None
        )

        # Should get max EMA points (20)
        assert result.ema_score == 20

        # No alignment: 62 > 38 > 24 (bearish)
        result_bearish = scorer.calculate_score(
            df=sample_df,
            ema_24=110,
            ema_38=120,
            ema_62=130,
            current_price=105,
            support_level=100,
            rsi_breakout=None,
            rsi_trendline=None
        )

        # Should get 0 EMA points
        assert result_bearish.ema_score == 0

    @pytest.mark.integration
    def test_support_proximity_scoring(self, scorer, sample_df):
        """Test support proximity component scoring"""
        # Price at support (0% distance)
        result_at_support = scorer.calculate_score(
            df=sample_df,
            ema_24=100,
            ema_38=100,
            ema_62=100,
            current_price=100,
            support_level=100,  # Same as price
            rsi_breakout=None,
            rsi_trendline=None
        )

        # Should get max support points
        assert result_at_support.support_score == 20

        # Price far from support (8%+)
        result_far = scorer.calculate_score(
            df=sample_df,
            ema_24=100,
            ema_38=100,
            ema_62=100,
            current_price=110,
            support_level=100,  # 10% away
            rsi_breakout=None,
            rsi_trendline=None
        )

        # Should get low or zero support points
        assert result_far.support_score <= 5

    @pytest.mark.integration
    def test_rsi_breakout_quality_scoring(self, scorer, sample_df):
        """Test RSI breakout quality component scoring"""
        # Strong breakout with high R²
        result_strong = scorer.calculate_score(
            df=sample_df,
            ema_24=100,
            ema_38=100,
            ema_62=100,
            current_price=100,
            support_level=100,
            rsi_breakout={'strength': 'STRONG', 'age_in_periods': 1, 'index': 95, 'volume_ratio': 1.5},
            rsi_trendline={'r_squared': 0.90}
        )

        # Weak breakout with low R²
        result_weak = scorer.calculate_score(
            df=sample_df,
            ema_24=100,
            ema_38=100,
            ema_62=100,
            current_price=100,
            support_level=100,
            rsi_breakout={'strength': 'WEAK', 'age_in_periods': 5, 'index': 90, 'volume_ratio': 0.8},
            rsi_trendline={'r_squared': 0.30}
        )

        # Strong should score higher than weak
        assert result_strong.rsi_score > result_weak.rsi_score

    @pytest.mark.integration
    def test_freshness_scoring(self, scorer, sample_df):
        """Test freshness/recency component scoring"""
        # Fresh breakout (age 0-1)
        result_fresh = scorer.calculate_score(
            df=sample_df,
            ema_24=100,
            ema_38=100,
            ema_62=100,
            current_price=100,
            support_level=100,
            rsi_breakout={'strength': 'MODERATE', 'age_in_periods': 0, 'index': 99, 'volume_ratio': 1.0},
            rsi_trendline={'r_squared': 0.70}
        )

        # Stale breakout (age 6+)
        result_stale = scorer.calculate_score(
            df=sample_df,
            ema_24=100,
            ema_38=100,
            ema_62=100,
            current_price=100,
            support_level=100,
            rsi_breakout={'strength': 'MODERATE', 'age_in_periods': 10, 'index': 90, 'volume_ratio': 1.0},
            rsi_trendline={'r_squared': 0.70}
        )

        # Fresh should score 20 points, stale should score 0
        assert result_fresh.freshness_score == 20
        assert result_stale.freshness_score == 0

    @pytest.mark.integration
    def test_volume_confirmation_scoring(self, scorer, sample_df):
        """Test volume confirmation component scoring"""
        # High volume ratio (2x+)
        result_high_vol = scorer.calculate_score(
            df=sample_df,
            ema_24=100,
            ema_38=100,
            ema_62=100,
            current_price=100,
            support_level=100,
            rsi_breakout={'strength': 'MODERATE', 'age_in_periods': 2, 'index': 95, 'volume_ratio': 2.5},
            rsi_trendline={'r_squared': 0.70}
        )

        # Low volume ratio (0.5x)
        result_low_vol = scorer.calculate_score(
            df=sample_df,
            ema_24=100,
            ema_38=100,
            ema_62=100,
            current_price=100,
            support_level=100,
            rsi_breakout={'strength': 'MODERATE', 'age_in_periods': 2, 'index': 95, 'volume_ratio': 0.5},
            rsi_trendline={'r_squared': 0.70}
        )

        # High volume should score 15, low volume should score 0
        assert result_high_vol.volume_score == 15
        assert result_low_vol.volume_score == 0

    @pytest.mark.integration
    def test_signal_classification_thresholds(self, scorer, sample_df):
        """Test signal classification based on thresholds"""
        # STRONG_BUY (75+)
        result_strong_buy = scorer.calculate_score(
            df=sample_df,
            ema_24=130, ema_38=125, ema_62=120,  # Bullish EMAs (+20)
            current_price=125,
            support_level=124,  # Very close (+~18)
            rsi_breakout={'strength': 'STRONG', 'age_in_periods': 1, 'index': 98, 'volume_ratio': 2.0},  # Strong (+13), fresh (+20), vol (+15)
            rsi_trendline={'r_squared': 0.90}  # Good R² (+10.8)
        )
        assert result_strong_buy.signal == 'STRONG_BUY'
        assert result_strong_buy.total_score >= 75

        # OBSERVE (< 35)
        result_observe = scorer.calculate_score(
            df=sample_df,
            ema_24=100, ema_38=110, ema_62=120,  # Bearish EMAs (+0)
            current_price=90,
            support_level=100,  # Far (+0)
            rsi_breakout=None,  # No breakout
            rsi_trendline=None  # No trendline
        )
        assert result_observe.signal == 'OBSERVE'
        assert result_observe.total_score < 35


class TestConfidenceScorerEdgeCases:
    """Edge case tests for ConfidenceScorer"""

    @pytest.fixture
    def scorer(self):
        return ConfidenceScorer()

    @pytest.fixture
    def empty_df(self):
        return pd.DataFrame()

    @pytest.fixture
    def minimal_df(self):
        """Minimal valid DataFrame"""
        return pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000000, 1100000, 1200000]
        })

    @pytest.mark.integration
    def test_no_rsi_data(self, scorer, minimal_df):
        """Test scoring without RSI breakout data"""
        result = scorer.calculate_score(
            df=minimal_df,
            ema_24=100,
            ema_38=100,
            ema_62=100,
            current_price=100,
            support_level=100,
            rsi_breakout=None,
            rsi_trendline=None
        )

        # Should still return valid score
        assert isinstance(result, ConfidenceScore)
        assert result.rsi_score == 0  # No RSI breakout
        assert result.freshness_score == 5  # Neutral (no breakout yet)

    @pytest.mark.integration
    def test_zero_support_level(self, scorer, minimal_df):
        """Test with zero support level"""
        result = scorer.calculate_score(
            df=minimal_df,
            ema_24=100,
            ema_38=100,
            ema_62=100,
            current_price=100,
            support_level=0,  # Invalid
            rsi_breakout=None,
            rsi_trendline=None
        )

        assert result.support_score == 0

    @pytest.mark.integration
    def test_missing_volume_in_breakout(self, scorer, minimal_df):
        """Test breakout without volume_ratio field"""
        rsi_breakout = {
            'strength': 'MODERATE',
            'age_in_periods': 2,
            'index': 2
            # No volume_ratio
        }

        result = scorer.calculate_score(
            df=minimal_df,
            ema_24=100,
            ema_38=100,
            ema_62=100,
            current_price=100,
            support_level=100,
            rsi_breakout=rsi_breakout,
            rsi_trendline=None
        )

        # Should handle gracefully (fallback to DataFrame volume or neutral)
        assert isinstance(result, ConfidenceScore)


class TestVolumeConfirmationIntegration:
    """Tests for volume confirmation across the pipeline"""

    @pytest.fixture
    def df_with_volume_spike(self):
        """DataFrame with volume spike at end"""
        np.random.seed(42)
        n = 50
        dates = pd.date_range(start='2023-01-01', periods=n, freq='D')

        close = np.linspace(100, 120, n)

        # Normal volume, then spike at end
        volume = np.concatenate([
            np.random.uniform(1000000, 1500000, n-5),  # Normal
            np.random.uniform(3000000, 4000000, 5)      # Spike
        ])

        df = pd.DataFrame({
            'Close': close,
            'High': close * 1.01,
            'Low': close * 0.99,
            'Open': close * 0.995,
            'Volume': volume
        }, index=dates)

        return df

    @pytest.mark.integration
    def test_volume_from_breakout_preferred(self):
        """Test that volume_ratio from breakout is used over calculation"""
        scorer = ConfidenceScorer()

        df = pd.DataFrame({
            'Close': [100] * 50,
            'Volume': [1000000] * 50  # Uniform volume
        })

        # Breakout says 2.0x volume
        rsi_breakout = {
            'strength': 'MODERATE',
            'age_in_periods': 1,
            'index': 45,
            'volume_ratio': 2.0  # This should be used
        }

        result = scorer.calculate_score(
            df=df,
            ema_24=100, ema_38=100, ema_62=100,
            current_price=100,
            support_level=100,
            rsi_breakout=rsi_breakout,
            rsi_trendline=None
        )

        # Should use 2.0x ratio -> 15 points
        assert result.volume_score == 15
        assert result.details['volume']['volume_ratio'] == 2.0
        assert result.details['volume']['source'] == 'breakout'


class TestConfidenceScoreDisplay:
    """Tests for score display formatting"""

    @pytest.mark.integration
    def test_format_score_display(self):
        """Test format_score_display method"""
        scorer = ConfidenceScorer()

        score = ConfidenceScore(
            total_score=87.5,
            ema_score=20,
            support_score=17,
            rsi_score=22,
            freshness_score=16,
            volume_score=12.5,
            signal='STRONG_BUY',
            details={}
        )

        display = scorer.format_score_display(score)

        assert 'STRONG_BUY' in display
        assert '87' in display or '88' in display  # Rounded


class TestSingletonInstance:
    """Tests for singleton confidence_scorer instance"""

    @pytest.mark.integration
    def test_singleton_exists(self):
        """Test singleton instance is available"""
        assert confidence_scorer is not None
        assert isinstance(confidence_scorer, ConfidenceScorer)

    @pytest.mark.integration
    def test_singleton_thresholds(self):
        """Test singleton has correct thresholds"""
        assert confidence_scorer.THRESHOLDS['STRONG_BUY'] == 75
        assert confidence_scorer.THRESHOLDS['BUY'] == 55
        assert confidence_scorer.THRESHOLDS['WATCH'] == 35
        assert confidence_scorer.THRESHOLDS['OBSERVE'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration'])
