"""Tests for ML Training Pipeline - V7."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


class TestFeatureExtractor:
    """Test feature extraction."""

    def _make_df(self, n=200):
        """Create synthetic OHLCV DataFrame."""
        dates = pd.bdate_range('2020-01-01', periods=n)
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 10)  # Ensure positive
        return pd.DataFrame({
            'Open': close * (1 + np.random.randn(n) * 0.001),
            'High': close * (1 + abs(np.random.randn(n) * 0.01)),
            'Low': close * (1 - abs(np.random.randn(n) * 0.01)),
            'Close': close,
            'Volume': np.random.randint(100000, 10000000, n).astype(float)
        }, index=dates)

    def test_extract_features_returns_dict(self):
        from src.ml.training_pipeline import FeatureExtractor
        fe = FeatureExtractor()
        df = self._make_df(200)
        features = fe.extract_at(df, 150)
        assert features is not None
        assert isinstance(features, dict)

    def test_extract_features_count(self):
        from src.ml.training_pipeline import FeatureExtractor, FEATURE_NAMES
        fe = FeatureExtractor()
        df = self._make_df(200)
        features = fe.extract_at(df, 150)
        # Should have all 40 features
        for fname in FEATURE_NAMES:
            assert fname in features, f"Missing feature: {fname}"

    def test_extract_too_early_returns_none(self):
        from src.ml.training_pipeline import FeatureExtractor
        fe = FeatureExtractor()
        df = self._make_df(200)
        features = fe.extract_at(df, 50)  # < 100 minimum
        assert features is None

    def test_no_nan_in_features(self):
        from src.ml.training_pipeline import FeatureExtractor
        fe = FeatureExtractor()
        df = self._make_df(300)
        features = fe.extract_at(df, 250)
        assert features is not None
        for k, v in features.items():
            assert not np.isnan(v), f"NaN in feature {k}"
            assert not np.isinf(v), f"Inf in feature {k}"

    def test_extract_at_boundary(self):
        """Extract at the last valid index."""
        from src.ml.training_pipeline import FeatureExtractor
        fe = FeatureExtractor()
        df = self._make_df(200)
        features = fe.extract_at(df, 199)
        assert features is not None

    def test_extract_out_of_bounds(self):
        """Extract beyond DataFrame length returns None."""
        from src.ml.training_pipeline import FeatureExtractor
        fe = FeatureExtractor()
        df = self._make_df(200)
        features = fe.extract_at(df, 200)
        assert features is None

    def test_feature_names_list_length(self):
        from src.ml.training_pipeline import FEATURE_NAMES
        assert len(FEATURE_NAMES) == 40


class TestWalkForwardTrainer:
    """Test walk-forward training logic."""

    def test_prepare_features_shape(self):
        from src.ml.walk_forward_trainer import WalkForwardTrainer, FEATURE_NAMES
        trainer = WalkForwardTrainer()

        # Create mock training data
        n = 100
        data = {f: np.random.randn(n) for f in FEATURE_NAMES}
        data['label'] = np.random.randint(0, 2, n)
        data['symbol'] = ['AAPL'] * n
        data['date'] = pd.date_range('2020-01-01', periods=n)
        data['regime'] = ['BULL'] * n
        data['forward_return'] = np.random.randn(n) * 0.05
        df = pd.DataFrame(data)

        X, y = trainer.prepare_features(df)
        assert X.shape == (n, len(FEATURE_NAMES))
        assert y.shape == (n,)
        assert set(y).issubset({0, 1})

    def test_prepare_features_replaces_nan(self):
        from src.ml.walk_forward_trainer import WalkForwardTrainer, FEATURE_NAMES
        trainer = WalkForwardTrainer()

        n = 50
        data = {f: np.random.randn(n) for f in FEATURE_NAMES}
        # Introduce some NaN values
        data[FEATURE_NAMES[0]][0] = np.nan
        data[FEATURE_NAMES[5]][10] = np.inf
        data['label'] = np.random.randint(0, 2, n)
        df = pd.DataFrame(data)

        X, y = trainer.prepare_features(df)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))

    def test_prepare_features_missing_columns(self):
        """If some feature columns are missing, only available ones are used."""
        from src.ml.walk_forward_trainer import WalkForwardTrainer, FEATURE_NAMES
        trainer = WalkForwardTrainer()

        n = 30
        # Only include half the features
        partial_features = FEATURE_NAMES[:20]
        data = {f: np.random.randn(n) for f in partial_features}
        data['label'] = np.random.randint(0, 2, n)
        df = pd.DataFrame(data)

        X, y = trainer.prepare_features(df)
        assert X.shape == (n, 20)
        assert y.shape == (n,)


class TestTrainingConfig:
    """Test training configuration defaults."""

    def test_default_config(self):
        from src.ml.training_pipeline import TrainingConfig
        config = TrainingConfig()
        assert config.train_start == '2015-01-01'
        assert config.train_end == '2022-01-01'
        assert config.min_data_points == 100
