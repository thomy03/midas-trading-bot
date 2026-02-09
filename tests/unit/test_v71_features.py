"""Unit tests for V7.1 features: improved exits, momentum, breadth, adaptive params, regime ML."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta


# ── Helpers ──

def _make_ohlcv(n=300, start_price=100.0, trend=0.0005):
    """Create a synthetic OHLCV DataFrame."""
    dates = pd.bdate_range(start='2020-01-01', periods=n)
    np.random.seed(42)
    prices = [start_price]
    for i in range(1, n):
        change = np.random.normal(trend, 0.015)
        prices.append(prices[-1] * (1 + change))

    close = np.array(prices)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volume = np.random.randint(1_000_000, 10_000_000, n).astype(float)

    df = pd.DataFrame({
        'Open': close * (1 + np.random.normal(0, 0.002, n)),
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    return df


def _make_position(entry_price=100.0, days_ago=30, trailing_atr=3.0):
    """Create a BacktestPosition for testing."""
    from src.backtesting.v6_backtester import BacktestPosition
    entry_date = pd.Timestamp('2020-06-01')
    return BacktestPosition(
        symbol='TEST',
        entry_date=entry_date,
        entry_price=entry_price,
        effective_entry=entry_price * 1.0015,
        shares=100,
        stop_loss=entry_price * 0.92,
        highest_price=entry_price * 1.10,
        confidence_score=60.0,
        regime_at_entry='BULL',
        trailing_atr=trailing_atr,
        sector='Technology',
        last_score_check_day=0
    )


# ── Phase 1.1: Profit Target Exit ──

class TestProfitTargetExit:
    """Test adaptive profit target exit."""

    def test_config_has_profit_target_fields(self):
        from src.backtesting.v6_backtester import V6BacktestConfig
        config = V6BacktestConfig()
        assert hasattr(config, 'profit_target_enabled')
        assert hasattr(config, 'profit_target_pct')
        assert config.profit_target_pct == 0.18

    def test_exit_on_profit_target_reached(self):
        from src.backtesting.v6_backtester import V6Backtester, V6BacktestConfig, MarketRegime
        config = V6BacktestConfig(profit_target_enabled=True, profit_target_pct=0.15)
        bt = V6Backtester(config)
        pos = _make_position(entry_price=100.0)

        # Price at +20% -> should trigger profit target
        should_exit, reason = bt._check_exit(
            pos, current_price=120.0, current_low=118.0,
            date=pd.Timestamp('2020-07-15'), regime=MarketRegime.BULL
        )
        assert should_exit
        assert reason == 'profit_target'

    def test_no_exit_below_profit_target(self):
        from src.backtesting.v6_backtester import V6Backtester, V6BacktestConfig, MarketRegime
        config = V6BacktestConfig(
            profit_target_enabled=True, profit_target_pct=0.18,
            regime_tightening_enabled=False, score_exit_enabled=False
        )
        bt = V6Backtester(config)
        pos = _make_position(entry_price=100.0)
        pos.highest_price = 110.0

        # Price at +10% -> below 18% target
        should_exit, reason = bt._check_exit(
            pos, current_price=110.0, current_low=109.0,
            date=pd.Timestamp('2020-07-01'), regime=MarketRegime.BULL
        )
        assert not should_exit

    def test_profit_target_disabled(self):
        from src.backtesting.v6_backtester import V6Backtester, V6BacktestConfig, MarketRegime
        config = V6BacktestConfig(
            profit_target_enabled=False,
            regime_tightening_enabled=False, score_exit_enabled=False
        )
        bt = V6Backtester(config)
        pos = _make_position(entry_price=100.0)
        pos.highest_price = 130.0

        # Price at +25% but profit target disabled
        should_exit, reason = bt._check_exit(
            pos, current_price=125.0, current_low=124.0,
            date=pd.Timestamp('2020-07-15'), regime=MarketRegime.BULL
        )
        assert not should_exit


# ── Phase 1.2: Score-Based Exit ──

class TestScoreBasedExit:
    """Test technical score degradation exit."""

    def test_config_has_score_exit_fields(self):
        from src.backtesting.v6_backtester import V6BacktestConfig
        config = V6BacktestConfig()
        assert hasattr(config, 'score_exit_enabled')
        assert hasattr(config, 'score_exit_threshold')
        assert config.score_exit_threshold == 35.0

    def test_exit_on_low_tech_score(self):
        from src.backtesting.v6_backtester import (
            V6Backtester, V6BacktestConfig, MarketRegime, BacktestTechnicalScorer
        )
        config = V6BacktestConfig(
            score_exit_enabled=True, score_exit_threshold=35.0,
            score_exit_check_interval=5,
            profit_target_enabled=False, regime_tightening_enabled=False,
            regime_adaptive_enabled=False
        )
        bt = V6Backtester(config)

        # Mock the tech scorer to return a low score
        bt.tech_scorer.score_at = MagicMock(return_value=25.0)

        pos = _make_position(entry_price=100.0, trailing_atr=0.0)
        pos.stop_loss = 80.0  # Very wide stop so it won't trigger
        pos.highest_price = 102.0  # Not far from entry
        pos.last_score_check_day = 0

        # Create symbol_data with the date present
        date = pd.Timestamp('2020-07-06')  # 35 days after entry
        df = _make_ohlcv()
        if date not in df.index:
            date = df.index[df.index.get_indexer([date], method='nearest')[0]]

        symbol_data = {'TEST': df}

        should_exit, reason = bt._check_exit(
            pos, current_price=98.0, current_low=97.0,
            date=date, regime=MarketRegime.RANGE,
            symbol_data=symbol_data
        )
        assert should_exit
        assert reason == 'score_degradation'

    def test_score_check_respects_interval(self):
        from src.backtesting.v6_backtester import V6Backtester, V6BacktestConfig, MarketRegime
        config = V6BacktestConfig(
            score_exit_enabled=True, score_exit_threshold=35.0,
            score_exit_check_interval=5,
            profit_target_enabled=False, regime_tightening_enabled=False,
            regime_adaptive_enabled=False
        )
        bt = V6Backtester(config)
        bt.tech_scorer.score_at = MagicMock(return_value=25.0)

        pos = _make_position(entry_price=100.0, trailing_atr=0.0)
        pos.stop_loss = 80.0  # Very wide stop
        pos.highest_price = 100.0
        pos.last_score_check_day = 2  # Last check at day 2

        # At day 3 (only 1 day since last check) -> should NOT check
        date = pd.Timestamp('2020-06-04')
        df = _make_ohlcv()
        if date not in df.index:
            date = df.index[df.index.get_indexer([date], method='nearest')[0]]

        should_exit, reason = bt._check_exit(
            pos, current_price=98.0, current_low=97.0,
            date=date, regime=MarketRegime.RANGE,
            symbol_data={'TEST': df}
        )
        # Hold_days = 3 - last_check=2 = 1 < interval=5, so no check
        assert not should_exit


# ── Phase 1.3: Regime Tightening ──

class TestRegimeTightening:
    """Test regime-change trailing stop tightening."""

    def test_config_has_regime_tightening_fields(self):
        from src.backtesting.v6_backtester import V6BacktestConfig
        config = V6BacktestConfig()
        assert hasattr(config, 'regime_tightening_enabled')
        assert hasattr(config, 'regime_tight_atr_multiplier')
        assert config.regime_tight_atr_multiplier == 2.0

    def test_tighter_stop_in_bear_regime(self):
        from src.backtesting.v6_backtester import V6Backtester, V6BacktestConfig, MarketRegime
        config = V6BacktestConfig(
            regime_tightening_enabled=True,
            regime_tight_atr_multiplier=2.0,
            trailing_atr_multiplier=3.5,
            profit_target_enabled=False, score_exit_enabled=False,
            regime_adaptive_enabled=False
        )
        bt = V6Backtester(config)

        pos = _make_position(entry_price=100.0, trailing_atr=3.0)
        pos.highest_price = 110.0  # highest reached

        # With ATR=3.0, tight mult=2.0: trailing_stop = 110 - 6 = 104
        # With ATR=3.0, normal mult=3.5: trailing_stop = 110 - 10.5 = 99.5
        # Price at 103 -> low enough to trigger tight stop but not normal

        should_exit_bear, reason_bear = bt._check_exit(
            pos, current_price=103.5, current_low=103.5,
            date=pd.Timestamp('2020-07-15'), regime=MarketRegime.BEAR
        )
        assert should_exit_bear
        assert reason_bear == 'trailing_stop'

        # Same price in BULL regime -> should NOT trigger (normal 3.5x)
        config_bull = V6BacktestConfig(
            regime_tightening_enabled=True,
            regime_tight_atr_multiplier=2.0,
            trailing_atr_multiplier=3.5,
            profit_target_enabled=False, score_exit_enabled=False,
            regime_adaptive_enabled=False
        )
        bt_bull = V6Backtester(config_bull)
        pos2 = _make_position(entry_price=100.0, trailing_atr=3.0)
        pos2.highest_price = 110.0

        should_exit_bull, _ = bt_bull._check_exit(
            pos2, current_price=103.5, current_low=103.5,
            date=pd.Timestamp('2020-07-15'), regime=MarketRegime.BULL
        )
        assert not should_exit_bull


# ── Phase 2.1: Momentum Scoring ──

class TestMomentumScoring:
    """Test cross-sectional momentum scoring."""

    def test_config_has_momentum_fields(self):
        from src.backtesting.v6_backtester import V6BacktestConfig
        config = V6BacktestConfig()
        assert hasattr(config, 'momentum_scoring_enabled')
        assert hasattr(config, 'momentum_lookback_months')
        assert config.momentum_bonus_max == 10.0
        assert config.momentum_penalty_max == 5.0

    def test_momentum_scores_computed(self):
        from src.backtesting.v6_backtester import V6Backtester, V6BacktestConfig
        config = V6BacktestConfig(momentum_scoring_enabled=True, momentum_lookback_months=3)
        bt = V6Backtester(config)

        # Create 10 symbols with different trends
        symbol_data = {}
        for i in range(10):
            trend = 0.001 * (i - 5)  # -0.005 to +0.004
            df = _make_ohlcv(n=200, trend=trend)
            symbol_data[f'SYM{i}'] = df

        date = list(symbol_data.values())[0].index[150]
        scores = bt._compute_momentum_scores(symbol_data, date)

        assert len(scores) > 0
        # Top performers should have positive bonus
        # Bottom performers should have negative penalty
        all_scores = list(scores.values())
        assert max(all_scores) > 0
        assert min(all_scores) < 0

    def test_momentum_disabled_returns_empty(self):
        from src.backtesting.v6_backtester import V6Backtester, V6BacktestConfig
        config = V6BacktestConfig(momentum_scoring_enabled=False)
        bt = V6Backtester(config)
        scores = bt._compute_momentum_scores({}, pd.Timestamp('2020-06-01'))
        assert scores == {}


# ── Phase 2.2: Market Breadth Filter ──

class TestBreadthFilter:
    """Test market breadth filter."""

    def test_config_has_breadth_fields(self):
        from src.backtesting.v6_backtester import V6BacktestConfig
        config = V6BacktestConfig()
        assert hasattr(config, 'breadth_filter_enabled')
        assert hasattr(config, 'breadth_bearish_threshold')
        assert config.breadth_bearish_threshold == 0.30

    def test_breadth_computation(self):
        from src.backtesting.v6_backtester import V6Backtester, V6BacktestConfig
        config = V6BacktestConfig(breadth_filter_enabled=True)
        bt = V6Backtester(config)

        # Create symbols, half trending up, half trending down
        symbol_data = {}
        for i in range(10):
            trend = 0.002 if i < 5 else -0.002
            df = _make_ohlcv(n=200, trend=trend)
            symbol_data[f'SYM{i}'] = df
            bt.tech_scorer.precompute(f'SYM{i}', df)

        date = list(symbol_data.values())[0].index[100]
        breadth = bt._compute_breadth(symbol_data, date)

        # Should be around 0.5 (half above EMA50, half below)
        assert 0.1 <= breadth <= 0.9


# ── Phase 4.2: Regime-Adaptive Thresholds ──

class TestRegimeAdaptiveThresholds:
    """Test regime-adaptive buy thresholds and ATR multipliers."""

    def test_config_has_adaptive_fields(self):
        from src.backtesting.v6_backtester import V6BacktestConfig
        config = V6BacktestConfig()
        assert hasattr(config, 'regime_adaptive_enabled')
        assert hasattr(config, 'regime_buy_thresholds')
        assert hasattr(config, 'regime_atr_multipliers')
        assert config.regime_buy_thresholds['BULL'] == 53.0
        assert config.regime_buy_thresholds['BEAR'] == 60.0
        assert config.regime_atr_multipliers['BULL'] == 3.5
        assert config.regime_atr_multipliers['BEAR'] == 2.5

    def test_bull_threshold_lower(self):
        from src.backtesting.v6_backtester import V6BacktestConfig
        config = V6BacktestConfig(regime_adaptive_enabled=True)
        # In BULL, threshold should be 53 (more permissive)
        assert config.regime_buy_thresholds['BULL'] < config.regime_buy_thresholds['BEAR']

    def test_bear_atr_tighter(self):
        from src.backtesting.v6_backtester import V6BacktestConfig
        config = V6BacktestConfig(regime_adaptive_enabled=True)
        # In BEAR, ATR multiplier should be tighter (smaller)
        assert config.regime_atr_multipliers['BEAR'] < config.regime_atr_multipliers['BULL']


# ── Phase 4.3: Volatility-Scaled Sizing ──

class TestVolScaledSizing:
    """Test volatility-scaled position sizing."""

    def test_config_has_vol_scaling_fields(self):
        from src.backtesting.v6_backtester import V6BacktestConfig
        config = V6BacktestConfig()
        assert hasattr(config, 'vol_scaling_enabled')
        assert hasattr(config, 'vol_target')
        assert config.vol_target == 0.15
        assert config.vol_scaling_min == 0.5
        assert config.vol_scaling_max == 1.5

    def test_vol_scaling_factor_computation(self):
        from src.backtesting.v6_backtester import V6Backtester, V6BacktestConfig
        config = V6BacktestConfig(vol_scaling_enabled=True, vol_target=0.15)
        bt = V6Backtester(config)

        # Create SPY data with known volatility
        spy_df = _make_ohlcv(n=200, trend=0.0003)
        # Put it in cache
        bt.data_mgr._cache['SPY_test'] = spy_df

        date = spy_df.index[150]
        factor = bt._get_vol_scaling_factor({'SPY_test': spy_df}, date)

        # Factor should be between min and max
        assert config.vol_scaling_min <= factor <= config.vol_scaling_max

    def test_vol_scaling_disabled_returns_1(self):
        from src.backtesting.v6_backtester import V6Backtester, V6BacktestConfig
        config = V6BacktestConfig(vol_scaling_enabled=False)
        bt = V6Backtester(config)
        factor = bt._get_vol_scaling_factor({}, pd.Timestamp('2020-06-01'))
        assert factor == 1.0


# ── Phase 3.1: Regime ML Predictor ──

def _try_import_sklearn():
    try:
        import sklearn
        return True
    except ImportError:
        return False


class TestRegimeML:
    """Test regime ML predictor."""

    def test_feature_extractor_basic(self):
        from src.ml.regime_ml import RegimeMLFeatureExtractor, REGIME_FEATURE_NAMES
        extractor = RegimeMLFeatureExtractor()

        spy_df = _make_ohlcv(n=300)
        vix_df = _make_ohlcv(n=300, start_price=20.0, trend=0.0)

        features = extractor.extract(
            spy_df, vix_df, {}, date_loc=150, spy_date=spy_df.index[150]
        )

        assert features is not None
        # All feature names should be present
        for name in REGIME_FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"

    def test_feature_extractor_insufficient_data(self):
        from src.ml.regime_ml import RegimeMLFeatureExtractor
        extractor = RegimeMLFeatureExtractor()

        spy_df = _make_ohlcv(n=100)
        features = extractor.extract(spy_df, None, {}, date_loc=30)
        assert features is None  # date_loc < 60

    def test_predictor_inactive_by_default(self):
        from src.ml.regime_ml import RegimeMLPredictor
        predictor = RegimeMLPredictor()
        assert not predictor.is_active
        prob, valid = predictor.predict(
            _make_ohlcv(), None, {}, date_loc=100
        )
        assert prob == 0.5
        assert not valid

    def test_feature_names_consistent(self):
        from src.ml.regime_ml import REGIME_FEATURE_NAMES
        assert len(REGIME_FEATURE_NAMES) == 17
        assert 'vix_level' in REGIME_FEATURE_NAMES
        assert 'breadth_ema50' in REGIME_FEATURE_NAMES
        assert 'spy_drawdown' in REGIME_FEATURE_NAMES

    def test_predictor_config_defaults(self):
        from src.ml.regime_ml import RegimeMLConfig
        config = RegimeMLConfig()
        assert config.forward_days == 20
        assert config.min_auc == 0.55
        assert config.max_depth == 5
        assert config.min_samples_leaf == 50

    @pytest.mark.skipif(
        not _try_import_sklearn(),
        reason="scikit-learn not available"
    )
    def test_train_with_synthetic_data(self):
        from src.ml.regime_ml import RegimeMLPredictor, RegimeMLConfig
        config = RegimeMLConfig(min_auc=0.0, n_estimators=10, max_depth=3)
        predictor = RegimeMLPredictor(config)

        spy_df = _make_ohlcv(n=500, trend=0.0005)
        vix_df = _make_ohlcv(n=500, start_price=20.0, trend=0.0)

        metrics = predictor.train(spy_df, vix_df, {})
        assert 'auc' in metrics
        assert 'accuracy' in metrics
        # With min_auc=0, model should be active
        assert predictor.is_active


# ── Integration: BacktestPosition has new field ──

class TestBacktestPositionV71:
    """Test BacktestPosition has V7.1 fields."""

    def test_last_score_check_day(self):
        from src.backtesting.v6_backtester import BacktestPosition
        pos = BacktestPosition(
            symbol='TEST',
            entry_date=pd.Timestamp('2020-01-01'),
            entry_price=100.0,
            effective_entry=100.15,
            shares=100,
            stop_loss=92.0,
            highest_price=100.0,
            confidence_score=60.0,
            regime_at_entry='BULL'
        )
        assert pos.last_score_check_day == 0


# ── Integration: All V7.1 config fields have defaults ──

class TestV71ConfigDefaults:
    """Test that all V7.1 config fields have sensible defaults."""

    def test_all_v71_fields_present(self):
        from src.backtesting.v6_backtester import V6BacktestConfig
        config = V6BacktestConfig()

        # Phase 1
        assert config.profit_target_enabled is True
        assert config.profit_target_pct == 0.18
        assert config.score_exit_enabled is True
        assert config.score_exit_threshold == 35.0
        assert config.score_exit_check_interval == 5
        assert config.regime_tightening_enabled is True
        assert config.regime_tight_atr_multiplier == 2.0

        # Phase 2
        assert config.momentum_scoring_enabled is True
        assert config.momentum_lookback_months == 6
        assert config.momentum_bonus_max == 10.0
        assert config.momentum_penalty_max == 5.0
        assert config.breadth_filter_enabled is True
        assert config.breadth_bearish_threshold == 0.30
        assert config.breadth_sizing_reduction == 0.50

        # Phase 4
        assert config.regime_adaptive_enabled is True
        assert config.vol_scaling_enabled is True
        assert config.vol_target == 0.15

    def test_v71_features_can_be_disabled(self):
        from src.backtesting.v6_backtester import V6BacktestConfig
        config = V6BacktestConfig(
            profit_target_enabled=False,
            score_exit_enabled=False,
            regime_tightening_enabled=False,
            momentum_scoring_enabled=False,
            breadth_filter_enabled=False,
            regime_adaptive_enabled=False,
            vol_scaling_enabled=False
        )
        assert not config.profit_target_enabled
        assert not config.score_exit_enabled
        assert not config.regime_tightening_enabled
        assert not config.momentum_scoring_enabled
        assert not config.breadth_filter_enabled
        assert not config.regime_adaptive_enabled
        assert not config.vol_scaling_enabled
