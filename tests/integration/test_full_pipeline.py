"""Integration tests for V7 full pipeline."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


class TestBacktesterModes:
    """Test that all backtester modes work."""

    def test_scoring_mode_e_exists(self):
        from src.backtesting.v6_backtester import ScoringMode
        assert hasattr(ScoringMode, 'ML_TRAINED')
        assert ScoringMode.ML_TRAINED.value == 'E'

    def test_all_scoring_modes(self):
        from src.backtesting.v6_backtester import ScoringMode
        assert ScoringMode.TECH_ONLY.value == 'A'
        assert ScoringMode.THREE_PILLARS.value == 'B'
        assert ScoringMode.THREE_PILLARS_VIX.value == 'C'
        assert ScoringMode.FULL_LIVE.value == 'D'
        assert ScoringMode.ML_TRAINED.value == 'E'

    def test_config_has_v7_fields(self):
        from src.backtesting.v6_backtester import V6BacktestConfig
        config = V6BacktestConfig()
        assert hasattr(config, 'dynamic_sizing')
        assert hasattr(config, 'defensive_mode')
        assert hasattr(config, 'correlation_checks')
        assert hasattr(config, 'fundamental_bias_penalty')
        assert hasattr(config, 'ml_model_path')

    def test_fundamental_bias_penalty(self):
        from src.backtesting.v6_backtester import BacktestFundamentalScorer
        # Without penalty
        scorer_full = BacktestFundamentalScorer(bias_penalty=1.0)
        # With 30% penalty
        scorer_penalized = BacktestFundamentalScorer(bias_penalty=0.7)

        info = {'trailingPE': 10, 'sector': 'Technology'}
        score_full = scorer_full.score(info)
        score_penalized = scorer_penalized.score(info)

        # Penalized should be closer to 50 (neutral)
        assert abs(score_penalized - 50) < abs(score_full - 50)

    def test_fundamental_empty_info_returns_neutral(self):
        from src.backtesting.v6_backtester import BacktestFundamentalScorer
        scorer = BacktestFundamentalScorer()
        assert scorer.score({}) == 50.0
        assert scorer.score(None) == 50.0


class TestVIXProxy:
    """Test enhanced VIX sentiment proxy."""

    def test_basic_levels(self):
        from src.backtesting.v6_backtester import BacktestVIXSentimentProxy
        proxy = BacktestVIXSentimentProxy()

        # Extreme panic -> bullish contrarian
        assert proxy.score(45) > 70
        # Normal -> neutral
        assert 45 < proxy.score(18) < 55
        # Extreme complacency -> bearish
        assert proxy.score(10) < 40

    def test_vix_spike_bonus(self):
        from src.backtesting.v6_backtester import BacktestVIXSentimentProxy
        proxy = BacktestVIXSentimentProxy()

        # VIX spike from 20 to 30 (50% increase) -> contrarian buy bonus
        score_spike = proxy.score(30, vix_5d_ago=20)
        score_no_spike = proxy.score(30)
        assert score_spike > score_no_spike

    def test_vix_collapse_penalty(self):
        from src.backtesting.v6_backtester import BacktestVIXSentimentProxy
        proxy = BacktestVIXSentimentProxy()

        # VIX collapse -> complacency
        score_collapse = proxy.score(14, vix_5d_ago=20)
        score_normal = proxy.score(14)
        assert score_collapse < score_normal

    def test_vix_levels_monotonic_contrarian(self):
        """Higher VIX should generally yield higher (more bullish) scores."""
        from src.backtesting.v6_backtester import BacktestVIXSentimentProxy
        proxy = BacktestVIXSentimentProxy()
        score_low = proxy.score(12)
        score_high = proxy.score(40)
        assert score_high > score_low

    def test_vix_trend_adjustment(self):
        """VIX that was high but is now falling signals recovery."""
        from src.backtesting.v6_backtester import BacktestVIXSentimentProxy
        proxy = BacktestVIXSentimentProxy()
        # VIX at 32, was 38 twenty days ago (falling from high)
        score_falling = proxy.score(32, vix_20d_ago=38)
        score_no_trend = proxy.score(32)
        assert score_falling >= score_no_trend


class TestPositionSizerIntegration:
    """Test position sizer in backtest context."""

    def test_sizer_creation(self):
        from src.execution.position_sizer import PositionSizer
        ps = PositionSizer()
        assert ps is not None
        assert ps.base_size_pct == 0.10

    def test_sizing_in_bear_market(self):
        from src.execution.position_sizer import PositionSizer
        ps = PositionSizer(use_kelly=False)

        bull_result = ps.calculate_size(100000, 100, regime='BULL')
        bear_result = ps.calculate_size(100000, 100, regime='BEAR')

        assert bear_result.adjusted_size_pct < bull_result.adjusted_size_pct

    def test_sizing_with_all_features(self):
        """Test sizing with all adjustments enabled."""
        from src.execution.position_sizer import PositionSizer
        ps = PositionSizer(use_kelly=False)

        result = ps.calculate_size(
            capital=100000,
            price=150,
            atr_pct=0.03,
            confidence_score=85,
            regime='BULL'
        )
        assert 0.03 <= result.adjusted_size_pct <= 0.20
        assert result.method == 'fixed'

    def test_capacity_limiting(self):
        """Position size should not exceed remaining capacity."""
        from src.execution.position_sizer import PositionSizer
        ps = PositionSizer(use_kelly=False, use_vol_adjust=False, use_confidence_scale=False)

        result = ps.calculate_size(
            capital=100000,
            price=100,
            regime='BULL',
            defensive_max_invested=0.50,
            current_invested_pct=0.48
        )
        # Only 2% capacity left (use approx for floating point)
        assert result.adjusted_size_pct <= 0.02 + 1e-10


class TestDefensiveInBacktest:
    """Test defensive mode parameters in backtest config."""

    def test_config_defensive_mode(self):
        from src.backtesting.v6_backtester import V6BacktestConfig
        config = V6BacktestConfig(
            defensive_mode=True,
            dynamic_sizing=True,
            correlation_checks=True,
            fundamental_bias_penalty=0.7
        )
        assert config.defensive_mode is True
        assert config.dynamic_sizing is True
        assert config.correlation_checks is True
        assert config.fundamental_bias_penalty == 0.7

    def test_config_defaults_v7_disabled(self):
        """V7 features are disabled by default for backward compatibility."""
        from src.backtesting.v6_backtester import V6BacktestConfig
        config = V6BacktestConfig()
        assert config.defensive_mode is False
        assert config.dynamic_sizing is False
        assert config.correlation_checks is False

    def test_regime_stops(self):
        from src.backtesting.v6_backtester import V6BacktestConfig
        config = V6BacktestConfig()
        assert 'BULL' in config.regime_stops
        assert 'BEAR' in config.regime_stops
        assert config.regime_stops['BEAR'] < config.regime_stops['BULL']

    def test_transaction_costs(self):
        from src.backtesting.v6_backtester import TransactionCosts
        costs = TransactionCosts()
        assert costs.entry_cost_factor() > 1.0
        assert costs.exit_cost_factor() < 1.0


class TestMonteCarloIntegration:
    """Test Monte Carlo simulation."""

    def test_monte_carlo_basic(self):
        from src.backtesting.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(n_simulations=100)

        # Generate some trades with clear positive expectation
        np.random.seed(42)
        pnls = list(np.random.randn(50) * 200 + 500)  # Clearly positive mean

        result = sim.run(pnls, total_days=1260)
        assert result.n_simulations == 100
        assert result.cagr_mean > -1  # Reasonable CAGR range

    def test_monte_carlo_empty(self):
        from src.backtesting.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(n_simulations=10)
        result = sim.run([])
        assert result.n_simulations == 0

    def test_monte_carlo_report(self):
        from src.backtesting.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(n_simulations=50)
        pnls = [100, 200, -50, 300, -100, 150, -30, 250]
        result = sim.run(pnls, total_days=252)
        assert len(result.report) > 0
        assert 'MONTE CARLO' in result.report

    def test_monte_carlo_all_winning(self):
        from src.backtesting.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(n_simulations=50)
        pnls = [500] * 20
        result = sim.run(pnls, total_days=252)
        assert result.prob_profitable == 1.0
        assert result.cagr_mean > 0

    def test_monte_carlo_all_losing(self):
        from src.backtesting.monte_carlo import MonteCarloSimulator
        sim = MonteCarloSimulator(n_simulations=50)
        pnls = [-500] * 20
        result = sim.run(pnls, total_days=252)
        assert result.prob_profitable == 0.0
        assert result.cagr_mean < 0


class TestRegimeDetection:
    """Test regime detection component."""

    def test_market_regime_enum(self):
        from src.backtesting.v6_backtester import MarketRegime
        assert MarketRegime.BULL.value == 'BULL'
        assert MarketRegime.BEAR.value == 'BEAR'
        assert MarketRegime.RANGE.value == 'RANGE'
        assert MarketRegime.VOLATILE.value == 'VOLATILE'

    def test_regime_weights_exist(self):
        from src.backtesting.v6_backtester import V6BacktestConfig
        config = V6BacktestConfig()
        for regime in ['BULL', 'BEAR', 'RANGE', 'VOLATILE']:
            assert regime in config.regime_weights
            weights = config.regime_weights[regime]
            assert 'technical' in weights
            assert 'fundamental' in weights
            assert 'ml' in weights


class TestMLGate:
    """Test ML Gate in backtester."""

    def test_ml_gate_boost(self):
        from src.backtesting.v6_backtester import BacktestMLGate, V6BacktestConfig
        config = V6BacktestConfig()
        gate = BacktestMLGate(config)
        score, mode = gate.apply(base_score=70, ml_score=65, volatility=0.02)
        assert mode == 'ML_BOOST'
        assert score > 70

    def test_ml_gate_block(self):
        from src.backtesting.v6_backtester import BacktestMLGate, V6BacktestConfig
        config = V6BacktestConfig()
        gate = BacktestMLGate(config)
        score, mode = gate.apply(base_score=70, ml_score=35, volatility=0.02)
        assert mode == 'ML_BLOCK'
        assert score == 0.0

    def test_ml_gate_high_vol_passthrough(self):
        from src.backtesting.v6_backtester import BacktestMLGate, V6BacktestConfig
        config = V6BacktestConfig()
        gate = BacktestMLGate(config)
        score, mode = gate.apply(base_score=70, ml_score=80, volatility=0.05)
        assert mode == '5P_ONLY'
        assert score == 70

    def test_ml_gate_disabled(self):
        from src.backtesting.v6_backtester import BacktestMLGate, V6BacktestConfig
        config = V6BacktestConfig(ml_gate_enabled=False)
        gate = BacktestMLGate(config)
        score, mode = gate.apply(base_score=70, ml_score=80, volatility=0.02)
        assert mode == 'DISABLED'
        assert score == 70


class TestCrossComponentIntegration:
    """Test that V7 components work together."""

    def test_defensive_plus_sizer(self):
        """DefensiveManager informs PositionSizer regime multiplier."""
        from src.execution.defensive_manager import DefensiveManager
        from src.execution.position_sizer import PositionSizer

        dm = DefensiveManager()
        level = dm.get_defensive_level('BEAR', vix_level=25)
        config = dm.get_config()

        ps = PositionSizer(use_kelly=False, use_vol_adjust=False, use_confidence_scale=False)
        result = ps.calculate_size(capital=100000, price=100, regime='BEAR')

        # Both systems should reduce exposure in BEAR
        assert config.max_invested_pct < 1.0
        assert result.defensive_multiplier < 1.0

    def test_correlation_plus_defensive(self):
        """Both systems independently filter trades."""
        from src.execution.defensive_manager import DefensiveManager
        from src.execution.correlation_manager import (
            CorrelationManager, CorrelationConfig, PortfolioPosition
        )

        dm = DefensiveManager()
        dm.get_defensive_level('BEAR', vix_level=25)

        cm = CorrelationManager(CorrelationConfig(max_sector_pct=0.25))
        positions = [
            PortfolioPosition('AAPL', 'Technology', 25000),
        ]
        cm._sector_cache['MSFT'] = 'Technology'

        # DefensiveManager blocks low score
        allowed_dm, _ = dm.should_enter_trade(score=80)
        assert allowed_dm is False

        # CorrelationManager blocks concentrated sector
        allowed_cm, _ = cm.check_new_position(
            'MSFT', positions, 100000, new_position_value=6000
        )
        assert allowed_cm is False
