"""Tests for DefensiveManager - V7 risk management."""
import pytest
from src.execution.defensive_manager import (
    DefensiveManager, DefensiveLevel, DEFENSIVE_CONFIGS
)


class TestDefensiveLevels:
    """Test defensive level detection."""

    def test_bull_regime_no_defense(self):
        dm = DefensiveManager()
        level = dm.get_defensive_level('BULL', vix_level=15)
        assert level == DefensiveLevel.NONE

    def test_bear_regime_defensive(self):
        dm = DefensiveManager()
        level = dm.get_defensive_level('BEAR', vix_level=22)
        assert level == DefensiveLevel.DEFENSIVE

    def test_volatile_regime_maximum(self):
        dm = DefensiveManager()
        level = dm.get_defensive_level('VOLATILE', vix_level=35)
        assert level == DefensiveLevel.MAXIMUM

    def test_range_with_high_vix_cautious(self):
        dm = DefensiveManager()
        level = dm.get_defensive_level('RANGE', vix_level=22)
        assert level == DefensiveLevel.CAUTIOUS

    def test_drawdown_escalation(self):
        dm = DefensiveManager()
        level = dm.get_defensive_level('BULL', vix_level=15, drawdown_pct=0.16)
        assert level == DefensiveLevel.MAXIMUM

    def test_high_vix_overrides_regime(self):
        """VIX > 35 forces MAXIMUM regardless of regime."""
        dm = DefensiveManager()
        level = dm.get_defensive_level('BULL', vix_level=36)
        assert level == DefensiveLevel.MAXIMUM

    def test_vix_above_25_forces_defensive(self):
        """VIX > 25 forces at least DEFENSIVE."""
        dm = DefensiveManager()
        level = dm.get_defensive_level('RANGE', vix_level=27)
        assert level == DefensiveLevel.DEFENSIVE

    def test_moderate_drawdown_escalation(self):
        """High drawdown in BEAR regime escalates to at least DEFENSIVE."""
        dm = DefensiveManager()
        level = dm.get_defensive_level('BEAR', vix_level=25, drawdown_pct=0.12)
        # BEAR + high VIX + drawdown -> at least DEFENSIVE
        assert level.value in ('defensive', 'maximum')


class TestShouldEnterTrade:
    """Test trade entry filtering."""

    def test_normal_mode_allows_trade(self):
        dm = DefensiveManager()
        dm.get_defensive_level('BULL', vix_level=15)
        allowed, reason = dm.should_enter_trade(score=80)
        assert allowed is True

    def test_defensive_mode_blocks_low_score(self):
        dm = DefensiveManager()
        dm.get_defensive_level('BEAR', vix_level=25)
        allowed, reason = dm.should_enter_trade(score=80)
        assert allowed is False  # Need 85+ in DEFENSIVE

    def test_maximum_mode_blocks_most(self):
        dm = DefensiveManager()
        dm.get_defensive_level('VOLATILE', vix_level=40)
        allowed, reason = dm.should_enter_trade(score=88)
        assert allowed is False  # Need 90+ in MAXIMUM

    def test_max_positions_respected(self):
        dm = DefensiveManager()
        dm.get_defensive_level('BEAR', vix_level=25)
        config = dm.get_config()
        allowed, reason = dm.should_enter_trade(
            score=90, current_positions=config.max_positions
        )
        assert allowed is False

    def test_max_invested_respected(self):
        dm = DefensiveManager()
        dm.get_defensive_level('BEAR', vix_level=25)
        allowed, reason = dm.should_enter_trade(
            score=90, total_invested_pct=0.35  # > 30% limit
        )
        assert allowed is False

    def test_high_score_passes_defensive(self):
        dm = DefensiveManager()
        dm.get_defensive_level('BEAR', vix_level=25)
        allowed, reason = dm.should_enter_trade(score=86)
        assert allowed is True

    def test_high_score_passes_maximum(self):
        dm = DefensiveManager()
        dm.get_defensive_level('VOLATILE', vix_level=40)
        allowed, reason = dm.should_enter_trade(score=92)
        assert allowed is True


class TestStopLoss:
    """Test stop loss adjustments."""

    def test_normal_stop(self):
        dm = DefensiveManager()
        dm.get_defensive_level('BULL', vix_level=15)
        stop = dm.get_adjusted_stop_loss(100.0, 0.08)
        assert stop == 92.0  # 8% stop

    def test_defensive_tighter_stop(self):
        dm = DefensiveManager()
        dm.get_defensive_level('BEAR', vix_level=25)
        stop = dm.get_adjusted_stop_loss(100.0, 0.08)
        assert stop == 96.0  # 4% stop (tighter)

    def test_maximum_tightest_stop(self):
        dm = DefensiveManager()
        dm.get_defensive_level('VOLATILE', vix_level=40)
        stop = dm.get_adjusted_stop_loss(100.0, 0.08)
        assert stop == 97.0  # 3% stop

    def test_already_tight_stop_unchanged(self):
        """If base stop is already tighter than config, use base."""
        dm = DefensiveManager()
        dm.get_defensive_level('BULL', vix_level=15)
        stop = dm.get_adjusted_stop_loss(100.0, 0.05)
        assert stop == 95.0  # 5% < 8% config, so use 5%


class TestConfigs:
    """Test defensive configurations are valid."""

    def test_all_levels_have_configs(self):
        for level in DefensiveLevel:
            assert level in DEFENSIVE_CONFIGS

    def test_max_invested_decreasing(self):
        none_cfg = DEFENSIVE_CONFIGS[DefensiveLevel.NONE]
        cautious_cfg = DEFENSIVE_CONFIGS[DefensiveLevel.CAUTIOUS]
        defensive_cfg = DEFENSIVE_CONFIGS[DefensiveLevel.DEFENSIVE]
        maximum_cfg = DEFENSIVE_CONFIGS[DefensiveLevel.MAXIMUM]
        assert none_cfg.max_invested_pct > cautious_cfg.max_invested_pct
        assert cautious_cfg.max_invested_pct > defensive_cfg.max_invested_pct
        assert defensive_cfg.max_invested_pct > maximum_cfg.max_invested_pct

    def test_score_threshold_increasing(self):
        none_cfg = DEFENSIVE_CONFIGS[DefensiveLevel.NONE]
        maximum_cfg = DEFENSIVE_CONFIGS[DefensiveLevel.MAXIMUM]
        assert maximum_cfg.min_score_threshold > none_cfg.min_score_threshold

    def test_stop_loss_tighter_at_higher_levels(self):
        none_cfg = DEFENSIVE_CONFIGS[DefensiveLevel.NONE]
        maximum_cfg = DEFENSIVE_CONFIGS[DefensiveLevel.MAXIMUM]
        assert maximum_cfg.stop_loss_pct < none_cfg.stop_loss_pct

    def test_max_positions_decreasing(self):
        none_cfg = DEFENSIVE_CONFIGS[DefensiveLevel.NONE]
        maximum_cfg = DEFENSIVE_CONFIGS[DefensiveLevel.MAXIMUM]
        assert maximum_cfg.max_positions < none_cfg.max_positions

    def test_inverse_etf_disabled_by_default(self):
        dm = DefensiveManager()
        config = dm.get_config()
        assert config.hedge_with_inverse_etf is False
        assert config.inverse_etf_allocation == 0.0


class TestSubscriberMessage:
    """Test subscriber-facing messages."""

    def test_normal_message(self):
        dm = DefensiveManager()
        dm.get_defensive_level('BULL', vix_level=15)
        msg = dm.get_subscriber_message()
        assert 'NORMAL' in msg

    def test_defensive_message(self):
        dm = DefensiveManager()
        dm.get_defensive_level('BEAR', vix_level=25)
        msg = dm.get_subscriber_message()
        assert 'DEFENSIVE' in msg

    def test_maximum_message(self):
        dm = DefensiveManager()
        dm.get_defensive_level('VOLATILE', vix_level=40)
        msg = dm.get_subscriber_message()
        assert 'MAXIMUM PROTECTION' in msg


class TestStats:
    """Test stats tracking."""

    def test_level_change_tracked(self):
        dm = DefensiveManager()
        dm.get_defensive_level('BULL', vix_level=15)
        dm.get_defensive_level('BEAR', vix_level=25)
        stats = dm.get_stats()
        assert stats['level_changes'] >= 1
        assert stats['current_level'] == 'defensive'
