"""
Strategy Registry - Manage active and archived strategies.

Handles:
- Active strategy slots (max 3)
- Strategy lifecycle (candidate -> active -> archived)
- Performance tracking per strategy
- Persistence to JSON
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio
from pathlib import Path


class StrategyStatus(str, Enum):
    """Strategy lifecycle status."""
    CANDIDATE = "candidate"      # Just created, needs validation
    VALIDATING = "validating"    # In sandbox (backtest/paper)
    ACTIVE = "active"            # In production
    PAUSED = "paused"            # Temporarily disabled
    ARCHIVED = "archived"        # No longer in use


@dataclass
class StrategyPerformance:
    """Track strategy performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    last_trade_date: Optional[str] = None
    consecutive_losses: int = 0
    best_trade_pct: float = 0.0
    worst_trade_pct: float = 0.0

    def update_from_trade(self, pnl: float, pnl_pct: float):
        """Update metrics after a trade."""
        self.total_trades += 1
        self.total_pnl += pnl
        self.total_pnl_pct += pnl_pct
        self.last_trade_date = datetime.now().isoformat()

        if pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
            self.avg_win = ((self.avg_win * (self.winning_trades - 1)) + pnl_pct) / self.winning_trades
            if pnl_pct > self.best_trade_pct:
                self.best_trade_pct = pnl_pct
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            if self.losing_trades > 0:
                self.avg_loss = ((self.avg_loss * (self.losing_trades - 1)) + abs(pnl_pct)) / self.losing_trades
            if pnl_pct < self.worst_trade_pct:
                self.worst_trade_pct = pnl_pct

        # Update derived metrics
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades

        total_wins = self.avg_win * self.winning_trades
        total_losses = self.avg_loss * self.losing_trades
        if total_losses > 0:
            self.profit_factor = total_wins / total_losses

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyPerformance':
        return cls(**data)


@dataclass
class RegisteredStrategy:
    """A strategy registered in the system."""
    id: str                                     # Unique ID
    name: str                                   # Human-readable name
    description: str                            # What the strategy does
    status: StrategyStatus                      # Current lifecycle status
    created_at: str                             # ISO timestamp
    updated_at: str                             # ISO timestamp
    created_by: str                             # "llm" or "manual"

    # Strategy definition
    entry_conditions: List[Dict[str, Any]] = field(default_factory=list)
    exit_conditions: List[Dict[str, Any]] = field(default_factory=list)
    indicators_used: List[str] = field(default_factory=list)
    timeframe: str = "1d"

    # Risk management
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    max_position_pct: float = 0.10

    # Validation results
    backtest_passed: bool = False
    backtest_metrics: Dict[str, Any] = field(default_factory=dict)
    paper_passed: bool = False
    paper_metrics: Dict[str, Any] = field(default_factory=dict)

    # Performance tracking
    performance: StrategyPerformance = field(default_factory=StrategyPerformance)

    # LLM reasoning
    creation_reasoning: str = ""
    archive_reason: str = ""

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['status'] = self.status.value
        data['performance'] = self.performance.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'RegisteredStrategy':
        data['status'] = StrategyStatus(data['status'])
        if 'performance' in data:
            data['performance'] = StrategyPerformance.from_dict(data['performance'])
        return cls(**data)


class StrategyRegistry:
    """
    Central registry for all strategies.

    Manages:
    - Max 3 active strategies at once
    - Strategy lifecycle transitions
    - Persistence to JSON
    - Performance tracking
    """

    MAX_ACTIVE_STRATEGIES = 3
    MAX_NEW_STRATEGIES_PER_WEEK = 2

    def __init__(self, data_dir: str = "data/strategies"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.strategies: Dict[str, RegisteredStrategy] = {}
        self.strategies_created_this_week: List[str] = []
        self.week_start: Optional[str] = None

        self._load()

    def _get_registry_path(self) -> Path:
        return self.data_dir / "registry.json"

    def _load(self):
        """Load registry from disk."""
        path = self._get_registry_path()
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for strategy_id, strategy_data in data.get('strategies', {}).items():
                    self.strategies[strategy_id] = RegisteredStrategy.from_dict(strategy_data)

                self.strategies_created_this_week = data.get('strategies_created_this_week', [])
                self.week_start = data.get('week_start')

                # Reset weekly counter if new week
                self._check_week_reset()

            except Exception as e:
                print(f"Error loading registry: {e}")

    def _save(self):
        """Save registry to disk."""
        path = self._get_registry_path()
        data = {
            'strategies': {sid: s.to_dict() for sid, s in self.strategies.items()},
            'strategies_created_this_week': self.strategies_created_this_week,
            'week_start': self.week_start,
            'updated_at': datetime.now().isoformat()
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _check_week_reset(self):
        """Reset weekly counter if it's a new week."""
        now = datetime.now()
        current_week = now.strftime('%Y-W%W')

        if self.week_start != current_week:
            self.week_start = current_week
            self.strategies_created_this_week = []
            self._save()

    def _generate_id(self, name: str) -> str:
        """Generate unique strategy ID."""
        base = name.lower().replace(' ', '_')[:20]
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"{base}_{timestamp}"

    # ========== Query Methods ==========

    def get_active_strategies(self) -> List[RegisteredStrategy]:
        """Get all active strategies."""
        return [s for s in self.strategies.values() if s.status == StrategyStatus.ACTIVE]

    def get_strategy(self, strategy_id: str) -> Optional[RegisteredStrategy]:
        """Get a strategy by ID."""
        return self.strategies.get(strategy_id)

    def get_strategies_by_status(self, status: StrategyStatus) -> List[RegisteredStrategy]:
        """Get all strategies with a given status."""
        return [s for s in self.strategies.values() if s.status == status]

    def get_best_performing_strategy(self) -> Optional[RegisteredStrategy]:
        """Get the best performing active strategy."""
        active = self.get_active_strategies()
        if not active:
            return None
        return max(active, key=lambda s: s.performance.total_pnl_pct)

    def get_worst_performing_strategy(self) -> Optional[RegisteredStrategy]:
        """Get the worst performing active strategy."""
        active = self.get_active_strategies()
        if not active:
            return None
        return min(active, key=lambda s: s.performance.total_pnl_pct)

    def can_create_new_strategy(self) -> tuple[bool, str]:
        """Check if we can create a new strategy this week."""
        self._check_week_reset()

        if len(self.strategies_created_this_week) >= self.MAX_NEW_STRATEGIES_PER_WEEK:
            return False, f"Weekly limit reached ({self.MAX_NEW_STRATEGIES_PER_WEEK} strategies/week)"

        return True, "OK"

    def has_active_slot(self) -> bool:
        """Check if there's room for another active strategy."""
        return len(self.get_active_strategies()) < self.MAX_ACTIVE_STRATEGIES

    # ========== Lifecycle Methods ==========

    def register_strategy(
        self,
        name: str,
        description: str,
        entry_conditions: List[Dict[str, Any]],
        exit_conditions: List[Dict[str, Any]],
        indicators_used: List[str],
        timeframe: str = "1d",
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.15,
        max_position_pct: float = 0.10,
        created_by: str = "llm",
        creation_reasoning: str = ""
    ) -> tuple[Optional[RegisteredStrategy], str]:
        """
        Register a new strategy as candidate.

        Returns:
            (strategy, message) - strategy is None if registration failed
        """
        # Check weekly limit
        can_create, reason = self.can_create_new_strategy()
        if not can_create:
            return None, reason

        now = datetime.now().isoformat()
        strategy_id = self._generate_id(name)

        strategy = RegisteredStrategy(
            id=strategy_id,
            name=name,
            description=description,
            status=StrategyStatus.CANDIDATE,
            created_at=now,
            updated_at=now,
            created_by=created_by,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            indicators_used=indicators_used,
            timeframe=timeframe,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_position_pct=max_position_pct,
            creation_reasoning=creation_reasoning
        )

        self.strategies[strategy_id] = strategy
        self.strategies_created_this_week.append(strategy_id)
        self._save()

        return strategy, f"Strategy '{name}' registered as candidate"

    def start_validation(self, strategy_id: str) -> tuple[bool, str]:
        """Move strategy to validating status."""
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            return False, f"Strategy {strategy_id} not found"

        if strategy.status != StrategyStatus.CANDIDATE:
            return False, f"Strategy must be CANDIDATE to validate (current: {strategy.status.value})"

        strategy.status = StrategyStatus.VALIDATING
        strategy.updated_at = datetime.now().isoformat()
        self._save()

        return True, f"Strategy '{strategy.name}' now validating"

    def update_backtest_results(
        self,
        strategy_id: str,
        passed: bool,
        metrics: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Update backtest results for a strategy."""
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            return False, f"Strategy {strategy_id} not found"

        strategy.backtest_passed = passed
        strategy.backtest_metrics = metrics
        strategy.updated_at = datetime.now().isoformat()
        self._save()

        status = "PASSED" if passed else "FAILED"
        return True, f"Backtest {status} for '{strategy.name}'"

    def update_paper_results(
        self,
        strategy_id: str,
        passed: bool,
        metrics: Dict[str, Any]
    ) -> tuple[bool, str]:
        """Update paper trading results for a strategy."""
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            return False, f"Strategy {strategy_id} not found"

        strategy.paper_passed = passed
        strategy.paper_metrics = metrics
        strategy.updated_at = datetime.now().isoformat()
        self._save()

        status = "PASSED" if passed else "FAILED"
        return True, f"Paper trading {status} for '{strategy.name}'"

    def activate_strategy(self, strategy_id: str) -> tuple[bool, str]:
        """
        Activate a validated strategy.

        Requirements:
        - Strategy must be in VALIDATING status
        - Backtest must have passed
        - Paper trading must have passed (or be skipped)
        - Must have available slot
        """
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            return False, f"Strategy {strategy_id} not found"

        if strategy.status != StrategyStatus.VALIDATING:
            return False, f"Strategy must be VALIDATING to activate (current: {strategy.status.value})"

        if not strategy.backtest_passed:
            return False, "Strategy must pass backtest before activation"

        if not self.has_active_slot():
            return False, f"No active slot available (max {self.MAX_ACTIVE_STRATEGIES})"

        strategy.status = StrategyStatus.ACTIVE
        strategy.updated_at = datetime.now().isoformat()
        self._save()

        return True, f"Strategy '{strategy.name}' is now ACTIVE"

    def pause_strategy(self, strategy_id: str, reason: str = "") -> tuple[bool, str]:
        """Temporarily pause an active strategy."""
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            return False, f"Strategy {strategy_id} not found"

        if strategy.status != StrategyStatus.ACTIVE:
            return False, f"Only ACTIVE strategies can be paused (current: {strategy.status.value})"

        strategy.status = StrategyStatus.PAUSED
        strategy.updated_at = datetime.now().isoformat()
        if reason:
            strategy.archive_reason = f"Paused: {reason}"
        self._save()

        return True, f"Strategy '{strategy.name}' is now PAUSED"

    def resume_strategy(self, strategy_id: str) -> tuple[bool, str]:
        """Resume a paused strategy."""
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            return False, f"Strategy {strategy_id} not found"

        if strategy.status != StrategyStatus.PAUSED:
            return False, f"Only PAUSED strategies can be resumed (current: {strategy.status.value})"

        if not self.has_active_slot():
            return False, f"No active slot available (max {self.MAX_ACTIVE_STRATEGIES})"

        strategy.status = StrategyStatus.ACTIVE
        strategy.updated_at = datetime.now().isoformat()
        self._save()

        return True, f"Strategy '{strategy.name}' is now ACTIVE again"

    def archive_strategy(self, strategy_id: str, reason: str) -> tuple[bool, str]:
        """Archive a strategy (permanently disable)."""
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            return False, f"Strategy {strategy_id} not found"

        if strategy.status == StrategyStatus.ARCHIVED:
            return False, "Strategy is already archived"

        strategy.status = StrategyStatus.ARCHIVED
        strategy.archive_reason = reason
        strategy.updated_at = datetime.now().isoformat()
        self._save()

        return True, f"Strategy '{strategy.name}' archived: {reason}"

    # ========== Performance Tracking ==========

    def record_trade(
        self,
        strategy_id: str,
        pnl: float,
        pnl_pct: float
    ) -> tuple[bool, str]:
        """Record a trade result for a strategy."""
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            return False, f"Strategy {strategy_id} not found"

        strategy.performance.update_from_trade(pnl, pnl_pct)
        strategy.updated_at = datetime.now().isoformat()
        self._save()

        return True, f"Trade recorded for '{strategy.name}'"

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all active strategies."""
        active = self.get_active_strategies()

        if not active:
            return {
                'active_count': 0,
                'total_pnl': 0.0,
                'total_trades': 0,
                'strategies': []
            }

        return {
            'active_count': len(active),
            'total_pnl': sum(s.performance.total_pnl for s in active),
            'total_pnl_pct': sum(s.performance.total_pnl_pct for s in active),
            'total_trades': sum(s.performance.total_trades for s in active),
            'avg_win_rate': sum(s.performance.win_rate for s in active) / len(active) if active else 0,
            'strategies': [
                {
                    'id': s.id,
                    'name': s.name,
                    'pnl': s.performance.total_pnl,
                    'pnl_pct': s.performance.total_pnl_pct,
                    'trades': s.performance.total_trades,
                    'win_rate': s.performance.win_rate
                }
                for s in active
            ]
        }

    # ========== Automatic Management ==========

    def check_and_archive_underperformers(
        self,
        min_trades: int = 5,
        max_consecutive_losses: int = 3,
        min_win_rate: float = 0.3
    ) -> List[str]:
        """
        Check active strategies and archive underperformers.

        Returns list of archived strategy IDs.
        """
        archived = []

        for strategy in self.get_active_strategies():
            perf = strategy.performance

            # Skip if not enough trades
            if perf.total_trades < min_trades:
                continue

            # Check consecutive losses
            if perf.consecutive_losses >= max_consecutive_losses:
                self.archive_strategy(
                    strategy.id,
                    f"Too many consecutive losses ({perf.consecutive_losses})"
                )
                archived.append(strategy.id)
                continue

            # Check win rate
            if perf.win_rate < min_win_rate:
                self.archive_strategy(
                    strategy.id,
                    f"Win rate too low ({perf.win_rate:.1%} < {min_win_rate:.1%})"
                )
                archived.append(strategy.id)
                continue

        return archived

    def get_strategies_needing_replacement(self) -> List[RegisteredStrategy]:
        """Get strategies that should be replaced (poor performance)."""
        result = []

        for strategy in self.get_active_strategies():
            perf = strategy.performance

            # Need at least some trades to evaluate
            if perf.total_trades < 3:
                continue

            # Flag if performing poorly
            if perf.win_rate < 0.4 or perf.total_pnl_pct < -5:
                result.append(strategy)

        return result


# Singleton instance
_registry_instance: Optional[StrategyRegistry] = None


def get_strategy_registry(data_dir: str = "data/strategies") -> StrategyRegistry:
    """Get the singleton StrategyRegistry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = StrategyRegistry(data_dir)
    return _registry_instance
