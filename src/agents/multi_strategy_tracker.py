"""
Multi-Strategy Tracker V8.1

Evaluates every signal against 4 strategy profiles simultaneously.
Each profile has its own virtual portfolio for A/B testing.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

INITIAL_CAPITAL = 15000.0
DATA_FILE = os.environ.get("MULTI_STRATEGY_FILE", "data/multi_strategy_state.json")


@dataclass
class VirtualPosition:
    symbol: str
    shares: int
    entry_price: float
    entry_date: str
    stop_loss: float
    take_profit: float
    score_at_entry: float
    strategy_id: str
    current_price: float = 0.0
    pnl_pct: float = 0.0

    def update_price(self, price: float):
        self.current_price = price
        if self.entry_price > 0:
            self.pnl_pct = ((price - self.entry_price) / self.entry_price) * 100

    def should_stop_loss(self, price: float) -> bool:
        return price <= self.stop_loss

    def should_take_profit(self, price: float) -> bool:
        return price >= self.take_profit

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'VirtualPosition':
        return cls(**data)


@dataclass
class StrategyState:
    strategy_id: str
    name: str
    color: str
    initial_capital: float = INITIAL_CAPITAL
    cash: float = INITIAL_CAPITAL
    positions: List[VirtualPosition] = field(default_factory=list)
    closed_trades: List[dict] = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    equity_curve: List[dict] = field(default_factory=list)  # [{date, equity}]
    signals_evaluated: int = 0
    signals_accepted: int = 0
    signals_rejected: int = 0
    max_drawdown: float = 0.0
    peak_equity: float = INITIAL_CAPITAL

    @property
    def total_equity(self) -> float:
        pos_value = sum(p.current_price * p.shares for p in self.positions)
        return self.cash + pos_value

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def total_return_pct(self) -> float:
        return ((self.total_equity - self.initial_capital) / self.initial_capital) * 100

    def record_equity(self):
        equity = self.total_equity
        self.equity_curve.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "equity": round(equity, 2)
        })
        # Track drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = ((self.peak_equity - equity) / self.peak_equity) * 100
        if dd > self.max_drawdown:
            self.max_drawdown = dd
        # Keep last 500 points
        if len(self.equity_curve) > 500:
            self.equity_curve = self.equity_curve[-500:]

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "color": self.color,
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions": [p.to_dict() for p in self.positions],
            "closed_trades": self.closed_trades[-100:],  # Keep last 100
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "equity_curve": self.equity_curve,
            "signals_evaluated": self.signals_evaluated,
            "signals_accepted": self.signals_accepted,
            "signals_rejected": self.signals_rejected,
            "max_drawdown": self.max_drawdown,
            "peak_equity": self.peak_equity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'StrategyState':
        positions = [VirtualPosition.from_dict(p) for p in data.get("positions", [])]
        state = cls(
            strategy_id=data["strategy_id"],
            name=data["name"],
            color=data.get("color", "#888"),
            initial_capital=data.get("initial_capital", INITIAL_CAPITAL),
            cash=data.get("cash", INITIAL_CAPITAL),
            positions=positions,
            closed_trades=data.get("closed_trades", []),
            total_trades=data.get("total_trades", 0),
            winning_trades=data.get("winning_trades", 0),
            losing_trades=data.get("losing_trades", 0),
            total_pnl=data.get("total_pnl", 0.0),
            equity_curve=data.get("equity_curve", []),
            signals_evaluated=data.get("signals_evaluated", 0),
            signals_accepted=data.get("signals_accepted", 0),
            signals_rejected=data.get("signals_rejected", 0),
            max_drawdown=data.get("max_drawdown", 0.0),
            peak_equity=data.get("peak_equity", INITIAL_CAPITAL),
        )
        return state


class MultiStrategyTracker:
    """
    Tracks 4 virtual portfolios in parallel.
    Each signal is evaluated against all 4 strategies.
    """

    def __init__(self, data_file: str = DATA_FILE):
        self.data_file = data_file
        self.strategies: Dict[str, StrategyState] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                for sid, sdata in data.get("strategies", {}).items():
                    self.strategies[sid] = StrategyState.from_dict(sdata)
                logger.info(f"[MULTI] Loaded {len(self.strategies)} strategy states")
            except Exception as e:
                logger.error(f"[MULTI] Error loading state: {e}")
                self._init_strategies()
        else:
            self._init_strategies()

    def _init_strategies(self):
        from config.strategies import get_all_profiles
        for pid, profile in get_all_profiles().items():
            self.strategies[pid] = StrategyState(
                strategy_id=pid,
                name=profile.name,
                color=profile.color,
            )
        self._save()
        logger.info(f"[MULTI] Initialized {len(self.strategies)} strategies")

    def _save(self):
        os.makedirs(os.path.dirname(self.data_file) or ".", exist_ok=True)
        data = {
            "strategies": {sid: s.to_dict() for sid, s in self.strategies.items()},
            "updated_at": datetime.now().isoformat(),
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def evaluate_signal(
        self,
        symbol: str,
        total_score: float,
        ml_score: Optional[float],
        pillar_scores: Dict[str, float],
        current_price: float,
        atr: float,
    ) -> Dict[str, str]:
        """
        Evaluate a signal against all 4 strategies.
        Returns {strategy_id: 'accepted'|'rejected'} with reasons.
        """
        from config.strategies import get_all_profiles
        profiles = get_all_profiles()
        results = {}

        for sid, profile in profiles.items():
            state = self.strategies.get(sid)
            if not state:
                continue

            state.signals_evaluated += 1

            # Check max positions
            if len(state.positions) >= profile.max_positions:
                state.signals_rejected += 1
                results[sid] = "rejected:max_positions"
                continue

            # Check if already holding this symbol
            if any(p.symbol == symbol for p in state.positions):
                state.signals_rejected += 1
                results[sid] = "rejected:already_holding"
                continue

            # Recalculate score with strategy-specific weights
            weighted_score = 0
            for pillar, weight in profile.pillar_weights.items():
                weighted_score += pillar_scores.get(pillar, 0) * weight

            # Apply ML Gate if enabled
            if profile.use_ml_gate and ml_score is not None:
                if ml_score < profile.ml_min_score:
                    state.signals_rejected += 1
                    results[sid] = f"rejected:ml_gate({ml_score:.0f}<{profile.ml_min_score})"
                    continue
                if ml_score > 60:
                    weighted_score += profile.ml_boost

            # Check min score
            if weighted_score < profile.min_score:
                state.signals_rejected += 1
                results[sid] = f"rejected:score({weighted_score:.1f}<{profile.min_score})"
                continue

            # ACCEPTED - open virtual position
            position_pct = profile.get_position_size(weighted_score)
            position_value = state.cash * (position_pct / 100)
            if position_value < 100 or position_value > state.cash:
                state.signals_rejected += 1
                results[sid] = "rejected:insufficient_cash"
                continue

            shares = int(position_value / current_price)
            if shares <= 0:
                state.signals_rejected += 1
                results[sid] = "rejected:shares=0"
                continue

            cost = shares * current_price
            sl = current_price - (atr * profile.stop_loss_atr_mult)
            tp = current_price + (atr * profile.take_profit_atr_mult)

            position = VirtualPosition(
                symbol=symbol,
                shares=shares,
                entry_price=current_price,
                entry_date=datetime.now().isoformat(),
                stop_loss=sl,
                take_profit=tp,
                score_at_entry=weighted_score,
                strategy_id=sid,
                current_price=current_price,
            )

            state.positions.append(position)
            state.cash -= cost
            state.signals_accepted += 1
            results[sid] = f"accepted:score={weighted_score:.1f},shares={shares}"
            logger.info(f"[MULTI] {profile.name} BOUGHT {symbol} x{shares} @{current_price:.2f} (score={weighted_score:.1f})")

        self._save()
        return results

    def update_prices(self, prices: Dict[str, float]):
        """Update all positions with current prices and check SL/TP."""
        from config.strategies import get_all_profiles
        profiles = get_all_profiles()

        for sid, state in self.strategies.items():
            profile = profiles.get(sid)
            closed = []

            for pos in state.positions:
                price = prices.get(pos.symbol)
                if price is None:
                    continue

                pos.update_price(price)

                # Check stop loss
                if pos.should_stop_loss(price):
                    pnl = (price - pos.entry_price) * pos.shares
                    pnl_pct = ((price - pos.entry_price) / pos.entry_price) * 100
                    state.cash += price * pos.shares
                    state.total_trades += 1
                    state.total_pnl += pnl
                    if pnl > 0:
                        state.winning_trades += 1
                    else:
                        state.losing_trades += 1
                    state.closed_trades.append({
                        "symbol": pos.symbol,
                        "entry_price": pos.entry_price,
                        "exit_price": price,
                        "shares": pos.shares,
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "reason": "stop_loss",
                        "entry_date": pos.entry_date,
                        "exit_date": datetime.now().isoformat(),
                    })
                    closed.append(pos)
                    logger.info(f"[MULTI] {state.name} SL {pos.symbol} pnl={pnl:.2f}")

                # Check take profit
                elif pos.should_take_profit(price):
                    pnl = (price - pos.entry_price) * pos.shares
                    pnl_pct = ((price - pos.entry_price) / pos.entry_price) * 100
                    state.cash += price * pos.shares
                    state.total_trades += 1
                    state.total_pnl += pnl
                    state.winning_trades += 1
                    state.closed_trades.append({
                        "symbol": pos.symbol,
                        "entry_price": pos.entry_price,
                        "exit_price": price,
                        "shares": pos.shares,
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "reason": "take_profit",
                        "entry_date": pos.entry_date,
                        "exit_date": datetime.now().isoformat(),
                    })
                    closed.append(pos)
                    logger.info(f"[MULTI] {state.name} TP {pos.symbol} pnl={pnl:.2f}")

            for c in closed:
                state.positions.remove(c)

            state.record_equity()

        self._save()

    def get_comparison(self) -> Dict[str, Any]:
        """Get comparison data for all 4 strategies."""
        result = []
        for sid, state in self.strategies.items():
            result.append({
                "id": sid,
                "name": state.name,
                "color": state.color,
                "equity": round(state.total_equity, 2),
                "cash": round(state.cash, 2),
                "return_pct": round(state.total_return_pct, 2),
                "total_trades": state.total_trades,
                "winning_trades": state.winning_trades,
                "losing_trades": state.losing_trades,
                "win_rate": round(state.win_rate * 100, 1),
                "total_pnl": round(state.total_pnl, 2),
                "max_drawdown": round(state.max_drawdown, 2),
                "open_positions": len(state.positions),
                "signals_evaluated": state.signals_evaluated,
                "signals_accepted": state.signals_accepted,
                "signals_rejected": state.signals_rejected,
                "positions": [p.to_dict() for p in state.positions],
                "equity_curve": state.equity_curve[-100:],
                "recent_trades": state.closed_trades[-20:],
            })
        
        # Sort by return
        result.sort(key=lambda x: x["return_pct"], reverse=True)
        return {"strategies": result, "updated_at": datetime.now().isoformat()}

    def reset(self):
        """Reset all strategies to initial state."""
        self._init_strategies()
        logger.info("[MULTI] All strategies reset")


# Singleton
_tracker: Optional[MultiStrategyTracker] = None

def get_multi_tracker() -> MultiStrategyTracker:
    global _tracker
    if _tracker is None:
        _tracker = MultiStrategyTracker()
    return _tracker
