"""
V8.1 Strategy Profiles - 2 strategies per agent.

Each agent (LLM and No-LLM) runs both profiles independently.
- Aggressive: lower threshold, bigger positions, tighter stops
- Moderate: higher threshold, smaller positions, wider stops

ML Gate is always active.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class StrategyProfile:
    """A strategy profile with specific parameters."""
    id: str
    name: str
    description: str
    color: str

    # Aggressiveness
    min_score: float
    max_positions: int
    position_size_pct: float

    # Risk management
    stop_loss_atr_mult: float
    take_profit_atr_mult: float
    trailing_stop: bool

    # ML Gate (always on)
    use_ml_gate: bool = True
    ml_min_score: float = 40.0
    ml_boost: float = 5.0

    # Pillar weights
    pillar_weights: Dict[str, float] = field(default_factory=dict)

    def get_position_size(self, score: float) -> float:
        if score >= 90:
            return self.position_size_pct * 2.0
        elif score >= 85:
            return self.position_size_pct * 1.6
        elif score >= 80:
            return self.position_size_pct * 1.2
        return self.position_size_pct


STRATEGY_PROFILES: Dict[str, StrategyProfile] = {

    "aggressive": StrategyProfile(
        id="aggressive",
        name="Aggressive",
        description="Lower threshold, bigger positions, tighter stops. More trades, higher risk/reward.",
        color="#ef4444",
        min_score=70,
        max_positions=10,
        position_size_pct=8.0,
        stop_loss_atr_mult=1.5,
        take_profit_atr_mult=3.0,
        trailing_stop=True,
        use_ml_gate=True,
        ml_min_score=40,
        ml_boost=5.0,
        pillar_weights={"technical": 0.55, "fundamental": 0.45},
    ),

    "moderate": StrategyProfile(
        id="moderate",
        name="Moderate",
        description="Higher threshold, smaller positions, wider stops. Fewer trades, more selective.",
        color="#22c55e",
        min_score=78,
        max_positions=6,
        position_size_pct=5.0,
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        trailing_stop=True,
        use_ml_gate=True,
        ml_min_score=50,
        ml_boost=3.0,
        pillar_weights={"technical": 0.55, "fundamental": 0.45},
    ),
}


def get_all_profiles() -> Dict[str, StrategyProfile]:
    return STRATEGY_PROFILES


def get_profile(profile_id: str) -> Optional[StrategyProfile]:
    return STRATEGY_PROFILES.get(profile_id)
