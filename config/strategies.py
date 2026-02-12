"""
V8.1 Strategy Profiles - 4 parallel strategies for A/B testing.

2x Aggressiveness (aggressive / moderate)
2x ML modes (with ML Gate / without ML Gate)
= 4 strategies total

Scoring: Technical + Fundamental only (pillar weights)
ML acts as a GATE (pass/block), not a scoring pillar
Orchestrator (Grok+Gemini) is an overlay on LLM agent (+/- 15 pts)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class StrategyProfile:
    """A strategy profile with specific parameters."""
    id: str
    name: str
    description: str
    color: str  # For dashboard display
    
    # Aggressiveness
    min_score: float           # Minimum score to enter
    max_positions: int         # Max simultaneous positions
    position_size_pct: float   # Base position size (% of capital)
    
    # Risk management
    stop_loss_atr_mult: float  # ATR multiplier for stop loss
    take_profit_atr_mult: float  # ATR multiplier for take profit
    trailing_stop: bool        # Use trailing stop
    
    # ML Gate
    use_ml_gate: bool          # Enable/disable ML Gate
    ml_min_score: float        # ML minimum to pass gate (if enabled)
    ml_boost: float            # Score boost if ML > threshold
    
    # Pillar weights (override defaults)
    pillar_weights: Dict[str, float] = field(default_factory=dict)
    
    # Position sizing by score
    def get_position_size(self, score: float) -> float:
        """Dynamic position sizing based on score."""
        if score >= 90:
            return self.position_size_pct * 2.0
        elif score >= 85:
            return self.position_size_pct * 1.6
        elif score >= 80:
            return self.position_size_pct * 1.2
        return self.position_size_pct


# ============================================================
# The 4 Strategy Profiles
# ============================================================

STRATEGY_PROFILES: Dict[str, StrategyProfile] = {
    
    # 1. AGGRESSIVE + ML GATE
    "aggressive_ml": StrategyProfile(
        id="aggressive_ml",
        name="🔴 Aggressive + ML Gate",
        description="High conviction entries with ML safety gate. Larger positions, tighter selection.",
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
        pillar_weights={
            "technical": 0.55,
            "fundamental": 0.45,
            "sentiment": 0.0,
            "news": 0.0,
            "ml": 0.0,
        },
    ),
    
    # 2. AGGRESSIVE + NO ML GATE
    "aggressive_no_ml": StrategyProfile(
        id="aggressive_no_ml",
        name="🟠 Aggressive (No ML Gate)",
        description="High conviction entries without ML gate. Pure fundamental + technical analysis.",
        color="#f97316",
        min_score=70,
        max_positions=10,
        position_size_pct=8.0,
        stop_loss_atr_mult=1.5,
        take_profit_atr_mult=3.0,
        trailing_stop=True,
        use_ml_gate=False,
        ml_min_score=0,
        ml_boost=0,
        pillar_weights={
            "technical": 0.55,
            "fundamental": 0.45,
            "sentiment": 0.0,
            "news": 0.0,
            "ml": 0.0,
        },
    ),
    
    # 3. MODERATE + ML GATE
    "moderate_ml": StrategyProfile(
        id="moderate_ml",
        name="🟢 Moderate + ML Gate",
        description="Selective entries with ML safety gate. Smaller positions, higher threshold.",
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
        pillar_weights={
            "technical": 0.55,
            "fundamental": 0.45,
            "sentiment": 0.0,
            "news": 0.0,
            "ml": 0.0,
        },
    ),
    
    # 4. MODERATE + NO ML GATE
    "moderate_no_ml": StrategyProfile(
        id="moderate_no_ml",
        name="🔵 Moderate (No ML Gate)",
        description="Selective entries without ML gate. Conservative approach, pure analysis.",
        color="#3b82f6",
        min_score=78,
        max_positions=6,
        position_size_pct=5.0,
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=4.0,
        trailing_stop=True,
        use_ml_gate=False,
        ml_min_score=0,
        ml_boost=0,
        pillar_weights={
            "technical": 0.55,
            "fundamental": 0.45,
            "sentiment": 0.0,
            "news": 0.0,
            "ml": 0.0,
        },
    ),
}


def get_all_profiles() -> Dict[str, StrategyProfile]:
    return STRATEGY_PROFILES


def get_profile(profile_id: str) -> Optional[StrategyProfile]:
    return STRATEGY_PROFILES.get(profile_id)
