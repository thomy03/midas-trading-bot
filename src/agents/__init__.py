"""
Agents Module - Système agentique V4
Composants pour le trading autonome.
"""

from .guardrails import TradingGuardrails, TradeRequest, ValidationResult
from .state import AgentState, StateManager, MarketRegime, AgentPhase
from .orchestrator import MarketAgent, get_market_agent
from .nightly_auditor import NightlyAuditor, get_nightly_auditor, LearnedGuideline
from .decision_journal import (
    DecisionJournal,
    TradeDecision,
    TechnicalFactors,
    SentimentFactors,
    get_decision_journal,
)
from .strategy_evolver import (
    StrategyEvolver,
    TradeResult,
    FailurePattern,
    StrategyHealth,
    AdaptationLevel,
    FailureType,
    get_strategy_evolver,
)
from .live_loop import (
    LiveLoop,
    LiveLoopConfig,
    LoopMetrics,
    MarketSession,
    get_live_loop,
    start_live_trading,
)
from .strategy_composer import (
    StrategyComposer,
    StrategyDefinition,
    IndicatorCondition,
    ComposerConfig,
    get_strategy_composer,
)
from .strategy_sandbox import (
    StrategySandbox,
    BacktestResult,
    PaperTradeResult,
    get_strategy_sandbox,
)
from .strategy_registry import (
    StrategyRegistry,
    RegisteredStrategy,
    StrategyStatus,
    StrategyPerformance,
    get_strategy_registry,
)
from .reasoning_engine import (
    ReasoningEngine,
    ReasoningResult,
    ReasoningConfig,
    DecisionType,
    get_reasoning_engine,
)
from .pillars import (
    BasePillar,
    PillarScore,
    PillarSignal,
    TechnicalPillar,
    FundamentalPillar,
    SentimentPillar,
    NewsPillar,
)

__all__ = [
    # Guardrails
    'TradingGuardrails',
    'TradeRequest',
    'ValidationResult',

    # State
    'AgentState',
    'StateManager',
    'MarketRegime',
    'AgentPhase',

    # Orchestrator
    'MarketAgent',
    'get_market_agent',

    # Nightly Auditor
    'NightlyAuditor',
    'get_nightly_auditor',
    'LearnedGuideline',

    # Decision Journal
    'DecisionJournal',
    'TradeDecision',
    'TechnicalFactors',
    'SentimentFactors',
    'get_decision_journal',

    # Strategy Evolver
    'StrategyEvolver',
    'TradeResult',
    'FailurePattern',
    'StrategyHealth',
    'AdaptationLevel',
    'FailureType',
    'get_strategy_evolver',

    # Live Loop
    'LiveLoop',
    'LiveLoopConfig',
    'LoopMetrics',
    'MarketSession',
    'get_live_loop',
    'start_live_trading',

    # Strategy Composer
    'StrategyComposer',
    'StrategyDefinition',
    'IndicatorCondition',
    'ComposerConfig',
    'get_strategy_composer',

    # Strategy Sandbox
    'StrategySandbox',
    'BacktestResult',
    'PaperTradeResult',
    'get_strategy_sandbox',

    # Strategy Registry
    'StrategyRegistry',
    'RegisteredStrategy',
    'StrategyStatus',
    'StrategyPerformance',
    'get_strategy_registry',

    # Reasoning Engine
    'ReasoningEngine',
    'ReasoningResult',
    'ReasoningConfig',
    'DecisionType',
    'get_reasoning_engine',

    # Pillars
    'BasePillar',
    'PillarScore',
    'PillarSignal',
    'TechnicalPillar',
    'FundamentalPillar',
    'SentimentPillar',
    'NewsPillar',
]
