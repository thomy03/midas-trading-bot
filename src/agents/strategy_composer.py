"""
Strategy Composer - LLM-powered strategy creation (V4.1).

This module allows the trading bot to autonomously create new trading
strategies by combining indicators from the library.

V4.1: Uses Gemini via OpenRouter for strategy composition.
Grok is reserved for sentiment analysis only (X/Twitter access).

The LLM (Gemini) can:
1. Analyze current strategy performance
2. Identify failure patterns
3. Propose new indicator combinations
4. Define entry/exit conditions
5. Set risk parameters

Strategies are validated through backtest + paper trading before production.
"""

import os
import json
import httpx
import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import logging

from src.indicators import get_indicator_registry, IndicatorCategory

logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """Status of a strategy in its lifecycle"""
    PROPOSED = "proposed"       # Just created by LLM
    BACKTESTING = "backtesting" # Running backtest
    PAPER_TRADING = "paper_trading"  # In paper validation
    ACTIVE = "active"           # Live production
    PAUSED = "paused"           # Temporarily disabled
    ARCHIVED = "archived"       # No longer used


class ConditionLogic(Enum):
    """How to combine multiple conditions"""
    AND = "AND"
    OR = "OR"


@dataclass
class IndicatorCondition:
    """A single condition on an indicator"""
    indicator: str
    params: Dict[str, Any]
    operator: str  # ">", "<", ">=", "<=", "==", "crosses_above", "crosses_below"
    value: Any  # Number or another indicator output
    output_column: Optional[str] = None  # Which output to use (for multi-output indicators)


@dataclass
class StrategyDefinition:
    """Complete definition of a trading strategy"""
    name: str
    description: str
    version: int = 1

    # Entry conditions
    entry_conditions: List[IndicatorCondition] = field(default_factory=list)
    entry_logic: ConditionLogic = ConditionLogic.AND

    # Exit conditions
    exit_conditions: List[IndicatorCondition] = field(default_factory=list)
    exit_logic: ConditionLogic = ConditionLogic.OR

    # Risk management
    stop_loss_type: str = "atr"  # "atr", "percent", "support"
    stop_loss_value: float = 2.0  # ATR multiplier or percentage
    take_profit_type: str = "percent"
    take_profit_value: float = 0.08  # 8%

    # Filters
    min_volume_ratio: float = 1.0
    market_regime: Optional[str] = None  # "bull", "bear", "range", None (any)
    sectors: List[str] = field(default_factory=list)  # Empty = all sectors
    min_market_cap: Optional[float] = None  # In billions

    # Metadata
    status: StrategyStatus = StrategyStatus.PROPOSED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = "llm"  # "llm" or "user"
    reasoning: str = ""  # LLM explanation for why this strategy

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['entry_logic'] = self.entry_logic.value
        data['exit_logic'] = self.exit_logic.value
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyDefinition':
        """Create from dictionary"""
        data = data.copy()
        data['entry_logic'] = ConditionLogic(data.get('entry_logic', 'AND'))
        data['exit_logic'] = ConditionLogic(data.get('exit_logic', 'OR'))
        data['status'] = StrategyStatus(data.get('status', 'proposed'))

        # Convert entry conditions
        entry_conditions = []
        for cond in data.get('entry_conditions', []):
            if isinstance(cond, dict):
                entry_conditions.append(IndicatorCondition(**cond))
            else:
                entry_conditions.append(cond)
        data['entry_conditions'] = entry_conditions

        # Convert exit conditions
        exit_conditions = []
        for cond in data.get('exit_conditions', []):
            if isinstance(cond, dict):
                exit_conditions.append(IndicatorCondition(**cond))
            else:
                exit_conditions.append(cond)
        data['exit_conditions'] = exit_conditions

        return cls(**data)


@dataclass
class ComposerConfig:
    """Configuration for the Strategy Composer (V4.1)"""
    # LLM via OpenRouter (Gemini 3 Flash)
    openrouter_api_key: str = ""
    llm_model: str = ""  # Read from OPENROUTER_MODEL env var

    # Legacy Grok (fallback/optional)
    grok_api_key: str = ""
    grok_model: str = "grok-4-1-fast-reasoning"

    # Strategy limits
    max_indicators_per_strategy: int = 5
    min_backtest_win_rate: float = 0.45
    min_backtest_sharpe: float = 0.5
    min_paper_trades: int = 3
    max_strategies_per_week: int = 2
    max_active_strategies: int = 3


class StrategyComposer:
    """
    LLM-powered strategy composition engine (V4.1).

    Uses Gemini via OpenRouter to analyze performance and create new trading strategies
    by combining indicators from the library.
    """

    def __init__(self, config: Optional[ComposerConfig] = None):
        """Initialize the composer"""
        self.config = config or ComposerConfig()

        # Primary: OpenRouter (Gemini)
        if not self.config.openrouter_api_key:
            self.config.openrouter_api_key = os.getenv('OPENROUTER_API_KEY', '')
        if not self.config.llm_model:
            self.config.llm_model = os.getenv('OPENROUTER_MODEL', 'google/gemini-3-flash-preview')

        # Fallback: Grok (optional)
        if not self.config.grok_api_key:
            self.config.grok_api_key = os.getenv('GROK_API_KEY', '')

        self.registry = get_indicator_registry()
        self._client: Optional[httpx.AsyncClient] = None

        # Load history
        self._strategies_created_this_week = 0
        self._last_creation_date: Optional[datetime] = None

        # Determine which backend to use
        self._use_openrouter = bool(self.config.openrouter_api_key)
        logger.info(f"StrategyComposer V4.1 initialized, backend: {'OpenRouter/Gemini' if self._use_openrouter else 'Grok'}")

    async def initialize(self):
        """Initialize the async client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)

    async def close(self):
        """Close the async client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_indicator_library_prompt(self) -> str:
        """Generate a description of available indicators for the LLM"""
        registry_dict = self.registry.to_dict()

        categories = {}
        for name, info in registry_dict.items():
            cat = info['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                'name': name,
                'description': info['description'],
                'params': info['params'],
                'outputs': info['outputs']
            })

        prompt_parts = ["AVAILABLE INDICATORS:\n"]
        for category, indicators in categories.items():
            prompt_parts.append(f"\n## {category.upper()}")
            for ind in indicators:
                params_str = ", ".join([f"{k}={v}" for k, v in ind['params'].items()])
                prompt_parts.append(f"- {ind['name']}({params_str})")
                prompt_parts.append(f"  Description: {ind['description']}")
                prompt_parts.append(f"  Outputs: {', '.join(ind['outputs'])}")

        return "\n".join(prompt_parts)

    def _get_composition_prompt(
        self,
        performance_data: Dict[str, Any],
        failure_patterns: List[str],
        market_context: Dict[str, Any]
    ) -> str:
        """Generate the prompt for strategy composition"""

        indicator_library = self._get_indicator_library_prompt()

        prompt = f"""Tu es un trader quantitatif expert. Analyse les données de performance et propose une nouvelle stratégie de trading.

{indicator_library}

PERFORMANCE ACTUELLE:
- Stratégies actives: {performance_data.get('active_strategies', [])}
- Win rates: {performance_data.get('win_rates', {})}
- Total P&L: {performance_data.get('total_pnl', 0):.2f}%

PATTERNS D'ÉCHEC IDENTIFIÉS:
{chr(10).join(['- ' + p for p in failure_patterns]) if failure_patterns else '- Aucun pattern identifié'}

CONTEXTE MARCHÉ:
- Régime: {market_context.get('regime', 'unknown')}
- VIX: {market_context.get('vix', 'N/A')}
- SPY trend: {market_context.get('spy_trend', 'N/A')}

CONTRAINTES:
- Maximum {self.config.max_indicators_per_strategy} indicateurs par stratégie
- Stop loss obligatoire (ATR-based ou %)
- Doit inclure au moins 1 confirmation volume
- Win rate backtest minimum requis: {self.config.min_backtest_win_rate * 100:.0f}%

TÂCHE:
1. Analyse pourquoi les stratégies actuelles sous-performent (si applicable)
2. Propose une nouvelle combinaison d'indicateurs adaptée au contexte de marché
3. Définis les conditions d'entrée et de sortie précises
4. Explique ton raisonnement

IMPORTANT: Réponds UNIQUEMENT avec un JSON valide au format suivant:
{{
    "name": "nom_strategie",
    "description": "Description courte de la stratégie",
    "reasoning": "Explication détaillée du raisonnement",
    "entry_conditions": [
        {{
            "indicator": "RSI",
            "params": {{"period": 14}},
            "operator": "<",
            "value": 30,
            "output_column": "RSI"
        }}
    ],
    "entry_logic": "AND",
    "exit_conditions": [
        {{
            "indicator": "RSI",
            "params": {{"period": 14}},
            "operator": ">",
            "value": 70,
            "output_column": "RSI"
        }}
    ],
    "exit_logic": "OR",
    "stop_loss_type": "atr",
    "stop_loss_value": 2.0,
    "take_profit_type": "percent",
    "take_profit_value": 0.08,
    "min_volume_ratio": 1.5,
    "market_regime": null
}}
"""
        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM API (OpenRouter/Gemini or Grok fallback)"""
        if not self._client:
            await self.initialize()

        if self._use_openrouter:
            return await self._call_openrouter(prompt)
        else:
            return await self._call_grok(prompt)

    async def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API with Gemini"""
        if not self.config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not configured")

        headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://tradingbot-v4.local",
            "X-Title": "TradingBot V4.1 Strategy Composer"
        }

        payload = {
            "model": self.config.llm_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }

        response = await self._client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
            raise Exception(f"OpenRouter API error: {response.status_code}")

        result = response.json()
        return result['choices'][0]['message']['content']

    async def _call_grok(self, prompt: str) -> str:
        """Call Grok API (fallback)"""
        if not self.config.grok_api_key:
            raise ValueError("GROK_API_KEY not configured")

        headers = {
            "Authorization": f"Bearer {self.config.grok_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.grok_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }

        response = await self._client.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            logger.error(f"Grok API error: {response.status_code} - {response.text}")
            raise Exception(f"Grok API error: {response.status_code}")

        result = response.json()
        return result['choices'][0]['message']['content']

    def _parse_strategy_response(self, response: str) -> StrategyDefinition:
        """Parse the LLM response into a StrategyDefinition"""
        # Extract JSON from response (handle markdown code blocks)
        json_str = response
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            json_str = response[start:end].strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse strategy JSON: {e}")
            logger.error(f"Response was: {response[:500]}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")

        # Validate required fields
        required = ['name', 'entry_conditions']
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Convert to StrategyDefinition
        return StrategyDefinition.from_dict(data)

    async def compose_strategy(
        self,
        performance_data: Dict[str, Any],
        failure_patterns: List[str] = None,
        market_context: Dict[str, Any] = None
    ) -> StrategyDefinition:
        """
        Have the LLM compose a new trading strategy.

        Args:
            performance_data: Current strategy performance metrics
            failure_patterns: Identified patterns in losing trades
            market_context: Current market conditions

        Returns:
            A new StrategyDefinition proposed by the LLM
        """
        # Check rate limits
        self._check_rate_limits()

        failure_patterns = failure_patterns or []
        market_context = market_context or {}

        # Generate prompt
        prompt = self._get_composition_prompt(
            performance_data,
            failure_patterns,
            market_context
        )

        logger.info(f"Requesting new strategy from {'Gemini' if self._use_openrouter else 'Grok'}...")

        # Call LLM
        response = await self._call_llm(prompt)

        # Parse response
        strategy = self._parse_strategy_response(response)

        # Validate strategy
        self._validate_strategy(strategy)

        # Update rate limit tracking
        self._strategies_created_this_week += 1
        self._last_creation_date = datetime.now()

        logger.info(f"New strategy proposed: {strategy.name}")
        logger.info(f"Reasoning: {strategy.reasoning[:200]}...")

        return strategy

    def _check_rate_limits(self):
        """Check if we can create more strategies this week"""
        now = datetime.now()

        # Reset weekly counter if new week
        if self._last_creation_date:
            days_since = (now - self._last_creation_date).days
            if days_since >= 7:
                self._strategies_created_this_week = 0

        if self._strategies_created_this_week >= self.config.max_strategies_per_week:
            raise ValueError(
                f"Rate limit: max {self.config.max_strategies_per_week} strategies/week reached"
            )

    def _validate_strategy(self, strategy: StrategyDefinition):
        """Validate that a strategy is well-formed"""
        # Check indicator count
        total_indicators = len(strategy.entry_conditions) + len(strategy.exit_conditions)
        if total_indicators > self.config.max_indicators_per_strategy * 2:
            raise ValueError(
                f"Too many indicators: {total_indicators} "
                f"(max {self.config.max_indicators_per_strategy * 2})"
            )

        # Verify all indicators exist
        for condition in strategy.entry_conditions + strategy.exit_conditions:
            if condition.indicator not in self.registry.get_names():
                raise ValueError(f"Unknown indicator: {condition.indicator}")

        # Verify stop loss is set
        if not strategy.stop_loss_type or strategy.stop_loss_value <= 0:
            raise ValueError("Stop loss must be configured")

        # Check for volume confirmation in entry
        has_volume = any(
            cond.indicator in ['VolumeRatio', 'OBV', 'VWAP', 'CMF', 'MFI']
            for cond in strategy.entry_conditions
        )
        if not has_volume:
            logger.warning("Strategy has no volume confirmation - adding VolumeRatio")
            strategy.entry_conditions.append(
                IndicatorCondition(
                    indicator="VolumeRatio",
                    params={"period": 20},
                    operator=">=",
                    value=1.0,
                    output_column="VOLUME_RATIO"
                )
            )

    async def analyze_and_improve(
        self,
        strategy: StrategyDefinition,
        trades: List[Dict[str, Any]],
        performance: Dict[str, Any]
    ) -> Tuple[StrategyDefinition, str]:
        """
        Have the LLM analyze a strategy's performance and suggest improvements.

        Returns:
            Tuple of (improved_strategy, reasoning)
        """
        prompt = f"""Analyse cette stratégie de trading et ses résultats.

STRATÉGIE: {strategy.name}
Description: {strategy.description}

CONDITIONS D'ENTRÉE:
{json.dumps([asdict(c) for c in strategy.entry_conditions], indent=2)}

CONDITIONS DE SORTIE:
{json.dumps([asdict(c) for c in strategy.exit_conditions], indent=2)}

PERFORMANCE:
- Trades: {performance.get('total_trades', 0)}
- Win rate: {performance.get('win_rate', 0) * 100:.1f}%
- P&L total: {performance.get('total_pnl', 0):.2f}%
- Sharpe: {performance.get('sharpe', 0):.2f}

DERNIERS TRADES PERDANTS:
{json.dumps([t for t in trades if t.get('pnl', 0) < 0][:5], indent=2)}

TÂCHE:
1. Identifie les patterns communs dans les trades perdants
2. Propose des ajustements aux conditions d'entrée/sortie
3. Explique ton raisonnement

Réponds avec un JSON contenant:
{{
    "analysis": "Analyse des problèmes identifiés",
    "improvements": [
        {{"type": "modify_condition", "target": "entry", "index": 0, "change": "description du changement"}},
        {{"type": "add_condition", "target": "entry", "condition": {{...}}}},
        {{"type": "remove_condition", "target": "exit", "index": 1}}
    ],
    "expected_improvement": "Estimation de l'amélioration attendue"
}}
"""

        response = await self._call_llm(prompt)

        # Parse and apply improvements
        try:
            json_str = response
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()

            data = json.loads(json_str)

            # Create improved strategy
            improved = StrategyDefinition.from_dict(strategy.to_dict())
            improved.version += 1
            improved.name = f"{strategy.name}_v{improved.version}"

            # Apply improvements
            for imp in data.get('improvements', []):
                self._apply_improvement(improved, imp)

            reasoning = f"""
ANALYSE: {data.get('analysis', '')}

AMÉLIORATIONS APPLIQUÉES:
{chr(10).join([f"- {i.get('change', str(i))}" for i in data.get('improvements', [])])}

AMÉLIORATION ATTENDUE: {data.get('expected_improvement', 'N/A')}
"""

            return improved, reasoning

        except Exception as e:
            logger.error(f"Failed to parse improvement response: {e}")
            return strategy, f"Échec de l'analyse: {e}"

    def _apply_improvement(self, strategy: StrategyDefinition, improvement: Dict):
        """Apply a single improvement to a strategy"""
        imp_type = improvement.get('type')
        target = improvement.get('target', 'entry')
        conditions = strategy.entry_conditions if target == 'entry' else strategy.exit_conditions

        if imp_type == 'modify_condition':
            idx = improvement.get('index', 0)
            if 0 <= idx < len(conditions):
                # Modify the condition based on the change description
                # This is a simplified version - in practice, you'd parse the change
                pass

        elif imp_type == 'add_condition':
            cond_data = improvement.get('condition', {})
            if cond_data:
                conditions.append(IndicatorCondition(**cond_data))

        elif imp_type == 'remove_condition':
            idx = improvement.get('index', 0)
            if 0 <= idx < len(conditions):
                conditions.pop(idx)


# Singleton instance
_composer_instance: Optional[StrategyComposer] = None


async def get_strategy_composer(config: Optional[ComposerConfig] = None) -> StrategyComposer:
    """Get or create the global StrategyComposer instance"""
    global _composer_instance
    if _composer_instance is None:
        _composer_instance = StrategyComposer(config)
        await _composer_instance.initialize()
    return _composer_instance
