"""
Trading Tools - Outils appelables par le LLM
Ces outils permettent à l'agent de prendre des décisions de trading.

Pattern: Tool Use / Function Calling
Le LLM peut appeler ces fonctions pour exécuter des actions.

Outils disponibles:
- approve_order: Approuver un ordre de trading
- reject_signal: Rejeter un signal
- add_to_watchlist: Ajouter un symbole à la watchlist
- request_human_review: Demander une validation humaine
- adjust_position_size: Ajuster la taille d'une position
- set_market_regime: Définir le régime de marché

Usage:
    from src.agents.tools import get_trading_tools

    tools = get_trading_tools(guardrails, executor, state_manager)

    # Le LLM appelle un outil
    result = await tools.approve_order(
        symbol='NVDA',
        quantity=10,
        reason='RSI breakout confirmed with volume'
    )
"""

import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Awaitable
from datetime import datetime
from pathlib import Path
import logging
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# TOOL RESULT
# =============================================================================

@dataclass
class ToolResult:
    """Résultat d'un appel d'outil"""
    success: bool
    message: str
    data: Optional[Dict] = None
    requires_human: bool = False
    action_taken: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'message': self.message,
            'data': self.data,
            'requires_human': self.requires_human,
            'action_taken': self.action_taken
        }


# =============================================================================
# TOOL DEFINITIONS (for LLM)
# =============================================================================

TOOL_DEFINITIONS = [
    {
        "name": "approve_order",
        "description": "Approuve et passe un ordre de trading. Utilisé quand les conditions sont favorables.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Le ticker du symbole (ex: 'NVDA')"
                },
                "action": {
                    "type": "string",
                    "enum": ["BUY", "SELL"],
                    "description": "Action d'achat ou vente"
                },
                "quantity": {
                    "type": "integer",
                    "description": "Nombre d'actions"
                },
                "order_type": {
                    "type": "string",
                    "enum": ["MARKET", "LIMIT"],
                    "description": "Type d'ordre"
                },
                "limit_price": {
                    "type": "number",
                    "description": "Prix limite (requis si order_type=LIMIT)"
                },
                "reason": {
                    "type": "string",
                    "description": "Raison de l'approbation"
                },
                "confidence": {
                    "type": "number",
                    "description": "Score de confiance (0-100)"
                }
            },
            "required": ["symbol", "action", "quantity", "reason"]
        }
    },
    {
        "name": "reject_signal",
        "description": "Rejette un signal de trading. Utilisé quand les conditions ne sont pas favorables.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Le ticker du symbole"
                },
                "reason": {
                    "type": "string",
                    "description": "Raison du rejet"
                },
                "add_to_avoid_list": {
                    "type": "boolean",
                    "description": "Ajouter à la liste des symboles à éviter temporairement"
                },
                "avoid_duration_hours": {
                    "type": "integer",
                    "description": "Durée d'évitement en heures"
                }
            },
            "required": ["symbol", "reason"]
        }
    },
    {
        "name": "add_to_watchlist",
        "description": "Ajoute un symbole à la watchlist pour surveillance future.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Le ticker du symbole"
                },
                "reason": {
                    "type": "string",
                    "description": "Raison de l'ajout"
                },
                "priority": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Priorité de surveillance"
                },
                "trigger_conditions": {
                    "type": "string",
                    "description": "Conditions à surveiller (ex: 'RSI < 30')"
                }
            },
            "required": ["symbol", "reason"]
        }
    },
    {
        "name": "request_human_review",
        "description": "Demande une validation humaine pour une décision importante.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Le ticker concerné"
                },
                "decision_type": {
                    "type": "string",
                    "enum": ["large_position", "unusual_signal", "risk_limit", "uncertain"],
                    "description": "Type de décision nécessitant validation"
                },
                "context": {
                    "type": "string",
                    "description": "Contexte de la décision"
                },
                "recommendation": {
                    "type": "string",
                    "description": "Recommandation de l'agent"
                },
                "urgency": {
                    "type": "string",
                    "enum": ["immediate", "today", "this_week"],
                    "description": "Urgence de la décision"
                }
            },
            "required": ["decision_type", "context", "recommendation"]
        }
    },
    {
        "name": "adjust_position_size",
        "description": "Ajuste la taille de position recommandée pour un trade.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Le ticker du symbole"
                },
                "adjustment_factor": {
                    "type": "number",
                    "description": "Facteur d'ajustement (0.5 = réduire de moitié, 1.5 = augmenter de 50%)"
                },
                "reason": {
                    "type": "string",
                    "description": "Raison de l'ajustement"
                }
            },
            "required": ["symbol", "adjustment_factor", "reason"]
        }
    },
    {
        "name": "set_market_regime",
        "description": "Définit le régime de marché actuel pour ajuster la stratégie.",
        "parameters": {
            "type": "object",
            "properties": {
                "regime": {
                    "type": "string",
                    "enum": ["bull_strong", "bull_weak", "bear_strong", "bear_weak", "ranging", "high_volatility"],
                    "description": "Régime de marché"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confiance dans l'évaluation (0-1)"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Raisonnement"
                }
            },
            "required": ["regime", "reasoning"]
        }
    },
    {
        "name": "update_stop_loss",
        "description": "Met à jour le stop loss d'une position existante.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Le ticker de la position"
                },
                "new_stop_loss": {
                    "type": "number",
                    "description": "Nouveau prix de stop loss"
                },
                "reason": {
                    "type": "string",
                    "description": "Raison de la mise à jour"
                }
            },
            "required": ["symbol", "new_stop_loss", "reason"]
        }
    }
]


# =============================================================================
# TRADING TOOLS CLASS
# =============================================================================

class TradingTools:
    """
    Collection d'outils de trading pour l'agent.

    Ces outils sont appelés par le LLM via le pattern tool use.
    Chaque outil valide les entrées et respecte les guardrails.
    """

    def __init__(
        self,
        guardrails=None,
        executor=None,
        state_manager=None,
        notification_callback: Optional[Callable] = None
    ):
        """
        Initialise les outils de trading.

        Args:
            guardrails: TradingGuardrails pour validation
            executor: IBKRExecutor pour exécution
            state_manager: StateManager pour l'état
            notification_callback: Fonction pour envoyer des notifications
        """
        self.guardrails = guardrails
        self.executor = executor
        self.state_manager = state_manager
        self.notify = notification_callback or self._default_notify

        # Historique des actions
        self._action_history: List[Dict] = []

        # Liste temporaire des symboles à éviter
        self._avoid_list: Dict[str, datetime] = {}

        # Watchlist enrichie
        self._watchlist_details: Dict[str, Dict] = {}

        # Pending human reviews
        self._pending_reviews: List[Dict] = []

    def get_tool_definitions(self) -> List[Dict]:
        """Retourne les définitions d'outils pour le LLM"""
        return TOOL_DEFINITIONS

    async def execute_tool(self, tool_name: str, parameters: Dict) -> ToolResult:
        """
        Exécute un outil par son nom.

        Args:
            tool_name: Nom de l'outil
            parameters: Paramètres de l'outil

        Returns:
            Résultat de l'exécution
        """
        tool_map = {
            'approve_order': self.approve_order,
            'reject_signal': self.reject_signal,
            'add_to_watchlist': self.add_to_watchlist,
            'request_human_review': self.request_human_review,
            'adjust_position_size': self.adjust_position_size,
            'set_market_regime': self.set_market_regime,
            'update_stop_loss': self.update_stop_loss
        }

        if tool_name not in tool_map:
            return ToolResult(
                success=False,
                message=f"Unknown tool: {tool_name}"
            )

        try:
            result = await tool_map[tool_name](**parameters)
            self._log_action(tool_name, parameters, result)
            return result

        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return ToolResult(
                success=False,
                message=f"Error executing {tool_name}: {str(e)}"
            )

    async def approve_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        reason: str,
        order_type: str = "LIMIT",
        limit_price: Optional[float] = None,
        confidence: float = 50.0
    ) -> ToolResult:
        """
        Approuve et passe un ordre de trading.

        Args:
            symbol: Ticker
            action: BUY ou SELL
            quantity: Nombre d'actions
            reason: Raison de l'approbation
            order_type: MARKET ou LIMIT
            limit_price: Prix limite
            confidence: Score de confiance

        Returns:
            ToolResult
        """
        logger.info(f"approve_order called: {action} {quantity} {symbol}")

        # 1. Vérifier si le symbole est dans la liste d'évitement
        if symbol in self._avoid_list:
            avoid_until = self._avoid_list[symbol]
            if datetime.now() < avoid_until:
                return ToolResult(
                    success=False,
                    message=f"{symbol} est dans la liste d'évitement jusqu'à {avoid_until}"
                )
            else:
                del self._avoid_list[symbol]

        # 2. Valider via guardrails
        if self.guardrails:
            from src.agents.guardrails import TradeRequest

            trade_request = TradeRequest(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=limit_price or 0,
                signal_source='llm_approval',
                confidence_score=confidence
            )

            validation = self.guardrails.validate_trade(trade_request)

            if not validation.approved:
                return ToolResult(
                    success=False,
                    message=f"Trade rejected by guardrails: {validation.reason}",
                    requires_human=validation.requires_human_approval
                )

        # 3. Exécuter l'ordre
        if self.executor and self.executor.is_connected():
            from src.execution.ibkr_executor import OrderRequest, OrderAction, OrderType

            request = OrderRequest(
                symbol=symbol,
                action=OrderAction[action],
                quantity=quantity,
                order_type=OrderType[order_type],
                limit_price=limit_price,
                signal_source='agent_approval',
                confidence_score=confidence,
                notes=reason
            )

            result = await self.executor.place_order(request)

            if result.status.value in ['filled', 'submitted']:
                # Mettre à jour l'état
                if self.state_manager:
                    await self.state_manager.add_trade({
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'price': result.fill_price or limit_price,
                        'reason': reason,
                        'order_id': result.order_id
                    })

                return ToolResult(
                    success=True,
                    message=f"Order {result.status.value}: {action} {quantity} {symbol}",
                    data={'order_id': result.order_id, 'fill_price': result.fill_price},
                    action_taken='order_placed'
                )
            else:
                return ToolResult(
                    success=False,
                    message=f"Order failed: {result.message}"
                )
        else:
            # Mode simulation
            logger.warning("Executor not connected - simulating order")
            return ToolResult(
                success=True,
                message=f"[SIMULATED] Would {action} {quantity} {symbol}",
                data={'simulated': True, 'reason': reason},
                action_taken='order_simulated'
            )

    async def reject_signal(
        self,
        symbol: str,
        reason: str,
        add_to_avoid_list: bool = False,
        avoid_duration_hours: int = 24
    ) -> ToolResult:
        """
        Rejette un signal de trading.

        Args:
            symbol: Ticker
            reason: Raison du rejet
            add_to_avoid_list: Ajouter à la liste d'évitement
            avoid_duration_hours: Durée d'évitement

        Returns:
            ToolResult
        """
        logger.info(f"reject_signal called: {symbol} - {reason}")

        if add_to_avoid_list:
            avoid_until = datetime.now() + timedelta(hours=avoid_duration_hours)
            self._avoid_list[symbol] = avoid_until
            message = f"Signal rejected for {symbol}. Added to avoid list until {avoid_until.strftime('%H:%M')}"
        else:
            message = f"Signal rejected for {symbol}: {reason}"

        # Mettre à jour l'état
        if self.state_manager:
            await self.state_manager.log_decision({
                'type': 'signal_rejected',
                'symbol': symbol,
                'reason': reason,
                'avoid_list': add_to_avoid_list
            })

        return ToolResult(
            success=True,
            message=message,
            data={'symbol': symbol, 'reason': reason, 'avoided': add_to_avoid_list},
            action_taken='signal_rejected'
        )

    async def add_to_watchlist(
        self,
        symbol: str,
        reason: str,
        priority: str = "medium",
        trigger_conditions: Optional[str] = None
    ) -> ToolResult:
        """
        Ajoute un symbole à la watchlist.

        Args:
            symbol: Ticker
            reason: Raison de l'ajout
            priority: high, medium, low
            trigger_conditions: Conditions à surveiller

        Returns:
            ToolResult
        """
        logger.info(f"add_to_watchlist called: {symbol} ({priority})")

        self._watchlist_details[symbol] = {
            'added_at': datetime.now().isoformat(),
            'reason': reason,
            'priority': priority,
            'trigger_conditions': trigger_conditions
        }

        # Mettre à jour l'état
        if self.state_manager:
            self.state_manager.state.watchlist.add(symbol)

        return ToolResult(
            success=True,
            message=f"Added {symbol} to watchlist (priority: {priority})",
            data={'symbol': symbol, 'priority': priority},
            action_taken='added_to_watchlist'
        )

    async def request_human_review(
        self,
        decision_type: str,
        context: str,
        recommendation: str,
        symbol: Optional[str] = None,
        urgency: str = "today"
    ) -> ToolResult:
        """
        Demande une validation humaine.

        Args:
            decision_type: Type de décision
            context: Contexte
            recommendation: Recommandation de l'agent
            symbol: Ticker concerné
            urgency: Niveau d'urgence

        Returns:
            ToolResult
        """
        logger.info(f"request_human_review called: {decision_type}")

        review_request = {
            'id': f"review_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'decision_type': decision_type,
            'symbol': symbol,
            'context': context,
            'recommendation': recommendation,
            'urgency': urgency,
            'status': 'pending'
        }

        self._pending_reviews.append(review_request)

        # Envoyer notification
        await self.notify(
            f"🔔 Review Required ({urgency})\n"
            f"Type: {decision_type}\n"
            f"Symbol: {symbol or 'N/A'}\n"
            f"Recommendation: {recommendation}"
        )

        return ToolResult(
            success=True,
            message=f"Human review requested for {decision_type}",
            data=review_request,
            requires_human=True,
            action_taken='human_review_requested'
        )

    async def adjust_position_size(
        self,
        symbol: str,
        adjustment_factor: float,
        reason: str
    ) -> ToolResult:
        """
        Ajuste la taille de position recommandée.

        Args:
            symbol: Ticker
            adjustment_factor: Facteur d'ajustement (0.5 à 2.0)
            reason: Raison

        Returns:
            ToolResult
        """
        logger.info(f"adjust_position_size called: {symbol} x{adjustment_factor}")

        # Valider le facteur
        if adjustment_factor < 0.25 or adjustment_factor > 2.0:
            return ToolResult(
                success=False,
                message=f"Invalid adjustment factor: {adjustment_factor}. Must be between 0.25 and 2.0"
            )

        # Stocker l'ajustement
        if not hasattr(self, '_size_adjustments'):
            self._size_adjustments = {}

        self._size_adjustments[symbol] = {
            'factor': adjustment_factor,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }

        return ToolResult(
            success=True,
            message=f"Position size for {symbol} adjusted by {adjustment_factor}x",
            data={'symbol': symbol, 'factor': adjustment_factor},
            action_taken='position_size_adjusted'
        )

    async def set_market_regime(
        self,
        regime: str,
        reasoning: str,
        confidence: float = 0.7
    ) -> ToolResult:
        """
        Définit le régime de marché.

        Args:
            regime: Type de régime
            reasoning: Raisonnement
            confidence: Confiance (0-1)

        Returns:
            ToolResult
        """
        logger.info(f"set_market_regime called: {regime}")

        valid_regimes = ['bull_strong', 'bull_weak', 'bear_strong', 'bear_weak', 'ranging', 'high_volatility']

        if regime not in valid_regimes:
            return ToolResult(
                success=False,
                message=f"Invalid regime: {regime}. Valid: {valid_regimes}"
            )

        # Mettre à jour l'état
        if self.state_manager:
            from src.agents.state import MarketRegime
            regime_enum = MarketRegime[regime.upper()]
            self.state_manager.state.market_regime = regime_enum

        return ToolResult(
            success=True,
            message=f"Market regime set to {regime} (confidence: {confidence:.0%})",
            data={'regime': regime, 'confidence': confidence, 'reasoning': reasoning},
            action_taken='market_regime_updated'
        )

    async def update_stop_loss(
        self,
        symbol: str,
        new_stop_loss: float,
        reason: str
    ) -> ToolResult:
        """
        Met à jour le stop loss d'une position.

        Args:
            symbol: Ticker
            new_stop_loss: Nouveau prix
            reason: Raison

        Returns:
            ToolResult
        """
        logger.info(f"update_stop_loss called: {symbol} -> {new_stop_loss}")

        # Vérifier si la position existe
        if self.state_manager:
            position = self.state_manager.state.positions.get(symbol)
            if not position:
                return ToolResult(
                    success=False,
                    message=f"No position found for {symbol}"
                )

            old_stop = position.stop_loss
            position.stop_loss = new_stop_loss

            return ToolResult(
                success=True,
                message=f"Stop loss for {symbol} updated: {old_stop} -> {new_stop_loss}",
                data={'symbol': symbol, 'old_stop': old_stop, 'new_stop': new_stop_loss},
                action_taken='stop_loss_updated'
            )
        else:
            return ToolResult(
                success=True,
                message=f"[SIMULATED] Stop loss for {symbol} would be updated to {new_stop_loss}",
                action_taken='stop_loss_simulated'
            )

    def get_position_size_adjustment(self, symbol: str) -> float:
        """Retourne le facteur d'ajustement pour un symbole"""
        if hasattr(self, '_size_adjustments') and symbol in self._size_adjustments:
            return self._size_adjustments[symbol]['factor']
        return 1.0

    def get_pending_reviews(self) -> List[Dict]:
        """Retourne les reviews en attente"""
        return [r for r in self._pending_reviews if r['status'] == 'pending']

    def approve_review(self, review_id: str, approved: bool, notes: str = "") -> bool:
        """Approuve ou rejette une review humaine"""
        for review in self._pending_reviews:
            if review['id'] == review_id:
                review['status'] = 'approved' if approved else 'rejected'
                review['resolved_at'] = datetime.now().isoformat()
                review['notes'] = notes
                return True
        return False

    def is_symbol_avoided(self, symbol: str) -> bool:
        """Vérifie si un symbole est dans la liste d'évitement"""
        if symbol in self._avoid_list:
            if datetime.now() < self._avoid_list[symbol]:
                return True
            del self._avoid_list[symbol]
        return False

    def _log_action(self, tool_name: str, parameters: Dict, result: ToolResult):
        """Log une action dans l'historique"""
        self._action_history.append({
            'timestamp': datetime.now().isoformat(),
            'tool': tool_name,
            'parameters': parameters,
            'result': result.to_dict()
        })

        # Garder les 1000 dernières actions
        if len(self._action_history) > 1000:
            self._action_history = self._action_history[-1000:]

    async def _default_notify(self, message: str):
        """Notification par défaut (log)"""
        logger.info(f"NOTIFICATION: {message}")

    def get_action_history(self, limit: int = 50) -> List[Dict]:
        """Retourne l'historique des actions"""
        return self._action_history[-limit:]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_trading_tools(
    guardrails=None,
    executor=None,
    state_manager=None,
    notification_callback=None
) -> TradingTools:
    """Factory pour créer les outils de trading"""
    return TradingTools(
        guardrails=guardrails,
        executor=executor,
        state_manager=state_manager,
        notification_callback=notification_callback
    )


# Import helper for datetime
from datetime import timedelta


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        print("=== Trading Tools Test ===\n")

        tools = TradingTools()

        # Test approve_order (simulated)
        print("1. Testing approve_order...")
        result = await tools.approve_order(
            symbol='NVDA',
            action='BUY',
            quantity=10,
            reason='RSI breakout with strong volume',
            confidence=85
        )
        print(f"   Result: {result.message}")

        # Test reject_signal
        print("\n2. Testing reject_signal...")
        result = await tools.reject_signal(
            symbol='TSLA',
            reason='RSI overbought, weak volume',
            add_to_avoid_list=True,
            avoid_duration_hours=4
        )
        print(f"   Result: {result.message}")

        # Test add_to_watchlist
        print("\n3. Testing add_to_watchlist...")
        result = await tools.add_to_watchlist(
            symbol='AMD',
            reason='Potential breakout forming',
            priority='high',
            trigger_conditions='RSI < 40 and Volume > 1.5x'
        )
        print(f"   Result: {result.message}")

        # Test request_human_review
        print("\n4. Testing request_human_review...")
        result = await tools.request_human_review(
            decision_type='large_position',
            context='Position would be 12% of portfolio',
            recommendation='Reduce to 8%',
            symbol='NVDA',
            urgency='today'
        )
        print(f"   Result: {result.message}")

        # Test set_market_regime
        print("\n5. Testing set_market_regime...")
        result = await tools.set_market_regime(
            regime='bull_strong',
            reasoning='SPY above all EMAs, breadth positive',
            confidence=0.8
        )
        print(f"   Result: {result.message}")

        # Check avoid list
        print("\n6. Checking avoid list...")
        is_avoided = tools.is_symbol_avoided('TSLA')
        print(f"   TSLA avoided: {is_avoided}")

        # Get pending reviews
        print("\n7. Pending reviews:")
        for review in tools.get_pending_reviews():
            print(f"   - {review['id']}: {review['decision_type']}")

        # Get tool definitions
        print(f"\n8. Available tools: {len(tools.get_tool_definitions())}")
        for tool in tools.get_tool_definitions():
            print(f"   - {tool['name']}: {tool['description'][:50]}...")

    asyncio.run(main())
