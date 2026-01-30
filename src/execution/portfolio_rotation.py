#!/usr/bin/env python3
"""
Portfolio Rotation Manager
Gère la rotation intelligente du portefeuille quand il est plein.

Logique:
- Si nouveau signal avec score élevé + portefeuille plein
- Comparer avec les positions existantes (score ACTUEL, pas d'entrée)
- Vendre la position la plus faible si le nouveau signal est significativement meilleur
"""
import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PositionScore:
    """Score actuel d'une position"""
    symbol: str
    entry_price: float
    current_price: float
    entry_score: float
    current_score: float
    pnl_pct: float
    days_held: int
    grok_sentiment: str = "neutral"
    
    @property
    def rotation_score(self) -> float:
        """
        Score composite pour décider de la rotation.
        Plus le score est bas, plus la position est candidate à la vente.
        """
        score = self.current_score
        
        # Bonus si en profit (on garde les winners)
        if self.pnl_pct > 10:
            score += 20  # Forte protection des gros gains
        elif self.pnl_pct > 5:
            score += 10  # Protection modérée
        elif self.pnl_pct > 0:
            score += 5   # Légère protection
        
        # Malus si détenue depuis peu (éviter churning)
        if self.days_held < 3:
            score += 15  # Ne pas vendre trop vite
        
        # Malus si sentiment bearish
        if self.grok_sentiment == "bearish":
            score -= 10
        
        return score


@dataclass
class RotationDecision:
    """Décision de rotation"""
    should_rotate: bool
    sell_symbol: Optional[str] = None
    sell_reason: str = ""
    buy_symbol: Optional[str] = None
    buy_score: float = 0
    confidence: float = 0  # 0-100


class PortfolioRotationManager:
    """
    Gestionnaire de rotation de portefeuille.
    
    Paramètres:
    - max_positions: Nombre max de positions simultanées
    - min_score_advantage: Avantage minimum de score pour rotation (default: 15)
    - min_hold_days: Jours minimum avant de pouvoir vendre (default: 3)
    - max_rotations_per_day: Limite de rotations quotidiennes (default: 2)
    - protect_winners_above: Ne pas vendre si P&L > ce % (default: 10)
    """
    
    def __init__(
        self,
        max_positions: int = 10,
        min_score_advantage: float = 15,
        min_hold_days: int = 3,
        max_rotations_per_day: int = 2,
        protect_winners_above: float = 10,
        position_manager=None,
        grok_scanner=None,
        reasoning_engine=None
    ):
        self.max_positions = max_positions
        self.min_score_advantage = min_score_advantage
        self.min_hold_days = min_hold_days
        self.max_rotations_per_day = max_rotations_per_day
        self.protect_winners_above = protect_winners_above
        
        self.position_manager = position_manager
        self.grok_scanner = grok_scanner
        self.reasoning_engine = reasoning_engine
        
        # Tracking des rotations du jour
        self._rotations_today: List[datetime] = []
        self._last_rotation_reset: datetime = datetime.now().date()
    
    def _reset_daily_counter(self):
        """Reset le compteur de rotations si nouveau jour."""
        today = datetime.now().date()
        if today != self._last_rotation_reset:
            self._rotations_today = []
            self._last_rotation_reset = today
    
    def can_rotate(self) -> bool:
        """Vérifie si une rotation est possible aujourd'hui."""
        self._reset_daily_counter()
        return len(self._rotations_today) < self.max_rotations_per_day
    
    async def get_current_position_scores(self) -> List[PositionScore]:
        """
        Récupère les scores ACTUELS de toutes les positions.
        Utilise le reasoning engine pour recalculer les scores.
        """
        positions = []
        
        if not self.position_manager:
            return positions
        
        for symbol, pos in self.position_manager.positions.items():
            # Calculer P&L
            pnl_pct = 0
            if pos.entry_price > 0 and pos.current_price > 0:
                pnl_pct = ((pos.current_price - pos.entry_price) / pos.entry_price) * 100
            
            # Jours détenus
            days_held = (datetime.now() - pos.entry_date).days
            
            # Score actuel (utiliser le score stocké ou recalculer)
            current_score = pos.current_score if pos.current_score > 0 else pos.entry_score
            
            # Recalculer le score si reasoning engine disponible
            if self.reasoning_engine and current_score == pos.entry_score:
                try:
                    # Recalculer les 4 piliers
                    new_score = await self._recalculate_score(symbol)
                    if new_score > 0:
                        current_score = new_score
                        pos.current_score = new_score
                except Exception as e:
                    logger.warning(f"Could not recalculate score for {symbol}: {e}")
            
            positions.append(PositionScore(
                symbol=symbol,
                entry_price=pos.entry_price,
                current_price=pos.current_price,
                entry_score=pos.entry_score,
                current_score=current_score,
                pnl_pct=pnl_pct,
                days_held=days_held,
                grok_sentiment=pos.last_grok_sentiment
            ))
        
        return positions
    
    async def _recalculate_score(self, symbol: str) -> float:
        """Recalcule le score d'un symbole via le reasoning engine."""
        if not self.reasoning_engine:
            return 0
        
        try:
            # Appeler le reasoning engine pour obtenir le nouveau score
            result = await self.reasoning_engine.analyze_symbol(symbol)
            if result and 'total_score' in result:
                return result['total_score']
        except Exception as e:
            logger.warning(f"Score recalculation failed for {symbol}: {e}")
        
        return 0
    
    def find_weakest_position(
        self, 
        positions: List[PositionScore]
    ) -> Optional[PositionScore]:
        """
        Trouve la position la plus faible candidate à la rotation.
        Exclut les positions protégées (gros gains, trop récentes).
        """
        candidates = []
        
        for pos in positions:
            # Exclure les positions trop récentes
            if pos.days_held < self.min_hold_days:
                logger.debug(f"{pos.symbol}: Too recent ({pos.days_held} days)")
                continue
            
            # Exclure les gros winners
            if pos.pnl_pct > self.protect_winners_above:
                logger.debug(f"{pos.symbol}: Protected winner ({pos.pnl_pct:.1f}%)")
                continue
            
            candidates.append(pos)
        
        if not candidates:
            return None
        
        # Trier par rotation_score (le plus bas = le plus faible)
        candidates.sort(key=lambda x: x.rotation_score)
        return candidates[0]
    
    async def evaluate_rotation(
        self,
        new_signal_symbol: str,
        new_signal_score: float,
        current_positions: List[PositionScore]
    ) -> RotationDecision:
        """
        Évalue si on doit faire une rotation.
        
        Args:
            new_signal_symbol: Symbole du nouveau signal
            new_signal_score: Score du nouveau signal
            current_positions: Positions actuelles avec leurs scores
        
        Returns:
            RotationDecision avec la décision et les détails
        """
        # Vérifier si on peut encore faire une rotation aujourd'hui
        if not self.can_rotate():
            logger.info(f"Max rotations reached for today ({self.max_rotations_per_day})")
            return RotationDecision(
                should_rotate=False,
                sell_reason="Max daily rotations reached"
            )
        
        # Vérifier si le portefeuille est plein
        if len(current_positions) < self.max_positions:
            logger.info(f"Portfolio not full ({len(current_positions)}/{self.max_positions})")
            return RotationDecision(
                should_rotate=False,
                sell_reason="Portfolio not full - can add without rotation"
            )
        
        # Vérifier si on a déjà ce symbole
        if any(p.symbol == new_signal_symbol for p in current_positions):
            return RotationDecision(
                should_rotate=False,
                sell_reason=f"Already holding {new_signal_symbol}"
            )
        
        # Trouver la position la plus faible
        weakest = self.find_weakest_position(current_positions)
        
        if not weakest:
            logger.info("No rotation candidate found (all positions protected)")
            return RotationDecision(
                should_rotate=False,
                sell_reason="All positions protected"
            )
        
        # Calculer l'avantage du nouveau signal
        score_advantage = new_signal_score - weakest.rotation_score
        
        logger.info(f"Rotation evaluation: {new_signal_symbol} (score {new_signal_score}) "
                   f"vs {weakest.symbol} (rotation_score {weakest.rotation_score:.1f})")
        logger.info(f"Score advantage: {score_advantage:.1f} (min required: {self.min_score_advantage})")
        
        # Décision
        if score_advantage >= self.min_score_advantage:
            confidence = min(100, 50 + score_advantage)
            return RotationDecision(
                should_rotate=True,
                sell_symbol=weakest.symbol,
                sell_reason=f"Score {weakest.current_score:.0f} < new signal {new_signal_score:.0f} "
                           f"(advantage: +{score_advantage:.0f})",
                buy_symbol=new_signal_symbol,
                buy_score=new_signal_score,
                confidence=confidence
            )
        else:
            return RotationDecision(
                should_rotate=False,
                sell_reason=f"Score advantage too low ({score_advantage:.1f} < {self.min_score_advantage})"
            )
    
    async def execute_rotation(
        self,
        decision: RotationDecision,
        executor=None
    ) -> bool:
        """
        Exécute une rotation (vente + achat).
        
        Returns:
            True si la rotation a été exécutée avec succès
        """
        if not decision.should_rotate:
            return False
        
        logger.info(f"🔄 EXECUTING ROTATION: Sell {decision.sell_symbol} → Buy {decision.buy_symbol}")
        logger.info(f"   Reason: {decision.sell_reason}")
        logger.info(f"   Confidence: {decision.confidence:.0f}%")
        
        # 1. Vendre la position existante
        sell_success = False
        if executor and decision.sell_symbol:
            try:
                # Obtenir la quantité à vendre
                if self.position_manager and decision.sell_symbol in self.position_manager.positions:
                    pos = self.position_manager.positions[decision.sell_symbol]
                    
                    from src.execution.ibkr_executor import OrderRequest, OrderType, OrderSide
                    sell_order = OrderRequest(
                        symbol=decision.sell_symbol,
                        side=OrderSide.SELL,
                        quantity=pos.quantity,
                        order_type=OrderType.MARKET
                    )
                    result = await executor.place_order(sell_order)
                    logger.info(f"✅ Sold {decision.sell_symbol}: {result}")
                    sell_success = True
                    
                    # Retirer du position manager
                    del self.position_manager.positions[decision.sell_symbol]
            except Exception as e:
                logger.error(f"❌ Failed to sell {decision.sell_symbol}: {e}")
                return False
        else:
            # Mode simulation
            logger.info(f"[SIMULATION] Would sell {decision.sell_symbol}")
            sell_success = True
        
        # 2. Enregistrer la rotation
        if sell_success:
            self._rotations_today.append(datetime.now())
            await self._send_rotation_notification(decision)
        
        return sell_success
    
    async def _send_rotation_notification(self, decision: RotationDecision):
        """Envoie une notification Telegram pour la rotation."""
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            return
        
        message = f"""
🔄 **PORTFOLIO ROTATION**

📤 SELL: {decision.sell_symbol}
   Reason: {decision.sell_reason}

📥 BUY: {decision.buy_symbol}
   Score: {decision.buy_score:.0f}

🎯 Confidence: {decision.confidence:.0f}%
⏰ {datetime.now().strftime('%H:%M:%S')}

Rotations today: {len(self._rotations_today)}/{self.max_rotations_per_day}
"""
        
        try:
            import aiohttp
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            async with aiohttp.ClientSession() as session:
                await session.post(url, json={
                    'chat_id': chat_id,
                    'text': message,
                    'parse_mode': 'Markdown'
                })
        except Exception as e:
            logger.warning(f"Telegram notification failed: {e}")
    
    async def check_and_rotate(
        self,
        new_signal_symbol: str,
        new_signal_score: float,
        executor=None
    ) -> RotationDecision:
        """
        Point d'entrée principal: vérifie et exécute une rotation si nécessaire.
        
        Args:
            new_signal_symbol: Symbole du nouveau signal
            new_signal_score: Score du nouveau signal
            executor: Executor pour passer les ordres (optionnel)
        
        Returns:
            RotationDecision avec le résultat
        """
        # Récupérer les scores actuels des positions
        current_positions = await self.get_current_position_scores()
        
        # Évaluer la rotation
        decision = await self.evaluate_rotation(
            new_signal_symbol,
            new_signal_score,
            current_positions
        )
        
        # Exécuter si décidé
        if decision.should_rotate:
            success = await self.execute_rotation(decision, executor)
            if not success:
                decision.should_rotate = False
                decision.sell_reason = "Execution failed"
        
        return decision
    
    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut du rotation manager."""
        self._reset_daily_counter()
        return {
            'max_positions': self.max_positions,
            'min_score_advantage': self.min_score_advantage,
            'rotations_today': len(self._rotations_today),
            'max_rotations_per_day': self.max_rotations_per_day,
            'can_rotate': self.can_rotate()
        }


# Singleton
_rotation_manager: Optional[PortfolioRotationManager] = None

def get_rotation_manager(**kwargs) -> PortfolioRotationManager:
    """Récupère ou crée le rotation manager singleton."""
    global _rotation_manager
    if _rotation_manager is None:
        _rotation_manager = PortfolioRotationManager(**kwargs)
    return _rotation_manager
