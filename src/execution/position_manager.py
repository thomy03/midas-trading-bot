#!/usr/bin/env python3
"""
Position Manager avec Adaptive Exit Strategy
Monitore les positions ouvertes et applique les sorties intelligentes.
"""
import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    SCORE_DROP = "score_below_threshold"
    TRAILING_STOP = "trailing_stop"
    ATR_STOP = "atr_stop"
    GROK_BEARISH = "grok_bearish_signal"
    MAX_HOLD = "max_hold_reached"
    MANUAL = "manual"


@dataclass
class TrackedPosition:
    """Position suivie avec données pour adaptive exit"""
    symbol: str
    entry_price: float
    entry_date: datetime
    quantity: int
    entry_score: float
    
    # Tracking pour adaptive exit
    highest_price: float = 0.0
    current_price: float = 0.0
    current_score: float = 0.0
    trailing_active: bool = False
    stop_level: float = 0.0
    atr: float = 0.02  # Default 2%
    
    # Grok sentiment
    last_grok_sentiment: str = "neutral"
    last_grok_score: float = 0.0
    
    def __post_init__(self):
        self.highest_price = self.entry_price


class AdaptivePositionManager:
    """
    Gestionnaire de positions avec sorties adaptatives.
    
    Stratégie:
    - Pas de TP fixe (laisser courir les gains)
    - Stop ATR dynamique (ATR × 2.5)
    - Trailing activé après +5%, distance 12%
    - Sortie si score descend sous 45
    - Sortie si Grok détecte sentiment très bearish
    """
    
    def __init__(
        self,
        score_exit_threshold: float = 45,
        atr_multiplier: float = 2.5,
        trailing_activation: float = 0.05,  # +5%
        trailing_distance: float = 0.12,     # 12%
        max_hold_days: int = 30,
        grok_scanner=None,
        executor=None
    ):
        self.score_exit_threshold = score_exit_threshold
        self.atr_multiplier = atr_multiplier
        self.trailing_activation = trailing_activation
        self.trailing_distance = trailing_distance
        self.max_hold_days = max_hold_days
        
        self.grok_scanner = grok_scanner
        self.executor = executor
        
        self.positions: Dict[str, TrackedPosition] = {}
        self._running = False
    
    def add_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        entry_score: float,
        atr: float = 0.02
    ):
        """Ajoute une nouvelle position à tracker."""
        pos = TrackedPosition(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=datetime.now(),
            quantity=quantity,
            entry_score=entry_score,
            atr=atr
        )
        # Calcul du stop initial (ATR-based)
        pos.stop_level = entry_price * (1 - atr * self.atr_multiplier)
        self.positions[symbol] = pos
        logger.info(f"📊 Position added: {symbol} @ ${entry_price:.2f}, "
                   f"Stop: ${pos.stop_level:.2f} (ATR-based)")
    
    def update_price(self, symbol: str, current_price: float, high_price: float = None):
        """Met à jour le prix et gère le trailing stop."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        pos.current_price = current_price
        
        # Update highest price
        check_high = high_price or current_price
        if check_high > pos.highest_price:
            pos.highest_price = check_high
        
        # Calcul du gain actuel
        gain_pct = (pos.highest_price - pos.entry_price) / pos.entry_price
        
        # Activation du trailing stop
        if not pos.trailing_active and gain_pct >= self.trailing_activation:
            pos.trailing_active = True
            logger.info(f"🎯 {symbol}: Trailing ACTIVATED at +{gain_pct*100:.1f}%")
        
        # Mise à jour du trailing stop
        if pos.trailing_active:
            new_stop = pos.highest_price * (1 - self.trailing_distance)
            if new_stop > pos.stop_level:
                pos.stop_level = new_stop
                logger.info(f"📈 {symbol}: Trailing stop raised to ${new_stop:.2f}")
        
        return pos
    
    def update_score(self, symbol: str, new_score: float):
        """Met à jour le score d'un symbole."""
        if symbol not in self.positions:
            return
        self.positions[symbol].current_score = new_score
    
    async def update_grok_sentiment(self, symbol: str):
        """Récupère le sentiment Grok pour un symbole."""
        if not self.grok_scanner or symbol not in self.positions:
            return
        
        try:
            insight = await self.grok_scanner.analyze_symbol(symbol)
            if insight:
                pos = self.positions[symbol]
                pos.last_grok_sentiment = insight.get('sentiment', 'neutral')
                pos.last_grok_score = insight.get('sentiment_score', 0)
                logger.info(f"🐦 {symbol}: Grok sentiment = {pos.last_grok_sentiment} "
                           f"({pos.last_grok_score:+.2f})")
        except Exception as e:
            logger.warning(f"Grok sentiment error for {symbol}: {e}")
    
    def check_exit(self, symbol: str) -> Optional[tuple]:
        """
        Vérifie si une position doit être fermée.
        
        Returns:
            (ExitReason, details) si exit, None sinon
        """
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        current_pnl = (pos.current_price - pos.entry_price) / pos.entry_price
        
        # 1. Check ATR/Trailing Stop
        if pos.current_price <= pos.stop_level:
            reason = ExitReason.TRAILING_STOP if pos.trailing_active else ExitReason.ATR_STOP
            return (reason, f"Price ${pos.current_price:.2f} hit stop ${pos.stop_level:.2f}")
        
        # 2. Check Score Drop
        if pos.current_score > 0 and pos.current_score < self.score_exit_threshold:
            return (ExitReason.SCORE_DROP, 
                   f"Score dropped to {pos.current_score:.0f} (threshold: {self.score_exit_threshold})")
        
        # 3. Check Grok Bearish Signal (sentiment très négatif)
        if pos.last_grok_score < -0.6 and pos.last_grok_sentiment == 'bearish':
            return (ExitReason.GROK_BEARISH,
                   f"Grok bearish signal: {pos.last_grok_sentiment} ({pos.last_grok_score:+.2f})")
        
        # 4. Check Max Hold
        days_held = (datetime.now() - pos.entry_date).days
        if days_held >= self.max_hold_days:
            return (ExitReason.MAX_HOLD, f"Position held for {days_held} days (max: {self.max_hold_days})")
        
        return None
    
    async def execute_exit(self, symbol: str, reason: ExitReason, details: str):
        """Exécute la sortie d'une position."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price * 100
        
        logger.info(f"🚨 EXIT {symbol}: {reason.value}")
        logger.info(f"   Details: {details}")
        logger.info(f"   Entry: ${pos.entry_price:.2f} → Current: ${pos.current_price:.2f}")
        logger.info(f"   P&L: {pnl_pct:+.1f}%")
        
        # Exécuter l'ordre de vente si executor disponible
        if self.executor:
            try:
                # Créer ordre de vente market
                from src.execution.ibkr_executor import OrderRequest, OrderType, OrderSide
                order = OrderRequest(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=pos.quantity,
                    order_type=OrderType.MARKET
                )
                result = await self.executor.place_order(order)
                logger.info(f"✅ Sell order placed for {symbol}: {result}")
            except Exception as e:
                logger.error(f"❌ Failed to execute exit for {symbol}: {e}")
        
        # Retirer de la liste des positions
        del self.positions[symbol]
        
        # Notification Telegram si configuré
        await self._send_notification(symbol, reason, details, pnl_pct)
    
    async def _send_notification(self, symbol: str, reason: ExitReason, details: str, pnl_pct: float):
        """Envoie une notification Telegram."""
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            return
        
        emoji = "🟢" if pnl_pct > 0 else "🔴"
        message = f"""
{emoji} **EXIT SIGNAL - {symbol}**

📍 Reason: {reason.value}
📝 {details}
💰 P&L: {pnl_pct:+.1f}%

⏰ {datetime.now().strftime('%H:%M:%S')}
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
    
    async def monitor_loop(self, interval_seconds: int = 60):
        """Boucle de monitoring des positions."""
        self._running = True
        logger.info(f"🔄 Position monitor started (interval: {interval_seconds}s)")
        
        while self._running:
            try:
                for symbol in list(self.positions.keys()):
                    # Check exit conditions
                    exit_signal = self.check_exit(symbol)
                    if exit_signal:
                        reason, details = exit_signal
                        await self.execute_exit(symbol, reason, details)
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(10)
    
    def stop(self):
        """Arrête le monitoring."""
        self._running = False
        logger.info("Position monitor stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut de toutes les positions."""
        status = {}
        for symbol, pos in self.positions.items():
            pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price * 100 if pos.current_price else 0
            status[symbol] = {
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'pnl_pct': round(pnl_pct, 2),
                'stop_level': round(pos.stop_level, 2),
                'trailing_active': pos.trailing_active,
                'score': pos.current_score,
                'grok_sentiment': pos.last_grok_sentiment,
                'days_held': (datetime.now() - pos.entry_date).days
            }
        return status


# Singleton instance
_position_manager: Optional[AdaptivePositionManager] = None

def get_position_manager(grok_scanner=None, executor=None) -> AdaptivePositionManager:
    """Récupère ou crée le position manager singleton."""
    global _position_manager
    if _position_manager is None:
        _position_manager = AdaptivePositionManager(
            grok_scanner=grok_scanner,
            executor=executor
        )
    return _position_manager
