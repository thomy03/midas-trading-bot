#!/usr/bin/env python3
"""
Position Manager avec Adaptive Exit Strategy - V8.2
Monitore les positions ouvertes et applique les sorties intelligentes.

V8.2 enhancements:
- Regime-adaptive trailing stop (tighter in VOLATILE, wider in BULL)
- Partial position reduction (sell 50% if score degrades but stays positive)
- Intelligence-based re-evaluation exits
"""
import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    SCORE_DROP = "score_below_threshold"
    TRAILING_STOP = "trailing_stop"
    ATR_STOP = "atr_stop"
    GROK_BEARISH = "grok_bearish_signal"
    MAX_HOLD = "max_hold_reached"
    PARTIAL_SCORE_DECAY = "partial_score_decay"
    INTEL_DEGRADATION = "intelligence_degradation"
    MANUAL = "manual"


# V8.2: Regime-specific trailing stop parameters
REGIME_TRAILING_CONFIG = {
    'BULL': {
        'activation_pct': 0.06,   # Activate at +6% (wider, let it run)
        'distance_pct': 0.07,     # 7% trailing distance
        'atr_multiplier': 3.0,    # Wider initial stop
    },
    'BEAR': {
        'activation_pct': 0.03,   # Activate at +3% (take profits early)
        'distance_pct': 0.04,     # 4% trailing distance (tight)
        'atr_multiplier': 2.0,    # Tighter initial stop
    },
    'RANGE': {
        'activation_pct': 0.05,   # Activate at +5%
        'distance_pct': 0.05,     # 5% trailing distance
        'atr_multiplier': 2.5,    # Standard
    },
    'VOLATILE': {
        'activation_pct': 0.04,   # Activate at +4%
        'distance_pct': 0.06,     # 6% trailing (wider to avoid whipsaws)
        'atr_multiplier': 2.0,    # Tighter initial stop (protect capital)
    },
}

DEFAULT_TRAILING_CONFIG = REGIME_TRAILING_CONFIG['RANGE']


@dataclass
class TrackedPosition:
    """Position suivie avec donnees pour adaptive exit"""
    symbol: str
    entry_price: float
    entry_date: datetime
    quantity: int
    initial_quantity: int  # V8.2: Track original quantity for partial reductions
    entry_score: float

    # Tracking pour adaptive exit
    highest_price: float = 0.0
    current_price: float = 0.0
    current_score: float = 0.0
    previous_score: float = 0.0  # V8.2: Track score trend
    trailing_active: bool = False
    stop_level: float = 0.0
    atr: float = 0.02  # Default 2%

    # V8.2: Regime-adaptive parameters
    regime: str = 'RANGE'
    partial_exit_done: bool = False  # Track if we already did a partial exit

    # Grok sentiment
    last_grok_sentiment: str = "neutral"
    last_grok_score: float = 0.0

    # V8.2: Intelligence re-evaluation tracking
    last_intel_check: Optional[datetime] = None
    intel_score_at_entry: float = 0.0
    current_intel_score: float = 0.0

    def __post_init__(self):
        self.highest_price = self.entry_price
        self.initial_quantity = self.initial_quantity or self.quantity


class AdaptivePositionManager:
    """
    Gestionnaire de positions avec sorties adaptatives - V8.2.

    Strategie:
    - Pas de TP fixe (laisser courir les gains)
    - Stop ATR dynamique adapte au regime (BULL=large, VOLATILE=serre)
    - Trailing adaptatif active a +X% selon regime
    - Sortie partielle (50%) si score se degrade mais reste positif
    - Sortie si score descend sous 45
    - Sortie si Grok detecte sentiment tres bearish
    - Re-evaluation intelligence a J+5, J+10
    """

    def __init__(
        self,
        score_exit_threshold: float = 45,
        partial_exit_threshold: float = 55,   # V8.2: Score below this -> partial exit
        partial_exit_fraction: float = 0.50,  # V8.2: Sell 50% on partial exit
        max_hold_days: int = 30,
        intel_reeval_days: List[int] = None,  # V8.2: Days to re-evaluate intelligence
        grok_scanner=None,
        executor=None
    ):
        self.score_exit_threshold = score_exit_threshold
        self.partial_exit_threshold = partial_exit_threshold
        self.partial_exit_fraction = partial_exit_fraction
        self.max_hold_days = max_hold_days
        self.intel_reeval_days = intel_reeval_days or [5, 10]

        self.grok_scanner = grok_scanner
        self.executor = executor

        self.positions: Dict[str, TrackedPosition] = {}
        self._running = False

    def _get_regime_config(self, regime: str) -> dict:
        """Get trailing stop config for the current regime."""
        return REGIME_TRAILING_CONFIG.get(regime, DEFAULT_TRAILING_CONFIG)

    def add_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        entry_score: float,
        atr: float = 0.02,
        regime: str = 'RANGE',
        intel_score: float = 0.0
    ):
        """Ajoute une nouvelle position a tracker."""
        config = self._get_regime_config(regime)

        pos = TrackedPosition(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=datetime.now(),
            quantity=quantity,
            initial_quantity=quantity,
            entry_score=entry_score,
            atr=atr,
            regime=regime,
            intel_score_at_entry=intel_score,
            current_intel_score=intel_score,
        )
        # Regime-adaptive initial stop (ATR-based)
        pos.stop_level = entry_price * (1 - atr * config['atr_multiplier'])
        self.positions[symbol] = pos
        logger.info(
            f"[POS] Position added: {symbol} @ ${entry_price:.2f}, "
            f"Stop: ${pos.stop_level:.2f} (ATR*{config['atr_multiplier']}, regime={regime})"
        )

    def update_regime(self, symbol: str, new_regime: str):
        """V8.2: Update regime for a position (adjusts trailing parameters)."""
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        if pos.regime != new_regime:
            old_regime = pos.regime
            pos.regime = new_regime
            logger.info(f"[POS] {symbol}: Regime changed {old_regime} -> {new_regime}")

    def update_price(self, symbol: str, current_price: float, high_price: float = None):
        """Met a jour le prix et gere le trailing stop adaptatif."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        pos.current_price = current_price
        config = self._get_regime_config(pos.regime)

        # Update highest price
        check_high = high_price or current_price
        if check_high > pos.highest_price:
            pos.highest_price = check_high

        # Calcul du gain actuel
        gain_pct = (pos.highest_price - pos.entry_price) / pos.entry_price

        # V8.2: Regime-adaptive trailing activation
        activation_pct = config['activation_pct']
        trailing_distance = config['distance_pct']

        if not pos.trailing_active and gain_pct >= activation_pct:
            pos.trailing_active = True
            logger.info(
                f"[POS] {symbol}: Trailing ACTIVATED at +{gain_pct*100:.1f}% "
                f"(regime={pos.regime}, distance={trailing_distance*100:.0f}%)"
            )

        # Mise a jour du trailing stop
        if pos.trailing_active:
            new_stop = pos.highest_price * (1 - trailing_distance)
            if new_stop > pos.stop_level:
                pos.stop_level = new_stop
                logger.debug(f"[POS] {symbol}: Trailing stop raised to ${new_stop:.2f}")

        return pos

    def update_score(self, symbol: str, new_score: float):
        """Met a jour le score d'un symbole."""
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        pos.previous_score = pos.current_score
        pos.current_score = new_score

    def update_intel_score(self, symbol: str, intel_score: float):
        """V8.2: Update intelligence score for a position."""
        if symbol not in self.positions:
            return
        pos = self.positions[symbol]
        pos.current_intel_score = intel_score
        pos.last_intel_check = datetime.now()

    async def update_grok_sentiment(self, symbol: str):
        """Recupere le sentiment Grok pour un symbole."""
        if not self.grok_scanner or symbol not in self.positions:
            return

        try:
            insight = await self.grok_scanner.analyze_symbol(symbol)
            if insight:
                pos = self.positions[symbol]
                pos.last_grok_sentiment = insight.get('sentiment', 'neutral')
                pos.last_grok_score = insight.get('sentiment_score', 0)
                logger.info(
                    f"[POS] {symbol}: Grok sentiment = {pos.last_grok_sentiment} "
                    f"({pos.last_grok_score:+.2f})"
                )
        except Exception as e:
            logger.warning(f"Grok sentiment error for {symbol}: {e}")

    def check_exit(self, symbol: str) -> Optional[Tuple[ExitReason, str, bool]]:
        """
        Verifie si une position doit etre fermee.

        Returns:
            (ExitReason, details, is_partial) si exit, None sinon
            is_partial=True means sell only partial_exit_fraction of position
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        if pos.current_price <= 0:
            return None

        current_pnl = (pos.current_price - pos.entry_price) / pos.entry_price

        # 1. Check ATR/Trailing Stop (full exit)
        if pos.current_price <= pos.stop_level:
            reason = ExitReason.TRAILING_STOP if pos.trailing_active else ExitReason.ATR_STOP
            return (reason, f"Price ${pos.current_price:.2f} hit stop ${pos.stop_level:.2f}", False)

        # 2. Check Score Drop (full exit if below threshold)
        if pos.current_score > 0 and pos.current_score < self.score_exit_threshold:
            return (
                ExitReason.SCORE_DROP,
                f"Score dropped to {pos.current_score:.0f} (threshold: {self.score_exit_threshold})",
                False,
            )

        # 3. V8.2: Partial exit if score degraded but still positive
        if (
            not pos.partial_exit_done
            and pos.current_score > 0
            and pos.current_score < self.partial_exit_threshold
            and pos.entry_score >= self.partial_exit_threshold
            and current_pnl > 0  # Only partial exit if in profit
        ):
            return (
                ExitReason.PARTIAL_SCORE_DECAY,
                f"Score degraded {pos.entry_score:.0f} -> {pos.current_score:.0f}, "
                f"partial exit (P&L: {current_pnl*100:+.1f}%)",
                True,  # Partial exit
            )

        # 4. V8.2: Intelligence degradation check
        days_held = (datetime.now() - pos.entry_date).days
        if (
            days_held in self.intel_reeval_days
            and pos.current_intel_score > 0
            and pos.intel_score_at_entry > 0
        ):
            intel_decay = (pos.intel_score_at_entry - pos.current_intel_score) / pos.intel_score_at_entry
            if intel_decay > 0.5:  # Intelligence score dropped by >50%
                return (
                    ExitReason.INTEL_DEGRADATION,
                    f"Intelligence score degraded by {intel_decay*100:.0f}% at J+{days_held}",
                    False,
                )

        # 5. Check Grok Bearish Signal (sentiment tres negatif)
        if pos.last_grok_score < -0.6 and pos.last_grok_sentiment == 'bearish':
            return (
                ExitReason.GROK_BEARISH,
                f"Grok bearish signal: {pos.last_grok_sentiment} ({pos.last_grok_score:+.2f})",
                False,
            )

        # 6. Check Max Hold
        if days_held >= self.max_hold_days:
            return (
                ExitReason.MAX_HOLD,
                f"Position held for {days_held} days (max: {self.max_hold_days})",
                False,
            )

        return None

    async def execute_exit(
        self, symbol: str, reason: ExitReason, details: str, is_partial: bool = False
    ):
        """Execute la sortie d'une position (totale ou partielle)."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price * 100

        # V8.2: Calculate quantity to sell
        if is_partial:
            sell_quantity = max(1, int(pos.quantity * self.partial_exit_fraction))
            exit_type = "PARTIAL EXIT"
        else:
            sell_quantity = pos.quantity
            exit_type = "FULL EXIT"

        logger.info(f"[POS] {exit_type} {symbol}: {reason.value}")
        logger.info(f"   Details: {details}")
        logger.info(f"   Entry: ${pos.entry_price:.2f} -> Current: ${pos.current_price:.2f}")
        logger.info(f"   P&L: {pnl_pct:+.1f}% | Selling {sell_quantity}/{pos.quantity} shares")

        # Execute sell order if executor available
        if self.executor:
            try:
                from src.execution.ibkr_executor import OrderRequest, OrderType, OrderSide
                order = OrderRequest(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=sell_quantity,
                    order_type=OrderType.MARKET
                )
                result = await self.executor.place_order(order)
                logger.info(f"[POS] Sell order placed for {symbol}: {result}")
            except Exception as e:
                logger.error(f"[POS] Failed to execute exit for {symbol}: {e}")

        # V8.2: Handle partial vs full exit
        if is_partial:
            pos.quantity -= sell_quantity
            pos.partial_exit_done = True
            logger.info(
                f"[POS] {symbol}: Partial exit done, remaining {pos.quantity} shares "
                f"(was {pos.initial_quantity})"
            )
        else:
            del self.positions[symbol]

        # Notification Telegram si configure
        await self._send_notification(symbol, reason, details, pnl_pct, exit_type)

    async def _send_notification(
        self, symbol: str, reason: ExitReason, details: str,
        pnl_pct: float, exit_type: str = "EXIT"
    ):
        """Envoie une notification Telegram."""
        bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')

        if not bot_token or not chat_id:
            return

        emoji = "🟢" if pnl_pct > 0 else "🔴"
        message = (
            f"{emoji} **{exit_type} - {symbol}**\n\n"
            f"Reason: {reason.value}\n"
            f"{details}\n"
            f"P&L: {pnl_pct:+.1f}%\n\n"
            f"{datetime.now().strftime('%H:%M:%S')}"
        )

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
        logger.info(f"[POS] Position monitor started (interval: {interval_seconds}s)")

        while self._running:
            try:
                for symbol in list(self.positions.keys()):
                    # Check exit conditions
                    exit_signal = self.check_exit(symbol)
                    if exit_signal:
                        reason, details, is_partial = exit_signal
                        await self.execute_exit(symbol, reason, details, is_partial)

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(10)

    def stop(self):
        """Arrete le monitoring."""
        self._running = False
        logger.info("Position monitor stopped")

    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut de toutes les positions."""
        status = {}
        for symbol, pos in self.positions.items():
            pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price * 100 if pos.current_price else 0
            config = self._get_regime_config(pos.regime)
            status[symbol] = {
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'pnl_pct': round(pnl_pct, 2),
                'stop_level': round(pos.stop_level, 2),
                'trailing_active': pos.trailing_active,
                'score': pos.current_score,
                'regime': pos.regime,
                'trailing_config': config,
                'grok_sentiment': pos.last_grok_sentiment,
                'days_held': (datetime.now() - pos.entry_date).days,
                'quantity': pos.quantity,
                'initial_quantity': pos.initial_quantity,
                'partial_exit_done': pos.partial_exit_done,
            }
        return status


# Singleton instance
_position_manager: Optional[AdaptivePositionManager] = None

def get_position_manager(grok_scanner=None, executor=None) -> AdaptivePositionManager:
    """Recupere ou cree le position manager singleton."""
    global _position_manager
    if _position_manager is None:
        _position_manager = AdaptivePositionManager(
            grok_scanner=grok_scanner,
            executor=executor
        )
    return _position_manager
