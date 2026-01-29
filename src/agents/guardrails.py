"""
Trading Guardrails - Protections Hard-Coded
Ces règles ne peuvent JAMAIS être modifiées par l'IA.

CRITIQUE: Ce fichier protège votre capital en trading réel.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class GuardrailViolation(Exception):
    """Exception levée quand un guardrail est violé"""
    def __init__(self, rule: str, message: str, severity: str = "CRITICAL"):
        self.rule = rule
        self.message = message
        self.severity = severity
        super().__init__(f"[{severity}] {rule}: {message}")


class TradeValidation(Enum):
    """Résultat de validation d'un trade"""
    APPROVED = "approved"           # Trade approuvé pour exécution autonome
    NEEDS_NOTIFICATION = "notify"   # Notification avant exécution (30 min délai)
    NEEDS_APPROVAL = "manual"       # Validation manuelle requise
    REJECTED = "rejected"           # Trade rejeté par guardrail


@dataclass
class TradeRequest:
    """Demande de trade à valider"""
    symbol: str
    action: str              # 'BUY' ou 'SELL'
    quantity: int
    price: float
    order_type: str          # 'MARKET', 'LIMIT'

    # Contexte
    current_capital: float
    position_value: float    # quantity * price
    daily_pnl: float         # P&L du jour en €
    current_drawdown: float  # Drawdown actuel en %

    # Optionnel
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    thesis: Optional[str] = None  # Raison du trade (pour audit)


@dataclass
class ValidationResult:
    """Résultat de la validation d'un trade"""
    status: TradeValidation
    trade: TradeRequest
    reason: str
    violated_rules: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def is_allowed(self) -> bool:
        return self.status in [TradeValidation.APPROVED, TradeValidation.NEEDS_NOTIFICATION]


class TradingGuardrails:
    """
    Guardrails de trading - Protections hard-coded.

    Ces limites sont ABSOLUES et ne peuvent pas être contournées par l'IA.
    Elles protègent le capital en cas de bug, hallucination, ou erreur.
    """

    # =========================================================================
    # HARD LIMITS - JAMAIS MODIFIABLES PAR L'IA
    # =========================================================================

    # Perte maximale par jour (% du capital)
    MAX_DAILY_LOSS_PCT = 0.03  # 3% → Kill switch si atteint

    # Taille maximale d'une position (% du capital)
    MAX_POSITION_PCT = 0.10  # 10% max par position

    # Drawdown maximal avant pause (% du capital initial)
    MAX_DRAWDOWN_PCT = 0.15  # 15% → Pause 24h

    # Nombre max de positions ouvertes simultanément
    MAX_OPEN_POSITIONS = 5

    # Nombre max de trades par jour
    MAX_TRADES_PER_DAY = 10

    # =========================================================================
    # SEUILS DE VALIDATION (Mode Hybride)
    # =========================================================================

    # < 5% capital = autonome
    AUTO_TRADE_THRESHOLD = 0.05

    # 5-10% capital = notification avant exécution
    NOTIFY_THRESHOLD = 0.10

    # > 10% capital = validation manuelle
    MANUAL_THRESHOLD = 0.10

    # Délai de notification avant exécution (secondes)
    NOTIFICATION_DELAY_SECONDS = 1800  # 30 minutes

    # =========================================================================
    # PROFIT OBJECTIVES (V4.1)
    # =========================================================================

    # Objectif mensuel de profit (% du capital)
    MONTHLY_PROFIT_TARGET = 0.08  # 8% mensuel

    # Objectif de win rate minimum
    MIN_WIN_RATE_TARGET = 0.50  # 50%

    # Ratio gain/perte minimum (reward/risk)
    MIN_REWARD_RISK_RATIO = 1.5  # 1.5:1

    # Take profit cible par trade
    DEFAULT_TAKE_PROFIT_PCT = 0.08  # 8%

    # =========================================================================
    # KELLY CRITERION (V4.1)
    # =========================================================================

    # Multiplicateur Kelly (fraction of full Kelly for safety)
    KELLY_FRACTION = 0.25  # Use 1/4 Kelly for safety (full Kelly is too aggressive)

    # Min/max Kelly-adjusted position size
    MIN_KELLY_POSITION_PCT = 0.02  # 2% minimum
    MAX_KELLY_POSITION_PCT = 0.10  # 10% maximum (same as MAX_POSITION_PCT)

    # Default win rate if no history available
    DEFAULT_WIN_RATE = 0.45

    # Default reward/risk ratio if no history
    DEFAULT_REWARD_RISK = 1.5

    # =========================================================================
    # SANITY CHECKS
    # =========================================================================

    # Rejeter si prix dévie > X% du dernier prix connu
    MAX_PRICE_DEVIATION = 0.05  # 5%

    # Rejeter si volume < X% de la moyenne 20j
    MIN_VOLUME_RATIO = 0.5  # 50%

    # Rejeter si spread > X%
    MAX_SPREAD_PCT = 0.02  # 2%

    # Market cap minimum (éviter penny stocks)
    MIN_MARKET_CAP = 500_000_000  # $500M

    # Volume minimum journalier
    MIN_DAILY_VOLUME = 500_000  # 500K actions

    # =========================================================================
    # KILL SWITCH
    # =========================================================================

    # Fichier de kill switch
    KILL_SWITCH_FILE = "data/kill_switch.json"

    def __init__(self, capital: float, data_dir: str = "data"):
        """
        Initialise les guardrails.

        Args:
            capital: Capital initial en €
            data_dir: Répertoire de données
        """
        self.initial_capital = capital
        self.current_capital = capital
        self.data_dir = data_dir

        # État quotidien
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._last_reset_date = date.today()
        self._open_positions: Dict[str, float] = {}  # symbol -> value

        # Kill switch
        self._kill_switch_active = False
        self._kill_switch_reason = ""

        # Historique des violations
        self._violations: List[Dict] = []

        # Charger l'état précédent
        self._load_state()

        logger.info(f"TradingGuardrails initialized: capital={capital}€, "
                   f"max_daily_loss={self.MAX_DAILY_LOSS_PCT*100}%, "
                   f"max_position={self.MAX_POSITION_PCT*100}%")

    # =========================================================================
    # VALIDATION PRINCIPALE
    # =========================================================================

    def validate_trade(self, trade: TradeRequest) -> ValidationResult:
        """
        Valide un trade contre tous les guardrails.

        Args:
            trade: La demande de trade à valider

        Returns:
            ValidationResult avec le statut et les détails
        """
        violations = []
        warnings = []

        # 0. Vérifier le kill switch
        if self._kill_switch_active:
            return ValidationResult(
                status=TradeValidation.REJECTED,
                trade=trade,
                reason=f"KILL SWITCH ACTIVE: {self._kill_switch_reason}",
                violated_rules=["KILL_SWITCH"]
            )

        # 1. Reset quotidien si nouveau jour
        self._check_daily_reset()

        # 2. Vérifier la perte quotidienne max
        if not self._check_daily_loss(trade, violations):
            self._trigger_kill_switch("Daily loss limit exceeded")
            return ValidationResult(
                status=TradeValidation.REJECTED,
                trade=trade,
                reason="Perte quotidienne maximale atteinte - Kill switch activé",
                violated_rules=violations
            )

        # 3. Vérifier le drawdown max
        if not self._check_drawdown(trade, violations):
            self._trigger_pause("Max drawdown exceeded - 24h pause")
            return ValidationResult(
                status=TradeValidation.REJECTED,
                trade=trade,
                reason="Drawdown maximum atteint - Pause 24h",
                violated_rules=violations
            )

        # 4. Vérifier la taille de position
        if not self._check_position_size(trade, violations):
            return ValidationResult(
                status=TradeValidation.REJECTED,
                trade=trade,
                reason="Position trop grande",
                violated_rules=violations
            )

        # 5. Vérifier le nombre de positions
        if not self._check_max_positions(trade, violations):
            return ValidationResult(
                status=TradeValidation.REJECTED,
                trade=trade,
                reason="Nombre maximum de positions atteint",
                violated_rules=violations
            )

        # 6. Vérifier le nombre de trades quotidiens
        if not self._check_daily_trades(trade, violations):
            return ValidationResult(
                status=TradeValidation.REJECTED,
                trade=trade,
                reason="Nombre maximum de trades quotidiens atteint",
                violated_rules=violations
            )

        # 7. Sanity checks (warnings, pas de rejet)
        self._sanity_checks(trade, warnings)

        # 8. Déterminer le niveau de validation requis
        position_pct = trade.position_value / self.current_capital

        if position_pct < self.AUTO_TRADE_THRESHOLD:
            status = TradeValidation.APPROVED
            reason = f"Trade autonome ({position_pct*100:.1f}% < {self.AUTO_TRADE_THRESHOLD*100}%)"
        elif position_pct < self.MANUAL_THRESHOLD:
            status = TradeValidation.NEEDS_NOTIFICATION
            reason = f"Notification requise ({position_pct*100:.1f}% capital)"
        else:
            status = TradeValidation.NEEDS_APPROVAL
            reason = f"Validation manuelle requise ({position_pct*100:.1f}% > {self.MANUAL_THRESHOLD*100}%)"

        return ValidationResult(
            status=status,
            trade=trade,
            reason=reason,
            violated_rules=violations,
            warnings=warnings
        )

    # =========================================================================
    # CHECKS INDIVIDUELS
    # =========================================================================

    def _check_daily_loss(self, trade: TradeRequest, violations: List[str]) -> bool:
        """Vérifie la perte quotidienne max"""
        max_loss = self.initial_capital * self.MAX_DAILY_LOSS_PCT

        if abs(trade.daily_pnl) >= max_loss:
            violations.append(f"MAX_DAILY_LOSS: {trade.daily_pnl:.2f}€ >= {max_loss:.2f}€ ({self.MAX_DAILY_LOSS_PCT*100}%)")
            return False

        return True

    def _check_drawdown(self, trade: TradeRequest, violations: List[str]) -> bool:
        """Vérifie le drawdown max"""
        if trade.current_drawdown >= self.MAX_DRAWDOWN_PCT:
            violations.append(f"MAX_DRAWDOWN: {trade.current_drawdown*100:.1f}% >= {self.MAX_DRAWDOWN_PCT*100}%")
            return False

        return True

    def _check_position_size(self, trade: TradeRequest, violations: List[str]) -> bool:
        """Vérifie la taille de position max"""
        max_position = self.current_capital * self.MAX_POSITION_PCT

        if trade.position_value > max_position:
            violations.append(f"MAX_POSITION: {trade.position_value:.2f}€ > {max_position:.2f}€ ({self.MAX_POSITION_PCT*100}%)")
            return False

        return True

    def _check_max_positions(self, trade: TradeRequest, violations: List[str]) -> bool:
        """Vérifie le nombre max de positions"""
        if trade.action == 'BUY':
            if len(self._open_positions) >= self.MAX_OPEN_POSITIONS:
                if trade.symbol not in self._open_positions:
                    violations.append(f"MAX_POSITIONS: {len(self._open_positions)} >= {self.MAX_OPEN_POSITIONS}")
                    return False

        return True

    def _check_daily_trades(self, trade: TradeRequest, violations: List[str]) -> bool:
        """Vérifie le nombre max de trades quotidiens"""
        if self._daily_trades >= self.MAX_TRADES_PER_DAY:
            violations.append(f"MAX_DAILY_TRADES: {self._daily_trades} >= {self.MAX_TRADES_PER_DAY}")
            return False

        return True

    def _sanity_checks(self, trade: TradeRequest, warnings: List[str]):
        """Vérifications de cohérence (ne bloquent pas, juste des warnings)"""
        # Vérifier que le stop-loss est défini
        if trade.stop_loss is None:
            warnings.append("WARNING: Pas de stop-loss défini")

        # Vérifier que la thèse d'investissement est fournie
        if not trade.thesis:
            warnings.append("WARNING: Pas de thèse d'investissement fournie")

    # =========================================================================
    # KELLY CRITERION (V4.1)
    # =========================================================================

    def calculate_kelly_position_size(
        self,
        win_rate: float = None,
        avg_win: float = None,
        avg_loss: float = None,
        confidence_score: float = 0.5
    ) -> Dict:
        """
        Calculate optimal position size using Kelly Criterion.

        Kelly Formula: f* = (bp - q) / b
        Where:
            f* = fraction of capital to bet
            b = odds (avg_win / avg_loss)
            p = probability of winning (win_rate)
            q = probability of losing (1 - p)

        Args:
            win_rate: Historical win rate (0-1). Uses default if None.
            avg_win: Average winning trade return (%). Uses default ratio if None.
            avg_loss: Average losing trade return (%). Uses default ratio if None.
            confidence_score: Confidence in the trade (0-1). Scales Kelly.

        Returns:
            Dict with kelly_pct, adjusted_pct, position_value, reasoning
        """
        # Use defaults if not provided
        p = win_rate if win_rate is not None else self.DEFAULT_WIN_RATE
        q = 1 - p

        # Calculate odds (b = reward/risk ratio)
        if avg_win is not None and avg_loss is not None and avg_loss != 0:
            b = abs(avg_win / avg_loss)
        else:
            b = self.DEFAULT_REWARD_RISK

        # Kelly formula: f* = (bp - q) / b
        # Rearranged: f* = p - (q / b)
        kelly_fraction = (b * p - q) / b

        # Handle negative Kelly (don't trade!)
        if kelly_fraction <= 0:
            return {
                "kelly_raw": kelly_fraction,
                "kelly_pct": 0.0,
                "adjusted_pct": 0.0,
                "position_value": 0.0,
                "should_trade": False,
                "reasoning": f"Negative Kelly ({kelly_fraction:.2%}) - edge is negative, avoid trade"
            }

        # Apply fractional Kelly for safety
        kelly_adjusted = kelly_fraction * self.KELLY_FRACTION

        # Scale by confidence score
        kelly_with_confidence = kelly_adjusted * confidence_score

        # Clamp to min/max bounds
        final_pct = max(
            self.MIN_KELLY_POSITION_PCT,
            min(self.MAX_KELLY_POSITION_PCT, kelly_with_confidence)
        )

        # Calculate position value
        position_value = self.current_capital * final_pct

        return {
            "kelly_raw": kelly_fraction,
            "kelly_pct": kelly_adjusted,
            "adjusted_pct": final_pct,
            "position_value": position_value,
            "should_trade": True,
            "win_rate_used": p,
            "reward_risk_used": b,
            "confidence_applied": confidence_score,
            "reasoning": (
                f"Full Kelly: {kelly_fraction:.2%}, "
                f"Fractional ({self.KELLY_FRACTION}): {kelly_adjusted:.2%}, "
                f"With confidence ({confidence_score:.0%}): {kelly_with_confidence:.2%}, "
                f"Final (clamped): {final_pct:.2%} = {position_value:.2f}€"
            )
        }

    def get_profit_objectives(self) -> Dict:
        """Get current profit objectives and progress"""
        monthly_target = self.current_capital * self.MONTHLY_PROFIT_TARGET
        current_month_pnl = self._get_monthly_pnl()

        progress_pct = (current_month_pnl / monthly_target * 100) if monthly_target > 0 else 0

        return {
            "monthly_target_pct": self.MONTHLY_PROFIT_TARGET * 100,
            "monthly_target_value": monthly_target,
            "current_month_pnl": current_month_pnl,
            "progress_pct": progress_pct,
            "on_track": progress_pct >= (self._days_into_month() / 30 * 100),
            "min_win_rate": self.MIN_WIN_RATE_TARGET * 100,
            "min_reward_risk": self.MIN_REWARD_RISK_RATIO,
            "take_profit_target": self.DEFAULT_TAKE_PROFIT_PCT * 100
        }

    def _get_monthly_pnl(self) -> float:
        """Get P&L for current month (from state file)"""
        # For now, return daily PnL as approximation
        # TODO: Implement full monthly tracking
        return self._daily_pnl

    def _days_into_month(self) -> int:
        """Get number of days into current month"""
        today = date.today()
        return today.day

    # =========================================================================
    # KILL SWITCH
    # =========================================================================

    def _trigger_kill_switch(self, reason: str):
        """Active le kill switch - arrête tout trading"""
        self._kill_switch_active = True
        self._kill_switch_reason = reason

        # Sauvegarder l'état
        kill_data = {
            "active": True,
            "reason": reason,
            "triggered_at": datetime.now().isoformat(),
            "capital_at_trigger": self.current_capital,
            "daily_pnl_at_trigger": self._daily_pnl
        }

        os.makedirs(self.data_dir, exist_ok=True)
        with open(os.path.join(self.data_dir, "kill_switch.json"), 'w') as f:
            json.dump(kill_data, f, indent=2)

        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def _trigger_pause(self, reason: str):
        """Active une pause de 24h"""
        pause_until = datetime.now() + timedelta(hours=24)

        pause_data = {
            "active": True,
            "reason": reason,
            "pause_until": pause_until.isoformat()
        }

        os.makedirs(self.data_dir, exist_ok=True)
        with open(os.path.join(self.data_dir, "trading_pause.json"), 'w') as f:
            json.dump(pause_data, f, indent=2)

        logger.warning(f"TRADING PAUSED until {pause_until}: {reason}")

    def reset_kill_switch(self, admin_override: bool = False):
        """Reset le kill switch (nécessite confirmation manuelle)"""
        if not admin_override:
            raise GuardrailViolation(
                "KILL_SWITCH_RESET",
                "Tentative de reset sans admin_override=True",
                "CRITICAL"
            )

        self._kill_switch_active = False
        self._kill_switch_reason = ""

        kill_file = os.path.join(self.data_dir, "kill_switch.json")
        if os.path.exists(kill_file):
            os.remove(kill_file)

        logger.warning("KILL SWITCH RESET by admin override")

    def is_kill_switch_active(self) -> Tuple[bool, str]:
        """Vérifie si le kill switch est actif"""
        return self._kill_switch_active, self._kill_switch_reason

    def activate_kill_switch(self, reason: str):
        """
        Activate kill switch externally (e.g., from IBKR watchdog).

        This is the public method to trigger kill switch from outside the class.
        Use this when:
        - IBKR connection fails repeatedly
        - External system detects critical issue
        - Manual emergency stop needed

        Args:
            reason: Reason for activating kill switch
        """
        logger.critical(f"EXTERNAL KILL SWITCH ACTIVATION: {reason}")
        self._trigger_kill_switch(reason)

    # =========================================================================
    # GESTION D'ÉTAT
    # =========================================================================

    def _check_daily_reset(self):
        """Reset les compteurs quotidiens si nouveau jour"""
        today = date.today()
        if today != self._last_reset_date:
            logger.info(f"Daily reset: trades={self._daily_trades}, pnl={self._daily_pnl:.2f}€")
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._last_reset_date = today
            self._save_state()

    def record_trade(self, symbol: str, action: str, value: float, pnl: float = 0.0):
        """Enregistre un trade exécuté"""
        self._daily_trades += 1
        self._daily_pnl += pnl

        if action == 'BUY':
            self._open_positions[symbol] = self._open_positions.get(symbol, 0) + value
        elif action == 'SELL':
            if symbol in self._open_positions:
                self._open_positions[symbol] -= value
                if self._open_positions[symbol] <= 0:
                    del self._open_positions[symbol]

        self._save_state()

    def update_capital(self, new_capital: float):
        """Met à jour le capital actuel"""
        self.current_capital = new_capital
        self._save_state()

    def _save_state(self):
        """Sauvegarde l'état actuel"""
        state = {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trades,
            "last_reset_date": self._last_reset_date.isoformat(),
            "open_positions": self._open_positions,
            "kill_switch_active": self._kill_switch_active,
            "updated_at": datetime.now().isoformat()
        }

        os.makedirs(self.data_dir, exist_ok=True)
        with open(os.path.join(self.data_dir, "guardrails_state.json"), 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Charge l'état précédent"""
        state_file = os.path.join(self.data_dir, "guardrails_state.json")

        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                self._daily_pnl = state.get("daily_pnl", 0.0)
                self._daily_trades = state.get("daily_trades", 0)
                self._last_reset_date = date.fromisoformat(state.get("last_reset_date", date.today().isoformat()))
                self._open_positions = state.get("open_positions", {})
                self._kill_switch_active = state.get("kill_switch_active", False)

                logger.info(f"Guardrails state loaded: trades={self._daily_trades}, pnl={self._daily_pnl:.2f}€")
            except Exception as e:
                logger.error(f"Error loading guardrails state: {e}")

        # Vérifier aussi le fichier kill switch
        kill_file = os.path.join(self.data_dir, "kill_switch.json")
        if os.path.exists(kill_file):
            try:
                with open(kill_file, 'r') as f:
                    kill_data = json.load(f)
                if kill_data.get("active"):
                    self._kill_switch_active = True
                    self._kill_switch_reason = kill_data.get("reason", "Unknown")
                    logger.critical(f"KILL SWITCH IS ACTIVE: {self._kill_switch_reason}")
            except Exception as e:
                logger.error(f"Error loading kill switch state: {e}")

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_status(self) -> Dict:
        """Retourne le statut actuel des guardrails (V4.1)"""
        max_daily_loss = self.initial_capital * self.MAX_DAILY_LOSS_PCT
        max_position = self.current_capital * self.MAX_POSITION_PCT

        # Kelly example calculation
        kelly_example = self.calculate_kelly_position_size(
            win_rate=self.DEFAULT_WIN_RATE,
            confidence_score=0.7
        )

        return {
            "version": "4.1",
            "capital": {
                "initial": self.initial_capital,
                "current": self.current_capital,
                "drawdown_pct": (self.initial_capital - self.current_capital) / self.initial_capital
            },
            "daily": {
                "pnl": self._daily_pnl,
                "max_loss": max_daily_loss,
                "pnl_pct_of_max": abs(self._daily_pnl) / max_daily_loss if max_daily_loss > 0 else 0,
                "trades": self._daily_trades,
                "max_trades": self.MAX_TRADES_PER_DAY
            },
            "positions": {
                "open": len(self._open_positions),
                "max": self.MAX_OPEN_POSITIONS,
                "max_size": max_position,
                "details": self._open_positions
            },
            "kill_switch": {
                "active": self._kill_switch_active,
                "reason": self._kill_switch_reason
            },
            "thresholds": {
                "auto_trade": f"< {self.AUTO_TRADE_THRESHOLD*100}%",
                "notify": f"< {self.NOTIFY_THRESHOLD*100}%",
                "manual": f">= {self.MANUAL_THRESHOLD*100}%"
            },
            # V4.1: Kelly Criterion
            "kelly_criterion": {
                "fraction_used": self.KELLY_FRACTION,
                "min_position_pct": self.MIN_KELLY_POSITION_PCT * 100,
                "max_position_pct": self.MAX_KELLY_POSITION_PCT * 100,
                "example_sizing": kelly_example
            },
            # V4.1: Profit Objectives
            "profit_objectives": self.get_profit_objectives()
        }


# =========================================================================
# FACTORY
# =========================================================================

_guardrails_instance: Optional[TradingGuardrails] = None


def get_guardrails(capital: float = 1500.0) -> TradingGuardrails:
    """Retourne l'instance singleton des guardrails"""
    global _guardrails_instance

    if _guardrails_instance is None:
        _guardrails_instance = TradingGuardrails(capital=capital)

    return _guardrails_instance


# =========================================================================
# TESTS
# =========================================================================

if __name__ == "__main__":
    # Test des guardrails
    logging.basicConfig(level=logging.INFO)

    guardrails = TradingGuardrails(capital=1500.0)

    # Test 1: Trade valide (petit montant)
    trade1 = TradeRequest(
        symbol="AAPL",
        action="BUY",
        quantity=1,
        price=150.0,
        order_type="LIMIT",
        current_capital=1500.0,
        position_value=150.0,  # 10% du capital
        daily_pnl=0.0,
        current_drawdown=0.0,
        stop_loss=145.0,
        thesis="RSI breakout confirmed"
    )

    result1 = guardrails.validate_trade(trade1)
    print(f"Trade 1: {result1.status.value} - {result1.reason}")

    # Test 2: Trade trop gros
    trade2 = TradeRequest(
        symbol="NVDA",
        action="BUY",
        quantity=2,
        price=500.0,
        order_type="MARKET",
        current_capital=1500.0,
        position_value=1000.0,  # 66% du capital!
        daily_pnl=0.0,
        current_drawdown=0.0
    )

    result2 = guardrails.validate_trade(trade2)
    print(f"Trade 2: {result2.status.value} - {result2.reason}")

    # Afficher le statut
    print("\nStatut des guardrails:")
    import json
    print(json.dumps(guardrails.get_status(), indent=2))
