"""
IBKR Executor Module
Exécution des ordres via Interactive Brokers (IBKR).

Ce module gère:
1. Connexion à TWS/IB Gateway via ib_insync
2. Passage d'ordres (market, limit, stop)
3. Gestion des positions
4. Récupération des données de marché

IMPORTANT: Trading RÉEL! Les guardrails DOIVENT être respectés.

Prérequis:
1. Compte IBKR (Pro ou Lite)
2. TWS ou IB Gateway installé et configuré
3. API activée dans TWS (Edit > Global Configuration > API)
4. pip install ib_insync

Usage:
    from src.execution.ibkr_executor import IBKRExecutor

    executor = IBKRExecutor(port=7497)  # 7497=TWS Paper, 7496=TWS Live
    await executor.connect()

    # Passer un ordre
    result = await executor.place_order(
        symbol='AAPL',
        quantity=10,
        order_type='limit',
        limit_price=175.50,
        stop_loss=168.00
    )

    # Obtenir les positions
    positions = await executor.get_positions()
"""

import os
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json

# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class OrderType(Enum):
    """Types d'ordres supportés"""
    MARKET = 'MKT'
    LIMIT = 'LMT'
    STOP = 'STP'
    STOP_LIMIT = 'STP LMT'
    TRAILING_STOP = 'TRAIL'


class OrderStatus(Enum):
    """Statuts d'ordres"""
    PENDING = 'pending'
    SUBMITTED = 'submitted'
    FILLED = 'filled'
    PARTIALLY_FILLED = 'partial'
    CANCELLED = 'cancelled'
    REJECTED = 'rejected'
    ERROR = 'error'


class OrderAction(Enum):
    """Actions d'ordres"""
    BUY = 'BUY'
    SELL = 'SELL'


@dataclass
class OrderRequest:
    """Requête d'ordre à passer"""
    symbol: str
    action: OrderAction
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trailing_percent: Optional[float] = None
    time_in_force: str = 'DAY'  # DAY, GTC, IOC, FOK
    outside_rth: bool = False  # Autoriser hors heures de marché

    # Bracket order (stop loss + take profit)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Métadonnées
    signal_source: str = ''
    confidence_score: float = 0.0
    notes: str = ''


@dataclass
class OrderResult:
    """Résultat d'un ordre passé"""
    order_id: int
    symbol: str
    action: str
    quantity: int
    status: OrderStatus
    fill_price: Optional[float] = None
    fill_quantity: int = 0
    commission: float = 0.0
    message: str = ''
    submitted_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None

    # Ordres liés (stop loss, take profit)
    stop_loss_order_id: Optional[int] = None
    take_profit_order_id: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'action': self.action,
            'quantity': self.quantity,
            'status': self.status.value,
            'fill_price': self.fill_price,
            'fill_quantity': self.fill_quantity,
            'commission': self.commission,
            'message': self.message,
            'submitted_at': self.submitted_at.isoformat(),
            'filled_at': self.filled_at.isoformat() if self.filled_at else None
        }


@dataclass
class Position:
    """Position ouverte"""
    symbol: str
    quantity: int  # Positif = long, négatif = short
    avg_cost: float
    market_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_cost': self.avg_cost,
            'market_price': self.market_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl
        }


@dataclass
class AccountInfo:
    """Informations du compte"""
    account_id: str
    net_liquidation: float
    total_cash: float
    buying_power: float
    gross_position_value: float
    unrealized_pnl: float
    realized_pnl: float
    cushion: float  # Marge de sécurité

    def to_dict(self) -> Dict:
        return {
            'account_id': self.account_id,
            'net_liquidation': self.net_liquidation,
            'total_cash': self.total_cash,
            'buying_power': self.buying_power,
            'gross_position_value': self.gross_position_value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'cushion': self.cushion
        }


# =============================================================================
# IBKR EXECUTOR
# =============================================================================

class IBKRExecutor:
    """
    Exécuteur d'ordres via Interactive Brokers.

    Utilise ib_insync pour communiquer avec TWS/IB Gateway.

    Configuration TWS requise:
    1. Edit > Global Configuration > API > Settings
    2. Cocher "Enable ActiveX and Socket Clients"
    3. Décocher "Read-Only API"
    4. Port: 7497 (paper) ou 7496 (live)

    IMPORTANT: Ce module exécute des ordres RÉELS!
    Toujours valider via les guardrails avant d'appeler place_order().
    """

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 7497,  # 7497=TWS Paper, 7496=TWS Live, 4001=Gateway Paper, 4002=Gateway Live
        client_id: int = 1,
        readonly: bool = False
    ):
        """
        Initialise l'exécuteur IBKR.

        Args:
            host: Hôte TWS/Gateway (généralement localhost)
            port: Port TWS/Gateway
                - 7497: TWS Paper Trading
                - 7496: TWS Live
                - 4001: IB Gateway Paper
                - 4002: IB Gateway Live
            client_id: ID client unique (changer si plusieurs connexions)
            readonly: Mode lecture seule (pas d'ordres)
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.readonly = readonly

        # Instance ib_insync (chargée dynamiquement)
        self.ib = None
        self._connected = False

        # Cache des ordres en cours
        self._pending_orders: Dict[int, OrderResult] = {}
        self._filled_orders: List[OrderResult] = []

        # Callbacks pour les events
        self._order_callbacks = []
        self._position_callbacks = []

        # Persistance
        self._data_dir = Path("data/execution")
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Historique des ordres
        self._order_history: List[Dict] = []

    async def connect(self, timeout: int = 30) -> bool:
        """
        Connecte à TWS/IB Gateway.

        Args:
            timeout: Timeout de connexion en secondes

        Returns:
            True si connecté, False sinon
        """
        try:
            # Import dynamique pour éviter les erreurs si ib_insync n'est pas installé
            from ib_insync import IB

            self.ib = IB()

            # Connecter
            logger.info(f"Connecting to IBKR at {self.host}:{self.port}...")

            await self.ib.connectAsync(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                readonly=self.readonly,
                timeout=timeout
            )

            if self.ib.isConnected():
                self._connected = True

                # Enregistrer les callbacks
                self.ib.orderStatusEvent += self._on_order_status
                self.ib.execDetailsEvent += self._on_execution

                # Log account info
                account_info = await self.get_account_info()
                if account_info:
                    logger.info(
                        f"Connected to IBKR account {account_info.account_id} | "
                        f"Net Liq: ${account_info.net_liquidation:,.2f}"
                    )

                # Déterminer si paper ou live
                if self.port in [7497, 4001]:
                    logger.warning("⚠️ PAPER TRADING MODE")
                else:
                    logger.warning("🔴 LIVE TRADING MODE - REAL MONEY!")

                return True
            else:
                logger.error("Failed to connect to IBKR")
                return False

        except ImportError:
            logger.error(
                "ib_insync not installed. Install with: pip install ib_insync"
            )
            return False

        except Exception as e:
            logger.error(f"IBKR connection error: {e}")
            return False

    async def disconnect(self):
        """Déconnecte de TWS/Gateway"""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    def is_connected(self) -> bool:
        """Vérifie si connecté"""
        return self._connected and self.ib and self.ib.isConnected()

    async def get_account_info(self) -> Optional[AccountInfo]:
        """
        Récupère les informations du compte.

        Returns:
            AccountInfo ou None si erreur
        """
        if not self.is_connected():
            logger.warning("Not connected to IBKR")
            return None

        try:
            # Récupérer les valeurs du compte
            account_values = self.ib.accountValues()

            # Parser les valeurs
            values = {}
            for av in account_values:
                if av.currency == 'USD' or av.currency == 'EUR':
                    values[av.tag] = float(av.value) if av.value else 0

            return AccountInfo(
                account_id=self.ib.managedAccounts()[0] if self.ib.managedAccounts() else 'Unknown',
                net_liquidation=values.get('NetLiquidation', 0),
                total_cash=values.get('TotalCashValue', 0),
                buying_power=values.get('BuyingPower', 0),
                gross_position_value=values.get('GrossPositionValue', 0),
                unrealized_pnl=values.get('UnrealizedPnL', 0),
                realized_pnl=values.get('RealizedPnL', 0),
                cushion=values.get('Cushion', 0)
            )

        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None

    async def get_positions(self) -> List[Position]:
        """
        Récupère toutes les positions ouvertes.

        Returns:
            Liste des positions
        """
        if not self.is_connected():
            logger.warning("Not connected to IBKR")
            return []

        try:
            positions = []
            ib_positions = self.ib.positions()

            for pos in ib_positions:
                positions.append(Position(
                    symbol=pos.contract.symbol,
                    quantity=int(pos.position),
                    avg_cost=pos.avgCost,
                    market_price=pos.marketPrice if hasattr(pos, 'marketPrice') else 0,
                    market_value=pos.marketValue if hasattr(pos, 'marketValue') else 0,
                    unrealized_pnl=pos.unrealizedPNL if hasattr(pos, 'unrealizedPNL') else 0,
                    realized_pnl=pos.realizedPNL if hasattr(pos, 'realizedPNL') else 0
                ))

            return positions

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    async def get_position(self, symbol: str) -> Optional[Position]:
        """Récupère une position spécifique"""
        positions = await self.get_positions()
        for pos in positions:
            if pos.symbol == symbol:
                return pos
        return None

    async def place_order(
        self,
        request: OrderRequest,
        validate_only: bool = False
    ) -> OrderResult:
        """
        Place un ordre sur IBKR.

        IMPORTANT: Cette méthode exécute des ordres RÉELS!
        Assurez-vous que les guardrails ont validé l'ordre AVANT d'appeler.

        Args:
            request: Requête d'ordre
            validate_only: Si True, valide seulement sans passer l'ordre

        Returns:
            Résultat de l'ordre
        """
        if not self.is_connected():
            return OrderResult(
                order_id=-1,
                symbol=request.symbol,
                action=request.action.value,
                quantity=request.quantity,
                status=OrderStatus.ERROR,
                message="Not connected to IBKR"
            )

        if self.readonly:
            return OrderResult(
                order_id=-1,
                symbol=request.symbol,
                action=request.action.value,
                quantity=request.quantity,
                status=OrderStatus.REJECTED,
                message="Executor is in readonly mode"
            )

        try:
            from ib_insync import Stock, Order, MarketOrder, LimitOrder, StopOrder

            # Créer le contrat
            contract = Stock(request.symbol, 'SMART', 'USD')

            # Qualifier le contrat (vérifier qu'il existe)
            await self.ib.qualifyContractsAsync(contract)

            # Créer l'ordre selon le type
            if request.order_type == OrderType.MARKET:
                order = MarketOrder(
                    action=request.action.value,
                    totalQuantity=request.quantity,
                    tif=request.time_in_force,
                    outsideRth=request.outside_rth
                )

            elif request.order_type == OrderType.LIMIT:
                if not request.limit_price:
                    return OrderResult(
                        order_id=-1,
                        symbol=request.symbol,
                        action=request.action.value,
                        quantity=request.quantity,
                        status=OrderStatus.ERROR,
                        message="Limit price required for limit order"
                    )

                order = LimitOrder(
                    action=request.action.value,
                    totalQuantity=request.quantity,
                    lmtPrice=request.limit_price,
                    tif=request.time_in_force,
                    outsideRth=request.outside_rth
                )

            elif request.order_type == OrderType.STOP:
                if not request.stop_price:
                    return OrderResult(
                        order_id=-1,
                        symbol=request.symbol,
                        action=request.action.value,
                        quantity=request.quantity,
                        status=OrderStatus.ERROR,
                        message="Stop price required for stop order"
                    )

                order = StopOrder(
                    action=request.action.value,
                    totalQuantity=request.quantity,
                    stopPrice=request.stop_price,
                    tif=request.time_in_force,
                    outsideRth=request.outside_rth
                )

            else:
                return OrderResult(
                    order_id=-1,
                    symbol=request.symbol,
                    action=request.action.value,
                    quantity=request.quantity,
                    status=OrderStatus.ERROR,
                    message=f"Unsupported order type: {request.order_type}"
                )

            # Validation seulement?
            if validate_only:
                # Vérifier avec whatIfOrder
                what_if = await self.ib.whatIfOrderAsync(contract, order)
                if what_if:
                    return OrderResult(
                        order_id=0,
                        symbol=request.symbol,
                        action=request.action.value,
                        quantity=request.quantity,
                        status=OrderStatus.PENDING,
                        commission=float(what_if.commission) if what_if.commission else 0,
                        message=f"Validation OK - Est. commission: ${what_if.commission}"
                    )

            # Passer l'ordre
            trade = self.ib.placeOrder(contract, order)

            # Créer le résultat
            result = OrderResult(
                order_id=trade.order.orderId,
                symbol=request.symbol,
                action=request.action.value,
                quantity=request.quantity,
                status=OrderStatus.SUBMITTED,
                message="Order submitted"
            )

            # Stocker dans les ordres en cours
            self._pending_orders[trade.order.orderId] = result

            # Log
            logger.info(
                f"Order submitted: {request.action.value} {request.quantity} {request.symbol} "
                f"@ {request.limit_price or 'MKT'} | ID: {trade.order.orderId}"
            )

            # Si bracket order, passer stop loss et take profit
            if request.stop_loss or request.take_profit:
                await self._place_bracket_orders(
                    request, trade.order.orderId, contract
                )

            # Sauvegarder l'historique
            self._save_order_to_history(request, result)

            return result

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return OrderResult(
                order_id=-1,
                symbol=request.symbol,
                action=request.action.value,
                quantity=request.quantity,
                status=OrderStatus.ERROR,
                message=str(e)
            )

    async def _place_bracket_orders(
        self,
        request: OrderRequest,
        parent_order_id: int,
        contract
    ):
        """Place les ordres stop loss et take profit pour un bracket"""
        from ib_insync import StopOrder, LimitOrder

        # Déterminer la direction
        is_buy = request.action == OrderAction.BUY

        # Stop Loss
        if request.stop_loss:
            sl_action = 'SELL' if is_buy else 'BUY'
            sl_order = StopOrder(
                action=sl_action,
                totalQuantity=request.quantity,
                stopPrice=request.stop_loss,
                parentId=parent_order_id,
                tif='GTC'  # Good Till Cancelled
            )
            sl_trade = self.ib.placeOrder(contract, sl_order)
            logger.info(f"Stop loss order placed: {request.stop_loss} | ID: {sl_trade.order.orderId}")

        # Take Profit
        if request.take_profit:
            tp_action = 'SELL' if is_buy else 'BUY'
            tp_order = LimitOrder(
                action=tp_action,
                totalQuantity=request.quantity,
                lmtPrice=request.take_profit,
                parentId=parent_order_id,
                tif='GTC'
            )
            tp_trade = self.ib.placeOrder(contract, tp_order)
            logger.info(f"Take profit order placed: {request.take_profit} | ID: {tp_trade.order.orderId}")

    async def cancel_order(self, order_id: int) -> bool:
        """
        Annule un ordre.

        Args:
            order_id: ID de l'ordre à annuler

        Returns:
            True si annulé, False sinon
        """
        if not self.is_connected():
            return False

        try:
            # Trouver le trade correspondant
            for trade in self.ib.trades():
                if trade.order.orderId == order_id:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Order {order_id} cancelled")
                    return True

            logger.warning(f"Order {order_id} not found")
            return False

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    async def cancel_all_orders(self) -> int:
        """
        Annule tous les ordres ouverts.

        Returns:
            Nombre d'ordres annulés
        """
        if not self.is_connected():
            return 0

        try:
            open_orders = self.ib.openOrders()
            count = 0

            for order in open_orders:
                try:
                    self.ib.cancelOrder(order)
                    count += 1
                except Exception as e:
                    logger.debug(f"Failed to cancel order: {e}")

            logger.info(f"Cancelled {count} orders")
            return count

        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return 0

    async def close_position(
        self,
        symbol: str,
        order_type: OrderType = OrderType.MARKET
    ) -> Optional[OrderResult]:
        """
        Ferme une position.

        Args:
            symbol: Symbole de la position
            order_type: Type d'ordre (MARKET recommandé pour fermeture immédiate)

        Returns:
            Résultat de l'ordre de fermeture
        """
        position = await self.get_position(symbol)

        if not position or position.quantity == 0:
            logger.warning(f"No position found for {symbol}")
            return None

        # Déterminer l'action (inverse de la position)
        action = OrderAction.SELL if position.quantity > 0 else OrderAction.BUY

        request = OrderRequest(
            symbol=symbol,
            action=action,
            quantity=abs(position.quantity),
            order_type=order_type,
            signal_source='close_position'
        )

        return await self.place_order(request)

    async def close_all_positions(self) -> List[OrderResult]:
        """
        Ferme toutes les positions.

        Returns:
            Liste des résultats d'ordres
        """
        results = []
        positions = await self.get_positions()

        for pos in positions:
            if pos.quantity != 0:
                result = await self.close_position(pos.symbol)
                if result:
                    results.append(result)

        logger.info(f"Closed {len(results)} positions")
        return results

    async def get_market_price(self, symbol: str) -> Optional[float]:
        """
        Récupère le prix de marché actuel.

        Args:
            symbol: Symbole

        Returns:
            Prix ou None
        """
        if not self.is_connected():
            return None

        try:
            from ib_insync import Stock

            contract = Stock(symbol, 'SMART', 'USD')
            await self.ib.qualifyContractsAsync(contract)

            # Demander les données de marché
            ticker = self.ib.reqMktData(contract)
            await asyncio.sleep(1)  # Attendre les données

            price = ticker.last or ticker.close or ticker.bid

            # Annuler la souscription
            self.ib.cancelMktData(contract)

            return float(price) if price else None

        except Exception as e:
            logger.error(f"Error getting market price for {symbol}: {e}")
            return None

    def _on_order_status(self, trade):
        """Callback quand le statut d'un ordre change"""
        order_id = trade.order.orderId

        if order_id in self._pending_orders:
            result = self._pending_orders[order_id]

            # Mettre à jour le statut
            status_map = {
                'Submitted': OrderStatus.SUBMITTED,
                'Filled': OrderStatus.FILLED,
                'Cancelled': OrderStatus.CANCELLED,
                'Inactive': OrderStatus.REJECTED,
                'PendingSubmit': OrderStatus.PENDING,
                'PreSubmitted': OrderStatus.PENDING
            }

            result.status = status_map.get(trade.orderStatus.status, OrderStatus.PENDING)

            if result.status == OrderStatus.FILLED:
                result.fill_price = trade.orderStatus.avgFillPrice
                result.fill_quantity = int(trade.orderStatus.filled)
                result.filled_at = datetime.now()

                # Déplacer vers filled
                self._filled_orders.append(result)
                del self._pending_orders[order_id]

                logger.info(
                    f"Order {order_id} FILLED: {result.quantity} {result.symbol} "
                    f"@ ${result.fill_price:.2f}"
                )

    def _on_execution(self, trade, fill):
        """Callback quand un ordre est exécuté"""
        logger.info(
            f"Execution: {fill.execution.side} {fill.execution.shares} "
            f"{trade.contract.symbol} @ ${fill.execution.price:.2f}"
        )

    def _save_order_to_history(self, request: OrderRequest, result: OrderResult):
        """Sauvegarde un ordre dans l'historique"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'request': {
                'symbol': request.symbol,
                'action': request.action.value,
                'quantity': request.quantity,
                'order_type': request.order_type.value,
                'limit_price': request.limit_price,
                'stop_loss': request.stop_loss,
                'take_profit': request.take_profit,
                'signal_source': request.signal_source,
                'confidence_score': request.confidence_score
            },
            'result': result.to_dict()
        }

        self._order_history.append(entry)

        # Sauvegarder en fichier
        try:
            history_file = self._data_dir / "order_history.json"
            with open(history_file, 'w') as f:
                json.dump(self._order_history[-1000:], f, indent=2)  # Garder 1000 derniers
        except Exception as e:
            logger.error(f"Error saving order history: {e}")

    def get_pending_orders(self) -> List[OrderResult]:
        """Retourne les ordres en attente"""
        return list(self._pending_orders.values())

    def get_filled_orders(self) -> List[OrderResult]:
        """Retourne les ordres exécutés"""
        return self._filled_orders.copy()


# =============================================================================
# MOCK EXECUTOR (pour tests sans IBKR)
# =============================================================================

class MockIBKRExecutor(IBKRExecutor):
    """
    Version mock pour les tests sans connexion IBKR réelle.

    Simule les comportements de base sans passer d'ordres réels.
    """

    def __init__(self, initial_cash: float = 10000.0):
        super().__init__()
        self._mock_cash = initial_cash
        self._mock_positions: Dict[str, Position] = {}
        self._mock_order_id = 1000

    async def connect(self, timeout: int = 30) -> bool:
        """Simule une connexion"""
        logger.info("MockIBKRExecutor: Simulated connection")
        self._connected = True
        return True

    async def disconnect(self):
        """Simule une déconnexion"""
        self._connected = False
        logger.info("MockIBKRExecutor: Disconnected")

    def is_connected(self) -> bool:
        return self._connected

    async def get_account_info(self) -> Optional[AccountInfo]:
        """Retourne des infos de compte simulées"""
        total_position_value = sum(
            p.quantity * p.market_price
            for p in self._mock_positions.values()
        )

        return AccountInfo(
            account_id='MOCK_ACCOUNT',
            net_liquidation=self._mock_cash + total_position_value,
            total_cash=self._mock_cash,
            buying_power=self._mock_cash,
            gross_position_value=total_position_value,
            unrealized_pnl=0,
            realized_pnl=0,
            cushion=0.5
        )

    async def get_positions(self) -> List[Position]:
        """Retourne les positions simulées"""
        return list(self._mock_positions.values())

    async def get_position(self, symbol: str) -> Optional[Position]:
        return self._mock_positions.get(symbol)

    async def place_order(
        self,
        request: OrderRequest,
        validate_only: bool = False
    ) -> OrderResult:
        """Simule le passage d'un ordre"""
        self._mock_order_id += 1
        order_id = self._mock_order_id

        # Simuler un prix de remplissage
        fill_price = request.limit_price or 100.0  # Prix par défaut

        # Simuler le coût
        cost = request.quantity * fill_price

        if request.action == OrderAction.BUY:
            if cost > self._mock_cash:
                return OrderResult(
                    order_id=order_id,
                    symbol=request.symbol,
                    action=request.action.value,
                    quantity=request.quantity,
                    status=OrderStatus.REJECTED,
                    message="Insufficient funds"
                )

            self._mock_cash -= cost

            # Ajouter ou mettre à jour la position
            if request.symbol in self._mock_positions:
                pos = self._mock_positions[request.symbol]
                new_qty = pos.quantity + request.quantity
                new_cost = (pos.avg_cost * pos.quantity + cost) / new_qty
                pos.quantity = new_qty
                pos.avg_cost = new_cost
            else:
                self._mock_positions[request.symbol] = Position(
                    symbol=request.symbol,
                    quantity=request.quantity,
                    avg_cost=fill_price,
                    market_price=fill_price,
                    market_value=cost,
                    unrealized_pnl=0,
                    realized_pnl=0
                )

        else:  # SELL
            if request.symbol not in self._mock_positions:
                return OrderResult(
                    order_id=order_id,
                    symbol=request.symbol,
                    action=request.action.value,
                    quantity=request.quantity,
                    status=OrderStatus.REJECTED,
                    message="No position to sell"
                )

            pos = self._mock_positions[request.symbol]
            if request.quantity > pos.quantity:
                return OrderResult(
                    order_id=order_id,
                    symbol=request.symbol,
                    action=request.action.value,
                    quantity=request.quantity,
                    status=OrderStatus.REJECTED,
                    message="Insufficient shares"
                )

            # Vendre
            self._mock_cash += cost
            pos.quantity -= request.quantity

            if pos.quantity == 0:
                del self._mock_positions[request.symbol]

        result = OrderResult(
            order_id=order_id,
            symbol=request.symbol,
            action=request.action.value,
            quantity=request.quantity,
            status=OrderStatus.FILLED,
            fill_price=fill_price,
            fill_quantity=request.quantity,
            commission=1.0,  # Commission fixe simulée
            message="Order filled (simulated)",
            filled_at=datetime.now()
        )

        logger.info(
            f"[MOCK] Order filled: {request.action.value} {request.quantity} "
            f"{request.symbol} @ ${fill_price:.2f}"
        )

        return result

    async def get_market_price(self, symbol: str) -> Optional[float]:
        """Retourne un prix simulé"""
        if symbol in self._mock_positions:
            return self._mock_positions[symbol].market_price
        return 100.0  # Prix par défaut


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_executor_instance: Optional[IBKRExecutor] = None


async def get_ibkr_executor(
    use_mock: bool = False,
    port: int = 7497,
    initial_cash: float = 10000.0
) -> IBKRExecutor:
    """
    Factory pour obtenir une instance de l'exécuteur IBKR.

    Args:
        use_mock: Utiliser le mock (pas de vraie connexion)
        port: Port TWS (7497=paper, 7496=live)
        initial_cash: Cash initial pour le mock

    Returns:
        Instance de l'exécuteur
    """
    global _executor_instance

    if _executor_instance is None:
        if use_mock:
            _executor_instance = MockIBKRExecutor(initial_cash=initial_cash)
        else:
            _executor_instance = IBKRExecutor(port=port)

        await _executor_instance.connect()

    return _executor_instance


async def close_ibkr_executor():
    """Ferme l'exécuteur global"""
    global _executor_instance

    if _executor_instance:
        await _executor_instance.disconnect()
        _executor_instance = None


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        print("=== IBKR Executor Test ===\n")

        # Utiliser le mock pour le test
        executor = MockIBKRExecutor(initial_cash=10000.0)
        await executor.connect()

        try:
            # Afficher les infos du compte
            account = await executor.get_account_info()
            print(f"Account: {account.account_id}")
            print(f"Cash: ${account.total_cash:,.2f}")
            print(f"Net Liquidation: ${account.net_liquidation:,.2f}")

            # Passer un ordre d'achat
            print("\n--- Placing buy order ---")
            request = OrderRequest(
                symbol='AAPL',
                action=OrderAction.BUY,
                quantity=10,
                order_type=OrderType.LIMIT,
                limit_price=175.50,
                stop_loss=168.00
            )
            result = await executor.place_order(request)
            print(f"Order result: {result.status.value}")
            print(f"Fill price: ${result.fill_price:.2f}")

            # Vérifier les positions
            print("\n--- Positions ---")
            positions = await executor.get_positions()
            for pos in positions:
                print(f"  {pos.symbol}: {pos.quantity} shares @ ${pos.avg_cost:.2f}")

            # Vérifier le cash restant
            account = await executor.get_account_info()
            print(f"\nCash after order: ${account.total_cash:,.2f}")

            # Vendre la position
            print("\n--- Closing position ---")
            result = await executor.close_position('AAPL')
            print(f"Close result: {result.status.value}")

            # Cash final
            account = await executor.get_account_info()
            print(f"\nFinal cash: ${account.total_cash:,.2f}")

        finally:
            await executor.disconnect()

    asyncio.run(main())
