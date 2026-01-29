"""
WebSocket support for real-time alerts

Provides real-time streaming of:
- New alerts as they are detected
- Screening progress updates
- Price updates for watched symbols
"""
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field, asdict

from fastapi import WebSocket, WebSocketDisconnect


@dataclass
class AlertMessage:
    """Real-time alert message"""
    type: str = "alert"
    symbol: str = ""
    recommendation: str = ""
    current_price: float = 0.0
    support_level: float = 0.0
    confidence_score: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ProgressMessage:
    """Screening progress message"""
    type: str = "progress"
    total: int = 0
    completed: int = 0
    current_symbol: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PriceUpdateMessage:
    """Price update message"""
    type: str = "price_update"
    symbol: str = ""
    price: float = 0.0
    change_pct: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ConnectionManager:
    """
    Manages WebSocket connections and broadcasting.

    Supports:
    - Multiple client connections
    - Topic-based subscriptions (alerts, progress, prices)
    - Broadcasting to specific topics or all clients
    """

    def __init__(self):
        # Active connections by client ID
        self.active_connections: Dict[str, WebSocket] = {}

        # Subscriptions: topic -> set of client IDs
        self.subscriptions: Dict[str, Set[str]] = {
            "alerts": set(),
            "progress": set(),
            "prices": set(),
            "all": set()
        }

        # Price watch list: symbol -> set of client IDs
        self.price_watchers: Dict[str, Set[str]] = {}

        self._client_counter = 0

    def _generate_client_id(self) -> str:
        """Generate unique client ID"""
        self._client_counter += 1
        return f"client_{self._client_counter}_{datetime.now().strftime('%H%M%S')}"

    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept new WebSocket connection.

        Returns:
            Client ID for this connection
        """
        await websocket.accept()
        client_id = self._generate_client_id()
        self.active_connections[client_id] = websocket

        # Subscribe to alerts by default
        self.subscriptions["alerts"].add(client_id)

        # Send welcome message
        await self.send_personal_message(
            client_id,
            {
                "type": "connected",
                "client_id": client_id,
                "message": "Connected to TradingBot WebSocket",
                "subscriptions": ["alerts"]
            }
        )

        return client_id

    def disconnect(self, client_id: str):
        """Remove client from all connections and subscriptions"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]

        # Remove from all subscriptions
        for topic in self.subscriptions.values():
            topic.discard(client_id)

        # Remove from price watchers
        for watchers in self.price_watchers.values():
            watchers.discard(client_id)

    async def subscribe(self, client_id: str, topics: List[str]):
        """Subscribe client to topics"""
        for topic in topics:
            if topic in self.subscriptions:
                self.subscriptions[topic].add(client_id)

        await self.send_personal_message(
            client_id,
            {
                "type": "subscribed",
                "topics": topics
            }
        )

    async def unsubscribe(self, client_id: str, topics: List[str]):
        """Unsubscribe client from topics"""
        for topic in topics:
            if topic in self.subscriptions:
                self.subscriptions[topic].discard(client_id)

        await self.send_personal_message(
            client_id,
            {
                "type": "unsubscribed",
                "topics": topics
            }
        )

    async def watch_symbol(self, client_id: str, symbol: str):
        """Add symbol to client's price watch list"""
        symbol = symbol.upper()
        if symbol not in self.price_watchers:
            self.price_watchers[symbol] = set()
        self.price_watchers[symbol].add(client_id)

        await self.send_personal_message(
            client_id,
            {
                "type": "watching",
                "symbol": symbol
            }
        )

    async def unwatch_symbol(self, client_id: str, symbol: str):
        """Remove symbol from client's price watch list"""
        symbol = symbol.upper()
        if symbol in self.price_watchers:
            self.price_watchers[symbol].discard(client_id)

    async def send_personal_message(self, client_id: str, message: dict):
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_json(message)
            except Exception:
                self.disconnect(client_id)

    async def broadcast_to_topic(self, topic: str, message: dict):
        """Broadcast message to all clients subscribed to a topic"""
        client_ids = self.subscriptions.get(topic, set()) | self.subscriptions.get("all", set())

        for client_id in list(client_ids):
            await self.send_personal_message(client_id, message)

    async def broadcast_alert(self, alert: AlertMessage):
        """Broadcast new alert to subscribers"""
        message = asdict(alert)
        await self.broadcast_to_topic("alerts", message)

    async def broadcast_progress(self, progress: ProgressMessage):
        """Broadcast screening progress to subscribers"""
        message = asdict(progress)
        await self.broadcast_to_topic("progress", message)

    async def broadcast_price_update(self, update: PriceUpdateMessage):
        """Broadcast price update to watchers of that symbol"""
        symbol = update.symbol.upper()
        message = asdict(update)

        # Send to symbol watchers
        if symbol in self.price_watchers:
            for client_id in list(self.price_watchers[symbol]):
                await self.send_personal_message(client_id, message)

        # Also send to 'prices' topic subscribers
        await self.broadcast_to_topic("prices", message)

    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all connected clients"""
        for client_id in list(self.active_connections.keys()):
            await self.send_personal_message(client_id, message)

    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)

    def get_client_subscriptions(self, client_id: str) -> List[str]:
        """Get topics a client is subscribed to"""
        return [
            topic for topic, clients in self.subscriptions.items()
            if client_id in clients
        ]


# Global connection manager instance
connection_manager = ConnectionManager()


async def handle_websocket_message(client_id: str, message: dict, manager: ConnectionManager):
    """
    Handle incoming WebSocket message from client.

    Supported commands:
    - subscribe: {"action": "subscribe", "topics": ["alerts", "progress"]}
    - unsubscribe: {"action": "unsubscribe", "topics": ["progress"]}
    - watch: {"action": "watch", "symbol": "AAPL"}
    - unwatch: {"action": "unwatch", "symbol": "AAPL"}
    - ping: {"action": "ping"}
    """
    action = message.get("action", "").lower()

    if action == "subscribe":
        topics = message.get("topics", [])
        await manager.subscribe(client_id, topics)

    elif action == "unsubscribe":
        topics = message.get("topics", [])
        await manager.unsubscribe(client_id, topics)

    elif action == "watch":
        symbol = message.get("symbol", "")
        if symbol:
            await manager.watch_symbol(client_id, symbol)

    elif action == "unwatch":
        symbol = message.get("symbol", "")
        if symbol:
            await manager.unwatch_symbol(client_id, symbol)

    elif action == "ping":
        await manager.send_personal_message(
            client_id,
            {"type": "pong", "timestamp": datetime.now().isoformat()}
        )

    elif action == "status":
        await manager.send_personal_message(
            client_id,
            {
                "type": "status",
                "client_id": client_id,
                "subscriptions": manager.get_client_subscriptions(client_id),
                "total_connections": manager.get_connection_count()
            }
        )

    else:
        await manager.send_personal_message(
            client_id,
            {
                "type": "error",
                "message": f"Unknown action: {action}",
                "valid_actions": ["subscribe", "unsubscribe", "watch", "unwatch", "ping", "status"]
            }
        )
