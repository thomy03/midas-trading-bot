"""
Reasoning Graph - Neural network-like structure for visualizing and learning from decision chains.

This module models the trading bot's reasoning as a directed graph where:
- Nodes = Decision points (sources, analyses, signals)
- Edges = Logical connections with weights (importance/confidence)
- Paths = Complete reasoning chains leading to buy/sell/reject decisions

The graph can be:
1. Visualized as a neural network diagram
2. Converted to vector embeddings for ML learning
3. Analyzed to understand which paths lead to successful trades
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the reasoning graph"""
    # Input layer (sources)
    SOURCE_REDDIT = "source_reddit"
    SOURCE_STOCKTWITS = "source_stocktwits"
    SOURCE_GROK = "source_grok"
    SOURCE_NEWS = "source_news"
    SOURCE_VOLUME = "source_volume"
    SOURCE_WATCHLIST = "source_watchlist"

    # Hidden layer 1 (detection/filtering)
    DETECTION_SOCIAL = "detection_social"
    DETECTION_SENTIMENT = "detection_sentiment"
    DETECTION_CATALYST = "detection_catalyst"
    DETECTION_ANOMALY = "detection_anomaly"

    # Hidden layer 2 (pillar analysis)
    PILLAR_TECHNICAL = "pillar_technical"
    PILLAR_FUNDAMENTAL = "pillar_fundamental"
    PILLAR_SENTIMENT = "pillar_sentiment"
    PILLAR_NEWS = "pillar_news"

    # Hidden layer 3 (aggregation)
    AGGREGATION_SCORE = "aggregation_score"
    AGGREGATION_CONVERGENCE = "aggregation_convergence"
    AGGREGATION_RISK = "aggregation_risk"

    # Output layer
    DECISION_BUY = "decision_buy"
    DECISION_WATCH = "decision_watch"
    DECISION_REJECT = "decision_reject"


@dataclass
class ReasoningNode:
    """A node in the reasoning graph (like a neuron)"""
    id: str                          # Unique ID
    node_type: NodeType              # Type of node
    label: str                       # Human-readable label
    value: float                     # Activation value [0, 1]
    confidence: float                # Confidence in this value [0, 1]
    reasoning: str                   # Why this value
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Vector embedding (20 dimensions like AdaptiveScorer)
    embedding: List[float] = field(default_factory=lambda: [0.0] * 20)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['node_type'] = self.node_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'ReasoningNode':
        data['node_type'] = NodeType(data['node_type'])
        return cls(**data)


@dataclass
class ReasoningEdge:
    """An edge connecting two nodes (like a synapse)"""
    source_id: str                   # Source node ID
    target_id: str                   # Target node ID
    weight: float                    # Connection strength [0, 1]
    reasoning: str                   # Why this connection matters
    contribution: float              # How much this edge contributed to target

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ReasoningGraph:
    """Complete reasoning graph for a symbol analysis"""
    symbol: str
    timestamp: str
    nodes: List[ReasoningNode] = field(default_factory=list)
    edges: List[ReasoningEdge] = field(default_factory=list)
    outcome: Optional[str] = None    # "profit", "loss", "pending"
    outcome_pct: float = 0.0         # Actual P&L percentage

    def add_node(self, node: ReasoningNode):
        """Add a node to the graph"""
        self.nodes.append(node)

    def add_edge(self, edge: ReasoningEdge):
        """Add an edge to the graph"""
        self.edges.append(edge)

    def connect(self, source_id: str, target_id: str, weight: float,
                reasoning: str = "", contribution: float = 0.0):
        """Create a connection between two nodes"""
        self.add_edge(ReasoningEdge(
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            reasoning=reasoning,
            contribution=contribution
        ))

    def get_node(self, node_id: str) -> Optional[ReasoningNode]:
        """Get a node by ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_nodes_by_type(self, node_type: NodeType) -> List[ReasoningNode]:
        """Get all nodes of a specific type"""
        return [n for n in self.nodes if n.node_type == node_type]

    def get_incoming_edges(self, node_id: str) -> List[ReasoningEdge]:
        """Get all edges pointing to a node"""
        return [e for e in self.edges if e.target_id == node_id]

    def get_outgoing_edges(self, node_id: str) -> List[ReasoningEdge]:
        """Get all edges from a node"""
        return [e for e in self.edges if e.source_id == node_id]

    def get_path_to_decision(self) -> List[List[str]]:
        """Get all paths from sources to decision"""
        # Find decision nodes
        decision_nodes = [n for n in self.nodes if n.node_type.value.startswith('decision_')]
        if not decision_nodes:
            return []

        # Find source nodes
        source_nodes = [n for n in self.nodes if n.node_type.value.startswith('source_')]

        # BFS to find all paths
        paths = []
        for source in source_nodes:
            for decision in decision_nodes:
                path = self._find_path(source.id, decision.id)
                if path:
                    paths.append(path)

        return paths

    def _find_path(self, start_id: str, end_id: str) -> List[str]:
        """Find a path between two nodes using BFS"""
        from collections import deque

        queue = deque([[start_id]])
        visited = set()

        while queue:
            path = queue.popleft()
            node_id = path[-1]

            if node_id == end_id:
                return path

            if node_id in visited:
                continue

            visited.add(node_id)

            for edge in self.get_outgoing_edges(node_id):
                new_path = list(path)
                new_path.append(edge.target_id)
                queue.append(new_path)

        return []

    def to_vector(self) -> List[float]:
        """
        Convert the entire graph to a fixed-size vector embedding.
        This enables ML learning on reasoning patterns.

        Vector structure (80 dimensions):
        - [0-19]: Aggregated source embeddings
        - [20-39]: Aggregated pillar embeddings
        - [40-59]: Aggregated decision embeddings
        - [60-79]: Graph structure features
        """
        vector = [0.0] * 80

        # Aggregate source embeddings (average)
        source_nodes = [n for n in self.nodes if n.node_type.value.startswith('source_')]
        if source_nodes:
            for i in range(20):
                vector[i] = sum(n.embedding[i] for n in source_nodes) / len(source_nodes)

        # Aggregate pillar embeddings
        pillar_nodes = [n for n in self.nodes if n.node_type.value.startswith('pillar_')]
        if pillar_nodes:
            for i in range(20):
                vector[20 + i] = sum(n.embedding[i] for n in pillar_nodes) / len(pillar_nodes)

        # Aggregate decision embeddings
        decision_nodes = [n for n in self.nodes if n.node_type.value.startswith('decision_')]
        if decision_nodes:
            for i in range(20):
                vector[40 + i] = sum(n.embedding[i] for n in decision_nodes) / len(decision_nodes)

        # Graph structure features
        vector[60] = len(self.nodes) / 20  # Normalized node count
        vector[61] = len(self.edges) / 50  # Normalized edge count
        vector[62] = sum(e.weight for e in self.edges) / max(len(self.edges), 1)  # Avg edge weight
        vector[63] = sum(n.value for n in self.nodes) / max(len(self.nodes), 1)  # Avg node value
        vector[64] = sum(n.confidence for n in self.nodes) / max(len(self.nodes), 1)  # Avg confidence

        # Path features
        paths = self.get_path_to_decision()
        vector[65] = len(paths) / 10  # Number of active paths
        if paths:
            vector[66] = sum(len(p) for p in paths) / len(paths) / 10  # Avg path length

        # Source diversity
        source_types = set(n.node_type for n in source_nodes)
        vector[67] = len(source_types) / 6  # Diversity of sources

        # Pillar convergence
        pillar_values = [n.value for n in pillar_nodes]
        if pillar_values:
            vector[68] = 1 - (max(pillar_values) - min(pillar_values))  # Convergence

        # Decision confidence
        if decision_nodes:
            vector[69] = max(n.value for n in decision_nodes)  # Decision strength

        return vector

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'nodes': [n.to_dict() for n in self.nodes],
            'edges': [e.to_dict() for e in self.edges],
            'outcome': self.outcome,
            'outcome_pct': self.outcome_pct
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ReasoningGraph':
        graph = cls(
            symbol=data['symbol'],
            timestamp=data['timestamp'],
            outcome=data.get('outcome'),
            outcome_pct=data.get('outcome_pct', 0.0)
        )
        graph.nodes = [ReasoningNode.from_dict(n) for n in data.get('nodes', [])]
        graph.edges = [ReasoningEdge(**e) for e in data.get('edges', [])]
        return graph


class ReasoningGraphBuilder:
    """
    Builds a reasoning graph from analysis data.
    Converts the raw analysis results into a neural network-like structure.
    """

    def __init__(self):
        self._node_counter = 0

    def _make_id(self, prefix: str) -> str:
        """Generate unique node ID"""
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"

    def build_from_journey(self, symbol: str, journey: Dict) -> ReasoningGraph:
        """
        Build a reasoning graph from a symbol journey.

        Args:
            symbol: Stock symbol
            journey: Journey data with discovery, analysis, decision steps

        Returns:
            ReasoningGraph representing the full reasoning chain
        """
        graph = ReasoningGraph(
            symbol=symbol,
            timestamp=datetime.now().isoformat()
        )

        steps = journey.get('journey', [])
        discovery_data = next((s.get('data', {}) for s in steps if s.get('step') == 'discovery'), {})
        analysis_data = next((s.get('data', {}) for s in steps if s.get('step') == 'analysis'), {})

        # === INPUT LAYER: Sources ===
        source_ids = []

        # Reddit source
        reddit_data = discovery_data.get('reddit', {}) or discovery_data.get('social', {}).get('reddit', {})
        if reddit_data:
            mentions = reddit_data.get('mentions', 0)
            sentiment = reddit_data.get('sentiment', 0.5)
            node = ReasoningNode(
                id=self._make_id('src_reddit'),
                node_type=NodeType.SOURCE_REDDIT,
                label=f"Reddit ({mentions} mentions)",
                value=min(mentions / 100, 1.0),
                confidence=0.7 if mentions > 20 else 0.4,
                reasoning=f"Détecté {mentions} mentions avec sentiment {sentiment:.2f}",
                metadata={'mentions': mentions, 'sentiment': sentiment},
                embedding=self._create_source_embedding('reddit', mentions, sentiment)
            )
            graph.add_node(node)
            source_ids.append(node.id)

        # Grok/X source
        grok_data = discovery_data.get('grok', {})
        if grok_data:
            grok_sentiment = grok_data.get('sentiment', 0.5)
            themes = grok_data.get('themes', [])
            node = ReasoningNode(
                id=self._make_id('src_grok'),
                node_type=NodeType.SOURCE_GROK,
                label=f"Grok/X ({len(themes)} themes)",
                value=grok_sentiment,
                confidence=0.8,
                reasoning=f"Analyse X: sentiment {grok_sentiment:.2f}, thèmes: {', '.join(themes[:3])}",
                metadata={'sentiment': grok_sentiment, 'themes': themes},
                embedding=self._create_source_embedding('grok', len(themes) * 10, grok_sentiment)
            )
            graph.add_node(node)
            source_ids.append(node.id)

        # Volume source
        volume_ratio = discovery_data.get('volume_ratio', 1.0)
        if volume_ratio > 1.2:
            node = ReasoningNode(
                id=self._make_id('src_volume'),
                node_type=NodeType.SOURCE_VOLUME,
                label=f"Volume ({volume_ratio:.1f}x)",
                value=min(volume_ratio / 3, 1.0),
                confidence=0.9,
                reasoning=f"Volume anormal: {volume_ratio:.1f}x la moyenne 20j",
                metadata={'ratio': volume_ratio},
                embedding=self._create_source_embedding('volume', volume_ratio * 30, 0.6)
            )
            graph.add_node(node)
            source_ids.append(node.id)

        # === HIDDEN LAYER 1: Detection ===
        detection_ids = []

        # Social detection (aggregates Reddit + StockTwits)
        if any(n.node_type in [NodeType.SOURCE_REDDIT, NodeType.SOURCE_STOCKTWITS]
               for n in graph.nodes):
            social_value = sum(n.value for n in graph.nodes
                             if n.node_type in [NodeType.SOURCE_REDDIT, NodeType.SOURCE_STOCKTWITS]) / 2
            node = ReasoningNode(
                id=self._make_id('det_social'),
                node_type=NodeType.DETECTION_SOCIAL,
                label="Social Buzz",
                value=social_value,
                confidence=0.6,
                reasoning="Agrégation des signaux sociaux (Reddit, StockTwits)",
                embedding=self._create_detection_embedding('social', social_value)
            )
            graph.add_node(node)
            detection_ids.append(node.id)

            # Connect sources to detection
            for src_id in source_ids:
                src_node = graph.get_node(src_id)
                if src_node and src_node.node_type in [NodeType.SOURCE_REDDIT, NodeType.SOURCE_STOCKTWITS]:
                    graph.connect(src_id, node.id, weight=0.7,
                                reasoning="Source sociale contribue au buzz")

        # === HIDDEN LAYER 2: Pillar Analysis ===
        pillar_ids = []
        pillar_details = analysis_data.get('pillar_details', {})

        pillar_configs = [
            (NodeType.PILLAR_TECHNICAL, 'technical', 'Technical', analysis_data.get('technical', 0)),
            (NodeType.PILLAR_FUNDAMENTAL, 'fundamental', 'Fundamental', analysis_data.get('fundamental', 0)),
            (NodeType.PILLAR_SENTIMENT, 'sentiment', 'Sentiment', analysis_data.get('sentiment', 0)),
            (NodeType.PILLAR_NEWS, 'news', 'News', analysis_data.get('news', 0)),
        ]

        for node_type, key, label, score in pillar_configs:
            detail = pillar_details.get(key, {})
            reasoning = detail.get('reasoning', f'Score: {score}')

            # Normalize score to [0, 1]
            normalized = (score + 100) / 200 if score < 0 or score > 1 else score
            normalized = max(0, min(1, normalized))

            node = ReasoningNode(
                id=self._make_id(f'pillar_{key}'),
                node_type=node_type,
                label=f"{label} ({normalized*100:.0f}%)",
                value=normalized,
                confidence=0.8,
                reasoning=reasoning[:200],
                metadata={'raw_score': score, 'factors': detail.get('factors', [])},
                embedding=self._create_pillar_embedding(key, normalized)
            )
            graph.add_node(node)
            pillar_ids.append(node.id)

        # Connect detection nodes to pillars
        for det_id in detection_ids:
            for pil_id in pillar_ids:
                graph.connect(det_id, pil_id, weight=0.5,
                            reasoning="Information filtrée alimente l'analyse")

        # Connect sources directly to relevant pillars
        for src_id in source_ids:
            src_node = graph.get_node(src_id)
            if src_node:
                if src_node.node_type == NodeType.SOURCE_GROK:
                    # Grok contributes to sentiment
                    sentiment_pillar = next((p for p in pillar_ids if 'sentiment' in p), None)
                    if sentiment_pillar:
                        graph.connect(src_id, sentiment_pillar, weight=0.8,
                                    reasoning="Grok/X → Sentiment direct")

        # === HIDDEN LAYER 3: Aggregation ===
        total_score = analysis_data.get('total', 0)
        normalized_total = (total_score + 100) / 200 if total_score < 0 or total_score > 1 else total_score / 100
        normalized_total = max(0, min(1, normalized_total))

        score_node = ReasoningNode(
            id=self._make_id('agg_score'),
            node_type=NodeType.AGGREGATION_SCORE,
            label=f"Score Total ({normalized_total*100:.0f}%)",
            value=normalized_total,
            confidence=0.85,
            reasoning=f"Agrégation des 4 piliers avec pondération adaptative",
            embedding=self._create_aggregation_embedding(normalized_total)
        )
        graph.add_node(score_node)

        # Connect pillars to aggregation
        for pil_id in pillar_ids:
            pil_node = graph.get_node(pil_id)
            if pil_node:
                graph.connect(pil_id, score_node.id, weight=0.25,
                            reasoning=f"{pil_node.label} contribue 25%",
                            contribution=pil_node.value * 0.25)

        # === OUTPUT LAYER: Decision ===
        status = journey.get('current_status', 'unknown')

        if status in ['signal', 'buy']:
            decision_type = NodeType.DECISION_BUY
            decision_label = "ACHETER"
            decision_value = normalized_total
        elif status == 'reject':
            decision_type = NodeType.DECISION_REJECT
            decision_label = "REJETER"
            decision_value = 1 - normalized_total
        else:
            decision_type = NodeType.DECISION_WATCH
            decision_label = "SURVEILLER"
            decision_value = 0.5

        decision_node = ReasoningNode(
            id=self._make_id('decision'),
            node_type=decision_type,
            label=decision_label,
            value=decision_value,
            confidence=0.9 if status in ['signal', 'buy', 'reject'] else 0.5,
            reasoning=f"Décision basée sur score {normalized_total*100:.0f}% vs seuil 55%",
            embedding=self._create_decision_embedding(decision_type, decision_value)
        )
        graph.add_node(decision_node)

        # Connect aggregation to decision
        graph.connect(score_node.id, decision_node.id, weight=0.9,
                     reasoning="Score total détermine la décision finale",
                     contribution=normalized_total)

        return graph

    def _create_source_embedding(self, source_type: str, intensity: float, sentiment: float) -> List[float]:
        """Create a 20-dim embedding for a source node"""
        embedding = [0.0] * 20

        # Source type encoding (one-hot-ish)
        source_map = {'reddit': 0, 'stocktwits': 1, 'grok': 2, 'news': 3, 'volume': 4}
        idx = source_map.get(source_type, 0)
        embedding[idx] = 1.0

        # Intensity features
        embedding[5] = min(intensity / 100, 1.0)
        embedding[6] = sentiment
        embedding[7] = intensity * sentiment / 100

        return embedding

    def _create_detection_embedding(self, detection_type: str, value: float) -> List[float]:
        """Create a 20-dim embedding for a detection node"""
        embedding = [0.0] * 20
        embedding[8] = value
        embedding[9] = 1.0 if detection_type == 'social' else 0.5
        return embedding

    def _create_pillar_embedding(self, pillar_type: str, value: float) -> List[float]:
        """Create a 20-dim embedding for a pillar node"""
        embedding = [0.0] * 20

        pillar_map = {'technical': 10, 'fundamental': 11, 'sentiment': 12, 'news': 13}
        idx = pillar_map.get(pillar_type, 10)
        embedding[idx] = value

        return embedding

    def _create_aggregation_embedding(self, value: float) -> List[float]:
        """Create a 20-dim embedding for an aggregation node"""
        embedding = [0.0] * 20
        embedding[14] = value
        embedding[15] = 1.0 if value >= 0.55 else 0.0  # Threshold pass
        return embedding

    def _create_decision_embedding(self, decision_type: NodeType, value: float) -> List[float]:
        """Create a 20-dim embedding for a decision node"""
        embedding = [0.0] * 20

        if decision_type == NodeType.DECISION_BUY:
            embedding[16] = 1.0
        elif decision_type == NodeType.DECISION_WATCH:
            embedding[17] = 1.0
        elif decision_type == NodeType.DECISION_REJECT:
            embedding[18] = 1.0

        embedding[19] = value

        return embedding


class ReasoningGraphStore:
    """
    Persistent storage for reasoning graphs.
    Enables learning from historical reasoning patterns.
    """

    def __init__(self, db_path: str = "data/reasoning_graphs.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS graphs (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    graph_json TEXT NOT NULL,
                    vector TEXT,
                    outcome TEXT,
                    outcome_pct REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON graphs(symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON graphs(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_outcome ON graphs(outcome)')

    def save(self, graph: ReasoningGraph) -> str:
        """Save a graph and return its ID"""
        graph_id = hashlib.md5(
            f"{graph.symbol}_{graph.timestamp}".encode()
        ).hexdigest()[:16]

        vector = json.dumps(graph.to_vector())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO graphs
                (id, symbol, timestamp, graph_json, vector, outcome, outcome_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                graph_id,
                graph.symbol,
                graph.timestamp,
                json.dumps(graph.to_dict()),
                vector,
                graph.outcome,
                graph.outcome_pct
            ))

        return graph_id

    def get(self, graph_id: str) -> Optional[ReasoningGraph]:
        """Get a graph by ID"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                'SELECT graph_json FROM graphs WHERE id = ?',
                (graph_id,)
            ).fetchone()

            if row:
                return ReasoningGraph.from_dict(json.loads(row[0]))
        return None

    def get_for_symbol(self, symbol: str, limit: int = 10) -> List[ReasoningGraph]:
        """Get recent graphs for a symbol"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute('''
                SELECT graph_json FROM graphs
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (symbol, limit)).fetchall()

            return [ReasoningGraph.from_dict(json.loads(r[0])) for r in rows]

    def update_outcome(self, graph_id: str, outcome: str, outcome_pct: float):
        """Update the outcome of a graph (for learning)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE graphs SET outcome = ?, outcome_pct = ?
                WHERE id = ?
            ''', (outcome, outcome_pct, graph_id))

    def get_successful_patterns(self, min_profit_pct: float = 5.0, limit: int = 100) -> List[ReasoningGraph]:
        """Get graphs that led to successful trades (for learning)"""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute('''
                SELECT graph_json FROM graphs
                WHERE outcome = 'profit' AND outcome_pct >= ?
                ORDER BY outcome_pct DESC
                LIMIT ?
            ''', (min_profit_pct, limit)).fetchall()

            return [ReasoningGraph.from_dict(json.loads(r[0])) for r in rows]

    def get_all_vectors_with_outcomes(self) -> List[Tuple[List[float], str, float]]:
        """
        Get all graph vectors with their outcomes.
        Used for training ML models.

        Returns:
            List of (vector, outcome, outcome_pct) tuples
        """
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute('''
                SELECT vector, outcome, outcome_pct FROM graphs
                WHERE outcome IS NOT NULL
            ''').fetchall()

            results = []
            for row in rows:
                if row[0]:
                    vector = json.loads(row[0])
                    results.append((vector, row[1], row[2]))

            return results


# Singleton instances
_graph_builder: Optional[ReasoningGraphBuilder] = None
_graph_store: Optional[ReasoningGraphStore] = None


def get_graph_builder() -> ReasoningGraphBuilder:
    """Get the singleton graph builder"""
    global _graph_builder
    if _graph_builder is None:
        _graph_builder = ReasoningGraphBuilder()
    return _graph_builder


def get_graph_store() -> ReasoningGraphStore:
    """Get the singleton graph store"""
    global _graph_store
    if _graph_store is None:
        _graph_store = ReasoningGraphStore()
    return _graph_store
