"""
Heat Detector - Detection temps reel de ce qui est "chaud"
Analyse les sources en continu pour identifier les topics/symboles en tendance.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HeatEvent:
    """Evenement detecte par une source"""
    source: str  # "social", "grok", "stocktwits", "price", "volume"
    symbol: str
    timestamp: datetime
    sentiment: float  # -1 to 1
    magnitude: float  # Force de l'evenement (mentions count, volume ratio, etc.)
    content: Optional[str] = None  # Texte original si applicable
    metadata: Dict = field(default_factory=dict)


@dataclass
class SymbolHeat:
    """Score de chaleur pour un symbole"""
    symbol: str
    heat_score: float  # 0 to 1

    # Composantes du score
    mention_velocity: float  # mentions/min vs baseline
    sentiment_shift: float  # Changement de sentiment
    volume_anomaly: float  # Volume vs moyenne
    price_momentum: float  # Mouvement de prix recent

    # Metadata
    sources: List[str]  # Sources qui ont contribue
    events_count: int
    last_update: datetime
    trending_since: Optional[datetime] = None

    # Seuils
    is_hot: bool = False
    is_warming: bool = False


@dataclass
class HeatSnapshot:
    """Snapshot de l'etat de chaleur du marche"""
    timestamp: datetime
    hot_symbols: List[SymbolHeat]
    warming_symbols: List[SymbolHeat]
    total_events: int
    market_heat: float  # Chaleur globale du marche


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HeatConfig:
    """Configuration du Heat Detector"""
    # Seuils de chaleur
    hot_threshold: float = 0.7
    warming_threshold: float = 0.4

    # Poids des composantes
    mention_velocity_weight: float = 0.30
    sentiment_shift_weight: float = 0.20
    volume_anomaly_weight: float = 0.25
    price_momentum_weight: float = 0.25

    # Parametres temporels
    baseline_window_minutes: int = 60  # Fenetre pour calculer la baseline
    decay_half_life_minutes: int = 15  # Demi-vie du score de chaleur
    min_events_for_heat: int = 3  # Minimum d'evenements pour calculer heat

    # Limites
    max_symbols_tracked: int = 200
    max_hot_symbols: int = 20

    # Frequences de polling (secondes)
    social_poll_interval: int = 30
    grok_poll_interval: int = 60
    stocktwits_poll_interval: int = 60
    price_check_interval: int = 10


# =============================================================================
# HEAT DETECTOR
# =============================================================================

class HeatDetector:
    """
    Detecte les symboles "chauds" en temps reel.

    Analyse plusieurs sources (Reddit, Grok, StockTwits, prix/volume)
    pour identifier ce qui est en tendance MAINTENANT.
    """

    def __init__(self, config: Optional[HeatConfig] = None):
        self.config = config or HeatConfig()

        # Stockage des evenements par symbole
        self._events: Dict[str, List[HeatEvent]] = defaultdict(list)

        # Baselines historiques
        self._baselines: Dict[str, Dict] = {}

        # Scores de chaleur actuels
        self._heat_scores: Dict[str, SymbolHeat] = {}

        # Historique pour l'apprentissage
        self._heat_history: List[HeatSnapshot] = []

        # Derniers polls par source
        self._last_poll: Dict[str, datetime] = {}

        # Lock pour thread safety
        self._lock = asyncio.Lock()

        # Chemin de persistence
        self._data_dir = Path("data/heat")
        self._data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("HeatDetector initialized")

    async def initialize(self):
        """Initialise le detecteur et charge les donnees historiques"""
        await self._load_baselines()
        logger.info("HeatDetector ready")

    # -------------------------------------------------------------------------
    # EVENT INGESTION
    # -------------------------------------------------------------------------

    async def add_event(self, event: HeatEvent):
        """Ajoute un nouvel evenement et recalcule le heat score"""
        async with self._lock:
            self._events[event.symbol].append(event)

            # Nettoyer les vieux evenements
            await self._cleanup_old_events(event.symbol)

            # Recalculer le score
            await self._update_heat_score(event.symbol)

    async def add_events(self, events: List[HeatEvent]):
        """Ajoute plusieurs evenements en batch"""
        for event in events:
            await self.add_event(event)

    async def ingest_social_data(self, mentions: Dict[str, Dict]):
        """
        Ingere les donnees Reddit.

        Args:
            mentions: Dict {symbol: {count, sentiment, posts: [...]}}
        """
        now = datetime.now()

        for symbol, data in mentions.items():
            count = data.get('mention_count', data.get('count', 1))
            sentiment = data.get('avg_sentiment', data.get('sentiment', 0))

            event = HeatEvent(
                source="social",
                symbol=symbol.upper(),
                timestamp=now,
                sentiment=sentiment,
                magnitude=count,
                metadata={'posts': data.get('posts', [])}
            )
            await self.add_event(event)

        self._last_poll['social'] = now

    async def ingest_grok_data(self, insights: Dict[str, Dict]):
        """
        Ingere les donnees Grok (X/Twitter).

        Args:
            insights: Dict {symbol: {sentiment, sentiment_score, summary}}
        """
        now = datetime.now()

        for symbol, data in insights.items():
            sentiment_score = data.get('sentiment_score', 0)

            event = HeatEvent(
                source="grok",
                symbol=symbol.upper(),
                timestamp=now,
                sentiment=sentiment_score,
                magnitude=abs(sentiment_score),  # Magnitude = intensite du sentiment
                content=data.get('summary', ''),
                metadata=data
            )
            await self.add_event(event)

        self._last_poll['grok'] = now

    async def ingest_price_data(self, price_changes: Dict[str, Dict]):
        """
        Ingere les donnees de prix/volume.

        Args:
            price_changes: Dict {symbol: {change_pct, volume_ratio, price}}
        """
        now = datetime.now()

        for symbol, data in price_changes.items():
            change_pct = data.get('change_pct', 0)
            volume_ratio = data.get('volume_ratio', 1.0)

            # Creer evenement prix si mouvement significatif
            if abs(change_pct) >= 1.0:  # >= 1% move
                event = HeatEvent(
                    source="price",
                    symbol=symbol.upper(),
                    timestamp=now,
                    sentiment=1.0 if change_pct > 0 else -1.0,
                    magnitude=abs(change_pct),
                    metadata={'change_pct': change_pct, 'price': data.get('price')}
                )
                await self.add_event(event)

            # Creer evenement volume si spike
            if volume_ratio >= 2.0:  # >= 2x volume moyen
                event = HeatEvent(
                    source="volume",
                    symbol=symbol.upper(),
                    timestamp=now,
                    sentiment=0,  # Volume neutre
                    magnitude=volume_ratio,
                    metadata={'volume_ratio': volume_ratio}
                )
                await self.add_event(event)

        self._last_poll['price'] = now

    # -------------------------------------------------------------------------
    # HEAT CALCULATION
    # -------------------------------------------------------------------------

    async def _update_heat_score(self, symbol: str):
        """Recalcule le score de chaleur pour un symbole"""
        events = self._events.get(symbol, [])

        if len(events) < self.config.min_events_for_heat:
            # Pas assez d'evenements
            if symbol in self._heat_scores:
                del self._heat_scores[symbol]
            return

        now = datetime.now()

        # Calculer les composantes
        mention_velocity = self._calculate_mention_velocity(symbol, events)
        sentiment_shift = self._calculate_sentiment_shift(symbol, events)
        volume_anomaly = self._calculate_volume_anomaly(symbol, events)
        price_momentum = self._calculate_price_momentum(symbol, events)

        # Score pondere
        heat_score = (
            mention_velocity * self.config.mention_velocity_weight +
            sentiment_shift * self.config.sentiment_shift_weight +
            volume_anomaly * self.config.volume_anomaly_weight +
            price_momentum * self.config.price_momentum_weight
        )

        # Normaliser entre 0 et 1
        heat_score = max(0, min(1, heat_score))

        # Appliquer decay si le symbole n'a pas eu d'evenement recent
        last_event_time = max(e.timestamp for e in events)
        age_minutes = (now - last_event_time).total_seconds() / 60
        decay_factor = 0.5 ** (age_minutes / self.config.decay_half_life_minutes)
        heat_score *= decay_factor

        # Determiner le statut
        is_hot = heat_score >= self.config.hot_threshold
        is_warming = heat_score >= self.config.warming_threshold

        # Sources contributrices
        sources = list(set(e.source for e in events))

        # Mettre a jour ou creer
        if symbol in self._heat_scores:
            existing = self._heat_scores[symbol]
            trending_since = existing.trending_since
            if is_hot and not existing.is_hot:
                trending_since = now
        else:
            trending_since = now if is_hot else None

        self._heat_scores[symbol] = SymbolHeat(
            symbol=symbol,
            heat_score=heat_score,
            mention_velocity=mention_velocity,
            sentiment_shift=sentiment_shift,
            volume_anomaly=volume_anomaly,
            price_momentum=price_momentum,
            sources=sources,
            events_count=len(events),
            last_update=now,
            trending_since=trending_since,
            is_hot=is_hot,
            is_warming=is_warming
        )

    def _calculate_mention_velocity(self, symbol: str, events: List[HeatEvent]) -> float:
        """Calcule la velocite des mentions vs baseline"""
        now = datetime.now()
        recent_window = timedelta(minutes=10)

        # Compter les mentions recentes
        recent_mentions = sum(
            1 for e in events
            if e.source in ('social', 'grok', 'stocktwits')
            and now - e.timestamp < recent_window
        )

        # Baseline (mentions attendues en 10 min)
        baseline = self._baselines.get(symbol, {}).get('mentions_per_10min', 1)

        # Ratio
        if baseline > 0:
            velocity = recent_mentions / baseline
        else:
            velocity = recent_mentions

        # Normaliser (1 = normal, 2+ = accelere)
        return min(1.0, (velocity - 1) / 5) if velocity > 1 else 0

    def _calculate_sentiment_shift(self, symbol: str, events: List[HeatEvent]) -> float:
        """Calcule le changement de sentiment recent"""
        now = datetime.now()

        # Sentiment recent (derniere heure)
        recent_window = timedelta(hours=1)
        recent_sentiments = [
            e.sentiment for e in events
            if now - e.timestamp < recent_window
            and e.sentiment != 0
        ]

        if not recent_sentiments:
            return 0

        current_sentiment = sum(recent_sentiments) / len(recent_sentiments)

        # Baseline sentiment
        baseline_sentiment = self._baselines.get(symbol, {}).get('avg_sentiment', 0)

        # Shift (valeur absolue du changement)
        shift = abs(current_sentiment - baseline_sentiment)

        return min(1.0, shift)

    def _calculate_volume_anomaly(self, symbol: str, events: List[HeatEvent]) -> float:
        """Calcule l'anomalie de volume"""
        volume_events = [e for e in events if e.source == 'volume']

        if not volume_events:
            return 0

        # Prendre le ratio de volume le plus recent
        latest = max(volume_events, key=lambda e: e.timestamp)
        volume_ratio = latest.magnitude

        # Normaliser (2x = 0.5, 4x = 1.0)
        return min(1.0, (volume_ratio - 1) / 3) if volume_ratio > 1 else 0

    def _calculate_price_momentum(self, symbol: str, events: List[HeatEvent]) -> float:
        """Calcule le momentum de prix"""
        price_events = [e for e in events if e.source == 'price']

        if not price_events:
            return 0

        # Prendre le mouvement le plus recent
        latest = max(price_events, key=lambda e: e.timestamp)
        change_pct = latest.magnitude  # Valeur absolue

        # Normaliser (1% = 0.33, 3%+ = 1.0)
        return min(1.0, change_pct / 3)

    # -------------------------------------------------------------------------
    # QUERIES
    # -------------------------------------------------------------------------

    def get_hot_symbols(self, limit: int = 10) -> List[SymbolHeat]:
        """Retourne les symboles les plus chauds"""
        hot = [h for h in self._heat_scores.values() if h.is_hot]
        hot.sort(key=lambda x: x.heat_score, reverse=True)
        return hot[:limit]

    def get_warming_symbols(self, limit: int = 20) -> List[SymbolHeat]:
        """Retourne les symboles en train de chauffer"""
        warming = [h for h in self._heat_scores.values() if h.is_warming and not h.is_hot]
        warming.sort(key=lambda x: x.heat_score, reverse=True)
        return warming[:limit]

    def get_symbol_heat(self, symbol: str) -> Optional[SymbolHeat]:
        """Retourne le heat score d'un symbole specifique"""
        return self._heat_scores.get(symbol.upper())

    def get_snapshot(self) -> HeatSnapshot:
        """Retourne un snapshot de l'etat actuel"""
        hot = self.get_hot_symbols(self.config.max_hot_symbols)
        warming = self.get_warming_symbols(20)

        total_events = sum(len(events) for events in self._events.values())

        # Chaleur globale du marche = moyenne des top 10
        if hot:
            market_heat = sum(h.heat_score for h in hot[:10]) / min(10, len(hot))
        else:
            market_heat = 0

        return HeatSnapshot(
            timestamp=datetime.now(),
            hot_symbols=hot,
            warming_symbols=warming,
            total_events=total_events,
            market_heat=market_heat
        )

    def is_symbol_hot(self, symbol: str) -> bool:
        """Verifie si un symbole est chaud"""
        heat = self._heat_scores.get(symbol.upper())
        return heat.is_hot if heat else False

    # -------------------------------------------------------------------------
    # MAINTENANCE
    # -------------------------------------------------------------------------

    async def _cleanup_old_events(self, symbol: str):
        """Nettoie les evenements trop vieux"""
        cutoff = datetime.now() - timedelta(minutes=self.config.baseline_window_minutes)
        self._events[symbol] = [
            e for e in self._events[symbol]
            if e.timestamp > cutoff
        ]

    async def update_baselines(self):
        """Met a jour les baselines historiques (appeler periodiquement)"""
        for symbol, events in self._events.items():
            if len(events) < 5:
                continue

            # Calculer les moyennes
            mention_events = [e for e in events if e.source in ('social', 'grok', 'stocktwits')]
            sentiments = [e.sentiment for e in events if e.sentiment != 0]

            self._baselines[symbol] = {
                'mentions_per_10min': len(mention_events) / 6,  # Sur 1h
                'avg_sentiment': sum(sentiments) / len(sentiments) if sentiments else 0,
                'updated_at': datetime.now().isoformat()
            }

        # Sauvegarder
        await self._save_baselines()

    async def _load_baselines(self):
        """Charge les baselines depuis le disque"""
        path = self._data_dir / "baselines.json"
        if path.exists():
            try:
                with open(path, 'r') as f:
                    self._baselines = json.load(f)
                logger.info(f"Loaded {len(self._baselines)} baselines")
            except Exception as e:
                logger.warning(f"Could not load baselines: {e}")

    async def _save_baselines(self):
        """Sauvegarde les baselines sur le disque"""
        path = self._data_dir / "baselines.json"
        try:
            with open(path, 'w') as f:
                json.dump(self._baselines, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save baselines: {e}")

    async def close(self):
        """Ferme proprement le detecteur"""
        await self._save_baselines()
        logger.info("HeatDetector closed")


# =============================================================================
# FACTORY
# =============================================================================

_heat_detector: Optional[HeatDetector] = None


async def get_heat_detector(config: Optional[HeatConfig] = None) -> HeatDetector:
    """Retourne l'instance singleton du HeatDetector"""
    global _heat_detector
    if _heat_detector is None:
        _heat_detector = HeatDetector(config)
        await _heat_detector.initialize()
    return _heat_detector
