"""Heat Detector - DEPRECATED (V8.1 cleanup)
Replaced by Intelligence Orchestrator V8.
This stub exists only to prevent import errors during transition.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class HeatLevel(Enum):
    COLD = "cold"
    WARMING = "warming"
    HOT = "hot"


@dataclass
class SymbolHeat:
    symbol: str
    heat_score: float = 0.0
    heat_level: HeatLevel = HeatLevel.COLD
    sources: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class HeatEvent:
    symbol: str
    event_type: str = ""
    score: float = 0.0


@dataclass
class HeatSnapshot:
    hot_symbols: List[SymbolHeat] = field(default_factory=list)
    warming_symbols: List[SymbolHeat] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HeatConfig:
    max_hot_symbols: int = 10


class HeatDetector:
    """Stub - does nothing. Kept for import compatibility."""

    def __init__(self, config: Optional[HeatConfig] = None):
        self.config = config or HeatConfig()
        logger.info("HeatDetector STUB initialized (deprecated)")

    async def initialize(self):
        pass

    async def close(self):
        pass

    def get_hot_symbols(self, limit: int = 10) -> List[SymbolHeat]:
        return []

    def get_warming_symbols(self, limit: int = 20) -> List[SymbolHeat]:
        return []

    def get_symbol_heat(self, symbol: str) -> Optional[SymbolHeat]:
        return None

    def get_snapshot(self) -> HeatSnapshot:
        return HeatSnapshot()

    async def ingest_grok_data(self, data: dict):
        pass

    async def ingest_social_data(self, data: dict):
        pass

    async def ingest_price_data(self, data: dict):
        pass


_heat_detector: Optional[HeatDetector] = None


async def get_heat_detector(config: Optional[HeatConfig] = None) -> HeatDetector:
    global _heat_detector
    if _heat_detector is None:
        _heat_detector = HeatDetector(config)
        await _heat_detector.initialize()
    return _heat_detector
