"""Social Scanner - DEPRECATED (V8.1 cleanup)
Replaced by Intelligence Orchestrator V8.
Stub for import compatibility.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrendingSymbol:
    symbol: str
    mention_count: int = 0
    avg_sentiment: float = 0.0


@dataclass
class SocialScanResult:
    trending_symbols: List[TrendingSymbol] = field(default_factory=list)


class SocialScanner:
    """Stub - does nothing."""
    def __init__(self, enable_stocktwits=False, enable_social=False, **kwargs):
        logger.info("SocialScanner STUB initialized (deprecated)")

    async def initialize(self):
        pass

    async def close(self):
        pass

    async def full_scan(self) -> SocialScanResult:
        return SocialScanResult()

    async def get_symbol_details_for_narrative(self, symbol: str) -> dict:
        return {}


_social_scanner_instance: Optional[SocialScanner] = None


async def get_social_scanner(enable_stocktwits=False, enable_social=False) -> SocialScanner:
    global _social_scanner_instance
    if _social_scanner_instance is None:
        _social_scanner_instance = SocialScanner(enable_stocktwits=enable_stocktwits, enable_social=enable_social)
        await _social_scanner_instance.initialize()
    return _social_scanner_instance


async def close_social_scanner():
    global _social_scanner_instance
    if _social_scanner_instance:
        await _social_scanner_instance.close()
        _social_scanner_instance = None
