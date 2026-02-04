"""
Market Scheduler - Time-based market filtering

Filters which markets to scan based on Paris time:
- 8h-15h30 Paris: Europe only (pre-US)
- 15h30-17h30 Paris: Europe + US (overlap)
- 17h30-22h Paris: US only (post-EU close)
"""

import logging
from datetime import datetime, time
from enum import Enum
from typing import Set, List
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

PARIS_TZ = ZoneInfo("Europe/Paris")
NY_TZ = ZoneInfo("America/New_York")


class MarketRegion(Enum):
    """Market regions"""
    EU = "europe"
    US = "us"
    ALL = "all"


class MarketScheduler:
    """
    Determines which markets should be scanned based on current time.
    
    Schedule (Paris time, Mon-Fri):
    - 08:00 - 15:30: Europe only (EU markets open, US pre-market)
    - 15:30 - 17:30: Europe + US (overlap period)
    - 17:30 - 22:00: US only (EU closed, US open)
    - 22:00 - 08:00: Closed (no scanning)
    """
    
    # Paris time boundaries
    EU_START = time(8, 0)      # 8h Paris
    US_PREMARKET = time(15, 30)  # 15h30 Paris (9h30 NY)
    EU_CLOSE = time(17, 30)    # 17h30 Paris
    US_CLOSE = time(22, 0)     # 22h Paris (16h NY)
    
    # Ticker files by region
    EU_TICKER_FILES = ["europe.json", "cac40.json", "eu_extended.json"]
    US_TICKER_FILES = ["nasdaq.json", "sp500.json", "nyse_full.json"]
    
    @classmethod
    def get_paris_time(cls) -> datetime:
        """Get current Paris time"""
        return datetime.now(PARIS_TZ)
    
    @classmethod
    def get_active_region(cls) -> MarketRegion:
        """
        Determine which market region should be active now.
        
        Returns:
            MarketRegion: EU, US, or ALL
        """
        now = cls.get_paris_time()
        current_time = now.time()
        weekday = now.weekday()
        
        # Weekend = closed
        if weekday >= 5:
            logger.debug("Weekend - markets closed")
            return MarketRegion.EU  # Default to EU for analysis
        
        # Time-based logic
        if current_time < cls.EU_START:
            logger.debug(f"Before EU open ({current_time}) - closed")
            return MarketRegion.EU
        elif current_time < cls.US_PREMARKET:
            logger.info(f"EU session ({current_time}) - Europe only")
            return MarketRegion.EU
        elif current_time < cls.EU_CLOSE:
            logger.info(f"Overlap session ({current_time}) - Europe + US")
            return MarketRegion.ALL
        elif current_time < cls.US_CLOSE:
            logger.info(f"US session ({current_time}) - US only")
            return MarketRegion.US
        else:
            logger.debug(f"After US close ({current_time}) - closed")
            return MarketRegion.US
    
    @classmethod
    def get_active_ticker_files(cls) -> List[str]:
        """
        Get list of ticker files to load based on current market session.
        
        Returns:
            List of ticker file names to use
        """
        region = cls.get_active_region()
        
        if region == MarketRegion.EU:
            return cls.EU_TICKER_FILES
        elif region == MarketRegion.US:
            return cls.US_TICKER_FILES
        else:  # ALL
            return cls.EU_TICKER_FILES + cls.US_TICKER_FILES
    
    @classmethod
    def should_scan_symbol(cls, symbol: str) -> bool:
        """
        Check if a symbol should be scanned in current session.
        
        Args:
            symbol: Stock symbol to check
            
        Returns:
            True if symbol should be scanned now
        """
        region = cls.get_active_region()
        
        # EU symbols typically have suffixes like .PA, .DE, .L, .AS
        is_eu_symbol = any(
            symbol.upper().endswith(suffix) 
            for suffix in [".PA", ".DE", ".L", ".AS", ".MI", ".MC", ".BR"]
        )
        
        if region == MarketRegion.ALL:
            return True
        elif region == MarketRegion.EU:
            return is_eu_symbol
        else:  # US
            return not is_eu_symbol
    
    @classmethod
    def get_session_info(cls) -> dict:
        """Get current session information for logging/display"""
        now = cls.get_paris_time()
        region = cls.get_active_region()
        
        return {
            "paris_time": now.strftime("%H:%M"),
            "weekday": now.strftime("%A"),
            "region": region.value,
            "ticker_files": cls.get_active_ticker_files(),
            "next_transition": cls._get_next_transition(now.time())
        }
    
    @classmethod
    def _get_next_transition(cls, current: time) -> str:
        """Get description of next session transition"""
        if current < cls.EU_START:
            return f"EU open at {cls.EU_START}"
        elif current < cls.US_PREMARKET:
            return f"US pre-market at {cls.US_PREMARKET}"
        elif current < cls.EU_CLOSE:
            return f"EU close at {cls.EU_CLOSE}"
        elif current < cls.US_CLOSE:
            return f"US close at {cls.US_CLOSE}"
        else:
            return f"EU open tomorrow at {cls.EU_START}"


# Convenience function
def get_market_scheduler() -> MarketScheduler:
    """Get MarketScheduler instance"""
    return MarketScheduler()


# Quick test
if __name__ == "__main__":
    import json
    info = MarketScheduler.get_session_info()
    print(json.dumps(info, indent=2))
