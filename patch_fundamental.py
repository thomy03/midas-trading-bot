#!/usr/bin/env python3
"""Patch fundamental_pillar.py to add Polygon integration"""
import re

# Read the file
with open("src/agents/pillars/fundamental_pillar.py", "r") as f:
    content = f.read()

# New method
new_method = '''    async def _fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch fundamental data from yfinance + Polygon (hybrid approach).
        
        yfinance: Primary source (no rate limit)
        Polygon: Enrichment for better data quality (5 calls/min limit)
        """
        result = {}
        
        # 1. Primary: yfinance (fast, no rate limit)
        try:
            ticker = yf.Ticker(symbol)
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, lambda: ticker.info)
            if info:
                result = info.copy()
                logger.debug(f"[FUNDAMENTAL] {symbol}: yfinance data OK")
        except Exception as e:
            logger.warning(f"yfinance failed for {symbol}: {e}")
        
        # 2. Enrich with Polygon (only for candidates, respects rate limit)
        try:
            if get_polygon_client is not None:
                polygon = get_polygon_client()
                if polygon.api_key:
                    # Get ticker details (sector, industry, market cap)
                    details = polygon.get_ticker_details(symbol)
                    if details:
                        # Merge Polygon data (prefer Polygon when yfinance is missing)
                        polygon_mappings = {
                            "market_cap": "marketCap",
                            "sic_description": "sector",
                            "total_employees": "fullTimeEmployees",
                            "description": "longBusinessSummary",
                            "homepage_url": "website",
                        }
                        for poly_key, yf_key in polygon_mappings.items():
                            if details.get(poly_key) and not result.get(yf_key):
                                result[yf_key] = details[poly_key]
                        
                        result["_polygon_details"] = details
                        logger.debug(f"[FUNDAMENTAL] {symbol}: Polygon enrichment OK")
        except Exception as e:
            logger.debug(f"Polygon enrichment skipped for {symbol}: {e}")
        
        return result'''

# Find and replace the old method using a simpler approach
# Find the method start
start_marker = '    async def _fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:'
end_marker = '            return {}'

start_idx = content.find(start_marker)
if start_idx != -1:
    # Find the end of the method (return {})
    end_idx = content.find(end_marker, start_idx)
    if end_idx != -1:
        end_idx += len(end_marker)
        new_content = content[:start_idx] + new_method + content[end_idx:]
        
        with open("src/agents/pillars/fundamental_pillar.py", "w") as f:
            f.write(new_content)
        print("✅ Method updated successfully!")
    else:
        print("❌ End marker not found")
else:
    print("❌ Start marker not found")
