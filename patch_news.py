#!/usr/bin/env python3
"""Patch news_pillar.py to add Polygon News integration"""
import re

# Read the file
with open("src/agents/pillars/news_pillar.py", "r") as f:
    content = f.read()

# Add import after existing imports
import_line = "from .base import BasePillar, PillarScore"
polygon_import = """from .base import BasePillar, PillarScore

# Polygon integration for hybrid news
try:
    from src.data.polygon_client import get_polygon_client
except ImportError:
    get_polygon_client = None"""

content = content.replace(import_line, polygon_import)

# New fetch method
new_fetch = '''    async def _fetch_news(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch news from yfinance + Polygon (hybrid approach).
        
        yfinance: Primary source (free, reliable for headlines)
        Polygon: Enrichment with additional news sources (5 calls/min)
        """
        headlines = []
        calendar = None
        recommendations = None
        
        # 1. Primary: yfinance news
        try:
            ticker = yf.Ticker(symbol)
            loop = asyncio.get_event_loop()
            news = await loop.run_in_executor(None, lambda: ticker.news)

            if news:
                for item in news[:10]:
                    headlines.append({
                        'title': item.get('title', ''),
                        'publisher': item.get('publisher', ''),
                        'link': item.get('link', ''),
                        'published': item.get('providerPublishTime', 0),
                        'type': item.get('type', 'STORY'),
                        'source': 'yfinance'
                    })

            # Get calendar events
            try:
                calendar = await loop.run_in_executor(None, lambda: ticker.calendar)
            except Exception:
                pass

            # Get recommendations
            try:
                recs = await loop.run_in_executor(None, lambda: ticker.recommendations)
                if recs is not None and not recs.empty:
                    recommendations = recs.tail(5).to_dict('records')
            except Exception:
                pass
                
            logger.debug(f"[NEWS] {symbol}: yfinance returned {len(headlines)} headlines")

        except Exception as e:
            logger.warning(f"Failed to fetch yfinance news for {symbol}: {e}")

        # 2. Enrich with Polygon News (additional sources)
        try:
            if get_polygon_client is not None:
                polygon = get_polygon_client()
                if polygon.api_key:
                    poly_news = polygon.get_news(symbol, limit=5)
                    if poly_news:
                        existing_titles = {h['title'].lower() for h in headlines}
                        for item in poly_news:
                            title = item.get('title', '')
                            # Avoid duplicates
                            if title.lower() not in existing_titles:
                                headlines.append({
                                    'title': title,
                                    'publisher': item.get('publisher', {}).get('name', 'Polygon'),
                                    'link': item.get('article_url', ''),
                                    'published': 0,  # Polygon uses ISO dates
                                    'type': 'NEWS',
                                    'source': 'polygon'
                                })
                                existing_titles.add(title.lower())
                        logger.debug(f"[NEWS] {symbol}: Polygon added {len(poly_news)} news items")
        except Exception as e:
            logger.debug(f"Polygon news skipped for {symbol}: {e}")

        return {
            'headlines': headlines,
            'count': len(headlines),
            'calendar': calendar,
            'recommendations': recommendations,
            'fetched_at': datetime.now().isoformat()
        }'''

# Find and replace the old _fetch_news method
start_marker = '    async def _fetch_news(self, symbol: str) -> Dict[str, Any]:'
# Find the next method or end of class
old_method_end = "            return {'headlines': [], 'count': 0}"

start_idx = content.find(start_marker)
if start_idx != -1:
    # Find where this method ends (the final return with empty dict)
    end_idx = content.find(old_method_end, start_idx)
    if end_idx != -1:
        end_idx += len(old_method_end)
        new_content = content[:start_idx] + new_fetch + content[end_idx:]
        
        with open("src/agents/pillars/news_pillar.py", "w") as f:
            f.write(new_content)
        print("✅ News pillar updated successfully!")
    else:
        print("❌ End marker not found, trying alternative...")
        # Try to find the next method definition
        next_method = content.find("    async def _analyze_with_llm", start_idx + 10)
        if next_method != -1:
            new_content = content[:start_idx] + new_fetch + "\n\n" + content[next_method:]
            with open("src/agents/pillars/news_pillar.py", "w") as f:
                f.write(new_content)
            print("✅ News pillar updated (alternative method)!")
        else:
            print("❌ Could not find method boundaries")
else:
    print("❌ Start marker not found")
