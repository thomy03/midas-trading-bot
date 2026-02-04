#!/usr/bin/env python3
"""Patch news_pillar.py to add Polygon News enrichment"""

with open("src/agents/pillars/news_pillar.py", "r") as f:
    content = f.read()

# Find the return statement in _fetch_news and inject Polygon code before it
old_return = """            return {
                'headlines': headlines,
                'count': len(headlines),
                'calendar': calendar,
                'recommendations': recommendations,
                'fetched_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.warning(f"Failed to fetch news for {symbol}: {e}")
            return {'headlines': [], 'count': 0}"""

new_return = """            # Enrich with Polygon News (additional sources)
            try:
                if get_polygon_client is not None:
                    polygon = get_polygon_client()
                    if polygon.api_key:
                        poly_news = polygon.get_news(symbol, limit=5)
                        if poly_news:
                            existing_titles = {h['title'].lower() for h in headlines}
                            added = 0
                            for item in poly_news:
                                title = item.get('title', '')
                                if title.lower() not in existing_titles:
                                    headlines.append({
                                        'title': title,
                                        'publisher': item.get('publisher', {}).get('name', 'Polygon'),
                                        'link': item.get('article_url', ''),
                                        'published': 0,
                                        'type': 'NEWS'
                                    })
                                    existing_titles.add(title.lower())
                                    added += 1
                            if added > 0:
                                logger.debug(f"Polygon added {added} news for {symbol}")
            except Exception as pe:
                logger.debug(f"Polygon news skipped for {symbol}: {pe}")

            return {
                'headlines': headlines,
                'count': len(headlines),
                'calendar': calendar,
                'recommendations': recommendations,
                'fetched_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.warning(f"Failed to fetch news for {symbol}: {e}")
            return {'headlines': [], 'count': 0}"""

if old_return in content:
    content = content.replace(old_return, new_return)
    with open("src/agents/pillars/news_pillar.py", "w") as f:
        f.write(content)
    print("✅ News pillar patched successfully!")
else:
    print("❌ Could not find target block")
    # Show what we're looking for
    print("Looking for:", repr(old_return[:100]))
