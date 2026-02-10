"""
News Pillar - News and events analysis using LLM.

V4.8: Uses GeminiClient (Google AI) directly instead of OpenRouter.

Analyzes:
- Recent news headlines (via yfinance)
- Earnings announcements
- Analyst upgrades/downgrades
- Sector/market news
- Event impact assessment

Contributes 25% to final decision score.
"""

import os
import yfinance as yf
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from dateutil.parser import parse as parse_date
import logging
import json

from .base import BasePillar, PillarScore

# Google Trends integration
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    TrendReq = None

# Polygon integration for hybrid news
try:
    from src.data.polygon_client import get_polygon_client
except ImportError:
    get_polygon_client = None

logger = logging.getLogger(__name__)


class NewsPillar(BasePillar):
    """
    News analysis pillar using Gemini (Google AI) for financial news interpretation.

    V4.8: Uses GeminiClient directly instead of OpenRouter.

    Analyzes:
    - Recent news headlines (via yfinance)
    - Earnings announcements
    - Analyst upgrades/downgrades
    - Company-specific news
    - Sector/market news impact

    Note: Grok is NOT used here - it's reserved for social media analysis
    (X/Twitter) via grok_scanner.py.
    """

    def __init__(self, weight: float = 0.25):
        super().__init__(weight)
        self._cache_ttl = 900  # 15 min cache for news

        # V4.8: Use Gemini direct instead of OpenRouter
        self.api_key = os.getenv('GOOGLE_AI_API_KEY', '')
        self.model = os.getenv('GEMINI_MODEL', 'gemini-3-flash-preview')

        self._gemini_client = None  # Will be GeminiClient

        # News type weights
        self.news_weights = {
            'earnings': 0.30,
            'analyst': 0.25,
            'company': 0.25,
            'sector': 0.20
        }

    def get_name(self) -> str:
        return "News"


    async def get_trends_score(self, symbol: str) -> Dict[str, Any]:
        """Get Google Trends interest score for a symbol."""
        if not PYTRENDS_AVAILABLE:
            return {"score": 50, "trend": "unknown", "error": "pytrends not available"}
        
        try:
            pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
            pytrends.build_payload([symbol], timeframe="now 7-d", geo="US")
            data = pytrends.interest_over_time()
            
            if data.empty:
                return {"score": 50, "trend": "no_data", "current": 0}
            
            values = data[symbol].values
            current = values[-1]
            avg_first_half = values[:len(values)//2].mean()
            avg_second_half = values[len(values)//2:].mean()
            
            # Calculate trend
            if avg_first_half > 0:
                change_pct = ((avg_second_half / avg_first_half) - 1) * 100
            else:
                change_pct = 0
            
            # Score based on trend
            if change_pct > 50:
                score = 85  # Strong buzz increase
                trend = "surging"
            elif change_pct > 20:
                score = 70  # Moderate increase
                trend = "rising"
            elif change_pct > -10:
                score = 50  # Stable
                trend = "stable"
            elif change_pct > -30:
                score = 35  # Declining
                trend = "declining"
            else:
                score = 20  # Strong decline
                trend = "crashing"
            
            return {
                "score": score,
                "trend": trend,
                "current_interest": int(current),
                "change_pct": round(change_pct, 1)
            }
        except Exception as e:
            logger.warning(f"Google Trends error for {symbol}: {e}")
            return {"score": 50, "trend": "error", "error": str(e)}
    async def initialize(self):
        """Initialize Gemini client"""
        if self._gemini_client is None and self.api_key:
            try:
                from src.intelligence.gemini_client import GeminiClient
                self._gemini_client = GeminiClient(api_key=self.api_key, model=self.model)
                await self._gemini_client.initialize()
                logger.info(f"NewsPillar initialized with Gemini: {self._gemini_client.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize GeminiClient for NewsPillar: {e}")

    async def close(self):
        """Close Gemini client"""
        if self._gemini_client:
            await self._gemini_client.close()
            self._gemini_client = None

    async def analyze(
        self,
        symbol: str,
        data: Dict[str, Any]
    ) -> PillarScore:
        """
        Analyze news for a symbol.

        Args:
            symbol: Stock symbol
            data: Optional pre-fetched news data

        Returns:
            PillarScore with news analysis result
        """
        # V8.1: If LLM disabled, return neutral score
        if os.environ.get("DISABLE_LLM", "false").lower() == "true":
            logger.info(f"[NEWS] {symbol}: LLM disabled, returning neutral (50)")
            return PillarScore(
                pillar_name="News",
                score=50.0,
                signal=PillarSignal.NEUTRAL,
                confidence=0.5,
                reasoning=f"LLM disabled - neutral news for {symbol}",
                factors=[],
                timestamp=datetime.now().isoformat(),
                data_quality=0.5,
            )

        await self.initialize()

        factors = []
        category_scores = {}

        logger.info(f"[NEWS] {symbol}: Starting news analysis (Gemini: {'configured' if self._gemini_client else 'NOT configured'})...")

        try:
            # Fetch news if not provided
            news_data = data.get('news')
            if news_data is None:
                news_data = await self._fetch_news(symbol)

            if not news_data or not news_data.get('headlines'):
                logger.warning(f"[NEWS] {symbol}: No recent news headlines found")
                return self._create_score(
                    score=0,
                    reasoning=f"No recent news for {symbol}",
                    data_quality=0.3  # Absence of news is also data
                )

            headlines_count = len(news_data.get('headlines', []))
            logger.info(f"[NEWS] {symbol}: Found {headlines_count} headlines to analyze")

            # V4.8: Analyze news with GeminiClient
            if self._gemini_client:
                analysis = await self._analyze_with_llm(symbol, news_data)
                if analysis:
                    raw_scores = analysis.get('category_scores', {})
                    # V4.9.5: Ensure all scores are numeric (Gemini may return strings)
                    category_scores = {}
                    for k, v in raw_scores.items():
                        try:
                            category_scores[k] = float(v) if v is not None else 0
                        except (ValueError, TypeError):
                            category_scores[k] = 0
                    factors = analysis.get('factors', [])

            # Calculate weighted total
            if category_scores:
                total_score = sum(
                    category_scores.get(cat, 0) * self.news_weights.get(cat, 0.25)
                    for cat in category_scores
                )
            else:
                total_score = 0

            # Generate reasoning
            reasoning = self._generate_reasoning(symbol, news_data, category_scores, factors)

            # Confidence based on news recency
            confidence = self._calculate_confidence(news_data)

            # V4.4: Verbose logging
            impact = "positive" if total_score > 20 else "negative" if total_score < -20 else "neutral"
            data_quality = 0.8 if category_scores else 0.3
            logger.info(f"[NEWS] {symbol}: Score={total_score:.1f}/100 ({impact}) | Headlines={headlines_count} | Quality={data_quality:.0%}")

            return self._create_score(
                score=total_score,
                reasoning=reasoning,
                factors=factors,
                confidence=confidence,
                data_quality=data_quality
            )

        except Exception as e:
            logger.error(f"News analysis failed for {symbol}: {e}")
            return self._create_score(
                score=0,
                reasoning=f"News analysis error: {str(e)}",
                data_quality=0.0
            )

    async def _fetch_news(self, symbol: str) -> Dict[str, Any]:
        """Fetch news from yfinance"""
        try:
            ticker = yf.Ticker(symbol)

            loop = asyncio.get_event_loop()
            news = await loop.run_in_executor(None, lambda: ticker.news)

            if not news:
                return {'headlines': [], 'count': 0}

            # Parse news items (handle both old and new yfinance formats)
            headlines = []
            for item in news[:10]:  # Limit to 10 most recent
                try:
                    # New format: item['content']['title']
                    # Old format: item['title']
                    content = item.get('content') or item
                    provider = content.get('provider') or {}
                    click_url = content.get('clickThroughUrl') or {}
                    
                    headlines.append({
                        'title': content.get('title') or item.get('title', ''),
                        'publisher': provider.get('displayName') or item.get('publisher', ''),
                        'link': click_url.get('url') or item.get('link', ''),
                        'published': content.get('pubDate') or item.get('providerPublishTime', ''),
                        'type': content.get('contentType') or item.get('type', 'STORY'),
                        'summary': content.get('summary', '')
                    })
                except Exception as e:
                    logger.warning(f'Error parsing news item: {e}')
                    continue

            # Get calendar events
            calendar = None
            try:
                calendar = await loop.run_in_executor(None, lambda: ticker.calendar)
            except Exception:
                pass

            # Get recommendations
            recommendations = None
            try:
                recs = await loop.run_in_executor(None, lambda: ticker.recommendations)
                if recs is not None and not recs.empty:
                    recommendations = recs.tail(5).to_dict('records')
            except Exception:
                pass

            # Enrich with Polygon News (additional sources)
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
            return {'headlines': [], 'count': 0}

    async def _analyze_with_llm(self, symbol: str, news_data: Dict) -> Optional[Dict]:
        """
        Analyze news using Gemini (Google AI) directly.

        V4.8: Uses GeminiClient instead of OpenRouter.

        Note: Grok is NOT used here - it's reserved for social media analysis
        (X/Twitter) via grok_scanner.py. Gemini handles all financial news.
        """
        if not self._gemini_client:
            logger.warning(f"GeminiClient not configured for news analysis of {symbol}")
            return None

        headlines = news_data.get('headlines', [])
        calendar = news_data.get('calendar', {})
        recommendations = news_data.get('recommendations', [])

        # Build context
        headlines_text = "\n".join([
            f"- {h.get('title', '')} ({h.get('publisher', '')})"
            for h in headlines[:8]
        ])

        calendar_text = ""
        if calendar:
            earnings_date = calendar.get('Earnings Date')
            if earnings_date:
                calendar_text = f"Upcoming earnings: {earnings_date}"

        recs_text = ""
        if recommendations:
            recent_recs = recommendations[-3:] if len(recommendations) > 3 else recommendations
            recs_text = "Recent analyst actions: " + ", ".join([
                f"{r.get('Firm', 'Unknown')}: {r.get('To Grade', 'N/A')}"
                for r in recent_recs
            ])

        prompt = f"""Analyze the following news and events for ${symbol} and assess their impact on the stock.

RECENT HEADLINES:
{headlines_text}

{calendar_text}
{recs_text}

For each category, provide a score from -100 (very negative) to +100 (very positive):

1. EARNINGS: Impact of earnings-related news (announcements, surprises, guidance)
2. ANALYST: Impact of analyst actions (upgrades, downgrades, price targets)
3. COMPANY: Impact of company-specific news (products, management, partnerships)
4. SECTOR: Impact of sector/market-wide news affecting this stock

Return your analysis as JSON:
{{
    "category_scores": {{
        "earnings": <-100 to 100>,
        "analyst": <-100 to 100>,
        "company": <-100 to 100>,
        "sector": <-100 to 100>
    }},
    "factors": [
        {{"category": "...", "headline": "...", "impact": "positive/negative/neutral", "score": <number>, "reason": "..."}}
    ],
    "overall_sentiment": "bullish/bearish/neutral",
    "key_catalyst": "Most important news item",
    "summary": "One sentence summary"
}}

Be objective and focus on material news that could impact the stock price."""

        try:
            # V4.8: Use GeminiClient's chat_json for automatic JSON parsing
            analysis = await self._gemini_client.chat_json(
                prompt,
                system_prompt="You are an expert financial analyst. Analyze news objectively and respond in valid JSON only.",
                temperature=0.3,
                max_tokens=1500  # V4.9.4: Increased to avoid truncation
            )

            if analysis:
                logger.info(f"Gemini news analysis complete for {symbol}")
                return analysis

        except Exception as e:
            logger.warning(f"Gemini news analysis failed for {symbol}: {e}")

        return None

    def _generate_reasoning(
        self,
        symbol: str,
        news_data: Dict,
        category_scores: Dict[str, float],
        factors: List[Dict]
    ) -> str:
        """Generate human-readable reasoning"""
        parts = [f"News analysis for ${symbol}"]

        headlines = news_data.get('headlines', [])
        parts.append(f"({len(headlines)} recent news items analyzed)")

        if not category_scores:
            if headlines:
                parts.append("\nRecent headlines:")
                for h in headlines[:3]:
                    parts.append(f"  - {h.get('title', 'N/A')[:60]}...")
            return "\n".join(parts)

        # Overall assessment
        avg_score = sum(category_scores.values()) / len(category_scores) if category_scores else 0

        if avg_score > 40:
            parts.append("News impact: VERY POSITIVE")
        elif avg_score > 15:
            parts.append("News impact: POSITIVE")
        elif avg_score > -15:
            parts.append("News impact: NEUTRAL")
        elif avg_score > -40:
            parts.append("News impact: NEGATIVE")
        else:
            parts.append("News impact: VERY NEGATIVE")

        # Category breakdown
        for cat, score in category_scores.items():
            impact = "positive" if score > 20 else "negative" if score < -20 else "neutral"
            parts.append(f"- {cat.capitalize()}: {impact} ({score:+.0f})")

        # Key factors
        if factors:
            key_factors = sorted(factors, key=lambda x: abs(x.get('score', 0)), reverse=True)[:2]
            if key_factors:
                parts.append("\nKey news:")
                for f in key_factors:
                    headline = f.get('headline', f.get('reason', 'N/A'))[:50]
                    parts.append(f"  * {headline}...")

        return "\n".join(parts)

    def _calculate_confidence(self, news_data: Dict) -> float:
        """Calculate confidence based on news recency and quantity"""
        headlines = news_data.get('headlines', [])

        if not headlines:
            return 0.3

        # Check recency of most recent news
        now = datetime.now().timestamp()
        recent_count = 0

        for h in headlines:
            published = h.get('published', 0)
            # Handle both timestamp (int) and ISO date string (new yfinance format)
            if isinstance(published, str):
                try:
                    published = parse_date(published).timestamp()
                except:
                    published = 0
            if published and published > 0:
                age_hours = (now - published) / 3600
                if age_hours < 24:
                    recent_count += 1

        # More recent news = higher confidence
        recency_factor = min(1.0, recent_count / 3)

        # More news = higher confidence (up to a point)
        quantity_factor = min(1.0, len(headlines) / 5)

        return 0.4 + (recency_factor * 0.3) + (quantity_factor * 0.3)


# Singleton
_news_pillar: Optional[NewsPillar] = None


async def get_news_pillar() -> NewsPillar:
    """Get or create the NewsPillar singleton"""
    global _news_pillar
    if _news_pillar is None:
        _news_pillar = NewsPillar()
        await _news_pillar.initialize()
    return _news_pillar
