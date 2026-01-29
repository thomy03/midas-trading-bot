"""
Sentiment Pillar - Social media sentiment analysis.

Analyzes:
- Grok X/Twitter sentiment (via xAI API)
- StockTwits sentiment (if available)
- Reddit mentions (r/wallstreetbets, r/stocks)
- Overall social buzz

Contributes 25% to final decision score.
"""

import os
import httpx
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import re

from .base import BasePillar, PillarScore

logger = logging.getLogger(__name__)


class SentimentPillar(BasePillar):
    """
    Sentiment analysis pillar using Grok for X/Twitter analysis.
    """

    def __init__(self, weight: float = 0.25):
        super().__init__(weight)
        self._cache_ttl = 600  # 10 min cache for sentiment

        # API config
        self.grok_api_key = os.getenv('GROK_API_KEY', '')
        self.grok_model = os.getenv('GROK_MODEL', 'grok-4-1-fast-reasoning')

        self._client: Optional[httpx.AsyncClient] = None

        # Sentiment weights
        self.source_weights = {
            'twitter': 0.50,    # X/Twitter via Grok
            'stocktwits': 0.30,
            'reddit': 0.20
        }

    def get_name(self) -> str:
        return "Sentiment"

    async def initialize(self):
        """Initialize HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def analyze(
        self,
        symbol: str,
        data: Dict[str, Any]
    ) -> PillarScore:
        """
        Analyze social sentiment for a symbol.

        Args:
            symbol: Stock symbol
            data: Optional pre-fetched sentiment data

        Returns:
            PillarScore with sentiment analysis result
        """
        await self.initialize()

        factors = []
        source_scores = {}
        total_weight = 0

        logger.info(f"[SENTIMENT] {symbol}: Starting sentiment analysis (Grok API: {'configured' if self.grok_api_key else 'NOT configured'})...")

        try:
            # 1. X/Twitter sentiment via Grok
            if self.grok_api_key:
                twitter_result = await self._analyze_twitter(symbol)
                if twitter_result:
                    source_scores['twitter'] = twitter_result['score']
                    factors.append(twitter_result)
                    total_weight += self.source_weights['twitter']
                    logger.info(f"[SENTIMENT] {symbol}: X/Twitter score={twitter_result['score']:.0f}")
                else:
                    logger.warning(f"[SENTIMENT] {symbol}: X/Twitter analysis returned no result")
            else:
                logger.warning(f"[SENTIMENT] {symbol}: Grok API key not configured - skipping X/Twitter analysis")

            # 2. Pre-fetched sentiment data (from data dict)
            if 'stocktwits' in data:
                st_result = self._parse_stocktwits(data['stocktwits'])
                if st_result:
                    source_scores['stocktwits'] = st_result['score']
                    factors.append(st_result)
                    total_weight += self.source_weights['stocktwits']

            if 'reddit' in data:
                reddit_result = self._parse_reddit(data['reddit'])
                if reddit_result:
                    source_scores['reddit'] = reddit_result['score']
                    factors.append(reddit_result)
                    total_weight += self.source_weights['reddit']

            # Calculate weighted average
            if total_weight > 0:
                total_score = sum(
                    source_scores.get(src, 0) * self.source_weights.get(src, 0)
                    for src in source_scores
                ) / total_weight * (total_weight / sum(self.source_weights.values()))
            else:
                total_score = 0

            # Data quality based on sources available
            data_quality = total_weight / sum(self.source_weights.values())

            # Generate reasoning
            reasoning = self._generate_reasoning(symbol, source_scores, factors)

            # Confidence based on agreement
            confidence = self._calculate_confidence(source_scores)

            # V4.4: Verbose logging
            sources_str = ", ".join(source_scores.keys()) if source_scores else "none"
            sentiment = "bullish" if total_score > 20 else "bearish" if total_score < -20 else "neutral"
            logger.info(f"[SENTIMENT] {symbol}: Score={total_score:.1f}/100 ({sentiment}) | Sources: {sources_str} | Quality={data_quality:.0%}")

            return self._create_score(
                score=total_score,
                reasoning=reasoning,
                factors=factors,
                confidence=confidence,
                data_quality=data_quality
            )

        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            return self._create_score(
                score=0,
                reasoning=f"Sentiment analysis error: {str(e)}",
                data_quality=0.0
            )

    async def _analyze_twitter(self, symbol: str) -> Optional[Dict]:
        """Analyze X/Twitter sentiment via Grok"""
        if not self.grok_api_key or not self._client:
            return None

        prompt = f"""Analyze the current sentiment on X/Twitter for the stock ${symbol}.

Consider:
1. Overall sentiment (bullish/bearish/neutral)
2. Volume of mentions (high/medium/low)
3. Key themes being discussed
4. Any notable influencers or institutional mentions
5. Recent news or events driving discussion

Return your analysis in this exact JSON format:
{{
    "sentiment_score": <number from -100 to 100>,
    "sentiment_label": "bullish" | "bearish" | "neutral",
    "volume": "high" | "medium" | "low",
    "confidence": <number from 0 to 1>,
    "key_themes": ["theme1", "theme2"],
    "summary": "One sentence summary"
}}

Be objective and base your analysis on actual social media trends."""

        try:
            headers = {
                "Authorization": f"Bearer {self.grok_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.grok_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 500
            }

            response = await self._client.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                logger.warning(f"Grok API error: {response.status_code}")
                return None

            result = response.json()
            content = result['choices'][0]['message']['content']

            # Parse JSON from response
            import json
            # Extract JSON from potential markdown
            json_match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
            if json_match:
                # V4.8: Fix common LLM JSON errors (trailing commas)
                json_str = json_match.group()
                json_str = re.sub(r',\s*]', ']', json_str)
                json_str = re.sub(r',\s*}', '}', json_str)
                sentiment_data = json.loads(json_str)

                score = sentiment_data.get('sentiment_score', 0)
                label = sentiment_data.get('sentiment_label', 'neutral')
                volume = sentiment_data.get('volume', 'medium')
                summary = sentiment_data.get('summary', '')

                # Adjust score based on volume
                volume_multiplier = {'high': 1.2, 'medium': 1.0, 'low': 0.8}
                adjusted_score = score * volume_multiplier.get(volume, 1.0)
                adjusted_score = max(-100, min(100, adjusted_score))

                return {
                    'source': 'X/Twitter',
                    'score': adjusted_score,
                    'message': f"{label.capitalize()} sentiment ({volume} volume): {summary}",
                    'raw_data': sentiment_data
                }

        except Exception as e:
            logger.warning(f"Twitter analysis failed: {e}")

        return None

    def _parse_stocktwits(self, data: Dict) -> Optional[Dict]:
        """Parse StockTwits sentiment data"""
        try:
            bullish = data.get('bullish', 0)
            bearish = data.get('bearish', 0)
            total = bullish + bearish

            if total == 0:
                return None

            # Calculate score (-100 to 100)
            ratio = bullish / total
            score = (ratio - 0.5) * 200  # 0.5 = neutral

            # Volume consideration
            volume = data.get('volume', 'medium')
            if volume == 'high' and total > 100:
                volume_mult = 1.2
            elif volume == 'low' or total < 20:
                volume_mult = 0.7
            else:
                volume_mult = 1.0

            adjusted_score = score * volume_mult
            adjusted_score = max(-100, min(100, adjusted_score))

            label = "bullish" if ratio > 0.55 else "bearish" if ratio < 0.45 else "neutral"

            return {
                'source': 'StockTwits',
                'score': adjusted_score,
                'message': f"StockTwits: {label} ({bullish} bullish / {bearish} bearish)",
                'raw_data': {'bullish': bullish, 'bearish': bearish, 'ratio': ratio}
            }

        except Exception as e:
            logger.warning(f"StockTwits parse failed: {e}")
            return None

    def _parse_reddit(self, data: Dict) -> Optional[Dict]:
        """Parse Reddit sentiment data"""
        try:
            mentions = data.get('mentions', 0)
            sentiment = data.get('sentiment', 0)  # Pre-calculated -1 to 1
            subreddits = data.get('subreddits', [])

            if mentions == 0:
                return None

            # Convert to -100 to 100 scale
            score = sentiment * 100

            # Adjust based on mention count
            if mentions > 50:
                volume = "high"
                score *= 1.1
            elif mentions > 10:
                volume = "medium"
            else:
                volume = "low"
                score *= 0.8

            score = max(-100, min(100, score))

            label = "bullish" if sentiment > 0.2 else "bearish" if sentiment < -0.2 else "mixed"
            subs = ", ".join(subreddits[:3]) if subreddits else "various"

            return {
                'source': 'Reddit',
                'score': score,
                'message': f"Reddit ({subs}): {label} ({mentions} mentions)",
                'raw_data': data
            }

        except Exception as e:
            logger.warning(f"Reddit parse failed: {e}")
            return None

    def _generate_reasoning(
        self,
        symbol: str,
        source_scores: Dict[str, float],
        factors: List[Dict]
    ) -> str:
        """Generate human-readable reasoning"""
        parts = [f"Social sentiment analysis for ${symbol}"]

        if not source_scores:
            parts.append("No sentiment data available")
            return "\n".join(parts)

        # Overall assessment
        avg_score = sum(source_scores.values()) / len(source_scores) if source_scores else 0

        if avg_score > 50:
            parts.append("Overall: VERY BULLISH")
        elif avg_score > 20:
            parts.append("Overall: BULLISH")
        elif avg_score > -20:
            parts.append("Overall: NEUTRAL")
        elif avg_score > -50:
            parts.append("Overall: BEARISH")
        else:
            parts.append("Overall: VERY BEARISH")

        # Source details
        for factor in factors:
            parts.append(f"- {factor.get('message', 'N/A')}")

        return "\n".join(parts)

    def _calculate_confidence(self, source_scores: Dict[str, float]) -> float:
        """Calculate confidence based on source agreement"""
        if len(source_scores) < 2:
            return 0.5

        scores = list(source_scores.values())

        # Check if all sources agree on direction
        all_positive = all(s > 0 for s in scores)
        all_negative = all(s < 0 for s in scores)

        if all_positive or all_negative:
            # High agreement
            return 0.8
        else:
            # Mixed signals, lower confidence
            return 0.5


# Singleton
_sentiment_pillar: Optional[SentimentPillar] = None


async def get_sentiment_pillar() -> SentimentPillar:
    """Get or create the SentimentPillar singleton"""
    global _sentiment_pillar
    if _sentiment_pillar is None:
        _sentiment_pillar = SentimentPillar()
        await _sentiment_pillar.initialize()
    return _sentiment_pillar
