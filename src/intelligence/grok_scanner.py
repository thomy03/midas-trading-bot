"""
Grok Scanner Module
Scanner utilisant l'API Grok (xAI) pour analyser les tendances sur X (Twitter).

L'API Grok offre:
1. Accès en temps réel aux posts X/Twitter
2. Capacités LLM pour analyser le sentiment et les tendances
3. Recherche de posts par mots-clés/cashtags

Coût: ~$25/mois (API key via console.x.ai)

Usage:
    from src.intelligence.grok_scanner import GrokScanner

    scanner = GrokScanner(api_key="xai-...")
    await scanner.initialize()

    # Recherche de tendances financières
    trends = await scanner.search_financial_trends()

    # Analyse d'un symbole spécifique
    analysis = await scanner.analyze_symbol("NVDA")

    # Scan complet avec LLM
    insights = await scanner.full_scan_with_analysis()
"""

import os
import json
import asyncio
import aiohttp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from collections import Counter
import logging
from pathlib import Path
import re

# Setup logging
logger = logging.getLogger(__name__)


def _safe_json_loads(json_str: str) -> dict:
    """Parse JSON with trailing comma fix (V4.8)

    LLMs often produce JSON with trailing commas which is invalid.
    This function cleans up the JSON before parsing.
    """
    if not json_str:
        return {}
    # Remove trailing commas before ] or }
    cleaned = re.sub(r',\s*]', ']', json_str)
    cleaned = re.sub(r',\s*}', '}', cleaned)
    return json.loads(cleaned)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class XPost:
    """Post X (Twitter) récupéré"""
    id: str
    text: str
    author: str
    author_followers: int
    created_at: datetime
    likes: int
    retweets: int
    replies: int
    cashtags: List[str]  # $AAPL, $NVDA, etc.
    hashtags: List[str]
    url: str

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'text': self.text[:500],
            'author': self.author,
            'author_followers': self.author_followers,
            'created_at': self.created_at.isoformat(),
            'likes': self.likes,
            'retweets': self.retweets,
            'replies': self.replies,
            'cashtags': self.cashtags,
            'hashtags': self.hashtags,
            'url': self.url
        }

    @property
    def engagement(self) -> int:
        """Score d'engagement total"""
        return self.likes + self.retweets * 2 + self.replies

    @property
    def influence_score(self) -> float:
        """Score d'influence basé sur followers et engagement"""
        # Log scale pour les followers
        import math
        followers_score = math.log10(max(self.author_followers, 1) + 1) / 7  # 0-1 scale
        engagement_score = min(self.engagement / 1000, 1.0)
        return (followers_score + engagement_score) / 2


@dataclass
class GrokInsight:
    """Insight généré par l'analyse LLM Grok"""
    topic: str
    summary: str
    sentiment: str  # 'bullish', 'bearish', 'neutral', 'mixed'
    sentiment_score: float  # -1 to +1
    confidence: float  # 0 to 1
    key_points: List[str]
    mentioned_symbols: List[str]
    catalysts: List[str]
    risk_factors: List[str]
    source_posts: List[str]  # IDs des posts analysés
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'topic': self.topic,
            'summary': self.summary,
            'sentiment': self.sentiment,
            'sentiment_score': self.sentiment_score,
            'confidence': self.confidence,
            'key_points': self.key_points,
            'mentioned_symbols': self.mentioned_symbols,
            'catalysts': self.catalysts,
            'risk_factors': self.risk_factors,
            'source_posts': self.source_posts,
            'generated_at': self.generated_at.isoformat()
        }


@dataclass
class GrokScanResult:
    """Résultat complet d'un scan Grok"""
    timestamp: datetime
    posts_analyzed: int
    trending_symbols: Dict[str, int]  # symbol -> mention count
    trending_hashtags: Dict[str, int]
    overall_sentiment: float
    insights: List[GrokInsight]
    top_influencers: List[Dict]
    emerging_narratives: List[str]

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'posts_analyzed': self.posts_analyzed,
            'trending_symbols': self.trending_symbols,
            'trending_hashtags': self.trending_hashtags,
            'overall_sentiment': self.overall_sentiment,
            'insights': [i.to_dict() for i in self.insights],
            'top_influencers': self.top_influencers,
            'emerging_narratives': self.emerging_narratives
        }


# =============================================================================
# GROK SCANNER
# =============================================================================

class GrokScanner:
    """
    Scanner utilisant l'API Grok (xAI) pour analyser X/Twitter.

    L'API Grok combine:
    1. Recherche X/Twitter en temps réel
    2. Analyse LLM native (pas besoin d'appeler un autre LLM)

    Configuration requise:
    - GROK_API_KEY dans .env (obtenir sur console.x.ai)
    """

    # API Endpoints
    BASE_URL = "https://api.x.ai/v1"

    ENDPOINTS = {
        'chat': '/chat/completions',      # LLM chat
        'search': '/search',               # Recherche X
        'embeddings': '/embeddings'        # Embeddings
    }

    # Requêtes de recherche prédéfinies
    FINANCIAL_QUERIES = [
        # Trading et marchés
        "$SPY OR $QQQ OR $DIA market",
        "stock market today bullish OR bearish",
        "earnings beat OR miss",
        # Secteurs
        "$NVDA OR $AMD OR $AVGO AI semiconductor",
        "$AAPL OR $MSFT OR $GOOGL tech",
        "biotech FDA approval",
        # Tendances
        "short squeeze gamma",
        "options flow unusual",
        "breaking news stock",
        # Macro
        "fed rate decision powell",
        "inflation CPI report"
    ]

    # Influenceurs financiers connus à surveiller
    FINANCIAL_INFLUENCERS = [
        'jimcramer', 'elonmusk', 'chaikigroup', 'unusual_whales',
        'stocktwits', 'thestockguy', 'traborjack', 'watchersguru',
        'dikiykru', 'whale_alert', 'zaboronsky'
    ]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialise le scanner Grok.

        Args:
            api_key: Clé API Grok (xAI). Si None, utilise GROK_API_KEY de .env
            model: Modèle Grok à utiliser. Si None, utilise GROK_MODEL de .env ou grok-4-1-fast-reasoning
        """
        self.api_key = api_key or os.getenv('GROK_API_KEY')
        self.model = model or os.getenv('GROK_MODEL', 'grok-4-1-fast-reasoning')
        self.session: Optional[aiohttp.ClientSession] = None

        # Rate limiting
        self._request_count = 0
        self._last_reset = datetime.now()
        self._rate_limit = 60  # requests per minute

        # Cache
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

        # Historique des scans
        self._scan_history: List[GrokScanResult] = []

        # Persistance
        self._data_dir = Path("data/grok")
        self._data_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """Initialise la session HTTP"""
        if not self.api_key:
            logger.warning("No Grok API key found. Set GROK_API_KEY in .env")
            return

        if self.session is None:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)

        logger.info(f"Grok scanner initialized (model: {self.model})")

    async def close(self):
        """Ferme la session"""
        if self.session:
            await self.session.close()
            self.session = None

    def is_available(self) -> bool:
        """Vérifie si l'API est disponible (clé configurée)"""
        return self.api_key is not None and len(self.api_key) > 0

    async def _make_request(
        self,
        endpoint: str,
        method: str = 'POST',
        data: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Effectue une requête à l'API Grok avec rate limiting.
        """
        if not self.is_available():
            logger.warning("Grok API not available (no API key)")
            return None

        # Rate limiting
        now = datetime.now()
        if (now - self._last_reset).total_seconds() > 60:
            self._request_count = 0
            self._last_reset = now

        if self._request_count >= self._rate_limit - 5:  # Marge
            wait_time = 60 - (now - self._last_reset).total_seconds()
            logger.warning(f"Grok rate limit proche, attente {wait_time:.0f}s")
            await asyncio.sleep(wait_time)
            self._request_count = 0
            self._last_reset = datetime.now()

        url = f"{self.BASE_URL}{endpoint}"

        try:
            if method == 'POST':
                async with self.session.post(url, json=data) as response:
                    self._request_count += 1
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Grok API error {response.status}: {error_text}")
                        return None
            else:
                async with self.session.get(url, params=data) as response:
                    self._request_count += 1
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Grok API error {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Grok request error: {e}")
            return None

    async def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Optional[str]:
        """
        Appelle le LLM Grok pour une analyse.

        Args:
            prompt: Question/prompt à analyser
            system_prompt: Prompt système optionnel
            temperature: Température (0-1)
            max_tokens: Tokens max de réponse

        Returns:
            Réponse du LLM ou None
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        result = await self._make_request(self.ENDPOINTS['chat'], data=data)

        if result and 'choices' in result:
            return result['choices'][0]['message']['content']

        return None

    async def search_x(
        self,
        query: str,
        max_results: int = 50,
        recent_only: bool = True
    ) -> List[XPost]:
        """
        Recherche des posts X/Twitter via Grok.

        Note: Cette fonctionnalité dépend de l'accès Grok à X.
        En attendant l'API de recherche directe, on utilise le LLM
        pour synthétiser les tendances.

        Args:
            query: Requête de recherche
            max_results: Nombre max de résultats
            recent_only: Seulement les posts récents (24h)

        Returns:
            Liste de posts X
        """
        # Pour l'instant, utiliser le LLM pour obtenir une synthèse
        # L'API de recherche directe X via Grok n'est pas encore publique

        system_prompt = """You are a financial analyst monitoring X (Twitter) for market trends.
When asked about a topic, provide:
1. Current sentiment (bullish/bearish/neutral)
2. Key recent posts/tweets about this topic
3. Notable mentions by influential accounts
4. Any breaking news or catalysts

Format your response as JSON with these fields:
- sentiment: "bullish" | "bearish" | "neutral" | "mixed"
- sentiment_score: -1.0 to 1.0
- summary: Brief summary (2-3 sentences)
- key_points: List of main points
- mentioned_symbols: List of stock tickers mentioned
- influencers_talking: List of notable accounts discussing
- catalysts: Any upcoming events/catalysts
"""

        prompt = f"""Analyze current X/Twitter sentiment and posts about: {query}

Focus on:
- Posts from the last 24 hours
- Financial/trading focused accounts
- Any market-moving news
- Unusual activity or volume mentions

Respond in JSON format."""

        response = await self.chat(prompt, system_prompt=system_prompt, temperature=0.3)

        if not response:
            return []

        # Parser la réponse (le LLM renvoie une synthèse, pas des posts individuels)
        # Pour les posts réels, on aurait besoin de l'API X directe
        try:
            # Essayer de parser le JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = _safe_json_loads(json_match.group())
                # Stocker dans le cache pour utilisation par analyze_trends
                self._cache[f"search:{query}"] = (data, datetime.now())
                logger.info(f"Grok search for '{query}': {data.get('sentiment', 'unknown')}")
        except json.JSONDecodeError:
            logger.warning(f"Could not parse Grok response as JSON")

        return []  # Pas de posts individuels sans API X directe

    async def analyze_symbol(self, symbol: str) -> Optional[GrokInsight]:
        """
        Analyse approfondie d'un symbole via Grok.

        Args:
            symbol: Ticker du symbole (ex: 'NVDA')

        Returns:
            Insight détaillé ou None
        """
        if not self.is_available():
            return None

        system_prompt = """You are an expert financial analyst with real-time access to X (Twitter).
Analyze the current social media sentiment and discussion around the given stock symbol.

Provide your analysis in JSON format with these exact fields:
{
    "sentiment": "bullish" | "bearish" | "neutral" | "mixed",
    "sentiment_score": float (-1.0 to 1.0),
    "confidence": float (0 to 1),
    "summary": "Brief 2-3 sentence summary",
    "key_points": ["point1", "point2", ...],
    "catalysts": ["upcoming earnings", "FDA decision", ...],
    "risk_factors": ["high valuation", "competition", ...],
    "notable_mentions": ["@trader1 said X", "@analyst2 noted Y", ...]
}
"""

        prompt = f"""Analyze current X/Twitter sentiment for ${symbol}

Consider:
1. Recent posts mentioning ${symbol} or the company
2. Notable financial influencers discussing it
3. Any breaking news or upcoming catalysts
4. Options flow mentions
5. Unusual activity signals

Be specific and cite actual trends you're seeing."""

        response = await self.chat(prompt, system_prompt=system_prompt, temperature=0.3)

        if not response:
            return None

        try:
            # Extraire le JSON de la réponse
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = _safe_json_loads(json_match.group())

                return GrokInsight(
                    topic=f"${symbol} Analysis",
                    summary=data.get('summary', ''),
                    sentiment=data.get('sentiment', 'neutral'),
                    sentiment_score=float(data.get('sentiment_score', 0)),
                    confidence=float(data.get('confidence', 0.5)),
                    key_points=data.get('key_points', []),
                    mentioned_symbols=[symbol],
                    catalysts=data.get('catalysts', []),
                    risk_factors=data.get('risk_factors', []),
                    source_posts=[]
                )
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Error parsing Grok response for {symbol}: {e}")

        return None

    async def search_financial_trends(self) -> List[GrokInsight]:
        """
        Recherche les tendances financières actuelles sur X.

        Returns:
            Liste d'insights sur les tendances
        """
        if not self.is_available():
            return []

        insights = []

        # Analyser plusieurs requêtes en parallèle
        for query in self.FINANCIAL_QUERIES[:5]:  # Limiter pour éviter rate limit
            await self.search_x(query)
            await asyncio.sleep(1)  # Petit délai entre requêtes

        # Synthèse globale
        system_prompt = """You are a financial market analyst monitoring X (Twitter) in real-time.
Identify the top emerging trends and narratives in financial discussions.

Return your analysis as a JSON array of trend objects:
[
    {
        "topic": "AI Semiconductors Rally",
        "summary": "Brief description",
        "sentiment": "bullish",
        "sentiment_score": 0.7,
        "confidence": 0.8,
        "key_points": ["point1", "point2"],
        "symbols": ["NVDA", "AMD"],
        "catalysts": ["earnings", "product launch"]
    },
    ...
]

Focus on actionable trends with clear trading implications."""

        prompt = """Analyze current financial trends on X/Twitter:

1. What are the top 5 trending financial topics right now?
2. Which stocks are generating the most buzz?
3. What's the overall market sentiment?
4. Any emerging narratives (AI, rate cuts, earnings, etc.)?
5. Notable calls from influential traders?

Return as JSON array."""

        response = await self.chat(prompt, system_prompt=system_prompt, temperature=0.3)

        if response:
            try:
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    trends_data = _safe_json_loads(json_match.group())

                    for trend in trends_data:
                        insight = GrokInsight(
                            topic=trend.get('topic', 'Unknown Trend'),
                            summary=trend.get('summary', ''),
                            sentiment=trend.get('sentiment', 'neutral'),
                            sentiment_score=float(trend.get('sentiment_score', 0)),
                            confidence=float(trend.get('confidence', 0.5)),
                            key_points=trend.get('key_points', []),
                            mentioned_symbols=trend.get('symbols', []),
                            catalysts=trend.get('catalysts', []),
                            risk_factors=trend.get('risk_factors', []),
                            source_posts=[]
                        )
                        insights.append(insight)

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing trends response: {e}")

        return insights

    async def full_scan_with_analysis(self) -> Optional[GrokScanResult]:
        """
        Scan complet avec analyse LLM des tendances X/Twitter.

        Combine:
        1. Recherche de trending topics
        2. Analyse de sentiment globale
        3. Identification des narratifs émergents
        4. Top influenceurs actifs

        Returns:
            Résultat complet du scan
        """
        if not self.is_available():
            logger.warning("Grok API not available")
            return None

        logger.info("Starting full Grok scan with analysis...")

        # 1. Obtenir les tendances financières
        insights = await self.search_financial_trends()

        # 2. Analyse globale du marché
        market_analysis = await self._analyze_market_sentiment()

        # 3. Extraire les symboles et hashtags trending
        trending_symbols = Counter()
        trending_hashtags = Counter()

        for insight in insights:
            for symbol in insight.mentioned_symbols:
                trending_symbols[symbol] += 1

        # 4. Calculer le sentiment global
        if insights:
            overall_sentiment = sum(i.sentiment_score for i in insights) / len(insights)
        else:
            overall_sentiment = 0.0

        # 5. Identifier les narratifs émergents
        emerging_narratives = self._extract_narratives(insights)

        # 6. Top influenceurs (basé sur l'analyse)
        top_influencers = await self._get_active_influencers()

        result = GrokScanResult(
            timestamp=datetime.now(),
            posts_analyzed=len(insights) * 10,  # Estimation
            trending_symbols=dict(trending_symbols.most_common(20)),
            trending_hashtags=dict(trending_hashtags.most_common(10)),
            overall_sentiment=overall_sentiment,
            insights=insights,
            top_influencers=top_influencers,
            emerging_narratives=emerging_narratives
        )

        # Sauvegarder le résultat
        self._scan_history.append(result)
        self._save_scan_result(result)

        logger.info(
            f"Grok scan complete: {len(insights)} insights, "
            f"sentiment={overall_sentiment:.2f}, "
            f"{len(trending_symbols)} trending symbols"
        )

        return result

    async def _analyze_market_sentiment(self) -> Dict:
        """Analyse le sentiment global du marché"""
        prompt = """What is the current overall market sentiment on X/Twitter?

Consider:
- Major index discussions ($SPY, $QQQ, $DIA)
- Fear vs greed indicators in posts
- Bullish vs bearish ratio
- Any macro concerns (Fed, inflation, geopolitics)

Respond with:
{
    "sentiment": "bullish" | "bearish" | "neutral" | "fearful",
    "score": -1.0 to 1.0,
    "main_drivers": ["driver1", "driver2"],
    "concerns": ["concern1", "concern2"]
}"""

        response = await self.chat(prompt, temperature=0.3)

        if response:
            try:
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    return _safe_json_loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {"sentiment": "neutral", "score": 0.0}

    async def _get_active_influencers(self) -> List[Dict]:
        """Identifie les influenceurs financiers actifs"""
        prompt = f"""Which financial influencers on X are most active today?

Known accounts to check: {', '.join(self.FINANCIAL_INFLUENCERS[:10])}

Return as JSON:
[
    {{"username": "name", "recent_topic": "what they're discussing", "sentiment": "bullish/bearish"}}
]"""

        response = await self.chat(prompt, temperature=0.3)

        if response:
            try:
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    return _safe_json_loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return []

    def _extract_narratives(self, insights: List[GrokInsight]) -> List[str]:
        """Extrait les narratifs des insights"""
        narratives = []

        for insight in insights:
            # Le topic est souvent le narratif
            if insight.topic and insight.confidence > 0.5:
                narratives.append(insight.topic)

            # Ajouter les catalyseurs comme narratifs potentiels
            for catalyst in insight.catalysts:
                if len(catalyst) > 10:  # Filtrer les courts
                    narratives.append(catalyst)

        # Dédupliquer et limiter
        seen = set()
        unique_narratives = []
        for n in narratives:
            n_lower = n.lower()
            if n_lower not in seen:
                seen.add(n_lower)
                unique_narratives.append(n)

        return unique_narratives[:10]

    def _save_scan_result(self, result: GrokScanResult):
        """Sauvegarde le résultat du scan"""
        filename = f"scan_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self._data_dir / filename

        try:
            with open(filepath, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving scan result: {e}")

    def get_last_scan(self) -> Optional[GrokScanResult]:
        """Retourne le dernier scan"""
        return self._scan_history[-1] if self._scan_history else None

    async def get_symbol_details_for_narrative(self, symbol: str) -> Dict:
        """
        Récupère les détails enrichis d'un symbole pour le NarrativeGenerator.

        Extrait les thèmes, catalyseurs et key tweets de l'analyse Grok.

        Args:
            symbol: Ticker du symbole

        Returns:
            Dict avec sentiment, themes, catalysts, key_tweets pour le narratif
        """
        result = {
            'symbol': symbol,
            'sentiment': 0.5,
            'sentiment_label': 'neutral',
            'themes': [],
            'catalysts': [],
            'key_tweets': [],
            'analyst_mentions': 0,
            'catalyst_linked': False,
            'confidence': 0.5
        }

        if not self.is_available():
            return result

        try:
            # Analyser le symbole via Grok LLM
            insight = await self.analyze_symbol(symbol)

            if insight:
                result['sentiment'] = (insight.sentiment_score + 1) / 2  # Convert -1/+1 to 0/1
                result['sentiment_label'] = insight.sentiment
                result['themes'] = insight.key_points[:5]  # Top 5 themes
                result['catalysts'] = insight.catalysts[:3]
                result['confidence'] = insight.confidence

                # Simuler des "key tweets" à partir du summary
                if insight.summary:
                    result['key_tweets'] = [insight.summary]

                # Check si catalyseurs liés
                result['catalyst_linked'] = len(insight.catalysts) > 0

                # Compter mentions d'analystes (basé sur notable_mentions du cache)
                cache_key = f"search:${symbol}"
                if cache_key in self._cache:
                    cached_data, _ = self._cache[cache_key]
                    influencers = cached_data.get('influencers_talking', [])
                    result['analyst_mentions'] = len(influencers)

            logger.debug(f"Grok details for {symbol}: sentiment={result['sentiment']:.2f}, "
                        f"themes={len(result['themes'])}, catalysts={len(result['catalysts'])}")

        except Exception as e:
            logger.warning(f"Failed to get Grok details for {symbol}: {e}")

        return result

    async def get_symbol_buzz(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Obtient le "buzz" pour une liste de symboles.

        Args:
            symbols: Liste de tickers

        Returns:
            Dict avec sentiment et mentions par symbole
        """
        results = {}

        # Analyser par lots de 5 pour éviter rate limit
        for i in range(0, len(symbols), 5):
            batch = symbols[i:i+5]

            prompt = f"""Analyze X/Twitter buzz for these stocks: {', '.join(f'${s}' for s in batch)}

For each stock, provide:
{{
    "symbol": {{
        "mention_level": "high" | "medium" | "low" | "none",
        "sentiment": "bullish" | "bearish" | "neutral",
        "sentiment_score": -1.0 to 1.0,
        "key_topic": "main discussion topic"
    }}
}}"""

            response = await self.chat(prompt, temperature=0.3)

            if response:
                try:
                    json_match = re.search(r'\{[\s\S]*\}', response)
                    if json_match:
                        data = _safe_json_loads(json_match.group())
                        results.update(data)
                except json.JSONDecodeError:
                    pass

            # Petit délai entre les batches
            if i + 5 < len(symbols):
                await asyncio.sleep(2)

        return results


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_grok_scanner_instance: Optional[GrokScanner] = None


async def get_grok_scanner(api_key: Optional[str] = None) -> GrokScanner:
    """
    Factory pour obtenir une instance du scanner Grok.

    Singleton pattern.
    """
    global _grok_scanner_instance

    if _grok_scanner_instance is None:
        _grok_scanner_instance = GrokScanner(api_key=api_key)
        await _grok_scanner_instance.initialize()

    return _grok_scanner_instance


async def close_grok_scanner():
    """Ferme le scanner global"""
    global _grok_scanner_instance

    if _grok_scanner_instance:
        await _grok_scanner_instance.close()
        _grok_scanner_instance = None


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        print("=== Grok Scanner Test ===\n")

        # Check if API key is available
        api_key = os.getenv('GROK_API_KEY')
        if not api_key:
            print("GROK_API_KEY not set in environment.")
            print("Get your API key at: https://console.x.ai")
            print("\nRunning in simulation mode...\n")

        scanner = GrokScanner()
        await scanner.initialize()

        try:
            if scanner.is_available():
                # Full scan
                print("Running full Grok scan...")
                result = await scanner.full_scan_with_analysis()

                if result:
                    print(f"\nPosts analyzed: {result.posts_analyzed}")
                    print(f"Overall sentiment: {result.overall_sentiment:.2f}")

                    print("\nTrending Symbols:")
                    for symbol, count in list(result.trending_symbols.items())[:10]:
                        print(f"  ${symbol}: {count} mentions")

                    print("\nEmerging Narratives:")
                    for narrative in result.emerging_narratives[:5]:
                        print(f"  - {narrative}")

                    print("\nInsights:")
                    for insight in result.insights[:3]:
                        print(f"  [{insight.sentiment}] {insight.topic}")
                        print(f"    {insight.summary}")
            else:
                print("Grok API not available. Set GROK_API_KEY to enable.")

        finally:
            await scanner.close()

    asyncio.run(main())
