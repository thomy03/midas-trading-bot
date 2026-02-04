"""
Grok Scanner V2 - Intelligent Discovery
Scanner autonome utilisant l'API Grok (xAI) pour découvrir ce qui bouge sur X/Twitter.

V2 Features:
1. DISCOVER PHASE: Grok demande lui-même ce qui bouge (pas de queries fixes)
2. DEEP DIVE: Pour chaque découverte, creuser automatiquement (pourquoi, qui, catalyseur)
3. CHAIN OF THOUGHT: Recherches en cascade (NVDA → AMD, AVGO, TSM)
4. MEMORY & FEEDBACK: Mémorise ce qui a marché pour améliorer les futures recherches

Usage:
    from src.intelligence.grok_scanner import GrokScanner
    
    scanner = GrokScanner(api_key="xai-...")
    await scanner.initialize()
    
    # Full intelligent scan
    result = await scanner.full_scan_with_analysis()
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

logger = logging.getLogger(__name__)


def _safe_json_loads(json_str: str) -> dict:
    """Parse JSON - try standard first, then with minimal fixes"""
    if not json_str:
        return {}
    
    # Remove markdown code blocks
    cleaned = json_str
    code_block = chr(96) * 3  # backticks
    if code_block + 'json' in cleaned:
        cleaned = cleaned.split(code_block + 'json')[1].split(code_block)[0]
    elif code_block in cleaned:
        parts = cleaned.split(code_block)
        if len(parts) >= 2:
            cleaned = parts[1]
    
    cleaned = cleaned.strip()
    
    # Try standard parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Minimal fixes only if standard fails
    # Remove trailing commas
    cleaned = re.sub(r',\s*]', ']', cleaned)
    cleaned = re.sub(r',\s*}', '}', cleaned)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse failed: {e}. Returning empty dict.")
        return {}



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
    cashtags: List[str]
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
        return self.likes + self.retweets * 2 + self.replies

    @property
    def influence_score(self) -> float:
        import math
        followers_score = math.log10(max(self.author_followers, 1) + 1) / 7
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
    source_posts: List[str]
    generated_at: datetime = field(default_factory=datetime.now)
    # V2: Ajout du "why" - pourquoi cette découverte est intéressante
    discovery_reason: str = ""
    related_symbols: List[str] = field(default_factory=list)

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
            'generated_at': self.generated_at.isoformat(),
            'discovery_reason': self.discovery_reason,
            'related_symbols': self.related_symbols
        }


@dataclass
class GrokScanResult:
    """Résultat complet d'un scan Grok"""
    timestamp: datetime
    posts_analyzed: int
    trending_symbols: Dict[str, int]
    trending_hashtags: Dict[str, int]
    overall_sentiment: float
    insights: List[GrokInsight]
    top_influencers: List[Dict]
    emerging_narratives: List[str]
    # V2: Metadata sur la découverte
    discovery_chain: List[str] = field(default_factory=list)
    memory_hits: int = 0  # Combien de découvertes basées sur la mémoire

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'posts_analyzed': self.posts_analyzed,
            'trending_symbols': self.trending_symbols,
            'trending_hashtags': self.trending_hashtags,
            'overall_sentiment': self.overall_sentiment,
            'insights': [i.to_dict() for i in self.insights],
            'top_influencers': self.top_influencers,
            'emerging_narratives': self.emerging_narratives,
            'discovery_chain': self.discovery_chain,
            'memory_hits': self.memory_hits
        }


# =============================================================================
# SEARCH MEMORY - V2 Feature
# =============================================================================

@dataclass
class SearchMemoryEntry:
    """Une entrée de mémoire de recherche"""
    query: str
    timestamp: datetime
    symbols_discovered: List[str]
    sentiment_at_discovery: float
    # Performance tracking - rempli plus tard
    performance_tracked: bool = False
    symbols_that_performed: List[str] = field(default_factory=list)
    performance_score: float = 0.0  # -1 to +1, how well did the discovery do
    
    def to_dict(self) -> Dict:
        return {
            'query': self.query,
            'timestamp': self.timestamp.isoformat(),
            'symbols_discovered': self.symbols_discovered,
            'sentiment_at_discovery': self.sentiment_at_discovery,
            'performance_tracked': self.performance_tracked,
            'symbols_that_performed': self.symbols_that_performed,
            'performance_score': self.performance_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SearchMemoryEntry':
        return cls(
            query=data['query'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            symbols_discovered=data.get('symbols_discovered', []),
            sentiment_at_discovery=data.get('sentiment_at_discovery', 0),
            performance_tracked=data.get('performance_tracked', False),
            symbols_that_performed=data.get('symbols_that_performed', []),
            performance_score=data.get('performance_score', 0)
        )


class SearchMemory:
    """Gestion de la mémoire des recherches pour le feedback loop"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.memory_file = data_dir / "search_memory.json"
        self.entries: List[SearchMemoryEntry] = []
        self._load()
    
    def _load(self):
        """Charge la mémoire depuis le fichier"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.entries = [SearchMemoryEntry.from_dict(e) for e in data.get('entries', [])]
                    logger.info(f"Loaded {len(self.entries)} search memory entries")
            except Exception as e:
                logger.error(f"Failed to load search memory: {e}")
                self.entries = []
    
    def _save(self):
        """Sauvegarde la mémoire"""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.memory_file, 'w') as f:
                json.dump({
                    'last_updated': datetime.now().isoformat(),
                    'total_entries': len(self.entries),
                    'entries': [e.to_dict() for e in self.entries[-500:]]  # Garde les 500 derniers
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save search memory: {e}")
    
    def add_discovery(self, query: str, symbols: List[str], sentiment: float):
        """Ajoute une nouvelle découverte"""
        entry = SearchMemoryEntry(
            query=query,
            timestamp=datetime.now(),
            symbols_discovered=symbols,
            sentiment_at_discovery=sentiment
        )
        self.entries.append(entry)
        self._save()
        logger.debug(f"Added memory entry: {query} -> {symbols}")
    
    def update_performance(self, symbol: str, performed_well: bool):
        """Met à jour la performance d'un symbole dans les entrées récentes"""
        cutoff = datetime.now() - timedelta(days=7)
        
        for entry in reversed(self.entries):
            if entry.timestamp < cutoff:
                break
            if symbol in entry.symbols_discovered and not entry.performance_tracked:
                if performed_well:
                    entry.symbols_that_performed.append(symbol)
                    entry.performance_score = min(1.0, entry.performance_score + 0.2)
                else:
                    entry.performance_score = max(-1.0, entry.performance_score - 0.1)
        
        self._save()
    
    def get_successful_patterns(self, limit: int = 10) -> List[Dict]:
        """Retourne les patterns de recherche qui ont bien fonctionné"""
        # Filtrer les entrées avec bonne performance
        successful = [e for e in self.entries if e.performance_score > 0.3]
        
        # Trier par score et date
        successful.sort(key=lambda x: (x.performance_score, x.timestamp), reverse=True)
        
        patterns = []
        for entry in successful[:limit]:
            patterns.append({
                'query': entry.query,
                'symbols_worked': entry.symbols_that_performed,
                'score': entry.performance_score
            })
        
        return patterns
    
    def get_failed_patterns(self, limit: int = 10) -> List[str]:
        """Retourne les queries qui n'ont pas bien fonctionné (à éviter)"""
        failed = [e for e in self.entries if e.performance_score < -0.2 and e.performance_tracked]
        failed.sort(key=lambda x: x.performance_score)
        return [e.query for e in failed[:limit]]
    
    def get_recent_discoveries(self, hours: int = 24) -> List[str]:
        """Retourne les symboles découverts récemment (pour éviter les redondances)"""
        cutoff = datetime.now() - timedelta(hours=hours)
        symbols = set()
        for entry in self.entries:
            if entry.timestamp > cutoff:
                symbols.update(entry.symbols_discovered)
        return list(symbols)
    
    def get_context_for_prompt(self) -> str:
        """Génère le contexte mémoire à injecter dans le prompt"""
        successful = self.get_successful_patterns(5)
        failed = self.get_failed_patterns(3)
        recent = self.get_recent_discoveries(12)
        
        context_parts = []
        
        if successful:
            context_parts.append("RECHERCHES QUI ONT BIEN MARCHÉ RÉCEMMENT:")
            for p in successful:
                context_parts.append(f"  - '{p['query']}' → {', '.join(p['symbols_worked'])} (score: {p['score']:.1f})")
        
        if failed:
            context_parts.append("\nRECHERCHES À ÉVITER (n'ont pas fonctionné):")
            for q in failed:
                context_parts.append(f"  - '{q}'")
        
        if recent:
            context_parts.append(f"\nSYMBOLES DÉJÀ DÉCOUVERTS (12h): {', '.join(recent[:20])}")
        
        return "\n".join(context_parts) if context_parts else "Aucun historique de recherche disponible."


# =============================================================================
# SECTOR MAPPING - Pour Chain of Thought
# =============================================================================

SECTOR_MAPPING = {
    'semiconductors': ['NVDA', 'AMD', 'AVGO', 'TSM', 'INTC', 'QCOM', 'MU', 'ASML', 'KLAC', 'AMAT'],
    'ai_cloud': ['NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'ORCL', 'CRM', 'PLTR', 'AI', 'PATH'],
    'ev': ['TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'F', 'GM', 'PLUG', 'CHPT'],
    'fintech': ['SQ', 'PYPL', 'COIN', 'HOOD', 'SOFI', 'AFRM', 'UPST', 'NU', 'BILL', 'FOUR'],
    'biotech': ['MRNA', 'BNTX', 'REGN', 'VRTX', 'BIIB', 'GILD', 'AMGN', 'ABBV', 'BMY', 'LLY'],
    'crypto': ['BTC', 'ETH', 'COIN', 'MSTR', 'RIOT', 'MARA', 'IBIT', 'GBTC', 'BITO', 'CLSK'],
    'defense': ['LMT', 'RTX', 'NOC', 'GD', 'BA', 'HII', 'LHX', 'TDG', 'LDOS', 'PLTR'],
    'mag7': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
    'retail': ['AMZN', 'WMT', 'TGT', 'COST', 'HD', 'LOW', 'EBAY', 'ETSY', 'W', 'BABA'],
}


def get_related_symbols(symbol: str, current_discovered: List[str]) -> List[str]:
    """Trouve les symboles liés à explorer (Chain of Thought)"""
    related = []
    for sector, symbols in SECTOR_MAPPING.items():
        if symbol.upper() in symbols:
            for s in symbols:
                if s != symbol.upper() and s not in current_discovered:
                    related.append(s)
    return related[:5]  # Max 5 pour éviter explosion


# =============================================================================
# GROK SCANNER V2
# =============================================================================

class GrokScanner:
    """
    Scanner V2 - Intelligent Discovery
    
    Au lieu de requêtes statiques, Grok:
    1. Demande ce qui bouge
    2. Creuse automatiquement ce qui est intéressant
    3. Suit les connexions sectorielles
    4. Apprend de ce qui a marché
    """

    BASE_URL = "https://api.x.ai/v1"
    ENDPOINTS = {
        'chat': '/chat/completions',
        'search': '/search',
        'embeddings': '/embeddings'
    }

    # V2: Plus de FINANCIAL_QUERIES statiques - on laisse Grok découvrir

    FINANCIAL_INFLUENCERS = [
        'jimcramer', 'elonmusk', 'chaikigroup', 'unusual_whales',
        'stocktwits', 'thestockguy', 'traborjack', 'watchersguru',
        'dikiykru', 'whale_alert', 'zaboronsky'
    ]

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv('GROK_API_KEY')
        self.model = model or os.getenv('GROK_MODEL', 'grok-3-fast')
        self.session: Optional[aiohttp.ClientSession] = None

        # Rate limiting
        self._request_count = 0
        self._last_reset = datetime.now()
        self._rate_limit = 60

        # Cache
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

        # Historique des scans
        self._scan_history: List[GrokScanResult] = []

        # Persistance
        self._data_dir = Path("data/grok")
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
        # V2: Search Memory
        self._memory = SearchMemory(self._data_dir)

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

        logger.info(f"Grok scanner V2 initialized (model: {self.model})")

    async def close(self):
        """Ferme la session"""
        if self.session:
            await self.session.close()
            self.session = None

    def is_available(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0

    async def _make_request(
        self,
        endpoint: str,
        method: str = 'POST',
        data: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Effectue une requête à l'API Grok avec rate limiting"""
        if not self.is_available():
            logger.warning("Grok API not available (no API key)")
            return None

        now = datetime.now()
        if (now - self._last_reset).total_seconds() > 60:
            self._request_count = 0
            self._last_reset = now

        if self._request_count >= self._rate_limit - 5:
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
        max_tokens: int = 1500
    ) -> Optional[str]:
        """Appelle le LLM Grok"""
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

    # =========================================================================
    # V2: DISCOVER PHASE - Autonomous Discovery
    # =========================================================================

    async def _autonomous_discover(self) -> List[Dict]:
        """
        Phase 1: Grok découvre par lui-même ce qui bouge.
        Pas de queries prédéfinies - on lui demande directement.
        """
        memory_context = self._memory.get_context_for_prompt()
        
        system_prompt = """You are a financial analyst scanning X/Twitter for trading opportunities.
Return ONLY valid JSON, no explanations."""

        prompt = """Scan X/Twitter NOW - What's trending in finance?

List 5-7 topics being discussed right now with:
- topic: short name
- symbols: array of tickers
- sentiment: bullish/bearish/mixed
- sentiment_score: -1.0 to 1.0
- urgency: high/medium/low
- why_interesting: why it matters
- confidence: 0.0 to 1.0

JSON format:
{
    "discoveries": [...],
    "market_mood": "bullish"|"bearish"|"neutral",
    "worth_deep_dive": ["SYM1", "SYM2"]
}"""

        response = await self.chat(prompt, system_prompt=system_prompt, temperature=0.5)
        
        if not response:
            return []
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = _safe_json_loads(json_match.group())
                
                # Enregistrer les découvertes en mémoire
                for disc in data.get('discoveries', []):
                    self._memory.add_discovery(
                        query=disc.get('topic', 'unknown'),
                        symbols=disc.get('symbols', []),
                        sentiment=disc.get('sentiment_score', 0)
                    )
                
                logger.info(f"Autonomous discovery: {len(data.get('discoveries', []))} topics found")
                return data.get('discoveries', []), data.get('worth_deep_dive', [])
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in autonomous discover: {e}")
        
        return [], []

    # =========================================================================
    # V2: DEEP DIVE PHASE - Explore discoveries
    # =========================================================================

    async def _deep_dive(self, symbol: str, context: str = "") -> Optional[GrokInsight]:
        """
        Phase 2: Creuse automatiquement une découverte.
        - Pourquoi ça bouge ?
        - Qui en parle ?
        - Catalyseur ?
        """
        system_prompt = """Tu es un analyste financier expert qui fait un DEEP DIVE sur un symbole.

TON OBJECTIF:
- Comprendre POURQUOI ce symbole est discuté
- Identifier QUI en parle (influenceurs, institutions)
- Trouver le CATALYSEUR (event, news, earnings)
- Évaluer les RISQUES
- Identifier les symboles LIÉS à surveiller

FORMAT JSON:
{
    "symbol": "TICKER",
    "analysis": {
        "why_moving": "Explication détaillée",
        "main_catalyst": "Le catalyseur principal",
        "secondary_catalysts": ["cat1", "cat2"],
        "key_voices": [
            {"name": "@user", "stance": "bullish/bearish", "quote": "ce qu'il dit"}
        ],
        "sentiment": "bullish/bearish/neutral/mixed",
        "sentiment_score": -1.0 à 1.0,
        "confidence": 0.0 à 1.0,
        "risk_factors": ["risk1", "risk2"],
        "related_symbols": ["SYM1", "SYM2"],
        "actionable_insight": "Ce qu'un trader devrait savoir",
        "timeframe": "immediate/days/weeks"
    }
}"""

        prompt = f"""DEEP DIVE: ${symbol}

{f'CONTEXTE: {context}' if context else ''}

ANALYSE EN PROFONDEUR:
1. Pourquoi ${symbol} est discuté maintenant sur X ?
2. Qui sont les voix influentes qui en parlent ?
3. Quel est le catalyseur principal ?
4. Quels symboles liés (même secteur) devraient bouger aussi ?
5. Quels sont les risques ?
6. Quelle est ton conviction (sentiment + confidence) ?

SOIS PRÉCIS et ACTIONNABLE. Un trader doit pouvoir utiliser ton analyse."""

        response = await self.chat(prompt, system_prompt=system_prompt, temperature=0.3)
        
        if not response:
            return None
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = _safe_json_loads(json_match.group())
                analysis = data.get('analysis', data)
                
                return GrokInsight(
                    topic=f"${symbol} Deep Dive",
                    summary=analysis.get('why_moving', analysis.get('actionable_insight', '')),
                    sentiment=analysis.get('sentiment', 'neutral'),
                    sentiment_score=float(analysis.get('sentiment_score', 0)),
                    confidence=float(analysis.get('confidence', 0.5)),
                    key_points=[analysis.get('actionable_insight', '')] if analysis.get('actionable_insight') else [],
                    mentioned_symbols=[symbol],
                    catalysts=[analysis.get('main_catalyst', '')] + analysis.get('secondary_catalysts', []),
                    risk_factors=analysis.get('risk_factors', []),
                    source_posts=[],
                    discovery_reason=analysis.get('why_moving', ''),
                    related_symbols=analysis.get('related_symbols', [])
                )
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Deep dive parse issue for {symbol}: {e}")
            # Return minimal insight from raw response
            if response:
                sentiment = 'neutral'
                if 'bullish' in response.lower():
                    sentiment = 'bullish'
                elif 'bearish' in response.lower():
                    sentiment = 'bearish'
                return GrokInsight(
                    topic=f"${symbol} Deep Dive",
                    summary=response[:500] if response else "",
                    sentiment=sentiment,
                    sentiment_score=0.3 if sentiment == 'bullish' else (-0.3 if sentiment == 'bearish' else 0),
                    confidence=0.3,
                    key_points=[],
                    mentioned_symbols=[symbol],
                    catalysts=[],
                    risk_factors=[],
                    source_posts=[],
                    discovery_reason="Analysis completed with partial data",
                    related_symbols=[]
                )
        return None

    # =========================================================================
    # V2: CHAIN OF THOUGHT - Cascade searches
    # =========================================================================

    async def _chain_of_thought_explore(
        self, 
        initial_symbols: List[str],
        max_depth: int = 2
    ) -> List[GrokInsight]:
        """
        Phase 3: Recherches en cascade.
        Si on trouve NVDA intéressant → chercher AMD, AVGO, TSM.
        """
        explored = set()
        insights = []
        to_explore = list(initial_symbols)
        depth = 0
        
        while to_explore and depth < max_depth:
            current_batch = to_explore[:3]  # Max 3 par niveau pour éviter explosion
            to_explore = to_explore[3:]
            
            for symbol in current_batch:
                if symbol in explored:
                    continue
                    
                explored.add(symbol)
                
                # Deep dive sur ce symbole
                insight = await self._deep_dive(symbol)
                if insight:
                    insights.append(insight)
                    
                    # Trouver les symboles liés pour la prochaine itération
                    related = insight.related_symbols or get_related_symbols(symbol, list(explored))
                    for rel in related:
                        if rel not in explored and rel not in to_explore:
                            to_explore.append(rel)
                
                await asyncio.sleep(1)  # Rate limit friendly
            
            depth += 1
            
        logger.info(f"Chain of thought: explored {len(explored)} symbols, {len(insights)} insights")
        return insights

    # =========================================================================
    # PUBLIC METHODS - Compatibilité avec le code existant
    # =========================================================================

    async def search_x(
        self,
        query: str,
        max_results: int = 50,
        recent_only: bool = True
    ) -> List[XPost]:
        """Compatibilité - recherche X via Grok"""
        # On utilise le deep dive à la place
        await self._deep_dive(query)
        return []

    async def analyze_symbol(self, symbol: str) -> Optional[GrokInsight]:
        """Analyse approfondie d'un symbole via Grok"""
        if not self.is_available():
            return None
        return await self._deep_dive(symbol)

    async def search_financial_trends(self) -> List[GrokInsight]:
        """
        V2: Découverte autonome des tendances.
        Remplace les queries statiques par une découverte intelligente.
        """
        if not self.is_available():
            return []

        # Phase 1: Découverte autonome
        discoveries, worth_deep_dive = await self._autonomous_discover()
        
        if not discoveries:
            return []
        
        insights = []
        
        # Convertir les découvertes en insights
        for disc in discoveries:
            insight = GrokInsight(
                topic=disc.get('topic', 'Unknown'),
                summary=disc.get('why_interesting', ''),
                sentiment=disc.get('sentiment', 'neutral'),
                sentiment_score=float(disc.get('sentiment_score', 0)),
                confidence=float(disc.get('confidence', 0.5)),
                key_points=[disc.get('catalyst', '')] if disc.get('catalyst') else [],
                mentioned_symbols=disc.get('symbols', []),
                catalysts=[disc.get('catalyst')] if disc.get('catalyst') else [],
                risk_factors=[],
                source_posts=[],
                discovery_reason=disc.get('why_interesting', '')
            )
            insights.append(insight)
        
        # Phase 2: Deep dive sur les plus intéressants
        if worth_deep_dive:
            for symbol in worth_deep_dive[:2]:  # Max 2 deep dives
                deep_insight = await self._deep_dive(symbol)
                if deep_insight:
                    # Remplacer l'insight basique par le deep dive
                    insights = [i for i in insights if symbol not in i.mentioned_symbols]
                    insights.append(deep_insight)
                await asyncio.sleep(1)
        
        return insights

    async def full_scan_with_analysis(self) -> Optional[GrokScanResult]:
        """
        V2: Scan complet intelligent.
        1. Découverte autonome
        2. Deep dive sur les plus intéressants
        3. Chain of thought pour explorer les connexions
        """
        if not self.is_available():
            logger.warning("Grok API not available")
            return None

        logger.info("Starting full Grok V2 intelligent scan...")
        
        # Phase 1: Découverte autonome
        discoveries, worth_deep_dive = await self._autonomous_discover()
        
        insights = []
        discovery_chain = []
        
        # Convertir découvertes en insights
        for disc in discoveries:
            insight = GrokInsight(
                topic=disc.get('topic', 'Unknown'),
                summary=disc.get('why_interesting', ''),
                sentiment=disc.get('sentiment', 'neutral'),
                sentiment_score=float(disc.get('sentiment_score', 0)),
                confidence=float(disc.get('confidence', 0.5)),
                key_points=[disc.get('catalyst', '')] if disc.get('catalyst') else [],
                mentioned_symbols=disc.get('symbols', []),
                catalysts=[disc.get('catalyst')] if disc.get('catalyst') else [],
                risk_factors=[],
                source_posts=[],
                discovery_reason=disc.get('why_interesting', '')
            )
            insights.append(insight)
            discovery_chain.append(f"DISCOVER: {disc.get('topic')}")
        
        # Phase 2 & 3: Deep dive + Chain of thought
        if worth_deep_dive:
            chain_insights = await self._chain_of_thought_explore(worth_deep_dive, max_depth=2)
            
            for ci in chain_insights:
                discovery_chain.append(f"DEEP_DIVE: {ci.topic}")
                # Éviter les doublons
                if not any(ci.topic == i.topic for i in insights):
                    insights.append(ci)
        
        # Calculer les trending symbols
        trending_symbols = Counter()
        for insight in insights:
            for symbol in insight.mentioned_symbols:
                trending_symbols[symbol] += 1
            for symbol in insight.related_symbols:
                trending_symbols[symbol] += 1
        
        # Sentiment global
        if insights:
            overall_sentiment = sum(i.sentiment_score for i in insights) / len(insights)
        else:
            overall_sentiment = 0.0
        
        # Narratifs
        emerging_narratives = list(set([i.topic for i in insights if i.confidence > 0.5]))
        
        # Influenceurs (basé sur les deep dives)
        top_influencers = await self._get_active_influencers()
        
        result = GrokScanResult(
            timestamp=datetime.now(),
            posts_analyzed=len(insights) * 15,  # Estimation
            trending_symbols=dict(trending_symbols.most_common(20)),
            trending_hashtags={},
            overall_sentiment=overall_sentiment,
            insights=insights,
            top_influencers=top_influencers,
            emerging_narratives=emerging_narratives[:10],
            discovery_chain=discovery_chain,
            memory_hits=len(self._memory.get_successful_patterns(5))
        )

        self._scan_history.append(result)
        self._save_scan_result(result)

        logger.info(
            f"Grok V2 scan complete: {len(insights)} insights, "
            f"sentiment={overall_sentiment:.2f}, "
            f"{len(trending_symbols)} trending symbols, "
            f"chain depth={len(discovery_chain)}"
        )

        return result

    async def _get_active_influencers(self) -> List[Dict]:
        """Identifie les influenceurs actifs"""
        prompt = f"""Quels influenceurs financiers sur X sont les plus actifs MAINTENANT?

Comptes connus: {', '.join(self.FINANCIAL_INFLUENCERS[:10])}

Retourne JSON:
[
    {{"username": "name", "recent_topic": "sujet discuté", "sentiment": "bullish/bearish"}}
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
        return self._scan_history[-1] if self._scan_history else None

    async def get_symbol_details_for_narrative(self, symbol: str) -> Dict:
        """Récupère les détails enrichis d'un symbole pour le NarrativeGenerator"""
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
            insight = await self._deep_dive(symbol)

            if insight:
                result['sentiment'] = (insight.sentiment_score + 1) / 2
                result['sentiment_label'] = insight.sentiment
                result['themes'] = insight.key_points[:5]
                result['catalysts'] = insight.catalysts[:3]
                result['confidence'] = insight.confidence

                if insight.summary:
                    result['key_tweets'] = [insight.summary]

                result['catalyst_linked'] = len(insight.catalysts) > 0

        except Exception as e:
            logger.warning(f"Failed to get Grok details for {symbol}: {e}")

        return result

    async def get_symbol_buzz(self, symbols: List[str]) -> Dict[str, Dict]:
        """Obtient le buzz pour une liste de symboles"""
        results = {}

        for i in range(0, len(symbols), 5):
            batch = symbols[i:i+5]

            prompt = f"""Analyse le buzz X/Twitter pour: {', '.join(f'${s}' for s in batch)}

Pour chaque stock:
{{
    "SYMBOL": {{
        "mention_level": "high" | "medium" | "low" | "none",
        "sentiment": "bullish" | "bearish" | "neutral",
        "sentiment_score": -1.0 à 1.0,
        "key_topic": "sujet principal"
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

            if i + 5 < len(symbols):
                await asyncio.sleep(2)

        return results
    
    # =========================================================================
    # V2: FEEDBACK LOOP - Apprendre des résultats
    # =========================================================================
    
    def record_symbol_performance(self, symbol: str, performed_well: bool):
        """
        Enregistre si un symbole découvert a bien performé.
        Appelé par le système de trading après évaluation.
        """
        self._memory.update_performance(symbol, performed_well)
        logger.info(f"Recorded performance for {symbol}: {'good' if performed_well else 'bad'}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_grok_scanner_instance: Optional[GrokScanner] = None


async def get_grok_scanner(api_key: Optional[str] = None) -> GrokScanner:
    """Factory pour obtenir une instance du scanner Grok"""
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
        print("=== Grok Scanner V2 Test ===\n")

        api_key = os.getenv('GROK_API_KEY')
        if not api_key:
            print("GROK_API_KEY not set in environment.")
            print("Get your API key at: https://console.x.ai")
            return

        scanner = GrokScanner()
        await scanner.initialize()

        try:
            if scanner.is_available():
                print("Running full V2 intelligent scan...")
                result = await scanner.full_scan_with_analysis()

                if result:
                    print(f"\n✅ Scan complete!")
                    print(f"   Overall sentiment: {result.overall_sentiment:.2f}")
                    print(f"   Discovery chain: {len(result.discovery_chain)} steps")
                    
                    print("\n📈 Trending Symbols:")
                    for symbol, count in list(result.trending_symbols.items())[:10]:
                        print(f"   ${symbol}: {count}")

                    print("\n🔍 Insights:")
                    for insight in result.insights[:5]:
                        print(f"   [{insight.sentiment}] {insight.topic}")
                        print(f"      → {insight.summary[:100]}...")
                        if insight.related_symbols:
                            print(f"      Related: {', '.join(insight.related_symbols)}")
                    
                    print(f"\n🔗 Discovery Chain:")
                    for step in result.discovery_chain[:10]:
                        print(f"   {step}")

        finally:
            await scanner.close()

    asyncio.run(main())
