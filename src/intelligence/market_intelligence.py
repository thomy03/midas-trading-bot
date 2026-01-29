"""
Market Intelligence Module
Intégration LLM via OpenRouter pour l'analyse de marché et la veille sectorielle.

Fonctionnalités:
- Recherche de news sectorielles
- Analyse de sentiment (réseaux sociaux, forums)
- Surveillance Arxiv/publications tech
- Identification d'opportunités émergentes
"""

import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import aiohttp


class IntelligenceType(Enum):
    """Types d'intelligence de marché"""
    SECTOR_NEWS = "sector_news"           # News sectorielles
    SOCIAL_SENTIMENT = "social_sentiment"  # Sentiment réseaux sociaux
    RESEARCH_PAPERS = "research_papers"    # Publications Arxiv/académiques
    ANALYST_REPORTS = "analyst_reports"    # Rapports d'analystes
    EARNINGS_PREVIEW = "earnings_preview"  # Prévisualisation résultats
    MACRO_EVENTS = "macro_events"          # Événements macro-économiques


@dataclass
class IntelligenceResult:
    """Résultat d'une requête d'intelligence"""
    type: IntelligenceType
    query: str
    timestamp: datetime
    summary: str
    sentiment_score: float  # -1.0 (très négatif) à +1.0 (très positif)
    confidence: float       # 0.0 à 1.0
    sources: List[str] = field(default_factory=list)
    key_points: List[str] = field(default_factory=list)
    symbols_mentioned: List[str] = field(default_factory=list)
    sectors_impacted: List[str] = field(default_factory=list)
    raw_data: Optional[Dict] = None


@dataclass
class SectorIntelligence:
    """Intelligence agrégée pour un secteur"""
    sector: str
    overall_sentiment: float
    news_count: int
    trending_topics: List[str]
    key_developments: List[str]
    emerging_opportunities: List[str]
    risk_factors: List[str]
    recommended_symbols: List[str]
    last_updated: datetime


class OpenRouterClient:
    """
    Client pour l'API OpenRouter.
    Permet d'utiliser différents modèles LLM (Claude, GPT-4, Llama, etc.)
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    # Modèles recommandés par cas d'usage
    MODELS = {
        'fast': 'anthropic/claude-3-haiku',      # Rapide, économique
        'balanced': 'anthropic/claude-3-sonnet', # Équilibré
        'powerful': 'anthropic/claude-3-opus',   # Plus puissant
        'gpt4': 'openai/gpt-4-turbo',            # Alternative OpenAI
        'llama': 'meta-llama/llama-3-70b',       # Open source
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le client OpenRouter.

        Args:
            api_key: Clé API OpenRouter (ou variable d'environnement OPENROUTER_API_KEY)
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )

        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        """Crée la session HTTP si nécessaire"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Ferme la session HTTP"""
        if self.session and not self.session.closed:
            await self.session.close()

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = 'balanced',
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> str:
        """
        Envoie une requête de chat au modèle.

        Args:
            messages: Liste de messages [{"role": "user/assistant/system", "content": "..."}]
            model: Nom du modèle ou clé dans MODELS
            temperature: Créativité (0.0 = déterministe, 1.0 = créatif)
            max_tokens: Nombre max de tokens en réponse

        Returns:
            Réponse du modèle
        """
        await self._ensure_session()

        # Résoudre le nom du modèle
        model_id = self.MODELS.get(model, model)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://tradingbot-v3.local",  # Requis par OpenRouter
            "X-Title": "TradingBot V3 Market Intelligence"
        }

        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        async with self.session.post(
            f"{self.BASE_URL}/chat/completions",
            headers=headers,
            json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"OpenRouter API error {response.status}: {error_text}")

            data = await response.json()
            return data['choices'][0]['message']['content']

    async def analyze_text(
        self,
        text: str,
        analysis_type: str = "sentiment",
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyse un texte avec le LLM.

        Args:
            text: Texte à analyser
            analysis_type: Type d'analyse (sentiment, summary, entities, opportunities)
            context: Contexte additionnel

        Returns:
            Résultat de l'analyse (JSON parsé)
        """
        prompts = {
            "sentiment": f"""Analyse le sentiment de ce texte concernant les marchés financiers.
Réponds en JSON avec:
- sentiment_score: float entre -1.0 (très négatif) et +1.0 (très positif)
- confidence: float entre 0.0 et 1.0
- key_points: liste des points clés
- symbols_mentioned: liste des symboles/tickers mentionnés
- sectors_impacted: liste des secteurs impactés

Texte: {text}

{f'Contexte: {context}' if context else ''}

Réponds uniquement en JSON valide.""",

            "summary": f"""Résume ce texte en identifiant les informations pertinentes pour un investisseur.
Réponds en JSON avec:
- summary: résumé concis (2-3 phrases)
- key_developments: liste des développements importants
- market_impact: impact potentiel sur les marchés (high/medium/low)
- actionable_insights: conseils actionnables

Texte: {text}

Réponds uniquement en JSON valide.""",

            "opportunities": f"""Identifie les opportunités d'investissement dans ce texte.
Réponds en JSON avec:
- opportunities: liste d'objets avec (symbol, sector, thesis, confidence, timeframe)
- risks: risques identifiés
- sectors_to_watch: secteurs à surveiller

Texte: {text}

{f'Contexte: {context}' if context else ''}

Réponds uniquement en JSON valide."""
        }

        prompt = prompts.get(analysis_type, prompts["sentiment"])

        messages = [
            {"role": "system", "content": "Tu es un analyste financier expert. Réponds toujours en JSON valide."},
            {"role": "user", "content": prompt}
        ]

        response = await self.chat(messages, model='fast', temperature=0.1)

        # Parser le JSON de la réponse
        try:
            # Nettoyer la réponse (enlever markdown si présent)
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.startswith("```"):
                clean_response = clean_response[3:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]

            # V4.8: Fix common LLM JSON errors (trailing commas)
            import re
            clean_response = re.sub(r',\s*]', ']', clean_response)
            clean_response = re.sub(r',\s*}', '}', clean_response)

            return json.loads(clean_response.strip())
        except json.JSONDecodeError:
            return {"raw_response": response, "error": "Failed to parse JSON"}


class MarketIntelligence:
    """
    Système d'intelligence de marché principal.
    Agrège différentes sources d'information et utilise le LLM pour l'analyse.
    """

    # Mapping secteurs -> mots-clés de recherche
    SECTOR_KEYWORDS = {
        'Technology': ['tech', 'software', 'cloud', 'SaaS', 'cybersecurity', 'digital'],
        'AI_Semiconductors': ['AI', 'artificial intelligence', 'GPU', 'chip', 'semiconductor', 'NVIDIA', 'AMD'],
        'Healthcare': ['pharma', 'biotech', 'FDA', 'drug', 'clinical trial', 'healthcare'],
        'Finance': ['bank', 'fintech', 'interest rate', 'Fed', 'credit', 'lending'],
        'Energy': ['oil', 'gas', 'renewable', 'solar', 'EV', 'battery', 'energy transition'],
        'Consumer_Discretionary': ['retail', 'luxury', 'consumer', 'e-commerce', 'spending'],
        'Industrials': ['manufacturing', 'aerospace', 'defense', 'infrastructure', 'automation'],
        'Communications': ['telecom', 'media', 'streaming', '5G', 'advertising', 'social media'],
    }

    def __init__(self, openrouter_api_key: Optional[str] = None):
        """
        Initialise le système d'intelligence.

        Args:
            openrouter_api_key: Clé API OpenRouter
        """
        self.llm_client: Optional[OpenRouterClient] = None
        self._api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')

        # Cache des résultats
        self._cache: Dict[str, IntelligenceResult] = {}
        self._sector_cache: Dict[str, SectorIntelligence] = {}
        self._cache_ttl = timedelta(hours=1)

    async def initialize(self):
        """Initialise les connexions"""
        if self._api_key:
            self.llm_client = OpenRouterClient(self._api_key)

    async def close(self):
        """Ferme les connexions"""
        if self.llm_client:
            await self.llm_client.close()

    def _is_cache_valid(self, key: str) -> bool:
        """Vérifie si une entrée du cache est encore valide"""
        if key not in self._cache:
            return False
        result = self._cache[key]
        return datetime.now() - result.timestamp < self._cache_ttl

    async def analyze_sector_news(
        self,
        sector: str,
        news_text: str
    ) -> IntelligenceResult:
        """
        Analyse les news d'un secteur.

        Args:
            sector: Nom du secteur
            news_text: Texte des news à analyser

        Returns:
            IntelligenceResult avec l'analyse
        """
        if not self.llm_client:
            raise ValueError("LLM client not initialized. Call initialize() first.")

        keywords = self.SECTOR_KEYWORDS.get(sector, [])
        context = f"Secteur: {sector}. Mots-clés pertinents: {', '.join(keywords)}"

        analysis = await self.llm_client.analyze_text(
            news_text,
            analysis_type="sentiment",
            context=context
        )

        return IntelligenceResult(
            type=IntelligenceType.SECTOR_NEWS,
            query=f"sector:{sector}",
            timestamp=datetime.now(),
            summary=analysis.get('summary', ''),
            sentiment_score=float(analysis.get('sentiment_score', 0.0)),
            confidence=float(analysis.get('confidence', 0.5)),
            key_points=analysis.get('key_points', []),
            symbols_mentioned=analysis.get('symbols_mentioned', []),
            sectors_impacted=analysis.get('sectors_impacted', [sector]),
            raw_data=analysis
        )

    async def identify_opportunities(
        self,
        text: str,
        focus_sectors: Optional[List[str]] = None
    ) -> IntelligenceResult:
        """
        Identifie les opportunités d'investissement dans un texte.

        Args:
            text: Texte à analyser (news, article, rapport)
            focus_sectors: Secteurs sur lesquels se concentrer

        Returns:
            IntelligenceResult avec les opportunités identifiées
        """
        if not self.llm_client:
            raise ValueError("LLM client not initialized. Call initialize() first.")

        context = None
        if focus_sectors:
            context = f"Focus sur les secteurs: {', '.join(focus_sectors)}"

        analysis = await self.llm_client.analyze_text(
            text,
            analysis_type="opportunities",
            context=context
        )

        opportunities = analysis.get('opportunities', [])
        symbols = [opp.get('symbol') for opp in opportunities if opp.get('symbol')]
        sectors = [opp.get('sector') for opp in opportunities if opp.get('sector')]

        return IntelligenceResult(
            type=IntelligenceType.SECTOR_NEWS,
            query="opportunities",
            timestamp=datetime.now(),
            summary=f"Identifié {len(opportunities)} opportunités potentielles",
            sentiment_score=0.5,  # Neutre par défaut
            confidence=0.7,
            key_points=[f"{opp.get('symbol', '?')}: {opp.get('thesis', '')}" for opp in opportunities[:5]],
            symbols_mentioned=list(set(symbols)),
            sectors_impacted=list(set(sectors)),
            raw_data=analysis
        )

    async def get_sector_intelligence(
        self,
        sector: str,
        force_refresh: bool = False
    ) -> SectorIntelligence:
        """
        Obtient l'intelligence complète pour un secteur.

        Args:
            sector: Nom du secteur
            force_refresh: Forcer le rafraîchissement du cache

        Returns:
            SectorIntelligence agrégée
        """
        cache_key = f"sector:{sector}"

        # Vérifier le cache
        if not force_refresh and cache_key in self._sector_cache:
            cached = self._sector_cache[cache_key]
            if datetime.now() - cached.last_updated < self._cache_ttl:
                return cached

        # TODO: Implémenter la récupération de news depuis des APIs externes
        # Pour l'instant, retourne un placeholder

        intelligence = SectorIntelligence(
            sector=sector,
            overall_sentiment=0.0,
            news_count=0,
            trending_topics=[],
            key_developments=[],
            emerging_opportunities=[],
            risk_factors=[],
            recommended_symbols=[],
            last_updated=datetime.now()
        )

        self._sector_cache[cache_key] = intelligence
        return intelligence

    async def generate_market_brief(
        self,
        sectors: Optional[List[str]] = None
    ) -> str:
        """
        Génère un brief de marché.

        Args:
            sectors: Secteurs à inclure (tous si None)

        Returns:
            Brief formaté en texte
        """
        if not self.llm_client:
            return "LLM client not initialized. Set OPENROUTER_API_KEY environment variable."

        target_sectors = sectors or list(self.SECTOR_KEYWORDS.keys())

        # Collecter l'intelligence de chaque secteur
        sector_data = []
        for sector in target_sectors:
            intel = await self.get_sector_intelligence(sector)
            sector_data.append({
                'sector': sector,
                'sentiment': intel.overall_sentiment,
                'opportunities': intel.emerging_opportunities,
                'risks': intel.risk_factors
            })

        # Générer le brief avec le LLM
        prompt = f"""Génère un brief de marché concis basé sur ces données sectorielles:

{json.dumps(sector_data, indent=2)}

Le brief doit inclure:
1. Vue d'ensemble du marché
2. Secteurs les plus prometteurs
3. Risques à surveiller
4. Actions recommandées

Réponds en français, format markdown."""

        messages = [
            {"role": "system", "content": "Tu es un stratégiste de marché senior."},
            {"role": "user", "content": prompt}
        ]

        return await self.llm_client.chat(messages, model='balanced')


# Singleton pour accès global
market_intelligence: Optional[MarketIntelligence] = None


async def get_market_intelligence() -> MarketIntelligence:
    """
    Retourne l'instance singleton de MarketIntelligence.
    Initialise si nécessaire.
    """
    global market_intelligence

    if market_intelligence is None:
        market_intelligence = MarketIntelligence()
        await market_intelligence.initialize()

    return market_intelligence
