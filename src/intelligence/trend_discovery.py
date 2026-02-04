"""
Trend Discovery Module
Découverte automatique des tendances de marché émergentes.

Approche hybride:
1. Détection quantitative - Anomalies de momentum sectoriel
2. Analyse LLM - Identification des narratifs dans les news
3. Intégration NarrativeTracker - Mise à jour dynamique des tendances

Usage:
    from src.intelligence.trend_discovery import TrendDiscovery

    discovery = TrendDiscovery(openrouter_api_key="...", model="anthropic/claude-3-haiku")
    await discovery.initialize()

    # Découverte quotidienne
    report = await discovery.daily_scan()

    # Scanner un secteur spécifique
    sector_trends = await discovery.scan_sector("Technology")
"""

import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Imports internes
try:
    from .market_intelligence import MarketIntelligence, OpenRouterClient, IntelligenceResult
    from .news_fetcher import NewsFetcher, NewsArticle
    from .dynamic_sectors import DynamicSectors, get_dynamic_sectors
except ImportError:
    from src.intelligence.market_intelligence import MarketIntelligence, OpenRouterClient, IntelligenceResult
    from src.intelligence.news_fetcher import NewsFetcher, NewsArticle
    from src.intelligence.dynamic_sectors import DynamicSectors, get_dynamic_sectors


class TrendStrength(Enum):
    """Force de la tendance détectée"""
    EMERGING = "emerging"       # Tendance naissante (signaux faibles)
    DEVELOPING = "developing"   # Tendance en développement
    ESTABLISHED = "established" # Tendance établie
    DECLINING = "declining"     # Tendance en déclin


class TrendType(Enum):
    """Types de tendances"""
    SECTOR_MOMENTUM = "sector_momentum"     # Momentum sectoriel
    THEMATIC = "thematic"                   # Thématique (AI, crypto, etc.)
    EVENT_DRIVEN = "event_driven"           # Événementiel (FDA, earnings)
    MACRO = "macro"                         # Macro-économique
    TECHNICAL = "technical"                 # Pattern technique


@dataclass
class EmergingTrend:
    """Tendance émergente détectée"""
    name: str
    type: TrendType
    strength: TrendStrength
    confidence: float  # 0.0 à 1.0
    description: str

    # Signaux quantitatifs
    momentum_score: float = 0.0  # -1 à +1
    volume_surge: float = 1.0    # Ratio vs moyenne
    breadth_score: float = 0.0   # % d'actions en hausse

    # Signaux qualitatifs (LLM)
    news_sentiment: float = 0.0  # -1 à +1
    news_count: int = 0
    key_catalysts: List[str] = field(default_factory=list)

    # Actions concernées
    sectors: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)

    # Métadonnées
    detected_at: datetime = field(default_factory=datetime.now)
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convertit en dictionnaire"""
        return {
            'name': self.name,
            'type': self.type.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'description': self.description,
            'momentum_score': self.momentum_score,
            'volume_surge': self.volume_surge,
            'breadth_score': self.breadth_score,
            'news_sentiment': self.news_sentiment,
            'news_count': self.news_count,
            'key_catalysts': self.key_catalysts,
            'sectors': self.sectors,
            'symbols': self.symbols,
            'detected_at': self.detected_at.isoformat(),
            'sources': self.sources
        }


@dataclass
class TrendReport:
    """Rapport quotidien de découverte de tendances"""
    date: datetime
    trends: List[EmergingTrend]
    sector_momentum: Dict[str, float]
    market_sentiment: float
    top_movers: List[str]
    watchlist_additions: List[str]
    narrative_updates: List[Dict]

    def to_dict(self) -> Dict:
        """Convertit en dictionnaire"""
        return {
            'date': self.date.isoformat(),
            'trends': [t.to_dict() for t in self.trends],
            'sector_momentum': self.sector_momentum,
            'market_sentiment': self.market_sentiment,
            'top_movers': self.top_movers,
            'watchlist_additions': self.watchlist_additions,
            'narrative_updates': self.narrative_updates
        }

    def summary(self) -> str:
        """Génère un résumé textuel"""
        lines = [
            f"=== Trend Discovery Report - {self.date.strftime('%Y-%m-%d')} ===",
            f"",
            f"Market Sentiment: {self.market_sentiment:+.2f}",
            f"Trends Detected: {len(self.trends)}",
            f""
        ]

        if self.trends:
            lines.append("Top Emerging Trends:")
            for i, trend in enumerate(self.trends[:5], 1):
                lines.append(
                    f"  {i}. {trend.name} ({trend.strength.value}) - "
                    f"Confidence: {trend.confidence:.0%}"
                )
                if trend.key_catalysts:
                    lines.append(f"     Catalysts: {', '.join(trend.key_catalysts[:3])}")

        if self.watchlist_additions:
            lines.append(f"\nWatchlist Additions: {', '.join(self.watchlist_additions[:10])}")

        return '\n'.join(lines)


class TrendDiscovery:
    """
    Système de découverte de tendances hybride.
    Combine analyse quantitative et intelligence LLM.

    V4.2: Utilise DynamicSectors au lieu de secteurs hardcodés.
    Les secteurs sont mis à jour quotidiennement via l'API NASDAQ et le LLM.
    """

    # Secteurs FALLBACK (utilisés uniquement si DynamicSectors échoue)
    SECTORS_FALLBACK = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'CRM', 'ADBE'],
        'AI_Semiconductors': ['NVDA', 'AMD', 'AVGO', 'QCOM', 'INTC', 'ARM', 'MRVL', 'MU'],
        'Healthcare': ['UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'DHR'],
        'Finance': ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK', 'SCHW'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO'],
    }

    # Propriété dynamique pour les secteurs (sera populée à l'initialisation)
    @property
    def SECTORS(self) -> Dict[str, List[str]]:
        """Retourne les secteurs dynamiques ou le fallback"""
        if hasattr(self, '_dynamic_sectors') and self._dynamic_sectors:
            sectors = {}
            for name, sector_info in self._dynamic_sectors.get_all_sectors().items():
                sectors[name] = sector_info.symbols
            if sectors:
                return sectors
        return self.SECTORS_FALLBACK

    # Thèmes à surveiller (pour analyse LLM)
    THEMES_KEYWORDS = {
        'AI_Revolution': ['artificial intelligence', 'ChatGPT', 'LLM', 'machine learning', 'generative AI', 'AI chips'],
        'Quantum_Computing': ['quantum', 'qubit', 'quantum supremacy', 'quantum computing'],
        'Space_Economy': ['SpaceX', 'satellite', 'space launch', 'orbital', 'rocket'],
        'Obesity_Drugs': ['GLP-1', 'Ozempic', 'Wegovy', 'Mounjaro', 'weight loss drug', 'obesity treatment'],
        'Nuclear_Renaissance': ['nuclear', 'SMR', 'small modular reactor', 'uranium', 'nuclear power'],
        'EV_Transition': ['electric vehicle', 'EV', 'battery', 'charging station', 'Tesla'],
        'Cybersecurity': ['cybersecurity', 'data breach', 'ransomware', 'zero trust', 'security threat'],
        'Reshoring': ['reshoring', 'nearshoring', 'supply chain', 'manufacturing USA', 'CHIPS Act'],
        'Crypto_Bull': ['Bitcoin', 'Ethereum', 'crypto', 'blockchain', 'DeFi', 'ETF approval'],
        'Rate_Pivot': ['Fed pivot', 'rate cut', 'interest rate', 'monetary policy', 'FOMC'],
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = None,
        data_dir: str = 'data/trends',
        # Legacy parameter for backward compatibility
        openrouter_api_key: Optional[str] = None
    ):
        """
        Initialise le système de découverte.

        Args:
            api_key: Clé API Google AI (ou openrouter_api_key pour compatibilité)
            model: Modèle LLM (default: gemini-3-flash-preview)
            data_dir: Répertoire pour stocker les rapports
        """
        # V4.8: Use Gemini direct, fallback to OpenRouter for compatibility
        self.api_key = api_key or openrouter_api_key or os.getenv('GOOGLE_AI_API_KEY')
        self.model = model or os.getenv('GEMINI_MODEL', 'gemini-3-flash-preview')
        self.data_dir = data_dir

        # Composants - V4.8: Use GeminiClient
        self.llm_client = None  # Will be GeminiClient
        self.news_fetcher: Optional[NewsFetcher] = None
        self.market_intelligence: Optional[MarketIntelligence] = None

        # V4.2: Secteurs dynamiques
        self._dynamic_sectors: Optional[DynamicSectors] = None

        # Cache
        self._sector_data_cache: Dict[str, Any] = {}
        self._last_scan: Optional[datetime] = None

        # Créer le répertoire de données
        os.makedirs(data_dir, exist_ok=True)

    async def initialize(self):
        """Initialise les connexions et composants"""
        # V4.8: Use Gemini direct
        if self.api_key:
            try:
                from .gemini_client import GeminiClient
                self.llm_client = GeminiClient(api_key=self.api_key, model=self.model)
                await self.llm_client.initialize()
            except Exception as e:
                logger.warning(f"GeminiClient init failed, trying OpenRouter: {e}")
                self.llm_client = OpenRouterClient(self.api_key)

        self.news_fetcher = NewsFetcher()
        self.market_intelligence = MarketIntelligence(self.api_key)

        if self.market_intelligence:
            await self.market_intelligence.initialize()

        # V4.2: Initialiser les secteurs dynamiques
        try:
            self._dynamic_sectors = DynamicSectors(
                openrouter_api_key=self.api_key,
                llm_model=self.model
            )
            await self._dynamic_sectors.initialize()
            stats = self._dynamic_sectors.get_stats()
            logger.info(f"DynamicSectors loaded: {stats['total_sectors']} sectors, {stats['total_symbols']} symbols")
        except Exception as e:
            logger.warning(f"DynamicSectors initialization failed, using fallback: {e}")
            self._dynamic_sectors = None

        model_name = getattr(self.llm_client, 'model_name', self.model)
        logger.info(f"TrendDiscovery initialized with model: {model_name}")

    async def close(self):
        """Ferme les connexions"""
        if self.llm_client:
            await self.llm_client.close()
        if self.news_fetcher:
            await self.news_fetcher.close()
        if self.market_intelligence:
            await self.market_intelligence.close()
        if self._dynamic_sectors:
            await self._dynamic_sectors.close()

    # =========================================================================
    # DETECTION QUANTITATIVE
    # =========================================================================

    async def detect_sector_momentum(self) -> Dict[str, float]:
        """
        Détecte les anomalies de momentum par secteur.
        Utilise yfinance pour récupérer les données de prix.

        Returns:
            Dict avec score de momentum par secteur (-1 à +1)
        """
        import yfinance as yf

        sector_momentum = {}

        for sector, symbols in self.SECTORS.items():
            try:
                # Récupérer les données des 20 derniers jours
                tickers = yf.Tickers(' '.join(symbols))

                momentum_scores = []
                for symbol in symbols:
                    try:
                        hist = tickers.tickers[symbol].history(period='1mo')
                        if len(hist) < 10:
                            continue

                        # Calculer le momentum (rendement 5j vs rendement 20j)
                        ret_5d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1) if len(hist) >= 5 else 0
                        ret_20d = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1)

                        # Score: momentum récent vs momentum long terme
                        # Positif = accélération, Négatif = décélération
                        if ret_20d != 0:
                            momentum = (ret_5d * 4 - ret_20d) / abs(ret_20d) if ret_20d != 0 else ret_5d
                        else:
                            momentum = ret_5d * 10  # Multiplier si pas de tendance long terme

                        momentum_scores.append(max(-1, min(1, momentum)))
                    except Exception as e:
                        logger.debug(f"Error fetching {symbol}: {e}")
                        continue

                if momentum_scores:
                    sector_momentum[sector] = sum(momentum_scores) / len(momentum_scores)
                else:
                    sector_momentum[sector] = 0.0

            except Exception as e:
                logger.error(f"Error analyzing sector {sector}: {e}")
                sector_momentum[sector] = 0.0

        return sector_momentum

    async def detect_volume_anomalies(self) -> List[Tuple[str, float]]:
        """
        Détecte les anomalies de volume (surges).

        Returns:
            Liste de (symbol, volume_ratio) triée par ratio décroissant
        """
        import yfinance as yf

        all_symbols = []
        for symbols in self.SECTORS.values():
            all_symbols.extend(symbols)
        all_symbols = list(set(all_symbols))

        volume_anomalies = []

        # Récupérer en batch
        try:
            tickers = yf.Tickers(' '.join(all_symbols))

            for symbol in all_symbols:
                try:
                    hist = tickers.tickers[symbol].history(period='1mo')
                    if len(hist) < 20:
                        continue

                    # Volume moyen sur 20j (excluant aujourd'hui)
                    avg_volume = hist['Volume'].iloc[:-1].mean()
                    current_volume = hist['Volume'].iloc[-1]

                    if avg_volume > 0:
                        ratio = current_volume / avg_volume
                        if ratio > 1.5:  # Volume 50% au-dessus de la moyenne
                            volume_anomalies.append((symbol, ratio))
                except (KeyError, IndexError, TypeError, ZeroDivisionError):
                    continue

        except Exception as e:
            logger.error(f"Error detecting volume anomalies: {e}")

        # Trier par ratio décroissant
        volume_anomalies.sort(key=lambda x: x[1], reverse=True)
        return volume_anomalies[:20]  # Top 20

    async def detect_breadth_divergences(self) -> Dict[str, float]:
        """
        Détecte les divergences de breadth (% d'actions en hausse vs indice).

        Returns:
            Dict avec breadth score par secteur
        """
        import yfinance as yf

        breadth_scores = {}

        for sector, symbols in self.SECTORS.items():
            try:
                tickers = yf.Tickers(' '.join(symbols))

                advancing = 0
                declining = 0

                for symbol in symbols:
                    try:
                        hist = tickers.tickers[symbol].history(period='5d')
                        if len(hist) >= 2:
                            change = hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1
                            if change > 0.01:  # +1%
                                advancing += 1
                            elif change < -0.01:  # -1%
                                declining += 1
                    except (KeyError, IndexError, TypeError, ZeroDivisionError):
                        continue

                total = advancing + declining
                if total > 0:
                    breadth_scores[sector] = (advancing - declining) / total
                else:
                    breadth_scores[sector] = 0.0

            except Exception as e:
                logger.error(f"Error calculating breadth for {sector}: {e}")
                breadth_scores[sector] = 0.0

        return breadth_scores

    # =========================================================================
    # ANALYSE LLM
    # =========================================================================

    async def analyze_news_themes(self, days_back: int = 3) -> Dict[str, Dict]:
        """
        Analyse les news pour détecter les thèmes émergents.
        Utilise les thèmes prédéfinis ET les thèmes appris précédemment.

        Args:
            days_back: Nombre de jours d'historique

        Returns:
            Dict avec analyse par thème
        """
        if not self.llm_client or not self.news_fetcher:
            return {}

        theme_analysis = {}

        # Utiliser TOUS les thèmes (prédéfinis + appris)
        all_themes = self.get_all_themes()
        logger.info(f"Analyzing {len(all_themes)} themes ({len(self.THEMES_KEYWORDS)} predefined + {len(all_themes) - len(self.THEMES_KEYWORDS)} learned)")

        for theme, keywords in all_themes.items():
            try:
                # Récupérer les news pour ce thème
                query = ' OR '.join(keywords[:3])  # Limiter pour éviter trop de résultats
                articles = await self.news_fetcher.fetch_newsapi(
                    query,
                    from_date=datetime.now() - timedelta(days=days_back),
                    page_size=10
                )

                if not articles:
                    theme_analysis[theme] = {
                        'sentiment': 0.0,
                        'news_count': 0,
                        'catalysts': [],
                        'symbols': []
                    }
                    continue

                # Combiner les contenus pour analyse
                combined_text = '\n\n'.join([
                    f"Title: {a.title}\nContent: {a.content[:500]}"
                    for a in articles[:5]
                ])

                # Analyser avec LLM
                analysis = await self._analyze_theme_with_llm(theme, combined_text)
                theme_analysis[theme] = analysis

            except Exception as e:
                logger.error(f"Error analyzing theme {theme}: {e}")
                theme_analysis[theme] = {
                    'sentiment': 0.0,
                    'news_count': 0,
                    'catalysts': [],
                    'symbols': []
                }

        return theme_analysis

    async def _analyze_theme_with_llm(self, theme: str, text: str) -> Dict:
        """Analyse un thème avec le LLM (V4.8: uses GeminiClient)"""
        if not self.llm_client:
            return {'sentiment': 0.0, 'news_count': 0, 'catalysts': [], 'symbols': []}

        prompt = f"""Analyse ces articles concernant le thème "{theme}" pour un investisseur.

Articles:
{text}

Réponds en JSON avec:
{{
    "sentiment": float entre -1.0 (très négatif) et +1.0 (très positif),
    "momentum": "accelerating" | "stable" | "decelerating",
    "catalysts": [liste des 3 principaux catalyseurs identifiés],
    "symbols": [tickers d'actions mentionnées ou bénéficiaires],
    "key_insight": "insight principal en une phrase",
    "investment_thesis": "thèse d'investissement courte si opportunité détectée"
}}

Réponds uniquement en JSON valide."""

        try:
            # V4.8: Use GeminiClient's chat_json for automatic JSON parsing
            result = await self.llm_client.chat_json(
                prompt,
                system_prompt="Tu es un analyste financier expert. Réponds toujours en JSON valide.",
                temperature=0.1,
                max_tokens=1500  # V4.9.4: Increased to avoid truncation
            )

            return {
                'sentiment': float(result.get('sentiment', 0.0)),
                'momentum': result.get('momentum', 'stable'),
                'catalysts': result.get('catalysts', []),
                'symbols': result.get('symbols', []),
                'key_insight': result.get('key_insight', ''),
                'investment_thesis': result.get('investment_thesis', '')
            }

        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return {'sentiment': 0.0, 'news_count': 0, 'catalysts': [], 'symbols': []}

    async def identify_emerging_narratives(self) -> List[Dict]:
        """
        Utilise le LLM pour identifier de nouveaux narratifs émergents
        non couverts par les thèmes prédéfinis.

        Returns:
            Liste de narratifs émergents avec détails
        """
        if not self.llm_client or not self.news_fetcher:
            return []

        # Récupérer les news générales du marché
        try:
            articles = await self.news_fetcher.fetch_newsapi(
                "stock market investing finance technology",
                from_date=datetime.now() - timedelta(days=2),
                page_size=20
            )

            # Ajouter les posts Reddit populaires
            social_posts = await self.news_fetcher.fetch_social_posts(
                subsocials=['wallstreetbets', 'stocks', 'investing'],
                limit=10
            )

            # Filtrer par relevance
            top_posts = [p for p in social_posts if p.relevance_score > 0.3][:5]

        except Exception as e:
            logger.error(f"Error fetching news for narrative detection: {e}")
            return []

        if not articles and not top_posts:
            return []

        # Combiner pour analyse
        combined_text = '\n\n'.join([
            f"[News] {a.title}: {a.content[:300]}"
            for a in articles[:10]
        ])

        if top_posts:
            combined_text += '\n\n' + '\n\n'.join([
                f"[Reddit] {p.title}: {p.content[:200]}"
                for p in top_posts
            ])

        # Demander au LLM d'identifier les narratifs émergents
        prompt = f"""Analyse ces articles et posts récents pour identifier des NOUVEAUX narratifs d'investissement émergents.

Un narratif émergent est une tendance ou thème qui commence à attirer l'attention mais n'est pas encore mainstream.

Articles et posts:
{combined_text}

Identifie jusqu'à 3 narratifs émergents (différents de: AI, crypto, EV, etc. qui sont déjà bien connus).

Réponds en JSON:
{{
    "narratives": [
        {{
            "name": "Nom court du narratif",
            "description": "Description en 2 phrases",
            "strength": "emerging" | "developing",
            "sectors": ["secteurs concernés"],
            "symbols": ["tickers potentiellement bénéficiaires"],
            "catalysts": ["catalyseurs identifiés"],
            "risk_level": "low" | "medium" | "high"
        }}
    ]
}}

Réponds uniquement en JSON valide. Si aucun narratif émergent clair, retourne {{"narratives": []}}."""

        try:
            # V4.8: Use GeminiClient's chat_json for automatic JSON parsing
            result = await self.llm_client.chat_json(
                prompt,
                system_prompt="Tu es un stratégiste de marché expert en identification de tendances émergentes.",
                temperature=0.3,
                max_tokens=1500  # V4.9.4: Increased to avoid truncation
            )

            return result.get('narratives', [])

        except Exception as e:
            logger.error(f"Error identifying narratives: {e}")
            return []

    # =========================================================================
    # SCAN QUOTIDIEN
    # =========================================================================

    async def daily_scan(self) -> TrendReport:
        """
        Effectue un scan quotidien complet.
        Combine analyse quantitative et LLM.

        Returns:
            TrendReport avec toutes les tendances détectées
        """
        logger.info("Starting daily trend discovery scan...")

        trends = []
        watchlist_additions = []
        narrative_updates = []

        # 1. Analyse quantitative
        logger.info("Step 1/4: Analyzing sector momentum...")
        sector_momentum = await self.detect_sector_momentum()

        logger.info("Step 2/4: Detecting volume anomalies...")
        volume_anomalies = await self.detect_volume_anomalies()

        logger.info("Step 3/4: Calculating breadth scores...")
        breadth_scores = await self.detect_breadth_divergences()

        # Identifier les secteurs avec momentum anormal
        for sector, momentum in sector_momentum.items():
            if abs(momentum) > 0.3:  # Momentum significatif
                breadth = breadth_scores.get(sector, 0)

                # Déterminer la force
                if abs(momentum) > 0.6 and breadth > 0.5:
                    strength = TrendStrength.ESTABLISHED
                    confidence = 0.8
                elif abs(momentum) > 0.4:
                    strength = TrendStrength.DEVELOPING
                    confidence = 0.6
                else:
                    strength = TrendStrength.EMERGING
                    confidence = 0.4

                trend = EmergingTrend(
                    name=f"{sector} Momentum",
                    type=TrendType.SECTOR_MOMENTUM,
                    strength=strength,
                    confidence=confidence,
                    description=f"{'Bullish' if momentum > 0 else 'Bearish'} momentum detected in {sector}",
                    momentum_score=momentum,
                    breadth_score=breadth,
                    sectors=[sector],
                    symbols=self.SECTORS.get(sector, [])[:5]
                )
                trends.append(trend)

                # Ajouter à la watchlist si bullish
                if momentum > 0.3:
                    watchlist_additions.extend(self.SECTORS.get(sector, [])[:3])

        # 2. Analyse LLM des thèmes
        if self.llm_client:
            logger.info("Step 4/4: Analyzing news themes with LLM...")
            theme_analysis = await self.analyze_news_themes()

            for theme, analysis in theme_analysis.items():
                sentiment = analysis.get('sentiment', 0)
                momentum_str = analysis.get('momentum', 'stable')

                if abs(sentiment) > 0.3 or momentum_str == 'accelerating':
                    # Déterminer la force
                    if momentum_str == 'accelerating' and abs(sentiment) > 0.5:
                        strength = TrendStrength.DEVELOPING
                        confidence = 0.7
                    elif abs(sentiment) > 0.3:
                        strength = TrendStrength.EMERGING
                        confidence = 0.5
                    else:
                        strength = TrendStrength.EMERGING
                        confidence = 0.4

                    trend = EmergingTrend(
                        name=theme.replace('_', ' '),
                        type=TrendType.THEMATIC,
                        strength=strength,
                        confidence=confidence,
                        description=analysis.get('key_insight', f"Theme {theme} showing activity"),
                        news_sentiment=sentiment,
                        key_catalysts=analysis.get('catalysts', []),
                        symbols=analysis.get('symbols', []),
                        sources=['NewsAPI', 'LLM Analysis']
                    )
                    trends.append(trend)

                    # Ajouter les symboles à la watchlist
                    if sentiment > 0.3:
                        watchlist_additions.extend(analysis.get('symbols', []))

            # Identifier de nouveaux narratifs (la partie clé!)
            new_narratives = await self.identify_emerging_narratives()
            for narrative in new_narratives:
                narrative_updates.append(narrative)

                # SAUVEGARDER le nouveau thème pour les scans futurs
                self._save_learned_theme(narrative)
                logger.info(f"New theme discovered and saved: {narrative.get('name')}")

                trend = EmergingTrend(
                    name=narrative.get('name', 'Unknown'),
                    type=TrendType.THEMATIC,
                    strength=TrendStrength.EMERGING,
                    confidence=0.5,
                    description=narrative.get('description', ''),
                    key_catalysts=narrative.get('catalysts', []),
                    sectors=narrative.get('sectors', []),
                    symbols=narrative.get('symbols', []),
                    sources=['LLM Discovery']
                )
                trends.append(trend)
                watchlist_additions.extend(narrative.get('symbols', []))

        # Calculer le sentiment global
        market_sentiment = sum(sector_momentum.values()) / len(sector_momentum) if sector_momentum else 0

        # Top movers (volume anomalies)
        top_movers = [symbol for symbol, _ in volume_anomalies[:10]]

        # Dédupliquer la watchlist
        watchlist_additions = list(set(watchlist_additions))

        # Trier les tendances par confidence
        trends.sort(key=lambda t: t.confidence, reverse=True)

        # Créer le rapport
        report = TrendReport(
            date=datetime.now(),
            trends=trends,
            sector_momentum=sector_momentum,
            market_sentiment=market_sentiment,
            top_movers=top_movers,
            watchlist_additions=watchlist_additions,
            narrative_updates=narrative_updates
        )

        # Sauvegarder le rapport
        self._save_report(report)

        self._last_scan = datetime.now()
        logger.info(f"Daily scan complete. Found {len(trends)} trends.")

        return report

    async def scan_sector(self, sector: str) -> Dict:
        """
        Scan approfondi d'un secteur spécifique.

        Args:
            sector: Nom du secteur

        Returns:
            Analyse détaillée du secteur
        """
        if sector not in self.SECTORS:
            return {'error': f'Unknown sector: {sector}'}

        symbols = self.SECTORS[sector]

        import yfinance as yf

        # Analyser chaque symbole
        stock_analysis = []

        tickers = yf.Tickers(' '.join(symbols))

        for symbol in symbols:
            try:
                hist = tickers.tickers[symbol].history(period='1mo')
                if len(hist) < 10:
                    continue

                # Métriques
                ret_5d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-5] - 1) * 100
                ret_20d = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
                avg_volume = hist['Volume'].iloc[:-1].mean()
                vol_ratio = hist['Volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1

                stock_analysis.append({
                    'symbol': symbol,
                    'return_5d': round(ret_5d, 2),
                    'return_20d': round(ret_20d, 2),
                    'volume_ratio': round(vol_ratio, 2),
                    'momentum': 'strong' if ret_5d > 5 and vol_ratio > 1.5 else 'moderate' if ret_5d > 0 else 'weak'
                })

            except Exception as e:
                logger.debug(f"Error analyzing {symbol}: {e}")

        # Trier par performance 5j
        stock_analysis.sort(key=lambda x: x['return_5d'], reverse=True)

        return {
            'sector': sector,
            'analyzed_at': datetime.now().isoformat(),
            'stock_count': len(stock_analysis),
            'top_performers': stock_analysis[:5],
            'worst_performers': stock_analysis[-3:] if len(stock_analysis) > 3 else [],
            'sector_momentum': sum(s['return_5d'] for s in stock_analysis) / len(stock_analysis) if stock_analysis else 0,
            'high_volume_count': len([s for s in stock_analysis if s['volume_ratio'] > 1.5])
        }

    # =========================================================================
    # INTEGRATION AVEC SCREENER
    # =========================================================================

    def get_focus_symbols(self, min_confidence: float = 0.5) -> List[str]:
        """
        Retourne les symboles à scanner en priorité basé sur les tendances.

        Args:
            min_confidence: Confidence minimum

        Returns:
            Liste de symboles
        """
        # Charger le dernier rapport
        report = self._load_latest_report()
        if not report:
            return []

        symbols = set()

        # Ajouter les symboles des tendances à haute confiance
        for trend in report.get('trends', []):
            if trend.get('confidence', 0) >= min_confidence:
                symbols.update(trend.get('symbols', []))

        # Ajouter les top movers
        symbols.update(report.get('top_movers', [])[:10])

        # Ajouter les additions watchlist
        symbols.update(report.get('watchlist_additions', []))

        return list(symbols)

    def get_sector_focus(self, min_momentum: float = 0.2) -> List[str]:
        """
        Retourne les secteurs à surveiller en priorité.

        Args:
            min_momentum: Momentum minimum absolu

        Returns:
            Liste de secteurs
        """
        report = self._load_latest_report()
        if not report:
            return list(self.SECTORS.keys())

        sector_momentum = report.get('sector_momentum', {})

        # Secteurs avec momentum significatif (positif ou négatif)
        focus_sectors = [
            sector for sector, momentum in sector_momentum.items()
            if abs(momentum) >= min_momentum
        ]

        return focus_sectors if focus_sectors else list(self.SECTORS.keys())[:5]

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _save_report(self, report: TrendReport):
        """Sauvegarde le rapport quotidien"""
        filename = f"trend_report_{report.date.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.data_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        # Aussi sauvegarder comme "latest"
        latest_path = os.path.join(self.data_dir, 'latest_report.json')
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Report saved to {filepath}")

    def _load_latest_report(self) -> Optional[Dict]:
        """Charge le dernier rapport"""
        latest_path = os.path.join(self.data_dir, 'latest_report.json')

        if os.path.exists(latest_path):
            try:
                with open(latest_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError, IOError):
                return None
        return None

    # =========================================================================
    # DYNAMIC THEME LEARNING
    # =========================================================================

    def _get_learned_themes_path(self) -> str:
        """Chemin du fichier des thèmes appris"""
        return os.path.join(self.data_dir, 'learned_themes.json')

    def _load_learned_themes(self) -> Dict[str, Dict]:
        """Charge les thèmes découverts par le LLM"""
        path = self._get_learned_themes_path()
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError, IOError):
                return {}
        return {}

    def _save_learned_theme(self, theme: Dict):
        """
        Sauvegarde un nouveau thème découvert par le LLM.
        Ces thèmes seront utilisés dans les scans futurs.
        """
        themes = self._load_learned_themes()

        theme_name = theme.get('name', 'Unknown').replace(' ', '_')

        # Ajouter ou mettre à jour le thème
        themes[theme_name] = {
            'name': theme.get('name'),
            'description': theme.get('description'),
            'keywords': self._extract_keywords_from_theme(theme),
            'sectors': theme.get('sectors', []),
            'symbols': theme.get('symbols', []),
            'discovered_at': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat(),
            'occurrence_count': themes.get(theme_name, {}).get('occurrence_count', 0) + 1,
            'strength': theme.get('strength', 'emerging')
        }

        # Sauvegarder
        path = self._get_learned_themes_path()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(themes, f, indent=2, ensure_ascii=False)

        logger.info(f"Learned theme saved: {theme_name}")

    def _extract_keywords_from_theme(self, theme: Dict) -> List[str]:
        """Extrait des mots-clés à partir d'un thème découvert"""
        keywords = []

        # Du nom
        name = theme.get('name', '')
        keywords.extend(name.lower().split())

        # De la description
        desc = theme.get('description', '')
        # Mots importants (> 4 caractères, pas des mots communs)
        stop_words = {'the', 'and', 'for', 'that', 'this', 'with', 'from', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'dans', 'pour', 'avec', 'sont', 'être', 'faire'}
        for word in desc.lower().split():
            word = word.strip('.,!?()[]{}')
            if len(word) > 4 and word not in stop_words:
                keywords.append(word)

        # Des catalyseurs
        for catalyst in theme.get('catalysts', []):
            keywords.extend(catalyst.lower().split()[:3])

        return list(set(keywords))[:10]  # Max 10 mots-clés

    def get_all_themes(self) -> Dict[str, List[str]]:
        """
        Retourne tous les thèmes (prédéfinis + appris).
        À utiliser pour le scan.
        """
        all_themes = dict(self.THEMES_KEYWORDS)  # Copie des thèmes prédéfinis

        # Ajouter les thèmes appris
        learned = self._load_learned_themes()
        for theme_name, theme_data in learned.items():
            if theme_data.get('occurrence_count', 0) >= 2:  # Thème vu au moins 2 fois
                all_themes[theme_name] = theme_data.get('keywords', [])

        return all_themes

    def get_learned_themes_summary(self) -> List[Dict]:
        """Retourne un résumé des thèmes appris"""
        learned = self._load_learned_themes()

        summary = []
        for name, data in learned.items():
            summary.append({
                'name': data.get('name', name),
                'description': data.get('description', '')[:100],
                'occurrence_count': data.get('occurrence_count', 1),
                'symbols': data.get('symbols', [])[:5],
                'discovered_at': data.get('discovered_at'),
                'strength': data.get('strength', 'emerging')
            })

        # Trier par nombre d'occurrences
        summary.sort(key=lambda x: x['occurrence_count'], reverse=True)
        return summary


# =========================================================================
# SCHEDULER INTEGRATION
# =========================================================================

class TrendDiscoveryScheduler:
    """
    Scheduler pour exécuter la découverte de tendances quotidiennement.
    """

    def __init__(
        self,
        discovery: TrendDiscovery,
        run_time: str = "06:00",  # Heure d'exécution (format HH:MM)
        timezone: str = "America/New_York"
    ):
        """
        Args:
            discovery: Instance TrendDiscovery
            run_time: Heure d'exécution quotidienne
            timezone: Fuseau horaire
        """
        self.discovery = discovery
        self.run_time = run_time
        self.timezone = timezone
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Démarre le scheduler"""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"TrendDiscovery scheduler started. Next run at {self.run_time}")

    async def stop(self):
        """Arrête le scheduler"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("TrendDiscovery scheduler stopped")

    async def _run_loop(self):
        """Boucle principale du scheduler"""
        while self._running:
            try:
                # Calculer le temps jusqu'à la prochaine exécution
                now = datetime.now()
                target_hour, target_minute = map(int, self.run_time.split(':'))
                target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)

                if target <= now:
                    target += timedelta(days=1)

                wait_seconds = (target - now).total_seconds()
                logger.info(f"Next trend scan in {wait_seconds/3600:.1f} hours")

                # Attendre
                await asyncio.sleep(wait_seconds)

                # Exécuter le scan
                if self._running:
                    logger.info("Running scheduled trend discovery scan...")
                    report = await self.discovery.daily_scan()
                    logger.info(f"Scheduled scan complete: {len(report.trends)} trends found")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(3600)  # Attendre 1h en cas d'erreur

    async def run_now(self) -> TrendReport:
        """Exécute un scan immédiat"""
        return await self.discovery.daily_scan()


# =========================================================================
# FACTORY FUNCTIONS
# =========================================================================

_discovery_instance: Optional[TrendDiscovery] = None


async def get_trend_discovery(
    api_key: Optional[str] = None,
    model: str = None
) -> TrendDiscovery:
    """
    Retourne une instance singleton de TrendDiscovery.

    Args:
        api_key: Clé API Google AI (ou GOOGLE_AI_API_KEY env var)
        model: Modèle LLM (default: gemini-3-flash-preview via GEMINI_MODEL env var)

    Returns:
        Instance TrendDiscovery initialisée
    """
    global _discovery_instance

    if _discovery_instance is None:
        _discovery_instance = TrendDiscovery(
            api_key=api_key,
            model=model or os.getenv('GEMINI_MODEL', 'gemini-3-flash-preview')
        )
        await _discovery_instance.initialize()

    return _discovery_instance
