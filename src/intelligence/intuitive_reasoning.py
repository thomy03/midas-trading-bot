"""
Intuitive Reasoning - Raisonnement LLM pour dériver des implications de trading

Ce module utilise un LLM (Gemini) pour:
1. Analyser les news et événements géopolitiques
2. Déduire les implications sectorielles (chaîne de raisonnement)
3. Identifier les actions à acheter/vendre

Exemple:
    News: "Trump menace d'annexer le Groenland"
    Raisonnement:
        → L'Europe se sent menacée
        → L'Europe va investir massivement dans sa défense
        → Les valeurs défense européennes vont monter
    Actions: Thales, Rheinmetall, Leonardo, BAE Systems

Auteur: TradingBot V5
Date: Janvier 2026
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum

import aiohttp

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    BUY = "buy"
    SELL = "sell"
    WATCH = "watch"


@dataclass
class ReasoningStep:
    """Une étape dans la chaîne de raisonnement"""
    step_number: int
    premise: str  # "Trump menace d'annexer le Groenland"
    implication: str  # "L'Europe investit dans sa défense"
    confidence: float  # 0-1


@dataclass
class TradingImplication:
    """Implication de trading dérivée du raisonnement"""
    sector: str  # "Defense", "Energy", etc.
    region: str  # "Europe", "US", "Asia"
    direction: str  # "buy", "sell", "watch"
    symbols: List[str]  # ["RHM.DE", "HO.PA", "LDO.MI"]
    reasoning_chain: List[ReasoningStep]
    confidence: float  # 0-1
    time_horizon: str  # "short_term", "medium_term", "long_term"
    catalyst: str  # L'événement déclencheur


@dataclass
class NewsAnalysis:
    """Analyse complète d'une news"""
    headline: str
    source: str
    timestamp: str
    raw_analysis: str  # Réponse brute du LLM
    implications: List[TradingImplication]
    sentiment: str  # "bullish", "bearish", "neutral"
    impact_score: float  # 0-10


# Mapping des secteurs vers les actions clés
SECTOR_STOCKS = {
    # Défense
    "defense_europe": {
        "stocks": ["RHM.DE", "HO.PA", "LDO.MI", "BA.L", "SAF.PA", "AIR.PA"],
        "names": {
            "RHM.DE": "Rheinmetall",
            "HO.PA": "Thales",
            "LDO.MI": "Leonardo",
            "BA.L": "BAE Systems",
            "SAF.PA": "Safran",
            "AIR.PA": "Airbus"
        }
    },
    "defense_us": {
        "stocks": ["LMT", "RTX", "NOC", "GD", "BA", "HII"],
        "names": {
            "LMT": "Lockheed Martin",
            "RTX": "Raytheon",
            "NOC": "Northrop Grumman",
            "GD": "General Dynamics",
            "BA": "Boeing",
            "HII": "Huntington Ingalls"
        }
    },

    # Énergie
    "energy_oil": {
        "stocks": ["XOM", "CVX", "COP", "SLB", "TTE.PA", "BP.L"],
        "names": {
            "XOM": "Exxon Mobil",
            "CVX": "Chevron",
            "COP": "ConocoPhillips",
            "SLB": "Schlumberger",
            "TTE.PA": "TotalEnergies",
            "BP.L": "BP"
        }
    },
    "energy_renewable": {
        "stocks": ["ENPH", "FSLR", "NEE", "SEDG", "PLUG", "BE"],
        "names": {
            "ENPH": "Enphase Energy",
            "FSLR": "First Solar",
            "NEE": "NextEra Energy",
            "SEDG": "SolarEdge",
            "PLUG": "Plug Power",
            "BE": "Bloom Energy"
        }
    },
    "energy_nuclear": {
        "stocks": ["CCJ", "UEC", "NNE", "LEU", "SMR"],
        "names": {
            "CCJ": "Cameco",
            "UEC": "Uranium Energy Corp",
            "NNE": "Nano Nuclear Energy",
            "LEU": "Centrus Energy",
            "SMR": "NuScale Power"
        }
    },

    # Tech
    "tech_ai": {
        "stocks": ["NVDA", "AMD", "AVGO", "MRVL", "PLTR", "AI"],
        "names": {
            "NVDA": "Nvidia",
            "AMD": "AMD",
            "AVGO": "Broadcom",
            "MRVL": "Marvell",
            "PLTR": "Palantir",
            "AI": "C3.ai"
        }
    },
    "tech_cybersecurity": {
        "stocks": ["CRWD", "PANW", "ZS", "FTNT", "S", "OKTA"],
        "names": {
            "CRWD": "CrowdStrike",
            "PANW": "Palo Alto Networks",
            "ZS": "Zscaler",
            "FTNT": "Fortinet",
            "S": "SentinelOne",
            "OKTA": "Okta"
        }
    },
    "tech_cloud": {
        "stocks": ["MSFT", "AMZN", "GOOGL", "SNOW", "NET", "MDB"],
        "names": {
            "MSFT": "Microsoft",
            "AMZN": "Amazon",
            "GOOGL": "Google",
            "SNOW": "Snowflake",
            "NET": "Cloudflare",
            "MDB": "MongoDB"
        }
    },

    # Santé
    "pharma_obesity": {
        "stocks": ["LLY", "NVO", "AMGN", "VKTX", "RYTM"],
        "names": {
            "LLY": "Eli Lilly",
            "NVO": "Novo Nordisk",
            "AMGN": "Amgen",
            "VKTX": "Viking Therapeutics",
            "RYTM": "Rhythm Pharmaceuticals"
        }
    },
    "pharma_biotech": {
        "stocks": ["MRNA", "BNTX", "REGN", "VRTX", "BIIB"],
        "names": {
            "MRNA": "Moderna",
            "BNTX": "BioNTech",
            "REGN": "Regeneron",
            "VRTX": "Vertex",
            "BIIB": "Biogen"
        }
    },

    # Finance
    "finance_banks_us": {
        "stocks": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
        "names": {
            "JPM": "JPMorgan",
            "BAC": "Bank of America",
            "WFC": "Wells Fargo",
            "GS": "Goldman Sachs",
            "MS": "Morgan Stanley",
            "C": "Citigroup"
        }
    },
    "finance_crypto": {
        "stocks": ["COIN", "MSTR", "RIOT", "MARA", "HUT"],
        "names": {
            "COIN": "Coinbase",
            "MSTR": "MicroStrategy",
            "RIOT": "Riot Platforms",
            "MARA": "Marathon Digital",
            "HUT": "Hut 8"
        }
    },

    # Consommation
    "consumer_luxury": {
        "stocks": ["MC.PA", "RMS.PA", "KER.PA", "LVMUY"],
        "names": {
            "MC.PA": "LVMH",
            "RMS.PA": "Hermès",
            "KER.PA": "Kering",
            "LVMUY": "LVMH ADR"
        }
    },
    "consumer_ev": {
        "stocks": ["TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI"],
        "names": {
            "TSLA": "Tesla",
            "RIVN": "Rivian",
            "LCID": "Lucid",
            "NIO": "NIO",
            "XPEV": "XPeng",
            "LI": "Li Auto"
        }
    },

    # Matériaux
    "materials_rare_earth": {
        "stocks": ["MP", "LAC", "ALB", "LTHM", "SQM"],
        "names": {
            "MP": "MP Materials",
            "LAC": "Lithium Americas",
            "ALB": "Albemarle",
            "LTHM": "Livent",
            "SQM": "SQM"
        }
    },
    "materials_gold": {
        "stocks": ["GOLD", "NEM", "AEM", "FNV", "WPM"],
        "names": {
            "GOLD": "Barrick Gold",
            "NEM": "Newmont",
            "AEM": "Agnico Eagle",
            "FNV": "Franco-Nevada",
            "WPM": "Wheaton Precious Metals"
        }
    },

    # Infrastructure
    "infrastructure_construction": {
        "stocks": ["CAT", "DE", "URI", "VMC", "MLM"],
        "names": {
            "CAT": "Caterpillar",
            "DE": "Deere",
            "URI": "United Rentals",
            "VMC": "Vulcan Materials",
            "MLM": "Martin Marietta"
        }
    },

    # Shipping & Logistics
    "shipping": {
        "stocks": ["ZIM", "GOGL", "SBLK", "FDX", "UPS"],
        "names": {
            "ZIM": "ZIM Shipping",
            "GOGL": "Golden Ocean",
            "SBLK": "Star Bulk",
            "FDX": "FedEx",
            "UPS": "UPS"
        }
    }
}

# Thèmes géopolitiques et leurs implications
GEOPOLITICAL_THEMES = {
    "us_china_tension": {
        "keywords": ["china", "taiwan", "tariff", "trade war", "xi jinping", "decoupling"],
        "bullish_sectors": ["defense_us", "tech_cybersecurity", "materials_rare_earth"],
        "bearish_sectors": ["tech_ai", "consumer_luxury"],  # Si dépendants de Chine
        "description": "Tensions US-Chine"
    },
    "europe_defense": {
        "keywords": ["nato", "defense spending", "ukraine", "russia", "european army", "groenland", "greenland", "trump threat"],
        "bullish_sectors": ["defense_europe", "defense_us", "tech_cybersecurity"],
        "bearish_sectors": [],
        "description": "Réarmement européen"
    },
    "middle_east_conflict": {
        "keywords": ["israel", "iran", "houthi", "red sea", "oil price", "opec"],
        "bullish_sectors": ["energy_oil", "defense_us", "shipping"],
        "bearish_sectors": ["consumer_ev"],  # Si pétrole monte
        "description": "Conflit Moyen-Orient"
    },
    "climate_policy": {
        "keywords": ["climate", "green deal", "carbon tax", "renewable", "cop", "paris agreement"],
        "bullish_sectors": ["energy_renewable", "energy_nuclear", "consumer_ev"],
        "bearish_sectors": ["energy_oil"],
        "description": "Politique climatique"
    },
    "ai_regulation": {
        "keywords": ["ai regulation", "artificial intelligence", "chatgpt", "openai", "nvidia", "chips act"],
        "bullish_sectors": ["tech_ai", "tech_cybersecurity"],
        "bearish_sectors": [],
        "description": "Révolution IA"
    },
    "interest_rates": {
        "keywords": ["fed", "interest rate", "inflation", "powell", "rate cut", "rate hike", "ecb", "lagarde"],
        "bullish_sectors": ["finance_banks_us"] if "cut" else [],
        "bearish_sectors": ["finance_banks_us"] if "hike" else [],
        "description": "Politique monétaire"
    },
    "crypto_adoption": {
        "keywords": ["bitcoin", "crypto", "etf", "coinbase", "sec approval", "btc"],
        "bullish_sectors": ["finance_crypto"],
        "bearish_sectors": [],
        "description": "Adoption crypto"
    }
}


class IntuitiveReasoning:
    """
    Module de raisonnement intuitif utilisant un LLM.

    Analyse les news et événements pour dériver des implications de trading
    via une chaîne de raisonnement logique.

    Usage:
        reasoner = IntuitiveReasoning()
        await reasoner.initialize()

        # Analyser une news
        analysis = await reasoner.analyze_news(
            "Trump menace d'annexer le Groenland",
            source="Reuters"
        )

        # Obtenir les implications de trading
        for implication in analysis.implications:
            print(f"BUY {implication.sector}: {implication.symbols}")

        # Scan automatique des news du jour
        implications = await reasoner.daily_news_scan()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "google/gemini-2.0-flash-001",
        data_dir: str = "data/intuitive_reasoning"
    ):
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.model = model
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.analyses_file = self.data_dir / "news_analyses.json"
        self.implications_file = self.data_dir / "trading_implications.json"

        self.analyses: List[NewsAnalysis] = []
        self.active_implications: List[TradingImplication] = []

        self._session: Optional[aiohttp.ClientSession] = None
        self._initialized = False

    async def initialize(self):
        """Initialiser le module"""
        if self._initialized:
            return

        # Charger les analyses passées
        if self.analyses_file.exists():
            try:
                with open(self.analyses_file, 'r') as f:
                    data = json.load(f)
                    self.analyses = [self._dict_to_analysis(a) for a in data]
                logger.info(f"Loaded {len(self.analyses)} past analyses")
            except Exception as e:
                logger.error(f"Error loading analyses: {e}")

        # Charger les implications actives
        if self.implications_file.exists():
            try:
                with open(self.implications_file, 'r') as f:
                    data = json.load(f)
                    self.active_implications = [self._dict_to_implication(i) for i in data]
            except Exception as e:
                logger.error(f"Error loading implications: {e}")

        self._session = aiohttp.ClientSession()
        self._initialized = True
        logger.info("IntuitiveReasoning initialized")

    async def close(self):
        """Fermer les connexions"""
        if self._session:
            await self._session.close()

    def _save(self):
        """Persister les données"""
        with open(self.analyses_file, 'w') as f:
            json.dump([self._analysis_to_dict(a) for a in self.analyses[-100:]], f, indent=2)

        with open(self.implications_file, 'w') as f:
            json.dump([self._implication_to_dict(i) for i in self.active_implications], f, indent=2)

    def _analysis_to_dict(self, analysis: NewsAnalysis) -> Dict:
        return {
            'headline': analysis.headline,
            'source': analysis.source,
            'timestamp': analysis.timestamp,
            'raw_analysis': analysis.raw_analysis,
            'implications': [self._implication_to_dict(i) for i in analysis.implications],
            'sentiment': analysis.sentiment,
            'impact_score': analysis.impact_score
        }

    def _dict_to_analysis(self, data: Dict) -> NewsAnalysis:
        return NewsAnalysis(
            headline=data['headline'],
            source=data['source'],
            timestamp=data['timestamp'],
            raw_analysis=data['raw_analysis'],
            implications=[self._dict_to_implication(i) for i in data.get('implications', [])],
            sentiment=data['sentiment'],
            impact_score=data['impact_score']
        )

    def _implication_to_dict(self, impl: TradingImplication) -> Dict:
        return {
            'sector': impl.sector,
            'region': impl.region,
            'direction': impl.direction,
            'symbols': impl.symbols,
            'reasoning_chain': [asdict(r) for r in impl.reasoning_chain],
            'confidence': impl.confidence,
            'time_horizon': impl.time_horizon,
            'catalyst': impl.catalyst
        }

    def _dict_to_implication(self, data: Dict) -> TradingImplication:
        return TradingImplication(
            sector=data['sector'],
            region=data['region'],
            direction=data['direction'],
            symbols=data['symbols'],
            reasoning_chain=[ReasoningStep(**r) for r in data.get('reasoning_chain', [])],
            confidence=data['confidence'],
            time_horizon=data['time_horizon'],
            catalyst=data['catalyst']
        )

    async def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Appeler le LLM via OpenRouter"""
        if not self._session:
            self._session = aiohttp.ClientSession()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,  # Bas pour cohérence
            "max_tokens": 2000
        }

        try:
            async with self._session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data['choices'][0]['message']['content']
                else:
                    error = await response.text()
                    logger.error(f"LLM API error: {response.status} - {error}")
                    return ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""

    async def analyze_news(
        self,
        headline: str,
        content: str = "",
        source: str = "Unknown"
    ) -> NewsAnalysis:
        """
        Analyser une news et dériver les implications de trading.

        Args:
            headline: Titre de la news
            content: Contenu complet (optionnel)
            source: Source de la news

        Returns:
            NewsAnalysis avec les implications de trading
        """
        if not self._initialized:
            await self.initialize()

        system_prompt = """Tu es un analyste financier expert en géopolitique et marchés.
Tu dois analyser les news et dériver les IMPLICATIONS CONCRÈTES pour le trading.

Pour chaque news, tu dois:
1. Identifier le thème géopolitique/économique
2. Construire une CHAÎNE DE RAISONNEMENT logique (3-5 étapes)
3. Identifier les SECTEURS impactés (bullish ou bearish)
4. Suggérer des ACTIONS CONCRÈTES à acheter ou vendre

Format de réponse OBLIGATOIRE (JSON):
{
    "theme": "nom du thème",
    "sentiment": "bullish|bearish|neutral",
    "impact_score": 7.5,
    "reasoning_chain": [
        {"step": 1, "premise": "...", "implication": "..."},
        {"step": 2, "premise": "...", "implication": "..."}
    ],
    "implications": [
        {
            "sector": "defense_europe",
            "direction": "buy",
            "confidence": 0.8,
            "time_horizon": "medium_term",
            "stocks": ["RHM.DE", "HO.PA"]
        }
    ]
}

IMPORTANT: Sois PRÉCIS et ACTIONNABLE. Pas de généralités."""

        prompt = f"""Analyse cette news et donne-moi les implications de trading:

HEADLINE: {headline}

{f"CONTENT: {content}" if content else ""}

SOURCE: {source}

Réponds UNIQUEMENT en JSON valide."""

        raw_response = await self._call_llm(prompt, system_prompt)

        # Parser la réponse
        implications = []
        sentiment = "neutral"
        impact_score = 5.0

        try:
            # Extraire le JSON de la réponse
            json_start = raw_response.find('{')
            json_end = raw_response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(raw_response[json_start:json_end])

                sentiment = parsed.get('sentiment', 'neutral')
                impact_score = parsed.get('impact_score', 5.0)

                # Construire la chaîne de raisonnement
                reasoning_chain = []
                for step in parsed.get('reasoning_chain', []):
                    reasoning_chain.append(ReasoningStep(
                        step_number=step.get('step', 0),
                        premise=step.get('premise', ''),
                        implication=step.get('implication', ''),
                        confidence=0.8
                    ))

                # Construire les implications
                for impl in parsed.get('implications', []):
                    sector = impl.get('sector', '')
                    stocks = impl.get('stocks', [])

                    # Si pas de stocks, utiliser ceux du mapping
                    if not stocks and sector in SECTOR_STOCKS:
                        stocks = SECTOR_STOCKS[sector]['stocks'][:4]

                    implications.append(TradingImplication(
                        sector=sector,
                        region=impl.get('region', 'global'),
                        direction=impl.get('direction', 'watch'),
                        symbols=stocks,
                        reasoning_chain=reasoning_chain,
                        confidence=impl.get('confidence', 0.5),
                        time_horizon=impl.get('time_horizon', 'medium_term'),
                        catalyst=headline
                    ))

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Fallback: analyse par mots-clés
            implications = self._fallback_keyword_analysis(headline)

        # Créer l'analyse
        analysis = NewsAnalysis(
            headline=headline,
            source=source,
            timestamp=datetime.now().isoformat(),
            raw_analysis=raw_response,
            implications=implications,
            sentiment=sentiment,
            impact_score=impact_score
        )

        # Sauvegarder
        self.analyses.append(analysis)
        self.active_implications.extend(implications)
        self._save()

        logger.info(f"[INTUITIVE] Analyzed: '{headline[:50]}...' -> {len(implications)} implications")

        return analysis

    def _fallback_keyword_analysis(self, headline: str) -> List[TradingImplication]:
        """Analyse de fallback basée sur les mots-clés"""
        headline_lower = headline.lower()
        implications = []

        for theme_name, theme_data in GEOPOLITICAL_THEMES.items():
            # Vérifier si des mots-clés correspondent
            matches = [kw for kw in theme_data['keywords'] if kw in headline_lower]

            if matches:
                # Secteurs bullish
                for sector in theme_data['bullish_sectors']:
                    if sector in SECTOR_STOCKS:
                        implications.append(TradingImplication(
                            sector=sector,
                            region="global",
                            direction="buy",
                            symbols=SECTOR_STOCKS[sector]['stocks'][:4],
                            reasoning_chain=[ReasoningStep(
                                step_number=1,
                                premise=headline,
                                implication=f"Bullish pour {sector}",
                                confidence=0.6
                            )],
                            confidence=0.6,
                            time_horizon="medium_term",
                            catalyst=headline
                        ))

                # Secteurs bearish
                for sector in theme_data['bearish_sectors']:
                    if sector in SECTOR_STOCKS:
                        implications.append(TradingImplication(
                            sector=sector,
                            region="global",
                            direction="sell",
                            symbols=SECTOR_STOCKS[sector]['stocks'][:4],
                            reasoning_chain=[ReasoningStep(
                                step_number=1,
                                premise=headline,
                                implication=f"Bearish pour {sector}",
                                confidence=0.5
                            )],
                            confidence=0.5,
                            time_horizon="short_term",
                            catalyst=headline
                        ))

        return implications

    async def daily_news_scan(self) -> List[TradingImplication]:
        """
        Scanner les news du jour et générer les implications de trading.

        Returns:
            Liste des implications de trading
        """
        if not self._initialized:
            await self.initialize()

        # Utiliser NewsAPI si disponible
        newsapi_key = os.getenv('NEWSAPI_KEY')
        if not newsapi_key:
            logger.warning("NEWSAPI_KEY not set, using sample headlines")
            return []

        try:
            async with self._session.get(
                "https://newsapi.org/v2/top-headlines",
                params={
                    "apiKey": newsapi_key,
                    "category": "business",
                    "language": "en",
                    "pageSize": 20
                }
            ) as response:
                if response.status != 200:
                    logger.error(f"NewsAPI error: {response.status}")
                    return []

                data = await response.json()
                articles = data.get('articles', [])

        except Exception as e:
            logger.error(f"NewsAPI request failed: {e}")
            return []

        # Analyser chaque article
        all_implications = []
        for article in articles[:10]:  # Max 10 pour limiter les appels LLM
            headline = article.get('title', '')
            content = article.get('description', '')
            source = article.get('source', {}).get('name', 'Unknown')

            if headline:
                analysis = await self.analyze_news(headline, content, source)
                all_implications.extend(analysis.implications)

                # Pause pour éviter rate limiting
                await asyncio.sleep(1)

        logger.info(f"[INTUITIVE] Daily scan complete: {len(all_implications)} implications from {len(articles)} articles")

        return all_implications

    def get_buy_recommendations(self, min_confidence: float = 0.6) -> List[Dict]:
        """
        Obtenir les recommandations d'achat actuelles.

        Args:
            min_confidence: Confiance minimum (0-1)

        Returns:
            Liste de recommandations avec symboles et raisonnement
        """
        recommendations = []

        for impl in self.active_implications:
            if impl.direction == "buy" and impl.confidence >= min_confidence:
                recommendations.append({
                    'sector': impl.sector,
                    'symbols': impl.symbols,
                    'confidence': impl.confidence,
                    'catalyst': impl.catalyst,
                    'reasoning': [
                        f"{r.step_number}. {r.premise} → {r.implication}"
                        for r in impl.reasoning_chain
                    ],
                    'time_horizon': impl.time_horizon
                })

        # Trier par confiance
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)

        return recommendations

    def get_symbols_to_watch(self) -> List[str]:
        """Obtenir tous les symboles à surveiller basés sur les implications"""
        symbols = set()
        for impl in self.active_implications:
            if impl.direction in ["buy", "watch"] and impl.confidence >= 0.5:
                symbols.update(impl.symbols)
        return list(symbols)

    def clear_old_implications(self, days: int = 7):
        """Supprimer les implications de plus de N jours"""
        cutoff = datetime.now() - timedelta(days=days)

        # Filtrer les analyses récentes
        self.analyses = [
            a for a in self.analyses
            if datetime.fromisoformat(a.timestamp) > cutoff
        ]

        # Reconstruire les implications actives
        self.active_implications = []
        for analysis in self.analyses:
            self.active_implications.extend(analysis.implications)

        self._save()

    def get_statistics(self) -> Dict:
        """Obtenir les statistiques du module"""
        buy_count = len([i for i in self.active_implications if i.direction == "buy"])
        sell_count = len([i for i in self.active_implications if i.direction == "sell"])

        return {
            'total_analyses': len(self.analyses),
            'active_implications': len(self.active_implications),
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'avg_confidence': sum(i.confidence for i in self.active_implications) / len(self.active_implications) if self.active_implications else 0,
            'sectors_covered': list(set(i.sector for i in self.active_implications)),
            'symbols_to_watch': len(self.get_symbols_to_watch())
        }


# Singleton
_intuitive_reasoning: Optional[IntuitiveReasoning] = None


async def get_intuitive_reasoning() -> IntuitiveReasoning:
    """Obtenir l'instance singleton"""
    global _intuitive_reasoning
    if _intuitive_reasoning is None:
        _intuitive_reasoning = IntuitiveReasoning()
        await _intuitive_reasoning.initialize()
    return _intuitive_reasoning
