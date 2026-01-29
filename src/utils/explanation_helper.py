"""
V4.5 Explanation Helper - AI-powered explanations for trading decisions.

Uses OpenRouter API to generate contextual explanations that help users
understand the trading system's decisions and reasoning.
"""

import os
import httpx
from typing import Dict, Optional, Any
from datetime import datetime

# Cache for explanations to avoid repeated API calls
_explanation_cache: Dict[str, Dict] = {}
_cache_ttl = 3600  # 1 hour


class ExplanationHelper:
    """
    Generates human-readable explanations for trading decisions using LLM.
    """

    # Pre-defined explanations (no API call needed)
    STATIC_EXPLANATIONS = {
        'four_pillars': """
## Les 4 Piliers d'Analyse

Le système analyse chaque action selon **4 piliers complémentaires**, chacun contribuant 25% au score final:

### 🔧 Technical (25%)
Analyse des indicateurs techniques: EMAs, RSI, patterns de prix, supports/résistances.
Un score élevé indique une configuration technique favorable (momentum positif, breakouts).

### 📊 Fundamental (25%)
Évalue les fondamentaux: P/E ratio, croissance, dette, marges.
Un bon score signifie que l'entreprise est fondamentalement solide.

### 💬 Sentiment (25%)
Mesure le sentiment de marché via X/Twitter (Grok), Reddit, StockTwits.
Score positif = sentiment haussier dominant.

### 📰 News (25%)
Analyse des actualités récentes via OpenRouter (Gemini).
Évalue l'impact des news sur le cours potentiel.

**Score Final**: Somme des 4 piliers (0-100). Seuil d'achat: 55+
        """,

        'score_interpretation': """
## Interprétation des Scores

| Score | Interprétation | Action |
|-------|---------------|--------|
| 80-100 | Excellent signal | STRONG BUY - Configuration idéale |
| 65-79 | Bon signal | BUY - Opportunité intéressante |
| 55-64 | Signal modéré | WATCH - Surveiller de près |
| 40-54 | Neutre | HOLD - Pas d'action recommandée |
| 25-39 | Faible | AVOID - Risque élevé |
| 0-24 | Très faible | SELL si position - Conditions défavorables |

**Note**: Un score de 50 représente la neutralité (ni haussier ni baissier).
        """,

        'discovery_process': """
## Processus de Découverte

Le bot découvre les opportunités en plusieurs étapes:

1. **Social Scanning** 🌐
   - Analyse Reddit (r/wallstreetbets, r/stocks, r/investing)
   - Détecte les symboles mentionnés fréquemment
   - Évalue le sentiment des discussions

2. **Grok/X Analysis** 🐦
   - Utilise l'API Grok (xAI) pour X/Twitter
   - Identifie les tendances financières
   - Extrait les thèmes émergents

3. **Volume Anomalies** 📈
   - Détecte les volumes anormaux (>1.5x moyenne 20j)
   - Signal souvent précurseur de mouvements

4. **Construction Watchlist** 📋
   - Fusionne toutes les sources
   - Élimine les doublons
   - Priorise par nombre de sources concordantes
        """,

        'guardrails': """
## Sécurités (Guardrails)

Le système inclut des **protections hard-coded** qui ne peuvent PAS être modifiées:

| Sécurité | Limite | Action |
|----------|--------|--------|
| Perte journalière | 3% max | Kill switch automatique |
| Taille position | 10% max | Blocage si dépassement |
| Drawdown total | 15% max | Pause 24h automatique |
| Trades/jour | 5 max | Blocage après 5 trades |
| Pertes consécutives | 3 max | Alerte et pause |

**Mode Hybride**:
- < 5% du capital: Automatique
- 5-10%: Notification requise
- > 10%: Validation manuelle
        """,

        'timeline_explanation': """
## Flux de Pensées

Ce panneau affiche le **raisonnement en temps réel** du bot:

- 🔍 **Discovery**: Recherche de nouvelles opportunités
- 🌐 **Social Media**: Analyse des réseaux sociaux
- 🐦 **Grok/X**: Insights de X/Twitter via Grok
- 📰 **News/LLM**: Analyse des actualités par IA
- 💡 **Insights**: Conclusions et observations
- 🎯 **Décisions**: Actions prises ou à prendre
- ❌ **Erreurs**: Problèmes rencontrés
- ⏳ **Attente**: Phases d'attente

Les entrées les plus récentes apparaissent en haut.
        """
    }

    def __init__(self):
        self.api_key = os.getenv('OPENROUTER_API_KEY', '')
        self.model = os.getenv('OPENROUTER_MODEL', 'google/gemini-2.0-flash-001')
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self):
        """Initialize HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    def get_static_explanation(self, key: str) -> Optional[str]:
        """Get a pre-defined static explanation"""
        return self.STATIC_EXPLANATIONS.get(key)

    async def generate_explanation(
        self,
        context: str,
        question: str,
        max_tokens: int = 300
    ) -> Optional[str]:
        """
        Generate a contextual explanation using LLM.

        Args:
            context: Context data (scores, factors, etc.)
            question: What to explain
            max_tokens: Maximum response length

        Returns:
            Generated explanation or None if failed
        """
        if not self.api_key:
            return None

        # Check cache
        cache_key = f"{question}:{hash(context)}"
        if cache_key in _explanation_cache:
            cached = _explanation_cache[cache_key]
            if (datetime.now().timestamp() - cached['timestamp']) < _cache_ttl:
                return cached['explanation']

        await self.initialize()

        prompt = f"""Tu es un expert en analyse financière. Explique de manière claire et concise
pour un investisseur débutant.

CONTEXTE:
{context}

QUESTION:
{question}

Réponds en français, de manière pédagogique et factuelle. Maximum 3 paragraphes.
"""

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://tradingbot-v4.local",
                "X-Title": "TradingBot V4 Explanation Helper"
            }

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": max_tokens
            }

            response = await self._client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                explanation = result['choices'][0]['message']['content']

                # Cache the result
                _explanation_cache[cache_key] = {
                    'timestamp': datetime.now().timestamp(),
                    'explanation': explanation
                }

                return explanation

        except Exception as e:
            print(f"Explanation generation failed: {e}")

        return None

    async def explain_decision(self, symbol: str, journey_data: Dict) -> str:
        """
        Generate an explanation for a trading decision.

        Args:
            symbol: Stock symbol
            journey_data: Complete journey data with scores and factors

        Returns:
            Explanation text
        """
        # Build context from journey data
        analysis_step = None
        for step in journey_data.get('journey', []):
            if step.get('step') == 'analysis':
                analysis_step = step
                break

        if not analysis_step:
            return f"Analyse de {symbol} en cours ou non disponible."

        data = analysis_step.get('data', {})
        reasoning = analysis_step.get('reasoning', '')

        context = f"""
Symbol: {symbol}
Score Total: {journey_data.get('current_score', 0)}/100
Status: {journey_data.get('current_status', 'unknown')}

Scores par pilier:
- Technical: {data.get('technical', 0)}/25
- Fundamental: {data.get('fundamental', 0)}/25
- Sentiment: {data.get('sentiment', 0)}/25
- News: {data.get('news', 0)}/25

Raisonnement système: {reasoning[:500]}
"""

        explanation = await self.generate_explanation(
            context=context,
            question=f"Pourquoi le système a attribué ce score à {symbol}? Quels sont les points forts et points faibles?"
        )

        if explanation:
            return explanation

        # Fallback to static explanation
        return self._generate_fallback_explanation(symbol, data, journey_data)

    def _generate_fallback_explanation(
        self,
        symbol: str,
        data: Dict,
        journey_data: Dict
    ) -> str:
        """Generate a basic explanation without LLM"""
        score = journey_data.get('current_score', 50)
        status = journey_data.get('current_status', 'unknown')

        tech = data.get('technical', 12.5)
        fund = data.get('fundamental', 12.5)
        sent = data.get('sentiment', 12.5)
        news = data.get('news', 12.5)

        # Find strongest and weakest pillars
        pillars = [
            ('Technical', tech),
            ('Fundamental', fund),
            ('Sentiment', sent),
            ('News', news)
        ]
        pillars.sort(key=lambda x: x[1], reverse=True)
        strongest = pillars[0]
        weakest = pillars[-1]

        if score >= 55:
            signal = "un signal d'achat"
            reason = f"Le score de {score}/100 dépasse le seuil de 55."
        elif score >= 40:
            signal = "un signal neutre"
            reason = f"Le score de {score}/100 est dans la zone neutre (40-55)."
        else:
            signal = "un signal de prudence"
            reason = f"Le score de {score}/100 est en dessous du seuil minimal."

        return f"""## Analyse de {symbol}

**Verdict**: {signal}
{reason}

**Point fort**: {strongest[0]} ({strongest[1]:.1f}/25)
**Point faible**: {weakest[0]} ({weakest[1]:.1f}/25)

Le système a analysé {symbol} selon les 4 piliers (Technical, Fundamental, Sentiment, News).
Chaque pilier contribue 25% au score final.
"""


# Singleton instance
_helper: Optional[ExplanationHelper] = None


def get_explanation_helper() -> ExplanationHelper:
    """Get or create the ExplanationHelper singleton"""
    global _helper
    if _helper is None:
        _helper = ExplanationHelper()
    return _helper


# Convenience functions for static explanations
def get_four_pillars_explanation() -> str:
    return ExplanationHelper.STATIC_EXPLANATIONS['four_pillars']


def get_score_interpretation() -> str:
    return ExplanationHelper.STATIC_EXPLANATIONS['score_interpretation']


def get_discovery_explanation() -> str:
    return ExplanationHelper.STATIC_EXPLANATIONS['discovery_process']


def get_guardrails_explanation() -> str:
    return ExplanationHelper.STATIC_EXPLANATIONS['guardrails']


def get_timeline_explanation() -> str:
    return ExplanationHelper.STATIC_EXPLANATIONS['timeline_explanation']
