"""
Narrative Generator - Generates human-readable analysis reports.

Uses LLM (Gemini via OpenRouter) to transform raw data into
structured, reasoned analysis reports that explain:
- WHY a stock is in the watchlist (sources, catalysts)
- WHY we're buying (pillar convergence, key factors)
- WHAT are the risks and reservations
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SourceEvidence:
    """Evidence from a single source"""
    source_name: str  # "Reddit", "Grok/X", "News", "Volume"
    raw_metric: str  # "47 mentions/h vs 11 baseline"
    themes: List[str]  # ["Blackwell shipping", "CES keynote"]
    key_posts: List[str]  # Top posts/tweets summarized
    sentiment: float  # 0-1
    reliability: str  # "high", "medium", "low"
    catalyst_linked: bool  # Is the buzz linked to a real catalyst?


@dataclass
class AnalysisNarrative:
    """Complete narrative analysis for a symbol"""
    symbol: str
    timestamp: str

    # Section 1: Why in watchlist
    detection_sources: List[SourceEvidence]
    watchlist_summary: str  # LLM-generated narrative

    # Section 2: Why buy/not buy
    pillar_scores: Dict[str, float]  # technical, fundamental, sentiment, news
    pillar_explanations: Dict[str, str]  # LLM explanation for each pillar
    decision: str  # "BUY", "WATCH", "PASS"
    decision_reasoning: str  # LLM-generated reasoning

    # Section 3: Risks and reservations
    risks: List[str]
    reservations: str  # LLM-generated concerns
    exit_triggers: List[str]  # When would we exit

    def to_markdown(self) -> str:
        """Generate full markdown report"""
        lines = []
        lines.append(f"# Analyse {self.symbol} - {self.timestamp}")
        lines.append("")

        # Section 1: Watchlist
        lines.append("## 📊 POURQUOI DANS MA WATCHLIST")
        lines.append("")
        lines.append(self.watchlist_summary)
        lines.append("")

        for source in self.detection_sources:
            lines.append(f"### {source.source_name}")
            lines.append(f"- **Métrique**: {source.raw_metric}")
            if source.themes:
                lines.append(f"- **Thèmes**: {', '.join(source.themes)}")
            lines.append(f"- **Fiabilité**: {source.reliability}")
            lines.append("")

        # Section 2: Decision
        lines.append("## 💰 MA DECISION")
        lines.append("")
        lines.append(f"**{self.decision}**")
        lines.append("")
        lines.append(self.decision_reasoning)
        lines.append("")

        lines.append("### Scores des 4 Piliers")
        for pillar, score in self.pillar_scores.items():
            explanation = self.pillar_explanations.get(pillar, "")
            lines.append(f"- **{pillar.capitalize()}** ({score}/100): {explanation}")
        lines.append("")

        # Section 3: Risks
        lines.append("## ⚠️ MES RESERVES")
        lines.append("")
        lines.append(self.reservations)
        lines.append("")

        if self.exit_triggers:
            lines.append("### Conditions de sortie")
            for trigger in self.exit_triggers:
                lines.append(f"- {trigger}")

        return "\n".join(lines)


class NarrativeGenerator:
    """
    Generates narrative analysis reports using LLM.

    Takes raw data from various sources (scanners, reasoning engine)
    and produces human-readable, reasoned analysis.
    """

    NARRATIVE_PROMPT = """Tu es un analyste financier expérimenté. Génère un rapport d'analyse NARRATIF et ETAYE.

DONNÉES BRUTES:
{raw_data}

INSTRUCTIONS:
1. Explique POURQUOI ce symbole est intéressant (pas juste "47 mentions" mais QUELS thèmes, QUELS catalyseurs)
2. Chaque conclusion doit être JUSTIFIÉE par des faits
3. Exprime tes DOUTES et RÉSERVES de manière honnête
4. Utilise un ton professionnel mais accessible

FORMAT DE RÉPONSE (JSON):
{{
    "watchlist_summary": "Paragraphe expliquant pourquoi ce symbole a attiré mon attention, avec les thèmes clés détectés",
    "pillar_explanations": {{
        "technical": "Explication du score technique (RSI, EMAs, patterns)",
        "fundamental": "Explication du score fondamental (P/E, croissance, etc.)",
        "sentiment": "Explication du sentiment (thèmes Reddit, qualité du buzz)",
        "news": "Explication des news (catalyseurs, événements)"
    }},
    "decision_reasoning": "Paragraphe expliquant la décision finale avec convergence des piliers",
    "reservations": "Paragraphe sur les risques et doutes",
    "exit_triggers": ["Condition 1 pour sortir", "Condition 2 pour sortir"]
}}
"""

    def __init__(self, api_key: str = None, model: str = None):
        # V4.8: Use Gemini direct instead of OpenRouter
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY')
        self.model = model or os.getenv('GEMINI_MODEL', 'gemini-3-flash-preview')
        self._client = None

    async def initialize(self):
        """Initialize the LLM client (Gemini direct)"""
        if not self.api_key:
            logger.warning("No GOOGLE_AI_API_KEY - narrative generation disabled")
            return

        try:
            from .gemini_client import GeminiClient
            self._client = GeminiClient(api_key=self.api_key, model=self.model)
            await self._client.initialize()
            logger.info(f"NarrativeGenerator initialized with Gemini: {self._client.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize NarrativeGenerator: {e}")

    async def close(self):
        """Close the LLM client"""
        if self._client:
            await self._client.close()

    async def generate_analysis(
        self,
        symbol: str,
        social_data: Dict = None,
        grok_data: Dict = None,
        technical_data: Dict = None,
        fundamental_data: Dict = None,
        news_data: Dict = None,
        reasoning_result: Any = None
    ) -> AnalysisNarrative:
        """
        Generate a complete narrative analysis for a symbol.

        Args:
            symbol: Stock symbol
            social_data: Data from SocialScanner (Reddit, StockTwits)
            grok_data: Data from GrokScanner (X/Twitter)
            technical_data: Technical indicators
            fundamental_data: Company fundamentals
            news_data: Recent news
            reasoning_result: Result from ReasoningEngine

        Returns:
            AnalysisNarrative with complete report
        """
        timestamp = datetime.now().strftime("%d %b %Y %H:%M")

        # Build source evidence
        sources = []

        # Reddit/Social source
        if social_data:
            social_data = social_data.get('social', {})
            sources.append(SourceEvidence(
                source_name="Reddit",
                raw_metric=f"{social_data.get('mentions', 0)} mentions/h (baseline: {social_data.get('baseline', 0)})",
                themes=social_data.get('themes', []),
                key_posts=social_data.get('top_posts', [])[:3],
                sentiment=social_data.get('sentiment', 0.5),
                reliability="medium" if social_data.get('mentions', 0) > 20 else "low",
                catalyst_linked=bool(social_data.get('catalysts', []))
            ))

        # Grok/X source
        if grok_data:
            sources.append(SourceEvidence(
                source_name="Grok/X",
                raw_metric=f"Sentiment: {grok_data.get('sentiment', 0):.2f}",
                themes=grok_data.get('themes', []),
                key_posts=grok_data.get('key_tweets', [])[:3],
                sentiment=grok_data.get('sentiment', 0.5),
                reliability="high" if grok_data.get('analyst_mentions', 0) > 0 else "medium",
                catalyst_linked=grok_data.get('catalyst_linked', False)
            ))

        # Volume source
        if technical_data and technical_data.get('volume_ratio', 1.0) > 1.5:
            sources.append(SourceEvidence(
                source_name="Volume",
                raw_metric=f"{technical_data.get('volume_ratio', 1.0):.1f}x la moyenne",
                themes=["Institutional interest" if technical_data.get('volume_ratio', 1.0) > 2 else "Elevated activity"],
                key_posts=[],
                sentiment=0.6,
                reliability="high",
                catalyst_linked=True
            ))

        # Extract pillar scores from reasoning result
        pillar_scores = {
            "technical": 0,
            "fundamental": 0,
            "sentiment": 0,
            "news": 0
        }

        if reasoning_result:
            if hasattr(reasoning_result, 'technical_score') and reasoning_result.technical_score:
                pillar_scores["technical"] = reasoning_result.technical_score.score
            if hasattr(reasoning_result, 'fundamental_score') and reasoning_result.fundamental_score:
                pillar_scores["fundamental"] = reasoning_result.fundamental_score.score
            if hasattr(reasoning_result, 'sentiment_score') and reasoning_result.sentiment_score:
                pillar_scores["sentiment"] = reasoning_result.sentiment_score.score
            if hasattr(reasoning_result, 'news_score') and reasoning_result.news_score:
                pillar_scores["news"] = reasoning_result.news_score.score

        # Generate LLM narrative if client available
        pillar_explanations = {}
        watchlist_summary = ""
        decision_reasoning = ""
        reservations = ""
        exit_triggers = []

        if self._client:
            try:
                llm_result = await self._generate_llm_narrative(
                    symbol=symbol,
                    sources=sources,
                    pillar_scores=pillar_scores,
                    social_data=social_data,
                    grok_data=grok_data,
                    technical_data=technical_data,
                    fundamental_data=fundamental_data,
                    news_data=news_data,
                    reasoning_result=reasoning_result
                )

                watchlist_summary = llm_result.get('watchlist_summary', '')
                pillar_explanations = llm_result.get('pillar_explanations', {})
                decision_reasoning = llm_result.get('decision_reasoning', '')
                reservations = llm_result.get('reservations', '')
                exit_triggers = llm_result.get('exit_triggers', [])

            except Exception as e:
                logger.error(f"LLM narrative generation failed: {e}")
                # Fallback to basic narrative
                watchlist_summary = self._generate_basic_watchlist_summary(symbol, sources)
                pillar_explanations = self._generate_basic_pillar_explanations(pillar_scores)
                decision_reasoning = self._generate_basic_decision(pillar_scores)
                reservations = "Analyse automatique - validation humaine recommandée."
        else:
            # No LLM - generate basic narrative
            watchlist_summary = self._generate_basic_watchlist_summary(symbol, sources)
            pillar_explanations = self._generate_basic_pillar_explanations(pillar_scores)
            decision_reasoning = self._generate_basic_decision(pillar_scores)
            reservations = "Analyse automatique - validation humaine recommandée."

        # Determine decision
        total_score = sum(pillar_scores.values()) / 4
        if total_score > 60:
            decision = "BUY"
        elif total_score > 40:
            decision = "WATCH"
        else:
            decision = "PASS"

        # Extract risks
        risks = []
        if reasoning_result and hasattr(reasoning_result, 'risk_factors'):
            risks = reasoning_result.risk_factors[:5]

        return AnalysisNarrative(
            symbol=symbol,
            timestamp=timestamp,
            detection_sources=sources,
            watchlist_summary=watchlist_summary,
            pillar_scores=pillar_scores,
            pillar_explanations=pillar_explanations,
            decision=decision,
            decision_reasoning=decision_reasoning,
            risks=risks,
            reservations=reservations,
            exit_triggers=exit_triggers
        )

    async def _generate_llm_narrative(
        self,
        symbol: str,
        sources: List[SourceEvidence],
        pillar_scores: Dict[str, float],
        **data
    ) -> Dict:
        """Generate narrative using LLM"""

        # Build raw data summary for prompt
        raw_data = {
            "symbol": symbol,
            "sources": [
                {
                    "name": s.source_name,
                    "metric": s.raw_metric,
                    "themes": s.themes,
                    "key_content": s.key_posts[:2],
                    "sentiment": s.sentiment,
                    "catalyst_linked": s.catalyst_linked
                }
                for s in sources
            ],
            "pillar_scores": pillar_scores,
        }

        # Add reasoning result key factors if available
        reasoning_result = data.get('reasoning_result')
        if reasoning_result:
            if hasattr(reasoning_result, 'key_factors'):
                raw_data["key_factors"] = reasoning_result.key_factors[:5]
            if hasattr(reasoning_result, 'risk_factors'):
                raw_data["risk_factors"] = reasoning_result.risk_factors[:5]
            if hasattr(reasoning_result, 'reasoning_summary'):
                raw_data["reasoning_summary"] = reasoning_result.reasoning_summary

        # Add fundamental data
        fundamental = data.get('fundamental_data')
        if fundamental:
            raw_data["fundamentals"] = {
                "pe_ratio": fundamental.get('pe_ratio'),
                "revenue_growth": fundamental.get('revenue_growth'),
                "eps_growth": fundamental.get('eps_growth'),
                "market_cap": fundamental.get('market_cap')
            }

        # Add technical data
        technical = data.get('technical_data')
        if technical:
            raw_data["technical"] = {
                "rsi": technical.get('rsi'),
                "ema_alignment": technical.get('ema_alignment'),
                "volume_ratio": technical.get('volume_ratio'),
                "trend": technical.get('trend')
            }

        prompt = self.NARRATIVE_PROMPT.format(raw_data=json.dumps(raw_data, indent=2, default=str))

        try:
            # V4.8: Use Gemini direct with chat_json for automatic JSON parsing
            result = await self._client.chat_json(
                prompt,
                system_prompt="Tu es un analyste financier expert. Réponds toujours en JSON valide.",
                max_tokens=1500,
                temperature=0.3
            )
            return result

        except Exception as e:
            logger.error(f"Gemini query failed: {e}")
            return {}

    def _generate_basic_watchlist_summary(self, symbol: str, sources: List[SourceEvidence]) -> str:
        """Generate basic watchlist summary without LLM"""
        parts = [f"J'ai détecté {symbol} via {len(sources)} source(s) convergentes:"]

        for source in sources:
            parts.append(f"- **{source.source_name}**: {source.raw_metric}")
            if source.themes:
                parts.append(f"  Thèmes: {', '.join(source.themes[:3])}")

        return "\n".join(parts)

    def _generate_basic_pillar_explanations(self, scores: Dict[str, float]) -> Dict[str, str]:
        """Generate basic pillar explanations without LLM"""
        explanations = {}

        for pillar, score in scores.items():
            if score > 70:
                explanations[pillar] = "Signal très positif"
            elif score > 50:
                explanations[pillar] = "Signal positif"
            elif score > 30:
                explanations[pillar] = "Signal neutre"
            else:
                explanations[pillar] = "Signal négatif"

        return explanations

    def _generate_basic_decision(self, scores: Dict[str, float]) -> str:
        """Generate basic decision reasoning without LLM"""
        avg = sum(scores.values()) / 4

        bullish = sum(1 for s in scores.values() if s > 50)
        bearish = sum(1 for s in scores.values() if s < 30)

        if bullish >= 3:
            return f"Convergence positive: {bullish}/4 piliers sont bullish. Score moyen: {avg:.0f}/100."
        elif bearish >= 2:
            return f"Signaux mixtes: {bearish}/4 piliers sont négatifs. Prudence recommandée."
        else:
            return f"Situation incertaine: les piliers ne convergent pas clairement. Score moyen: {avg:.0f}/100."


# Singleton instance
_generator_instance: Optional[NarrativeGenerator] = None


async def get_narrative_generator() -> NarrativeGenerator:
    """Get or create the NarrativeGenerator singleton"""
    global _generator_instance

    if _generator_instance is None:
        _generator_instance = NarrativeGenerator()
        await _generator_instance.initialize()

    return _generator_instance
