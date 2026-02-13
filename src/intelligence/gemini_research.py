"""
Gemini Research Module - Autonomous research capabilities for GeminiClient.

Adds:
1. Google Search Grounding - Gemini searches autonomously
2. Google Trends analysis via Gemini grounding
3. Contextual memory (gemini_memory.json)
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

MEMORY_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'gemini_memory.json')


class GeminiMemory:
    """Persistent memory for Gemini research - avoids redundant searches."""

    def __init__(self, path: str = MEMORY_PATH):
        self.path = path
        self.data = {"discoveries": [], "events": [], "lessons": [], "last_updated": None}
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, 'r') as f:
                    self.data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load gemini memory: {e}")

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'w') as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save gemini memory: {e}")

    def add_discoveries(self, discoveries: list):
        """Add new discoveries, keep last 50."""
        now = datetime.now().isoformat()
        for d in discoveries:
            d['timestamp'] = now
        self.data['discoveries'] = (discoveries + self.data.get('discoveries', []))[:50]
        self.data['last_updated'] = now
        self._save()

    def add_event(self, event: dict):
        """Track important event."""
        event['timestamp'] = datetime.now().isoformat()
        self.data.setdefault('events', []).insert(0, event)
        self.data['events'] = self.data['events'][:100]
        self._save()

    def add_lesson(self, lesson: str):
        """Store a lesson learned."""
        self.data.setdefault('lessons', []).append({
            'lesson': lesson, 'timestamp': datetime.now().isoformat()
        })
        self.data['lessons'] = self.data['lessons'][:30]
        self._save()

    def get_context_summary(self, max_chars: int = 2000) -> str:
        """Return memory summary for prompt injection."""
        parts = []
        # Recent discoveries (last 24h)
        recent = []
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        for d in self.data.get('discoveries', [])[:10]:
            if d.get('timestamp', '') > cutoff:
                recent.append(d)
        if recent:
            parts.append("RECENT DISCOVERIES (last 24h):")
            for d in recent[:5]:
                sym = d.get('symbol', '?')
                summary = d.get('summary', d.get('insight', ''))[:100]
                parts.append(f"  - [{sym}] {summary}")

        # Known upcoming events
        events = self.data.get('events', [])[:5]
        if events:
            parts.append("KNOWN EVENTS:")
            for e in events:
                parts.append(f"  - {e.get('summary', str(e))[:100]}")

        # Lessons
        lessons = self.data.get('lessons', [])[:5]
        if lessons:
            parts.append("LESSONS LEARNED:")
            for l in lessons:
                parts.append(f"  - {l.get('lesson', '')[:80]}")

        result = '\n'.join(parts)
        return result[:max_chars] if result else "No prior research memory."

    def cleanup_old(self, days: int = 7):
        """Remove entries older than N days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        for key in ['discoveries', 'events']:
            self.data[key] = [
                x for x in self.data.get(key, [])
                if x.get('timestamp', '') > cutoff
            ]
        self._save()


class GeminiResearchMixin:
    """Mixin that adds research capabilities to GeminiClient.
    
    Expects self._client, self.model_name, self.is_available(), self.initialize()
    """

    def _get_memory(self) -> GeminiMemory:
        if not hasattr(self, '_memory'):
            self._memory = GeminiMemory()
        return self._memory

    async def research(self, context: dict) -> dict:
        """
        Autonomous research using Google Search Grounding.
        
        Args:
            context: dict with keys like 'regime', 'symbols', 'vix', 'events', etc.
            
        Returns:
            dict with 'discoveries', 'symbol_adjustments', 'macro_insight', 'sources'
        """
        if not self.is_available():
            if not await self.initialize():
                return {"error": "Gemini not available", "discoveries": [], "symbol_adjustments": {}}

        memory = self._get_memory()
        memory_context = memory.get_context_summary()

        regime = context.get('regime', 'UNKNOWN')
        symbols = context.get('symbols', [])
        vix = context.get('vix', 'N/A')
        events = context.get('events', [])

        symbols_str = ', '.join(symbols) if symbols else 'aucun symbole spécifique'
        events_str = '\n'.join(f"  - {e}" for e in events[:10]) if events else '  Aucun événement récent connu'

        prompt = f"""Tu es l'analyste en chef du fonds Midas.

CONTEXTE MARCHÉ:
- Régime: {regime} | VIX: {vix}
- Symboles en focus: {symbols_str}
- Derniers événements connus:
{events_str}

MÉMOIRE CONTEXTUELLE:
{memory_context}

MISSION:
Utilise Google Search pour rechercher les informations les PLUS RÉCENTES et pertinentes pour ces symboles et le contexte de marché actuel.

Recherche ce que TU juges le plus important parmi :
1. News breaking ou événements récents pour ces symboles
2. Résultats financiers / earnings surprises récentes
3. Changements d'analystes (upgrades/dowgrades)
4. Tendances sectorielles émergentes
5. Risques macro (taux, géopolitique, régulation)
6. Rumeurs de M&A, restructurations, layoffs
7. Changements réglementaires impactant ces secteurs

NE RÉPÈTE PAS ce qui est déjà dans la mémoire contextuelle. Cherche du NOUVEAU.

Pour chaque découverte importante, structure ainsi :
- symbol: le ticker impacté (ou "MACRO" si général)
- summary: résumé en 1-2 phrases
- impact_score: score de -15 à +15 (négatif=bearish, positif=bullish)
- confidence: 0.0 à 1.0
- source: d'où vient l'info
- category: earnings|news|analyst|sector|macro|regulation|rumor

Réponds UNIQUEMENT en JSON valide :
{{
  "discoveries": [
    {{"symbol": "XXX", "summary": "...", "impact_score": 0, "confidence": 0.5, "source": "...", "category": "..."}}
  ],
  "symbol_adjustments": {{"SYMBOL": 0.0}},
  "macro_insight": "résumé macro en 1-2 phrases",
  "lessons": ["leçon apprise si pertinent"]
}}"""

        try:
            # Call Gemini with Google Search grounding
            loop = asyncio.get_event_loop()

            search_tool = types.Tool(google_search=types.GoogleSearch())

            contents = [
                types.Content(role="user", parts=[types.Part(text=prompt)])
            ]

            config = types.GenerateContentConfig(
                temperature=1.0,
                max_output_tokens=3000,
                tools=[search_tool]
            )

            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=config
                    )
                ),
                timeout=30.0
            )

            result = self._parse_research_response(response)

            # Update memory
            if result.get('discoveries'):
                memory.add_discoveries(result['discoveries'])
            for lesson in result.get('lessons', []):
                memory.add_lesson(lesson)

            logger.info(f"Gemini research: {len(result.get('discoveries', []))} discoveries, "
                       f"{len(result.get('symbol_adjustments', {}))} adjustments")
            return result

        except asyncio.TimeoutError:
            logger.warning("Gemini research timed out (30s)")
            return {"error": "timeout", "discoveries": [], "symbol_adjustments": {}}
        except Exception as e:
            logger.error(f"Gemini research failed: {e}")
            # Fallback: try without grounding
            return await self._research_fallback(context)

    def _parse_research_response(self, response) -> dict:
        """Parse research response (may contain grounding metadata + text)."""
        from src.intelligence.gemini_client import _clean_json

        text = ""
        if response and response.text:
            text = response.text

        if not text:
            return {"discoveries": [], "symbol_adjustments": {}, "macro_insight": "No response"}

        try:
            cleaned = _clean_json(text)
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse research JSON: {text[:300]}")
            result = {"discoveries": [], "symbol_adjustments": {}, "macro_insight": text[:200]}

        # Extract grounding sources if available
        sources = []
        try:
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                gm = getattr(candidate, 'grounding_metadata', None)
                if gm and hasattr(gm, 'grounding_chunks'):
                    for chunk in (gm.grounding_chunks or []):
                        if hasattr(chunk, 'web') and chunk.web:
                            sources.append({
                                'title': getattr(chunk.web, 'title', ''),
                                'uri': getattr(chunk.web, 'uri', '')
                            })
        except Exception:
            pass

        result['sources'] = sources
        return result

    async def _research_fallback(self, context: dict) -> dict:
        """Fallback research without grounding (uses Gemini's internal knowledge)."""
        logger.info("Using fallback research (no grounding)")
        try:
            symbols = context.get('symbols', [])
            prompt = (f"Based on your knowledge, what are the most important recent developments "
                     f"for these stocks: {', '.join(symbols)}? "
                     f"Market regime: {context.get('regime', 'UNKNOWN')}. "
                     f"Respond in JSON with 'discoveries' list and 'symbol_adjustments' dict.")

            from src.intelligence.gemini_client import _clean_json
            response = await self.chat(prompt, max_tokens=2000)
            if response:
                cleaned = _clean_json(response)
                return json.loads(cleaned)
        except Exception as e:
            logger.error(f"Fallback research also failed: {e}")

        return {"discoveries": [], "symbol_adjustments": {}, "macro_insight": "Research unavailable"}

    async def analyze_trends(self, symbols: list) -> dict:
        """
        Analyze search trends for symbols using Gemini with Google Search grounding.
        Detects abnormal search spikes that may indicate imminent events.
        """
        if not self.is_available():
            if not await self.initialize():
                return {"trends": {}, "alerts": []}

        symbols_str = ', '.join(symbols)
        prompt = f"""Analyse les tendances de recherche Google récentes pour ces symboles boursiers : {symbols_str}

Pour chaque symbole, recherche :
1. Y a-t-il un pic anormal de recherches récemment ?
2. Quels termes associés sont en tendance ? (ex: "AAPL earnings", "TSLA recall")
3. Le volume de recherche suggère-t-il un événement imminent ?

Un pic de recherche peut indiquer : earnings imminents, scandale, M&A, changement de direction, produit majeur...

Réponds en JSON :
{{
  "trends": {{
    "SYMBOL": {{
      "search_spike": true/false,
      "spike_magnitude": "low|medium|high",
      "trending_terms": ["term1", "term2"],
      "likely_catalyst": "explication courte",
      "alert_level": "none|watch|warning|critical"
    }}
  }},
  "alerts": ["alerte 1 si pertinent"]
}}"""

        try:
            loop = asyncio.get_event_loop()
            search_tool = types.Tool(google_search=types.GoogleSearch())

            config = types.GenerateContentConfig(
                temperature=1.0,
                max_output_tokens=2000,
                tools=[search_tool]
            )

            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._client.models.generate_content(
                        model=self.model_name,
                        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                        config=config
                    )
                ),
                timeout=30.0
            )

            if response and response.text:
                from src.intelligence.gemini_client import _clean_json
                cleaned = _clean_json(response.text)
                return json.loads(cleaned)

        except asyncio.TimeoutError:
            logger.warning("Gemini trends analysis timed out")
        except Exception as e:
            logger.error(f"Gemini trends analysis failed: {e}")

        return {"trends": {}, "alerts": []}
