"""
Intelligence Orchestrator - LLM-as-Conductor for market intelligence.

V8: Central brain that collects data from ALL existing sources (Grok Scanner,
News Fetcher, Trend Discovery, Market Context) and asks Gemini to REASON
about portfolio implications.

Live-only for the LLM reasoning part.
ETF momentum bonus is backtestable via calculate_etf_momentum_bonus().

Usage:
    orchestrator = IntelligenceOrchestrator(
        gemini_client=gemini,
        grok_scanner=grok,
        news_fetcher=news,
        trend_discovery=trends,
        market_context=market_ctx,
        portfolio_symbols=['AAPL', 'MSFT', ...]
    )
    brief = await orchestrator.get_brief()
    adj = orchestrator.get_symbol_adjustment('NVDA', brief)
"""

import json
import time
import asyncio
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────

@dataclass
class MarketEvent:
    """An event detected by the orchestrator."""
    event_type: str              # LLM-generated (not a fixed enum)
    summary: str                 # 1-sentence description
    affected_symbols: Dict[str, float]  # symbol -> adjustment (-15 to +15)
    confidence: float            # 0-1
    reasoning: str               # Chain of thought from LLM
    source: str                  # grok/news/trends
    ttl_hours: int               # Validity duration


@dataclass
class MegatrendSignal:
    """A megatrend detected by the orchestrator."""
    theme: str                   # LLM-generated (e.g. "AI infrastructure buildout")
    strength: str                # emerging/developing/established
    symbols: Dict[str, float]    # symbol -> bonus (0 to 10)
    etf_momentum: float          # Quantitative: ETF proxy vs SPY (backtestable)
    reasoning: str


@dataclass
class IntelligenceBrief:
    """Output of the LLM brain - produced every 15 minutes."""
    timestamp: datetime
    events: List[MarketEvent] = field(default_factory=list)
    megatrends: List[MegatrendSignal] = field(default_factory=list)
    macro_regime_bias: str = 'neutral'  # bullish / bearish / neutral
    portfolio_alerts: List[str] = field(default_factory=list)
    reasoning_summary: str = ''

    @property
    def is_empty(self):
        # type: () -> bool
        return not self.events and not self.megatrends


# ── ETF Momentum (backtestable) ──────────────────────────────────────────

# Theme ETF -> symbols mapping for backtestable momentum signal
THEME_ETF_PROXIES = {
    'SMH': ['NVDA', 'AMD', 'AVGO', 'INTC', 'QCOM', 'TXN', 'AMAT', 'LRCX'],
    'XBI': ['MRNA', 'GILD', 'AMGN'],
    'XLK': ['AAPL', 'MSFT', 'GOOGL', 'CRM', 'ADBE', 'INTU'],
    'XLF': ['JPM', 'GS', 'BAC', 'V', 'MA'],
    'XLE': ['XOM', 'CVX', 'COP'],
    'XLV': ['UNH', 'JNJ', 'PFE', 'ABBV', 'MRK'],
    'XLY': ['AMZN', 'TSLA', 'COST', 'NFLX', 'DIS'],
    'XLI': ['CAT', 'HON', 'UPS'],
    'XLP': ['PG', 'KO', 'PEP', 'WMT'],
}

# Reverse mapping: symbol -> ETF
_SYMBOL_TO_ETF = {}
for _etf, _syms in THEME_ETF_PROXIES.items():
    for _s in _syms:
        _SYMBOL_TO_ETF[_s] = _etf


def calculate_etf_momentum_bonus(symbol, date, data_mgr):
    # type: (str, pd.Timestamp, object) -> float
    """Backtestable signal: bonus if the sector/theme ETF outperforms SPY.

    ETF vs SPY 60-day relative performance:
    - Outperformance > 5% -> +3 pts
    - Outperformance > 10% -> +5 pts
    - Outperformance > 15% -> +8 pts

    Bonus only (never negative), cap at +8.

    Args:
        symbol: Stock symbol
        date: Current backtest date
        data_mgr: BacktestDataManager instance (has get_ohlcv method)

    Returns:
        Bonus points (0 to 8)
    """
    etf = _SYMBOL_TO_ETF.get(symbol)
    if etf is None:
        return 0.0

    # Get ETF and SPY data from cache
    etf_df = None
    spy_df = None
    for key, df in data_mgr._cache.items():
        if key.startswith(etf + '_'):
            etf_df = df
        if key.startswith('SPY_'):
            spy_df = df

    if etf_df is None or spy_df is None:
        return 0.0

    if date not in etf_df.index or date not in spy_df.index:
        return 0.0

    etf_loc = etf_df.index.get_loc(date)
    spy_loc = spy_df.index.get_loc(date)

    if etf_loc < 60 or spy_loc < 60:
        return 0.0

    etf_now = float(etf_df['Close'].iloc[etf_loc])
    etf_past = float(etf_df['Close'].iloc[etf_loc - 60])
    spy_now = float(spy_df['Close'].iloc[spy_loc])
    spy_past = float(spy_df['Close'].iloc[spy_loc - 60])

    if etf_past <= 0 or spy_past <= 0:
        return 0.0

    etf_ret = (etf_now - etf_past) / etf_past
    spy_ret = (spy_now - spy_past) / spy_past
    outperf = etf_ret - spy_ret

    if outperf > 0.15:
        return 8.0
    elif outperf > 0.10:
        return 5.0
    elif outperf > 0.05:
        return 3.0
    return 0.0


# ── LLM System Prompt ────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM_PROMPT = """
Tu es l'analyste en chef d'un fonds d'investissement algorithmique.
Tu recois des donnees brutes de multiples sources (news, reseaux sociaux,
donnees de marche, tendances). Ton role est de RAISONNER sur les implications
concretes pour notre portefeuille.

REGLES :
1. Identifie les evenements MAJEURS qui impactent les marches (geopolitique,
   macro-economie, sectoriel, reglementaire)
2. Pour chaque evenement, raisonne sur QUELS secteurs et symboles sont impactes,
   DANS QUEL SENS, et AVEC QUELLE MAGNITUDE (-15 a +15 points)
3. Identifie les MEGATRENDS emergents (themes qui durent des mois/annees)
4. Donne un biais macro (bullish/bearish/neutral) base sur le contexte global
5. NE reponds QUE sur les symboles de notre portefeuille : {symbols}
6. Sois CONSERVATEUR : en cas de doute, adjustment = 0
7. Les ajustements sont des OVERLAYS sur un score technique existant (0-100)
   Ils doivent etre proportionnels a la conviction

IMPORTANT : Raisonne etape par etape. Explique POURQUOI chaque ajustement.

Reponds en JSON strict :
{{
  "events": [{{"event_type": "str", "summary": "str",
              "affected_symbols": {{"SYM": 0.0}}, "confidence": 0.0,
              "reasoning": "str", "ttl_hours": 24}}],
  "megatrends": [{{"theme": "str", "strength": "emerging|developing|established",
                   "symbols": {{"SYM": 0.0}}, "reasoning": "str"}}],
  "macro_regime_bias": "bullish|bearish|neutral",
  "portfolio_alerts": ["str"],
  "reasoning_summary": "str"
}}
"""


# ── Intelligence Orchestrator ────────────────────────────────────────────

SYMBOL_INTEL_PROMPT = """
Tu es un analyste financier. On te donne des articles de news recents concernant
un symbole boursier specifique. Analyse l'impact potentiel sur le cours.

Symbole: {symbol}
Company: {company_name}

Articles:
{articles}

Brief macro actuel: {macro_bias}

Reponds en JSON strict:
{{
  "adjustment": 0.0,
  "reasoning": "explication courte",
  "sentiment": "positive|negative|neutral|mixed"
}}

Regles:
- adjustment entre -10 et +10 (conservateur)
- 0 si pas d'impact clair
- Ne specule pas, base-toi uniquement sur les faits des articles
"""


class IntelligenceOrchestrator:
    """
    LLM conductor. Collects data from ALL existing sources, then asks
    Gemini to REASON about portfolio implications.

    Cycle: every 15 minutes (cached).
    Cost: 2-4 Gemini Flash calls/cycle = ~10/hour = within free tier.
    """

    CACHE_TTL_SECONDS = 900  # 15 minutes

    def __init__(self, gemini_client, grok_scanner=None, news_fetcher=None,
                 trend_discovery=None, market_context=None,
                 portfolio_symbols=None):
        self.gemini = gemini_client
        self.grok = grok_scanner
        self.news = news_fetcher
        self.trends = trend_discovery
        self.market = market_context
        self.symbols = portfolio_symbols or []
        self._cache = None  # type: Optional[IntelligenceBrief]
        self._symbol_intel_cache = {}  # type: Dict[str, dict]  # symbol -> {result, time}
        self._cache_time = None  # type: Optional[datetime]

    async def get_brief(self, force=False):
        # type: (bool) -> IntelligenceBrief
        """Main entry point. Returns cached brief (15 min TTL)."""
        now = datetime.now()
        if (not force and self._cache is not None and self._cache_time is not None
                and (now - self._cache_time).total_seconds() < self.CACHE_TTL_SECONDS):
            return self._cache

        try:
            brief = await self._generate_brief()
        except Exception as e:
            logger.error("Intelligence brief generation failed: %s", e)
            brief = IntelligenceBrief(timestamp=now)

        self._cache = brief
        self._cache_time = now
        return brief

    async def _generate_brief(self):
        # type: () -> IntelligenceBrief
        """Collect intelligence from all sources, then ask LLM to reason."""
        raw = await self._collect_intelligence()
        brief = await self._llm_reasoning(raw)
        return brief

    async def _collect_intelligence(self):
        # type: () -> Dict
        """Parallel collection from all existing sources."""
        results = {}  # type: Dict
        tasks = []
        task_names = []

        if self.grok is not None:
            tasks.append(self._safe_call(self.grok.full_scan_with_analysis))
            task_names.append('grok')

        if self.news is not None:
            tasks.append(self._safe_call(self.news.fetch_latest))
            task_names.append('news')

        if self.trends is not None:
            tasks.append(self._safe_call(self.trends.daily_scan))
            task_names.append('trends')

        if self.market is not None:
            tasks.append(self._safe_call(self.market.get_context))
            task_names.append('market_context')

        # Gemini autonomous research (with Google Search grounding)
        if self.gemini is not None and self.gemini.is_available() and hasattr(self.gemini, 'research'):
            research_context = {
                'regime': 'UNKNOWN',
                'symbols': self.symbols[:20],  # Batch, not per-symbol
                'vix': None,
                'events': []
            }
            # Try to get regime/vix from market context if available
            if self.market is not None:
                try:
                    mkt = self.market
                    if hasattr(mkt, 'regime'):
                        research_context['regime'] = str(mkt.regime)
                    if hasattr(mkt, 'vix'):
                        research_context['vix'] = mkt.vix
                except Exception:
                    pass
            tasks.append(self._safe_call(lambda: self.gemini.research(research_context)))
            task_names.append('gemini_research')

        if tasks:
            gathered = await asyncio.gather(*tasks)
            for name, result in zip(task_names, gathered):
                results[name] = result

        return results

    @staticmethod
    async def _safe_call(coro_func):
        """Call an async function, return None on error."""
        try:
            return await coro_func()
        except Exception as e:
            logger.warning("Intelligence source failed: %s", e)
            return None

    async def _llm_reasoning(self, raw_data):
        # type: (Dict) -> IntelligenceBrief
        """THE CORE: Gemini reasons about the raw data."""
        if self.gemini is None or not self.gemini.is_available():
            logger.warning("Gemini not available - returning empty brief")
            return IntelligenceBrief(timestamp=datetime.now())

        prompt = self._build_reasoning_prompt(raw_data)
        system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(
            symbols=', '.join(self.symbols)
        )

        try:
            response = await asyncio.wait_for(
                self.gemini.chat_json(
                    prompt,
                    system_prompt=system_prompt,
                    max_tokens=3000
                ),
                timeout=60
            )
        except asyncio.TimeoutError:
            logger.warning('Gemini reasoning timed out (60s)')
            return IntelligenceBrief(timestamp=datetime.now())

        return self._parse_brief(response)

    def _build_reasoning_prompt(self, raw_data):
        # type: (Dict) -> str
        """Build structured prompt from all raw intelligence sources."""
        sections = []

        # Market context
        mkt = raw_data.get('market_context')
        if mkt is not None:
            try:
                summary = mkt.get_summary() if hasattr(mkt, 'get_summary') else str(mkt)
                sections.append("=== MARKET CONTEXT ===\n%s" % summary)
            except Exception:
                pass

        # Grok (X/Twitter)
        grok = raw_data.get('grok')
        if grok is not None:
            try:
                if hasattr(grok, 'insights'):
                    insights_text = '\n'.join(
                        "- %s" % str(i) for i in (grok.insights or [])[:10]
                    )
                    sections.append("=== X/TWITTER BUZZ (Grok) ===\n%s" % insights_text)
                elif isinstance(grok, dict):
                    sections.append("=== X/TWITTER BUZZ (Grok) ===\n%s" % json.dumps(grok, default=str)[:2000])
            except Exception:
                pass

        # News
        news = raw_data.get('news')
        if news is not None:
            try:
                if isinstance(news, list):
                    headlines = '\n'.join(
                        "- [%s] %s" % (getattr(a, 'source', '?'), getattr(a, 'title', str(a)))
                        for a in news[:15]
                    )
                    sections.append("=== TOP NEWS HEADLINES ===\n%s" % headlines)
                elif isinstance(news, dict):
                    sections.append("=== NEWS ===\n%s" % json.dumps(news, default=str)[:2000])
            except Exception:
                pass

        # Trends
        trends = raw_data.get('trends')
        if trends is not None:
            try:
                if hasattr(trends, 'trends'):
                    trend_text = '\n'.join(
                        "- %s (%s, confidence=%.2f)" % (
                            getattr(t, 'name', '?'),
                            getattr(t, 'strength', '?'),
                            getattr(t, 'confidence', 0)
                        )
                        for t in (trends.trends or [])[:10]
                    )
                    sections.append("=== DETECTED TRENDS ===\n%s" % trend_text)
                elif isinstance(trends, dict):
                    sections.append("=== TRENDS ===\n%s" % json.dumps(trends, default=str)[:2000])
            except Exception:
                pass

        # Gemini autonomous research results
        gemini_research = raw_data.get('gemini_research')
        if gemini_research is not None:
            try:
                if isinstance(gemini_research, dict):
                    discoveries = gemini_research.get('discoveries', [])
                    if discoveries:
                        disc_text = '\n'.join(
                            "- [%s] %s (impact: %s, confidence: %s)" % (
                                d.get('symbol', '?'), d.get('summary', ''),
                                d.get('impact_score', 0), d.get('confidence', 0)
                            )
                            for d in discoveries[:10]
                        )
                        sections.append("=== GEMINI AUTONOMOUS RESEARCH (Google Search) ===\n%s" % disc_text)
                    macro = gemini_research.get('macro_insight')
                    if macro:
                        sections.append("Macro insight: %s" % macro)
            except Exception:
                pass

        # RSS news (if available via news dict)
        rss = raw_data.get('news', {})
        if isinstance(rss, dict):
            rss_articles = rss.get('rss', [])
            if rss_articles:
                rss_text = chr(10).join(
                    '- [%s] %s' % (getattr(a, 'source', '?'), getattr(a, 'title', str(a)))
                    for a in rss_articles[:20]
                )
                sections.append("=== RSS NEWS FEEDS ===\n%s" % rss_text)

        if not sections:
            sections.append("No intelligence data available at this time.")

        sections.append(
            "\n=== PORTFOLIO SYMBOLS ===\n%s" % ', '.join(self.symbols)
        )
        sections.append(
            "\nAnalyze the above data. Identify events, megatrends, and macro bias. "
            "Return JSON as specified."
        )

        return '\n\n'.join(sections)

    def _parse_brief(self, data):
        # type: (Dict) -> IntelligenceBrief
        """Parse LLM JSON response into IntelligenceBrief."""
        now = datetime.now()
        if not data:
            return IntelligenceBrief(timestamp=now)

        events = []
        for e in data.get('events', []):
            if not isinstance(e, dict):
                continue
            # Filter affected_symbols to only portfolio symbols, cap adjustments
            raw_syms = e.get('affected_symbols', {})
            if not isinstance(raw_syms, dict):
                raw_syms = {}
            filtered = {}
            for sym, adj in raw_syms.items():
                if sym in self.symbols:
                    try:
                        filtered[sym] = max(-15.0, min(15.0, float(adj)))
                    except (ValueError, TypeError):
                        pass

            events.append(MarketEvent(
                event_type=str(e.get('event_type', 'unknown')),
                summary=str(e.get('summary', '')),
                affected_symbols=filtered,
                confidence=min(1.0, max(0.0, float(e.get('confidence', 0.5)))),
                reasoning=str(e.get('reasoning', '')),
                source=str(e.get('source', 'llm')),
                ttl_hours=int(e.get('ttl_hours', 24))
            ))

        megatrends = []
        for t in data.get('megatrends', []):
            if not isinstance(t, dict):
                continue
            raw_syms = t.get('symbols', {})
            if not isinstance(raw_syms, dict):
                raw_syms = {}
            filtered = {}
            for sym, bonus in raw_syms.items():
                if sym in self.symbols:
                    try:
                        filtered[sym] = max(0.0, min(10.0, float(bonus)))
                    except (ValueError, TypeError):
                        pass

            megatrends.append(MegatrendSignal(
                theme=str(t.get('theme', '')),
                strength=str(t.get('strength', 'emerging')),
                symbols=filtered,
                etf_momentum=0.0,  # Filled in backtest path
                reasoning=str(t.get('reasoning', ''))
            ))

        bias = str(data.get('macro_regime_bias', 'neutral')).lower()
        if bias not in ('bullish', 'bearish', 'neutral'):
            bias = 'neutral'

        alerts = []
        for a in data.get('portfolio_alerts', []):
            alerts.append(str(a))

        return IntelligenceBrief(
            timestamp=now,
            events=events,
            megatrends=megatrends,
            macro_regime_bias=bias,
            portfolio_alerts=alerts[:10],
            reasoning_summary=str(data.get('reasoning_summary', ''))
        )

    async def get_symbol_intelligence(self, symbol: str, company_name: str = None) -> dict:
        """Get intelligence specific to a symbol during screening.
        Returns: {'adjustment': float, 'reasoning': str, 'news': list, 'sentiment': str}
        """
        now = time.time()
        # Check cache (30 min TTL)
        cached = self._symbol_intel_cache.get(symbol)
        if cached and (now - cached['time']) < 1800:
            return cached['result']

        result = {'adjustment': 0.0, 'reasoning': '', 'news': [], 'sentiment': 'neutral'}

        # Fetch symbol-specific news
        if self.news is None:
            self._symbol_intel_cache[symbol] = {'result': result, 'time': now}
            return result

        try:
            articles = await self.news.fetch_symbol_news(symbol, company_name)
        except Exception as e:
            logger.debug(f"Symbol news fetch failed for {symbol}: {e}")
            self._symbol_intel_cache[symbol] = {'result': result, 'time': now}
            return result

        if not articles:
            self._symbol_intel_cache[symbol] = {'result': result, 'time': now}
            return result

        result['news'] = [a.title for a in articles[:10]]

        # If Gemini available, ask for analysis
        if self.gemini is not None and self.gemini.is_available():
            try:
                articles_text = "\n".join(
                    f"- [{a.source}] {a.title}" + (f": {a.summary[:150]}" if a.summary else "")
                    for a in articles[:10]
                )
                macro_bias = 'neutral'
                if self._cache:
                    macro_bias = self._cache.macro_regime_bias

                prompt = SYMBOL_INTEL_PROMPT.format(
                    symbol=symbol,
                    company_name=company_name or symbol,
                    articles=articles_text,
                    macro_bias=macro_bias,
                )

                response = await asyncio.wait_for(
                    self.gemini.chat_json(prompt, max_tokens=500),
                    timeout=30
                )

                if isinstance(response, dict):
                    adj = float(response.get('adjustment', 0))
                    result['adjustment'] = max(-10.0, min(10.0, adj))
                    result['reasoning'] = str(response.get('reasoning', ''))
                    result['sentiment'] = str(response.get('sentiment', 'neutral'))
                    logger.info(f"🔍 SYMBOL INTEL {symbol}: adj={result['adjustment']:+.1f} sentiment={result['sentiment']} ({len(articles)} articles)")
            except asyncio.TimeoutError:
                logger.warning(f"Gemini timeout for symbol intel {symbol}")
            except Exception as e:
                logger.debug(f"Gemini symbol intel failed for {symbol}: {e}")
        else:
            # No Gemini - just note we found news
            result['reasoning'] = f"Found {len(articles)} relevant articles (no LLM analysis)"

        self._symbol_intel_cache[symbol] = {'result': result, 'time': now}
        return result

    def get_symbol_adjustment(self, symbol, brief):
        # type: (str, IntelligenceBrief) -> float
        """Return total score adjustment for a symbol (capped +/- 15)."""
        adj = 0.0
        for event in brief.events:
            adj += event.affected_symbols.get(symbol, 0.0)
        for trend in brief.megatrends:
            adj += trend.symbols.get(symbol, 0.0)
        return max(-15.0, min(15.0, adj))
