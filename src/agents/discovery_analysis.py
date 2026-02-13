"""
Discovery & Analysis Coordinator - Extracted from orchestrator.py (V8.2 Sprint 4)

Handles:
- Discovery Phase (06:00): Social scanning, Grok, Intuitive Reasoning, FMP screening
- Analysis Phase (07:00): Trend analysis, regime detection, focus symbol selection
- FMP Pre-screening and IBKR validation
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.orchestrator import MarketAgent

from src.agents.state import AgentPhase, MarketRegime

logger = logging.getLogger(__name__)


class DiscoveryAnalysisCoordinator:
    """Coordinates discovery and analysis phases for the MarketAgent."""

    def __init__(self, agent: 'MarketAgent'):
        self.agent = agent

    @property
    def config(self):
        return self.agent.config

    @property
    def state_manager(self):
        return self.agent.state_manager

    # =========================================================================
    # HYBRID SCREENING (FMP + IBKR)
    # =========================================================================

    async def run_fmp_prescreening(self) -> List[str]:
        """
        Run FMP pre-screening to get candidates before IBKR validation.
        This avoids IBKR pacing violations by pre-filtering with FMP.
        """
        if not self.agent.hybrid_screener:
            logger.warning("HybridScreener not available - using existing watchlist")
            return self.state_manager.state.watchlist[:100]

        try:
            logger.info("=== FMP PRE-SCREENING STARTED ===")
            candidates = await self.agent.hybrid_screener.run_prescreening()

            if not candidates:
                logger.warning("No candidates from FMP pre-screening")
                return []

            symbols = [c.symbol for c in candidates]

            existing = set(self.state_manager.state.watchlist)
            new_symbols = [s for s in symbols if s not in existing]
            if new_symbols:
                self.state_manager.state.watchlist.extend(
                    new_symbols[:self.config.max_watchlist_size]
                )
                logger.info(f"Added {len(new_symbols)} new symbols to watchlist from FMP")

            self.state_manager.state.focus_symbols = symbols[:self.config.max_focus_symbols]
            logger.info(f"FMP pre-screening complete: {len(symbols)} candidates, {len(new_symbols)} new")
            return symbols

        except Exception as e:
            logger.error(f"FMP pre-screening error: {e}")
            return []

    async def validate_with_ibkr(self, symbols: List[str]) -> List[str]:
        """Validate FMP candidates with IBKR official data."""
        if not self.agent.hybrid_screener:
            logger.warning("HybridScreener not available - returning unvalidated symbols")
            return symbols

        if not self.agent.hybrid_screener.ibkr:
            logger.warning("IBKR not connected - skipping validation")
            return symbols

        try:
            logger.info(f"=== IBKR VALIDATION: {len(symbols)} candidates ===")
            from src.screening.hybrid_screener import ScreeningCandidate
            candidates = [
                ScreeningCandidate(
                    symbol=s, signal='BUY', rsi_at_breakout=0,
                    strength='UNKNOWN', source='FMP'
                )
                for s in symbols
            ]

            validated = await self.agent.hybrid_screener.run_ibkr_validation(candidates)
            validated_symbols = [c.symbol for c in validated]
            logger.info(f"IBKR validation complete: {len(validated_symbols)}/{len(symbols)} validated")
            return validated_symbols

        except Exception as e:
            logger.error(f"IBKR validation error: {e}")
            return symbols

    # =========================================================================
    # PHASE: DISCOVERY (06:00)
    # =========================================================================

    async def run_discovery_phase(self) -> Dict:
        """
        Phase de decouverte matinale.
        Scanne les reseaux sociaux et les news pour identifier les opportunites.
        """
        logger.info("=== DISCOVERY PHASE STARTED ===")
        self.state_manager.set_phase(AgentPhase.DISCOVERY)

        results = {
            "timestamp": datetime.now().isoformat(),
            "social_trending": [],
            "grok_insights": [],
            "discovered_stocks": [],
            "volume_anomalies": [],
            "watchlist": []
        }

        # 1. Scanner Grok (X/Twitter via xAI)
        if self.agent.grok_scanner:
            try:
                grok_insights = await self.agent.grok_scanner.search_financial_trends()
                results["grok_insights"] = []
                for insight in grok_insights:
                    if hasattr(insight, 'to_dict'):
                        results["grok_insights"].append(insight.to_dict())
                    if hasattr(insight, 'symbols') and insight.symbols:
                        results["social_trending"].extend(insight.symbols)
                logger.info(f"Grok scan: {len(results['grok_insights'])} insights")
            except Exception as e:
                logger.error(f"Grok scan error: {e}")
                self.state_manager.log_error(f"Grok scan failed: {e}")

        # 2. Intuitive Reasoning: News generales -> Secteurs -> Actions
        news_derived_symbols = []
        if self.agent.intuitive_reasoning:
            try:
                logger.info("Running IntuitiveReasoning: Analyzing macro news for sector implications...")
                implications = await self.agent.intuitive_reasoning.daily_news_scan()
                buy_recs = self.agent.intuitive_reasoning.get_buy_recommendations(min_confidence=0.5)
                news_derived_symbols = self.agent.intuitive_reasoning.get_symbols_to_watch()

                results["intuitive_reasoning"] = {
                    "implications_count": len(implications),
                    "buy_recommendations": len(buy_recs),
                    "symbols_derived": news_derived_symbols,
                    "top_sectors": list(set(impl.sector for impl in implications))[:5] if implications else []
                }

                for rec in buy_recs[:3]:
                    logger.info(f"[INTUITIVE] BUY Signal: {rec['sector']} -> {rec['symbols'][:3]} (conf={rec['confidence']:.0%})")
                    logger.info(f"  Catalyst: {rec['catalyst'][:80]}...")

                logger.info(f"IntuitiveReasoning: {len(implications)} implications, {len(news_derived_symbols)} symbols derived")
            except Exception as e:
                logger.error(f"IntuitiveReasoning error: {e}")
                self.state_manager.log_error(f"IntuitiveReasoning failed: {e}")

        # 3. Decouverte dynamique de stocks (volume/momentum)
        if self.agent.stock_discovery:
            try:
                discovered = await self.agent.stock_discovery.discover_trending()
                results["discovered_stocks"] = discovered
                logger.info(f"Stock discovery: {len(discovered)} new symbols")
            except Exception as e:
                logger.error(f"Stock discovery error: {e}")
                self.state_manager.log_error(f"Stock discovery failed: {e}")

        # 4. Scanner les anomalies de volume
        if self.agent.trend_discovery:
            try:
                anomalies = await self.agent.trend_discovery.detect_volume_anomalies()
                results["volume_anomalies"] = [s for s, _ in anomalies[:20]]
                logger.info(f"Volume anomalies: {len(results['volume_anomalies'])} detected")
            except Exception as e:
                logger.error(f"Volume anomaly detection error: {e}")

        # 5. FMP Pre-screening (RSI breakout candidates)
        fmp_candidates = []
        if self.agent.hybrid_screener:
            try:
                logger.info("Running FMP pre-screening for RSI breakout candidates...")
                fmp_candidates = await self.run_fmp_prescreening()
                results["fmp_candidates"] = fmp_candidates
                logger.info(f"FMP pre-screening: {len(fmp_candidates)} RSI breakout candidates")
            except Exception as e:
                logger.error(f"FMP pre-screening error: {e}")
                self.state_manager.log_error(f"FMP pre-screening failed: {e}")

        # 6. Construire la watchlist dynamique
        all_symbols = set()
        all_symbols.update(results["social_trending"])
        all_symbols.update(results["discovered_stocks"])
        all_symbols.update(results["volume_anomalies"])

        if news_derived_symbols:
            all_symbols.update(news_derived_symbols)
            logger.info(f"Added {len(news_derived_symbols)} symbols from macro news analysis")

        all_symbols.update(fmp_candidates)

        results["watchlist"] = list(all_symbols)[:self.config.max_watchlist_size]

        # Set focus symbols prioritizing FMP candidates
        if fmp_candidates:
            focus = fmp_candidates[:self.config.max_focus_symbols]
            remaining_slots = self.config.max_focus_symbols - len(focus)
            if remaining_slots > 0:
                other_symbols = [s for s in results["watchlist"] if s not in focus]
                focus.extend(other_symbols[:remaining_slots])
        else:
            focus = list(all_symbols)[:self.config.max_focus_symbols]
            logger.warning(f"FMP unavailable - using {len(focus)} symbols from social sources")

        self.state_manager.state.focus_symbols = focus
        results["focus_symbols"] = focus

        self.state_manager.update_watchlist(results["watchlist"])
        self.state_manager.update_last_scan()

        # Auto-generate report
        try:
            import subprocess
            import json as json_lib
            logger.info("Generating market analysis report...")
            report_data_path = "/app/data/latest_scan.json"
            with open(report_data_path, "w") as f:
                json_lib.dump(results, f, indent=2, default=str)
            proc = subprocess.run(
                ["python", "/app/src/intelligence/report_generator.py"],
                capture_output=True, text=True, timeout=120
            )
            if proc.returncode == 0:
                logger.info("Market analysis report generated successfully")
            else:
                logger.warning(f"Report generation warning: {proc.stderr[:200]}")
        except Exception as e:
            logger.error(f"Report generation error: {e}")

        logger.info(f"=== DISCOVERY COMPLETE: {len(results['watchlist'])} symbols in watchlist, "
                    f"{len(fmp_candidates)} FMP candidates ===")
        return results

    # =========================================================================
    # PHASE: ANALYSIS (07:00)
    # =========================================================================

    async def run_analysis_phase(self) -> Dict:
        """
        Phase d'analyse des tendances.
        Utilise le LLM pour identifier les narratifs et prioriser les symboles.
        """
        logger.info("=== ANALYSIS PHASE STARTED ===")
        self.state_manager.set_phase(AgentPhase.ANALYSIS)

        results = {
            "timestamp": datetime.now().isoformat(),
            "trends": [],
            "narratives": [],
            "market_sentiment": 0.0,
            "focus_symbols": [],
            "market_regime": MarketRegime.UNKNOWN.value
        }

        if self.agent.trend_discovery:
            try:
                report = await self.agent.trend_discovery.daily_scan()
                results["trends"] = [t.to_dict() for t in report.trends]
                results["narratives"] = report.narrative_updates
                results["market_sentiment"] = report.market_sentiment

                focus = report.watchlist_additions[:self.config.max_focus_symbols]
                results["focus_symbols"] = focus

                if report.market_sentiment > 0.3:
                    regime = MarketRegime.BULL_STRONG
                    confidence = min(0.9, abs(report.market_sentiment))
                elif report.market_sentiment > 0.1:
                    regime = MarketRegime.BULL_WEAK
                    confidence = 0.6
                elif report.market_sentiment < -0.3:
                    regime = MarketRegime.BEAR_STRONG
                    confidence = min(0.9, abs(report.market_sentiment))
                elif report.market_sentiment < -0.1:
                    regime = MarketRegime.BEAR_WEAK
                    confidence = 0.6
                else:
                    regime = MarketRegime.RANGING
                    confidence = 0.5

                results["market_regime"] = regime.value
                self.state_manager.update_market_regime(regime, confidence)

                logger.info(f"Trend analysis: {len(results['trends'])} trends, "
                           f"sentiment={results['market_sentiment']:.2f}, regime={regime.value}")

            except Exception as e:
                logger.error(f"Trend analysis error: {e}")
                self.state_manager.log_error(f"Trend analysis failed: {e}")

        self.state_manager.update_focus_symbols(results["focus_symbols"])
        self.state_manager.update_trends(results["trends"])
        self.state_manager.update_narratives([n.get("name", "") for n in results["narratives"]])

        logger.info(f"=== ANALYSIS COMPLETE: {len(results['focus_symbols'])} focus symbols ===")
        return results
