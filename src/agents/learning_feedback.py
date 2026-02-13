"""
Learning & Feedback Coordinator - Extracted from orchestrator.py (V8.2 Sprint 4)

Handles:
- Audit Phase (20:00): Trade analysis, memory storage, pattern extraction
- Adaptive Learning: Weight updates from trade outcomes
- Shadow Tracking: Autonomous signal verification
- Strategy Composition: LLM-based strategy improvement
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.orchestrator import MarketAgent

from src.agents.state import AgentPhase

logger = logging.getLogger(__name__)


class LearningFeedbackCoordinator:
    """Coordinates audit, learning, and feedback for the MarketAgent."""

    def __init__(self, agent: 'MarketAgent'):
        self.agent = agent

    @property
    def config(self):
        return self.agent.config

    @property
    def state_manager(self):
        return self.agent.state_manager

    # =========================================================================
    # PHASE: AUDIT (20:00)
    # =========================================================================

    async def run_audit_phase(self) -> Dict:
        """
        Phase d'audit nocturne.
        Analyse les trades du jour, stocke dans TradeMemory, et genere des regles apprises.
        """
        logger.info("=== AUDIT PHASE STARTED ===")
        self.state_manager.set_phase(AgentPhase.AUDIT)

        results = {
            "timestamp": datetime.now().isoformat(),
            "trades_analyzed": 0,
            "trades_stored": 0,
            "patterns_extracted": 0,
            "new_rules_generated": 0,
            "rules": [],
            "memory_cleanup": {},
            "summary": {}
        }

        # 1. Get closed trades from today
        closed_trades = self.state_manager.get_closed_trades_today()
        results["trades_analyzed"] = len(closed_trades)
        logger.info(f"Analyzing {len(closed_trades)} closed trades from today")

        # 2. Store in TradeMemory (RAG)
        if self.agent.trade_memory and closed_trades:
            results["trades_stored"] = await self._store_trades_in_memory(closed_trades)

        # 3. Update adaptive learning
        if self.agent.adaptive_scorer and closed_trades:
            self._update_adaptive_learning(closed_trades, results)

        # 4. Extract patterns
        if self.agent.pattern_extractor and self.agent.trade_memory:
            await self._extract_patterns(results)

        # 5. Memory cleanup
        if self.agent.trade_memory:
            await self._cleanup_memory(results)

        # 6. Shadow tracker verification
        if self.agent.shadow_tracker:
            await self._run_shadow_audit(results)

        # 7. LLM strategy analysis for losing trades
        results["strategy_improvements"] = []
        if closed_trades:
            losing_trades = [t for t in closed_trades if t.pnl_pct < 0]
            if self.agent.strategy_composer and len(losing_trades) >= 2:
                await self._analyze_losing_trades(losing_trades, closed_trades, results)

        # 8. Generate summary
        if results["trades_analyzed"] > 0 and closed_trades:
            wins = sum(1 for t in closed_trades if t.pnl_pct > 0)
            total_pnl = sum(t.pnl_pct for t in closed_trades)
            results["summary"] = {
                "trades_today": results["trades_analyzed"],
                "wins": wins,
                "losses": results["trades_analyzed"] - wins,
                "win_rate": wins / results["trades_analyzed"] if results["trades_analyzed"] > 0 else 0,
                "total_pnl_pct": total_pnl,
                "avg_pnl_pct": total_pnl / results["trades_analyzed"]
            }

        self.state_manager.update_last_audit()

        logger.info(f"=== AUDIT COMPLETE: {results['trades_stored']} stored, "
                   f"{results['new_rules_generated']} new rules ===")
        return results

    # =========================================================================
    # TRADE MEMORY
    # =========================================================================

    async def _store_trades_in_memory(self, closed_trades: List) -> int:
        """Store closed trades in TradeMemory (RAG)."""
        from src.intelligence.trade_memory import TradeContext, TradeOutcome

        stored = 0
        for trade in closed_trades:
            try:
                context = TradeContext(
                    symbol=trade.symbol,
                    timestamp=trade.entry_date,
                    market_regime=trade.metadata.get("market_regime", "unknown"),
                    vix_level=trade.metadata.get("vix_level", 20.0),
                    spy_trend=trade.metadata.get("spy_trend", "sideways"),
                    sector_momentum=trade.metadata.get("sector_momentum", 0.0),
                    rsi=trade.metadata.get("rsi", 50),
                    ema_alignment=trade.metadata.get("ema_alignment", "neutral"),
                    volume_ratio=trade.metadata.get("volume_ratio", 1.0),
                    breakout_strength=trade.metadata.get("breakout_strength"),
                    social_sentiment=trade.metadata.get("social_sentiment", 0),
                    grok_sentiment=trade.metadata.get("grok_sentiment", 0),
                    stocktwits_sentiment=trade.metadata.get("stocktwits_sentiment", 0),
                    heat_score=trade.metadata.get("heat_score", 0),
                    news_sentiment=trade.metadata.get("news_sentiment", 0),
                    catalyst_present=trade.metadata.get("catalyst_present", False),
                    decision_score=trade.metadata.get("decision_score", 0),
                    confidence=trade.metadata.get("confidence", 0)
                )

                outcome = TradeOutcome(
                    pnl_pct=trade.pnl_pct,
                    hold_days=trade.hold_days,
                    exit_type=trade.exit_type,
                    max_drawdown=trade.metadata.get("max_drawdown", 0),
                    max_gain=trade.metadata.get("max_gain", 0)
                )

                if self.agent.trade_memory.store_trade(
                    trade_id=trade.id, context=context, outcome=outcome
                ):
                    stored += 1

            except Exception as e:
                logger.warning(f"Failed to store trade {trade.id} in memory: {e}")

        logger.info(f"Stored {stored} trades in memory")
        return stored

    # =========================================================================
    # ADAPTIVE LEARNING
    # =========================================================================

    def _update_adaptive_learning(self, closed_trades: List, results: Dict):
        """Update adaptive learning with closed trades."""
        adaptive_recorded = 0
        for trade in closed_trades:
            try:
                trade_id_prefix = f"{trade.symbol}_"
                self.agent.adaptive_scorer.record_exit(
                    trade_id=trade_id_prefix + trade.entry_date.replace('-', '').replace(':', '').replace('T', '')[:14],
                    exit_price=trade.exit_price or trade.entry_price,
                    exit_reason=trade.exit_type or "unknown"
                )
                adaptive_recorded += 1
            except Exception as e:
                logger.debug(f"[V5] Could not record exit for {trade.symbol}: {e}")

        if adaptive_recorded > 0:
            logger.info(f"[V5] Recorded {adaptive_recorded} trade exits for adaptive learning")

        weights = self.agent.adaptive_scorer.weights.to_dict()
        results["adaptive_weights"] = weights
        logger.info(f"[V5] Current adaptive weights: {weights}")

    # =========================================================================
    # PATTERN EXTRACTION
    # =========================================================================

    async def _extract_patterns(self, results: Dict):
        """Extract patterns from recent trades."""
        try:
            recent_trades = await self.agent.trade_memory.get_recent_trades(days=30)

            if len(recent_trades) >= 10:
                extraction_result = await self.agent.pattern_extractor.run_extraction(recent_trades)

                results["patterns_extracted"] = len(extraction_result.candidates)
                results["new_rules_generated"] = len(extraction_result.rules)
                results["rules"] = [r.to_dict() for r in extraction_result.rules]

                logger.info(f"Extracted {results['patterns_extracted']} patterns, "
                           f"generated {results['new_rules_generated']} rules")

                if extraction_result.rules:
                    await self._save_learned_rules(extraction_result.rules)

        except Exception as e:
            logger.warning(f"Pattern extraction failed: {e}")

    # =========================================================================
    # MEMORY CLEANUP
    # =========================================================================

    async def _cleanup_memory(self, results: Dict):
        """Remove stale data from trade memory."""
        try:
            cleanup_result = await self.agent.trade_memory.cleanup_stale_memory()
            results["memory_cleanup"] = cleanup_result
            logger.info(f"Memory cleanup: removed {sum(cleanup_result.values())} stale trades")
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")

    # =========================================================================
    # SHADOW TRACKING AUDIT
    # =========================================================================

    async def _run_shadow_audit(self, results: Dict):
        """Verify shadow tracking outcomes and learn."""
        try:
            shadow_stats = await self.agent.shadow_tracker.verify_outcomes()
            results["shadow_tracking"] = shadow_stats
            logger.info(f"[SHADOW] Verified {shadow_stats.get('verified', 0)} signals: "
                       f"{shadow_stats.get('winners', 0)} winners, "
                       f"{shadow_stats.get('losers', 0)} losers")

            insights = self.agent.shadow_tracker.generate_learning_insights()
            if insights:
                results["learning_insights"] = len(insights)
                logger.info(f"[SHADOW] Generated {len(insights)} learning insights")

                self.agent.shadow_tracker.apply_weight_adjustments(max_change=0.03)
                new_weights = self.agent.shadow_tracker.pillar_weights
                results["shadow_weights"] = new_weights
                logger.info(f"[SHADOW] Updated pillar weights: {new_weights}")

            overall_stats = self.agent.shadow_tracker.get_statistics()
            results["shadow_overall"] = overall_stats
            logger.info(f"[SHADOW] Overall: {overall_stats['total_signals']} signals, "
                       f"Win Rate: {overall_stats['win_rate']:.1f}%, "
                       f"Profit Factor: {overall_stats['profit_factor']:.2f}")

        except Exception as e:
            logger.warning(f"Shadow tracker audit failed: {e}")

    # =========================================================================
    # STRATEGY IMPROVEMENT (LLM)
    # =========================================================================

    async def _analyze_losing_trades(self, losing_trades: List, closed_trades: List, results: Dict):
        """Use LLM to analyze losing trades and propose strategy improvements."""
        try:
            market_context = {}
            if self.agent.trend_discovery:
                try:
                    market_ctx = await self.agent.trend_discovery.get_market_context()
                    market_context = market_ctx if isinstance(market_ctx, dict) else {}
                except Exception:
                    pass

            failure_patterns = self._extract_failure_patterns(losing_trades)

            performance_data = {
                "total_trades": len(closed_trades),
                "win_rate": results["summary"].get("win_rate", 0) if results.get("summary") else 0,
                "total_pnl": results["summary"].get("total_pnl_pct", 0) if results.get("summary") else 0,
                "losing_trades": len(losing_trades)
            }

            if performance_data["win_rate"] < 0.5 or performance_data["total_pnl"] < 0:
                logger.info("Win rate or P&L below target, requesting LLM strategy analysis...")

                new_strategy = await self.agent.strategy_composer.compose_strategy(
                    performance_data=performance_data,
                    failure_patterns=failure_patterns,
                    market_context=market_context
                )

                results["strategy_improvements"].append({
                    "strategy_name": new_strategy.name,
                    "description": new_strategy.description,
                    "reasoning": new_strategy.reasoning[:500],
                    "status": "proposed"
                })

                await self._save_proposed_strategy(new_strategy)
                logger.info(f"LLM proposed new strategy: {new_strategy.name}")

        except Exception as e:
            logger.warning(f"Strategy Composer analysis failed: {e}")

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    async def _save_learned_rules(self, rules: List) -> None:
        """Save learned rules to JSON file."""
        rules_file = os.path.join(self.config.data_dir, "learned_guidelines.json")

        existing_rules = []
        if os.path.exists(rules_file):
            try:
                with open(rules_file, 'r') as f:
                    existing_rules = json.load(f)
            except Exception:
                pass

        for rule in rules:
            rule_dict = rule.to_dict() if hasattr(rule, 'to_dict') else rule
            rule_dict["learned_at"] = datetime.now().isoformat()
            existing_rules.append(rule_dict)

        with open(rules_file, 'w') as f:
            json.dump(existing_rules, f, indent=2)

        logger.info(f"Saved {len(rules)} new rules to {rules_file}")

    async def _save_proposed_strategy(self, strategy) -> None:
        """Save a proposed strategy to JSON file."""
        strategies_file = os.path.join(self.config.data_dir, "proposed_strategies.json")

        existing = []
        if os.path.exists(strategies_file):
            try:
                with open(strategies_file, 'r') as f:
                    existing = json.load(f)
            except Exception:
                pass

        strategy_dict = strategy.to_dict() if hasattr(strategy, 'to_dict') else strategy
        strategy_dict["proposed_at"] = datetime.now().isoformat()
        existing.append(strategy_dict)

        existing = existing[-10:]

        with open(strategies_file, 'w') as f:
            json.dump(existing, f, indent=2)

        logger.info(f"Saved proposed strategy: {strategy.name}")

    # =========================================================================
    # ANALYSIS HELPERS
    # =========================================================================

    @staticmethod
    def _extract_failure_patterns(losing_trades: List) -> List[str]:
        """Extract common patterns from losing trades."""
        patterns = []

        if not losing_trades:
            return patterns

        exit_types = {}
        for t in losing_trades:
            exit_type = t.exit_type if hasattr(t, 'exit_type') else 'unknown'
            exit_types[exit_type] = exit_types.get(exit_type, 0) + 1

        most_common_exit = max(exit_types, key=exit_types.get) if exit_types else None
        if most_common_exit == "stop_loss":
            patterns.append("Majority of losses hit stop loss - consider wider stops or better entry timing")
        elif most_common_exit == "max_hold_period":
            patterns.append("Many trades expired at max hold - consider shorter timeframes or momentum confirmation")

        avg_hold = sum(t.hold_days for t in losing_trades if hasattr(t, 'hold_days')) / len(losing_trades)
        if avg_hold < 3:
            patterns.append(f"Short average hold ({avg_hold:.1f} days) - entries may be too early")
        elif avg_hold > 30:
            patterns.append(f"Long average hold ({avg_hold:.1f} days) - consider trailing stops")

        high_vix_losses = sum(1 for t in losing_trades
                             if hasattr(t, 'metadata') and t.metadata.get('vix_level', 0) > 25)
        if high_vix_losses > len(losing_trades) * 0.6:
            patterns.append("High VIX correlation with losses - reduce position size in high volatility")

        overbought_entries = sum(1 for t in losing_trades
                                 if hasattr(t, 'metadata') and t.metadata.get('rsi', 50) > 70)
        if overbought_entries > len(losing_trades) * 0.4:
            patterns.append("Many entries at overbought RSI - wait for pullbacks")

        return patterns
