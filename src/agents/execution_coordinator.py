"""
Execution Coordinator - Extracted from orchestrator.py (V8.2 Sprint 4)

Handles:
- Trading Scan (09:30-16:00): Symbol scanning, scoring, signal detection
- Alert Processing: Guardrail validation, ML veto, memory insight
- Trade Execution: IBKR orders, simulated trades, position tracking
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.orchestrator import MarketAgent

from src.agents.state import AgentPhase, Position
from src.agents.guardrails import TradeRequest, TradeValidation

# V5.3 - ML Pillar for trade validation
try:
    from src.agents.ml_integration import validate_trade_ml, get_ml_validator
    ML_VALIDATOR_AVAILABLE = True
except ImportError:
    ML_VALIDATOR_AVAILABLE = False

logger = logging.getLogger(__name__)


class ExecutionCoordinator:
    """Coordinates trading scans, alert processing, and trade execution."""

    def __init__(self, agent: 'MarketAgent'):
        self.agent = agent

    @property
    def config(self):
        return self.agent.config

    @property
    def state_manager(self):
        return self.agent.state_manager

    @property
    def guardrails(self):
        return self.agent.guardrails

    # =========================================================================
    # TRADING SCAN
    # =========================================================================

    async def run_trading_scan(self, execute_trades: bool = True) -> Dict:
        """
        Un cycle de scan pendant les heures de trading.
        Detecte les signaux et execute les trades (si execute_trades=True).
        """
        logger.info("--- Trading Scan Started ---")
        self.state_manager.set_phase(AgentPhase.TRADING)

        results = {
            "timestamp": datetime.now().isoformat(),
            "scanned_symbols": 0,
            "signals_found": 0,
            "trades_executed": 0,
            "trades_rejected": 0,
            "alerts": [],
            "analyzed_symbols": [],
            "scoring_details": {},
            "rejected_symbols": {},
            "guardrails_blocked": [],
            "narrative_reports": {}
        }

        # Check kill switch
        kill_active, kill_reason = self.guardrails.is_kill_switch_active()
        if kill_active:
            logger.warning(f"Kill switch active: {kill_reason}")
            return results

        # Get symbols to scan
        focus_symbols = self.state_manager.state.focus_symbols
        if not focus_symbols:
            focus_symbols = self.state_manager.state.watchlist[:50]

        if not focus_symbols:
            logger.warning("No symbols to scan")
            return results

        # Filter recently analyzed symbols
        original_count = len(focus_symbols)
        cooldown_hours = getattr(self.config, 'analysis_cooldown_hours', 1)
        fresh_symbols = self.agent.analysis_store.get_unanalyzed_symbols(
            focus_symbols, hours=cooldown_hours
        )

        if len(fresh_symbols) < len(focus_symbols):
            skipped = original_count - len(fresh_symbols)
            logger.info(f"Skipping {skipped} recently analyzed symbols (cooldown: {cooldown_hours}h)")
            results["skipped_recent"] = skipped

        if not fresh_symbols:
            stale_symbols = self.agent.analysis_store.get_symbols_needing_reanalysis(hours=24)
            if stale_symbols:
                fresh_symbols = stale_symbols[:15]
                logger.info(f"All focus symbols recently analyzed. Refreshing {len(fresh_symbols)} stale symbols.")
            else:
                logger.info("All symbols recently analyzed. Waiting for cooldown.")
                results["all_symbols_recent"] = True
                return results

        focus_symbols = fresh_symbols
        results["scanned_symbols"] = len(focus_symbols)

        # Scan with ReasoningEngine (4 Pillars) - PRIMARY
        if self.agent.reasoning_engine:
            try:
                for symbol in focus_symbols:
                    try:
                        results["analyzed_symbols"].append(symbol)

                        # Fetch OHLCV data for technical analysis
                        logger.info(f"Fetching OHLCV data for {symbol}...")
                        df = self.agent.market_data_fetcher.get_historical_data(
                            symbol=symbol, period='6mo', interval='1d'
                        )
                        if df is None or len(df) < 50:
                            logger.warning(f"Insufficient data for {symbol}: "
                                         f"{len(df) if df is not None else 0} rows (need 50+)")
                            results["rejected_symbols"][symbol] = (
                                f"Insufficient historical data ({len(df) if df is not None else 0} rows)"
                            )
                            continue

                        # Analyze with 4 Pillars
                        reasoning_result = await self.agent.reasoning_engine.analyze(symbol, df=df)

                        # Store detailed scoring
                        scoring = self._build_scoring_details(symbol, reasoning_result)
                        results["scoring_details"][symbol] = scoring

                        # Persist analysis to store
                        self._persist_analysis(symbol, reasoning_result, results)

                        # Record in orchestrator memory
                        self._record_decision(symbol, reasoning_result)

                        # Skip non-actionable signals
                        if reasoning_result.decision.value not in ["strong_buy", "buy"]:
                            decision_reason = (
                                f"Decision: {reasoning_result.decision.value} "
                                f"(score: {reasoning_result.total_score:.1f}/100)"
                            )
                            if reasoning_result.risk_factors:
                                decision_reason += f" | Risks: {', '.join(reasoning_result.risk_factors[:2])}"
                            results["rejected_symbols"][symbol] = decision_reason
                            continue

                        # Consult Trade Memory
                        memory_insight = await self._get_memory_insight(symbol, reasoning_result)

                        # Convert to alert
                        alert = self._reasoning_to_alert(symbol, reasoning_result, memory_insight)
                        results["alerts"].append(alert)
                        results["signals_found"] += 1

                        # Shadow track signal
                        await self._shadow_track(symbol, reasoning_result, alert)

                        # Generate narrative report
                        await self._generate_narrative(symbol, reasoning_result, results)

                        logger.info(f"Signal: {symbol} - {reasoning_result.decision.value} "
                                   f"(score={reasoning_result.total_score:.1f}, "
                                   f"conf={reasoning_result.confidence:.2f})")

                    except Exception as e:
                        logger.warning(f"Failed to analyze {symbol}: {e}")
                        continue

                logger.info(f"ReasoningEngine found {results['signals_found']} signals")

                # Process alerts
                if execute_trades:
                    for alert in results["alerts"]:
                        trade_result = await self._process_alert(alert)
                        if trade_result.get("executed"):
                            results["trades_executed"] += 1
                        elif trade_result.get("rejected"):
                            results["trades_rejected"] += 1
                            reason = trade_result.get("reason", "Unknown")
                            if any(kw in reason.lower() for kw in ("guardrail", "kill", "limit")):
                                results["guardrails_blocked"].append({
                                    "symbol": alert.get("symbol"),
                                    "reason": reason
                                })
                else:
                    logger.info(f"Analysis mode: {len(results['alerts'])} signals prepared (no execution)")
                    results["analysis_only"] = True
                    results["pending_signals"] = len(results["alerts"])

            except Exception as e:
                logger.error(f"ReasoningEngine error: {e}")
                self.state_manager.log_error(f"Reasoning failed: {e}")

        # FALLBACK: V3 Screener
        elif self.agent.market_screener:
            try:
                alerts = self.agent.market_screener.screen_multiple_stocks(focus_symbols)
                results["signals_found"] = len(alerts)
                results["alerts"] = alerts
                logger.info(f"V3 Screener found {len(alerts)} signals")

                if execute_trades:
                    for alert in alerts:
                        trade_result = await self._process_alert(alert)
                        if trade_result.get("executed"):
                            results["trades_executed"] += 1
                        elif trade_result.get("rejected"):
                            results["trades_rejected"] += 1
                else:
                    logger.info(f"Analysis mode: {len(alerts)} signals prepared (no execution)")
                    results["analysis_only"] = True
                    results["pending_signals"] = len(alerts)

            except Exception as e:
                logger.error(f"Screening error: {e}")
                self.state_manager.log_error(f"Screening failed: {e}")

        logger.info(f"--- Trading Scan Complete: {results['trades_executed']} executed, "
                   f"{results['trades_rejected']} rejected ---")
        return results

    # =========================================================================
    # SCORING & PERSISTENCE HELPERS
    # =========================================================================

    def _build_scoring_details(self, symbol: str, reasoning_result) -> Dict:
        """Build detailed scoring dict from reasoning result."""
        return {
            "total_score": round(reasoning_result.total_score, 2),
            "technical": round(reasoning_result.technical_score.score, 2) if reasoning_result.technical_score else 0,
            "fundamental": round(reasoning_result.fundamental_score.score, 2) if reasoning_result.fundamental_score else 0,
            "sentiment": round(reasoning_result.sentiment_score.score, 2) if reasoning_result.sentiment_score else 0,
            "news": round(reasoning_result.news_score.score, 2) if reasoning_result.news_score else 0,
            "decision": reasoning_result.decision.value,
            "key_factors": reasoning_result.key_factors[:5] if reasoning_result.key_factors else [],
            "risk_factors": reasoning_result.risk_factors[:5] if reasoning_result.risk_factors else [],
            "pillar_details": {
                pillar: {
                    "score": round(getattr(reasoning_result, f"{pillar}_score").score, 2)
                            if getattr(reasoning_result, f"{pillar}_score") else 0,
                    "reasoning": getattr(reasoning_result, f"{pillar}_score").reasoning
                                if getattr(reasoning_result, f"{pillar}_score") else "",
                    "factors": (getattr(reasoning_result, f"{pillar}_score").factors[:5]
                               if getattr(reasoning_result, f"{pillar}_score")
                               and getattr(reasoning_result, f"{pillar}_score").factors else []),
                    "signal": (getattr(reasoning_result, f"{pillar}_score").signal.value
                              if getattr(reasoning_result, f"{pillar}_score") else "neutral"),
                    "data_quality": (getattr(reasoning_result, f"{pillar}_score").data_quality
                                    if getattr(reasoning_result, f"{pillar}_score") else 0)
                }
                for pillar in ["technical", "fundamental", "sentiment", "news"]
            },
            "reasoning_summary": reasoning_result.reasoning_summary
        }

    def _persist_analysis(self, symbol: str, reasoning_result, results: Dict):
        """Persist analysis to store and add journey step."""
        try:
            market_regime = None
            if reasoning_result.market_context:
                market_regime = reasoning_result.market_context.regime.value

            self.agent.analysis_store.save_analysis(
                symbol=symbol,
                reasoning_result=reasoning_result,
                market_regime=market_regime
            )

            pillar_details = results["scoring_details"][symbol].get("pillar_details", {})
            step_type = "analysis" if reasoning_result.decision.value in ["strong_buy", "buy"] else "rejected"

            self.agent.analysis_store.add_journey_step(
                symbol=symbol,
                step=step_type,
                title=f"Score {round(reasoning_result.total_score, 1)}/100 - {reasoning_result.decision.value}",
                reasoning=reasoning_result.reasoning_summary,
                data={
                    "total": round(reasoning_result.total_score, 2),
                    "technical": round(reasoning_result.technical_score.score, 2) if reasoning_result.technical_score else 0,
                    "fundamental": round(reasoning_result.fundamental_score.score, 2) if reasoning_result.fundamental_score else 0,
                    "sentiment": round(reasoning_result.sentiment_score.score, 2) if reasoning_result.sentiment_score else 0,
                    "news": round(reasoning_result.news_score.score, 2) if reasoning_result.news_score else 0,
                    "decision": reasoning_result.decision.value,
                    "pillar_details": pillar_details,
                    "key_factors": reasoning_result.key_factors[:5] if reasoning_result.key_factors else [],
                    "risk_factors": reasoning_result.risk_factors[:5] if reasoning_result.risk_factors else []
                }
            )
            logger.info(f"Persisted analysis for {symbol}")
        except Exception as e:
            logger.error(f"Failed to persist analysis for {symbol}: {e}")

    def _record_decision(self, symbol: str, reasoning_result):
        """Record decision in orchestrator memory."""
        if not self.agent.orchestrator_memory:
            return
        try:
            pillar_scores_dict = {
                "technical": round(reasoning_result.technical_score.score, 2) if reasoning_result.technical_score else 0,
                "fundamental": round(reasoning_result.fundamental_score.score, 2) if reasoning_result.fundamental_score else 0,
                "sentiment": round(reasoning_result.sentiment_score.score, 2) if reasoning_result.sentiment_score else 0,
                "news": round(reasoning_result.news_score.score, 2) if reasoning_result.news_score else 0,
            }
            self.agent.orchestrator_memory.record_decision(
                symbol=symbol,
                total_score=reasoning_result.total_score,
                action=reasoning_result.decision.value,
                pillar_scores=pillar_scores_dict,
                entry_price=0
            )
        except Exception as mem_err:
            logger.warning(f"[MEMORY] Failed to record decision for {symbol}: {mem_err}")

    async def _get_memory_insight(self, symbol: str, reasoning_result) -> Optional[Dict]:
        """Consult Trade Memory for similar past trades."""
        if not self.agent.trade_memory:
            return None

        try:
            from src.intelligence.trade_memory import TradeContext
            context = self._build_trade_context(symbol, reasoning_result)
            similar_trades = self.agent.trade_memory.find_similar(
                context=context,
                top_k=self.config.similar_trades_k,
                current_regime=context.market_regime
            )
            if similar_trades:
                wins = sum(1 for t in similar_trades if t.outcome.is_win)
                return {
                    "similar_count": len(similar_trades),
                    "historical_win_rate": wins / len(similar_trades),
                    "avg_pnl": sum(t.outcome.pnl_pct for t in similar_trades) / len(similar_trades),
                    "avg_relevance": sum(t.relevance_score for t in similar_trades) / len(similar_trades)
                }
        except Exception as e:
            logger.warning(f"Trade memory lookup failed: {e}")
        return None

    async def _shadow_track(self, symbol: str, reasoning_result, alert: Dict):
        """Shadow track BUY signal for autonomous learning."""
        if not self.agent.shadow_tracker:
            return

        try:
            entry_price = alert.get("price", 0)
            if entry_price > 0:
                await self.agent.shadow_tracker.track_signal(
                    symbol=symbol,
                    signal_type=alert.get("signal", "BUY"),
                    total_score=reasoning_result.total_score,
                    technical_score=reasoning_result.technical_score.score if reasoning_result.technical_score else 0,
                    fundamental_score=reasoning_result.fundamental_score.score if reasoning_result.fundamental_score else 0,
                    sentiment_score=reasoning_result.sentiment_score.score if reasoning_result.sentiment_score else 0,
                    news_score=reasoning_result.news_score.score if reasoning_result.news_score else 0,
                    entry_price=entry_price,
                    stop_loss=alert.get("stop_loss"),
                    take_profit=entry_price * 1.10,
                    key_factors=reasoning_result.key_factors[:5] if reasoning_result.key_factors else [],
                    risk_factors=reasoning_result.risk_factors[:5] if reasoning_result.risk_factors else []
                )
                logger.info(f"[SHADOW] Auto-tracked {symbol} @ ${entry_price:.2f}")
        except Exception as e:
            logger.warning(f"Shadow tracking failed for {symbol}: {e}")

    async def _generate_narrative(self, symbol: str, reasoning_result, results: Dict):
        """Generate narrative report for actionable signal."""
        if not self.agent.narrative_generator:
            return

        try:
            _pillar_context = (
                f"\n=== PILLAR SCORES for {symbol} ===\n"
                f"Technical: {reasoning_result.technical_score.score:.1f}/100\n" if reasoning_result.technical_score else ""
                f"Fundamental: {reasoning_result.fundamental_score.score:.1f}/100\n" if reasoning_result.fundamental_score else ""
                f"Sentiment: {reasoning_result.sentiment_score.score:.1f}/100\n" if reasoning_result.sentiment_score else ""
                f"News: {reasoning_result.news_score.score:.1f}/100\n" if reasoning_result.news_score else ""
                f"TOTAL: {reasoning_result.total_score:.1f}/100 -> {reasoning_result.decision.value}\n"
            )
            _memory_context = ""
            if self.agent.orchestrator_memory:
                _memory_context = self.agent.orchestrator_memory.get_summary_for_prompt(days=14)

            grok_data = None
            if self.agent.grok_scanner:
                try:
                    grok_data = await self.agent.grok_scanner.get_symbol_details_for_narrative(symbol)
                except Exception:
                    pass

            narrative = await self.agent.narrative_generator.generate_analysis(
                symbol=symbol,
                social_data=None,
                grok_data=grok_data,
                technical_data={
                    'rsi': reasoning_result.technical_score.factors[0].get('rsi', 50)
                           if reasoning_result.technical_score and reasoning_result.technical_score.factors else 50,
                    'ema_alignment': 'bullish' if reasoning_result.technical_score
                                     and reasoning_result.technical_score.score > 50 else 'neutral',
                    'volume_ratio': 1.0,
                    'trend': 'up' if reasoning_result.technical_score
                             and reasoning_result.technical_score.score > 50 else 'sideways'
                },
                fundamental_data={
                    'score': reasoning_result.fundamental_score.score if reasoning_result.fundamental_score else 0
                },
                news_data={
                    'score': reasoning_result.news_score.score if reasoning_result.news_score else 0,
                    'pillar_context': _pillar_context,
                    'memory_context': _memory_context
                },
                reasoning_result=reasoning_result
            )
            results["narrative_reports"][symbol] = narrative.to_markdown()
            logger.info(f"Narrative report generated for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to generate narrative for {symbol}: {e}")

    # =========================================================================
    # CONTEXT BUILDERS
    # =========================================================================

    def _build_trade_context(self, symbol: str, reasoning_result):
        """Build a TradeContext from ReasoningResult for memory lookup."""
        from src.intelligence.trade_memory import TradeContext

        market_ctx = reasoning_result.market_context
        regime = "unknown"
        vix = 20.0
        spy_trend = "sideways"

        if market_ctx:
            regime = market_ctx.regime.value if hasattr(market_ctx, 'regime') else "unknown"
            vix = market_ctx.vix_level if hasattr(market_ctx, 'vix_level') else 20.0
            spy_trend = market_ctx.spy_trend if hasattr(market_ctx, 'spy_trend') else "sideways"

        tech_score = reasoning_result.technical_score
        sent_score = reasoning_result.sentiment_score

        return TradeContext(
            symbol=symbol,
            timestamp=reasoning_result.timestamp,
            market_regime=regime,
            vix_level=vix,
            spy_trend=spy_trend,
            sector_momentum=0.0,
            rsi=tech_score.factors[0].get('rsi', 50) if tech_score.factors else 50,
            ema_alignment="bullish" if tech_score.score > 0 else "bearish",
            volume_ratio=1.0,
            breakout_strength=tech_score.score / 100 if tech_score.score > 0 else None,
            social_sentiment=sent_score.factors[0].get('social', 0) if sent_score.factors else 0,
            grok_sentiment=sent_score.factors[0].get('grok', 0) if sent_score.factors else 0,
            stocktwits_sentiment=sent_score.factors[0].get('stocktwits', 0) if sent_score.factors else 0,
            heat_score=0.0,
            news_sentiment=reasoning_result.news_score.score / 100,
            catalyst_present=reasoning_result.news_score.score > 30,
            decision_score=reasoning_result.total_score,
            confidence=reasoning_result.confidence
        )

    def _reasoning_to_alert(self, symbol: str, reasoning_result,
                            memory_insight: Optional[Dict] = None) -> Dict:
        """Convert ReasoningResult to alert format compatible with _process_alert."""
        decision_to_signal = {
            "strong_buy": "STRONG_BUY",
            "buy": "BUY",
            "hold": "HOLD",
            "sell": "SELL",
            "strong_sell": "STRONG_SELL"
        }
        signal = decision_to_signal.get(reasoning_result.decision.value, "HOLD")
        confidence_score = int(reasoning_result.confidence * 100)

        if memory_insight:
            if memory_insight.get("historical_win_rate", 0) > 0.6:
                confidence_score = min(100, confidence_score + 5)
            elif memory_insight.get("historical_win_rate", 0) < 0.4:
                confidence_score = max(0, confidence_score - 10)

        # V8.2 Sprint 4: Calculate stop-loss from technical analysis
        price = 0.0
        stop_loss = None
        tech = reasoning_result.technical_score
        if tech and tech.factors:
            for factor_dict in tech.factors:
                if isinstance(factor_dict, dict):
                    if 'close' in factor_dict:
                        price = float(factor_dict['close'])
                    elif 'price' in factor_dict:
                        price = float(factor_dict['price'])
                    atr = factor_dict.get('atr', factor_dict.get('ATR'))
                    if atr and price > 0:
                        stop_loss = round(price - (2.0 * float(atr)), 2)
        if price > 0 and stop_loss is None:
            stop_loss = round(price * 0.95, 2)

        return {
            "symbol": symbol,
            "signal": signal,
            "confidence_score": confidence_score,
            "price": price,
            "stop_loss": stop_loss,
            "reasoning_summary": reasoning_result.reasoning_summary,
            "total_score": reasoning_result.total_score,
            "pillar_scores": {
                "technical": reasoning_result.technical_score.score,
                "fundamental": reasoning_result.fundamental_score.score,
                "sentiment": reasoning_result.sentiment_score.score,
                "news": reasoning_result.news_score.score
            },
            "key_factors": reasoning_result.key_factors[:3],
            "risk_factors": reasoning_result.risk_factors,
            "memory_insight": memory_insight,
            "timestamp": reasoning_result.timestamp
        }

    # =========================================================================
    # ALERT PROCESSING & TRADE EXECUTION
    # =========================================================================

    async def _process_alert(self, alert: Dict) -> Dict:
        """Process an alert: validate with guardrails and execute if approved."""
        symbol = alert.get("symbol")
        signal = alert.get("signal", "")
        confidence = alert.get("confidence_score", 0)
        price = alert.get("price", 0)
        stop_loss = alert.get("stop_loss")

        result = {
            "symbol": symbol, "signal": signal,
            "executed": False, "rejected": False, "reason": ""
        }

        if signal not in ["STRONG_BUY", "BUY"]:
            result["rejected"] = True
            result["reason"] = f"Signal trop faible: {signal} (requis: BUY ou STRONG_BUY)"
            return result

        # ML Pillar Validation
        if ML_VALIDATOR_AVAILABLE:
            try:
                scoring = alert.get("scoring_details", {})
                ml_result = validate_trade_ml(
                    symbol=symbol,
                    technical_score=scoring.get("technical", 50),
                    fundamental_score=scoring.get("fundamental", 50),
                    sentiment_score=scoring.get("sentiment", 50),
                    news_score=scoring.get("news", 50),
                    combined_score=confidence,
                    rsi=scoring.get("rsi", 50),
                    market_regime=alert.get("market_regime", "neutral")
                )
                result["ml_score"] = ml_result.get("ml_score", 50)
                result["ml_probability"] = ml_result.get("ml_probability", 0.5)
                if ml_result.get("ml_veto", False):
                    result["rejected"] = True
                    result["reason"] = f"ML Veto: {ml_result['ml_probability']*100:.0f}% win prob (min 30%)"
                    logger.info(f"[ML] Trade {symbol} vetoed: {ml_result['ml_probability']*100:.1f}%")
                    return result
                logger.info(f"[ML] Trade {symbol} approved: {ml_result['ml_probability']*100:.1f}%")
            except Exception as e:
                logger.warning(f"[ML] Validation error for {symbol}: {e}")

        if confidence < 55:
            result["rejected"] = True
            result["reason"] = f"Score confiance trop bas: {confidence}/100 (seuil: 55)"
            return result

        # Calculate position size
        capital = self.state_manager.state.current_capital
        position_value = capital * 0.05
        quantity = int(position_value / price) if price > 0 else 0

        if quantity < 1:
            result["rejected"] = True
            result["reason"] = f"Position trop petite: prix ${price:.2f} > capital disponible pour 1 action"
            return result

        trade_request = TradeRequest(
            symbol=symbol, action="BUY", quantity=quantity, price=price,
            order_type="LIMIT", current_capital=capital,
            position_value=quantity * price,
            daily_pnl=self.guardrails._daily_pnl,
            current_drawdown=self.state_manager.get_drawdown(),
            stop_loss=stop_loss,
            thesis=f"{signal} signal, confidence {confidence}/100"
        )

        # Guardrail validation
        validation = self.guardrails.validate_trade(trade_request)

        if not validation.is_allowed():
            if self.agent.rotation_manager and "position" in validation.reason.lower():
                logger.info(f"Portfolio full, checking rotation for {symbol}...")
                try:
                    rotation = await self.agent.rotation_manager.check_and_rotate(
                        new_signal_symbol=symbol,
                        new_signal_score=confidence,
                        executor=self.agent.ibkr_executor
                    )
                    if rotation.should_rotate:
                        logger.info(f"Rotation OK: sold {rotation.sell_symbol}")
                        validation = self.guardrails.validate_trade(trade_request)
                    else:
                        logger.info(f"No rotation: {rotation.sell_reason}")
                except Exception as e:
                    logger.warning(f"Rotation check failed: {e}")

            if not validation.is_allowed():
                result["rejected"] = True
                result["reason"] = f"Guardrail: {validation.reason}"
                logger.info(f"Trade rejected for {symbol}: {validation.reason}")
                return result

        # Execute trade
        if validation.status == TradeValidation.APPROVED:
            executed = await self._execute_trade(trade_request)
            result["executed"] = executed
            result["reason"] = "Autonomous execution" if executed else "Execution failed"
        elif validation.status == TradeValidation.NEEDS_NOTIFICATION:
            result["rejected"] = True
            result["reason"] = "Notification required (not implemented)"
        elif validation.status == TradeValidation.NEEDS_APPROVAL:
            result["rejected"] = True
            result["reason"] = "Manual approval required"

        return result

    async def _execute_trade(self, trade: TradeRequest) -> bool:
        """Execute a trade via IBKR."""
        if not self.agent.ibkr_executor:
            logger.warning("IBKR executor not available - simulating trade")
            self._simulate_trade(trade)
            return True

        try:
            if not self.agent.ibkr_executor.is_connected():
                await self.agent.ibkr_executor.connect()

            order_id = await self.agent.ibkr_executor.place_order(
                symbol=trade.symbol, action=trade.action,
                quantity=trade.quantity, order_type=trade.order_type,
                limit_price=trade.price if trade.order_type == "LIMIT" else None
            )

            if order_id:
                self._record_position(trade)
                logger.info(f"Trade executed: {trade.action} {trade.quantity} {trade.symbol} @ {trade.price}")
                return True

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            self.state_manager.log_error(f"Trade execution failed: {e}")

        return False

    def _simulate_trade(self, trade: TradeRequest):
        """Simulate a trade (for tests without IBKR)."""
        self._record_position(trade)
        logger.info(f"Trade SIMULATED: {trade.action} {trade.quantity} {trade.symbol} @ {trade.price}")

    def _record_position(self, trade: TradeRequest):
        """Record a new position in state and adaptive learning."""
        position = Position(
            symbol=trade.symbol,
            quantity=trade.quantity,
            entry_price=trade.price,
            entry_date=datetime.now().isoformat(),
            stop_loss=trade.stop_loss,
            thesis=trade.thesis
        )
        self.state_manager.add_position(position)

        if self.agent.position_manager:
            self.agent.position_manager.add_position(
                symbol=trade.symbol,
                entry_price=trade.price,
                quantity=trade.quantity,
                entry_score=trade.confidence_score or 50
            )
            logger.info(f"[V5.1] Position tracked: {trade.symbol}")

        self.guardrails.record_trade(trade.symbol, trade.action, trade.position_value)

        if self.agent.adaptive_scorer and hasattr(trade, 'pillar_scores'):
            try:
                self.agent.adaptive_scorer.record_entry(
                    trade_id=f"{trade.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    symbol=trade.symbol,
                    entry_price=trade.price,
                    pillar_scores=trade.pillar_scores or {},
                    entry_score=trade.confidence_score or 50,
                    decision=trade.action
                )
                logger.info(f"[V5] Trade recorded for adaptive learning: {trade.symbol}")
            except Exception as e:
                logger.warning(f"[V5] Failed to record trade for learning: {e}")
