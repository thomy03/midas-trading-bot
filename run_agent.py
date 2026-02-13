#!/usr/bin/env python3
"""
TradingBot V4 - Agent Runner
Script principal pour lancer l'agent de trading autonome.

Usage:
    # Mode LIVE - Boucle event-driven temps réel (RECOMMANDÉ)
    python run_agent.py --mode live
    python run_agent.py --mode live --paper   # Paper trading

    # Mode découverte (scan social + tendances)
    python run_agent.py --mode discovery

    # Mode analyse (analyse LLM des tendances)
    python run_agent.py --mode analysis

    # Mode trading (screening + exécution one-shot)
    python run_agent.py --mode trading

    # Mode complet (cycle quotidien one-shot)
    python run_agent.py --mode full

    # Mode test (avec mock IBKR)
    python run_agent.py --mode test
"""

import os
import sys
import asyncio
import argparse
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Ajouter le répertoire racine au path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('agent')


# =============================================================================
# AGENT RUNNER
# =============================================================================

class AgentRunner:
    """
    Runner pour l'agent de trading V4.

    Modes disponibles:
    - discovery: Scan social + découverte de stocks
    - analysis: Analyse LLM des tendances
    - trading: Screening + exécution
    - full: Cycle quotidien complet
    - test: Mode test avec mock
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.agent = None
        self.social_scanner = None
        self.grok_scanner = None
        self.stock_discovery = None
        self.executor = None

    async def initialize(self):
        """Initialise tous les composants"""
        logger.info("=" * 60)
        logger.info("🤖 TradingBot V4 - Agent Autonome")
        logger.info("=" * 60)

        capital = float(os.getenv('TRADING_CAPITAL', 1500))
        logger.info(f"Capital: {capital}€")
        logger.info(f"Mode: {'MOCK' if self.use_mock else 'LIVE'}")
        logger.info("")

        # 1. Social Scanner - REMOVED (V8.2: replaced by Intelligence Orchestrator)

        # 2. Grok Scanner (si API key disponible)
        grok_key = os.getenv('GROK_API_KEY', '')
        if grok_key and not grok_key.startswith('your_'):
            logger.info("Initializing Grok Scanner...")
            from src.intelligence.grok_scanner import GrokScanner
            self.grok_scanner = GrokScanner()
            await self.grok_scanner.initialize()
        else:
            logger.warning("Grok API key not configured - skipping")

        # 3. Stock Discovery
        logger.info("Initializing Stock Discovery...")
        from src.intelligence.stock_discovery import StockDiscovery
        self.stock_discovery = StockDiscovery()
        await self.stock_discovery.initialize()

        # 4. IBKR Executor
        if self.use_mock:
            logger.info("Initializing Mock IBKR Executor...")
            from src.execution.ibkr_executor import MockIBKRExecutor
            self.executor = MockIBKRExecutor(initial_cash=capital)
        else:
            logger.info("Initializing IBKR Executor...")
            from src.execution.ibkr_executor import IBKRExecutor
            port = int(os.getenv('IBKR_PORT', 7497))
            self.executor = IBKRExecutor(port=port)

        await self.executor.connect()

        # 5. Market Agent (Orchestrator)
        logger.info("Initializing Market Agent...")
        from src.agents.orchestrator import MarketAgent, OrchestratorConfig
        config = OrchestratorConfig(initial_capital=capital)
        self.agent = MarketAgent(config=config)
        # Injecter les composants
        self.agent.social_scanner = self.social_scanner
        self.agent.grok_scanner = self.grok_scanner
        self.agent.stock_discovery = self.stock_discovery
        self.agent.executor = self.executor

        logger.info("")
        logger.info("✅ All components initialized!")
        logger.info("")

    async def run_discovery(self):
        """Mode découverte: scan social + stocks"""
        logger.info("📡 Running Discovery Phase...")
        logger.info("-" * 40)

        # 1. Scan social (Reddit)
        logger.info("Scanning Reddit...")
        social_result = await self.social_scanner.full_scan()
        logger.info(f"  Found {social_result.total_mentions} mentions")
        logger.info(f"  Overall sentiment: {social_result.overall_sentiment:.2f}")

        # Top symbols
        logger.info("  Top trending:")
        for ts in social_result.trending_symbols[:5]:
            logger.info(f"    ${ts.symbol}: {ts.mention_count} mentions ({ts.sentiment_label})")

        # 2. Grok scan (si disponible)
        if self.grok_scanner and self.grok_scanner.is_available():
            logger.info("")
            logger.info("Scanning X/Twitter via Grok...")
            grok_result = await self.grok_scanner.full_scan_with_analysis()
            if grok_result:
                logger.info(f"  Sentiment: {grok_result.overall_sentiment:.2f}")
                logger.info(f"  Insights: {len(grok_result.insights)}")
                for insight in grok_result.insights[:3]:
                    logger.info(f"    - {insight.topic}: {insight.sentiment}")

        # 3. Découverte de stocks
        logger.info("")
        logger.info("Discovering stocks...")

        # Convertir social_result en dict pour update_social_scores
        social_data = {}
        for ts in social_result.trending_symbols:
            social_data[ts.symbol] = {
                'mention_count': ts.mention_count,
                'avg_sentiment': ts.avg_sentiment
            }

        self.stock_discovery.update_social_scores(social_data)
        candidates = await self.stock_discovery.get_screening_candidates(max_candidates=50)
        logger.info(f"  Universe: {candidates.total_universe} stocks")
        logger.info(f"  Candidates: {candidates.filtered_count}")

        logger.info("")
        logger.info("Top 10 candidates for screening:")
        for stock in candidates.candidates[:10]:
            logger.info(f"  ${stock.symbol}: {stock.name[:25]} | MCap: ${stock.market_cap/1e9:.1f}B")

        return {
            'social': social_result,
            'candidates': candidates
        }

    async def run_analysis(self):
        """Mode analyse: analyse LLM des tendances"""
        logger.info("🔬 Running Analysis Phase...")
        logger.info("-" * 40)

        if not self.grok_scanner or not self.grok_scanner.is_available():
            logger.warning("Grok not available - analysis limited")
            return None

        # Analyser les top symboles
        symbols_to_analyze = ['NVDA', 'AAPL', 'TSLA', 'AMD', 'META']

        results = {}
        for symbol in symbols_to_analyze:
            logger.info(f"Analyzing ${symbol}...")
            insight = await self.grok_scanner.analyze_symbol(symbol)
            if insight:
                results[symbol] = insight
                logger.info(f"  Sentiment: {insight.sentiment} ({insight.sentiment_score:.2f})")
                logger.info(f"  Summary: {insight.summary[:80]}...")
            await asyncio.sleep(2)  # Rate limiting

        return results

    async def run_trading(self):
        """Mode trading: screening + exécution"""
        logger.info("💰 Running Trading Phase...")
        logger.info("-" * 40)

        # 1. Obtenir les candidats
        discovery_result = await self.run_discovery()
        candidates = [c.symbol for c in discovery_result['candidates'].candidates[:20]]

        # 2. Screening technique
        logger.info("")
        logger.info("Running technical screening...")

        from src.screening.screener import MarketScreener
        screener = MarketScreener()

        alerts = []
        for symbol in candidates[:10]:  # Limiter pour le test
            try:
                result = screener.screen_stock(symbol, timeframe='daily')
                if result and result.get('signal') in ['BUY', 'STRONG_BUY']:
                    alerts.append(result)
                    logger.info(f"  🎯 {symbol}: {result.get('signal')} (confidence: {result.get('confidence_score', 0):.0f})")
            except Exception as e:
                logger.debug(f"  Error screening {symbol}: {e}")

        # 3. Exécution (si alerts)
        if alerts:
            logger.info("")
            logger.info(f"Found {len(alerts)} trading signals!")

            # Valider via guardrails
            from src.agents.guardrails import TradingGuardrails
            guardrails = TradingGuardrails(capital=float(os.getenv('TRADING_CAPITAL', 1500)))

            for alert in alerts[:3]:  # Max 3 trades
                symbol = alert.get('symbol')
                confidence = alert.get('confidence_score', 50)

                # Vérifier les guardrails
                if guardrails.check_daily_loss_limit():
                    logger.warning("Daily loss limit reached - stopping")
                    break

                if not guardrails.can_open_position():
                    logger.warning("Max positions reached - stopping")
                    break

                # Calculer la taille
                from src.agents.guardrails import TradeRequest
                trade_req = TradeRequest(
                    symbol=symbol,
                    action='BUY',
                    quantity=0,  # À calculer
                    price=alert.get('current_price', 100),
                    signal_source='screener',
                    confidence_score=confidence
                )

                validation = guardrails.validate_trade(trade_req)

                if validation.approved:
                    logger.info(f"  ✅ Executing: BUY {validation.adjusted_quantity} {symbol}")

                    if self.executor.is_connected():
                        from src.execution.ibkr_executor import OrderRequest, OrderAction, OrderType
                        order = OrderRequest(
                            symbol=symbol,
                            action=OrderAction.BUY,
                            quantity=validation.adjusted_quantity,
                            order_type=OrderType.MARKET
                        )
                        result = await self.executor.place_order(order)
                        logger.info(f"     Order: {result.status.value}")
                else:
                    logger.info(f"  ❌ Rejected: {symbol} - {validation.reason}")

        else:
            logger.info("No trading signals found")

        return alerts

    async def run_full_cycle(self):
        """Mode complet: cycle quotidien"""
        logger.info("🔄 Running Full Daily Cycle...")
        logger.info("=" * 60)

        # Phase 1: Discovery (06:00 ET)
        await self.run_discovery()

        # Phase 2: Analysis (07:00 ET)
        await self.run_analysis()

        # Phase 3: Trading (09:30-16:00 ET)
        await self.run_trading()

        # Phase 4: Audit (20:00 ET)
        await self.run_audit()

        logger.info("")
        logger.info("✅ Daily cycle complete!")

    async def run_audit(self):
        """Mode audit: analyse des trades"""
        logger.info("📊 Running Nightly Audit...")
        logger.info("-" * 40)

        from src.agents.nightly_auditor import NightlyAuditor

        auditor = NightlyAuditor(llm_client=self.grok_scanner)

        # Récupérer les trades du jour (depuis l'état)
        # Pour le moment, simuler
        if hasattr(self, '_today_trades') and self._today_trades:
            report = await auditor.run_audit(self._today_trades)
            logger.info(f"  Trades analyzed: {report.trades_analyzed}")
            logger.info(f"  Win rate: {report.win_rate:.0%}")
            logger.info(f"  New rules: {len(report.new_guidelines)}")
        else:
            logger.info("  No trades to audit today")

        # Afficher les guidelines actuelles
        guidelines = auditor.get_learned_guidelines()
        if guidelines:
            logger.info("")
            logger.info(f"  Active guidelines: {len(guidelines)}")
            for g in guidelines[:3]:
                logger.info(f"    - {g.rule} ({g.confidence:.0%} confidence)")

    async def run_live(self, paper_trading: bool = True):
        """Mode live: boucle event-driven temps réel"""
        logger.info("🔴 LIVE MODE - Event-Driven Trading Loop")
        logger.info("=" * 60)
        logger.info(f"Paper Trading: {'YES' if paper_trading else 'NO - REAL MONEY!'}")
        logger.info("")

        from src.agents.live_loop import LiveLoop, LiveLoopConfig

        # Configuration
        config = LiveLoopConfig(
            heat_poll_interval=int(os.getenv('HEAT_POLL_INTERVAL', 60)),
            screening_interval=int(os.getenv('SCREENING_INTERVAL', 300)),
            paper_trading=paper_trading,
            notification_callback=self._send_notification
        )

        # Créer et initialiser la boucle
        live_loop = LiveLoop(config)
        await live_loop.initialize()

        # Callbacks
        live_loop.set_signal_callback(self._on_signal)
        live_loop.set_trade_callback(self._on_trade)

        logger.info("Starting live loop...")
        logger.info("Press Ctrl+C to stop")
        logger.info("")

        try:
            await live_loop.start()
        except KeyboardInterrupt:
            logger.info("\nStopping live loop...")
            await live_loop.stop()

    async def _send_notification(self, message: str):
        """Envoie une notification (Telegram, etc.)"""
        logger.info(f"📱 {message}")

        # TODO: Intégrer avec notification_manager pour Telegram/Email
        try:
            from src.utils.notification_manager import NotificationManager
            notif = NotificationManager()
            await notif.send_message(message)
        except Exception:
            pass  # Notification manager optionnel

    async def _on_signal(self, alert: dict):
        """Callback quand un signal est détecté"""
        symbol = alert.get('symbol', 'UNKNOWN')
        signal = alert.get('signal', 'UNKNOWN')
        confidence = alert.get('confidence_score', 0)

        logger.info(f"🎯 SIGNAL: {symbol} - {signal} (confidence: {confidence:.0f})")

    async def _on_trade(self, alert: dict, decision):
        """Callback quand un trade est exécuté"""
        symbol = alert.get('symbol', 'UNKNOWN')
        logger.info(f"💰 TRADE EXECUTED: {symbol}")

        # Enregistrer pour l'audit
        if not hasattr(self, '_today_trades'):
            self._today_trades = []
        self._today_trades.append({
            'alert': alert,
            'decision_id': decision.get('id') if isinstance(decision, dict) else (decision.decision_id if hasattr(decision, 'decision_id') else None),
            'timestamp': datetime.now().isoformat()
        })

    async def run_test(self):
        """Mode test: démo rapide"""
        logger.info("🧪 Running Test Mode...")
        logger.info("-" * 40)

        # Test rapide de chaque composant
        logger.info("1. Testing Social Scanner...")
        result = await self.social_scanner.full_scan()
        logger.info(f"   ✅ {result.total_mentions} mentions found")

        if self.grok_scanner and self.grok_scanner.is_available():
            logger.info("2. Testing Grok Scanner...")
            insight = await self.grok_scanner.analyze_symbol('NVDA')
            if insight:
                logger.info(f"   ✅ NVDA sentiment: {insight.sentiment}")
        else:
            logger.info("2. Grok Scanner: ⚠️ Not configured")

        logger.info("3. Testing Stock Discovery...")
        stats = self.stock_discovery.get_universe_stats()
        logger.info(f"   ✅ Universe: {stats['total']} stocks")

        logger.info("4. Testing IBKR Executor...")
        account = await self.executor.get_account_info()
        if account:
            logger.info(f"   ✅ Account: {account.account_id}, Cash: ${account.total_cash:,.2f}")
        else:
            logger.info("   ⚠️ Not connected")

        logger.info("")
        logger.info("✅ All tests passed!")

    async def shutdown(self):
        """Ferme tous les composants"""
        logger.info("Shutting down...")

        if self.social_scanner:
            await self.social_scanner.close()
        if self.grok_scanner:
            await self.grok_scanner.close()
        if self.stock_discovery:
            await self.stock_discovery.close()
        if self.executor:
            await self.executor.disconnect()

        logger.info("Goodbye!")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description='TradingBot V4 Agent')
    parser.add_argument(
        '--mode',
        choices=['live', 'discovery', 'analysis', 'trading', 'full', 'test'],
        default='live',
        help='Mode d\'exécution (live = boucle event-driven recommandée)'
    )
    parser.add_argument(
        '--mock',
        action='store_true',
        help='Utiliser le mock IBKR (pas de trading réel)'
    )
    parser.add_argument(
        '--paper',
        action='store_true',
        default=True,
        help='Paper trading (défaut: True)'
    )
    parser.add_argument(
        '--real',
        action='store_true',
        help='Trading réel (ATTENTION: argent réel!)'
    )

    args = parser.parse_args()

    # Déterminer si paper trading
    paper_trading = not args.real

    # Forcer mock en mode test
    use_mock = args.mock or args.mode == 'test'

    # En mode live, pas besoin d'initialiser tous les composants legacy
    if args.mode == 'live':
        logger.info("=" * 60)
        logger.info("🤖 TradingBot V4 - Live Event-Driven Mode")
        logger.info("=" * 60)

        capital = float(os.getenv('TRADING_CAPITAL', 1500))
        logger.info(f"Capital: {capital}€")
        logger.info(f"Paper Trading: {'YES' if paper_trading else 'NO - REAL MONEY!'}")
        logger.info("")

        runner = AgentRunner(use_mock=use_mock)

        try:
            # Initialisation minimale pour le mode live
            await runner.run_live(paper_trading=paper_trading)
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user")
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            raise
        finally:
            await runner.shutdown()

        return

    # Modes legacy (one-shot)
    runner = AgentRunner(use_mock=use_mock)

    try:
        await runner.initialize()

        if args.mode == 'discovery':
            await runner.run_discovery()
        elif args.mode == 'analysis':
            await runner.run_analysis()
        elif args.mode == 'trading':
            await runner.run_trading()
        elif args.mode == 'full':
            await runner.run_full_cycle()
        elif args.mode == 'test':
            await runner.run_test()

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        await runner.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
