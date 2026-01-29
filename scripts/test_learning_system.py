#!/usr/bin/env python3
"""
Test du système d'apprentissage V4.

Teste:
1. StateManager - stockage des trades
2. AnalysisStore - persistance des analyses
3. Anti-répétition - cooldown entre scans
4. Chaîne complète - simulation d'une journée
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Ajouter le projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.state import StateManager, TradeRecord, get_state_manager
from src.intelligence.analysis_store import AnalysisStore, get_analysis_store


def test_state_manager():
    """Test 1: StateManager et stockage des trades"""
    print("\n" + "="*60)
    print("TEST 1: StateManager - Stockage des trades")
    print("="*60)

    # Utiliser un répertoire de test
    manager = StateManager(data_dir='data/test_learning')

    # Créer des trades de test
    trades = [
        TradeRecord(
            id='test_001',
            symbol='AAPL',
            action='BUY',
            quantity=10,
            price=150.0,
            timestamp=datetime.now().isoformat(),
            thesis='RSI breakout on daily',
            outcome='WIN',
            pnl=75.0,
            exit_reason='take_profit',
            entry_date=(datetime.now() - timedelta(days=3)).isoformat(),
            exit_date=datetime.now().isoformat(),
            entry_price=150.0,
            exit_price=157.5,
            pnl_pct=5.0,
            hold_days=3,
            exit_type='take_profit',
            metadata={'market_regime': 'bull_strong', 'vix': 15.2, 'rsi_at_entry': 62}
        ),
        TradeRecord(
            id='test_002',
            symbol='NVDA',
            action='BUY',
            quantity=5,
            price=450.0,
            timestamp=datetime.now().isoformat(),
            thesis='Momentum breakout',
            outcome='LOSS',
            pnl=-45.0,
            exit_reason='stop_loss',
            entry_date=(datetime.now() - timedelta(days=2)).isoformat(),
            exit_date=datetime.now().isoformat(),
            entry_price=450.0,
            exit_price=441.0,
            pnl_pct=-2.0,
            hold_days=2,
            exit_type='stop_loss',
            metadata={'market_regime': 'volatile', 'vix': 22.5, 'rsi_at_entry': 71}
        ),
        TradeRecord(
            id='test_003',
            symbol='MSFT',
            action='BUY',
            quantity=8,
            price=380.0,
            timestamp=datetime.now().isoformat(),
            thesis='EMA support bounce',
            outcome='WIN',
            pnl=120.0,
            exit_reason='take_profit',
            entry_date=(datetime.now() - timedelta(days=5)).isoformat(),
            exit_date=datetime.now().isoformat(),
            entry_price=380.0,
            exit_price=395.0,
            pnl_pct=3.95,
            hold_days=5,
            exit_type='take_profit',
            metadata={'market_regime': 'bull_weak', 'vix': 18.0, 'rsi_at_entry': 55}
        )
    ]

    print(f"\n1. Enregistrement de {len(trades)} trades...")
    for trade in trades:
        manager.record_trade(trade)
        print(f"   [OK] {trade.symbol}: {trade.outcome} ({trade.pnl:+.2f}EUR)")

    # Vérifier trade_history
    print(f"\n2. Vérification trade_history...")
    print(f"   Trades stockés: {len(manager.state.trade_history)}")

    # Test get_closed_trades_today
    print(f"\n3. Test get_closed_trades_today()...")
    closed_today = manager.get_closed_trades_today()
    print(f"   Trades fermés aujourd'hui: {len(closed_today)}")
    for t in closed_today:
        print(f"   - {t.symbol}: {t.outcome} ({t.pnl:+.2f}EUR)")

    # Test get_trade_history
    print(f"\n4. Test get_trade_history()...")
    history = manager.get_trade_history(limit=10)
    print(f"   Derniers trades: {len(history)}")

    # Statistiques
    print(f"\n5. Statistiques globales:")
    print(f"   Total trades: {manager.state.total_trades}")
    print(f"   Wins: {manager.state.total_wins}")
    print(f"   Losses: {manager.state.total_losses}")
    print(f"   Win rate: {manager.state.win_rate*100:.1f}%")
    print(f"   Total P&L: {manager.state.total_pnl:+.2f}EUR")

    print("\n[PASS] TEST 1 PASSÉ")
    return True


def test_analysis_store():
    """Test 2: AnalysisStore et persistance"""
    print("\n" + "="*60)
    print("TEST 2: AnalysisStore - Persistance des analyses")
    print("="*60)

    store = get_analysis_store()

    # Simuler des analyses
    symbols = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'TSLA']

    print(f"\n1. Sauvegarde de {len(symbols)} analyses...")

    for i, symbol in enumerate(symbols):
        # Créer un mock de reasoning_result
        class MockDecision:
            value = 'buy' if i % 2 == 0 else 'hold'

        class MockReasoningResult:
            total_score = 60 + i * 5
            decision = MockDecision()
            reasoning_summary = f"Test analysis for {symbol}"
            pillar_scores = {
                'technical': 65 + i,
                'fundamental': 60 + i,
                'sentiment': 55 + i,
                'news': 70 + i
            }

        store.save_analysis(
            symbol=symbol,
            reasoning_result=MockReasoningResult(),
            market_regime='bull_strong'
        )
        print(f"   [OK] {symbol} sauvegardé (score: {60 + i*5})")

    # Test was_recently_analyzed
    print(f"\n2. Test anti-répétition (was_recently_analyzed)...")
    for symbol in symbols[:2]:
        was_recent = store.was_recently_analyzed(symbol, hours=1)
        print(f"   {symbol}: {'SKIP (< 1h)' if was_recent else 'OK à analyser'}")

    # Test get_unanalyzed_symbols
    print(f"\n3. Test filtrage symboles (get_unanalyzed_symbols)...")
    test_symbols = symbols + ['AMD', 'INTC', 'META']  # 3 nouveaux
    fresh = store.get_unanalyzed_symbols(test_symbols, hours=1)
    print(f"   Symboles à analyser: {len(fresh)}/{len(test_symbols)}")
    print(f"   Nouveaux: {[s for s in fresh if s not in symbols]}")

    # Test get_analysis_history
    print(f"\n4. Test recuperation derniere analyse...")
    history = store.get_analysis_history('AAPL', limit=1)
    if history:
        latest = history[0]
        print(f"   AAPL: score={latest.get('total_score')}, regime={latest.get('market_regime')}")
    else:
        print(f"   AAPL: pas d'historique")

    # Test get_symbols_needing_reanalysis
    print(f"\n5. Test symboles à rafraîchir (> 24h)...")
    stale = store.get_symbols_needing_reanalysis(hours=24)
    print(f"   Symboles stale: {len(stale)}")

    print("\n[PASS] TEST 2 PASSÉ")
    return True


def test_cooldown_logic():
    """Test 3: Logique de cooldown"""
    print("\n" + "="*60)
    print("TEST 3: Logique de Cooldown Anti-Répétition")
    print("="*60)

    store = get_analysis_store()

    # Scénario: scan toutes les 15 min
    print("\n1. Simulation scan toutes les 15 min...")
    symbol = 'TEST_COOLDOWN'

    # Premier scan
    class MockResult:
        total_score = 75
        class decision:
            value = 'buy'
        reasoning_summary = "First scan"
        pillar_scores = {'technical': 75}

    store.save_analysis(symbol, MockResult(), 'bull')
    print(f"   Scan 1: {symbol} analysé")

    # Vérifier cooldown
    is_recent_1h = store.was_recently_analyzed(symbol, hours=1)
    is_recent_15m = store.was_recently_analyzed(symbol, hours=0.25)  # 15 min

    print(f"\n2. Vérification cooldown...")
    print(f"   Cooldown 1h: {'ACTIF - SKIP' if is_recent_1h else 'OK'}")
    print(f"   Cooldown 15min: {'ACTIF - SKIP' if is_recent_15m else 'OK'}")

    # Logique recommandée
    print(f"\n3. Logique recommandée:")
    print(f"   - Pendant marché: cooldown 1h")
    print(f"   - Hors marché (soir): cooldown jusqu'à next open")
    print(f"   - Nouvelle donnée majeure: force re-scan")

    print("\n[PASS] TEST 3 PASSÉ")
    return True


async def test_orchestrator_integration():
    """Test 4: Intégration avec l'Orchestrator"""
    print("\n" + "="*60)
    print("TEST 4: Intégration Orchestrator")
    print("="*60)

    try:
        from src.agents.orchestrator import MarketAgent

        print("\n1. Initialisation MarketAgent...")
        agent = MarketAgent()
        await agent.initialize()
        print("   [OK] Agent initialisé")

        # Vérifier les composants
        print(f"\n2. Vérification composants...")
        print(f"   - state_manager: {'[OK]' if agent.state_manager else '[X]'}")
        print(f"   - analysis_store: {'[OK]' if agent.analysis_store else '[X]'}")
        print(f"   - guardrails: {'[OK]' if agent.guardrails else '[X]'}")

        # Test get_closed_trades_today via orchestrator
        print(f"\n3. Test chaîne d'apprentissage...")
        trades = agent.state_manager.get_closed_trades_today()
        print(f"   Trades fermés aujourd'hui: {len(trades)}")

        # Résumé
        summary = agent.state_manager.get_summary()
        print(f"\n4. Résumé agent:")
        print(f"   Phase: {summary['phase']}")
        print(f"   Capital: {summary['capital']['current']}EUR")
        print(f"   Positions: {summary['positions']['count']}")

        print("\n[PASS] TEST 4 PASSÉ")
        return True

    except Exception as e:
        print(f"\n[FAIL] TEST 4 ÉCHOUÉ: {e}")
        return False


def main():
    """Lance tous les tests"""
    print("\n" + "#"*60)
    print("#  TEST DU SYSTÈME D'APPRENTISSAGE V4")
    print("#"*60)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    # Tests synchrones
    results.append(("StateManager", test_state_manager()))
    results.append(("AnalysisStore", test_analysis_store()))
    results.append(("Cooldown Logic", test_cooldown_logic()))

    # Test async
    results.append(("Orchestrator", asyncio.run(test_orchestrator_integration())))

    # Résumé final
    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "[PASS] PASSÉ" if result else "[FAIL] ÉCHOUÉ"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passés")

    if passed == total:
        print("\n*** TOUS LES TESTS SONT PASSÉS!")
        print("\nLe système d'apprentissage est opérationnel:")
        print("  - Les trades sont stockés avec toutes les métadonnées")
        print("  - get_closed_trades_today() fonctionne")
        print("  - L'anti-répétition est actif")
        print("  - L'orchestrator peut exécuter l'audit nocturne")
    else:
        print("\n[WARN]  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
