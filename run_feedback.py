#!/usr/bin/env python3
"""
Script pour exécuter le Feedback Loop manuellement ou via cron.

Usage:
    python run_feedback.py              # Exécuter le cycle de feedback
    python run_feedback.py --summary    # Voir le résumé de l'apprentissage
    python run_feedback.py --top        # Voir les meilleurs indicateurs
"""

import asyncio
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.learning.feedback_loop import get_feedback_loop, run_daily_feedback
from src.learning.market_learner import get_market_learner


async def main():
    args = sys.argv[1:]
    
    if "--summary" in args:
        learner = get_market_learner()
        summary = learner.get_learning_summary()
        print("\n📊 RÉSUMÉ DE L'APPRENTISSAGE")
        print("=" * 40)
        print(f"Status: {summary['status']}")
        print(f"Indicateurs suivis: {summary.get('total_indicators', 0)}")
        print(f"Signaux analysés: {summary.get('total_signals_analyzed', 0)}")
        print(f"Accuracy moyenne: {summary.get('average_accuracy', 0.5):.1%}")
        print("\n🏆 Top indicateurs:")
        for ind in summary.get('top_indicators', []):
            print(f"  • {ind['name']}: accuracy={ind['accuracy']:.1%}, poids={ind['weight']:.2f}")
        return
    
    if "--top" in args:
        loop = get_feedback_loop()
        top = loop.get_top_indicators(10)
        print("\n🏆 TOP 10 INDICATEURS PAR ACCURACY")
        print("=" * 50)
        for i, ind in enumerate(top, 1):
            print(f"{i}. {ind['indicator']}")
            print(f"   Accuracy: {ind['accuracy']:.1%} | Poids: {ind['weight']:.2f} | Signaux: {ind['total_signals']}")
        return
    
    # Exécuter le cycle de feedback
    print("\n🔄 EXÉCUTION DU FEEDBACK LOOP")
    print("=" * 40)
    print("Analyse des mouvements du marché...")
    
    result = await run_daily_feedback()
    
    print(f"\n✅ Cycle terminé!")
    print(f"Date: {result.get('date', 'N/A')}")
    print(f"Gainers analysés: {result.get('gainers_analyzed', 0)}")
    print(f"Losers analysés: {result.get('losers_analyzed', 0)}")
    
    if result.get('top_gainer'):
        g = result['top_gainer']
        print(f"\n📈 Top Gainer: {g['symbol']} (+{g['pct_change']}%)")
    
    if result.get('top_loser'):
        l = result['top_loser']
        print(f"📉 Top Loser: {l['symbol']} ({l['pct_change']}%)")
    
    reinforced = result.get('indicators_reinforced', [])
    penalized = result.get('indicators_penalized', [])
    
    if reinforced:
        print(f"\n✅ Indicateurs renforcés ({len(reinforced)}): {', '.join(reinforced[:5])}...")
    if penalized:
        print(f"⚠️  Indicateurs pénalisés ({len(penalized)}): {', '.join(penalized[:5])}...")


if __name__ == "__main__":
    asyncio.run(main())
