"""
Test de l'intégration du détecteur amélioré dans le screener

Compare les résultats entre :
- Détecteur standard (ancien)
- Détecteur amélioré mode LOW
- Détecteur amélioré mode MEDIUM (par défaut)
- Détecteur amélioré mode HIGH
"""

from src.screening.screener import MarketScreener
from src.utils.logger import logger


def test_single_stock(symbol: str):
    """
    Test un symbole avec les 4 configurations de détecteur

    Args:
        symbol: Symbole à tester
    """
    print(f"\n{'='*80}")
    print(f"TEST INTÉGRATION SCREENER - {symbol}")
    print('='*80)

    configs = [
        ('STANDARD', False, None),
        ('ENHANCED LOW', True, 'low'),
        ('ENHANCED MEDIUM', True, 'medium'),
        ('ENHANCED HIGH', True, 'high'),
    ]

    results = []

    for config_name, use_enhanced, precision_mode in configs:
        print(f"\n🔍 Configuration: {config_name}")
        print(f"   {'─'*60}")

        try:
            # Créer screener avec la configuration
            if use_enhanced:
                screener = MarketScreener(use_enhanced_detector=True, precision_mode=precision_mode)
            else:
                screener = MarketScreener(use_enhanced_detector=False)

            # Screening
            alert = screener.screen_single_stock(symbol, symbol)

            if alert:
                print(f"   ✅ ALERTE DÉTECTÉE:")
                print(f"      - Recommandation: {alert.get('recommendation', 'N/A')}")
                print(f"      - Support level: ${alert.get('support_level', 0):.2f}")
                print(f"      - Distance: {alert.get('distance_to_support_pct', 0):.2f}%")
                print(f"      - Timeframe: {alert.get('timeframe', 'N/A')}")

                if 'has_rsi_breakout' in alert:
                    print(f"      - RSI breakout: {'OUI' if alert['has_rsi_breakout'] else 'NON'}")
                    print(f"      - RSI signal: {alert.get('rsi_signal', 'N/A')}")

                    if alert.get('has_rsi_breakout'):
                        print(f"      - RSI breakout date: {alert.get('rsi_breakout_date', 'N/A')}")
                        print(f"      - RSI breakout strength: {alert.get('rsi_breakout_strength', 'N/A')}")

                if 'rsi_trendline_peaks' in alert:
                    print(f"      - Oblique RSI: {alert['rsi_trendline_peaks']} pics, R²={alert.get('rsi_trendline_r2', 0):.3f}")

                results.append({
                    'config': config_name,
                    'found': True,
                    'recommendation': alert.get('recommendation'),
                    'has_rsi': alert.get('has_rsi_breakout', False),
                    'rsi_peaks': alert.get('rsi_trendline_peaks', 0),
                    'rsi_r2': alert.get('rsi_trendline_r2', 0)
                })
            else:
                print(f"   ❌ Aucune alerte détectée")
                results.append({
                    'config': config_name,
                    'found': False
                })

        except Exception as e:
            print(f"   ⚠️  ERREUR: {e}")
            results.append({
                'config': config_name,
                'found': False,
                'error': str(e)
            })

    # Résumé comparatif
    print(f"\n{'='*80}")
    print(f"RÉSUMÉ COMPARATIF - {symbol}")
    print('='*80)

    print(f"\n📊 Détection d'alertes:")
    for r in results:
        status = "✅" if r['found'] else "❌"
        print(f"   {status} {r['config']:<20} : {'Alerte trouvée' if r['found'] else 'Aucune alerte'}")

    rsi_results = [r for r in results if r['found'] and r.get('has_rsi')]
    if rsi_results:
        print(f"\n🎯 Obliques RSI détectées:")
        for r in rsi_results:
            print(f"   - {r['config']:<20} : {r['rsi_peaks']} pics, R²={r['rsi_r2']:.3f}")

    print()


def test_multiple_stocks():
    """Test sur plusieurs actions"""

    test_symbols = [
        'TSLA',   # Connue pour avoir une bonne oblique
        'MSFT',   # Connue pour avoir une bonne oblique
        'NVDA',   # Connue pour avoir une bonne oblique
        'AAPL',   # Pas d'oblique en HIGH
        'AMD',    # Bonne oblique
    ]

    print("\n" + "="*80)
    print("TEST INTÉGRATION COMPLÈTE - SCREENER AMÉLIORÉ")
    print("="*80)
    print(f"\nTest de {len(test_symbols)} actions")
    print("Configurations testées:")
    print("  1. STANDARD (ancien détecteur)")
    print("  2. ENHANCED LOW (R²>0.35)")
    print("  3. ENHANCED MEDIUM (R²>0.50) ← PAR DÉFAUT")
    print("  4. ENHANCED HIGH (R²>0.65)")

    for symbol in test_symbols:
        test_single_stock(symbol)

    print("\n" + "="*80)
    print("TESTS TERMINÉS")
    print("="*80)


def test_default_screener():
    """Test avec le screener par défaut (singleton)"""

    print("\n" + "="*80)
    print("TEST SCREENER PAR DÉFAUT (Singleton)")
    print("="*80)

    # Importer le singleton
    from src.screening.screener import market_screener

    print(f"\nConfiguration du screener:")
    print(f"  - Détecteur amélioré: {market_screener.use_enhanced_detector}")
    print(f"  - Mode précision: {market_screener.precision_mode}")

    if market_screener.use_enhanced_detector:
        detector_info = market_screener.rsi_analyzer.get_detector_info()
        print(f"\n  Paramètres du détecteur:")
        print(f"    - Min R²: {detector_info['min_r_squared']}")
        print(f"    - Max distance résiduelle: {detector_info['max_residual_distance']}")
        print(f"    - RANSAC: {'Activé' if detector_info['ransac_enabled'] else 'Désactivé'}")
        print(f"    - Prominence adaptative: {'Activée' if detector_info['adaptive_prominence'] else 'Désactivée'}")

    # Test sur TSLA
    print(f"\n📊 Test sur TSLA:")
    alert = market_screener.screen_single_stock('TSLA', 'Tesla Inc')

    if alert:
        print(f"  ✅ Alerte détectée:")
        print(f"     - Recommandation: {alert['recommendation']}")
        print(f"     - RSI breakout: {'OUI' if alert.get('has_rsi_breakout') else 'NON'}")
        if alert.get('rsi_trendline_peaks'):
            print(f"     - Oblique: {alert['rsi_trendline_peaks']} pics, R²={alert.get('rsi_trendline_r2', 0):.3f}")
    else:
        print(f"  ❌ Aucune alerte")

    print()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        # Test sur un symbole spécifique
        symbol = sys.argv[1].upper()
        test_single_stock(symbol)
    else:
        # Test complet
        print("\n🚀 Lancement des tests d'intégration...\n")

        # Test 1: Screener par défaut
        test_default_screener()

        # Test 2: Comparaison sur plusieurs symboles
        test_multiple_stocks()

        print("\n✅ Tous les tests d'intégration sont terminés !")
        print("\n💡 Pour tester un symbole spécifique:")
        print("   python test_screener_integration.py TSLA")
