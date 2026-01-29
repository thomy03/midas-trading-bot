"""
Script pour générer des exemples visuels d'obliques RSI
Montre comment les obliques baissières sont détectées et matérialisées
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Import des modules du projet
from src.data.market_data import market_data_fetcher
from trendline_analysis.core.trendline_detector import RSITrendlineDetector
from trendline_analysis.core.rsi_breakout_analyzer import RSIBreakoutAnalyzer

def plot_rsi_trendline_example(symbol: str, timeframe: str = 'weekly', save_path: str = None):
    """
    Génère un graphique montrant l'oblique RSI pour un symbole donné

    Args:
        symbol: Symbole de l'action (ex: 'AAPL', 'NVDA')
        timeframe: 'weekly' ou 'daily'
        save_path: Chemin pour sauvegarder l'image (optionnel)
    """
    print(f"\n{'='*80}")
    print(f"Analyse RSI pour {symbol} - Timeframe: {timeframe}")
    print('='*80)

    # Récupérer les données
    if timeframe == 'weekly':
        df = market_data_fetcher.get_historical_data(symbol, period='2y', interval='1wk')
        lookback = 104
    else:
        df = market_data_fetcher.get_historical_data(symbol, period='1y', interval='1d')
        lookback = 252

    if df is None or df.empty:
        print(f"❌ Pas de données disponibles pour {symbol}")
        return

    # Initialiser les analyseurs
    rsi_detector = RSITrendlineDetector()
    rsi_analyzer = RSIBreakoutAnalyzer()

    # Calculer le RSI
    rsi = rsi_detector.calculate_rsi(df)

    # Détecter les pics
    peaks, properties = rsi_detector.detect_peaks(rsi)

    print(f"\n📊 Données disponibles:")
    print(f"   - Périodes: {len(df)}")
    print(f"   - Pics RSI détectés: {len(peaks)}")

    if len(peaks) < 3:
        print(f"❌ Pas assez de pics pour tracer une oblique (minimum 3)")
        return

    # Détecter l'oblique
    trendline = rsi_detector.find_best_trendline(rsi, peaks, lookback_periods=lookback)

    if trendline is None:
        print(f"❌ Aucune oblique valide détectée pour {symbol}")
        return

    # Analyser le breakout
    result = rsi_analyzer.analyze(df, lookback_periods=lookback)

    print(f"\n✅ OBLIQUE RSI DÉTECTÉE:")
    print(f"   - Nombre de pics: {len(trendline.peak_indices)}")
    print(f"   - R² (qualité): {trendline.r_squared:.3f}")
    print(f"   - Pente: {trendline.slope:.4f}")
    print(f"   - Score qualité: {trendline.quality_score:.1f}/100")
    print(f"\n   📍 Pics de l'oblique:")
    for i, (idx, date, value) in enumerate(zip(trendline.peak_indices,
                                                 trendline.peak_dates,
                                                 trendline.peak_values)):
        print(f"      {i+1}. Date: {date.strftime('%Y-%m-%d')} | RSI: {value:.1f}")

    if result and result.has_rsi_breakout:
        print(f"\n🚀 BREAKOUT DÉTECTÉ:")
        print(f"   - Date: {result.rsi_breakout.date.strftime('%Y-%m-%d')}")
        print(f"   - RSI au breakout: {result.rsi_breakout.rsi_value:.1f}")
        print(f"   - Force: {result.rsi_breakout.strength}")
        print(f"   - Age: {result.rsi_breakout.age_in_periods} périodes")
        print(f"   - Signal: {result.signal}")
    elif result and result.has_rsi_trendline:
        print(f"\n👀 OBLIQUE PRÉSENTE (pas encore cassée)")
        print(f"   - Signal: {result.signal}")

    # Créer le graphique
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'{symbol} - Oblique RSI Baissière - {timeframe.upper()}',
                 fontsize=16, fontweight='bold')

    # Subplot 1: Prix
    ax1.plot(df.index, df['Close'], linewidth=2, label='Prix', color='#2196F3')
    ax1.set_ylabel('Prix ($)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Evolution du Prix', fontsize=12)

    # Subplot 2: RSI avec oblique
    ax2.plot(rsi.index, rsi, linewidth=1.5, label='RSI(14)', color='#9C27B0', alpha=0.7)

    # Lignes de surachat/survente
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.3, label='Surachat (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.3, label='Survente (30)')
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.2)

    # Marquer TOUS les pics détectés (en gris clair)
    all_peak_values = rsi.iloc[peaks]
    ax2.scatter(rsi.index[peaks], all_peak_values,
               color='lightgray', s=80, zorder=3, alpha=0.5,
               label=f'Tous les pics ({len(peaks)})')

    # Marquer les pics de l'oblique (en orange)
    peak_dates = trendline.peak_dates
    peak_values_trendline = trendline.peak_values
    ax2.scatter(peak_dates, peak_values_trendline,
               color='#FF9800', s=150, zorder=4, edgecolors='black', linewidths=2,
               label=f'Pics de l\'oblique ({len(peak_dates)})')

    # Annoter chaque pic de l'oblique
    for i, (date, value) in enumerate(zip(peak_dates, peak_values_trendline)):
        ax2.annotate(f'{i+1}',
                    xy=(date, value),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=10,
                    fontweight='bold',
                    color='#FF9800',
                    bbox=dict(boxstyle='circle', facecolor='white', edgecolor='#FF9800', linewidth=2))

    # Tracer l'oblique de résistance
    trendline_x = np.arange(trendline.start_idx, trendline.end_idx + 1)
    trendline_y = trendline.slope * trendline_x + trendline.intercept
    trendline_dates = rsi.index[trendline_x]

    ax2.plot(trendline_dates, trendline_y,
            color='#FF9800', linestyle='--', linewidth=3,
            label=f'Oblique baissière (R²={trendline.r_squared:.2f})',
            zorder=5)

    # Marquer le breakout si présent
    if result and result.has_rsi_breakout:
        breakout_date = result.rsi_breakout.date
        breakout_value = result.rsi_breakout.rsi_value
        ax2.scatter([breakout_date], [breakout_value],
                   color='#4CAF50', s=300, zorder=6,
                   marker='*', edgecolors='black', linewidths=2,
                   label=f'BREAKOUT ({result.rsi_breakout.strength})')

        ax2.annotate('🚀 BREAKOUT',
                    xy=(breakout_date, breakout_value),
                    xytext=(20, -20),
                    textcoords='offset points',
                    ha='left',
                    fontsize=11,
                    fontweight='bold',
                    color='#4CAF50',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                             edgecolor='#4CAF50', linewidth=2),
                    arrowprops=dict(arrowstyle='->', color='#4CAF50', linewidth=2))

    ax2.set_ylabel('RSI', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    ax2.set_title('RSI avec Oblique de Résistance Baissière', fontsize=12)

    # Ajouter zone de validation (entre les pics, le RSI ne doit pas toucher l'oblique)
    for i in range(len(trendline.peak_indices) - 1):
        start_idx = trendline.peak_indices[i]
        end_idx = trendline.peak_indices[i + 1]

        # Zone entre deux pics
        zone_x = rsi.index[start_idx:end_idx+1]
        zone_y_min = [0] * len(zone_x)
        zone_y_max = trendline.slope * np.arange(start_idx, end_idx+1) + trendline.intercept

        ax2.fill_between(zone_x, zone_y_min, zone_y_max,
                        alpha=0.05, color='red',
                        label='Zone interdite' if i == 0 else '')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Graphique sauvegardé: {save_path}")
    else:
        plt.savefig(f'rsi_oblique_{symbol}_{timeframe}.png', dpi=150, bbox_inches='tight')
        print(f"\n💾 Graphique sauvegardé: rsi_oblique_{symbol}_{timeframe}.png")

    plt.close()

    return result


def generate_multiple_examples():
    """Génère plusieurs exemples sur différentes actions"""

    # Liste d'actions à analyser
    symbols_to_test = [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'NVDA',   # NVIDIA
        'TSLA',   # Tesla
        'GOOGL',  # Alphabet
        'META',   # Meta
        'AMD',    # AMD
        'NFLX',   # Netflix
    ]

    print("\n" + "="*80)
    print("GÉNÉRATION D'EXEMPLES D'OBLIQUES RSI")
    print("="*80)
    print(f"\nAnalyse de {len(symbols_to_test)} actions...")
    print("Recherche d'obliques RSI baissières avec au moins 3 points de contact\n")

    results = []

    for symbol in symbols_to_test:
        try:
            # Tester sur weekly d'abord
            result = plot_rsi_trendline_example(symbol, 'weekly',
                                               f'rsi_oblique_{symbol}_weekly.png')
            if result:
                results.append({
                    'symbol': symbol,
                    'timeframe': 'weekly',
                    'has_trendline': result.has_rsi_trendline,
                    'has_breakout': result.has_rsi_breakout,
                    'signal': result.signal
                })
        except Exception as e:
            print(f"❌ Erreur pour {symbol}: {e}")

    # Résumé final
    print("\n" + "="*80)
    print("RÉSUMÉ DES EXEMPLES GÉNÉRÉS")
    print("="*80)

    obliques_found = [r for r in results if r['has_trendline']]
    breakouts_found = [r for r in results if r['has_breakout']]

    print(f"\n📊 Statistiques:")
    print(f"   - Actions analysées: {len(symbols_to_test)}")
    print(f"   - Obliques détectées: {len(obliques_found)}")
    print(f"   - Breakouts détectés: {len(breakouts_found)}")

    if obliques_found:
        print(f"\n✅ Actions avec obliques RSI:")
        for r in obliques_found:
            status = "🚀 BREAKOUT" if r['has_breakout'] else "👀 En formation"
            print(f"   - {r['symbol']}: {status} - Signal: {r['signal']}")

    if breakouts_found:
        print(f"\n🚀 Actions avec BREAKOUT confirmé:")
        for r in breakouts_found:
            print(f"   - {r['symbol']}: {r['signal']}")

    print("\n" + "="*80)
    print("Graphiques sauvegardés dans le répertoire courant")
    print("="*80)


if __name__ == '__main__':
    # Générer les exemples
    generate_multiple_examples()
