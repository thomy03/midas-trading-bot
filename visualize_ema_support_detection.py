"""
Visualisation de la détection des supports EMA

Montre comment les croisements d'EMA (24/38/62) créent des niveaux de support
et comment le prix approche ces supports pour générer des signaux d'achat.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from src.data.market_data import market_data_fetcher
from src.indicators.ema_analyzer import ema_analyzer
from config.settings import TIMEFRAMES


def visualize_ema_support_detection(symbol: str, timeframe: str = 'weekly', save_path: str = None):
    """
    Visualise la détection des supports EMA pour un symbole donné

    Args:
        symbol: Symbole de l'action
        timeframe: 'weekly' ou 'daily'
        save_path: Chemin pour sauvegarder (optionnel)
    """
    print(f"\n{'='*80}")
    print(f"DÉTECTION SUPPORTS EMA - {symbol} ({timeframe})")
    print('='*80)

    # Récupérer données
    if timeframe == 'weekly':
        df = market_data_fetcher.get_historical_data(symbol, period='2y', interval=TIMEFRAMES['weekly'])
    else:
        df = market_data_fetcher.get_historical_data(symbol, period='2y', interval=TIMEFRAMES['daily'])

    if df is None or df.empty:
        print(f"❌ Pas de données disponibles pour {symbol}")
        return None

    # Calculer les EMA
    df = ema_analyzer.calculate_emas(df)
    current_price = float(df['Close'].iloc[-1])

    print(f"\n📊 Prix actuel: ${current_price:.2f}")
    print(f"   EMA 24: ${df['EMA_24'].iloc[-1]:.2f}")
    print(f"   EMA 38: ${df['EMA_38'].iloc[-1]:.2f}")
    print(f"   EMA 62: ${df['EMA_62'].iloc[-1]:.2f}")

    # Détecter tous les croisements historiques
    crossovers = ema_analyzer.detect_crossovers(df, timeframe)

    print(f"\n🔍 Croisements EMA détectés: {len(crossovers)}")

    if not crossovers:
        print("❌ Aucun croisement trouvé")
        return None

    # Afficher détails des croisements
    print(f"\n📍 Détails des croisements:")
    for i, cross in enumerate(crossovers[:10]):  # Afficher les 10 premiers
        print(f"   {i+1}. {cross['date'].strftime('%Y-%m-%d')} | "
              f"Type: {cross['type']} | "
              f"EMA{cross['fast_ema']}xEMA{cross['slow_ema']} @ ${cross['price']:.2f} | "
              f"Age: {cross.get('age_in_periods', 0)} périodes")

    # Trouver les niveaux de support historiques
    historical_levels = ema_analyzer.find_historical_support_levels(df, crossovers, current_price)

    print(f"\n🎯 Niveaux de support historiques: {len(historical_levels)}")

    if not historical_levels:
        print("❌ Aucun niveau de support valide trouvé")
        return None

    # Classifier les niveaux par distance
    near_levels = [l for l in historical_levels if l['is_near']]
    far_levels = [l for l in historical_levels if not l['is_near']]

    print(f"\n   Niveaux PROCHES (< 8%): {len(near_levels)}")
    print(f"   Niveaux éloignés (> 8%): {len(far_levels)}")

    # Afficher niveaux proches
    if near_levels:
        print(f"\n✅ NIVEAUX DE SUPPORT PROCHES (signaux potentiels):")
        for i, level in enumerate(near_levels):
            cross = level['crossover_info']
            print(f"   {i+1}. Support: ${level['level']:.2f} | "
                  f"Distance: {level['distance_pct']:.2f}% | "
                  f"Force: {level['strength']:.1f}% | "
                  f"Type: {cross['type']} | "
                  f"Date: {cross['date'].strftime('%Y-%m-%d')}")

    # Afficher niveaux éloignés (top 5)
    if far_levels:
        print(f"\n⏳ NIVEAUX ÉLOIGNÉS (pas encore de signal):")
        for i, level in enumerate(far_levels[:5]):
            cross = level['crossover_info']
            print(f"   {i+1}. Support: ${level['level']:.2f} | "
                  f"Distance: {level['distance_pct']:.2f}% | "
                  f"Date: {cross['date'].strftime('%Y-%m-%d')}")

    # Créer graphique
    plot_ema_support_analysis(
        df, symbol, timeframe, crossovers, historical_levels,
        current_price, near_levels, save_path
    )

    return {
        'symbol': symbol,
        'current_price': current_price,
        'crossovers': crossovers,
        'historical_levels': historical_levels,
        'near_levels': near_levels,
        'far_levels': far_levels
    }


def plot_ema_support_analysis(
    df, symbol, timeframe, crossovers, historical_levels,
    current_price, near_levels, save_path=None
):
    """Génère un graphique détaillé de l'analyse EMA"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})

    fig.suptitle(
        f'{symbol} - Détection Supports EMA (Croisements 24/38/62) - {timeframe.upper()}',
        fontsize=16, fontweight='bold'
    )

    # ========== GRAPHIQUE 1: Prix + EMA + Supports ==========
    ax1.plot(df.index, df['Close'], linewidth=2.5, label='Prix', color='#2196F3', zorder=5)

    # Tracer les 3 EMA
    ax1.plot(df.index, df['EMA_24'], linewidth=2, label='EMA 24', color='#FF5722', alpha=0.8)
    ax1.plot(df.index, df['EMA_38'], linewidth=2, label='EMA 38', color='#FF9800', alpha=0.8)
    ax1.plot(df.index, df['EMA_62'], linewidth=2, label='EMA 62', color='#FFC107', alpha=0.8)

    # Marquer tous les croisements
    for cross in crossovers:
        cross_date = cross['date']
        cross_price = cross['price']

        # Couleur selon type de croisement
        if cross['type'] == 'bullish':
            color = '#4CAF50'
            marker = '^'
            size = 120
        else:
            color = '#F44336'
            marker = 'v'
            size = 100

        ax1.scatter([cross_date], [cross_price],
                   color=color, marker=marker, s=size,
                   edgecolors='black', linewidths=1.5,
                   zorder=10, alpha=0.7)

    # Tracer les niveaux de support historiques
    for i, level in enumerate(historical_levels):
        support_price = level['level']
        is_near = level['is_near']

        # Style selon proximité
        if is_near:
            color = '#4CAF50'
            linestyle = '-'
            linewidth = 2.5
            alpha = 0.8
            label_prefix = '🎯 PROCHE'
        else:
            color = '#9E9E9E'
            linestyle = '--'
            linewidth = 1.5
            alpha = 0.4
            label_prefix = 'Éloigné'

        # Ligne horizontale de support
        ax1.axhline(y=support_price, color=color, linestyle=linestyle,
                   linewidth=linewidth, alpha=alpha, zorder=3)

        # Annotation
        cross = level['crossover_info']
        annotation_text = f"{label_prefix}: ${support_price:.2f} ({level['distance_pct']:.1f}%)\n" \
                         f"{cross['type']} {cross['date'].strftime('%Y-%m-%d')}"

        # Position de l'annotation (à droite du graphique)
        ax1.text(df.index[-1], support_price, f"  {annotation_text}",
                verticalalignment='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3),
                color='black', fontweight='bold' if is_near else 'normal')

    # Ligne de prix actuel
    ax1.axhline(y=current_price, color='#2196F3', linestyle='-',
               linewidth=3, label=f'Prix actuel: ${current_price:.2f}', zorder=6)

    ax1.set_ylabel('Prix ($)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Prix avec EMA et Niveaux de Support', fontsize=12, fontweight='bold')

    # ========== GRAPHIQUE 2: Distances aux supports ==========
    if historical_levels:
        # Trier par distance
        sorted_levels = sorted(historical_levels, key=lambda x: abs(x['distance_pct']))[:10]

        support_labels = []
        distances = []
        colors_bar = []

        for level in sorted_levels:
            support_labels.append(f"${level['level']:.2f}")
            distances.append(level['distance_pct'])
            colors_bar.append('#4CAF50' if level['is_near'] else '#9E9E9E')

        bars = ax2.barh(support_labels, distances, color=colors_bar, alpha=0.7, edgecolor='black')

        # Ligne de seuil 8%
        ax2.axvline(x=8.0, color='red', linestyle='--', linewidth=2, label='Seuil proximité (8%)')
        ax2.axvline(x=-8.0, color='red', linestyle='--', linewidth=2)

        # Zone verte (proximité)
        ax2.axvspan(-8, 8, alpha=0.1, color='green', label='Zone de signal')

        ax2.set_xlabel('Distance au prix actuel (%)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Niveau de support', fontsize=11, fontweight='bold')
        ax2.set_title('Distance des Supports au Prix Actuel (Top 10)', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='x')

        # Annotations de distance
        for bar, dist in zip(bars, distances):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f' {dist:.1f}%',
                    ha='left' if width > 0 else 'right',
                    va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Graphique sauvegardé: {save_path}")
    else:
        filename = f'ema_support_{symbol}_{timeframe}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n💾 Graphique sauvegardé: {filename}")

    plt.close()


def generate_multiple_examples():
    """Génère plusieurs exemples sur différentes actions"""

    symbols_to_test = [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'NVDA',   # NVIDIA
        'TSLA',   # Tesla
        'GOOGL',  # Alphabet
        'META',   # Meta
        'AMD',    # AMD
        'AMZN',   # Amazon
        'NFLX',   # Netflix
        'CRM',    # Salesforce
    ]

    print("\n" + "="*80)
    print("GÉNÉRATION D'EXEMPLES - DÉTECTION SUPPORTS EMA")
    print("="*80)
    print(f"\nAnalyse de {len(symbols_to_test)} actions")
    print("Recherche de croisements EMA (24/38/62) formant des supports\n")

    results = []

    for symbol in symbols_to_test:
        try:
            result = visualize_ema_support_detection(
                symbol, 'weekly',
                f'ema_support_{symbol}_weekly.png'
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Erreur pour {symbol}: {e}")

    # Résumé final
    print("\n" + "="*80)
    print("RÉSUMÉ DES DÉTECTIONS")
    print("="*80)

    total_analyzed = len(results)
    with_near_levels = sum(1 for r in results if r['near_levels'])

    print(f"\n📊 Statistiques:")
    print(f"   - Actions analysées: {total_analyzed}")
    print(f"   - Actions avec supports PROCHES (< 8%): {with_near_levels}")
    print(f"   - Taux de signaux: {with_near_levels/total_analyzed*100:.1f}%")

    if with_near_levels > 0:
        print(f"\n✅ Actions avec signaux d'achat potentiels (support proche):")
        for r in results:
            if r['near_levels']:
                best_level = min(r['near_levels'], key=lambda x: abs(x['distance_pct']))
                print(f"   - {r['symbol']}: "
                      f"{len(r['near_levels'])} support(s) proche(s) | "
                      f"Meilleur: ${best_level['level']:.2f} "
                      f"({best_level['distance_pct']:.2f}%) | "
                      f"Force: {best_level['strength']:.0f}%")

    # Statistiques globales
    all_crossovers = sum(len(r['crossovers']) for r in results)
    all_levels = sum(len(r['historical_levels']) for r in results)
    all_near = sum(len(r['near_levels']) for r in results)

    print(f"\n📈 Statistiques globales:")
    print(f"   - Croisements totaux détectés: {all_crossovers}")
    print(f"   - Niveaux de support historiques: {all_levels}")
    print(f"   - Supports proches (signaux): {all_near}")
    print(f"   - Taux de conversion: {all_near/all_levels*100:.1f}% des supports sont proches")

    print("\n" + "="*80)
    print("Graphiques sauvegardés dans le répertoire courant")
    print("="*80)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        # Test sur un symbole spécifique
        symbol = sys.argv[1].upper()
        visualize_ema_support_detection(symbol, 'weekly')
    else:
        # Générer tous les exemples
        generate_multiple_examples()
