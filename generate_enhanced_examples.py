"""
Générateur d'exemples d'obliques RSI avec le détecteur amélioré

Montre la qualité et la précision des obliques détectées avec :
- RANSAC
- Prominence adaptative
- Validation stricte de distance
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from src.data.market_data import market_data_fetcher
from trendline_analysis.core.enhanced_trendline_detector import EnhancedRSITrendlineDetector


def generate_enhanced_example(
    symbol: str,
    timeframe: str = 'weekly',
    precision_mode: str = 'high',
    save_path: str = None
):
    """
    Génère un exemple d'oblique RSI avec le détecteur amélioré

    Args:
        symbol: Symbole de l'action
        timeframe: 'weekly' ou 'daily'
        precision_mode: 'high', 'medium', ou 'low'
        save_path: Chemin pour sauvegarder (optionnel)
    """
    print(f"\n{'='*80}")
    print(f"DÉTECTION OBLIQUE RSI HAUTE PRÉCISION - {symbol} ({timeframe})")
    print('='*80)

    # Récupérer données
    if timeframe == 'weekly':
        df = market_data_fetcher.get_historical_data(symbol, period='2y', interval='1wk')
        lookback = 104
    else:
        df = market_data_fetcher.get_historical_data(symbol, period='1y', interval='1d')
        lookback = 252

    if df is None or df.empty:
        print(f"❌ Pas de données disponibles pour {symbol}")
        return None

    # Initialiser détecteur amélioré
    detector = EnhancedRSITrendlineDetector(
        precision_mode=precision_mode,
        use_ransac=True,
        adaptive_prominence=True
    )

    print(f"\n⚙️  CONFIGURATION:")
    print(f"   Mode précision: {precision_mode.upper()}")
    print(f"   RANSAC: Activé")
    print(f"   Prominence adaptative: Activée")

    # Calculer RSI
    rsi = detector.calculate_rsi(df)

    # Détecter pics avec prominence adaptative
    peaks, properties = detector.detect_peaks_adaptive(rsi)
    prominence_used = properties.get('used_prominence', 'N/A')
    rsi_volatility = rsi.std()

    print(f"\n📊 ANALYSE DU RSI:")
    print(f"   Volatilité RSI: {rsi_volatility:.2f}")
    print(f"   Prominence utilisée: {prominence_used:.1f} (adaptée à la volatilité)")
    print(f"   Pics bruts détectés: {len(peaks)}")

    # Filtrage qualité
    filtered_peaks = detector.filter_peaks_by_quality(rsi, peaks)
    print(f"   Pics après filtrage qualité: {len(filtered_peaks)}")

    # Détecter oblique
    trendline = detector.find_best_trendline(rsi, peaks, lookback_periods=lookback)

    if trendline is None:
        print(f"\n❌ AUCUNE OBLIQUE DÉTECTÉE")
        print(f"   Les standards de précision ne sont pas atteints pour {symbol}")
        print(f"   (R² < {detector.min_r_squared}, distance > {detector.max_residual} points)")
        return None

    # Calculer métriques de précision
    x = np.array(trendline.peak_indices)
    y = np.array(trendline.peak_values)
    y_pred = trendline.slope * x + trendline.intercept
    residuals = np.abs(y - y_pred)

    print(f"\n✅ OBLIQUE HAUTE PRÉCISION DÉTECTÉE:")
    print(f"\n   📍 Caractéristiques géométriques:")
    print(f"      - Nombre de pics: {len(trendline.peak_indices)}")
    print(f"      - R² (coefficient de détermination): {trendline.r_squared:.4f}")
    print(f"      - Pente: {trendline.slope:.4f} (descendante)")
    print(f"      - Score qualité global: {trendline.quality_score:.1f}/100")

    print(f"\n   📏 Métriques de précision:")
    print(f"      - Distance moyenne pics/oblique: {np.mean(residuals):.3f} points RSI")
    print(f"      - Distance maximale: {np.max(residuals):.3f} points RSI")
    print(f"      - Écart-type des distances: {np.std(residuals):.3f} points RSI")
    print(f"      - Distance minimale: {np.min(residuals):.3f} points RSI")

    # Validation
    is_precise, reason = detector.validate_trendline_precision(trendline, rsi)
    print(f"\n   ✓ Validation stricte: {reason}")

    print(f"\n   🎯 Points de contact (pics de l'oblique):")
    for i, (idx, date, value) in enumerate(zip(trendline.peak_indices,
                                                 trendline.peak_dates,
                                                 trendline.peak_values)):
        pred_value = trendline.slope * idx + trendline.intercept
        distance = abs(value - pred_value)
        print(f"      {i+1}. {date.strftime('%Y-%m-%d')} | RSI: {value:.2f} | "
              f"Oblique: {pred_value:.2f} | Écart: {distance:.3f}")

    # Durée de l'oblique
    duration_periods = trendline.peak_dates[-1] - trendline.peak_dates[0]
    print(f"\n   ⏱  Durée de l'oblique: {duration_periods.days} jours "
          f"({trendline.peak_dates[0].strftime('%Y-%m-%d')} → "
          f"{trendline.peak_dates[-1].strftime('%Y-%m-%d')})")

    # Créer graphique détaillé
    plot_enhanced_trendline(
        df, rsi, symbol, timeframe, trendline, peaks, filtered_peaks,
        detector, save_path
    )

    return trendline


def plot_enhanced_trendline(
    df, rsi, symbol, timeframe, trendline, peaks, filtered_peaks, detector, save_path=None
):
    """Génère un graphique détaillé de l'oblique détectée"""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 0.5], hspace=0.3, wspace=0.3)

    fig.suptitle(
        f'{symbol} - Oblique RSI Haute Précision (R²={trendline.r_squared:.3f}) - {timeframe.upper()}',
        fontsize=16, fontweight='bold'
    )

    # Subplot 1: Prix
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df.index, df['Close'], linewidth=2.5, label='Prix', color='#2196F3')
    ax1.set_ylabel('Prix ($)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Évolution du Prix', fontsize=12)

    # Marquer les dates des pics RSI sur le graphique prix
    for date in trendline.peak_dates:
        ax1.axvline(x=date, color='orange', alpha=0.2, linestyle='--', linewidth=1)

    # Subplot 2: RSI avec oblique
    ax2 = fig.add_subplot(gs[1, :])

    # RSI
    ax2.plot(rsi.index, rsi, linewidth=2, label='RSI(14)', color='#9C27B0', alpha=0.8)

    # Lignes de référence
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.3, linewidth=1.5, label='Surachat (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.3, linewidth=1.5, label='Survente (30)')
    ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.2, linewidth=1)

    # Tous les pics bruts (gris très clair)
    ax2.scatter(rsi.index[peaks], rsi.iloc[peaks],
               color='#E0E0E0', s=50, zorder=3, alpha=0.4,
               label=f'Pics bruts ({len(peaks)})')

    # Pics filtrés (gris moyen)
    ax2.scatter(rsi.index[filtered_peaks], rsi.iloc[filtered_peaks],
               color='#9E9E9E', s=80, zorder=4, alpha=0.6,
               label=f'Pics filtrés ({len(filtered_peaks)})')

    # Pics de l'oblique (orange vif)
    peak_dates = trendline.peak_dates
    peak_values = trendline.peak_values
    ax2.scatter(peak_dates, peak_values,
               color='#FF9800', s=200, zorder=5, edgecolors='black', linewidths=2.5,
               label=f'Pics de l\'oblique ({len(peak_dates)})', marker='o')

    # Annoter chaque pic de l'oblique avec numéro et distance
    x_vals = np.array(trendline.peak_indices)
    y_vals = np.array(trendline.peak_values)
    y_pred = trendline.slope * x_vals + trendline.intercept
    distances = np.abs(y_vals - y_pred)

    for i, (date, value, dist) in enumerate(zip(peak_dates, peak_values, distances)):
        # Numéro du pic
        ax2.annotate(f'{i+1}',
                    xy=(date, value),
                    xytext=(0, 15),
                    textcoords='offset points',
                    ha='center',
                    fontsize=11,
                    fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='circle,pad=0.3', facecolor='#FF9800',
                             edgecolor='black', linewidth=2))

        # Distance à l'oblique
        ax2.annotate(f'Δ={dist:.2f}',
                    xy=(date, value),
                    xytext=(0, -20),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                    color='#FF6600',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='#FF9800', linewidth=1))

    # Tracer l'oblique
    trendline_x = np.arange(trendline.start_idx, trendline.end_idx + 1)
    trendline_y = trendline.slope * trendline_x + trendline.intercept
    trendline_dates = rsi.index[trendline_x]

    ax2.plot(trendline_dates, trendline_y,
            color='#FF9800', linestyle='--', linewidth=3.5,
            label=f'Oblique résistance', zorder=6, alpha=0.9)

    # Zone entre RSI et oblique (validation)
    for i in range(len(trendline.peak_indices) - 1):
        start_idx = trendline.peak_indices[i]
        end_idx = trendline.peak_indices[i + 1]

        zone_x = rsi.index[start_idx:end_idx+1]
        zone_y_bottom = [0] * len(zone_x)
        zone_y_top = trendline.slope * np.arange(start_idx, end_idx+1) + trendline.intercept

        ax2.fill_between(zone_x, zone_y_bottom, zone_y_top,
                        alpha=0.03, color='red',
                        label='Zone validation' if i == 0 else '')

    ax2.set_ylabel('RSI', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    ax2.set_title('RSI avec Oblique de Résistance Haute Précision', fontsize=12, fontweight='bold')

    # Subplot 3: Métriques (gauche)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis('off')

    metrics_text = f"""
    MÉTRIQUES DE QUALITÉ

    Ajustement statistique:
    • R² (coefficient): {trendline.r_squared:.4f}
    • Score qualité: {trendline.quality_score:.1f}/100
    • Pente: {trendline.slope:.4f}

    Précision géométrique:
    • Distance moyenne: {np.mean(distances):.3f} points RSI
    • Distance max: {np.max(distances):.3f} points RSI
    • Distance min: {np.min(distances):.3f} points RSI
    • Écart-type: {np.std(distances):.3f} points RSI

    Configuration détecteur:
    • Seuils: R²>{detector.min_r_squared}, Dist<{detector.max_residual}
    • Prominence: {rsi.std():.1f} (adaptative)
    • RANSAC: Activé
    """

    ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='wheat', alpha=0.8))

    # Subplot 4: Histogramme des distances (droite)
    ax4 = fig.add_subplot(gs[2, 1])

    ax4.hist(distances, bins=min(len(distances), 10), color='#FF9800', alpha=0.7, edgecolor='black')
    ax4.axvline(x=np.mean(distances), color='red', linestyle='--', linewidth=2,
               label=f'Moyenne: {np.mean(distances):.3f}')
    ax4.axvline(x=detector.max_mean_residual, color='green', linestyle='--', linewidth=2,
               label=f'Seuil max: {detector.max_mean_residual:.1f}')

    ax4.set_xlabel('Distance à l\'oblique (points RSI)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Fréquence', fontsize=10, fontweight='bold')
    ax4.set_title('Distribution des distances pics/oblique', fontsize=10, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Graphique sauvegardé: {save_path}")
    else:
        filename = f'enhanced_oblique_{symbol}_{timeframe}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n💾 Graphique sauvegardé: {filename}")

    plt.close()


def generate_multiple_examples(precision_mode='high'):
    """Génère plusieurs exemples sur différentes actions"""

    symbols_to_test = [
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'NVDA',   # NVIDIA
        'TSLA',   # Tesla
        'GOOGL',  # Alphabet
        'META',   # Meta
        'AMD',    # AMD
        'NFLX',   # Netflix
        'AMZN',   # Amazon
        'CRM',    # Salesforce
    ]

    print("\n" + "="*80)
    print(f"GÉNÉRATION D'EXEMPLES - DÉTECTEUR HAUTE PRÉCISION")
    print(f"Mode: {precision_mode.upper()}")
    print("="*80)

    results = []
    trendlines_found = []

    for symbol in symbols_to_test:
        try:
            trendline = generate_enhanced_example(
                symbol, 'weekly', precision_mode,
                f'enhanced_oblique_{symbol}_weekly.png'
            )
            if trendline:
                trendlines_found.append({
                    'symbol': symbol,
                    'r_squared': trendline.r_squared,
                    'num_peaks': len(trendline.peak_indices),
                    'mean_distance': np.mean(np.abs(
                        np.array(trendline.peak_values) -
                        (trendline.slope * np.array(trendline.peak_indices) + trendline.intercept)
                    )),
                    'quality_score': trendline.quality_score
                })
            results.append({'symbol': symbol, 'found': trendline is not None})
        except Exception as e:
            print(f"❌ Erreur pour {symbol}: {e}")
            results.append({'symbol': symbol, 'found': False})

    # Résumé final
    print("\n" + "="*80)
    print("RÉSUMÉ DES DÉTECTIONS")
    print("="*80)

    found_count = sum(1 for r in results if r['found'])
    total_count = len(results)

    print(f"\n📊 Statistiques globales:")
    print(f"   - Actions analysées: {total_count}")
    print(f"   - Obliques détectées: {found_count} ({found_count/total_count*100:.1f}%)")

    if trendlines_found:
        avg_r2 = np.mean([t['r_squared'] for t in trendlines_found])
        avg_distance = np.mean([t['mean_distance'] for t in trendlines_found])
        avg_quality = np.mean([t['quality_score'] for t in trendlines_found])

        print(f"\n📈 Qualité moyenne des obliques détectées:")
        print(f"   - R² moyen: {avg_r2:.3f}")
        print(f"   - Distance moyenne: {avg_distance:.3f} points RSI")
        print(f"   - Score qualité moyen: {avg_quality:.1f}/100")

        print(f"\n✅ Actions avec obliques haute précision:")
        for t in sorted(trendlines_found, key=lambda x: x['r_squared'], reverse=True):
            print(f"   - {t['symbol']}: R²={t['r_squared']:.3f}, "
                  f"{t['num_peaks']} pics, "
                  f"dist={t['mean_distance']:.3f}, "
                  f"score={t['quality_score']:.1f}")

    print("\n" + "="*80)
    print("Graphiques sauvegardés dans le répertoire courant")
    print("="*80)


if __name__ == '__main__':
    # Générer exemples avec différents modes de précision

    # Mode HIGH (recommandé pour trading)
    generate_multiple_examples(precision_mode='high')

    # Décommenter pour tester d'autres modes:
    # generate_multiple_examples(precision_mode='medium')
    # generate_multiple_examples(precision_mode='low')
