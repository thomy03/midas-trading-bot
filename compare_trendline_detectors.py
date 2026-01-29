"""
Script de comparaison : Détecteur original vs Détecteur amélioré

Compare la précision et la qualité des obliques RSI détectées
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from src.data.market_data import market_data_fetcher
from trendline_analysis.core.trendline_detector import RSITrendlineDetector
from trendline_analysis.core.enhanced_trendline_detector import EnhancedRSITrendlineDetector


def compare_detectors(symbol: str, timeframe: str = 'weekly'):
    """
    Compare les deux détecteurs sur un symbole donné

    Args:
        symbol: Symbole de l'action
        timeframe: 'weekly' ou 'daily'
    """
    print(f"\n{'='*80}")
    print(f"COMPARAISON DES DÉTECTEURS - {symbol} ({timeframe})")
    print('='*80)

    # Récupérer données
    if timeframe == 'weekly':
        df = market_data_fetcher.get_historical_data(symbol, period='2y', interval='1wk')
        lookback = 104
    else:
        df = market_data_fetcher.get_historical_data(symbol, period='1y', interval='1d')
        lookback = 252

    if df is None or df.empty:
        print(f"❌ Pas de données pour {symbol}")
        return None

    # Initialiser les deux détecteurs
    original_detector = RSITrendlineDetector(
        prominence=2.0,
        min_r_squared=0.25
    )

    enhanced_detector = EnhancedRSITrendlineDetector(
        precision_mode='high',
        use_ransac=True,
        adaptive_prominence=True
    )

    # Calculer RSI (partagé)
    rsi = original_detector.calculate_rsi(df)

    # Détection avec ORIGINAL
    print("\n📊 DÉTECTEUR ORIGINAL:")
    print(f"   Paramètres: prominence=2.0, min_R²=0.25, max_residual=8.0")

    peaks_orig, _ = original_detector.detect_peaks(rsi)
    print(f"   Pics détectés: {len(peaks_orig)}")

    trendline_orig = original_detector.find_best_trendline(rsi, peaks_orig, lookback)

    if trendline_orig:
        x_orig = np.array(trendline_orig.peak_indices)
        y_orig = np.array(trendline_orig.peak_values)
        y_pred_orig = trendline_orig.slope * x_orig + trendline_orig.intercept
        residuals_orig = np.abs(y_orig - y_pred_orig)

        print(f"   ✅ Oblique trouvée:")
        print(f"      - Nombre de pics: {len(trendline_orig.peak_indices)}")
        print(f"      - R²: {trendline_orig.r_squared:.3f}")
        print(f"      - Distance moyenne: {np.mean(residuals_orig):.2f} points RSI")
        print(f"      - Distance max: {np.max(residuals_orig):.2f} points RSI")
        print(f"      - Écart-type: {np.std(residuals_orig):.2f} points RSI")
        print(f"      - Score qualité: {trendline_orig.quality_score:.1f}/100")
    else:
        print(f"   ❌ Aucune oblique détectée")

    # Détection avec AMÉLIORÉ
    print("\n🎯 DÉTECTEUR AMÉLIORÉ:")
    print(f"   Paramètres: prominence adaptative (2.5-4.5), min_R²=0.65, max_residual=4.0")
    print(f"   Améliorations: RANSAC, filtrage qualité, validation stricte")

    peaks_enh, props = enhanced_detector.detect_peaks_adaptive(rsi)
    prominence_used = props.get('used_prominence', 'N/A')
    print(f"   Pics détectés: {len(peaks_enh)} (prominence={prominence_used:.1f})")

    trendline_enh = enhanced_detector.find_best_trendline(rsi, peaks_enh, lookback)

    if trendline_enh:
        x_enh = np.array(trendline_enh.peak_indices)
        y_enh = np.array(trendline_enh.peak_values)
        y_pred_enh = trendline_enh.slope * x_enh + trendline_enh.intercept
        residuals_enh = np.abs(y_enh - y_pred_enh)

        print(f"   ✅ Oblique trouvée:")
        print(f"      - Nombre de pics: {len(trendline_enh.peak_indices)}")
        print(f"      - R²: {trendline_enh.r_squared:.3f}")
        print(f"      - Distance moyenne: {np.mean(residuals_enh):.2f} points RSI")
        print(f"      - Distance max: {np.max(residuals_enh):.2f} points RSI")
        print(f"      - Écart-type: {np.std(residuals_enh):.2f} points RSI")
        print(f"      - Score qualité: {trendline_enh.quality_score:.1f}/100")

        # Validation précision
        is_precise, reason = enhanced_detector.validate_trendline_precision(trendline_enh, rsi)
        print(f"      - Validation: {reason}")
    else:
        print(f"   ❌ Aucune oblique détectée (standards de précision non atteints)")

    # Comparaison
    print("\n📈 COMPARAISON:")

    if trendline_orig and trendline_enh:
        improvements = {
            'R²': (trendline_enh.r_squared - trendline_orig.r_squared) / trendline_orig.r_squared * 100,
            'Distance moyenne': (np.mean(residuals_orig) - np.mean(residuals_enh)) / np.mean(residuals_orig) * 100,
            'Distance max': (np.max(residuals_orig) - np.max(residuals_enh)) / np.max(residuals_orig) * 100,
        }

        for metric, improvement in improvements.items():
            symbol_imp = "📈" if improvement > 0 else "📉"
            print(f"   {symbol_imp} {metric}: {improvement:+.1f}%")

    elif trendline_orig and not trendline_enh:
        print(f"   ⚠️  Détecteur amélioré plus strict: oblique originale ne respecte pas les standards de précision")
    elif not trendline_orig and trendline_enh:
        print(f"   ✅ Détecteur amélioré trouve une oblique de qualité là où l'original échoue")
    else:
        print(f"   ℹ️  Aucune oblique valide pour les deux détecteurs")

    # Graphique comparatif
    if trendline_orig or trendline_enh:
        plot_comparison(
            df, rsi, symbol, timeframe,
            trendline_orig, trendline_enh,
            peaks_orig, peaks_enh
        )

    return {
        'symbol': symbol,
        'original': trendline_orig,
        'enhanced': trendline_enh
    }


def plot_comparison(df, rsi, symbol, timeframe, trendline_orig, trendline_enh, peaks_orig, peaks_enh):
    """Génère graphique de comparaison côte à côte"""

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'{symbol} - Comparaison Détecteurs RSI - {timeframe.upper()}',
                 fontsize=16, fontweight='bold')

    # Prix (commun aux deux)
    for ax in axes[0]:
        ax.plot(df.index, df['Close'], linewidth=2, color='#2196F3')
        ax.set_ylabel('Prix ($)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    axes[0, 0].set_title('DÉTECTEUR ORIGINAL', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('DÉTECTEUR AMÉLIORÉ', fontsize=12, fontweight='bold')

    # RSI avec obliques
    colors = ['#2196F3', '#4CAF50']

    for idx, (ax, trendline, peaks, title) in enumerate([
        (axes[1, 0], trendline_orig, peaks_orig, 'Original'),
        (axes[1, 1], trendline_enh, peaks_enh, 'Amélioré')
    ]):
        # RSI
        ax.plot(rsi.index, rsi, linewidth=1.5, color='#9C27B0', alpha=0.7, label='RSI(14)')

        # Lignes de référence
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.3)
        ax.axhline(y=30, color='green', linestyle='--', alpha=0.3)
        ax.axhline(y=50, color='gray', linestyle='-', alpha=0.2)

        # Tous les pics (gris clair)
        ax.scatter(rsi.index[peaks], rsi.iloc[peaks],
                  color='lightgray', s=60, zorder=3, alpha=0.5,
                  label=f'Tous les pics ({len(peaks)})')

        if trendline:
            # Pics de l'oblique
            peak_dates = trendline.peak_dates
            peak_values = trendline.peak_values
            ax.scatter(peak_dates, peak_values,
                      color='#FF9800', s=150, zorder=4, edgecolors='black', linewidths=2,
                      label=f'Pics oblique ({len(peak_dates)})')

            # Annoter
            for i, (date, value) in enumerate(zip(peak_dates, peak_values)):
                ax.annotate(f'{i+1}', xy=(date, value), xytext=(0, 10),
                           textcoords='offset points', ha='center',
                           fontsize=9, fontweight='bold', color='#FF9800',
                           bbox=dict(boxstyle='circle', facecolor='white',
                                   edgecolor='#FF9800', linewidth=1.5))

            # Oblique
            trendline_x = np.arange(trendline.start_idx, trendline.end_idx + 1)
            trendline_y = trendline.slope * trendline_x + trendline.intercept
            trendline_dates = rsi.index[trendline_x]

            ax.plot(trendline_dates, trendline_y,
                   color='#FF9800', linestyle='--', linewidth=3,
                   label=f'Oblique (R²={trendline.r_squared:.2f})', zorder=5)

            # Métriques
            x_vals = np.array(trendline.peak_indices)
            y_vals = np.array(trendline.peak_values)
            y_pred = trendline.slope * x_vals + trendline.intercept
            residuals = np.abs(y_vals - y_pred)

            metrics_text = f"R² = {trendline.r_squared:.3f}\n"
            metrics_text += f"Dist. moy = {np.mean(residuals):.2f}\n"
            metrics_text += f"Dist. max = {np.max(residuals):.2f}\n"
            metrics_text += f"Score = {trendline.quality_score:.1f}/100"

            ax.text(0.02, 0.98, metrics_text,
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Aucune oblique détectée',
                   transform=ax.transAxes, fontsize=14, ha='center',
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

        ax.set_ylabel('RSI', fontsize=11, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

    plt.tight_layout()
    filename = f'comparison_{symbol}_{timeframe}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n💾 Graphique sauvegardé: {filename}")
    plt.close()


def run_comparison_suite():
    """Compare sur plusieurs actions"""

    symbols = ['MSFT', 'TSLA', 'AMD', 'NFLX', 'GOOGL', 'META', 'NVDA', 'AAPL']

    print("\n" + "="*80)
    print("SUITE DE COMPARAISON - DÉTECTEUR ORIGINAL VS AMÉLIORÉ")
    print("="*80)

    results = []

    for symbol in symbols:
        try:
            result = compare_detectors(symbol, 'weekly')
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ Erreur pour {symbol}: {e}")

    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES AMÉLIORATIONS")
    print("="*80)

    orig_found = sum(1 for r in results if r['original'])
    enh_found = sum(1 for r in results if r['enhanced'])

    print(f"\n📊 Détections:")
    print(f"   - Original: {orig_found}/{len(results)} obliques")
    print(f"   - Amélioré: {enh_found}/{len(results)} obliques")

    # Comparaison qualité (pour obliques trouvées par les deux)
    both_found = [r for r in results if r['original'] and r['enhanced']]

    if both_found:
        print(f"\n📈 Amélioration de qualité ({len(both_found)} obliques communes):")

        avg_r2_orig = np.mean([r['original'].r_squared for r in both_found])
        avg_r2_enh = np.mean([r['enhanced'].r_squared for r in both_found])

        print(f"   - R² moyen:")
        print(f"      Original: {avg_r2_orig:.3f}")
        print(f"      Amélioré: {avg_r2_enh:.3f}")
        print(f"      Gain: {(avg_r2_enh - avg_r2_orig) / avg_r2_orig * 100:+.1f}%")

    print("\n" + "="*80)


if __name__ == '__main__':
    # Test sur une action spécifique
    # compare_detectors('MSFT', 'weekly')

    # Ou lancer la suite complète
    run_comparison_suite()
