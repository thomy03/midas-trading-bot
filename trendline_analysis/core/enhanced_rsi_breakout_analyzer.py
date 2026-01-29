"""
Enhanced RSI Breakout Analyzer - Version Haute Précision

Compatible avec RSIBreakoutAnalyzer mais utilise le détecteur amélioré en interne.
Peut être utilisé comme drop-in replacement dans le screener.
"""

import pandas as pd
from typing import Optional

from .enhanced_trendline_detector import EnhancedRSITrendlineDetector
from .breakout_analyzer import TrendlineBreakoutAnalyzer
from .rsi_breakout_analyzer import RSIBreakoutResult


class EnhancedRSIBreakoutAnalyzer:
    """
    Analyseur RSI amélioré avec haute précision

    Drop-in replacement pour RSIBreakoutAnalyzer avec :
    - RANSAC pour ajustement robuste
    - Prominence adaptative
    - Validation stricte de distance
    - 3 modes de précision configurables

    Usage dans le screener :
        # Remplacer :
        # from trendline_analysis.core.rsi_breakout_analyzer import RSIBreakoutAnalyzer
        # self.rsi_analyzer = RSIBreakoutAnalyzer()

        # Par :
        from trendline_analysis.core.enhanced_rsi_breakout_analyzer import EnhancedRSIBreakoutAnalyzer
        self.rsi_analyzer = EnhancedRSIBreakoutAnalyzer(precision_mode='medium')
    """

    def __init__(self, precision_mode: str = 'medium', enable_ransac: bool = True):
        """
        Initialize enhanced RSI breakout analyzer

        Args:
            precision_mode: 'high', 'medium', ou 'low'
                - 'high': R²>0.65, dist<4.0 (stricte, moins d'obliques mais excellente qualité)
                - 'medium': R²>0.50, dist<5.0 (équilibré, recommandé pour screening)
                - 'low': R²>0.35, dist<6.0 (permissif, plus d'obliques)
            enable_ransac: Utiliser RANSAC (recommandé: True)
        """
        self.precision_mode = precision_mode
        self.enable_ransac = enable_ransac

        # Détecteur d'obliques amélioré
        self.rsi_detector = EnhancedRSITrendlineDetector(
            precision_mode=precision_mode,
            use_ransac=enable_ransac,
            adaptive_prominence=True
        )

        # Analyseur de breakout (inchangé)
        self.breakout_analyzer = TrendlineBreakoutAnalyzer()

    def analyze(
        self,
        df: pd.DataFrame,
        lookback_periods: int = 252
    ) -> Optional[RSIBreakoutResult]:
        """
        Analyse un DataFrame pour détecter oblique + breakout RSI

        IMPORTANT: Vérifie le breakout sur TOUTES les obliques détectées (jusqu'à 3)
        et retourne le breakout le plus significatif.

        COMPATIBLE avec RSIBreakoutAnalyzer.analyze()

        Args:
            df: DataFrame avec OHLCV data
            lookback_periods: Période de lookback (252 pour daily, 104 pour weekly)

        Returns:
            RSIBreakoutResult ou None si aucune oblique détectée
        """
        # Step 1: Calculer RSI
        rsi = self.rsi_detector.calculate_rsi(df)

        # Step 2: Détecter les pics
        peaks, _ = self.rsi_detector.detect_peaks(rsi)

        if len(peaks) < 3:
            return None

        # Step 3: Détecter TOUTES les obliques RSI (pas juste la meilleure)
        all_trendlines = self.rsi_detector.find_all_trendlines(rsi, peaks, lookback_periods)

        if not all_trendlines:
            return None  # Pas d'oblique de qualité suffisante

        # Step 4: Vérifier breakout sur CHAQUE oblique (avec volume confirmation)
        best_breakout = None
        best_trendline = None
        best_signal = 'WATCH'

        for trendline in all_trendlines:
            breakout = self.breakout_analyzer.detect_breakout(rsi=rsi, trendline=trendline, df=df)

            if breakout:
                signal = self._get_signal(trendline, breakout)

                # Garder le breakout le plus significatif
                if self._is_better_signal(signal, best_signal):
                    best_breakout = breakout
                    best_trendline = trendline
                    best_signal = signal

        # Si aucun breakout trouvé, retourner la meilleure oblique avec WATCH
        if best_breakout is None:
            return RSIBreakoutResult(
                has_rsi_trendline=True,
                has_rsi_breakout=False,
                rsi_trendline=all_trendlines[0],
                rsi_breakout=None,
                signal='WATCH'
            )

        return RSIBreakoutResult(
            has_rsi_trendline=True,
            has_rsi_breakout=True,
            rsi_trendline=best_trendline,
            rsi_breakout=best_breakout,
            signal=best_signal
        )

    def _is_better_signal(self, new_signal: str, current_signal: str) -> bool:
        """Compare deux signaux et retourne True si new_signal est meilleur"""
        signal_priority = {'STRONG_BUY': 4, 'BUY': 3, 'WATCH': 2, 'NO_SIGNAL': 1}
        return signal_priority.get(new_signal, 0) > signal_priority.get(current_signal, 0)

    def _get_signal(self, rsi_trendline, rsi_breakout) -> str:
        """
        Génère un signal de trading basé sur le breakout RSI

        Args:
            rsi_trendline: Oblique RSI détectée
            rsi_breakout: Breakout RSI (ou None)

        Returns:
            Signal: 'STRONG_BUY', 'BUY', 'WATCH', 'NO_SIGNAL'
        """
        if rsi_breakout is None:
            # Oblique présente mais pas encore cassée
            return 'WATCH'

        # Breakout détecté - évaluer la force
        strength = rsi_breakout.strength
        age = rsi_breakout.age_in_periods

        # STRONG_BUY: breakout STRONG et récent (≤3 périodes)
        if strength == 'STRONG' and age <= 3:
            return 'STRONG_BUY'

        # BUY: breakout MODERATE ou STRONG un peu plus ancien
        if strength in ['STRONG', 'MODERATE'] and age <= 5:
            return 'BUY'

        # WATCH: breakout WEAK ou trop ancien
        return 'WATCH'

    def get_detector_info(self) -> dict:
        """
        Retourne les informations de configuration du détecteur

        Utile pour debugging et logging
        """
        return {
            'precision_mode': self.precision_mode,
            'ransac_enabled': self.enable_ransac,
            'min_r_squared': self.rsi_detector.min_r_squared,
            'max_residual_distance': self.rsi_detector.max_residual,
            'max_mean_residual': self.rsi_detector.max_mean_residual,
            'adaptive_prominence': True
        }


# Alias pour compatibilité
class PrecisionRSIBreakoutAnalyzer(EnhancedRSIBreakoutAnalyzer):
    """
    Alias pour EnhancedRSIBreakoutAnalyzer
    Nom alternatif plus explicite
    """
    pass
