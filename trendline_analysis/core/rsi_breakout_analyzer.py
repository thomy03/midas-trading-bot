"""
RSI Breakout Analyzer - Simplified version for EMA Screener integration

Détecte UNIQUEMENT les obliques RSI et leurs breakouts (pas de price trendline).
Conçu pour être intégré avec le screener EMA.
"""

import pandas as pd
from typing import Optional, Dict
from dataclasses import dataclass

from .trendline_detector import RSITrendlineDetector, Trendline
from .breakout_analyzer import TrendlineBreakoutAnalyzer, Breakout


@dataclass
class RSIBreakoutResult:
    """Résultat de l'analyse RSI breakout"""
    has_rsi_trendline: bool
    has_rsi_breakout: bool
    rsi_trendline: Optional[Trendline] = None
    rsi_breakout: Optional[Breakout] = None
    signal: str = 'NO_SIGNAL'  # 'STRONG_BUY', 'BUY', 'WATCH', 'NO_SIGNAL'


class RSIBreakoutAnalyzer:
    """
    Analyseur simplifié pour breakouts RSI uniquement

    Usage:
        analyzer = RSIBreakoutAnalyzer()
        result = analyzer.analyze(df, lookback_periods=252)

        if result.has_rsi_breakout:
            print(f"RSI Breakout détecté: {result.signal}")
    """

    def __init__(self):
        """Initialize RSI breakout analyzer"""
        self.rsi_detector = RSITrendlineDetector()
        self.breakout_analyzer = TrendlineBreakoutAnalyzer()

    def analyze(
        self,
        df: pd.DataFrame,
        lookback_periods: int = 252
    ) -> Optional[RSIBreakoutResult]:
        """
        Analyse un DataFrame pour détecter un breakout d'oblique RSI

        Args:
            df: DataFrame avec OHLCV data
            lookback_periods: Période de lookback (252 pour daily ~1an, 104 pour weekly ~2ans)

        Returns:
            RSIBreakoutResult ou None si aucune oblique RSI détectée
        """
        # Step 1: Détecter oblique RSI
        rsi_trendline = self.rsi_detector.detect(df, lookback_periods=lookback_periods)

        if rsi_trendline is None:
            return None  # Pas d'oblique RSI = pas de signal possible

        # Step 2: Calculer RSI pour détecter breakout
        rsi = self.rsi_detector.calculate_rsi(df)

        # Step 3: Détecter breakout RSI
        rsi_breakout = self.breakout_analyzer.detect_breakout(
            rsi=rsi,
            trendline=rsi_trendline
        )

        # Step 4: Générer signal
        signal = self._get_signal(rsi_trendline, rsi_breakout)

        return RSIBreakoutResult(
            has_rsi_trendline=True,
            has_rsi_breakout=(rsi_breakout is not None),
            rsi_trendline=rsi_trendline,
            rsi_breakout=rsi_breakout,
            signal=signal
        )

    def _get_signal(
        self,
        rsi_trendline: Trendline,
        rsi_breakout: Optional[Breakout]
    ) -> str:
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

    def get_summary(self, result: RSIBreakoutResult) -> Dict:
        """
        Génère un résumé lisible du résultat

        Args:
            result: Résultat de l'analyse

        Returns:
            Dictionnaire avec résumé
        """
        summary = {
            'has_rsi_trendline': result.has_rsi_trendline,
            'has_rsi_breakout': result.has_rsi_breakout,
            'signal': result.signal
        }

        if result.rsi_trendline:
            summary['rsi_trendline'] = {
                'num_peaks': len(result.rsi_trendline.peak_indices),
                'r_squared': result.rsi_trendline.r_squared,
                'slope': result.rsi_trendline.slope,
                'start_date': result.rsi_trendline.peak_dates[0].strftime('%Y-%m-%d'),
                'end_date': result.rsi_trendline.peak_dates[-1].strftime('%Y-%m-%d')
            }

        if result.rsi_breakout:
            summary['rsi_breakout'] = {
                'date': result.rsi_breakout.date.strftime('%Y-%m-%d'),
                'rsi_value': result.rsi_breakout.rsi_value,
                'strength': result.rsi_breakout.strength,
                'age_periods': result.rsi_breakout.age_in_periods
            }

        return summary
