"""
Multi-Trendline Detector

Détecte PLUSIEURS trendlines sur le prix en utilisant une approche par fenêtre glissante.
Au lieu d'une seule trendline globale, détecte plusieurs segments qui peuvent avoir des pentes différentes.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from scipy.signal import find_peaks

from .price_trendline_detector import PriceTrendline


class MultiTrendlineDetector:
    """Détecte plusieurs trendlines sur des segments temporels"""

    def __init__(
        self,
        min_peaks: int = 3,
        window_size: int = 60,  # 60 jours de fenêtre glissante
        step_size: int = 15,     # Décalage de 15 jours entre fenêtres
        min_r_squared: float = 0.70,  # R² élevé pour trendlines claires
        max_residual_pct: float = 5.0
    ):
        """
        Initialize multi-trendline detector

        Args:
            min_peaks: Minimum 3 sommets pour une trendline
            window_size: Taille de la fenêtre glissante (en jours)
            step_size: Décalage entre fenêtres
            min_r_squared: R² minimum pour une trendline valide
            max_residual_pct: Distance max des sommets à la ligne (%)
        """
        self.min_peaks = min_peaks
        self.window_size = window_size
        self.step_size = step_size
        self.min_r_squared = min_r_squared
        self.max_residual_pct = max_residual_pct

    def detect_all_resistance_trendlines(
        self,
        df: pd.DataFrame,
        all_peaks: np.ndarray,  # Indices de TOUS les sommets détectés
        peak_values: np.ndarray  # Valeurs des sommets
    ) -> List[PriceTrendline]:
        """
        Détecte toutes les trendlines de résistance possibles

        Utilise une fenêtre glissante pour trouver des segments de sommets
        qui forment des lignes claires.

        Args:
            df: DataFrame avec OHLCV
            all_peaks: Indices de tous les sommets détectés
            peak_values: Valeurs des sommets

        Returns:
            Liste de PriceTrendline trouvées
        """
        trendlines = []
        total_length = len(df)

        # Fenêtre glissante sur la période
        for start_idx in range(0, total_length - self.window_size, self.step_size):
            end_idx = start_idx + self.window_size

            # Filtrer les sommets dans cette fenêtre
            window_peaks_mask = (all_peaks >= start_idx) & (all_peaks < end_idx)
            window_peaks = all_peaks[window_peaks_mask]
            window_values = peak_values[window_peaks_mask]

            if len(window_peaks) < self.min_peaks:
                continue

            # Essayer de trouver la meilleure trendline dans cette fenêtre
            best_tl = self._find_best_in_window(
                df, window_peaks, window_values, trendline_type='resistance'
            )

            if best_tl is not None:
                # Vérifier qu'elle n'est pas déjà couverte par une trendline existante
                if not self._is_overlapping(best_tl, trendlines):
                    trendlines.append(best_tl)

        return trendlines

    def detect_all_support_trendlines(
        self,
        df: pd.DataFrame,
        all_valleys: np.ndarray,
        valley_values: np.ndarray
    ) -> List[PriceTrendline]:
        """
        Détecte toutes les trendlines de support possibles

        Args:
            df: DataFrame avec OHLCV
            all_valleys: Indices de tous les creux détectés
            valley_values: Valeurs des creux

        Returns:
            Liste de PriceTrendline trouvées
        """
        trendlines = []
        total_length = len(df)

        for start_idx in range(0, total_length - self.window_size, self.step_size):
            end_idx = start_idx + self.window_size

            window_valleys_mask = (all_valleys >= start_idx) & (all_valleys < end_idx)
            window_valleys = all_valleys[window_valleys_mask]
            window_values = valley_values[window_valleys_mask]

            if len(window_valleys) < self.min_peaks:
                continue

            best_tl = self._find_best_in_window(
                df, window_valleys, window_values, trendline_type='support'
            )

            if best_tl is not None:
                if not self._is_overlapping(best_tl, trendlines):
                    trendlines.append(best_tl)

        return trendlines

    def _find_best_in_window(
        self,
        df: pd.DataFrame,
        peaks: np.ndarray,
        values: np.ndarray,
        trendline_type: str
    ) -> Optional[PriceTrendline]:
        """
        Trouve la meilleure trendline dans une fenêtre donnée

        Args:
            df: DataFrame
            peaks: Indices des sommets dans cette fenêtre
            values: Valeurs des sommets
            trendline_type: 'support' or 'resistance'

        Returns:
            Meilleure PriceTrendline ou None
        """
        best_tl = None
        best_score = -np.inf

        # Essayer différentes combinaisons de sommets
        for start in range(len(peaks) - self.min_peaks + 1):
            for end in range(start + self.min_peaks, len(peaks) + 1):
                selected_peaks = peaks[start:end]
                selected_values = values[start:end]

                if len(selected_peaks) < self.min_peaks:
                    continue

                # Fit trendline
                x = selected_peaks.astype(float)
                y = selected_values

                slope, intercept, r_squared = self._fit_trendline(x, y)

                # Valider R²
                if r_squared < self.min_r_squared:
                    continue

                # Valider résidus
                max_residual = max(abs(y[i] - (slope * x[i] + intercept)) for i in range(len(y)))
                avg_price = np.mean(y)
                residual_threshold = (self.max_residual_pct / 100) * avg_price

                if max_residual > residual_threshold:
                    continue

                # Score de qualité
                score = r_squared * 50 + len(selected_peaks) * 10

                if score > best_score:
                    best_score = score
                    best_tl = PriceTrendline(
                        peak_indices=selected_peaks,
                        peak_values=selected_values,
                        peak_dates=df.index[selected_peaks],
                        slope=slope,
                        intercept=intercept,
                        r_squared=r_squared,
                        start_idx=selected_peaks[0],
                        end_idx=selected_peaks[-1],
                        trendline_type=trendline_type,
                        quality_score=score
                    )

        return best_tl

    def _fit_trendline(self, x: np.ndarray, y: np.ndarray):
        """Fit linear regression et calcule R²"""
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs

        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return slope, intercept, r_squared

    def _is_overlapping(self, new_tl: PriceTrendline, existing_tls: List[PriceTrendline]) -> bool:
        """
        Vérifie si une trendline chevauche significativement une existante

        Args:
            new_tl: Nouvelle trendline
            existing_tls: Trendlines déjà détectées

        Returns:
            True si chevauchement significatif
        """
        if not existing_tls:
            return False

        new_start = new_tl.start_idx
        new_end = new_tl.end_idx

        for existing in existing_tls:
            ex_start = existing.start_idx
            ex_end = existing.end_idx

            # Calcul du chevauchement
            overlap_start = max(new_start, ex_start)
            overlap_end = min(new_end, ex_end)

            if overlap_start < overlap_end:
                overlap_length = overlap_end - overlap_start
                new_length = new_end - new_start
                existing_length = ex_end - ex_start

                # Si plus de 70% de chevauchement, considérer comme dupliqué
                if (overlap_length / new_length > 0.7) or (overlap_length / existing_length > 0.7):
                    return True

        return False
