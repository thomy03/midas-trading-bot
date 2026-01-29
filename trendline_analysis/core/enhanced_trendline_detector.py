"""
Enhanced RSI Trendline Detector - Version Précision

Améliorations par rapport à trendline_detector.py:
1. Détection adaptative des pics selon volatilité RSI
2. RANSAC pour ajustement robuste aux outliers
3. Validation stricte de la distance pics/oblique
4. Filtrage qualité des pics
5. Scoring raffiné basé sur précision géométrique
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    from sklearn.linear_model import RANSACRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available, RANSAC disabled")

from .trendline_detector import Trendline, RSITrendlineDetector


class EnhancedRSITrendlineDetector(RSITrendlineDetector):
    """
    Détecteur d'obliques RSI avec précision améliorée

    Usage:
        detector = EnhancedRSITrendlineDetector(
            precision_mode='high'  # 'high', 'medium', 'low'
        )
        trendline = detector.detect(df)
    """

    def __init__(
        self,
        precision_mode: str = 'high',
        use_ransac: bool = True,
        adaptive_prominence: bool = True
    ):
        """
        Initialize enhanced detector

        Args:
            precision_mode: 'high' (R²>0.65, strict), 'medium' (R²>0.50), 'low' (R²>0.35)
            use_ransac: Use RANSAC instead of simple linear regression
            adaptive_prominence: Adapt peak detection to RSI volatility
        """
        # Paramètres selon mode de précision
        if precision_mode == 'high':
            prominence = 3.5
            min_r_squared = 0.65
            max_residual = 4.0
            max_mean_residual = 2.0
            max_std_residual = 1.5
        elif precision_mode == 'medium':
            prominence = 3.0
            min_r_squared = 0.50
            max_residual = 5.0
            max_mean_residual = 2.5
            max_std_residual = 2.0
        else:  # low
            prominence = 2.5
            min_r_squared = 0.35
            max_residual = 6.0
            max_mean_residual = 3.0
            max_std_residual = 2.5

        super().__init__(
            prominence=prominence,
            distance=5,
            min_peaks=3,
            min_r_squared=min_r_squared
        )

        self.precision_mode = precision_mode
        self.use_ransac = use_ransac and HAS_SKLEARN
        self.adaptive_prominence = adaptive_prominence
        self.max_residual = max_residual
        self.max_mean_residual = max_mean_residual
        self.max_std_residual = max_std_residual

    def detect_peaks_adaptive(self, rsi: pd.Series) -> Tuple[np.ndarray, Dict]:
        """
        Détection adaptative des pics selon volatilité RSI

        Returns:
            peaks: Indices des pics
            properties: Propriétés des pics (dont prominence utilisée)
        """
        if not self.adaptive_prominence:
            return self.detect_peaks(rsi)

        # Calculer volatilité RSI
        rsi_volatility = rsi.std()

        # Ajuster prominence selon volatilité
        if rsi_volatility > 15:
            prominence = 4.5  # RSI très volatile
        elif rsi_volatility > 12:
            prominence = 3.5
        elif rsi_volatility > 10:
            prominence = 3.0
        else:
            prominence = 2.5  # RSI stable

        peaks, properties = find_peaks(
            rsi.to_numpy().ravel(),
            prominence=prominence,
            distance=self.distance
        )

        # Stocker la prominence utilisée
        properties['used_prominence'] = prominence

        return peaks, properties

    def filter_peaks_by_quality(
        self,
        rsi: pd.Series,
        peaks: np.ndarray
    ) -> np.ndarray:
        """
        Filtrage secondaire : garde seulement les pics bien formés

        Critères:
        1. Maximum local sur fenêtre ±3 périodes
        2. Au moins 3 points RSI au-dessus des creux adjacents
        3. Pas d'autre pic trop proche (< 4 périodes)
        """
        filtered_peaks = []

        for peak_idx in peaks:
            # 1. Vérifier max local
            window_start = max(0, peak_idx - 3)
            window_end = min(len(rsi), peak_idx + 4)
            window = rsi.iloc[window_start:window_end]

            if rsi.iloc[peak_idx] != window.max():
                continue

            # 2. Vérifier hauteur par rapport aux creux
            left_min = rsi.iloc[max(0, peak_idx - 5):peak_idx].min() if peak_idx > 5 else 0
            right_min = rsi.iloc[peak_idx + 1:min(len(rsi), peak_idx + 6)].min() if peak_idx < len(rsi) - 1 else 0

            peak_value = rsi.iloc[peak_idx]
            min_prominence = min(peak_value - left_min, peak_value - right_min) if left_min > 0 or right_min > 0 else 10

            if min_prominence < 3.0:
                continue

            # 3. Vérifier séparation avec pic précédent
            if len(filtered_peaks) > 0:
                if peak_idx - filtered_peaks[-1] < 4:
                    # Garder le plus haut
                    if rsi.iloc[peak_idx] > rsi.iloc[filtered_peaks[-1]]:
                        filtered_peaks[-1] = peak_idx
                    continue

            filtered_peaks.append(peak_idx)

        return np.array(filtered_peaks)

    def fit_trendline_ransac(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float, float, np.ndarray]:
        """
        Ajustement d'oblique avec RANSAC (robuste aux outliers)

        Returns:
            slope, intercept, r_squared, inlier_mask
        """
        if not self.use_ransac or not HAS_SKLEARN:
            # Fallback sur régression simple
            slope, intercept, r_squared = self.fit_trendline(x, y)
            inlier_mask = np.ones(len(x), dtype=bool)
            return slope, intercept, r_squared, inlier_mask

        X = x.reshape(-1, 1)
        y_reshaped = y.reshape(-1, 1)

        try:
            ransac = RANSACRegressor(
                min_samples=2,
                residual_threshold=3.0,  # 3 points RSI
                max_trials=200,
                random_state=42
            )

            ransac.fit(X, y_reshaped)

            slope = ransac.estimator_.coef_[0][0]
            intercept = ransac.estimator_.intercept_[0]
            inlier_mask = ransac.inlier_mask_

            # R² sur inliers seulement
            if inlier_mask.sum() > 0:
                y_pred = slope * x + intercept
                y_inliers = y[inlier_mask]
                y_pred_inliers = y_pred[inlier_mask]
                ss_res = np.sum((y_inliers - y_pred_inliers) ** 2)
                ss_tot = np.sum((y_inliers - np.mean(y_inliers)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            else:
                r_squared = 0

            return slope, intercept, r_squared, inlier_mask

        except Exception as e:
            # Fallback en cas d'erreur
            slope, intercept, r_squared = self.fit_trendline(x, y)
            inlier_mask = np.ones(len(x), dtype=bool)
            return slope, intercept, r_squared, inlier_mask

    def validate_trendline_precision(
        self,
        trendline: Trendline,
        rsi: pd.Series
    ) -> Tuple[bool, str]:
        """
        Validation stricte de la précision de l'oblique

        Critères:
        1. R² > seuil (selon precision_mode)
        2. Distance moyenne < seuil
        3. Distance max < seuil
        4. Écart-type des distances < seuil
        """
        x = np.array(trendline.peak_indices)
        y = np.array(trendline.peak_values)
        y_pred = trendline.slope * x + trendline.intercept

        residuals = np.abs(y - y_pred)

        # 1. R²
        if trendline.r_squared < self.min_r_squared:
            return False, f"R² trop bas: {trendline.r_squared:.3f} < {self.min_r_squared}"

        # 2. Distance moyenne
        mean_residual = np.mean(residuals)
        if mean_residual > self.max_mean_residual:
            return False, f"Distance moyenne trop grande: {mean_residual:.2f} > {self.max_mean_residual}"

        # 3. Distance max
        max_residual = np.max(residuals)
        if max_residual > self.max_residual:
            return False, f"Distance max trop grande: {max_residual:.2f} > {self.max_residual}"

        # 4. Écart-type
        std_residual = np.std(residuals)
        if std_residual > self.max_std_residual:
            return False, f"Écart-type trop élevé: {std_residual:.2f} > {self.max_std_residual}"

        return True, f"Oblique précise (R²={trendline.r_squared:.3f}, dist_moy={mean_residual:.2f})"

    def find_best_trendline(
        self,
        rsi: pd.Series,
        peaks: np.ndarray,
        lookback_periods: int = 52
    ) -> Optional[Trendline]:
        """
        Trouve la meilleure oblique avec validation de précision

        Amélioration par rapport à la version de base:
        - Utilise RANSAC si activé
        - Filtre par qualité des pics
        - Validation stricte de précision
        - Scoring raffiné
        """
        # Filtrage qualité des pics
        peaks = self.filter_peaks_by_quality(rsi, peaks)

        if len(peaks) < self.min_peaks:
            return None

        # Filtrer pics récents
        recent_threshold = len(rsi) - lookback_periods
        recent_peaks = peaks[peaks >= max(0, recent_threshold)]

        if len(recent_peaks) < self.min_peaks:
            return None

        best_trendline = None
        best_score = -np.inf

        min_bars_after_last_peak = 2

        # Tester différentes combinaisons
        for start_idx in range(len(recent_peaks) - self.min_peaks + 1):
            for num_peaks in range(self.min_peaks, min(len(recent_peaks) - start_idx + 1, 10)):
                selected_peaks = recent_peaks[start_idx:start_idx + num_peaks]

                last_peak_idx = selected_peaks[-1]
                bars_after = len(rsi) - 1 - last_peak_idx

                if bars_after < min_bars_after_last_peak:
                    continue

                x = selected_peaks.astype(float)
                y = rsi.iloc[selected_peaks].to_numpy()

                # Vérifier tendance descendante globale
                if y[-1] >= y[0]:
                    continue

                # Ajustement avec RANSAC ou régression simple
                if self.use_ransac:
                    slope, intercept, r_squared, inlier_mask = self.fit_trendline_ransac(x, y)

                    # Ne garder que les inliers
                    if inlier_mask.sum() < self.min_peaks:
                        continue

                    selected_peaks = selected_peaks[inlier_mask]
                    y = y[inlier_mask]
                    x = selected_peaks.astype(float)
                else:
                    slope, intercept, r_squared = self.fit_trendline(x, y)

                # Validation pente et R²
                if not (slope >= -1.0 and slope <= 0.0 and r_squared >= self.min_r_squared):
                    continue

                # Validation résistance
                if not self.validate_resistance(rsi, slope, intercept, selected_peaks, tolerance=2.0):
                    continue

                # Créer trendline temporaire pour validation précision
                temp_trendline = Trendline(
                    slope=slope,
                    intercept=intercept,
                    r_squared=r_squared,
                    peak_indices=selected_peaks.tolist(),
                    peak_dates=rsi.index[selected_peaks].tolist(),
                    peak_values=y.tolist(),
                    start_idx=selected_peaks[0],
                    end_idx=len(rsi) - 1,
                    quality_score=0
                )

                # Validation stricte de précision
                is_precise, reason = self.validate_trendline_precision(temp_trendline, rsi)
                if not is_precise:
                    continue

                # Score de qualité raffiné
                quality_score = self._calculate_enhanced_quality_score(
                    temp_trendline, rsi, len(rsi)
                )

                if quality_score > best_score:
                    best_score = quality_score
                    temp_trendline.quality_score = quality_score
                    best_trendline = temp_trendline

        return best_trendline

    def _calculate_enhanced_quality_score(
        self,
        trendline: Trendline,
        rsi: pd.Series,
        total_length: int
    ) -> float:
        """
        Score de qualité raffiné avec focus sur la précision géométrique

        Pondération:
        - R² (50%)
        - Distance moyenne aux pics (25%)
        - Nombre de pics (15%)
        - Pente optimale (5%)
        - Récence (5%)

        + Bonus pour obliques très précises
        """
        x = np.array(trendline.peak_indices)
        y = np.array(trendline.peak_values)
        y_pred = trendline.slope * x + trendline.intercept
        residuals = np.abs(y - y_pred)

        scores = {}

        # 1. R² (50 points max)
        scores['r_squared'] = trendline.r_squared * 50

        # 2. Distance moyenne (25 points max)
        mean_residual = np.mean(residuals)
        # Score inversé : plus petite distance = meilleur score
        residual_score = max(0, (1 - mean_residual / self.max_mean_residual)) * 25
        scores['mean_residual'] = residual_score

        # 3. Nombre de pics (15 points max)
        num_peaks = len(trendline.peak_indices)
        scores['num_peaks'] = min(num_peaks * 3, 15)

        # 4. Pente optimale (5 points max)
        optimal_slope = -0.15
        slope_deviation = abs(trendline.slope - optimal_slope)
        scores['slope'] = max(0, (1 - slope_deviation) * 5)

        # 5. Récence (5 points max)
        scores['recency'] = (trendline.end_idx / total_length) * 5

        total_score = sum(scores.values())

        # BONUS pour obliques très précises
        if trendline.r_squared > 0.85:
            total_score += 10  # Bonus excellent fit

        if mean_residual < 1.0:
            total_score += 10  # Bonus précision extrême

        max_residual = np.max(residuals)
        if max_residual < 2.5:
            total_score += 10  # Bonus tous les pics très proches

        return total_score

    def detect(
        self,
        df: pd.DataFrame,
        lookback_periods: int = 52
    ) -> Optional[Trendline]:
        """
        Détection principale avec toutes les améliorations

        Returns:
            Trendline de haute précision ou None
        """
        # Calculer RSI
        rsi = self.calculate_rsi(df)

        # Détection adaptative des pics
        peaks, properties = self.detect_peaks_adaptive(rsi)

        if len(peaks) == 0:
            return None

        # Trouver meilleure oblique
        trendline = self.find_best_trendline(rsi, peaks, lookback_periods)

        return trendline
