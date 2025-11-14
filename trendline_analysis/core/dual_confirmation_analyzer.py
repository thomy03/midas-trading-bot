"""
Dual Confirmation Analyzer

Validates trading signals by requiring BOTH RSI trendline breakout AND price trendline breakout
within a close time window (default: ±6 periods).

This dramatically reduces false signals by ensuring both momentum (RSI) and price action
confirm the breakout.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from .trendline_detector import RSITrendlineDetector, Trendline
from .breakout_analyzer import TrendlineBreakoutAnalyzer, Breakout
from .price_trendline_detector import PriceTrendlineDetector, PriceTrendline
from .synchronized_price_detector import SynchronizedPriceDetector
from .multi_trendline_detector import MultiTrendlineDetector
from scipy.signal import find_peaks


@dataclass
class PriceBreakout:
    """Represents a detected price trendline breakout"""
    date: pd.Timestamp
    index: int
    price_value: float
    trendline_value: float
    distance_above: float
    trendline_type: str  # 'support' or 'resistance'
    strength: str


@dataclass
class DualConfirmation:
    """Represents a double-confirmed signal (RSI + Price)"""
    rsi_breakout: Breakout
    price_breakout: PriceBreakout
    time_difference: int  # Number of periods between breakouts
    is_synchronized: bool  # True if within acceptable window
    confirmation_strength: str  # 'VERY_STRONG', 'STRONG', 'MODERATE'


class DualConfirmationAnalyzer:
    """Analyzes dual confirmation signals from RSI and Price trendlines"""

    def __init__(self, sync_window: int = 6, use_multi_trendlines: bool = True):
        """
        Initialize dual confirmation analyzer

        Args:
            sync_window: Maximum periods between RSI and Price breakouts (default: 6)
            use_multi_trendlines: Use multi-trendline detection (default: True)
        """
        self.sync_window = sync_window
        self.use_multi_trendlines = use_multi_trendlines
        self.rsi_detector = RSITrendlineDetector()
        self.rsi_breakout_analyzer = TrendlineBreakoutAnalyzer()
        self.price_detector = PriceTrendlineDetector()
        self.multi_trendline_detector = MultiTrendlineDetector(
            min_peaks=3,
            window_size=90,
            step_size=30,
            min_r_squared=0.70,
            max_residual_pct=5.0
        )

    def analyze(
        self,
        df: pd.DataFrame,
        lookback_periods: int = 52
    ) -> Optional[Dict]:
        """
        Complete dual confirmation analysis

        LOGIQUE FLEXIBLE (HORIZON TEMPOREL):
        1. Cherche TOUJOURS une oblique RSI ET une oblique Prix
        2. Si les deux existent, vérifie qu'elles sont sur le MÊME HORIZON TEMPOREL
        3. Le signal peut venir de:
           - RSI breakout seul (prix pas encore cassé mais présent)
           - Prix breakout seul (RSI pas encore cassé mais présent)
           - Dual confirmation (les deux cassées)
        4. Retourne un résultat si AU MOINS une oblique est détectée

        Args:
            df: DataFrame with OHLCV data
            lookback_periods: How far back to look

        Returns:
            Dictionary with analysis results or None
        """
        # Step 1: Detect RSI trendline
        rsi = self.rsi_detector.calculate_rsi(df)
        rsi_trendline = self.rsi_detector.detect(df, lookback_periods)
        has_rsi_trendline = rsi_trendline is not None

        # Step 2: Detect Price trendline
        # Chercher une oblique prix (temporellement alignée avec RSI si possible)
        if self.use_multi_trendlines:
            price_trendline = self._detect_best_descending_resistance(df, rsi_trendline)
        else:
            price_trendline = self.price_detector.detect_resistance(df, lookback_periods)
        has_price_trendline = price_trendline is not None

        # SI AUCUNE oblique détectée, pas de signal possible
        if not has_rsi_trendline and not has_price_trendline:
            return None

        # Step 3: Vérifier l'alignement temporel si les DEUX obliques existent
        temporal_alignment = 'none'
        if has_rsi_trendline and has_price_trendline:
            temporal_alignment = self._check_temporal_alignment(rsi_trendline, price_trendline)

        # Step 4: Detect RSI breakout (si oblique RSI existe)
        rsi_breakout = None
        if has_rsi_trendline:
            rsi_breakout = self.rsi_breakout_analyzer.detect_breakout(rsi, rsi_trendline)
        has_rsi_breakout = rsi_breakout is not None

        # Step 5: Detect Price breakout
        price_breakout = None
        if has_price_trendline:
            if has_rsi_breakout:
                # Si RSI breakout existe, chercher prix breakout proche
                price_breakout = self._detect_price_breakout(df, price_trendline, rsi_breakout)
            else:
                # Sinon, chercher le breakout prix standalone
                price_breakout = self._detect_price_breakout_standalone(df, price_trendline)
        has_price_breakout = price_breakout is not None

        # Step 6: Check dual confirmation (si les deux breakouts existent)
        dual_confirmation = None
        if rsi_breakout is not None and price_breakout is not None:
            dual_confirmation = self._check_synchronization(rsi_breakout, price_breakout)

        # Build comprehensive result
        result = {
            'has_rsi_trendline': has_rsi_trendline,
            'has_rsi_breakout': rsi_breakout is not None,
            'has_price_trendline': has_price_trendline,
            'has_price_breakout': price_breakout is not None,
            'has_dual_confirmation': dual_confirmation is not None,
            'temporal_alignment': temporal_alignment,  # 'good', 'weak', or 'none'
        }

        # Add optional fields only if they exist
        if rsi_trendline is not None:
            result['rsi_trendline'] = rsi_trendline
        if rsi_breakout is not None:
            result['rsi_breakout'] = rsi_breakout
        if price_trendline is not None:
            result['price_trendline'] = price_trendline
        if price_breakout is not None:
            result['price_breakout'] = price_breakout
        if dual_confirmation is not None:
            result['dual_confirmation'] = dual_confirmation

        return result

    def _detect_price_breakout(
        self,
        df: pd.DataFrame,
        trendline: PriceTrendline,
        rsi_breakout: Breakout
    ) -> Optional[PriceBreakout]:
        """
        Detect if and when price broke above/below the trendline

        Selects the breakout CLOSEST to the RSI breakout (smart selection)

        Args:
            df: DataFrame with OHLCV data
            trendline: PriceTrendline to check
            rsi_breakout: RSI breakout to synchronize with

        Returns:
            PriceBreakout or None
        """
        if trendline.trendline_type == 'resistance':
            # For resistance: check if Close crossed ABOVE
            price_series = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
            direction = 'above'
        else:
            # For support: check if Close crossed BELOW
            price_series = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
            direction = 'below'

        # Start checking from last peak
        start_idx = max(trendline.peak_indices[-1] + 1, trendline.start_idx)
        end_idx = len(price_series) - 1

        breakout_candidates = []

        for i in range(start_idx, end_idx + 1):
            price_val = price_series.iloc[i]
            trendline_val = self.price_detector.get_trendline_value(trendline, i)

            if direction == 'above':
                distance = price_val - trendline_val

                if distance >= 0:
                    # Check if it's a true crossing (was below before)
                    if i > start_idx:
                        prev_price = price_series.iloc[i - 1]
                        prev_trendline = self.price_detector.get_trendline_value(trendline, i - 1)

                        if prev_price <= prev_trendline:
                            breakout_candidates.append({
                                'index': i,
                                'date': price_series.index[i],
                                'price_value': price_val,
                                'trendline_value': trendline_val,
                                'distance': distance
                            })
            else:
                distance = trendline_val - price_val

                if distance >= 0:
                    # Check if it's a true crossing (was above before)
                    if i > start_idx:
                        prev_price = price_series.iloc[i - 1]
                        prev_trendline = self.price_detector.get_trendline_value(trendline, i - 1)

                        if prev_price >= prev_trendline:
                            breakout_candidates.append({
                                'index': i,
                                'date': price_series.index[i],
                                'price_value': price_val,
                                'trendline_value': trendline_val,
                                'distance': distance
                            })

        if not breakout_candidates:
            return None

        # SMART SELECTION: Get breakout CLOSEST to RSI breakout (not just first!)
        # This dramatically improves synchronization detection
        rsi_breakout_idx = rsi_breakout.index

        closest_breakout = min(
            breakout_candidates,
            key=lambda bo: abs(bo['index'] - rsi_breakout_idx)
        )

        # Calculate strength based on distance
        avg_price = np.mean(price_series.iloc[start_idx:end_idx])
        distance_pct = (closest_breakout['distance'] / avg_price) * 100

        if distance_pct > 2.0:
            strength = 'STRONG'
        elif distance_pct > 1.0:
            strength = 'MODERATE'
        else:
            strength = 'WEAK'

        return PriceBreakout(
            date=closest_breakout['date'],
            index=closest_breakout['index'],
            price_value=closest_breakout['price_value'],
            trendline_value=closest_breakout['trendline_value'],
            distance_above=closest_breakout['distance'],
            trendline_type=trendline.trendline_type,
            strength=strength
        )

    def _detect_price_breakout_standalone(
        self,
        df: pd.DataFrame,
        trendline: PriceTrendline
    ) -> Optional[PriceBreakout]:
        """
        Detect price breakout WITHOUT synchronizing to RSI breakout

        Returns the MOST RECENT price breakout (standalone analysis)

        Args:
            df: DataFrame with OHLCV data
            trendline: PriceTrendline to check

        Returns:
            PriceBreakout or None
        """
        if trendline.trendline_type == 'resistance':
            # For resistance: check if Close crossed ABOVE
            price_series = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
            direction = 'above'
        else:
            # For support: check if Close crossed BELOW
            price_series = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
            direction = 'below'

        # Start checking from last peak
        start_idx = max(trendline.peak_indices[-1] + 1, trendline.start_idx)
        end_idx = len(price_series) - 1

        breakout_candidates = []

        for i in range(start_idx, end_idx + 1):
            price_val = price_series.iloc[i]
            trendline_val = self.price_detector.get_trendline_value(trendline, i)

            if direction == 'above':
                distance = price_val - trendline_val

                if distance >= 0:
                    # Check if it's a true crossing (was below before)
                    if i > start_idx:
                        prev_price = price_series.iloc[i - 1]
                        prev_trendline = self.price_detector.get_trendline_value(trendline, i - 1)

                        if prev_price <= prev_trendline:
                            breakout_candidates.append({
                                'index': i,
                                'date': price_series.index[i],
                                'price_value': price_val,
                                'trendline_value': trendline_val,
                                'distance': distance
                            })
            else:
                distance = trendline_val - price_val

                if distance >= 0:
                    # Check if it's a true crossing (was above before)
                    if i > start_idx:
                        prev_price = price_series.iloc[i - 1]
                        prev_trendline = self.price_detector.get_trendline_value(trendline, i - 1)

                        if prev_price >= prev_trendline:
                            breakout_candidates.append({
                                'index': i,
                                'date': price_series.index[i],
                                'price_value': price_val,
                                'trendline_value': trendline_val,
                                'distance': distance
                            })

        if not breakout_candidates:
            return None

        # STANDALONE: Get the MOST RECENT breakout (last in list)
        selected_breakout = breakout_candidates[-1]

        # Calculate strength based on distance
        avg_price = np.mean(price_series.iloc[start_idx:end_idx])
        distance_pct = (selected_breakout['distance'] / avg_price) * 100

        if distance_pct > 2.0:
            strength = 'STRONG'
        elif distance_pct > 1.0:
            strength = 'MODERATE'
        else:
            strength = 'WEAK'

        return PriceBreakout(
            date=selected_breakout['date'],
            index=selected_breakout['index'],
            price_value=selected_breakout['price_value'],
            trendline_value=selected_breakout['trendline_value'],
            distance_above=selected_breakout['distance'],
            trendline_type=trendline.trendline_type,
            strength=strength
        )

    def _check_synchronization(
        self,
        rsi_breakout: Breakout,
        price_breakout: PriceBreakout
    ) -> Optional[DualConfirmation]:
        """
        Check if RSI and Price breakouts are synchronized (within window)

        Args:
            rsi_breakout: RSI breakout
            price_breakout: Price breakout

        Returns:
            DualConfirmation or None if not synchronized
        """
        # Calculate time difference (in periods)
        time_diff = abs(rsi_breakout.index - price_breakout.index)

        # Check if within sync window
        is_synchronized = time_diff <= self.sync_window

        if not is_synchronized:
            return None

        # Calculate confirmation strength
        # Combine RSI strength, Price strength, and synchronization quality
        strength_score = 0

        # RSI strength component (0-3 points)
        if rsi_breakout.strength == 'STRONG':
            strength_score += 3
        elif rsi_breakout.strength == 'MODERATE':
            strength_score += 2
        else:
            strength_score += 1

        # Price strength component (0-3 points)
        if price_breakout.strength == 'STRONG':
            strength_score += 3
        elif price_breakout.strength == 'MODERATE':
            strength_score += 2
        else:
            strength_score += 1

        # Synchronization quality (0-2 points)
        # Closer breakouts = higher score
        if time_diff <= 2:
            strength_score += 2
        elif time_diff <= 4:
            strength_score += 1

        # Determine overall confirmation strength
        if strength_score >= 7:
            confirmation_strength = 'VERY_STRONG'
        elif strength_score >= 5:
            confirmation_strength = 'STRONG'
        else:
            confirmation_strength = 'MODERATE'

        return DualConfirmation(
            rsi_breakout=rsi_breakout,
            price_breakout=price_breakout,
            time_difference=time_diff,
            is_synchronized=is_synchronized,
            confirmation_strength=confirmation_strength
        )

    def _detect_best_descending_resistance(
        self,
        df: pd.DataFrame,
        rsi_trendline: Optional[Trendline]
    ) -> Optional[PriceTrendline]:
        """
        Détecte la meilleure résistance descendante (oblique baissière)
        pour une stratégie de cassure par le haut.

        Utilise multi-trendline detection et sélectionne celle qui correspond
        le mieux à l'oblique RSI (temporellement).

        Args:
            df: DataFrame avec OHLCV
            rsi_trendline: Oblique RSI (pour synchronisation temporelle)

        Returns:
            Meilleure PriceTrendline descendante ou None
        """
        # 1. Détecter TOUS les sommets prix
        high = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        open_price = df['Open'].iloc[:, 0] if isinstance(df['Open'], pd.DataFrame) else df['Open']

        price_range = high.max() - high.min()
        prominence_value = (2.0 / 100) * price_range

        high_peaks, _ = find_peaks(high.values, prominence=prominence_value, distance=5)
        green_candles = close > open_price
        close_for_peaks = np.where(green_candles, close.values, -np.inf)
        close_peaks, _ = find_peaks(close_for_peaks, prominence=prominence_value, distance=5)

        all_peaks = np.union1d(high_peaks, close_peaks)
        peak_values = np.array([max(high.iloc[i], close.iloc[i]) for i in all_peaks])

        if len(all_peaks) < 3:
            return None

        # 2. Détecter TOUTES les résistances avec multi-trendline
        all_resistances = self.multi_trendline_detector.detect_all_resistance_trendlines(
            df, all_peaks, peak_values
        )

        if not all_resistances:
            return None

        # 3. Ne garder QUE les DESCENDANTES (slope < 0) pour cassure par le haut
        descending_resistances = [tl for tl in all_resistances if tl.slope < 0]

        if not descending_resistances:
            return None

        # 4. Sélectionner la meilleure selon contexte
        if rsi_trendline is not None:
            # Si on a une oblique RSI, prendre celle qui chevauche temporellement
            return self._select_by_temporal_overlap(descending_resistances, rsi_trendline)
        else:
            # Sinon, prendre la plus récente avec meilleur quality score
            return max(descending_resistances, key=lambda tl: (tl.end_idx, tl.quality_score))

    def _check_temporal_alignment(
        self,
        rsi_trendline: Trendline,
        price_trendline: PriceTrendline
    ) -> str:
        """
        Vérifie si les obliques RSI et Prix sont sur le même horizon temporel

        Args:
            rsi_trendline: Oblique RSI
            price_trendline: Oblique Prix

        Returns:
            'good': Bon chevauchement (>50%)
            'weak': Faible chevauchement (20-50%)
            'none': Pas de chevauchement (<20%)
        """
        rsi_start = rsi_trendline.start_idx
        rsi_end = rsi_trendline.end_idx
        price_start = price_trendline.start_idx
        price_end = price_trendline.end_idx

        # Calculer le chevauchement
        overlap_start = max(rsi_start, price_start)
        overlap_end = min(rsi_end, price_end)

        if overlap_start >= overlap_end:
            return 'none'  # Pas de chevauchement

        overlap_length = overlap_end - overlap_start
        rsi_length = rsi_end - rsi_start
        price_length = price_end - price_start

        # Pourcentage de chevauchement par rapport à la plus courte oblique
        min_length = min(rsi_length, price_length)
        overlap_pct = (overlap_length / min_length) * 100 if min_length > 0 else 0

        if overlap_pct >= 50:
            return 'good'
        elif overlap_pct >= 20:
            return 'weak'
        else:
            return 'none'

    def _select_by_temporal_overlap(
        self,
        trendlines: list,
        rsi_trendline: Trendline
    ) -> Optional[PriceTrendline]:
        """
        Sélectionne la trendline prix qui chevauche le mieux l'oblique RSI temporellement

        Args:
            trendlines: Liste de PriceTrendline candidates
            rsi_trendline: Oblique RSI de référence

        Returns:
            Meilleure trendline ou None
        """
        rsi_start = rsi_trendline.start_idx
        rsi_end = rsi_trendline.end_idx

        best_tl = None
        best_overlap = 0

        for tl in trendlines:
            # Calculer chevauchement temporel
            overlap_start = max(tl.start_idx, rsi_start)
            overlap_end = min(tl.end_idx, rsi_end)

            if overlap_start < overlap_end:
                overlap = overlap_end - overlap_start

                # Favoriser les obliques avec grand chevauchement ET bon quality score
                score = overlap * tl.quality_score

                if score > best_overlap:
                    best_overlap = score
                    best_tl = tl

        # Si aucun chevauchement, prendre la plus récente
        if best_tl is None and trendlines:
            best_tl = max(trendlines, key=lambda tl: (tl.end_idx, tl.quality_score))

        return best_tl

    def get_signal(self, analysis: Dict) -> str:
        """
        Get trading signal - LOGIQUE FLEXIBLE

        3 SCÉNARIOS acceptés:
        1. RSI breakout seul (si alignement temporel avec oblique prix)
        2. Prix breakout seul (si alignement temporel avec oblique RSI)
        3. Dual confirmation (RSI + Prix breakout)

        Args:
            analysis: Result from analyze()

        Returns:
            Signal: 'VERY_STRONG_BUY', 'STRONG_BUY', 'BUY', 'WATCH', or 'NO_SIGNAL'
        """
        if not analysis:
            return 'NO_SIGNAL'

        has_rsi_tl = analysis.get('has_rsi_trendline', False)
        has_price_tl = analysis.get('has_price_trendline', False)
        has_rsi_bo = analysis.get('has_rsi_breakout', False)
        has_price_bo = analysis.get('has_price_breakout', False)
        alignment = analysis.get('temporal_alignment', 'none')

        # SI AUCUNE oblique = NO_SIGNAL
        if not has_rsi_tl and not has_price_tl:
            return 'NO_SIGNAL'

        # Vérifier récence du breakout (RSI ou Prix)
        recent_breakout = False
        breakout_age = 999

        if has_rsi_bo:
            rsi_breakout = analysis['rsi_breakout']
            breakout_age = min(breakout_age, rsi_breakout.age_in_periods)
            recent_breakout = rsi_breakout.age_in_periods <= 10

        if has_price_bo:
            # Prix breakout est toujours considéré récent si détecté
            recent_breakout = True
            breakout_age = 0  # Prix breakout = signal immédiat

        # Si breakout trop vieux, WATCH ou NO_SIGNAL
        if not recent_breakout:
            # Si obliques présentes mais pas de breakout récent = WATCH
            if has_rsi_tl or has_price_tl:
                return 'WATCH'
            else:
                return 'NO_SIGNAL'

        # SCÉNARIO 1: Dual Confirmation (RSI + Prix breakout)
        if has_rsi_bo and has_price_bo:
            dual_conf = analysis.get('dual_confirmation')
            if dual_conf:
                if dual_conf.confirmation_strength == 'VERY_STRONG':
                    return 'VERY_STRONG_BUY'
                elif dual_conf.confirmation_strength == 'STRONG':
                    return 'STRONG_BUY'
                else:
                    return 'BUY'
            # Dual breakout sans synchronization = STRONG_BUY quand même
            return 'STRONG_BUY'

        # SCÉNARIO 2: RSI breakout seul (avec oblique prix présente)
        if has_rsi_bo and has_price_tl:
            # Vérifier alignement temporel
            if alignment == 'good':
                rsi_breakout = analysis['rsi_breakout']
                if rsi_breakout.strength == 'STRONG':
                    return 'STRONG_BUY'
                else:
                    return 'BUY'
            elif alignment == 'weak':
                return 'BUY'
            # else: pas d'alignement = WATCH

        # SCÉNARIO 3: Prix breakout seul (avec oblique RSI présente)
        if has_price_bo and has_rsi_tl:
            # Vérifier alignement temporel
            if alignment == 'good':
                price_breakout = analysis['price_breakout']
                if price_breakout.strength == 'STRONG':
                    return 'STRONG_BUY'
                else:
                    return 'BUY'
            elif alignment == 'weak':
                return 'BUY'
            # else: pas d'alignement = WATCH

        # Par défaut: WATCH (obliques présentes mais conditions non remplies)
        return 'WATCH'
