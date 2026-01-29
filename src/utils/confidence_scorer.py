"""
Confidence Scorer - Calculate trading signal confidence (0-100)

Combines multiple factors:
- EMA alignment (20 points max)
- Support proximity (20 points max)
- RSI breakout quality (25 points max)
- Freshness/recency (20 points max) - NEW
- Volume confirmation (15 points max)
"""

import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConfidenceScore:
    """Result of confidence scoring"""
    total_score: float  # 0-100
    ema_score: float  # 0-20
    support_score: float  # 0-20
    rsi_score: float  # 0-25
    freshness_score: float  # 0-20 - NEW
    volume_score: float  # 0-15
    signal: str  # STRONG_BUY, BUY, WATCH, OBSERVE
    details: Dict  # Breakdown details


class ConfidenceScorer:
    """
    Calculate confidence score for trading signals

    Score composition:
    - EMA Alignment: 20 points (EMAs croissantes)
    - Support Proximity: 20 points (distance au support)
    - RSI Breakout Quality: 25 points (R², strength)
    - Freshness: 20 points (recency of signal) - NEW
    - Volume Confirmation: 15 points (volume ratio)
    """

    # Score thresholds for signal classification
    THRESHOLDS = {
        'STRONG_BUY': 75,
        'BUY': 55,
        'WATCH': 35,
        'OBSERVE': 0
    }

    def __init__(self):
        pass

    def calculate_score(
        self,
        df: pd.DataFrame,
        ema_24: float,
        ema_38: float,
        ema_62: float,
        current_price: float,
        support_level: float,
        rsi_breakout: Optional[Dict] = None,
        rsi_trendline: Optional[Dict] = None
    ) -> ConfidenceScore:
        """
        Calculate comprehensive confidence score

        Args:
            df: DataFrame with OHLCV data
            ema_24, ema_38, ema_62: Current EMA values
            current_price: Current stock price
            support_level: Nearest support level
            rsi_breakout: RSI breakout info (optional)
            rsi_trendline: RSI trendline info (optional)

        Returns:
            ConfidenceScore with total and component scores
        """
        # Calculate each component
        ema_score, ema_details = self._score_ema_alignment(ema_24, ema_38, ema_62)
        support_score, support_details = self._score_support_proximity(
            current_price, support_level
        )
        rsi_score, rsi_details = self._score_rsi_breakout(rsi_breakout, rsi_trendline)
        freshness_score, freshness_details = self._score_freshness(rsi_breakout)
        volume_score, volume_details = self._score_volume_confirmation(df, rsi_breakout)

        # Total score
        total_score = ema_score + support_score + rsi_score + freshness_score + volume_score

        # Determine signal based on total score
        signal = self._get_signal_from_score(total_score)

        return ConfidenceScore(
            total_score=round(total_score, 1),
            ema_score=round(ema_score, 1),
            support_score=round(support_score, 1),
            rsi_score=round(rsi_score, 1),
            freshness_score=round(freshness_score, 1),
            volume_score=round(volume_score, 1),
            signal=signal,
            details={
                'ema': ema_details,
                'support': support_details,
                'rsi': rsi_details,
                'freshness': freshness_details,
                'volume': volume_details
            }
        )

    def _score_ema_alignment(
        self,
        ema_24: float,
        ema_38: float,
        ema_62: float
    ) -> Tuple[float, Dict]:
        """
        Score EMA alignment (0-20 points)

        +6.67 points for each favorable condition:
        - EMA 24 > EMA 38
        - EMA 24 > EMA 62
        - EMA 38 > EMA 62

        Max 20 points (capped)
        """
        score = 0
        conditions = {
            '24>38': ema_24 > ema_38,
            '24>62': ema_24 > ema_62,
            '38>62': ema_38 > ema_62
        }

        for condition, is_met in conditions.items():
            if is_met:
                score += 6.67  # ~6.67 * 3 = 20

        score = min(score, 20)  # Cap at 20

        return score, {
            'conditions_met': sum(conditions.values()),
            'conditions': conditions,
            'max_points': 20
        }

    def _score_support_proximity(
        self,
        current_price: float,
        support_level: float
    ) -> Tuple[float, Dict]:
        """
        Score support proximity (0-20 points)

        Formula: 20 - (distance_pct * 2.5)
        - Distance 0%: 20 points
        - Distance 3%: 12.5 points
        - Distance 5%: 7.5 points
        - Distance 8%+: 0 points
        """
        if support_level <= 0:
            return 0, {'distance_pct': None, 'reason': 'No support level'}

        distance_pct = abs((current_price - support_level) / support_level * 100)
        score = max(0, 20 - (distance_pct * 2.5))

        return score, {
            'distance_pct': round(distance_pct, 2),
            'max_points': 20
        }

    def _score_rsi_breakout(
        self,
        rsi_breakout: Optional[Dict],
        rsi_trendline: Optional[Dict]
    ) -> Tuple[float, Dict]:
        """
        Score RSI breakout quality (0-25 points)

        Components:
        - Trendline R² quality: up to 12 points
        - Breakout strength bonus: STRONG=13, MODERATE=8, WEAK=4
        (Age penalty moved to _score_freshness)
        """
        score = 0
        details = {'has_breakout': False, 'has_trendline': False}

        # Trendline quality (0-12 points)
        if rsi_trendline:
            details['has_trendline'] = True
            r_squared = rsi_trendline.get('r_squared', 0)
            trendline_score = r_squared * 12  # R²=1.0 → 12 points
            score += trendline_score
            details['r_squared'] = round(r_squared, 3)
            details['trendline_score'] = round(trendline_score, 1)

        # Breakout strength bonus (0-13 points)
        if rsi_breakout:
            details['has_breakout'] = True
            strength = rsi_breakout.get('strength', 'WEAK')
            strength_bonus = {
                'STRONG': 13,
                'MODERATE': 8,
                'WEAK': 4
            }.get(strength, 4)

            score += strength_bonus
            details['strength'] = strength
            details['strength_score'] = strength_bonus

        score = min(score, 25)  # Cap at 25

        details['max_points'] = 25
        return score, details

    def _score_freshness(
        self,
        rsi_breakout: Optional[Dict]
    ) -> Tuple[float, Dict]:
        """
        Score signal freshness/recency (0-20 points)

        A fresh signal is more valuable than an old one:
        - Age 0-1 periods: 20 points (just happened!)
        - Age 2 periods: 16 points
        - Age 3 periods: 12 points
        - Age 4 periods: 8 points
        - Age 5 periods: 4 points
        - Age 6+ periods: 0 points (stale signal)

        No breakout = 5 points (neutral, has trendline but no breakout yet)
        """
        details = {'age_in_periods': None, 'max_points': 20}

        if not rsi_breakout:
            # No breakout yet = neutral score (trendline only)
            return 5, {'reason': 'No breakout yet (trendline only)', 'max_points': 20}

        age = rsi_breakout.get('age_in_periods', 0)
        details['age_in_periods'] = age

        # Freshness scoring: newer = better
        if age <= 1:
            score = 20  # Just happened!
        elif age == 2:
            score = 16
        elif age == 3:
            score = 12
        elif age == 4:
            score = 8
        elif age == 5:
            score = 4
        else:
            score = 0  # Stale signal (6+ periods old)

        details['freshness_label'] = 'FRESH' if age <= 2 else ('RECENT' if age <= 4 else 'STALE')

        return score, details

    def _score_volume_confirmation(
        self,
        df: pd.DataFrame,
        rsi_breakout: Optional[Dict]
    ) -> Tuple[float, Dict]:
        """
        Score volume confirmation (0-15 points)

        Uses volume_ratio from Breakout if available (preferred),
        otherwise falls back to calculating from DataFrame.

        Scoring:
        - Volume > 2.0x average: 15 points
        - Volume > 1.5x average: 12 points
        - Volume > 1.0x average: 8 points
        - Volume 0.7-1.0x average: 4 points
        - Volume < 0.7x average: 0 points (weak signal)
        """
        details = {'volume_ratio': None, 'max_points': 15}

        # PRIORITY 1: Use volume_ratio from Breakout if available
        if rsi_breakout and 'volume_ratio' in rsi_breakout:
            volume_ratio = rsi_breakout['volume_ratio']
            details['volume_ratio'] = round(volume_ratio, 2)
            details['source'] = 'breakout'
            return self._volume_ratio_to_score(volume_ratio, details)

        # PRIORITY 2: Calculate from DataFrame (fallback)
        if df is None or df.empty or 'Volume' not in df.columns:
            return 7, {'reason': 'No volume data', 'max_points': 15}

        try:
            avg_volume = df['Volume'].rolling(20).mean()

            # Get breakout index or use last bar
            if rsi_breakout and 'index' in rsi_breakout:
                breakout_idx = rsi_breakout['index']
                if breakout_idx >= len(df):
                    breakout_idx = len(df) - 1
            else:
                breakout_idx = len(df) - 1

            breakout_volume = df['Volume'].iloc[breakout_idx]
            avg_at_breakout = avg_volume.iloc[breakout_idx]

            if pd.isna(avg_at_breakout) or avg_at_breakout <= 0:
                return 7, {'reason': 'Insufficient volume history', 'max_points': 15}

            volume_ratio = breakout_volume / avg_at_breakout
            details['volume_ratio'] = round(volume_ratio, 2)
            details['source'] = 'calculated'
            return self._volume_ratio_to_score(volume_ratio, details)

        except Exception as e:
            return 7, {'reason': f'Error: {str(e)[:50]}', 'max_points': 15}

    def _volume_ratio_to_score(self, volume_ratio: float, details: Dict) -> Tuple[float, Dict]:
        """Convert volume ratio to score (0-15 points)"""
        if volume_ratio >= 2.0:
            score = 15
        elif volume_ratio >= 1.5:
            score = 12
        elif volume_ratio >= 1.0:
            score = 8
        elif volume_ratio >= 0.7:
            score = 4
        else:
            score = 0  # Very low volume = weak signal

        return score, details

    def _get_signal_from_score(self, score: float) -> str:
        """Convert score to signal classification"""
        if score >= self.THRESHOLDS['STRONG_BUY']:
            return 'STRONG_BUY'
        elif score >= self.THRESHOLDS['BUY']:
            return 'BUY'
        elif score >= self.THRESHOLDS['WATCH']:
            return 'WATCH'
        else:
            return 'OBSERVE'

    def format_score_display(self, confidence: ConfidenceScore) -> str:
        """Format score for display: STRONG_BUY (87/100)"""
        return f"{confidence.signal} ({confidence.total_score:.0f}/100)"


# Singleton instance
confidence_scorer = ConfidenceScorer()
