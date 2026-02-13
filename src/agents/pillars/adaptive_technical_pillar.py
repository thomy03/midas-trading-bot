"""
Adaptive Technical Pillar - V8.1

Replaces the legacy if/else technical pillar with a living system:
- 18 raw indicators (trend, momentum, volume, structure, divergences)
- Contextual normalization via smooth interpolation (no if/else)
- Learned weights per regime × sector group
- Fully compatible with BasePillar interface
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .base import BasePillar, PillarScore
from .indicators_calculator import IndicatorsCalculator, RawIndicators
from .indicator_normalizer import IndicatorNormalizer

logger = logging.getLogger(__name__)

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'learned_weights', 'technical_weights.json')

# Sector → group mapping
SECTOR_GROUPS = {
    'Technology': 'growth',
    'Communication Services': 'growth',
    'Consumer Discretionary': 'cyclical',
    'Industrials': 'cyclical',
    'Materials': 'cyclical',
    'Energy': 'cyclical',
    'Consumer Staples': 'defensive',
    'Utilities': 'defensive',
    'Health Care': 'defensive',
    'Healthcare': 'defensive',
    'Financials': 'financial',
    'Real Estate': 'financial',
}

# All indicator names used in weighting
INDICATOR_NAMES = [
    'ema_alignment', 'macd_histogram', 'adx_value', 'adx_direction',
    'vwap_distance', 'ichimoku_cloud',
    'rsi', 'stochastic_k', 'cci', 'williams_r', 'roc',
    'volume_ratio', 'obv_slope', 'mfi', 'cmf',
    'support_distance', 'resistance_distance', 'bollinger_pct_b',
    'rsi_divergence', 'macd_divergence',
]

DEFAULT_WEIGHTS = {
    "BULL": {
        "growth":    {"ema_alignment": 0.09, "macd_histogram": 0.07, "adx_value": 0.04, "adx_direction": 0.05, "vwap_distance": 0.05, "ichimoku_cloud": 0.04, "rsi": 0.05, "stochastic_k": 0.04, "cci": 0.03, "williams_r": 0.03, "roc": 0.05, "volume_ratio": 0.08, "obv_slope": 0.05, "mfi": 0.04, "cmf": 0.04, "support_distance": 0.04, "resistance_distance": 0.04, "bollinger_pct_b": 0.04, "rsi_divergence": 0.05, "macd_divergence": 0.04},
        "defensive": {"ema_alignment": 0.07, "macd_histogram": 0.05, "adx_value": 0.04, "adx_direction": 0.04, "vwap_distance": 0.04, "ichimoku_cloud": 0.04, "rsi": 0.08, "stochastic_k": 0.05, "cci": 0.04, "williams_r": 0.04, "roc": 0.04, "volume_ratio": 0.06, "obv_slope": 0.05, "mfi": 0.06, "cmf": 0.05, "support_distance": 0.05, "resistance_distance": 0.05, "bollinger_pct_b": 0.05, "rsi_divergence": 0.05, "macd_divergence": 0.05},
        "cyclical":  {"ema_alignment": 0.08, "macd_histogram": 0.06, "adx_value": 0.05, "adx_direction": 0.05, "vwap_distance": 0.05, "ichimoku_cloud": 0.04, "rsi": 0.06, "stochastic_k": 0.04, "cci": 0.04, "williams_r": 0.03, "roc": 0.05, "volume_ratio": 0.07, "obv_slope": 0.05, "mfi": 0.05, "cmf": 0.04, "support_distance": 0.05, "resistance_distance": 0.05, "bollinger_pct_b": 0.04, "rsi_divergence": 0.05, "macd_divergence": 0.05},
        "financial": {"ema_alignment": 0.08, "macd_histogram": 0.06, "adx_value": 0.05, "adx_direction": 0.05, "vwap_distance": 0.04, "ichimoku_cloud": 0.04, "rsi": 0.06, "stochastic_k": 0.05, "cci": 0.04, "williams_r": 0.03, "roc": 0.04, "volume_ratio": 0.07, "obv_slope": 0.05, "mfi": 0.05, "cmf": 0.05, "support_distance": 0.05, "resistance_distance": 0.05, "bollinger_pct_b": 0.04, "rsi_divergence": 0.05, "macd_divergence": 0.05}
    },
    "BEAR": {
        "growth":    {"ema_alignment": 0.07, "macd_histogram": 0.06, "adx_value": 0.05, "adx_direction": 0.06, "vwap_distance": 0.04, "ichimoku_cloud": 0.04, "rsi": 0.07, "stochastic_k": 0.05, "cci": 0.04, "williams_r": 0.04, "roc": 0.04, "volume_ratio": 0.06, "obv_slope": 0.05, "mfi": 0.05, "cmf": 0.05, "support_distance": 0.06, "resistance_distance": 0.04, "bollinger_pct_b": 0.05, "rsi_divergence": 0.04, "macd_divergence": 0.04},
        "defensive": {"ema_alignment": 0.06, "macd_histogram": 0.05, "adx_value": 0.05, "adx_direction": 0.05, "vwap_distance": 0.04, "ichimoku_cloud": 0.04, "rsi": 0.08, "stochastic_k": 0.06, "cci": 0.04, "williams_r": 0.04, "roc": 0.04, "volume_ratio": 0.05, "obv_slope": 0.05, "mfi": 0.06, "cmf": 0.05, "support_distance": 0.06, "resistance_distance": 0.04, "bollinger_pct_b": 0.05, "rsi_divergence": 0.05, "macd_divergence": 0.04},
        "cyclical":  {"ema_alignment": 0.07, "macd_histogram": 0.06, "adx_value": 0.05, "adx_direction": 0.06, "vwap_distance": 0.04, "ichimoku_cloud": 0.04, "rsi": 0.07, "stochastic_k": 0.05, "cci": 0.04, "williams_r": 0.04, "roc": 0.04, "volume_ratio": 0.06, "obv_slope": 0.05, "mfi": 0.05, "cmf": 0.05, "support_distance": 0.06, "resistance_distance": 0.04, "bollinger_pct_b": 0.05, "rsi_divergence": 0.04, "macd_divergence": 0.04},
        "financial": {"ema_alignment": 0.07, "macd_histogram": 0.06, "adx_value": 0.05, "adx_direction": 0.06, "vwap_distance": 0.04, "ichimoku_cloud": 0.04, "rsi": 0.07, "stochastic_k": 0.05, "cci": 0.04, "williams_r": 0.04, "roc": 0.04, "volume_ratio": 0.06, "obv_slope": 0.05, "mfi": 0.05, "cmf": 0.05, "support_distance": 0.06, "resistance_distance": 0.04, "bollinger_pct_b": 0.05, "rsi_divergence": 0.04, "macd_divergence": 0.04}
    },
    "RANGE": {
        "growth":    {"ema_alignment": 0.05, "macd_histogram": 0.05, "adx_value": 0.04, "adx_direction": 0.04, "vwap_distance": 0.05, "ichimoku_cloud": 0.04, "rsi": 0.07, "stochastic_k": 0.06, "cci": 0.05, "williams_r": 0.05, "roc": 0.04, "volume_ratio": 0.05, "obv_slope": 0.04, "mfi": 0.05, "cmf": 0.05, "support_distance": 0.06, "resistance_distance": 0.06, "bollinger_pct_b": 0.06, "rsi_divergence": 0.05, "macd_divergence": 0.04},
        "defensive": {"ema_alignment": 0.05, "macd_histogram": 0.04, "adx_value": 0.04, "adx_direction": 0.04, "vwap_distance": 0.04, "ichimoku_cloud": 0.04, "rsi": 0.08, "stochastic_k": 0.06, "cci": 0.05, "williams_r": 0.05, "roc": 0.04, "volume_ratio": 0.05, "obv_slope": 0.04, "mfi": 0.06, "cmf": 0.05, "support_distance": 0.06, "resistance_distance": 0.06, "bollinger_pct_b": 0.06, "rsi_divergence": 0.05, "macd_divergence": 0.04},
        "cyclical":  {"ema_alignment": 0.05, "macd_histogram": 0.05, "adx_value": 0.04, "adx_direction": 0.04, "vwap_distance": 0.05, "ichimoku_cloud": 0.04, "rsi": 0.07, "stochastic_k": 0.06, "cci": 0.05, "williams_r": 0.05, "roc": 0.04, "volume_ratio": 0.05, "obv_slope": 0.04, "mfi": 0.05, "cmf": 0.05, "support_distance": 0.06, "resistance_distance": 0.06, "bollinger_pct_b": 0.06, "rsi_divergence": 0.05, "macd_divergence": 0.04},
        "financial": {"ema_alignment": 0.05, "macd_histogram": 0.05, "adx_value": 0.04, "adx_direction": 0.04, "vwap_distance": 0.04, "ichimoku_cloud": 0.04, "rsi": 0.07, "stochastic_k": 0.06, "cci": 0.05, "williams_r": 0.05, "roc": 0.04, "volume_ratio": 0.05, "obv_slope": 0.04, "mfi": 0.05, "cmf": 0.05, "support_distance": 0.06, "resistance_distance": 0.06, "bollinger_pct_b": 0.06, "rsi_divergence": 0.05, "macd_divergence": 0.04}
    },
    "VOLATILE": {
        "growth":    {"ema_alignment": 0.06, "macd_histogram": 0.05, "adx_value": 0.05, "adx_direction": 0.05, "vwap_distance": 0.04, "ichimoku_cloud": 0.04, "rsi": 0.06, "stochastic_k": 0.05, "cci": 0.05, "williams_r": 0.04, "roc": 0.04, "volume_ratio": 0.07, "obv_slope": 0.05, "mfi": 0.06, "cmf": 0.05, "support_distance": 0.06, "resistance_distance": 0.05, "bollinger_pct_b": 0.05, "rsi_divergence": 0.04, "macd_divergence": 0.04},
        "defensive": {"ema_alignment": 0.05, "macd_histogram": 0.05, "adx_value": 0.05, "adx_direction": 0.05, "vwap_distance": 0.04, "ichimoku_cloud": 0.04, "rsi": 0.07, "stochastic_k": 0.06, "cci": 0.05, "williams_r": 0.05, "roc": 0.03, "volume_ratio": 0.06, "obv_slope": 0.05, "mfi": 0.06, "cmf": 0.05, "support_distance": 0.06, "resistance_distance": 0.05, "bollinger_pct_b": 0.05, "rsi_divergence": 0.04, "macd_divergence": 0.04},
        "cyclical":  {"ema_alignment": 0.06, "macd_histogram": 0.05, "adx_value": 0.05, "adx_direction": 0.05, "vwap_distance": 0.04, "ichimoku_cloud": 0.04, "rsi": 0.06, "stochastic_k": 0.05, "cci": 0.05, "williams_r": 0.04, "roc": 0.04, "volume_ratio": 0.07, "obv_slope": 0.05, "mfi": 0.06, "cmf": 0.05, "support_distance": 0.06, "resistance_distance": 0.05, "bollinger_pct_b": 0.05, "rsi_divergence": 0.04, "macd_divergence": 0.04},
        "financial": {"ema_alignment": 0.06, "macd_histogram": 0.05, "adx_value": 0.05, "adx_direction": 0.05, "vwap_distance": 0.04, "ichimoku_cloud": 0.04, "rsi": 0.06, "stochastic_k": 0.05, "cci": 0.05, "williams_r": 0.04, "roc": 0.04, "volume_ratio": 0.07, "obv_slope": 0.05, "mfi": 0.06, "cmf": 0.05, "support_distance": 0.06, "resistance_distance": 0.05, "bollinger_pct_b": 0.05, "rsi_divergence": 0.04, "macd_divergence": 0.04}
    }
}


def _map_regime(regime_str: str) -> str:
    """Map MarketRegime enum values to weight keys."""
    regime_upper = str(regime_str).upper().replace('MARKETREGIME.', '')
    if regime_upper in ('STRONG_BULL', 'BULL'):
        return 'BULL'
    elif regime_upper in ('STRONG_BEAR', 'BEAR', 'CRASH'):
        return 'BEAR'
    elif regime_upper in ('HIGH', 'EXTREME'):
        return 'VOLATILE'
    else:
        return 'RANGE'


class AdaptiveTechnicalPillar(BasePillar):
    """
    Adaptive Technical Pillar v8.1
    
    - 18 indicators (raw values, no scoring)
    - Contextual normalization (volatility-blended anchors)
    - Learned weights per regime × sector group
    - Returns PillarScore compatible with existing interface
    """

    def __init__(self, weight: float = 0.25):
        super().__init__(weight)
        self.calculator = IndicatorsCalculator()
        self.normalizer = IndicatorNormalizer()
        self.weights = self._load_weights()
    
    def _load_weights(self) -> Dict:
        try:
            with open(WEIGHTS_PATH, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("Using default technical weights")
            return DEFAULT_WEIGHTS
    
    def save_weights(self):
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        with open(WEIGHTS_PATH, 'w') as f:
            json.dump(self.weights, f, indent=2)

    def get_name(self) -> str:
        return "Technical"

    async def analyze(self, symbol: str, data: Dict[str, Any]) -> PillarScore:
        """
        Full adaptive technical analysis.
        
        data keys:
            'df': OHLCV DataFrame (required)
            'sector': str (optional, e.g. 'Technology')
            'regime': str (optional, e.g. 'BULL')
            'market_cap_category': str (optional)
        """
        df = data.get('df')
        if df is None or len(df) < 30:
            row_count = len(df) if df is not None else 0
            logger.warning(f"[TECHNICAL-ADAPTIVE] {symbol}: Insufficient data ({row_count} rows)")
            return self._create_score(score=0, reasoning=f"Insufficient data ({row_count} rows)", data_quality=0.0)

        logger.info(f"[TECHNICAL-ADAPTIVE] {symbol}: Analyzing {len(df)} data points with 18 indicators...")

        # 1. Calculate all raw indicators
        raw = self.calculator.calculate_all(df)

        # 2. Build context
        sector = data.get('sector', 'Technology')
        regime_raw = data.get('regime', 'RANGE')
        context = {
            'volatility_percentile': raw.atr_percentile or 0.5,
            'sector': sector,
            'regime': str(regime_raw),
            'market_cap_category': data.get('market_cap_category', 'large'),
        }

        # 3. Normalize all indicators
        raw_dict = raw.to_dict()
        # Remove metadata keys
        raw_dict.pop('atr_value', None)
        raw_dict.pop('atr_percentile', None)
        normalized = self.normalizer.normalize_all(raw_dict, context)

        # 4. Get weights for this regime × sector
        regime_key = _map_regime(regime_raw)
        sector_group = SECTOR_GROUPS.get(sector, 'growth')
        
        regime_weights = self.weights.get(regime_key, self.weights.get('RANGE', {}))
        group_weights = regime_weights.get(sector_group, regime_weights.get('growth', {}))

        # 5. Compute weighted score
        total_score = 0.0
        total_weight = 0.0
        factors = []

        for ind_name, norm_score in normalized.items():
            w = group_weights.get(ind_name, 0.05)
            total_score += norm_score * w
            total_weight += w
            
            raw_val = raw_dict.get(ind_name)
            factors.append({
                'indicator': ind_name,
                'raw_value': round(raw_val, 4) if raw_val is not None else None,
                'normalized': round(norm_score, 1),
                'weight': round(w, 3),
                'contribution': round(norm_score * w, 2),
            })

        # Normalize by total weight (should be ~1.0 but safety)
        if total_weight > 0:
            total_score /= total_weight

        # total_score is now 0-100 (50 = neutral)
        # Convert to internal scale (-100 to +100) for BasePillar compatibility
        internal_score = (total_score - 50) * 2  # 0→-100, 50→0, 100→+100

        # Sort factors by contribution
        factors.sort(key=lambda x: abs(x['contribution']), reverse=True)

        # Confidence from indicator agreement
        confidence = self._calc_confidence(normalized)

        # Quality based on how many indicators computed
        quality = min(1.0, raw.available_count() / 15.0)

        # Reasoning
        reasoning = self._build_reasoning(symbol, total_score, regime_key, sector_group, factors, raw)

        # Log summary
        signal = "bullish" if total_score > 55 else "bearish" if total_score < 45 else "neutral"
        logger.info(
            f"[TECHNICAL-ADAPTIVE] {symbol}: Score={total_score:.1f}/100 ({signal}) | "
            f"Regime={regime_key} Sector={sector_group} Indicators={raw.available_count()}/18 "
            f"Confidence={confidence:.2f}"
        )

        return self._create_score(
            score=internal_score,
            reasoning=reasoning,
            factors=factors,
            confidence=confidence,
            data_quality=quality
        )

    def _calc_confidence(self, normalized: Dict[str, float]) -> float:
        """Confidence based on indicator agreement."""
        if not normalized:
            return 0.5
        values = list(normalized.values())
        bullish = sum(1 for v in values if v > 55)
        bearish = sum(1 for v in values if v < 45)
        total = len(values)
        agreement = max(bullish, bearish) / total if total > 0 else 0.5
        return 0.5 + agreement * 0.5

    def _build_reasoning(self, symbol, score, regime, sector_group, factors, raw: RawIndicators) -> str:
        """Build human-readable reasoning."""
        parts = []
        
        if score > 65:
            outlook = "STRONGLY BULLISH"
        elif score > 55:
            outlook = "BULLISH"
        elif score > 45:
            outlook = "NEUTRAL"
        elif score > 35:
            outlook = "BEARISH"
        else:
            outlook = "STRONGLY BEARISH"
        
        parts.append(f"Adaptive Technical ({regime}/{sector_group}): {outlook} ({score:.1f}/100)")
        parts.append(f"Indicators computed: {raw.available_count()}/18")
        
        # Top 5 contributors
        top = factors[:5]
        parts.append("\nTop signals:")
        for f in top:
            direction = "↑" if f['normalized'] > 55 else "↓" if f['normalized'] < 45 else "→"
            parts.append(f"  {direction} {f['indicator']}: raw={f['raw_value']} → {f['normalized']:.0f}/100 (w={f['weight']:.2f})")
        
        # Divergences
        if raw.rsi_divergence and raw.rsi_divergence != 0:
            div_type = "BULLISH" if raw.rsi_divergence > 0 else "BEARISH"
            parts.append(f"\n⚡ RSI Divergence: {div_type}")
        if raw.macd_divergence and raw.macd_divergence != 0:
            div_type = "BULLISH" if raw.macd_divergence > 0 else "BEARISH"
            parts.append(f"⚡ MACD Divergence: {div_type}")
        
        return "\n".join(parts)

    def update_weight(self, regime: str, sector_group: str, indicator: str, new_weight: float):
        """Update a single weight (for learning feedback)."""
        if regime in self.weights and sector_group in self.weights[regime]:
            old = self.weights[regime][sector_group].get(indicator, 0.05)
            # EMA update
            self.weights[regime][sector_group][indicator] = round(old * 0.9 + new_weight * 0.1, 4)
            # Re-normalize
            total = sum(self.weights[regime][sector_group].values())
            if total > 0:
                self.weights[regime][sector_group] = {
                    k: round(v / total, 4) for k, v in self.weights[regime][sector_group].items()
                }


# Singleton
_adaptive_pillar: Optional[AdaptiveTechnicalPillar] = None


def get_adaptive_technical_pillar() -> AdaptiveTechnicalPillar:
    """Get or create the AdaptiveTechnicalPillar singleton."""
    global _adaptive_pillar
    if _adaptive_pillar is None:
        _adaptive_pillar = AdaptiveTechnicalPillar()
    return _adaptive_pillar
