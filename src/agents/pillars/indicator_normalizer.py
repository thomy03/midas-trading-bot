"""
Indicator Normalizer - Contextual normalization of raw indicators to 0-100 scores.

Uses smooth interpolation with adaptive anchor points.
No if/else scoring — continuous functions only.
"""

import json
import os
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Path to anchor points file
ANCHORS_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'learned_weights', 'indicator_anchors.json')


def _smooth_interpolate(value: float, anchors: Dict[float, float]) -> float:
    """
    Smooth interpolation between anchor points.
    Uses piecewise linear interpolation (numpy.interp).
    
    Args:
        value: Raw indicator value
        anchors: {raw_value: normalized_score} mapping
    
    Returns:
        Interpolated score (0-100)
    """
    if not anchors:
        return 50.0
    
    xs = sorted(anchors.keys())
    ys = [anchors[x] for x in xs]
    
    # numpy interp handles extrapolation at boundaries
    result = float(np.interp(value, xs, ys))
    return max(0.0, min(100.0, result))


def _get_volatility_key(vol_percentile: float) -> str:
    """Map volatility percentile to key for anchor lookup."""
    if vol_percentile < 0.3:
        return 'low_vol'
    elif vol_percentile < 0.7:
        return 'mid_vol'
    else:
        return 'high_vol'


def _blend_anchors(anchors_dict: Dict[str, Dict], vol_percentile: float) -> Dict[float, float]:
    """
    Blend between volatility-keyed anchor sets using smooth weighting.
    Instead of hard cutoffs, interpolate between adjacent anchor sets.
    """
    if len(anchors_dict) == 1:
        return {float(k): v for k, v in list(anchors_dict.values())[0].items()}
    
    # Define vol centers for each key
    vol_centers = {'low_vol': 0.15, 'mid_vol': 0.5, 'high_vol': 0.85}
    
    # Compute weights based on distance to each center
    weights = {}
    for key, center in vol_centers.items():
        if key in anchors_dict:
            dist = abs(vol_percentile - center)
            weights[key] = max(0, 1.0 - dist * 2.5)  # Smooth falloff
    
    total_w = sum(weights.values())
    if total_w == 0:
        # Fallback to nearest
        key = _get_volatility_key(vol_percentile)
        if key in anchors_dict:
            return {float(k): v for k, v in anchors_dict[key].items()}
        return {float(k): v for k, v in list(anchors_dict.values())[0].items()}
    
    # Normalize weights
    weights = {k: v / total_w for k, v in weights.items()}
    
    # Collect all x-points across all sets
    all_xs = set()
    for key in weights:
        all_xs.update(float(x) for x in anchors_dict[key].keys())
    all_xs = sorted(all_xs)
    
    # Blend y-values at each x-point
    blended = {}
    for x in all_xs:
        y = 0.0
        for key, w in weights.items():
            anchor_set = {float(k): v for k, v in anchors_dict[key].items()}
            xs_sorted = sorted(anchor_set.keys())
            ys_sorted = [anchor_set[xi] for xi in xs_sorted]
            y += w * float(np.interp(x, xs_sorted, ys_sorted))
        blended[x] = y
    
    return blended


class IndicatorNormalizer:
    """Normalizes raw indicator values to 0-100 scores using adaptive anchor points."""
    
    def __init__(self, anchors_path: str = None):
        self.anchors_path = anchors_path or ANCHORS_PATH
        self._anchors = self._load_anchors()
    
    def _load_anchors(self) -> Dict:
        """Load anchor points from JSON file."""
        try:
            with open(self.anchors_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load anchors from {self.anchors_path}: {e}. Using defaults.")
            return self._default_anchors()
    
    def save_anchors(self):
        """Persist current anchors to disk."""
        os.makedirs(os.path.dirname(self.anchors_path), exist_ok=True)
        with open(self.anchors_path, 'w') as f:
            json.dump(self._anchors, f, indent=2)
    
    def normalize(self, indicator_name: str, raw_value: float, context: Dict[str, Any]) -> float:
        """
        Normalize a raw indicator value to 0-100 score.
        
        Args:
            indicator_name: Name matching keys in anchors dict
            raw_value: Raw indicator value
            context: {'volatility_percentile': 0-1, 'regime': str, ...}
        
        Returns:
            Normalized score 0-100 (50 = neutral)
        """
        if raw_value is None:
            return 50.0
        
        vol_pct = context.get('volatility_percentile', 0.5)
        
        indicator_anchors = self._anchors.get(indicator_name, {})
        if not indicator_anchors:
            logger.debug(f"No anchors for {indicator_name}, returning 50")
            return 50.0
        
        # Check if anchors are volatility-keyed
        first_key = list(indicator_anchors.keys())[0]
        if first_key in ('low_vol', 'mid_vol', 'high_vol'):
            blended = _blend_anchors(indicator_anchors, vol_pct)
        else:
            blended = {float(k): v for k, v in indicator_anchors.items()}
        
        return _smooth_interpolate(raw_value, blended)
    
    def normalize_all(self, raw_indicators: Dict[str, float], context: Dict[str, Any]) -> Dict[str, float]:
        """Normalize all indicators at once."""
        result = {}
        for name, value in raw_indicators.items():
            if value is not None:
                result[name] = self.normalize(name, value, context)
        return result
    
    def update_anchor(self, indicator_name: str, vol_key: str, raw_value: float, new_score: float):
        """
        Update an anchor point (for learning from feedback).
        Adds/adjusts the nearest anchor point.
        """
        if indicator_name not in self._anchors:
            return
        
        anchors = self._anchors[indicator_name]
        if vol_key in anchors:
            target = anchors[vol_key]
        else:
            target = anchors
        
        # Find nearest anchor and adjust it slightly toward new_score
        str_key = str(int(round(raw_value)))
        if str_key in target:
            old = target[str_key]
            # Exponential moving average update (learning rate 0.1)
            target[str_key] = round(old * 0.9 + new_score * 0.1, 1)
    
    @staticmethod
    def _default_anchors() -> Dict:
        """Default anchor points for all 18 indicators."""
        return {
            # ─── TREND ───
            "ema_alignment": {
                "low_vol": {"-10": 10, "-5": 25, "-2": 35, "0": 50, "2": 65, "5": 75, "10": 90},
                "mid_vol": {"-10": 15, "-5": 28, "-2": 38, "0": 50, "2": 62, "5": 72, "10": 85},
                "high_vol": {"-10": 20, "-5": 32, "-2": 40, "0": 50, "2": 60, "5": 68, "10": 80}
            },
            "macd_histogram": {
                "low_vol": {"-1": 5, "-0.5": 20, "-0.2": 35, "0": 50, "0.2": 65, "0.5": 80, "1": 95},
                "mid_vol": {"-1": 10, "-0.5": 25, "-0.2": 38, "0": 50, "0.2": 62, "0.5": 75, "1": 90},
                "high_vol": {"-1": 15, "-0.5": 30, "-0.2": 40, "0": 50, "0.2": 60, "0.5": 70, "1": 85}
            },
            "adx_value": {
                "0": 30, "15": 40, "20": 45, "25": 55, "30": 65, "40": 75, "50": 85, "70": 90
            },
            "adx_direction": {
                "-40": 5, "-20": 20, "-10": 35, "0": 50, "10": 65, "20": 80, "40": 95
            },
            "vwap_distance": {
                "low_vol": {"-5": 10, "-2": 25, "-1": 35, "0": 50, "1": 65, "2": 75, "5": 90},
                "mid_vol": {"-5": 15, "-2": 28, "-1": 38, "0": 50, "1": 62, "2": 72, "5": 85},
                "high_vol": {"-5": 20, "-2": 32, "-1": 40, "0": 50, "1": 60, "2": 68, "5": 80}
            },
            "ichimoku_cloud": {
                "-5": 10, "-2": 25, "-1": 35, "0": 50, "1": 65, "2": 75, "5": 90
            },
            # ─── MOMENTUM ───
            "rsi": {
                "low_vol": {"0": 95, "20": 80, "30": 65, "40": 55, "50": 50, "60": 45, "70": 35, "80": 20, "100": 5},
                "mid_vol": {"0": 90, "20": 75, "30": 60, "40": 53, "50": 50, "60": 47, "70": 40, "80": 25, "100": 10},
                "high_vol": {"0": 80, "20": 65, "30": 55, "40": 52, "50": 50, "60": 48, "70": 45, "80": 35, "100": 20}
            },
            "stochastic_k": {
                "low_vol": {"0": 90, "10": 80, "20": 65, "50": 50, "80": 35, "90": 20, "100": 10},
                "high_vol": {"0": 75, "10": 65, "20": 58, "50": 50, "80": 42, "90": 35, "100": 25}
            },
            "cci": {
                "low_vol": {"-200": 90, "-100": 75, "-50": 60, "0": 50, "50": 40, "100": 25, "200": 10},
                "high_vol": {"-200": 80, "-100": 65, "-50": 55, "0": 50, "50": 45, "100": 35, "200": 20}
            },
            "williams_r": {
                "-100": 90, "-80": 70, "-60": 55, "-50": 50, "-40": 45, "-20": 30, "0": 10
            },
            "roc": {
                "low_vol": {"-10": 5, "-5": 20, "-2": 35, "0": 50, "2": 65, "5": 80, "10": 95},
                "mid_vol": {"-10": 10, "-5": 25, "-2": 38, "0": 50, "2": 62, "5": 75, "10": 90},
                "high_vol": {"-10": 20, "-5": 32, "-2": 42, "0": 50, "2": 58, "5": 68, "10": 80}
            },
            # ─── VOLUME ───
            "volume_ratio": {
                "0": 20, "0.5": 35, "0.8": 45, "1": 50, "1.5": 60, "2": 70, "3": 80, "5": 90
            },
            "obv_slope": {
                "-10": 10, "-5": 25, "-2": 38, "0": 50, "2": 62, "5": 75, "10": 90
            },
            "mfi": {
                "low_vol": {"0": 95, "20": 80, "30": 65, "50": 50, "70": 35, "80": 20, "100": 5},
                "high_vol": {"0": 80, "20": 65, "30": 55, "50": 50, "70": 45, "80": 35, "100": 20}
            },
            "cmf": {
                "-1": 5, "-0.5": 15, "-0.2": 30, "-0.1": 40, "0": 50, "0.1": 60, "0.2": 70, "0.5": 85, "1": 95
            },
            # ─── STRUCTURE ───
            "support_distance": {
                "0": 30, "1": 40, "2": 50, "3": 55, "5": 60, "10": 65
            },
            "resistance_distance": {
                "0": 70, "1": 60, "2": 50, "3": 45, "5": 40, "10": 35
            },
            "bollinger_pct_b": {
                "low_vol": {"-0.2": 90, "0": 80, "0.2": 65, "0.5": 50, "0.8": 35, "1": 20, "1.2": 10},
                "high_vol": {"-0.2": 75, "0": 65, "0.2": 58, "0.5": 50, "0.8": 42, "1": 35, "1.2": 25}
            },
            # ─── DIVERGENCES ───
            "rsi_divergence": {
                "-1": 15, "0": 50, "1": 85
            },
            "macd_divergence": {
                "-1": 15, "0": 50, "1": 85
            }
        }
