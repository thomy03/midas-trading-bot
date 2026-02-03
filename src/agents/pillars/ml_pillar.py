"""
ML Pillar - Adaptive Machine Learning scoring.

Uses Random Forest to:
1. Extract 40+ technical features
2. Detect market regime (bull/bear/range/volatile)
3. Predict trade success probability
4. Adjust weights of other pillars dynamically

Contributes 20-25% to final decision score.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
import os
from dataclasses import dataclass

from .base import BasePillar, PillarScore

# ML imports - graceful fallback if not installed
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    RandomForestClassifier = None
    StandardScaler = None

logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Current market regime detection result"""
    regime: str  # 'BULL', 'BEAR', 'RANGE', 'VOLATILE'
    confidence: float  # 0-1
    spy_trend: str  # 'up', 'down', 'flat'
    vix_level: float
    details: Dict[str, Any]


class MLPillar(BasePillar):
    """
    Adaptive ML Pillar that learns which indicators work best.
    
    Features:
    - 40+ technical indicators as features
    - Market regime detection for weight adjustment
    - Random Forest for trade success prediction
    - Monthly retrain on trade history
    """
    
    # Feature groups
    FEATURE_GROUPS = {
        'trend': [
            'ema_cross_20_50', 'ema_cross_50_200', 'macd_histogram',
            'macd_signal_cross', 'adx_value', 'adx_direction',
            'supertrend_signal', 'aroon_oscillator', 'price_vs_ema20',
            'price_vs_ema50'
        ],
        'momentum': [
            'rsi_14', 'rsi_slope', 'stoch_k', 'stoch_d',
            'williams_r', 'cci_20', 'roc_10', 'momentum_10',
            'rsi_oversold', 'rsi_overbought'
        ],
        'volume': [
            'volume_ratio_20', 'obv_trend', 'obv_slope',
            'cmf_20', 'mfi_14', 'volume_trend_5d',
            'volume_breakout', 'price_volume_trend'
        ],
        'volatility': [
            'atr_percent', 'atr_ratio', 'bb_width', 'bb_percent',
            'volatility_20d', 'volatility_expansion'
        ],
        'regime': [
            'spy_above_ema50', 'vix_level', 'vix_percentile',
            'sector_momentum', 'market_breadth', 'correlation_spy'
        ]
    }
    
    # Weight adjustments by regime
    REGIME_WEIGHTS = {
        'BULL': {
            'technical': 0.22,
            'fundamental': 0.25,
            'sentiment': 0.18,
            'news': 0.05,
            'ml': 0.30
        },
        'BEAR': {
            'technical': 0.30,
            'fundamental': 0.18,
            'sentiment': 0.22,
            'news': 0.05,
            'ml': 0.25
        },
        'RANGE': {
            'technical': 0.27,
            'fundamental': 0.22,
            'sentiment': 0.16,
            'news': 0.05,
            'ml': 0.30
        },
        'VOLATILE': {
            'technical': 0.35,
            'fundamental': 0.15,
            'sentiment': 0.15,
            'news': 0.05,
            'ml': 0.30
        }
    }
    
    def __init__(self, weight: float = 0.25, model_path: str = None):
        super().__init__(weight)
        self.model_path = model_path or '/root/tradingbot-github/data/ml_model.joblib'
        self.scaler_path = self.model_path.replace('.joblib', '_scaler.joblib')
        
        self.model = None
        self.scaler = None
        self.feature_names = self._get_all_features()
        self.last_regime: Optional[MarketRegime] = None
        
        # Load existing model if available
        self._load_model()
    
    def _get_all_features(self) -> List[str]:
        """Get flat list of all feature names"""
        features = []
        for group_features in self.FEATURE_GROUPS.values():
            features.extend(group_features)
        return features
    
    def _load_model(self):
        """Load pre-trained model if exists"""
        if not ML_AVAILABLE:
            logger.warning("[ML] sklearn not available - ML pillar will use heuristics")
            return
            
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                if os.path.exists(self.scaler_path):
                    self.scaler = joblib.load(self.scaler_path)
                logger.info(f"[ML] Loaded model from {self.model_path}")
            except Exception as e:
                logger.warning(f"[ML] Failed to load model: {e}")
    
    def get_name(self) -> str:
        return "ML_Adaptive"
    
    async def analyze(
        self,
        symbol: str,
        data: Dict[str, Any]
    ) -> PillarScore:
        """
        Perform ML-based analysis on a symbol.
        
        Args:
            symbol: Stock symbol
            data: Must contain 'df' with OHLCV DataFrame
            
        Returns:
            PillarScore with ML prediction and regime info
        """
        df = data.get('df')
        if df is None or len(df) < 50:
            return self._create_score(
                score=50,  # Neutral score
                reasoning="Insufficient data for ML analysis",
                data_quality=0.0
            )
        
        logger.info(f"[ML] {symbol}: Analyzing with {len(df)} data points...")
        
        # 1. Extract features
        features, feature_details = self._extract_features(df)
        
        # 2. Detect market regime
        regime = self._detect_regime(df, data)
        self.last_regime = regime
        
        # 3. ML Prediction or heuristic fallback
        if self.model is not None and ML_AVAILABLE:
            score, prediction_details = self._ml_predict(features)
        else:
            score, prediction_details = self._heuristic_score(features, feature_details)
        
        # 4. Generate reasoning
        reasoning = self._generate_reasoning(
            symbol, score, regime, feature_details, prediction_details
        )
        
        # 5. Calculate confidence
        confidence = self._calculate_confidence(features, regime)
        
        # V4.4: Verbose logging
        logger.info(
            f"[ML] {symbol}: Score={score:.1f}/100 | "
            f"Regime={regime.regime} | "
            f"Confidence={confidence:.0%}"
        )
        
        return self._create_score(
            score=score,
            reasoning=reasoning,
            factors=[{
                'regime': regime.regime,
                'regime_confidence': regime.confidence,
                'feature_summary': feature_details,
                'prediction': prediction_details
            }],
            confidence=confidence,
        )
    
    def _extract_features(self, df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Extract all features from OHLCV data"""
        features = {}
        details = {}
        
        try:
            close = df['Close']
            high = df['High']
            low = df['Low']
            volume = df['Volume']
            
            # === TREND FEATURES ===
            
            # EMAs
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            ema200 = close.ewm(span=200, adjust=False).mean() if len(df) >= 200 else ema50
            
            features['ema_cross_20_50'] = 1 if ema20.iloc[-1] > ema50.iloc[-1] else -1
            features['ema_cross_50_200'] = 1 if ema50.iloc[-1] > ema200.iloc[-1] else -1
            features['price_vs_ema20'] = (close.iloc[-1] - ema20.iloc[-1]) / ema20.iloc[-1] * 100
            features['price_vs_ema50'] = (close.iloc[-1] - ema50.iloc[-1]) / ema50.iloc[-1] * 100
            
            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            features['macd_histogram'] = macd.iloc[-1] - signal.iloc[-1]
            features['macd_signal_cross'] = 1 if macd.iloc[-1] > signal.iloc[-1] else -1
            
            # ADX
            adx, plus_di, minus_di = self._calculate_adx(high, low, close)
            features['adx_value'] = adx
            features['adx_direction'] = 1 if plus_di > minus_di else -1
            
            # Supertrend (simplified)
            atr = self._calculate_atr(high, low, close)
            upper_band = (high + low) / 2 + (2 * atr)
            features['supertrend_signal'] = 1 if close.iloc[-1] > upper_band.iloc[-1] else -1
            
            # Aroon
            aroon_up = ((14 - (14 - high.rolling(14).apply(lambda x: x.argmax()))) / 14) * 100
            aroon_down = ((14 - (14 - low.rolling(14).apply(lambda x: x.argmin()))) / 14) * 100
            features['aroon_oscillator'] = aroon_up.iloc[-1] - aroon_down.iloc[-1]
            
            details['trend'] = {
                'ema_alignment': 'bullish' if features['ema_cross_20_50'] > 0 and features['ema_cross_50_200'] > 0 else 'bearish',
                'macd': 'bullish' if features['macd_signal_cross'] > 0 else 'bearish',
                'adx': adx
            }
            
            # === MOMENTUM FEATURES ===
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            features['rsi_14'] = rsi.iloc[-1]
            features['rsi_slope'] = rsi.iloc[-1] - rsi.iloc[-5] if len(rsi) > 5 else 0
            features['rsi_oversold'] = 1 if rsi.iloc[-1] < 30 else 0
            features['rsi_overbought'] = 1 if rsi.iloc[-1] > 70 else 0
            
            # Stochastic
            lowest_low = low.rolling(14).min()
            highest_high = high.rolling(14).max()
            stoch_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
            stoch_d = stoch_k.rolling(3).mean()
            
            features['stoch_k'] = stoch_k.iloc[-1]
            features['stoch_d'] = stoch_d.iloc[-1]
            
            # Williams %R
            features['williams_r'] = -100 * (highest_high.iloc[-1] - close.iloc[-1]) / (highest_high.iloc[-1] - lowest_low.iloc[-1])
            
            # CCI
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(20).mean()
            mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
            features['cci_20'] = (tp.iloc[-1] - sma_tp.iloc[-1]) / (0.015 * mad.iloc[-1]) if mad.iloc[-1] != 0 else 0
            
            # ROC & Momentum
            features['roc_10'] = ((close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]) * 100 if len(close) > 10 else 0
            features['momentum_10'] = close.iloc[-1] - close.iloc[-10] if len(close) > 10 else 0
            
            details['momentum'] = {
                'rsi': features['rsi_14'],
                'stoch': features['stoch_k'],
                'cci': features['cci_20']
            }
            
            # === VOLUME FEATURES ===
            
            avg_vol_20 = volume.rolling(20).mean()
            features['volume_ratio_20'] = volume.iloc[-1] / avg_vol_20.iloc[-1] if avg_vol_20.iloc[-1] > 0 else 1
            features['volume_trend_5d'] = (volume.iloc[-1] - volume.iloc[-5]) / volume.iloc[-5] if len(volume) > 5 and volume.iloc[-5] > 0 else 0
            features['volume_breakout'] = 1 if features['volume_ratio_20'] > 2.0 else 0
            
            # OBV
            obv = (np.sign(close.diff()) * volume).cumsum()
            features['obv_trend'] = 1 if obv.iloc[-1] > obv.iloc[-20] else -1 if len(obv) > 20 else 0
            features['obv_slope'] = (obv.iloc[-1] - obv.iloc[-5]) / abs(obv.iloc[-5]) * 100 if len(obv) > 5 and obv.iloc[-5] != 0 else 0
            
            # CMF
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)
            mfv = mfm * volume
            features['cmf_20'] = mfv.rolling(20).sum().iloc[-1] / volume.rolling(20).sum().iloc[-1] if volume.rolling(20).sum().iloc[-1] > 0 else 0
            
            # MFI
            tp = (high + low + close) / 3
            mf = tp * volume
            pos_mf = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
            neg_mf = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
            features['mfi_14'] = 100 - (100 / (1 + pos_mf.iloc[-1] / neg_mf.iloc[-1])) if neg_mf.iloc[-1] > 0 else 50
            
            # Price Volume Trend
            features['price_volume_trend'] = features['volume_ratio_20'] * np.sign(close.iloc[-1] - close.iloc[-2])
            
            details['volume'] = {
                'ratio': features['volume_ratio_20'],
                'obv_trend': 'accumulation' if features['obv_trend'] > 0 else 'distribution',
                'cmf': features['cmf_20']
            }
            
            # === VOLATILITY FEATURES ===
            
            atr_series = self._calculate_atr(high, low, close)
            features['atr_percent'] = atr_series.iloc[-1] / close.iloc[-1] * 100
            features['atr_ratio'] = atr_series.iloc[-1] / atr_series.rolling(20).mean().iloc[-1] if len(atr_series) > 20 else 1
            
            # Bollinger Bands
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            bb_upper = sma20 + (2 * std20)
            bb_lower = sma20 - (2 * std20)
            features['bb_width'] = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / sma20.iloc[-1] * 100
            features['bb_percent'] = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) if bb_upper.iloc[-1] != bb_lower.iloc[-1] else 0.5
            
            # Historical volatility
            returns = close.pct_change()
            features['volatility_20d'] = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            features['volatility_expansion'] = 1 if features['atr_ratio'] > 1.5 else 0
            
            details['volatility'] = {
                'atr_pct': features['atr_percent'],
                'bb_position': features['bb_percent'],
                'hv_20d': features['volatility_20d']
            }
            
            # === REGIME FEATURES ===
            # These are placeholders - will be filled by _detect_regime
            features['spy_above_ema50'] = 0
            features['vix_level'] = 20
            features['vix_percentile'] = 50
            features['sector_momentum'] = 0
            features['market_breadth'] = 50
            features['correlation_spy'] = 0.5
            
        except Exception as e:
            logger.warning(f"[ML] Feature extraction error: {e}")
        
        return features, details
    
    def _calculate_adx(self, high, low, close, period=14) -> Tuple[float, float, float]:
        """Calculate ADX indicator"""
        try:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
            minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
            
            plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            return adx.iloc[-1], plus_di.iloc[-1], minus_di.iloc[-1]
        except:
            return 25, 25, 25
    
    def _calculate_atr(self, high, low, close, period=14):
        """Calculate ATR"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def _detect_regime(self, df: pd.DataFrame, data: Dict) -> MarketRegime:
        """Detect current market regime"""
        try:
            # Try to get SPY and VIX data from context
            spy_data = data.get('spy_df')
            vix_level = data.get('vix_level', 20)
            
            # Heuristic regime detection based on available data
            close = df['Close']
            ema50 = close.ewm(span=50, adjust=False).mean()
            
            # Price trend
            price_above_ema50 = close.iloc[-1] > ema50.iloc[-1]
            price_trend = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100 if len(close) > 20 else 0
            
            # Volatility
            returns = close.pct_change()
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
            
            # Determine regime
            if vix_level > 30 or volatility > 40:
                regime = 'VOLATILE'
                confidence = 0.8
            elif price_above_ema50 and price_trend > 5:
                regime = 'BULL'
                confidence = min(0.9, 0.5 + price_trend / 20)
            elif not price_above_ema50 and price_trend < -5:
                regime = 'BEAR'
                confidence = min(0.9, 0.5 + abs(price_trend) / 20)
            else:
                regime = 'RANGE'
                confidence = 0.6
            
            return MarketRegime(
                regime=regime,
                confidence=confidence,
                spy_trend='up' if price_above_ema50 else 'down',
                vix_level=vix_level,
                details={
                    'price_trend_20d': price_trend,
                    'volatility': volatility,
                    'above_ema50': price_above_ema50
                }
            )
            
        except Exception as e:
            logger.warning(f"[ML] Regime detection error: {e}")
            return MarketRegime(
                regime='RANGE',
                confidence=0.5,
                spy_trend='flat',
                vix_level=20,
                details={'error': str(e)}
            )
    
    def _ml_predict(self, features: Dict[str, float]) -> Tuple[float, Dict]:
        """Make prediction using trained model"""
        try:
            # Prepare feature vector
            X = np.array([[features.get(f, 0) for f in self.feature_names]])
            
            # Scale features
            if self.scaler:
                X = self.scaler.transform(X)
            
            # Predict probability
            proba = self.model.predict_proba(X)[0]
            
            # Convert to score (0-100)
            # Assuming class 1 = successful trade
            success_proba = proba[1] if len(proba) > 1 else proba[0]
            score = success_proba * 100
            
            # Get feature importances
            importances = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            top_features = sorted(
                importances.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            return score, {
                'method': 'ml_model',
                'success_probability': success_proba,
                'top_features': top_features
            }
            
        except Exception as e:
            logger.warning(f"[ML] Prediction error: {e}, falling back to heuristics")
            return self._heuristic_score(features, {})
    
    def _heuristic_score(self, features: Dict[str, float], details: Dict) -> Tuple[float, Dict]:
        """Fallback heuristic scoring when model not available"""
        score = 50  # Start neutral
        factors = []
        
        # Trend factors (+/- 15 max)
        if features.get('ema_cross_20_50', 0) > 0:
            score += 5
            factors.append('EMA 20>50 bullish')
        else:
            score -= 5
            
        if features.get('macd_signal_cross', 0) > 0:
            score += 5
            factors.append('MACD bullish')
        else:
            score -= 5
            
        if features.get('adx_value', 0) > 25:
            direction_boost = 5 if features.get('adx_direction', 0) > 0 else -5
            score += direction_boost
            factors.append(f'Strong trend ADX={features.get("adx_value", 0):.0f}')
        
        # Momentum factors (+/- 15 max)
        rsi = features.get('rsi_14', 50)
        if rsi < 30:
            score += 10
            factors.append(f'Oversold RSI={rsi:.0f}')
        elif rsi > 70:
            score -= 10
            factors.append(f'Overbought RSI={rsi:.0f}')
        elif rsi > 50:
            score += 3
            
        if features.get('stoch_k', 50) < 20:
            score += 5
        elif features.get('stoch_k', 50) > 80:
            score -= 5
        
        # Volume factors (+/- 10 max)
        vol_ratio = features.get('volume_ratio_20', 1)
        if vol_ratio > 2:
            volume_boost = 10 if features.get('price_volume_trend', 0) > 0 else -10
            score += volume_boost
            factors.append(f'High volume {vol_ratio:.1f}x')
        elif vol_ratio > 1.5:
            score += 5 if features.get('price_volume_trend', 0) > 0 else -5
            
        # Volatility factors (+/- 10 max)
        if features.get('bb_percent', 0.5) < 0.2:
            score += 8
            factors.append('Near lower BB')
        elif features.get('bb_percent', 0.5) > 0.8:
            score -= 8
            factors.append('Near upper BB')
        
        # Clamp score to 0-100
        score = max(0, min(100, score))
        
        return score, {
            'method': 'heuristic',
            'factors': factors
        }
    
    def _generate_reasoning(
        self,
        symbol: str,
        score: float,
        regime: MarketRegime,
        feature_details: Dict,
        prediction_details: Dict
    ) -> str:
        """Generate human-readable reasoning"""
        parts = []
        
        # Score interpretation
        if score >= 70:
            parts.append(f"ML Score: STRONGLY BULLISH ({score:.0f}/100)")
        elif score >= 55:
            parts.append(f"ML Score: BULLISH ({score:.0f}/100)")
        elif score >= 45:
            parts.append(f"ML Score: NEUTRAL ({score:.0f}/100)")
        elif score >= 30:
            parts.append(f"ML Score: BEARISH ({score:.0f}/100)")
        else:
            parts.append(f"ML Score: STRONGLY BEARISH ({score:.0f}/100)")
        
        # Regime info
        parts.append(f"Market Regime: {regime.regime} (confidence: {regime.confidence:.0%})")
        parts.append(f"Recommended pillar weights: {self.REGIME_WEIGHTS.get(regime.regime, {})}")
        
        # Key features
        if prediction_details.get('method') == 'ml_model':
            top_features = prediction_details.get('top_features', [])[:3]
            if top_features:
                parts.append("Top predictive features:")
                for feat, imp in top_features:
                    parts.append(f"  - {feat}: {imp:.3f}")
        else:
            factors = prediction_details.get('factors', [])[:5]
            if factors:
                parts.append("Key signals:")
                for f in factors:
                    parts.append(f"  - {f}")
        
        return "\n".join(parts)
    
    def _calculate_confidence(self, features: Dict, regime: MarketRegime) -> float:
        """Calculate confidence in the ML prediction"""
        confidence = 0.5
        
        # Higher confidence if regime is clear
        confidence += regime.confidence * 0.2
        
        # Higher confidence if indicators agree
        bullish_count = sum(1 for k, v in features.items() 
                          if isinstance(v, (int, float)) and v > 0)
        bearish_count = sum(1 for k, v in features.items() 
                          if isinstance(v, (int, float)) and v < 0)
        total = bullish_count + bearish_count
        
        if total > 0:
            agreement = max(bullish_count, bearish_count) / total
            confidence += agreement * 0.3
        
        return min(confidence, 1.0)
    
    def get_regime_weights(self) -> Dict[str, float]:
        """Get current recommended weights based on last detected regime"""
        if self.last_regime:
            return self.REGIME_WEIGHTS.get(self.last_regime.regime, self.REGIME_WEIGHTS['RANGE'])
        return self.REGIME_WEIGHTS['RANGE']
    
    def train(self, trade_history: List[Dict], min_trades: int = 50) -> bool:
        """
        Train/retrain the model on trade history.
        
        Args:
            trade_history: List of completed trades with features and outcomes
            min_trades: Minimum trades required to train
            
        Returns:
            True if training successful
        """
        if not ML_AVAILABLE:
            logger.warning("[ML] sklearn not available - cannot train")
            return False
            
        if len(trade_history) < min_trades:
            logger.warning(f"[ML] Not enough trades ({len(trade_history)} < {min_trades})")
            return False
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for trade in trade_history:
                features = trade.get('features', {})
                if not features:
                    continue
                    
                feature_vector = [features.get(f, 0) for f in self.feature_names]
                X.append(feature_vector)
                
                # Target: 1 if profitable, 0 otherwise
                y.append(1 if trade.get('pnl_pct', 0) > 0 else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42
            )
            self.model.fit(X_scaled, y)
            
            # Save model
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            # Log performance
            accuracy = self.model.score(X_scaled, y)
            logger.info(f"[ML] Model trained. Accuracy: {accuracy:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"[ML] Training error: {e}")
            return False


# Singleton
_ml_pillar: Optional[MLPillar] = None


def get_ml_pillar() -> MLPillar:
    """Get or create the MLPillar singleton"""
    global _ml_pillar
    if _ml_pillar is None:
        _ml_pillar = MLPillar()
    return _ml_pillar
