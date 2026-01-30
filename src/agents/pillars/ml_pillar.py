"""
ML Pillar - Machine Learning scoring pillar using Gradient Boosting

Uses historical trade data to predict probability of trade success.
Features: technical indicators, fundamental metrics, sentiment scores, market regime.

Lightweight implementation optimized for VPS (< 100MB RAM).
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

logger = logging.getLogger(__name__)


@dataclass
class MLPrediction:
    """ML model prediction result"""
    probability: float  # 0.0 to 1.0 probability of winning trade
    confidence: str     # 'high', 'medium', 'low'
    score: float        # 0-100 score for pillar
    features_used: int
    model_version: str
    veto: bool          # True if ML recommends NOT trading


class MLPillar:
    """
    Machine Learning pillar for trade scoring.
    
    Uses Gradient Boosting to predict trade success probability
    based on technical, fundamental, sentiment, and market features.
    """
    
    MODEL_PATH = Path("/app/data/models/ml_pillar_model.joblib")
    SCALER_PATH = Path("/app/data/models/ml_pillar_scaler.joblib")
    HISTORY_PATH = Path("/app/data/ml_training_history.csv")
    
    # Feature columns expected by the model
    FEATURE_COLUMNS = [
        # Technical features
        'technical_score',
        'rsi',
        'rsi_oversold',      # 1 if RSI < 30
        'rsi_overbought',    # 1 if RSI > 70
        'macd_signal',       # 1 bullish, -1 bearish, 0 neutral
        'ema_trend',         # 1 above EMAs, -1 below
        'atr_percent',       # ATR as % of price
        'volume_ratio',      # Current vol / avg vol
        'price_vs_52w_high', # Price / 52w high
        'price_vs_52w_low',  # Price / 52w low
        
        # Fundamental features
        'fundamental_score',
        'pe_ratio_norm',     # Normalized P/E (0-1)
        'revenue_growth',    # YoY revenue growth
        'profit_margin',     # Net margin
        'debt_to_equity',    # Debt/Equity ratio
        
        # Sentiment features
        'sentiment_score',
        'twitter_sentiment', # -1 to 1
        'news_sentiment',    # -1 to 1
        
        # Market regime features
        'market_regime',     # 1 bull, 0 neutral, -1 bear
        'vix_level',         # VIX normalized
        'sector_momentum',   # Sector relative strength
        
        # Combined score from other pillars
        'combined_score',    # Weighted average of other pillars
    ]
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize ML Pillar"""
        self.model_path = model_path or self.MODEL_PATH
        self.scaler_path = self.SCALER_PATH
        self.model = None
        self.scaler = None
        self.model_version = "1.0.0"
        self.is_trained = False
        
        # Ensure directories exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            if self.model_path.exists() and self.scaler_path.exists():
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.is_trained = True
                logger.info(f"ML Pillar model loaded from {self.model_path}")
                return True
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}")
        return False
    
    def _save_model(self):
        """Save trained model to disk"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"ML Pillar model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Could not save ML model: {e}")
    
    def extract_features(
        self,
        symbol: str,
        technical_data: Dict,
        fundamental_data: Dict,
        sentiment_data: Dict,
        market_data: Dict,
        combined_score: float
    ) -> np.ndarray:
        """
        Extract feature vector from raw data.
        
        Returns numpy array of features in correct order.
        """
        features = {}
        
        # Technical features
        features['technical_score'] = technical_data.get('score', 50) / 100
        features['rsi'] = technical_data.get('rsi', 50) / 100
        features['rsi_oversold'] = 1 if technical_data.get('rsi', 50) < 30 else 0
        features['rsi_overbought'] = 1 if technical_data.get('rsi', 50) > 70 else 0
        
        macd = technical_data.get('macd_signal', 'neutral')
        features['macd_signal'] = 1 if macd == 'bullish' else (-1 if macd == 'bearish' else 0)
        
        features['ema_trend'] = 1 if technical_data.get('above_ema', False) else -1
        features['atr_percent'] = min(technical_data.get('atr_percent', 2) / 10, 1)  # Cap at 10%
        features['volume_ratio'] = min(technical_data.get('volume_ratio', 1) / 3, 1)  # Cap at 3x
        features['price_vs_52w_high'] = technical_data.get('price_vs_52w_high', 0.8)
        features['price_vs_52w_low'] = technical_data.get('price_vs_52w_low', 1.2)
        
        # Fundamental features
        features['fundamental_score'] = fundamental_data.get('score', 50) / 100
        features['pe_ratio_norm'] = min(fundamental_data.get('pe_ratio', 20) / 50, 1)  # Cap at 50
        features['revenue_growth'] = np.tanh(fundamental_data.get('revenue_growth', 0) / 50)  # Normalize
        features['profit_margin'] = np.tanh(fundamental_data.get('profit_margin', 0) / 30)
        features['debt_to_equity'] = min(fundamental_data.get('debt_to_equity', 0.5) / 2, 1)
        
        # Sentiment features
        features['sentiment_score'] = sentiment_data.get('score', 50) / 100
        features['twitter_sentiment'] = sentiment_data.get('twitter', 0)  # -1 to 1
        features['news_sentiment'] = sentiment_data.get('news', 0)  # -1 to 1
        
        # Market regime
        regime = market_data.get('regime', 'neutral')
        features['market_regime'] = 1 if regime == 'bull' else (-1 if regime == 'bear' else 0)
        features['vix_level'] = min(market_data.get('vix', 20) / 40, 1)  # Cap at 40
        features['sector_momentum'] = np.tanh(market_data.get('sector_momentum', 0) / 10)
        
        # Combined score
        features['combined_score'] = combined_score / 100
        
        # Build feature vector in correct order
        feature_vector = [features.get(col, 0) for col in self.FEATURE_COLUMNS]
        
        return np.array(feature_vector).reshape(1, -1)
    
    def predict(
        self,
        symbol: str,
        technical_data: Dict,
        fundamental_data: Dict,
        sentiment_data: Dict,
        market_data: Dict,
        combined_score: float
    ) -> MLPrediction:
        """
        Predict probability of successful trade.
        
        Returns MLPrediction with probability, score, and veto recommendation.
        """
        # Extract features
        features = self.extract_features(
            symbol, technical_data, fundamental_data,
            sentiment_data, market_data, combined_score
        )
        
        # If no trained model, return neutral prediction
        if not self.is_trained or self.model is None:
            return MLPrediction(
                probability=0.5,
                confidence='low',
                score=50,
                features_used=len(self.FEATURE_COLUMNS),
                model_version=self.model_version,
                veto=False
            )
        
        try:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get probability prediction
            proba = self.model.predict_proba(features_scaled)[0]
            win_probability = proba[1] if len(proba) > 1 else proba[0]
            
            # Determine confidence based on probability distance from 0.5
            distance = abs(win_probability - 0.5)
            if distance > 0.3:
                confidence = 'high'
            elif distance > 0.15:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            # Convert to 0-100 score
            score = win_probability * 100
            
            # Veto if probability < 30%
            veto = win_probability < 0.30
            
            return MLPrediction(
                probability=float(win_probability),
                confidence=confidence,
                score=float(score),
                features_used=len(self.FEATURE_COLUMNS),
                model_version=self.model_version,
                veto=veto
            )
            
        except Exception as e:
            logger.error(f"ML prediction error for {symbol}: {e}")
            return MLPrediction(
                probability=0.5,
                confidence='low',
                score=50,
                features_used=0,
                model_version=self.model_version,
                veto=False
            )
    
    def train(
        self,
        training_data: pd.DataFrame,
        target_column: str = 'is_winner',
        test_size: float = 0.2
    ) -> Dict:
        """
        Train the ML model on historical trade data.
        
        Args:
            training_data: DataFrame with features and target
            target_column: Name of target column (1=winning trade, 0=losing)
            test_size: Fraction of data for testing
            
        Returns:
            Dict with training metrics
        """
        logger.info(f"Training ML Pillar on {len(training_data)} samples...")
        
        # Prepare features and target
        feature_cols = [c for c in self.FEATURE_COLUMNS if c in training_data.columns]
        X = training_data[feature_cols].fillna(0)
        y = training_data[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Gradient Boosting model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': len(feature_cols),
            'feature_importance': dict(zip(feature_cols, self.model.feature_importances_))
        }
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        
        self.is_trained = True
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Save model
        self._save_model()
        
        logger.info(f"ML Pillar trained - Accuracy: {metrics['accuracy']:.2%}, F1: {metrics['f1']:.2%}")
        
        return metrics
    
    def train_from_backtest(self, backtest_results: List[Dict]) -> Dict:
        """
        Train model from backtest results.
        
        Args:
            backtest_results: List of trade dicts with features and outcome
            
        Returns:
            Training metrics
        """
        if len(backtest_results) < 50:
            logger.warning(f"Not enough data to train: {len(backtest_results)} samples (need 50+)")
            return {'error': 'insufficient_data', 'samples': len(backtest_results)}
        
        # Convert to DataFrame
        df = pd.DataFrame(backtest_results)
        
        # Ensure target column exists
        if 'is_winner' not in df.columns and 'profit_pct' in df.columns:
            df['is_winner'] = (df['profit_pct'] > 0).astype(int)
        
        return self.train(df)
    
    def retrain_weekly(self, new_trades: List[Dict]) -> Optional[Dict]:
        """
        Incrementally retrain with new trade data (called weekly).
        
        Appends new trades to history and retrains if enough data.
        """
        if not new_trades:
            return None
        
        # Load existing history
        if self.HISTORY_PATH.exists():
            history = pd.read_csv(self.HISTORY_PATH)
        else:
            history = pd.DataFrame()
        
        # Append new trades
        new_df = pd.DataFrame(new_trades)
        history = pd.concat([history, new_df], ignore_index=True)
        
        # Keep last 12 months of data
        if 'date' in history.columns:
            history['date'] = pd.to_datetime(history['date'])
            cutoff = datetime.now() - timedelta(days=365)
            history = history[history['date'] > cutoff]
        
        # Save updated history
        history.to_csv(self.HISTORY_PATH, index=False)
        
        # Retrain if enough data
        if len(history) >= 100:
            return self.train(history)
        
        return {'status': 'accumulated', 'samples': len(history)}


# Singleton instance
_ml_pillar: Optional[MLPillar] = None


def get_ml_pillar() -> MLPillar:
    """Get or create ML Pillar singleton"""
    global _ml_pillar
    if _ml_pillar is None:
        _ml_pillar = MLPillar()
    return _ml_pillar
