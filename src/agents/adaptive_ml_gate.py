"""
Adaptive ML Gate - S'active selon la volatilité
Créé le 2026-02-07 basé sur backtest 10/15/20 ans
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from dataclasses import dataclass
from typing import Optional, Tuple
import json
import os

@dataclass
class MLGateResult:
    original_score: float
    gated_score: float
    mode: str  # '5P_ONLY', 'ML_BOOST', 'ML_BLOCK', 'ML_NEUTRAL'
    volatility: float
    ml_confidence: float

class AdaptiveMLGate:
    """
    ML Gate qui s'active uniquement quand la volatilité est basse.
    
    - Vol > threshold: 5 Piliers seuls (pas de ML)
    - Vol <= threshold: Applique ML Gate (boost/block)
    """
    
    def __init__(self, volatility_threshold: float = 0.03, model_path: str = None):
        self.volatility_threshold = volatility_threshold
        self.model: Optional[GradientBoostingClassifier] = None
        self.X_train = []
        self.y_train = []
        self.min_samples = 100
        self.model_path = model_path or '/app/data/ml_gate_model.json'
        self.stats = {
            'total_signals': 0,
            '5p_only_count': 0,
            'ml_boost_count': 0,
            'ml_block_count': 0,
            'ml_neutral_count': 0
        }
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self):
        """Load trained model from disk if exists"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'r') as f:
                    data = json.load(f)
                    self.X_train = data.get('X_train', [])
                    self.y_train = data.get('y_train', [])
                    if len(self.X_train) >= self.min_samples:
                        self._train_model()
        except Exception as e:
            print(f'[ML Gate] Could not load model: {e}')
    
    def _save_model(self):
        """Save training data to disk"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'w') as f:
                json.dump({
                    'X_train': self.X_train[-5000:],  # Keep last 5000 samples
                    'y_train': self.y_train[-5000:]
                }, f)
        except Exception as e:
            print(f'[ML Gate] Could not save model: {e}')
    
    def _train_model(self) -> bool:
        """Train the ML model on collected data"""
        if len(self.X_train) < self.min_samples:
            return False
        
        try:
            X = np.array(self.X_train)
            y = np.array(self.y_train)
            
            n_estimators = min(100 + len(self.X_train) // 100, 200)
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=4,
                random_state=42
            )
            self.model.fit(X, y)
            print(f'[ML Gate] Model trained on {len(X)} samples with {n_estimators} trees')
            return True
        except Exception as e:
            print(f'[ML Gate] Training failed: {e}')
            return False
    
    def add_feedback(self, pillar_scores: dict, outcome: int):
        """
        Add trade outcome for learning
        
        Args:
            pillar_scores: Dict with technical, fundamental, sentiment, news, ml_regime
            outcome: 1 if profitable trade, 0 otherwise
        """
        features = [
            pillar_scores.get('technical', 50),
            pillar_scores.get('fundamental', 50),
            pillar_scores.get('sentiment', 50),
            pillar_scores.get('news', 50),
            pillar_scores.get('ml_regime', 50)
        ]
        
        self.X_train.append(features)
        self.y_train.append(outcome)
        
        # Retrain periodically
        if len(self.X_train) % 200 == 0:
            self._train_model()
            self._save_model()
    
    def get_ml_confidence(self, features: list) -> float:
        """Get ML model confidence for given features"""
        if self.model is None:
            return 0.5
        
        try:
            return self.model.predict_proba([features])[0][1]
        except:
            return 0.5
    
    def apply(self, base_score: float, pillar_scores: dict, volatility: float) -> MLGateResult:
        """
        Apply adaptive ML Gate to a signal
        
        Args:
            base_score: Original 5-pillar combined score (0-100)
            pillar_scores: Individual pillar scores dict
            volatility: Current 20-day volatility (as decimal, e.g., 0.03 = 3%)
        
        Returns:
            MLGateResult with gated score and metadata
        """
        self.stats['total_signals'] += 1
        
        features = [
            pillar_scores.get('technical', 50),
            pillar_scores.get('fundamental', 50),
            pillar_scores.get('sentiment', 50),
            pillar_scores.get('news', 50),
            pillar_scores.get('ml_regime', 50)
        ]
        
        ml_confidence = self.get_ml_confidence(features)
        
        # HIGH VOLATILITY: Skip ML Gate, use 5 Pillars only
        if volatility > self.volatility_threshold:
            self.stats['5p_only_count'] += 1
            return MLGateResult(
                original_score=base_score,
                gated_score=base_score,
                mode='5P_ONLY',
                volatility=volatility,
                ml_confidence=ml_confidence
            )
        
        # LOW VOLATILITY: Apply ML Gate
        if ml_confidence > 0.6:
            # ML is confident this is a good signal -> BOOST
            gated_score = base_score + 5
            mode = 'ML_BOOST'
            self.stats['ml_boost_count'] += 1
        elif ml_confidence < 0.4:
            # ML thinks this is a bad signal -> BLOCK
            gated_score = 0
            mode = 'ML_BLOCK'
            self.stats['ml_block_count'] += 1
        else:
            # ML is neutral -> pass through
            gated_score = base_score
            mode = 'ML_NEUTRAL'
            self.stats['ml_neutral_count'] += 1
        
        return MLGateResult(
            original_score=base_score,
            gated_score=gated_score,
            mode=mode,
            volatility=volatility,
            ml_confidence=ml_confidence
        )
    
    def get_stats(self) -> dict:
        """Get usage statistics"""
        total = self.stats['total_signals']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            '5p_only_pct': self.stats['5p_only_count'] / total * 100,
            'ml_boost_pct': self.stats['ml_boost_count'] / total * 100,
            'ml_block_pct': self.stats['ml_block_count'] / total * 100,
            'ml_neutral_pct': self.stats['ml_neutral_count'] / total * 100,
            'training_samples': len(self.X_train),
            'model_trained': self.model is not None
        }


# Global instance
_ml_gate_instance = None

def get_ml_gate() -> AdaptiveMLGate:
    """Get singleton ML Gate instance"""
    global _ml_gate_instance
    if _ml_gate_instance is None:
        _ml_gate_instance = AdaptiveMLGate(volatility_threshold=0.03)
    return _ml_gate_instance
