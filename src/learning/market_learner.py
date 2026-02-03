"""
Market Learner - Intègre les poids appris dans le scoring.

Fournit une interface pour ajuster les scores en fonction de l'apprentissage.
"""

import logging
from typing import Dict, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
LEARNED_WEIGHTS_FILE = DATA_DIR / "learned_weights.json"


class MarketLearner:
    """
    Utilise les poids appris pour ajuster les scores des indicateurs.
    """
    
    def __init__(self):
        self.weights = self._load_weights()
        self.learning_rate = 0.1  # Taux d'apprentissage pour les ajustements
        
    def _load_weights(self) -> Dict:
        """Charge les poids appris."""
        if LEARNED_WEIGHTS_FILE.exists():
            try:
                with open(LEARNED_WEIGHTS_FILE) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading weights: {e}")
        return {}
    
    def reload_weights(self):
        """Recharge les poids depuis le fichier."""
        self.weights = self._load_weights()
        logger.info(f"Reloaded {len(self.weights)} indicator weights")
    
    def get_weight(self, indicator: str) -> float:
        """
        Retourne le poids appris pour un indicateur.
        
        Returns:
            Poids entre 0.1 et 2.0 (1.0 par défaut)
        """
        if indicator in self.weights:
            return self.weights[indicator].get("weight", 1.0)
        return 1.0
    
    def get_accuracy(self, indicator: str) -> float:
        """
        Retourne l'accuracy historique d'un indicateur.
        
        Returns:
            Accuracy entre 0 et 1 (0.5 par défaut)
        """
        if indicator in self.weights:
            return self.weights[indicator].get("accuracy", 0.5)
        return 0.5
    
    def adjust_score(self, base_score: float, indicator: str) -> float:
        """
        Ajuste un score en fonction du poids appris.
        
        Args:
            base_score: Score original (0-100)
            indicator: Nom de l'indicateur
            
        Returns:
            Score ajusté
        """
        weight = self.get_weight(indicator)
        
        # Le poids multiplie l'écart par rapport à 50 (neutre)
        deviation = base_score - 50
        adjusted_deviation = deviation * weight
        
        return max(0, min(100, 50 + adjusted_deviation))
    
    def get_weighted_technical_score(self, technical_factors: Dict[str, float]) -> float:
        """
        Calcule un score technique pondéré par les poids appris.
        
        Args:
            technical_factors: Dict {indicator_name: score}
            
        Returns:
            Score pondéré
        """
        if not technical_factors:
            return 50.0
        
        weighted_sum = 0
        total_weight = 0
        
        for indicator, score in technical_factors.items():
            weight = self.get_weight(indicator)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 50.0
        
        return weighted_sum / total_weight
    
    def get_confidence_multiplier(self, indicators: list) -> float:
        """
        Retourne un multiplicateur de confiance basé sur la fiabilité
        historique des indicateurs actifs.
        
        Args:
            indicators: Liste des indicateurs actifs
            
        Returns:
            Multiplicateur entre 0.5 et 1.5
        """
        if not indicators:
            return 1.0
        
        accuracies = [self.get_accuracy(ind) for ind in indicators]
        avg_accuracy = sum(accuracies) / len(accuracies)
        
        # Convertir accuracy (0.5-1.0) en multiplicateur (0.5-1.5)
        # accuracy 0.5 -> multiplier 0.5
        # accuracy 0.75 -> multiplier 1.0
        # accuracy 1.0 -> multiplier 1.5
        return 0.5 + (avg_accuracy * 2 - 0.5)
    
    def get_learning_summary(self) -> Dict:
        """Retourne un résumé de l'apprentissage."""
        if not self.weights:
            return {"status": "no_data", "indicators": 0}
        
        total_signals = sum(w.get("total_signals", 0) for w in self.weights.values())
        avg_accuracy = sum(w.get("accuracy", 0.5) for w in self.weights.values()) / len(self.weights)
        
        # Top 5 indicators
        sorted_weights = sorted(
            self.weights.items(),
            key=lambda x: x[1].get("accuracy", 0.5) * x[1].get("total_signals", 0),
            reverse=True
        )
        top_indicators = [
            {"name": k, "accuracy": v.get("accuracy", 0.5), "weight": v.get("weight", 1.0)}
            for k, v in sorted_weights[:5]
        ]
        
        return {
            "status": "active",
            "total_indicators": len(self.weights),
            "total_signals_analyzed": total_signals,
            "average_accuracy": round(avg_accuracy, 3),
            "top_indicators": top_indicators
        }


# Singleton
_market_learner: Optional[MarketLearner] = None

def get_market_learner() -> MarketLearner:
    global _market_learner
    if _market_learner is None:
        _market_learner = MarketLearner()
    return _market_learner
