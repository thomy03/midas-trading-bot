"""
Multi-Pillar Feedback Loop - Apprentissage sur TOUS les piliers.

Analyse les résultats du marché et ajuste les poids de:
- Technical: EMA, RSI, MACD, Volume, ATR, Bollinger
- Fundamental: P/E, PEG, Revenue Growth, Margins, ROE
- Sentiment: Grok/Twitter scores
- News: Headline sentiment
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
MULTI_PILLAR_WEIGHTS_FILE = DATA_DIR / "multi_pillar_weights.json"
PILLAR_HISTORY_FILE = DATA_DIR / "pillar_feedback_history.json"


class MultiPillarFeedback:
    """
    Feedback Loop qui apprend de TOUS les piliers du système.
    """
    
    def __init__(self):
        self.weights = self._load_weights()
        
    def _load_weights(self) -> Dict:
        """Charge les poids multi-piliers."""
        if MULTI_PILLAR_WEIGHTS_FILE.exists():
            try:
                with open(MULTI_PILLAR_WEIGHTS_FILE) as f:
                    return json.load(f)
            except:
                pass
        return self._init_default_weights()
    
    def _init_default_weights(self) -> Dict:
        """Initialise les poids par défaut pour tous les piliers."""
        return {
            # PILIER TECHNIQUE
            "technical": {
                "pillar_weight": 0.25,
                "indicators": {
                    # Trend
                    "ema_cross_20_50": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "ema_cross_50_200": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "price_above_sma200": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "adx_strong": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    # Momentum
                    "rsi_oversold": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "rsi_overbought": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "macd_bullish": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "stoch_oversold": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    # Volume
                    "volume_spike": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "obv_rising": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    # Volatility
                    "bollinger_squeeze": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "atr_expansion": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                }
            },
            # PILIER FONDAMENTAL
            "fundamental": {
                "pillar_weight": 0.25,
                "indicators": {
                    # Valuation
                    "pe_undervalued": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "peg_attractive": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "pb_undervalued": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    # Growth
                    "revenue_growth_strong": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "earnings_growth_strong": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "eps_growth_positive": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    # Profitability
                    "high_gross_margin": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "high_roe": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "high_roa": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    # Health
                    "low_debt_ratio": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "positive_free_cash_flow": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                }
            },
            # PILIER SENTIMENT (Grok/Twitter)
            "sentiment": {
                "pillar_weight": 0.25,
                "indicators": {
                    "twitter_bullish": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "twitter_volume_spike": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "social_momentum_up": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "influencer_mention": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                }
            },
            # PILIER NEWS (Gemini)
            "news": {
                "pillar_weight": 0.25,
                "indicators": {
                    "news_sentiment_positive": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "earnings_beat": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "analyst_upgrade": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "product_launch": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                    "partnership_announced": {"weight": 1.0, "accuracy": 0.5, "signals": 0, "correct": 0},
                }
            },
            "last_updated": "",
            "total_feedback_cycles": 0
        }
    
    def _save_weights(self):
        """Sauvegarde les poids."""
        self.weights["last_updated"] = datetime.now().isoformat()
        with open(MULTI_PILLAR_WEIGHTS_FILE, "w") as f:
            json.dump(self.weights, f, indent=2)
    
    def reinforce_indicator(self, pillar: str, indicator: str, correct: bool):
        """
        Met à jour le poids d'un indicateur.
        
        Args:
            pillar: 'technical', 'fundamental', 'sentiment', 'news'
            indicator: Nom de l'indicateur
            correct: True si le signal était correct
        """
        if pillar not in self.weights or "indicators" not in self.weights[pillar]:
            return
        
        indicators = self.weights[pillar]["indicators"]
        if indicator not in indicators:
            return
        
        ind = indicators[indicator]
        ind["signals"] += 1
        
        if correct:
            ind["correct"] += 1
            ind["weight"] = min(2.0, ind["weight"] * 1.05)  # +5%
        else:
            ind["weight"] = max(0.1, ind["weight"] * 0.95)  # -5%
        
        # Recalculer accuracy
        if ind["signals"] > 0:
            ind["accuracy"] = ind["correct"] / ind["signals"]
    
    def adjust_pillar_weights(self):
        """
        Ajuste les poids des piliers en fonction de leur performance globale.
        """
        pillar_accuracies = {}
        total_accuracy = 0
        
        for pillar in ["technical", "fundamental", "sentiment", "news"]:
            if pillar in self.weights and "indicators" in self.weights[pillar]:
                indicators = self.weights[pillar]["indicators"]
                if indicators:
                    accuracies = [ind["accuracy"] for ind in indicators.values() if ind["signals"] > 0]
                    if accuracies:
                        pillar_accuracies[pillar] = sum(accuracies) / len(accuracies)
                        total_accuracy += pillar_accuracies[pillar]
        
        # Redistribuer les poids proportionnellement à l'accuracy
        if total_accuracy > 0 and pillar_accuracies:
            for pillar, accuracy in pillar_accuracies.items():
                # Poids proportionnel mais borné entre 0.1 et 0.4
                new_weight = max(0.1, min(0.4, accuracy / total_accuracy))
                self.weights[pillar]["pillar_weight"] = new_weight
    
    def get_indicator_weight(self, pillar: str, indicator: str) -> float:
        """Retourne le poids d'un indicateur."""
        try:
            return self.weights[pillar]["indicators"][indicator]["weight"]
        except:
            return 1.0
    
    def get_pillar_weight(self, pillar: str) -> float:
        """Retourne le poids d'un pilier."""
        try:
            return self.weights[pillar]["pillar_weight"]
        except:
            return 0.25
    
    def get_summary(self) -> Dict:
        """Retourne un résumé de l'apprentissage multi-piliers."""
        summary = {
            "total_cycles": self.weights.get("total_feedback_cycles", 0),
            "last_updated": self.weights.get("last_updated", ""),
            "pillars": {}
        }
        
        for pillar in ["technical", "fundamental", "sentiment", "news"]:
            if pillar in self.weights:
                p = self.weights[pillar]
                indicators = p.get("indicators", {})
                total_signals = sum(ind["signals"] for ind in indicators.values())
                avg_accuracy = sum(ind["accuracy"] for ind in indicators.values()) / len(indicators) if indicators else 0.5
                
                # Top 3 indicators
                sorted_inds = sorted(indicators.items(), key=lambda x: x[1]["accuracy"] * x[1]["signals"], reverse=True)
                top_3 = [{"name": k, "accuracy": v["accuracy"], "weight": v["weight"]} for k, v in sorted_inds[:3]]
                
                summary["pillars"][pillar] = {
                    "weight": p.get("pillar_weight", 0.25),
                    "total_signals": total_signals,
                    "avg_accuracy": round(avg_accuracy, 3),
                    "top_indicators": top_3
                }
        
        return summary


# Singleton
_multi_pillar_feedback: Optional[MultiPillarFeedback] = None

def get_multi_pillar_feedback() -> MultiPillarFeedback:
    global _multi_pillar_feedback
    if _multi_pillar_feedback is None:
        _multi_pillar_feedback = MultiPillarFeedback()
    return _multi_pillar_feedback


async def run_multi_pillar_feedback():
    """
    Exécute le feedback loop complet sur tous les piliers.
    Utilise les scores des piliers du ReasoningEngine.
    """
    from src.learning.feedback_loop import get_feedback_loop
    
    logger.info("Starting Multi-Pillar Feedback Loop...")
    
    # 1. Run the technical feedback (existing)
    feedback = get_feedback_loop()
    tech_result = await feedback.run_feedback_cycle()
    
    # 2. Update multi-pillar weights based on results
    mpf = get_multi_pillar_feedback()
    
    # Reinforce technical indicators
    for ind in tech_result.get("indicators_reinforced", []):
        mpf.reinforce_indicator("technical", ind, True)
    
    for ind in tech_result.get("indicators_penalized", []):
        mpf.reinforce_indicator("technical", ind, False)
    
    # 3. Adjust pillar weights based on performance
    mpf.adjust_pillar_weights()
    
    # 4. Save
    mpf.weights["total_feedback_cycles"] += 1
    mpf._save_weights()
    
    # 5. Get summary
    summary = mpf.get_summary()
    
    logger.info(f"Multi-Pillar Feedback complete: {summary['total_cycles']} cycles")
    
    return {
        "technical_result": tech_result,
        "multi_pillar_summary": summary
    }
