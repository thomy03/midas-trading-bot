# Feedback Loop - Système d'Apprentissage Adaptatif

*Spécification créée le 2026-02-02*
*Auteur: Jarvis pour Thomas*

---

## Objectif

Créer un système qui **apprend des vrais résultats du marché** pour améliorer les prédictions.

## Architecture

```
FEEDBACK LOOP QUOTIDIEN

📈 ÉTAPE 1: RÉCUPÉRER RÉSULTATS RÉELS
   • Top 20 gainers du jour (+5% min)
   • Top 20 losers du jour (-5% min)
   • Via Polygon API ou yfinance

🔍 ÉTAPE 2: ANALYSE RÉTROSPECTIVE
   Pour chaque gainer:
   • Récupérer les features de J-1 (HIER)
   • Quels indicateurs étaient bullish ?
   • RSI < 30 ? MACD cross ? Volume spike ?

🧠 ÉTAPE 3: RENFORCEMENT
   • Indicateurs qui ont prédit → +poids
   • Indicateurs qui ont raté → -poids
   • Sauvegarde dans learned_weights.json

📊 ÉTAPE 4: PATTERNS DISCOVERY
   • Quels combos d'indicateurs gagnent ?
   • Quels patterns récurrents ?
   • Stockage dans patterns_db.json
```

## Fichiers à créer

| Fichier | Description |
|---------|-------------|
| `src/learning/feedback_loop.py` | Récupération et analyse des résultats |
| `src/learning/market_learner.py` | Apprentissage et ajustement des poids |
| `data/learned_weights.json` | Poids appris des indicateurs |
| `data/patterns_db.json` | Patterns découverts |

## Fichiers à modifier

| Fichier | Modification |
|---------|--------------|
| NiceGUI webapp | Ajouter bouton "Run Full Scan" |
| `src/agents/orchestrator.py` | Ordre: Tech → Fund → Sent → News → ML |
| `src/agents/pillars/ml_pillar.py` | Utiliser les poids appris |

## Acceptance Criteria

- [ ] AC1: Bouton "Run Full Scan" visible sur /control
- [ ] AC2: Scan dans l'ordre correct
- [ ] AC3: Feedback Loop récupère top gainers/losers
- [ ] AC4: Système identifie quels indicateurs ont prédit
- [ ] AC5: Poids s'ajustent automatiquement
- [ ] AC6: Logs visibles dans /activity

## Indicateurs à tracker

### Trend
- EMA cross 20/50, 50/200
- MACD histogram, signal cross
- ADX value et direction
- Supertrend signal

### Momentum
- RSI (valeur, divergence, oversold/overbought)
- Stochastic %K, %D
- Williams %R
- CCI, ROC, Momentum

### Volume
- Volume ratio vs 20-day avg
- OBV trend et divergence
- CMF, MFI
- Volume breakout

### Volatility
- ATR percent et expansion
- Bollinger Bands width et position
- Historical volatility

## Logique d'apprentissage

```python
LEARNING_RATE = 0.01

def learn_from_gainer(symbol, features_yesterday):
    """
    Pour chaque gainer, on regarde quels indicateurs
    étaient bullish hier et on renforce leur poids.
    """
    for indicator, value in features_yesterday.items():
        if was_bullish_signal(indicator, value):
            weights[indicator] += LEARNING_RATE
            
def learn_from_loser(symbol, features_yesterday):
    """
    Pour chaque loser, on regarde quels indicateurs
    étaient bullish hier (faux positifs) et on réduit leur poids.
    """
    for indicator, value in features_yesterday.items():
        if was_bullish_signal(indicator, value):
            weights[indicator] -= LEARNING_RATE * 0.5  # Pénalité plus douce
```

## Schedule

- **Quotidien (après clôture 22h Paris)** : Feedback Loop analyse la journée
- **Hebdomadaire (dimanche)** : Rapport des patterns découverts
- **Mensuel (1er du mois)** : Retrain complet du modèle ML

## Ordre du scan (corrigé)

1. **Technical** - Filtrage initial par indicateurs techniques
2. **Fundamental** - Santé financière des candidats
3. **Sentiment** - Analyse X/Twitter via Grok
4. **News** - Actualités récentes
5. **ML** - Score final avec poids appris

---

*Ce document est la spec de référence pour le système de Feedback Loop.*
