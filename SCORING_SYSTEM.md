# Système de Scoring TradingBot - 5 Piliers Adaptatifs

*Version: 2.0 - Février 2026*
*Mis à jour par: Jarvis*

---

## Architecture Globale

Le système utilise **5 piliers** pour évaluer chaque opportunité de trading.
Le pilier ML ajuste dynamiquement les poids selon le régime de marché.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SCORING ENGINE (100 pts)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐│
│  │TECHNICAL │ │FUNDAMENT │ │SENTIMENT │ │  NEWS    │ │   ML   ││
│  │  20-25%  │ │  20-25%  │ │  15-20%  │ │  10-15%  │ │ 20-25% ││
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘│
│       │            │            │            │           │      │
│       └────────────┴────────────┴────────────┴───────────┘      │
│                              │                                  │
│                    ┌─────────▼─────────┐                       │
│                    │  REGIME DETECTOR  │                       │
│                    │ (Bull/Bear/Range) │                       │
│                    └─────────┬─────────┘                       │
│                              │                                  │
│                    ┌─────────▼─────────┐                       │
│                    │   FINAL SCORE     │                       │
│                    │    (0-100)        │                       │
│                    └───────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pilier 1: TECHNICAL (20-25%)

Analyse technique basée sur les indicateurs de prix et volume.

### Indicateurs utilisés

| Catégorie | Indicateurs | Poids interne |
|-----------|-------------|---------------|
| **Trend** | EMA (20/50/200), MACD, ADX, Supertrend | 30% |
| **Momentum** | RSI, Stochastic, Williams %R, CCI | 25% |
| **Volume** | OBV, Volume Ratio, CMF, MFI | 25% |
| **Volatility** | ATR, Bollinger Bands, Keltner | 20% |

### Scoring
- Score positif (+) = Signal bullish
- Score négatif (-) = Signal bearish
- Normalisé sur 0-100

---

## Pilier 2: FUNDAMENTAL (20-25%)

Analyse fondamentale de la santé financière.

### Métriques

| Métrique | Description | Poids |
|----------|-------------|-------|
| **P/E Ratio** | Valorisation vs earnings | 20% |
| **Revenue Growth** | Croissance CA YoY | 25% |
| **Profit Margins** | Marges bénéficiaires | 20% |
| **ROE** | Return on Equity | 15% |
| **Debt/Equity** | Niveau d'endettement | 10% |
| **Earnings Surprise** | Surprise vs consensus | 10% |

### Kill Switches
- Earnings dans < 5 jours → EXCLUDE
- Volume moyen < 200k → EXCLUDE

---

## Pilier 3: SENTIMENT (15-20%)

Sentiment des réseaux sociaux via Grok API (X/Twitter).

### Sources
- X/Twitter (via Grok API)
- StockTwits
- Reddit (WSB, stocks, investing)

### Métriques
- Sentiment score (-1 à +1)
- Mention trend (rising/stable/falling)
- Catalysts identifiés

---

## Pilier 4: NEWS (10-15%)

Analyse des actualités financières via OpenRouter/Gemini.

### Sources
- NewsAPI
- Financial news feeds
- Press releases

### Scoring
- Headlines sentiment
- Recency (news récentes > anciennes)
- Relevance au symbole

---

## Pilier 5: ML ADAPTATIF (20-25%) 🆕

**Machine Learning qui apprend quels indicateurs performent le mieux.**

### Architecture

```python
class MLPillar:
    """
    Pilier ML adaptatif qui:
    1. Extrait 40+ features techniques
    2. Détecte le régime de marché
    3. Prédit la probabilité de succès
    4. Ajuste les poids des autres piliers
    """
```

### Features (40+ indicateurs)

```python
FEATURES = {
    # Trend (10 features)
    'trend': [
        'ema_cross_20_50',      # Croisement EMA 20/50
        'ema_cross_50_200',     # Croisement EMA 50/200 (Golden/Death cross)
        'macd_histogram',       # MACD histogram value
        'macd_signal_cross',    # MACD vs Signal line
        'adx_value',            # ADX trend strength
        'adx_direction',        # +DI vs -DI
        'supertrend_signal',    # Supertrend direction
        'aroon_oscillator',     # Aroon Up - Aroon Down
        'ichimoku_cloud',       # Price vs Cloud position
        'parabolic_sar',        # SAR position vs price
    ],
    
    # Momentum (10 features)
    'momentum': [
        'rsi_14',               # RSI value
        'rsi_divergence',       # RSI vs Price divergence
        'stoch_k',              # Stochastic %K
        'stoch_crossover',      # %K vs %D crossover
        'williams_r',           # Williams %R
        'cci_20',               # Commodity Channel Index
        'roc_10',               # Rate of Change
        'momentum_10',          # Momentum indicator
        'ultimate_oscillator',  # Ultimate Oscillator
        'tsi',                  # True Strength Index
    ],
    
    # Volume (8 features)
    'volume': [
        'volume_ratio_20',      # Volume vs 20-day avg
        'obv_trend',            # OBV direction
        'obv_divergence',       # OBV vs Price divergence
        'cmf_20',               # Chaikin Money Flow
        'mfi_14',               # Money Flow Index
        'ad_line_trend',        # A/D Line direction
        'vwap_distance',        # Price vs VWAP
        'volume_price_trend',   # VPT indicator
    ],
    
    # Volatility (6 features)
    'volatility': [
        'atr_percent',          # ATR as % of price
        'atr_expansion',        # ATR vs 20-day avg ATR
        'bb_width',             # Bollinger Band width
        'bb_percent',           # Position in BB (0-1)
        'keltner_position',     # Position in Keltner Channel
        'historical_vol_20',    # 20-day historical volatility
    ],
    
    # Market Regime (6 features)
    'regime': [
        'spy_trend',            # SPY above/below EMA50
        'vix_level',            # VIX absolute level
        'vix_trend',            # VIX direction (rising/falling)
        'sector_relative',      # Stock vs Sector ETF
        'market_breadth',       # Advance/Decline ratio
        'correlation_spy',      # 20-day correlation with SPY
    ],
}
```

### Modèle ML

```python
# Modèle: Random Forest ou XGBoost (léger et interprétable)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)

# Target: Trade réussi (profit > 0) dans les 5-10 jours
# Features: 40 indicateurs normalisés
# Training: Sur historique des trades (winners vs losers)
```

### Régimes de Marché

| Régime | Détection | Poids ajustés |
|--------|-----------|---------------|
| **BULL** | SPY > EMA50, VIX < 20 | Tech↓ Fund↑ Sent↓ ML↑ |
| **BEAR** | SPY < EMA50, VIX > 25 | Tech↑ Fund↓ Sent↑ ML↑ |
| **RANGE** | SPY ≈ EMA50, VIX 15-25 | Équilibré |
| **VOLATILE** | VIX > 30 | Tech↑↑ ML↓ (prudence) |

### Poids Adaptatifs par Régime

```python
REGIME_WEIGHTS = {
    'BULL': {
        'technical': 0.20,
        'fundamental': 0.25,
        'sentiment': 0.15,
        'news': 0.15,
        'ml': 0.25
    },
    'BEAR': {
        'technical': 0.30,
        'fundamental': 0.15,
        'sentiment': 0.20,
        'news': 0.10,
        'ml': 0.25
    },
    'RANGE': {
        'technical': 0.25,
        'fundamental': 0.20,
        'sentiment': 0.15,
        'news': 0.15,
        'ml': 0.25
    },
    'VOLATILE': {
        'technical': 0.35,
        'fundamental': 0.15,
        'sentiment': 0.15,
        'news': 0.10,
        'ml': 0.25
    }
}
```

### Retrain Automatique

- **Fréquence**: Mensuel (1er du mois)
- **Data**: Derniers 6 mois de trades
- **Validation**: 70/30 train/test split
- **Métriques**: Accuracy, Precision, F1-score

---

## Flow Complet

```
1. SCREENING (FMP + Social)
   → 3000+ titres → filtre → ~100 candidats

2. SCORING (5 Piliers)
   → Technical: analyse indicateurs
   → Fundamental: santé financière
   → Sentiment: réseaux sociaux
   → News: actualités
   → ML: prédiction adaptative

3. RÉGIME DETECTION
   → Analyse SPY, VIX, breadth
   → Détermine weights par pilier

4. SCORE FINAL
   → Weighted sum des 5 piliers
   → Classement: ELITE (>75) / BUY (>55) / WATCH (>35)

5. PAPER TRADING
   → Position sizing automatique
   → Passage d'ordres fictifs
   → Suivi P&L temps réel
```

---

## Signals de Trading

| Signal | Score | Action |
|--------|-------|--------|
| **ELITE** | > 75 | BUY prioritaire, position normale |
| **STRONG_BUY** | 65-75 | BUY, position réduite |
| **BUY** | 55-65 | BUY si capital dispo |
| **WATCH** | 35-55 | Surveiller, pas d'action |
| **AVOID** | < 35 | Ne pas trader |

---

## Fichiers Clés

| Fichier | Rôle |
|---------|------|
| `src/agents/pillars/technical_pillar.py` | Pilier technique |
| `src/agents/pillars/fundamental_pillar.py` | Pilier fondamental |
| `src/agents/pillars/sentiment_pillar.py` | Pilier sentiment |
| `src/agents/pillars/news_pillar.py` | Pilier news |
| `src/agents/pillars/ml_pillar.py` | **Pilier ML adaptatif** 🆕 |
| `src/agents/reasoning_engine.py` | Orchestration des piliers |
| `src/agents/orchestrator.py` | Cerveau principal |

---

## Webapp

**URL**: https://tradingbot.46-225-58-233.sslip.io

| Page | Description |
|------|-------------|
| /portfolio | État du portefeuille fictif |
| /activity | Historique des trades |
| /analysis | Analyse des symboles |
| /control | Start/Stop du bot |

---

*Ce document est la référence pour le système de scoring. Toute modification doit être documentée ici.*
