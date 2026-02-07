# TradingBot V3 - Documentation Technique et Métier

**Version:** 3.0
**Date:** 27 Décembre 2024
**Auteur:** Système automatisé
**Objectif:** Audit par ingénieur financier

---

## Table des Matières

1. [Vue d'Ensemble](#1-vue-densemble)
2. [Architecture Technique](#2-architecture-technique)
3. [Stratégie de Trading](#3-stratégie-de-trading)
4. [Analyse RSI Trendline Breakout](#4-analyse-rsi-trendline-breakout)
5. [Analyse EMA et Supports](#5-analyse-ema-et-supports)
6. [Système de Scoring](#6-système-de-scoring)
7. [Position Sizing et Gestion du Risque](#7-position-sizing-et-gestion-du-risque)
8. [Module de Backtesting](#8-module-de-backtesting)
9. [Trend Discovery (IA)](#9-trend-discovery-ia)
10. [Sources de Données](#10-sources-de-données)
11. [Base de Données](#11-base-de-données)
12. [Configuration et Paramètres](#12-configuration-et-paramètres)
13. [Limites et Risques](#13-limites-et-risques)

---

## 1. Vue d'Ensemble

### 1.1 Objectif du Système

TradingBot V3 est un système de **screening d'actions** conçu pour identifier des opportunités d'achat basées sur une combinaison de:
- Analyse technique via EMAs (Exponential Moving Averages)
- Détection de cassures d'obliques RSI (Relative Strength Index)
- Confirmation par le volume
- Analyse de tendances de marché via IA

### 1.2 Philosophie de Trading

Le système repose sur l'hypothèse que:
1. Les **supports EMA** constituent des zones de rebond potentiel
2. Une **cassure d'oblique RSI descendante** signale un retournement de momentum
3. La **confirmation par le volume** valide la conviction du mouvement
4. L'**alignement des EMAs** confirme la tendance sous-jacente

### 1.3 Marchés Couverts

| Marché | Symboles | Source |
|--------|----------|--------|
| NASDAQ | ~100 valeurs | yfinance |
| S&P 500 | ~500 valeurs | yfinance |
| CAC 40 | 40 valeurs | yfinance |
| DAX | 40 valeurs | yfinance |
| Europe | 50 valeurs pan-européennes | yfinance |
| Crypto | 20 cryptomonnaies | yfinance |

---

## 2. Architecture Technique

### 2.1 Structure des Répertoires

```
TradingBot_V3/
├── config/
│   └── settings.py              # Configuration globale
├── src/
│   ├── data/
│   │   └── market_data.py       # Fetching données yfinance
│   ├── database/
│   │   └── db_manager.py        # Gestion SQLite
│   ├── indicators/
│   │   ├── ema_analyzer.py      # Calcul EMAs et crossovers
│   │   └── adaptive_exit.py     # Sorties adaptatives
│   ├── screening/
│   │   └── screener.py          # Logique de screening principale
│   ├── backtesting/
│   │   ├── backtester.py        # Moteur de backtest
│   │   └── metrics.py           # Calcul métriques performance
│   ├── intelligence/
│   │   └── trend_discovery.py   # Découverte tendances IA
│   └── utils/
│       ├── confidence_scorer.py # Scoring de confiance
│       ├── position_sizing.py   # Calcul taille positions
│       ├── interactive_chart.py # Graphiques Plotly
│       └── ...
├── trendline_analysis/
│   ├── core/
│   │   ├── trendline_detector.py         # Détection obliques RSI
│   │   ├── breakout_analyzer.py          # Analyse breakouts
│   │   ├── rsi_breakout_analyzer.py      # Analyseur simplifié
│   │   └── enhanced_rsi_breakout_analyzer.py # Analyseur avancé
│   └── config/
│       └── settings.py                   # Paramètres trendlines
├── dashboard.py                 # Interface Streamlit (legacy)
├── dashboard_nicegui.py         # Interface NiceGUI (moderne)
└── data/
    ├── screener.db              # Base SQLite
    ├── portfolio.json           # Positions ouvertes
    ├── tickers/                 # Listes de symboles
    └── trends/                  # Rapports trend discovery
```

### 2.2 Stack Technologique

| Composant | Technologie | Version |
|-----------|-------------|---------|
| Langage | Python | 3.10+ |
| Interface | NiceGUI | 1.4+ |
| Graphiques | Plotly | 5.x |
| Données marché | yfinance | 0.2+ |
| Base de données | SQLite | 3.x |
| Calcul scientifique | NumPy, Pandas, SciPy | - |
| IA/LLM | OpenRouter API | - |

### 2.3 Flux de Données

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  yfinance   │────▶│ market_data  │────▶│  ema_analyzer   │
│  (OHLCV)    │     │  .py         │     │  (calcul EMAs)  │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                    ┌──────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                      screener.py                             │
│  ┌───────────────┐  ┌────────────────┐  ┌────────────────┐  │
│  │ RSI Breakout  │  │ Support Zones  │  │ Confidence     │  │
│  │ Analyzer      │  │ Detection      │  │ Scorer         │  │
│  └───────────────┘  └────────────────┘  └────────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  Position Sizer  │
                    │  (risk mgmt)     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  SQLite DB       │
                    │  (alertes)       │
                    └──────────────────┘
```

---

## 3. Stratégie de Trading

### 3.1 Conditions d'Entrée (LONG uniquement)

Un signal d'achat est généré lorsque **TOUTES** les conditions suivantes sont réunies:

| # | Condition | Description | Paramètre |
|---|-----------|-------------|-----------|
| 1 | RSI Breakout | Le RSI casse une oblique descendante | `BREAKOUT_THRESHOLD = 0.0` |
| 2 | EMAs Croissantes | Au moins une EMA courte > EMA longue | EMA24 > EMA38 ou EMA38 > EMA62 |
| 3 | Prix proche support | Prix < 8% d'un niveau EMA historique | `SUPPORT_ZONE_THRESHOLD = 0.08` |
| 4 | Breakout récent | Âge du breakout ≤ N périodes | `MAX_BREAKOUT_AGE = 3` |

### 3.2 Classification des Signaux

| Signal | Conditions |
|--------|------------|
| **STRONG_BUY** | Breakout STRONG + âge ≤ 3 + score confiance > 70 |
| **BUY** | Breakout MODERATE/STRONG + âge ≤ 5 |
| **WATCH** | Oblique présente, pas encore de breakout |
| **NO_SIGNAL** | Aucune condition remplie |

### 3.3 Conditions de Sortie

Le backtester implémente plusieurs mécanismes de sortie:

| Type | Description | Paramètre par défaut |
|------|-------------|---------------------|
| **Stop-Loss** | Prix atteint le stop fixe | Basé sur RSI peaks |
| **Take-Profit** | Profit atteint le seuil | `take_profit_pct = 15%` |
| **Trailing Stop** | Stop suiveur dynamique | Optionnel |
| **Time Stop** | Durée max de détention | `max_hold_days = 60` |

---

## 4. Analyse RSI Trendline Breakout

### 4.1 Principe Mathématique

L'analyse repose sur la détection d'une **droite de résistance descendante** sur le RSI, puis de sa cassure.

#### Étape 1: Calcul du RSI (Wilder, 14 périodes)

```
RS = Moyenne(Gains, 14) / Moyenne(Pertes, 14)
RSI = 100 - (100 / (1 + RS))
```

#### Étape 2: Détection des Pics RSI

Utilisation de `scipy.signal.find_peaks`:

```python
peaks, _ = find_peaks(
    rsi,
    prominence=2.0,    # Amplitude minimum du pic
    distance=5         # Distance minimum entre pics
)
```

#### Étape 3: Régression Linéaire

Pour N pics sélectionnés (minimum 3), calcul de la droite:

```
y = slope × x + intercept
```

Où:
- `x` = indice temporel du pic
- `y` = valeur RSI du pic

**Critères de validation:**
- R² ≥ 0.25 (qualité de la régression)
- slope < 0 (pente négative = résistance descendante)
- Pas de croisement significatif au-dessus entre les pics

### 4.2 Détection du Breakout

```python
# À chaque période t:
trendline_value = slope * t + intercept
distance = rsi[t] - trendline_value

# Breakout si:
if distance >= BREAKOUT_THRESHOLD (0.0):
    if rsi[t-1] <= trendline_value[t-1] + tolerance:
        # BREAKOUT CONFIRMÉ
```

### 4.3 Classification de la Force

| Force | Condition |
|-------|-----------|
| **STRONG** | RSI > 60 ET distance > 3 points |
| **MODERATE** | RSI > 50 OU distance > 2 points |
| **WEAK** | Breakout marginal |

### 4.4 Détection Multi-Obliques

Le système détecte jusqu'à **3 obliques distinctes** par actif:

```python
# Algorithme par combinaisons
from itertools import combinations

for num_peaks in range(3, 7):  # 3 à 6 pics par oblique
    for combo in combinations(peaks[:20], num_peaks):
        if is_valid_trendline(combo):
            trendlines.append(create_trendline(combo))

# Filtrage des obliques chevauchantes
trendlines = filter_overlapping(trendlines, max_trendlines=3)
```

### 4.5 Paramètres Configurables

```python
# trendline_analysis/config/settings.py

# Détection des pics
PEAK_PROMINENCE = 2.0      # Prominence minimum
PEAK_DISTANCE = 5          # Distance minimum entre pics

# Validation de l'oblique
MIN_R_SQUARED = 0.25       # R² minimum
MIN_PEAKS_FOR_TRENDLINE = 3
MAX_RESIDUAL_DISTANCE = 8.0

# Breakout
BREAKOUT_THRESHOLD = 0.0   # Seuil de cassure
MAX_BREAKOUT_AGE = 3       # Âge max pour signal
```

---

## 5. Analyse EMA et Supports

### 5.1 EMAs Utilisées

| EMA | Périodes | Rôle |
|-----|----------|------|
| EMA 24 | 24 | Court terme |
| EMA 38 | 38 | Moyen terme |
| EMA 62 | 62 | Long terme |

### 5.2 Détection des Crossovers

Un crossover est détecté lorsque deux EMAs se croisent:

```python
# Crossover haussier (Golden Cross)
if ema_short[t-1] < ema_long[t-1] and ema_short[t] > ema_long[t]:
    crossover = "BULLISH"

# Crossover baissier (Death Cross)
if ema_short[t-1] > ema_long[t-1] and ema_short[t] < ema_long[t]:
    crossover = "BEARISH"
```

### 5.3 Zones de Support

Le système identifie les **niveaux de prix historiques** où des crossovers haussiers se sont produits:

```python
support_zones = []
for crossover in bullish_crossovers:
    if abs(current_price - crossover.price) / current_price < 0.08:
        support_zones.append({
            'price': crossover.price,
            'date': crossover.date,
            'distance_pct': distance
        })
```

### 5.4 Validation du Contexte EMA

Un breakout RSI n'est **valide** que si:

```python
ema_rising = (ema_24 > ema_38) or (ema_24 > ema_62) or (ema_38 > ema_62)
near_support = len(support_zones) > 0 or was_near_ema_recently()

is_valid = ema_rising and near_support
```

---

## 6. Système de Scoring

### 6.1 Score de Confiance (0-100)

Le score agrège plusieurs facteurs avec pondération:

```python
confidence_score = (
    ema_alignment_score * 0.25 +      # 0-25 points
    support_proximity_score * 0.25 +   # 0-25 points
    rsi_breakout_score * 0.30 +        # 0-30 points
    volume_confirmation_score * 0.20   # 0-20 points
)
```

### 6.2 Détail des Composantes

#### EMA Alignment Score (0-25)

| Configuration | Score |
|---------------|-------|
| EMA24 > EMA38 > EMA62 (parfait) | 25 |
| 2 EMAs alignées | 15 |
| 1 EMA au-dessus | 8 |
| Aucune | 0 |

#### Support Proximity Score (0-25)

| Distance au support | Score |
|---------------------|-------|
| < 2% | 25 |
| 2-4% | 20 |
| 4-6% | 12 |
| 6-8% | 5 |
| > 8% | 0 |

#### RSI Breakout Score (0-30)

| Critère | Points |
|---------|--------|
| Breakout détecté | +10 |
| Force STRONG | +10 |
| Force MODERATE | +5 |
| R² > 0.5 | +5 |
| Âge ≤ 2 périodes | +5 |

#### Volume Confirmation Score (0-20)

| Volume / Moyenne 20j | Score |
|----------------------|-------|
| > 2.0x | 20 |
| > 1.5x | 15 |
| > 1.0x | 10 |
| 0.7-1.0x | 5 |
| < 0.7x | 0 |

### 6.3 Interprétation

| Score | Interprétation | Action suggérée |
|-------|----------------|-----------------|
| 80-100 | Très forte conviction | Entrée immédiate |
| 65-79 | Bonne opportunité | Entrée avec confirmation |
| 50-64 | Signal modéré | Surveiller |
| < 50 | Signal faible | Ignorer |

---

## 7. Position Sizing et Gestion du Risque

### 7.1 Calcul du Stop-Loss

Le stop-loss est fixé au **prix le plus bas correspondant au pic RSI le plus bas** de l'oblique:

```python
def calculate_stop_loss(df, rsi_peak_indices):
    peak_lows = [df['Low'].iloc[i] for i in rsi_peak_indices]
    stop_loss = min(peak_lows)
    return stop_loss
```

**Justification:** Le plus bas pic RSI représente le niveau de support le plus significatif dans la construction de l'oblique.

### 7.2 Risque par Trade (Variable selon Market Cap)

| Market Cap | Catégorie | Risque Max |
|------------|-----------|------------|
| > 200B $ | Mega Cap | 4% du capital |
| 50-200B $ | Large Cap | 3% |
| 10-50B $ | Mid Cap | 2.5% |
| 2-10B $ | Small Cap | 2% |
| < 2B $ | Micro Cap | 1.5% |

### 7.3 Calcul de la Taille de Position

```python
def calculate_position_size(capital, entry_price, stop_loss, risk_pct):
    risk_per_share = entry_price - stop_loss
    risk_amount = capital * risk_pct
    shares = risk_amount / risk_per_share
    return int(shares)
```

### 7.4 Exemple Numérique

```
Capital: 10,000 €
Entry Price: 150 €
Stop-Loss: 142 € (basé sur RSI peaks)
Market Cap: 50B € (Large Cap → 3% risque)

Risk Amount = 10,000 × 0.03 = 300 €
Risk per Share = 150 - 142 = 8 €
Position Size = 300 / 8 = 37 actions
Position Value = 37 × 150 = 5,550 €
```

### 7.5 Gestion du Portfolio

```python
class PortfolioTracker:
    - total_capital: float      # Capital initial
    - positions: Dict           # Positions ouvertes
    - available_capital: float  # Capital disponible

    def open_position(symbol, shares, entry, stop):
        # Vérifie capital disponible
        # Enregistre la position
        # Met à jour capital disponible

    def close_position(symbol, exit_price):
        # Calcule P&L
        # Libère le capital
```

---

## 8. Module de Backtesting

### 8.1 Configuration

```python
@dataclass
class BacktestConfig:
    initial_capital: float = 10000
    position_size_pct: float = 0.10      # 10% par position
    max_positions: int = 5
    min_confidence_score: float = 55
    require_volume_confirmation: bool = True

    # Sorties
    take_profit_pct: float = 0.15        # 15%
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0.08
    max_hold_days: int = 60

    # Détecteur
    use_enhanced_detector: bool = True
    precision_mode: str = 'medium'       # low, medium, high
    timeframe: str = 'weekly'
```

### 8.2 Métriques Calculées

| Métrique | Formule |
|----------|---------|
| **Total Return** | (Final - Initial) / Initial |
| **Win Rate** | Trades gagnants / Total trades |
| **Profit Factor** | Somme gains / Somme pertes |
| **Max Drawdown** | Max(Peak - Trough) / Peak |
| **Sharpe Ratio** | (Return - Rf) / Std(Returns) |
| **Sortino Ratio** | (Return - Rf) / Std(Negative Returns) |
| **Avg Trade Duration** | Moyenne des jours de détention |

### 8.3 Conditions de Sortie

```python
def check_exit_conditions(position, current_bar):
    # 1. Stop-Loss
    if current_bar['Low'] <= position.stop_loss:
        return ('stop_loss', position.stop_loss)

    # 2. Take-Profit
    profit_pct = (current_bar['Close'] - position.entry) / position.entry
    if profit_pct >= config.take_profit_pct:
        return ('take_profit', current_bar['Close'])

    # 3. Trailing Stop
    if config.use_trailing_stop:
        if current_bar['Close'] < position.trailing_stop:
            return ('trailing_stop', position.trailing_stop)

    # 4. Time Stop
    days_held = (current_bar.date - position.entry_date).days
    if days_held >= config.max_hold_days:
        return ('max_hold_period', current_bar['Close'])

    return None  # Garder la position
```

---

## 9. Trend Discovery (IA)

### 9.1 Architecture Hybride

Le module combine:
1. **Analyse quantitative** (données de marché)
2. **Analyse LLM** (interprétation des news)

### 9.2 Détection Quantitative

```python
# Momentum sectoriel
momentum = (performance_5d - performance_20d) / abs(performance_20d)

# Anomalies de volume
volume_ratio = current_volume / avg_volume_20d
is_anomaly = volume_ratio > 1.5

# Breadth (largeur de marché)
breadth = stocks_up / total_stocks
```

### 9.3 Analyse LLM via OpenRouter

```python
prompt = f"""
Analyze these market news and identify:
1. Emerging themes and narratives
2. Sector rotations
3. Sentiment shifts

News: {news_articles}

Return structured JSON with themes, confidence, and affected symbols.
"""

response = await openrouter.chat(model="anthropic/claude-3-haiku", prompt=prompt)
```

### 9.4 Thèmes Prédéfinis

| Thème | Mots-clés |
|-------|-----------|
| AI_Revolution | artificial intelligence, machine learning, neural network |
| Quantum_Computing | quantum, qubit, supremacy |
| Space_Economy | space, satellite, rocket, orbit |
| Obesity_Drugs | GLP-1, Ozempic, Wegovy, obesity |
| Nuclear_Renaissance | nuclear, uranium, SMR |
| EV_Transition | electric vehicle, EV, charging |
| Cybersecurity | cyber, ransomware, zero trust |
| Reshoring | reshoring, nearshoring, supply chain |

### 9.5 Apprentissage de Nouveaux Thèmes

Le LLM peut découvrir de **nouveaux narratifs** non prédéfinis:

```python
# Stockage dans learned_themes.json
{
    "theme_id": {
        "name": "Renaissance des Matières Premières",
        "description": "Retour de l'intérêt pour les commodities...",
        "keywords": ["commodities", "copper", "lithium"],
        "symbols": ["FCX", "ALB", "MP"],
        "discovered_at": "2024-12-27",
        "occurrence_count": 3
    }
}
```

---

## 10. Sources de Données

### 10.1 Données de Marché

| Source | Type | Latence | Limitations |
|--------|------|---------|-------------|
| **yfinance** | OHLCV, Fondamentaux | ~15 min | Rate limiting, pas de tick data |

### 10.2 News et Sentiment

| Source | Type | API Key | Usage |
|--------|------|---------|-------|
| **NewsAPI** | Articles de presse | Requis | Analyse thématique |
| **AlphaVantage** | News + Sentiment | Requis | Sentiment pré-calculé |

### 10.3 Intelligence Artificielle

| Service | Modèles | Usage |
|---------|---------|-------|
| **OpenRouter** | Claude, GPT-4, Gemini | Analyse narrative |

---

## 11. Base de Données

### 11.1 Schéma SQLite

```sql
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    timeframe TEXT,
    signal TEXT,
    confidence_score REAL,
    entry_price REAL,
    stop_loss REAL,
    rsi_value REAL,
    volume_ratio REAL,
    market_cap REAL,
    sector TEXT,
    notes TEXT
);

CREATE INDEX idx_alerts_symbol ON alerts(symbol);
CREATE INDEX idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX idx_alerts_signal ON alerts(signal);
```

### 11.2 Fichiers JSON

| Fichier | Contenu |
|---------|---------|
| `data/portfolio.json` | Positions ouvertes |
| `data/trends/trend_report_*.json` | Rapports de scan |
| `data/trends/learned_themes.json` | Thèmes découverts par IA |
| `data/trends/focus_symbols.json` | Symboles à surveiller |

---

## 12. Configuration et Paramètres

### 12.1 Variables d'Environnement (.env)

```bash
# Capital
CAPITAL=10000

# APIs
OPENROUTER_API_KEY=sk-...
OPENROUTER_MODEL=google/gemini-3-flash-preview
NEWSAPI_KEY=...
ALPHAVANTAGE_KEY=...

# Scheduling
TREND_SCAN_TIME=06:00
```

### 12.2 Paramètres Critiques (config/settings.py)

```python
# Screening
EMA_PERIODS = [24, 38, 62]
SUPPORT_ZONE_THRESHOLD = 0.08  # 8%

# RSI Trendline
MIN_R_SQUARED = 0.25
BREAKOUT_THRESHOLD = 0.0
MAX_BREAKOUT_AGE = 3

# Position Sizing
RISK_SCALING = {
    'mega_cap': 0.04,   # > 200B
    'large_cap': 0.03,  # 50-200B
    'mid_cap': 0.025,   # 10-50B
    'small_cap': 0.02,  # 2-10B
    'micro_cap': 0.015  # < 2B
}
```

---

## 13. Limites et Risques

### 13.1 Limites Techniques

| Limite | Impact | Mitigation |
|--------|--------|------------|
| Données retardées (15 min) | Entrées imprécises | Utiliser ordres limites |
| Pas de données tick | Slippage non modélisé | Marge de sécurité sur stop |
| Dépendance yfinance | Interruption de service | Retry avec backoff exponentiel |
| Rate limiting APIs | Scans lents | Batch processing, cache |

### 13.2 Limites Méthodologiques

| Limite | Description |
|--------|-------------|
| **Biais de survie** | Les backtests n'incluent pas les entreprises délistées |
| **Overfitting** | Les paramètres sont optimisés sur données historiques |
| **Corrélation** | Pas de contrôle de corrélation inter-positions |
| **Liquidité** | Pas de vérification de la liquidité avant sizing |
| **Coûts** | Frais de transaction non inclus dans backtest |

### 13.3 Risques Opérationnels

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| Panne API externe | Moyenne | Moyen | Fallback, alertes |
| Erreur de calcul | Faible | Élevé | Tests unitaires |
| Faux signaux | Élevée | Moyen | Score de confiance |
| Black swan event | Faible | Très élevé | Diversification, stops |

### 13.4 Avertissements

> **AVERTISSEMENT LÉGAL**
>
> Ce système est fourni à titre éducatif et informationnel uniquement.
> Il ne constitue en aucun cas un conseil en investissement.
> Les performances passées ne préjugent pas des performances futures.
> Le trading comporte des risques de perte en capital.

---

## Annexes

### A. Formules Mathématiques

#### RSI (Relative Strength Index)

$$RSI = 100 - \frac{100}{1 + RS}$$

où $RS = \frac{\text{Moyenne des gains sur N périodes}}{\text{Moyenne des pertes sur N périodes}}$

#### EMA (Exponential Moving Average)

$$EMA_t = \alpha \times Price_t + (1 - \alpha) \times EMA_{t-1}$$

où $\alpha = \frac{2}{N + 1}$

#### Régression Linéaire (Oblique RSI)

$$y = \beta_0 + \beta_1 x$$

$$\beta_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$$

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

#### Sharpe Ratio

$$Sharpe = \frac{R_p - R_f}{\sigma_p}$$

#### Max Drawdown

$$MDD = \max_{t \in [0,T]} \left( \frac{Peak_t - Trough_t}{Peak_t} \right)$$

### B. Glossaire

| Terme | Définition |
|-------|------------|
| **Breakout** | Cassure d'un niveau de résistance |
| **Crossover** | Croisement de deux moyennes mobiles |
| **Drawdown** | Perte depuis le plus haut |
| **EMA** | Moyenne mobile exponentielle |
| **Momentum** | Vitesse de variation du prix |
| **RSI** | Indice de force relative (0-100) |
| **Stop-loss** | Ordre de vente automatique à perte |
| **Support** | Niveau de prix où la demande domine |
| **Trendline** | Droite reliant des sommets ou creux |

---

**Document généré le:** 27 Décembre 2024
**Version:** 1.0
**Statut:** Pour audit
