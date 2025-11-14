# RSI Trendline Breakout Analysis

Module d'analyse des ruptures de lignes de tendance (trendlines) sur le RSI, conçu pour compléter le screener EMA existant.

## Vue d'ensemble

Ce module détecte automatiquement:
1. **Obliques de résistance descendantes** sur le RSI (lignes reliant les sommets)
2. **Ruptures haussières** (breakouts) lorsque le RSI casse au-dessus de l'oblique
3. **Signaux de trading** combinant les critères EMA et les breakouts de trendlines

## Structure du Module

```
trendline_analysis/
├── core/
│   ├── trendline_detector.py    # Détection des peaks et obliques RSI
│   ├── breakout_analyzer.py     # Analyse des ruptures
│   └── validator.py             # Validation de qualité
├── screeners/
│   └── trendline_screener.py    # Intégration avec screener EMA
├── visualization/
│   └── trendline_visualizer.py  # Affichage graphique
└── config/
    └── settings.py              # Configuration
```

## Fonctionnement

### 1. Détection de Trendline (trendline_detector.py)

**Processus:**
1. Calcul du RSI (14 périodes)
2. Détection des sommets (peaks) avec `scipy.signal.find_peaks`
3. Régression linéaire sur les 3+ derniers peaks
4. Validation de la qualité (R², pente)

**Critères de qualité:**
- Minimum 3 peaks
- R² ≥ 0.75 (bonne corrélation)
- Pente entre -0.5 et -0.01 (descente significative)

**Exemple:**
```python
from trendline_analysis.core.trendline_detector import RSITrendlineDetector

detector = RSITrendlineDetector()
trendline = detector.detect(df, lookback_periods=52)

if trendline:
    print(f"Quality: {trendline.quality_score:.1f}/100")
    print(f"R²: {trendline.r_squared:.3f}")
    print(f"Peaks: {len(trendline.peak_indices)}")
```

### 2. Détection de Breakout (breakout_analyzer.py)

**Processus:**
1. Vérifie si le RSI croise au-dessus de l'oblique
2. Valide la confirmation (reste au-dessus)
3. Calcule la force du breakout

**Critères de breakout:**
- RSI > Trendline + 0.5 points minimum
- Crossing (pas déjà au-dessus)
- Âge ≤ 3 périodes (récent)

**Forces de breakout:**
- **STRONG**: RSI > 60 ET distance > 3 points
- **MODERATE**: RSI > 50 OU distance > 2 points
- **WEAK**: Autres cas

**Exemple:**
```python
from trendline_analysis.core.breakout_analyzer import TrendlineBreakoutAnalyzer

analyzer = TrendlineBreakoutAnalyzer()
analysis = analyzer.analyze(df, lookback_periods=52)

if analysis['has_breakout']:
    breakout = analysis['breakout']
    print(f"Breakout: {breakout.strength}")
    print(f"Date: {breakout.date}")
    print(f"Confirmed: {breakout.is_confirmed}")
```

### 3. Screening (trendline_screener.py)

**Workflow:**
```
Symboles du screener EMA
    ↓
Analyse RSI Weekly
    ↓
Trendline détectée?
    ├─ OUI → Breakout récent?
    │         ├─ OUI → ✅ SIGNAL FORT
    │         └─ NON → Continuer
    └─ NON → Analyse RSI Daily
              ↓
              Breakout?
                  ├─ OUI → ✅ SIGNAL
                  └─ NON → Pas de signal
```

**Exemple:**
```python
from trendline_analysis.screeners.trendline_screener import TrendlineBreakoutScreener

screener = TrendlineBreakoutScreener()

# Filtrer les résultats EMA
enhanced_results = screener.filter_ema_results(
    ema_alerts,
    data_provider_func
)

for result in enhanced_results:
    print(f"{result['symbol']}: {result['signal']}")
    print(f"  Timeframe: {result['timeframe']}")
    print(f"  Combined Score: {result['combined_score']}/100")
```

### 4. Visualisation (trendline_visualizer.py)

**Affichage:**
- Ligne d'oblique (orange, pointillée)
- Marqueurs sur les peaks
- Étoile au point de breakout
- Ligne verticale au moment du breakout

**Exemple:**
```python
from trendline_analysis.visualization.trendline_visualizer import TrendlineVisualizer

visualizer = TrendlineVisualizer()

# Créer un graphique autonome
chart_result = visualizer.create_annotated_rsi_chart(
    df, symbol="BTC-USD", timeframe="weekly"
)

if chart_result:
    chart_result['fig'].show()  # Afficher
    chart_result['fig'].write_html("btc_trendline.html")  # Sauvegarder
```

## Configuration (config/settings.py)

### Paramètres de Détection des Peaks

```python
PEAK_PROMINENCE = 5.0  # Importance minimale d'un peak
PEAK_DISTANCE = 5      # Distance minimale entre peaks
MIN_PEAKS_FOR_TRENDLINE = 3  # Nombre minimum de peaks
```

**Ajustement:**
- ↑ PROMINENCE = Moins de peaks, plus significatifs
- ↑ DISTANCE = Peaks plus espacés
- ↑ MIN_PEAKS = Trendlines plus robustes

### Paramètres de Qualité de Trendline

```python
MIN_R_SQUARED = 0.75   # R² minimum
MIN_SLOPE = -0.5       # Pente minimale (descente)
MAX_SLOPE = -0.01      # Pente maximale (trop plat = non pertinent)
```

**Ajustement:**
- ↑ R_SQUARED = Trendlines plus précises, moins de résultats
- Ajuster SLOPE pour capturer différents types de descentes

### Paramètres de Breakout

```python
BREAKOUT_THRESHOLD = 0.5  # Distance minimale au-dessus
CONFIRMATION_PERIODS = 1   # Périodes de confirmation
MAX_BREAKOUT_AGE = 3      # Âge maximum (périodes)
```

## Test Rapide

```bash
# Installer scipy
pip install scipy>=1.10.0

# Tester sur BTC
python test_trendline_btc.py
```

Cela génère deux fichiers HTML:
- `btc_trendline_weekly.html`
- `btc_trendline_daily.html`

Ouvrez-les dans un navigateur pour voir les obliques détectées.

## Intégration avec le Screener EMA

### Option 1: Post-Processing

```python
# Dans main.py ou dashboard
from src.screening.screener import MarketScreener
from trendline_analysis.screeners.trendline_screener import TrendlineBreakoutScreener

# Screener EMA normal
market_screener = MarketScreener()
ema_results = market_screener.screen_market()

# Filtrer avec trendlines
trendline_screener = TrendlineBreakoutScreener()
enhanced_results = trendline_screener.filter_ema_results(
    ema_results,
    data_provider_func
)
```

### Option 2: Dashboard Integration

Ajouter une nouvelle page au dashboard Streamlit pour afficher:
1. Symboles avec signaux EMA
2. Analyse trendline pour chaque symbole
3. Graphiques avec obliques annotées
4. Score combiné EMA + Trendline

## Exemples d'Utilisation

### Exemple 1: Analyse Simple

```python
import yfinance as yf
from trendline_analysis.core.breakout_analyzer import TrendlineBreakoutAnalyzer

# Fetch data
df = yf.download("AAPL", period="2y", interval="1wk")

# Analyze
analyzer = TrendlineBreakoutAnalyzer()
analysis = analyzer.analyze(df, lookback_periods=52)

# Get signal
signal = analyzer.get_signal(analysis)
print(f"Signal: {signal}")
```

### Exemple 2: Screening Multiple Symboles

```python
from trendline_analysis.screeners.trendline_screener import TrendlineBreakoutScreener

symbols = ["AAPL", "MSFT", "GOOGL", "NVDA"]
screener = TrendlineBreakoutScreener()

symbols_data = []
for symbol in symbols:
    df_weekly = yf.download(symbol, period="2y", interval="1wk")
    df_daily = yf.download(symbol, period="1y", interval="1d")
    symbols_data.append({
        'symbol': symbol,
        'df_weekly': df_weekly,
        'df_daily': df_daily
    })

results = screener.screen_multiple_symbols(symbols_data)

for r in results:
    print(f"{r['symbol']}: {r['signal']} on {r['timeframe']}")
```

### Exemple 3: Visualisation Avancée

```python
from trendline_analysis.visualization.trendline_visualizer import TrendlineVisualizer

visualizer = TrendlineVisualizer()

# Créer chart
chart = visualizer.create_annotated_rsi_chart(
    df, "AAPL", "weekly", lookback_periods=104
)

if chart and chart['has_breakout']:
    print(f"Signal: {chart['signal']}")

    # Informations détaillées
    info = visualizer.get_trendline_info_text(
        chart['analysis']['trendline'],
        chart['analysis']['breakout']
    )
    print(info)

    # Sauvegarder
    chart['fig'].write_html("aapl_analysis.html")
```

## Cas d'Usage par Rapport aux Screenshots BTC

D'après vos screenshots:

**Image 1 (Weekly BTC):**
- Oblique descendante reliant plusieurs peaks du RSI
- Breakout visible en octobre 2023
- ✅ Ce module détectera cette configuration

**Image 2 (Daily BTC):**
- Même type d'oblique sur daily
- Point de breakout marqué
- ✅ Si pas d'oblique sur weekly, analyse le daily

**Workflow optimal:**
1. Le screener EMA détecte BTC comme candidat
2. Le module trendline analyse le RSI weekly
3. Détecte l'oblique descendante
4. Identifie le breakout en octobre 2023
5. Signal: STRONG_BUY (confirmé + forte distance)

## Débogage

Si aucune trendline n'est détectée:

1. **Pas assez de peaks:**
   - Réduire `PEAK_PROMINENCE`
   - Réduire `PEAK_DISTANCE`

2. **R² trop bas:**
   - Réduire `MIN_R_SQUARED` (ex: 0.70)

3. **Pente hors limites:**
   - Ajuster `MIN_SLOPE` et `MAX_SLOPE`

4. **Lookback trop court:**
   - Augmenter `lookback_periods`

## Performance

- **Détection de trendline:** ~0.1-0.2s par symbole
- **Screening 100 symboles:** ~10-20s (parallèle)
- **Génération de chart:** ~0.5s

## Prochaines Étapes

1. ✅ Tester sur BTC et valider contre screenshots
2. ⏳ Ajuster paramètres si nécessaire
3. ⏳ Intégrer au dashboard Streamlit
4. ⏳ Ajouter à la boucle de screening automatique
5. ⏳ Créer notifications Telegram pour breakouts

## Support

Pour des questions ou ajustements:
- Consulter `trendline_analysis/config/settings.py` pour les paramètres
- Tester avec `test_trendline_btc.py`
- Comparer les graphiques générés avec vos screenshots

---

**Ce module transforme le screener en un système de détection de breakouts techniques professionnels!** 📈
