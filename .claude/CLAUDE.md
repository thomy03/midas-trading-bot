# TradingBot V3 - Guide de Contexte Claude

## Apercu du Projet

TradingBot V3 est un systeme de screening d'actions base sur:
1. **Analyse EMA** - Detection de supports/resistances via croisements EMA (24/38/62)
2. **RSI Trendline Breakout** - Detection d'obliques descendantes sur le RSI et leurs cassures
3. **Dashboard Interactif** - Interface Streamlit avec graphiques Plotly interactifs

## Architecture

```
TradingBot_V3/
├── dashboard.py                 # Interface principale Streamlit
├── config/settings.py           # Configuration globale
├── src/
│   ├── screening/screener.py    # Logique de screening principale
│   ├── indicators/ema_analyzer.py # Calcul EMAs et crossovers
│   ├── data/market_data.py      # Fetching donnees yfinance
│   └── utils/
│       ├── interactive_chart.py # Graphiques interactifs Plotly
│       └── visualizer.py        # Graphiques statiques
└── trendline_analysis/
    ├── core/
    │   ├── trendline_detector.py      # Detection obliques RSI
    │   ├── breakout_analyzer.py       # Detection breakouts
    │   └── rsi_breakout_analyzer.py   # Analyseur simplifie
    └── config/settings.py             # Parametres trendlines
```

## Fonctionnalites Dashboard (Dec 2024)

### Pages Disponibles
- **Pro Chart**: Graphique interactif avec pan/zoom/dessins + zones crossover EMA
- **Chart Analyzer**: Analyse technique avec EMAs et supports
- **Signaux Historiques**: Niveaux historiques + obliques RSI
- **Trendline Analysis**: Double confirmation RSI + Prix
- **Screening**: Scan manuel single/multi symboles
- **Scheduler**: Scans automatiques programmes
- **Backtesting**: Simulation de strategie sur donnees historiques
- **Watchlists**: Gestion de listes personnalisees
- **Portfolio**: Suivi des positions
- **Performance**: Analyse de performance
- **Calendar**: Calendrier economique
- **Settings**: Configuration (notifications, email, Telegram)

### Interactivite Graphiques
- **Pan**: Click + drag pour deplacer le graphique
- **Zoom**: Molette souris pour zoomer
- **Double-click**: Reset vue
- **Crossovers EMA**: Affichage des zones de croisement (diamants verts/rouges)

## Mecanisme RSI Breakout

### 1. Detection des Pics RSI
```python
# Utilise scipy.signal.find_peaks
peaks, _ = find_peaks(rsi, prominence=2.0, distance=5)
```

### 2. Construction de l'Oblique
- Minimum 3 pics pour former une trendline
- Regression lineaire (y = slope * x + intercept)
- R² minimum 0.25 pour valider la qualite
- Pente negative obligatoire (MAX_SLOPE = 0.0)

### 3. Validation Resistance
L'oblique doit etre une VRAIE resistance:
- Le RSI ne doit pas croiser significativement AU-DESSUS entre les pics
- Tolerance de 2.0 points RSI

### 4. Detection Breakout
```python
# Breakout = RSI croise au-dessus de l'oblique
distance_above = rsi_value - trendline_value
if distance_above >= 0:  # BREAKOUT_THRESHOLD = 0.0
    # Verifier que c'etait en-dessous avant
    if prev_rsi <= prev_trendline + threshold:
        # BREAKOUT DETECTE!
```

### 5. Force du Breakout
| Force    | Condition                           |
|----------|-------------------------------------|
| STRONG   | RSI > 60 ET distance > 3 points     |
| MODERATE | RSI > 50 OU distance > 2 points     |
| WEAK     | Breakout marginal                   |

### 6. Signal Trading
| Signal      | Condition                                    |
|-------------|----------------------------------------------|
| STRONG_BUY  | STRONG breakout, age ≤ 3 periodes            |
| BUY         | MODERATE/STRONG breakout, age ≤ 5 periodes   |
| WATCH       | Oblique presente mais pas de breakout        |

## Parametres Cles (trendline_analysis/config/settings.py)

```python
# Detection pics
PEAK_PROMINENCE = 2.0      # Prominence minimum
PEAK_DISTANCE = 5          # Distance min entre pics

# Validation oblique
MIN_R_SQUARED = 0.25       # Qualite regression
MIN_PEAKS_FOR_TRENDLINE = 3
MAX_RESIDUAL_DISTANCE = 8.0

# Breakout
BREAKOUT_THRESHOLD = 0.0   # Sensibilite detection
MAX_BREAKOUT_AGE = 3       # Age max pour signal
```

## Integration avec le Screener EMA

Le systeme combine:
1. **Niveaux EMA**: Prix proche d'un support (crossover EMA historique)
2. **RSI Breakout**: Cassure de l'oblique RSI descendante

Signal final = Niveau EMA proche + RSI Breakout = STRONG_BUY

## VALIDATION DU BREAKOUT RSI (IMPORTANTE!)

Un breakout RSI n'est **VALIDE** que si TOUTES ces conditions sont remplies:

### Condition 1: EMAs Croissantes
Au moins une EMA courte doit etre au-dessus d'une EMA longue au moment du breakout:
- EMA 24 > EMA 38, OU
- EMA 24 > EMA 62, OU
- EMA 38 > EMA 62

```python
ema_rising = (ema_24 > ema_38) or (ema_24 > ema_62) or (ema_38 > ema_62)
```

### Condition 2: Prix proche d'un Support EMA
Le prix doit avoir touche une zone de support (crossover EMA ou EMA directe) recemment:
- Soit `support_zones` non vide (prix < 8% d'un niveau)
- Soit le prix etait proche d'une EMA dans les 20 dernieres periodes

### Visualisation
- ★ **Etoile verte**: Breakout VALIDE (conditions remplies)
- ⚠ **Cercle jaune**: Breakout detecte mais NON VALIDE
  - Raison affichee: "EMAs non croissantes" ou "Prix loin des supports"

### Code de validation (`_validate_rsi_breakout`)
```python
# Dans interactive_chart.py
def _validate_rsi_breakout(self, df, df_with_emas, support_zones, breakout):
    # 1. Verifier EMAs croissantes au moment du breakout
    # 2. Verifier si prix proche d'un support
    return (is_valid, reason)
```

## Commandes Utiles

```bash
# Lancer le dashboard
streamlit run dashboard.py

# Tests
python -m pytest tests/

# Screening rapide
python run_screener.py
```

## Detection Multi-Obliques RSI (Dec 2024)

### Principe
Le systeme detecte desormais **jusqu'a 3 obliques RSI distinctes** par timeframe:
- Permet de ne manquer aucune cassure d'oblique repondant aux criteres
- Chaque oblique est validee independamment
- Le screener verifie le breakout sur TOUTES les obliques detectees

### Algorithme de Selection des Pics
```python
# Utilise itertools.combinations pour tester TOUTES les combinaisons de pics
# (pas seulement les pics consecutifs)
from itertools import combinations

max_peaks_to_consider = 20  # Augmente de 12 a 20 pour plus d'historique
for num_peaks in range(3, 7):  # De 3 a 6 pics par oblique
    for combo in combinations(range(len(peaks)), num_peaks):
        # Valider chaque combinaison comme oblique potentielle
```

### Filtrage des Obliques Chevauchantes
```python
def _filter_overlapping_trendlines(trendlines, max_trendlines=3):
    # Garde les obliques de DIFFERENTES periodes temporelles
    # Rejette si > 50% de pics en commun ET meme periode
    # Trie par R² descendant (meilleure qualite en premier)
```

### Visualisation Multi-Obliques
- **3 couleurs distinctes**: orange, violet, cyan
- Chaque oblique affichee avec ses propres pics
- Breakouts marques sur l'oblique correspondante
- Extension jusqu'a la fin des donnees (pas juste au dernier pic)

### Integration Screener
Les deux analyseurs (`RSIBreakoutAnalyzer` et `EnhancedRSIBreakoutAnalyzer`) :
1. Detectent TOUTES les obliques valides
2. Verifient le breakout sur CHAQUE oblique
3. Retournent le **meilleur signal** trouve

```python
# Priorite des signaux
signal_priority = {'STRONG_BUY': 4, 'BUY': 3, 'WATCH': 2, 'NO_SIGNAL': 1}

# Pour chaque oblique, si breakout detecte et meilleur que l'actuel
if self._is_better_signal(new_signal, best_signal):
    best_breakout = breakout
    best_signal = signal
```

## Derniere Session (Dec 2024)

### Modifications recentes:
- **Volume Confirmation au Breakout**: Le ratio volume/moyenne 20j est maintenant calcule au moment du breakout
  - Nouveau champs `volume_ratio` et `volume_confirmed` dans `Breakout` dataclass
  - Methode `_calculate_volume_confirmation()` dans `breakout_analyzer.py`
  - Propagation dans les alertes du screener
- Ajout pan/zoom interactif sur Pro Chart et Chart Analyzer
- Visualisation zones de croisement EMA (diamants verts/rouges)
- Integration crossovers dans `create_interactive_chart()`
- Config Plotly: `scrollZoom: True`, `dragmode: 'pan'`
- **Visualisation RSI Trendline Breakout** sur les graphiques:
  - Ligne orange pointillee = oblique de resistance RSI
  - Triangles rouges = pics RSI utilises pour l'oblique
  - Etoile verte = breakout VALIDE avec force (WEAK/MODERATE/STRONG)
  - Cercle jaune = breakout NON VALIDE (EMAs non croissantes ou prix loin supports)
  - Checkbox "RSI Oblique" pour activer/desactiver
  - Methode `add_rsi_trendline_breakout()` dans interactive_chart.py
  - Methode `_validate_rsi_breakout()` pour valider les conditions EMA + support
- **Detection Multi-Obliques** (jusqu'a 3 par timeframe):
  - Algorithme par combinaisons (pas seulement pics consecutifs)
  - `max_peaks_to_consider = 20` pour plus d'historique
  - Filtrage par periode temporelle (garde obliques distinctes)
  - Screener verifie breakout sur TOUTES les obliques
  - Retourne le meilleur signal trouve parmi toutes les obliques

## Ameliorations Trading (Dec 2024)

### Scoring de Confiance (0-100)
Nouveau systeme de scoring numerique en plus des signaux categoriques:
```python
confidence_score = (
    ema_alignment_score * 0.25 +      # EMAs croissantes (0-25)
    support_proximity_score * 0.25 +   # Distance au support (0-25)
    rsi_breakout_score * 0.30 +        # Qualite du breakout (0-30)
    volume_confirmation_score * 0.20   # Volume relatif (0-20)
)
```
- Affichage: `STRONG_BUY (87/100)` ou `BUY (65/100)`
- Fichier: `src/utils/confidence_scorer.py`

### Confirmation Volume au Breakout (Dec 2024)
Le volume au moment du breakout RSI est maintenant calcule et stocke directement dans le `Breakout` dataclass:

```python
@dataclass
class Breakout:
    # ... autres champs ...
    volume_ratio: float = 1.0      # Volume breakout / moyenne 20j
    volume_confirmed: bool = True   # True si volume_ratio >= 1.0
```

**Calcul du volume_ratio:**
```python
# Dans breakout_analyzer.py
def _calculate_volume_confirmation(self, df, breakout_idx):
    avg_volume = df['Volume'].rolling(20, min_periods=10).mean()
    breakout_volume = df['Volume'].iloc[breakout_idx]
    volume_ratio = breakout_volume / avg_at_breakout
    volume_confirmed = volume_ratio >= 1.0
    return (volume_ratio, volume_confirmed)
```

**Interpretation:**
- `volume_ratio >= 2.0`: Volume fort, breakout tres fiable
- `volume_ratio >= 1.5`: Volume eleve, bonne confirmation
- `volume_ratio >= 1.0`: Volume normal, confirmation OK
- `volume_ratio < 1.0`: Volume faible, breakout potentiellement faux

**Integration dans les alertes:**
```python
alert['volume_ratio'] = breakout.volume_ratio
alert['volume_confirmed'] = breakout.volume_confirmed
```

### Validation par Volume (Scoring)
Le volume contribue au score de confiance (0-15 points):
- Volume > 2.0x moyenne 20j → 15 points
- Volume > 1.5x moyenne → 12 points
- Volume > 1.0x moyenne → 8 points
- Volume 0.7-1.0x moyenne → 4 points
- Volume < 0.7x moyenne → 0 points (signal faible)

### Position Sizing Automatique (REFACTORISE Dec 2024)
Calcul automatique de la taille de position basee sur:
- **Stop-loss FIXE**: Prix le plus bas au niveau du plus bas pic RSI de l'oblique
- **Risque variable selon market cap**:
  - Mega cap (>200B): 4% du capital
  - Large cap (50-200B): 3%
  - Mid cap (10-50B): 2.5%
  - Small cap (2-10B): 2%
  - Micro cap (<2B): 1.5%
- **Portfolio tracking**: Suivi des positions ouvertes, capital restant

```python
from src.utils.position_sizing import PositionSizer

# Initialisation avec capital
sizer = PositionSizer(total_capital=10000)

# Calcul position
position = sizer.calculate(
    df=df,
    entry_price=150.0,
    symbol='AAPL',
    market_cap=3000,  # En milliards
    rsi_peaks_indices=[10, 25, 40]  # Indices des pics RSI
)
# Retourne: shares, stop_loss, stop_source, risk_amount, risk_pct

# Gestion portfolio
sizer.open_position('AAPL', position)  # Enregistrer position
sizer.get_available_capital()  # Capital disponible
sizer.close_position('AAPL')  # Fermer position
```

**Fichiers**:
- `src/utils/position_sizing.py` - Classes PositionSizer, PortfolioTracker
- `data/portfolio.json` - Persistance des positions ouvertes

### Multi-Assets Support
Extension a plusieurs marches:
- **NASDAQ/S&P500**: US stocks (existant)
- **CRYPTO**: 20 cryptos (BTC, ETH, SOL, etc.) via yfinance
- **EUROPE**: 50 valeurs pan-europeennes (FR, DE, NL, IT, ES, BE, FI, DK, CH)
- **CAC40**: Actions francaises (subset de EUROPE)
- **DAX**: Actions allemandes (subset de EUROPE)

Fichiers tickers:
- `data/tickers/crypto.json` - 20 cryptos
- `data/tickers/europe.json` - 50 valeurs europeennes
- `data/tickers/cac40.json` - 40 valeurs CAC40

### Performance Backend
- **Batch insert DB**: 100x plus rapide pour sauvegarder les alertes
- **Exponential backoff**: Retry intelligent avec delai croissant
- **Validation donnees**: Verification qualite (NaN, dates, colonnes)
- Fichier modifie: `src/database/db_manager.py`, `src/data/market_data.py`

### Module de Backtesting (Dec 2024)
Nouveau module pour tester la strategie sur donnees historiques.

**Fichiers:**
- `src/backtesting/backtester.py` - Classe principale `Backtester`
- `src/backtesting/metrics.py` - Calcul des metriques de performance

**Usage:**
```python
from src.backtesting.backtester import Backtester, BacktestConfig

config = BacktestConfig(
    initial_capital=10000,
    min_confidence_score=55,
    require_volume_confirmation=True,
    use_enhanced_detector=True,
    precision_mode='medium',
    timeframe='weekly',
    take_profit_pct=0.15,
    max_hold_days=60
)

backtester = Backtester(config)
result = backtester.run(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2023-01-01',
    end_date='2024-12-01'
)

print(result)  # Affiche le rapport complet
```

**Metriques calculees:**
- Win rate, Profit factor
- Total return, Max drawdown
- Sharpe ratio, Sortino ratio
- Average trade duration
- Best/Worst trade

**Conditions de sortie:**
- `stop_loss`: Prix touche le stop-loss (base sur RSI peaks)
- `take_profit`: Profit >= take_profit_pct
- `max_hold_period`: Duree max atteinte
- `trailing_stop`: Stop suiveur (optionnel)
- `end_of_period`: Fin de la periode de backtest

### Page Backtesting Dashboard (Dec 2024)
Interface graphique pour le backtesting dans le dashboard Streamlit.

**4 onglets:**
1. **Configuration** - Parametres du backtest:
   - Capital initial, taille position, max positions
   - Score confiance minimum, confirmation volume
   - Take profit, trailing stop, duree max
   - Precision detecteur, timeframe (daily/weekly)

2. **Run Backtest** - Execution:
   - Selection dates debut/fin
   - Selection symboles: Manuel, Market (NASDAQ/SP500/Europe), Watchlist
   - Progress bar et statut

3. **Results** - Resultats:
   - Metriques: Return, Win Rate, Profit Factor, Max Drawdown, Sharpe
   - Courbe d'equity interactive (Plotly)
   - Table des trades avec export CSV
   - Rapport complet

4. **Mass Simulation** - Simulation de masse:
   - Selection marches: NASDAQ + Europe + SP500 + CAC40 + DAX
   - Periode: 6 mois, 1 an, 2 ans
   - Timeframe: Weekly ou Daily
   - Limite symboles par marche
   - Warning: peut prendre 30-60 minutes!

**Tests:** `tests/integration/test_backtesting.py` (40 tests)

### Configuration
Nouveaux parametres dans `config/settings.py`:
```python
CAPITAL = 10000  # Capital en euros
RISK_PER_TRADE = 0.02  # 2% max par trade (obsolete, maintenant dynamique)
MARKETS_EXTENDED = {
    'NASDAQ': True, 'SP500': True, 'CRYPTO': True,
    'CAC40': True, 'DAX': False, 'EUROPE': True
}
```

### Integration Scoring + Position Sizing dans Screener
Le screener integre maintenant automatiquement:
1. **Confidence Score** sur chaque alerte (0-100)
2. **Position sizing** avec stop-loss base sur RSI peaks
3. **Gestion du portfolio** avec capital restant

```python
from src.screening.screener import market_screener

# Screening avec scoring + sizing automatique
alerts = market_screener.screen_multiple_stocks(stocks)
# Chaque alerte contient:
# - confidence_score, confidence_signal (ex: "BUY (67/100)")
# - position_shares, position_value, stop_loss, stop_source
# - risk_amount, risk_pct, market_cap_b

# Gestion portfolio via screener
market_screener.update_capital(15000)  # Modifier capital
market_screener.add_position('AAPL', 10, 175.0, 168.0)  # Ajouter position
market_screener.close_position('AAPL')  # Fermer position
summary = market_screener.get_portfolio_summary()  # Voir etat portfolio
available = market_screener.get_available_capital()  # Capital disponible
```

## Trend Discovery (Dec 2024)

### Module de Decouverte de Tendances
Systeme hybride pour detecter automatiquement les tendances emergentes du marche.

**Fichiers:**
- `src/intelligence/trend_discovery.py` - Module principal
- `scripts/run_trend_discovery.py` - Script d'execution
- `tests/integration/test_trend_discovery.py` - Tests (19 tests)

### Approche Hybride

1. **Detection Quantitative**:
   - Momentum sectoriel (performance 5j vs 20j)
   - Anomalies de volume (>1.5x moyenne 20j)
   - Divergences de breadth (% actions en hausse)

2. **Analyse LLM (via OpenRouter)**:
   - Analyse des news par theme
   - Detection de sentiment
   - Identification de narratifs emergents

### Usage

```python
from src.intelligence.trend_discovery import TrendDiscovery, get_trend_discovery

# Initialisation
discovery = TrendDiscovery(
    openrouter_api_key="sk-...",
    model="anthropic/claude-3-haiku"  # ou sonnet, opus, gpt-4-turbo
)
await discovery.initialize()

# Scan quotidien complet
report = await discovery.daily_scan()
print(report.summary())

# Scan d'un secteur specifique
sector_data = await discovery.scan_sector("AI_Semiconductors")

# Obtenir les symboles a surveiller
focus_symbols = discovery.get_focus_symbols(min_confidence=0.5)
focus_sectors = discovery.get_sector_focus(min_momentum=0.2)

# Fermer
await discovery.close()
```

### Scheduler Quotidien

```python
from src.intelligence.trend_discovery import TrendDiscoveryScheduler

scheduler = TrendDiscoveryScheduler(
    discovery=discovery,
    run_time="06:00",  # Heure d'execution
    timezone="America/New_York"
)

await scheduler.start()  # Lance le scheduler
report = await scheduler.run_now()  # Scan immediat
await scheduler.stop()  # Arrete le scheduler
```

### Script CLI

```bash
# Scan immediat
python scripts/run_trend_discovery.py

# Scan d'un secteur
python scripts/run_trend_discovery.py --sector Technology

# Demarrer le scheduler
python scripts/run_trend_discovery.py --schedule

# Avec un modele specifique
python scripts/run_trend_discovery.py --model anthropic/claude-3-sonnet

# Lister les secteurs disponibles
python scripts/run_trend_discovery.py --list-sectors
```

### Configuration (.env)

```bash
# API OpenRouter
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=anthropic/claude-3-haiku

# Heure du scan quotidien
TREND_SCAN_TIME=06:00
```

### Secteurs Surveilles

| Secteur | Symboles Cles |
|---------|--------------|
| Technology | AAPL, MSFT, GOOGL, META, NVDA |
| AI_Semiconductors | NVDA, AMD, AVGO, QCOM, ARM |
| Healthcare | UNH, JNJ, PFE, LLY, MRK |
| Biotech | AMGN, GILD, REGN, MRNA, VRTX |
| Finance | JPM, BAC, GS, MS, WFC |
| Energy | XOM, CVX, COP, SLB |
| Clean_Energy | ENPH, FSLR, PLUG, NEE |
| Consumer | AMZN, TSLA, HD, NKE, SBUX |
| Crypto_Adjacent | COIN, MSTR, SQ, RIOT, MARA |
| Defense | LMT, RTX, NOC, GD, BA |

### Themes Detectes par LLM

- AI_Revolution, Quantum_Computing
- Space_Economy, Obesity_Drugs
- Nuclear_Renaissance, EV_Transition
- Cybersecurity, Reshoring
- Crypto_Bull, Rate_Pivot

### Integration avec le Screener

```python
# Dans le screener, utiliser les symboles focus
from src.intelligence.trend_discovery import get_trend_discovery

discovery = await get_trend_discovery()
focus_symbols = discovery.get_focus_symbols()

# Scanner prioritairement ces symboles
alerts = market_screener.screen_multiple_stocks(focus_symbols)
```

## V4 - SYSTÈME AUTONOME AGENTIQUE (Dec 2024)

### Vision V4

Evolution vers un robot de trading **100% autonome** avec:
- Decouverte dynamique de stocks (tout NASDAQ/NYSE, pas juste SP500)
- Intelligence sociale (StockTwits + Reddit + X via Grok)
- Execution reelle via IBKR
- Auto-apprentissage via boucle nightly audit

### Architecture V4

```
src/
├── agents/                      # NOUVEAU - Systeme agentique
│   ├── guardrails.py           # Securites hard-coded (CRITIQUE)
│   ├── state.py                # Etat persistant agent
│   ├── orchestrator.py         # Cerveau central
│   └── tools/                  # Outils LLM (Phase 2)
├── intelligence/                # Intelligence augmentee
│   ├── trend_discovery.py      # Existant V3
│   ├── social_scanner.py       # NOUVEAU - StockTwits + Reddit
│   ├── grok_scanner.py         # NOUVEAU - X/Twitter via Grok API
│   └── stock_discovery.py      # NOUVEAU - Univers dynamique
└── execution/                   # NOUVEAU - Execution ordres
    └── ibkr_executor.py        # Connexion Interactive Brokers
```

### Guardrails - Securites Hard-Coded

**CRITIQUE: Ces limites ne peuvent PAS etre modifiees par l'IA**

```python
# Dans src/agents/guardrails.py
MAX_DAILY_LOSS_PCT = 0.03      # 3% perte/jour → Kill switch
MAX_POSITION_PCT = 0.10        # 10% max par position
MAX_DRAWDOWN_PCT = 0.15        # 15% drawdown → Pause 24h
MIN_POSITION_SIZE = 50         # 50€ minimum
MAX_TRADES_PER_DAY = 5         # 5 trades max/jour

# Mode hybride d'autonomie
AUTO_TRADE_THRESHOLD = 0.05    # < 5% capital = automatique
NOTIFY_THRESHOLD = 0.10        # 5-10% = notification
MANUAL_THRESHOLD = 0.10        # > 10% = validation manuelle

# Kill switch automatique
KILL_SWITCH_TRIGGERS = [
    "daily_loss >= 3%",
    "drawdown >= 15%",
    "consecutive_losses >= 3"
]
```

### Orchestrateur (MarketAgent)

Phases quotidiennes:
1. **DISCOVERY (06:00 ET)** - Scan social + decouverte stocks
2. **ANALYSIS (07:00 ET)** - Analyse tendances avec LLM
3. **TRADING (09:30-16:00 ET)** - Screening + execution
4. **AUDIT (20:00 ET)** - Boucle auto-apprentissage

```python
from src.agents.orchestrator import MarketAgent, get_market_agent

agent = await get_market_agent(capital=1500)
await agent.initialize()

# Demarrer le cycle quotidien
await agent.start()

# Ou executer une phase manuellement
await agent.run_discovery_phase()
await agent.run_trading_scan()
```

### Social Scanner

Scan StockTwits + Reddit ameliore:

```python
from src.intelligence.social_scanner import SocialScanner

scanner = SocialScanner()
await scanner.initialize()

result = await scanner.full_scan()
print(f"Trending: {result.get_top_symbols(10)}")
print(f"Bullish: {result.get_bullish_symbols(min_sentiment=0.3)}")
```

Subreddits surveilles:
- r/wallstreetbets, r/stocks, r/investing
- r/options, r/pennystocks, r/stockmarket
- r/ValueInvesting, r/Daytrading

### Grok Scanner (X/Twitter)

Analyse X via l'API Grok (~25$/mois):

```python
from src.intelligence.grok_scanner import GrokScanner

scanner = GrokScanner(api_key=os.getenv('GROK_API_KEY'))
await scanner.initialize()

# Analyse d'un symbole
insight = await scanner.analyze_symbol("NVDA")
print(f"Sentiment: {insight.sentiment} ({insight.sentiment_score:.2f})")

# Tendances financieres
trends = await scanner.search_financial_trends()
```

### Stock Discovery

Decouverte dynamique au-dela du SP500:

```python
from src.intelligence.stock_discovery import StockDiscovery

discovery = StockDiscovery()
await discovery.initialize()

# Rafraichir l'univers (NASDAQ + NYSE + AMEX)
await discovery.refresh_universe()  # ~5000+ stocks

# Obtenir les candidats pour screening
result = await discovery.get_screening_candidates(max_candidates=200)
symbols = [c.symbol for c in result.candidates]
```

### IBKR Executor

Execution d'ordres reels via Interactive Brokers:

```python
from src.execution.ibkr_executor import IBKRExecutor, OrderRequest, OrderAction, OrderType

executor = IBKRExecutor(port=7497)  # 7497=Paper, 7496=Live
await executor.connect()

# Passer un ordre
request = OrderRequest(
    symbol='AAPL',
    action=OrderAction.BUY,
    quantity=10,
    order_type=OrderType.LIMIT,
    limit_price=175.50,
    stop_loss=168.00
)
result = await executor.place_order(request)
```

**Prerequis IBKR:**
1. Compte IBKR (Pro ou Lite)
2. TWS ou IB Gateway installe
3. API activee (Edit > Global Configuration > API)
4. `pip install ib_insync`

### Configuration V4 (.env)

```bash
# Grok API (xAI) - ~25$/mois
GROK_API_KEY=xai-...

# StockTwits (optionnel)
STOCKTWITS_ACCESS_TOKEN=...

# IBKR Connection
IBKR_HOST=127.0.0.1
IBKR_PORT=7497          # 7497=Paper, 7496=Live
IBKR_CLIENT_ID=1

# Trading
TRADING_CAPITAL=1500    # Capital initial EUR
AUTONOMY_MODE=hybrid    # full, hybrid, manual
ENABLE_LIVE_TRADING=false
```

### Roadmap V4

| Phase | Duree | Contenu |
|-------|-------|---------|
| 1 | 2 sem | Core modules (guardrails, orchestrator, scanners) ✅ |
| 2 | 2 sem | LLM Tool Use (approve_order, reject_signal) |
| 3 | 2 sem | Nightly Auditor + learned_guidelines.json |
| 4 | 2 sem | Backtesting V4 + validation signaux |
| 5 | 4 sem | Paper trading → Live progressif |

### Capital Evolution

Progression prevue (si rentable apres 6 mois):
- Debut: **1500€**
- Phase 2: **3000€**
- Phase 3: **6000€**
- Cible: **15000€**

## Corrections Critiques V4 (29 Dec 2024)

### Bug Critique Corrige: Systeme d'Apprentissage

**Probleme**: La methode `get_closed_trades_today()` n'existait pas dans `StateManager`.
- L'orchestrator l'appelait a la ligne 1066 de `orchestrator.py`
- Cela faisait crasher la phase d'audit nocturne
- Consequence: **ZERO apprentissage** car TradeMemory jamais alimente

**Corrections appliquees dans `src/agents/state.py`:**

1. **TradeRecord ameliore** avec champs d'audit:
```python
@dataclass
class TradeRecord:
    # Champs existants...

    # NOUVEAUX champs pour l'apprentissage:
    entry_date: Optional[str] = None
    exit_date: Optional[str] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    pnl_pct: float = 0.0
    hold_days: int = 0
    exit_type: Optional[str] = None  # 'stop_loss', 'take_profit', etc.
    metadata: Dict[str, Any] = field(default_factory=dict)  # market_regime, vix, rsi

    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeRecord':
        """Reconstruction depuis dictionnaire"""
```

2. **AgentState** - nouveau champ:
```python
trade_history: List[Dict] = field(default_factory=list)  # 5000 trades max
```

3. **record_trade()** - stockage dans trade_history:
```python
def record_trade(self, trade: TradeRecord):
    # ... stats existantes ...

    # NOUVEAU: Stockage complet pour apprentissage
    self.state.trade_history.append(trade.to_dict())

    # Limite a 5000 trades (suffisant pour ML)
    if len(self.state.trade_history) > 5000:
        self.state.trade_history = self.state.trade_history[-5000:]
```

4. **NOUVELLES methodes ajoutees**:
```python
def get_closed_trades_today(self) -> List[TradeRecord]:
    """Trades fermes aujourd'hui (pour audit nocturne)"""

def get_trades_by_date_range(self, start_date, end_date) -> List[TradeRecord]:
    """Trades dans une plage de dates"""

def get_trade_history(self, limit=100) -> List[TradeRecord]:
    """Derniers N trades"""
```

### Architecture LLM Clarifiee

**Separation des responsabilites:**
- **Grok (xAI)**: Sentiment reseaux sociaux (X/Twitter UNIQUEMENT)
  - Via `src/intelligence/grok_scanner.py`
- **Gemini (OpenRouter)**: Analyse news financieres + valuations
  - Via `src/agents/pillars/news_pillar.py`
  - Via `src/intelligence/trend_discovery.py`

**Correction dans `news_pillar.py`:**
- Supprime le fallback Grok (qui n'avait pas sa place)
- Utilise UNIQUEMENT OpenRouter/Gemini pour les news

### Securite: .env.example Nettoye

Cles API reelles remplacees par placeholders:
- `GROK_API_KEY=xai-your_grok_api_key_here`
- `OPENROUTER_API_KEY=sk-or-v1-your_openrouter_api_key_here`
- `NEWSAPI_KEY=your_newsapi_key_here`
- `ALPHAVANTAGE_KEY=your_alphavantage_key_here`

### Anti-Repetition des Scans

**Mecanisme existant dans `analysis_store.py`:**
```python
# Cooldown par defaut: 1 heure
cooldown_hours = getattr(self.config, 'analysis_cooldown_hours', 1)

# Filtrer les symboles analyses recemment
fresh_symbols = self.analysis_store.get_unanalyzed_symbols(symbols, hours=cooldown_hours)

# Si tous analyses: rafraichir les "stales" (> 24h)
stale_symbols = self.analysis_store.get_symbols_needing_reanalysis(hours=24)
```

**Persistance:**
- Analyses stockees dans SQLite: `data/analyses.db`
- Survit aux redemarrages

## Corrections V4.4 (29 Dec 2024) - Bug Pilliers 0/25

### Probleme Critique: Pilliers Retournant 0/25

**Symptomes observes:**
- Tous les 4 pilliers affichaient 0/25 dans le dashboard
- Scores avec precision flottante bizarre (4.925000000001/100)
- Aucun log detaille pour comprendre ce qui se passait
- Watchlist avec valeurs bizarres (SCHD, LLY)

**Cause racine:**
L'orchestrator appelait `reasoning_engine.analyze(symbol)` **SANS fournir les donnees OHLCV**.
Le Technical Pillar retournait donc 0 car `df is None`.

### Corrections Appliquees

**1. Orchestrator - Fetch OHLCV avant analyse (`src/agents/orchestrator.py`):**

```python
# AVANT (bug):
reasoning_result = await self.reasoning_engine.analyze(symbol)

# APRES (corrige):
df = self.market_data_fetcher.get_historical_data(
    symbol=symbol,
    period='6mo',  # 6 mois pour avoir >50 data points
    interval='1d'
)
if df is None or len(df) < 50:
    results["rejected_symbols"][symbol] = f"Insufficient historical data"
    continue
reasoning_result = await self.reasoning_engine.analyze(symbol, df=df)
```

**2. Ajout MarketDataFetcher dans l'Orchestrator:**
```python
from src.data.market_data import MarketDataFetcher

# Dans __init__:
self.market_data_fetcher = MarketDataFetcher()
```

**3. Arrondi des Scores (precision flottante):**
Tous les scores sont maintenant arrondis a 2 decimales:
```python
"total_score": round(reasoning_result.total_score, 2),
"technical": round(reasoning_result.technical_score.score, 2),
# etc.
```

**4. Logs Verbeux dans les 4 Pilliers:**

Chaque pillier affiche maintenant son analyse en detail:
```
[TECHNICAL] AAPL: Analyzing 126 data points...
[TECHNICAL] AAPL: Score=42.5/100 (bullish) | Trend=55 Momentum=38 Volume=30 Volatility=25
[FUNDAMENTAL] AAPL: Score=35.2/100 (good) | Valuation=40 Growth=45 Profit=25 Health=30 | Quality=92%
[SENTIMENT] AAPL: Score=28.0/100 (bullish) | Sources: twitter | Quality=50%
[NEWS] AAPL: Score=15.5/100 (neutral) | Headlines=8 | Quality=80%
```

### Fichiers Modifies
- `src/agents/orchestrator.py` - Ajout fetch OHLCV + arrondi scores
- `src/agents/pillars/technical_pillar.py` - Logs verbeux
- `src/agents/pillars/fundamental_pillar.py` - Logs verbeux
- `src/agents/pillars/sentiment_pillar.py` - Logs verbeux
- `src/agents/pillars/news_pillar.py` - Logs verbeux

### Verification
Apres ces corrections, les pilliers devraient afficher des scores reels:
- Technical: 0-100 (basé sur EMA, MACD, RSI, Volume, etc.)
- Fundamental: 0-100 (basé sur P/E, ROE, marges, etc.)
- Sentiment: 0-100 (basé sur X/Twitter via Grok)
- News: 0-100 (basé sur headlines via Gemini/OpenRouter)
- Scans du soir disponibles le matin

### Script de Test

Nouveau script: `scripts/test_learning_system.py`

```bash
python scripts/test_learning_system.py
```

Teste:
1. StateManager - stockage trades
2. AnalysisStore - persistance + anti-repetition
3. Cooldown Logic - verification skip
4. Orchestrator - integration complete

### Etat du Systeme Apres Corrections

| Composant | Avant | Apres |
|-----------|-------|-------|
| get_closed_trades_today() | MANQUANT (crash) | OK |
| trade_history | Non stocke | 5000 trades max |
| Audit nocturne | Crashait | Operationnel |
| TradeMemory (RAG) | Jamais alimente | Pret a recevoir |
| Anti-repetition | Existait | Verifie OK |
| Scans persistants | OK | OK |

**Le systeme d'apprentissage est maintenant fonctionnel.**

## Corrections V4.7 (29 Dec 2024) - Visualisation Neuronale + Analyse Hors-Marché

### Nouvelles Fonctionnalités

#### 1. Analyse Complète Même Quand le Marché Est Fermé

**Problème**: Le bot ne faisait que la découverte (Discovery) quand le marché était fermé, sans lancer l'analyse 4 piliers.

**Solution**: Mode AFTER_HOURS/WEEKEND lance maintenant le scan complet:

```python
# Dans webapp.py - bot loop pour AFTER_HOURS/WEEKEND
result = await state.agent.run_trading_scan(execute_trades=False)
```

**Nouvelle signature de `run_trading_scan`**:
```python
async def run_trading_scan(self, execute_trades: bool = True) -> Dict:
    """
    Args:
        execute_trades: Si False, analyse seulement sans exécuter de trades.
                       Utile pour préparer l'ouverture du marché.
    """
```

**Comportement**:
- `execute_trades=True` (défaut): Comportement normal, exécute les trades
- `execute_trades=False`: Analyse complète, stocke les signaux comme "PREPARED_BUY"
- Les signaux sont stockés dans `state.premarket_queue` pour exécution à l'ouverture

#### 2. Visualisation Réseau Neuronal sur /reasoning

**Nouvelle section**: "🧠 Réseau de Raisonnement Global"

Affiche le flux de décision comme un réseau neuronal vertical:
- **Layer 1 - Sources**: Reddit/StockTwits, Grok/X, News API, Volume
- **Layer 2 - 4 Piliers**: Technical, Fundamental, Sentiment, News
- **Layer 3 - Agrégation**: Score total
- **Layer 4 - Décisions**: BUY, WATCH, REJECT

**Activation des noeuds**:
- Chaque noeud affiche son % d'activation basé sur les journeys analysés
- Plus le noeud est activé, plus il est opaque
- Vue agrégée de TOUS les symboles analysés

**CSS ajouté**:
```css
.neural-graph-vertical {
    display: flex;
    flex-direction: column;
    align-items: center;
}
.neural-connector {
    height: 30px;
    background: linear-gradient(180deg, #8b5cf6, #6366f1);
}
```

#### 3. Section Apprentissage & Vectorisation sur /reasoning

**Nouvelle section**: "📚 Apprentissage & Vectorisation"

Affiche les statistiques du système d'apprentissage:

**Métriques affichées**:
- Vecteurs Stockés (80 dimensions chacun)
- Trades Gagnants (avec P&L moyen)
- Trades Perdants (avec P&L moyen)
- Win Rate global

**Structure du Vecteur (80 dimensions)**:
| Range | Nom | Description |
|-------|-----|-------------|
| [0-19] | Sources | Embeddings des sources (Reddit, Grok, etc.) |
| [20-39] | Piliers | Embeddings des 4 piliers |
| [40-59] | Décision | Embeddings de la décision finale |
| [60-79] | Structure | Métriques du graphe (chemins, convergence) |

**États d'Optimisation**:
1. **Phase de collecte** (<10 vecteurs): Collecte des données
2. **En attente d'outcomes** (<5 trades clôturés): Besoin de feedback
3. **Optimisation active**: Ajustement des poids basé sur les résultats

### Fichiers Modifiés

- `webapp.py`:
  - Ajout analyse hors-marché dans le bot loop (lignes 1520-1610)
  - Section "Réseau de Raisonnement Global" (lignes 2743-2906)
  - Section "Apprentissage & Vectorisation" (lignes 2908-3026)
  - CSS `.neural-graph-vertical` et `.neural-connector` (lignes 3259-3289)

- `src/agents/orchestrator.py`:
  - Paramètre `execute_trades=True` ajouté à `run_trading_scan()`
  - Condition skip d'exécution si `execute_trades=False`

### Utilisation

**Lancer le bot en mode weekend/after-hours**:
Le bot analysera automatiquement les symboles et préparera les trades pour l'ouverture.

**Voir le réseau neuronal**:
1. Aller sur http://localhost:8080/reasoning
2. Scroller jusqu'à "🧠 Réseau de Raisonnement Global"
3. Observer l'activation de chaque noeud

**Voir les stats d'apprentissage**:
1. Aller sur http://localhost:8080/reasoning
2. Scroller jusqu'à "📚 Apprentissage & Vectorisation"
3. Observer le Win Rate et les vecteurs stockés
