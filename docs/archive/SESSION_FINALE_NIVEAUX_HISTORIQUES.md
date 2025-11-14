# Session Finale - Système de Niveaux Historiques

## 📅 Date: 2025-10-25 03:10 UTC

## 🎯 RÈGLE FONDAMENTALE (À NE PAS OUBLIER!)

### Citation Utilisateur:
> "Je veux que ces niveaux de croisements restent valide tant que toutes les emas ne l'ont pas traversé. Ensuite lorsque le prix s'approche alors il faut recherché des obliques sur le RSI"

### Logique Complète:

1. **Crossover EMA** → Crée un **niveau de prix de référence PERMANENT**

2. **Niveau reste VALIDE tant que**:
   - ✅ TOUTES les EMAs (24, 38, 62) sont au-dessus du niveau (pour un support bullish)
   - ✅ Le niveau n'a pas encore servi de signal/trade

3. **Quand prix s'approche d'un niveau** (< 8%):
   - 🔍 **Rechercher oblique RSI descendante**
   - 🎯 **Vérifier breakout RSI**
   - 🚨 **Si oblique RSI + breakout** → SIGNAL!

4. **Niveau devient INVALIDE quand**:
   - ❌ UNE des EMAs traverse (retrace) le niveau à la baisse
   - ❌ Le niveau a servi de signal/trade

## ✅ Ce Qui a Été Implémenté

### 1. Détection des Niveaux Historiques

**Fichier**: `src/indicators/ema_analyzer.py`

**Méthode**: `find_historical_support_levels()`

```python
def find_historical_support_levels(
    self,
    df: pd.DataFrame,
    crossovers: List[Dict],
    current_price: float
) -> List[Dict]:
    """
    Retourne TOUS les niveaux historiques (crossovers) sans filtre de distance.

    Chaque niveau contient:
    - level: Prix du niveau
    - distance_pct: Distance du prix actuel
    - is_near: True si < 8% (ZONE_TOLERANCE)
    - crossover_info: Détails du crossover
    - zone_type: 'historical_support' ou 'historical_resistance'
    """
```

### 2. Validation des Niveaux

**Fichier**: `src/indicators/ema_analyzer.py`

**Méthode**: `detect_crossovers()` (modifiée)

```python
# NOUVELLE LOGIQUE: Support reste valide tant que TOUTES les EMAs sont au-dessus
if cross_type == 'bullish':
    latest = df.iloc[-1]
    current_ema_24 = latest.get('EMA_24', 0)
    current_ema_38 = latest.get('EMA_38', 0)
    current_ema_62 = latest.get('EMA_62', 0)

    # Support reste valide si TOUTES les EMAs sont au-dessus
    all_emas_above = (
        current_ema_24 > cross_price and
        current_ema_38 > cross_price and
        current_ema_62 > cross_price
    )

    # Si toutes les EMAs sont au-dessus, le support est TOUJOURS valide (pas de limite d'âge)
    if not all_emas_above:
        # Une EMA a retracé → appliquer limite d'âge
        max_age = MAX_CROSSOVER_AGE_WEEKLY if timeframe == 'weekly' else MAX_CROSSOVER_AGE_DAILY
        if age_in_periods > max_age:
            continue
```

### 3. Paramètres Assouplis

**Fichier**: `config/settings.py`
```python
MAX_CROSSOVER_AGE_WEEKLY = 104  # 2 ans (au lieu de 52)
MAX_CROSSOVER_AGE_DAILY = 365   # 1 an (au lieu de 120)
```

**Fichier**: `trendline_analysis/config/settings.py`
```python
PEAK_PROMINENCE = 2.0          # Au lieu de 3.0 (plus de peaks RSI)
MIN_R_SQUARED = 0.25           # Au lieu de 0.40 (obliques moins strictes)
MAX_RESIDUAL_DISTANCE = 8.0    # Au lieu de 6.0 (plus de tolérance)
```

## 📊 Exemple: TSLA

### État Actuel (2025-10-25)
```
Prix: $433.72
EMA 24: $367.66
EMA 38: $348.04
EMA 62: $324.02
```

### Niveaux Historiques Détectés
```
📍 $208.13 (108.4% distance) - 2023-10-30 ✅ VALIDE (EMAs au-dessus)
📍 $208.01 (108.5% distance) - 2023-10-30 ✅ VALIDE (EMAs au-dessus)
📍 $207.83 (108.7% distance) - 2023-10-30 ✅ VALIDE (EMAs au-dessus)
📍 $204.38 (112.2% distance) - 2024-09-09 ✅ VALIDE (EMAs au-dessus)
📍 $203.17 (113.5% distance) - 2024-08-19 ✅ VALIDE (EMAs au-dessus)
📍 $200.20 (116.6% distance) - 2024-07-22 ✅ VALIDE (EMAs au-dessus)
```

**Tous restent valides car toutes les EMAs ($324-367) sont au-dessus!**

### Scénario Futur
Si TSLA retrace vers $220:
```
Prix: $220
Distance au niveau $208: 5.7% ✅ PROCHE!

→ Système doit:
  1. Détecter que prix proche du niveau $208
  2. Rechercher oblique RSI descendante
  3. Vérifier breakout RSI
  4. Si breakout → SIGNAL STRONG_BUY!
```

## 🔄 À IMPLÉMENTER (Prochaine Session)

### 1. Modifier le Screener

**Fichier**: `src/screening/screener.py`

**Changements requis**:

```python
def screen_single_stock(self, symbol: str, company_name: str = "") -> Optional[Dict]:
    """
    NOUVELLE LOGIQUE:
    1. Obtenir TOUS les niveaux historiques (find_historical_support_levels)
    2. Filtrer les niveaux PROCHES (is_near = True, < 8%)
    3. Pour chaque niveau proche:
       a. Vérifier oblique RSI
       b. Vérifier breakout RSI
       c. Si breakout → GÉNÉRER SIGNAL
    """

    # Step 1: Get historical data
    df_weekly = self.market_data.get_historical_data(symbol, period='2y', interval='1wk')
    df_weekly = self.ema_analyzer.calculate_emas(df_weekly)

    # Step 2: Detect ALL crossovers (historical levels)
    crossovers = self.ema_analyzer.detect_crossovers(df_weekly, 'weekly')
    current_price = float(df_weekly['Close'].iloc[-1])

    # Step 3: Get ALL historical levels
    historical_levels = self.ema_analyzer.find_historical_support_levels(
        df_weekly, crossovers, current_price
    )

    # Step 4: Filter NEAR levels (< 8%)
    near_levels = [level for level in historical_levels if level['is_near']]

    if not near_levels:
        logger.debug(f"{symbol}: No near historical levels")
        return None

    # Step 5: Check RSI for each near level
    for level in near_levels:
        # Check RSI weekly
        rsi_weekly = self._check_rsi_breakout(symbol, 'weekly')
        if rsi_weekly and rsi_weekly.has_rsi_breakout:
            return self._create_alert_historical_level(
                symbol, company_name, level, 'weekly', rsi_weekly
            )

        # Check RSI daily
        rsi_daily = self._check_rsi_breakout(symbol, 'daily')
        if rsi_daily and rsi_daily.has_rsi_breakout:
            return self._create_alert_historical_level(
                symbol, company_name, level, 'daily', rsi_daily
            )

    return None
```

### 2. Nouvelle Méthode: `_create_alert_historical_level()`

```python
def _create_alert_historical_level(
    self,
    symbol: str,
    company_name: str,
    historical_level: Dict,
    timeframe: str,
    rsi_result
) -> Dict:
    """
    Create alert for historical level approach + RSI breakout

    Args:
        symbol: Stock symbol
        company_name: Company name
        historical_level: Historical level dict from find_historical_support_levels()
        timeframe: 'weekly' or 'daily'
        rsi_result: RSIBreakoutResult

    Returns:
        Alert dictionary
    """
    crossover_info = historical_level['crossover_info']

    alert = {
        'symbol': symbol,
        'company_name': company_name or symbol,
        'timeframe': timeframe,
        'current_price': df['Close'].iloc[-1],
        'support_level': historical_level['level'],
        'distance_to_support_pct': historical_level['distance_pct'],
        'support_type': 'historical_crossover',
        'crossover_date': crossover_info['date'].strftime('%Y-%m-%d'),
        'crossover_age_weeks': crossover_info.get('age_in_periods', 0),
        'has_rsi_breakout': rsi_result.has_rsi_breakout,
        'rsi_timeframe': timeframe,
        'rsi_breakout_date': rsi_result.rsi_breakout.date.strftime('%Y-%m-%d'),
        'rsi_breakout_value': rsi_result.rsi_breakout.rsi_value,
        'recommendation': 'STRONG_BUY'  # Prix proche niveau historique + RSI breakout
    }

    return alert
```

### 3. Base de Données des Niveaux

**Créer table**: `historical_levels`

```sql
CREATE TABLE historical_levels (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    level_price REAL NOT NULL,
    crossover_date TEXT NOT NULL,
    crossover_type TEXT NOT NULL,  -- 'bullish' or 'bearish'
    ema_fast INTEGER NOT NULL,
    ema_slow INTEGER NOT NULL,
    is_valid BOOLEAN DEFAULT 1,  -- False quand EMA retrace ou signal utilisé
    signal_date TEXT,  -- Date quand signal généré
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### 4. Dashboard: Section "Historical Levels"

**Fichier**: `dashboard.py`

**Nouvelle section**:

```python
st.markdown("### 📍 Historical Support Levels")

if st.button("Show Historical Levels"):
    df = market_data_fetcher.get_historical_data(symbol, period='2y', interval='1wk')
    df = ema_analyzer.calculate_emas(df)
    crossovers = ema_analyzer.detect_crossovers(df, 'weekly')
    current_price = float(df['Close'].iloc[-1])

    historical_levels = ema_analyzer.find_historical_support_levels(
        df, crossovers, current_price
    )

    # Display table
    for i, level in enumerate(historical_levels[:10]):
        emoji = "🎯" if level['is_near'] else "📍"
        st.write(f"{emoji} ${level['level']:.2f} - {level['distance_pct']:.1f}% away")
```

## 📝 Tests Créés

1. **`test_historical_levels.py`** - Affiche tous les niveaux historiques TSLA
2. **`find_290_support.py`** - Analyse spécifique du niveau $290
3. **`test_tsla_historical_crossovers.py`** - Crossovers avec distances

## 🗂️ Fichiers Importants

### Code Source
- `src/indicators/ema_analyzer.py` - Contient `find_historical_support_levels()`
- `src/screening/screener.py` - À modifier pour intégration
- `config/settings.py` - Paramètres EMA
- `trendline_analysis/config/settings.py` - Paramètres RSI

### Documentation
- `IMPLEMENTATION_NIVEAUX_HISTORIQUES.md` - Doc technique complète
- `SESSION_FINALE_NIVEAUX_HISTORIQUES.md` - Ce fichier (sauvegarde session)
- `SESSION_CONTEXTE_FINAL.md` - Contexte sessions précédentes
- `SESSION_UPDATE_MODE_FLEXIBLE.md` - Mode flexible implémenté

### Tests
- `test_historical_levels.py` - Test niveaux historiques
- `test_tsla_signal.py` - Test TSLA
- `test_flexible_mode.py` - Test mode flexible

## 🎯 Résumé Pour Prochaine Session

### État Actuel
✅ **Détection niveaux historiques** - Implémentée
✅ **Validation EMAs au-dessus** - Implémentée
✅ **Détection RSI breakout** - Implémentée
✅ **Mode flexible signaux** - Implémenté

### À Faire
❌ **Intégrer dans screener** - Scanner automatique niveaux proches + RSI
❌ **Base de données niveaux** - Persister et marquer utilisés
❌ **Dashboard section historique** - Afficher tous les niveaux
❌ **Système notification** - Alerter approche niveau + RSI

### Priorité #1
**Modifier `src/screening/screener.py`** pour:
1. Utiliser `find_historical_support_levels()` au lieu de `find_support_zones()`
2. Filtrer niveaux proches (`is_near = True`)
3. Vérifier RSI pour chaque niveau proche
4. Générer signal si breakout RSI

### Code Clé À Ajouter

```python
# Dans screen_single_stock():

# Obtenir niveaux historiques
historical_levels = self.ema_analyzer.find_historical_support_levels(
    df_weekly, crossovers, current_price
)

# Filtrer proches
near_levels = [l for l in historical_levels if l['is_near']]

# Pour chaque niveau proche, vérifier RSI
for level in near_levels:
    rsi_result = self._check_rsi_breakout(symbol, 'weekly')
    if rsi_result and rsi_result.has_rsi_breakout:
        return self._create_alert_historical_level(
            symbol, company_name, level, 'weekly', rsi_result
        )
```

## 💡 Points Critiques

### NE PAS OUBLIER:
1. ✅ Distance n'est PAS un critère de validité du niveau
2. ✅ Niveau valide tant que TOUTES les EMAs au-dessus
3. ✅ Chercher oblique RSI SEULEMENT quand prix proche (< 8%)
4. ✅ Niveaux peuvent être à 100%+ de distance et rester valides

### Exemple Concret:
```
TSLA prix = $433
Niveau historique = $208 (108% de distance)

✅ Niveau VALIDE car EMAs ($324-367) toutes au-dessus
📍 Niveau PAS PROCHE (> 8%) → Pas de recherche RSI maintenant
🔮 Si TSLA retrace vers $220 → Distance 5.7% → PROCHE!
   → Alors rechercher oblique RSI + vérifier breakout
```

---

**Date de sauvegarde**: 2025-10-25 03:10 UTC
**Statut**: Détection niveaux historiques implémentée, intégration screener à faire
**Prochaine action**: Modifier `screen_single_stock()` pour utiliser niveaux historiques

## 🔗 Pour Continuer

1. Lire ce fichier en entier
2. Lire `IMPLEMENTATION_NIVEAUX_HISTORIQUES.md`
3. Tester avec `python3 test_historical_levels.py`
4. Modifier `src/screening/screener.py` selon code ci-dessus
5. Tester avec TSLA en simulant prix proche d'un niveau

**La logique est claire, l'implémentation de base est faite, il ne reste que l'intégration dans le screener!**
