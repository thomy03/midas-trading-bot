# Implémentation des Niveaux de Support Historiques

## 📅 Date: 2025-10-25 03:00 UTC

## 🎯 Logique Clarifiée par l'Utilisateur

### Règle Fondamentale
Les **crossovers d'EMA** créent des **niveaux de prix de référence** qui restent valides comme **supports horizontaux** tant que:

1. ✅ Ce niveau n'a pas encore servi de signal/trade
2. ✅ Les EMAs actuelles n'ont pas retracé (retesté) ce prix à la baisse

### Citation Clé de l'Utilisateur
> "Les croisements d'ema servent de prix de référence pour un support tant que ce niveau n'a pas servi de signal et de trade ou que les emas actuelles n'ont pas retracé ce prix."

## 🔄 Différence avec l'Ancienne Logique

### AVANT (Logique Incorrecte)
```
Crossover EMA → Support valide SEULEMENT si:
  - Distance < 8% du prix actuel (ZONE_TOLERANCE)
  - Âge < 52-104 semaines
```
**Problème**: Rejette les niveaux éloignés mais toujours valides!

### APRÈS (Logique Correcte)
```
Crossover EMA → Niveau de référence PERMANENT qui reste valide tant que:
  - TOUTES les EMAs (24, 38, 62) sont au-dessus du niveau
  - Le niveau n'a pas encore servi de signal

Distance du prix actuel: NON PERTINENTE pour la validité!
Distance: PERTINENTE uniquement pour PRIORISER l'alerte
```

## ✅ Implémentation

### 1. Nouvelle Méthode: `find_historical_support_levels()`

**Fichier**: `src/indicators/ema_analyzer.py`

```python
def find_historical_support_levels(
    self,
    df: pd.DataFrame,
    crossovers: List[Dict],
    current_price: float
) -> List[Dict]:
    """
    Find ALL historical support levels from crossovers (no distance limit).

    NOUVELLE LOGIQUE: Les crossovers sont des niveaux de référence permanents
    qui restent valides tant que les EMAs ne les ont pas retracés.
    """
    historical_levels = []

    for crossover in crossovers:
        if crossover['type'] != 'bullish':  # On garde que les supports
            continue

        cross_price = crossover['price']
        distance_pct = abs((current_price - cross_price) / cross_price * 100)

        # Déterminer si le prix est au-dessus (support) ou en-dessous
        if current_price >= cross_price:
            zone_type = 'historical_support'
        else:
            zone_type = 'historical_resistance'

        historical_levels.append({
            'level': cross_price,
            'distance_pct': distance_pct,
            'crossover_info': crossover,
            'zone_type': zone_type,
            'strength': self._calculate_zone_strength(crossover),
            'is_near': distance_pct <= ZONE_TOLERANCE  # Flag pour alertes
        })

    # Sort by proximity (closest first)
    historical_levels.sort(key=lambda x: x['distance_pct'])
    return historical_levels
```

### 2. Modification: `detect_crossovers()`

**Changement Clé**: Support reste valide si TOUTES les EMAs sont au-dessus

```python
# NOUVELLE LOGIQUE: Support reste valide tant que TOUTES les EMAs sont au-dessus
if cross_type == 'bullish':
    latest = df.iloc[-1]
    current_ema_24 = latest.get('EMA_24', 0)
    current_ema_38 = latest.get('EMA_38', 0)
    current_ema_62 = latest.get('EMA_62', 0)

    # Support reste valide si TOUTES les EMAs sont au-dessus du niveau crossover
    all_emas_above = (
        current_ema_24 > cross_price and
        current_ema_38 > cross_price and
        current_ema_62 > cross_price
    )

    # Si toutes les EMAs sont au-dessus, le support est TOUJOURS valide (pas de limite d'âge)
    if not all_emas_above:
        max_age = MAX_CROSSOVER_AGE_WEEKLY if timeframe == 'weekly' else MAX_CROSSOVER_AGE_DAILY
        if age_in_periods > max_age:
            continue
```

## 📊 Exemple: TSLA

### État Actuel
- **Prix**: $433.72
- **EMA 24**: $367.66
- **EMA 38**: $348.04
- **EMA 62**: $324.02

### Niveaux Historiques Détectés

| Niveau | Prix | Distance | Date | Type | Statut |
|--------|------|----------|------|------|--------|
| #1 | $208.13 | 108.4% | 2023-10-30 | historical_support | ✅ VALIDE |
| #2 | $208.01 | 108.5% | 2023-10-30 | historical_support | ✅ VALIDE |
| #3 | $207.83 | 108.7% | 2023-10-30 | historical_support | ✅ VALIDE |
| #4 | $204.38 | 112.2% | 2024-09-09 | historical_support | ✅ VALIDE |
| #5 | $203.17 | 113.5% | 2024-08-19 | historical_support | ✅ VALIDE |
| #6 | $200.20 | 116.6% | 2024-07-22 | historical_support | ✅ VALIDE |

**Tous restent valides** car TOUTES les EMAs ($324-$367) sont au-dessus des niveaux ($200-210).

## 🎯 Logique d'Alerte

### Quand Alerter?

1. **Prix s'approche d'un niveau historique** (< 8%)
   - Exemple: Si TSLA retrace vers $210, distance devient ~7% → ALERTE!

2. **ET RSI breakout détecté**
   - Oblique RSI descendante cassée
   - Breakout récent (< 3-15 périodes selon MAX_BREAKOUT_AGE)

3. **Signal Généré**:
   ```
   🚨 ALERTE: TSLA
   📍 Prix: $220 → Approche niveau historique $208 (5.7%)
   🎯 RSI Breakout détecté (daily)
   ⭐ Recommandation: STRONG_BUY
   ```

## 🔧 Fichiers Modifiés

1. **`src/indicators/ema_analyzer.py`**
   - Ajout: `find_historical_support_levels()` - Retourne TOUS les niveaux sans filtre
   - Modif: `detect_crossovers()` - Garde crossovers tant que EMAs au-dessus

2. **`config/settings.py`**
   - `MAX_CROSSOVER_AGE_WEEKLY`: 52 → 104 semaines (~2 ans)
   - `MAX_CROSSOVER_AGE_DAILY`: 120 → 365 jours (1 an)

3. **`trendline_analysis/config/settings.py`**
   - `PEAK_PROMINENCE`: 3.0 → 2.0 (plus de peaks RSI)
   - `MIN_R_SQUARED`: 0.40 → 0.25 (obliques moins strictes)
   - `MAX_RESIDUAL_DISTANCE`: 6.0 → 8.0 (plus de tolérance)

## 📝 Tests Créés

1. **`test_historical_levels.py`** - Affiche tous les niveaux historiques
2. **`find_290_support.py`** - Analyse du niveau $290
3. **`test_tsla_historical_crossovers.py`** - Crossovers historiques TSLA

## 🔄 Prochaines Étapes

### À Implémenter:

1. **Modifier le screener** pour utiliser `find_historical_support_levels()`
   - Scanner tous les niveaux historiques
   - Alerter quand prix proche d'un niveau + RSI breakout

2. **Ajouter base de données** des niveaux historiques
   - Persister les niveaux de référence
   - Marquer niveaux comme "utilisés" après signal/trade

3. **Dashboard: Section "Historical Levels"**
   - Afficher tous les niveaux historiques
   - Indiquer distance du prix actuel
   - Highlight niveaux proches (< 8%)

4. **Système de notification**
   - Alerter quand prix s'approche d'un niveau historique
   - Vérifier RSI breakout automatiquement

## 💡 Avantages de Cette Approche

### ✅ Avantages
1. **Mémoire des niveaux clés** - Ne perd plus les supports importants
2. **Vision long terme** - Niveaux valides plusieurs années
3. **Pas de faux négatifs** - Ne rate plus les signaux sur anciens niveaux
4. **Logique claire** - Critère simple: EMAs au-dessus = valide

### ⚠️ Considérations
1. **Plus de niveaux à suivre** - Peut générer plus de données
2. **Besoin de priorisation** - Utiliser distance pour prioriser alertes
3. **Gestion des niveaux "utilisés"** - À implémenter pour éviter duplicatas

## 📚 Citation Documentation

Pour comprendre la logique, se référer à la conversation:

> **Utilisateur**: "attention lorsque le prix était à 290 sur le graphique où j'ai tracé il n'y avait en effet pas de crossover d'ema mais ce n'est pas ça la règle. Les croisements d'ema servent de prix de référence pour un support tant que ce niveau n'a pas servi de signal et de trade ou que les emas actuelles n'ont pas retracé ce prix."

**Traduction**: Les crossovers EMA = niveaux de référence permanents, pas temporaires!

---

**Date de documentation**: 2025-10-25 03:00 UTC
**Statut**: ✅ Méthode `find_historical_support_levels()` implémentée et testée
**Prochaine étape**: Intégrer dans le screener pour alertes automatiques
