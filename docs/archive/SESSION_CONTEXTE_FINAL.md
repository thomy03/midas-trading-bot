# Contexte de Session - Système de Trading Intégré EMA + RSI

## 📅 Date: 2025-10-25

## 🎯 Objectif Principal
Créer un système de trading qui combine:
1. **Screener EMA** (détection via crossovers et zones de support)
2. **RSI Breakout** (validation par cassure d'oblique RSI)

## ✅ Ce qui a été Accompli

### 1. Simplification du Système Trendline
- ❌ **Supprimé**: Dual confirmation (RSI + Prix)
- ✅ **Gardé**: Uniquement RSI breakout
- **Fichier créé**: `trendline_analysis/core/rsi_breakout_analyzer.py`
  - Analyse RSI uniquement (pas de price trendline)
  - Détecte obliques RSI descendantes
  - Détecte breakouts (cassures d'obliques)

### 2. Intégration dans le Screener EMA
- **Fichier modifié**: `src/screening/screener.py`
- **Nouvelle logique cascade**:
  ```
  Signal EMA Weekly → Check RSI Weekly → Check RSI Daily
  Signal EMA Daily (si EMAs weekly alignées) → Check RSI Daily
  ```
- **Méthodes ajoutées**:
  - `_check_rsi_breakout()`: Vérifie breakout RSI sur un timeframe
  - `_create_alert()`: Modifié pour inclure infos RSI
  - `_get_recommendation()`: Priorise signal RSI si présent

### 3. Assouplissement des Critères
- **Distance au support**: 5% → 8% (`config/settings.py`)
- **Seuils de recommandation** (`src/screening/screener.py`):
  - STRONG_BUY: ≤2% (au lieu de 1%)
  - BUY: ≤4% (au lieu de 2%)
  - WATCH: ≤8% (au lieu de 3.5%)

### 4. Détection de Rebond sur EMA (NOUVEAU!)
- **Fichier modifié**: `src/indicators/ema_analyzer.py`
- **Nouvelle méthode**: `find_ema_support_levels()`
  - Détecte quand le prix rebondit sur une EMA
  - **Fonctionne SANS crossover récent**
  - Vérifie que l'EMA agit comme support (prix au-dessus)
- **Modification**: `find_support_zones()`
  - Combine supports de crossovers + supports EMA

### 5. Dashboard Amélioré
- **Fichier modifié**: `dashboard.py`
- **TOUJOURS affiche un graphique**, même sans signal
- **Affiche**:
  - 📈 Prix avec EMAs (24, 38, 62)
  - 📊 RSI avec oblique (si détectée)
  - ⭐ Marqueur de breakout (étoile verte)
  - ✅ Résumé de l'analyse RSI

## 🔧 Fichiers Créés
1. `trendline_analysis/core/rsi_breakout_analyzer.py` - Analyseur RSI simplifié
2. `test_integrated_ema_rsi_screener.py` - Test du système intégré
3. `test_simple_screening.py` - Test simple avec détails
4. `find_active_signals.py` - Scanner de signaux actifs
5. `diagnostic_aapl.py` - Diagnostic détaillé pour AAPL
6. `test_tsla_signal.py` - Test TSLA avec nouvelle logique

## 🔧 Fichiers Modifiés
1. `config/settings.py` - ZONE_TOLERANCE: 5.0 → 8.0
2. `src/screening/screener.py`:
   - Import RSIBreakoutAnalyzer
   - Ajout `_check_rsi_breakout()`
   - Modification `screen_single_stock()` avec cascade
   - Modification `_create_alert()` avec infos RSI
   - Modification `_get_recommendation()` avec priorité RSI
3. `src/indicators/ema_analyzer.py`:
   - Ajout `find_ema_support_levels()`
   - Modification `find_support_zones()`
   - Modification `analyze_stock()` (ne retourne plus None sans crossover)
4. `dashboard.py`:
   - Section "🔍 Screening" complètement refaite
   - Affiche graphique même sans signal
   - Graphique avec Prix + EMAs + RSI + Oblique RSI

## ⚠️ Problème Identifié (TSLA)

### Cas d'usage: TSLA
Sur le screenshot fourni, on voit:
- ✅ Oblique RSI descendante (ligne verte)
- ✅ Breakout RSI (étoile verte)
- ✅ Support de prix ~$290
- ✅ Prix rebondit sur support

**MAIS le système ne détecte PAS de signal!**

### Diagnostic
1. ✅ **Signal EMA détecté** (Prix: $433.72, Support: $416.01, 4.3%)
2. ✅ **EMAs alignées** (24>38, 24>62, 38>62)
3. ❌ **Pas de breakout RSI récent** (< 3 périodes)

Le breakout RSI visible sur le screenshot est **trop ancien** (probablement il y a plusieurs semaines).

## 🔮 Prochaines Étapes Suggérées

### Option 1: Augmenter MAX_BREAKOUT_AGE
```python
# trendline_analysis/config/settings.py
MAX_BREAKOUT_AGE = 15  # Au lieu de 3
```
- **Avantage**: Détecte les breakouts plus anciens
- **Inconvénient**: Peut donner des signaux trop tardifs

### Option 2: Accepter Signaux EMA sans Breakout RSI (Recommandé)
- Générer signal si:
  - ✅ Signal EMA (support détecté)
  - ✅ Oblique RSI présente (même sans breakout)
- **Avantage**: Plus de signaux, détection précoce
- **Inconvénient**: Moins strict

### Option 3: Mode Flexible
- **STRONG_BUY**: EMA + RSI breakout récent
- **BUY**: EMA + oblique RSI (sans breakout)
- **WATCH**: EMA seul (sans RSI)

## 📊 État Actuel du Système

### Logique de Détection
```
1. Vérifier EMA Weekly:
   ├─ Crossover récent OU Prix proche EMA (rebond)
   └─ Si signal:
      ├─ Check RSI Weekly
      │  └─ Si breakout → SIGNAL
      └─ Check RSI Daily
         └─ Si breakout → SIGNAL

2. Si pas de signal weekly:
   └─ Vérifier EMAs weekly alignées:
      └─ Si alignées:
         └─ Check EMA Daily:
            ├─ Crossover récent OU Prix proche EMA
            └─ Si signal:
               └─ Check RSI Daily
                  └─ Si breakout → SIGNAL
```

### Critères Actuels
- **EMA alignement**: 2 conditions sur 3 (24>38, 24>62, 38>62)
- **Support EMA**: Prix à ≤8% d'une EMA
- **Support crossover**: Prix à ≤8% d'un crossover
- **RSI breakout**: Age ≤3 périodes

## 🌐 Accès Dashboard
- **URL**: http://localhost:8501
- **Section**: "🔍 Screening" → "Single Symbol"
- **Fonctionnalité**: Affiche TOUJOURS graphique (même sans signal)

## 💡 Points Clés pour la Suite

1. **Le système fonctionne** mais est **très strict**:
   - Nécessite Signal EMA + Breakout RSI récent
   - Peu de signaux actuellement

2. **La nouvelle logique de rebond EMA fonctionne**:
   - Détecte TSLA avec signal EMA (4.3% du support)
   - Mais bloqué par absence de breakout RSI récent

3. **Décision à prendre**:
   - Assouplir critères RSI (MAX_BREAKOUT_AGE)
   - OU accepter signaux EMA sans breakout RSI
   - OU créer système à plusieurs niveaux de signaux

## 📝 Question en Suspens

**"Peux tu garder le contexte de la conversation pour la prochaine session?"**

→ Ce fichier sert de contexte complet. La prochaine session peut:
1. Décider quelle option implémenter (1, 2, ou 3)
2. Tester avec TSLA et d'autres symboles
3. Affiner les critères selon les résultats

## 🔗 Fichiers Importants

### Configuration
- `config/settings.py` - Paramètres globaux
- `trendline_analysis/config/settings.py` - Paramètres RSI

### Analyse
- `src/indicators/ema_analyzer.py` - Analyse EMA
- `trendline_analysis/core/rsi_breakout_analyzer.py` - Analyse RSI
- `src/screening/screener.py` - Screener intégré

### Interface
- `dashboard.py` - Dashboard Streamlit

### Tests
- `test_tsla_signal.py` - Test TSLA
- `find_active_signals.py` - Scanner multi-symboles
- `diagnostic_aapl.py` - Diagnostic détaillé

---

**Date de sauvegarde**: 2025-10-25 02:02 UTC
**Statut**: Système opérationnel, décision à prendre sur critères RSI
