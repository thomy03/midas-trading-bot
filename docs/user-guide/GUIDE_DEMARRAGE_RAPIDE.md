# Guide de Démarrage Rapide - Prochaine Session

## 📋 CONTEXTE EN 30 SECONDES

**Objectif**: Système de trading détectant niveau historique EMA proche + RSI breakout = STRONG_BUY

**Problème actuel**:
1. ❌ Niveau $290 sur TSLA non détecté (visible sur screenshot utilisateur)
2. ❌ Niveaux de support horizontaux ne s'affichent PAS sur le graphique dashboard

**Ce qui fonctionne**:
- ✅ Détection crossovers EMA historiques
- ✅ RSI trendlines + breakouts
- ✅ Screener intégré
- ✅ Dashboard avec page "Signaux Historiques"

---

## 🚀 DÉMARRAGE IMMÉDIAT

### 1. Lire la synthèse complète
```bash
cat SYNTHESE_FINALE_COMPLETE.md
```

### 2. Vérifier état actuel
```bash
# Test rapide niveaux TSLA
python3 -c "
import sys; sys.path.insert(0, '.')
from src.data.market_data import market_data_fetcher
from src.indicators.ema_analyzer import ema_analyzer

df = market_data_fetcher.get_historical_data('TSLA', period='2y', interval='1wk')
df = ema_analyzer.calculate_emas(df)
crossovers = ema_analyzer.detect_crossovers(df, 'weekly')
current_price = float(df['Close'].iloc[-1])
levels = ema_analyzer.find_historical_support_levels(df, crossovers, current_price)

print(f'Prix: \${current_price:.2f}')
print(f'Niveaux détectés: {len(levels)}')
for l in levels[:3]:
    print(f'  \${l[\"level\"]:.2f} - {l[\"distance_pct\"]:.1f}% - {l[\"crossover_info\"][\"date\"].strftime(\"%Y-%m-%d\")}')
"
```

### 3. Lancer dashboard
```bash
# Tuer anciens processus
pkill -9 -f streamlit

# Lancer nouveau
streamlit run dashboard.py --server.port 8501
```

**URL**: http://localhost:8501

---

## 🎯 ACTIONS PRIORITAIRES

### Problème #1: Niveaux horizontaux manquants sur graphique

**Fichier à vérifier**: `src/utils/visualizer.py`

**Méthode**: `create_historical_chart()` (ligne ~677)

**Vérifications**:
```python
# 1. Les niveaux sont-ils bien passés à la fonction?
print(f"Niveaux reçus: {len(historical_levels)}")

# 2. Les lignes horizontales sont-elles créées?
for level in historical_levels:
    print(f"Ajout ligne à ${level['level']:.2f}")
    fig.add_hline(y=level['level'], ...)  # Cette ligne s'exécute?

# 3. Le graphique est-il retourné correctement?
return fig  # Pas d'erreur avant?
```

**Test visuel**:
1. Ouvrir dashboard → "📈 Signaux Historiques"
2. Entrer TSLA, weekly, 2y
3. Cliquer "Afficher"
4. **Vérifier**: Lignes horizontales vertes/rouges apparaissent?

### Problème #2: Niveau $290 non détecté

**Screenshot utilisateur montre**: Support horizontal à ~$290

**Système détecte**: $200-210 uniquement

**Hypothèses**:
1. Niveau $290 hors période (2y) → Tester avec 5y
2. Niveau $290 crossover bearish ignoré → Vérifier tous types
3. EMAs ont retracé sous $290 → Afficher aussi niveaux invalidés

**Test étendu**:
```python
# Test avec période 5y
df = market_data_fetcher.get_historical_data('TSLA', period='5y', interval='1wk')
df = ema_analyzer.calculate_emas(df)
crossovers = ema_analyzer.detect_crossovers(df, 'weekly')

print(f"Crossovers (5y): {len(crossovers)}")
for c in crossovers:
    if 280 <= c['price'] <= 300:
        print(f"  ${c['price']:.2f} - {c['date']} - {c['type']}")
```

---

## 📁 FICHIERS CLÉS

### Code Source (À NE PAS TOUCHER sauf debug)
```
src/indicators/ema_analyzer.py     # Niveaux historiques ✅
src/screening/screener.py          # Screener modifié ✅
src/utils/visualizer.py            # Graphiques ❌ À CORRIGER
trendline_analysis/core/rsi_breakout_analyzer.py  # RSI ✅
dashboard.py                       # Interface ✅
```

### Configuration
```
config/settings.py                 # Paramètres EMA
trendline_analysis/config/settings.py  # Paramètres RSI
```

### Documentation
```
SYNTHESE_FINALE_COMPLETE.md        # ⭐ LIRE EN PREMIER
GUIDE_DEMARRAGE_RAPIDE.md         # Ce fichier
SESSION_FINALE_NIVEAUX_HISTORIQUES.md  # Logique détaillée
IMPLEMENTATION_NIVEAUX_HISTORIQUES.md  # Implémentation
```

---

## 🔍 DEBUG RAPIDE

### Si niveaux ne s'affichent pas

```python
# Dans dashboard.py, page "Signaux Historiques"
# Ajouter après ligne 281:

print(f"DEBUG: Niveaux historiques = {len(historical_levels)}")
for i, level in enumerate(historical_levels[:5]):
    print(f"  {i+1}. ${level['level']:.2f} - near={level['is_near']}")
```

### Si niveau $290 manque

```python
# Dans src/indicators/ema_analyzer.py
# Méthode detect_crossovers(), ajouter:

print(f"Crossover détecté: ${cross_price:.2f} - {cross_date} - {cross_type}")
```

---

## 💡 RAPPELS IMPORTANTS

1. **Distance ≠ Validité**
   - Niveau à 100% peut être valide si EMAs au-dessus
   - Distance sert uniquement pour alertes (< 8%)

2. **Niveaux = Références permanentes**
   - Pas de limite de distance pour affichage
   - Limite d'âge SEULEMENT si EMAs retracées

3. **Signal complet = 3 conditions**
   - Niveau proche (< 8%)
   - RSI trendline présente
   - RSI breakout détecté

---

## ⚡ COMMANDES ULTRA-RAPIDES

```bash
# Dashboard
streamlit run dashboard.py --server.port 8501

# Test niveaux
python3 test_historical_levels.py  # Si existe

# Nettoyer
bash cleanup_temp_files.sh

# Supprimer archive (si nettoyage OK)
rm -rf archive_temp_files/
```

---

## 📊 EXEMPLE TSLA ATTENDU

**Sur le graphique, on DEVRAIT voir**:

```
Graphique Prix:
├─ Candlesticks ✅
├─ EMA 24 (orange) ✅
├─ EMA 38 (bleu) ✅
├─ EMA 62 (violet) ✅
├─ Ligne horizontale verte pointillée à $208 ❌ MANQUANTE
├─ Ligne horizontale verte pointillée à $204 ❌ MANQUANTE
└─ Étoiles aux crossovers ❌ MANQUANTES?

Graphique RSI:
├─ Ligne RSI (bleu) ✅
├─ Ligne oblique rouge (trendline) ✅
├─ Triangles rouges (peaks) ✅
└─ Étoile verte (breakout) ✅
```

**Actuellement**:
- RSI: Tout s'affiche ✅
- Prix: Lignes horizontales MANQUENT ❌

---

## 🎯 SUCCÈS =

1. [ ] Lignes horizontales vertes/rouges visibles sur graphique prix
2. [ ] Étoiles aux points de crossover visibles
3. [ ] Niveau $290 TSLA détecté (ou expliqué pourquoi absent)
4. [ ] Dashboard complet et fonctionnel

---

**Prochaine étape**: Débugger `visualizer.py:create_historical_chart()` ligne par ligne pour comprendre pourquoi les niveaux ne s'affichent pas!

**Bon courage!** 🚀
