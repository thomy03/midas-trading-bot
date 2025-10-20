# 🧪 Guide de Test - Vérification des Règles de Trading

## 📊 Règles du Système (Rappel)

### Critères de Détection

Le système recherche des opportunités d'achat basées sur:

1. **EMAs Alignées** (24, 38, 62 périodes)
   - Au moins 2 EMAs dans l'ordre haussier
   - Exemples valides:
     - 24 > 38 > 62 (alignement parfait)
     - 24 > 38 (38 non alignée avec 62)
     - 24 > 62 (38 non alignée)

2. **Support Détecté**
   - Prix proche d'un croisement d'EMAs (zone de support)
   - Distance: 0-5% du support

3. **Timeframe**
   - **Weekly en priorité**: Signal plus fort
   - **Daily**: Si weekly aligné mais sans signal proche

### Niveaux de Recommandation

| Recommandation | Distance | Force Support | Signification |
|---------------|----------|---------------|---------------|
| **STRONG_BUY** 🔥 | ≤ 1% | ≥ 70% | Achat immédiat à considérer |
| **BUY** ✅ | ≤ 2% | ≥ 50% | Bonne opportunité |
| **WATCH** 👀 | ≤ 3.5% | Quelconque | À surveiller |
| **OBSERVE** 📊 | > 3.5% | Quelconque | EMAs alignées mais loin |

---

## 🎯 Tests à Effectuer

### Test 1: Interface Dashboard

**Dans votre navigateur (http://localhost:8501):**

1. **Page Home:**
   - Vérifiez que la page s'affiche correctement
   - Pas d'alertes récentes (base vide pour le moment)

2. **Page Chart Analyzer:**
   - Entrez le symbole: `AAPL`
   - Timeframe: `Weekly`
   - Period: `1 year`
   - Cliquez sur "Analyze"

**À vérifier:**
- ✅ Graphique candlestick s'affiche
- ✅ 3 EMAs visibles (bleu, orange, rose)
- ✅ Zones de support visualisées (lignes vertes)
- ✅ Croisements marqués (triangles verts/rouges)
- ✅ Volume en bas du graphique

3. **Panneau d'analyse (sous le graphique):**
   - Prix actuel
   - Valeurs des EMAs
   - Statut d'alignement (✅ ou ❌)
   - Support le plus proche
   - Distance en %

### Test 2: Screening Manuel

**Page Screening → Tab "Single Symbol":**

Testez ces symboles un par un:

#### Test A: Action avec Signal Fort (exemple)
```
Symbole: AAPL
```

**Vérifications:**
1. **EMAs:**
   - Vérifiez l'ordre: EMA24, EMA38, EMA62
   - Au moins 2 doivent être alignées (ordre croissant)

2. **Support:**
   - Un support doit être détecté
   - Distance affichée (en %)
   - Force du support (0-100%)

3. **Recommandation:**
   - Doit correspondre à la distance:
     - Si ≤ 1% et force ≥ 70% → STRONG_BUY
     - Si ≤ 2% et force ≥ 50% → BUY
     - Si ≤ 3.5% → WATCH
     - Si > 3.5% → OBSERVE

#### Test B: Multiple Symboles
**Page Screening → Tab "Multiple Symbols":**

Entrez (un par ligne):
```
AAPL
MSFT
GOOGL
TSLA
NVDA
```

Cliquez sur "Screen All".

**À vérifier:**
- ✅ Barre de progression s'affiche
- ✅ Résultats pour chaque symbole
- ✅ Graphiques générés pour ceux avec alertes
- ✅ Cohérence des recommandations

### Test 3: Vérification Manuelle des Règles

Pour un symbole donné (ex: AAPL), vérifiez manuellement:

1. **Graphique Weekly:**
   - Regardez les 3 EMAs
   - Sont-elles alignées? (au moins 2)
   - Y a-t-il des croisements récents?
   - Le prix est-il proche d'un croisement?

2. **Calcul de distance:**
   ```
   Distance = ((Prix actuel - Support) / Support) × 100
   ```
   - Correspond-elle à ce qu'affiche le système?

3. **Cohérence de la recommandation:**
   - STRONG_BUY: distance ≤ 1% + force ≥ 70%
   - BUY: distance ≤ 2% + force ≥ 50%
   - WATCH: distance ≤ 3.5%
   - OBSERVE: distance > 3.5%

---

## 🔍 Test 4: Screening Complet (Optionnel)

**Dans le terminal WSL:**

```bash
cd /mnt/c/Users/tkado/Documents/Tradingbot_V3
source venv/bin/activate
python main.py run
```

Ce screening va:
1. Analyser ~700 actions (NASDAQ, S&P 500, Europe)
2. Filtrer selon:
   - Capitalisation min (100M$ NASDAQ, 500M$ autres)
   - Volume quotidien min (750k$)
3. Appliquer les règles EMAs + Support
4. Générer des alertes

**Durée:** 3-5 minutes

**Résultats:**
- Affichés dans le terminal
- Sauvegardés dans la base de données
- Visibles ensuite dans le Dashboard → Alerts History

---

## ✅ Checklist de Validation

### Interface
- [ ] Dashboard s'ouvre correctement
- [ ] Graphiques s'affichent avec EMAs
- [ ] Zones de support visualisées
- [ ] Croisements marqués
- [ ] Volume affiché

### Règles de Trading
- [ ] EMAs calculées correctement (24, 38, 62)
- [ ] Alignement détecté (au moins 2 EMAs)
- [ ] Support détecté aux croisements
- [ ] Distance calculée correctement
- [ ] Recommandations cohérentes avec les règles

### Filtres de Marché
- [ ] Capitalisation min respectée
- [ ] Volume min respecté
- [ ] Marchés configurés (NASDAQ, S&P500, Europe)

### Données
- [ ] Prix en temps réel récupérés
- [ ] Données historiques suffisantes
- [ ] Pas d'erreurs d'API

---

## 🐛 Problèmes Possibles

### "No data available for symbol"
→ Symbole invalide ou indisponible sur Yahoo Finance

### "Not enough data to calculate EMAs"
→ Symbole trop récent, pas assez d'historique

### EMAs ne s'affichent pas
→ Vérifiez la période sélectionnée (min 6 mois recommandé)

### Recommandation incohérente
→ Vérifiez manuellement:
  1. Distance au support
  2. Force du support
  3. Alignement des EMAs

---

## 📊 Exemples de Vérification Manuelle

### Cas 1: STRONG_BUY Attendu

**Conditions:**
- Prix: $175.00
- Support: $174.00
- Distance: 0.57% ✅ (< 1%)
- Force: 85% ✅ (> 70%)
- EMAs: 24>38>62 ✅

**Recommandation attendue:** STRONG_BUY 🔥

### Cas 2: BUY Attendu

**Conditions:**
- Prix: $100.00
- Support: $98.50
- Distance: 1.52% ✅ (< 2%)
- Force: 60% ✅ (> 50%)
- EMAs: 24>38 ✅

**Recommandation attendue:** BUY ✅

### Cas 3: WATCH Attendu

**Conditions:**
- Prix: $150.00
- Support: $145.00
- Distance: 3.45% ✅ (< 3.5%)
- Force: 40%
- EMAs: 24>38 ✅

**Recommandation attendue:** WATCH 👀

### Cas 4: OBSERVE Attendu

**Conditions:**
- Prix: $200.00
- Support: $190.00
- Distance: 5.26% ❌ (> 3.5%)
- EMAs: 24>38>62 ✅

**Recommandation attendue:** OBSERVE 📊

---

## 🎯 Résultat Attendu

Si tous les tests passent:
- ✅ Interface fonctionnelle
- ✅ Règles de trading respectées
- ✅ Filtres appliqués correctement
- ✅ Recommandations cohérentes

**Vous êtes prêt à lancer le scheduler automatique!** 🚀

---

## 📝 Notes de Test

Utilisez cette section pour noter vos observations:

```
Date du test: _____________

Symboles testés:
- AAPL: _____________
- MSFT: _____________
- GOOGL: _____________

Incohérences détectées:
_______________________
_______________________

Règles à ajuster:
_______________________
_______________________
```
