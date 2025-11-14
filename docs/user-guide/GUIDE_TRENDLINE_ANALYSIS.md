# 🎯 Guide: Trendline Analysis Dashboard

## Vue d'ensemble

Cette fonctionnalité analyse la **double confirmation** pour valider les signaux d'achat:
1. **RSI Breakout**: Cassure de trendline descendante sur RSI
2. **Price Breakout**: Cassure de trendline sur prix (support/resistance)
3. **Synchronisation**: Les deux breakouts doivent se produire dans une fenêtre de ±6 périodes

## Stratégie de Trading

### Signal d'ACHAT validé quand:
```
✅ RSI breakout détecté (cassure résistance descendante)
   ET
✅ Price trendline détectée (support ou résistance)
   ET
✅ Price breakout détecté (cassure de la trendline)
   ET
🎯 Synchronisation: Les deux breakouts sont à ±6 périodes
```

### Workflow complet:
1. **Screening EMA** → Identifie les candidats potentiels
2. **Trendline Analysis** → Valide avec double confirmation RSI + Prix
3. **Signal d'achat** → Seulement si les deux conditions sont remplies

## Utilisation du Dashboard

### 1. Accéder à la page
- Ouvrir le dashboard: `http://localhost:8501`
- Naviguer vers: **🎯 Trendline Analysis**

### 2. Paramètres d'analyse
- **Symbol**: Symbole à analyser (ex: AAPL, MSFT, TSLA)
- **Timeframe**:
  - `daily` - Données journalières (recommandé pour trading court/moyen terme)
  - `weekly` - Données hebdomadaires (pour tendances long terme)
- **Lookback**: Profondeur d'analyse
  - `104` - ~6 mois (daily) / ~2 ans (weekly)
  - `252` - ~1 an (daily) / ~5 ans (weekly) ⭐ **Recommandé**
  - `500` - ~2 ans (daily) / ~10 ans (weekly)

### 3. Cartes de statut
Après analyse, 4 cartes montrent l'état:
- ✅ **RSI Breakout** - Cassure de résistance RSI détectée
- ✅ **Price Trendline** - Trendline sur prix détectée (support/resistance)
- ✅ **Price Breakout** - Cassure de la trendline prix
- 🎯 **Dual Confirmation** - Les deux breakouts synchronisés

### 4. Métriques détaillées

#### RSI Analysis
```
📈 RSI Trendline:
   - Peaks: Nombre de pics formant la trendline (min 3)
   - R²: Qualité du fit (>0.6 = bon)
   - Slope: Pente (négatif pour résistance descendante)
   - Quality: Score global /100

🚀 RSI Breakout:
   - Date: Quand le RSI a cassé la trendline
   - RSI: Valeur du RSI au moment du breakout
   - Trendline: Valeur de la trendline à ce moment
   - Distance: Écart au-dessus de la trendline
   - Strength: WEAK/MODERATE/STRONG
   - Age: Nombre de périodes depuis le breakout
```

#### Price Analysis
```
📈 Price Trendline (SUPPORT ou RESISTANCE):
   - Peaks: Nombre de points de contact
   - R²: Qualité du fit (>0.5 = bon pour prix)
   - Slope: Pente de la trendline
   - Quality: Score global /100

🚀 Price Breakout:
   - Date: Quand le prix a cassé la trendline
   - Price: Valeur du prix au breakout
   - Trendline: Valeur de la trendline à ce moment
   - Distance: Écart par rapport à la trendline
   - Strength: WEAK/MODERATE/STRONG
   - Age: Nombre de périodes depuis le breakout
```

### 5. Graphique interactif

Le graphique montre 2 rangées:

**Rangée 1 - Prix:**
- Chandelier japonais (OHLC)
- Trendline prix (ligne violette pointillée)
- Pics de la trendline (cercles violets)
- Breakout prix (étoile violette ⭐)

**Rangée 2 - RSI:**
- Courbe RSI (ligne bleue)
- Trendline RSI (ligne orange pointillée)
- Pics RSI (cercles orange)
- Breakout RSI (étoile verte ⭐)
- Niveaux 70/30 (zones surachat/survente)

**Interactivité:**
- Zoom: Sélectionner une zone avec la souris
- Pan: Glisser pour naviguer
- Hover: Afficher les valeurs exactes
- Reset: Double-clic pour réinitialiser

## Détection de Trendline Prix - Stratégie Mixte

### Pour RESISTANCE (trendline descendante):
Combine deux sources de pics:
1. **High (wicks)** - Mèches hautes des bougies
2. **Close des bougies VERTES** - Close > Open

Avantages:
- Plus de points de contact potentiels
- Détection plus flexible et robuste
- Capture à la fois rejets violents (wicks) et rejets doux (close)

### Pour SUPPORT (trendline ascendante):
Combine deux sources de vallées:
1. **Low (wicks)** - Mèches basses des bougies
2. **Close des bougies ROUGES** - Close < Open

### Paramètres de validation:
- R² minimum: **0.50** (vs 0.60 pour RSI - plus tolérant pour volatilité prix)
- Résiduel max: **5%** du prix moyen (vs 3% pour RSI)
- Direction: Premier et dernier pic montrent la tendance
- Pente: Validation du signe uniquement (pas de magnitude absolue)

## Exemples d'utilisation

### Cas 1: Signal d'achat confirmé
```
Symbol: MSFT
✅ RSI Breakout: 2025-06-15
✅ Price Trendline: RESISTANCE (R²=0.966)
✅ Price Breakout: 2025-06-18
🎯 DUAL CONFIRMATION: 3 periods apart
→ SIGNAL D'ACHAT VALIDÉ! 🚀
```

### Cas 2: En attente de confirmation prix
```
Symbol: AAPL
✅ RSI Breakout: 2025-06-30
✅ Price Trendline: RESISTANCE (R²=0.999)
❌ Price Breakout: Pas encore
⏳ Attente de cassure de la résistance prix
→ PAS DE SIGNAL D'ACHAT (incomplet)
```

### Cas 3: Pas de trendline prix
```
Symbol: XYZ
✅ RSI Breakout: 2025-07-01
❌ Price Trendline: Aucune détectée
→ PAS DE SIGNAL D'ACHAT (prix trop volatile ou pas de trend clair)
```

### Cas 4: Breakouts non synchronisés
```
Symbol: TSLA
✅ RSI Breakout: 2025-05-10
✅ Price Trendline: SUPPORT (R²=0.750)
✅ Price Breakout: 2025-06-25
⚠️ NOT SYNCHRONIZED: 32 periods apart (>6)
→ PAS DE SIGNAL D'ACHAT (trop d'écart temporel)
```

## Intégration avec le Screener EMA

### Workflow recommandé:

1. **Page 🔍 Screening**
   - Lancer le screening avec critères EMA (24/38/62)
   - Obtenir liste des candidats qui passent les critères EMA

2. **Page 🎯 Trendline Analysis**
   - Pour chaque candidat du screening
   - Vérifier la double confirmation RSI + Prix
   - Ne prendre positions que sur symboles avec 🎯 DUAL CONFIRMATION

3. **Critères cumulatifs**
   ```
   ✅ Passe le screening EMA (support + bougie baissière)
      ET
   🎯 Dual confirmation trendline (RSI + Prix synchronisés)
      →  SIGNAL D'ACHAT FINAL
   ```

## Paramètres techniques

### RSI Trendline Detection
- Prominence: 1.5 (1.5% du range RSI 0-100)
- Distance: 3 périodes minimum entre pics
- Min peaks: 3 pics minimum
- R² minimum: 0.60
- Résiduel max: 3.0 (3% du range RSI)

### Price Trendline Detection
- Prominence: 1.5% du range de prix
- Distance: 3 périodes minimum entre pics/vallées
- Min peaks: 3 points de contact minimum
- R² minimum: 0.50 (plus tolérant que RSI)
- Résiduel max: 5% du prix moyen (volatilité prix > RSI)

### Synchronization
- Fenêtre: ±6 périodes
- Exemple: Si RSI breakout au jour 100, prix breakout doit être entre jour 94-106

## Limitations et précautions

### ⚠️ Quand la détection peut échouer:
1. **Volatilité extrême** - Prix très chaotique, pas de trend clair
2. **Données insuffisantes** - Lookback trop court, pas assez de pics
3. **Trends trop faibles** - R² < seuils minimum
4. **Breakout trop ancien** - Signal périmé (age > 20-30 périodes)

### 💡 Bonnes pratiques:
- ✅ Utiliser lookback=252 (1 an) pour balance data/recency
- ✅ Vérifier l'âge du breakout (<20 périodes pour fraîcheur)
- ✅ Combiner avec analyse volume et contexte macro
- ✅ Valider visuellement sur le graphique interactif
- ✅ Ne trader que les symboles avec dual confirmation FORTE
- ❌ Ne pas trader sur breakouts anciens (>30 périodes)
- ❌ Ne pas ignorer le contexte de marché général

## Démarrage du Dashboard

```bash
# Depuis le répertoire du projet
./start_dashboard.sh

# Ou manuellement
source venv/bin/activate
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Dashboard disponible à: **http://localhost:8501**

## Support et dépannage

### Dashboard ne démarre pas:
```bash
# Vérifier processus Streamlit
ps aux | grep streamlit

# Tuer processus si nécessaire
pkill -f streamlit

# Redémarrer
./start_dashboard.sh
```

### Erreurs d'import:
```bash
# Réinstaller dépendances
source venv/bin/activate
pip install -r requirements.txt
```

### Tester l'analyse hors dashboard:
```bash
# Tester avec script standalone
python test_streamlit_trendline.py
```

---

**Créé par:** Claude Code
**Dernière mise à jour:** 2025-10-22
**Version:** 2.0 (Dual Confirmation avec Mixed Peak Detection)
