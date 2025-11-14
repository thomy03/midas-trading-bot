# 📊 Dashboard - Guide d'Utilisation

## Vue d'ensemble

Le Market Screener Dashboard est une interface web interactive type **TradingView** qui vous permet de visualiser, analyser et screener des actions en temps réel.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Dashboard+Preview)

## 🚀 Lancement Rapide

### Méthode 1: Scripts automatiques

**Windows:**
```bash
start_dashboard.bat
```

**Linux/Mac:**
```bash
./start_dashboard.sh
```

### Méthode 2: Commande directe

```bash
streamlit run dashboard.py
```

Le dashboard s'ouvrira automatiquement dans votre navigateur à l'adresse:
```
http://localhost:8501
```

## 📑 Pages du Dashboard

### 🏠 Home

**Page d'accueil** avec vue d'ensemble:

- **Quick Stats**: Nombre d'alertes récentes
- **Recent Alerts**: Tableau des 20 dernières alertes avec:
  - Symbole et nom de la société
  - Timeframe (Weekly/Daily)
  - Prix actuel et niveau de support
  - Distance au support
  - Recommandation (STRONG_BUY, BUY, WATCH, OBSERVE)
  - Date de l'alerte

- **Alert Distribution**: Graphique circulaire montrant la répartition des recommandations

**Utilisation:**
- Vue rapide de l'activité récente
- Identification des meilleures opportunités
- Export CSV des alertes

---

### 📊 Chart Analyzer

**Analyse graphique interactive** type TradingView:

#### Fonctionnalités:

1. **Graphiques en chandelier (Candlestick)**
   - Prix d'ouverture, haut, bas, fermeture
   - Couleurs: Vert (hausse) / Rouge (baisse)

2. **EMAs (Moyennes Mobiles Exponentielles)**
   - EMA 24 (Bleu)
   - EMA 38 (Orange)
   - EMA 62 (Rose)
   - Affichage en temps réel

3. **Zones de Support**
   - Lignes horizontales vertes
   - Zone de tolérance ±5% ombrée
   - Force du support indiquée

4. **Croisements d'EMAs**
   - Marqueurs triangulaires
   - Vert = Croisement haussier (bullish)
   - Rouge = Croisement baissier (bearish)

5. **Volume**
   - Histogramme sous le graphique principal
   - Couleur correspondant au mouvement du prix

#### Options:

- **Symbol**: Entrez n'importe quel ticker (AAPL, MSFT, etc.)
- **Timeframe**: Daily ou Weekly
- **Period**: 6 mois, 1 an, 2 ans, 5 ans
- **Show Volume**: Afficher/masquer le volume

#### Panneau d'Analyse:

Sous le graphique, vous trouverez:

**Colonne 1 - Prix & EMAs:**
- Prix actuel
- Valeurs des 3 EMAs

**Colonne 2 - Alignement:**
- Statut d'alignement des EMAs (✅/❌)
- Description de l'alignement
- Nombre de croisements trouvés
- Nombre de zones de support

**Colonne 3 - Support:**
- Niveau du support le plus proche
- Distance en %
- Force de la zone (0-100%)

**Tableau des Croisements:**
- 10 derniers croisements d'EMAs
- Date, type, prix, âge

#### Interactivité:

- **Zoom**: Sélectionnez une zone avec la souris
- **Pan**: Cliquez et glissez
- **Hover**: Survolez pour voir les valeurs exactes
- **Légende**: Cliquez pour masquer/afficher des indicateurs
- **Reset**: Double-cliquez pour réinitialiser la vue

---

### 🔍 Screening

**Screening manuel** de symboles:

#### Tab 1: Single Symbol

Screenez un symbole à la fois:

1. Entrez le symbole (ex: AAPL)
2. Cliquez sur "🔍 Screen"
3. Résultats affichés:
   - Métriques clés (timeframe, prix, support, recommandation)
   - Détails complets en JSON
   - Graphique interactif automatique

**Cas d'usage:**
- Vérifier rapidement une action spécifique
- Valider un signal avant d'acheter
- Explorer une nouvelle opportunité

#### Tab 2: Multiple Symbols

Screenez plusieurs symboles en batch:

1. Entrez les symboles:
   - Un par ligne
   - Ou séparés par des virgules
   - Ex:
   ```
   AAPL
   MSFT
   GOOGL, TSLA, NVDA
   ```

2. Cliquez sur "🔍 Screen All"

3. Barre de progression en temps réel

4. Résultats avec expandeurs:
   - Chaque alerte dans un expandeur
   - Métriques clés
   - Graphique interactif

**Cas d'usage:**
- Analyser votre watchlist personnelle
- Comparer plusieurs actions
- Screening rapide d'un secteur

---

### 🚨 Alerts History

**Historique complet** des alertes:

#### Filtres:

- **Days to look back**: Slider 1-90 jours
- **Timeframe**: Daily et/ou Weekly

#### Tableau:

Colonnes:
- Date & heure
- Symbole & société
- Timeframe
- Prix
- Support
- Distance %
- EMAs (24, 38, 62)
- Recommandation
- Statut notification (✅/❌)

#### Fonctionnalités:

1. **Formatage automatique**:
   - Prix en dollars ($)
   - Distances en pourcentage (%)
   - Couleurs par recommandation

2. **Export CSV**:
   - Bouton "📥 Download CSV"
   - Nom de fichier avec date

3. **Statistiques**:
   - Nombre de Strong Buys
   - Nombre de Buys
   - Signaux Weekly vs Daily

**Cas d'usage:**
- Analyser les performances passées
- Identifier les patterns
- Exporter pour analyse externe (Excel, etc.)
- Vérifier si une action a déjà été alertée

---

### ⚙️ Settings

**Configuration et diagnostics**:

#### Current Configuration:

1. **Screening Parameters**:
   - Périodes des EMAs
   - Tolérance de zone de support

2. **Market Filters**:
   - Capitalisation min par marché (NASDAQ, SP500, Europe)
   - Volume quotidien minimum

3. **Notification Settings**:
   - Statut Telegram (✅/⚠️)
   - Instructions de configuration

4. **Scheduling**:
   - Heure du rapport quotidien
   - Fuseau horaire

#### Quick Actions:

1. **🧪 Test Notifications**:
   - Envoie un message de test
   - Vérifie la configuration Telegram
   - Résultat immédiat

2. **🗄️ View Database**:
   - Emplacement de la base de données
   - Nombre d'alertes récentes

3. **📊 System Info**:
   - Version Python
   - Système d'exploitation

---

## 💡 Conseils d'Utilisation

### Workflow Recommandé:

1. **Matin**:
   - Ouvrez le Dashboard
   - Vérifiez Home pour les nouvelles alertes
   - Analysez les STRONG_BUY dans Chart Analyzer

2. **Recherche d'opportunités**:
   - Utilisez Screening > Multiple Symbols avec votre watchlist
   - Filtrez par recommandation
   - Étudiez les graphiques

3. **Analyse approfondie**:
   - Chart Analyzer pour comprendre le contexte
   - Vérifiez l'historique dans Alerts History
   - Confirmez l'alignement des EMAs

4. **Suivi**:
   - Alerts History pour tracker les performances
   - Export CSV pour analyse externe

### Raccourcis Clavier:

Les raccourcis Streamlit standards:
- `Ctrl + R`: Rafraîchir la page
- `Ctrl + Shift + R`: Rafraîchir en effaçant le cache

### Performance:

**Chargement initial:**
- Premier symbole: 2-5 secondes (téléchargement données)
- Symboles suivants: 1-2 secondes (cache)

**Screening multiple:**
- ~3-5 secondes par symbole
- Peut être optimisé avec API premium

**Graphiques:**
- Rendu instantané avec Plotly
- Zoom et pan fluides

---

## 🎨 Personnalisation

### Thème:

Le dashboard utilise le thème sombre par défaut (optimisé pour le trading).

Pour changer:
```bash
# Créer .streamlit/config.toml
mkdir .streamlit
cat > .streamlit/config.toml << EOF
[theme]
primaryColor = "#00C853"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1E1E1E"
textColor = "#FAFAFA"
font = "sans serif"
EOF
```

### Layout:

Modifiez `dashboard.py`:
- Largeur des colonnes: `st.columns([2, 1])`
- Hauteur des graphiques: `height=800`
- Nombre d'alertes affichées: `[:20]`

---

## 🔧 Dépannage

### Le dashboard ne démarre pas

**Erreur: `streamlit: command not found`**

Solution:
```bash
pip install streamlit
```

**Erreur: Port déjà utilisé**

Solution:
```bash
streamlit run dashboard.py --server.port 8502
```

### Graphiques ne s'affichent pas

**Erreur: `No data available`**

Causes possibles:
1. Symbole invalide → Vérifiez le ticker
2. Connexion Internet → Testez avec un navigateur
3. Limites API yfinance → Attendez quelques minutes

### Performance lente

**Le chargement prend trop de temps**

Solutions:
1. Réduisez la période (6mo au lieu de 5y)
2. Fermez les onglets inutilisés
3. Videz le cache: `Ctrl + Shift + R`
4. Redémarrez le dashboard

---

## 📱 Accès à Distance

### Sur votre réseau local:

```bash
streamlit run dashboard.py --server.address 0.0.0.0
```

Accédez depuis un autre appareil:
```
http://[votre-ip-locale]:8501
```

Trouvez votre IP:
- Windows: `ipconfig`
- Linux/Mac: `ifconfig` ou `ip addr`

### Via Internet (Avancé):

Options:
1. **Streamlit Cloud** (gratuit, public)
2. **VPS** avec reverse proxy (Nginx)
3. **Tunnel SSH** (sécurisé, temporaire)

⚠️ **Attention**: Ne pas exposer publiquement sans authentification!

---

## 🔐 Sécurité

Le dashboard est conçu pour un usage **local uniquement**.

**Bonnes pratiques:**
- Ne partagez pas votre URL publiquement
- Utilisez un VPN si accès à distance
- Ne stockez pas de données sensibles
- Gardez vos credentials (.env) privés

---

## 📞 Support

**Problèmes courants:**
1. Consultez la section Dépannage ci-dessus
2. Vérifiez les logs dans `logs/screener.log`
3. Testez avec un symbole simple (AAPL)

**Fonctionnalités futures:**
- Comparaison multi-symboles
- Alertes configurables
- Backtesting intégré
- Export PDF des graphiques
- Mode mobile optimisé

---

**Bon trading! 📈**
