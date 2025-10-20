# Market Screener - Automated Stock Screening System

Un système de screening automatisé de marché qui analyse les actions (et bientôt les cryptos) en utilisant des moyennes mobiles exponentielles (EMA) pour identifier des opportunités d'achat.

## Fonctionnalités

- **Screening Multi-Timeframe**: Analyse en Weekly et Daily
- **Indicateurs Techniques**: 3 EMAs (24, 38, 62 périodes)
- **Détection de Support/Résistance**: Basée sur les croisements d'EMAs
- **Filtrage Intelligent Différencié**:
  - NASDAQ: Capitalisation > 100M$
  - S&P 500: Capitalisation > 500M$
  - Europe: Capitalisation > 500M$
  - Volume quotidien > 750k$ (tous marchés)
  - Marchés: NASDAQ, S&P 500, Europe, ADR Asiatiques
- **Notifications Automatiques**: Telegram et Email
- **Rapports Quotidiens**: Envoyés chaque matin à 8h
- **Base de Données**: Historique des alertes et analyses
- **🎨 Dashboard Interactif**: Interface web type TradingView avec graphiques interactifs

## Stratégie de Trading

### Critères de Détection

1. **EMAs alignées**: Au moins 2 EMAs dans l'ordre haussier (24>38, 24>62, ou 38>62)
2. **Support détecté**: Prix proche d'un croisement d'EMAs (zone de 0-5%)
3. **Timeframe**:
   - Screening Weekly en priorité
   - Si EMAs alignées en Weekly mais pas de signal → analyse Daily

### Recommandations

- **STRONG_BUY**: Distance ≤ 1% du support, force ≥ 70%
- **BUY**: Distance ≤ 2% du support, force ≥ 50%
- **WATCH**: Distance ≤ 3.5% du support
- **OBSERVE**: Distance > 3.5% du support

## Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de packages Python)

### Étapes d'Installation

1. **Cloner ou télécharger le projet**

```bash
cd Tradingbot_V3
```

2. **Créer un environnement virtuel (recommandé)**

```bash
python -m venv venv

# Sur Windows
venv\Scripts\activate

# Sur Linux/Mac
source venv/bin/activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Configurer les variables d'environnement**

Copiez le fichier `.env.example` vers `.env`:

```bash
cp .env.example .env
```

Éditez `.env` et configurez vos paramètres:

```env
# Configuration Telegram (RECOMMANDÉ)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Configuration Email (Optionnel)
EMAIL_ENABLED=False
EMAIL_FROM=your_email@example.com
EMAIL_TO=recipient@example.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@example.com
SMTP_PASSWORD=your_app_password
```

### Configuration de Telegram (Recommandé)

1. **Créer un bot Telegram**:
   - Ouvrez Telegram et cherchez `@BotFather`
   - Envoyez `/newbot`
   - Suivez les instructions et notez le **token**

2. **Obtenir votre Chat ID**:
   - Cherchez `@userinfobot` sur Telegram
   - Envoyez `/start`
   - Notez votre **Chat ID**

3. **Configurer dans .env**:
   ```env
   TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
   TELEGRAM_CHAT_ID=123456789
   ```

## Utilisation

### Commandes Disponibles

#### 1. Exécuter un screening unique

```bash
python main.py run
```

Lance un screening complet une seule fois et envoie le rapport.

#### 2. Activer le scheduler (Mode automatique)

```bash
python main.py schedule
```

Lance le screener en mode automatique. Il s'exécutera automatiquement chaque jour à 8h (configurable dans `config/settings.py`).

#### 3. Tester les notifications

```bash
python main.py test
```

Envoie un message de test via Telegram/Email pour vérifier la configuration.

#### 4. Screener un symbole spécifique

```bash
python main.py screen --symbol AAPL
```

Analyse un symbole spécifique et affiche les résultats.

#### 5. Lancer le Dashboard Web 🎨

```bash
# Sur Windows
start_dashboard.bat

# Sur Linux/Mac
./start_dashboard.sh

# Ou directement avec streamlit
streamlit run dashboard.py
```

Lance l'interface web interactive sur http://localhost:8501

**Fonctionnalités du Dashboard:**
- 📊 **Chart Analyzer**: Graphiques interactifs type TradingView avec EMAs et zones de support
- 🔍 **Screening Manual**: Screener des symboles individuels ou multiples
- 🚨 **Alerts History**: Historique complet des alertes avec filtres
- ⚙️ **Settings**: Configuration et tests du système

#### 6. Voir les alertes récentes

```bash
python main.py alerts --days 7
```

Affiche les alertes des 7 derniers jours depuis la base de données.

### Exemples d'Utilisation

```bash
# Screening manuel
python main.py run

# Mode automatique (tourne en continu)
python main.py schedule

# Analyser Apple
python main.py screen --symbol AAPL

# Analyser Tesla
python main.py screen --symbol TSLA

# Voir les alertes des 30 derniers jours
python main.py alerts --days 30

# Tester les notifications
python main.py test
```

## Configuration

### Fichier `config/settings.py`

Vous pouvez personnaliser:

- **EMAs**: Modifier les périodes (`EMA_PERIODS`)
- **Filtres**: Capitalisation min, volume min (`MIN_MARKET_CAP`, `MIN_DAILY_VOLUME`)
- **Tolérance**: Zone de support/résistance (`ZONE_TOLERANCE`)
- **Horaire**: Heure du rapport quotidien (`DAILY_REPORT_TIME`)
- **Timezone**: Fuseau horaire (`TIMEZONE`)
- **Symboles personnalisés**: Ajouter des symboles dans `CUSTOM_SYMBOLS`

Exemple:

```python
# Ajouter vos symboles favoris
CUSTOM_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'TSLA',
    'NVDA', 'AMD', 'META', 'AMZN'
]

# Changer l'heure du rapport (format 24h)
DAILY_REPORT_TIME = time(7, 30)  # 7h30 du matin
```

## Structure du Projet

```
trading-screener/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration principale
├── src/
│   ├── data/
│   │   └── market_data.py   # Récupération données marché
│   ├── indicators/
│   │   └── ema_analyzer.py  # Analyse des EMAs
│   ├── screening/
│   │   └── screener.py      # Logique de screening
│   ├── notifications/
│   │   └── notifier.py      # Système de notifications
│   ├── database/
│   │   └── db_manager.py    # Gestion base de données
│   └── utils/
│       └── logger.py        # Système de logging
├── data/
│   └── screener.db          # Base de données SQLite
├── logs/
│   └── screener.log         # Fichiers de logs
├── main.py                  # Point d'entrée
├── requirements.txt         # Dépendances
├── .env                     # Variables d'environnement
└── README.md               # Documentation
```

## Format des Notifications

### Rapport Quotidien

```
📊 DAILY MARKET SCREENING REPORT
🗓 2025-10-19 08:00:00

========================================

📈 Summary:
  • Stocks Analyzed: 523
  • Alerts Generated: 12
  • Execution Time: 245.3s
  • Status: SUCCESS

========================================

🔥 STRONG BUY (3)
  • AAPL @ $175.50 (weekly)
  • MSFT @ $380.20 (daily)
  • NVDA @ $495.75 (weekly)

✅ BUY (5)
  • GOOGL @ $145.30 (daily)
  ...

💡 Top 3 Opportunities:

1. 🔥 AAPL - Apple Inc.
📊 Timeframe: WEEKLY
💰 Current Price: $175.50
🎯 Support Level: $174.20
📏 Distance: 0.74%
📈 EMAs:
  • EMA 24: $176.30
  • EMA 38: $174.50
  • EMA 62: $172.80
  • Alignment: 24>38, 24>62, 38>62
💡 Recommendation: STRONG_BUY
```

## Base de Données

Le système utilise SQLite pour stocker:

- **Alertes**: Tous les signaux d'achat générés
- **Historique**: Résultats de chaque screening
- **Statistiques**: Performances et métriques

### Consulter la Base de Données

```bash
sqlite3 data/screener.db

# Voir les alertes récentes
SELECT symbol, current_price, recommendation, alert_date
FROM stock_alerts
ORDER BY alert_date DESC
LIMIT 10;

# Statistiques
SELECT COUNT(*), recommendation
FROM stock_alerts
GROUP BY recommendation;
```

## Logs

Les logs sont sauvegardés dans `logs/screener.log` avec:

- Logs colorés dans la console
- Logs détaillés dans le fichier
- Niveaux: DEBUG, INFO, WARNING, ERROR, CRITICAL

## Dépannage

### Problème: Pas de données récupérées

**Solution**: Vérifiez votre connexion Internet. yfinance nécessite une connexion active.

### Problème: Telegram ne fonctionne pas

**Solutions**:
1. Vérifiez que le token et chat_id sont corrects dans `.env`
2. Testez avec `python main.py test`
3. Assurez-vous d'avoir envoyé `/start` au bot

### Problème: Trop de stocks, execution lente

**Solutions**:
1. Réduisez `MAX_STOCKS` dans `config/settings.py`
2. Augmentez `MIN_MARKET_CAP` ou `MIN_DAILY_VOLUME`
3. Désactivez certains marchés dans `MARKETS`

### Problème: Erreurs de calcul EMA

**Solution**: Assurez-vous d'avoir assez de données historiques. Le système a besoin d'au moins 62 périodes pour calculer l'EMA 62.

## Améliorations Futures

- [ ] Support des cryptomonnaies
- [ ] Interface web de visualisation
- [ ] Backtesting des signaux
- [ ] Machine Learning pour optimisation
- [ ] API REST pour intégrations externes
- [ ] Alertes en temps réel (websockets)
- [ ] Support de plus d'indicateurs techniques
- [ ] Gestion de portefeuille intégrée

## Performance

Le screening de ~500 actions prend environ 3-5 minutes selon:
- Vitesse de connexion
- Nombre de workers (`MAX_WORKERS`)
- Filtres appliqués

## Sécurité

- Ne commitez JAMAIS le fichier `.env`
- Utilisez des App Passwords pour Gmail
- Limitez les permissions du bot Telegram
- Sauvegardez régulièrement `data/screener.db`

## Support

Pour toute question ou problème:
1. Vérifiez les logs dans `logs/screener.log`
2. Consultez cette documentation
3. Testez avec un symbole unique: `python main.py screen --symbol AAPL`

## Licence

Ce projet est fourni tel quel pour usage personnel. Utilisez-le à vos propres risques. Ce n'est PAS un conseil financier.

## Avertissement

Ce système est fourni à des fins éducatives et d'information uniquement. Il ne constitue en aucun cas un conseil en investissement. Faites toujours vos propres recherches avant d'investir.

---

**Bonne analyse! 📈**
