# Guide de Démarrage Rapide

## 🚀 Installation en 5 Minutes

### 1. Installer Python

Assurez-vous d'avoir Python 3.8+ installé:

```bash
python --version
```

### 2. Installer les Dépendances

```bash
pip install -r requirements.txt
```

### 3. Configurer Telegram

1. Ouvrez Telegram
2. Cherchez `@BotFather`
3. Envoyez `/newbot`
4. Suivez les instructions
5. Notez le **token** reçu

6. Cherchez `@userinfobot`
7. Envoyez `/start`
8. Notez votre **Chat ID**

### 4. Configurer les Variables

Créez le fichier `.env`:

```bash
cp .env.example .env
```

Éditez `.env` et ajoutez:

```env
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

### 5. Tester

```bash
# Tester les notifications
python main.py test

# Tester avec un symbole
python main.py screen --symbol AAPL
```

### 6. Premier Screening

```bash
# Lancer un screening complet
python main.py run
```

Vous recevrez un rapport sur Telegram! 📱

### 7. Lancer le Dashboard 🎨

```bash
# Sur Windows
start_dashboard.bat

# Sur Linux/Mac
./start_dashboard.sh
```

Le dashboard s'ouvrira dans votre navigateur à http://localhost:8501

**Interface TradingView-like avec:**
- Graphiques interactifs avec EMAs
- Zones de support/résistance visualisées
- Screening manuel en temps réel
- Historique des alertes

## 📋 Commandes Utiles

```bash
# Screening unique
python main.py run

# Mode automatique (tous les jours à 8h)
python main.py schedule

# Analyser une action spécifique
python main.py screen --symbol MSFT

# Voir les alertes récentes
python main.py alerts --days 7
```

## ⚙️ Personnalisation Rapide

### Ajouter vos Actions Favorites

Éditez `config/settings.py`:

```python
CUSTOM_SYMBOLS = [
    'AAPL', 'MSFT', 'GOOGL', 'TSLA',
    'NVDA', 'AMD', 'META', 'AMZN'
]
```

### Changer l'Heure du Rapport

Dans `config/settings.py`:

```python
DAILY_REPORT_TIME = time(7, 30)  # 7h30 du matin
```

### Ajuster les Filtres

```python
MIN_MARKET_CAP = 1000  # Minimum 1B$
MIN_DAILY_VOLUME = 1_000_000  # Minimum 1M$/jour
```

## 🎯 Exemples de Notifications

### Message d'Alerte

```
🔥 AAPL - Apple Inc.

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

## 🔧 Résolution de Problèmes

### Pas de notifications Telegram?

1. Vérifiez que le bot token est correct
2. Envoyez `/start` à votre bot sur Telegram
3. Testez: `python main.py test`

### Erreurs d'installation?

```bash
# Sur Windows, installer Visual C++
# Télécharger depuis microsoft.com

# Sur Linux
sudo apt-get install python3-dev

# Sur Mac
xcode-select --install
```

### Trop lent?

Réduisez le nombre d'actions dans `config/settings.py`:

```python
MAX_STOCKS = 300  # Au lieu de 700
```

## 📊 Comprendre les Recommandations

| Recommandation | Signification | Action Suggérée |
|---------------|---------------|-----------------|
| **STRONG_BUY** 🔥 | Prix très proche du support, signal fort | Analyser pour achat immédiat |
| **BUY** ✅ | Prix proche du support | Ajouter à la watchlist, surveiller |
| **WATCH** 👀 | Prix s'approche du support | Observer l'évolution |
| **OBSERVE** 📊 | EMAs alignées mais prix éloigné | Garder sur le radar |

## 🎓 Stratégie en Bref

1. Le système analyse les EMAs (24, 38, 62)
2. Détecte les croisements = zones de support
3. Quand le prix revient sur le support + EMAs alignées = Signal d'achat
4. Analyse Weekly d'abord, puis Daily si besoin

## 📱 Mode Automatique

Pour laisser tourner 24/7:

```bash
# Sur Linux/Mac avec screen
screen -S screener
python main.py schedule
# Ctrl+A puis D pour détacher

# Sur Windows avec Task Scheduler
# Créer une tâche planifiée pointant vers main.py
```

## 🐳 Docker (Avancé)

Un Dockerfile sera ajouté prochainement pour faciliter le déploiement!

## 🆘 Besoin d'Aide?

1. Consultez `README.md` pour la documentation complète
2. Vérifiez les logs dans `logs/screener.log`
3. Testez avec: `python tests/test_basic.py`

## ✅ Checklist de Vérification

- [ ] Python 3.8+ installé
- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] Bot Telegram créé
- [ ] Fichier `.env` configuré
- [ ] Test des notifications réussi (`python main.py test`)
- [ ] Premier screening exécuté (`python main.py run`)

Vous êtes prêt! 🎉

---

**Note**: Ce n'est PAS un conseil financier. Faites toujours vos propres recherches avant d'investir.
