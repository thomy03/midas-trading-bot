# 🐧 Guide de Lancement WSL - Market Screener

## ✅ Installation Terminée

Votre environnement WSL est déjà configuré:
- ✅ Python 3.11 installé
- ✅ Environnement virtuel créé (`venv`)
- ✅ Toutes les dépendances installées
- ✅ Fichier `.env` créé

## 📋 Configuration Requise AVANT le Lancement

### 1️⃣ Configuration Telegram (OBLIGATOIRE)

Le système envoie les alertes via Telegram. Vous devez configurer:

**Étape A: Créer un Bot Telegram**
1. Ouvrez Telegram sur votre téléphone/ordinateur
2. Cherchez `@BotFather`
3. Envoyez: `/newbot`
4. Suivez les instructions (choisissez un nom pour votre bot)
5. **Notez le TOKEN** reçu (format: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

**Étape B: Obtenir votre Chat ID**
1. Cherchez `@userinfobot` sur Telegram
2. Envoyez: `/start`
3. **Notez votre Chat ID** (format: `123456789`)

**Étape C: Configurer le fichier .env**

Éditez le fichier `.env`:
```bash
nano .env
```

Remplacez les lignes suivantes:
```env
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

Sauvegardez: `Ctrl+O` puis `Enter`, puis `Ctrl+X` pour quitter.

### 2️⃣ Configuration de l'Heure (OPTIONNEL)

Par défaut, le screening se lance à **8h00 Europe/Paris**.

Pour changer l'heure, éditez `config/settings.py`:
```bash
nano config/settings.py
```

Modifiez la ligne 96:
```python
DAILY_REPORT_TIME = time(9, 0)  # Pour 9h00
```

Et/ou la ligne 99 pour le fuseau horaire:
```python
TIMEZONE = 'Europe/Paris'  # Ou votre timezone
```

## 🚀 Lancement du Screening Automatique

### Option 1: Mode Screen (RECOMMANDÉ pour 24/7)

Cette méthode permet de détacher le processus et de fermer le terminal.

```bash
# Naviguer vers le projet
cd /mnt/c/Users/tkado/Documents/Tradingbot_V3

# Activer l'environnement virtuel
source venv/bin/activate

# Créer une session screen nommée "screener"
screen -S screener

# Lancer le scheduler
python main.py schedule

# Détacher la session: Appuyez sur Ctrl+A puis D
```

Le processus tourne maintenant en arrière-plan! ✅

**Commandes utiles pour Screen:**
```bash
# Réattacher à la session pour voir les logs
screen -r screener

# Lister toutes les sessions screen
screen -ls

# Tuer la session (si besoin)
screen -S screener -X quit
```

### Option 2: Mode Direct (pour tests)

Pour tester sans screen:
```bash
cd /mnt/c/Users/tkado/Documents/Tradingbot_V3
source venv/bin/activate
python main.py schedule
```

**CTRL+C** pour arrêter.

## 🧪 Tests Avant le Lancement 24/7

### 1. Test des Notifications Telegram
```bash
cd /mnt/c/Users/tkado/Documents/Tradingbot_V3
source venv/bin/activate
python main.py test
```

Vous devriez recevoir un message de test sur Telegram.

### 2. Test d'un Screening Manuel (optionnel)
```bash
python main.py screen --symbol AAPL
```

Analyse une action spécifique (Apple).

### 3. Screening Complet Unique
```bash
python main.py run
```

Lance un screening complet une seule fois (peut prendre 3-5 minutes).

## 📊 Consultation des Résultats

### Via le Dashboard Web

Ouvrez un **NOUVEAU terminal WSL** (sans arrêter le scheduler):
```bash
cd /mnt/c/Users/tkado/Documents/Tradingbot_V3
source venv/bin/activate
streamlit run dashboard.py --server.address 0.0.0.0
```

Puis ouvrez votre navigateur Windows à: **http://localhost:8501**

### Via Telegram

Vous recevrez automatiquement:
- Un rapport quotidien à 8h (ou l'heure configurée)
- Les alertes en temps réel quand détectées

## 📝 Logs et Dépannage

**Voir les logs:**
```bash
tail -f logs/screener.log
```

**Vérifier la base de données:**
```bash
python main.py alerts --days 7
```

Affiche les alertes des 7 derniers jours.

## ⚠️ Points Importants

1. **Votre PC doit rester allumé** pour que le scheduler fonctionne
2. **WSL doit rester actif** (ne pas arrêter WSL)
3. **Connexion Internet** requise pour récupérer les données de marché
4. **Telegram configuré** sinon aucune notification ne sera envoyée

## 🛑 Arrêter le Scheduler

```bash
# Réattacher à la session screen
screen -r screener

# Appuyer sur Ctrl+C pour arrêter

# Ou tuer directement la session
screen -S screener -X quit
```

## 🔄 Commandes Récapitulatives

```bash
# LANCER LE SCHEDULER 24/7
cd /mnt/c/Users/tkado/Documents/Tradingbot_V3
source venv/bin/activate
screen -S screener
python main.py schedule
# Ctrl+A puis D pour détacher

# CONSULTER LE DASHBOARD (nouveau terminal)
cd /mnt/c/Users/tkado/Documents/Tradingbot_V3
source venv/bin/activate
streamlit run dashboard.py --server.address 0.0.0.0
# http://localhost:8501 dans le navigateur

# VÉRIFIER LES LOGS
tail -f logs/screener.log

# RÉATTACHER AU SCHEDULER
screen -r screener
```

## 📞 Support

En cas de problème:
1. Vérifiez les logs: `tail -f logs/screener.log`
2. Testez Telegram: `python main.py test`
3. Vérifiez que l'environnement est activé: `which python` doit afficher le chemin vers `venv/bin/python`

---

**Vous êtes prêt! 🚀** Configurez Telegram, testez, puis lancez le scheduler en mode screen.
