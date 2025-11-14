# 🚀 DÉMARRAGE RAPIDE - Market Screener WSL

## ⚡ En 3 Étapes Simples

### 1️⃣ Configurer Telegram (2 minutes)

**Créer votre bot:**
1. Ouvrez Telegram → Cherchez `@BotFather`
2. Envoyez `/newbot` et suivez les instructions
3. **Copiez le TOKEN** reçu

**Obtenir votre Chat ID:**
1. Cherchez `@userinfobot` sur Telegram
2. Envoyez `/start`
3. **Copiez le Chat ID** affiché

**Configurer le projet:**
```bash
cd /mnt/c/Users/tkado/Documents/Tradingbot_V3
nano .env
```

Remplacez:
```
TELEGRAM_BOT_TOKEN=VOTRE_TOKEN_ICI
TELEGRAM_CHAT_ID=VOTRE_CHAT_ID_ICI
```

Sauvegardez: `Ctrl+O` → `Enter` → `Ctrl+X`

### 2️⃣ Tester Telegram

```bash
source venv/bin/activate
python main.py test
```

Vous devriez recevoir un message de test sur Telegram! ✅

### 3️⃣ Lancer le Scheduler

**Méthode Facile (recommandé):**
```bash
./start_scheduler.sh
```
Choisissez l'option 1 (lancer avec screen).

**Méthode Manuelle:**
```bash
source venv/bin/activate
screen -S screener
python main.py schedule
# Appuyez sur Ctrl+A puis D pour détacher
```

**C'est tout! 🎉** Le screening tournera automatiquement tous les jours à 8h.

---

## 📋 Commandes Utiles

| Action | Commande |
|--------|----------|
| **Lancer le scheduler** | `./start_scheduler.sh` |
| **Voir le dashboard** | `./start_dashboard.sh` ou `streamlit run dashboard.py` |
| **Tester Telegram** | `python main.py test` |
| **Screening manuel** | `python main.py run` |
| **Voir les logs** | `screen -r screener` |
| **Arrêter le scheduler** | `screen -S screener -X quit` |

---

## ⚙️ Configuration (Optionnel)

**Changer l'heure du screening:**
```bash
nano config/settings.py
```
Modifiez la ligne 96:
```python
DAILY_REPORT_TIME = time(9, 0)  # Pour 9h
```

**Fuseau horaire:**
Ligne 99:
```python
TIMEZONE = 'Europe/Paris'  # Votre timezone
```

---

## 📁 Documentation Complète

- **WSL_SETUP.md** - Guide complet WSL avec tous les détails
- **README.md** - Documentation générale du projet
- **QUICKSTART.md** - Guide de démarrage rapide
- **DASHBOARD.md** - Guide du dashboard web

---

## 🆘 Problèmes Courants

**Pas de notification Telegram?**
→ Vérifiez `.env` et testez avec `python main.py test`

**Le scheduler ne démarre pas?**
→ Vérifiez que l'environnement est activé: `source venv/bin/activate`

**Imports échouent?**
→ Vérifiez Python 3.11: `python --version`

**Session screen introuvable?**
→ Listez les sessions: `screen -ls`

---

**Support:** Consultez les logs avec `tail -f logs/screener.log`
