# Tradingbot_V3 Documentation

Bienvenue dans la documentation du Tradingbot_V3, un système de screening automatique de marchés financiers.

## Navigation Rapide

### 📚 Guides Utilisateur
- [Guide de Démarrage Rapide](user-guide/GUIDE_DEMARRAGE_RAPIDE.md) - Commencez ici !
- [Configuration du Dashboard](user-guide/DASHBOARD.md) - Interface web Streamlit
- [Analyse de Trendlines](user-guide/GUIDE_TRENDLINE_ANALYSIS.md) - Détection RSI
- [Paramètres Universels](user-guide/PARAMETRES_UNIVERSELS.md) - Configuration avancée
- [Visualisations Disponibles](user-guide/VISUALISATIONS_DISPONIBLES.md) - Types de graphiques
- [Configuration WSL](user-guide/WSL_SETUP.md) - Windows Subsystem for Linux
- [Guide de Test](user-guide/TEST_GUIDE.md) - Tests et validation
- [GitHub Push](user-guide/GITHUB_PUSH.md) - Déploiement

### 🔧 Documentation Développeur
- [Architecture](development/) - À venir
- [Tests](development/) - À venir
- [API Reference](development/) - À venir

### 📦 Archive
Anciennes notes de session et documentation historique disponibles dans [archive/](archive/).

## Vue d'Ensemble

Le Tradingbot_V3 offre :
- ✅ Screening de 700+ actions (NASDAQ, S&P 500, Europe, Asie)
- ✅ Détection de supports/résistances via EMAs (24/38/62)
- ✅ Analyse de cassures de trendlines RSI
- ✅ Confirmation duale RSI + Prix
- ✅ Alertes Telegram/Email automatiques
- ✅ Dashboard web interactif
- ✅ Historique en base SQLite

## Installation Rapide

```bash
# Cloner le projet
git clone <your-repo-url>
cd Tradingbot_V3

# Installer les dépendances
pip install -r requirements.txt

# Configurer l'environnement
cp .env.example .env
# Éditer .env avec vos credentials

# Lancer le dashboard
streamlit run dashboard.py

# OU lancer le screening
python main.py
```

## Support

Pour toute question ou problème :
1. Consultez d'abord les guides utilisateur
2. Vérifiez les [problèmes résolus](archive/PROBLEMES_RESOLUS.md)
3. Créez une issue GitHub

## Licence

Voir le fichier LICENSE à la racine du projet.
