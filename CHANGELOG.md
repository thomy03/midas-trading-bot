# Changelog

Toutes les modifications notables de ce projet seront documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

## [Non publié]

### Ajouté
- Structure modulaire `src/dashboard/` pour le dashboard Streamlit
  - Séparation en pages indépendantes
  - Composants réutilisables
  - Utilitaires de styling
- Documentation organisée dans `docs/`
  - Guide utilisateur complet
  - Archive des notes de session
  - Index de navigation
- Fichiers `REFACTORING.md` et `CHANGELOG.md`

### Modifié
- Réorganisation de la documentation
  - 18 fichiers MD déplacés vers `docs/`
  - Structure claire : user-guide/ development/ archive/

### Supprimé
- 282 MB de fichiers temporaires (`archive_temp_files/`)
- 9.3 MB de graphiques HTML de debug
- Scripts de diagnostic (`diagnostic_*.py`)
- Images PNG non utilisées

### Optimisé
- Réduction de 292 MB d'espace disque
- Structure de projet plus claire
- Meilleure organisation du code

## [1.0.0] - Date du premier commit

### Ajouté
- Système de screening EMA initial
- Détection de supports/résistances via EMAs (24, 38, 62)
- Analyse de trendlines RSI
- Confirmation duale (RSI + Prix)
- Dashboard Streamlit interactif
- Base de données SQLite pour l'historique
- Alertes Telegram/Email
- Screening multi-marchés (NASDAQ, S&P 500, Europe, Asie)
- Module `trendline_analysis/` pour l'analyse RSI avancée
- Configuration centralisée via `config/settings.py`

### Fonctionnalités principales
- `main.py` - Point d'entrée du screening automatique
- `dashboard.py` - Interface web Streamlit
- `src/screening/screener.py` - Logique de screening
- `src/indicators/ema_analyzer.py` - Analyse EMA
- `src/data/market_data.py` - Récupération données marché
- `trendline_analysis/core/` - Détection trendlines et breakouts RSI

## Types de changements

- `Ajouté` pour les nouvelles fonctionnalités
- `Modifié` pour les changements aux fonctionnalités existantes
- `Déprécié` pour les fonctionnalités bientôt supprimées
- `Supprimé` pour les fonctionnalités maintenant retirées
- `Corrigé` pour les corrections de bugs
- `Sécurité` pour les vulnérabilités corrigées
- `Optimisé` pour les améliorations de performance
