# 🏛️ MIDAS - Multi-Intelligence Decision & Analysis System

Système autonome d'analyse et de trading utilisant une architecture multi-piliers avec intelligence artificielle.

## Architecture

### 🎯 Les 5 Piliers d'Analyse

Chaque signal est évalué par 5 piliers indépendants qui votent ensemble :

| Pilier | Rôle | Sources |
|--------|------|---------|
| **📊 Technical** | Analyse technique (EMAs, patterns, S/R) | Prix, volumes, indicateurs |
| **📈 Fundamental** | Santé financière, valorisation | Ratios, earnings, croissance |
| **📰 News** | Actualités et catalyseurs | News feeds, SEC filings |
| **💬 Sentiment** | Sentiment marché et social | Social media, options flow |
| **🤖 ML** | Patterns et prédictions ML | Modèles entraînés, features |

### 🔍 Grok Scanner (xAI)

Scanner intelligent autonome utilisant l'API Grok pour découvrir les opportunités sur X/Twitter :

- **Discover Phase** : Grok identifie ce qui bouge (pas de queries fixes)
- **Deep Dive** : Analyse approfondie automatique (pourquoi, qui, catalyseur)
- **Chain of Thought** : Recherches en cascade (NVDA → AMD, AVGO, TSM)
- **Memory & Feedback** : Mémorise ce qui a marché pour s'améliorer

### 🧠 Intelligence Layer

- **Attention Manager** : Gère les priorités et le focus
- **Market Context** : Comprend le régime de marché actuel
- **Narrative Generator** : Génère des analyses lisibles
- **Trade Memory** : Historique et apprentissage des trades

### ⚙️ Agents

- **Orchestrator** : Coordonne tous les agents
- **Live Loop** : Boucle de trading temps réel
- **Nightly Auditor** : Audit quotidien des performances
- **Strategy Evolver** : Fait évoluer les stratégies automatiquement
- **Guardrails** : Limites de risque et protections

## Quick Start

```bash
# Installation
pip install -r requirements.txt

# Configuration
cp .env.example .env
# Éditer .env avec vos API keys

# Lancer le screening
python -m src.screening.scanner

# Lancer la webapp
python webapp.py
```

## Configuration Requise

- Python 3.10+
- API Keys : Polygon, Alpha Vantage, Grok (xAI), Telegram

## Structure

```
src/
├── agents/           # Agents autonomes
│   ├── pillars/      # Les 5 piliers d'analyse
│   ├── orchestrator  # Coordination
│   └── live_loop     # Trading temps réel
├── intelligence/     # Couche IA
│   ├── grok_scanner  # Scanner X/Twitter
│   ├── attention_*   # Gestion attention
│   └── narrative_*   # Génération texte
├── screening/        # Scanning de marché
├── execution/        # Exécution des trades
└── dashboard/        # Interface web
```

## License

Private - All rights reserved
