# TradingBot V4.1 - Documentation Complète

*Dernière mise à jour: Décembre 2024*

---

## Table des Matières

1. [Vue d'Ensemble](#1-vue-densemble)
2. [Architecture Technique](#2-architecture-technique)
3. [Composants Principaux](#3-composants-principaux)
4. [Flux de Données](#4-flux-de-données)
5. [Configuration](#5-configuration)
6. [APIs et Services Externes](#6-apis-et-services-externes)
7. [Interface Utilisateur](#7-interface-utilisateur)
8. [Stratégie de Trading](#8-stratégie-de-trading)
9. [Sécurité (Guardrails)](#9-sécurité-guardrails)
10. [Installation et Déploiement](#10-installation-et-déploiement)

---

## 1. Vue d'Ensemble

### 1.1 Description

TradingBot V4.1 est un système de trading automatisé qui combine:
- **Analyse technique** (EMAs, RSI, trendlines)
- **Intelligence artificielle** (LLMs via OpenRouter/Gemini et Grok)
- **Analyse sociale** (Reddit, StockTwits, X/Twitter)
- **Mémoire des trades** (RAG avec ChromaDB)
- **Exécution automatisée** (Interactive Brokers)

### 1.2 Objectifs

| Objectif | Description |
|----------|-------------|
| Autonomie | Fonctionnement 24/7 avec modes adaptés (pré-market, market, after-hours) |
| Intelligence | Utilisation de LLMs pour l'analyse de sentiment et la découverte de tendances |
| Sécurité | Guardrails stricts pour limiter les pertes |
| Apprentissage | Mémoire RAG pour apprendre des trades passés |

### 1.3 Capital et Risque

- **Capital initial**: 1500€ (configurable)
- **Risque par trade**: 2-4% selon la market cap
- **Max daily loss**: 3%
- **Max drawdown**: 15%

---

## 2. Architecture Technique

### 2.1 Structure des Dossiers

```
TradingBot_V3/
├── webapp.py                    # Dashboard NiceGUI (interface principale)
├── dashboard.py                 # Dashboard Streamlit (legacy)
├── config/
│   └── settings.py              # Configuration globale
├── src/
│   ├── agents/                  # Système agentique V4.1
│   │   ├── orchestrator.py      # MarketAgent - Cerveau central
│   │   ├── guardrails.py        # Sécurités hard-coded
│   │   ├── state.py             # État persistant
│   │   ├── reasoning_engine.py  # Moteur de décision 4 piliers
│   │   ├── strategy_composer.py # Composition de stratégies LLM
│   │   └── pillars/             # Les 4 piliers d'analyse
│   │       ├── technical_pillar.py
│   │       ├── fundamental_pillar.py
│   │       ├── sentiment_pillar.py
│   │       └── news_pillar.py
│   ├── intelligence/            # Intelligence de marché
│   │   ├── trend_discovery.py   # Découverte de tendances (OpenRouter)
│   │   ├── social_scanner.py    # Scan Reddit/StockTwits
│   │   ├── grok_scanner.py      # Scan X/Twitter via Grok
│   │   ├── news_fetcher.py      # Récupération news (NewsAPI, ArXiv)
│   │   ├── market_intelligence.py # Analyse LLM
│   │   ├── trade_memory.py      # RAG / Mémoire des trades
│   │   ├── stock_discovery.py   # Découverte dynamique de stocks
│   │   └── report_generator.py  # Génération de rapports .md
│   ├── execution/               # Exécution des ordres
│   │   └── ibkr_executor.py     # Connexion Interactive Brokers
│   ├── screening/               # Screener V3 (fallback)
│   │   └── screener.py
│   ├── indicators/              # Indicateurs techniques
│   │   └── ema_analyzer.py
│   ├── data/                    # Données de marché
│   │   └── market_data.py       # Fetch yfinance
│   ├── database/                # Base de données
│   │   └── db_manager.py        # SQLAlchemy
│   └── utils/                   # Utilitaires
│       ├── confidence_scorer.py
│       ├── position_sizing.py
│       ├── sector_analyzer.py
│       └── ...
├── trendline_analysis/          # Module d'analyse RSI trendlines
│   ├── core/
│   │   ├── trendline_detector.py
│   │   ├── breakout_analyzer.py
│   │   └── rsi_breakout_analyzer.py
│   └── config/
│       └── settings.py
├── data/                        # Données persistantes
│   ├── trends/                  # Rapports de tendances (JSON)
│   ├── reports/                 # Rapports .md générés
│   ├── tickers/                 # Listes de tickers (JSON)
│   ├── vector_store/            # ChromaDB (RAG)
│   └── portfolio.json           # Positions ouvertes
└── tests/                       # Tests
    └── integration/
```

### 2.2 Stack Technologique

| Composant | Technologie |
|-----------|-------------|
| Language | Python 3.11+ |
| Web Framework | NiceGUI (+ Streamlit legacy) |
| Database | SQLAlchemy + ChromaDB (RAG) |
| Data | yfinance, pandas, numpy |
| LLMs | OpenRouter (Gemini), Grok (xAI) |
| Broker | Interactive Brokers (ib_insync) |
| Embeddings | sentence-transformers |
| Notifications | Telegram Bot API |

---

## 3. Composants Principaux

### 3.1 MarketAgent (Orchestrateur)

Le cerveau central du système. Coordonne toutes les phases:

```python
# Phases quotidiennes
DISCOVERY   = "06:00 ET"  # Scan social, news, stocks trending
ANALYSIS    = "07:00 ET"  # Analyse LLM, identification des narratifs
TRADING     = "09:30-16:00 ET"  # Scans toutes les 15 min, exécution
AUDIT       = "20:00 ET"  # Analyse des trades, stockage RAG, règles apprises
```

**Fichier**: `src/agents/orchestrator.py`

### 3.2 ReasoningEngine (4 Piliers)

Système de scoring sur 100 points répartis en 4 piliers:

| Pilier | Poids | Description |
|--------|-------|-------------|
| Technical | 25% | EMAs, RSI, MACD, ADX, Supports/Résistances |
| Fundamental | 25% | PE ratio, EPS growth, debt/equity |
| Sentiment | 25% | Reddit, StockTwits, Grok/X sentiment |
| News | 25% | NewsAPI, Alpha Vantage, analyse LLM |

**Fichiers**: `src/agents/pillars/*.py`

### 3.3 TrendDiscovery

Découverte automatique des tendances émergentes:

- **Quantitatif**: Momentum sectoriel, anomalies de volume, breadth divergences
- **LLM**: Analyse des news par thème, identification de narratifs émergents

**Thèmes surveillés**:
- AI_Revolution, Quantum_Computing, Space_Economy
- Obesity_Drugs, Nuclear_Renaissance, EV_Transition
- Cybersecurity, Reshoring, Crypto_Bull, Rate_Pivot

**Fichier**: `src/intelligence/trend_discovery.py`

### 3.4 SocialScanner

Scan des réseaux sociaux:

- **StockTwits**: Messages, sentiment, symboles mentionnés
- **Reddit**: r/wallstreetbets, r/stocks, r/investing, r/options

**Fichier**: `src/intelligence/social_scanner.py`

### 3.5 GrokScanner

Analyse X/Twitter via l'API Grok (xAI):

- Tendances financières en temps réel
- Analyse de sentiment par symbole
- Identification des catalyseurs

**Fichier**: `src/intelligence/grok_scanner.py`

### 3.6 TradeMemory (RAG)

Mémoire à long terme des trades:

- **Stockage**: ChromaDB (vector database)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Requêtes**: Recherche de trades similaires par contexte
- **Apprentissage**: Extraction de patterns, génération de règles

**Fichier**: `src/intelligence/trade_memory.py`

### 3.7 IBKRExecutor

Exécution des ordres via Interactive Brokers:

- Connexion TWS/IB Gateway
- Ordres LIMIT, MARKET, STOP
- Gestion des positions
- Paper trading (port 7497) / Live (port 7496)

**Fichier**: `src/execution/ibkr_executor.py`

---

## 4. Flux de Données

### 4.1 Cycle Quotidien

```
┌──────────────────────────────────────────────────────────────┐
│                    PHASE DISCOVERY (06:00 ET)                 │
├──────────────────────────────────────────────────────────────┤
│  SocialScanner ──┐                                           │
│  GrokScanner ────┼──► Watchlist dynamique (max 200 symboles) │
│  StockDiscovery ─┘                                           │
│  VolumeAnomalies ─► Top movers détectés                      │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    PHASE ANALYSIS (07:00 ET)                  │
├──────────────────────────────────────────────────────────────┤
│  TrendDiscovery ──► Analyse LLM des news (OpenRouter/Gemini) │
│                  ──► Identification narratifs émergents      │
│                  ──► Détermination régime de marché          │
│                                                              │
│  Output: focus_symbols (30 max), market_sentiment, trends    │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│              PHASE TRADING (09:30-16:00 ET)                   │
├──────────────────────────────────────────────────────────────┤
│  Toutes les 15 minutes:                                      │
│                                                              │
│  1. ReasoningEngine analyse chaque focus_symbol              │
│     ├── Technical Pillar ──► Score technique                 │
│     ├── Fundamental Pillar ──► Score fondamental             │
│     ├── Sentiment Pillar ──► Score sentiment                 │
│     └── News Pillar ──► Score actualité                      │
│                                                              │
│  2. TradeMemory consulté pour trades similaires              │
│                                                              │
│  3. Guardrails valident le trade                             │
│                                                              │
│  4. IBKRExecutor exécute si approuvé                         │
│                                                              │
│  5. Notifications Telegram si score >= 70                    │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    PHASE AUDIT (20:00 ET)                     │
├──────────────────────────────────────────────────────────────┤
│  1. Récupération trades fermés du jour                       │
│  2. Stockage dans TradeMemory (RAG)                          │
│  3. Extraction de patterns (PatternExtractor)                │
│  4. Génération de nouvelles règles                           │
│  5. Sauvegarde learned_guidelines.json                       │
│  6. Si pertes: StrategyComposer propose améliorations        │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Flux de Décision

```
Signal détecté (score >= 55)
        │
        ▼
┌───────────────────┐
│ Position sizing   │  Capital * (2-4% selon market cap)
│ Stop-loss calculé │  Basé sur RSI peaks
└─────────┬─────────┘
          │
          ▼
┌───────────────────────────────────────────────────┐
│                   GUARDRAILS                       │
├───────────────────────────────────────────────────┤
│ □ Max position size < 10% capital                 │
│ □ Daily loss < 3%                                 │
│ □ Drawdown < 15%                                  │
│ □ Trades today < 5                                │
│ □ Consecutive losses < 3                          │
└─────────┬─────────────────────────────────────────┘
          │
          ▼ Validation OK?
    ┌─────┴─────┐
    │           │
   OUI         NON
    │           │
    ▼           ▼
EXECUTE     REJECT
(IBKR)      (log reason)
```

---

## 5. Configuration

### 5.1 Variables d'Environnement (.env)

```bash
# Capital
TRADING_CAPITAL=1500

# OpenRouter (Intelligence LLM)
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=google/gemini-3-flash-preview

# Grok (Sentiment X/Twitter)
GROK_API_KEY=xai-...
GROK_MODEL=grok-4-1-fast-reasoning

# News APIs
NEWSAPI_KEY=...
ALPHAVANTAGE_KEY=...

# Telegram
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# Interactive Brokers
IBKR_HOST=127.0.0.1
IBKR_PORT=7497  # 7497=Paper, 7496=Live
IBKR_CLIENT_ID=1
```

### 5.2 Configuration Guardrails

```python
# src/agents/guardrails.py
MAX_DAILY_LOSS_PCT = 0.03      # 3% perte max/jour
MAX_POSITION_PCT = 0.10        # 10% max par position
MAX_DRAWDOWN_PCT = 0.15        # 15% drawdown → Pause
MAX_TRADES_PER_DAY = 5         # 5 trades max/jour
AUTO_TRADE_THRESHOLD = 0.05    # < 5% capital = auto
```

---

## 6. APIs et Services Externes

### 6.1 OpenRouter (LLM Intelligence)

- **Usage**: Analyse de news, identification de tendances, composition de stratégies
- **Modèle**: google/gemini-3-flash-preview (ou claude-3-haiku)
- **Endpoint**: https://openrouter.ai/api/v1/chat/completions
- **Coût**: ~$0.001 / 1K tokens (Gemini Flash)

### 6.2 Grok (xAI)

- **Usage**: Analyse sentiment X/Twitter
- **Modèle**: grok-4-1-fast-reasoning
- **Coût**: ~$25/mois
- **Avantage**: Accès temps réel à X/Twitter

### 6.3 NewsAPI

- **Usage**: Récupération d'articles de news
- **Limite gratuite**: 100 requêtes/jour
- **Endpoint**: https://newsapi.org/v2/everything

### 6.4 Alpha Vantage

- **Usage**: News avec sentiment, données financières
- **Limite gratuite**: 5 requêtes/minute
- **Endpoint**: https://www.alphavantage.co/query

### 6.5 Interactive Brokers

- **Usage**: Exécution des ordres
- **Prérequis**: TWS ou IB Gateway installé
- **API**: ib_insync (Python wrapper)
- **Ports**: 7497 (Paper), 7496 (Live)

### 6.6 Telegram

- **Usage**: Notifications en temps réel
- **Seuil**: Score >= 70 pour notification
- **Format**: Signal, symbole, score, prix, stop-loss

---

## 7. Interface Utilisateur

### 7.1 Dashboard NiceGUI (webapp.py)

Accessible sur http://localhost:8080

**Pages disponibles**:

| Page | Description |
|------|-------------|
| Dashboard | Vue d'ensemble, contrôle du bot, stats |
| Raisonnement | Chain of Thought, flux de pensées en temps réel |
| Alertes | Liste des alertes détectées |
| Rapports | Génération et visualisation des rapports .md |
| Configuration | Capital, Telegram, IBKR |
| Logs | Logs complets du système |

### 7.2 Fonctionnalités Clés

- **Mode Market**: Indicateur temps réel (MARKET, PRE_MARKET, AFTER_HOURS, WEEKEND)
- **Chain of Thought**: Visualisation du raisonnement par catégorie
- **Génération Rapports**: Export .md avec tous les résultats
- **Heartbeat**: Messages toutes les 5 minutes pendant l'attente

---

## 8. Stratégie de Trading

### 8.1 Signaux

| Signal | Score | Action |
|--------|-------|--------|
| STRONG_BUY | >= 80 | Exécution auto (< 5% capital) |
| BUY | 55-79 | Notification puis exécution |
| HOLD | 30-54 | Pas d'action |
| SELL | < 30 | Vente si position ouverte |

### 8.2 Position Sizing

```python
# Risque variable selon market cap
MEGA_CAP  (> 200B): 4% du capital
LARGE_CAP (50-200B): 3% du capital
MID_CAP   (10-50B): 2.5% du capital
SMALL_CAP (2-10B): 2% du capital
MICRO_CAP (< 2B): 1.5% du capital
```

### 8.3 Stop-Loss

- **Méthode**: Basé sur le plus bas pic RSI de l'oblique
- **Calcul**: Prix au moment du pic RSI le plus bas
- **Trailing**: Optionnel (configurable)

### 8.4 Conditions de Sortie

1. **stop_loss**: Prix touche le stop-loss
2. **take_profit**: Gain >= target (ex: 15%)
3. **trailing_stop**: Stop suiveur déclenché
4. **max_hold_period**: Durée max atteinte (ex: 60 jours)
5. **signal_reversal**: Signal de vente détecté

---

## 9. Sécurité (Guardrails)

### 9.1 Limites Hard-Coded

Ces limites NE PEUVENT PAS être modifiées par l'IA:

```python
MAX_DAILY_LOSS_PCT = 0.03      # Kill switch à -3%/jour
MAX_POSITION_PCT = 0.10        # Max 10% par position
MAX_DRAWDOWN_PCT = 0.15        # Pause à -15% drawdown
MIN_POSITION_SIZE = 50         # Min 50€ par trade
MAX_TRADES_PER_DAY = 5         # Max 5 trades/jour
```

### 9.2 Kill Switch

Déclenché automatiquement si:
- Perte journalière >= 3%
- Drawdown >= 15%
- 3 pertes consécutives

**Action**: Pause de 24h, notification Telegram

### 9.3 Validation Trade

Chaque trade passe par 3 niveaux:

| Niveau | Condition | Action |
|--------|-----------|--------|
| APPROVED | < 5% capital | Exécution auto |
| NEEDS_NOTIFICATION | 5-10% capital | Notification + exécution |
| NEEDS_APPROVAL | > 10% capital | Attente validation manuelle |

---

## 10. Installation et Déploiement

### 10.1 Prérequis

- Python 3.11+
- TWS ou IB Gateway (pour trading réel)
- Compte OpenRouter
- Compte Grok (optionnel)

### 10.2 Installation

```bash
# Cloner le projet
git clone <repo-url>
cd TradingBot_V3

# Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt

# Configurer les variables d'environnement
copy .env.example .env
# Éditer .env avec vos clés API
```

### 10.3 Lancement

```bash
# Dashboard NiceGUI (recommandé)
python webapp.py

# Ou Dashboard Streamlit (legacy)
streamlit run dashboard.py

# Ou CLI directe
python -m src.agents.orchestrator --daemon
```

### 10.4 Tests

```bash
# Tous les tests
python -m pytest tests/

# Tests d'intégration spécifiques
python -m pytest tests/integration/test_trend_discovery.py
python -m pytest tests/integration/test_backtesting.py
```

---

## Annexes

### A. Secteurs Dynamiques (V4.2)

**NOUVEAU: Les secteurs ne sont plus hardcodés!**

Le système utilise maintenant `DynamicSectors` qui:
1. Récupère automatiquement les secteurs depuis l'API NASDAQ
2. Identifie les thèmes émergents via le LLM (Gemini)
3. Met à jour le cache toutes les 24h
4. Conserve un fallback hardcodé en cas d'erreur API

**Sources de données:**
| Source | Type | Fréquence |
|--------|------|-----------|
| API NASDAQ | Tous les stocks NASDAQ/NYSE/AMEX | Quotidien |
| LLM (Gemini) | Thèmes émergents (AI, Quantum, etc.) | Quotidien |
| Fallback | 5 secteurs hardcodés | Backup uniquement |

**Thèmes identifiés dynamiquement par le LLM:**
- AI_Semiconductors, Obesity_Drugs, Space_Economy
- Nuclear_Renaissance, Quantum_Computing
- Crypto_Adjacent, Defense_Tech, Clean_Energy

**Fichier:** `src/intelligence/dynamic_sectors.py`

### B. Subreddits Surveillés

- r/wallstreetbets
- r/stocks
- r/investing
- r/stockmarket
- r/options
- r/pennystocks
- r/ValueInvesting
- r/Daytrading

### C. Changelog

**V4.1 (Décembre 2024)**
- ReasoningEngine 4 piliers
- TradeMemory (RAG) avec ChromaDB
- Dual LLM: Gemini (intelligence) + Grok (sentiment)
- Dashboard NiceGUI avec Chain of Thought
- Génération de rapports .md
- StrategyComposer pour amélioration continue

**V4.0 (Décembre 2024)**
- Architecture agentique
- Guardrails stricts
- Social/Grok scanners
- IBKR integration

**V3.x (2024)**
- RSI Trendline Breakout
- Multi-obliques (jusqu'à 3)
- Scoring confiance 0-100
- Position sizing dynamique

---

*Documentation générée automatiquement par TradingBot V4.1*
