"""
Configuration settings for the Market Screener
"""
import os
from datetime import time
from typing import List

# ===========================
# MARKET SCREENING PARAMETERS
# ===========================

# EMA Periods
EMA_PERIODS = [24, 38, 62]

# Timeframes to analyze
TIMEFRAMES = {
    'weekly': '1wk',
    'daily': '1d'
}

# Support/Resistance zone tolerance (%)
ZONE_TOLERANCE = 8.0  # Assoupli de 5% à 8% pour détecter plus de signaux

# How many days back to look for EMA crossovers
LOOKBACK_DAYS_WEEKLY = 260  # ~1 year of weekly data
LOOKBACK_DAYS_DAILY = 365   # ~1 year of daily data

# Maximum age of crossover to be considered relevant (in weeks for weekly, days for daily)
# AUGMENTÉ pour garder les anciens supports valides (comme dans screenshot TSLA à $290)
MAX_CROSSOVER_AGE_WEEKLY = 104  # weeks (~2 ans) - Les anciens crossovers restent des supports valides
MAX_CROSSOVER_AGE_DAILY = 365   # days (1 an) - Les anciens crossovers restent des supports valides

# ===========================
# STOCK FILTERING CRITERIA
# ===========================

# Minimum market cap criteria (in millions USD) - differentiated by market
MIN_MARKET_CAP_NASDAQ = 100  # 100M$ for NASDAQ stocks
MIN_MARKET_CAP_SP500 = 500   # 500M$ for S&P 500 stocks
MIN_MARKET_CAP_EUROPE = 500  # 500M$ for European stocks
MIN_MARKET_CAP_ASIA_ADR = 500  # 500M$ for Asian ADRs

# Legacy setting (kept for backward compatibility)
MIN_MARKET_CAP = 500  # Default 500M$

# Minimum average daily volume (in USD)
MIN_DAILY_VOLUME = 750_000  # 750k$/day

# Maximum number of stocks to screen
MAX_STOCKS = 700

# Markets to screen
MARKETS = {
    'NASDAQ': True,   # NASDAQ stocks
    'SP500': True,    # S&P 500 stocks (excluding NASDAQ)
    'EUROPE': True,   # European stocks
    'ASIA_ADR': True  # Asian ADRs
}

# Major US indices tickers (for initial stock list)
US_INDICES = [
    '^GSPC',  # S&P 500
    '^DJI',   # Dow Jones
    '^IXIC',  # Nasdaq
]

# European indices
EUROPEAN_INDICES = [
    '^STOXX50E',  # Euro Stoxx 50
    '^FTSE',      # FTSE 100
    '^GDAXI',     # DAX
    '^FCHI',      # CAC 40
]

# ===========================
# NOTIFICATION SETTINGS
# ===========================

# Telegram Bot Configuration (set in .env file)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Email Configuration (alternative to Telegram)
EMAIL_ENABLED = os.getenv('EMAIL_ENABLED', 'False').lower() == 'true'
EMAIL_FROM = os.getenv('EMAIL_FROM', '')
EMAIL_TO = os.getenv('EMAIL_TO', '')
SMTP_SERVER = os.getenv('SMTP_SERVER', '')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USERNAME = os.getenv('SMTP_USERNAME', '')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')

# ===========================
# SCHEDULING
# ===========================

# Time to send daily report (24-hour format)
DAILY_REPORT_TIME = time(8, 0)  # 8:00 AM

# Timezone
TIMEZONE = 'Europe/Paris'  # Adjust based on your location

# ===========================
# DATABASE
# ===========================

DATABASE_PATH = 'data/screener.db'

# ===========================
# LOGGING
# ===========================

LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/screener.log'

# ===========================
# CUSTOM WATCHLIST
# ===========================

# Add your custom symbols here
CUSTOM_SYMBOLS: List[str] = [
    # Example: 'AAPL', 'MSFT', 'GOOGL', 'TSLA'
]

# ===========================
# ADVANCED SETTINGS
# ===========================

# Number of threads for parallel data fetching
MAX_WORKERS = 10

# Cache duration for market data (in minutes)
CACHE_DURATION = 60

# Retry settings for API calls
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# ===========================
# TICKERS DATA SOURCE
# ===========================

# Directory containing ticker JSON files
# These files are used instead of scraping Wikipedia (more reliable)
# Update using: .\scripts\Update-Tickers.ps1
TICKERS_JSON_DIR = 'data/tickers'

# Maximum age of ticker files before warning (in days)
TICKERS_MAX_AGE_DAYS = 7

# ===========================
# POSITION SIZING
# ===========================

# Trading capital in euros
CAPITAL = 10000

# Maximum risk per trade (as decimal: 0.02 = 2%)
RISK_PER_TRADE = 0.02

# Stop-loss distance in ATR units
ATR_MULTIPLIER = 2.0

# Maximum single position as % of capital
MAX_POSITION_PCT = 0.25

# ===========================
# CONFIDENCE SCORING
# ===========================

# Score thresholds for signal classification
CONFIDENCE_THRESHOLDS = {
    'STRONG_BUY': 75,  # Score >= 75
    'BUY': 55,         # Score >= 55
    'WATCH': 35,       # Score >= 35
    'OBSERVE': 0       # Score < 35
}

# Score weights (should sum to 100)
SCORE_WEIGHTS = {
    'ema_alignment': 25,      # EMAs croissantes
    'support_proximity': 25,  # Distance au support
    'rsi_breakout': 30,       # Qualité du breakout
    'volume_confirmation': 20  # Volume relatif
}

# ===========================
# MULTI-ASSETS MARKETS
# ===========================

# Extended markets configuration
MARKETS_EXTENDED = {
    'NASDAQ': True,
    'SP500': True,
    'CRYPTO': True,    # Top cryptos (BTC, ETH, SOL, etc.)
    'CAC40': True,     # French CAC40
    'DAX': False,      # German DAX (disabled by default)
    'EUROPE': True,    # Euro Stoxx 50
    'ASIA_ADR': True   # Asian ADRs
}

# ===========================
# GROK API (xAI) - LAYER 2 SENTIMENT
# ===========================

# Enable Grok sentiment analysis (Phase 2)
GROK_ENABLED = True

# xAI API Configuration
# SECURITY: API key must be set in .env file or environment variable
GROK_API_KEY = os.getenv('GROK_API_KEY', '')
GROK_API_URL = 'https://api.x.ai/v1/chat/completions'
GROK_MODEL = 'grok-4-1-fast-reasoning'  # Fast reasoning model

# Grok request timeout (seconds)
GROK_TIMEOUT = 30

# Rate limiting (requests per minute)
GROK_RATE_LIMIT = 60

# ===========================
# OPENROUTER API - TREND DISCOVERY
# ===========================

# OpenRouter API Configuration for LLM-based trend analysis
# SECURITY: API key must be set in .env file or environment variable
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')

# Model to use for trend discovery and V4.1 intelligence
# Options: 'google/gemini-3-flash-preview' (V4.1 default), 'anthropic/claude-3-sonnet' (balanced),
#          'anthropic/claude-3-opus' (powerful), 'openai/gpt-4-turbo'
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'google/gemini-3-flash-preview')

# ===========================
# TREND DISCOVERY SETTINGS
# ===========================

# Enable automatic trend discovery
TREND_DISCOVERY_ENABLED = True

# Daily scan time (24-hour format HH:MM)
TREND_SCAN_TIME = '06:00'

# Minimum confidence for trend to be included
TREND_MIN_CONFIDENCE = 0.4

# Minimum momentum score to flag a sector
TREND_MIN_MOMENTUM = 0.2

# Days of news to analyze
TREND_NEWS_DAYS = 3

# Data directory for trend reports
TREND_DATA_DIR = 'data/trends'

# ===========================
# NEWS API KEYS
# ===========================

# NewsAPI key for news fetching
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')

# Alpha Vantage key for news sentiment
ALPHAVANTAGE_KEY = os.getenv('ALPHAVANTAGE_KEY', '')
