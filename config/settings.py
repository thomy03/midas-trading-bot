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
ZONE_TOLERANCE = 5.0

# How many days back to look for EMA crossovers
LOOKBACK_DAYS_WEEKLY = 260  # ~1 year of weekly data
LOOKBACK_DAYS_DAILY = 365   # ~1 year of daily data

# Maximum age of crossover to be considered relevant (in weeks for weekly, days for daily)
MAX_CROSSOVER_AGE_WEEKLY = 52  # weeks
MAX_CROSSOVER_AGE_DAILY = 120  # days

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
