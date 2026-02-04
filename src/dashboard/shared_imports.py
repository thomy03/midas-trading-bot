"""
Shared imports for dashboard pages

This module provides all common imports needed by dashboard page modules.
Import from here to ensure consistency across pages.
"""
import sys
import os

# Ensure src is in path
_src_path = os.path.join(os.path.dirname(__file__), '../..')
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Streamlit
import streamlit as st

# Standard library
import json
import asyncio
import time
import glob as glob_module
from datetime import datetime, timedelta

# Data manipulation
import pandas as pd

# Project imports - Core
from src.utils.visualizer import chart_visualizer
from src.utils.interactive_chart import interactive_chart_builder
from src.data.market_data import market_data_fetcher
from src.screening.screener import market_screener
from src.database.db_manager import db_manager
from src.indicators.ema_analyzer import ema_analyzer

# Project imports - Utils
from src.utils.background_scanner import background_scanner, ScanStatus, scan_scheduler
from src.utils.watchlist_manager import watchlist_manager
from src.utils.trade_tracker import trade_tracker
from src.utils.sector_heatmap import sector_heatmap_builder
from src.utils.economic_calendar import economic_calendar, EventType, EventImpact
from src.utils.notification_manager import notification_manager, NotificationPriority
from src.utils.sector_analyzer import SectorAnalyzer

# Project imports - Intelligence
from src.intelligence import NewsFetcher, get_news_fetcher
from src.intelligence.trend_discovery import TrendDiscovery, get_trend_discovery

# Config
from config.settings import EMA_PERIODS, ZONE_TOLERANCE, MARKETS_EXTENDED, CAPITAL
from config import settings

# Environment
from dotenv import load_dotenv
load_dotenv()

__all__ = [
    # Streamlit
    'st',
    # Standard
    'json', 'asyncio', 'time', 'glob_module', 'datetime', 'timedelta',
    # Data
    'pd',
    # Core
    'chart_visualizer', 'interactive_chart_builder', 'market_data_fetcher',
    'market_screener', 'db_manager', 'ema_analyzer',
    # Utils
    'background_scanner', 'ScanStatus', 'scan_scheduler',
    'watchlist_manager', 'trade_tracker', 'sector_heatmap_builder',
    'economic_calendar', 'EventType', 'EventImpact',
    'notification_manager', 'NotificationPriority', 'SectorAnalyzer',
    # Intelligence
    'NewsFetcher', 'get_news_fetcher', 'TrendDiscovery', 'get_trend_discovery',
    # Config
    'EMA_PERIODS', 'ZONE_TOLERANCE', 'MARKETS_EXTENDED', 'CAPITAL', 'settings',
]
