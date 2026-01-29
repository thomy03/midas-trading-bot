"""
Market Screener Dashboard - Dash Version

DEPRECATED: This dashboard is deprecated in favor of webapp.py
Please use: python webapp.py
This file is kept for reference only and will be removed in a future version.

Run with: python dashboard_dash.py
"""
import warnings
warnings.warn(
    "dashboard_dash.py is DEPRECATED. Please use 'python webapp.py' instead.",
    DeprecationWarning,
    stacklevel=2
)
import sys
import os
from datetime import datetime, timedelta
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict
from urllib.parse import parse_qs, urlparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

import dash
from dash import dcc, html, callback, Input, Output, State, ctx, dash_table, ALL, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.utils.visualizer import chart_visualizer
from src.utils.interactive_chart import interactive_chart_builder
from src.data.market_data import market_data_fetcher
from src.screening.screener import market_screener
from src.database.db_manager import db_manager
from src.utils.fundamental_scorer import fundamental_scorer
from src.utils.position_sizing import PortfolioTracker, find_stop_loss_from_rsi_trendline
from src.utils.sector_analyzer import SectorAnalyzer
from config import settings

# Determine Phase 1 vs Phase 2 based on Grok configuration
GROK_ACTIVE = getattr(settings, 'GROK_ENABLED', False) and getattr(settings, 'GROK_API_KEY', None)
ELITE_THRESHOLD = 45 if GROK_ACTIVE else 22
ELITE_MAX_SCORE = 60 if GROK_ACTIVE else 30
ELITE_PHASE = "Phase 2 - Grok" if GROK_ACTIVE else "Phase 1"
from src.indicators.ema_analyzer import ema_analyzer
from config.settings import EMA_PERIODS, ZONE_TOLERANCE, MARKETS_EXTENDED, CAPITAL

# ============================================================
# BACKGROUND SCANNER STATE (Thread-safe)
# ============================================================
import tempfile
SCAN_STATE_FILE = os.path.join(tempfile.gettempdir(), 'tradingbot_dash_scan.json')
_scan_lock = threading.Lock()
_scan_thread = None


def _convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.bool_, )):
        return bool(obj)
    elif isinstance(obj, (np.integer, )):
        return int(obj)
    elif isinstance(obj, (np.floating, )):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _save_scan_state(state: dict):
    """Save scan state to file - atomic write to prevent corruption"""
    import copy
    with _scan_lock:
        try:
            # Deep copy to avoid issues with nested lists
            state_copy = copy.deepcopy(state)

            # Convert numpy types to Python native types
            state_copy = _convert_numpy_types(state_copy)

            if state_copy.get('start_time') and hasattr(state_copy['start_time'], 'isoformat'):
                state_copy['start_time'] = state_copy['start_time'].isoformat()
            if state_copy.get('end_time') and hasattr(state_copy['end_time'], 'isoformat'):
                state_copy['end_time'] = state_copy['end_time'].isoformat()

            # Atomic write: write to temp file then rename
            temp_file = SCAN_STATE_FILE + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(state_copy, f, ensure_ascii=False)

            # Atomic rename (on Windows, need to remove first)
            if os.path.exists(SCAN_STATE_FILE):
                os.remove(SCAN_STATE_FILE)
            os.rename(temp_file, SCAN_STATE_FILE)
        except Exception as e:
            print(f"Error saving scan state: {e}")
            import traceback
            traceback.print_exc()


def _load_scan_state() -> dict:
    """Load scan state from file with retry for concurrent access"""
    default = {
        'status': 'idle',
        'total': 0,
        'completed': 0,
        'alerts': [],
        'pending': [],
        'start_time': None,
        'end_time': None,
        'error': None,
        'current_symbol': '',
        'pause_requested': False,
        'cancel_requested': False
    }

    # Retry up to 3 times for file read errors (concurrent access)
    for attempt in range(3):
        try:
            if os.path.exists(SCAN_STATE_FILE):
                with _scan_lock:
                    with open(SCAN_STATE_FILE, 'r') as f:
                        state = json.load(f)
                if state.get('start_time'):
                    state['start_time'] = datetime.fromisoformat(state['start_time'])
                if state.get('end_time'):
                    state['end_time'] = datetime.fromisoformat(state['end_time'])
                return state
            return default
        except json.JSONDecodeError:
            if attempt < 2:
                import time
                time.sleep(0.05)  # Wait 50ms and retry
                continue
        except Exception as e:
            print(f"Error loading scan state: {e}")
            break
    return default


def _run_background_scan(stocks: list, num_workers: int = 10):
    """Run scan in background thread with real-time L2 scoring"""
    global _scan_thread

    def scan_worker():
        from queue import Queue
        import threading

        completed_count = 0
        alerts_list = []
        l2_queue = Queue()  # Queue for STRONG_BUY signals to score
        l2_results = {}  # Symbol -> L2 result
        l2_lock = threading.Lock()

        # L2 worker thread - processes STRONG_BUY signals in real-time
        def l2_worker():
            while True:
                item = l2_queue.get()
                if item is None:  # Poison pill
                    break
                try:
                    symbol = item.get('symbol', '?')
                    print(f"[L2] Scoring {symbol}...")
                    # Score single alert
                    scored = fundamental_scorer.score_strong_buys([item])
                    if scored:
                        with l2_lock:
                            l2_results[symbol] = scored[0]
                        is_elite = scored[0].get('l2_is_elite', False)
                        l2_score = scored[0].get('l2_score', 0)
                        print(f"[L2] {symbol}: {l2_score}/60 {'*** ELITE ***' if is_elite else ''}")
                except Exception as e:
                    print(f"[L2] Error scoring {item.get('symbol', '?')}: {e}")
                finally:
                    l2_queue.task_done()

        # Start L2 worker thread
        l2_thread = threading.Thread(target=l2_worker, daemon=True)
        l2_thread.start()
        print(f"[WORKER] Starting scan of {len(stocks)} stocks with {num_workers} workers + real-time L2")

        try:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all jobs at once
                futures = {
                    executor.submit(market_screener.screen_single_stock, s['symbol'], s.get('name', s['symbol'])): s
                    for s in stocks
                }

                print(f"[WORKER] Submitted {len(futures)} jobs")

                for future in as_completed(futures):
                    stock = futures[future]

                    # Check for cancel or pause
                    state = _load_scan_state()
                    if state.get('cancel_requested'):
                        print("[WORKER] Cancel requested, stopping...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        l2_queue.put(None)  # Stop L2 worker
                        state['status'] = 'cancelled'
                        state['end_time'] = datetime.now()
                        state['completed'] = completed_count
                        state['alerts'] = alerts_list
                        _save_scan_state(state)
                        return

                    if state.get('pause_requested'):
                        print("[WORKER] Pause requested, saving state...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        l2_queue.put(None)  # Stop L2 worker
                        state['status'] = 'paused'
                        state['pause_requested'] = False
                        state['completed'] = completed_count
                        state['alerts'] = alerts_list
                        completed_symbols = {futures[f]['symbol'] for f in futures if f.done()}
                        state['pending'] = [s for s in stocks if s['symbol'] not in completed_symbols]
                        _save_scan_state(state)
                        return

                    try:
                        result = future.result(timeout=60)
                        completed_count += 1

                        if result:
                            result['market'] = stock.get('market', 'N/A')
                            alerts_list.append(result)
                            print(f"[WORKER] ALERT #{len(alerts_list)}: {stock['symbol']}")

                            # IMMEDIATE L2: Queue STRONG_BUY for real-time scoring
                            if result.get('confidence_signal', '').startswith('STRONG'):
                                l2_queue.put(result.copy())
                                print(f"[WORKER] -> Queued for L2: {stock['symbol']}")

                        # Update state every 10 completions
                        if completed_count % 10 == 0:
                            # Merge L2 results into alerts
                            with l2_lock:
                                for alert in alerts_list:
                                    sym = alert.get('symbol')
                                    if sym in l2_results:
                                        alert.update(l2_results[sym])

                            state = _load_scan_state()
                            state['completed'] = completed_count
                            state['alerts'] = alerts_list
                            state['current_symbol'] = stock['symbol']
                            elite_count = sum(1 for a in alerts_list if a.get('l2_is_elite', False))
                            state['elite_count'] = elite_count
                            _save_scan_state(state)
                            print(f"[WORKER] Progress: {completed_count}/{len(stocks)} - {len(alerts_list)} alerts - {elite_count} ELITE")

                    except Exception as e:
                        completed_count += 1

            # Wait for remaining L2 scoring to complete
            print(f"[WORKER] Phase 1 done! Waiting for L2 queue to finish...")
            l2_queue.join()  # Wait for all L2 tasks to complete
            l2_queue.put(None)  # Stop L2 worker thread

            # Final merge of L2 results
            with l2_lock:
                for alert in alerts_list:
                    sym = alert.get('symbol')
                    if sym in l2_results:
                        alert.update(l2_results[sym])

            state = _load_scan_state()
            state['status'] = 'completed'
            state['end_time'] = datetime.now()
            state['completed'] = completed_count
            state['alerts'] = alerts_list
            state['pending'] = []
            elite_count = sum(1 for a in alerts_list if a.get('l2_is_elite', False))
            state['elite_count'] = elite_count
            _save_scan_state(state)

            print(f"[WORKER] DONE! {completed_count} scanned, {len(alerts_list)} alerts, {elite_count} ELITE")

        except Exception as e:
            print(f"[WORKER] FATAL ERROR: {e}")
            state = _load_scan_state()
            state['status'] = 'error'
            state['error'] = str(e)
            state['end_time'] = datetime.now()
            _save_scan_state(state)

    _scan_thread = threading.Thread(target=scan_worker, daemon=True)
    _scan_thread.start()
    print(f"[MAIN] Scan thread started: {_scan_thread.is_alive()}")


# ============================================================
# DASH APP
# ============================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)
app.title = "Market Screener Dashboard"

# ============================================================
# SIDEBAR
# ============================================================
sidebar = dbc.Col([
    html.H3("Market Screener", className="text-success mb-4"),
    html.Hr(),
    dbc.Nav([
        dbc.NavLink([html.I(className="fas fa-home me-2"), "Home"], href="/", active="exact"),
        dbc.NavLink([html.I(className="fas fa-chart-line me-2"), "Pro Chart"], href="/pro-chart", active="exact"),
        dbc.NavLink([html.I(className="fas fa-chart-bar me-2"), "Chart Analyzer"], href="/chart-analyzer", active="exact"),
        dbc.NavLink([html.I(className="fas fa-search me-2"), "Screening"], href="/screening", active="exact"),
        dbc.NavLink([html.I(className="fas fa-star me-2"), "ELITE Signals"], href="/elite", active="exact", className="text-warning"),
        dbc.NavLink([html.I(className="fas fa-briefcase me-2"), "Portfolio"], href="/portfolio", active="exact"),
        dbc.NavLink([html.I(className="fas fa-bell me-2"), "Alerts History"], href="/alerts", active="exact"),
        dbc.NavLink([html.I(className="fas fa-brain me-2"), "Intelligence"], href="/intelligence", active="exact", className="text-info"),
        dbc.NavLink([html.I(className="fas fa-cog me-2"), "Settings"], href="/settings", active="exact"),
    ], vertical=True, pills=True),
    html.Hr(),
    html.Div([
        html.H6("Quick Stats", className="text-muted"),
        html.Div(id="sidebar-stats")
    ]),
    html.Hr(),
    html.Small("Developed with Dash", className="text-muted")
], width=2, className="bg-dark p-3", style={"minHeight": "100vh"})

# ============================================================
# PAGE LAYOUTS
# ============================================================

def home_layout():
    """Home page layout"""
    try:
        alerts = db_manager.get_recent_alerts(days=7)
        alert_count = len(alerts) if alerts else 0
    except:
        alert_count = 0
        alerts = []

    # Build alerts table data
    alert_data = []
    if alerts:
        for alert in alerts[:20]:
            alert_data.append({
                'Symbol': alert.symbol,
                'Company': alert.company_name,
                'Timeframe': alert.timeframe.upper(),
                'Price': f"${alert.current_price:.2f}",
                'Support': f"${alert.support_level:.2f}",
                'Distance': f"{alert.distance_to_support_pct:.2f}%",
                'Recommendation': alert.recommendation,
                'Date': alert.alert_date.strftime('%Y-%m-%d %H:%M')
            })

    return html.Div([
        html.H2("Market Screener Dashboard", className="text-success mb-4"),
        html.P("Welcome to your automated stock screening system!", className="lead"),

        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Strategy", className="card-title"),
                    html.P("EMA-based screening on Weekly and Daily timeframes")
                ])
            ], className="bg-info text-white"), width=4),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("EMAs Used", className="card-title"),
                    html.P(', '.join(map(str, EMA_PERIODS)))
                ])
            ], className="bg-success text-white"), width=4),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H5("Support Zone", className="card-title"),
                    html.P(f"±{ZONE_TOLERANCE}% tolerance")
                ])
            ], className="bg-warning text-dark"), width=4),
        ], className="mb-4"),

        html.Hr(),
        html.H4(f"Recent Alerts ({alert_count})", className="mb-3"),

        dash_table.DataTable(
            id='home-alerts-table',
            columns=[{"name": col, "id": col} for col in ['Symbol', 'Company', 'Timeframe', 'Price', 'Support', 'Distance', 'Recommendation', 'Date']],
            data=alert_data,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#303030', 'color': 'white'},
            style_header={'backgroundColor': '#404040', 'fontWeight': 'bold'},
            style_data_conditional=[
                {'if': {'filter_query': '{Recommendation} = "STRONG_BUY"'}, 'backgroundColor': '#00C853', 'color': 'white'},
                {'if': {'filter_query': '{Recommendation} = "BUY"'}, 'backgroundColor': '#64DD17', 'color': 'white'},
                {'if': {'filter_query': '{Recommendation} = "WATCH"'}, 'backgroundColor': '#FDD835', 'color': 'black'},
            ],
            page_size=15
        ) if alert_data else html.Div(
            dbc.Alert("No recent alerts. Run a screening to generate alerts!", color="info"),
            className="mt-3"
        )
    ])


def pro_chart_layout():
    """Pro Chart page layout"""
    return html.Div([
        html.H2("Interactive Pro Chart", className="text-primary mb-4"),
        html.P("TradingView-style interactive charting", className="lead"),

        dbc.Row([
            dbc.Col([
                dbc.Label("Symbol"),
                dbc.Input(id="pro-symbol", type="text", value="AAPL", placeholder="Enter symbol")
            ], width=3),
            dbc.Col([
                dbc.Label("Timeframe"),
                dbc.Select(id="pro-timeframe", options=[
                    {"label": "Daily", "value": "daily"},
                    {"label": "Weekly", "value": "weekly"}
                ], value="daily")
            ], width=2),
            dbc.Col([
                dbc.Label("Period"),
                dbc.Select(id="pro-period", options=[
                    {"label": "3 months", "value": "3mo"},
                    {"label": "6 months", "value": "6mo"},
                    {"label": "1 year", "value": "1y"},
                    {"label": "2 years", "value": "2y"},
                    {"label": "5 years", "value": "5y"},
                ], value="1y")
            ], width=2),
            dbc.Col([
                dbc.Label(" "),
                dbc.Button("Load Chart", id="pro-load-btn", color="primary", className="d-block")
            ], width=2),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dbc.Checklist(
                id="pro-options",
                options=[
                    {"label": " Volume", "value": "volume"},
                    {"label": " RSI", "value": "rsi"},
                    {"label": " EMAs", "value": "emas"},
                    {"label": " Supports", "value": "supports"},
                    {"label": " RSI Trendline", "value": "rsi_trendline"},
                ],
                value=["volume", "rsi", "emas", "supports", "rsi_trendline"],
                inline=True,
                className="mb-3"
            ))
        ]),

        dcc.Loading(
            id="pro-chart-loading",
            type="circle",
            children=html.Div(id="pro-chart-container")
        ),

        html.Hr(),
        html.Div(id="pro-price-info"),

        # Hidden stores for URL params from ELITE signal
        dcc.Store(id="pro-stop-loss", data=None),
        dcc.Store(id="pro-url-timeframe", data=None),  # Timeframe from URL (to match signal's RSI analysis)
        dcc.Store(id="pro-url-peaks", data=None)  # Peak indices from URL (to draw exact same trendline)
    ])


def screening_layout():
    """Screening page layout with background scanner"""
    return html.Div([
        html.H2("Market Screening", className="text-warning mb-4"),

        # Market Selection
        html.H5("Market Selection", className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Checklist(
                id="market-selection",
                options=[
                    {"label": " NASDAQ", "value": "NASDAQ"},
                    {"label": " S&P 500", "value": "SP500"},
                    {"label": " Crypto", "value": "CRYPTO"},
                    {"label": " Europe", "value": "EUROPE"},
                    {"label": " Asia ADR", "value": "ASIA_ADR"},
                ],
                value=["NASDAQ", "SP500"],
                inline=True
            ), width=10),
            dbc.Col(dbc.Checklist(
                id="full-exchange-mode",
                options=[{"label": " Full NASDAQ", "value": "full"}],
                value=[],
                inline=True
            ), width=2),
        ], className="mb-3"),

        html.Hr(),

        # Scan Controls
        html.H5("Scan Controls", className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Button("Start Scan", id="start-scan-btn", color="success", className="me-2"),
                dbc.Button("Pause", id="pause-scan-btn", color="warning", className="me-2", disabled=True),
                dbc.Button("Resume", id="resume-scan-btn", color="info", className="me-2", disabled=True),
                dbc.Button("Cancel", id="cancel-scan-btn", color="danger", className="me-2", disabled=True),
                dbc.Button("Reset", id="reset-scan-btn", color="secondary", className="me-2"),
            ], width=8),
            dbc.Col([
                dbc.Label("Workers"),
                dbc.Select(id="num-workers", options=[
                    {"label": "5", "value": "5"},
                    {"label": "10", "value": "10"},
                    {"label": "15", "value": "15"},
                    {"label": "20", "value": "20"},
                ], value="10")
            ], width=2),
        ], className="mb-3"),

        # Progress Display
        html.Div(id="scan-status-display", className="mb-3"),
        dbc.Progress(id="scan-progress-bar", value=0, striped=True, animated=True, className="mb-3"),

        # Auto-refresh interval (updates every 2 seconds during scan)
        dcc.Interval(id="scan-interval", interval=2000, n_intervals=0, disabled=True),

        html.Hr(),

        # Alerts Display
        html.H5("Detected Signals", className="mb-3"),
        html.Div(id="scan-alerts-container")
    ])


def portfolio_layout():
    """Portfolio page layout"""
    # Reload from file to get latest positions
    from src.utils.position_sizing import PortfolioTracker
    tracker = PortfolioTracker()
    portfolio = tracker.get_summary()

    return html.Div([
        html.H2("Portfolio Management", className="text-info mb-4"),

        # Capital Overview
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Total Capital", className="text-muted"),
                    html.H3(f"{portfolio['total_capital']:,.0f}€")
                ])
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Invested", className="text-muted"),
                    html.H3(f"{portfolio['invested_capital']:,.0f}€")
                ])
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Available", className="text-muted"),
                    html.H3(f"{portfolio['available_capital']:,.0f}€")
                ])
            ]), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Positions", className="text-muted"),
                    html.H3(f"{portfolio['num_positions']}")
                ])
            ]), width=3),
        ], className="mb-4"),

        html.Hr(),

        # Update Capital
        html.H5("Update Capital", className="mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Input(id="new-capital-input", type="number", value=portfolio['total_capital'], placeholder="New capital")
            ], width=4),
            dbc.Col([
                dbc.Button("Update", id="update-capital-btn", color="primary")
            ], width=2),
        ], className="mb-4"),

        html.Div(id="capital-update-feedback"),

        html.Hr(),

        # Positions
        html.H5("Open Positions", className="mb-3"),
        html.Div(id="positions-display")
    ])


def _create_score_legend():
    """Create the collapsible score legend component"""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Button([
                html.I(className="fas fa-question-circle me-2"),
                "Comprendre les scores"
            ], id="legend-toggle-btn", color="link", className="text-decoration-none p-0")
        ], className="bg-dark"),
        dbc.Collapse([
            dbc.CardBody([
                dbc.Row([
                    # Score L1 - Technical
                    dbc.Col([
                        html.H6("SCORE L1 (0-100) - Analyse Technique", className="text-primary mb-3"),
                        html.Ul([
                            html.Li([
                                html.Strong("EMA Alignment (20 pts): "),
                                "EMAs 24>38>62 = tendance haussiere"
                            ]),
                            html.Li([
                                html.Strong("Support Proximity (20 pts): "),
                                "Prix proche d'un support EMA"
                            ]),
                            html.Li([
                                html.Strong("RSI Breakout (25 pts): "),
                                "Cassure d'oblique RSI descendante"
                            ]),
                            html.Li([
                                html.Strong("Freshness (20 pts): "),
                                "Signal recent = meilleur score"
                            ]),
                            html.Li([
                                html.Strong("Volume (15 pts): "),
                                "Volume > moyenne = confirmation"
                            ]),
                        ], className="small")
                    ], md=4),
                    # Score L2 - Fundamental
                    dbc.Col([
                        html.H6("SCORE L2 (0-60) - Fondamental + Sentiment", className="text-warning mb-3"),
                        html.Ul([
                            html.Li([
                                html.Strong("Sante (20 pts): "),
                                "Revenue growth >15%, marges positives, earnings"
                            ]),
                            html.Li([
                                html.Strong("Contexte (10 pts): "),
                                "SPY > EMA20w, ETF secteur > SMA50"
                            ]),
                            html.Li([
                                html.Strong("Sentiment Grok (30 pts): "),
                                "Sentiment social + catalyseurs identifies"
                            ]),
                        ], className="small"),
                        html.Hr(),
                        html.P([
                            html.Strong("Seuils: "),
                            "STRONG_BUY: L1 >= 75 | ELITE: L2 > 45/60"
                        ], className="small text-muted mb-0")
                    ], md=4),
                    # Asset Types
                    dbc.Col([
                        html.H6("Types d'Actifs", className="text-info mb-3"),
                        html.Div([
                            html.Span("[STOCK] ", className="text-success"),
                            html.Span("Action standard - scoring complet", className="small")
                        ], className="mb-2"),
                        html.Div([
                            html.Span("[CRYPTO] ", className="text-warning"),
                            html.Span("Crypto pure - pas de sante, context BTC/ATH", className="small")
                        ], className="mb-2"),
                        html.Div([
                            html.Span("[MINER] ", className="text-info"),
                            html.Span("Crypto-proxy - hyper-growth OK, context BTC", className="small")
                        ], className="mb-2"),
                        html.Hr(),
                        html.P([
                            html.Strong("Kill Switches: "),
                            html.Br(),
                            "Earnings < 5j, Volume < 200K"
                        ], className="small text-danger mb-0")
                    ], md=4),
                ])
            ], className="bg-dark")
        ], id="legend-collapse", is_open=False)
    ], className="mb-4 border-secondary")


def _create_portfolio_summary() -> dbc.Card:
    """Create the portfolio summary card for ELITE page"""
    try:
        tracker = PortfolioTracker()
        summary = tracker.get_summary()
    except Exception:
        summary = {
            'total_capital': CAPITAL,
            'invested_capital': 0,
            'available_capital': CAPITAL,
            'num_positions': 0,
            'positions': []
        }

    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-wallet me-2"),
            "Mon Portfolio"
        ], className="bg-primary text-white"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5(f"{summary['total_capital']:.0f}EUR", className="text-primary mb-0"),
                    html.Small("Capital Total", className="text-muted")
                ], md=3, className="text-center"),
                dbc.Col([
                    html.H5(f"{summary['invested_capital']:.0f}EUR", className="text-warning mb-0"),
                    html.Small("Investi", className="text-muted")
                ], md=3, className="text-center"),
                dbc.Col([
                    html.H5(f"{summary['available_capital']:.0f}EUR", className="text-success mb-0"),
                    html.Small("Disponible", className="text-muted")
                ], md=3, className="text-center"),
                dbc.Col([
                    html.H5(f"{summary['num_positions']}", className="text-info mb-0"),
                    html.Small("Positions", className="text-muted")
                ], md=3, className="text-center"),
            ]),
            # List of open positions
            html.Hr(className="my-2") if summary['positions'] else None,
            html.Div([
                dbc.Badge(
                    f"{p['symbol']} ({p['shares']} @ ${p['entry_price']:.2f})",
                    color="secondary",
                    className="me-2 mb-1"
                )
                for p in summary['positions']
            ]) if summary['positions'] else html.Small("Aucune position ouverte", className="text-muted fst-italic")
        ])
    ], className="mb-4 border-primary", id="portfolio-summary-card")


def _create_portfolio_button(symbol: str, signal: Dict):
    """Create contextual portfolio button based on current state"""
    try:
        tracker = PortfolioTracker()
        in_portfolio = tracker.get_position(symbol) is not None
        available = tracker.get_available_capital()
        position_value = signal.get('position_value', 0)

        if in_portfolio:
            # Already in portfolio - show disabled button with option to remove
            return dbc.Button([
                html.I(className="fas fa-check-circle me-1"),
                "Dans portfolio"
            ], id={"type": "remove-portfolio-btn", "symbol": symbol}, color="secondary", size="sm", className="me-2")
        elif position_value > available:
            # Not enough capital
            return dbc.Button([
                html.I(className="fas fa-exclamation-triangle me-1"),
                f"Capital insuf. ({available:.0f}EUR dispo)"
            ], disabled=True, color="warning", size="sm", className="me-2", outline=True)
        else:
            # Available - can add
            return dbc.Button([
                html.I(className="fas fa-plus-circle me-1"),
                "Ajouter"
            ], id={"type": "add-portfolio-btn", "symbol": symbol}, color="success", size="sm", className="me-2")
    except Exception:
        # Fallback button
        return dbc.Button([
            html.I(className="fas fa-plus-circle me-1"),
            "Ajouter"
        ], id={"type": "add-portfolio-btn", "symbol": symbol}, color="success", size="sm", className="me-2")


def _create_elite_card(signal: Dict) -> dbc.Card:
    """Create a structured card for an ELITE signal"""
    symbol = signal.get('symbol', '?')
    details = signal.get('l2_details', {})
    health = details.get('health', {})
    context = details.get('context', {})
    sentiment = details.get('sentiment', {})
    asset_type = signal.get('l2_asset_type', 'EQUITY')

    # Timeframe info
    rsi_timeframe = signal.get('rsi_timeframe', 'daily')
    is_weekly = rsi_timeframe == 'weekly'

    # Determine max scores based on asset type
    if asset_type == 'CRYPTO':
        l2_max = 20
        health_max = 0
        context_max = 20
    else:
        l2_max = 60 if GROK_ACTIVE else 30
        health_max = 20
        context_max = 10

    # Asset type badge
    type_badges = {
        'CRYPTO': ('[CRYPTO]', 'warning'),
        'CRYPTO_PROXY': ('[MINER]', 'info'),
        'EQUITY': ('[STOCK]', 'success')
    }
    type_text, type_color = type_badges.get(asset_type, ('[STOCK]', 'success'))

    # Build technical section
    l1_score = signal.get('confidence_score', 0)
    rsi_signal = signal.get('rsi_signal', 'N/A')
    rsi_age = signal.get('rsi_breakout_age', '-')
    support_dist = signal.get('distance_to_support_pct', 0)

    # Build health section (skip for CRYPTO)
    health_items = []
    if asset_type != 'CRYPTO':
        rev_growth = health.get('revenue_growth')
        if rev_growth is not None:
            status = "[OK]" if rev_growth > 15 else "[--]"
            health_items.append(html.Li(f"Revenue Growth: {rev_growth:.1f}% {status}"))

        margin = health.get('profit_margins')
        if margin is not None:
            pardoned = " [HYPER]" if health.get('profit_margins_pardoned') else ""
            status = "[OK]" if margin > 0 or health.get('profit_margins_pardoned') else "[--]"
            health_items.append(html.Li(f"Marge: {margin:.1f}%{pardoned} {status}"))

        earnings = health.get('earnings_growth')
        if earnings is not None:
            status = "[OK]" if earnings > 0 else "[--]"
            health_items.append(html.Li(f"Earnings Growth: {earnings:.1f}% {status}"))

    # Build context section
    context_items = []
    if asset_type == 'CRYPTO':
        btc_ok = context.get('btc_is_self') or context.get('btc_above_ema20', False)
        context_items.append(html.Li(f"BTC > EMA20: {'[OK]' if btc_ok else '[--]'}"))
        ath_dist = context.get('ath_distance_pct')
        if ath_dist is not None:
            status = "[OK]" if ath_dist < 50 else "[--]"
            context_items.append(html.Li(f"Distance ATH: {ath_dist:.0f}% {status}"))
    elif asset_type == 'CRYPTO_PROXY':
        spy_ok = context.get('spy_above_ema20w', False)
        btc_ok = context.get('btc_above_ema20', False)
        context_items.append(html.Li(f"SPY > EMA20w: {'[OK]' if spy_ok else '[--]'}"))
        context_items.append(html.Li(f"BTC > EMA20: {'[OK]' if btc_ok else '[--]'}"))
    else:
        spy_ok = context.get('spy_above_ema20w', False)
        etf = context.get('sector_etf', '-')
        etf_ok = context.get('etf_above_sma50', False)
        context_items.append(html.Li(f"SPY > EMA20w: {'[OK]' if spy_ok else '[--]'}"))
        context_items.append(html.Li(f"{etf} > SMA50: {'[OK]' if etf_ok else '[--]'}"))

    # Build sentiment section
    sentiment_items = []
    if sentiment.get('status') == 'success':
        sent = sentiment.get('sentiment', 'neutral')
        conf = sentiment.get('sentiment_confidence', 0)
        trend = sentiment.get('mention_trend', 'stable')
        catalysts = sentiment.get('catalysts', [])

        sent_status = "[OK]" if sent == 'bullish' else ("[--]" if sent == 'bearish' else "[~~]")
        sentiment_items.append(html.Li(f"Sentiment: {sent.upper()} ({conf}%) {sent_status}"))

        trend_status = "[OK]" if trend == 'rising' else "[~~]"
        sentiment_items.append(html.Li(f"Mentions: {trend.upper()} {trend_status}"))

        if catalysts:
            for cat in catalysts[:2]:
                sentiment_items.append(html.Li(f"-> {cat[:40]}..."))
    else:
        sentiment_items.append(html.Li("Grok non configure", className="text-muted"))

    # Earnings warning
    earnings_info = details.get('earnings', {})
    days_to_earn = earnings_info.get('days_to_earnings')
    risk_items = []
    if days_to_earn is not None and days_to_earn < 30:
        risk_items.append(html.Li(f"Earnings dans {days_to_earn}j", className="text-warning"))

    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col([
                    html.H5([
                        html.Span(type_text, className=f"text-{type_color} me-2"),
                        html.Strong(symbol)
                    ], className="mb-0")
                ], width=6),
                dbc.Col([
                    html.Div([
                        dbc.Badge(f"L1: {l1_score:.0f}/100", color="primary", className="me-2"),
                        dbc.Badge(f"L2: {signal.get('l2_score', 0):.0f}/{l2_max}", color="warning"),
                        dbc.Badge("ELITE", color="success", className="ms-2"),
                        dbc.Badge(
                            "WEEKLY" if is_weekly else "DAILY",
                            color="info" if is_weekly else "secondary",
                            className="ms-2"
                        )
                    ], className="text-end")
                ], width=6)
            ])
        ], className="bg-dark"),
        dbc.CardBody([
            dbc.Row([
                # Technical Column
                dbc.Col([
                    html.H6("[TECHNIQUE]", className="text-primary mb-2"),
                    html.Ul([
                        html.Li(f"Prix: ${signal.get('current_price', 0):.2f}"),
                        html.Li(f"Support: ${signal.get('support_level', 0):.2f} ({support_dist:.1f}%)"),
                        html.Li(f"RSI: {rsi_signal}" + (f" ({rsi_age}p)" if rsi_age != '-' else "")),
                    ], className="small mb-0")
                ], md=3),
                # Health Column
                dbc.Col([
                    html.H6(f"[SANTE] {health.get('total', 0)}/{health_max}", className="text-success mb-2"),
                    html.Ul(health_items, className="small mb-0") if health_items else html.P("N/A", className="text-muted small")
                ], md=3),
                # Context Column
                dbc.Col([
                    html.H6(f"[CONTEXTE] {context.get('total', 0)}/{context_max}", className="text-info mb-2"),
                    html.Ul(context_items, className="small mb-0")
                ], md=3),
                # Sentiment Column
                dbc.Col([
                    html.H6(f"[SENTIMENT] {sentiment.get('total', 0)}/30", className="text-warning mb-2"),
                    html.Ul(sentiment_items, className="small mb-0")
                ], md=3),
            ]),
            # Position Sizing Row
            html.Div([
                html.Hr(className="my-2"),
                dbc.Row([
                    dbc.Col([
                        html.H6("[POSITION RECOMMANDEE]", className="text-success mb-2"),
                        html.Div([
                            dbc.Badge(f"{signal.get('position_shares', 0)} actions", color="success", className="me-2"),
                            dbc.Badge(f"{signal.get('position_value', 0):.0f}€", color="primary", className="me-2"),
                            dbc.Badge(f"Stop: ${signal.get('stop_loss', 0):.2f}", color="danger", className="me-2"),
                            dbc.Badge(f"Risque: {signal.get('risk_amount', 0):.0f}€ ({signal.get('risk_pct', 0)*100:.1f}%)", color="warning"),
                        ])
                    ], md=8),
                    dbc.Col([
                        html.Small(signal.get('stop_source', ''), className="text-muted fst-italic")
                    ], md=4, className="text-end align-self-center"),
                ])
            ]) if signal.get('position_shares', 0) > 0 else None,
            # Risk row if applicable
            html.Div([
                html.Hr(className="my-2"),
                html.Div([
                    html.Strong("[RISQUES] ", className="text-danger"),
                    html.Ul(risk_items, className="small mb-0 d-inline")
                ])
            ]) if risk_items else None,
        ]),
        dbc.CardFooter([
            dbc.Button([
                html.I(className="fas fa-robot me-1"),
                "Analyse Grok"
            ], id={"type": "grok-analysis-btn", "symbol": symbol}, color="info", size="sm", className="me-2"),
            # Portfolio button - contextual based on state
            _create_portfolio_button(symbol, signal),
            dbc.Button([
                html.I(className="fas fa-chart-line me-1"),
                "Graphique"
            ], href=f"/pro-chart?symbol={symbol}&stop={signal.get('stop_loss', 0):.2f}&tf={signal.get('rsi_timeframe', 'daily')}&peaks={','.join(map(str, signal.get('rsi_peak_indices', [])))}", color="secondary", size="sm"),
        ], className="bg-dark")
    ], className="mb-3 border-warning")


def elite_layout():
    """ELITE Signals page layout - Layer 2 filtered signals"""
    return html.Div([
        # Toast for portfolio feedback
        dbc.Toast(
            id="portfolio-toast",
            header="Portfolio",
            icon="success",
            is_open=False,
            dismissable=True,
            duration=5000,
            style={"position": "fixed", "top": 66, "right": 10, "width": 400, "zIndex": 1000}
        ),

        html.H2([
            html.I(className="fas fa-star text-warning me-2"),
            "ELITE Signals"
        ], className="text-warning mb-2"),
        html.P("Signaux STRONG_BUY filtres par analyse fondamentale (Layer 2)", className="text-muted mb-4"),

        # Portfolio Summary
        html.Div(id="elite-portfolio-summary", children=_create_portfolio_summary()),

        # Status bar
        dbc.Alert([
            html.I(className="fas fa-info-circle me-2"),
            "Les signaux ELITE sont automatiquement generes apres chaque scan. ",
            f"Score L2 > {ELITE_THRESHOLD}/{ELITE_MAX_SCORE} ({ELITE_PHASE}) requis.",
        ], color="info" if not GROK_ACTIVE else "success", className="mb-4"),

        # Legend (collapsible)
        _create_score_legend(),

        # Refresh button
        dbc.Button([
            html.I(className="fas fa-sync me-2"),
            "Rafraichir"
        ], id="elite-refresh-btn", color="warning", className="mb-4"),

        # Stats cards
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Total STRONG_BUY", className="text-muted"),
                    html.H3(id="elite-total-strong-buy", children="0")
                ])
            ], color="dark", outline=True), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6(f"ELITE (L2 > {ELITE_THRESHOLD})", className="text-muted"),
                    html.H3(id="elite-count", children="0", className="text-warning")
                ])
            ], color="warning", outline=True), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Exclus (Kill Switch)", className="text-muted"),
                    html.H3(id="elite-excluded", children="0", className="text-danger")
                ])
            ], color="danger", outline=True), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6("Contexte SPY", className="text-muted"),
                    html.H3(id="elite-spy-status", children="-")
                ])
            ], color="dark", outline=True), width=3),
        ], className="mb-4"),

        html.Hr(),

        # ELITE signals - now uses cards instead of table
        html.H5("Signaux ELITE", className="mb-3"),
        html.Div(id="elite-signals-table"),

        html.Hr(),

        # Non-ELITE signals (passed kill switches but < threshold)
        html.H5("Signaux en attente (L2 < seuil)", className="mb-3 text-muted"),
        html.Div(id="elite-waiting-table"),

        html.Hr(),

        # Excluded signals (collapsed)
        dbc.Accordion([
            dbc.AccordionItem([
                html.Div(id="elite-excluded-table")
            ], title="Signaux Exclus (Kill Switches)", item_id="excluded")
        ], id="elite-accordion", start_collapsed=True, className="mb-4"),

        # Grok Analysis Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(id="grok-modal-title")),
            dbc.ModalBody(id="grok-modal-body"),
            dbc.ModalFooter(
                dbc.Button("Fermer", id="grok-modal-close", className="ms-auto")
            )
        ], id="grok-analysis-modal", size="lg"),

        # Hidden store for selected symbol
        dcc.Store(id="grok-selected-symbol", data=None),
    ])


def intelligence_layout():
    """Market Intelligence page - Sector Momentum Analysis"""
    # Initialize sector analyzer
    sector_analyzer = SectorAnalyzer(min_momentum_vs_spy=0.0)

    # Load sector ETF data
    try:
        sector_analyzer.load_sector_data(market_data_fetcher, period='1y')
        momentum_data = sector_analyzer.calculate_sector_momentum()
    except Exception as e:
        momentum_data = {}

    # Build sector table data
    sector_table_data = []
    if momentum_data:
        for sector, sm in sorted(momentum_data.items(), key=lambda x: x[1].rank):
            sector_table_data.append({
                'Rank': sm.rank,
                'Sector': sector,
                'ETF': sm.etf_symbol,
                'Perf 20d': f"{sm.perf_20d:+.1f}%",
                'Perf 50d': f"{sm.perf_50d:+.1f}%",
                'vs SPY': f"{sm.vs_spy:+.1f}%",
                'Status': 'BULLISH' if sm.is_bullish else 'BEARISH'
            })

    bullish_count = sum(1 for sm in momentum_data.values() if sm.is_bullish) if momentum_data else 0
    bearish_count = len(momentum_data) - bullish_count if momentum_data else 0

    return html.Div([
        html.H2([html.I(className="fas fa-brain me-2"), "Market Intelligence"], className="text-info mb-4"),
        html.P("Sector momentum analysis and market intelligence", className="lead"),

        # Summary Cards
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(bullish_count, className="card-title text-success"),
                    html.P("Bullish Sectors", className="mb-0")
                ])
            ], className="bg-dark"), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(bearish_count, className="card-title text-danger"),
                    html.P("Bearish Sectors", className="mb-0")
                ])
            ], className="bg-dark"), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4(len(momentum_data), className="card-title text-info"),
                    html.P("Sectors Analyzed", className="mb-0")
                ])
            ], className="bg-dark"), width=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H4("0.0%", className="card-title text-warning"),
                    html.P("Min vs SPY Threshold", className="mb-0")
                ])
            ], className="bg-dark"), width=3),
        ], className="mb-4"),

        html.Hr(),

        # Tabs for different views
        dbc.Tabs([
            # Tab 1: Sector Momentum Table
            dbc.Tab([
                html.Br(),
                html.H4("Sector Momentum Rankings", className="text-info mb-3"),
                html.P("Sectors ranked by relative performance vs SPY (20-day). "
                       "BULLISH = positive performance AND outperforming SPY.", className="text-muted mb-3"),

                dash_table.DataTable(
                    id='intel-sector-table',
                    columns=[{"name": col, "id": col} for col in ['Rank', 'Sector', 'ETF', 'Perf 20d', 'Perf 50d', 'vs SPY', 'Status']],
                    data=sector_table_data,
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left', 'padding': '12px', 'backgroundColor': '#303030', 'color': 'white'},
                    style_header={'backgroundColor': '#404040', 'fontWeight': 'bold', 'color': '#17a2b8'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{Status} = "BULLISH"'}, 'backgroundColor': '#1e4620', 'color': '#4ade80'},
                        {'if': {'filter_query': '{Status} = "BEARISH"'}, 'backgroundColor': '#4a1e1e', 'color': '#f87171'},
                    ],
                    page_size=15
                ) if sector_table_data else html.Div("No sector data available", className="alert alert-warning")
            ], label="Sector Momentum", tab_id="tab-sectors"),

            # Tab 2: Symbol Sector Mapping
            dbc.Tab([
                html.Br(),
                html.H4("Symbol to Sector Mapping", className="text-info mb-3"),
                html.P("Shows which sector each tracked symbol belongs to.", className="text-muted mb-3"),

                dbc.Row([
                    dbc.Col([
                        html.H6("Technology", className="text-success"),
                        html.Ul([html.Li(s) for s in ['AAPL', 'MSFT', 'GOOGL', 'CRM', 'SAP.DE']])
                    ], width=3),
                    dbc.Col([
                        html.H6("AI/Semiconductors", className="text-warning"),
                        html.Ul([html.Li(s) for s in ['NVDA', 'AMD']])
                    ], width=3),
                    dbc.Col([
                        html.H6("Communications", className="text-info"),
                        html.Ul([html.Li(s) for s in ['META', 'NFLX', 'DTE.DE']])
                    ], width=3),
                    dbc.Col([
                        html.H6("Consumer Discretionary", className="text-primary"),
                        html.Ul([html.Li(s) for s in ['TSLA', 'AMZN', 'MC.PA', 'BMW.DE', 'MBG.DE']])
                    ], width=3),
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col([
                        html.H6("Finance", className="text-success"),
                        html.Ul([html.Li(s) for s in ['JPM', 'BNP.PA', 'ALV.DE']])
                    ], width=3),
                    dbc.Col([
                        html.H6("Healthcare", className="text-warning"),
                        html.Ul([html.Li(s) for s in ['SAN.PA', 'JNJ']])
                    ], width=3),
                    dbc.Col([
                        html.H6("Energy", className="text-info"),
                        html.Ul([html.Li(s) for s in ['XOM', 'TTE.PA']])
                    ], width=3),
                    dbc.Col([
                        html.H6("Industrials", className="text-primary"),
                        html.Ul([html.Li(s) for s in ['AI.PA', 'AIR.PA', 'SIE.DE']])
                    ], width=3),
                ])
            ], label="Symbol Mapping", tab_id="tab-mapping"),

            # Tab 3: Filter Explanation
            dbc.Tab([
                html.Br(),
                html.H4("Sector Momentum Filter", className="text-info mb-3"),

                dbc.Card([
                    dbc.CardBody([
                        html.H5("How it works", className="card-title"),
                        html.P([
                            "The sector momentum filter helps avoid taking trades in weak sectors. ",
                            "It uses sector ETFs as proxies to measure relative strength vs SPY."
                        ]),
                        html.Hr(),
                        html.H6("Bullish Sector Criteria:"),
                        html.Ul([
                            html.Li("20-day performance > 0%"),
                            html.Li("Performance vs SPY >= 0% (or configured threshold)")
                        ]),
                        html.Hr(),
                        html.H6("ETF Proxies Used:"),
                        html.Ul([
                            html.Li("XLK - Technology"),
                            html.Li("SMH - AI/Semiconductors"),
                            html.Li("XLF - Finance"),
                            html.Li("XLE - Energy"),
                            html.Li("XLV - Healthcare"),
                            html.Li("XLY - Consumer Discretionary"),
                            html.Li("XLI - Industrials"),
                            html.Li("XLC - Communications"),
                        ])
                    ])
                ], className="bg-dark")
            ], label="How It Works", tab_id="tab-help"),
        ], id="intel-tabs", active_tab="tab-sectors"),

        # Refresh button
        html.Div([
            html.Hr(),
            dbc.Button([html.I(className="fas fa-sync-alt me-2"), "Refresh Sector Data"],
                      id="intel-refresh-btn", color="info", className="mt-3"),
            html.Span(id="intel-refresh-status", className="ms-3 text-muted")
        ])
    ])


# ============================================================
# MAIN LAYOUT
# ============================================================
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.Row([
        sidebar,
        dbc.Col(html.Div(id='page-content', className="p-4"), width=10)
    ])
])


# ============================================================
# CALLBACKS
# ============================================================

@callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    """Route to different pages"""
    if pathname == '/' or pathname == '/home':
        return home_layout()
    elif pathname == '/pro-chart':
        return pro_chart_layout()
    elif pathname == '/chart-analyzer':
        return html.Div([
            html.H2("Chart Analyzer", className="text-primary"),
            html.P("Coming soon...")
        ])
    elif pathname == '/screening':
        return screening_layout()
    elif pathname == '/elite':
        return elite_layout()
    elif pathname == '/portfolio':
        return portfolio_layout()
    elif pathname == '/alerts':
        return html.Div([
            html.H2("Alerts History", className="text-warning"),
            html.P("Coming soon...")
        ])
    elif pathname == '/settings':
        return html.Div([
            html.H2("Settings", className="text-muted"),
            html.P("Coming soon...")
        ])
    elif pathname == '/intelligence':
        return intelligence_layout()
    else:
        return html.Div([
            html.H2("404 - Page Not Found"),
            html.P(f"The page '{pathname}' does not exist.")
        ])


@callback(Output('sidebar-stats', 'children'), Input('url', 'pathname'))
def update_sidebar_stats(_):
    """Update sidebar stats"""
    try:
        alerts = db_manager.get_recent_alerts(days=7)
        count = len(alerts) if alerts else 0
    except:
        count = 0

    return html.Div([
        html.P(f"Alerts (7 days): {count}", className="text-light mb-0")
    ])


# ============================================================
# PRO CHART CALLBACKS
# ============================================================

@callback(
    [Output('pro-chart-container', 'children'),
     Output('pro-price-info', 'children')],
    [Input('pro-load-btn', 'n_clicks'),
     Input('pro-url-peaks', 'data')],  # Changed to Input to ensure latest value
    [State('pro-symbol', 'value'),
     State('pro-timeframe', 'value'),
     State('pro-period', 'value'),
     State('pro-options', 'value'),
     State('pro-stop-loss', 'data')],
    prevent_initial_call=True
)
def load_pro_chart(n_clicks, url_peak_indices, symbol, timeframe, period, options, stop_loss_price):
    """Load and display pro chart"""
    # Debug: Print received parameters
    print(f"[DEBUG] load_pro_chart called: symbol={symbol}, url_peak_indices={url_peak_indices}, options={options}")

    if not symbol:
        return html.Div("Please enter a symbol"), html.Div()

    symbol = symbol.upper()
    options = options or []

    try:
        interval = '1wk' if timeframe == 'weekly' else '1d'
        df = market_data_fetcher.get_historical_data(symbol, period=period, interval=interval)

        if df is None or len(df) == 0:
            return html.Div(f"Could not load data for {symbol}", className="text-danger"), html.Div()

        # Calculate EMAs and crossovers
        df_with_emas = ema_analyzer.calculate_emas(df)
        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        current_price = float(close.iloc[-1])
        crossovers = ema_analyzer.detect_crossovers(df_with_emas, timeframe)
        support_zones = ema_analyzer.find_support_zones(df_with_emas, crossovers, current_price)

        # Create chart
        fig = interactive_chart_builder.create_interactive_chart(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            show_volume='volume' in options,
            show_rsi='rsi' in options,
            show_emas='emas' in options,
            ema_periods=EMA_PERIODS,
            height=800,
            crossovers=crossovers,
            show_crossover_zones=True
        )

        # Add support levels
        if 'supports' in options and support_zones:
            levels = [{'price': zone['level'], 'type': 'support', 'strength': zone['strength']}
                     for zone in support_zones[:5]]
            fig = interactive_chart_builder.add_support_resistance_levels(fig, levels, current_price)

        # Add RSI trendline visualization
        rsi_trendline_active = 'rsi' in options and 'rsi_trendline' in options
        print(f"[DEBUG] RSI trendline check: rsi={('rsi' in options)}, rsi_trendline={('rsi_trendline' in options)}, active={rsi_trendline_active}")
        print(f"[DEBUG] url_peak_indices for trendline: {url_peak_indices}")
        if rsi_trendline_active:
            rsi_row = 3 if 'volume' in options else 2
            traces_before = len(fig.data)
            # ALWAYS calculate stop dynamically using the forced_peak_indices
            # This ensures stop is at RSI minimum from first peak to END of trendline
            # (not just to last peak, but to the breakout/current date)
            fig = interactive_chart_builder.add_rsi_trendline_breakout(
                fig, df, rsi_row=rsi_row,
                df_with_emas=df_with_emas,
                support_zones=support_zones,
                show_stop_loss=True,  # Always recalculate dynamically
                forced_peak_indices=url_peak_indices  # Use exact trendline from signal if available
            )
            traces_after = len(fig.data)
            print(f"[DEBUG] RSI trendline added: {traces_after - traces_before} traces")

        chart = dcc.Graph(
            figure=fig,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True,
                'doubleClick': 'reset'
            }
        )

        # Price info
        high = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
        low = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
        volume = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']

        prev_close = float(close.iloc[-2]) if len(close) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0

        price_info = dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("Current Price"),
                html.H4(f"${current_price:.2f}"),
                html.Small(f"{change:+.2f} ({change_pct:+.2f}%)",
                          className="text-success" if change >= 0 else "text-danger")
            ])]), width=2),
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("Period High"),
                html.H4(f"${float(high.max()):.2f}")
            ])]), width=2),
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("Period Low"),
                html.H4(f"${float(low.min()):.2f}")
            ])]), width=2),
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("Avg Volume"),
                html.H4(f"{float(volume.mean())/1e6:.2f}M")
            ])]), width=2),
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("EMA 24"),
                html.H4(f"${float(df_with_emas['EMA_24'].iloc[-1]):.2f}")
            ])]), width=2),
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("EMA 62"),
                html.H4(f"${float(df_with_emas['EMA_62'].iloc[-1]):.2f}")
            ])]), width=2),
        ])

        return chart, price_info

    except Exception as e:
        return html.Div(f"Error loading chart: {e}", className="text-danger"), html.Div()


@callback(
    [Output('pro-symbol', 'value'),
     Output('pro-load-btn', 'n_clicks'),
     Output('pro-stop-loss', 'data'),
     Output('pro-url-timeframe', 'data'),
     Output('pro-url-peaks', 'data'),
     Output('pro-timeframe', 'value')],  # Also set the timeframe dropdown
    [Input('url', 'pathname'),
     Input('url', 'search')],
    [State('pro-load-btn', 'n_clicks')],
    prevent_initial_call=False
)
def load_symbol_from_url(pathname, search, current_clicks):
    """Auto-load chart when symbol is passed in URL query string"""
    if pathname != '/pro-chart':
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    if not search:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Parse query string (search includes the '?')
    params = parse_qs(search.lstrip('?'))
    symbol = params.get('symbol', [None])[0]
    stop_loss = params.get('stop', [None])[0]
    url_timeframe = params.get('tf', [None])[0]  # 'daily' or 'weekly'
    peaks_str = params.get('peaks', [None])[0]  # comma-separated indices

    # Convert stop loss to float if present
    stop_value = None
    if stop_loss:
        try:
            stop_value = float(stop_loss)
        except ValueError:
            pass

    # Convert peaks string to list of integers
    peaks_list = None
    if peaks_str:
        try:
            peaks_list = [int(p) for p in peaks_str.split(',') if p]
        except ValueError:
            pass

    if symbol:
        # Return symbol, increment clicks to trigger chart load, and URL params
        new_clicks = (current_clicks or 0) + 1
        # Set dropdown timeframe to match URL if provided
        dropdown_tf = url_timeframe if url_timeframe in ['daily', 'weekly'] else dash.no_update
        return symbol.upper(), new_clicks, stop_value, url_timeframe, peaks_list, dropdown_tf

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


# ============================================================
# SCREENING CALLBACKS
# ============================================================

@callback(
    [Output('scan-status-display', 'children'),
     Output('scan-progress-bar', 'value'),
     Output('scan-alerts-container', 'children'),
     Output('start-scan-btn', 'disabled'),
     Output('pause-scan-btn', 'disabled'),
     Output('resume-scan-btn', 'disabled'),
     Output('cancel-scan-btn', 'disabled'),
     Output('scan-interval', 'disabled')],
    [Input('scan-interval', 'n_intervals'),
     Input('start-scan-btn', 'n_clicks'),
     Input('pause-scan-btn', 'n_clicks'),
     Input('resume-scan-btn', 'n_clicks'),
     Input('cancel-scan-btn', 'n_clicks'),
     Input('reset-scan-btn', 'n_clicks')],
    [State('market-selection', 'value'),
     State('full-exchange-mode', 'value'),
     State('num-workers', 'value')]
)
def manage_scan(n_intervals, start_click, pause_click, resume_click, cancel_click, reset_click,
                markets, full_mode, num_workers):
    """Manage background scan"""
    triggered = ctx.triggered_id
    state = _load_scan_state()

    # Detect stale 'running' state (app restarted while scan was running)
    if state.get('status') == 'running' and (_scan_thread is None or not _scan_thread.is_alive()):
        print("[STATE] Detected stale 'running' state - resetting to cancelled")
        state['status'] = 'cancelled'
        state['error'] = 'Scan interrupted (app restarted)'
        _save_scan_state(state)

    if state.get('status') not in ['idle', 'completed', 'cancelled']:
        status_msg = state.get('status')
        if status_msg == 'l2_scoring':
            print(f"[SCAN] L2 scoring in progress - {len(state.get('alerts', []))} alerts")
        else:
            print(f"[SCAN] {state.get('completed', 0)}/{state.get('total', 0)} - status={status_msg} - alerts={len(state.get('alerts', []))}")

    # Handle Reset button - clear state file
    if triggered == 'reset-scan-btn' and reset_click:
        print("[SCAN] Resetting scan state...")
        if os.path.exists(SCAN_STATE_FILE):
            os.remove(SCAN_STATE_FILE)
        state = _load_scan_state()  # Will return default idle state

    # Handle button clicks
    if triggered == 'start-scan-btn' and start_click:
        # Build stock list
        stocks = []
        markets = markets or []
        full_exchange = 'full' in (full_mode or [])

        if 'NASDAQ' in markets:
            for t in market_data_fetcher.get_nasdaq_tickers(full_exchange=full_exchange):
                stocks.append({'symbol': t, 'name': t, 'market': 'NASDAQ'})

        if 'SP500' in markets:
            for t in market_data_fetcher.get_sp500_tickers(include_nyse_full=False):
                if not any(s['symbol'] == t for s in stocks):
                    stocks.append({'symbol': t, 'name': t, 'market': 'SP500'})

        if 'CRYPTO' in markets:
            for t in market_data_fetcher.get_crypto_tickers():
                stocks.append({'symbol': t, 'name': t, 'market': 'CRYPTO'})

        if 'EUROPE' in markets:
            for t in market_data_fetcher.get_european_tickers():
                stocks.append({'symbol': t, 'name': t, 'market': 'EUROPE'})

        if 'ASIA_ADR' in markets:
            for t in market_data_fetcher.get_asian_adr_tickers():
                stocks.append({'symbol': t, 'name': t, 'market': 'ASIA_ADR'})

        if stocks:
            print(f"[SCAN] Starting scan with {len(stocks)} stocks...")
            print(f"[SCAN] State file: {SCAN_STATE_FILE}")

            # Initialize state
            state = {
                'status': 'running',
                'total': len(stocks),
                'completed': 0,
                'alerts': [],
                'pending': stocks,
                'start_time': datetime.now(),
                'end_time': None,
                'error': None,
                'current_symbol': '',
                'pause_requested': False,
                'cancel_requested': False
            }
            _save_scan_state(state)
            print(f"[SCAN] Initial state saved to {SCAN_STATE_FILE}")

            # Verify state was saved
            test_state = _load_scan_state()
            print(f"[SCAN] State verification: status={test_state.get('status')}, total={test_state.get('total')}")

            # Start background scan
            _run_background_scan(stocks, int(num_workers or 10))
            print(f"[SCAN] Background scan thread started!")
        else:
            print(f"[SCAN] No stocks to scan - markets={markets}")

    elif triggered == 'pause-scan-btn' and pause_click:
        state['pause_requested'] = True
        _save_scan_state(state)

    elif triggered == 'resume-scan-btn' and resume_click:
        if state.get('status') == 'paused' and state.get('pending'):
            state['status'] = 'running'
            state['pause_requested'] = False
            _save_scan_state(state)
            _run_background_scan(state['pending'], int(num_workers or 10))

    elif triggered == 'cancel-scan-btn' and cancel_click:
        state['cancel_requested'] = True
        _save_scan_state(state)

    # Reload state after actions
    state = _load_scan_state()
    status = state.get('status', 'idle')

    # Calculate progress
    total = state.get('total', 0)
    completed = state.get('completed', 0)
    progress = (completed / total * 100) if total > 0 else 0

    # Fix: If total is 0 but status is 'running', it's a stale state - reset to idle
    if status == 'running' and total == 0:
        status = 'idle'
        if os.path.exists(SCAN_STATE_FILE):
            os.remove(SCAN_STATE_FILE)

    # Status display
    if status == 'idle':
        status_display = dbc.Alert("Ready to scan. Select markets and click Start Scan.", color="info")
    elif status == 'running':
        elapsed = 0
        if state.get('start_time'):
            elapsed = (datetime.now() - state['start_time']).total_seconds()
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rate if rate > 0 else 0

        status_display = dbc.Alert([
            html.Strong("Scan in progress... "),
            html.Span(f"{completed}/{total} ({progress:.1f}%) - "),
            html.Span(f"Speed: {rate:.1f}/sec - "),
            html.Span(f"ETA: {eta/60:.1f} min - "),
            html.Span(f"Signals: {len(state.get('alerts', []))} - "),
            html.Span(f"Current: {state.get('current_symbol', '')}")
        ], color="warning")
    elif status == 'paused':
        remaining = len(state.get('pending', []))
        status_display = dbc.Alert(f"Scan paused. {completed}/{total} completed, {remaining} remaining. {len(state.get('alerts', []))} signals found.", color="info")
    elif status == 'completed':
        elapsed = 0
        if state.get('start_time') and state.get('end_time'):
            elapsed = (state['end_time'] - state['start_time']).total_seconds() / 60
        status_display = dbc.Alert(f"Scan completed! {len(state.get('alerts', []))} signals found in {elapsed:.1f} minutes.", color="success")
    elif status == 'l2_scoring':
        strong_buy_count = sum(1 for a in state.get('alerts', []) if a.get('confidence_signal', '').startswith('STRONG'))
        status_display = dbc.Alert([
            html.Strong("🔬 Layer 2 Analysis (Grok AI)... "),
            html.Span(f"Analyzing {strong_buy_count} STRONG_BUY signals for sentiment & fundamentals")
        ], color="info")
    elif status == 'cancelled':
        status_display = dbc.Alert(f"Scan cancelled. {completed} symbols processed, {len(state.get('alerts', []))} signals found.", color="warning")
    elif status == 'error':
        status_display = dbc.Alert(f"Error: {state.get('error', 'Unknown error')}", color="danger")
    else:
        status_display = dbc.Alert("Unknown status", color="secondary")

    # Alerts display
    alerts = state.get('alerts', [])
    if alerts:
        alert_data = []
        for alert in sorted(alerts, key=lambda x: x.get('confidence_score', 0), reverse=True):
            # Determine timeframe info
            rsi_tf = alert.get('rsi_timeframe', 'weekly').upper()[0]  # W or D
            breakout_age = alert.get('rsi_breakout_age', '-')
            if breakout_age != '-':
                freshness = f"{rsi_tf} ({breakout_age}p)"  # e.g., "W (2p)" = Weekly, 2 periods ago
            else:
                freshness = f"{rsi_tf}"

            # RSI breakout date or crossover date
            signal_date = alert.get('rsi_breakout_date', alert.get('crossover_date', '-'))

            alert_data.append({
                'Symbol': alert['symbol'],
                'Market': alert.get('market', 'N/A'),
                'Price': f"${alert['current_price']:.2f}",
                'Support': f"${alert['support_level']:.2f}",
                'Dist.': f"{alert['distance_to_support_pct']:.1f}%",
                'TF': freshness,  # Timeframe + freshness
                'Date': signal_date,  # Signal date
                'Score': f"{alert.get('confidence_score', 0):.0f}",
                'Signal': alert.get('recommendation', 'N/A'),
                'RSI': alert.get('rsi_signal', '-')
            })

        alerts_display = dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in alert_data[0].keys()] if alert_data else [],
            data=alert_data,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'padding': '10px', 'backgroundColor': '#303030', 'color': 'white'},
            style_header={'backgroundColor': '#404040', 'fontWeight': 'bold'},
            page_size=20,
            sort_action='native',
            filter_action='native'
        )
    else:
        alerts_display = dbc.Alert("No signals detected yet.", color="secondary")

    # Button states
    is_running = status in ['running', 'l2_scoring']  # L2 scoring is part of the running process
    is_paused = status == 'paused'
    is_idle = status in ['idle', 'completed', 'cancelled', 'error']

    return (
        status_display,
        progress,
        alerts_display,
        not is_idle,  # start disabled when not idle
        not is_running,  # pause disabled when not running
        not is_paused,  # resume disabled when not paused
        not (is_running or is_paused),  # cancel disabled when idle
        not is_running  # interval disabled when not running
    )


# ============================================================
# PORTFOLIO CALLBACKS
# ============================================================

@callback(
    Output('capital-update-feedback', 'children'),
    Input('update-capital-btn', 'n_clicks'),
    State('new-capital-input', 'value'),
    prevent_initial_call=True
)
def update_capital(n_clicks, new_capital):
    """Update portfolio capital"""
    if new_capital and new_capital > 0:
        market_screener.update_capital(new_capital)
        return dbc.Alert(f"Capital updated to {new_capital:,.0f}€", color="success", duration=3000)
    return dbc.Alert("Please enter a valid amount", color="warning", duration=3000)


@callback(
    Output('positions-display', 'children'),
    Input('url', 'pathname')
)
def display_positions(pathname):
    """Display portfolio positions"""
    if pathname != '/portfolio':
        return html.Div()

    # Reload from file to get latest positions (including those added from ELITE page)
    from src.utils.position_sizing import PortfolioTracker
    tracker = PortfolioTracker()  # Creates new instance, reloads from file
    portfolio = tracker.get_summary()
    positions = portfolio.get('positions', [])

    if positions:
        # Create cards for each position with delete buttons
        position_cards = []
        for pos in positions:
            card = dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5(pos.get('symbol', '?'), className="text-primary mb-0"),
                            html.Small(f"Entry: {pos.get('entry_date', 'N/A')}", className="text-muted")
                        ], md=2),
                        dbc.Col([
                            html.Div([
                                html.Small("Shares", className="text-muted d-block"),
                                html.Strong(f"{pos.get('shares', 0)}")
                            ])
                        ], md=2),
                        dbc.Col([
                            html.Div([
                                html.Small("Entry Price", className="text-muted d-block"),
                                html.Strong(f"${pos.get('entry_price', 0):.2f}")
                            ])
                        ], md=2),
                        dbc.Col([
                            html.Div([
                                html.Small("Stop Loss", className="text-muted d-block"),
                                html.Strong(f"${pos.get('stop_loss', 0):.2f}", className="text-danger")
                            ])
                        ], md=2),
                        dbc.Col([
                            html.Div([
                                html.Small("Value", className="text-muted d-block"),
                                html.Strong(f"{pos.get('position_value', 0):.0f}EUR")
                            ])
                        ], md=2),
                        dbc.Col([
                            dbc.Button([
                                html.I(className="fas fa-times me-1"),
                                "Fermer"
                            ], id={"type": "close-position-btn", "symbol": pos.get('symbol', '')},
                               color="danger", size="sm", className="mt-2")
                        ], md=2),
                    ], align="center")
                ])
            ], className="mb-2 bg-dark")
            position_cards.append(card)
        return html.Div(position_cards)

    return dbc.Alert("No open positions", color="info")


@callback(
    Output('positions-display', 'children', allow_duplicate=True),
    Input({"type": "close-position-btn", "symbol": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def close_position_from_portfolio(n_clicks_list):
    """Close a position from Portfolio Management page"""
    if not n_clicks_list or not any(n_clicks_list):
        return dash.no_update

    # Find which button was clicked
    triggered = ctx.triggered_id
    if not triggered or not isinstance(triggered, dict):
        return dash.no_update

    symbol = triggered.get('symbol', '')
    if not symbol:
        return dash.no_update

    # Close the position
    from src.utils.position_sizing import PortfolioTracker
    tracker = PortfolioTracker()
    closed = tracker.close_position(symbol)

    if not closed:
        return dash.no_update

    # Reload and return updated positions display
    portfolio = tracker.get_summary()
    positions = portfolio.get('positions', [])

    if positions:
        position_cards = []
        for pos in positions:
            card = dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5(pos.get('symbol', '?'), className="text-primary mb-0"),
                            html.Small(f"Entry: {pos.get('entry_date', 'N/A')}", className="text-muted")
                        ], md=2),
                        dbc.Col([
                            html.Div([
                                html.Small("Shares", className="text-muted d-block"),
                                html.Strong(f"{pos.get('shares', 0)}")
                            ])
                        ], md=2),
                        dbc.Col([
                            html.Div([
                                html.Small("Entry Price", className="text-muted d-block"),
                                html.Strong(f"${pos.get('entry_price', 0):.2f}")
                            ])
                        ], md=2),
                        dbc.Col([
                            html.Div([
                                html.Small("Stop Loss", className="text-muted d-block"),
                                html.Strong(f"${pos.get('stop_loss', 0):.2f}", className="text-danger")
                            ])
                        ], md=2),
                        dbc.Col([
                            html.Div([
                                html.Small("Value", className="text-muted d-block"),
                                html.Strong(f"{pos.get('position_value', 0):.0f}EUR")
                            ])
                        ], md=2),
                        dbc.Col([
                            dbc.Button([
                                html.I(className="fas fa-times me-1"),
                                "Fermer"
                            ], id={"type": "close-position-btn", "symbol": pos.get('symbol', '')},
                               color="danger", size="sm", className="mt-2")
                        ], md=2),
                    ], align="center")
                ])
            ], className="mb-2 bg-dark")
            position_cards.append(card)
        return html.Div(position_cards)

    return dbc.Alert("No open positions", color="info")


# ============================================================
# ELITE PAGE CALLBACKS
# ============================================================

# Store for ELITE signals (populated by scan)
_elite_store = {'signals': [], 'last_update': None}


def _get_elite_signals_from_scan():
    """Get ELITE signals from the last scan state"""
    state = _load_scan_state()
    alerts = state.get('alerts', [])

    # Filter STRONG_BUY with L2 data
    strong_buys = [a for a in alerts if a.get('confidence_signal', '').startswith('STRONG_BUY')]

    return strong_buys


@callback(
    [Output('elite-total-strong-buy', 'children'),
     Output('elite-count', 'children'),
     Output('elite-excluded', 'children'),
     Output('elite-spy-status', 'children'),
     Output('elite-signals-table', 'children'),
     Output('elite-waiting-table', 'children'),
     Output('elite-excluded-table', 'children')],
    [Input('elite-refresh-btn', 'n_clicks'),
     Input('url', 'pathname')],
    prevent_initial_call=False
)
def update_elite_page(n_clicks, pathname):
    """Update ELITE page with L2 signals"""
    if pathname != '/elite':
        return "0", "0", "0", "-", html.Div(), html.Div(), html.Div()

    # Get signals from scan state
    signals = _get_elite_signals_from_scan()

    if not signals:
        return (
            "0", "0", "0", "-",
            dbc.Alert("Aucun signal STRONG_BUY disponible. Lancez un scan depuis la page Screening.", color="warning"),
            html.Div(),
            html.Div()
        )

    # Count stats
    total = len(signals)
    elite_signals = [s for s in signals if s.get('l2_is_elite', False) and not s.get('l2_excluded', False)]
    excluded_signals = [s for s in signals if s.get('l2_excluded', False)]
    # NEW: Waiting signals = passed kill switches but not ELITE
    waiting_signals = [s for s in signals if not s.get('l2_is_elite', False) and not s.get('l2_excluded', False)]
    elite_count = len(elite_signals)
    excluded_count = len(excluded_signals)

    # SPY status
    spy_status = "-"
    for s in signals:
        details = s.get('l2_details', {})
        context = details.get('context', {})
        if 'spy_above_ema20w' in context:
            spy_status = "Bullish" if context['spy_above_ema20w'] else "Bearish"
            break

    # Build ELITE table (cards)
    if elite_signals:
        elite_table = _build_elite_table(elite_signals)
    else:
        elite_table = dbc.Alert(f"Aucun signal ELITE (L2 > {ELITE_THRESHOLD}) trouve", color="info")

    # Build waiting table (signals close to ELITE)
    if waiting_signals:
        waiting_table = _build_waiting_table(waiting_signals)
    else:
        waiting_table = html.P("Tous les signaux valides sont ELITE!", className="text-success")

    # Build excluded table
    if excluded_signals:
        excluded_table = _build_excluded_table(excluded_signals)
    else:
        excluded_table = html.P("Aucun signal exclu", className="text-muted")

    return str(total), str(elite_count), str(excluded_count), spy_status, elite_table, waiting_table, excluded_table


def _recalculate_stop_loss_dynamic(signal: Dict) -> Dict:
    """
    Recalculate stop-loss dynamically using rsi_peak_indices and fresh market data.
    Uses the period from first peak to END OF DATA (breakout), NOT just to last peak.
    """
    symbol = signal.get('symbol', '')
    rsi_peak_indices = signal.get('rsi_peak_indices', [])
    rsi_timeframe = signal.get('rsi_timeframe', 'daily')

    if not rsi_peak_indices or not symbol:
        return signal  # No peaks to recalculate from

    try:
        # Determine period based on timeframe
        if rsi_timeframe == 'weekly':
            period = '2y'
            interval = '1wk'
        else:
            period = '1y'
            interval = '1d'

        # Fetch fresh market data
        df = market_data_fetcher.get_historical_data(symbol, period=period, interval=interval)

        if df is None or len(df) == 0:
            return signal

        # Recalculate stop-loss using the correct logic (first peak to END OF DATA)
        stop_loss, stop_source = find_stop_loss_from_rsi_trendline(df, rsi_peak_indices)

        if stop_loss and stop_loss > 0:
            # Update signal with recalculated values
            signal = signal.copy()  # Don't modify original
            signal['stop_loss'] = stop_loss
            signal['stop_source'] = stop_source

    except Exception as e:
        print(f"Error recalculating stop-loss for {symbol}: {e}")

    return signal


def _build_elite_table(signals):
    """Build cards grid for ELITE signals (replaces table)"""
    if not signals:
        return dbc.Alert("Aucun signal ELITE", color="warning")

    # Sort by L2 score descending
    sorted_signals = sorted(signals, key=lambda x: x.get('l2_score', 0), reverse=True)

    # Recalculate stop-loss dynamically for each signal (uses correct period: first peak to END)
    recalculated_signals = [_recalculate_stop_loss_dynamic(s) for s in sorted_signals]

    # Create cards for each signal
    cards = [_create_elite_card(s) for s in recalculated_signals]

    return html.Div(cards)


def _build_waiting_table(signals):
    """Build table for signals waiting (passed kill switches but not ELITE)"""
    if not signals:
        return html.P("Aucun signal en attente", className="text-muted")

    # Sort by L2 score descending (closest to ELITE first)
    sorted_signals = sorted(signals, key=lambda x: x.get('l2_score', 0), reverse=True)

    rows = []
    for s in sorted_signals:
        details = s.get('l2_details', {})
        health = details.get('health', {})
        context = details.get('context', {})
        sentiment = details.get('sentiment', {})
        asset_type = s.get('l2_asset_type', 'EQUITY')

        # Asset type badge
        type_badge = {
            'CRYPTO': '[CRYPTO]',
            'CRYPTO_PROXY': '[MINER]',
            'EQUITY': '[STOCK]'
        }.get(asset_type, '[STOCK]')

        # L2 max depends on asset type
        if asset_type == 'CRYPTO':
            l2_max = 60 if GROK_ACTIVE else 20
        else:
            l2_max = 60 if GROK_ACTIVE else 30

        # Calculate distance to ELITE threshold
        l2_score = s.get('l2_score', 0)
        distance_to_elite = ELITE_THRESHOLD - l2_score

        # Health details
        rev_growth = health.get('revenue_growth')
        rev_str = f"{rev_growth:.1f}%" if rev_growth is not None else "N/A"

        # Sentiment status
        if sentiment.get('status') == 'success':
            sent_str = sentiment.get('sentiment', 'neutral').upper()[:4]
        else:
            sent_str = "N/A"

        # Timeframe
        rsi_tf = s.get('rsi_timeframe', 'daily')
        tf_str = "W" if rsi_tf == 'weekly' else "D"

        rows.append({
            'Type': type_badge,
            'Symbol': s.get('symbol', '?'),
            'TF': tf_str,
            'Price': f"${s.get('current_price', 0):.2f}",
            'L1': f"{s.get('confidence_score', 0):.0f}",
            'L2': f"{l2_score:.0f}/{l2_max}",
            'Health': f"{health.get('total', 0):.0f}/20",
            'Context': f"{context.get('total', 0):.0f}/10",
            'Sentiment': sent_str,
            'Gap': f"-{distance_to_elite:.0f} pts"
        })

    df = pd.DataFrame(rows)

    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '8px', 'backgroundColor': '#2d2d2d', 'color': 'white'},
        style_header={'backgroundColor': '#6c757d', 'color': 'white', 'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'column_id': 'L2'}, 'fontWeight': 'bold', 'color': '#6c757d'},
            {'if': {'column_id': 'Symbol'}, 'fontWeight': 'bold'},
            {'if': {'column_id': 'Gap'}, 'color': '#ffc107'},
        ],
        sort_action='native',
        filter_action='native',
        page_size=10
    )


def _build_excluded_table(signals):
    """Build table for excluded signals"""
    rows = []
    for s in signals:
        asset_type = s.get('l2_asset_type', 'EQUITY')
        type_badge = {
            'CRYPTO': '🪙',
            'CRYPTO_PROXY': '⛏️',
            'EQUITY': '📈'
        }.get(asset_type, '📈')

        rows.append({
            'Type': type_badge,
            'Symbol': s.get('symbol', '?'),
            'Price': f"${s.get('current_price', 0):.2f}",
            'L1 Score': f"{s.get('confidence_score', 0):.0f}",
            'Reason': s.get('l2_exclusion_reason', 'Unknown')
        })

    df = pd.DataFrame(rows)

    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '8px', 'backgroundColor': '#2d2d2d', 'color': 'white'},
        style_header={'backgroundColor': '#dc3545', 'color': 'white', 'fontWeight': 'bold'},
        style_data_conditional=[
            {'if': {'column_id': 'Reason'}, 'color': '#dc3545'},
            {'if': {'column_id': 'Type'}, 'textAlign': 'center'},
        ],
        page_size=10
    )


# ============================================================
# ELITE PAGE - LEGEND TOGGLE CALLBACK
# ============================================================

@callback(
    Output('legend-collapse', 'is_open'),
    Input('legend-toggle-btn', 'n_clicks'),
    State('legend-collapse', 'is_open'),
    prevent_initial_call=True
)
def toggle_legend(n_clicks, is_open):
    """Toggle the score legend collapse"""
    return not is_open


# ============================================================
# ELITE PAGE - GROK ANALYSIS MODAL CALLBACKS
# ============================================================

@callback(
    [Output('grok-analysis-modal', 'is_open'),
     Output('grok-modal-title', 'children'),
     Output('grok-modal-body', 'children')],
    [Input({'type': 'grok-analysis-btn', 'symbol': dash.ALL}, 'n_clicks'),
     Input('grok-modal-close', 'n_clicks')],
    State('grok-analysis-modal', 'is_open'),
    prevent_initial_call=True
)
def handle_grok_modal(btn_clicks, close_click, is_open):
    """Handle Grok analysis modal open/close"""
    triggered = ctx.triggered_id

    # Close button clicked
    if triggered == 'grok-modal-close':
        return False, "", ""

    # Grok button clicked - check which one
    if isinstance(triggered, dict) and triggered.get('type') == 'grok-analysis-btn':
        symbol = triggered.get('symbol', '?')

        # Check if any button was actually clicked (not just initialized)
        if not any(btn_clicks):
            return False, "", ""

        # Generate Grok analysis
        title = f"Analyse Approfondie - {symbol}"
        body = _generate_grok_analysis(symbol)

        return True, title, body

    return is_open, dash.no_update, dash.no_update


def _generate_grok_analysis(symbol: str):
    """Generate comprehensive Grok analysis for a symbol"""
    # Get signal data from scan state
    state = _load_scan_state()
    alerts = state.get('alerts', [])

    # Find the signal
    signal = None
    for a in alerts:
        if a.get('symbol') == symbol:
            signal = a
            break

    if not signal:
        return dbc.Alert(f"Signal {symbol} non trouve dans le scan actuel", color="warning")

    details = signal.get('l2_details', {})
    sentiment = details.get('sentiment', {})

    # Check if Grok is configured
    if not GROK_ACTIVE:
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Grok API non configure. Configurez GROK_API_KEY dans settings.py pour activer l'analyse approfondie."
            ], color="warning"),
            html.Hr(),
            html.H5("Donnees Disponibles"),
            _build_signal_summary(signal)
        ])

    # If sentiment was already analyzed
    if sentiment.get('status') == 'success':
        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                "Analyse Grok deja effectuee lors du scan"
            ], color="success"),
            html.Hr(),
            _build_grok_report(signal, sentiment)
        ])

    # Call Grok API for deep analysis
    try:
        report = _call_grok_deep_analysis(symbol, signal)
        return report
    except Exception as e:
        return dbc.Alert(f"Erreur lors de l'analyse Grok: {e}", color="danger")


def _build_signal_summary(signal: Dict):
    """Build a summary of signal data without Grok"""
    details = signal.get('l2_details', {})
    health = details.get('health', {})
    context = details.get('context', {})
    asset_type = signal.get('l2_asset_type', 'EQUITY')

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H6("Technique", className="text-primary"),
                html.P(f"Score L1: {signal.get('confidence_score', 0):.0f}/100"),
                html.P(f"Prix: ${signal.get('current_price', 0):.2f}"),
                html.P(f"Support: ${signal.get('support_level', 0):.2f}"),
            ], md=4),
            dbc.Col([
                html.H6("Fondamental", className="text-success"),
                html.P(f"Sante: {health.get('total', 0)}/20"),
                html.P(f"Revenue: {health.get('revenue_growth', 'N/A')}%") if health.get('revenue_growth') else None,
                html.P(f"Marge: {health.get('profit_margins', 'N/A')}%") if health.get('profit_margins') else None,
            ], md=4),
            dbc.Col([
                html.H6("Contexte", className="text-info"),
                html.P(f"Context: {context.get('total', 0)}/10"),
                html.P(f"SPY bullish: {'Oui' if context.get('spy_above_ema20w') else 'Non'}"),
            ], md=4),
        ])
    ])


def _build_grok_report(signal: Dict, sentiment: Dict):
    """Build the Grok analysis report"""
    return html.Div([
        # Sentiment Summary
        dbc.Card([
            dbc.CardHeader(html.H5("Sentiment Social", className="mb-0 text-warning")),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H3([
                            sentiment.get('sentiment', 'neutral').upper(),
                            html.Small(f" ({sentiment.get('sentiment_confidence', 0)}%)", className="text-muted")
                        ], className="text-center"),
                    ], md=4),
                    dbc.Col([
                        html.P([
                            html.Strong("Tendance mentions: "),
                            sentiment.get('mention_trend', 'stable').upper()
                        ]),
                        html.P([
                            html.Strong("Score: "),
                            f"{sentiment.get('total', 0)}/30 points"
                        ]),
                    ], md=4),
                    dbc.Col([
                        html.P(html.Strong("Catalyseurs:")),
                        html.Ul([
                            html.Li(cat) for cat in sentiment.get('catalysts', [])[:3]
                        ]) if sentiment.get('catalysts') else html.P("Aucun", className="text-muted")
                    ], md=4),
                ])
            ])
        ], className="mb-3"),

        # Summary if available
        html.Div([
            html.H6("Resume"),
            html.P(sentiment.get('summary', 'Non disponible'), className="text-muted fst-italic")
        ]) if sentiment.get('summary') else None,

        # Signal data summary
        html.Hr(),
        html.H5("Donnees Completes"),
        _build_signal_summary(signal)
    ])


def _call_grok_deep_analysis(symbol: str, signal: Dict):
    """Call Grok API for deep analysis (on-demand)"""
    import requests

    details = signal.get('l2_details', {})
    health = details.get('health', {})
    context = details.get('context', {})

    # Build prompt
    prompt = f"""Tu es un analyste financier expert. Genere un resume d'investissement pour {symbol} base sur ces donnees:

SCORES:
- Technique L1: {signal.get('confidence_score', 0):.0f}/100
- Fondamental L2: {signal.get('l2_score', 0):.0f}/60
- Sante: {health.get('total', 0)}/20
- Contexte: {context.get('total', 0)}/10

DONNEES:
- Prix actuel: ${signal.get('current_price', 0):.2f}
- Support EMA: ${signal.get('support_level', 0):.2f}
- Revenue Growth: {health.get('revenue_growth', 'N/A')}%
- Profit Margin: {health.get('profit_margins', 'N/A')}%

Reponds en francais avec:
1. RESUME (2-3 phrases)
2. POINTS FORTS (3 bullet points max)
3. RISQUES (2 bullet points max)
4. CONCLUSION (1 phrase: recommandation claire)"""

    try:
        headers = {
            'Authorization': f'Bearer {settings.GROK_API_KEY}',
            'Content-Type': 'application/json'
        }

        payload = {
            'model': getattr(settings, 'GROK_MODEL', 'grok-beta'),
            'messages': [
                {'role': 'system', 'content': 'Tu es un analyste financier expert. Reponds de maniere structuree.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.5,
            'max_tokens': 800
        }

        response = requests.post(
            getattr(settings, 'GROK_API_URL', 'https://api.x.ai/v1/chat/completions'),
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            return dbc.Alert(f"Erreur API Grok: {response.status_code}", color="danger")

        result = response.json()
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')

        return html.Div([
            dbc.Alert([
                html.I(className="fas fa-robot me-2"),
                "Analyse generee par Grok AI"
            ], color="info"),
            html.Hr(),
            dcc.Markdown(content, className="grok-analysis-content"),
            html.Hr(),
            html.Small("Analyse generee a la demande - non sauvegardee", className="text-muted")
        ])

    except requests.exceptions.Timeout:
        return dbc.Alert("Timeout - Grok n'a pas repondu a temps", color="warning")
    except Exception as e:
        return dbc.Alert(f"Erreur: {str(e)}", color="danger")


# ============================================================
# PORTFOLIO CALLBACKS
# ============================================================

@callback(
    [Output("portfolio-toast", "is_open"),
     Output("portfolio-toast", "children"),
     Output("portfolio-toast", "icon"),
     Output("portfolio-toast", "header"),
     Output("elite-portfolio-summary", "children"),
     Output("elite-signals-table", "children", allow_duplicate=True)],
    [Input({"type": "add-portfolio-btn", "symbol": ALL}, "n_clicks"),
     Input({"type": "remove-portfolio-btn", "symbol": ALL}, "n_clicks")],
    prevent_initial_call=True
)
def handle_portfolio_action(add_clicks, remove_clicks):
    """Handle add/remove from portfolio actions"""
    # Get which button was clicked
    triggered = ctx.triggered_id

    if triggered is None:
        return False, "", "success", "Portfolio", dash.no_update, dash.no_update

    action_type = triggered.get('type') if isinstance(triggered, dict) else None
    symbol = triggered.get('symbol') if isinstance(triggered, dict) else None

    if not action_type or not symbol:
        return False, "", "success", "Portfolio", dash.no_update, dash.no_update

    # Check if any button was actually clicked
    all_clicks = (add_clicks or []) + (remove_clicks or [])
    if not any(all_clicks):
        return False, "", "success", "Portfolio", dash.no_update, dash.no_update

    tracker = PortfolioTracker()

    if action_type == "add-portfolio-btn":
        # Add position
        state = _load_scan_state()
        alerts = state.get('alerts', [])
        signal = next((a for a in alerts if a.get('symbol') == symbol), None)

        if not signal:
            return True, f"Signal {symbol} non trouve", "danger", "Erreur", _create_portfolio_summary(), dash.no_update

        # Check if already in portfolio
        if tracker.get_position(symbol):
            return True, f"{symbol} est deja dans le portfolio", "warning", "Attention", _create_portfolio_summary(), dash.no_update

        # Check available capital
        available = tracker.get_available_capital()
        position_value = signal.get('position_value', 0)

        if position_value > available:
            return True, f"Capital insuffisant: {available:.0f}EUR disponible, {position_value:.0f}EUR requis", "danger", "Capital", _create_portfolio_summary(), dash.no_update

        # Add the position
        tracker.add_position(
            symbol=symbol,
            shares=signal.get('position_shares', 0),
            entry_price=signal.get('current_price', 0),
            stop_loss=signal.get('stop_loss', 0)
        )

        # Rebuild elite signals table to update button states
        signals = _get_elite_signals_from_scan()
        elite_signals = [s for s in signals if s.get('l2_is_elite', False) and not s.get('l2_excluded', False)]
        elite_table = _build_elite_table(elite_signals) if elite_signals else dbc.Alert(f"Aucun signal ELITE", color="info")

        msg = f"{symbol} ajoute: {signal.get('position_shares')} actions @ ${signal.get('current_price', 0):.2f}"
        return True, msg, "success", "Position Ajoutee", _create_portfolio_summary(), elite_table

    elif action_type == "remove-portfolio-btn":
        # Remove position
        position = tracker.close_position(symbol)

        if not position:
            return True, f"Position {symbol} non trouvee", "warning", "Attention", _create_portfolio_summary(), dash.no_update

        # Rebuild elite signals table to update button states
        signals = _get_elite_signals_from_scan()
        elite_signals = [s for s in signals if s.get('l2_is_elite', False) and not s.get('l2_excluded', False)]
        elite_table = _build_elite_table(elite_signals) if elite_signals else dbc.Alert(f"Aucun signal ELITE", color="info")

        msg = f"{symbol} retire du portfolio ({position.shares} actions)"
        return True, msg, "info", "Position Fermee", _create_portfolio_summary(), elite_table

    return False, "", "success", "Portfolio", dash.no_update, dash.no_update


# ============================================================
# INTELLIGENCE PAGE CALLBACKS
# ============================================================
@callback(
    Output('intel-refresh-status', 'children'),
    Input('intel-refresh-btn', 'n_clicks'),
    prevent_initial_call=True
)
def refresh_sector_data(n_clicks):
    """Refresh sector momentum data"""
    from datetime import datetime
    return f"Last refreshed: {datetime.now().strftime('%H:%M:%S')}"


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == '__main__':
    print("Starting Market Screener Dashboard (Dash version)...")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run(debug=True, use_reloader=False, port=8050)
