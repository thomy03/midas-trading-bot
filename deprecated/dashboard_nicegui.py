"""
TradingBot V3 - Modern Dashboard (NiceGUI)
Interface moderne, fluide et réactive

DEPRECATED: This dashboard is deprecated in favor of webapp.py
Please use: python webapp.py
This file was an early NiceGUI prototype - all features are now in webapp.py.
"""
import warnings
warnings.warn(
    "dashboard_nicegui.py is DEPRECATED. Please use 'python webapp.py' instead.",
    DeprecationWarning,
    stacklevel=2
)

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from nicegui import ui, app
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List

# Import trading components
from src.data.market_data import market_data_fetcher
from src.screening.screener import market_screener
from src.indicators.ema_analyzer import ema_analyzer
from trendline_analysis.core.enhanced_rsi_breakout_analyzer import EnhancedRSIBreakoutAnalyzer
from src.intelligence.trend_discovery import TrendDiscovery
from config import settings
import json
import asyncio
import glob

# Global state
current_symbol = 'AAPL'
current_timeframe = '1d'
current_period = '1y'
chart_container = None
rsi_analyzer = EnhancedRSIBreakoutAnalyzer(precision_mode='medium')


def create_chart(symbol: str, timeframe: str, period: str, show_emas: bool = True,
                 show_rsi: bool = True, show_trendline: bool = True) -> go.Figure:
    """Create interactive candlestick chart with indicators"""

    # Fetch data
    df = market_data_fetcher.get_historical_data(symbol, period=period, interval=timeframe)

    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No data for {symbol}", x=0.5, y=0.5,
                          xref="paper", yref="paper", showarrow=False, font_size=20)
        return fig

    # Convert index to strings for JSON serialization
    df = df.copy()
    df.index = df.index.strftime('%Y-%m-%d %H:%M')

    # Add EMAs
    df = ema_analyzer.calculate_emas(df)

    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Create subplots
    row_heights = [0.5, 0.15, 0.35] if show_rsi else [0.7, 0.3]
    rows = 3 if show_rsi else 2

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=(f'{symbol} - {timeframe.upper()}', 'Volume', 'RSI (14)') if show_rsi
                       else (f'{symbol} - {timeframe.upper()}', 'Volume')
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)

    # EMAs
    if show_emas:
        colors = {'EMA_24': '#2196F3', 'EMA_38': '#FF9800', 'EMA_62': '#9C27B0'}
        for ema, color in colors.items():
            if ema in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[ema],
                    name=ema.replace('_', ' '),
                    line=dict(color=color, width=1.5),
                    opacity=0.8
                ), row=1, col=1)

    # Volume
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.5
    ), row=2, col=1)

    # RSI
    if show_rsi:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'],
            name='RSI',
            line=dict(color='#7C4DFF', width=2)
        ), row=3, col=1)

        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)

        # RSI Trendline
        if show_trendline:
            try:
                result = rsi_analyzer.analyze(df, lookback_periods=len(df)-50)
                if result and result.rsi_trendline:
                    tl = result.rsi_trendline
                    # Draw trendline
                    start_idx = tl.peak_indices[0] if tl.peak_indices else 0
                    end_idx = len(df) - 1

                    x_vals = [df.index[start_idx], df.index[end_idx]]
                    y_vals = [tl.slope * start_idx + tl.intercept,
                              tl.slope * end_idx + tl.intercept]

                    fig.add_trace(go.Scatter(
                        x=x_vals, y=y_vals,
                        name='RSI Trendline',
                        line=dict(color='#FF6B6B', width=2, dash='dash'),
                        mode='lines'
                    ), row=3, col=1)

                    # Peak markers
                    peak_x = [df.index[i] for i in tl.peak_indices if i < len(df)]
                    peak_y = [df['RSI'].iloc[i] for i in tl.peak_indices if i < len(df)]
                    fig.add_trace(go.Scatter(
                        x=peak_x, y=peak_y,
                        mode='markers',
                        name='RSI Peaks',
                        marker=dict(color='#FF6B6B', size=10, symbol='triangle-down')
                    ), row=3, col=1)

                    # Breakout marker
                    if result.has_rsi_breakout:
                        bo = result.rsi_breakout
                        if bo.index < len(df):
                            fig.add_trace(go.Scatter(
                                x=[df.index[bo.index]],
                                y=[df['RSI'].iloc[bo.index]],
                                mode='markers',
                                name=f'BREAKOUT ({bo.strength})',
                                marker=dict(
                                    color='#4CAF50' if bo.strength == 'STRONG' else '#FFC107',
                                    size=15,
                                    symbol='star'
                                )
                            ), row=3, col=1)
            except Exception as e:
                print(f"Trendline error: {e}")

    # Layout - Dark theme
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#16213e',
        font=dict(color='#e0e0e0'),
        height=700,
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )

    # Grid styling
    for i in range(1, rows + 1):
        fig.update_xaxes(
            gridcolor='#2a2a4a',
            zerolinecolor='#2a2a4a',
            row=i, col=1
        )
        fig.update_yaxes(
            gridcolor='#2a2a4a',
            zerolinecolor='#2a2a4a',
            row=i, col=1
        )

    return fig


def update_chart():
    """Update chart with current settings"""
    global chart_container
    if chart_container:
        fig = create_chart(
            current_symbol,
            current_timeframe,
            current_period,
            show_emas=True,
            show_rsi=True,
            show_trendline=True
        )
        chart_container.figure = fig
        chart_container.update()


# ============================================================
# UI COMPONENTS
# ============================================================

@ui.page('/')
def main_page():
    global chart_container, current_symbol, current_timeframe, current_period

    # Dark theme CSS
    ui.add_head_html('''
        <style>
            body { background: #0f0f23 !important; }
            .q-card { background: #1a1a2e !important; border: 1px solid #2a2a4a; }
            .q-input, .q-select { background: #16213e !important; }
            .q-btn { border-radius: 8px !important; }
            .nicegui-plotly { border-radius: 12px; overflow: hidden; }
        </style>
    ''')

    with ui.header().classes('bg-[#1a1a2e] border-b border-[#2a2a4a]'):
        with ui.row().classes('w-full items-center justify-between px-4'):
            ui.label('TradingBot V3').classes('text-2xl font-bold text-[#7C4DFF]')

            with ui.row().classes('gap-2'):
                ui.button('Home', icon='home', on_click=lambda: ui.navigate.to('/')).props('flat color=white')
                ui.button('Trends', icon='trending_up', on_click=lambda: ui.navigate.to('/trends')).props('flat color=white')
                ui.button('Screening', icon='search', on_click=lambda: ui.navigate.to('/screening')).props('flat color=white')
                ui.button('Portfolio', icon='account_balance_wallet', on_click=lambda: ui.navigate.to('/portfolio')).props('flat color=white')

    with ui.column().classes('w-full p-4 gap-4'):
        # Controls row
        with ui.card().classes('w-full'):
            with ui.row().classes('w-full items-center gap-4 flex-wrap'):
                symbol_input = ui.input(
                    'Symbol',
                    value=current_symbol,
                    on_change=lambda e: set_symbol(e.value)
                ).classes('w-32')

                timeframe_select = ui.select(
                    ['1d', '1wk', '1h', '4h'],
                    value=current_timeframe,
                    label='Timeframe',
                    on_change=lambda e: set_timeframe(e.value)
                ).classes('w-28')

                period_select = ui.select(
                    ['1mo', '3mo', '6mo', '1y', '2y', '5y'],
                    value=current_period,
                    label='Period',
                    on_change=lambda e: set_period(e.value)
                ).classes('w-28')

                ui.button('Load Chart', icon='refresh', on_click=update_chart).props('color=primary')

                ui.space()

                # Quick symbols
                with ui.row().classes('gap-1'):
                    for sym in ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'MC.PA']:
                        ui.button(sym, on_click=lambda s=sym: quick_load(s)).props('flat dense color=grey')

        # Main chart
        with ui.card().classes('w-full'):
            chart_container = ui.plotly(create_chart(current_symbol, current_timeframe, current_period))
            chart_container.classes('w-full h-[700px]')

        # Info cards
        with ui.row().classes('w-full gap-4'):
            with ui.card().classes('flex-1'):
                ui.label('Signal Status').classes('text-lg font-bold text-[#7C4DFF] mb-2')
                signal_label = ui.label('Loading...').classes('text-2xl')

            with ui.card().classes('flex-1'):
                ui.label('RSI').classes('text-lg font-bold text-[#7C4DFF] mb-2')
                rsi_label = ui.label('--').classes('text-2xl')

            with ui.card().classes('flex-1'):
                ui.label('EMA Status').classes('text-lg font-bold text-[#7C4DFF] mb-2')
                ema_label = ui.label('--').classes('text-2xl')

            with ui.card().classes('flex-1'):
                ui.label('Price').classes('text-lg font-bold text-[#7C4DFF] mb-2')
                price_label = ui.label('--').classes('text-2xl')

        # Update info
        update_info_cards(signal_label, rsi_label, ema_label, price_label)


def set_symbol(value):
    global current_symbol
    current_symbol = value.upper()


def set_timeframe(value):
    global current_timeframe
    current_timeframe = value


def set_period(value):
    global current_period
    current_period = value


def quick_load(symbol):
    global current_symbol
    current_symbol = symbol
    update_chart()


def update_info_cards(signal_label, rsi_label, ema_label, price_label):
    """Update information cards"""
    try:
        df = market_data_fetcher.get_historical_data(current_symbol, period='3mo', interval='1d')
        if df is not None and not df.empty:
            df = ema_analyzer.calculate_emas(df)

            # Price
            price = df['Close'].iloc[-1]
            price_label.text = f'${price:.2f}'

            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]
            rsi_label.text = f'{rsi:.1f}'

            # EMA status
            ema24 = df['EMA_24'].iloc[-1]
            ema38 = df['EMA_38'].iloc[-1]
            ema62 = df['EMA_62'].iloc[-1]

            if ema24 > ema38 > ema62:
                ema_label.text = 'Bullish'
                ema_label.classes('text-green-400', remove='text-red-400 text-yellow-400')
            elif ema24 < ema38 < ema62:
                ema_label.text = 'Bearish'
                ema_label.classes('text-red-400', remove='text-green-400 text-yellow-400')
            else:
                ema_label.text = 'Mixed'
                ema_label.classes('text-yellow-400', remove='text-green-400 text-red-400')

            # Signal
            result = rsi_analyzer.analyze(df, lookback_periods=100)
            if result and result.has_rsi_breakout:
                signal_label.text = f'{result.signal}'
                signal_label.classes('text-green-400', remove='text-gray-400')
            else:
                signal_label.text = 'No Signal'
                signal_label.classes('text-gray-400', remove='text-green-400')

    except Exception as e:
        print(f"Error updating cards: {e}")


@ui.page('/screening')
def screening_page():
    ui.add_head_html('''<style>body { background: #0f0f23 !important; }</style>''')

    with ui.header().classes('bg-[#1a1a2e] border-b border-[#2a2a4a]'):
        with ui.row().classes('w-full items-center px-4'):
            ui.label('TradingBot V3 - Screening').classes('text-2xl font-bold text-[#7C4DFF]')
            ui.space()
            ui.button('Back to Chart', icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat color=white')

    with ui.column().classes('w-full p-4 gap-4'):
        with ui.card().classes('w-full'):
            ui.label('Market Screening').classes('text-xl font-bold mb-4')

            with ui.row().classes('gap-4'):
                market_select = ui.select(
                    ['NASDAQ', 'SP500', 'CAC40', 'EUROPE'],
                    value='NASDAQ',
                    label='Market'
                ).classes('w-40')

                ui.button('Start Scan', icon='play_arrow', on_click=lambda: start_scan(market_select.value)).props('color=primary')

        results_container = ui.column().classes('w-full gap-2')


def start_scan(market: str):
    ui.notify(f'Scanning {market}...', type='info')


@ui.page('/portfolio')
def portfolio_page():
    ui.add_head_html('''<style>body { background: #0f0f23 !important; }</style>''')

    with ui.header().classes('bg-[#1a1a2e] border-b border-[#2a2a4a]'):
        with ui.row().classes('w-full items-center px-4'):
            ui.label('TradingBot V3 - Portfolio').classes('text-2xl font-bold text-[#7C4DFF]')
            ui.space()
            ui.button('Back to Chart', icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat color=white')

    with ui.column().classes('w-full p-4'):
        with ui.card().classes('w-full'):
            ui.label('Portfolio Tracker').classes('text-xl font-bold mb-4')
            ui.label('Coming soon...').classes('text-gray-400')


# ============================================================
# TREND DISCOVERY PAGE
# ============================================================

def load_trend_report():
    """Load the latest trend report."""
    report_dir = settings.TREND_DATA_DIR
    if not os.path.exists(report_dir):
        return None

    report_files = glob.glob(os.path.join(report_dir, 'trend_report_*.json'))
    if not report_files:
        return None

    latest_file = max(report_files, key=os.path.getmtime)
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data['_file_path'] = latest_file
            data['_file_date'] = datetime.fromtimestamp(os.path.getmtime(latest_file))
            return data
    except Exception:
        return None


def load_learned_themes():
    """Load themes discovered by AI."""
    path = os.path.join(settings.TREND_DATA_DIR, 'learned_themes.json')
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@ui.page('/trends')
def trends_page():
    ui.add_head_html('''<style>body { background: #0f0f23 !important; }</style>''')

    # Header
    with ui.header().classes('bg-[#1a1a2e] border-b border-[#2a2a4a]'):
        with ui.row().classes('w-full items-center px-4'):
            ui.label('🔮 Trend Discovery').classes('text-2xl font-bold text-[#7C4DFF]')
            ui.space()
            ui.button('Home', icon='home', on_click=lambda: ui.navigate.to('/')).props('flat color=white')
            ui.button('Screening', icon='search', on_click=lambda: ui.navigate.to('/screening')).props('flat color=white')

    # Load data
    report_data = load_trend_report()
    learned_themes = load_learned_themes()

    with ui.column().classes('w-full p-4 gap-4'):
        # Tabs
        with ui.tabs().classes('w-full') as tabs:
            overview_tab = ui.tab('Overview', icon='dashboard')
            trends_tab = ui.tab('Tendances', icon='trending_up')
            themes_tab = ui.tab('Thèmes IA', icon='psychology')
            sources_tab = ui.tab('Sources', icon='source')

        with ui.tab_panels(tabs, value=overview_tab).classes('w-full'):
            # ============================================
            # TAB 1: Overview
            # ============================================
            with ui.tab_panel(overview_tab):
                if report_data:
                    trends = report_data.get('trends', [])
                    sentiment = report_data.get('market_sentiment', 0)
                    narratives = report_data.get('narrative_updates', [])
                    file_date = report_data.get('_file_date', datetime.now())

                    # Metrics row
                    with ui.row().classes('w-full gap-4'):
                        with ui.card().classes('flex-1 text-center'):
                            ui.label('📈 Tendances').classes('text-gray-400')
                            ui.label(str(len(trends))).classes('text-4xl font-bold text-[#7C4DFF]')

                        with ui.card().classes('flex-1 text-center'):
                            sentiment_color = 'text-green-400' if sentiment > 0.2 else 'text-yellow-400' if sentiment > -0.2 else 'text-red-400'
                            ui.label('💹 Sentiment').classes('text-gray-400')
                            ui.label(f'{sentiment:+.2f}').classes(f'text-4xl font-bold {sentiment_color}')

                        with ui.card().classes('flex-1 text-center'):
                            ui.label('🆕 Nouveaux Narratifs').classes('text-gray-400')
                            ui.label(str(len(narratives))).classes('text-4xl font-bold text-[#FF6B6B]')

                        with ui.card().classes('flex-1 text-center'):
                            ui.label('🧠 Thèmes Appris').classes('text-gray-400')
                            ui.label(str(len(learned_themes))).classes('text-4xl font-bold text-[#4CAF50]')

                    # Last scan info + Run button
                    with ui.card().classes('w-full'):
                        with ui.row().classes('w-full items-center justify-between'):
                            ui.label(f"📅 Dernier scan: {file_date.strftime('%d/%m/%Y %H:%M')}").classes('text-lg')

                            async def run_scan():
                                ui.notify('Scan en cours...', type='info')
                                try:
                                    discovery = TrendDiscovery(
                                        openrouter_api_key=settings.OPENROUTER_API_KEY,
                                        model=settings.OPENROUTER_MODEL,
                                        data_dir=settings.TREND_DATA_DIR
                                    )
                                    await discovery.initialize()
                                    report = await discovery.daily_scan()
                                    await discovery.close()
                                    ui.notify(f'✅ Scan terminé! {len(report.trends)} tendances', type='positive')
                                    ui.navigate.reload()
                                except Exception as e:
                                    ui.notify(f'Erreur: {e}', type='negative')

                            ui.button('🚀 Lancer un Scan', on_click=run_scan).props('color=primary')

                    # Top Trends Table
                    ui.label('🔥 Top Tendances').classes('text-xl font-bold mt-4 mb-2')

                    columns = [
                        {'name': 'name', 'label': 'Nom', 'field': 'name', 'align': 'left'},
                        {'name': 'type', 'label': 'Type', 'field': 'type'},
                        {'name': 'strength', 'label': 'Force', 'field': 'strength'},
                        {'name': 'confidence', 'label': 'Confiance', 'field': 'confidence'},
                        {'name': 'symbols', 'label': 'Symboles', 'field': 'symbols', 'align': 'left'},
                    ]

                    rows = []
                    for t in sorted(trends, key=lambda x: x.get('confidence', 0), reverse=True)[:10]:
                        rows.append({
                            'name': t.get('name', 'N/A'),
                            'type': t.get('type', '').replace('_', ' ').title(),
                            'strength': t.get('strength', '').title(),
                            'confidence': f"{t.get('confidence', 0)*100:.0f}%",
                            'symbols': ', '.join(t.get('symbols', [])[:4])
                        })

                    ui.table(columns=columns, rows=rows).classes('w-full')

                else:
                    with ui.card().classes('w-full p-8 text-center'):
                        ui.icon('cloud_off', size='xl').classes('text-gray-500')
                        ui.label('Aucun rapport disponible').classes('text-xl text-gray-400 mt-4')
                        ui.label('Lancez un scan pour découvrir les tendances').classes('text-gray-500')

            # ============================================
            # TAB 2: Tendances Détaillées
            # ============================================
            with ui.tab_panel(trends_tab):
                if report_data:
                    trends = report_data.get('trends', [])

                    # Filters
                    with ui.row().classes('w-full gap-4 mb-4'):
                        type_filter = ui.select(
                            ['Tous', 'Sector Momentum', 'Thematic'],
                            value='Tous',
                            label='Type'
                        ).classes('w-40')

                        strength_filter = ui.select(
                            ['Tous', 'Established', 'Developing', 'Emerging'],
                            value='Tous',
                            label='Force'
                        ).classes('w-40')

                    # Trends list
                    for trend in sorted(trends, key=lambda x: x.get('confidence', 0), reverse=True):
                        conf = trend.get('confidence', 0) * 100
                        strength = trend.get('strength', 'emerging')
                        icon = '🔥' if strength == 'established' else '📈' if strength == 'developing' else '🌱'

                        strength_color = 'border-green-500' if strength == 'established' else 'border-yellow-500' if strength == 'developing' else 'border-blue-500'

                        with ui.expansion(f"{icon} {trend.get('name', 'N/A')} ({conf:.0f}%)").classes(f'w-full border-l-4 {strength_color}'):
                            with ui.row().classes('w-full gap-8'):
                                with ui.column().classes('flex-1'):
                                    ui.label(trend.get('description', 'N/A')).classes('text-gray-300')

                                    catalysts = trend.get('key_catalysts', [])
                                    if catalysts:
                                        ui.label('Catalyseurs:').classes('font-bold mt-2')
                                        for cat in catalysts:
                                            ui.label(f'• {cat}').classes('text-gray-400 ml-4')

                                with ui.column().classes('w-48'):
                                    ui.label(f"Type: {trend.get('type', 'N/A').replace('_', ' ').title()}").classes('text-sm')
                                    ui.label(f"Force: {strength.title()}").classes('text-sm')
                                    ui.label(f"Momentum: {trend.get('momentum_score', 0):.1%}").classes('text-sm')

                                    sources = trend.get('sources', [])
                                    if sources:
                                        ui.label(f"Sources: {', '.join(sources)}").classes('text-sm text-gray-500')

                            # Symbols
                            symbols = trend.get('symbols', [])
                            if symbols:
                                with ui.row().classes('gap-2 mt-2 flex-wrap'):
                                    for sym in symbols[:8]:
                                        ui.badge(sym).props('color=primary')
                else:
                    ui.label('Aucune donnée').classes('text-gray-400')

            # ============================================
            # TAB 3: Thèmes IA
            # ============================================
            with ui.tab_panel(themes_tab):
                with ui.row().classes('w-full gap-4'):
                    # Predefined themes
                    with ui.column().classes('flex-1'):
                        ui.label('📚 Thèmes Prédéfinis').classes('text-xl font-bold mb-4')
                        ui.label(f'{len(TrendDiscovery.THEMES_KEYWORDS)} thèmes de base').classes('text-gray-400 mb-4')

                        for theme_name, keywords in TrendDiscovery.THEMES_KEYWORDS.items():
                            with ui.card().classes('w-full mb-2 bg-[#16213e]'):
                                ui.label(theme_name).classes('font-bold text-[#7C4DFF]')
                                ui.label(', '.join(keywords[:4]) + '...').classes('text-sm text-gray-400')

                    # Learned themes
                    with ui.column().classes('flex-1'):
                        ui.label('🆕 Thèmes Découverts par IA').classes('text-xl font-bold mb-4')
                        ui.label(f'{len(learned_themes)} thèmes appris').classes('text-gray-400 mb-4')

                        if learned_themes:
                            for theme_name, data in learned_themes.items():
                                occ = data.get('occurrence_count', 1)
                                discovered = data.get('discovered_at', '')[:10]
                                desc = data.get('description', '')[:100]
                                symbols = data.get('symbols', [])

                                with ui.card().classes('w-full mb-2 bg-[#1a3a2e] border-l-4 border-green-500'):
                                    with ui.row().classes('justify-between'):
                                        ui.label(data.get('name', theme_name)).classes('font-bold text-green-400')
                                        ui.badge(f'vu {occ}x').props('color=positive')

                                    ui.label(f'📅 {discovered}').classes('text-xs text-gray-500')
                                    ui.label(desc + '...').classes('text-sm text-gray-300 mt-1')

                                    if symbols:
                                        with ui.row().classes('gap-1 mt-2 flex-wrap'):
                                            for sym in symbols[:5]:
                                                ui.badge(sym, color='teal').props('outline')
                        else:
                            with ui.card().classes('w-full p-4 text-center bg-[#16213e]'):
                                ui.icon('lightbulb', size='lg').classes('text-yellow-500')
                                ui.label('Aucun thème découvert').classes('text-gray-400')
                                ui.label('Lancez un scan pour que l\'IA identifie de nouveaux narratifs').classes('text-sm text-gray-500')

            # ============================================
            # TAB 4: Sources & Stats
            # ============================================
            with ui.tab_panel(sources_tab):
                ui.label('📡 Sources de Données').classes('text-xl font-bold mb-4')

                # Sources status table
                sources_data = [
                    {'source': 'yfinance', 'type': 'Prix & Volume', 'status': '✅ Active', 'config': 'N/A'},
                    {'source': 'NewsAPI', 'type': 'Actualités', 'status': '✅ Active' if settings.NEWSAPI_KEY else '⚠️ Non configuré', 'config': 'NEWSAPI_KEY'},
                    {'source': 'OpenRouter (LLM)', 'type': 'Analyse IA', 'status': '✅ Active' if settings.OPENROUTER_API_KEY else '⚠️ Non configuré', 'config': 'OPENROUTER_API_KEY'},
                    {'source': 'AlphaVantage', 'type': 'News Sentiment', 'status': '✅ Active' if getattr(settings, 'ALPHAVANTAGE_KEY', '') else '⚠️ Non configuré', 'config': 'ALPHAVANTAGE_KEY'},
                ]

                columns = [
                    {'name': 'source', 'label': 'Source', 'field': 'source', 'align': 'left'},
                    {'name': 'type', 'label': 'Type', 'field': 'type'},
                    {'name': 'status', 'label': 'Statut', 'field': 'status'},
                    {'name': 'config', 'label': 'Variable', 'field': 'config'},
                ]

                ui.table(columns=columns, rows=sources_data).classes('w-full')

                # Scan history
                ui.label('📜 Historique des Scans').classes('text-xl font-bold mt-8 mb-4')

                report_dir = settings.TREND_DATA_DIR
                if os.path.exists(report_dir):
                    report_files = glob.glob(os.path.join(report_dir, 'trend_report_*.json'))
                    report_files = sorted(report_files, key=os.path.getmtime, reverse=True)[:10]

                    if report_files:
                        for rf in report_files:
                            file_date = datetime.fromtimestamp(os.path.getmtime(rf))
                            try:
                                with open(rf, 'r', encoding='utf-8') as f:
                                    rdata = json.load(f)
                                n_trends = len(rdata.get('trends', []))
                                n_narratives = len(rdata.get('narrative_updates', []))
                                sentiment = rdata.get('market_sentiment', 0)

                                with ui.row().classes('w-full items-center gap-4 py-2 border-b border-[#2a2a4a]'):
                                    ui.label(file_date.strftime('%d/%m/%Y %H:%M')).classes('w-40 font-mono')
                                    ui.badge(f'{n_trends} trends').props('color=primary')
                                    ui.badge(f'{n_narratives} narratifs').props('color=accent')
                                    sentiment_color = 'positive' if sentiment > 0 else 'negative'
                                    ui.badge(f'sentiment: {sentiment:+.2f}').props(f'color={sentiment_color}')
                            except Exception:
                                ui.label(f'{file_date.strftime("%d/%m/%Y %H:%M")} - Erreur').classes('text-red-400')
                    else:
                        ui.label('Aucun historique disponible').classes('text-gray-400')
                else:
                    ui.label(f'Répertoire {report_dir} non trouvé').classes('text-yellow-400')

                # Focus symbols
                ui.label('🎯 Symboles Focus').classes('text-xl font-bold mt-8 mb-4')

                focus_path = os.path.join(settings.TREND_DATA_DIR, 'focus_symbols.json')
                if os.path.exists(focus_path):
                    try:
                        with open(focus_path, 'r', encoding='utf-8') as f:
                            focus_data = json.load(f)
                        symbols = focus_data.get('symbols', [])
                        if symbols:
                            with ui.row().classes('gap-2 flex-wrap'):
                                for sym in symbols:
                                    ui.badge(sym).props('color=primary clickable')
                    except Exception as e:
                        ui.label(f'Erreur: {e}').classes('text-red-400')
                else:
                    ui.label('Aucun symbole focus. Lancez un scan.').classes('text-gray-400')


# ============================================================
# RUN
# ============================================================

if __name__ in {"__main__", "__mp_main__"}:
    print("\n" + "="*60)
    print("TradingBot V3 - Modern Dashboard (NiceGUI)")
    print("="*60)
    print("\nOpen http://localhost:8080 in your browser")
    print("="*60 + "\n")

    ui.run(
        title='TradingBot V3',
        port=8080,
        dark=True,
        reload=False,
        show=False
    )
