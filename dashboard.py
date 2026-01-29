"""
Market Screener Dashboard - TradingView-like Interface

DEPRECATED: This dashboard is deprecated in favor of webapp.py
Please use: python webapp.py
This file is kept for reference only and will be removed in a future version.

Run with: streamlit run dashboard.py
"""
import warnings
warnings.warn(
    "dashboard.py is DEPRECATED. Please use 'python webapp.py' instead.",
    DeprecationWarning,
    stacklevel=2
)
import streamlit as st
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from src.utils.visualizer import chart_visualizer
from src.utils.interactive_chart import interactive_chart_builder
from src.data.market_data import market_data_fetcher
from src.screening.screener import market_screener
from src.database.db_manager import db_manager
from src.indicators.ema_analyzer import ema_analyzer
from src.utils.background_scanner import background_scanner, ScanStatus, scan_scheduler
from src.utils.watchlist_manager import watchlist_manager
from src.utils.trade_tracker import trade_tracker
from src.utils.sector_heatmap import sector_heatmap_builder
from src.utils.economic_calendar import economic_calendar, EventType, EventImpact
from src.utils.notification_manager import notification_manager, NotificationPriority
from src.utils.sector_analyzer import SectorAnalyzer
from src.intelligence import NewsFetcher, get_news_fetcher
from src.intelligence.trend_discovery import TrendDiscovery, get_trend_discovery
from config.settings import EMA_PERIODS, ZONE_TOLERANCE, MARKETS_EXTENDED, CAPITAL
from config import settings
import json
import asyncio
import glob as glob_module

# Page config
st.set_page_config(
    page_title="Market Screener Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #00C853;
    }
    h2 {
        color: #2962FF;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📈 Market Screener")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📈 Pro Chart", "📊 Chart Analyzer", "📈 Signaux Historiques", "🔍 Screening", "⏰ Scheduler", "🔬 Backtesting", "📋 Watchlists", "💼 Portfolio", "📊 Performance", "🗺️ Sector Map", "📅 Calendar", "🎯 Trendline Analysis", "🧠 Intelligence", "🔮 Trend Discovery", "🚨 Alerts History", "⚙️ Settings"]
    )

    st.markdown("---")
    st.markdown("### Quick Stats")

    # Get recent alerts count
    try:
        recent_alerts = db_manager.get_recent_alerts(days=7)
        st.metric("Alerts (7 days)", len(recent_alerts))
    except:
        st.metric("Alerts (7 days)", "N/A")

    st.markdown("---")
    st.caption("Developed with ❤️")

# Main content
if page == "🏠 Home":
    st.title("🏠 Market Screener Dashboard")
    st.markdown("### Welcome to your automated stock screening system!")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("**Strategy**\n\nEMA-based screening on Weekly and Daily timeframes")

    with col2:
        st.success(f"**EMAs Used**\n\n{', '.join(map(str, EMA_PERIODS))}")

    with col3:
        st.warning(f"**Support Zone**\n\n±{ZONE_TOLERANCE}% tolerance")

    st.markdown("---")

    # Recent alerts
    st.subheader("📊 Recent Alerts")

    try:
        alerts = db_manager.get_recent_alerts(days=7)

        if alerts:
            # Create dataframe
            alert_data = []
            for alert in alerts[:20]:  # Show last 20
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

            df = pd.DataFrame(alert_data)

            # Color code recommendations
            def color_recommendation(val):
                colors = {
                    'STRONG_BUY': 'background-color: #00C853; color: white',
                    'BUY': 'background-color: #64DD17; color: white',
                    'WATCH': 'background-color: #FDD835; color: black',
                    'OBSERVE': 'background-color: #FF6D00; color: white'
                }
                return colors.get(val, '')

            styled_df = df.style.applymap(color_recommendation, subset=['Recommendation'])
            st.dataframe(styled_df, use_container_width=True, height=400)

            # Alert summary chart
            st.subheader("Alert Distribution")
            fig = chart_visualizer.create_alert_summary_chart(
                [{'recommendation': a.recommendation} for a in alerts]
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recent alerts found. Run a screening to generate alerts!")

    except Exception as e:
        st.error(f"Error loading alerts: {e}")

elif page == "📈 Pro Chart":
    st.title("📈 Interactive Pro Chart")
    st.markdown("### TradingView-style interactive charting with drawing tools")

    # Toolbar CSS
    st.markdown("""
    <style>
        .chart-toolbar {
            background-color: #1E1E1E;
            padding: 0.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .stButton > button {
            background-color: #2962FF;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            margin-right: 0.5rem;
        }
        .stButton > button:hover {
            background-color: #1E88E5;
        }
    </style>
    """, unsafe_allow_html=True)

    # Symbol input row
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        pro_symbol = st.text_input(
            "Symbol",
            value=st.session_state.get('pro_symbol', 'AAPL'),
            key="pro_symbol_input",
            placeholder="Enter symbol (e.g., AAPL, TSLA)"
        ).upper()
        st.session_state['pro_symbol'] = pro_symbol

    with col2:
        pro_timeframe = st.selectbox(
            "Timeframe",
            ["daily", "weekly"],
            index=0 if st.session_state.get('pro_timeframe', 'daily') == 'daily' else 1,
            key="pro_timeframe_select"
        )
        st.session_state['pro_timeframe'] = pro_timeframe

    with col3:
        pro_period = st.selectbox(
            "Period",
            ["3mo", "6mo", "1y", "2y", "5y", "max"],
            index=2,
            key="pro_period_select"
        )

    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        load_chart = st.button("🔄 Load Chart", type="primary", key="load_pro_chart")

    # Chart options row
    st.markdown("---")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        show_volume = st.checkbox("📊 Volume", value=True, key="pro_volume")

    with col2:
        show_rsi = st.checkbox("📈 RSI", value=True, key="pro_rsi")

    with col3:
        show_emas = st.checkbox("〰️ EMAs", value=True, key="pro_emas")

    with col4:
        show_supports = st.checkbox("🔻 Supports", value=True, key="pro_supports")

    with col5:
        show_rsi_trendline = st.checkbox("📉 RSI Oblique", value=True, key="pro_rsi_trendline")

    with col6:
        chart_height = st.selectbox("Height", [600, 800, 1000, 1200], index=1, key="pro_height")

    with col7:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📐 Reset View", key="reset_view"):
            st.rerun()

    # Drawing tools info
    st.info("""
    **Interactive Controls:**
    - ✋ **Pan**: Click and drag to move the chart left/right
    - 🔍 **Zoom**: Use mouse scroll wheel to zoom in/out
    - 📏 **Draw Line**: Use toolbar button to draw trendlines
    - ⬜ **Draw Rectangle**: Mark support/resistance zones
    - 🔄 **Double-click**: Reset view to original

    **Indicateurs:**
    - ◇ **EMA Crossovers**: Diamants verts (bullish) / rouges (bearish)
    - 📉 **RSI Oblique**: Ligne orange pointillee = resistance RSI, ★ verte = breakout
    """)

    # Load and display chart
    if pro_symbol:
        with st.spinner(f"Loading {pro_symbol}..."):
            try:
                interval = '1wk' if pro_timeframe == 'weekly' else '1d'
                df = market_data_fetcher.get_historical_data(pro_symbol, period=pro_period, interval=interval)

                if df is not None and len(df) > 0:
                    # Load saved drawings
                    drawings = interactive_chart_builder.load_drawings(pro_symbol)

                    # Calculate EMAs and detect crossovers first (needed for both chart and support zones)
                    df_with_emas = ema_analyzer.calculate_emas(df)
                    current_price = float(df['Close'].iloc[-1] if not isinstance(df['Close'], pd.DataFrame) else df['Close'].iloc[-1, 0])
                    crossovers = ema_analyzer.detect_crossovers(df_with_emas, pro_timeframe)

                    # Create interactive chart with crossover zones
                    fig = interactive_chart_builder.create_interactive_chart(
                        df=df,
                        symbol=pro_symbol,
                        timeframe=pro_timeframe,
                        show_volume=show_volume,
                        show_rsi=show_rsi,
                        show_emas=show_emas,
                        ema_periods=EMA_PERIODS,
                        drawings=drawings,
                        height=chart_height,
                        crossovers=crossovers,
                        show_crossover_zones=True
                    )

                    # Calculate support zones (needed for RSI validation AND display)
                    support_zones = ema_analyzer.find_support_zones(df_with_emas, crossovers, current_price)

                    # Add support levels if enabled
                    if show_supports and support_zones:
                        levels = [
                            {
                                'price': zone['level'],
                                'type': 'support',
                                'strength': zone['strength']
                            }
                            for zone in support_zones[:5]  # Top 5 supports
                        ]
                        fig = interactive_chart_builder.add_support_resistance_levels(fig, levels, current_price)

                    # Add RSI trendline and breakout visualization
                    if show_rsi and show_rsi_trendline:
                        rsi_row = 3 if show_volume else 2
                        # Passer df_with_emas et support_zones pour validation du breakout
                        fig = interactive_chart_builder.add_rsi_trendline_breakout(
                            fig, df, rsi_row=rsi_row,
                            df_with_emas=df_with_emas,
                            support_zones=support_zones
                        )

                    # Display chart
                    st.plotly_chart(fig, use_container_width=True, config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToAdd': [
                            'drawline',
                            'drawopenpath',
                            'drawrect',
                            'eraseshape'
                        ],
                        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                        'scrollZoom': True,
                        'doubleClick': 'reset'
                    })

                    # Price info panel
                    st.markdown("---")
                    st.subheader("📋 Price Information")

                    # Extract data
                    close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
                    high = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
                    low = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']
                    volume = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']

                    current_price = float(close.iloc[-1])
                    prev_close = float(close.iloc[-2]) if len(close) > 1 else current_price
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close != 0 else 0

                    col1, col2, col3, col4, col5, col6 = st.columns(6)

                    with col1:
                        st.metric(
                            "Current Price",
                            f"${current_price:.2f}",
                            f"{change:+.2f} ({change_pct:+.2f}%)"
                        )

                    with col2:
                        period_high = float(high.max())
                        st.metric("Period High", f"${period_high:.2f}")

                    with col3:
                        period_low = float(low.min())
                        st.metric("Period Low", f"${period_low:.2f}")

                    with col4:
                        avg_volume = float(volume.mean())
                        st.metric("Avg Volume", f"{avg_volume/1e6:.2f}M")

                    with col5:
                        # Calculate EMAs
                        df_emas = ema_analyzer.calculate_emas(df)
                        ema24 = float(df_emas['EMA_24'].iloc[-1])
                        st.metric("EMA 24", f"${ema24:.2f}")

                    with col6:
                        ema62 = float(df_emas['EMA_62'].iloc[-1])
                        st.metric("EMA 62", f"${ema62:.2f}")

                    # Support zones table
                    if show_supports and support_zones:
                        st.markdown("---")
                        st.subheader("🔻 Detected Support Zones")

                        support_data = []
                        for zone in support_zones[:10]:
                            distance = ((current_price - zone['level']) / current_price) * 100
                            support_data.append({
                                'Level': f"${zone['level']:.2f}",
                                'Distance': f"{distance:.2f}%",
                                'Strength': f"{zone['strength']:.0f}%",
                                'Type': zone.get('type', 'EMA Cross'),
                                'Status': '🔴 NEAR' if distance < 8 else '🟢 Valid'
                            })

                        st.dataframe(pd.DataFrame(support_data), use_container_width=True)

                    # Crossovers info
                    if crossovers:
                        st.markdown("---")
                        with st.expander("📊 EMA Crossovers History"):
                            cross_data = []
                            for cross in crossovers[:15]:
                                cross_data.append({
                                    'Date': cross['date'].strftime('%Y-%m-%d') if hasattr(cross['date'], 'strftime') else str(cross['date']),
                                    'Type': cross['type'].upper(),
                                    'EMAs': f"{cross['fast_ema']}/{cross['slow_ema']}",
                                    'Price': f"${cross['price']:.2f}",
                                    'Age': f"{cross['days_ago']} {'weeks' if pro_timeframe == 'weekly' else 'days'}"
                                })

                            st.dataframe(pd.DataFrame(cross_data), use_container_width=True)

                else:
                    st.error(f"Could not load data for {pro_symbol}")

            except Exception as e:
                st.error(f"Error loading chart: {e}")
                import traceback
                st.code(traceback.format_exc())

elif page == "📊 Chart Analyzer":
    st.title("📊 Interactive Chart Analyzer")
    st.markdown("Visualize stocks with EMAs and support zones")

    col1, col2 = st.columns([2, 1])

    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", key="chart_symbol").upper()

    with col2:
        timeframe = st.selectbox("Timeframe", ["daily", "weekly"], key="chart_timeframe")

    col3, col4, col5, col6 = st.columns(4)

    with col3:
        period = st.selectbox("Period", ["6mo", "1y", "2y", "5y", "10y", "max"], index=3, key="chart_period")

    with col4:
        show_volume = st.checkbox("Show Volume", value=True, key="chart_volume")

    with col5:
        show_rsi_trendline_analyzer = st.checkbox("RSI Oblique", value=True, key="chart_rsi_trendline")

    with col6:
        if st.button("🔄 Refresh Chart", key="refresh_chart"):
            st.rerun()

    if symbol:
        with st.spinner(f"Loading chart for {symbol}..."):
            # Get market data
            interval = '1wk' if timeframe == 'weekly' else '1d'
            df = market_data_fetcher.get_historical_data(symbol, period=period, interval=interval)

            if df is not None and len(df) > 0:
                # Calculate EMAs and detect crossovers
                df_with_emas = ema_analyzer.calculate_emas(df)
                current_price = float(df['Close'].iloc[-1] if not isinstance(df['Close'], pd.DataFrame) else df['Close'].iloc[-1, 0])
                crossovers = ema_analyzer.detect_crossovers(df_with_emas, timeframe)
                support_zones = ema_analyzer.find_support_zones(df_with_emas, crossovers, current_price)

                # Create interactive chart with crossover zones
                fig = interactive_chart_builder.create_interactive_chart(
                    df=df,
                    symbol=symbol,
                    timeframe=timeframe,
                    show_volume=show_volume,
                    show_rsi=True,
                    show_emas=True,
                    ema_periods=EMA_PERIODS,
                    height=800,
                    crossovers=crossovers,
                    show_crossover_zones=True
                )

                # Add support levels
                if support_zones:
                    levels = [
                        {
                            'price': zone['level'],
                            'type': 'support',
                            'strength': zone['strength']
                        }
                        for zone in support_zones[:5]
                    ]
                    fig = interactive_chart_builder.add_support_resistance_levels(fig, levels, current_price)

                # Add RSI trendline and breakout visualization
                if show_rsi_trendline_analyzer:
                    rsi_row = 3 if show_volume else 2
                    # Passer df_with_emas et support_zones pour validation du breakout
                    fig = interactive_chart_builder.add_rsi_trendline_breakout(
                        fig, df, rsi_row=rsi_row,
                        df_with_emas=df_with_emas,
                        support_zones=support_zones
                    )

                # Display chart with interactive config
                st.plotly_chart(fig, use_container_width=True, config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawrect', 'eraseshape'],
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    'scrollZoom': True,
                    'doubleClick': 'reset'
                })

                # Show analysis details
                st.markdown("---")
                st.subheader("📋 Analysis Details")

                col1, col2, col3 = st.columns(3)

                # Use already calculated data
                is_aligned, alignment_desc = ema_analyzer.check_ema_alignment(df_with_emas, for_buy=True)

                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                    st.metric("EMA 24", f"${df_with_emas['EMA_24'].iloc[-1]:.2f}")
                    st.metric("EMA 38", f"${df_with_emas['EMA_38'].iloc[-1]:.2f}")
                    st.metric("EMA 62", f"${df_with_emas['EMA_62'].iloc[-1]:.2f}")

                with col2:
                    st.metric("EMA Alignment", "✅ Aligned" if is_aligned else "❌ Not Aligned")
                    st.write(f"**Alignment:** {alignment_desc}")
                    st.metric("Crossovers Found", len(crossovers))
                    st.metric("Support Zones", len(support_zones))

                with col3:
                    if support_zones:
                        best_support = support_zones[0]
                        st.metric("Nearest Support", f"${best_support['level']:.2f}")
                        st.metric("Distance", f"{best_support['distance_pct']:.2f}%")
                        st.metric("Zone Strength", f"{best_support['strength']:.0f}%")
                    else:
                        st.info("No support zones detected")

                # Show crossovers table
                if crossovers:
                    st.markdown("### Recent EMA Crossovers")
                    crossover_data = []
                    for cross in crossovers[:10]:
                        crossover_data.append({
                            'Date': cross['date'].strftime('%Y-%m-%d') if isinstance(cross['date'], pd.Timestamp) else str(cross['date']),
                            'Type': cross['type'].upper(),
                            'EMAs': f"{cross['fast_ema']}/{cross['slow_ema']}",
                            'Price': f"${cross['price']:.2f}",
                            'Days Ago': cross['days_ago']
                        })
                    st.dataframe(pd.DataFrame(crossover_data), use_container_width=True)
            else:
                st.error(f"Could not load chart for {symbol}. Please check the symbol and try again.")

elif page == "📈 Signaux Historiques":
    st.title("📈 Signaux Historiques - Niveaux + RSI Trendlines")
    st.markdown("### Visualiser TOUS les anciens signaux avec les niveaux historiques et les obliques RSI")

    st.info("""
    **Ce graphique affiche:**
    - 🟢 **Lignes vertes pointillées**: Niveaux historiques valides (> 8%)
    - 🔴 **Lignes rouges pleines**: Niveaux historiques PROCHES (< 8%)
    - ⭐ **Étoiles sur prix**: Points de crossover EMA
    - 📉 **Ligne rouge oblique (RSI)**: Trendline RSI descendante
    - 🔺 **Triangles rouges (RSI)**: Pics RSI utilisés pour la trendline
    - ⭐ **Étoile verte (RSI)**: RSI Breakout (cassure de l'oblique)
    """)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        hist_symbol = st.text_input("Enter Stock Symbol", value="TSLA", key="hist_symbol").upper()

    with col2:
        hist_timeframe = st.selectbox("Timeframe", ["weekly", "daily"], key="hist_timeframe")

    with col3:
        hist_period = st.selectbox("Period", ["1y", "2y", "5y", "max"], index=2, key="hist_period")  # Default to 5y

    if st.button("📊 Afficher les Signaux Historiques", type="primary"):
        with st.spinner(f"Chargement des signaux historiques pour {hist_symbol}..."):
            fig = chart_visualizer.create_historical_chart(
                symbol=hist_symbol,
                timeframe=hist_timeframe,
                period=hist_period
            )

            if fig:
                st.plotly_chart(fig, use_container_width=True)

                # Show summary
                st.markdown("---")
                st.subheader("📋 Résumé")

                # Get the data
                interval = '1wk' if hist_timeframe == 'weekly' else '1d'
                df = market_data_fetcher.get_historical_data(hist_symbol, period=hist_period, interval=interval)

                if df is not None:
                    df = ema_analyzer.calculate_emas(df)
                    current_price = float(df['Close'].iloc[-1])
                    crossovers = ema_analyzer.detect_crossovers(df, hist_timeframe)
                    historical_levels = ema_analyzer.find_historical_support_levels(df, crossovers, current_price)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Prix Actuel", f"${current_price:.2f}")
                        st.metric("Niveaux Historiques", len(historical_levels))

                    with col2:
                        near_levels = [l for l in historical_levels if l['is_near']]
                        st.metric("Niveaux PROCHES (< 8%)", len(near_levels))
                        if near_levels:
                            st.success(f"✅ Prix s'approche de {len(near_levels)} niveau(x)!")
                        else:
                            st.info("📍 Aucun niveau proche actuellement")

                    with col3:
                        from trendline_analysis.core.rsi_breakout_analyzer import RSIBreakoutAnalyzer
                        rsi_analyzer = RSIBreakoutAnalyzer()
                        lookback = 104 if hist_timeframe == 'weekly' else 252
                        rsi_result = rsi_analyzer.analyze(df, lookback_periods=lookback)

                        if rsi_result and rsi_result.has_rsi_breakout:
                            st.metric("RSI Breakout", "✅ OUI")
                            st.success(f"Strength: {rsi_result.rsi_breakout.strength}")
                        elif rsi_result and rsi_result.has_rsi_trendline:
                            st.metric("RSI Trendline", "✅ OUI")
                            st.info("Pas encore de breakout")
                        else:
                            st.metric("RSI Signal", "❌ AUCUN")

                    # Show signal interpretation
                    st.markdown("---")
                    st.subheader("🎯 Interprétation du Signal")

                    if near_levels and rsi_result and rsi_result.has_rsi_breakout:
                        st.success("""
                        🚨 **SIGNAL POTENTIEL DÉTECTÉ!**

                        Le prix s'approche d'un niveau historique ET il y a un RSI breakout.
                        → C'est exactement le type de signal que le système recherche!
                        → Recommandation: STRONG_BUY
                        """)
                    elif near_levels:
                        st.warning("""
                        ⚠️ **Prix proche d'un niveau historique**

                        Mais pas de RSI breakout détecté pour le moment.
                        → Surveiller l'évolution du RSI
                        """)
                    elif rsi_result and rsi_result.has_rsi_breakout:
                        st.info("""
                        📊 **RSI Breakout détecté**

                        Mais le prix est LOIN des niveaux historiques (> 8%).
                        → Pas de signal car le prix doit s'approcher d'un niveau
                        """)
                    else:
                        st.info("""
                        📍 **Aucun signal actuellement**

                        Les signaux apparaîtront quand:
                        1. Le prix retracera vers un niveau historique (< 8%)
                        2. ET qu'il y aura un RSI breakout
                        """)
            else:
                st.error(f"Impossible de charger les données pour {hist_symbol}")

elif page == "🔍 Screening":
    st.title("🔍 Manual Screening")
    st.markdown("Run screening on specific symbols or full market scan")

    # Market selection
    st.markdown("### 🌍 Market Selection")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        market_nasdaq = st.checkbox("NASDAQ", value=MARKETS_EXTENDED.get('NASDAQ', True), key="mkt_nasdaq")
    with col2:
        market_sp500 = st.checkbox("S&P 500 / NYSE", value=MARKETS_EXTENDED.get('SP500', True), key="mkt_sp500")
    with col3:
        market_crypto = st.checkbox("Crypto", value=MARKETS_EXTENDED.get('CRYPTO', False), key="mkt_crypto")
    with col4:
        market_europe = st.checkbox("Europe", value=MARKETS_EXTENDED.get('EUROPE', False), key="mkt_europe")
    with col5:
        market_asia = st.checkbox("Asia ADR", value=MARKETS_EXTENDED.get('ASIA_ADR', False), key="mkt_asia")
    with col6:
        full_exchange_mode = st.checkbox("📊 Full NASDAQ", value=False, key="full_exchange",
                                         help="NASDAQ complet: ~3000+ stocks (vs NASDAQ-100: ~100)")

    if full_exchange_mode:
        st.info("🔥 Mode Full NASDAQ activé: accès à TOUS les symboles NASDAQ (~3000+) au lieu du NASDAQ-100 (~100). Le S&P 500 reste à ~500 valeurs.")

    # Portfolio summary
    st.markdown("---")
    st.markdown("### 💼 Portfolio Status")
    portfolio = market_screener.get_portfolio_summary()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Capital Total", f"{portfolio['total_capital']:,.0f}€")
    with col2:
        st.metric("Capital Investi", f"{portfolio['invested_capital']:,.0f}€")
    with col3:
        st.metric("Capital Disponible", f"{portfolio['available_capital']:,.0f}€")
    with col4:
        st.metric("Positions Ouvertes", portfolio['num_positions'])

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Single Symbol", "Multiple Symbols", "Market Scan"])

    with tab1:
        st.subheader("Screen Single Symbol")

        col1, col2 = st.columns([3, 1])

        with col1:
            single_symbol = st.text_input("Enter Symbol to Screen", value="", key="single_screen").upper()

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_single = st.button("🔍 Screen", key="run_single_screen", type="primary")

        if run_single and single_symbol:
            with st.spinner(f"Screening {single_symbol}..."):
                try:
                    from trendline_analysis.core.rsi_breakout_analyzer import RSIBreakoutAnalyzer
                    from plotly.subplots import make_subplots
                    import plotly.graph_objects as go

                    info = market_data_fetcher.get_stock_info(single_symbol)
                    if info:
                        company_name = info.get('longName', single_symbol)
                        alert = market_screener.screen_single_stock(single_symbol, company_name)

                        if alert:
                            st.success(f"✅ BUY SIGNAL DETECTED for {single_symbol}!")

                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Timeframe", alert['timeframe'].upper())

                            with col2:
                                st.metric("Price", f"${alert['current_price']:.2f}")

                            with col3:
                                st.metric("Support", f"${alert['support_level']:.2f}")

                            with col4:
                                st.metric("Recommendation", alert['recommendation'])

                            # Afficher infos RSI si disponibles
                            if alert.get('has_rsi_breakout'):
                                st.markdown("### 🎯 RSI Breakout Detected!")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("RSI Timeframe", alert.get('rsi_timeframe', 'N/A').upper())
                                with col2:
                                    st.metric("RSI Breakout", alert.get('rsi_breakout_date', 'N/A'))
                                with col3:
                                    st.metric("RSI Signal", alert.get('rsi_signal', 'N/A'))

                            st.json(alert)
                        else:
                            st.info(f"ℹ️ No signal detected for {single_symbol}, but showing analysis...")

                        # TOUJOURS afficher un graphique avec RSI
                        st.markdown("---")
                        st.markdown("### 📊 Technical Analysis")

                        # Get data for both timeframes
                        df_daily = market_data_fetcher.get_historical_data(single_symbol, period='1y', interval='1d')

                        if df_daily is not None and len(df_daily) > 0:
                            # Analyze RSI
                            rsi_analyzer = RSIBreakoutAnalyzer()
                            rsi_result = rsi_analyzer.analyze(df_daily, lookback_periods=252)

                            # Extract price data
                            close = df_daily['Close'].iloc[:, 0] if isinstance(df_daily['Close'], pd.DataFrame) else df_daily['Close']
                            open_price = df_daily['Open'].iloc[:, 0] if isinstance(df_daily['Open'], pd.DataFrame) else df_daily['Open']
                            high = df_daily['High'].iloc[:, 0] if isinstance(df_daily['High'], pd.DataFrame) else df_daily['High']
                            low = df_daily['Low'].iloc[:, 0] if isinstance(df_daily['Low'], pd.DataFrame) else df_daily['Low']

                            # Create subplot
                            fig = make_subplots(
                                rows=2, cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.05,
                                subplot_titles=(f'{single_symbol} - Price with EMAs', 'RSI with Trendline'),
                                row_heights=[0.65, 0.35]
                            )

                            # Price candlestick
                            fig.add_trace(
                                go.Candlestick(
                                    x=df_daily.index,
                                    open=open_price,
                                    high=high,
                                    low=low,
                                    close=close,
                                    name='Price'
                                ),
                                row=1, col=1
                            )

                            # Add EMAs
                            df_with_emas = ema_analyzer.calculate_emas(df_daily)
                            for ema in [24, 38, 62]:
                                fig.add_trace(
                                    go.Scatter(
                                        x=df_with_emas.index,
                                        y=df_with_emas[f'EMA_{ema}'],
                                        mode='lines',
                                        name=f'EMA {ema}',
                                        line=dict(width=1.5)
                                    ),
                                    row=1, col=1
                                )

                            # RSI
                            rsi = rsi_analyzer.rsi_detector.calculate_rsi(df_daily)
                            fig.add_trace(
                                go.Scatter(
                                    x=df_daily.index,
                                    y=rsi,
                                    mode='lines',
                                    name='RSI',
                                    line=dict(color='blue', width=2)
                                ),
                                row=2, col=1
                            )

                            # RSI trendline if exists
                            if rsi_result and rsi_result.has_rsi_trendline:
                                rsi_tl = rsi_result.rsi_trendline
                                last_rsi_peak_idx = rsi_tl.peak_indices[-1]
                                trendline_x = df_daily.index[rsi_tl.start_idx:last_rsi_peak_idx + 1]
                                trendline_y = [rsi_analyzer.rsi_detector.get_trendline_value(rsi_tl, i)
                                             for i in range(rsi_tl.start_idx, last_rsi_peak_idx + 1)]

                                fig.add_trace(
                                    go.Scatter(
                                        x=trendline_x,
                                        y=trendline_y,
                                        mode='lines',
                                        name='RSI Resistance',
                                        line=dict(color='orange', width=3, dash='dash')
                                    ),
                                    row=2, col=1
                                )

                                # RSI breakout marker
                                if rsi_result.has_rsi_breakout:
                                    rsi_bo = rsi_result.rsi_breakout
                                    fig.add_trace(
                                        go.Scatter(
                                            x=[df_daily.index[rsi_bo.index]],
                                            y=[rsi_bo.rsi_value],
                                            mode='markers',
                                            name='RSI Breakout',
                                            marker=dict(color='green', size=15, symbol='star'),
                                            showlegend=True
                                        ),
                                        row=2, col=1
                                    )

                            # RSI levels
                            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.3, row=2, col=1)
                            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.3, row=2, col=1)

                            # Layout
                            fig.update_layout(
                                height=900,
                                showlegend=True,
                                hovermode='x unified',
                                xaxis_rangeslider_visible=False
                            )

                            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                            fig.update_yaxes(title_text="RSI", row=2, col=1)

                            st.plotly_chart(fig, use_container_width=True)

                            # Show analysis summary
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("#### RSI Analysis")
                                if rsi_result:
                                    st.write(f"RSI Trendline: {'✅ YES' if rsi_result.has_rsi_trendline else '❌ NO'}")
                                    if rsi_result.has_rsi_trendline:
                                        st.write(f"Peaks: {len(rsi_result.rsi_trendline.peak_indices)}")
                                        st.write(f"R²: {rsi_result.rsi_trendline.r_squared:.3f}")
                                    st.write(f"RSI Breakout: {'✅ YES' if rsi_result.has_rsi_breakout else '❌ NO'}")
                                    if rsi_result.has_rsi_breakout:
                                        st.write(f"Strength: {rsi_result.rsi_breakout.strength}")
                                        st.write(f"Signal: {rsi_result.signal}")
                                else:
                                    st.write("❌ No RSI trendline detected")

                    else:
                        st.error(f"Could not fetch data for {single_symbol}")
                except Exception as e:
                    st.error(f"Error screening {single_symbol}: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    with tab2:
        st.subheader("Screen Multiple Symbols")

        symbols_input = st.text_area(
            "Enter symbols (one per line or comma-separated)",
            value="AAPL\nMSFT\nGOOGL\nTSLA",
            height=150,
            key="multi_symbols"
        )

        run_multi = st.button("🔍 Screen All", key="run_multi_screen", type="primary")

        if run_multi and symbols_input:
            # Parse symbols
            symbols = []
            for line in symbols_input.split('\n'):
                symbols.extend([s.strip().upper() for s in line.split(',') if s.strip()])

            symbols = list(set(symbols))  # Remove duplicates

            st.info(f"Screening {len(symbols)} symbols...")

            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()

            alerts = []
            for i, symbol in enumerate(symbols):
                status_text.text(f"Screening {symbol}... ({i+1}/{len(symbols)})")

                try:
                    info = market_data_fetcher.get_stock_info(symbol)
                    if info:
                        company_name = info.get('longName', symbol)
                        alert = market_screener.screen_single_stock(symbol, company_name)
                        if alert:
                            alerts.append(alert)
                except Exception as e:
                    st.warning(f"Error screening {symbol}: {e}")

                progress_bar.progress((i + 1) / len(symbols))

            status_text.text("Screening complete!")

            with results_container:
                st.markdown("---")
                st.subheader(f"Results: {len(alerts)} Alerts Generated")

                if alerts:
                    for alert in alerts:
                        with st.expander(f"🔥 {alert['symbol']} - {alert['recommendation']}"):
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Price", f"${alert['current_price']:.2f}")
                                st.metric("Support", f"${alert['support_level']:.2f}")

                            with col2:
                                st.metric("Distance", f"{alert['distance_to_support_pct']:.2f}%")
                                st.metric("Timeframe", alert['timeframe'].upper())

                            with col3:
                                st.metric("EMA Alignment", alert['ema_alignment'])

                            # Mini chart
                            fig = chart_visualizer.create_chart(
                                symbol=alert['symbol'],
                                timeframe=alert['timeframe'],
                                period='6mo',
                                show_volume=False
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, key=f"chart_{alert['symbol']}")
                else:
                    st.info("No alerts generated from the screening.")

    with tab3:
        st.subheader("🌍 Full Market Scan")
        st.markdown("Screen all symbols from selected markets")

        # Show selected markets
        selected_markets = []
        if market_nasdaq:
            selected_markets.append("NASDAQ")
        if market_sp500:
            selected_markets.append("SP500")
        if market_crypto:
            selected_markets.append("CRYPTO")
        if market_europe:
            selected_markets.append("EUROPE")
        if market_asia:
            selected_markets.append("ASIA_ADR")

        if selected_markets:
            mode_label = "Full NASDAQ" if full_exchange_mode else "Index Only"
            st.info(f"Selected markets: {', '.join(selected_markets)} ({mode_label})")

            # Get ticker counts based on mode
            ticker_counts = {}
            total_tickers = 0

            with st.spinner("Loading ticker counts..."):
                if market_nasdaq:
                    nasdaq_tickers = market_data_fetcher.get_nasdaq_tickers(full_exchange=full_exchange_mode)
                    ticker_counts['NASDAQ'] = len(nasdaq_tickers)
                    total_tickers += len(nasdaq_tickers)

                if market_sp500:
                    # S&P 500 toujours limité à ~500 (pas de full NYSE)
                    sp500_tickers = market_data_fetcher.get_sp500_tickers(include_nyse_full=False)
                    ticker_counts['S&P 500'] = len(sp500_tickers)
                    total_tickers += len(sp500_tickers)

                if market_crypto:
                    crypto_tickers = market_data_fetcher.get_crypto_tickers()
                    ticker_counts['CRYPTO'] = len(crypto_tickers)
                    total_tickers += len(crypto_tickers)

                if market_europe:
                    europe_tickers = market_data_fetcher.get_european_tickers()
                    ticker_counts['EUROPE'] = len(europe_tickers)
                    total_tickers += len(europe_tickers)

                if market_asia:
                    asia_tickers = market_data_fetcher.get_asian_adr_tickers()
                    ticker_counts['ASIA_ADR'] = len(asia_tickers)
                    total_tickers += len(asia_tickers)

            # Display ticker counts
            cols = st.columns(len(ticker_counts))
            for i, (market, count) in enumerate(ticker_counts.items()):
                with cols[i]:
                    st.metric(market, f"{count:,}")

            st.markdown(f"**Total: {total_tickers:,} symbols to scan**")

            # Filter options
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                only_fresh_signals = st.checkbox(
                    "🔥 Signaux frais uniquement (dernière clôture)",
                    value=False,
                    help="Ne montrer que les breakouts RSI de la dernière bougie (age=0)"
                )
            with col_filter2:
                max_signal_age = st.selectbox(
                    "Âge max du breakout RSI",
                    options=[0, 1, 2, 3, 5, 10],
                    index=3,  # Default: 3 périodes
                    format_func=lambda x: f"{x} période{'s' if x > 1 else ''}" if x > 0 else "Dernière clôture uniquement",
                    help="Filtrer les signaux dont le breakout RSI est plus ancien"
                )

            if full_exchange_mode and total_tickers > 1000:
                estimated_time = total_tickers * 1.5 / 60 / 10  # ~1.5s par symbole avec 10 workers
                st.info(f"⚡ Scan parallélisé de {total_tickers:,} symboles - temps estimé: ~{estimated_time:.0f} minutes avec 10 workers parallèles")

            # ============================================================
            # BACKGROUND SCANNER - Permet de naviguer pendant le scan
            # ============================================================

            # Get current scanner state
            scan_state = background_scanner.get_state()
            scan_progress = background_scanner.get_progress_info()

            # Parallel scanning options
            col_scan1, col_scan2, col_scan3, col_scan4 = st.columns([2, 1, 1, 1])

            with col_scan1:
                # Show different button based on scan state
                if scan_state.status == ScanStatus.RUNNING:
                    if st.button("⏸️ PAUSE", type="secondary", key="pause_scan"):
                        background_scanner.pause_scan()
                        st.rerun()
                elif scan_state.status == ScanStatus.PAUSED:
                    remaining = len(scan_state.pending_stocks)
                    if st.button(f"▶️ Reprendre ({remaining} restants)", type="primary", key="resume_scan"):
                        background_scanner.resume_scan(
                            screen_function=market_screener.screen_single_stock,
                            num_workers=st.session_state.get('num_workers', 10)
                        )
                        st.rerun()
                else:
                    if st.button("🚀 Run Full Market Scan", type="primary", key="run_market_scan"):
                        # Build stock list from selected markets
                        stocks_to_scan = []

                        if market_nasdaq:
                            for ticker in market_data_fetcher.get_nasdaq_tickers(full_exchange=full_exchange_mode):
                                stocks_to_scan.append({'symbol': ticker, 'name': ticker, 'market': 'NASDAQ'})

                        if market_sp500:
                            for ticker in market_data_fetcher.get_sp500_tickers(include_nyse_full=False):
                                if not any(s['symbol'] == ticker for s in stocks_to_scan):
                                    stocks_to_scan.append({'symbol': ticker, 'name': ticker, 'market': 'SP500'})

                        if market_crypto:
                            for ticker in market_data_fetcher.get_crypto_tickers():
                                stocks_to_scan.append({'symbol': ticker, 'name': ticker, 'market': 'CRYPTO'})

                        if market_europe:
                            for ticker in market_data_fetcher.get_european_tickers():
                                stocks_to_scan.append({'symbol': ticker, 'name': ticker, 'market': 'EUROPE'})

                        if market_asia:
                            for ticker in market_data_fetcher.get_asian_adr_tickers():
                                stocks_to_scan.append({'symbol': ticker, 'name': ticker, 'market': 'ASIA_ADR'})

                        if stocks_to_scan:
                            num_workers = st.session_state.get('num_workers', 10)
                            background_scanner.start_scan(
                                stocks=stocks_to_scan,
                                screen_function=market_screener.screen_single_stock,
                                num_workers=num_workers
                            )
                            st.rerun()

            with col_scan2:
                num_workers = st.selectbox("Workers parallèles", [5, 10, 15, 20], index=1, key="num_workers",
                                          help="Plus de workers = plus rapide, mais plus de requêtes simultanées")

            with col_scan3:
                if scan_state.status == ScanStatus.PAUSED:
                    if st.button("🔄 Nouveau scan", type="secondary", key="new_scan"):
                        background_scanner.reset()
                        st.rerun()
                elif scan_state.status == ScanStatus.RUNNING:
                    if st.button("🛑 ANNULER", type="secondary", key="cancel_scan"):
                        background_scanner.cancel_scan()
                        st.rerun()

            with col_scan4:
                # Auto-refresh toggle for running scans
                if scan_state.status == ScanStatus.RUNNING:
                    auto_refresh = st.checkbox("🔄 Auto-refresh", value=True, key="auto_refresh",
                                              help="Actualise automatiquement toutes les 2 secondes")

            # ============================================================
            # SCAN STATUS DISPLAY
            # ============================================================

            if scan_state.status == ScanStatus.RUNNING:
                st.info(f"🔄 **Scan en cours** - Vous pouvez naviguer librement sur d'autres pages!")

                # Progress bar
                progress_pct = scan_progress['progress_pct'] / 100
                st.progress(progress_pct)

                # Status metrics
                col_status1, col_status2, col_status3, col_status4 = st.columns(4)
                with col_status1:
                    st.metric("Progression", f"{scan_progress['completed']}/{scan_progress['total']}")
                with col_status2:
                    st.metric("Signaux trouvés", f"{scan_progress['alerts_count']}")
                with col_status3:
                    st.metric("Vitesse", f"{scan_progress['rate_per_second']:.1f}/sec")
                with col_status4:
                    eta_min = scan_progress['eta_seconds'] / 60
                    st.metric("ETA", f"{eta_min:.1f} min")

                # Auto-refresh
                if st.session_state.get('auto_refresh', True):
                    time.sleep(2)
                    st.rerun()

            elif scan_state.status == ScanStatus.PAUSED:
                completed = scan_progress['completed']
                remaining = len(scan_state.pending_stocks)
                total = completed + remaining
                st.warning(f"⏸️ **Scan en pause**: {completed}/{total} traités - {len(scan_state.alerts)} signaux trouvés")
                st.caption("Vous pouvez naviguer librement et reprendre le scan plus tard.")

            elif scan_state.status == ScanStatus.COMPLETED:
                elapsed_min = scan_progress['elapsed_seconds'] / 60
                st.success(f"✅ **Scan terminé**: {len(scan_state.alerts)} signaux trouvés sur {scan_state.total_stocks} symboles en {elapsed_min:.1f} minutes")

            elif scan_state.status == ScanStatus.CANCELLED:
                st.warning(f"⚠️ **Scan annulé**: {scan_progress['completed']} symboles traités - {len(scan_state.alerts)} signaux trouvés")

            elif scan_state.status == ScanStatus.ERROR:
                st.error(f"❌ **Erreur**: {scan_state.error_message}")

            # ============================================================
            # DISPLAY ALERTS (from background scanner)
            # ============================================================

            alerts = scan_state.alerts
            if alerts:
                st.markdown("### 📡 Signaux détectés")

                # Apply freshness filter
                if only_fresh_signals:
                    filtered_alerts = [a for a in alerts if a.get('rsi_breakout_age', 999) == 0]
                else:
                    filtered_alerts = [a for a in alerts
                                     if a.get('rsi_breakout_age', 0) <= max_signal_age
                                     or a.get('rsi_breakout_age') is None]

                if not filtered_alerts:
                    st.warning(f"🔍 {len(alerts)} signaux trouvés mais aucun ne correspond au filtre de fraîcheur")
                else:
                    # Sort by freshness then confidence
                    sorted_alerts = sorted(filtered_alerts,
                                         key=lambda x: (x.get('rsi_breakout_age', 999), -x.get('confidence_score', 0)))

                    # Build display data
                    signal_data = []
                    for alert in sorted_alerts:
                        support_type = alert.get('support_type', 'weekly')
                        timeframe = "📅 Daily" if support_type == 'daily_fallback' else "📆 Weekly"

                        rsi_age = alert.get('rsi_breakout_age', None)
                        if rsi_age is not None:
                            if rsi_age == 0:
                                freshness = "🔥 Nouveau!"
                            elif rsi_age == 1:
                                freshness = "✨ 1 période"
                            elif rsi_age <= 3:
                                freshness = f"⏰ {rsi_age:.0f} périodes"
                            else:
                                freshness = f"📅 {rsi_age:.0f} périodes"
                        else:
                            freshness = "🔍 Trendline"

                        signal_data.append({
                            "Symbole": alert['symbol'],
                            "Marché": alert.get('market', 'N/A'),
                            "TF": timeframe,
                            "Fraîcheur": freshness,
                            "Score": f"{alert.get('confidence_score', 0):.0f}/100",
                            "Signal": alert.get('confidence_signal', 'N/A'),
                            "Prix": f"${alert['current_price']:.2f}",
                            "Distance": f"{alert['distance_to_support_pct']:.1f}%"
                        })

                    # Display as dataframe
                    df_signals = pd.DataFrame(signal_data)
                    st.dataframe(df_signals, use_container_width=True, hide_index=True)

                    # Detailed view for top signals
                    st.markdown("### 📊 Détails des meilleurs signaux")
                    alerts_sorted = sorted(filtered_alerts, key=lambda x: x.get('confidence_score', 0), reverse=True)

                    for alert in alerts_sorted[:20]:
                        confidence = alert.get('confidence_signal', alert.get('recommendation', 'N/A'))
                        with st.expander(f"🎯 {alert['symbol']} ({alert.get('market', 'N/A')}) - {confidence}"):
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Price", f"${alert['current_price']:.2f}")
                                st.metric("Support", f"${alert['support_level']:.2f}")

                            with col2:
                                st.metric("Score", f"{alert.get('confidence_score', 'N/A')}/100")
                                st.metric("Distance", f"{alert['distance_to_support_pct']:.2f}%")

                            with col3:
                                st.metric("Position Size", f"{alert.get('position_shares', 'N/A')} shares")
                                st.metric("Position Value", f"{alert.get('position_value', 0):.0f}€")

                            with col4:
                                st.metric("Stop Loss", f"${alert.get('stop_loss', 'N/A'):.2f}" if alert.get('stop_loss') else "N/A")
                                st.metric("Risk", f"{alert.get('risk_amount', 0):.0f}€")

                            st.caption(f"Stop source: {alert.get('stop_source', 'N/A')}")
        else:
            st.warning("Please select at least one market to scan")

elif page == "⏰ Scheduler":
    st.title("⏰ Scheduled Scans")
    st.markdown("Configure automatic market scanning on a schedule")

    # Initialize scheduler with screen function and stocks provider
    def get_stocks_for_scheduler():
        """Get stocks based on scheduler config"""
        stocks = []
        config = scan_scheduler.config
        for market in config.markets:
            if market == "NASDAQ":
                stocks.extend(market_data_fetcher.get_nasdaq_symbols()[:200])
            elif market == "SP500":
                stocks.extend(market_data_fetcher.get_sp500_symbols())
            elif market == "EUROPE":
                stocks.extend(market_data_fetcher.get_europe_symbols())
            elif market == "CRYPTO":
                stocks.extend(market_data_fetcher.get_crypto_symbols())
        return stocks

    scan_scheduler.set_screen_function(market_screener.screen_single_stock_weekly)
    scan_scheduler.set_stocks_provider(get_stocks_for_scheduler)
    scan_scheduler.set_notification_manager(notification_manager)

    # Create tabs
    sched_tab1, sched_tab2, sched_tab3 = st.tabs(["📊 Status", "⚙️ Configuration", "📜 History"])

    with sched_tab1:
        st.markdown("### Scheduler Status")

        status = scan_scheduler.get_status()

        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if status['running']:
                st.success("🟢 Running")
            else:
                st.warning("🟡 Stopped")
        with col2:
            if status['enabled']:
                st.success("✅ Enabled")
            else:
                st.info("⏸️ Disabled")
        with col3:
            st.metric("Total Runs", status['total_runs'])
        with col4:
            st.metric("Last Alerts", status['last_alerts'])

        # Next/Last run info
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ⏭️ Next Scheduled Run")
            if status['next_run']:
                next_run = datetime.fromisoformat(status['next_run'])
                time_until = next_run - datetime.now()
                st.info(f"**{next_run.strftime('%Y-%m-%d %H:%M')}**")
                if time_until.total_seconds() > 0:
                    hours, remainder = divmod(int(time_until.total_seconds()), 3600)
                    minutes = remainder // 60
                    st.caption(f"In {hours}h {minutes}m")
            else:
                st.info("Not scheduled")

        with col2:
            st.markdown("#### ⏮️ Last Run")
            if status['last_run']:
                last_run = datetime.fromisoformat(status['last_run'])
                st.info(f"**{last_run.strftime('%Y-%m-%d %H:%M')}**")
                st.caption(f"Found {status['last_alerts']} alerts")
            else:
                st.info("No previous runs")

        # Control buttons
        st.markdown("---")
        st.markdown("### Controls")
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

        with ctrl_col1:
            if st.button("▶️ Start Scheduler", key="start_scheduler", disabled=status['running']):
                if scan_scheduler.start():
                    st.success("Scheduler started!")
                    st.rerun()
                else:
                    st.error("Failed to start scheduler")

        with ctrl_col2:
            if st.button("⏹️ Stop Scheduler", key="stop_scheduler", disabled=not status['running']):
                scan_scheduler.stop()
                st.success("Scheduler stopped!")
                st.rerun()

        with ctrl_col3:
            if st.button("🔄 Run Now", key="run_now"):
                if scan_scheduler.trigger_now():
                    st.success("Scan triggered!")
                else:
                    st.warning("Could not trigger scan (already running?)")

        with ctrl_col4:
            if st.button("🔃 Refresh Status", key="refresh_scheduler"):
                st.rerun()

        # Current scan progress if running
        scan_state = background_scanner.get_state()
        if scan_state.status == ScanStatus.RUNNING:
            st.markdown("---")
            st.markdown("### 🔄 Scan in Progress")
            progress = scan_state.completed_count / scan_state.total_stocks if scan_state.total_stocks > 0 else 0
            st.progress(progress)
            st.caption(f"Scanning: {scan_state.current_symbol} ({scan_state.completed_count}/{scan_state.total_stocks})")
            st.caption(f"Alerts found: {len(scan_state.alerts)}")

    with sched_tab2:
        st.markdown("### Schedule Configuration")

        current_config = scan_scheduler.config

        # Enable/Disable
        enabled = st.checkbox(
            "Enable Scheduled Scanning",
            value=current_config.enabled,
            key="scheduler_enabled"
        )

        st.markdown("---")
        st.markdown("#### Schedule Type")

        schedule_type = st.selectbox(
            "Run scans:",
            options=["daily", "weekly", "hourly", "interval"],
            index=["daily", "weekly", "hourly", "interval"].index(current_config.schedule_type),
            format_func=lambda x: {
                "daily": "📅 Daily (at specific time)",
                "weekly": "📆 Weekly (on specific days)",
                "hourly": "⏰ Hourly",
                "interval": "🔄 Every N minutes"
            }.get(x, x),
            key="schedule_type"
        )

        # Time configuration based on type
        if schedule_type in ["daily", "weekly"]:
            time_col1, time_col2 = st.columns(2)
            with time_col1:
                time_of_day = st.time_input(
                    "Time of day:",
                    value=datetime.strptime(current_config.time_of_day, "%H:%M").time(),
                    key="time_of_day"
                )
            with time_col2:
                if schedule_type == "weekly":
                    days_options = {
                        0: "Monday", 1: "Tuesday", 2: "Wednesday",
                        3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
                    }
                    days_of_week = st.multiselect(
                        "Days of week:",
                        options=list(days_options.keys()),
                        default=current_config.days_of_week,
                        format_func=lambda x: days_options[x],
                        key="days_of_week"
                    )
                else:
                    # Daily - weekdays only option
                    weekdays_only = st.checkbox(
                        "Weekdays only (Mon-Fri)",
                        value=current_config.days_of_week == [0, 1, 2, 3, 4],
                        key="weekdays_only"
                    )
                    days_of_week = [0, 1, 2, 3, 4] if weekdays_only else list(range(7))
        elif schedule_type == "interval":
            interval_minutes = st.number_input(
                "Interval (minutes):",
                min_value=15,
                max_value=480,
                value=current_config.interval_minutes,
                step=15,
                key="interval_minutes"
            )
            time_of_day = current_config.time_of_day
            days_of_week = list(range(7))
        else:  # hourly
            time_of_day = current_config.time_of_day
            days_of_week = list(range(7))
            interval_minutes = 60

        st.markdown("---")
        st.markdown("#### Markets to Scan")

        markets_options = ["NASDAQ", "SP500", "EUROPE", "CRYPTO", "CAC40", "DAX"]
        markets = st.multiselect(
            "Select markets:",
            options=markets_options,
            default=current_config.markets,
            key="scheduler_markets"
        )

        st.markdown("---")
        st.markdown("#### Notifications")

        notif_col1, notif_col2 = st.columns(2)
        with notif_col1:
            notify_on_completion = st.checkbox(
                "Notify when scan completes",
                value=current_config.notify_on_completion,
                key="notify_completion"
            )
        with notif_col2:
            notify_on_signals = st.checkbox(
                "Notify for each signal found",
                value=current_config.notify_on_signals,
                key="notify_signals"
            )

        # Save button
        if st.button("💾 Save Configuration", key="save_scheduler_config"):
            time_str = time_of_day.strftime("%H:%M") if hasattr(time_of_day, 'strftime') else time_of_day

            scan_scheduler.update_config(
                enabled=enabled,
                schedule_type=schedule_type,
                time_of_day=time_str,
                days_of_week=days_of_week,
                interval_minutes=interval_minutes if schedule_type == "interval" else 60,
                markets=markets,
                notify_on_completion=notify_on_completion,
                notify_on_signals=notify_on_signals
            )
            st.success("✅ Configuration saved!")

            # Show next run time
            if enabled:
                next_run = scan_scheduler._calculate_next_run()
                st.info(f"Next scheduled run: {next_run.strftime('%Y-%m-%d %H:%M')}")

    with sched_tab3:
        st.markdown("### Scan History")

        # Get alerts from recent scans
        scan_state = background_scanner.get_state()

        if scan_state.alerts:
            st.markdown(f"**Last scan found {len(scan_state.alerts)} alerts:**")

            for alert in scan_state.alerts[:20]:  # Show last 20
                with st.expander(f"🎯 {alert.get('symbol', 'N/A')} - {alert.get('recommendation', 'N/A')}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Price", f"${alert.get('current_price', 0):.2f}")
                    with col2:
                        st.metric("Support", f"${alert.get('support_level', 0):.2f}")
                    with col3:
                        st.metric("Confidence", f"{alert.get('confidence_score', 0):.0f}/100")
        else:
            st.info("No alerts from recent scans. Run a scan to see results here.")

        # Show scheduler statistics
        st.markdown("---")
        st.markdown("### Statistics")

        stats = scan_scheduler.get_status()
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Total Scans Run", stats['total_runs'])
        with stat_col2:
            st.metric("Last Scan Alerts", stats['last_alerts'])
        with stat_col3:
            if stats['last_run']:
                last_run = datetime.fromisoformat(stats['last_run'])
                st.metric("Last Run", last_run.strftime('%m/%d %H:%M'))
            else:
                st.metric("Last Run", "Never")

elif page == "🔬 Backtesting":
    st.title("🔬 Backtesting")
    st.markdown("Simulate trading strategy on historical data to evaluate performance")

    # Import backtesting modules
    from src.backtesting import Backtester, BacktestResult
    from src.backtesting.backtester import BacktestConfig
    from src.backtesting.metrics import format_metrics_report

    # Initialize session state for backtest results
    if 'backtest_result' not in st.session_state:
        st.session_state['backtest_result'] = None
    if 'backtest_running' not in st.session_state:
        st.session_state['backtest_running'] = False

    # Create tabs
    bt_tab1, bt_tab2, bt_tab3, bt_tab4 = st.tabs([
        "⚙️ Configuration", "▶️ Run Backtest", "📊 Results", "📈 Mass Simulation"
    ])

    with bt_tab1:
        st.markdown("### Backtest Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Capital & Position Sizing")
            bt_initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=1000000,
                value=st.session_state.get('bt_initial_capital', 10000),
                step=1000,
                key="bt_initial_capital_input"
            )
            st.session_state['bt_initial_capital'] = bt_initial_capital

            bt_position_size = st.slider(
                "Max Position Size (%)",
                min_value=5,
                max_value=50,
                value=st.session_state.get('bt_position_size', 20),
                step=5,
                key="bt_position_size_slider"
            )
            st.session_state['bt_position_size'] = bt_position_size

            bt_max_positions = st.number_input(
                "Max Concurrent Positions",
                min_value=1,
                max_value=20,
                value=st.session_state.get('bt_max_positions', 5),
                key="bt_max_positions_input"
            )
            st.session_state['bt_max_positions'] = bt_max_positions

        with col2:
            st.markdown("#### Signal Filters")
            bt_min_confidence = st.slider(
                "Min Confidence Score",
                min_value=40,
                max_value=80,
                value=st.session_state.get('bt_min_confidence', 55),
                step=5,
                key="bt_min_confidence_slider"
            )
            st.session_state['bt_min_confidence'] = bt_min_confidence

            bt_require_volume = st.checkbox(
                "Require Volume Confirmation",
                value=st.session_state.get('bt_require_volume', True),
                key="bt_require_volume_check"
            )
            st.session_state['bt_require_volume'] = bt_require_volume

            bt_min_volume_ratio = st.slider(
                "Min Volume Ratio",
                min_value=0.5,
                max_value=2.0,
                value=st.session_state.get('bt_min_volume_ratio', 1.0),
                step=0.1,
                key="bt_min_volume_slider",
                disabled=not bt_require_volume
            )
            st.session_state['bt_min_volume_ratio'] = bt_min_volume_ratio

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### Exit Conditions")
            bt_take_profit = st.slider(
                "Take Profit (%)",
                min_value=5,
                max_value=50,
                value=st.session_state.get('bt_take_profit', 15),
                step=5,
                key="bt_take_profit_slider"
            )
            st.session_state['bt_take_profit'] = bt_take_profit

            bt_use_trailing = st.checkbox(
                "Use Trailing Stop",
                value=st.session_state.get('bt_use_trailing', False),
                key="bt_use_trailing_check"
            )
            st.session_state['bt_use_trailing'] = bt_use_trailing

            bt_trailing_pct = st.slider(
                "Trailing Stop (%)",
                min_value=3,
                max_value=15,
                value=st.session_state.get('bt_trailing_pct', 5),
                step=1,
                key="bt_trailing_slider",
                disabled=not bt_use_trailing
            )
            st.session_state['bt_trailing_pct'] = bt_trailing_pct

            bt_max_hold = st.number_input(
                "Max Hold Days",
                min_value=10,
                max_value=180,
                value=st.session_state.get('bt_max_hold', 60),
                key="bt_max_hold_input"
            )
            st.session_state['bt_max_hold'] = bt_max_hold

        with col4:
            st.markdown("#### Strategy Settings")
            bt_use_enhanced = st.checkbox(
                "Use Enhanced Detector",
                value=st.session_state.get('bt_use_enhanced', True),
                key="bt_use_enhanced_check"
            )
            st.session_state['bt_use_enhanced'] = bt_use_enhanced

            bt_precision = st.selectbox(
                "Detection Precision",
                options=["low", "medium", "high"],
                index=["low", "medium", "high"].index(st.session_state.get('bt_precision', 'medium')),
                key="bt_precision_select"
            )
            st.session_state['bt_precision'] = bt_precision

            bt_timeframe = st.selectbox(
                "Timeframe",
                options=["weekly", "daily"],
                index=["weekly", "daily"].index(st.session_state.get('bt_timeframe', 'weekly')),
                key="bt_timeframe_select"
            )
            st.session_state['bt_timeframe'] = bt_timeframe

            # Dual-timeframe option (like real screener)
            bt_use_daily_fallback = st.checkbox(
                "Use Daily Fallback (if Weekly EMAs bullish)",
                value=st.session_state.get('bt_use_daily_fallback', True),
                key="bt_use_daily_fallback_check",
                help="Like the real screener: check weekly first, then daily if EMAs are bullish"
            )
            st.session_state['bt_use_daily_fallback'] = bt_use_daily_fallback

    with bt_tab2:
        st.markdown("### Run Backtest")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Date Range")
            bt_start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                key="bt_start_date"
            )
            bt_end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                key="bt_end_date"
            )

        with col2:
            st.markdown("#### Symbols")
            bt_symbol_mode = st.radio(
                "Symbol Selection",
                ["Manual", "Market", "Watchlist"],
                key="bt_symbol_mode"
            )

        if bt_symbol_mode == "Manual":
            bt_symbols_text = st.text_area(
                "Enter symbols (comma-separated)",
                value="AAPL, MSFT, GOOGL, AMZN, NVDA",
                height=100,
                key="bt_symbols_text"
            )
            bt_symbols = [s.strip().upper() for s in bt_symbols_text.split(",") if s.strip()]
        elif bt_symbol_mode == "Market":
            bt_market = st.selectbox(
                "Select Market",
                ["NASDAQ Top 50", "S&P 500", "Europe", "CAC40", "DAX", "Crypto"],
                key="bt_market_select"
            )
            if bt_market == "NASDAQ Top 50":
                bt_symbols = market_data_fetcher.get_nasdaq_symbols()[:50]
            elif bt_market == "S&P 500":
                bt_symbols = market_data_fetcher.get_sp500_symbols()
            elif bt_market == "Europe":
                bt_symbols = market_data_fetcher.get_europe_symbols()
            elif bt_market == "CAC40":
                bt_symbols = market_data_fetcher.get_cac40_symbols()
            elif bt_market == "DAX":
                bt_symbols = market_data_fetcher.get_dax_symbols()
            elif bt_market == "Crypto":
                bt_symbols = market_data_fetcher.get_crypto_symbols()
            st.info(f"Selected {len(bt_symbols)} symbols from {bt_market}")
        else:  # Watchlist
            watchlist_names = watchlist_manager.get_watchlist_names()
            if watchlist_names:
                bt_watchlist = st.selectbox(
                    "Select Watchlist",
                    watchlist_names,
                    key="bt_watchlist_select"
                )
                bt_symbols = watchlist_manager.get_symbols(bt_watchlist)
                st.info(f"Selected {len(bt_symbols)} symbols from {bt_watchlist}")
            else:
                st.warning("No watchlists found. Create one first.")
                bt_symbols = []

        st.markdown("---")

        if st.button("🚀 Run Backtest", key="run_backtest", disabled=st.session_state.get('backtest_running', False)):
            if not bt_symbols:
                st.error("No symbols selected!")
            else:
                st.session_state['backtest_running'] = True

                # Build config from session state
                config = BacktestConfig(
                    initial_capital=st.session_state.get('bt_initial_capital', 10000),
                    max_positions=st.session_state.get('bt_max_positions', 5),
                    position_size_pct=st.session_state.get('bt_position_size', 20) / 100,
                    min_confidence_score=st.session_state.get('bt_min_confidence', 55),
                    require_volume_confirmation=st.session_state.get('bt_require_volume', True),
                    min_volume_ratio=st.session_state.get('bt_min_volume_ratio', 1.0),
                    use_trailing_stop=st.session_state.get('bt_use_trailing', False),
                    trailing_stop_pct=st.session_state.get('bt_trailing_pct', 5) / 100,
                    take_profit_pct=st.session_state.get('bt_take_profit', 15) / 100,
                    max_hold_days=st.session_state.get('bt_max_hold', 60),
                    use_enhanced_detector=st.session_state.get('bt_use_enhanced', True),
                    precision_mode=st.session_state.get('bt_precision', 'medium'),
                    timeframe=st.session_state.get('bt_timeframe', 'weekly')
                )

                backtester = Backtester(config)

                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.text(f"Starting backtest on {len(bt_symbols)} symbols...")

                    result = backtester.run(
                        symbols=bt_symbols,
                        start_date=bt_start_date.strftime('%Y-%m-%d'),
                        end_date=bt_end_date.strftime('%Y-%m-%d')
                    )

                    st.session_state['backtest_result'] = result
                    progress_bar.progress(100)
                    status_text.text("Backtest complete!")

                    st.success(f"✅ Backtest complete! {result.metrics.total_trades} trades executed.")
                    st.info("Go to the 'Results' tab to see detailed metrics and equity curve.")

                except Exception as e:
                    st.error(f"Backtest failed: {e}")
                finally:
                    st.session_state['backtest_running'] = False

    with bt_tab3:
        st.markdown("### Backtest Results")

        result = st.session_state.get('backtest_result')

        if result is None:
            st.info("No backtest results yet. Run a backtest first!")
        else:
            # Summary metrics
            metrics = result.metrics
            st.markdown("#### Performance Summary")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", f"{metrics.total_return:+.2f}%",
                         delta_color="normal" if metrics.total_return >= 0 else "inverse")
            with col2:
                st.metric("Win Rate", f"{metrics.win_rate:.1f}%")
            with col3:
                st.metric("Profit Factor", f"{metrics.profit_factor:.2f}")
            with col4:
                st.metric("Max Drawdown", f"{metrics.max_drawdown:.2f}%")

            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.metric("Total Trades", metrics.total_trades)
            with col6:
                st.metric("Avg Win", f"{metrics.avg_win:+.2f}%")
            with col7:
                st.metric("Avg Loss", f"{metrics.avg_loss:.2f}%")
            with col8:
                st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")

            st.markdown("---")

            # Equity curve
            st.markdown("#### Equity Curve")
            if result.equity_curve is not None and len(result.equity_curve) > 0:
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=result.equity_curve.index,
                    y=result.equity_curve.values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#00C853', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(0, 200, 83, 0.1)'
                ))

                # Add initial capital line
                fig.add_hline(
                    y=metrics.initial_capital,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Initial Capital"
                )

                fig.update_layout(
                    title="Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Value ($)",
                    template="plotly_dark",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No equity curve data available")

            st.markdown("---")

            # Trade list
            st.markdown("#### Trade History")
            if result.trades:
                trade_data = []
                for trade in result.trades:
                    trade_data.append({
                        'Symbol': trade.symbol,
                        'Entry Date': trade.entry_date.strftime('%Y-%m-%d'),
                        'Entry Price': f"${trade.entry_price:.2f}",
                        'Exit Date': trade.exit_date.strftime('%Y-%m-%d'),
                        'Exit Price': f"${trade.exit_price:.2f}",
                        'P&L %': f"{trade.profit_loss_pct:+.2f}%",
                        'P&L $': f"${trade.profit_loss:+.2f}",
                        'Exit Reason': trade.exit_reason,
                        'Confidence': f"{trade.confidence_score:.0f}"
                    })

                trade_df = pd.DataFrame(trade_data)

                # Color the P&L column
                def color_pnl(val):
                    try:
                        num = float(val.replace('%', '').replace('$', '').replace('+', ''))
                        if num > 0:
                            return 'background-color: rgba(0, 200, 83, 0.3)'
                        elif num < 0:
                            return 'background-color: rgba(255, 82, 82, 0.3)'
                    except:
                        pass
                    return ''

                styled_df = trade_df.style.applymap(color_pnl, subset=['P&L %', 'P&L $'])
                st.dataframe(styled_df, use_container_width=True, height=400)

                # Export button
                csv = trade_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Trades CSV",
                    data=csv,
                    file_name=f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No trades executed during this backtest period")

            # Full report
            st.markdown("---")
            with st.expander("📋 Full Report"):
                st.code(format_metrics_report(metrics))

    with bt_tab4:
        st.markdown("### Mass Simulation")
        st.markdown("Simulate screening over 1 year on NASDAQ + Europe (daily/weekly)")

        # Import PortfolioSimulator for realistic mode
        from src.backtesting import PortfolioSimulator, SimulationResult

        # Simulation mode toggle
        simulation_mode = st.radio(
            "Simulation Mode",
            ["🚀 Basic (Fast)", "🎯 Realistic (Capital Management)"],
            horizontal=True,
            key="simulation_mode",
            help="Realistic mode simulates day-by-day with proper capital management and position limits"
        )

        is_realistic = "Realistic" in simulation_mode

        if is_realistic:
            st.success("""
            **✅ Realistic Portfolio Simulation**
            - Chronological day-by-day simulation
            - Proper capital management (positions block capital)
            - Max positions enforced at all times
            - Signals prioritized by confidence score
            - Uses pre-cached data for speed
            """)
        else:
            st.info("""
            **⚡ Basic Simulation Mode**
            - Processes each symbol independently
            - Faster but doesn't track concurrent positions
            - Good for initial signal validation
            """)

        mass_col1, mass_col2 = st.columns(2)

        with mass_col1:
            mass_markets = st.multiselect(
                "Markets to Include",
                ["NASDAQ", "SP500", "EUROPE", "CAC40", "DAX"],
                default=["NASDAQ", "EUROPE"],
                key="mass_markets"
            )

            mass_period = st.selectbox(
                "Simulation Period",
                ["1 Year", "6 Months", "2 Years"],
                key="mass_period"
            )

        with mass_col2:
            mass_timeframe = st.selectbox(
                "Screening Timeframe",
                ["Weekly", "Daily"],
                key="mass_timeframe"
            )

            mass_limit = st.number_input(
                "Max Symbols per Market",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                key="mass_limit"
            )

        # Calculate date range
        if mass_period == "1 Year":
            mass_start = datetime.now() - timedelta(days=365)
        elif mass_period == "6 Months":
            mass_start = datetime.now() - timedelta(days=180)
        else:
            mass_start = datetime.now() - timedelta(days=730)

        st.markdown(f"**Date Range:** {mass_start.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}")

        # Get symbol count
        total_symbols = 0
        for market in mass_markets:
            if market == "NASDAQ":
                count = min(len(market_data_fetcher.get_nasdaq_symbols()), mass_limit)
            elif market == "SP500":
                count = len(market_data_fetcher.get_sp500_symbols())
            elif market == "EUROPE":
                count = len(market_data_fetcher.get_europe_symbols())
            elif market == "CAC40":
                count = len(market_data_fetcher.get_cac40_symbols())
            elif market == "DAX":
                count = len(market_data_fetcher.get_dax_symbols())
            else:
                count = 0
            total_symbols += count

        st.info(f"Total symbols to analyze: ~{total_symbols}")

        # Cache statistics (for realistic mode)
        if is_realistic:
            st.markdown("---")
            st.markdown("#### 📦 Data Cache Status")
            cache_stats = market_data_fetcher.get_cache_stats()

            if 'error' not in cache_stats:
                cache_col1, cache_col2, cache_col3 = st.columns(3)
                with cache_col1:
                    st.metric("Cached Symbols", cache_stats.get('total_files', 0))
                with cache_col2:
                    st.metric("Cache Size", f"{cache_stats.get('total_size_mb', 0):.1f} MB")
                with cache_col3:
                    valid = cache_stats.get('valid_files', 0)
                    total = cache_stats.get('total_files', 1)
                    st.metric("Valid Cache", f"{valid}/{total}")

                if cache_stats.get('newest_file'):
                    st.caption(f"Last updated: {cache_stats['newest_file']}")

                # Prefetch button
                if st.button("🔄 Refresh Cache", key="refresh_cache"):
                    with st.spinner("Prefetching market data... This may take 15-30 minutes."):
                        try:
                            data = market_data_fetcher.prefetch_all_markets(
                                markets=mass_markets,
                                period='5y',
                                interval='1wk' if mass_timeframe == "Weekly" else '1d',
                                exclude_crypto=True
                            )
                            st.success(f"✅ Prefetch complete! {len(data)} symbols cached.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Prefetch failed: {e}")
            else:
                st.warning("Could not load cache statistics")

        st.markdown("---")

        if st.button("🚀 Start Mass Simulation", key="start_mass_simulation"):
            # Gather all symbols
            all_symbols = []
            for market in mass_markets:
                if market == "NASDAQ":
                    all_symbols.extend(market_data_fetcher.get_nasdaq_symbols()[:mass_limit])
                elif market == "SP500":
                    all_symbols.extend(market_data_fetcher.get_sp500_symbols())
                elif market == "EUROPE":
                    all_symbols.extend(market_data_fetcher.get_europe_symbols())
                elif market == "CAC40":
                    all_symbols.extend(market_data_fetcher.get_cac40_symbols())
                elif market == "DAX":
                    all_symbols.extend(market_data_fetcher.get_dax_symbols())

            # Remove duplicates
            all_symbols = list(set(all_symbols))

            st.write(f"Starting {'realistic ' if is_realistic else ''}simulation on {len(all_symbols)} symbols...")

            # Build config
            use_daily_fallback = st.session_state.get('bt_use_daily_fallback', True)
            config = BacktestConfig(
                initial_capital=st.session_state.get('bt_initial_capital', 10000),
                max_positions=st.session_state.get('bt_max_positions', 5),
                position_size_pct=st.session_state.get('bt_position_size', 20) / 100,
                min_confidence_score=st.session_state.get('bt_min_confidence', 55),
                require_volume_confirmation=st.session_state.get('bt_require_volume', True),
                min_volume_ratio=st.session_state.get('bt_min_volume_ratio', 1.0),
                use_trailing_stop=st.session_state.get('bt_use_trailing', False),
                trailing_stop_pct=st.session_state.get('bt_trailing_pct', 5) / 100,
                take_profit_pct=st.session_state.get('bt_take_profit', 15) / 100,
                max_hold_days=st.session_state.get('bt_max_hold', 60),
                use_enhanced_detector=st.session_state.get('bt_use_enhanced', True),
                precision_mode=st.session_state.get('bt_precision', 'medium'),
                timeframe='weekly' if mass_timeframe == "Weekly" else 'daily',
                use_daily_fallback=use_daily_fallback,
                min_ema_conditions_for_fallback=2
            )

            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()

            try:
                if is_realistic:
                    # Realistic mode: Use PortfolioSimulator
                    status_text.text("Loading weekly market data...")
                    progress_bar.progress(10)

                    # Always load weekly data (primary timeframe)
                    all_data_weekly = market_data_fetcher.get_batch_historical_data(
                        all_symbols,
                        period='5y',
                        interval='1wk',
                        batch_size=100
                    )

                    # Load daily data if dual-timeframe is enabled
                    all_data_daily = None
                    if use_daily_fallback:
                        status_text.text(f"Loaded {len(all_data_weekly)} weekly. Loading daily data...")
                        progress_bar.progress(20)
                        all_data_daily = market_data_fetcher.get_batch_historical_data(
                            all_symbols,
                            period='5y',
                            interval='1d',
                            batch_size=100
                        )
                        status_text.text(f"Loaded {len(all_data_weekly)} weekly + {len(all_data_daily)} daily. Running...")
                    else:
                        status_text.text(f"Loaded {len(all_data_weekly)} symbols. Running simulation...")

                    progress_bar.progress(30)

                    # Run realistic simulation with dual-timeframe
                    simulator = PortfolioSimulator(config)

                    def progress_callback(current, total, msg):
                        pct = 30 + int((current / max(total, 1)) * 60)
                        progress_bar.progress(min(pct, 90))
                        status_text.text(f"Day {current}/{total}: {msg}")

                    result = simulator.run_simulation(
                        all_data=all_data_weekly,
                        start_date=mass_start.strftime('%Y-%m-%d'),
                        end_date=datetime.now().strftime('%Y-%m-%d'),
                        progress_callback=progress_callback,
                        all_data_daily=all_data_daily  # Pass daily data for fallback
                    )

                    # Store as SimulationResult (compatible with backtest_result)
                    st.session_state['backtest_result'] = result
                    st.session_state['simulation_result'] = result  # Extra for realistic stats

                else:
                    # Basic mode: Use regular Backtester
                    backtester = Backtester(config)
                    status_text.text(f"Running mass simulation... This may take a while.")

                    result = backtester.run(
                        symbols=all_symbols,
                        start_date=mass_start.strftime('%Y-%m-%d'),
                        end_date=datetime.now().strftime('%Y-%m-%d')
                    )

                    st.session_state['backtest_result'] = result

                progress_bar.progress(100)
                status_text.text("Simulation complete!")

                # Show summary
                st.success("✅ Mass Simulation Complete!")

                st.markdown("### Results Summary")
                metrics = result.metrics

                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                with res_col1:
                    st.metric("Total Return", f"{metrics.total_return:+.2f}%")
                with res_col2:
                    st.metric("Total Trades", metrics.total_trades)
                with res_col3:
                    st.metric("Win Rate", f"{metrics.win_rate:.1f}%")
                with res_col4:
                    st.metric("Final Capital", f"${metrics.final_capital:,.2f}")

                # Extra stats for realistic mode
                if is_realistic and hasattr(result, 'signals_detected'):
                    st.markdown("### Signal Statistics")
                    sig_col1, sig_col2, sig_col3, sig_col4 = st.columns(4)
                    with sig_col1:
                        st.metric("Signals Detected", result.signals_detected)
                    with sig_col2:
                        st.metric("Signals Taken", result.signals_taken)
                    with sig_col3:
                        st.metric("Skipped (Capital)", result.signals_skipped_capital)
                    with sig_col4:
                        st.metric("Skipped (Max Pos)", result.signals_skipped_max_positions)

                st.info("Go to the 'Results' tab for detailed analysis and equity curve.")

            except Exception as e:
                st.error(f"Simulation failed: {e}")
                import traceback
                st.code(traceback.format_exc())

elif page == "📋 Watchlists":
    st.title("📋 Watchlists")
    st.markdown("Manage your custom watchlists and scan them for signals")

    # Create tabs for different actions
    tab1, tab2, tab3 = st.tabs(["📊 View & Scan", "➕ Manage", "📥 Import/Export"])

    with tab1:
        st.markdown("### Select a Watchlist")

        watchlist_names = watchlist_manager.get_watchlist_names()

        if not watchlist_names:
            st.info("No watchlists found. Create one in the 'Manage' tab.")
        else:
            col1, col2 = st.columns([2, 1])

            with col1:
                selected_watchlist = st.selectbox(
                    "Choose watchlist",
                    watchlist_names,
                    key="view_watchlist_select"
                )

            with col2:
                st.metric("Symbols", watchlist_manager.get_symbol_count(selected_watchlist))

            # Display watchlist symbols
            symbols = watchlist_manager.get_watchlist(selected_watchlist)

            if symbols:
                st.markdown(f"**Symbols:** {', '.join(symbols)}")

                st.markdown("---")
                st.markdown("### Quick Scan")

                col1, col2 = st.columns([1, 2])
                with col1:
                    scan_timeframe = st.selectbox("Timeframe", ["weekly", "daily"], key="wl_scan_tf")

                if st.button("🔍 Scan Watchlist", type="primary", key="scan_watchlist_btn"):
                    with st.spinner(f"Scanning {len(symbols)} symbols..."):
                        # Prepare stocks list for screener
                        stocks = [{'symbol': s, 'name': s} for s in symbols]

                        # Run screening
                        alerts = market_screener.screen_multiple_stocks(stocks)

                        if alerts:
                            st.success(f"Found {len(alerts)} signals!")

                            # Sort by confidence score
                            alerts_sorted = sorted(alerts, key=lambda x: x.get('confidence_score', 0), reverse=True)

                            # Display results
                            for alert in alerts_sorted:
                                signal = alert.get('confidence_signal', 'N/A')
                                score = alert.get('confidence_score', 0)

                                with st.expander(f"🎯 {alert['symbol']} - {signal} ({score:.0f}/100)"):
                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        st.metric("Price", f"${alert['current_price']:.2f}")
                                        st.metric("Support", f"${alert['support_level']:.2f}")

                                    with col2:
                                        st.metric("Distance", f"{alert['distance_to_support_pct']:.1f}%")
                                        if alert.get('has_rsi_breakout'):
                                            st.metric("RSI Breakout", f"✅ {alert.get('rsi_breakout_strength', 'N/A')}")

                                    with col3:
                                        st.metric("Position", f"{alert.get('position_shares', 'N/A')} shares")
                                        st.metric("Stop Loss", f"${alert.get('stop_loss', 0):.2f}")
                        else:
                            st.info("No signals found in this watchlist")

                # Display symbols as cards
                st.markdown("---")
                st.markdown("### Symbols in Watchlist")

                cols = st.columns(4)
                for i, symbol in enumerate(symbols):
                    with cols[i % 4]:
                        st.markdown(f"""
                        <div style="background: #1E1E1E; padding: 10px; border-radius: 8px; margin: 5px 0; text-align: center;">
                            <b>{symbol}</b>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("This watchlist is empty. Add symbols in the 'Manage' tab.")

    with tab2:
        st.markdown("### Manage Watchlists")

        # Create new watchlist
        st.markdown("#### Create New Watchlist")
        col1, col2 = st.columns([2, 1])

        with col1:
            new_wl_name = st.text_input("Watchlist name", key="new_wl_name")

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Create", key="create_wl_btn"):
                if new_wl_name:
                    if watchlist_manager.create_watchlist(new_wl_name):
                        st.success(f"Created watchlist '{new_wl_name}'")
                        st.rerun()
                    else:
                        st.error("Watchlist already exists")
                else:
                    st.warning("Enter a name for the watchlist")

        st.markdown("---")

        # Manage existing watchlists
        st.markdown("#### Edit Watchlist")

        watchlist_names = watchlist_manager.get_watchlist_names()

        if watchlist_names:
            selected_wl = st.selectbox(
                "Select watchlist to edit",
                watchlist_names,
                key="edit_wl_select"
            )

            symbols = watchlist_manager.get_watchlist(selected_wl)
            st.markdown(f"**Current symbols ({len(symbols)}):** {', '.join(symbols) if symbols else 'None'}")

            # Add symbol
            col1, col2 = st.columns([2, 1])
            with col1:
                new_symbol = st.text_input("Add symbol", key="add_symbol_input").upper()
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("➕ Add", key="add_symbol_btn"):
                    if new_symbol:
                        if watchlist_manager.add_symbol(selected_wl, new_symbol):
                            st.success(f"Added {new_symbol}")
                            st.rerun()
                        else:
                            st.warning("Symbol already in watchlist or invalid")

            # Remove symbol
            if symbols:
                col1, col2 = st.columns([2, 1])
                with col1:
                    symbol_to_remove = st.selectbox("Remove symbol", symbols, key="remove_symbol_select")
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🗑️ Remove", key="remove_symbol_btn"):
                        if watchlist_manager.remove_symbol(selected_wl, symbol_to_remove):
                            st.success(f"Removed {symbol_to_remove}")
                            st.rerun()

            st.markdown("---")

            # Delete watchlist
            st.markdown("#### Delete Watchlist")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.warning(f"⚠️ This will permanently delete '{selected_wl}'")
            with col2:
                if st.button("🗑️ Delete Watchlist", key="delete_wl_btn", type="secondary"):
                    if watchlist_manager.delete_watchlist(selected_wl):
                        st.success(f"Deleted '{selected_wl}'")
                        st.rerun()

    with tab3:
        st.markdown("### Import / Export")

        watchlist_names = watchlist_manager.get_watchlist_names()

        # Export
        st.markdown("#### Export Watchlist")
        if watchlist_names:
            export_wl = st.selectbox("Select watchlist to export", watchlist_names, key="export_wl_select")
            export_str = watchlist_manager.export_watchlist(export_wl)
            if export_str:
                st.text_area("Symbols (copy this)", export_str, key="export_text", height=100)
            else:
                st.info("Watchlist is empty")

        st.markdown("---")

        # Import
        st.markdown("#### Import Symbols")
        col1, col2 = st.columns([2, 1])

        with col1:
            import_wl_name = st.text_input("Watchlist name (new or existing)", key="import_wl_name")

        import_symbols = st.text_area(
            "Paste symbols (comma, space, or newline separated)",
            placeholder="AAPL, MSFT, GOOGL\nNVDA\nTSLA",
            key="import_symbols_text",
            height=150
        )

        if st.button("📥 Import", key="import_btn"):
            if import_wl_name and import_symbols:
                count = watchlist_manager.import_from_string(import_wl_name, import_symbols)
                st.success(f"Imported {count} symbols to '{import_wl_name}'")
                st.rerun()
            else:
                st.warning("Enter watchlist name and symbols to import")

elif page == "💼 Portfolio":
    st.title("💼 Portfolio Management")
    st.markdown("Manage your trading capital and open positions")

    # Portfolio summary
    portfolio = market_screener.get_portfolio_summary()

    st.markdown("### 📊 Capital Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Capital Total", f"{portfolio['total_capital']:,.0f}€")
    with col2:
        st.metric("Capital Investi", f"{portfolio['invested_capital']:,.0f}€")
    with col3:
        st.metric("Capital Disponible", f"{portfolio['available_capital']:,.0f}€")
    with col4:
        pct_invested = (portfolio['invested_capital'] / portfolio['total_capital'] * 100) if portfolio['total_capital'] > 0 else 0
        st.metric("% Investi", f"{pct_invested:.1f}%")

    st.markdown("---")

    # Capital management
    st.markdown("### 💰 Update Capital")
    col1, col2 = st.columns([2, 1])

    with col1:
        new_capital = st.number_input(
            "New total capital (€)",
            min_value=0.0,
            value=float(portfolio['total_capital']),
            step=1000.0,
            key="new_capital"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Update Capital", key="update_capital"):
            market_screener.update_capital(new_capital)
            st.success(f"Capital updated to {new_capital:,.0f}€")
            st.rerun()

    st.markdown("---")

    # Open positions
    st.markdown("### 📈 Open Positions")
    positions = portfolio.get('positions', [])

    if positions:
        positions_df = pd.DataFrame(positions)
        positions_df['P&L Est.'] = "N/A"  # Could be calculated with current prices

        st.dataframe(
            positions_df.style.format({
                'entry_price': '${:.2f}',
                'stop_loss': '${:.2f}',
                'position_value': '{:,.0f}€'
            }),
            use_container_width=True
        )

        # Close position
        st.markdown("#### Close Position")
        col1, col2 = st.columns([2, 1])

        with col1:
            symbol_to_close = st.selectbox(
                "Select position to close",
                [p['symbol'] for p in positions],
                key="close_symbol"
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔴 Close Position", key="close_position"):
                if market_screener.close_position(symbol_to_close):
                    st.success(f"Position {symbol_to_close} closed")
                    st.rerun()
                else:
                    st.error(f"Could not close position {symbol_to_close}")
    else:
        st.info("No open positions")

    st.markdown("---")

    # Add manual position
    st.markdown("### ➕ Add Position Manually")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        add_symbol = st.text_input("Symbol", key="add_symbol").upper()
    with col2:
        add_shares = st.number_input("Shares", min_value=1, value=10, key="add_shares")
    with col3:
        add_entry = st.number_input("Entry Price (€)", min_value=0.01, value=100.0, key="add_entry")
    with col4:
        add_stop = st.number_input("Stop Loss (€)", min_value=0.01, value=95.0, key="add_stop")

    if st.button("➕ Add Position", key="add_position"):
        if add_symbol:
            market_screener.add_position(add_symbol, add_shares, add_entry, add_stop)
            st.success(f"Position added: {add_symbol} {add_shares} shares @ {add_entry}€")
            st.rerun()
        else:
            st.warning("Please enter a symbol")

elif page == "📊 Performance":
    st.title("📊 Trading Performance")
    st.markdown("Track your trades and analyze performance metrics")

    # Performance summary
    perf = trade_tracker.get_performance_summary()

    # Key metrics row
    st.markdown("### 📈 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        delta_color = "normal" if perf['total_pnl'] >= 0 else "inverse"
        st.metric(
            "Total P&L",
            f"${perf['total_pnl']:,.2f}",
            delta=f"{perf['roi_pct']:.1f}%",
            delta_color=delta_color
        )

    with col2:
        st.metric("Win Rate", f"{perf['win_rate']:.1f}%", delta=f"{perf['wins']}W / {perf['losses']}L")

    with col3:
        st.metric("Profit Factor", f"{perf['profit_factor']:.2f}")

    with col4:
        st.metric("Sharpe Ratio", f"{perf['sharpe_ratio']:.2f}")

    with col5:
        st.metric("Max Drawdown", f"-{perf['max_drawdown_pct']:.1f}%")

    # Secondary metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Initial Capital", f"${perf['initial_capital']:,.0f}")
    with col2:
        st.metric("Current Equity", f"${perf['current_equity']:,.0f}")
    with col3:
        st.metric("Avg Win", f"${perf['avg_win']:,.2f}")
    with col4:
        st.metric("Avg Loss", f"${perf['avg_loss']:,.2f}")

    st.markdown("---")

    # Charts tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Equity Curve", "📊 Monthly Returns", "🏷️ By Symbol", "📋 Trade History"])

    with tab1:
        st.markdown("### 📈 Equity Curve & Drawdown")
        equity_df = trade_tracker.get_equity_curve()
        fig_equity = interactive_chart_builder.create_equity_curve_chart(
            equity_df,
            initial_capital=perf['initial_capital'],
            height=500
        )
        st.plotly_chart(fig_equity, use_container_width=True)

        # Win/Loss and Distribution side by side
        col1, col2 = st.columns(2)

        with col1:
            fig_pie = interactive_chart_builder.create_win_loss_pie(
                perf['wins'],
                perf['losses'],
                height=300
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            closed_trades = trade_tracker.get_closed_trades()
            fig_dist = interactive_chart_builder.create_trade_distribution_chart(
                closed_trades,
                height=300
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    with tab2:
        st.markdown("### 📊 Monthly P&L")
        monthly_df = trade_tracker.get_monthly_returns()
        fig_monthly = interactive_chart_builder.create_monthly_returns_chart(
            monthly_df,
            height=400
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Monthly table
        if not monthly_df.empty:
            st.markdown("#### Monthly Breakdown")
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_df['Month'] = monthly_df.apply(
                lambda r: f"{month_names[int(r['month'])-1]} {int(r['year'])}", axis=1
            )
            st.dataframe(
                monthly_df[['Month', 'pnl', 'trades', 'win_rate']].style.format({
                    'pnl': '${:,.2f}',
                    'win_rate': '{:.1f}%'
                }),
                use_container_width=True
            )

    with tab3:
        st.markdown("### 🏷️ Performance by Symbol")
        symbol_df = trade_tracker.get_symbol_performance()
        fig_symbol = interactive_chart_builder.create_symbol_performance_chart(
            symbol_df,
            height=400
        )
        st.plotly_chart(fig_symbol, use_container_width=True)

        # Symbol table
        if not symbol_df.empty:
            st.markdown("#### Symbol Breakdown")
            st.dataframe(
                symbol_df.style.format({
                    'pnl': '${:,.2f}',
                    'win_rate': '{:.1f}%',
                    'avg_pnl': '${:,.2f}'
                }),
                use_container_width=True
            )

    with tab4:
        st.markdown("### 📋 Trade History")

        # Filter options
        col1, col2 = st.columns([1, 3])
        with col1:
            filter_status = st.selectbox(
                "Status",
                ["All", "Open", "Closed"],
                key="trade_filter"
            )

        # Get trades based on filter
        if filter_status == "Open":
            trades = trade_tracker.get_open_trades()
        elif filter_status == "Closed":
            trades = trade_tracker.get_closed_trades()
        else:
            trades = trade_tracker.trades

        if trades:
            trades_data = []
            for t in trades:
                trades_data.append({
                    'ID': t.trade_id,
                    'Symbol': t.symbol,
                    'Status': t.status,
                    'Entry Date': t.entry_date,
                    'Entry Price': t.entry_price,
                    'Shares': t.shares,
                    'Stop Loss': t.stop_loss,
                    'Exit Date': t.exit_date or '-',
                    'Exit Price': t.exit_price or '-',
                    'P&L': f"${t.pnl:,.2f}" if t.pnl != 0 else '-',
                    'P&L %': f"{t.pnl_pct:.1f}%" if t.pnl_pct != 0 else '-'
                })

            df_trades = pd.DataFrame(trades_data)
            st.dataframe(df_trades, use_container_width=True)
        else:
            st.info("No trades to display")

        st.markdown("---")

        # Add new trade
        st.markdown("### ➕ Record New Trade")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            new_symbol = st.text_input("Symbol", key="perf_new_symbol").upper()
        with col2:
            new_entry_price = st.number_input("Entry Price", min_value=0.01, value=100.0, key="perf_entry")
        with col3:
            new_shares = st.number_input("Shares", min_value=1, value=10, key="perf_shares")
        with col4:
            new_stop = st.number_input("Stop Loss", min_value=0.01, value=95.0, key="perf_stop")
        with col5:
            new_target = st.number_input("Target (optional)", min_value=0.0, value=0.0, key="perf_target")

        if st.button("➕ Open Trade", key="perf_open_trade"):
            if new_symbol:
                target = new_target if new_target > 0 else None
                trade = trade_tracker.open_trade(
                    symbol=new_symbol,
                    entry_price=new_entry_price,
                    shares=new_shares,
                    stop_loss=new_stop,
                    target_price=target
                )
                st.success(f"Trade opened: {trade.trade_id}")
                st.rerun()
            else:
                st.warning("Please enter a symbol")

        # Close trade section
        open_trades = trade_tracker.get_open_trades()
        if open_trades:
            st.markdown("### 🔴 Close Trade")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                trade_to_close = st.selectbox(
                    "Select Trade",
                    [f"{t.trade_id} ({t.symbol})" for t in open_trades],
                    key="perf_close_trade"
                )
            with col2:
                close_price = st.number_input("Exit Price", min_value=0.01, value=100.0, key="perf_close_price")
            with col3:
                close_reason = st.selectbox(
                    "Reason",
                    ["manual", "target", "stop_loss", "trailing_stop"],
                    key="perf_close_reason"
                )

            with col4:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔴 Close Trade", key="perf_close_btn"):
                    trade_id = trade_to_close.split(" ")[0]
                    trade_tracker.close_trade(trade_id, close_price, close_reason)
                    st.success(f"Trade {trade_id} closed")
                    st.rerun()

        st.markdown("---")

        # Capital management
        st.markdown("### 💰 Initial Capital")
        col1, col2 = st.columns([2, 1])

        with col1:
            new_initial = st.number_input(
                "Set Initial Capital",
                min_value=0.0,
                value=float(perf['initial_capital']),
                step=1000.0,
                key="perf_initial_capital"
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 Update", key="perf_update_capital"):
                trade_tracker.set_initial_capital(new_initial)
                st.success(f"Initial capital set to ${new_initial:,.0f}")
                st.rerun()

elif page == "🗺️ Sector Map":
    st.title("🗺️ Sector Heatmap")
    st.markdown("Market overview by sector with performance and rotation analysis")

    # Controls
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        metric = st.selectbox(
            "Performance Period",
            ["1D %", "1W %", "1M %", "YTD %"],
            key="sector_metric"
        )

    with col2:
        view_type = st.selectbox(
            "View Type",
            ["Treemap", "Bar Chart", "Rotation"],
            key="sector_view"
        )

    with col3:
        if st.button("🔄 Refresh Data", key="refresh_sectors"):
            st.cache_data.clear()

    st.markdown("---")

    # Fetch sector data
    with st.spinner("Loading sector data..."):
        try:
            sector_data = sector_heatmap_builder.get_sector_performance(use_cache=True)
            breadth = sector_heatmap_builder.get_market_breadth()
            rotation = sector_heatmap_builder.get_sector_rotation_signal()
        except Exception as e:
            st.error(f"Error loading sector data: {e}")
            sector_data = []
            breadth = {'advancing': 0, 'declining': 0, 'total': 0}
            rotation = {'signal': 'NEUTRAL', 'reason': 'No data'}

    if sector_data:
        # Market overview row
        st.markdown("### 📊 Market Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_perf = breadth.get('avg_performance', 0)
            st.metric(
                "Avg Performance",
                f"{avg_perf:+.2f}%",
                delta=f"{breadth['advancing']} advancing"
            )

        with col2:
            ratio = breadth.get('breadth_ratio', 1)
            st.metric(
                "Breadth Ratio",
                f"{ratio:.2f}",
                delta="Bullish" if ratio > 1 else "Bearish"
            )

        with col3:
            signal = rotation.get('signal', 'NEUTRAL')
            signal_color = {"RISK_ON": "green", "RISK_OFF": "red", "NEUTRAL": "gray"}
            st.metric("Rotation Signal", signal)

        with col4:
            st.metric(
                "Sectors",
                f"{breadth['advancing']} / {breadth['declining']}",
                delta="Adv/Dec"
            )

        st.markdown("---")

        # Main visualization
        if view_type == "Treemap":
            st.markdown("### 🗺️ Sector Treemap")
            fig = interactive_chart_builder.create_sector_heatmap(
                sector_data,
                metric=metric,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        elif view_type == "Bar Chart":
            st.markdown("### 📊 Sector Performance")
            fig = interactive_chart_builder.create_sector_bar_chart(
                sector_data,
                metric=metric,
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

        else:  # Rotation
            st.markdown("### 🔄 Sector Rotation Analysis")
            st.info(f"**{rotation['signal']}**: {rotation['reason']}")

            fig = interactive_chart_builder.create_sector_rotation_chart(
                sector_data,
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Quadrant Interpretation:**
            - **Leading** (top-right): Strong 1W & 1M performance - momentum sectors
            - **Improving** (top-left): Strong 1W, weak 1M - early rotation targets
            - **Weakening** (bottom-right): Weak 1W, strong 1M - taking profits
            - **Lagging** (bottom-left): Weak 1W & 1M - avoid or watch for reversal
            """)

        st.markdown("---")

        # Side-by-side: Breadth gauge + Table
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 📈 Market Breadth")
            fig_gauge = interactive_chart_builder.create_market_breadth_gauge(
                breadth,
                height=280
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            st.markdown("### 📋 Sector Details")
            # Create DataFrame for table
            table_data = []
            for s in sector_data:
                table_data.append({
                    'Sector': s.name,
                    'ETF': s.etf,
                    '1D': f"{s.perf_1d:+.2f}%",
                    '1W': f"{s.perf_1w:+.2f}%",
                    '1M': f"{s.perf_1m:+.2f}%",
                    'YTD': f"{s.perf_ytd:+.2f}%"
                })

            df_sectors = pd.DataFrame(table_data)
            st.dataframe(df_sectors, use_container_width=True, height=280)

        st.markdown("---")

        # Top movers section
        st.markdown("### 🚀 Top Movers by Sector")

        cols = st.columns(3)
        for i, s in enumerate(sector_data[:6]):
            with cols[i % 3]:
                with st.expander(f"**{s.name}** ({s.etf})", expanded=False):
                    if s.top_gainers:
                        st.markdown("**Top Gainers:**")
                        for g in s.top_gainers:
                            st.markdown(f"- {g['symbol']}: {g['perf']:+.2f}%")
                    if s.top_losers:
                        st.markdown("**Top Losers:**")
                        for l in s.top_losers:
                            st.markdown(f"- {l['symbol']}: {l['perf']:+.2f}%")

    else:
        st.warning("No sector data available. Please check your internet connection.")

elif page == "📅 Calendar":
    st.title("📅 Economic Calendar")
    st.markdown("Track earnings dates, FOMC meetings, CPI releases, and other market-moving events")

    # Controls
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        days_ahead = st.selectbox(
            "Time Range",
            [7, 14, 30, 60],
            index=2,
            format_func=lambda x: f"Next {x} days",
            key="calendar_days"
        )

    with col2:
        view_type = st.selectbox(
            "View Type",
            ["Timeline", "Table", "Week View"],
            key="calendar_view"
        )

    with col3:
        # Get tracked symbols from watchlist
        tracked_symbols = []
        try:
            watchlists = watchlist_manager.get_all_watchlists()
            for wl in watchlists:
                tracked_symbols.extend(wl.symbols)
            tracked_symbols = list(set(tracked_symbols))
        except:
            pass

        if st.button("🔄 Refresh Calendar", key="refresh_calendar"):
            st.cache_data.clear()

    st.markdown("---")

    # Event type filter
    st.markdown("### 🎯 Filter Events")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_fomc = st.checkbox("FOMC Meetings", value=True, key="show_fomc")
    with col2:
        show_cpi = st.checkbox("CPI Reports", value=True, key="show_cpi")
    with col3:
        show_jobs = st.checkbox("Jobs Reports", value=True, key="show_jobs")
    with col4:
        show_earnings = st.checkbox("Earnings", value=True, key="show_earnings")

    # Build event type filter
    event_types = []
    if show_fomc:
        event_types.append(EventType.FOMC)
    if show_cpi:
        event_types.append(EventType.CPI)
    if show_jobs:
        event_types.append(EventType.JOBS)

    st.markdown("---")

    # Fetch events
    with st.spinner("Loading economic events..."):
        try:
            events = economic_calendar.get_upcoming_events(
                days_ahead=days_ahead,
                event_types=event_types if event_types else None,
                include_earnings=show_earnings,
                symbols=tracked_symbols[:20] if show_earnings else None  # Limit to 20 symbols
            )
        except Exception as e:
            st.error(f"Error loading events: {e}")
            events = []

    # Summary metrics
    st.markdown("### 📊 Calendar Summary")
    col1, col2, col3, col4 = st.columns(4)

    high_impact = [e for e in events if e.impact.value == 'high']
    earnings_events = [e for e in events if e.event_type == EventType.EARNINGS]
    fomc_events = [e for e in events if e.event_type == EventType.FOMC]

    with col1:
        st.metric("Total Events", len(events))
    with col2:
        st.metric("High Impact", len(high_impact), delta="⚠️" if high_impact else None)
    with col3:
        st.metric("Earnings", len(earnings_events))
    with col4:
        next_fomc = economic_calendar.get_next_event(EventType.FOMC)
        if next_fomc:
            days_to_fomc = (next_fomc.date - datetime.now()).days
            st.metric("Next FOMC", f"{days_to_fomc}d")
        else:
            st.metric("Next FOMC", "N/A")

    st.markdown("---")

    # Main visualization
    if view_type == "Timeline":
        st.markdown("### 📈 Event Timeline")
        if events:
            fig = interactive_chart_builder.create_calendar_timeline(events, height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No events found for the selected filters.")

    elif view_type == "Table":
        st.markdown("### 📋 Event List")
        if events:
            fig = interactive_chart_builder.create_calendar_table(events, height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Also show DataFrame for filtering/sorting
            df_events = economic_calendar.get_calendar_dataframe(
                days_ahead=days_ahead,
                symbols=tracked_symbols[:20] if show_earnings else None
            )
            if not df_events.empty:
                st.dataframe(df_events, use_container_width=True, height=300)
        else:
            st.info("No events found for the selected filters.")

    else:  # Week View
        st.markdown("### 📅 This Week")
        week_events = economic_calendar.get_week_events(week_offset=0)
        fig = interactive_chart_builder.create_week_calendar(week_events, height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Next week
        st.markdown("### 📅 Next Week")
        next_week_events = economic_calendar.get_week_events(week_offset=1)
        fig_next = interactive_chart_builder.create_week_calendar(next_week_events, height=300)
        st.plotly_chart(fig_next, use_container_width=True)

    st.markdown("---")

    # Detailed sections
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 High Impact Events")
        if high_impact:
            for event in high_impact[:5]:
                days_until = (event.date - datetime.now()).days
                icon = "🔴" if days_until <= 3 else "🟡" if days_until <= 7 else "🟢"
                st.markdown(f"{icon} **{event.title}** - {event.date.strftime('%Y-%m-%d')}")
                st.caption(f"   {event.description[:60]}..." if event.description else "")
        else:
            st.info("No high impact events in the selected period.")

    with col2:
        st.markdown("### 💰 Upcoming Earnings")
        if earnings_events:
            for event in earnings_events[:5]:
                days_until = (event.date - datetime.now()).days
                icon = "🔴" if days_until <= 3 else "🟡" if days_until <= 7 else "🟢"
                st.markdown(f"{icon} **{event.symbol}** - {event.date.strftime('%Y-%m-%d')}")
        else:
            st.info("No earnings in tracked symbols for this period.")

    st.markdown("---")

    # Impact distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Events by Impact")
        if events:
            fig = interactive_chart_builder.create_event_impact_chart(events, height=280)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### ⚙️ Custom Events")
        st.caption("Add custom events to your calendar")

        custom_title = st.text_input("Event Title", key="custom_event_title")
        custom_date = st.date_input("Event Date", key="custom_event_date")
        custom_impact = st.selectbox(
            "Impact Level",
            ["HIGH", "MEDIUM", "LOW"],
            key="custom_event_impact"
        )

        if st.button("➕ Add Custom Event", key="add_custom_event"):
            if custom_title:
                try:
                    economic_calendar.add_custom_event(
                        title=custom_title,
                        date=datetime.combine(custom_date, datetime.min.time()),
                        impact=EventImpact(custom_impact.lower()),
                        description="Custom event added via dashboard"
                    )
                    st.success(f"Added: {custom_title}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding event: {e}")
            else:
                st.warning("Please enter an event title")

    # Earnings warnings section
    st.markdown("---")
    st.markdown("### ⚠️ Earnings Warnings for Watchlist")

    warnings = []
    for symbol in tracked_symbols[:10]:
        warning = economic_calendar.get_earnings_warning(symbol)
        if warning:
            warnings.append(warning)

    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.success("No imminent earnings for tracked symbols (next 7 days)")

elif page == "🎯 Trendline Analysis":
    st.title("🎯 Trendline Breakout Analysis")
    st.markdown("Double confirmation: RSI + Price trendline breakouts within ±6 periods")

    # Import trendline modules
    from trendline_analysis.core.dual_confirmation_analyzer import DualConfirmationAnalyzer
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Tab selection
    tab1, tab2 = st.tabs(["📊 Multi-Symbol Scan", "🔬 Single Symbol Analysis"])

    with tab2:
        # Single symbol analysis (original)
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            symbol = st.text_input("Enter Symbol", value="AAPL", key="trendline_symbol").upper()

        with col2:
            timeframe = st.selectbox("Timeframe", ["daily", "weekly"], key="trendline_timeframe")

        with col3:
            lookback = st.selectbox("Lookback", [104, 252, 500], index=1, key="trendline_lookback")

        analyze_single = st.button("🔍 Analyze Trendlines", type="primary", key="analyze_single")

    with tab1:
        # Multi-symbol scan
        st.markdown("### 📊 Breakout Scanner")
        st.markdown("Analyze multiple symbols and categorize by breakout type")

        col1, col2, col3 = st.columns(3)

        with col1:
            scan_type = st.selectbox(
                "Symbol Set",
                ["Predefined Mix", "Custom List", "Major Indices", "Tech Stocks", "Crypto"],
                key="scan_type"
            )

        with col2:
            scan_lookback = st.selectbox("Lookback", [126, 252], index=1, key="scan_lookback")

        with col3:
            scan_period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=2, key="scan_period")

        # Symbol selection
        if scan_type == "Predefined Mix":
            symbols_to_scan = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA', 'NVDA', 'BTC-USD', 'ETH-USD',
                              'DIS', 'META', 'NFLX', 'AMD']
        elif scan_type == "Major Indices":
            symbols_to_scan = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI']
        elif scan_type == "Tech Stocks":
            symbols_to_scan = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'TSLA', 'NFLX']
        elif scan_type == "Crypto":
            symbols_to_scan = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD']
        else:
            custom_symbols = st.text_input("Enter symbols (comma-separated)", "SPY,QQQ,AAPL", key="custom_symbols")
            symbols_to_scan = [s.strip().upper() for s in custom_symbols.split(',') if s.strip()]

        st.info(f"Ready to scan {len(symbols_to_scan)} symbols")

        run_scan = st.button("🚀 Run Breakout Scan", type="primary", key="run_scan")

        if run_scan:
            with st.spinner("Scanning symbols..."):
                analyzer = DualConfirmationAnalyzer(sync_window=6)

                # Categories
                dual_confirmations = []
                rsi_only = []
                price_only = []
                trendlines_only = []
                no_signals = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, sym in enumerate(symbols_to_scan):
                    status_text.text(f"Analyzing {sym}... ({i+1}/{len(symbols_to_scan)})")

                    try:
                        df = market_data_fetcher.get_historical_data(sym, period=scan_period, interval='1d')

                        if df is not None and len(df) >= 50:
                            result = analyzer.analyze(df, lookback_periods=scan_lookback)

                            if result is not None:
                                case = {
                                    'symbol': sym,
                                    'result': result,
                                    'df': df
                                }

                                # Categorize
                                if result['has_dual_confirmation']:
                                    dual_confirmations.append(case)
                                elif result['has_rsi_breakout'] and not result.get('has_price_breakout'):
                                    rsi_only.append(case)
                                elif result.get('has_price_breakout') and not result['has_rsi_breakout']:
                                    price_only.append(case)
                                elif result['has_rsi_trendline'] or result.get('has_price_trendline'):
                                    trendlines_only.append(case)
                                else:
                                    no_signals.append(case)
                            else:
                                no_signals.append({'symbol': sym, 'result': None, 'df': None})

                    except Exception as e:
                        st.warning(f"Error analyzing {sym}: {e}")

                    progress_bar.progress((i + 1) / len(symbols_to_scan))

                status_text.text("Scan complete!")
                st.success("✅ Scan completed successfully!")

                # Display results by category
                st.markdown("---")
                st.markdown("## 📊 Scan Results")

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("🎯 Dual Confirmations", len(dual_confirmations))

                with col2:
                    st.metric("⚠️ RSI Only", len(rsi_only))

                with col3:
                    st.metric("📈 Price Only", len(price_only))

                with col4:
                    st.metric("⏳ Trendlines Only", len(trendlines_only))

                # Store in session state for visualization
                if 'scan_results' not in st.session_state:
                    st.session_state.scan_results = {}

                st.session_state.scan_results = {
                    'dual_confirmations': dual_confirmations,
                    'rsi_only': rsi_only,
                    'price_only': price_only,
                    'trendlines_only': trendlines_only
                }

        # Display categorized results
        if 'scan_results' in st.session_state and st.session_state.scan_results:
            results = st.session_state.scan_results

            # Create analyzer for signal generation
            temp_analyzer = DualConfirmationAnalyzer(sync_window=6)

            # Category 1: Dual Confirmations
            if results['dual_confirmations']:
                st.markdown("---")
                st.markdown("### 🎯 Dual Confirmations (STRONGEST SIGNALS)")

                for case in results['dual_confirmations']:
                    with st.expander(f"✨ {case['symbol']} - {case['result']['dual_confirmation'].confirmation_strength}"):
                        result = case['result']
                        dual = result['dual_confirmation']

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("RSI Breakout", result['rsi_breakout'].date.strftime('%Y-%m-%d'))
                            st.metric("RSI Strength", result['rsi_breakout'].strength)

                        with col2:
                            st.metric("Price Breakout", result['price_breakout'].date.strftime('%Y-%m-%d'))
                            st.metric("Price Strength", result['price_breakout'].strength)

                        with col3:
                            st.metric("Time Difference", f"{dual.time_difference} periods")
                            st.metric("Synchronized", "✅ YES")

                        with col4:
                            signal = temp_analyzer.get_signal(result)
                            st.metric("Signal", signal)
                            st.metric("Confirmation", dual.confirmation_strength)

                        if st.button(f"📊 View Chart for {case['symbol']}", key=f"chart_dual_{case['symbol']}"):
                            st.session_state.selected_symbol = case['symbol']
                            st.session_state.selected_result = result
                            st.session_state.selected_df = case['df']

            # Category 2: RSI Only
            if results['rsi_only']:
                st.markdown("---")
                st.markdown("### ⚠️ RSI Breakout Only (Waiting for Price Confirmation)")

                for case in results['rsi_only']:
                    with st.expander(f"📊 {case['symbol']} - RSI Breakout"):
                        result = case['result']
                        rsi_bo = result['rsi_breakout']

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("RSI Breakout", rsi_bo.date.strftime('%Y-%m-%d'))
                            st.metric("RSI Value", f"{rsi_bo.rsi_value:.2f}")

                        with col2:
                            st.metric("Strength", rsi_bo.strength)
                            st.metric("Age (periods)", rsi_bo.age_in_periods)

                        with col3:
                            st.metric("Price Trendline", "✅" if result.get('has_price_trendline') else "❌")
                            st.metric("Price Breakout", "❌ Waiting")

                        if st.button(f"📊 View Chart for {case['symbol']}", key=f"chart_rsi_{case['symbol']}"):
                            st.session_state.selected_symbol = case['symbol']
                            st.session_state.selected_result = result
                            st.session_state.selected_df = case['df']

            # Category 3: Price Only
            if results['price_only']:
                st.markdown("---")
                st.markdown("### 📈 Price Breakout Only (No RSI Confirmation)")

                for case in results['price_only']:
                    with st.expander(f"📉 {case['symbol']} - Price Breakout"):
                        result = case['result']
                        price_bo = result['price_breakout']

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Price Breakout", price_bo.date.strftime('%Y-%m-%d'))
                            st.metric("Price", f"${price_bo.price_value:.2f}")

                        with col2:
                            st.metric("Type", price_bo.trendline_type.upper())
                            st.metric("Strength", price_bo.strength)

                        with col3:
                            st.metric("RSI Trendline", "✅" if result.get('has_rsi_trendline') else "❌")
                            st.metric("RSI Breakout", "❌ Not Yet")

                        if st.button(f"📊 View Chart for {case['symbol']}", key=f"chart_price_{case['symbol']}"):
                            st.session_state.selected_symbol = case['symbol']
                            st.session_state.selected_result = result
                            st.session_state.selected_df = case['df']

            # Category 4: Trendlines Only
            if results['trendlines_only']:
                st.markdown("---")
                with st.expander(f"⏳ Trendlines Only ({len(results['trendlines_only'])} symbols) - No Breakouts"):
                    for case in results['trendlines_only']:
                        st.write(f"- {case['symbol']}: RSI={case['result']['has_rsi_trendline']}, Price={case['result'].get('has_price_trendline', False)}")

        # Display selected chart from scan
        if 'selected_symbol' in st.session_state and 'selected_result' in st.session_state:
            st.markdown("---")
            st.markdown(f"### 📈 Detailed Chart: {st.session_state.selected_symbol}")

            # Use stored data
            result = st.session_state.selected_result
            df = st.session_state.selected_df
            symbol = st.session_state.selected_symbol
            analyzer = DualConfirmationAnalyzer(sync_window=6)

            # Create interactive chart for selected symbol
            if result is not None:
                # Extract data
                close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
                open_price = df['Open'].iloc[:, 0] if isinstance(df['Open'], pd.DataFrame) else df['Open']
                high = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
                low = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']

                # Create subplot
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=(f'{symbol} - Price Chart', 'RSI with Trendline'),
                    row_heights=[0.65, 0.35]
                )

                # Price candlestick
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=open_price,
                        high=high,
                        low=low,
                        close=close,
                        name='Price'
                    ),
                    row=1, col=1
                )

                # Price trendline if exists
                if result.get('has_price_trendline'):
                    price_tl = result['price_trendline']
                    from trendline_analysis.core.price_trendline_detector import PriceTrendlineDetector

                    price_detector = PriceTrendlineDetector()
                    # IMPORTANT: Arrêter l'oblique au dernier sommet, pas à la fin des données
                    last_peak_idx = price_tl.peak_indices[-1]
                    trendline_x = df.index[price_tl.start_idx:last_peak_idx + 1]
                    trendline_y = [price_detector.get_trendline_value(price_tl, i)
                                 for i in range(price_tl.start_idx, last_peak_idx + 1)]

                    fig.add_trace(
                        go.Scatter(
                            x=trendline_x,
                            y=trendline_y,
                            mode='lines',
                            name=f'Price {price_tl.trendline_type.title()}',
                            line=dict(color='red', width=3)  # Trait plein rouge
                        ),
                        row=1, col=1
                    )

                    # Price breakout marker
                    if result.get('has_price_breakout'):
                        price_bo = result['price_breakout']
                        fig.add_trace(
                            go.Scatter(
                                x=[df.index[price_bo.index]],
                                y=[price_bo.price_value],
                                mode='markers',
                                name='Price Breakout',
                                marker=dict(color='purple', size=15, symbol='star'),
                                showlegend=True
                            ),
                            row=1, col=1
                        )

                # RSI
                rsi = analyzer.rsi_detector.calculate_rsi(df)
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=rsi,
                        mode='lines',
                        name='RSI',
                        line=dict(color='blue', width=2)
                    ),
                    row=2, col=1
                )

                # RSI trendline (only if exists)
                if result.get('has_rsi_trendline'):
                    rsi_tl = result['rsi_trendline']
                    # IMPORTANT: Arrêter l'oblique RSI au dernier sommet RSI
                    last_rsi_peak_idx = rsi_tl.peak_indices[-1]
                    trendline_x = df.index[rsi_tl.start_idx:last_rsi_peak_idx + 1]
                    trendline_y = [analyzer.rsi_detector.get_trendline_value(rsi_tl, i)
                                 for i in range(rsi_tl.start_idx, last_rsi_peak_idx + 1)]

                    fig.add_trace(
                        go.Scatter(
                            x=trendline_x,
                            y=trendline_y,
                            mode='lines',
                            name='RSI Resistance',
                            line=dict(color='orange', width=3, dash='dash')
                        ),
                        row=2, col=1
                    )

                # RSI breakout marker
                if result['has_rsi_breakout']:
                    rsi_bo = result['rsi_breakout']
                    fig.add_trace(
                        go.Scatter(
                            x=[df.index[rsi_bo.index]],
                            y=[rsi_bo.rsi_value],
                            mode='markers',
                            name='RSI Breakout',
                            marker=dict(color='green', size=15, symbol='star'),
                            showlegend=True
                        ),
                        row=2, col=1
                    )

                # RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.3, row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.3, row=2, col=1)

                # Layout
                fig.update_layout(
                    height=900,
                    showlegend=True,
                    hovermode='x unified',
                    xaxis_rangeslider_visible=False
                )

                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="RSI", row=2, col=1)

                st.plotly_chart(fig, use_container_width=True)

    # Handle single symbol analysis from tab2
    if analyze_single:
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                # Download data
                interval = '1wk' if timeframe == 'weekly' else '1d'
                period_map = {104: '2y', 252: '1y', 500: '5y'}
                period = period_map.get(lookback, '1y')

                df = market_data_fetcher.get_historical_data(symbol, period=period, interval=interval)

                if df is None or len(df) < 50:
                    st.error(f"Could not fetch data for {symbol}")
                else:
                    # Analyze
                    analyzer = DualConfirmationAnalyzer(sync_window=6)
                    result = analyzer.analyze(df, lookback_periods=lookback)

                    if result is None:
                        st.warning(f"No RSI trendline detected for {symbol}")
                    else:
                        # Display results
                        st.markdown("---")

                        # Status cards
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            if result['has_rsi_breakout']:
                                st.success("✅ RSI Breakout")
                            else:
                                st.info("⏳ RSI Trendline Only")

                        with col2:
                            if result.get('has_price_trendline'):
                                st.success("✅ Price Trendline")
                            else:
                                st.warning("❌ No Price Trendline")

                        with col3:
                            if result.get('has_price_breakout'):
                                st.success("✅ Price Breakout")
                            else:
                                st.info("⏳ No Price Breakout")

                        with col4:
                            if result['has_dual_confirmation']:
                                st.success("🎯 DUAL CONFIRMED!")
                            else:
                                st.warning("⏳ Not Synchronized")

                        # Detailed metrics
                        st.markdown("### 📊 Analysis Details")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### RSI Analysis")
                            if result['has_rsi_breakout']:
                                rsi_bo = result['rsi_breakout']
                                st.metric("Breakout Date", rsi_bo.date.strftime('%Y-%m-%d'))
                                st.metric("RSI Value", f"{rsi_bo.rsi_value:.2f}")
                                st.metric("Strength", rsi_bo.strength)
                                st.metric("Age (days)", f"{rsi_bo.age_in_periods}")

                            if result.get('has_rsi_trendline'):
                                rsi_tl = result['rsi_trendline']
                                st.metric("Trendline Peaks", len(rsi_tl.peak_indices))
                                st.metric("R² Quality", f"{rsi_tl.r_squared:.3f}")
                            else:
                                st.info("No RSI trendline detected")

                        with col2:
                            st.markdown("#### Price Analysis")
                            if result.get('has_price_trendline'):
                                price_tl = result['price_trendline']
                                st.metric("Trendline Type", price_tl.trendline_type.upper())
                                st.metric("Peaks", len(price_tl.peak_indices))
                                st.metric("R² Quality", f"{price_tl.r_squared:.3f}")
                                st.metric("Quality Score", f"{price_tl.quality_score:.1f}/100")

                            if result.get('has_price_breakout'):
                                price_bo = result['price_breakout']
                                st.metric("Breakout Date", price_bo.date.strftime('%Y-%m-%d'))
                                st.metric("Price", f"${price_bo.price_value:.2f}")
                                st.metric("Strength", price_bo.strength)

                        # Dual confirmation details
                        if result['has_dual_confirmation']:
                            st.markdown("---")
                            st.markdown("### 🎯 Dual Confirmation Validated")
                            dual = result['dual_confirmation']

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Time Difference", f"{dual.time_difference} periods")
                            with col2:
                                st.metric("Confirmation Strength", dual.confirmation_strength)
                            with col3:
                                signal = analyzer.get_signal(result)
                                st.metric("Trading Signal", signal)

                        elif result.get('has_rsi_breakout') and result.get('has_price_breakout'):
                            rsi_idx = result['rsi_breakout'].index
                            price_idx = result['price_breakout'].index
                            time_diff = abs(rsi_idx - price_idx)
                            st.warning(f"⏳ Breakouts not synchronized: {time_diff} periods apart (max: 6)")

                        # Create interactive chart
                        st.markdown("---")
                        st.markdown("### 📈 Interactive Chart")

                        # Extract data
                        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
                        open_price = df['Open'].iloc[:, 0] if isinstance(df['Open'], pd.DataFrame) else df['Open']
                        high = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High']
                        low = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low']

                        # Create subplot
                        fig = make_subplots(
                            rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=(f'{symbol} - Price Chart', 'RSI with Trendline'),
                            row_heights=[0.65, 0.35]
                        )

                        # Price candlestick
                        fig.add_trace(
                            go.Candlestick(
                                x=df.index,
                                open=open_price,
                                high=high,
                                low=low,
                                close=close,
                                name='Price'
                            ),
                            row=1, col=1
                        )

                        # Price trendline if exists
                        if result.get('has_price_trendline'):
                            price_tl = result['price_trendline']
                            from trendline_analysis.core.price_trendline_detector import PriceTrendlineDetector

                            price_detector = PriceTrendlineDetector()
                            # IMPORTANT: Arrêter l'oblique au dernier sommet, pas à la fin des données
                            last_peak_idx = price_tl.peak_indices[-1]
                            trendline_x = df.index[price_tl.start_idx:last_peak_idx + 1]
                            trendline_y = [price_detector.get_trendline_value(price_tl, i)
                                         for i in range(price_tl.start_idx, last_peak_idx + 1)]

                            fig.add_trace(
                                go.Scatter(
                                    x=trendline_x,
                                    y=trendline_y,
                                    mode='lines',
                                    name=f'Price {price_tl.trendline_type.title()}',
                                    line=dict(color='red', width=3)  # Trait plein rouge
                                ),
                                row=1, col=1
                            )

                            # Price breakout marker
                            if result.get('has_price_breakout'):
                                price_bo = result['price_breakout']
                                fig.add_trace(
                                    go.Scatter(
                                        x=[df.index[price_bo.index]],
                                        y=[price_bo.price_value],
                                        mode='markers',
                                        name='Price Breakout',
                                        marker=dict(color='purple', size=15, symbol='star'),
                                        showlegend=True
                                    ),
                                    row=1, col=1
                                )

                        # RSI
                        rsi = analyzer.rsi_detector.calculate_rsi(df)
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=rsi,
                                mode='lines',
                                name='RSI',
                                line=dict(color='blue', width=2)
                            ),
                            row=2, col=1
                        )

                        # RSI trendline (only if exists)
                        if result.get('has_rsi_trendline'):
                            rsi_tl = result['rsi_trendline']
                            # IMPORTANT: Arrêter l'oblique RSI au dernier sommet RSI
                            last_rsi_peak_idx = rsi_tl.peak_indices[-1]
                            trendline_x = df.index[rsi_tl.start_idx:last_rsi_peak_idx + 1]
                            trendline_y = [analyzer.rsi_detector.get_trendline_value(rsi_tl, i)
                                         for i in range(rsi_tl.start_idx, last_rsi_peak_idx + 1)]

                            fig.add_trace(
                                go.Scatter(
                                    x=trendline_x,
                                    y=trendline_y,
                                    mode='lines',
                                    name='RSI Resistance',
                                    line=dict(color='orange', width=3, dash='dash')
                                ),
                                row=2, col=1
                            )

                        # RSI breakout marker
                        if result['has_rsi_breakout']:
                            rsi_bo = result['rsi_breakout']
                            fig.add_trace(
                                go.Scatter(
                                    x=[df.index[rsi_bo.index]],
                                    y=[rsi_bo.rsi_value],
                                    mode='markers',
                                    name='RSI Breakout',
                                    marker=dict(color='green', size=15, symbol='star'),
                                    showlegend=True
                                ),
                                row=2, col=1
                            )

                        # RSI levels
                        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.3, row=2, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.3, row=2, col=1)

                        # Layout
                        fig.update_layout(
                            height=900,
                            showlegend=True,
                            hovermode='x unified',
                            xaxis_rangeslider_visible=False
                        )

                        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                        fig.update_yaxes(title_text="RSI", row=2, col=1)

                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error analyzing {symbol}: {e}")
                import traceback
                st.code(traceback.format_exc())

elif page == "🧠 Intelligence":
    st.title("🧠 Market Intelligence")
    st.markdown("### Veille sectorielle et analyse de momentum")

    # Initialize sector analyzer
    if 'sector_analyzer' not in st.session_state:
        st.session_state.sector_analyzer = SectorAnalyzer(min_momentum_vs_spy=0.0)

    # Tabs for different features
    intel_tab1, intel_tab2, intel_tab3 = st.tabs(["📊 Sector Momentum", "📰 News Search", "📚 Research Papers"])

    with intel_tab1:
        st.subheader("Sector Momentum vs SPY")
        st.markdown("Analyse le momentum de chaque secteur par rapport au S&P 500")

        col1, col2 = st.columns([1, 1])

        with col1:
            min_vs_spy = st.slider(
                "Min. performance vs SPY (%)",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                help="Seuil minimum de surperformance par rapport au SPY"
            )

        with col2:
            if st.button("🔄 Refresh Sector Data", type="primary"):
                st.session_state.sector_data_loaded = False

        # Load sector data if needed
        if 'sector_data_loaded' not in st.session_state or not st.session_state.sector_data_loaded:
            with st.spinner("Loading sector ETF data..."):
                try:
                    st.session_state.sector_analyzer = SectorAnalyzer(min_momentum_vs_spy=min_vs_spy)
                    success = st.session_state.sector_analyzer.load_sector_data(market_data_fetcher, period='1y')
                    st.session_state.sector_data_loaded = success
                    if success:
                        st.success("Sector data loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading sector data: {e}")
                    st.session_state.sector_data_loaded = False

        # Display sector momentum if data is loaded
        if st.session_state.get('sector_data_loaded', False):
            try:
                momentum_data = st.session_state.sector_analyzer.calculate_sector_momentum()

                if momentum_data:
                    # Create dataframe for display
                    sector_df_data = []
                    for sector, sm in sorted(momentum_data.items(), key=lambda x: x[1].rank):
                        status = "BULLISH" if sm.is_bullish else "BEARISH"
                        status_color = "green" if sm.is_bullish else "red"
                        sector_df_data.append({
                            'Rank': sm.rank,
                            'Sector': sector,
                            'ETF': sm.etf_symbol,
                            'Perf 20d': f"{sm.perf_20d:+.1f}%",
                            'vs SPY': f"{sm.vs_spy:+.1f}%",
                            'Status': status
                        })

                    sector_df = pd.DataFrame(sector_df_data)

                    # Color code status
                    def color_status(val):
                        if val == 'BULLISH':
                            return 'background-color: #00C853; color: white'
                        elif val == 'BEARISH':
                            return 'background-color: #FF5252; color: white'
                        return ''

                    styled_sector_df = sector_df.style.applymap(color_status, subset=['Status'])
                    st.dataframe(styled_sector_df, use_container_width=True, height=500)

                    # Summary metrics
                    st.markdown("---")
                    bullish_count = sum(1 for sm in momentum_data.values() if sm.is_bullish)
                    bearish_count = len(momentum_data) - bullish_count

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Bullish Sectors", bullish_count)
                    with col2:
                        st.metric("Bearish Sectors", bearish_count)
                    with col3:
                        market_breadth = (bullish_count / len(momentum_data)) * 100 if momentum_data else 0
                        st.metric("Market Breadth", f"{market_breadth:.0f}%")

                    # Top sectors recommendation
                    st.markdown("---")
                    st.subheader("Recommended Sectors")
                    bullish_sectors = st.session_state.sector_analyzer.get_bullish_sectors()
                    if bullish_sectors:
                        for sm in bullish_sectors[:5]:
                            st.success(f"**{sm.sector}** ({sm.etf_symbol}): +{sm.perf_20d:.1f}% | vs SPY: +{sm.vs_spy:.1f}%")
                    else:
                        st.warning("No bullish sectors detected with current threshold")

                else:
                    st.info("No sector data available. Click 'Refresh Sector Data' to load.")

            except Exception as e:
                st.error(f"Error calculating momentum: {e}")

    with intel_tab2:
        st.subheader("Financial News Search")
        st.markdown("Search for news from various sources (Reddit, NewsAPI, Alpha Vantage)")

        col1, col2 = st.columns([3, 1])

        with col1:
            news_query = st.text_input(
                "Search keywords",
                placeholder="e.g., AI semiconductor, Tesla earnings, Fed rate",
                key="news_query"
            )

        with col2:
            news_source = st.selectbox(
                "Source",
                ["Reddit", "All Sources"],
                key="news_source"
            )

        if st.button("🔍 Search News", type="primary"):
            if news_query:
                with st.spinner(f"Searching for '{news_query}'..."):
                    import asyncio

                    async def fetch_news():
                        fetcher = get_news_fetcher()
                        try:
                            if news_source == "Reddit":
                                posts = await fetcher.fetch_reddit_posts(
                                    query=news_query,
                                    limit=20,
                                    time_filter='week'
                                )
                                return posts
                            else:
                                # Fetch from multiple sources
                                posts = await fetcher.fetch_reddit_posts(query=news_query, limit=10)
                                return posts
                        finally:
                            await fetcher.close()

                    try:
                        # Run async function
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        articles = loop.run_until_complete(fetch_news())
                        loop.close()

                        if articles:
                            st.success(f"Found {len(articles)} results")
                            for article in articles[:15]:
                                with st.expander(f"📰 {article.title[:80]}..."):
                                    st.markdown(f"**Source:** {article.source}")
                                    st.markdown(f"**Date:** {article.published_at.strftime('%Y-%m-%d %H:%M')}")
                                    if article.content:
                                        st.markdown(article.content[:500] + "..." if len(article.content) > 500 else article.content)
                                    st.markdown(f"[Read more]({article.url})")
                                    st.markdown(f"**Relevance:** {article.relevance_score:.2f}")
                        else:
                            st.info("No results found. Try different keywords.")

                    except Exception as e:
                        st.error(f"Error fetching news: {e}")
            else:
                st.warning("Please enter search keywords")

    with intel_tab3:
        st.subheader("Research Papers (Arxiv)")
        st.markdown("Search for academic publications on AI, Finance, and Technology")

        col1, col2 = st.columns([3, 1])

        with col1:
            arxiv_query = st.text_input(
                "Search query",
                placeholder="e.g., machine learning stock prediction, LLM finance",
                key="arxiv_query"
            )

        with col2:
            arxiv_categories = st.multiselect(
                "Categories",
                ["cs.AI", "cs.LG", "q-fin.ST", "q-fin.TR", "cs.CL"],
                default=["cs.AI", "cs.LG"],
                key="arxiv_categories"
            )

        if st.button("🔍 Search Papers", type="primary"):
            if arxiv_query:
                with st.spinner(f"Searching Arxiv for '{arxiv_query}'..."):
                    import asyncio

                    async def fetch_papers():
                        fetcher = get_news_fetcher()
                        try:
                            papers = await fetcher.fetch_arxiv_papers(
                                query=arxiv_query,
                                categories=arxiv_categories if arxiv_categories else None,
                                max_results=20
                            )
                            return papers
                        finally:
                            await fetcher.close()

                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        papers = loop.run_until_complete(fetch_papers())
                        loop.close()

                        if papers:
                            st.success(f"Found {len(papers)} papers")
                            for paper in papers:
                                with st.expander(f"📄 {paper.title[:80]}..."):
                                    st.markdown(f"**Authors:** {', '.join(paper.authors[:5])}")
                                    st.markdown(f"**Date:** {paper.published_at.strftime('%Y-%m-%d')}")
                                    st.markdown(f"**Categories:** {', '.join(paper.categories)}")
                                    st.markdown("**Abstract:**")
                                    st.markdown(paper.abstract[:800] + "..." if len(paper.abstract) > 800 else paper.abstract)
                                    if paper.url:
                                        st.markdown(f"[Read PDF]({paper.url})")
                        else:
                            st.info("No papers found. Try different keywords.")

                    except Exception as e:
                        st.error(f"Error fetching papers: {e}")
            else:
                st.warning("Please enter a search query")

    # API Configuration hint
    st.markdown("---")
    with st.expander("🔧 API Configuration"):
        st.markdown("""
        **Optional API keys for enhanced functionality:**

        - `NEWSAPI_KEY`: For NewsAPI access (newsapi.org)
        - `ALPHAVANTAGE_KEY`: For Alpha Vantage news sentiment
        - `OPENROUTER_API_KEY`: For LLM-powered analysis

        Set these in your `.env` file or environment variables.
        """)

elif page == "🔮 Trend Discovery":
    st.title("🔮 Trend Discovery")
    st.markdown("### Découverte automatique de tendances par IA")

    # Helper function to load latest report
    def load_latest_trend_report():
        """Charge le dernier rapport de tendances."""
        report_dir = settings.TREND_DATA_DIR
        if not os.path.exists(report_dir):
            return None

        # Find most recent report
        report_files = glob_module.glob(os.path.join(report_dir, 'trend_report_*.json'))
        if not report_files:
            return None

        latest_file = max(report_files, key=os.path.getmtime)
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['_file_path'] = latest_file
                data['_file_date'] = datetime.fromtimestamp(os.path.getmtime(latest_file))
                return data
        except Exception as e:
            st.error(f"Erreur de chargement: {e}")
            return None

    def load_learned_themes():
        """Charge les thèmes appris par l'IA."""
        path = os.path.join(settings.TREND_DATA_DIR, 'learned_themes.json')
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    # Load data
    report_data = load_latest_trend_report()
    learned_themes = load_learned_themes()

    # Tabs
    td_tab1, td_tab2, td_tab3, td_tab4 = st.tabs([
        "📊 Overview", "🎯 Tendances", "🧠 Thèmes IA", "📈 Sources & Stats"
    ])

    # ============================================
    # TAB 1: Overview
    # ============================================
    with td_tab1:
        if report_data:
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)

            trends = report_data.get('trends', [])
            sentiment = report_data.get('market_sentiment', 0)
            narratives = report_data.get('narrative_updates', [])

            with col1:
                st.metric("📈 Tendances", len(trends))
            with col2:
                sentiment_color = "🟢" if sentiment > 0.2 else "🟡" if sentiment > -0.2 else "🔴"
                st.metric("💹 Sentiment", f"{sentiment_color} {sentiment:+.2f}")
            with col3:
                st.metric("🆕 Nouveaux Narratifs", len(narratives))
            with col4:
                st.metric("🧠 Thèmes Appris", len(learned_themes))

            # Last scan info
            file_date = report_data.get('_file_date', datetime.now())
            st.info(f"📅 Dernier scan: **{file_date.strftime('%d/%m/%Y %H:%M')}**")

            # Run scan button
            col_btn1, col_btn2 = st.columns([1, 4])
            with col_btn1:
                if st.button("🚀 Lancer un Scan", type="primary", key="run_trend_scan"):
                    with st.spinner("Analyse en cours... (peut prendre 1-2 minutes)"):
                        try:
                            # Run async scan
                            async def run_scan():
                                discovery = TrendDiscovery(
                                    openrouter_api_key=settings.OPENROUTER_API_KEY,
                                    model=settings.OPENROUTER_MODEL,
                                    data_dir=settings.TREND_DATA_DIR
                                )
                                await discovery.initialize()
                                report = await discovery.daily_scan()
                                await discovery.close()
                                return report

                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            report = loop.run_until_complete(run_scan())
                            loop.close()

                            st.success(f"✅ Scan terminé! {len(report.trends)} tendances détectées.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erreur lors du scan: {e}")

            st.markdown("---")
            st.subheader("🔥 Top Tendances")

            # Create dataframe for trends
            if trends:
                trend_data = []
                for t in sorted(trends, key=lambda x: x.get('confidence', 0), reverse=True)[:10]:
                    trend_data.append({
                        'Nom': t.get('name', 'N/A'),
                        'Type': t.get('type', 'N/A').replace('_', ' ').title(),
                        'Force': t.get('strength', 'N/A').title(),
                        'Confiance': f"{t.get('confidence', 0)*100:.0f}%",
                        'Symboles': ', '.join(t.get('symbols', [])[:5])
                    })

                df_trends = pd.DataFrame(trend_data)

                # Style the dataframe
                def color_strength(val):
                    colors = {
                        'Established': 'background-color: #00C853; color: white',
                        'Developing': 'background-color: #FDD835; color: black',
                        'Emerging': 'background-color: #2196F3; color: white'
                    }
                    return colors.get(val, '')

                st.dataframe(
                    df_trends.style.applymap(color_strength, subset=['Force']),
                    use_container_width=True,
                    height=400
                )
            else:
                st.warning("Aucune tendance dans le rapport.")
        else:
            st.warning("⚠️ Aucun rapport de tendances trouvé. Lancez un scan pour commencer.")
            if st.button("🚀 Lancer le premier scan", type="primary"):
                st.info("Veuillez utiliser la commande: `python scripts/run_trend_discovery.py`")

    # ============================================
    # TAB 2: Tendances Détaillées
    # ============================================
    with td_tab2:
        if report_data:
            trends = report_data.get('trends', [])

            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                type_filter = st.selectbox(
                    "Type",
                    ["Tous", "Sector Momentum", "Thematic"],
                    key="td_type_filter"
                )
            with col2:
                strength_filter = st.selectbox(
                    "Force",
                    ["Tous", "Established", "Developing", "Emerging"],
                    key="td_strength_filter"
                )
            with col3:
                min_confidence = st.slider(
                    "Confiance min (%)",
                    0, 100, 40,
                    key="td_conf_filter"
                )

            # Filter trends
            filtered = trends
            if type_filter != "Tous":
                type_key = type_filter.lower().replace(' ', '_')
                filtered = [t for t in filtered if t.get('type', '') == type_key]
            if strength_filter != "Tous":
                filtered = [t for t in filtered if t.get('strength', '').lower() == strength_filter.lower()]
            filtered = [t for t in filtered if t.get('confidence', 0) * 100 >= min_confidence]

            st.markdown(f"**{len(filtered)} tendances** correspondent aux filtres")
            st.markdown("---")

            # Display each trend as expander
            for trend in sorted(filtered, key=lambda x: x.get('confidence', 0), reverse=True):
                conf = trend.get('confidence', 0) * 100
                strength = trend.get('strength', 'emerging')
                icon = "🔥" if strength == "established" else "📈" if strength == "developing" else "🌱"

                with st.expander(f"{icon} **{trend.get('name', 'N/A')}** ({conf:.0f}%)"):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"**Description:** {trend.get('description', 'N/A')}")

                        catalysts = trend.get('key_catalysts', [])
                        if catalysts:
                            st.markdown("**Catalyseurs:**")
                            for cat in catalysts:
                                st.markdown(f"  • {cat}")

                    with col2:
                        st.markdown(f"**Type:** {trend.get('type', 'N/A').replace('_', ' ').title()}")
                        st.markdown(f"**Force:** {strength.title()}")
                        st.markdown(f"**Momentum:** {trend.get('momentum_score', 0):.2%}")

                        sources = trend.get('sources', [])
                        if sources:
                            st.markdown(f"**Sources:** {', '.join(sources)}")

                    # Symbols
                    symbols = trend.get('symbols', [])
                    if symbols:
                        st.markdown("**Symboles:**")
                        st.code(', '.join(symbols))
        else:
            st.warning("Aucun rapport disponible.")

    # ============================================
    # TAB 3: Thèmes IA
    # ============================================
    with td_tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📚 Thèmes Prédéfinis")
            st.caption(f"{len(TrendDiscovery.THEMES_KEYWORDS)} thèmes de base")

            for theme_name, keywords in TrendDiscovery.THEMES_KEYWORDS.items():
                with st.container():
                    st.info(f"**{theme_name}**\n\n_{', '.join(keywords[:4])}..._")

        with col2:
            st.subheader("🆕 Thèmes Découverts par IA")
            st.caption(f"{len(learned_themes)} thèmes appris")

            if learned_themes:
                for theme_name, data in learned_themes.items():
                    occ = data.get('occurrence_count', 1)
                    discovered = data.get('discovered_at', '')[:10]
                    desc = data.get('description', '')[:100]
                    symbols = data.get('symbols', [])

                    with st.container():
                        st.success(f"""**{data.get('name', theme_name)}** (vu {occ}x)

📅 Découvert: {discovered}

{desc}...

🎯 Symboles: {', '.join(symbols[:5]) if symbols else 'N/A'}""")
            else:
                st.info("Aucun thème découvert. Lancez un scan pour que l'IA identifie de nouveaux narratifs.")

    # ============================================
    # TAB 4: Sources & Stats
    # ============================================
    with td_tab4:
        st.subheader("📡 Sources de Données")

        # Check API keys status
        sources_data = [
            {
                'Source': 'yfinance',
                'Type': 'Prix & Volume',
                'Statut': '✅ Active',
                'Config': 'N/A'
            },
            {
                'Source': 'NewsAPI',
                'Type': 'Actualités',
                'Statut': '✅ Active' if settings.NEWSAPI_KEY else '⚠️ Non configuré',
                'Config': 'NEWSAPI_KEY'
            },
            {
                'Source': 'OpenRouter (LLM)',
                'Type': 'Analyse IA',
                'Statut': '✅ Active' if settings.OPENROUTER_API_KEY else '⚠️ Non configuré',
                'Config': 'OPENROUTER_API_KEY'
            },
            {
                'Source': 'AlphaVantage',
                'Type': 'News Sentiment',
                'Statut': '✅ Active' if getattr(settings, 'ALPHAVANTAGE_KEY', '') else '⚠️ Non configuré',
                'Config': 'ALPHAVANTAGE_KEY'
            }
        ]

        df_sources = pd.DataFrame(sources_data)
        st.dataframe(df_sources, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📜 Historique des Scans")

        # List all report files
        report_dir = settings.TREND_DATA_DIR
        if os.path.exists(report_dir):
            report_files = glob_module.glob(os.path.join(report_dir, 'trend_report_*.json'))
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

                        st.markdown(f"""
**{file_date.strftime('%d/%m/%Y %H:%M')}** - {n_trends} tendances, {n_narratives} nouveaux narratifs (sentiment: {sentiment:+.2f})
""")
                    except:
                        st.markdown(f"**{file_date.strftime('%d/%m/%Y %H:%M')}** - Erreur de lecture")
            else:
                st.info("Aucun historique de scan disponible.")
        else:
            st.warning(f"Répertoire {report_dir} non trouvé.")

        st.markdown("---")
        st.subheader("🎯 Symboles Focus")

        # Load focus symbols
        focus_path = os.path.join(settings.TREND_DATA_DIR, 'focus_symbols.json')
        if os.path.exists(focus_path):
            try:
                with open(focus_path, 'r', encoding='utf-8') as f:
                    focus_data = json.load(f)
                symbols = focus_data.get('symbols', [])
                if symbols:
                    st.code(', '.join(symbols))

                    # Download button
                    csv_content = "Symbol\n" + "\n".join(symbols)
                    st.download_button(
                        "📥 Télécharger CSV",
                        data=csv_content,
                        file_name="focus_symbols.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Erreur: {e}")
        else:
            st.info("Aucun symbole focus. Lancez un scan pour générer la liste.")

elif page == "🚨 Alerts History":
    st.title("🚨 Alerts History")
    st.markdown("View and analyze historical alerts")

    col1, col2 = st.columns([1, 3])

    with col1:
        days_back = st.slider("Days to look back", 1, 90, 30, key="alerts_days")

    with col2:
        timeframe_filter = st.multiselect(
            "Filter by timeframe",
            ["daily", "weekly"],
            default=["daily", "weekly"],
            key="alerts_timeframe"
        )

    try:
        alerts = db_manager.get_recent_alerts(days=days_back)

        # Filter by timeframe
        if timeframe_filter:
            alerts = [a for a in alerts if a.timeframe in timeframe_filter]

        if alerts:
            st.info(f"Found {len(alerts)} alerts in the last {days_back} days")

            # Create detailed dataframe
            alert_data = []
            for alert in alerts:
                alert_data.append({
                    'Date': alert.alert_date.strftime('%Y-%m-%d %H:%M'),
                    'Symbol': alert.symbol,
                    'Company': alert.company_name,
                    'Timeframe': alert.timeframe.upper(),
                    'Price': alert.current_price,
                    'Support': alert.support_level,
                    'Distance %': alert.distance_to_support_pct,
                    'EMA 24': alert.ema_24,
                    'EMA 38': alert.ema_38,
                    'EMA 62': alert.ema_62,
                    'Recommendation': alert.recommendation,
                    'Notified': '✅' if alert.is_notified else '❌'
                })

            df = pd.DataFrame(alert_data)

            # Display with formatting
            st.dataframe(
                df.style.format({
                    'Price': '${:.2f}',
                    'Support': '${:.2f}',
                    'Distance %': '{:.2f}%',
                    'EMA 24': '${:.2f}',
                    'EMA 38': '${:.2f}',
                    'EMA 62': '${:.2f}'
                }),
                use_container_width=True,
                height=400
            )

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"alerts_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

            # Statistics
            st.markdown("---")
            st.subheader("📊 Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                strong_buys = sum(1 for a in alerts if a.recommendation == 'STRONG_BUY')
                st.metric("Strong Buys", strong_buys)

            with col2:
                buys = sum(1 for a in alerts if a.recommendation == 'BUY')
                st.metric("Buys", buys)

            with col3:
                weekly_signals = sum(1 for a in alerts if a.timeframe == 'weekly')
                st.metric("Weekly Signals", weekly_signals)

            with col4:
                daily_signals = sum(1 for a in alerts if a.timeframe == 'daily')
                st.metric("Daily Signals", daily_signals)

        else:
            st.info("No alerts found for the selected criteria.")

    except Exception as e:
        st.error(f"Error loading alerts: {e}")

elif page == "⚙️ Settings":
    st.title("⚙️ Settings & Configuration")

    # Create tabs for different settings sections
    settings_tab1, settings_tab2, settings_tab3, settings_tab4 = st.tabs([
        "📊 Screening", "🔔 Notifications", "📧 Email/Telegram", "🛠️ System"
    ])

    with settings_tab1:
        st.markdown("### Screening Parameters")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### EMA Settings")
            st.code(f"""
EMA Periods: {EMA_PERIODS}
Support Zone Tolerance: ±{ZONE_TOLERANCE}%
            """)

        with col2:
            st.markdown("#### Market Filters")
            from config.settings import (
                MIN_MARKET_CAP_NASDAQ,
                MIN_MARKET_CAP_SP500,
                MIN_MARKET_CAP_EUROPE,
                MIN_DAILY_VOLUME
            )
            st.code(f"""
NASDAQ Min Cap: ${MIN_MARKET_CAP_NASDAQ}M
SP500 Min Cap: ${MIN_MARKET_CAP_SP500}M
Europe Min Cap: ${MIN_MARKET_CAP_EUROPE}M
Min Daily Volume: ${MIN_DAILY_VOLUME/1e3:.0f}k
            """)

        st.markdown("#### Scheduling")
        from config.settings import DAILY_REPORT_TIME, TIMEZONE
        st.code(f"""
Daily Report Time: {DAILY_REPORT_TIME.strftime('%H:%M')}
Timezone: {TIMEZONE}
        """)

    with settings_tab2:
        st.markdown("### Notification Preferences")

        # Get current config from notification manager
        current_config = notification_manager.config

        # Notification channels enable/disable
        st.markdown("#### Enabled Channels")
        col1, col2, col3 = st.columns(3)

        with col1:
            telegram_enabled = st.checkbox(
                "📱 Telegram",
                value=current_config.telegram_enabled,
                key="notif_telegram_enabled"
            )
        with col2:
            email_enabled = st.checkbox(
                "📧 Email",
                value=current_config.email_enabled,
                key="notif_email_enabled"
            )
        with col3:
            in_app_enabled = st.checkbox(
                "🔔 In-App",
                value=current_config.in_app_enabled,
                key="notif_in_app_enabled"
            )

        st.markdown("---")
        st.markdown("#### Alert Type Filters")
        st.caption("Only receive notifications for selected alert types")

        alert_types_options = ["STRONG_BUY", "BUY", "WATCH", "OBSERVE"]
        selected_alert_types = st.multiselect(
            "Notify for these alerts:",
            options=alert_types_options,
            default=current_config.alert_types if current_config.alert_types else ["STRONG_BUY", "BUY"],
            key="notif_alert_types"
        )

        st.markdown("---")
        st.markdown("#### Priority Filter")

        priority_options = {"low": "Low (all)", "medium": "Medium+", "high": "High+", "urgent": "Urgent only"}
        min_priority = st.select_slider(
            "Minimum priority level:",
            options=list(priority_options.keys()),
            value=current_config.min_priority,
            format_func=lambda x: priority_options[x],
            key="notif_min_priority"
        )

        st.markdown("---")
        st.markdown("#### Quiet Hours")
        st.caption("No notifications during these hours (except urgent)")

        quiet_col1, quiet_col2 = st.columns(2)
        with quiet_col1:
            quiet_start = st.selectbox(
                "Start (hour):",
                options=[None] + list(range(24)),
                index=0 if current_config.quiet_hours_start is None else current_config.quiet_hours_start + 1,
                format_func=lambda x: "Disabled" if x is None else f"{x:02d}:00",
                key="notif_quiet_start"
            )
        with quiet_col2:
            quiet_end = st.selectbox(
                "End (hour):",
                options=[None] + list(range(24)),
                index=0 if current_config.quiet_hours_end is None else current_config.quiet_hours_end + 1,
                format_func=lambda x: "Disabled" if x is None else f"{x:02d}:00",
                key="notif_quiet_end"
            )

        # Save button
        if st.button("💾 Save Notification Settings", key="save_notif_settings"):
            notification_manager.update_config(
                telegram_enabled=telegram_enabled,
                email_enabled=email_enabled,
                in_app_enabled=in_app_enabled,
                alert_types=selected_alert_types,
                min_priority=min_priority,
                quiet_hours_start=quiet_start,
                quiet_hours_end=quiet_end
            )
            st.success("✅ Notification settings saved!")

        # Show notification stats
        st.markdown("---")
        st.markdown("#### Notification Statistics")
        stats = notification_manager.get_notification_stats()

        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        with stats_col1:
            st.metric("Total Sent", stats['sent'])
        with stats_col2:
            st.metric("Failed", stats['failed'])
        with stats_col3:
            tg_stats = stats['by_channel'].get('telegram', {})
            st.metric("Telegram", tg_stats.get('sent', 0))
        with stats_col4:
            email_stats = stats['by_channel'].get('email', {})
            st.metric("Email", email_stats.get('sent', 0))

        # Recent notifications
        recent_notifs = notification_manager.get_recent_notifications(limit=10)
        if recent_notifs:
            st.markdown("#### Recent Notifications")
            for notif in recent_notifs[:5]:
                status = "✅" if notif['sent'] else "❌"
                channel_emoji = {"telegram": "📱", "email": "📧", "in_app": "🔔"}.get(notif['channel'], "📊")
                st.text(f"{status} {channel_emoji} {notif['title']} ({notif['timestamp'][:16]})")

    with settings_tab3:
        st.markdown("### Telegram Configuration")

        from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

        telegram_configured = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

        if telegram_configured:
            st.success("✅ Telegram configured via environment variables")
            st.code(f"""
Bot Token: {TELEGRAM_BOT_TOKEN[:10]}...{TELEGRAM_BOT_TOKEN[-5:] if len(TELEGRAM_BOT_TOKEN) > 15 else ''}
Chat ID: {TELEGRAM_CHAT_ID}
            """)
        else:
            st.warning("⚠️ Telegram not configured")
            st.info("""
To configure Telegram:
1. Create a bot with @BotFather on Telegram
2. Get your chat ID by messaging @userinfobot
3. Add to .env file:
   - TELEGRAM_BOT_TOKEN=your_bot_token
   - TELEGRAM_CHAT_ID=your_chat_id
            """)

        if st.button("🧪 Test Telegram", key="test_telegram", disabled=not telegram_configured):
            with st.spinner("Sending test message..."):
                success = notification_manager.test_telegram()
                if success:
                    st.success("✅ Telegram test successful!")
                else:
                    st.error("❌ Telegram test failed. Check your configuration.")

        st.markdown("---")
        st.markdown("### Email Configuration")

        email_configured = bool(notification_manager.config.smtp_username)

        if email_configured:
            st.success("✅ Email configured")
            st.code(f"""
SMTP Server: {notification_manager.config.smtp_server}:{notification_manager.config.smtp_port}
From: {notification_manager.config.email_from}
To: {', '.join(notification_manager.config.email_to)}
            """)
        else:
            st.warning("⚠️ Email not configured")
            st.info("""
To configure Email:
Add to .env file:
- SMTP_SERVER=smtp.gmail.com
- SMTP_PORT=587
- SMTP_USERNAME=your_email@gmail.com
- SMTP_PASSWORD=your_app_password
- EMAIL_FROM=your_email@gmail.com
- EMAIL_TO=recipient@email.com
            """)

        if st.button("🧪 Test Email", key="test_email", disabled=not email_configured):
            with st.spinner("Sending test email..."):
                success = notification_manager.test_email()
                if success:
                    st.success("✅ Email test successful!")
                else:
                    st.error("❌ Email test failed. Check your configuration.")

    with settings_tab4:
        st.markdown("### System Information")

        col1, col2 = st.columns(2)

        with col1:
            import platform
            st.markdown("#### Environment")
            st.code(f"""
Python: {platform.python_version()}
OS: {platform.system()} {platform.release()}
            """)

        with col2:
            st.markdown("#### Database")
            st.info("Location: data/screener.db")
            try:
                recent = db_manager.get_recent_alerts(days=1)
                st.metric("Alerts (24h)", len(recent))
            except:
                st.error("Could not access database")

        st.markdown("---")
        st.markdown("### Quick Actions")

        action_col1, action_col2, action_col3 = st.columns(3)

        with action_col1:
            if st.button("🔄 Refresh Config", key="refresh_config"):
                st.cache_data.clear()
                st.success("Configuration refreshed!")
                st.rerun()

        with action_col2:
            if st.button("🗑️ Clear Cache", key="clear_cache"):
                st.cache_data.clear()
                st.success("Cache cleared!")

        with action_col3:
            if st.button("📋 Export Config", key="export_config"):
                config_data = notification_manager.config.to_dict()
                # Remove sensitive data
                config_data['telegram_bot_token'] = '***'
                config_data['smtp_password'] = '***'
                st.json(config_data)

# Footer
st.markdown("---")
st.caption("Market Screener Dashboard | Powered by Streamlit & Plotly")
