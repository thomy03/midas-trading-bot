"""
Market Screener Dashboard - TradingView-like Interface

Run with: streamlit run dashboard.py
"""
import streamlit as st
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from src.utils.visualizer import chart_visualizer
from src.data.market_data import market_data_fetcher
from src.screening.screener import market_screener
from src.database.db_manager import db_manager
from src.indicators.ema_analyzer import ema_analyzer
from config.settings import EMA_PERIODS, ZONE_TOLERANCE

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
        ["🏠 Home", "📊 Chart Analyzer", "📈 Signaux Historiques", "🔍 Screening", "🎯 Trendline Analysis", "🚨 Alerts History", "⚙️ Settings"]
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

elif page == "📊 Chart Analyzer":
    st.title("📊 Interactive Chart Analyzer")
    st.markdown("Visualize stocks with EMAs and support zones")

    col1, col2 = st.columns([2, 1])

    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", key="chart_symbol").upper()

    with col2:
        timeframe = st.selectbox("Timeframe", ["daily", "weekly"], key="chart_timeframe")

    col3, col4, col5 = st.columns(3)

    with col3:
        period = st.selectbox("Period", ["6mo", "1y", "2y", "5y", "10y", "max"], index=3, key="chart_period")

    with col4:
        show_volume = st.checkbox("Show Volume", value=True, key="chart_volume")

    with col5:
        if st.button("🔄 Refresh Chart", key="refresh_chart"):
            st.rerun()

    if symbol:
        with st.spinner(f"Loading chart for {symbol}..."):
            fig = chart_visualizer.create_chart(
                symbol=symbol,
                timeframe=timeframe,
                period=period,
                show_volume=show_volume
            )

            if fig:
                st.plotly_chart(fig, use_container_width=True)

                # Show analysis details
                st.markdown("---")
                st.subheader("📋 Analysis Details")

                col1, col2, col3 = st.columns(3)

                # Get current data
                interval = '1wk' if timeframe == 'weekly' else '1d'
                df = market_data_fetcher.get_historical_data(symbol, period=period, interval=interval)

                if df is not None:
                    df = ema_analyzer.calculate_emas(df)
                    current_price = df['Close'].iloc[-1]
                    is_aligned, alignment_desc = ema_analyzer.check_ema_alignment(df, for_buy=True)
                    crossovers = ema_analyzer.detect_crossovers(df, timeframe)
                    support_zones = ema_analyzer.find_support_zones(df, crossovers, current_price)

                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                        st.metric("EMA 24", f"${df['EMA_24'].iloc[-1]:.2f}")
                        st.metric("EMA 38", f"${df['EMA_38'].iloc[-1]:.2f}")
                        st.metric("EMA 62", f"${df['EMA_62'].iloc[-1]:.2f}")

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

    tab1, tab2 = st.tabs(["Single Symbol", "Multiple Symbols"])

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

    st.markdown("### Current Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Screening Parameters")
        st.code(f"""
EMA Periods: {EMA_PERIODS}
Support Zone Tolerance: ±{ZONE_TOLERANCE}%
        """)

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

    with col2:
        st.markdown("#### Notification Settings")
        from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

        telegram_configured = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

        if telegram_configured:
            st.success("✅ Telegram configured")
        else:
            st.warning("⚠️ Telegram not configured")
            st.info("Configure in .env file:\n- TELEGRAM_BOT_TOKEN\n- TELEGRAM_CHAT_ID")

        st.markdown("#### Scheduling")
        from config.settings import DAILY_REPORT_TIME, TIMEZONE
        st.code(f"""
Daily Report Time: {DAILY_REPORT_TIME.strftime('%H:%M')}
Timezone: {TIMEZONE}
        """)

    st.markdown("---")
    st.markdown("### Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🧪 Test Notifications", key="test_notif"):
            from src.notifications.notifier import notifier
            with st.spinner("Testing notifications..."):
                results = notifier.test_notifications()
                if results.get('telegram'):
                    st.success("✅ Telegram test successful!")
                elif results.get('telegram') is False:
                    st.error("❌ Telegram test failed")
                else:
                    st.info("Telegram not enabled")

    with col2:
        if st.button("🗄️ View Database", key="view_db"):
            st.info("Database location: data/screener.db")

            # Show table info
            try:
                recent = db_manager.get_recent_alerts(days=1)
                st.metric("Alerts (24h)", len(recent))
            except:
                st.error("Could not access database")

    with col3:
        if st.button("📊 System Info", key="sys_info"):
            import platform
            st.code(f"""
Python: {platform.python_version()}
OS: {platform.system()} {platform.release()}
            """)

# Footer
st.markdown("---")
st.caption("Market Screener Dashboard | Powered by Streamlit & Plotly")
