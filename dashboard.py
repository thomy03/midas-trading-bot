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
        ["🏠 Home", "📊 Chart Analyzer", "🔍 Screening", "🚨 Alerts History", "⚙️ Settings"]
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

                            st.json(alert)

                            # Show chart
                            st.markdown("---")
                            fig = chart_visualizer.create_chart(
                                symbol=single_symbol,
                                timeframe=alert['timeframe'],
                                period='1y',
                                show_volume=True
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No buy signal detected for {single_symbol}")
                    else:
                        st.error(f"Could not fetch data for {single_symbol}")
                except Exception as e:
                    st.error(f"Error screening {single_symbol}: {e}")

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
