"""
📊 Chart Analyzer page for Market Screener Dashboard
"""
import sys
import os

# Ensure src is in path
_src_path = os.path.join(os.path.dirname(__file__), '../../..')
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Import all shared dependencies
from src.dashboard.shared_imports import *


def render():
    """Render the 📊 Chart Analyzer page"""
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

