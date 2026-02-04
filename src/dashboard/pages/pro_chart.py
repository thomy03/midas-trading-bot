"""
📈 Pro Chart page for Market Screener Dashboard
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
    """Render the 📈 Pro Chart page"""
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

