"""
🎯 Trendline Analysis page for Market Screener Dashboard
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
    """Render the 🎯 Trendline Analysis page"""
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

