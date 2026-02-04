"""
🔍 Screening page for Market Screener Dashboard
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
    """Render the 🔍 Screening page"""
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

