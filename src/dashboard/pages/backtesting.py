"""
🔬 Backtesting page for Market Screener Dashboard
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
    """Render the 🔬 Backtesting page"""
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

