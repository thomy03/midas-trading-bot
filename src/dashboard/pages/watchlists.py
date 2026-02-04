"""
📋 Watchlists page for Market Screener Dashboard
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
    """Render the 📋 Watchlists page"""
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

