"""
📊 Performance page for Market Screener Dashboard
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
    """Render the 📊 Performance page"""
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

