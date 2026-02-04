"""
💼 Portfolio page for Market Screener Dashboard
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
    """Render the 💼 Portfolio page"""
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

