"""
🗺️ Sector Map page for Market Screener Dashboard
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
    """Render the 🗺️ Sector Map page"""
    st.title("🗺️ Sector Heatmap")
    st.markdown("Market overview by sector with performance and rotation analysis")

    # Controls
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        metric = st.selectbox(
            "Performance Period",
            ["1D %", "1W %", "1M %", "YTD %"],
            key="sector_metric"
        )

    with col2:
        view_type = st.selectbox(
            "View Type",
            ["Treemap", "Bar Chart", "Rotation"],
            key="sector_view"
        )

    with col3:
        if st.button("🔄 Refresh Data", key="refresh_sectors"):
            st.cache_data.clear()

    st.markdown("---")

    # Fetch sector data
    with st.spinner("Loading sector data..."):
        try:
            sector_data = sector_heatmap_builder.get_sector_performance(use_cache=True)
            breadth = sector_heatmap_builder.get_market_breadth()
            rotation = sector_heatmap_builder.get_sector_rotation_signal()
        except Exception as e:
            st.error(f"Error loading sector data: {e}")
            sector_data = []
            breadth = {'advancing': 0, 'declining': 0, 'total': 0}
            rotation = {'signal': 'NEUTRAL', 'reason': 'No data'}

    if sector_data:
        # Market overview row
        st.markdown("### 📊 Market Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_perf = breadth.get('avg_performance', 0)
            st.metric(
                "Avg Performance",
                f"{avg_perf:+.2f}%",
                delta=f"{breadth['advancing']} advancing"
            )

        with col2:
            ratio = breadth.get('breadth_ratio', 1)
            st.metric(
                "Breadth Ratio",
                f"{ratio:.2f}",
                delta="Bullish" if ratio > 1 else "Bearish"
            )

        with col3:
            signal = rotation.get('signal', 'NEUTRAL')
            signal_color = {"RISK_ON": "green", "RISK_OFF": "red", "NEUTRAL": "gray"}
            st.metric("Rotation Signal", signal)

        with col4:
            st.metric(
                "Sectors",
                f"{breadth['advancing']} / {breadth['declining']}",
                delta="Adv/Dec"
            )

        st.markdown("---")

        # Main visualization
        if view_type == "Treemap":
            st.markdown("### 🗺️ Sector Treemap")
            fig = interactive_chart_builder.create_sector_heatmap(
                sector_data,
                metric=metric,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

        elif view_type == "Bar Chart":
            st.markdown("### 📊 Sector Performance")
            fig = interactive_chart_builder.create_sector_bar_chart(
                sector_data,
                metric=metric,
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

        else:  # Rotation
            st.markdown("### 🔄 Sector Rotation Analysis")
            st.info(f"**{rotation['signal']}**: {rotation['reason']}")

            fig = interactive_chart_builder.create_sector_rotation_chart(
                sector_data,
                height=450
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            **Quadrant Interpretation:**
            - **Leading** (top-right): Strong 1W & 1M performance - momentum sectors
            - **Improving** (top-left): Strong 1W, weak 1M - early rotation targets
            - **Weakening** (bottom-right): Weak 1W, strong 1M - taking profits
            - **Lagging** (bottom-left): Weak 1W & 1M - avoid or watch for reversal
            """)

        st.markdown("---")

        # Side-by-side: Breadth gauge + Table
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### 📈 Market Breadth")
            fig_gauge = interactive_chart_builder.create_market_breadth_gauge(
                breadth,
                height=280
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            st.markdown("### 📋 Sector Details")
            # Create DataFrame for table
            table_data = []
            for s in sector_data:
                table_data.append({
                    'Sector': s.name,
                    'ETF': s.etf,
                    '1D': f"{s.perf_1d:+.2f}%",
                    '1W': f"{s.perf_1w:+.2f}%",
                    '1M': f"{s.perf_1m:+.2f}%",
                    'YTD': f"{s.perf_ytd:+.2f}%"
                })

            df_sectors = pd.DataFrame(table_data)
            st.dataframe(df_sectors, use_container_width=True, height=280)

        st.markdown("---")

        # Top movers section
        st.markdown("### 🚀 Top Movers by Sector")

        cols = st.columns(3)
        for i, s in enumerate(sector_data[:6]):
            with cols[i % 3]:
                with st.expander(f"**{s.name}** ({s.etf})", expanded=False):
                    if s.top_gainers:
                        st.markdown("**Top Gainers:**")
                        for g in s.top_gainers:
                            st.markdown(f"- {g['symbol']}: {g['perf']:+.2f}%")
                    if s.top_losers:
                        st.markdown("**Top Losers:**")
                        for l in s.top_losers:
                            st.markdown(f"- {l['symbol']}: {l['perf']:+.2f}%")

    else:
        st.warning("No sector data available. Please check your internet connection.")

