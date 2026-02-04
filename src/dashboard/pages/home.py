"""
🏠 Home page for Market Screener Dashboard
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
    """Render the 🏠 Home page"""
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

