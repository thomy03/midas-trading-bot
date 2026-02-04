"""
🚨 Alerts History page for Market Screener Dashboard
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
    """Render the 🚨 Alerts History page"""
    st.title("🚨 Alerts History")
    st.markdown("View and analyze historical alerts")

    col1, col2 = st.columns([1, 3])

    with col1:
        days_back = st.slider("Days to look back", 1, 90, 30, key="alerts_days")

    with col2:
        timeframe_filter = st.multiselect(
            "Filter by timeframe",
            ["daily", "weekly"],
            default=["daily", "weekly"],
            key="alerts_timeframe"
        )

    try:
        alerts = db_manager.get_recent_alerts(days=days_back)

        # Filter by timeframe
        if timeframe_filter:
            alerts = [a for a in alerts if a.timeframe in timeframe_filter]

        if alerts:
            st.info(f"Found {len(alerts)} alerts in the last {days_back} days")

            # Create detailed dataframe
            alert_data = []
            for alert in alerts:
                alert_data.append({
                    'Date': alert.alert_date.strftime('%Y-%m-%d %H:%M'),
                    'Symbol': alert.symbol,
                    'Company': alert.company_name,
                    'Timeframe': alert.timeframe.upper(),
                    'Price': alert.current_price,
                    'Support': alert.support_level,
                    'Distance %': alert.distance_to_support_pct,
                    'EMA 24': alert.ema_24,
                    'EMA 38': alert.ema_38,
                    'EMA 62': alert.ema_62,
                    'Recommendation': alert.recommendation,
                    'Notified': '✅' if alert.is_notified else '❌'
                })

            df = pd.DataFrame(alert_data)

            # Display with formatting
            st.dataframe(
                df.style.format({
                    'Price': '${:.2f}',
                    'Support': '${:.2f}',
                    'Distance %': '{:.2f}%',
                    'EMA 24': '${:.2f}',
                    'EMA 38': '${:.2f}',
                    'EMA 62': '${:.2f}'
                }),
                use_container_width=True,
                height=400
            )

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"alerts_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

            # Statistics
            st.markdown("---")
            st.subheader("📊 Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                strong_buys = sum(1 for a in alerts if a.recommendation == 'STRONG_BUY')
                st.metric("Strong Buys", strong_buys)

            with col2:
                buys = sum(1 for a in alerts if a.recommendation == 'BUY')
                st.metric("Buys", buys)

            with col3:
                weekly_signals = sum(1 for a in alerts if a.timeframe == 'weekly')
                st.metric("Weekly Signals", weekly_signals)

            with col4:
                daily_signals = sum(1 for a in alerts if a.timeframe == 'daily')
                st.metric("Daily Signals", daily_signals)

        else:
            st.info("No alerts found for the selected criteria.")

    except Exception as e:
        st.error(f"Error loading alerts: {e}")

