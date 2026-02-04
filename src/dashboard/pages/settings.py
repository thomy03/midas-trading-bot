"""
⚙️ Settings page for Market Screener Dashboard
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
    """Render the ⚙️ Settings page"""
    st.title("⚙️ Settings & Configuration")

    # Create tabs for different settings sections
    settings_tab1, settings_tab2, settings_tab3, settings_tab4 = st.tabs([
        "📊 Screening", "🔔 Notifications", "📧 Email/Telegram", "🛠️ System"
    ])

    with settings_tab1:
        st.markdown("### Screening Parameters")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### EMA Settings")
            st.code(f"""
EMA Periods: {EMA_PERIODS}
Support Zone Tolerance: ±{ZONE_TOLERANCE}%
            """)

        with col2:
            st.markdown("#### Market Filters")
            from config.settings import (
                MIN_MARKET_CAP_NASDAQ,
                MIN_MARKET_CAP_SP500,
                MIN_MARKET_CAP_EUROPE,
                MIN_DAILY_VOLUME
            )
            st.code(f"""
NASDAQ Min Cap: ${MIN_MARKET_CAP_NASDAQ}M
SP500 Min Cap: ${MIN_MARKET_CAP_SP500}M
Europe Min Cap: ${MIN_MARKET_CAP_EUROPE}M
Min Daily Volume: ${MIN_DAILY_VOLUME/1e3:.0f}k
            """)

        st.markdown("#### Scheduling")
        from config.settings import DAILY_REPORT_TIME, TIMEZONE
        st.code(f"""
Daily Report Time: {DAILY_REPORT_TIME.strftime('%H:%M')}
Timezone: {TIMEZONE}
        """)

    with settings_tab2:
        st.markdown("### Notification Preferences")

        # Get current config from notification manager
        current_config = notification_manager.config

        # Notification channels enable/disable
        st.markdown("#### Enabled Channels")
        col1, col2, col3 = st.columns(3)

        with col1:
            telegram_enabled = st.checkbox(
                "📱 Telegram",
                value=current_config.telegram_enabled,
                key="notif_telegram_enabled"
            )
        with col2:
            email_enabled = st.checkbox(
                "📧 Email",
                value=current_config.email_enabled,
                key="notif_email_enabled"
            )
        with col3:
            in_app_enabled = st.checkbox(
                "🔔 In-App",
                value=current_config.in_app_enabled,
                key="notif_in_app_enabled"
            )

        st.markdown("---")
        st.markdown("#### Alert Type Filters")
        st.caption("Only receive notifications for selected alert types")

        alert_types_options = ["STRONG_BUY", "BUY", "WATCH", "OBSERVE"]
        selected_alert_types = st.multiselect(
            "Notify for these alerts:",
            options=alert_types_options,
            default=current_config.alert_types if current_config.alert_types else ["STRONG_BUY", "BUY"],
            key="notif_alert_types"
        )

        st.markdown("---")
        st.markdown("#### Priority Filter")

        priority_options = {"low": "Low (all)", "medium": "Medium+", "high": "High+", "urgent": "Urgent only"}
        min_priority = st.select_slider(
            "Minimum priority level:",
            options=list(priority_options.keys()),
            value=current_config.min_priority,
            format_func=lambda x: priority_options[x],
            key="notif_min_priority"
        )

        st.markdown("---")
        st.markdown("#### Quiet Hours")
        st.caption("No notifications during these hours (except urgent)")

        quiet_col1, quiet_col2 = st.columns(2)
        with quiet_col1:
            quiet_start = st.selectbox(
                "Start (hour):",
                options=[None] + list(range(24)),
                index=0 if current_config.quiet_hours_start is None else current_config.quiet_hours_start + 1,
                format_func=lambda x: "Disabled" if x is None else f"{x:02d}:00",
                key="notif_quiet_start"
            )
        with quiet_col2:
            quiet_end = st.selectbox(
                "End (hour):",
                options=[None] + list(range(24)),
                index=0 if current_config.quiet_hours_end is None else current_config.quiet_hours_end + 1,
                format_func=lambda x: "Disabled" if x is None else f"{x:02d}:00",
                key="notif_quiet_end"
            )

        # Save button
        if st.button("💾 Save Notification Settings", key="save_notif_settings"):
            notification_manager.update_config(
                telegram_enabled=telegram_enabled,
                email_enabled=email_enabled,
                in_app_enabled=in_app_enabled,
                alert_types=selected_alert_types,
                min_priority=min_priority,
                quiet_hours_start=quiet_start,
                quiet_hours_end=quiet_end
            )
            st.success("✅ Notification settings saved!")

        # Show notification stats
        st.markdown("---")
        st.markdown("#### Notification Statistics")
        stats = notification_manager.get_notification_stats()

        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        with stats_col1:
            st.metric("Total Sent", stats['sent'])
        with stats_col2:
            st.metric("Failed", stats['failed'])
        with stats_col3:
            tg_stats = stats['by_channel'].get('telegram', {})
            st.metric("Telegram", tg_stats.get('sent', 0))
        with stats_col4:
            email_stats = stats['by_channel'].get('email', {})
            st.metric("Email", email_stats.get('sent', 0))

        # Recent notifications
        recent_notifs = notification_manager.get_recent_notifications(limit=10)
        if recent_notifs:
            st.markdown("#### Recent Notifications")
            for notif in recent_notifs[:5]:
                status = "✅" if notif['sent'] else "❌"
                channel_emoji = {"telegram": "📱", "email": "📧", "in_app": "🔔"}.get(notif['channel'], "📊")
                st.text(f"{status} {channel_emoji} {notif['title']} ({notif['timestamp'][:16]})")

    with settings_tab3:
        st.markdown("### Telegram Configuration")

        from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

        telegram_configured = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

        if telegram_configured:
            st.success("✅ Telegram configured via environment variables")
            st.code(f"""
Bot Token: {TELEGRAM_BOT_TOKEN[:10]}...{TELEGRAM_BOT_TOKEN[-5:] if len(TELEGRAM_BOT_TOKEN) > 15 else ''}
Chat ID: {TELEGRAM_CHAT_ID}
            """)
        else:
            st.warning("⚠️ Telegram not configured")
            st.info("""
To configure Telegram:
1. Create a bot with @BotFather on Telegram
2. Get your chat ID by messaging @userinfobot
3. Add to .env file:
   - TELEGRAM_BOT_TOKEN=your_bot_token
   - TELEGRAM_CHAT_ID=your_chat_id
            """)

        if st.button("🧪 Test Telegram", key="test_telegram", disabled=not telegram_configured):
            with st.spinner("Sending test message..."):
                success = notification_manager.test_telegram()
                if success:
                    st.success("✅ Telegram test successful!")
                else:
                    st.error("❌ Telegram test failed. Check your configuration.")

        st.markdown("---")
        st.markdown("### Email Configuration")

        email_configured = bool(notification_manager.config.smtp_username)

        if email_configured:
            st.success("✅ Email configured")
            st.code(f"""
SMTP Server: {notification_manager.config.smtp_server}:{notification_manager.config.smtp_port}
From: {notification_manager.config.email_from}
To: {', '.join(notification_manager.config.email_to)}
            """)
        else:
            st.warning("⚠️ Email not configured")
            st.info("""
To configure Email:
Add to .env file:
- SMTP_SERVER=smtp.gmail.com
- SMTP_PORT=587
- SMTP_USERNAME=your_email@gmail.com
- SMTP_PASSWORD=your_app_password
- EMAIL_FROM=your_email@gmail.com
- EMAIL_TO=recipient@email.com
            """)

        if st.button("🧪 Test Email", key="test_email", disabled=not email_configured):
            with st.spinner("Sending test email..."):
                success = notification_manager.test_email()
                if success:
                    st.success("✅ Email test successful!")
                else:
                    st.error("❌ Email test failed. Check your configuration.")

    with settings_tab4:
        st.markdown("### System Information")

        col1, col2 = st.columns(2)

        with col1:
            import platform
            st.markdown("#### Environment")
            st.code(f"""
Python: {platform.python_version()}
OS: {platform.system()} {platform.release()}
            """)

        with col2:
            st.markdown("#### Database")
            st.info("Location: data/screener.db")
            try:
                recent = db_manager.get_recent_alerts(days=1)
                st.metric("Alerts (24h)", len(recent))
            except:
                st.error("Could not access database")

        st.markdown("---")
        st.markdown("### Quick Actions")

        action_col1, action_col2, action_col3 = st.columns(3)

        with action_col1:
            if st.button("🔄 Refresh Config", key="refresh_config"):
                st.cache_data.clear()
                st.success("Configuration refreshed!")
                st.rerun()

        with action_col2:
            if st.button("🗑️ Clear Cache", key="clear_cache"):
                st.cache_data.clear()
                st.success("Cache cleared!")

        with action_col3:
            if st.button("📋 Export Config", key="export_config"):
                config_data = notification_manager.config.to_dict()
                # Remove sensitive data
                config_data['telegram_bot_token'] = '***'
                config_data['smtp_password'] = '***'
                st.json(config_data)

# Footer
st.markdown("---")
st.caption("Market Screener Dashboard | Powered by Streamlit & Plotly")
