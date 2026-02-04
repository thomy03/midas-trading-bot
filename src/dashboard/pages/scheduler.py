"""
⏰ Scheduler page for Market Screener Dashboard
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
    """Render the ⏰ Scheduler page"""
    st.title("⏰ Scheduled Scans")
    st.markdown("Configure automatic market scanning on a schedule")

    # Initialize scheduler with screen function and stocks provider
    def get_stocks_for_scheduler():
        """Get stocks based on scheduler config"""
        stocks = []
        config = scan_scheduler.config
        for market in config.markets:
            if market == "NASDAQ":
                stocks.extend(market_data_fetcher.get_nasdaq_symbols()[:200])
            elif market == "SP500":
                stocks.extend(market_data_fetcher.get_sp500_symbols())
            elif market == "EUROPE":
                stocks.extend(market_data_fetcher.get_europe_symbols())
            elif market == "CRYPTO":
                stocks.extend(market_data_fetcher.get_crypto_symbols())
        return stocks

    scan_scheduler.set_screen_function(market_screener.screen_single_stock_weekly)
    scan_scheduler.set_stocks_provider(get_stocks_for_scheduler)
    scan_scheduler.set_notification_manager(notification_manager)

    # Create tabs
    sched_tab1, sched_tab2, sched_tab3 = st.tabs(["📊 Status", "⚙️ Configuration", "📜 History"])

    with sched_tab1:
        st.markdown("### Scheduler Status")

        status = scan_scheduler.get_status()

        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if status['running']:
                st.success("🟢 Running")
            else:
                st.warning("🟡 Stopped")
        with col2:
            if status['enabled']:
                st.success("✅ Enabled")
            else:
                st.info("⏸️ Disabled")
        with col3:
            st.metric("Total Runs", status['total_runs'])
        with col4:
            st.metric("Last Alerts", status['last_alerts'])

        # Next/Last run info
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### ⏭️ Next Scheduled Run")
            if status['next_run']:
                next_run = datetime.fromisoformat(status['next_run'])
                time_until = next_run - datetime.now()
                st.info(f"**{next_run.strftime('%Y-%m-%d %H:%M')}**")
                if time_until.total_seconds() > 0:
                    hours, remainder = divmod(int(time_until.total_seconds()), 3600)
                    minutes = remainder // 60
                    st.caption(f"In {hours}h {minutes}m")
            else:
                st.info("Not scheduled")

        with col2:
            st.markdown("#### ⏮️ Last Run")
            if status['last_run']:
                last_run = datetime.fromisoformat(status['last_run'])
                st.info(f"**{last_run.strftime('%Y-%m-%d %H:%M')}**")
                st.caption(f"Found {status['last_alerts']} alerts")
            else:
                st.info("No previous runs")

        # Control buttons
        st.markdown("---")
        st.markdown("### Controls")
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

        with ctrl_col1:
            if st.button("▶️ Start Scheduler", key="start_scheduler", disabled=status['running']):
                if scan_scheduler.start():
                    st.success("Scheduler started!")
                    st.rerun()
                else:
                    st.error("Failed to start scheduler")

        with ctrl_col2:
            if st.button("⏹️ Stop Scheduler", key="stop_scheduler", disabled=not status['running']):
                scan_scheduler.stop()
                st.success("Scheduler stopped!")
                st.rerun()

        with ctrl_col3:
            if st.button("🔄 Run Now", key="run_now"):
                if scan_scheduler.trigger_now():
                    st.success("Scan triggered!")
                else:
                    st.warning("Could not trigger scan (already running?)")

        with ctrl_col4:
            if st.button("🔃 Refresh Status", key="refresh_scheduler"):
                st.rerun()

        # Current scan progress if running
        scan_state = background_scanner.get_state()
        if scan_state.status == ScanStatus.RUNNING:
            st.markdown("---")
            st.markdown("### 🔄 Scan in Progress")
            progress = scan_state.completed_count / scan_state.total_stocks if scan_state.total_stocks > 0 else 0
            st.progress(progress)
            st.caption(f"Scanning: {scan_state.current_symbol} ({scan_state.completed_count}/{scan_state.total_stocks})")
            st.caption(f"Alerts found: {len(scan_state.alerts)}")

    with sched_tab2:
        st.markdown("### Schedule Configuration")

        current_config = scan_scheduler.config

        # Enable/Disable
        enabled = st.checkbox(
            "Enable Scheduled Scanning",
            value=current_config.enabled,
            key="scheduler_enabled"
        )

        st.markdown("---")
        st.markdown("#### Schedule Type")

        schedule_type = st.selectbox(
            "Run scans:",
            options=["daily", "weekly", "hourly", "interval"],
            index=["daily", "weekly", "hourly", "interval"].index(current_config.schedule_type),
            format_func=lambda x: {
                "daily": "📅 Daily (at specific time)",
                "weekly": "📆 Weekly (on specific days)",
                "hourly": "⏰ Hourly",
                "interval": "🔄 Every N minutes"
            }.get(x, x),
            key="schedule_type"
        )

        # Time configuration based on type
        if schedule_type in ["daily", "weekly"]:
            time_col1, time_col2 = st.columns(2)
            with time_col1:
                time_of_day = st.time_input(
                    "Time of day:",
                    value=datetime.strptime(current_config.time_of_day, "%H:%M").time(),
                    key="time_of_day"
                )
            with time_col2:
                if schedule_type == "weekly":
                    days_options = {
                        0: "Monday", 1: "Tuesday", 2: "Wednesday",
                        3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"
                    }
                    days_of_week = st.multiselect(
                        "Days of week:",
                        options=list(days_options.keys()),
                        default=current_config.days_of_week,
                        format_func=lambda x: days_options[x],
                        key="days_of_week"
                    )
                else:
                    # Daily - weekdays only option
                    weekdays_only = st.checkbox(
                        "Weekdays only (Mon-Fri)",
                        value=current_config.days_of_week == [0, 1, 2, 3, 4],
                        key="weekdays_only"
                    )
                    days_of_week = [0, 1, 2, 3, 4] if weekdays_only else list(range(7))
        elif schedule_type == "interval":
            interval_minutes = st.number_input(
                "Interval (minutes):",
                min_value=15,
                max_value=480,
                value=current_config.interval_minutes,
                step=15,
                key="interval_minutes"
            )
            time_of_day = current_config.time_of_day
            days_of_week = list(range(7))
        else:  # hourly
            time_of_day = current_config.time_of_day
            days_of_week = list(range(7))
            interval_minutes = 60

        st.markdown("---")
        st.markdown("#### Markets to Scan")

        markets_options = ["NASDAQ", "SP500", "EUROPE", "CRYPTO", "CAC40", "DAX"]
        markets = st.multiselect(
            "Select markets:",
            options=markets_options,
            default=current_config.markets,
            key="scheduler_markets"
        )

        st.markdown("---")
        st.markdown("#### Notifications")

        notif_col1, notif_col2 = st.columns(2)
        with notif_col1:
            notify_on_completion = st.checkbox(
                "Notify when scan completes",
                value=current_config.notify_on_completion,
                key="notify_completion"
            )
        with notif_col2:
            notify_on_signals = st.checkbox(
                "Notify for each signal found",
                value=current_config.notify_on_signals,
                key="notify_signals"
            )

        # Save button
        if st.button("💾 Save Configuration", key="save_scheduler_config"):
            time_str = time_of_day.strftime("%H:%M") if hasattr(time_of_day, 'strftime') else time_of_day

            scan_scheduler.update_config(
                enabled=enabled,
                schedule_type=schedule_type,
                time_of_day=time_str,
                days_of_week=days_of_week,
                interval_minutes=interval_minutes if schedule_type == "interval" else 60,
                markets=markets,
                notify_on_completion=notify_on_completion,
                notify_on_signals=notify_on_signals
            )
            st.success("✅ Configuration saved!")

            # Show next run time
            if enabled:
                next_run = scan_scheduler._calculate_next_run()
                st.info(f"Next scheduled run: {next_run.strftime('%Y-%m-%d %H:%M')}")

    with sched_tab3:
        st.markdown("### Scan History")

        # Get alerts from recent scans
        scan_state = background_scanner.get_state()

        if scan_state.alerts:
            st.markdown(f"**Last scan found {len(scan_state.alerts)} alerts:**")

            for alert in scan_state.alerts[:20]:  # Show last 20
                with st.expander(f"🎯 {alert.get('symbol', 'N/A')} - {alert.get('recommendation', 'N/A')}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Price", f"${alert.get('current_price', 0):.2f}")
                    with col2:
                        st.metric("Support", f"${alert.get('support_level', 0):.2f}")
                    with col3:
                        st.metric("Confidence", f"{alert.get('confidence_score', 0):.0f}/100")
        else:
            st.info("No alerts from recent scans. Run a scan to see results here.")

        # Show scheduler statistics
        st.markdown("---")
        st.markdown("### Statistics")

        stats = scan_scheduler.get_status()
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Total Scans Run", stats['total_runs'])
        with stat_col2:
            st.metric("Last Scan Alerts", stats['last_alerts'])
        with stat_col3:
            if stats['last_run']:
                last_run = datetime.fromisoformat(stats['last_run'])
                st.metric("Last Run", last_run.strftime('%m/%d %H:%M'))
            else:
                st.metric("Last Run", "Never")

