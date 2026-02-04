"""
📅 Calendar page for Market Screener Dashboard
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
    """Render the 📅 Calendar page"""
    st.title("📅 Economic Calendar")
    st.markdown("Track earnings dates, FOMC meetings, CPI releases, and other market-moving events")

    # Controls
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        days_ahead = st.selectbox(
            "Time Range",
            [7, 14, 30, 60],
            index=2,
            format_func=lambda x: f"Next {x} days",
            key="calendar_days"
        )

    with col2:
        view_type = st.selectbox(
            "View Type",
            ["Timeline", "Table", "Week View"],
            key="calendar_view"
        )

    with col3:
        # Get tracked symbols from watchlist
        tracked_symbols = []
        try:
            watchlists = watchlist_manager.get_all_watchlists()
            for wl in watchlists:
                tracked_symbols.extend(wl.symbols)
            tracked_symbols = list(set(tracked_symbols))
        except:
            pass

        if st.button("🔄 Refresh Calendar", key="refresh_calendar"):
            st.cache_data.clear()

    st.markdown("---")

    # Event type filter
    st.markdown("### 🎯 Filter Events")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        show_fomc = st.checkbox("FOMC Meetings", value=True, key="show_fomc")
    with col2:
        show_cpi = st.checkbox("CPI Reports", value=True, key="show_cpi")
    with col3:
        show_jobs = st.checkbox("Jobs Reports", value=True, key="show_jobs")
    with col4:
        show_earnings = st.checkbox("Earnings", value=True, key="show_earnings")

    # Build event type filter
    event_types = []
    if show_fomc:
        event_types.append(EventType.FOMC)
    if show_cpi:
        event_types.append(EventType.CPI)
    if show_jobs:
        event_types.append(EventType.JOBS)

    st.markdown("---")

    # Fetch events
    with st.spinner("Loading economic events..."):
        try:
            events = economic_calendar.get_upcoming_events(
                days_ahead=days_ahead,
                event_types=event_types if event_types else None,
                include_earnings=show_earnings,
                symbols=tracked_symbols[:20] if show_earnings else None  # Limit to 20 symbols
            )
        except Exception as e:
            st.error(f"Error loading events: {e}")
            events = []

    # Summary metrics
    st.markdown("### 📊 Calendar Summary")
    col1, col2, col3, col4 = st.columns(4)

    high_impact = [e for e in events if e.impact.value == 'high']
    earnings_events = [e for e in events if e.event_type == EventType.EARNINGS]
    fomc_events = [e for e in events if e.event_type == EventType.FOMC]

    with col1:
        st.metric("Total Events", len(events))
    with col2:
        st.metric("High Impact", len(high_impact), delta="⚠️" if high_impact else None)
    with col3:
        st.metric("Earnings", len(earnings_events))
    with col4:
        next_fomc = economic_calendar.get_next_event(EventType.FOMC)
        if next_fomc:
            days_to_fomc = (next_fomc.date - datetime.now()).days
            st.metric("Next FOMC", f"{days_to_fomc}d")
        else:
            st.metric("Next FOMC", "N/A")

    st.markdown("---")

    # Main visualization
    if view_type == "Timeline":
        st.markdown("### 📈 Event Timeline")
        if events:
            fig = interactive_chart_builder.create_calendar_timeline(events, height=450)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No events found for the selected filters.")

    elif view_type == "Table":
        st.markdown("### 📋 Event List")
        if events:
            fig = interactive_chart_builder.create_calendar_table(events, height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Also show DataFrame for filtering/sorting
            df_events = economic_calendar.get_calendar_dataframe(
                days_ahead=days_ahead,
                symbols=tracked_symbols[:20] if show_earnings else None
            )
            if not df_events.empty:
                st.dataframe(df_events, use_container_width=True, height=300)
        else:
            st.info("No events found for the selected filters.")

    else:  # Week View
        st.markdown("### 📅 This Week")
        week_events = economic_calendar.get_week_events(week_offset=0)
        fig = interactive_chart_builder.create_week_calendar(week_events, height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Next week
        st.markdown("### 📅 Next Week")
        next_week_events = economic_calendar.get_week_events(week_offset=1)
        fig_next = interactive_chart_builder.create_week_calendar(next_week_events, height=300)
        st.plotly_chart(fig_next, use_container_width=True)

    st.markdown("---")

    # Detailed sections
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 High Impact Events")
        if high_impact:
            for event in high_impact[:5]:
                days_until = (event.date - datetime.now()).days
                icon = "🔴" if days_until <= 3 else "🟡" if days_until <= 7 else "🟢"
                st.markdown(f"{icon} **{event.title}** - {event.date.strftime('%Y-%m-%d')}")
                st.caption(f"   {event.description[:60]}..." if event.description else "")
        else:
            st.info("No high impact events in the selected period.")

    with col2:
        st.markdown("### 💰 Upcoming Earnings")
        if earnings_events:
            for event in earnings_events[:5]:
                days_until = (event.date - datetime.now()).days
                icon = "🔴" if days_until <= 3 else "🟡" if days_until <= 7 else "🟢"
                st.markdown(f"{icon} **{event.symbol}** - {event.date.strftime('%Y-%m-%d')}")
        else:
            st.info("No earnings in tracked symbols for this period.")

    st.markdown("---")

    # Impact distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Events by Impact")
        if events:
            fig = interactive_chart_builder.create_event_impact_chart(events, height=280)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### ⚙️ Custom Events")
        st.caption("Add custom events to your calendar")

        custom_title = st.text_input("Event Title", key="custom_event_title")
        custom_date = st.date_input("Event Date", key="custom_event_date")
        custom_impact = st.selectbox(
            "Impact Level",
            ["HIGH", "MEDIUM", "LOW"],
            key="custom_event_impact"
        )

        if st.button("➕ Add Custom Event", key="add_custom_event"):
            if custom_title:
                try:
                    economic_calendar.add_custom_event(
                        title=custom_title,
                        date=datetime.combine(custom_date, datetime.min.time()),
                        impact=EventImpact(custom_impact.lower()),
                        description="Custom event added via dashboard"
                    )
                    st.success(f"Added: {custom_title}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding event: {e}")
            else:
                st.warning("Please enter an event title")

    # Earnings warnings section
    st.markdown("---")
    st.markdown("### ⚠️ Earnings Warnings for Watchlist")

    warnings = []
    for symbol in tracked_symbols[:10]:
        warning = economic_calendar.get_earnings_warning(symbol)
        if warning:
            warnings.append(warning)

    if warnings:
        for w in warnings:
            st.warning(w)
    else:
        st.success("No imminent earnings for tracked symbols (next 7 days)")

