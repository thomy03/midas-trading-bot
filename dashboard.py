"""
Market Screener Dashboard - TradingView-like Interface

DEPRECATED: This dashboard is deprecated in favor of webapp.py
Please use: python webapp.py
This file is kept for reference only and will be removed in a future version.

Run with: streamlit run dashboard.py

REFACTORED: Page logic has been extracted to src/dashboard/pages/
This file is now a minimal entry point.
"""
import warnings
warnings.warn(
    "dashboard.py is DEPRECATED. Please use 'python webapp.py' instead.",
    DeprecationWarning,
    stacklevel=2
)
import streamlit as st
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

# Import page modules
from src.dashboard.pages import PAGE_MODULES, PAGE_NAMES
from src.database.db_manager import db_manager

# Page config
st.set_page_config(
    page_title="Market Screener Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #00C853;
    }
    h2 {
        color: #2962FF;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📈 Market Screener")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        PAGE_NAMES
    )

    st.markdown("---")
    st.markdown("### Quick Stats")

    # Get recent alerts count
    try:
        recent_alerts = db_manager.get_recent_alerts(days=7)
        st.metric("Alerts (7 days)", len(recent_alerts))
    except:
        st.metric("Alerts (7 days)", "N/A")

    st.markdown("---")
    st.caption("Developed with ❤️")

# Render selected page
page_module = PAGE_MODULES.get(page)
if page_module:
    page_module.render()
else:
    st.error(f"Page not found: {page}")
