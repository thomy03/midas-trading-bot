"""
Dashboard styling utilities
"""
import streamlit as st


def apply_custom_css():
    """Apply custom CSS to the dashboard"""
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


def color_recommendation(val: str) -> str:
    """
    Get color styling for recommendation values

    Args:
        val: Recommendation string

    Returns:
        CSS style string
    """
    colors = {
        'STRONG_BUY': 'background-color: #00C853; color: white',
        'BUY': 'background-color: #64DD17; color: white',
        'WATCH': 'background-color: #FDD835; color: black',
        'OBSERVE': 'background-color: #FF6D00; color: white'
    }
    return colors.get(val, '')


RECOMMENDATION_COLORS = {
    'STRONG_BUY': '#00C853',
    'BUY': '#64DD17',
    'WATCH': '#FDD835',
    'OBSERVE': '#FF6D00'
}
