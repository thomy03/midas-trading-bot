"""
📈 Signaux Historiques page for Market Screener Dashboard
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
    """Render the 📈 Signaux Historiques page"""
    st.title("📈 Signaux Historiques - Niveaux + RSI Trendlines")
    st.markdown("### Visualiser TOUS les anciens signaux avec les niveaux historiques et les obliques RSI")

    st.info("""
    **Ce graphique affiche:**
    - 🟢 **Lignes vertes pointillées**: Niveaux historiques valides (> 8%)
    - 🔴 **Lignes rouges pleines**: Niveaux historiques PROCHES (< 8%)
    - ⭐ **Étoiles sur prix**: Points de crossover EMA
    - 📉 **Ligne rouge oblique (RSI)**: Trendline RSI descendante
    - 🔺 **Triangles rouges (RSI)**: Pics RSI utilisés pour la trendline
    - ⭐ **Étoile verte (RSI)**: RSI Breakout (cassure de l'oblique)
    """)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        hist_symbol = st.text_input("Enter Stock Symbol", value="TSLA", key="hist_symbol").upper()

    with col2:
        hist_timeframe = st.selectbox("Timeframe", ["weekly", "daily"], key="hist_timeframe")

    with col3:
        hist_period = st.selectbox("Period", ["1y", "2y", "5y", "max"], index=2, key="hist_period")  # Default to 5y

    if st.button("📊 Afficher les Signaux Historiques", type="primary"):
        with st.spinner(f"Chargement des signaux historiques pour {hist_symbol}..."):
            fig = chart_visualizer.create_historical_chart(
                symbol=hist_symbol,
                timeframe=hist_timeframe,
                period=hist_period
            )

            if fig:
                st.plotly_chart(fig, use_container_width=True)

                # Show summary
                st.markdown("---")
                st.subheader("📋 Résumé")

                # Get the data
                interval = '1wk' if hist_timeframe == 'weekly' else '1d'
                df = market_data_fetcher.get_historical_data(hist_symbol, period=hist_period, interval=interval)

                if df is not None:
                    df = ema_analyzer.calculate_emas(df)
                    current_price = float(df['Close'].iloc[-1])
                    crossovers = ema_analyzer.detect_crossovers(df, hist_timeframe)
                    historical_levels = ema_analyzer.find_historical_support_levels(df, crossovers, current_price)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Prix Actuel", f"${current_price:.2f}")
                        st.metric("Niveaux Historiques", len(historical_levels))

                    with col2:
                        near_levels = [l for l in historical_levels if l['is_near']]
                        st.metric("Niveaux PROCHES (< 8%)", len(near_levels))
                        if near_levels:
                            st.success(f"✅ Prix s'approche de {len(near_levels)} niveau(x)!")
                        else:
                            st.info("📍 Aucun niveau proche actuellement")

                    with col3:
                        from trendline_analysis.core.rsi_breakout_analyzer import RSIBreakoutAnalyzer
                        rsi_analyzer = RSIBreakoutAnalyzer()
                        lookback = 104 if hist_timeframe == 'weekly' else 252
                        rsi_result = rsi_analyzer.analyze(df, lookback_periods=lookback)

                        if rsi_result and rsi_result.has_rsi_breakout:
                            st.metric("RSI Breakout", "✅ OUI")
                            st.success(f"Strength: {rsi_result.rsi_breakout.strength}")
                        elif rsi_result and rsi_result.has_rsi_trendline:
                            st.metric("RSI Trendline", "✅ OUI")
                            st.info("Pas encore de breakout")
                        else:
                            st.metric("RSI Signal", "❌ AUCUN")

                    # Show signal interpretation
                    st.markdown("---")
                    st.subheader("🎯 Interprétation du Signal")

                    if near_levels and rsi_result and rsi_result.has_rsi_breakout:
                        st.success("""
                        🚨 **SIGNAL POTENTIEL DÉTECTÉ!**

                        Le prix s'approche d'un niveau historique ET il y a un RSI breakout.
                        → C'est exactement le type de signal que le système recherche!
                        → Recommandation: STRONG_BUY
                        """)
                    elif near_levels:
                        st.warning("""
                        ⚠️ **Prix proche d'un niveau historique**

                        Mais pas de RSI breakout détecté pour le moment.
                        → Surveiller l'évolution du RSI
                        """)
                    elif rsi_result and rsi_result.has_rsi_breakout:
                        st.info("""
                        📊 **RSI Breakout détecté**

                        Mais le prix est LOIN des niveaux historiques (> 8%).
                        → Pas de signal car le prix doit s'approcher d'un niveau
                        """)
                    else:
                        st.info("""
                        📍 **Aucun signal actuellement**

                        Les signaux apparaîtront quand:
                        1. Le prix retracera vers un niveau historique (< 8%)
                        2. ET qu'il y aura un RSI breakout
                        """)
            else:
                st.error(f"Impossible de charger les données pour {hist_symbol}")

