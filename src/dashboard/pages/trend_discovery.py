"""
🔮 Trend Discovery page for Market Screener Dashboard
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
    """Render the 🔮 Trend Discovery page"""
    st.title("🔮 Trend Discovery")
    st.markdown("### Découverte automatique de tendances par IA")

    # Helper function to load latest report
    def load_latest_trend_report():
        """Charge le dernier rapport de tendances."""
        report_dir = settings.TREND_DATA_DIR
        if not os.path.exists(report_dir):
            return None

        # Find most recent report
        report_files = glob_module.glob(os.path.join(report_dir, 'trend_report_*.json'))
        if not report_files:
            return None

        latest_file = max(report_files, key=os.path.getmtime)
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['_file_path'] = latest_file
                data['_file_date'] = datetime.fromtimestamp(os.path.getmtime(latest_file))
                return data
        except Exception as e:
            st.error(f"Erreur de chargement: {e}")
            return None

    def load_learned_themes():
        """Charge les thèmes appris par l'IA."""
        path = os.path.join(settings.TREND_DATA_DIR, 'learned_themes.json')
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    # Load data
    report_data = load_latest_trend_report()
    learned_themes = load_learned_themes()

    # Tabs
    td_tab1, td_tab2, td_tab3, td_tab4 = st.tabs([
        "📊 Overview", "🎯 Tendances", "🧠 Thèmes IA", "📈 Sources & Stats"
    ])

    # ============================================
    # TAB 1: Overview
    # ============================================
    with td_tab1:
        if report_data:
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)

            trends = report_data.get('trends', [])
            sentiment = report_data.get('market_sentiment', 0)
            narratives = report_data.get('narrative_updates', [])

            with col1:
                st.metric("📈 Tendances", len(trends))
            with col2:
                sentiment_color = "🟢" if sentiment > 0.2 else "🟡" if sentiment > -0.2 else "🔴"
                st.metric("💹 Sentiment", f"{sentiment_color} {sentiment:+.2f}")
            with col3:
                st.metric("🆕 Nouveaux Narratifs", len(narratives))
            with col4:
                st.metric("🧠 Thèmes Appris", len(learned_themes))

            # Last scan info
            file_date = report_data.get('_file_date', datetime.now())
            st.info(f"📅 Dernier scan: **{file_date.strftime('%d/%m/%Y %H:%M')}**")

            # Run scan button
            col_btn1, col_btn2 = st.columns([1, 4])
            with col_btn1:
                if st.button("🚀 Lancer un Scan", type="primary", key="run_trend_scan"):
                    with st.spinner("Analyse en cours... (peut prendre 1-2 minutes)"):
                        try:
                            # Run async scan
                            async def run_scan():
                                discovery = TrendDiscovery(
                                    openrouter_api_key=settings.OPENROUTER_API_KEY,
                                    model=settings.OPENROUTER_MODEL,
                                    data_dir=settings.TREND_DATA_DIR
                                )
                                await discovery.initialize()
                                report = await discovery.daily_scan()
                                await discovery.close()
                                return report

                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            report = loop.run_until_complete(run_scan())
                            loop.close()

                            st.success(f"✅ Scan terminé! {len(report.trends)} tendances détectées.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erreur lors du scan: {e}")

            st.markdown("---")
            st.subheader("🔥 Top Tendances")

            # Create dataframe for trends
            if trends:
                trend_data = []
                for t in sorted(trends, key=lambda x: x.get('confidence', 0), reverse=True)[:10]:
                    trend_data.append({
                        'Nom': t.get('name', 'N/A'),
                        'Type': t.get('type', 'N/A').replace('_', ' ').title(),
                        'Force': t.get('strength', 'N/A').title(),
                        'Confiance': f"{t.get('confidence', 0)*100:.0f}%",
                        'Symboles': ', '.join(t.get('symbols', [])[:5])
                    })

                df_trends = pd.DataFrame(trend_data)

                # Style the dataframe
                def color_strength(val):
                    colors = {
                        'Established': 'background-color: #00C853; color: white',
                        'Developing': 'background-color: #FDD835; color: black',
                        'Emerging': 'background-color: #2196F3; color: white'
                    }
                    return colors.get(val, '')

                st.dataframe(
                    df_trends.style.applymap(color_strength, subset=['Force']),
                    use_container_width=True,
                    height=400
                )
            else:
                st.warning("Aucune tendance dans le rapport.")
        else:
            st.warning("⚠️ Aucun rapport de tendances trouvé. Lancez un scan pour commencer.")
            if st.button("🚀 Lancer le premier scan", type="primary"):
                st.info("Veuillez utiliser la commande: `python scripts/run_trend_discovery.py`")

    # ============================================
    # TAB 2: Tendances Détaillées
    # ============================================
    with td_tab2:
        if report_data:
            trends = report_data.get('trends', [])

            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                type_filter = st.selectbox(
                    "Type",
                    ["Tous", "Sector Momentum", "Thematic"],
                    key="td_type_filter"
                )
            with col2:
                strength_filter = st.selectbox(
                    "Force",
                    ["Tous", "Established", "Developing", "Emerging"],
                    key="td_strength_filter"
                )
            with col3:
                min_confidence = st.slider(
                    "Confiance min (%)",
                    0, 100, 40,
                    key="td_conf_filter"
                )

            # Filter trends
            filtered = trends
            if type_filter != "Tous":
                type_key = type_filter.lower().replace(' ', '_')
                filtered = [t for t in filtered if t.get('type', '') == type_key]
            if strength_filter != "Tous":
                filtered = [t for t in filtered if t.get('strength', '').lower() == strength_filter.lower()]
            filtered = [t for t in filtered if t.get('confidence', 0) * 100 >= min_confidence]

            st.markdown(f"**{len(filtered)} tendances** correspondent aux filtres")
            st.markdown("---")

            # Display each trend as expander
            for trend in sorted(filtered, key=lambda x: x.get('confidence', 0), reverse=True):
                conf = trend.get('confidence', 0) * 100
                strength = trend.get('strength', 'emerging')
                icon = "🔥" if strength == "established" else "📈" if strength == "developing" else "🌱"

                with st.expander(f"{icon} **{trend.get('name', 'N/A')}** ({conf:.0f}%)"):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"**Description:** {trend.get('description', 'N/A')}")

                        catalysts = trend.get('key_catalysts', [])
                        if catalysts:
                            st.markdown("**Catalyseurs:**")
                            for cat in catalysts:
                                st.markdown(f"  • {cat}")

                    with col2:
                        st.markdown(f"**Type:** {trend.get('type', 'N/A').replace('_', ' ').title()}")
                        st.markdown(f"**Force:** {strength.title()}")
                        st.markdown(f"**Momentum:** {trend.get('momentum_score', 0):.2%}")

                        sources = trend.get('sources', [])
                        if sources:
                            st.markdown(f"**Sources:** {', '.join(sources)}")

                    # Symbols
                    symbols = trend.get('symbols', [])
                    if symbols:
                        st.markdown("**Symboles:**")
                        st.code(', '.join(symbols))
        else:
            st.warning("Aucun rapport disponible.")

    # ============================================
    # TAB 3: Thèmes IA
    # ============================================
    with td_tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📚 Thèmes Prédéfinis")
            st.caption(f"{len(TrendDiscovery.THEMES_KEYWORDS)} thèmes de base")

            for theme_name, keywords in TrendDiscovery.THEMES_KEYWORDS.items():
                with st.container():
                    st.info(f"**{theme_name}**\n\n_{', '.join(keywords[:4])}..._")

        with col2:
            st.subheader("🆕 Thèmes Découverts par IA")
            st.caption(f"{len(learned_themes)} thèmes appris")

            if learned_themes:
                for theme_name, data in learned_themes.items():
                    occ = data.get('occurrence_count', 1)
                    discovered = data.get('discovered_at', '')[:10]
                    desc = data.get('description', '')[:100]
                    symbols = data.get('symbols', [])

                    with st.container():
                        st.success(f"""**{data.get('name', theme_name)}** (vu {occ}x)

📅 Découvert: {discovered}

{desc}...

🎯 Symboles: {', '.join(symbols[:5]) if symbols else 'N/A'}""")
            else:
                st.info("Aucun thème découvert. Lancez un scan pour que l'IA identifie de nouveaux narratifs.")

    # ============================================
    # TAB 4: Sources & Stats
    # ============================================
    with td_tab4:
        st.subheader("📡 Sources de Données")

        # Check API keys status
        sources_data = [
            {
                'Source': 'yfinance',
                'Type': 'Prix & Volume',
                'Statut': '✅ Active',
                'Config': 'N/A'
            },
            {
                'Source': 'NewsAPI',
                'Type': 'Actualités',
                'Statut': '✅ Active' if settings.NEWSAPI_KEY else '⚠️ Non configuré',
                'Config': 'NEWSAPI_KEY'
            },
            {
                'Source': 'OpenRouter (LLM)',
                'Type': 'Analyse IA',
                'Statut': '✅ Active' if settings.OPENROUTER_API_KEY else '⚠️ Non configuré',
                'Config': 'OPENROUTER_API_KEY'
            },
            {
                'Source': 'AlphaVantage',
                'Type': 'News Sentiment',
                'Statut': '✅ Active' if getattr(settings, 'ALPHAVANTAGE_KEY', '') else '⚠️ Non configuré',
                'Config': 'ALPHAVANTAGE_KEY'
            }
        ]

        df_sources = pd.DataFrame(sources_data)
        st.dataframe(df_sources, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("📜 Historique des Scans")

        # List all report files
        report_dir = settings.TREND_DATA_DIR
        if os.path.exists(report_dir):
            report_files = glob_module.glob(os.path.join(report_dir, 'trend_report_*.json'))
            report_files = sorted(report_files, key=os.path.getmtime, reverse=True)[:10]

            if report_files:
                for rf in report_files:
                    file_date = datetime.fromtimestamp(os.path.getmtime(rf))
                    try:
                        with open(rf, 'r', encoding='utf-8') as f:
                            rdata = json.load(f)
                        n_trends = len(rdata.get('trends', []))
                        n_narratives = len(rdata.get('narrative_updates', []))
                        sentiment = rdata.get('market_sentiment', 0)

                        st.markdown(f"""
**{file_date.strftime('%d/%m/%Y %H:%M')}** - {n_trends} tendances, {n_narratives} nouveaux narratifs (sentiment: {sentiment:+.2f})
""")
                    except:
                        st.markdown(f"**{file_date.strftime('%d/%m/%Y %H:%M')}** - Erreur de lecture")
            else:
                st.info("Aucun historique de scan disponible.")
        else:
            st.warning(f"Répertoire {report_dir} non trouvé.")

        st.markdown("---")
        st.subheader("🎯 Symboles Focus")

        # Load focus symbols
        focus_path = os.path.join(settings.TREND_DATA_DIR, 'focus_symbols.json')
        if os.path.exists(focus_path):
            try:
                with open(focus_path, 'r', encoding='utf-8') as f:
                    focus_data = json.load(f)
                symbols = focus_data.get('symbols', [])
                if symbols:
                    st.code(', '.join(symbols))

                    # Download button
                    csv_content = "Symbol\n" + "\n".join(symbols)
                    st.download_button(
                        "📥 Télécharger CSV",
                        data=csv_content,
                        file_name="focus_symbols.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Erreur: {e}")
        else:
            st.info("Aucun symbole focus. Lancez un scan pour générer la liste.")

